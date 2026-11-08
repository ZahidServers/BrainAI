# BrainAI - built by Mohammed Zahid.
"""Is AMSA actually better than GELU or SwiGLU?  Measure it.

Trains the same small non-spiking transformer LM once per activation, changing
only the feed-forward nonlinearity, on the same data with the same seeds and a
matched parameter budget.  Non-spiking on purpose: mixing in the SNN would
confound the activation's contribution with the spiking dynamics.

Parameter matching matters and is easy to get wrong.  SwiGLU and GatedAMSA use
three projections (up, gate, down); GELU uses two.  With the same hidden
multiplier GELU would be handed a third fewer parameters and would lose for the
wrong reason, so its hidden size is widened to 6d against their 4d.

The ``amsa+state`` arm exists because AMSA's adaptive terms multiply buffers
that are zero unless something writes them.  Benchmarking only the static form
would test a different function than the one the architecture actually uses.

    python benchmarks/activation_bench.py --steps 300

Result as of the last run: **SwiGLU wins on both quality and speed**, and
driving AMSA's state changes its loss by ~0.0001.  A single small run on one
corpus is weak evidence, but it is the evidence there is, and it does not
support AMSA.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai.activation import AMSA  # noqa: E402
from brainai.attention import RMSNorm  # noqa: E402
from brainai.data import load_corpus  # noqa: E402


class GELUFFN(nn.Module):
    def __init__(self, d: int, mult: int = 6) -> None:
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, d: int, mult: int = 4) -> None:
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.gate = nn.Linear(d, mult * d, bias=False)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AMSAFFN(nn.Module):
    def __init__(self, d: int, mult: int = 4) -> None:
        super().__init__()
        self.up = nn.Linear(d, mult * d, bias=False)
        self.gate = nn.Linear(d, mult * d, bias=False)
        self.act = AMSA(mult * d)
        self.down = nn.Linear(mult * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)) * torch.sigmoid(self.gate(x)))


FFNS: Dict[str, Callable[[int], nn.Module]] = {
    "gelu": GELUFFN,
    "swiglu": SwiGLUFFN,
    "amsa": AMSAFFN,
    # Same module, but with its memory / plasticity / confidence state actually
    # driven during training.  Without this the adaptive terms are multiplied by
    # zero buffers and AMSA is just a parameterised activation -- so measuring
    # only the row above would be testing the wrong thing.
    "amsa+state": AMSAFFN,
}


@torch.no_grad()
def drive_amsa_state(model: nn.Module) -> None:
    """Miniature of PlasticityController.sync_activations for the benchmark.

    Per output channel of `up`: confidence from the Fisher (squared gradient),
    plasticity as its complement, memory from the running activation.
    """
    for module in model.modules():
        if not isinstance(module, AMSAFFN):
            continue
        grad = module.up.weight.grad
        if grad is None:
            continue
        fisher = grad.pow(2).mean(dim=1)
        confidence = (fisher / fisher.quantile(0.95).clamp_min(1e-12)).clamp(max=1.0)
        activity = module.act.activity
        memory = (activity / activity.quantile(0.95).clamp_min(1e-12)).clamp(max=1.0)
        module.act.set_state(
            memory=memory, plasticity=1.0 - confidence, confidence=confidence
        )


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: nn.Module) -> None:
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)
        self.n2 = RMSNorm(d)
        self.ffn = ffn

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, kind: str, vocab: int, d: int = 128, layers: int = 4,
                 heads: int = 4, seq: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d, heads, FFNS[kind](d)) for _ in range(layers)]
        )
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        s = tokens.shape[1]
        mask = torch.ones(s, s, dtype=torch.bool, device=tokens.device).triu(1)
        x = self.embed(tokens) + self.pos[:s]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


def run(kind: str, corpus, steps: int, batch: int, lr: float, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    drive_state = kind == "amsa+state"
    model = TinyLM(kind.split("+")[0], vocab=260, seq=corpus.train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)

    start = time.time()
    losses: List[float] = []
    for step in range(steps):
        batch_tokens = corpus.batch(batch, generator=gen)
        inputs, targets = batch_tokens[:, :-1], batch_tokens[:, 1:]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(inputs).reshape(-1, 260), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if drive_state and step % 25 == 0:
            drive_amsa_state(model)
        opt.step()
        losses.append(float(loss.detach()))
    train_seconds = time.time() - start

    model.eval()
    with torch.no_grad():
        val = [
            float(F.cross_entropy(
                model(b[:, :-1]).reshape(-1, 260), b[:, 1:].reshape(-1)
            ))
            for b in corpus.val_batches(batch, limit=8)
        ]

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "first": sum(losses[:10]) / 10,
        "final": sum(losses[-20:]) / 20,
        "val": sum(val) / len(val),
        "seconds": train_seconds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    corpus = load_corpus("english", seq_len=args.seq)

    print("Activation comparison - identical model, identical data, only the")
    print("feed-forward nonlinearity differs.  Built by Mohammed Zahid.\n")
    print(f"  {args.steps} steps x batch {args.batch} x seq {args.seq}, "
          f"{args.seeds} seed(s)\n")

    print(f"{'ffn':<9}{'params':>10}{'start':>9}{'train':>9}{'val':>9}"
          f"{'bits/byte':>11}{'sec':>8}")
    print("-" * 65)

    for kind in FFNS:
        runs = [run(kind, corpus, args.steps, args.batch, args.lr, seed)
                for seed in range(args.seeds)]
        avg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        print(
            f"{kind:<9}{int(avg['params']):>10,}{avg['first']:>9.4f}"
            f"{avg['final']:>9.4f}{avg['val']:>9.4f}"
            f"{avg['val'] / math.log(2):>11.3f}{avg['seconds']:>8.1f}"
        )

    print("\n  Lower val is better.  Compare 'sec' too: an activation that wins")
    print("  by 0.01 nats while costing 30% more time is not a win.")


if __name__ == "__main__":
    main()
