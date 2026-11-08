# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""AMSA vs SwiGLU inside the real spiking model, on the 64-domain corpus.

Every earlier AMSA test was either static single-domain language modelling or an
English -> French continual pair, both on a tiny non-spiking proxy. This runs
the actual comparison people keep asking for: the *real* BrainAI nano model,
once with ``ffn="amsa"`` and once with ``ffn="swiglu"``, trained on the full
multi-domain corpus for an identical budget from an identical initialisation,
and evaluated per domain.

Fairness is enforced, not assumed:
  * same seed, so both models start from the same weights (only the FFN differs)
  * the corpus RNG is reseeded identically before each arm, so both see the
    *same* sequence of domains and batches
  * equal step budget, equal optimiser settings

    python benchmarks/amsa_multidomain.py --steps 800

This is the honest "does AMSA win this time" test. It is a modest budget on one
model size; read it as one more data point, not a verdict for all scales.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brainai import BrainAI, BrainConfig  # noqa: E402
from brainai.corpus import evaluate_domains, load_multidomain  # noqa: E402


def train_arm(ffn: str, corpus, steps: int, batch: int, seq: int,
              lr: float, seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)                      # identical init for both arms
    model = BrainAI(BrainConfig.for_scale("nano", max_seq_len=max(seq, 128),
                                          ffn=ffn))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Reseed the corpus RNG so this arm sees the same domain/batch stream.
    corpus._gen = torch.Generator().manual_seed(1234)
    model.train()
    start = time.time()
    for step in range(steps):
        _, tokens = corpus.sample(batch)
        opt.zero_grad(set_to_none=True)
        loss, out = model.lm_loss(tokens)
        wl, _ = model.world_loss_from(out)
        if wl is not None:
            loss = loss + 0.2 * wl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    elapsed = time.time() - start

    per_domain = evaluate_domains(model, corpus, batch_size=batch, batches=4)
    return {
        "params": model.parameter_count(),
        "seconds": elapsed,
        "mean": per_domain["mean"],
        "per_domain": per_domain,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print("AMSA vs SwiGLU in the real BrainAI model, on the 64-domain corpus.")
    print("Built by Mohammed Zahid Imtiyaz Wadiwale.\n")

    corpus = load_multidomain(seq_len=args.seq, verbose=False)
    print(f"corpus: {len(corpus.names)} domains, "
          f"{corpus.total_tokens / 1e6:.1f}M tokens")
    print(f"budget: {args.steps} steps x {args.batch} x {args.seq}, "
          f"identical init + batch stream per arm\n")

    results = {}
    for ffn in ("swiglu", "amsa"):
        print(f"  training ffn={ffn} ...", flush=True)
        results[ffn] = train_arm(ffn, corpus, args.steps, args.batch,
                                 args.seq, args.lr, args.seed)
        r = results[ffn]
        print(f"    params {r['params']:,}  mean {r['mean']:.4f} nats "
              f"({r['mean'] / math.log(2):.3f} bits/byte)  {r['seconds']/60:.1f} min\n")

    a, s = results["amsa"], results["swiglu"]
    delta = a["mean"] - s["mean"]
    print(f"{'ffn':<10}{'params':>12}{'mean loss':>12}{'bits/byte':>12}{'min':>8}")
    print("-" * 54)
    for ffn in ("swiglu", "amsa"):
        r = results[ffn]
        print(f"{ffn:<10}{r['params']:>12,}{r['mean']:>12.4f}"
              f"{r['mean']/math.log(2):>12.3f}{r['seconds']/60:>8.1f}")

    print()
    verdict = "AMSA wins" if delta < 0 else "SwiGLU wins"
    print(f"  delta (amsa - swiglu): {delta:+.4f} nats  ->  {verdict}")
    faster = "AMSA" if a["seconds"] < s["seconds"] else "SwiGLU"
    print(f"  faster: {faster} "
          f"({min(a['seconds'], s['seconds'])/max(a['seconds'],s['seconds']):.2f}x)")

    # Where each wins, per domain.
    wins_a = [d for d in a["per_domain"] if d != "mean"
              and a["per_domain"][d] < s["per_domain"][d]]
    print(f"  AMSA lower on {len(wins_a)}/{len(a['per_domain'])-1} domains")
    print("\n  One model size, modest budget. A data point, not a verdict for "
          "all scales.")


if __name__ == "__main__":
    main()
