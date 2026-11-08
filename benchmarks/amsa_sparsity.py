# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""AMSA vs SwiGLU on the axis AMSA was designed for: sparsity / energy.

Every previous AMSA test measured loss, which is not what a brain optimises. A
brain's advantage is doing the same work with fewer neurons firing, and AMSA's
mechanism -- the ``(1 - e^(-beta|z|))`` factor -- is explicitly a small-activation
suppressor. So the fair question is not "which reaches lower loss" but "at
comparable loss, which lights up less of the network."

Two models, identical init and identical batch stream, only the FFN differing.
After an equal training budget we measure, on held-out data:

  loss            so we know they are at a comparable operating point
  spike rate      the ALIF firing fraction -- in an SNN this *is* the energy,
                  the count of synaptic operations that actually happen
  ffn density     fraction of feed-forward hidden units that are meaningfully
                  active (|activation| > 1e-3); its complement is sparsity

Spike rate is the primary, unambiguous metric: it is binary and
activation-function-agnostic. FFN density is secondary and magnitude-sensitive
(the two nonlinearities output on different scales), so it is read as a
tendency, not a decimal to over-trust. Mean |activation| is printed alongside so
that scale context is visible.

    python benchmarks/amsa_sparsity.py --steps 800

A win here -- similar loss at a lower spike rate -- is a real advantage on the
efficiency axis AMSA was built for, and would be reported as such. A loss on
both loss and sparsity would close the question.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brainai import BrainAI, BrainConfig  # noqa: E402
from brainai.activation import GatedAMSA, SwiGLU  # noqa: E402
from brainai.blocks import mean_spike_rate  # noqa: E402
from brainai.corpus import load_multidomain  # noqa: E402

ACTIVE_EPS = 1e-3   # a hidden unit counts as "active" above this magnitude


class FFNProbe:
    """Captures feed-forward hidden-activation density via down-proj input hooks.

    The input to each FFN's ``down`` linear is the gated hidden activation for
    both AMSA (``act(up) * sigmoid(gate)``) and SwiGLU (``silu(gate) * up``), so
    hooking there compares the same quantity across the two designs.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.active = 0
        self.total = 0
        self.abs_sum = 0.0
        self._handles = []
        for module in model.modules():
            if isinstance(module, (GatedAMSA, SwiGLU)):
                self._handles.append(module.down.register_forward_pre_hook(self._hook))

    def _hook(self, module, inputs):
        x = inputs[0].detach()
        self.active += int((x.abs() > ACTIVE_EPS).sum())
        self.total += x.numel()
        self.abs_sum += float(x.abs().sum())

    @property
    def density(self) -> float:
        return self.active / max(self.total, 1)

    @property
    def mean_abs(self) -> float:
        return self.abs_sum / max(self.total, 1)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def train_arm(ffn: str, corpus, steps: int, batch: int, seq: int,
              lr: float, seed: int, stream_seed: int = 1234) -> Dict[str, float]:
    torch.manual_seed(seed)
    model = BrainAI(BrainConfig.for_scale("nano", max_seq_len=max(seq, 128),
                                          ffn=ffn))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Same stream for both arms within a seed; different across seeds.
    corpus._gen = torch.Generator().manual_seed(stream_seed)
    model.train()
    start = time.time()
    for _ in range(steps):
        _, tokens = corpus.sample(batch)
        opt.zero_grad(set_to_none=True)
        loss, out = model.lm_loss(tokens)
        wl, _ = model.world_loss_from(out)
        if wl is not None:
            loss = loss + 0.2 * wl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    train_s = time.time() - start

    # Held-out measurement pass, with the FFN probe attached.
    model.eval()
    probe = FFNProbe(model)
    eval_seed = torch.Generator().manual_seed(9999 + seed)
    losses: List[float] = []
    spikes: List[float] = []
    with torch.no_grad():
        for _ in range(12):
            idx_domain = corpus.names[int(torch.randint(len(corpus.names), (1,),
                                                        generator=eval_seed))]
            tokens = corpus.splits[idx_domain].batch(batch, generator=eval_seed)
            loss, _ = model.lm_loss(tokens, aux_weight=0.0, write_memory=False)
            losses.append(float(loss))
            spikes.append(mean_spike_rate(model))
    probe.remove()

    return {
        "loss": sum(losses) / len(losses),
        "spike_rate": sum(spikes) / len(spikes),
        "ffn_density": probe.density,
        "ffn_mean_abs": probe.mean_abs,
        "train_s": train_s,
    }


def _agg(values: List[float]) -> str:
    m = statistics.mean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{m:.4f}+-{sd:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print("AMSA vs SwiGLU on sparsity / energy -- the axis AMSA was built for.")
    print("Built by Mohammed Zahid Imtiyaz Wadiwale.\n")

    corpus = load_multidomain(seq_len=args.seq, verbose=False)
    print(f"corpus: {len(corpus.names)} domains, "
          f"{corpus.total_tokens / 1e6:.1f}M tokens")
    print(f"budget: {args.steps} steps x {args.seeds} seeds, "
          f"identical init + batch stream per arm within a seed\n")

    runs: Dict[str, Dict[str, List[float]]] = {
        ffn: {"loss": [], "spike_rate": [], "ffn_density": [], "ffn_mean_abs": []}
        for ffn in ("swiglu", "amsa")
    }
    wins = 0  # seeds where AMSA is sparser (spike rate) at comparable loss
    for seed in range(args.seeds):
        seed_res = {}
        for ffn in ("swiglu", "amsa"):
            print(f"  seed {seed}  ffn={ffn} ...", flush=True)
            r = train_arm(ffn, corpus, args.steps, args.batch, args.seq,
                          args.lr, seed, stream_seed=1234 + seed)
            seed_res[ffn] = r
            for k in runs[ffn]:
                runs[ffn][k].append(r[k])
        da = seed_res["amsa"]["spike_rate"] - seed_res["swiglu"]["spike_rate"]
        dl = seed_res["amsa"]["loss"] - seed_res["swiglu"]["loss"]
        sparser = da < 0 and dl < 0.03
        wins += int(sparser)
        print(f"    -> seed {seed}: dloss {dl:+.4f}, dspike {da:+.4f} "
              f"({'AMSA sparser' if sparser else 'not sparser'})\n")
        # Persist after every seed so a crash in the summary never discards a
        # multi-hour run (it did once: a missing import lost three seeds).
        Path("checkpoints").mkdir(exist_ok=True)
        Path("checkpoints/amsa_sparsity.json").write_text(json.dumps(runs, indent=1))

    print(f"{'ffn':<9}{'loss':>18}{'spike rate':>18}{'ffn active %':>18}")
    print("-" * 63)
    for ffn in ("swiglu", "amsa"):
        print(f"{ffn:<9}{_agg(runs[ffn]['loss']):>18}"
              f"{_agg(runs[ffn]['spike_rate']):>18}"
              f"{_agg([d*100 for d in runs[ffn]['ffn_density']]):>18}")

    a, s = runs["amsa"], runs["swiglu"]
    dloss = statistics.mean(a["loss"]) - statistics.mean(s["loss"])
    dspike = statistics.mean(a["spike_rate"]) - statistics.mean(s["spike_rate"])
    dspike_pct = abs(dspike) / max(statistics.mean(s["spike_rate"]), 1e-9) * 100
    dens_gap = (statistics.mean(s["ffn_density"]) -
                statistics.mean(a["ffn_density"])) * 100

    print(f"\n  across {args.seeds} seeds:")
    print(f"    loss:        AMSA {dloss:+.4f} nats "
          f"({'comparable' if abs(dloss) < 0.03 else 'not comparable'})")
    print(f"    spike rate:  AMSA {'sparser' if dspike < 0 else 'denser'} by "
          f"{dspike_pct:.1f}%, sparser in {wins}/{args.seeds} seeds")
    print(f"    ffn active:  AMSA lower by {dens_gap:.1f} points "
          f"(structural, from the activation shape)")


if __name__ == "__main__":
    main()
