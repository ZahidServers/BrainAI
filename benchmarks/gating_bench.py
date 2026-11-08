# BrainAI - built by Mohammed Zahid.
"""Does hard gating actually save anything?

The README claims the conscious pathway is "genuinely skipped" at inference
rather than masked.  That is a wall-clock claim and it deserves a wall-clock
measurement, because a gate that computes everything and multiplies by zero
looks identical from the outside.

Measures per-forward latency against the fraction of the batch that reaches the
conscious stack, by sweeping the gate threshold.

    python benchmarks/gating_bench.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai import BrainAI, BrainConfig  # noqa: E402

CKPT = Path("checkpoints/brainai_pretrained.pt")


@torch.no_grad()
def time_forward(model: BrainAI, tokens: torch.Tensor, repeats: int) -> tuple[float, float]:
    """Median seconds per forward, and the conscious fraction observed."""
    model(tokens=tokens)                       # warm up
    times, fractions = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = model(tokens=tokens)
        times.append(time.perf_counter() - t0)
        fractions.append(out.trace.conscious_fraction)
    return statistics.median(times), statistics.mean(fractions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model = (BrainAI.load(CKPT) if CKPT.exists()
             else BrainAI(BrainConfig.for_scale("nano", max_seq_len=args.seq)))
    model.eval()

    print("Gating latency - is the conscious pathway really skipped?")
    print("Built by Mohammed Zahid.\n")
    print(f"  batch {args.batch} x seq {args.seq}, median of {args.repeats} "
          f"forwards, CPU\n")

    tokens = torch.randint(0, 255, (args.batch, args.seq))

    print(f"{'threshold':>10}{'conscious %':>14}{'sec/forward':>14}"
          f"{'vs always-on':>14}")
    print("-" * 52)

    baseline = None
    for threshold in (0.0, 0.25, 0.5, 0.75, 1.01):
        model.cfg.conscious_gate_threshold = threshold
        secs, fraction = time_forward(model, tokens, args.repeats)
        if baseline is None:
            baseline = secs
        print(f"{threshold:>10.2f}{fraction * 100:>13.0f}%{secs:>14.4f}"
              f"{baseline / secs:>13.2f}x")

    model.cfg.conscious_gate_threshold = 0.5

    # Per-row gating: only the rows above threshold should pay.
    print("\n  per-row selection (threshold 1.01 = nothing deliberates):")
    model.cfg.conscious_gate_threshold = 1.01
    off, _ = time_forward(model, tokens, args.repeats)
    model.cfg.conscious_gate_threshold = 0.0
    on, _ = time_forward(model, tokens, args.repeats)
    print(f"    all rows deliberate : {on:.4f}s")
    print(f"    no rows deliberate  : {off:.4f}s")
    print(f"    saving              : {(1 - off / on) * 100:.1f}% of forward time")
    model.cfg.conscious_gate_threshold = 0.5

    print("\n  The conscious stack is the widest part of the model and is "
          "expert-routed,\n  so this is the largest single lever on inference "
          "cost.")


if __name__ == "__main__":
    main()
