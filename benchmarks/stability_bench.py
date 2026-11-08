# BrainAI - built by Mohammed Zahid.
"""Does online learning stay stable, or does it quietly destroy the model?

`learn_online` updates weights during inference.  The README asserts that is
safe because of plasticity gating, a trust region and rollback.  Six updates in
a demo does not test that.  This runs hundreds against a live stream and tracks
what actually matters:

    old-task loss   is the pretrained ability still there
    weight drift    how far the parameters have travelled from the anchor
    rollback rate   how often the guard fires
    spike rate      has the network gone dense (a classic SNN failure)

The control arm disables plasticity gating and the trust region, so "the guards
are doing something" is measured rather than assumed.

    python benchmarks/stability_bench.py --updates 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai import BrainAI, BrainConfig  # noqa: E402
from brainai.data import evaluate, load_corpus  # noqa: E402

CKPT = Path("checkpoints/brainai_pretrained.pt")

STREAM = [
    "the workshop key is kept in the third drawer",
    "the north door is locked after eighteen hundred hours",
    "an M10 bolt is torqued to forty newton metres",
    "the blue crate holds spare servo controllers",
    "battery pack two reports twelve percent capacity",
    "the calibration jig lives on the upper shelf",
    "coolant is changed every two hundred hours",
    "the left elbow encoder drifts when it is cold",
]


def anchor_snapshot(model: BrainAI) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def drift_from(model: BrainAI, anchor: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if name in anchor:
            total += float((p.detach() - anchor[name]).pow(2).sum())
    return total ** 0.5


def run(guarded: bool, updates: int, every: int, seq: int,
        english, threads: int) -> List[Dict[str, float]]:
    torch.manual_seed(0)
    torch.set_num_threads(threads)

    model = (BrainAI.load(CKPT) if CKPT.exists()
             else BrainAI(BrainConfig.for_scale("nano", max_seq_len=seq)))
    model.eval()

    if not guarded:
        # Control: every weight fully plastic, no trust region, no rollback.
        for _, key in model.plasticity._entries:
            getattr(model.plasticity, f"plast__{key}").fill_(1.0)

    anchor = anchor_snapshot(model)
    tok = model.tokenizer
    history: List[Dict[str, float]] = []
    rollbacks = 0
    start = time.time()

    for step in range(updates):
        text = STREAM[step % len(STREAM)]
        tokens = tok.encode_batch([text, text], seq)
        stats = model.learn_online(
            tokens,
            reward=1.0,
            task_id=1,
            verify_update=guarded,
            # Control gets an effectively unbounded trust region, so the
            # comparison isolates the guards rather than the learning rate.
            max_relative_norm=0.01 if guarded else 1e6,
        )
        rollbacks += int(stats.get("rolled_back", 0.0))

        if step % every == 0 or step == updates - 1:
            val = evaluate(model, english, batch_size=4, batches=3)
            history.append({
                "step": float(step),
                "old_task_loss": val,
                "drift": drift_from(model, anchor),
                "rollback_rate": rollbacks / (step + 1),
                "spike_rate": model.energy_report()["mean_spike_rate"],
                "new_fact_loss": stats["loss"],
            })
    for row in history:
        row["seconds"] = time.time() - start
    return history


def report(name: str, history: List[Dict[str, float]]) -> None:
    print(f"\n{name}")
    print(f"{'update':>8}{'old-task loss':>15}{'new-fact loss':>15}"
          f"{'drift':>10}{'rollback':>10}{'spike':>8}")
    print("-" * 66)
    for row in history:
        print(f"{int(row['step']):>8}{row['old_task_loss']:>15.4f}"
              f"{row['new_fact_loss']:>15.4f}{row['drift']:>10.4f}"
              f"{row['rollback_rate'] * 100:>9.0f}%{row['spike_rate']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--control", action="store_true",
                    help="also run the ungated control arm")
    args = ap.parse_args()

    english = load_corpus("english", seq_len=args.seq)

    print("Online-learning stability.  Built by Mohammed Zahid.")
    print(f"  {args.updates} inference-time updates, seq {args.seq}, CPU")

    guarded = run(True, args.updates, args.every, args.seq, english, args.threads)
    report("guarded (plasticity + trust region + rollback)", guarded)

    first, last = guarded[0], guarded[-1]
    delta = last["old_task_loss"] - first["old_task_loss"]
    print(f"\n  old-task loss moved {delta:+.4f} nats over {args.updates} updates")
    print(f"  final drift {last['drift']:.4f}, "
          f"spike rate {first['spike_rate']:.3f} -> {last['spike_rate']:.3f}")
    print(f"  verdict: {'STABLE' if abs(delta) < 0.25 else 'DEGRADING'}")

    if args.control:
        ungated = run(False, args.updates, args.every, args.seq, english,
                      args.threads)
        report("control (no plasticity gating, no rollback)", ungated)
        c_delta = ungated[-1]["old_task_loss"] - ungated[0]["old_task_loss"]
        print(f"\n  control old-task loss moved {c_delta:+.4f} nats "
              f"(guarded: {delta:+.4f})")
        print(f"  control drift {ungated[-1]['drift']:.4f} "
              f"(guarded: {last['drift']:.4f})")


if __name__ == "__main__":
    main()
