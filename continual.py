# BrainAI - built by Mohammed Zahid.
"""Does BrainAI forget?  The measurement.

Takes the English-pretrained checkpoint, trains it on French three different
ways, and measures what happened to English.

    naive   plain next-token loss                       (the control)
    ewc     + elastic weight consolidation penalty
    full    + EWC + experience replay + one sleep cycle (the architecture)

Every arm starts from identical weights and sees the same number of French
sequences over the same number of optimizer steps.  The ``full`` arm
additionally replays English from its buffer, which costs ~1.5x compute per
step and the memory to hold the buffer -- that cost is the price of the method
and is reported alongside the result, not hidden.

    python continual.py --steps 120
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List

import torch

from brainai import BrainAI
from brainai.data import evaluate, load_corpus
from brainai.identity import attribution_line


def run_arm(
    arm: str,
    ckpt: str,
    french,
    english,
    steps: int,
    batch: int,
    replay_batch: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    """Train one arm on French and report what it cost English."""
    torch.manual_seed(seed)
    model = BrainAI.load(ckpt)

    before_en = evaluate(model, english)
    before_fr = evaluate(model, french)

    use_ewc = arm in ("ewc", "full")
    use_replay = arm == "full"

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    start = time.time()
    ewc_total = 0.0

    for step in range(steps):
        new_batch = french.batch(batch, generator=gen)

        if use_replay:
            old = model.replay.sample(replay_batch)
            batch_tokens = (
                torch.cat([new_batch, old], dim=0) if old is not None else new_batch
            )
        else:
            batch_tokens = new_batch

        opt.zero_grad(set_to_none=True)
        loss, _ = model.lm_loss(batch_tokens)

        if use_ewc:
            penalty = model.plasticity.ewc_penalty(model)
            ewc_total += float(penalty.detach())
            loss = loss + penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if use_ewc:
            model.plasticity.accumulate_fisher(model)
        opt.step()

        # New material also goes into the buffer, tagged as a second task, so
        # the buffer stays balanced instead of becoming all-English.
        if use_replay and step % 10 == 0:
            model.replay.add(new_batch, importance=1.0, task_id=1)

    slept = {}
    if arm == "full":
        slept = model.sleep(cycles=10, batch_size=4, lr=1e-5)

    elapsed = time.time() - start
    after_en = evaluate(model, english)
    after_fr = evaluate(model, french)

    model.save(f"checkpoints/continual_{arm}.pt")

    return {
        "arm": arm,
        "before_en": before_en,
        "after_en": after_en,
        "forgetting": after_en - before_en,
        "before_fr": before_fr,
        "after_fr": after_fr,
        "learned_fr": before_fr - after_fr,
        "seconds": elapsed,
        "ewc_mean": ewc_total / max(steps, 1),
        "sleep_replay_last": slept.get("replay_loss_last", float("nan")),
        "protected": model.plasticity.report()["protected_fraction"],
        "spike_rate": model.energy_report()["mean_spike_rate"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="BrainAI continual-learning benchmark")
    ap.add_argument("--ckpt", default="checkpoints/brainai_pretrained.pt")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--replay-batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arms", default="naive,ewc,full")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    english = load_corpus("english", seq_len=args.seq)
    french = load_corpus("french", seq_len=args.seq)

    print("BrainAI continual-learning benchmark")
    print(attribution_line("BrainAI"))
    print()
    print(f"  task A (pretrained): {english.description}")
    print(f"  task B (new):        {french.description}")
    print(f"  {args.steps} steps x {args.batch} sequences of {args.seq} bytes\n")

    results: List[Dict[str, float]] = []
    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"  running arm: {arm} ...", flush=True)
        results.append(
            run_arm(
                arm, args.ckpt, french, english,
                args.steps, args.batch, args.replay_batch, args.lr, args.seed,
            )
        )

    ln2 = math.log(2)
    print()
    print(f"{'arm':<8}{'EN before':>11}{'EN after':>10}{'forgot':>9}"
          f"{'FR before':>11}{'FR after':>10}{'learned':>9}{'both':>8}{'sec':>7}")
    print("-" * 83)
    for r in results:
        r["both"] = 0.5 * (r["after_en"] + r["after_fr"])
        print(
            f"{r['arm']:<8}{r['before_en']:>11.4f}{r['after_en']:>10.4f}"
            f"{r['forgetting']:>+9.4f}{r['before_fr']:>11.4f}{r['after_fr']:>10.4f}"
            f"{r['learned_fr']:>+9.4f}{r['both']:>8.4f}{r['seconds']:>7.0f}"
        )
    print("\n  'both' = mean final loss over the two tasks. Read it before")
    print("  celebrating the forgetting column: protecting old weights also")
    print("  slows down learning the new task, and 'both' is where that shows.")

    base = next((r for r in results if r["arm"] == "naive"), None)
    print()
    for r in results:
        bits = r["forgetting"] / ln2
        line = (
            f"  {r['arm']:<6} forgetting {r['forgetting']:+.4f} nats "
            f"({bits:+.3f} bits/byte)"
        )
        if base is not None and r is not base and base["forgetting"] > 1e-6:
            reduction = 1.0 - r["forgetting"] / base["forgetting"]
            line += f"   -> {reduction:.0%} less than naive"
        print(line)

    print()
    for r in results:
        print(f"  {r['arm']:<6} protected weights {r['protected']:.1%}   "
              f"spike rate {r['spike_rate']:.3f}   "
              f"mean EWC penalty {r['ewc_mean']:.4f}")

    print(f"\n{attribution_line('BrainAI')}")


if __name__ == "__main__":
    main()
