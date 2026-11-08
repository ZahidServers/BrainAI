# BrainAI - built by Mohammed Zahid.
"""AMSA on the problem it was actually designed for.

``activation_bench.py`` measures static language-model quality and AMSA loses
there, clearly, on both loss and speed.  But that benchmark cannot test AMSA's
hypothesis, because the thing that is supposed to make it different --
per-channel threshold and gain driven by plasticity, confidence and memory
state -- is inert in a single-task run.  Those buffers only carry information
once there is a *previous task* whose importance has been estimated.

So this is the real test.  Train on English, consolidate, train on French, and
measure how much English survived:

    swiglu        the strong baseline
    amsa          identical wiring, state buffers left at zero
    amsa+state    plasticity and confidence driven from the task-A Fisher

If AMSA's adaptive state is worth anything, ``amsa+state`` forgets less than
both.  If it does not, AMSA has no demonstrated advantage anywhere and the
honest move is to use SwiGLU.

    python benchmarks/amsa_continual.py --steps 250 --seeds 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai.activation import AMSA  # noqa: E402
from brainai.data import load_corpus  # noqa: E402

from activation_bench import AMSAFFN, TinyLM  # noqa: E402


@torch.no_grad()
def drive_state(model: torch.nn.Module, fisher: Dict[str, torch.Tensor]) -> None:
    """Write task-A importance into every AMSA's state buffers.

    ``confidence`` is the normalised Fisher of the channel -- how much task A
    depended on it.  ``plasticity`` is its complement.  This is the mechanism
    under test: channels that mattered before should sharpen and resist being
    moved, channels that did not should stay free for the new task.
    """
    for name, module in model.named_modules():
        if not isinstance(module, AMSAFFN):
            continue
        key = f"{name}.up.weight"
        if key not in fisher:
            continue
        rows = fisher[key].mean(dim=1)
        scale = rows.quantile(0.95).clamp_min(1e-12)
        confidence = (rows / scale).clamp(max=1.0)
        activity = module.act.activity
        memory = (activity / activity.quantile(0.95).clamp_min(1e-12)).clamp(max=1.0)
        module.act.set_state(
            memory=memory, plasticity=1.0 - confidence, confidence=confidence
        )


def train(
    model: torch.nn.Module,
    corpus,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    fisher: Dict[str, torch.Tensor] | None = None,
    anchors: Dict[str, torch.Tensor] | None = None,
    ewc_lambda: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Train one task; return the diagonal Fisher accumulated over it."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)
    accumulated: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p) for n, p in model.named_parameters()
    }

    model.train()
    for _ in range(steps):
        tokens = corpus.batch(batch, generator=gen)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(inputs).reshape(-1, 260), targets.reshape(-1)
        )
        if ewc_lambda and fisher is not None and anchors is not None:
            penalty = sum(
                (fisher[n] * (p - anchors[n]).pow(2)).sum()
                for n, p in model.named_parameters() if n in fisher
            )
            loss = loss + ewc_lambda * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    accumulated[n].mul_(0.95).addcmul_(p.grad, p.grad, value=0.05)
        opt.step()
    return accumulated


@torch.no_grad()
def evaluate(model: torch.nn.Module, corpus, batch: int = 8) -> float:
    model.eval()
    losses = [
        float(F.cross_entropy(
            model(b[:, :-1]).reshape(-1, 260), b[:, 1:].reshape(-1)
        ))
        for b in corpus.val_batches(batch, limit=6)
    ]
    return sum(losses) / len(losses)


def run_arm(
    kind: str,
    english,
    french,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    width: int = 128,
    layers: int = 4,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    use_state = kind == "amsa+state"
    model = TinyLM(
        kind.split("+")[0], vocab=260, d=width, layers=layers,
        heads=max(2, width // 32), seq=english.train.shape[1],
    )

    fisher = train(model, english, steps, batch, lr, seed)
    before_en = evaluate(model, english)
    before_fr = evaluate(model, french)

    if use_state:
        drive_state(model, fisher)

    anchors = {n: p.detach().clone() for n, p in model.named_parameters()}
    train(model, french, steps, batch, lr, seed + 1)

    after_en = evaluate(model, english)
    after_fr = evaluate(model, french)
    drift = sum(
        float((p.detach() - anchors[n]).pow(2).sum())
        for n, p in model.named_parameters()
    ) ** 0.5

    return {
        "before_en": before_en,
        "after_en": after_en,
        "forgetting": after_en - before_en,
        "learned_fr": before_fr - after_fr,
        "both": 0.5 * (after_en + after_fr),
        "drift": drift,
    }


def sweep(english, french, args) -> None:
    """Does AMSA's disadvantage shrink with scale, or is it structural?

    The obvious defence of a losing result is "the model was too small and the
    run too short".  That is a real possibility and it is also testable, so it
    should be tested rather than asserted.  If the gap narrows as width, depth
    and training length grow, the defence holds and AMSA deserves a bigger
    experiment.  If it is flat or widening, it does not.
    """
    print("\n\nScale sweep - is the gap shrinking, flat, or widening?\n")
    print(f"{'width':>7}{'layers':>8}{'steps':>8}{'swiglu':>10}{'amsa':>10}"
          f"{'amsa+st':>10}{'gap':>9}{'gap %':>8}")
    print("-" * 70)

    grid = [
        (96, 3, args.steps // 2),
        (128, 4, args.steps),
        (192, 4, args.steps),
        (256, 6, args.steps * 2),
    ]

    trend = []
    for width, layers, steps in grid:
        row = {}
        for kind in ("swiglu", "amsa", "amsa+state"):
            runs = [
                run_arm(kind, english, french, steps, args.batch, args.lr, s,
                        width=width, layers=layers)
                for s in range(args.seeds)
            ]
            row[kind] = statistics.mean(r["forgetting"] for r in runs)
        gap = row["amsa"] - row["swiglu"]
        relative = 100.0 * gap / abs(row["swiglu"]) if row["swiglu"] else 0.0
        trend.append(relative)
        print(f"{width:>7}{layers:>8}{steps:>8}{row['swiglu']:>+10.4f}"
              f"{row['amsa']:>+10.4f}{row['amsa+state']:>+10.4f}"
              f"{gap:>+9.4f}{relative:>7.1f}%")

    direction = (
        "shrinking with scale" if trend[-1] < trend[0] - 0.2
        else "widening with scale" if trend[-1] > trend[0] + 0.2
        else "flat across scale"
    )
    print(f"\n  relative gap {trend[0]:.1f}% -> {trend[-1]:.1f}%  ({direction})")
    print("  Shrinking would support 'too small to show its value'.")
    print("  Flat or widening would not, and this range is still small either way.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="test whether the gap is scale-dependent")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    english = load_corpus("english", seq_len=args.seq)
    french = load_corpus("french", seq_len=args.seq)

    print("AMSA on continual learning - the hypothesis it was built for.")
    print("Built by Mohammed Zahid.\n")
    print(f"  English -> French, {args.steps} steps each, "
          f"{args.seeds} seeds, identical wiring\n")

    print(f"{'ffn':<12}{'EN forgot':>12}{'FR learned':>12}{'both':>9}"
          f"{'drift':>9}")
    print("-" * 54)

    results: Dict[str, Dict[str, float]] = {}
    for kind in ("swiglu", "amsa", "amsa+state"):
        runs = [
            run_arm(kind, english, french, args.steps, args.batch, args.lr, s)
            for s in range(args.seeds)
        ]
        avg = {k: statistics.mean(r[k] for r in runs) for k in runs[0]}
        spread = statistics.pstdev([r["forgetting"] for r in runs])
        results[kind] = dict(avg, spread=spread)
        print(f"{kind:<12}{avg['forgetting']:>+11.4f}{avg['learned_fr']:>+12.4f}"
              f"{avg['both']:>9.4f}{avg['drift']:>9.3f}")

    base = results["swiglu"]
    print(f"\n  seed-to-seed spread on forgetting: "
          + "  ".join(f"{k} +-{v['spread']:.4f}" for k, v in results.items()))
    print()
    for kind in ("amsa", "amsa+state"):
        delta = results[kind]["forgetting"] - base["forgetting"]
        verdict = "better than swiglu" if delta < 0 else "worse than swiglu"
        significant = abs(delta) > 2 * max(base["spread"], results[kind]["spread"])
        print(f"  {kind:<12}{delta:+.4f} nats vs swiglu  ({verdict}"
              f"{', outside seed noise' if significant else ', within seed noise'})")

    print("\n  The question is whether amsa+state beats plain amsa. If driving")
    print("  the state changes nothing here either, AMSA's adaptive machinery")
    print("  has no demonstrated benefit on any benchmark in this repo.")

    if args.sweep:
        sweep(english, french, args)


if __name__ == "__main__":
    main()
