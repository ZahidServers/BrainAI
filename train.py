# BrainAI - built by Mohammed Zahid.
"""Pretrain BrainAI on one corpus.

    python train.py --scale nano --steps 400 --batch 8 --seq 128

Writes a checkpoint to ``checkpoints/`` that ``continual.py`` and ``demo.py``
both load.  Every 50 steps it reports loss, held-out loss, and spike rate --
the last one matters as much as the loss, because a spiking model that has
quietly become dense has thrown away the thing that makes it worth building.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from brainai import BrainAI, BrainConfig
from brainai.data import evaluate, load_corpus
from brainai.identity import attribution_line


def cosine_lr(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretrain BrainAI")
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--task", default="english")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--world-weight", type=float, default=0.2,
                    help="weight on the self-supervised dynamics loss")
    ap.add_argument("--name", default="BrainAI", help="deployment display name")
    ap.add_argument("--out", default="checkpoints/brainai_pretrained.pt")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    cfg = BrainConfig.for_scale(args.scale, max_seq_len=max(args.seq, 128))
    model = BrainAI(cfg).rename(args.name)

    print(model.describe())
    print(f"\ntrainable parameters: {model.parameter_count():,}\n")

    corpus = load_corpus(args.task, seq_len=args.seq)
    print(f"corpus: {corpus}  [{corpus.description}]\n")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    model.train()
    gen = torch.Generator().manual_seed(args.seed)
    start = time.time()
    history = []

    print(f"{'step':>6}{'loss':>9}{'val':>9}{'world':>9}{'spike':>8}"
          f"{'lr':>10}{'s/step':>9}")
    print("-" * 60)

    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr

        batch = corpus.batch(args.batch, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss, out = model.lm_loss(batch)

        # Self-supervised dynamics, from the same forward pass.
        world_loss, world_parts = model.world_loss_from(out)
        if world_loss is not None:
            loss = loss + args.world_weight * world_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        model.plasticity.accumulate_fisher(model)
        opt.step()

        history.append(float(loss.detach()))

        if step % 50 == 0 or step == args.steps - 1:
            val = evaluate(model, corpus, batch_size=args.batch, batches=4)
            elapsed = (time.time() - start) / (step + 1)
            print(
                f"{step:>6}{float(loss.detach()):>9.4f}{val:>9.4f}"
                f"{world_parts.get('dynamics', float('nan')):>9.4f}"
                f"{out.trace.extras['spike_rate']:>8.3f}{lr:>10.2e}{elapsed:>9.2f}"
            )
            # Keep the replay buffer stocked as we go; sleep and the continual
            # phase both draw from it.
            model.replay.add(batch, importance=1.0, task_id=0)

    # Consolidate: anchor what was learned and compute per-weight plasticity.
    stats = model.plasticity.consolidate(model)
    print(
        f"\nconsolidated: {stats['protected_fraction']:.1%} of "
        f"{int(stats['total_weights']):,} weights are now protected"
    )

    final_val = evaluate(model, corpus, batch_size=args.batch, batches=8)
    print(f"final held-out loss: {final_val:.4f} "
          f"({final_val / math.log(2):.3f} bits/byte)")

    out_path = Path(args.out)
    model.save(out_path)
    print(f"\nsaved -> {out_path}")
    print(f"\n{model.status()}")
    print(f"\n{attribution_line(model.name)}")


if __name__ == "__main__":
    main()
