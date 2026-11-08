# BrainAI - built by Mohammed Zahid.
"""Train across twelve domains, and measure each one separately.

    python train_multidomain.py --steps 12000

The single-domain run in ``train.py`` proves the model learns. It cannot say
anything about breadth. This trains on code, physics, mathematics, biology,
chemistry, science, history, economics, psychology, philosophy, literature and
French, and reports **held-out loss per domain** throughout.

Per-domain reporting is the point. One averaged number cannot distinguish a
model that is uniformly mediocre from one that learned Python and forgot
chemistry, and those two failure modes want opposite fixes.

A history file is written alongside the checkpoint so the per-domain curves can
be inspected after the fact rather than scraped from a log.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch

from brainai import BrainAI, BrainConfig
from brainai.corpus import evaluate_domains, load_multidomain
from brainai.identity import attribution_line


def cosine_lr(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-domain BrainAI training")
    ap.add_argument("--scale", default="nano")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--world-weight", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="domain sampling temperature; 0=balanced, 1=proportional")
    ap.add_argument("--init", default=None, help="warm-start from a checkpoint")
    ap.add_argument("--out", default="checkpoints/brainai_multidomain.pt")
    ap.add_argument("--history", default="checkpoints/multidomain_history.json")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    print("BrainAI multi-domain training")
    print(attribution_line("BrainAI"))
    print()

    corpus = load_multidomain(seq_len=args.seq, temperature=args.temperature,
                              seed=args.seed)
    print()
    print(corpus.describe())
    print()

    if args.init and Path(args.init).exists():
        model = BrainAI.load(args.init)
        print(f"warm-started from {args.init}")
    else:
        model = BrainAI(BrainConfig.for_scale(args.scale,
                                              max_seq_len=max(args.seq, 128)))
    print(f"parameters: {model.parameter_count():,}")

    tokens_per_step = args.batch * args.seq
    print(f"budget: {args.steps:,} steps x {tokens_per_step} tokens = "
          f"{args.steps * tokens_per_step / 1e6:.1f}M tokens "
          f"({args.steps * tokens_per_step / corpus.total_tokens:.2f} epochs)\n")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    model.train()

    history: List[Dict] = []
    domain_steps: Dict[str, int] = {name: 0 for name in corpus.names}
    start = time.time()
    running = 0.0

    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr

        domain, batch = corpus.sample(args.batch)
        domain_steps[domain] += 1

        opt.zero_grad(set_to_none=True)
        loss, out = model.lm_loss(batch)
        world_loss, _ = model.world_loss_from(out)
        if world_loss is not None:
            loss = loss + args.world_weight * world_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        model.plasticity.accumulate_fisher(model)
        opt.step()

        running += float(loss.detach())

        # Keep the replay buffer stocked across domains, so a later sleep or
        # continual phase rehearses breadth rather than whichever domain
        # happened to come last.
        if step % 25 == 0:
            model.replay.add(batch, importance=1.0,
                             task_id=corpus.names.index(domain))

        if step % args.eval_every == 0 or step == args.steps - 1:
            per_domain = evaluate_domains(model, corpus, batch_size=args.batch)
            elapsed = time.time() - start
            record = {
                "step": step,
                "train_loss": running / max(1, min(args.eval_every, step + 1)),
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
                "tokens": step * tokens_per_step,
                "spike_rate": out.trace.extras["spike_rate"],
                "per_domain": per_domain,
            }
            history.append(record)
            running = 0.0

            worst = max((k for k in per_domain if k != "mean"),
                        key=lambda k: per_domain[k])
            best = min((k for k in per_domain if k != "mean"),
                       key=lambda k: per_domain[k])
            print(
                f"step {step:>6}  mean {per_domain['mean']:.4f}  "
                f"({per_domain['mean'] / math.log(2):.2f} bits/byte)  "
                f"best {best}={per_domain[best]:.3f}  "
                f"worst {worst}={per_domain[worst]:.3f}  "
                f"{elapsed / 60:.0f}m",
                flush=True,
            )
            Path(args.history).parent.mkdir(parents=True, exist_ok=True)
            Path(args.history).write_text(json.dumps(history, indent=1))

        if step and step % args.save_every == 0:
            model.save(args.out)

    stats = model.plasticity.consolidate(model)
    final = evaluate_domains(model, corpus, batch_size=args.batch, batches=8)

    print(f"\nconsolidated: {stats['protected_fraction']:.1%} of weights protected")
    print(f"\n{'domain':<14}{'loss':>9}{'bits/byte':>12}{'train steps':>13}")
    print("-" * 48)
    for name in sorted(corpus.names, key=lambda n: final[n]):
        print(f"{name:<14}{final[name]:>9.4f}{final[name] / math.log(2):>12.3f}"
              f"{domain_steps[name]:>13,}")
    print("-" * 48)
    print(f"{'mean':<14}{final['mean']:>9.4f}{final['mean'] / math.log(2):>12.3f}")

    model.save(args.out)
    history.append({"final": final, "domain_steps": domain_steps})
    Path(args.history).write_text(json.dumps(history, indent=1))
    print(f"\nsaved -> {args.out}")
    print(f"history -> {args.history}")
    print(f"\n{model.status()}")


if __name__ == "__main__":
    main()
