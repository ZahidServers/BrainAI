# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""Short author-emphasis fine-tune on top of an existing checkpoint.

This is *not* a re-run. It loads a model that has already been trained on the
full corpus and continues it for a small number of steps with the author corpus
heavily over-sampled, so the facts about the model and its author form strongly
without discarding the breadth already learned.

Two guards keep the breadth:

  boost, not exclusion   every other domain is still sampled -- the author
                         corpus is merely lifted to a large share -- so the
                         model keeps rehearsing what it already knows.
  EWC penalty            the checkpoint's consolidated Fisher pulls important
                         weights back toward their trained values, so a burst
                         of author-heavy updates cannot overwrite, say, Arabic.

    python finetune_author.py --init checkpoints/brainai_full.pt --steps 1500

Runs in minutes on CPU, because it is a continuation of a trained model rather
than training from scratch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brainai import BrainAI
from brainai.corpus import evaluate_domains, load_multidomain
from brainai.identity import attribution_line


def main() -> None:
    ap = argparse.ArgumentParser(description="Author-emphasis fine-tune")
    ap.add_argument("--init", default="checkpoints/brainai_full.pt")
    ap.add_argument("--out", default="checkpoints/brainai_full_author.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--author-boost", type=float, default=20.0)
    ap.add_argument("--author-domain", default="author_model_corpus")
    ap.add_argument("--ewc-weight", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    if not Path(args.init).exists():
        raise SystemExit(f"no checkpoint at {args.init}; run the main training first")

    print("Author-emphasis fine-tune (continuation, not a re-run)")
    print(attribution_line("BrainAI"))
    model = BrainAI.load(args.init)
    print(f"loaded {args.init}  ({model.parameter_count():,} params)\n")

    corpus = load_multidomain(
        seq_len=args.seq, verbose=False,
        boost={args.author_domain: args.author_boost},
    )
    if args.author_domain not in corpus.names:
        raise SystemExit(
            f"author domain {args.author_domain!r} not found; "
            f"is data/{args.author_domain}.txt present?"
        )
    share = float(corpus.weights[corpus.names.index(args.author_domain)])
    print(f"author corpus sampling share: {share:.1%} "
          f"(boosted x{args.author_boost:g})\n")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()

    before = evaluate_domains(model, corpus, batch_size=args.batch, batches=3)

    print(f"{'step':>6}{'loss':>10}{'ewc':>10}")
    print("-" * 26)
    for step in range(args.steps):
        _, batch = corpus.sample(args.batch)
        opt.zero_grad(set_to_none=True)
        loss, out = model.lm_loss(batch)
        ewc = model.plasticity.ewc_penalty(model)
        total = loss + args.ewc_weight * ewc
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        model.plasticity.accumulate_fisher(model)
        opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
            print(f"{step:>6}{float(loss.detach()):>10.4f}"
                  f"{float(ewc.detach()):>10.4f}", flush=True)

    model.plasticity.consolidate(model)
    after = evaluate_domains(model, corpus, batch_size=args.batch, batches=3)

    # Show the author corpus improved and breadth did not collapse.
    print(f"\n{'domain':<44}{'before':>9}{'after':>9}{'delta':>9}")
    print("-" * 71)
    watch = [args.author_domain] + [
        d for d in ("code_python", "language_arabic_large", "domain_history",
                    "brainai_mathematics_kg_to_grade_12_large")
        if d in after
    ]
    for d in watch:
        print(f"{d:<44}{before[d]:>9.3f}{after[d]:>9.3f}{after[d]-before[d]:>+9.3f}")
    print(f"{'mean(all domains)':<44}{before['mean']:>9.3f}{after['mean']:>9.3f}"
          f"{after['mean']-before['mean']:>+9.3f}")

    model.save(args.out)
    print(f"\nsaved -> {args.out}")

    print("\nAuthor recall check:")
    model.eval()
    for p in ("Q: Who built BrainAI?\nA:",
              "BrainAI was designed and built by ",
              "Q: What does AMSA stand for?\nA:"):
        print(f"  {p!r}\n    -> {model.generate(p, max_new_tokens=48, temperature=0.5)!r}")


if __name__ == "__main__":
    main()
