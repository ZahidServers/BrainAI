# BrainAI - built by Mohammed Zahid.
"""Distil a teacher into BrainAI, in two phases.

One constraint determines the whole design: **you cannot KL-match two models
with different tokenizers.** BrainAI's byte tokenizer has 260 symbols; a
typical BPE vocabulary has ~150k. Their output distributions live in different
spaces, so the usual logit objective is undefined between them.

    generate   teacher writes a corpus, once, cached to disk
    train      student trains on it, for as many epochs as you like

The split matters. Generation is the expensive half -- on a GPU a 0.5B teacher
produces maybe 2k tokens/s, on CPU nearer 5. Calling the teacher inside the
training loop, as an earlier version of this script did, regenerates the same
distribution every step and throws away the ability to do a second epoch. The
corpus is written as JSONL so a run can be resumed, inspected, or shared.

    python distill.py generate --teacher Qwen/Qwen2.5-0.5B --samples 20000
    python distill.py train    --steps 20000
    python distill.py generate --dry-run --samples 200      # no GPU needed

**On this hardware, do not use this.** Measured: 888 tok/s of real text
training versus ~5-10 tok/s of CPU teacher generation. Reading Project
Gutenberg and the Python standard library is ~100x more data per hour than
generating from a 0.5B teacher. Distillation earns its place on a GPU; see
``train_multidomain.py`` for what to run here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from brainai import BrainAI, BrainConfig
from brainai.corpus import evaluate_domains, load_multidomain
from brainai.identity import attribution_line
from brainai.prompts import DOMAIN_PROMPTS, sample as sample_prompts

DEFAULT_CORPUS = Path("data/teacher_corpus.jsonl")


# ---------------------------------------------------------------- teachers
class StubTeacher:
    """Offline stand-in, so the pipeline can be exercised without a download."""

    name = "stub-teacher"
    vocab_size = 260

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        filler = (
            " and the resulting value is computed from the preceding terms, "
            "which follow directly from the definition given above."
        )
        return prompt + filler * (1 + max_new_tokens // 96)


class HFTeacher:
    """Frozen Hugging Face causal LM. Weights are never updated."""

    def __init__(self, name: str, device: str = "cpu", dtype: str = "float32") -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "distillation needs `transformers`:\n"
                "    pip install transformers accelerate\n"
                "or pass --dry-run to exercise the pipeline offline."
            ) from exc

        self.name = name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=getattr(torch, dtype)
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.8, top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=tokens.to(self.device)).logits.float()


def build_teacher(args) -> object:
    return StubTeacher() if args.dry_run else HFTeacher(args.teacher, args.device)


# --------------------------------------------------------------- generate
def phase_generate(args) -> None:
    teacher = build_teacher(args)
    path = Path(args.corpus)
    path.parent.mkdir(parents=True, exist_ok=True)

    already = 0
    if path.exists() and not args.overwrite:
        already = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"resuming: {already:,} samples already in {path}")

    wanted = sample_prompts(args.samples, args.domains.split(",") if args.domains
                            else None, seed=args.seed)
    todo = wanted[already:]
    if not todo:
        print("nothing to generate")
        return

    print(f"teacher: {teacher.name}")
    print(f"generating {len(todo):,} samples across "
          f"{len({d for d, _ in wanted})} domains -> {path}\n")

    start = time.time()
    chars = 0
    with path.open("a", encoding="utf-8") as fh:
        for i, (domain, prompt) in enumerate(todo):
            text = teacher.generate(prompt, args.max_new_tokens)
            fh.write(json.dumps({"domain": domain, "prompt": prompt,
                                 "text": text}) + "\n")
            chars += len(text)
            if (i + 1) % max(1, len(todo) // 20) == 0:
                done = i + 1
                rate = chars / max(time.time() - start, 1e-9)
                eta = (len(todo) - done) * (time.time() - start) / done
                print(f"  {done:>7,}/{len(todo):,}  {rate:>7.0f} char/s  "
                      f"eta {eta / 60:>5.1f}m", flush=True)
                fh.flush()

    total = time.time() - start
    print(f"\ndone: {chars / 1e6:.2f}M characters in {total / 60:.1f}m")
    print(f"corpus -> {path}")


# ------------------------------------------------------------------ train
def read_corpus(path: Path, seq_len: int) -> Tuple[torch.Tensor, List[str]]:
    """Load the cached teacher corpus into windows, keeping domain labels."""
    from brainai.tokenizer import ByteTokenizer

    tok = ByteTokenizer()
    windows: List[torch.Tensor] = []
    labels: List[str] = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        chunk = tok.chunks(record["text"], seq_len, stride=seq_len)
        windows.append(chunk)
        labels.extend([record["domain"]] * chunk.shape[0])
    if not windows:
        raise SystemExit(f"{path} is empty; run `distill.py generate` first")
    return torch.cat(windows, dim=0), labels


def phase_train(args) -> None:
    path = Path(args.corpus)
    if not path.exists():
        raise SystemExit(
            f"no teacher corpus at {path}\n"
            f"run: python distill.py generate --dry-run --samples 500"
        )

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    tokens, labels = read_corpus(path, args.seq)
    counts: Dict[str, int] = {}
    for d in labels:
        counts[d] = counts.get(d, 0) + 1
    print(f"teacher corpus: {tokens.shape[0]:,} windows of {args.seq} bytes "
          f"({tokens.numel() / 1e6:.2f}M tokens) across {len(counts)} domains")

    model = (BrainAI.load(args.init) if args.init and Path(args.init).exists()
             else BrainAI(BrainConfig.for_scale(args.scale,
                                                max_seq_len=max(args.seq, 128))))
    model.rename(args.name)
    print(f"student: {model.name}, {model.parameter_count():,} params\n")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()

    print(f"{'step':>7}{'loss':>10}{'bits/byte':>12}{'s/step':>9}")
    print("-" * 38)
    start = time.time()

    for step in range(args.steps):
        idx = torch.randint(0, tokens.shape[0], (args.batch,), generator=gen)
        batch = tokens[idx]

        opt.zero_grad(set_to_none=True)
        loss, out = model.lm_loss(batch)
        world_loss, _ = model.world_loss_from(out)
        if world_loss is not None:
            loss = loss + 0.2 * world_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        model.plasticity.accumulate_fisher(model)
        opt.step()

        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            v = float(loss.detach())
            print(f"{step:>7}{v:>10.4f}{v / math.log(2):>12.3f}"
                  f"{(time.time() - start) / (step + 1):>9.2f}", flush=True)

    stats = model.plasticity.consolidate(model)
    print(f"\nconsolidated: {stats['protected_fraction']:.1%} of weights protected")

    if args.eval_real:
        corpus = load_multidomain(seq_len=args.seq, verbose=False)
        per_domain = evaluate_domains(model, corpus, batch_size=args.batch)
        print("\nheld-out loss on *real* text (not teacher output):")
        for name in sorted(corpus.names, key=lambda n: per_domain[n]):
            print(f"  {name:<14}{per_domain[name]:>9.4f}")
        print(f"  {'mean':<14}{per_domain['mean']:>9.4f}")

    model.save(args.out)
    print(f"\nsaved -> {args.out}")
    print(f"\n{attribution_line(model.name)}")
    print("\nQwen and most open-weight teachers carry licence terms; if you "
          "redistribute a model distilled from one, check that checkpoint's "
          "licence and attribute it.")


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Distil a teacher into BrainAI")
    sub = ap.add_subparsers(dest="phase", required=True)

    g = sub.add_parser("generate", help="teacher writes a corpus (cached)")
    g.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    g.add_argument("--samples", type=int, default=20000)
    g.add_argument("--max-new-tokens", type=int, default=96)
    g.add_argument("--domains", default="", help="comma-separated subset")
    g.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    g.add_argument("--device", default="cpu")
    g.add_argument("--dry-run", action="store_true", help="stub teacher, no download")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=phase_generate)

    t = sub.add_parser("train", help="student trains on the cached corpus")
    t.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    t.add_argument("--scale", default="nano")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--seq", type=int, default=128)
    t.add_argument("--lr", type=float, default=6e-4)
    t.add_argument("--init", default=None)
    t.add_argument("--out", default="checkpoints/brainai_distilled.pt")
    t.add_argument("--name", default="BrainAI")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--eval-real", action="store_true",
                   help="also score held-out *real* text, not just teacher output")
    t.set_defaults(func=phase_train)

    p = sub.add_parser("prompts", help="show the domain prompt bank")
    p.set_defaults(func=lambda a: print(__import__(
        "brainai.prompts", fromlist=["report"]).report()))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
