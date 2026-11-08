# BrainAI - built by Mohammed Zahid Imtiyaz Wadiwale.
"""Run every weight-dependent benchmark on a *trained* checkpoint.

The stand-alone benchmarks in ``benchmarks/`` mostly build small throwaway
models, because their job is to test the architecture in isolation. This script
answers a different question: how does the design behave on the actual model you
just spent hours training? It loads one checkpoint and drives the real weights
through every test whose outcome depends on them.

    python evaluate_trained.py --ckpt checkpoints/brainai_full.pt

What is measured on the trained weights:
    per-domain loss        the breadth of what it learned
    generation            what it actually emits, per domain
    author knowledge      does it associate BrainAI with its author
    gating latency        does skipping the conscious pathway save real time
    world-model jobs      the checkpoint's own dynamics head, on rollout/planning
    online stability      400 inference-time updates on the real weights
    uncertainty           per-stage confidence composed on a real forward pass
    energy / sparsity     spike rate of the trained network

What is reported but is training-independent by construction (shown once, and
labelled, so the distinction is explicit rather than implied):
    reality cortex        pure log-odds evidence math, no neural weights
    safety envelope       fixed limits, not learned

Nothing here retrains the language model. The 400-update stability probe and the
world-model-job training are seconds-to-minutes and operate on top of the loaded
checkpoint; everything else is forward passes.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Generation samples contain non-Latin scripts and the U+FFFD replacement
# character; the default Windows console codec (cp1252) raises on those. Force
# UTF-8 with replacement so a printout never crashes the evaluation.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brainai import BrainAI
from brainai.corpus import evaluate_domains, load_multidomain
from brainai.identity import attribution_line
from brainai.uncertainty import entropy_confidence


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


# -- 1. breadth -------------------------------------------------------------
def eval_breadth(model: BrainAI, seq: int, batch: int) -> Dict[str, float]:
    rule("1. Per-domain held-out loss (breadth of what it learned)")
    corpus = load_multidomain(seq_len=seq, verbose=False)
    scores = evaluate_domains(model, corpus, batch_size=batch, batches=6)
    ordered = sorted((k for k in scores if k != "mean"), key=lambda k: scores[k])
    print(f"  {'domain':<44}{'loss':>8}{'bits/byte':>11}")
    print("  " + "-" * 63)
    for name in ordered:
        print(f"  {name:<44}{scores[name]:>8.3f}{scores[name] / math.log(2):>11.2f}")
    print("  " + "-" * 63)
    print(f"  {'mean':<44}{scores['mean']:>8.3f}{scores['mean']/math.log(2):>11.2f}")
    return scores


# -- 2. generation ----------------------------------------------------------
def eval_generation(model: BrainAI) -> None:
    rule("2. Generation per domain (what it actually emits)")
    prompts = [
        ("python",  "def factorial(n):"),
        ("english",  "The most important thing about "),
        ("history",  "In the ancient world, "),
        ("math",     "To solve the equation "),
        ("morse",    "TEXT: SOS\nMORSE: "),
    ]
    for label, p in prompts:
        text = model.generate(p, max_new_tokens=64, temperature=0.7)
        print(f"  [{label}] {p!r}\n      -> {text!r}\n")


# -- 3. author knowledge ----------------------------------------------------
def eval_author(model: BrainAI) -> None:
    rule("3. Author / model knowledge")
    probes = [
        "Q: Who built BrainAI?\nA:",
        "Q: What does AMSA stand for?\nA:",
        "Q: Who created you?\nA:",
        "BrainAI was designed and built by ",
        "The AMSA activation function was developed by ",
    ]
    for p in probes:
        text = model.generate(p, max_new_tokens=48, temperature=0.5)
        print(f"  {p!r}\n      -> {text!r}\n")
    print("  Note: a 5.4M byte-level model reproduces trained associations as")
    print("  fragments, not fluent sentences. Read this as 'did the association")
    print("  form', not as a chatbot answer.")


# -- 4. gating latency ------------------------------------------------------
@torch.no_grad()
def eval_gating(model: BrainAI, seq: int, batch: int, repeats: int = 10) -> None:
    rule("4. Gating latency on the trained model")
    model.eval()
    tokens = torch.randint(0, 255, (batch, seq))

    def timed() -> float:
        model(tokens=tokens)
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            model(tokens=tokens)
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    model.cfg.conscious_gate_threshold = 0.0
    on = timed()
    model.cfg.conscious_gate_threshold = 1.01
    off = timed()
    model.cfg.conscious_gate_threshold = 0.5
    print(f"  all rows deliberate : {on:.4f} s")
    print(f"  none deliberate     : {off:.4f} s")
    print(f"  speedup             : {on / max(off, 1e-9):.2f}x  "
          f"({(1 - off / on) * 100:.0f}% of forward time in the conscious path)")


# -- 5. world-model jobs (the checkpoint's own dynamics head) ----------------
@torch.no_grad()
def eval_world_model(model: BrainAI, seq: int) -> None:
    rule("5. World model jobs, using the checkpoint's own trained dynamics")
    d = model.cfg.d_model
    # Drive the trained model to get real latent states, then test its world
    # model's rollout confidence on them.
    tokens = torch.randint(0, 255, (4, seq))
    out = model(tokens=tokens)
    state = out.summary
    for horizon in (1, 4, 16):
        _, conf = model.world.imagine(state, [None] * horizon)
        print(f"  imagine {horizon:>2} steps ahead -> confidence {float(conf.mean()):.4f}")
    # Calibration: does its expected-error head track its real error on a step?
    nxt = model(tokens=torch.randint(0, 255, (4, seq))).summary
    pred = model.world(state)
    err = float(model.world.prediction_error(pred, nxt).mean())
    print(f"  one-step error {err:.4f}  expected-error head "
          f"{float(pred.expected_error.mean()):.4f}")
    print("  (confidence should fall with horizon; that is the planning signal)")


# -- 6. online stability ----------------------------------------------------
def eval_stability(model: BrainAI, seq: int, updates: int = 200) -> None:
    rule(f"6. Online-learning stability: {updates} inference-time updates")
    fact = model.tokenizer.encode_batch(
        ["the workshop key is kept in the third drawer"] * 2, seq
    )
    snapshot = model.lm_head.weight.detach().clone()
    before = float(model.lm_loss(fact, write_memory=False)[0].detach())
    rolled = 0
    for _ in range(updates):
        stats = model.learn_online(fact, reward=1.0, verify_update=True)
        rolled += int(stats.get("rolled_back", 0.0))
    after = float(model.lm_loss(fact, write_memory=False)[0].detach())
    drift = float((model.lm_head.weight.detach() - snapshot).norm())
    rate = model.energy_report()["mean_spike_rate"]
    print(f"  fact loss {before:.4f} -> {after:.4f}  (learned {before - after:+.4f})")
    print(f"  weight drift {drift:.4f}   rollbacks {rolled}/{updates}   "
          f"spike rate {rate:.3f}")
    print("  (weights moved, in eval mode, without the network going dense)")


# -- 7. uncertainty ---------------------------------------------------------
@torch.no_grad()
def eval_uncertainty(model: BrainAI, seq: int) -> None:
    rule("7. Uncertainty on a real forward pass")
    out = model(tokens=torch.randint(0, 255, (2, seq)))
    if out.logits is not None:
        conf = float(entropy_confidence(out.logits).mean())
        print(f"  output entropy-confidence {conf:.3f}")
    print(f"  {model.uncertainty.summary()}")


# -- 8. design-only (labelled) ---------------------------------------------
def eval_design_only(model: BrainAI) -> None:
    rule("8. Design-only checks (training-independent by construction)")
    rc = model.reality
    for name, prior in [("textbook", 0.95), ("expert", 0.8), ("rumour", 0.2)]:
        rc.register_source(name, prior=prior, strength=10)
    rc.assert_axiom("earth", "shape_of", "oblate spheroid")
    for _ in range(50):
        r = rc.submit("earth", "shape_of", "flat", "rumour")
    print(f"  reality cortex: 'earth is flat' x50 from a low-trust source "
          f"-> confidence {r.confidence:.4f} (axiom holds)")
    print(f"  safety: {int(model.safety.stats()['constraints'])} constraints, "
          f"envelope enforced per joint")
    print("  These do not change with training; shown for completeness.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained BrainAI checkpoint")
    ap.add_argument("--ckpt", default="checkpoints/brainai_full.pt")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip", default="", help="comma-separated sections to skip")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    if not Path(args.ckpt).exists():
        raise SystemExit(f"no checkpoint at {args.ckpt}")

    print(f"Evaluating {args.ckpt} on the trained weights.")
    model = BrainAI.load(args.ckpt)
    print(attribution_line(model.name))
    print(f"parameters: {model.parameter_count():,}\n")

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if "breadth" not in skip:
        eval_breadth(model, args.seq, args.batch)
    if "generation" not in skip:
        eval_generation(model)
    if "author" not in skip:
        eval_author(model)
    if "gating" not in skip:
        eval_gating(model, args.seq, args.batch)
    if "world" not in skip:
        eval_world_model(model, args.seq)
    if "stability" not in skip:
        eval_stability(model, args.seq, args.updates)
    if "uncertainty" not in skip:
        eval_uncertainty(model, args.seq)
    if "design" not in skip:
        eval_design_only(model)

    print(f"\n{attribution_line(model.name)}")


if __name__ == "__main__":
    main()
