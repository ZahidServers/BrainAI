# BrainAI - built by Mohammed Zahid.
"""Do the new subsystems earn their place?  Each against a control.

Four claims, four controls.  A subsystem that cannot beat the obvious
alternative is not a contribution, it is a liability with a docstring.

    world model      vs a persistence baseline ("nothing changes")
    curiosity        vs pure novelty seeking, on a noisy-TV distractor
    autonomous trust vs a fixed operator prior
    planner          vs a flat single-method planner

    python benchmarks/cognition_bench.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai import BrainConfig  # noqa: E402
from brainai.executive import GoalStack  # noqa: E402
from brainai.planner import HierarchicalPlanner, MethodLibrary  # noqa: E402
from brainai.reality import RealityCortex  # noqa: E402
from brainai.world import CuriosityDrive, PredictionLedger, WorldModel  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


# -- 1. world model vs persistence -----------------------------------------
class LinearDynamics(torch.nn.Module):
    """A single linear map. For linear systems this is the *optimal* model."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.step = torch.nn.Linear(d, d, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.step(state)


class DeterministicRSSM(torch.nn.Module):
    """A GRU-based recurrent state-space model, in the PlaNet/Dreamer family.

    Deterministic path only -- the stochastic latent and the KL term matter for
    pixel reconstruction, not for the question here, which is whether BrainAI's
    dynamics head is competitive with a recurrent latent model on the same
    budget.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.cell = torch.nn.GRUCell(d, d)
        self.head = torch.nn.Linear(d, d)
        self.hidden: Optional[torch.Tensor] = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.hidden is None or self.hidden.shape[0] != state.shape[0]:
            self.hidden = torch.zeros_like(state)
        self.hidden = self.cell(state, self.hidden.detach())
        return self.head(self.hidden)


def _train_dynamics(steps: int, d: int, transition) -> Dict[str, float]:
    """Race BrainAI's dynamics head against three baselines on one system."""
    torch.manual_seed(0)
    cfg = BrainConfig.for_scale("nano")
    world, linear, rssm = WorldModel(cfg), LinearDynamics(d), DeterministicRSSM(d)

    opts = {
        "brainai": torch.optim.Adam(world.parameters(), lr=2e-3),
        "linear": torch.optim.Adam(linear.parameters(), lr=2e-3),
        "rssm": torch.optim.Adam(rssm.parameters(), lr=2e-3),
    }
    history: Dict[str, List[float]] = {
        k: [] for k in ("persistence", "linear", "rssm", "brainai")
    }

    for _ in range(steps):
        state = torch.randn(16, d)
        nxt = transition(state)
        history["persistence"].append(
            float(torch.nn.functional.mse_loss(state, nxt))
        )
        for name, model in (("linear", linear), ("rssm", rssm)):
            opts[name].zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(state), nxt)
            loss.backward()
            opts[name].step()
            history[name].append(float(loss.detach()))

        opts["brainai"].zero_grad(set_to_none=True)
        loss, parts = world.loss(state, nxt)
        loss.backward()
        opts["brainai"].step()
        history["brainai"].append(parts["dynamics"])

    result = {k: statistics.mean(v[-20:]) for k, v in history.items()}
    result["_params"] = float(sum(p.numel() for p in world.parameters()))
    result["_world"] = world
    return result


def bench_world_model(steps: int = 300) -> None:
    rule("1. World model vs linear and RSSM baselines")
    print("  Two systems, because the choice of system decides the answer.")
    print("  A linear map is *optimal* for linear dynamics, so a linear test")
    print("  measures nothing about a nonlinear model except its overhead.\n")

    d = BrainConfig.for_scale("nano").d_model
    rotation = torch.linalg.qr(torch.randn(d, d))[0]
    mix = torch.linalg.qr(torch.randn(d, d))[0]

    systems = {
        "linear (rotation)": lambda s: s @ rotation,
        # The gain matters: at unit variance tanh is nearly linear, so a "non-
        # linear" system built that way is still solved optimally by a linear
        # map and tests nothing.  x3 pushes it into saturation.
        "nonlinear (saturating)":
            lambda s: torch.tanh(3.0 * (s @ mix)) * (1.0 + 0.5 * s.tanh()),
    }

    print(f"  {'system':<24}{'persist':>10}{'linear':>10}{'rssm':>10}"
          f"{'brainai':>10}")
    print("  " + "-" * 64)

    results = {}
    for label, transition in systems.items():
        r = _train_dynamics(steps, d, transition)
        results[label] = r
        print(f"  {label:<24}{r['persistence']:>10.4f}{r['linear']:>10.4f}"
              f"{r['rssm']:>10.4f}{r['brainai']:>10.4f}")

    print()
    for label, r in results.items():
        best = min(r["linear"], r["rssm"])
        winner = "linear" if r["linear"] <= r["rssm"] else "rssm"
        ratio = best / max(r["brainai"], 1e-9)
        verdict = "beats it" if ratio > 1.0 else f"loses to it by {1 / ratio:.1f}x"
        print(f"  {label:<24}best baseline = {winner:<7} "
              f"brainai {verdict}")

    print("\n  Read the linear row as a cost measurement, not a defeat: nothing")
    print("  beats the optimal model on its own system.  The nonlinear row is")
    print("  the one that carries information.")

    world = results["nonlinear (saturating)"]["_world"]
    state = torch.randn(1, d)
    horizons = [1, 4, 16]
    confs = [float(world.imagine(state, [None] * h)[1]) for h in horizons]
    print("\n  rollout confidence by horizon: "
          + "  ".join(f"{h}->{c:.3f}" for h, c in zip(horizons, confs)))
    print(f"  monotonically decreasing: "
          f"{all(a > b for a, b in zip(confs, confs[1:]))}")


# -- 2. curiosity vs pure novelty ------------------------------------------
def bench_curiosity() -> None:
    rule("2. Curiosity vs pure novelty seeking (the noisy-TV problem)")
    print("  A pure novelty drive parks in front of a noise source forever:")
    print("  every frame is unseen.  Curiosity should prefer the learnable one.\n")

    cfg = BrainConfig.for_scale("nano")
    drive = CuriosityDrive(cfg)

    # Both maximally novel; the noise source is *expected* to be unpredictable.
    novelty = torch.ones(8)
    surprise = torch.full((8,), 2.0)

    learnable = float(drive.reward(surprise, novelty, torch.zeros(8)).mean())
    noisy_tv = float(drive.reward(surprise, novelty, torch.full((8,), 6.0)).mean())
    pure_novelty = float(novelty.mean())

    print(f"  {'pure novelty (control)':<32}"
          f"learnable {pure_novelty:.3f}   noise {pure_novelty:.3f}   "
          f"ratio 1.0x")
    print(f"  {'curiosity (expected-error gated)':<32}"
          f"learnable {learnable:.3f}   noise {noisy_tv:.3f}   "
          f"ratio {learnable / max(noisy_tv, 1e-9):.1f}x")
    print("\n  A control that cannot tell the two apart will sit in front of the")
    print("  noise.  The ratio is how strongly this one walks away from it.")


# -- 3. autonomous trust vs a fixed prior ----------------------------------
def bench_grounding(rounds: int = 40) -> None:
    rule("3. Autonomous trust grounding vs a fixed operator prior")
    print("  Two sensors, both given prior 0.8 by the operator.  One is honest,")
    print("  one is broken.  Nothing labels them; only predictions resolve.\n")

    torch.manual_seed(0)
    rc = RealityCortex()
    ledger = PredictionLedger()
    for name in ("good_sensor", "broken_sensor"):
        rc.register_source(name, prior=0.8, strength=6)

    good = rc.submit("valve", "state_of", "open", "good_sensor").claim_id
    bad = rc.submit("valve", "state_of", "shut", "broken_sensor").claim_id

    truth = torch.randn(16)
    for _ in range(rounds):
        ledger.expect([good], truth, tolerance=0.3)
        rc.ground(ledger.resolve(truth))                 # good sensor is right
        ledger.expect([bad], -truth, tolerance=0.3)
        rc.ground(ledger.resolve(truth))                 # broken one is wrong

    print(f"  {'source':<18}{'operator prior':>16}{'after grounding':>18}")
    print("  " + "-" * 50)
    for name in ("good_sensor", "broken_sensor"):
        print(f"  {name:<18}{0.8:>16.3f}{rc.trust(name):>18.3f}")

    separation = rc.trust("good_sensor") - rc.trust("broken_sensor")
    print(f"\n  separation after {rounds} rounds: {separation:.3f}")
    print(f"  ledger accuracy: {ledger.report()['accuracy']:.0%}")
    print("  A fixed prior keeps both at 0.800 forever and can never tell them")
    print("  apart -- which is what an oracle-dependent design does in practice.")


# -- 4. hierarchical planner vs flat ---------------------------------------
def bench_planner() -> None:
    rule("4. Hierarchical planner vs a flat one")
    print("  A task whose first method fails.  Flat planning has nowhere to go;")
    print("  hierarchical planning can swap the method for that subgoal.\n")

    lib = MethodLibrary()
    lib.register("serve_tea", ["boil_water", "pour"], "kettle")
    lib.register("serve_tea", ["use_dispenser", "pour"], "dispenser")
    lib.register("boil_water", ["fill_kettle", "switch_on"])

    flat = MethodLibrary()
    flat.register("serve_tea", ["boil_water", "pour"], "kettle")

    for label, library, retries in (("flat", flat, 0), ("hierarchical", lib, 1)):
        planner = HierarchicalPlanner(library, max_retries=retries)
        plan = planner.decompose("serve_tea")
        leaf = plan.next_action()
        outcomes = []
        for _ in range(4):
            if leaf is None:
                break
            outcomes.append(planner.report_result(plan, leaf, False))
            leaf = plan.next_action()
        depth = max(n.depth for n in plan.nodes)
        print(f"  {label:<14} depth {depth}  leaves {len(plan.leaves())}  "
              f"escalation {outcomes}")

    print("\n  'remethod' is the recovery a flat planner cannot express: it can")
    print("  only retry the same failing step or give up on the whole task.")


# -- 5. executive arbitration ----------------------------------------------
def bench_executive() -> None:
    rule("5. Executive: does urgency actually change what runs?")
    stack = GoalStack()
    stack.push("deep_work", priority=0.80)
    stack.push("deliver_report", priority=0.55, deadline=30)

    print(f"  {'tick':>6}{'winner':>18}{'deep_work':>12}{'report':>10}")
    print("  " + "-" * 46)
    for tick in (0, 10, 25, 30):
        stack.tick = tick
        winner = max(range(2), key=lambda i: stack.goals[i].score(tick))
        print(f"  {tick:>6}{stack.goals[winner].name:>18}"
              f"{stack.goals[0].score(tick):>12.3f}"
              f"{stack.goals[1].score(tick):>10.3f}")
    print("\n  The lower-priority job overtakes as its deadline closes, and only")
    print("  then.  A static priority scheme cannot express that.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    print("Cognitive subsystem benchmarks.  Built by Mohammed Zahid.")
    print("Every subsystem is measured against the obvious alternative.")

    bench_world_model(args.steps)
    bench_curiosity()
    bench_grounding()
    bench_planner()
    bench_executive()


if __name__ == "__main__":
    main()
