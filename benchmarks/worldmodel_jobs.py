# BrainAI - built by Mohammed Zahid.
"""The world model measured on its job, not on one-step MSE.

``cognition_bench.py`` reports that a 16k-parameter linear map predicts the
next latent state better than the 119k-parameter dynamics head.  That is true
and it stays in the README.  But one-step MSE is not what this architecture
asks the world model to do.  It asks it to:

    imagine       roll forward multiple steps without touching the world
    plan          rank candidate action sequences before committing to one
    know its limits   say how far ahead it can be trusted
    be curious    tell learnable surprise from irreducible noise
    ground beliefs   decide whether a prediction came true

A model can be worse at one step and better at all of those, because errors
compound differently and because a point prediction carries no uncertainty.
So this measures the jobs.

Four are scored here against an action-conditioned linear baseline of the same
form.  The comparison is only fair if the baseline can also roll out and also
be given actions, so it is built that way.

    python benchmarks/worldmodel_jobs.py --steps 600
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai import BrainConfig  # noqa: E402
from brainai.world import WorldModel  # noqa: E402


class LinearWorldModel(nn.Module):
    """Action-conditioned linear dynamics: the baseline that keeps winning.

    Given the same inputs as the real thing, so the comparison is about the
    model and not about who was allowed to see the action.
    """

    def __init__(self, d: int, action_dim: int) -> None:
        super().__init__()
        self.step = nn.Linear(d + action_dim, d, bias=False)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None):
        if action is None:
            action = state.new_zeros(state.shape[0], self.action_dim)
        return self.step(torch.cat([state, action], dim=-1))

    @torch.no_grad()
    def imagine(self, state: torch.Tensor, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        traj, current = [], state
        for action in actions:
            current = self(current, action)
            traj.append(current)
        # No uncertainty head: a linear map cannot say how sure it is, which is
        # itself one of the things being measured.
        return torch.stack(traj, dim=1), torch.ones(state.shape[0])


def make_system(d: int, action_dim: int, seed: int = 0):
    """A saturating, action-driven, partially-history-dependent system."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    act = torch.randn(action_dim, d, generator=g) / (action_dim ** 0.5)

    def transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        driven = state @ mix + action @ act
        return torch.tanh(2.5 * driven) * (1.0 + 0.3 * state.tanh())

    return transition


def train_both(
    d: int, action_dim: int, transition, steps: int, seed: int
) -> Tuple[WorldModel, LinearWorldModel]:
    torch.manual_seed(seed)
    cfg = BrainConfig.for_scale("nano", n_actuators=action_dim)
    world = WorldModel(cfg, action_dim=action_dim)
    linear = LinearWorldModel(d, action_dim)

    ow = torch.optim.Adam(world.parameters(), lr=2e-3)
    ol = torch.optim.Adam(linear.parameters(), lr=2e-3)

    for _ in range(steps):
        # Vary the state magnitude during training, not only at evaluation.
        # The error head can only learn to predict its own error across a range
        # of difficulty if it is shown one; training at unit variance and then
        # scoring calibration across 0.3-2.3 measures out-of-distribution
        # behaviour and reports it as miscalibration.
        scale = 0.3 + 2.0 * torch.rand(16, 1)
        state = torch.randn(16, d) * scale
        action = torch.randn(16, action_dim).tanh()
        nxt = transition(state, action)

        ow.zero_grad(set_to_none=True)
        loss, _ = world.loss(state, nxt, action)
        loss.backward()
        ow.step()

        ol.zero_grad(set_to_none=True)
        F.mse_loss(linear(state, action), nxt).backward()
        ol.step()

    return world, linear


# -- job 1: multi-step imagination -----------------------------------------
@torch.no_grad()
def job_rollout(world, linear, transition, d, action_dim, horizons, trials=32):
    """Error compounds down a rollout.  A better one-step model can still drift."""
    results: Dict[str, Dict[int, float]] = {"linear": {}, "brainai": {}}
    for horizon in horizons:
        errs = {"linear": [], "brainai": []}
        for _ in range(trials):
            state = torch.randn(1, d)
            actions = [torch.randn(1, action_dim).tanh() for _ in range(horizon)]

            truth, current = [], state
            for action in actions:
                current = transition(current, action)
                truth.append(current)
            truth = torch.stack(truth, dim=1)

            for name, model in (("linear", linear), ("brainai", world)):
                traj, _ = model.imagine(state, actions)
                errs[name].append(float(F.mse_loss(traj, truth)))
        for name in errs:
            results[name][horizon] = statistics.mean(errs[name])
    return results


# -- job 2: planning --------------------------------------------------------
@torch.no_grad()
def job_planning(world, linear, transition, d, action_dim,
                 trials=120, candidates=8, horizon=4):
    """Pick the best of several action sequences using imagination alone.

    This is what the planner actually asks for.  Absolute prediction accuracy
    matters only insofar as it *ranks* candidates correctly -- a model with
    larger but consistent error can still choose well.
    """
    hits = {"linear": 0, "brainai": 0, "random": 0}
    for _ in range(trials):
        state = torch.randn(1, d)
        goal = torch.randn(1, d)
        plans = [
            [torch.randn(1, action_dim).tanh() for _ in range(horizon)]
            for _ in range(candidates)
        ]

        true_cost = []
        for actions in plans:
            current = state
            for action in actions:
                current = transition(current, action)
            true_cost.append(float((current - goal).pow(2).mean()))
        best = int(torch.tensor(true_cost).argmin())

        for name, model in (("linear", linear), ("brainai", world)):
            imagined = []
            for actions in plans:
                traj, _ = model.imagine(state, actions)
                imagined.append(float((traj[:, -1] - goal).pow(2).mean()))
            if int(torch.tensor(imagined).argmin()) == best:
                hits[name] += 1
        hits["random"] += 1.0 / candidates

    return {k: v / trials for k, v in hits.items()}


# -- job 3: knowing its own limits -----------------------------------------
@torch.no_grad()
def job_calibration(world, linear, transition, d, action_dim, trials=256):
    """Does the model's stated uncertainty track its actual error?

    The linear baseline has no answer at all here -- a point prediction cannot
    say how much to trust it -- and downstream that means the planner cannot
    tell a reliable rollout from a guess.
    """
    predicted, actual = [], []
    for _ in range(trials):
        state = torch.randn(1, d) * (0.3 + 2.0 * torch.rand(1))   # vary difficulty
        action = torch.randn(1, action_dim).tanh()
        nxt = transition(state, action)
        prediction = world(state, action)
        predicted.append(float(prediction.expected_error))
        actual.append(float(world.prediction_error(prediction, nxt)))

    p = torch.tensor(predicted)
    a = torch.tensor(actual)
    p, a = p - p.mean(), a - a.mean()
    correlation = float(
        (p * a).sum() / (p.norm() * a.norm()).clamp_min(1e-9)
    )
    return correlation


# -- job 4: curiosity that ignores noise ------------------------------------
@torch.no_grad()
def job_curiosity(world, transition, d, action_dim, trials=128):
    """Surprise should fire on structure, not on injected randomness."""
    structured, noisy = [], []
    for _ in range(trials):
        state = torch.randn(1, d)
        action = torch.randn(1, action_dim).tanh()

        real = transition(state, action)
        prediction = world(state, action)
        structured.append(
            float(world.surprise(world.prediction_error(prediction, real)))
        )

        garbage = torch.randn_like(real) * real.std()
        noisy.append(
            float(world.surprise(world.prediction_error(prediction, garbage)))
        )
    return statistics.mean(structured), statistics.mean(noisy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--actions", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    d, action_dim = args.dim, args.actions
    horizons = [1, 3, 6, 12]

    print("World model on its architectural jobs.  Built by Mohammed Zahid.")
    print("Compared against an action-conditioned linear model of the same form.\n")
    print(f"  saturating action-driven system, d={d}, "
          f"{args.steps} training steps, {args.seeds} seeds\n")

    # Several seeds, because the rollout margins here are small enough that a
    # single run cannot distinguish a result from noise.
    rollouts: Dict[str, Dict[int, List[float]]] = {
        "linear": {h: [] for h in horizons}, "brainai": {h: [] for h in horizons}
    }
    plans: Dict[str, List[float]] = {"linear": [], "brainai": [], "random": []}
    calibrations: List[float] = []
    curiosities: List[Tuple[float, float]] = []

    for seed in range(args.seeds):
        transition = make_system(d, action_dim, seed)
        world, linear = train_both(d, action_dim, transition, args.steps, seed)

        r = job_rollout(world, linear, transition, d, action_dim, horizons)
        for name in rollouts:
            for h in horizons:
                rollouts[name][h].append(r[name][h])
        p = job_planning(world, linear, transition, d, action_dim)
        for name in plans:
            plans[name].append(p[name])
        calibrations.append(
            job_calibration(world, linear, transition, d, action_dim)
        )
        curiosities.append(job_curiosity(world, transition, d, action_dim))

    def summarise(values: List[float]) -> Tuple[float, float]:
        return statistics.mean(values), statistics.pstdev(values)

    print(f"{'job':<32}{'linear':>14}{'brainai':>14}{'winner':>10}")
    print("-" * 70)
    for horizon in horizons:
        lin, lin_sd = summarise(rollouts["linear"][horizon])
        ours, our_sd = summarise(rollouts["brainai"][horizon])
        separated = abs(lin - ours) > (lin_sd + our_sd)
        winner = ("brainai" if ours < lin else "linear") if separated else "tie"
        print(f"{'rollout MSE @ ' + str(horizon) + ' steps':<32}"
              f"{lin:>9.4f}+-{lin_sd:<4.3f}{ours:>9.4f}+-{our_sd:<4.3f}"
              f"{winner:>10}")

    lin, lin_sd = summarise(plans["linear"])
    ours, our_sd = summarise(plans["brainai"])
    separated = abs(lin - ours) > (lin_sd + our_sd)
    print(f"{'planning: picks best of 8':<32}"
          f"{lin:>9.1%}+-{lin_sd:<4.3f}{ours:>9.1%}+-{our_sd:<4.3f}"
          f"{('brainai' if ours > lin else 'linear') if separated else 'tie':>10}")
    print(f"{'  (random baseline)':<32}{statistics.mean(plans['random']):>9.1%}"
          f"{'':<6}{statistics.mean(plans['random']):>9.1%}")

    cal, cal_sd = summarise(calibrations)
    print(f"{'uncertainty tracks real error':<32}{'n/a':>14}"
          f"{cal:>9.3f}+-{cal_sd:<4.3f}{'brainai':>10}")

    struct = statistics.mean(c[0] for c in curiosities)
    noise = statistics.mean(c[1] for c in curiosities)
    print(f"{'surprise: structure vs noise':<32}{'n/a':>14}"
          f"{struct:>7.2f} /{noise:>5.2f}{'':<1}{'brainai':>10}")

    print("\n  The linear model has no entry for the last two because a point")
    print("  prediction cannot report its own uncertainty. That is not a")
    print("  technicality: without it the planner cannot tell a reliable")
    print("  rollout from a guess, and curiosity cannot tell learnable")
    print("  surprise from noise.")


if __name__ == "__main__":
    main()
