# BrainAI - built by Mohammed Zahid.
"""The predictive world model: imagine before acting, learn from being wrong.

Everything else in this architecture is reactive -- something arrives, the
pathways respond, memory records it.  This module is the part that runs *ahead*
of the world: given the current latent state and a candidate action, it predicts
the next state, and the gap between that prediction and what actually happened
becomes a learning signal that needs no labels and no human.

Three things fall out of that one mechanism, which is why it is worth more than
its size suggests:

**Grounding.**  :class:`PredictionLedger` lets a belief stake a claim on the
future.  When the observation arrives, the ledger resolves it and tells the
Reality Cortex which sources were right.  That removes the oracle -- the
environment, not an operator, decides who to trust.

**Curiosity.**  Prediction error normalised against a running baseline is an
intrinsic reward.  It goes to zero for both mastered states (error low) and
irreducibly random ones (error high but *expected* high), so the drive points at
what is learnable rather than merely surprising.

**Planning.**  :meth:`WorldModel.imagine` rolls the dynamics forward without
touching the environment, which is what lets the planner score a course of
action before committing to it.

The body model is the same idea applied inward: predict the proprioceptive
consequence of a motor command, and the error is a calibration signal for the
robot's own dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import RMSNorm
from .config import BrainConfig


@dataclass
class Prediction:
    """What the model expects to happen next, and how sure it is."""

    state: torch.Tensor            # (B, d) predicted next latent state
    confidence: torch.Tensor       # (B,) in [0, 1]
    expected_error: torch.Tensor   # (B,) predicted magnitude of its own error

    def detach(self) -> "Prediction":
        return Prediction(
            self.state.detach(), self.confidence.detach(),
            self.expected_error.detach(),
        )


class WorldModel(nn.Module):
    """Forward dynamics over the shared latent bus.

    Predicts the next state *and* its own error magnitude.  That second head is
    what separates "I do not know" from "this is inherently noisy": a die roll
    has high error and high expected error, so surprise -- and therefore
    curiosity -- is near zero.  Without it the agent parks in front of static
    and calls it learning.
    """

    def __init__(self, cfg: BrainConfig, action_dim: Optional[int] = None) -> None:
        super().__init__()
        d = cfg.d_model
        self.action_dim = action_dim or cfg.n_actuators

        self.action_embed = nn.Linear(self.action_dim, d, bias=False)
        self.dynamics = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d),
        )
        self.norm = RMSNorm(d)
        self.error_head = nn.Sequential(
            nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1),
        )

        # Running error statistics, for normalising surprise.
        self.register_buffer("error_mean", torch.zeros(()))
        self.register_buffer("error_var", torch.ones(()))
        self.register_buffer("observations", torch.zeros((), dtype=torch.long))

    def forward(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Prediction:
        """``state``: (B, d).  ``action``: (B, action_dim) or None for passive."""
        if action is None:
            action = state.new_zeros(state.shape[0], self.action_dim)
        joint = torch.cat([state, self.action_embed(action)], dim=-1)

        delta = self.dynamics(joint)
        predicted = self.norm(state + delta)          # residual dynamics
        expected_error = F.softplus(self.error_head(joint)).squeeze(-1)
        confidence = torch.exp(-expected_error).clamp(0.0, 1.0)
        return Prediction(predicted, confidence, expected_error)

    def prediction_error(
        self, prediction: Prediction, actual: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample normalised L2 error between prediction and reality."""
        diff = prediction.state - actual
        scale = actual.norm(dim=-1).clamp_min(1e-6)
        return (diff.norm(dim=-1) / scale)

    def loss(
        self,
        state: torch.Tensor,
        actual_next: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Self-supervised: predict the next state, and predict your own error.

        No labels involved -- the next observation is the target.
        """
        prediction = self(state, action)
        error = self.prediction_error(prediction, actual_next)

        dynamics_loss = F.mse_loss(prediction.state, actual_next)
        # The error head regresses onto the error it just made, detached so it
        # cannot cheat by making the dynamics worse to be right about it.
        calibration_loss = F.mse_loss(prediction.expected_error, error.detach())

        self._update_stats(error)
        return dynamics_loss + 0.1 * calibration_loss, {
            "dynamics": float(dynamics_loss.detach()),
            "calibration": float(calibration_loss.detach()),
            "error": float(error.detach().mean()),
        }

    @torch.no_grad()
    def _update_stats(self, error: torch.Tensor) -> None:
        e = error.detach().mean()
        n = self.observations.item() + 1
        self.observations += 1
        rate = max(0.01, 1.0 / n)
        delta = e - self.error_mean
        self.error_mean += rate * delta
        self.error_var = (1 - rate) * (self.error_var + rate * delta.pow(2))

    @torch.no_grad()
    def surprise(self, error: torch.Tensor) -> torch.Tensor:
        """Error in units of its own standard deviation.

        Being wrong is not surprising.  Being wrong *by more than usual* is.
        """
        sd = self.error_var.clamp_min(1e-8).sqrt()
        return ((error - self.error_mean) / sd).clamp(-8.0, 8.0)

    @torch.no_grad()
    def imagine(
        self,
        state: torch.Tensor,
        actions: Sequence[Optional[torch.Tensor]],
        collapse: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll the dynamics forward without touching the environment.

        Returns ``(trajectory, confidence)`` where trajectory is
        ``(B, horizon, d)``.  Confidence compounds multiplicatively down the
        rollout, so a long plan through uncertain dynamics is correctly reported
        as barely worth trusting.
        """
        trajectory: List[torch.Tensor] = []
        confidence = torch.ones(state.shape[0], device=state.device)
        current = state
        for action in actions:
            prediction = self(current, action)
            current = prediction.state
            confidence = confidence * prediction.confidence
            trajectory.append(current)
        if not trajectory:
            return state.unsqueeze(1), confidence
        stacked = torch.stack(trajectory, dim=1)
        return (stacked if collapse else stacked), confidence


class CuriosityDrive(nn.Module):
    """Intrinsic motivation from learnable prediction error.

    ``reward = surprise x novelty``, where surprise is error relative to the
    model's *expected* error.  Both terms are needed: novelty alone chases
    anything unseen including noise, surprise alone re-chases the same
    unpredictable corner forever.

    ``boredom`` rises while surprise stays low, which is what makes an idle
    agent go look for something instead of sitting in a mastered state.
    """

    def __init__(self, cfg: BrainConfig, boredom_rate: float = 0.02) -> None:
        super().__init__()
        self.cfg = cfg
        self.boredom_rate = boredom_rate
        self.register_buffer("boredom", torch.zeros(()))
        self.register_buffer("total_reward", torch.zeros(()))

    @torch.no_grad()
    def reward(
        self,
        surprise: torch.Tensor,
        novelty: torch.Tensor,
        expected_error: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Intrinsic reward per sample, in [0, 1]."""
        learnable = torch.sigmoid(surprise)          # high only if worse than usual
        if expected_error is not None:
            # Irreducible noise: high expected error means do not bother.
            learnable = learnable * torch.exp(-expected_error).clamp(0.0, 1.0)
        r = (learnable * novelty.to(learnable.device)).clamp(0.0, 1.0)

        mean_r = r.mean()
        self.total_reward += mean_r
        if float(mean_r) < 0.05:
            self.boredom.add_(self.boredom_rate).clamp_(0.0, 1.0)
        else:
            self.boredom.mul_(1.0 - self.boredom_rate)
        return r

    @property
    def exploring(self) -> bool:
        """True when the agent should go looking rather than exploiting."""
        return float(self.boredom) > 0.5

    def report(self) -> Dict[str, float]:
        return {
            "boredom": float(self.boredom),
            "exploring": float(self.exploring),
            "cumulative_intrinsic_reward": float(self.total_reward),
        }


class BodyModel(nn.Module):
    """The world model turned inward: what will my own body do?

    Predicts the proprioceptive result of a motor command.  The residual between
    prediction and measurement is a calibration signal for the robot's actual
    dynamics -- backlash, a sticking joint, a drifting encoder -- learned from
    operation rather than from a datasheet.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        n = cfg.n_actuators
        self.predict = nn.Sequential(
            nn.Linear(2 * n, 4 * n), nn.GELU(), nn.Linear(4 * n, n), nn.Tanh(),
        )
        self.register_buffer("joint_error", torch.zeros(n))
        self.register_buffer("calibrations", torch.zeros((), dtype=torch.long))

    def forward(
        self, joint_state: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        return self.predict(torch.cat([joint_state, command], dim=-1))

    def calibration_loss(
        self,
        joint_state: torch.Tensor,
        command: torch.Tensor,
        achieved: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self(joint_state, command)
        with torch.no_grad():
            per_joint = (predicted - achieved).abs().mean(dim=0)
            self.joint_error.mul_(0.95).add_(0.05 * per_joint)
            self.calibrations += 1
        return F.mse_loss(predicted, achieved)

    def worst_joints(self, k: int = 3) -> List[Tuple[int, float]]:
        """Which joints the body model understands least - a maintenance signal."""
        k = min(k, self.joint_error.numel())
        values, indices = self.joint_error.topk(k)
        return [(int(i), float(v)) for i, v in zip(indices, values)]


@dataclass
class OpenPrediction:
    """A belief staking a claim on the future."""

    claim_ids: List[int]
    predicted: torch.Tensor
    tolerance: float
    description: str = ""
    age: int = 0


class PredictionLedger:
    """Open predictions awaiting reality's verdict.

    This is what removes the oracle from :meth:`RealityCortex.observe_outcome`.
    A belief that implies something observable registers that expectation here;
    when the observation lands, the ledger scores it and reports which claims
    were vindicated.  Nobody has to label anything -- the world does it.

    Predictions that are never resolved expire, and expiry is *not* treated as
    a refutation.  Silently marking unobserved predictions wrong would make
    source trust decay just for talking about things the robot never checked.
    """

    def __init__(self, max_age: int = 64) -> None:
        self.max_age = max_age
        self.open: List[OpenPrediction] = []
        self.resolved = 0
        self.expired = 0
        self.correct = 0

    def expect(
        self,
        claim_ids: Sequence[int],
        predicted: torch.Tensor,
        tolerance: float = 0.25,
        description: str = "",
    ) -> None:
        if not claim_ids:
            return
        self.open.append(OpenPrediction(
            claim_ids=list(claim_ids),
            predicted=predicted.detach().reshape(-1),
            tolerance=tolerance,
            description=description,
        ))

    def resolve(self, actual: torch.Tensor) -> List[Tuple[int, bool]]:
        """Score every open prediction against an observation.

        Returns ``(claim_id, was_correct)`` pairs for the Reality Cortex.
        """
        actual = actual.detach().reshape(-1)
        verdicts: List[Tuple[int, bool]] = []
        still_open: List[OpenPrediction] = []

        for entry in self.open:
            if entry.predicted.shape != actual.shape:
                entry.age += 1
                if entry.age < self.max_age:
                    still_open.append(entry)
                else:
                    self.expired += 1
                continue

            scale = actual.norm().clamp_min(1e-6)
            error = float((entry.predicted - actual).norm() / scale)
            was_correct = error <= entry.tolerance
            self.resolved += 1
            self.correct += int(was_correct)
            verdicts.extend((cid, was_correct) for cid in entry.claim_ids)

        self.open = still_open
        return verdicts

    def tick(self) -> int:
        """Age open predictions; expire the stale ones.  Returns how many expired."""
        kept, expired = [], 0
        for entry in self.open:
            entry.age += 1
            if entry.age < self.max_age:
                kept.append(entry)
            else:
                expired += 1
        self.open = kept
        self.expired += expired
        return expired

    def report(self) -> Dict[str, float]:
        return {
            "open": float(len(self.open)),
            "resolved": float(self.resolved),
            "expired": float(self.expired),
            "accuracy": self.correct / self.resolved if self.resolved else 0.0,
        }
