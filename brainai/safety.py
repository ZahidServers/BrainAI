# BrainAI - built by Mohammed Zahid.
"""The safety controller: the last thing between a decision and an actuator.

    executive  ->  motor cortex  ->  SafetyController  ->  joints

A learned policy that updates its own weights at runtime cannot also be the
thing that decides what it is allowed to do.  This layer is deliberately *not*
learned: fixed limits, explicit constraints, and a hard interlock, all
evaluated after the network has produced its command and before anything moves.
Nothing in :mod:`brainai.plasticity` can change it, because online learning
must not be able to widen its own operating envelope.

Three kinds of check, cheapest first:

**Envelope** - position, per-step velocity and effort clamped to the robot's
declared limits.  Catches the common case: a confident network commanding a
joint through its own stop.

**Constraints** - named predicates over the proposed action and its context.
Each returns allow / clamp / veto.  This is where task-specific rules live
("no motion while a human is inside the work envelope").

**Interlock** - a hard e-stop.  When engaged, the controller holds position and
zeroes effort regardless of what the network asked for; it cannot be released
from inside a forward pass.

Every intervention is counted and logged.  A controller that silently clamps is
nearly as dangerous as no controller, because the operator never learns that
the policy is asking for something impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

ALLOW = "allow"
CLAMP = "clamp"
VETO = "veto"


@dataclass
class JointLimits:
    """Physical envelope of the robot, in normalised [-1, 1] joint units."""

    n_actuators: int
    lower: torch.Tensor = field(default=None)          # type: ignore[assignment]
    upper: torch.Tensor = field(default=None)          # type: ignore[assignment]
    max_step: torch.Tensor = field(default=None)       # type: ignore[assignment]
    max_effort: torch.Tensor = field(default=None)     # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = self.n_actuators
        if self.lower is None:
            self.lower = torch.full((n,), -1.0)
        if self.upper is None:
            self.upper = torch.full((n,), 1.0)
        if self.max_step is None:
            # 15% of full range per control step: a joint cannot slam.
            self.max_step = torch.full((n,), 0.15)
        if self.max_effort is None:
            self.max_effort = torch.full((n,), 0.8)

    def to(self, device: torch.device) -> "JointLimits":
        return JointLimits(
            self.n_actuators, self.lower.to(device), self.upper.to(device),
            self.max_step.to(device), self.max_effort.to(device),
        )


@dataclass
class Constraint:
    """A named rule over a proposed action.

    ``check(action, context) -> (verdict, reason)`` where verdict is one of
    ``ALLOW``, ``CLAMP`` or ``VETO``.
    """

    name: str
    check: Callable[[Dict[str, torch.Tensor], Dict[str, Any]], Tuple[str, str]]
    description: str = ""


@dataclass
class SafetyReport:
    """What the controller did to a command, and why."""

    allowed: bool = True
    estop: bool = False
    position_clamped: int = 0
    rate_clamped: int = 0
    effort_clamped: int = 0
    violations: List[str] = field(default_factory=list)
    vetoed_by: Optional[str] = None

    @property
    def intervened(self) -> bool:
        return bool(
            self.estop or self.vetoed_by or self.violations
            or self.position_clamped or self.rate_clamped or self.effort_clamped
        )

    def summary(self) -> str:
        if self.estop:
            return "E-STOP engaged - holding position, effort zeroed"
        if self.vetoed_by:
            return f"VETOED by {self.vetoed_by}: {'; '.join(self.violations)}"
        if not self.intervened:
            return "command passed unmodified"
        parts = []
        if self.position_clamped:
            parts.append(f"{self.position_clamped} joints clamped to limits")
        if self.rate_clamped:
            parts.append(f"{self.rate_clamped} joints rate-limited")
        if self.effort_clamped:
            parts.append(f"{self.effort_clamped} joints effort-capped")
        if self.violations:
            parts.append("; ".join(self.violations))
        return "clamped: " + ", ".join(parts)


class SafetyController:
    """Enforces the operating envelope.  Not a neural network, not trainable."""

    def __init__(
        self,
        limits: JointLimits,
        constraints: Optional[List[Constraint]] = None,
    ) -> None:
        self.limits = limits
        self.constraints: List[Constraint] = list(constraints or [])
        self._estop = False
        # Assume the robot starts at home so the very first command is
        # rate-limited like any other.  Leaving this None would let the first
        # command after boot jump the joints anywhere in one step.  Call
        # set_current_position() with the real measured pose at startup.
        self._last_command: Optional[torch.Tensor] = torch.zeros(limits.n_actuators)
        self.interventions = 0
        self.vetoes = 0
        self.commands = 0
        self.log: List[SafetyReport] = []

    # -- interlock -------------------------------------------------------
    @property
    def estop_engaged(self) -> bool:
        return self._estop

    def engage_estop(self, reason: str = "") -> None:
        self._estop = True
        self.log.append(SafetyReport(allowed=False, estop=True,
                                     violations=[reason or "manual e-stop"]))

    def release_estop(self) -> None:
        """Deliberately a separate, explicit call - never automatic."""
        self._estop = False

    def set_current_position(self, position: torch.Tensor) -> None:
        """Sync to the robot's measured pose.

        Call at startup and after any external motion.  Rate limiting is
        relative to this, so a stale value means the first command afterwards
        is limited against the wrong reference.
        """
        self._last_command = position.detach().clone()

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    # -- the gate --------------------------------------------------------
    @torch.no_grad()
    def filter(
        self,
        joint_targets: torch.Tensor,
        effort: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, SafetyReport]:
        """Return a safe command, plus a report of what was changed."""
        self.commands += 1
        context = context or {}
        report = SafetyReport()
        lim = self.limits.to(joint_targets.device)

        if self._estop:
            report.allowed = False
            report.estop = True
            hold = self._hold_position(joint_targets)
            self._record(report)
            return hold, torch.zeros_like(effort), report

        action = {"joint_targets": joint_targets, "effort": effort}
        for constraint in self.constraints:
            verdict, reason = constraint.check(action, context)
            if verdict == VETO:
                report.allowed = False
                report.vetoed_by = constraint.name
                report.violations.append(reason)
                self.vetoes += 1
                hold = self._hold_position(joint_targets)
                self._record(report)
                return hold, torch.zeros_like(effort), report
            if verdict == CLAMP:
                report.violations.append(f"{constraint.name}: {reason}")

        safe = joint_targets
        report.position_clamped = int(
            ((safe < lim.lower) | (safe > lim.upper)).sum()
        )
        safe = torch.max(torch.min(safe, lim.upper), lim.lower)

        last = self._reference(safe)
        if last is not None:
            delta = safe - last
            report.rate_clamped = int((delta.abs() > lim.max_step).sum())
            safe = last + delta.clamp(-lim.max_step, lim.max_step)

        report.effort_clamped = int((effort > lim.max_effort).sum())
        safe_effort = torch.min(effort, lim.max_effort).clamp_min(0.0)

        self._remember(safe)
        self._record(report)
        return safe, safe_effort, report

    def _reference(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        """Last commanded pose, broadcast to ``like``'s shape.

        Stored as ``(n,)`` and expanded on use.  Comparing shapes for equality
        instead silently disables rate limiting whenever the command arrives
        batched as ``(1, n)``, which is exactly how it always arrives.
        """
        if self._last_command is None:
            return None
        ref = self._last_command.to(like.device)
        return ref.expand_as(like) if ref.dim() < like.dim() else ref

    def _remember(self, command: torch.Tensor) -> None:
        """Record the issued pose as ``(n,)``, taking the last row if batched."""
        self._last_command = (
            command.detach().reshape(-1, command.shape[-1])[-1].clone()
        )

    def _hold_position(self, like: torch.Tensor) -> torch.Tensor:
        """The command issued when motion is refused: stay where we are.

        This also becomes the new rate-limiting reference.  Forgetting that is
        a live hazard -- without it the command *after* a veto has no reference
        to be limited against and the joints can slam.
        """
        ref = self._reference(like)
        hold = torch.zeros_like(like) if ref is None else ref.clone()
        self._remember(hold)
        return hold

    def _record(self, report: SafetyReport) -> None:
        if report.intervened:
            self.interventions += 1
        self.log.append(report)
        if len(self.log) > 1024:
            del self.log[:512]

    def stats(self) -> Dict[str, float]:
        return {
            "commands": float(self.commands),
            "interventions": float(self.interventions),
            "vetoes": float(self.vetoes),
            "intervention_rate": self.interventions / max(self.commands, 1),
            "estop_engaged": float(self._estop),
            "constraints": float(len(self.constraints)),
        }


# -- ready-made constraints ------------------------------------------------

def human_in_envelope(margin: float = 0.0) -> Constraint:
    """Veto motion while a human is detected inside the work envelope.

    Expects ``context['human_distance']`` in metres.
    """
    def check(action, context):
        d = context.get("human_distance")
        if d is None:
            return ALLOW, ""
        if float(d) <= margin:
            return VETO, f"human at {float(d):.2f}m, inside {margin:.2f}m envelope"
        return ALLOW, ""

    return Constraint("human_in_envelope", check,
                      f"no motion with a human within {margin}m")


def effort_ceiling(ceiling: float = 0.6) -> Constraint:
    """Veto commands demanding more force than the task should ever need."""
    def check(action, context):
        peak = float(action["effort"].max())
        if peak > ceiling:
            return VETO, f"peak effort {peak:.2f} exceeds ceiling {ceiling:.2f}"
        return ALLOW, ""

    return Constraint("effort_ceiling", check, f"peak effort <= {ceiling}")


def requires_confidence(minimum: float = 0.3) -> Constraint:
    """Refuse to act on a decision the model itself is unsure about.

    Expects ``context['salience']`` or ``context['confidence']``.
    """
    def check(action, context):
        conf = context.get("confidence", context.get("salience"))
        if conf is None:
            return ALLOW, ""
        if float(conf) < minimum:
            return VETO, f"confidence {float(conf):.3f} below {minimum:.2f}"
        return ALLOW, ""

    return Constraint("requires_confidence", check,
                      f"act only above confidence {minimum}")


def hazardous_joint_ceiling(bank, ceiling: float = 0.5) -> Constraint:
    """A tighter effort cap on stiff, high-force joints.

    A compliant actuator gives way against an unexpected obstacle.  Hydraulic
    and geared-stiff joints do not, and they have the force budget to keep
    going through one, so the envelope that is reasonable for a pneumatic hand
    is not reasonable for a hydraulic hip.
    """
    indices = bank.hazardous()

    def check(action, context):
        if not indices:
            return ALLOW, ""
        effort = action["effort"]
        if effort.shape[-1] <= max(indices):
            return ALLOW, ""
        peak = float(effort[..., indices].max())
        if peak > ceiling:
            return VETO, (f"stiff-and-strong joints at effort {peak:.2f}, "
                          f"ceiling {ceiling:.2f}")
        return ALLOW, ""

    return Constraint("hazardous_joint_ceiling", check,
                      f"{len(indices)} stiff joints capped at {ceiling}")


def default_constraints() -> List[Constraint]:
    """A conservative starting set.  Tune per robot; do not remove blindly."""
    return [human_in_envelope(margin=0.5), effort_ceiling(0.85)]
