# BrainAI - built by Mohammed Zahid.
"""Actuator types, and why one set of limits for all of them is unsafe.

``JointLimits(n_actuators)`` gave every joint the same envelope: same travel,
same rate cap, same effort ceiling.  On a robot with one kind of servo that is
fine.  On a humanoid with hydraulic hips, geared servos in the arms and a
pneumatic gripper it is wrong in both directions at once -- the rate cap that
protects a fingertip servo makes a hydraulic leg useless, and the effort
ceiling that lets the leg stand up will crush whatever the gripper is holding.

Each type here carries the properties that actually change how it must be
commanded and bounded:

**DC brushed / brushless** -- continuous rotation, commanded by velocity.  No
position feedback of its own; needs an encoder.

**Stepper** -- moves in fixed increments and holds position with no feedback
loop.  A command between steps is not achievable, so it is quantised rather
than silently rounded by the driver.

**Servo** -- position-commanded with an integrated loop.  The well-behaved
case, and the one uniform limits were implicitly written for.

**Hydraulic** -- very high force, low bandwidth, poor back-drivability.
Dangerous precisely because it will not stall against an obstacle: it has the
force budget to keep going through one.

**Pneumatic** -- compressible working fluid, so it is naturally compliant and
fast but cannot hold a precise position under varying load.  Good for grasping,
bad for anything that needs to be where you asked.

``compliance`` is the field that matters most for safety.  A compliant actuator
gives way against an unexpected obstacle; a stiff high-force one does not.
:meth:`ActuatorBank.limits` derives a per-joint envelope from the type, so the
safety controller enforces the right bound on each joint instead of the same
wrong one everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

DC_BRUSHED = "dc_brushed"
DC_BRUSHLESS = "dc_brushless"
STEPPER = "stepper"
SERVO = "servo"
HYDRAULIC = "hydraulic"
PNEUMATIC = "pneumatic"

#: How the actuator is commanded.  Sending a position to a DC motor is a
#: category error, not a tuning problem.
POSITION = "position"
VELOCITY = "velocity"
FORCE = "force"


@dataclass(frozen=True)
class ActuatorSpec:
    """One actuator type and the envelope it must be held to."""

    kind: str
    control: str
    max_travel: float = 1.0        # normalised position range, +-
    max_step: float = 0.15         # per control cycle
    max_effort: float = 0.8        # normalised force/torque
    compliance: float = 0.3        # 0 = rigid, 1 = fully back-drivable
    bandwidth_hz: float = 50.0
    discrete_steps: int = 0        # 0 = continuous
    backlash: float = 0.0
    description: str = ""

    @property
    def stiff_and_strong(self) -> bool:
        """Will not yield to an obstacle and has the force to go through it."""
        return self.compliance < 0.2 and self.max_effort > 0.7

    def quantise(self, command: torch.Tensor) -> torch.Tensor:
        """Snap to achievable positions.

        A stepper cannot sit between steps.  Rounding in the driver instead
        means the commanded and achieved positions silently disagree, which
        shows up later as an unexplained body-model error.
        """
        if self.discrete_steps <= 0:
            return command
        step = 2.0 * self.max_travel / self.discrete_steps
        return torch.round(command / step) * step


STANDARD_ACTUATORS: Dict[str, ActuatorSpec] = {
    DC_BRUSHED: ActuatorSpec(
        DC_BRUSHED, VELOCITY, max_step=0.25, max_effort=0.6, compliance=0.5,
        bandwidth_hz=100, backlash=0.02,
        description="cheap continuous rotation, needs an external encoder"),
    DC_BRUSHLESS: ActuatorSpec(
        DC_BRUSHLESS, VELOCITY, max_step=0.30, max_effort=0.85, compliance=0.35,
        bandwidth_hz=500, backlash=0.005,
        description="high power density, electronic commutation"),
    STEPPER: ActuatorSpec(
        STEPPER, POSITION, max_step=0.10, max_effort=0.5, compliance=0.15,
        bandwidth_hz=50, discrete_steps=200,
        description="repeatable open-loop steps; stalls silently if overloaded"),
    SERVO: ActuatorSpec(
        SERVO, POSITION, max_step=0.15, max_effort=0.7, compliance=0.25,
        bandwidth_hz=100, backlash=0.01,
        description="closed-loop angle control, the well-behaved default"),
    HYDRAULIC: ActuatorSpec(
        HYDRAULIC, FORCE, max_step=0.05, max_effort=1.0, compliance=0.05,
        bandwidth_hz=20, backlash=0.0,
        description="very high force, slow, will not stall against an obstacle"),
    PNEUMATIC: ActuatorSpec(
        PNEUMATIC, FORCE, max_step=0.35, max_effort=0.45, compliance=0.85,
        bandwidth_hz=30, backlash=0.05,
        description="compliant and fast; cannot hold position under load"),
}


def actuator(kind: str) -> ActuatorSpec:
    try:
        return STANDARD_ACTUATORS[kind]
    except KeyError:
        raise KeyError(
            f"unknown actuator {kind!r}; known: {sorted(STANDARD_ACTUATORS)}"
        ) from None


@dataclass
class ActuatorBank:
    """The actual actuator complement of one robot.

    Built as a list of ``(kind, count)`` so a humanoid can declare hydraulic
    hips, servo arms and a pneumatic gripper, and get a per-joint envelope that
    respects each.
    """

    layout: List[Tuple[str, int]] = field(default_factory=list)

    @classmethod
    def uniform(cls, kind: str, count: int) -> "ActuatorBank":
        return cls([(kind, count)])

    @classmethod
    def humanoid(cls, scale: int = 32) -> "ActuatorBank":
        """A representative mixed humanoid: strong legs, precise arms, soft hands."""
        legs = max(2, scale // 4)
        arms = max(2, scale // 3)
        hands = max(2, scale // 4)
        neck = max(1, scale - legs - arms - hands)
        return cls([
            (HYDRAULIC, legs),       # take the weight
            (DC_BRUSHLESS, arms),    # fast, strong enough, back-drivable
            (PNEUMATIC, hands),      # compliant grasping
            (SERVO, neck),           # precise pointing
        ])

    @property
    def n_actuators(self) -> int:
        return sum(count for _, count in self.layout)

    def specs(self) -> List[ActuatorSpec]:
        out: List[ActuatorSpec] = []
        for kind, count in self.layout:
            out.extend([actuator(kind)] * count)
        return out

    def kinds(self) -> List[str]:
        return [kind for kind, count in self.layout for _ in range(count)]

    def tensor(self, attribute: str) -> torch.Tensor:
        """Per-joint vector of one spec attribute."""
        return torch.tensor(
            [getattr(s, attribute) for s in self.specs()], dtype=torch.float32
        )

    def limits(self):
        """A per-joint :class:`brainai.safety.JointLimits` for this bank.

        This is the point of the whole module: the safety controller stops
        applying one envelope to hardware that does not share one.
        """
        from .safety import JointLimits

        travel = self.tensor("max_travel")
        return JointLimits(
            n_actuators=self.n_actuators,
            lower=-travel,
            upper=travel,
            max_step=self.tensor("max_step"),
            max_effort=self.tensor("max_effort"),
        )

    def quantise(self, command: torch.Tensor) -> torch.Tensor:
        """Apply each joint's own achievable-position grid."""
        out = command.clone()
        for index, spec in enumerate(self.specs()):
            if spec.discrete_steps > 0:
                out[..., index] = spec.quantise(command[..., index])
        return out

    def hazardous(self) -> List[int]:
        """Joints that are stiff *and* strong - the ones that injure.

        A compliant actuator gives way against an unexpected obstacle. These do
        not, and they have the force budget to keep going through one, so they
        deserve tighter constraints than the rest of the body.
        """
        return [i for i, s in enumerate(self.specs()) if s.stiff_and_strong]

    def describe(self) -> str:
        rows = ["  kind             n  control     step  effort  compliance  bw",
                "  " + "-" * 62]
        for kind, count in self.layout:
            s = actuator(kind)
            rows.append(
                f"  {kind:<15}{count:>3}  {s.control:<9}{s.max_step:>7.2f}"
                f"{s.max_effort:>8.2f}{s.compliance:>12.2f}{s.bandwidth_hz:>6.0f}"
            )
        hazard = self.hazardous()
        rows.append(f"  {self.n_actuators} actuators, "
                    f"{len(hazard)} stiff-and-strong (tighter limits apply)")
        return "\n".join(rows)
