# BrainAI - built by Mohammed Zahid.
"""Confidence that survives the trip from perception to actuator.

The Reality Cortex has calibrated confidence and nothing else did.  Vision was
either right or wrong with no opinion on which; the planner produced actions
with no sense of whether the state it planned from was trustworthy.  So a blurry
frame and a clear one produced equally assertive motor commands.

This module carries a scalar confidence alongside each stage's output and
composes them.  Composition is **multiplicative with a floor**, not averaging:
a chain is no more trustworthy than its weakest link, and averaging lets a
confident planner paper over a perception failure it cannot see.

Where the numbers come from, cheapest first:

- **Predictive entropy** of a distribution -- free, no training, well founded.
- **Spike-rate deviation** -- an SNN layer firing far from its usual rate is
  out of distribution.  Also free.
- **Expert disagreement** -- when routed experts diverge, the model is
  internally unsure.  See :mod:`brainai.critique`.
- **Learned heads** where a supervision signal exists, as the reflex pathway
  already does.

The end-to-end number is handed to the safety controller, whose
``requires_confidence`` constraint refuses to act on a decision the system is
not sure about.  That closes the loop: uncertainty stops being a diagnostic and
becomes something that can veto a motor command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def entropy_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``1 - H(p) / log(n)`` -- 1 for a spike, 0 for uniform.

    Requires no training and no extra parameters: the distribution already
    contains the model's uncertainty, it just is not usually read out.
    """
    log_p = F.log_softmax(logits, dim=dim)
    entropy = -(log_p.exp() * log_p).sum(dim=dim)
    max_entropy = torch.log(
        torch.tensor(float(logits.shape[dim]), device=logits.device)
    )
    return (1.0 - entropy / max_entropy).clamp(0.0, 1.0)


def margin_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gap between the top two options, squashed to [0, 1].

    Complements entropy: a distribution can have moderate entropy while still
    having one clear winner, and for a decision that is what matters.
    """
    if logits.shape[dim] < 2:
        return torch.ones(logits.shape[:-1], device=logits.device)
    top2 = logits.topk(2, dim=dim).values
    return torch.sigmoid(top2.select(dim, 0) - top2.select(dim, 1))


def spike_rate_confidence(
    rate: float, expected: float = 0.03, tolerance: float = 0.05
) -> float:
    """How normal is this firing rate?

    A spiking layer running far from its usual sparsity is seeing something
    unlike its training distribution.  Cheap out-of-distribution signal that
    only exists because the network is spiking.
    """
    deviation = abs(rate - expected) / max(tolerance, 1e-6)
    return float(torch.exp(-torch.tensor(deviation)).clamp(0.0, 1.0))


@dataclass
class Uncertain:
    """A value with a confidence and a record of where it lost confidence."""

    value: torch.Tensor
    confidence: torch.Tensor
    stage: str = ""
    trace: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def certain(cls, value: torch.Tensor, stage: str = "") -> "Uncertain":
        batch = value.shape[0] if value.dim() else 1
        return cls(value, torch.ones(batch, device=value.device), stage)

    def then(
        self, value: torch.Tensor, confidence: torch.Tensor, stage: str
    ) -> "Uncertain":
        """Chain another stage on: confidences multiply."""
        conf = confidence.to(self.confidence.device).reshape(-1)
        if conf.numel() == 1:
            conf = conf.expand_as(self.confidence)
        trace = dict(self.trace)
        trace[stage] = float(conf.mean())
        return Uncertain(value, self.confidence * conf, stage, trace)

    @property
    def weakest_link(self) -> str:
        return min(self.trace, key=self.trace.get) if self.trace else self.stage

    def summary(self) -> str:
        chain = " -> ".join(f"{k} {v:.2f}" for k, v in self.trace.items())
        return (f"confidence {float(self.confidence.mean()):.3f}"
                f"  weakest: {self.weakest_link}"
                + (f"  [{chain}]" if chain else ""))


class UncertaintyTracker:
    """Collects per-stage confidence for one forward pass.

    Composition is multiplicative with a floor.  Pure multiplication over many
    stages drives everything to zero and stops being informative; the floor
    keeps a long chain of individually-confident stages usable while still
    letting one genuinely bad stage dominate.
    """

    def __init__(self, floor: float = 0.05) -> None:
        self.floor = floor
        self.stages: Dict[str, float] = {}

    def record(self, stage: str, confidence) -> float:
        value = (
            float(confidence.detach().mean())
            if isinstance(confidence, torch.Tensor) else float(confidence)
        )
        value = max(self.floor, min(1.0, value))
        self.stages[stage] = value
        return value

    def overall(self) -> float:
        if not self.stages:
            return 1.0
        product = 1.0
        for value in self.stages.values():
            product *= value
        # Geometric mean rather than raw product: otherwise a ten-stage
        # pipeline of 0.9s reports 0.35 and every action looks doubtful.
        return product ** (1.0 / len(self.stages))

    def bottleneck(self) -> str:
        return min(self.stages, key=self.stages.get) if self.stages else ""

    def report(self) -> Dict[str, float]:
        out = dict(self.stages)
        out["overall"] = self.overall()
        return out

    def summary(self) -> str:
        if not self.stages:
            return "no uncertainty recorded"
        chain = "  ".join(f"{k}={v:.2f}" for k, v in self.stages.items())
        return (f"overall {self.overall():.3f}  "
                f"bottleneck {self.bottleneck()}  |  {chain}")

    def reset(self) -> None:
        self.stages.clear()
