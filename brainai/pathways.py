# BrainAI - built by Mohammed Zahid.
"""The three response pathways, in order of latency.

    reflex        1 shallow stack, always runs, can answer on its own
    subconscious  always runs, sparse, decides whether thinking is needed
    conscious     expensive, gated, only runs when salience clears threshold

The gate is soft during training (so the executive can learn *when* thinking
pays) and hard at inference (so skipping the conscious pathway actually skips
the FLOPs).  ``ConsciousPathway`` selects the batch rows that need it and runs
only those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .attention import RMSNorm
from .blocks import SpikingStack
from .config import BrainConfig


@dataclass
class PathwayTrace:
    """What each pathway did on this forward pass."""

    salience: float = 0.0
    reflex_confidence: float = 0.0
    conscious_ran: bool = False
    conscious_fraction: float = 0.0
    reflex_weight: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        route = "reflex only" if not self.conscious_ran else (
            f"conscious ({self.conscious_fraction:.0%} of batch)"
        )
        return (
            f"salience={self.salience:.3f} "
            f"reflex_conf={self.reflex_confidence:.3f} "
            f"reflex_weight={self.reflex_weight:.3f} -> {route}"
        )


class ReflexPathway(nn.Module):
    """Spinal cord.

    One shallow stack straight from perception to a candidate output.  It never
    consults memory and never waits for the conscious pathway; its only job is
    to have *something* correct-enough available immediately.  It also reports
    how confident it is, which is what lets the executive skip deliberation.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.stack = SpikingStack("reflex", cfg)
        self.norm = RMSNorm(cfg.d_model)
        self.confidence = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(self.stack(x))
        conf = torch.sigmoid(self.confidence(h))     # (B, S, 1)
        return h, conf


class SubconsciousPathway(nn.Module):
    """Always-on background processing and the salience estimate.

    Everything passes through here.  Most inputs never go further -- that is
    the point, and it is where the compute budget is saved.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.stack = SpikingStack("subconscious", cfg)
        self.norm = RMSNorm(cfg.d_model)
        self.salience = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.Tanh(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(self.stack(x))
        # One salience score per sequence, pooled over time.
        sal = torch.sigmoid(self.salience(h.mean(dim=1)))   # (B, 1)
        return h, sal


class ConsciousPathway(nn.Module):
    """Deliberate reasoning: wide expert-routed stack plus a planner.

    Reads working memory before reasoning and writes back after, so a chain of
    forward passes can carry intermediate results the way a scratchpad does.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.reason = SpikingStack("conscious", cfg, use_experts=True)
        self.planner = SpikingStack("planner", cfg)
        self.ltm = SpikingStack("ltm_interface", cfg, causal=False)
        self.norm = RMSNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        working_read: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x if working_read is None else x + working_read
        if memory_context is not None:
            # Cross-attend to what was recalled; non-causal, it is not a sequence.
            h = h + self.ltm(h, context=memory_context)
        h = self.reason(h)
        h = self.planner(h)
        return self.norm(h)

    def aux_loss(self) -> torch.Tensor:
        return self.reason.aux_loss()


class MotorCortex(nn.Module):
    """Actuator commands for the robot body.

    Emits bounded joint targets plus a per-joint effort estimate.  Separate from
    the language head so the same cognitive core drives both speech and motion.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.stack = SpikingStack("motor", cfg)
        self.norm = RMSNorm(cfg.d_model)
        self.joints = nn.Linear(cfg.d_model, cfg.n_actuators)
        self.effort = nn.Linear(cfg.d_model, cfg.n_actuators)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.norm(self.stack(x))
        pooled = h[:, -1]                       # act on the latest state
        return {
            "joint_targets": torch.tanh(self.joints(pooled)),
            "effort": torch.sigmoid(self.effort(pooled)),
        }


class ExecutiveController(nn.Module):
    """Arbitration.

    Decides how much of the final answer comes from the reflex arc versus
    deliberation, given both candidates and the reflex's own confidence.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stack = SpikingStack("executive", cfg)
        self.arbiter = nn.Linear(2 * cfg.d_model + 1, 1)
        self.norm = RMSNorm(cfg.d_model)

    def forward(
        self,
        reflex_out: torch.Tensor,
        deliberate_out: torch.Tensor,
        reflex_conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = torch.cat([reflex_out, deliberate_out, reflex_conf], dim=-1)
        w = torch.sigmoid(self.arbiter(feats))          # (B, S, 1)
        mixed = w * reflex_out + (1.0 - w) * deliberate_out
        return self.norm(self.stack(mixed)), w


def run_gated(
    pathway: ConsciousPathway,
    x: torch.Tensor,
    salience: torch.Tensor,
    threshold: float,
    training: bool,
    memory_context: Optional[torch.Tensor] = None,
    working_read: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float]:
    """Run the conscious pathway, soft-gated in training, hard-gated at inference.

    Returns ``(output, fraction_of_batch_that_ran)``.
    """
    if training:
        out = pathway(x, memory_context, working_read)
        # Soft gate: the salience head gets gradient, so it learns when to fire.
        return salience.unsqueeze(-1) * out, 1.0

    active = (salience.reshape(-1) >= threshold).nonzero(as_tuple=True)[0]
    out = torch.zeros_like(x)
    if active.numel() == 0:
        return out, 0.0

    sub_ctx = None if memory_context is None else memory_context[active]
    sub_read = None if working_read is None else working_read[active]
    out[active] = pathway(x[active], sub_ctx, sub_read)
    return out, active.numel() / x.shape[0]
