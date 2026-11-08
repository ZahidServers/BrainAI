# BrainAI - built by Mohammed Zahid.
"""AMSA - Adaptive Memory Spike Activation.

    f(x) = z . sigma(alpha.z) . (1 - exp(-beta.|z|)) . (1 + gamma.m)

with ``z = x - theta`` and

    theta = theta0 + lambda . p      (p = per-channel plasticity)
    alpha = softplus(alpha0 + delta . c)   (c = per-channel confidence)
    beta  = softplus(beta0)

Every one of ``theta0, alpha0, beta0, gamma, lambda, delta`` is a learned
per-channel parameter; ``m``, ``p`` and ``c`` are *state* buffers written by the
plasticity controller, not by gradient descent.  That is what makes the shape of
the activation itself change as the network's knowledge stabilises: confident
channels sharpen, plastic channels stay easy to move, and channels holding
important memories get amplified.

This is an original proposal, not a validated activation.  It is implemented
here so it can be measured -- ``benchmarks/activation_bench.py`` compares it
against GELU and SwiGLU on identical budgets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSA(nn.Module):
    """Adaptive Memory Spike Activation over the last dimension."""

    def __init__(
        self,
        width: int,
        gamma_init: float = 0.10,
        beta_init: float = 1.0,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.width = width

        # Learned shape parameters, one per channel.
        self.theta0 = nn.Parameter(torch.zeros(width))
        self.alpha0 = nn.Parameter(torch.full((width,), _inv_softplus(alpha_init)))
        self.beta0 = nn.Parameter(torch.full((width,), _inv_softplus(beta_init)))
        self.gamma = nn.Parameter(torch.full((width,), gamma_init))
        self.lam = nn.Parameter(torch.zeros(width))
        self.delta = nn.Parameter(torch.zeros(width))

        # Cognitive state, written by PlasticityController - never by autograd.
        self.register_buffer("memory", torch.zeros(width))
        self.register_buffer("plasticity", torch.ones(width))
        self.register_buffer("confidence", torch.zeros(width))
        # Running per-channel activity, the raw signal the controller turns
        # into `memory`.  Updated in the forward pass, outside autograd.
        self.register_buffer("activity", torch.zeros(width))

    def effective_theta(self) -> torch.Tensor:
        return self.theta0 + self.lam * self.plasticity

    def effective_alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha0 + self.delta * self.confidence) + 1e-4

    def effective_beta(self) -> torch.Tensor:
        return F.softplus(self.beta0) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x - self.effective_theta()
        gate = torch.sigmoid(self.effective_alpha() * z)
        # -expm1(-b|z|) == 1 - exp(-b|z|), but accurate for small b|z|.
        amp = -torch.expm1(-self.effective_beta() * z.abs())
        modulation = 1.0 + self.gamma * self.memory
        out = z * gate * amp * modulation
        if self.training:
            with torch.no_grad():
                dims = tuple(range(out.dim() - 1))
                self.activity.mul_(0.99).add_(
                    0.01 * out.detach().abs().mean(dim=dims)
                )
        return out

    @torch.no_grad()
    def set_state(
        self,
        memory: torch.Tensor | None = None,
        plasticity: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> None:
        """Write cognitive state.  Values are clamped to sane ranges."""
        if memory is not None:
            self.memory.copy_(memory.detach().reshape(-1)[: self.width].clamp(-1.0, 1.0))
        if plasticity is not None:
            self.plasticity.copy_(
                plasticity.detach().reshape(-1)[: self.width].clamp(0.0, 1.0)
            )
        if confidence is not None:
            self.confidence.copy_(
                confidence.detach().reshape(-1)[: self.width].clamp(0.0, 1.0)
            )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"width={self.width}"


class SwiGLU(nn.Module):
    """The standard gated feed-forward, for comparison and for use.

    Present because AMSA loses to it on every benchmark in this repo -- static
    perplexity, wall-clock, and continual learning, the last being the problem
    AMSA was designed for.  ``BrainConfig(ffn="swiglu")`` switches the whole
    model over, so the measurement is actionable rather than merely reported.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        d_out = d_out if d_out is not None else d_in
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class GatedAMSA(nn.Module):
    """SwiGLU-shaped feed-forward where the nonlinearity is AMSA.

    ``W2( AMSA(W1 x) * (Wg x) )`` -- the gate branch keeps the smooth,
    well-conditioned optimisation behaviour of SwiGLU while AMSA supplies the
    adaptive threshold and memory modulation.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        d_out = d_out if d_out is not None else d_in
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.act = AMSA(d_hidden)
        self.down = nn.Linear(d_hidden, d_out, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.up(x)) * torch.sigmoid(self.gate(x))
        return self.down(self.drop(h))


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) == y."""
    import math

    return math.log(math.expm1(y))
