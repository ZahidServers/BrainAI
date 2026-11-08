# BrainAI - built by Mohammed Zahid.
"""Adaptive Leaky Integrate-and-Fire neurons with surrogate gradients.

The spike itself is a Heaviside step -- non-differentiable, correct, and what
makes the network event-driven.  Training uses an arctan surrogate on the
backward pass only, so the forward computation stays genuinely binary and the
measured spike rate is a real sparsity number rather than a soft approximation.

Adaptation is what distinguishes ALIF from plain LIF: each spike raises the
neuron's own firing threshold, which decays back over ``tau_adapt``.  A neuron
that has just fired is harder to re-fire, giving the layer a short temporal
memory across the SNN timesteps without any recurrent weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ArcTanSpike(torch.autograd.Function):
    """Heaviside forward, arctan-derivative backward."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, slope: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (v,) = ctx.saved_tensors
        s = ctx.slope
        surrogate = s / (2.0 * (1.0 + (math.pi / 2.0 * s * v) ** 2))
        return grad_out * surrogate, None


def spike(v: torch.Tensor, slope: float = 2.0) -> torch.Tensor:
    """Emit a binary spike where ``v >= 0``, differentiable via surrogate."""
    return _ArcTanSpike.apply(v, slope)


class ALIF(nn.Module):
    """Adaptive Leaky Integrate-and-Fire layer.

    State is per-call: :meth:`reset` is invoked at the start of every multi-step
    sublayer pass, so no autograd graph is retained between forward calls.
    """

    def __init__(
        self,
        width: int,
        v_threshold: float = 1.0,
        adapt_scale: float = 0.18,
        tau_mem: float = 2.0,
        tau_adapt: float = 20.0,
        surrogate_slope: float = 2.0,
        learnable_dynamics: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.surrogate_slope = surrogate_slope

        mem_decay = math.exp(-1.0 / tau_mem)
        adapt_decay = math.exp(-1.0 / tau_adapt)

        def make(t: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(t, requires_grad=learnable_dynamics)

        self.mem_decay_logit = make(torch.full((width,), _logit(mem_decay)))
        self.adapt_decay_logit = make(torch.full((width,), _logit(adapt_decay)))
        self.v_threshold = make(torch.full((width,), v_threshold))
        self.adapt_scale = make(torch.full((width,), adapt_scale))

        self._v: torch.Tensor | None = None
        self._a: torch.Tensor | None = None
        self._s: torch.Tensor | None = None
        self.register_buffer("spike_rate", torch.zeros(()), persistent=False)

    def reset(self) -> None:
        """Clear membrane state.  Must be called before each timestep loop."""
        self._v = self._a = self._s = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """Integrate one timestep of input current and return binary spikes."""
        if self._v is None or self._v.shape != current.shape:
            self._v = torch.zeros_like(current)
            self._a = torch.zeros_like(current)
            self._s = torch.zeros_like(current)

        mem_decay = torch.sigmoid(self.mem_decay_logit)
        adapt_decay = torch.sigmoid(self.adapt_decay_logit)

        # Soft reset: subtract the threshold's worth of charge on the step after
        # a spike, rather than zeroing, which preserves sub-threshold detail.
        v = mem_decay * self._v * (1.0 - self._s) + (1.0 - mem_decay) * current
        a = adapt_decay * self._a + self._s
        threshold = self.v_threshold + self.adapt_scale * a

        s = spike(v - threshold, self.surrogate_slope)

        self._v, self._a, self._s = v, a, s
        self.spike_rate = s.detach().mean()
        return s

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"width={self.width}, slope={self.surrogate_slope}"


class SpikingLSTMCell(nn.Module):
    """LSTM cell whose hidden output is emitted as spikes.

    Used for short-term memory: gates stay analogue (they need smooth
    gradients), the exported hidden state is binary (it feeds spiking layers).
    """

    def __init__(self, d_in: int, d_hidden: int, surrogate_slope: float = 2.0) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.x2h = nn.Linear(d_in, 4 * d_hidden)
        self.h2h = nn.Linear(d_hidden, 4 * d_hidden, bias=False)
        # Threshold starts at 0, not 0.5.  h = o * tanh(c) is near zero at init,
        # so any positive threshold makes the readout completely silent -- and a
        # layer that emits all zeros gives whatever consumes it no gradient at
        # all.  At 0 roughly half the units fire and the threshold learns from
        # there.
        self.threshold = nn.Parameter(torch.zeros(d_hidden))
        self.surrogate_slope = surrogate_slope

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b = x.shape[0]
        if state is None:
            h = x.new_zeros(b, self.d_hidden)
            c = x.new_zeros(b, self.d_hidden)
        else:
            h, c = state

        gates = self.x2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        h = o * torch.tanh(c)
        s = spike(h - self.threshold, self.surrogate_slope)
        return s, (h, c)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))
