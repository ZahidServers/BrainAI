# BrainAI - built by Mohammed Zahid.
"""Spiking transformer blocks and expert routing.

Each sublayer runs the SNN for ``T`` timesteps over a *constant* input current
and returns the mean of its per-timestep outputs -- rate coding.  Because the
ALIF threshold adapts within those timesteps the spike pattern is genuinely
time-varying, so T > 1 buys temporal structure rather than T copies of the same
computation.

The residual stream stays analogue; only the sublayer interiors are binary.
Trying to make the residual stream itself spiking destroys the gradient path
and the model does not train.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import GatedAMSA, SwiGLU
from .attention import RMSNorm, SpikingAttention
from .config import BrainConfig
from .neurons import ALIF


def make_ffn(cfg: BrainConfig, d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    """The feed-forward the config asked for.

    Every block goes through here so ``BrainConfig(ffn="swiglu")`` swaps the
    whole model, including the mixture-of-experts pool.
    """
    if cfg.ffn == "swiglu":
        return SwiGLU(d_in, d_hidden, d_out, cfg.dropout)
    if cfg.ffn == "amsa":
        return GatedAMSA(d_in, d_hidden, d_out, cfg.dropout)
    raise ValueError(f"unknown ffn {cfg.ffn!r}; choose 'amsa' or 'swiglu'")


class SpikingAttentionSublayer(nn.Module):
    """ALIF encode -> attention -> accumulate, over T timesteps."""

    def __init__(self, width: int, heads: int, cfg: BrainConfig) -> None:
        super().__init__()
        self.timesteps = cfg.timesteps
        self.neuron = ALIF(
            width,
            v_threshold=cfg.v_threshold,
            adapt_scale=cfg.adapt_scale,
            surrogate_slope=cfg.surrogate_slope,
        )
        self.attn = SpikingAttention(width, heads, cfg.attention, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.neuron.reset()
        acc = torch.zeros_like(x)
        for _ in range(self.timesteps):
            s = self.neuron(x)
            acc = acc + self.attn(s, causal=causal, context=context)
        return acc / self.timesteps


class SpikingFFNSublayer(nn.Module):
    """ALIF encode -> gated AMSA feed-forward -> accumulate, over T timesteps."""

    def __init__(self, width: int, cfg: BrainConfig, hidden_mult: int = 4) -> None:
        super().__init__()
        self.timesteps = cfg.timesteps
        self.neuron = ALIF(
            width,
            v_threshold=cfg.v_threshold,
            adapt_scale=cfg.adapt_scale,
            surrogate_slope=cfg.surrogate_slope,
        )
        self.ffn = make_ffn(cfg, width, hidden_mult * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.neuron.reset()
        acc = torch.zeros_like(x)
        for _ in range(self.timesteps):
            acc = acc + self.ffn(self.neuron(x))
        return acc / self.timesteps


class MoEFFN(nn.Module):
    """Top-k routed mixture of AMSA experts with a load-balancing loss.

    Only the selected experts run, so adding domain experts adds capacity
    without adding per-token compute.
    """

    def __init__(self, width: int, cfg: BrainConfig, hidden_mult: int = 4) -> None:
        super().__init__()
        self.timesteps = cfg.timesteps
        self.n_experts = cfg.n_experts
        self.top_k = min(cfg.expert_top_k, cfg.n_experts)
        self.neuron = ALIF(
            width,
            v_threshold=cfg.v_threshold,
            adapt_scale=cfg.adapt_scale,
            surrogate_slope=cfg.surrogate_slope,
        )
        self.router = nn.Linear(width, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [make_ffn(cfg, width, hidden_mult * width, width)
             for _ in range(cfg.n_experts)]
        )
        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)
        self.register_buffer(
            "expert_load", torch.zeros(cfg.n_experts), persistent=False
        )
        # Growth: experts start dormant beyond an initial pool and are woken by
        # GrowthController when sustained novelty finds no good existing fit.
        active = torch.zeros(cfg.n_experts, dtype=torch.bool)
        active[: max(self.top_k, cfg.n_experts // 2)] = True
        self.register_buffer("active_experts", active)

    @torch.no_grad()
    def wake_expert(self) -> int:
        """Activate one dormant expert.  Returns its index, or -1 if none left."""
        dormant = (~self.active_experts).nonzero(as_tuple=True)[0]
        if dormant.numel() == 0:
            return -1
        idx = int(dormant[0])
        self.active_experts[idx] = True
        # Seed it from the busiest active expert so it starts useful, then
        # diverges - a fresh random expert would just emit noise for a while.
        donor = int(self.expert_load.argmax())
        for tgt, src in zip(self.experts[idx].parameters(),
                            self.experts[donor].parameters()):
            tgt.copy_(src + 0.01 * torch.randn_like(src))
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.neuron.reset()
        acc = torch.zeros_like(x)
        aux = x.new_zeros(())
        for _ in range(self.timesteps):
            out, step_aux = self._route(self.neuron(x))
            acc = acc + out
            aux = aux + step_aux
        self.aux_loss = aux / self.timesteps
        return acc / self.timesteps

    def _route(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, seq, w = s.shape
        flat = s.reshape(-1, w)

        n_active = int(self.active_experts.sum())
        k = max(1, min(self.top_k, n_active))

        logits = self.router(flat).masked_fill(~self.active_experts, -1e4)
        probs = F.softmax(logits, dim=-1)
        top_val, top_idx = probs.topk(k, dim=-1)
        top_val = top_val / top_val.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            hit = top_idx == e                       # (N, k)
            token_sel = hit.any(dim=-1)
            if not bool(token_sel.any()):
                continue
            idx = token_sel.nonzero(as_tuple=True)[0]
            weight = (top_val * hit).sum(dim=-1)[idx].unsqueeze(-1)
            out.index_add_(0, idx, weight * self.experts[e](flat[idx]))

        # Switch-Transformer load balance: fraction routed x mean router prob.
        with torch.no_grad():
            frac = torch.zeros(self.n_experts, device=s.device)
            frac.scatter_add_(
                0,
                top_idx.reshape(-1),
                torch.ones_like(top_idx.reshape(-1), dtype=frac.dtype),
            )
            frac = frac / frac.sum().clamp_min(1.0)
            self.expert_load = frac
        aux = n_active * (frac * probs.mean(dim=0)).sum()

        return out.view(b, seq, w), aux


class SpikingTransformerBlock(nn.Module):
    """Pre-norm spiking block: attention sublayer then feed-forward sublayer."""

    def __init__(
        self,
        width: int,
        heads: int,
        cfg: BrainConfig,
        use_experts: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SpikingAttentionSublayer(width, heads, cfg)
        self.norm2 = RMSNorm(width)
        self.ffn: nn.Module = (
            MoEFFN(width, cfg) if use_experts else SpikingFFNSublayer(width, cfg)
        )
        self.uses_experts = use_experts

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal, context=context)
        x = x + self.ffn(self.norm2(x))
        return x


class SpikingStack(nn.Module):
    """A named section: project into the section width, run N blocks, project back.

    Sections have their own widths (the conscious pathway is wider than the
    reflex arc) but all communicate over the shared ``d_model`` bus.
    """

    def __init__(
        self,
        name: str,
        cfg: BrainConfig,
        use_experts: bool = False,
        causal: bool = True,
    ) -> None:
        super().__init__()
        spec = cfg.section(name)
        self.name = name
        self.causal = causal
        self.width = spec.width

        self.in_proj = (
            nn.Identity() if spec.width == cfg.d_model
            else nn.Linear(cfg.d_model, spec.width, bias=False)
        )
        self.blocks = nn.ModuleList(
            [SpikingTransformerBlock(spec.width, spec.heads, cfg, use_experts)
             for _ in range(spec.layers)]
        )
        self.norm = RMSNorm(spec.width)
        self.out_proj = (
            nn.Identity() if spec.width == cfg.d_model
            else nn.Linear(spec.width, cfg.d_model, bias=False)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, causal=self.causal, context=context)
        return self.out_proj(self.norm(h))

    def aux_loss(self) -> torch.Tensor:
        total = None
        for block in self.blocks:
            if block.uses_experts:
                a = block.ffn.aux_loss
                total = a if total is None else total + a
        return total if total is not None else torch.zeros(())


def mean_spike_rate(module: nn.Module) -> float:
    """Average firing rate across every ALIF layer in ``module``.

    This is the sparsity figure: 0.08 means 8% of neurons fired, so 92% of the
    downstream multiply-accumulates were against zero.
    """
    rates = [
        float(m.spike_rate) for m in module.modules() if isinstance(m, ALIF)
    ]
    return sum(rates) / len(rates) if rates else 0.0
