# BrainAI - built by Mohammed Zahid.
"""Attention over spike trains.

Queries, keys and values are projected from *binary* spike tensors, so the
projection matmuls see mostly zeros -- that is where the event-driven energy
saving comes from on neuromorphic hardware, and where the sparsity statistics
in :func:`brainai.model.BrainAI.energy_report` come from on this one.

Two normalisers are available.  ``softmax`` is the stable default.
``sparsemax`` (Martins & Astudillo, 2016) produces exactly-zero attention
weights, which keeps the attention pattern itself sparse and matches the
event-driven story end to end.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Finite stand-in for -inf, so masked logits survive sparsemax's sort/cumsum.
NEG_MASK = -1e4


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection of ``z`` onto the probability simplex.

    Unlike softmax this returns exact zeros for low-scoring entries.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    zs, _ = torch.sort(z, dim=dim, descending=True)

    n = z.size(dim)
    shape = [1] * z.dim()
    shape[dim] = n
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype).view(shape)

    cumsum = zs.cumsum(dim)
    support = (1.0 + k * zs) > cumsum
    k_max = support.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1.0)

    tau = (cumsum.gather(dim, k_max.long() - 1) - 1.0) / k_max
    return torch.clamp(z - tau, min=0.0)


class SpikingAttention(nn.Module):
    """Multi-head attention whose inputs are spikes."""

    def __init__(
        self,
        width: int,
        heads: int,
        mode: str = "softmax",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} not divisible by heads {heads}")
        if mode not in ("softmax", "sparsemax"):
            raise ValueError(f"unknown attention mode {mode!r}")

        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.mode = mode
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.Linear(width, width, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        spikes: torch.Tensor,
        causal: bool = True,
        key_padding_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``spikes``: (B, S, W).  ``context``: optional (B, M, W) to attend to."""
        b, s, _ = spikes.shape
        src = spikes if context is None else context

        q = self._split(self.to_q(spikes), b, s)
        k = self._split(self.to_k(src), b, src.shape[1])
        v = self._split(self.to_v(src), b, src.shape[1])

        if self.mode == "softmax" and key_padding_mask is None:
            # Fused kernel: never materialises the (B, H, S, S) score matrix, which
            # at batch 32 / seq 256 x 3 timesteps x ~14 layers was ~14 GB and ran a
            # 15 GB T4 out of memory. Same maths as the explicit path below.
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop.p if isinstance(self.drop, nn.Dropout) and self.training else 0.0,
                is_causal=bool(causal and context is None),
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if causal and context is None:
                causal_mask = torch.ones(s, s, device=spikes.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(causal_mask, NEG_MASK)
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask[:, None, None, :], NEG_MASK
                )

            if self.mode == "softmax":
                attn = F.softmax(scores, dim=-1)
            else:
                attn = sparsemax(scores, dim=-1)
            attn = self.drop(attn)

            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, s, self.width)
        return self.to_out(out)

    def _split(self, t: torch.Tensor, b: int, s: int) -> torch.Tensor:
        return t.view(b, s, self.heads, self.head_dim).transpose(1, 2)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"width={self.width}, heads={self.heads}, mode={self.mode}"


class RMSNorm(nn.Module):
    """Root-mean-square layer norm - no mean subtraction, no bias."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
