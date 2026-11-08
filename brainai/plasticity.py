# BrainAI - built by Mohammed Zahid.
"""Per-parameter plasticity, importance tracking, and inference-time learning.

Every trainable weight carries three extra tensors:

    fisher      how much the loss cared about this weight on past tasks
    anchor      the value it settled on when that task was consolidated
    plasticity  1 - normalised fisher, floored: how far it is allowed to move

Weights that mattered become nearly frozen; weights that never mattered stay
free.  That is the anti-forgetting mechanism, and :meth:`ewc_penalty` is what
enforces it during ordinary training.

:meth:`online_step` is the part that updates weights *during inference*.  It is
deliberately conservative -- plasticity-scaled gradients, a hard trust region on
the update norm, and an optional verification callback that rolls the update
back if it made things worse.  Unbounded online updates destabilise the model
within a few hundred steps; this does not.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .activation import AMSA, GatedAMSA
from .blocks import MoEFFN
from .config import BrainConfig


def _key(name: str) -> str:
    return name.replace(".", "__")


def _robust_scale(t: torch.Tensor, q: float = 0.95, max_elems: int = 1_000_000) -> torch.Tensor:
    """High-quantile magnitude of ``t``, subsampling very large tensors.

    Normalising the Fisher by its *max* is useless in practice: the Fisher is
    heavy-tailed, so ``f / f.max()`` is near zero for almost every weight and
    nothing ends up protected.  A high quantile puts a meaningful slice of each
    tensor above the line.  ``torch.quantile`` also refuses inputs beyond ~16M
    elements, hence the subsample.
    """
    flat = t.detach().flatten().float()
    if flat.numel() == 0:
        return torch.ones((), device=t.device)
    if flat.numel() > max_elems:
        idx = torch.randint(0, flat.numel(), (max_elems,), device=flat.device)
        flat = flat[idx]
    return torch.quantile(flat, q).clamp_min(1e-12)


class PlasticityController(nn.Module):
    """Decides what is allowed to change, and by how much.

    Holds only buffers, never parameters, so constructing it against a model it
    is then attached to does not create a recursive parameter set.
    """

    def __init__(self, model: nn.Module, cfg: BrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._entries: List[Tuple[str, str]] = []
        self.register_buffer("consolidations", torch.zeros((), dtype=torch.long))
        # Calibration factor for the EWC penalty, set at consolidation time.
        self.register_buffer("ewc_scale", torch.ones(()))

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = _key(name)
            self.register_buffer(f"fisher__{k}", torch.zeros_like(p))
            self.register_buffer(f"anchor__{k}", p.detach().clone())
            self.register_buffer(f"plast__{k}", torch.ones_like(p))
            self._entries.append((name, k))

    # -- importance estimation -------------------------------------------
    @torch.no_grad()
    def accumulate_fisher(self, model: nn.Module, decay: float = 0.95) -> None:
        """Update the diagonal Fisher estimate.  Call after ``loss.backward()``."""
        params = dict(model.named_parameters())
        for name, k in self._entries:
            p = params.get(name)
            if p is None or p.grad is None:
                continue
            f = getattr(self, f"fisher__{k}")
            f.mul_(decay).addcmul_(p.grad, p.grad, value=1.0 - decay)

    @torch.no_grad()
    def consolidate(self, model: nn.Module) -> Dict[str, float]:
        """Freeze in what was just learned: re-anchor, recompute plasticity.

        Call at a task boundary, or at the end of a sleep cycle.
        """
        params = dict(model.named_parameters())
        floor = self.cfg.plasticity_floor
        protected = 0
        total = 0

        for name, k in self._entries:
            p = params.get(name)
            if p is None:
                continue
            f = getattr(self, f"fisher__{k}")
            norm_f = (f / _robust_scale(f)).clamp_(max=1.0)
            plast = (1.0 - norm_f).clamp_(min=floor, max=1.0)
            getattr(self, f"plast__{k}").copy_(plast)
            getattr(self, f"anchor__{k}").copy_(p.detach())
            protected += int((plast <= floor + 1e-6).sum())
            total += plast.numel()

        # Calibrate the penalty against a reference drift.
        #
        # A raw Fisher is uninterpretable as a loss term: at convergence the
        # squared gradients are ~1e-8, so `lambda * sum(F * dw^2)` lands around
        # 1e-4 against a task loss of ~2.5 and EWC silently does nothing --
        # which looks exactly like a working implementation until you compare it
        # against the no-EWC control.  So instead of picking lambda blind, scale
        # it here such that if every weight drifted by 1% of its own magnitude,
        # the penalty would be lambda/2.  lambda then has real units: "how many
        # nats is a 1% drift worth".
        ref = 0.0
        for name, k in self._entries:
            p = params.get(name)
            if p is None:
                continue
            f = getattr(self, f"fisher__{k}")
            ref += float((f * (0.01 * p.detach()).pow(2)).sum())
        self.ewc_scale.fill_(1.0 / max(ref, 1e-12))

        self.consolidations += 1
        self.sync_activations(model)
        return {
            "consolidations": float(self.consolidations),
            "ewc_scale": float(self.ewc_scale),
            "protected_weights": float(protected),
            "total_weights": float(total),
            "protected_fraction": protected / total if total else 0.0,
        }

    def ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        """Quadratic pull back toward the anchors, weighted by importance."""
        if int(self.consolidations) == 0:
            return torch.zeros((), device=next(model.parameters()).device)

        params = dict(model.named_parameters())
        total = None
        for name, k in self._entries:
            p = params.get(name)
            if p is None or not p.requires_grad:
                continue
            f = getattr(self, f"fisher__{k}")
            anchor = getattr(self, f"anchor__{k}")
            term = (f * (p - anchor).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), device=next(model.parameters()).device)
        return 0.5 * self.cfg.ewc_lambda * self.ewc_scale * total

    # -- push state into the activations ---------------------------------
    @torch.no_grad()
    def sync_activations(self, model: nn.Module) -> int:
        """Write per-channel memory / plasticity / confidence into every AMSA.

        This is what makes AMSA adaptive rather than just parameterised: the
        activation's threshold and gain follow the network's own estimate of
        which channels are load-bearing.
        """
        touched = 0
        for mod_name, module in model.named_modules():
            # Only AMSA has state to sync; a SwiGLU model simply has none.
            if not isinstance(module, GatedAMSA):
                continue
            k = _key(f"{mod_name}.up.weight" if mod_name else "up.weight")
            fisher = getattr(self, f"fisher__{k}", None)
            plast = getattr(self, f"plast__{k}", None)
            if fisher is None or plast is None:
                continue

            act: AMSA = module.act
            row_fisher = fisher.mean(dim=1)
            confidence = (row_fisher / _robust_scale(row_fisher)).clamp(max=1.0)
            activity = act.activity
            memory = (activity / _robust_scale(activity)).clamp(max=1.0)
            act.set_state(
                memory=memory,
                plasticity=plast.mean(dim=1),
                confidence=confidence,
            )
            touched += 1
        return touched

    # -- learning at inference time --------------------------------------
    def online_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        lr: Optional[float] = None,
        max_relative_norm: float = 0.01,
        verify: Optional[Callable[[], float]] = None,
    ) -> Dict[str, float]:
        """Take one plasticity-gated gradient step outside of training.

        ``loss`` must be built from the current forward pass with grad enabled.
        ``verify`` (optional) re-evaluates the loss after the step; if it got
        worse the update is rolled back.
        """
        lr = self.cfg.online_lr if lr is None else lr
        params = dict(model.named_parameters())

        named = [(n, k, params[n]) for n, k in self._entries
                 if n in params and params[n].requires_grad]
        grads = torch.autograd.grad(
            loss, [p for _, _, p in named], allow_unused=True, retain_graph=False
        )

        # Scale by plasticity first, then measure - the trust region should
        # bound what we actually apply, not the raw gradient.
        scaled: List[Optional[torch.Tensor]] = []
        sq = 0.0
        for (_, k, _), g in zip(named, grads):
            if g is None:
                scaled.append(None)
                continue
            s = g * getattr(self, f"plast__{k}")
            sq += float(s.pow(2).sum())
            scaled.append(s)
        update_norm = sq ** 0.5

        param_norm = sum(
            float(p.detach().pow(2).sum()) for _, _, p in named
        ) ** 0.5
        budget = max_relative_norm * max(param_norm, 1e-6)
        clip = min(1.0, budget / max(lr * update_norm, 1e-12))

        backup = None
        if verify is not None:
            backup = {n: p.detach().clone() for n, _, p in named}

        with torch.no_grad():
            for (_, _, p), s in zip(named, scaled):
                if s is not None:
                    p.add_(s, alpha=-lr * clip)

        result = {
            "update_norm": update_norm,
            "clip": clip,
            "applied_norm": lr * clip * update_norm,
            "rolled_back": 0.0,
        }

        if verify is not None and backup is not None:
            before = float(loss.detach())
            after = verify()
            result["loss_before"] = before
            result["loss_after"] = after
            if after > before:
                with torch.no_grad():
                    for n, _, p in named:
                        p.copy_(backup[n])
                result["rolled_back"] = 1.0
        return result

    # -- reporting -------------------------------------------------------
    def report(self) -> Dict[str, float]:
        floor = self.cfg.plasticity_floor
        protected = 0
        total = 0
        plast_sum = 0.0
        for _, k in self._entries:
            plast = getattr(self, f"plast__{k}")
            protected += int((plast <= floor + 1e-6).sum())
            total += plast.numel()
            plast_sum += float(plast.sum())
        return {
            "tracked_tensors": float(len(self._entries)),
            "tracked_weights": float(total),
            "protected_fraction": protected / total if total else 0.0,
            "mean_plasticity": plast_sum / total if total else 0.0,
            "consolidations": float(self.consolidations),
        }


class GrowthController(nn.Module):
    """Adds capacity when the existing network genuinely has no room for something.

    Growth is triggered by *sustained* novelty, not a single surprising input --
    a novelty EMA above threshold for ``patience`` observations wakes one
    dormant expert in each MoE layer.  Waking is seeded from the busiest expert
    so the new module starts useful instead of emitting noise.
    """

    def __init__(self, cfg: BrainConfig, patience: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.patience = patience
        self.register_buffer("novelty_ema", torch.zeros(()))
        self.register_buffer("streak", torch.zeros((), dtype=torch.long))
        self.register_buffer("grown", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def observe(self, novelty: torch.Tensor, model: nn.Module) -> Dict[str, float]:
        n = float(novelty.mean())
        self.novelty_ema.mul_(0.95).add_(0.05 * n)

        if float(self.novelty_ema) > self.cfg.novelty_threshold:
            self.streak += 1
        else:
            self.streak.zero_()

        woke = 0
        if int(self.streak) >= self.patience:
            for module in model.modules():
                if isinstance(module, MoEFFN) and module.wake_expert() >= 0:
                    woke += 1
            self.streak.zero_()
            if woke:
                self.grown += 1
        return {
            "novelty": n,
            "novelty_ema": float(self.novelty_ema),
            "streak": float(self.streak),
            "experts_woken": float(woke),
            "growth_events": float(self.grown),
        }

    def capacity(self, model: nn.Module) -> Dict[str, float]:
        active = total = 0
        for module in model.modules():
            if isinstance(module, MoEFFN):
                active += int(module.active_experts.sum())
                total += module.n_experts
        return {
            "active_experts": float(active),
            "total_experts": float(total),
            "headroom": (total - active) / total if total else 0.0,
        }
