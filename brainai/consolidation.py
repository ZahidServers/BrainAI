# BrainAI - built by Mohammed Zahid.
"""Sleep: offline replay, memory hygiene, and re-consolidation.

Nothing here runs while the system is responding.  During idle time it replays
stored experience against the *current* weights with the EWC penalty active,
merges redundant memories, lets unimportant ones decay, and then re-anchors --
so what was learned online during the day is folded into the weights slowly and
under supervision, instead of being written straight into them at inference.

The replay buffer keeps raw token sequences rather than embeddings, because
replaying an embedding only rehearses the layers above where it was captured.
Replaying tokens rehearses the whole network, which is the point.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import BrainConfig
from .plasticity import PlasticityController


class ReplayBuffer(nn.Module):
    """Importance-weighted experience store, balanced across tasks.

    When full it evicts the least important entry belonging to the *most
    represented* task, so a long run on one domain cannot squeeze an older
    domain out of the buffer entirely.  That balance is what makes replay work.
    """

    def __init__(self, capacity: int, seq_len: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.seq_len = seq_len
        self.register_buffer("tokens", torch.zeros(capacity, seq_len, dtype=torch.long))
        self.register_buffer("importance", torch.zeros(capacity))
        self.register_buffer("task", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))

    @property
    def size(self) -> int:
        return int(self.occupied.sum())

    def task_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for t in self.task[self.occupied].tolist():
            counts[t] = counts.get(t, 0) + 1
        return counts

    @torch.no_grad()
    def add(
        self,
        tokens: torch.Tensor,
        importance: torch.Tensor | float = 1.0,
        task_id: int = 0,
    ) -> int:
        """Store a (B, S) batch of sequences.  Returns how many were kept."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens[:, : self.seq_len]
        if tokens.shape[1] < self.seq_len:
            pad = self.seq_len - tokens.shape[1]
            tokens = torch.cat(
                [tokens, tokens.new_zeros(tokens.shape[0], pad)], dim=1
            )

        if isinstance(importance, torch.Tensor):
            imps = importance.detach().float().reshape(-1)
        else:
            imps = torch.full((tokens.shape[0],), float(importance))

        kept = 0
        for i in range(tokens.shape[0]):
            slot = self._slot(float(imps[i]))
            if slot is None:
                continue
            self.tokens[slot] = tokens[i].to(self.tokens.device)
            self.importance[slot] = imps[i]
            self.task[slot] = task_id
            self.occupied[slot] = True
            kept += 1
        return kept

    def _slot(self, incoming: float) -> Optional[int]:
        empty = (~self.occupied).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            return int(empty[0])

        counts = self.task_counts()
        if not counts:
            return 0
        crowded = max(counts, key=lambda t: counts[t])
        mask = self.occupied & (self.task == crowded)
        idx = mask.nonzero(as_tuple=True)[0]
        victim = int(idx[self.importance[idx].argmin()])
        return victim if float(self.importance[victim]) <= incoming else None

    @torch.no_grad()
    def sample(self, n: int) -> Optional[torch.Tensor]:
        """Importance-weighted sample of ``n`` sequences, or None if empty."""
        if self.size == 0:
            return None
        idx = self.occupied.nonzero(as_tuple=True)[0]
        weights = self.importance[idx].clamp_min(1e-4)
        pick = torch.multinomial(weights, min(n, idx.numel()), replacement=idx.numel() < n)
        return self.tokens[idx[pick]]

    def stats(self) -> Dict[str, float]:
        counts = self.task_counts()
        return {
            "entries": float(self.size),
            "capacity": float(self.capacity),
            "tasks": float(len(counts)),
            "balance": (min(counts.values()) / max(counts.values()))
            if len(counts) > 1 else 1.0,
        }


class SleepConsolidator:
    """Runs the offline consolidation cycle."""

    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        self.cycles_run = 0

    def sleep(
        self,
        model: nn.Module,
        replay: ReplayBuffer,
        plasticity: PlasticityController,
        step_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, float]]],
        cycles: int = 8,
        batch_size: int = 4,
        lr: float = 3e-5,
        memory_controller: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """One full sleep: replay -> hygiene -> re-consolidate.

        ``step_fn`` maps a (B, S) token batch to ``(loss, components)``.
        """
        report: Dict[str, float] = {"cycles": float(cycles)}
        was_training = model.training
        model.train()

        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        # -- phase 1: replay against the current weights, EWC active --------
        losses: List[float] = []
        scorer_losses: List[float] = []
        for _ in range(cycles):
            batch = replay.sample(batch_size)
            if batch is None:
                break
            opt.zero_grad(set_to_none=True)
            task_loss, parts = step_fn(batch)
            total = task_loss + plasticity.ewc_penalty(model)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            plasticity.accumulate_fisher(model)
            opt.step()
            losses.append(parts.get("lm", float(task_loss.detach())))
            if "importance" in parts:
                scorer_losses.append(parts["importance"])

        if losses:
            report["replay_loss_first"] = losses[0]
            report["replay_loss_last"] = losses[-1]
            report["replay_loss_mean"] = sum(losses) / len(losses)
        if scorer_losses:
            report["scorer_loss_first"] = scorer_losses[0]
            report["scorer_loss_last"] = scorer_losses[-1]
        report["replayed_batches"] = float(len(losses))

        # -- phase 2: memory hygiene ----------------------------------------
        if memory_controller is not None:
            merged_ep = memory_controller.episodic.merge_duplicates()
            merged_se = memory_controller.semantic.merge_duplicates()
            memory_controller.episodic.decay(0.995)
            memory_controller.semantic.decay(0.999)
            report["episodic_merged"] = float(merged_ep)
            report["semantic_merged"] = float(merged_se)

        # -- phase 3: re-anchor and recompute plasticity --------------------
        report.update(plasticity.consolidate(model))

        self.cycles_run += 1
        report["sleeps"] = float(self.cycles_run)
        if not was_training:
            model.eval()
        return report
