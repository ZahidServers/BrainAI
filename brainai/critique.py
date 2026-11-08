# BrainAI - built by Mohammed Zahid.
"""Second thoughts: self-critique, and experts that are allowed to disagree.

A mixture of experts routes tokens and then *averages* the chosen experts'
outputs.  If two experts propose contradictory things, the average is a blend
of both -- and the fact that the model was internally torn is thrown away at
exactly the moment it was most informative.

:class:`InternalDebate` runs the routed experts separately instead of blending
them, scores each with a judge, and reports how far apart they were.  That
disagreement is one of the better-calibrated uncertainty signals available for
free: it is not the model's opinion about its confidence, it is a measurement of
whether its own components agree.

:class:`SelfCritique` is the outer loop -- generate, criticise, revise, repeat,
and keep the best version rather than the last one.  Revision does not
monotonically improve, so a loop that returns its final iterate is worse than
one that returns its best.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MoEFFN
from .uncertainty import entropy_confidence


@dataclass
class DebateResult:
    """What the experts said, and how much they disagreed."""

    winner: torch.Tensor
    scores: torch.Tensor              # (n_experts,) judge score per candidate
    disagreement: float               # 0 = unanimous, 1 = maximally split
    participants: List[int] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Agreement *is* the confidence signal."""
        return max(0.0, 1.0 - self.disagreement)

    def summary(self) -> str:
        return (f"{len(self.participants)} experts, "
                f"disagreement {self.disagreement:.3f}, "
                f"confidence {self.confidence:.3f}")


class InternalDebate(nn.Module):
    """Run routed experts independently and let a judge choose between them."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.judge = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @torch.no_grad()
    def deliberate(self, moe: MoEFFN, x: torch.Tensor, top_k: int = 3) -> DebateResult:
        """``x``: (B, S, d).  Returns the judged-best expert output."""
        moe.neuron.reset()
        spikes = moe.neuron(x)
        flat = spikes.reshape(-1, spikes.shape[-1])

        logits = moe.router(flat).masked_fill(~moe.active_experts, -1e4)
        active = moe.active_experts.nonzero(as_tuple=True)[0].tolist()
        chosen = sorted(
            active, key=lambda e: float(logits[:, e].mean()), reverse=True
        )[:top_k]
        if not chosen:
            return DebateResult(x, torch.zeros(1), 0.0, [])

        candidates = [moe.experts[e](flat).view_as(x) for e in chosen]
        scores = torch.stack([
            self.judge(c.mean(dim=1)).mean() for c in candidates
        ])

        # Disagreement: mean pairwise cosine distance between the proposals.
        pooled = torch.stack([c.mean(dim=(0, 1)) for c in candidates])
        normed = F.normalize(pooled, dim=-1)
        similarity = normed @ normed.t()
        n = len(chosen)
        if n > 1:
            off_diagonal = (similarity.sum() - similarity.diagonal().sum())
            mean_similarity = float(off_diagonal / (n * (n - 1)))
            disagreement = float(max(0.0, min(1.0, (1.0 - mean_similarity) / 2.0)))
        else:
            disagreement = 0.0

        winner = candidates[int(scores.argmax())]
        return DebateResult(winner, scores, disagreement, chosen)


@dataclass
class CritiqueRound:
    text: str
    score: float
    issues: List[str] = field(default_factory=list)


class SelfCritique:
    """Generate, criticise, revise -- and return the best round, not the last.

    The critic is a scoring callable rather than a learned head, because a
    critic trained on the same objective as the generator tends to agree with
    it.  Perplexity under the model itself is a weak but honest default: it
    catches degenerate output, and it does not pretend to judge truth.
    """

    def __init__(
        self,
        generate: Callable[[str], str],
        critic: Callable[[str], Tuple[float, List[str]]],
        rounds: int = 3,
        accept_score: float = 0.9,
    ) -> None:
        self.generate = generate
        self.critic = critic
        self.rounds = rounds
        self.accept_score = accept_score
        self.history: List[CritiqueRound] = []

    def run(self, prompt: str) -> CritiqueRound:
        self.history = []
        current_prompt = prompt

        for _ in range(self.rounds):
            text = self.generate(current_prompt)
            score, issues = self.critic(text)
            self.history.append(CritiqueRound(text, score, issues))

            if score >= self.accept_score or not issues:
                break
            current_prompt = (
                f"{prompt}\n\nPrevious attempt: {text}\n"
                f"Problems: {'; '.join(issues)}\nRevised:"
            )

        # Best, not last: revision is not monotone.
        return max(self.history, key=lambda r: r.score)

    def improvement(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return max(r.score for r in self.history) - self.history[0].score

    def report(self) -> Dict[str, float]:
        return {
            "rounds": float(len(self.history)),
            "first_score": self.history[0].score if self.history else 0.0,
            "best_score": max((r.score for r in self.history), default=0.0),
            "improvement": self.improvement(),
        }


def perplexity_critic(
    model, max_len: Optional[int] = None
) -> Callable[[str], Tuple[float, List[str]]]:
    """A default critic: score by the model's own perplexity, flag degeneracy.

    Deliberately modest.  It detects repetition, emptiness and gibberish -- the
    failure modes a small model actually has -- and makes no claim to judge
    whether the content is true.  Truth is the Reality Cortex's job.
    """
    # Must not exceed the model's positional embedding, or scoring a long
    # candidate crashes instead of criticising it.
    window = min(max_len or model.cfg.max_seq_len, model.cfg.max_seq_len)

    @torch.no_grad()
    def critic(text: str) -> Tuple[float, List[str]]:
        issues: List[str] = []
        stripped = text.strip()
        if not stripped:
            return 0.0, ["empty output"]

        words = stripped.split()
        if len(words) > 4:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                issues.append(f"repetitive ({unique_ratio:.0%} unique words)")

        tokens = model.tokenizer.encode_batch([stripped], window)
        loss, out = model.lm_loss(tokens, aux_weight=0.0, write_memory=False)
        perplexity = float(torch.exp(loss.detach().clamp(max=20.0)))
        if perplexity > 60.0:
            issues.append(f"high perplexity ({perplexity:.0f})")

        confidence = 0.0
        if out.logits is not None:
            confidence = float(entropy_confidence(out.logits).mean())
            if confidence < 0.05:
                issues.append("near-uniform token distribution")

        score = float(max(0.0, min(1.0, 1.0 - loss.detach().item() / 6.0)))
        return score, issues

    return critic
