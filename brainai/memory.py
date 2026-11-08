# BrainAI - built by Mohammed Zahid.
"""The memory hierarchy: sensory -> short-term -> working -> episodic/semantic.

Two of these are differentiable neural state (short-term, working) and two are
addressable stores that live in buffers (episodic, semantic).  That split is
deliberate and it is the whole anti-forgetting strategy: facts the system
learns after training go into the *stores*, where writing them cannot damage a
weight that some older skill depends on.  Only slow, importance-gated
consolidation is ever allowed to move knowledge from a store into the weights.

Because the stores are registered buffers they are saved and restored with the
checkpoint -- a robot's episodic history survives a restart.  At ``full`` scale
the capacities here (10M entries) should be backed by an external ANN index
rather than a dense tensor; :class:`VectorStore` keeps the same API either way.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import RMSNorm
from .config import BrainConfig
from .neurons import SpikingLSTMCell


class VectorStore(nn.Module):
    """Fixed-capacity associative store with importance-based eviction.

    Not differentiable by design.  Retrieval returns detached values that are
    injected as *context*, so a bad memory can mislead the model but can never
    corrupt a gradient.
    """

    def __init__(
        self,
        capacity: int,
        dim: int,
        name: str = "store",
        index_after: int = 500_000,
        n_tables: int = 8,
        n_bits: int = 10,
        n_probes: int = 3,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.name = name

        # Approximate index, built only once a brute-force scan stops being
        # cheap -- and that threshold is much higher than it looks.  Exact
        # search is a single BLAS matmul: measured at d=128, 12 ms for 50k
        # entries, 62 ms for 200k, 290 ms for 1M.  Python-side bucket assembly
        # cannot beat that below ~500k, so switching earlier trades recall for
        # a slowdown.
        #
        # The wall that actually forces an external index at full scale is
        # memory, not latency: these buffers are dense, so 1M entries costs
        # 512 MB at d=128 and ~8 GB at d=2048.  A 50M-entry store belongs on
        # disk behind FAISS/DiskANN; this index keeps the same API so the swap
        # is local.
        self.index_after = index_after
        self.n_tables = n_tables
        self.n_bits = n_bits
        # Multi-probe: also inspect buckets reached by flipping the least
        # confident bits.  A hash bit whose projection sits near zero was
        # nearly a coin toss, so the neighbouring bucket is about as likely to
        # hold the true neighbour -- checking a few of those recovers most of
        # the recall a single-probe scheme throws away.
        self.n_probes = n_probes
        self.register_buffer(
            "hyperplanes", torch.randn(n_tables, n_bits, dim) / (dim ** 0.5)
        )
        self.register_buffer(
            "codes", torch.zeros(n_tables, capacity, dtype=torch.long)
        )
        self._buckets: List[Dict[int, List[int]]] = [{} for _ in range(n_tables)]
        self._indexed = False

        self.register_buffer("keys", torch.zeros(capacity, dim))
        self.register_buffer("values", torch.zeros(capacity, dim))
        self.register_buffer("importance", torch.zeros(capacity))
        self.register_buffer("last_access", torch.zeros(capacity))
        self.register_buffer("hits", torch.zeros(capacity))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("clock", torch.zeros((), dtype=torch.long))

    @property
    def size(self) -> int:
        return int(self.occupied.sum())

    @torch.no_grad()
    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: torch.Tensor,
    ) -> List[int]:
        """Store a batch of (key, value) pairs.  Returns the slot indices used."""
        keys = F.normalize(keys.detach().float(), dim=-1)
        values = values.detach().float()
        importance = importance.detach().float().reshape(-1)

        written: List[int] = []
        for i in range(keys.shape[0]):
            slot = self._free_slot(importance[i])
            if slot is None:
                continue
            self.clock += 1
            self.keys[slot] = keys[i]
            self.values[slot] = values[i]
            self.importance[slot] = importance[i]
            self.last_access[slot] = self.clock.float()
            self.hits[slot] = 0.0
            self.occupied[slot] = True
            self._index_slot(slot)
            written.append(slot)
        return written

    # -- approximate index ------------------------------------------------
    @torch.no_grad()
    def _hash(self, vectors: torch.Tensor) -> torch.Tensor:
        """Sign of the projection onto random hyperplanes, packed into ints.

        Random-hyperplane LSH: the probability two vectors land in the same
        bucket falls off with the angle between them, which is exactly the
        cosine ordering retrieval already uses.
        """
        # (tables, bits, dim) x (n, dim) -> (tables, n, bits)
        projected = torch.einsum("tbd,nd->tnb", self.hyperplanes, vectors)
        bits = (projected > 0).long()
        weights = (2 ** torch.arange(self.n_bits, device=vectors.device)).long()
        return (bits * weights).sum(dim=-1)          # (tables, n)

    @torch.no_grad()
    def _index_slot(self, slot: int) -> None:
        if not self._indexed:
            if self.size < self.index_after:
                return
            self.rebuild_index()
            return
        codes = self._hash(self.keys[slot : slot + 1])
        for table in range(self.n_tables):
            code = int(codes[table, 0])
            self.codes[table, slot] = code
            self._buckets[table].setdefault(code, []).append(slot)

    @torch.no_grad()
    def rebuild_index(self) -> None:
        """(Re)build every hash table from the current contents."""
        idx = self.occupied.nonzero(as_tuple=True)[0]
        self._buckets = [{} for _ in range(self.n_tables)]
        if idx.numel():
            codes = self._hash(self.keys[idx])
            for table in range(self.n_tables):
                self.codes[table, idx] = codes[table]
                for position, slot in enumerate(idx.tolist()):
                    self._buckets[table].setdefault(
                        int(codes[table, position]), []
                    ).append(slot)
        self._indexed = True

    @torch.no_grad()
    def _candidates(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """Slots sharing a bucket with the query in any table, or None."""
        if not self._indexed:
            return None
        projected = torch.einsum("tbd,nd->tnb", self.hyperplanes, query)
        weights = (2 ** torch.arange(self.n_bits, device=query.device)).long()
        codes = ((projected > 0).long() * weights).sum(dim=-1)   # (tables, B)

        # The bits closest to the hyperplane are the ones most likely wrong.
        uncertain = projected.abs().argsort(dim=-1)[..., : self.n_probes]

        found: Set[int] = set()
        for table in range(self.n_tables):
            buckets = self._buckets[table]
            for b in range(query.shape[0]):
                base = int(codes[table, b])
                found.update(buckets.get(base, ()))
                for bit in uncertain[table, b].tolist():
                    found.update(buckets.get(base ^ (1 << bit), ()))
        if not found:
            return None
        slots = torch.tensor(sorted(found), device=query.device, dtype=torch.long)
        # Buckets can hold slots that were since evicted or merged away; the
        # scoring pass below would happily return a dead entry otherwise.
        return slots[self.occupied[slots]]

    def _free_slot(self, incoming_importance: torch.Tensor) -> Optional[int]:
        empty = (~self.occupied).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            return int(empty[0])

        # Full: evict the least valuable entry, where value combines stored
        # importance, how often it has been retrieved, and how recently.
        recency = (self.last_access - self.last_access.min()) / (
            self.last_access.max() - self.last_access.min() + 1e-6
        )
        value = self.importance + 0.3 * torch.log1p(self.hits) + 0.2 * recency
        victim = int(value.argmin())
        if float(self.importance[victim]) > float(incoming_importance):
            return None  # incoming memory is not worth what it would displace
        return victim

    @torch.no_grad()
    def search(
        self, query: torch.Tensor, k: int = 4, exact: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cosine retrieval.  Returns ``(values, scores, indices)``.

        ``query``: (B, dim).  ``values``: (B, k, dim), zero-padded when the
        store holds fewer than ``k`` entries.

        Uses the LSH index once the store is large enough, falling back to an
        exact scan otherwise or when the index returns too few candidates.
        ``exact=True`` forces a brute-force scan -- used to measure the index's
        recall against ground truth.
        """
        b = query.shape[0]
        k = max(1, k)
        if self.size == 0:
            return (
                query.new_zeros(b, k, self.dim),
                query.new_zeros(b, k),
                query.new_full((b, k), -1, dtype=torch.long),
            )

        q = F.normalize(query.detach().float(), dim=-1)

        candidates = self._candidates(q) if exact is False else None
        if candidates is not None and candidates.numel() >= k:
            # Score only the LSH candidates, then map back to real slot ids.
            sim_sub = q @ self.keys[candidates].t()
            k_eff = min(k, candidates.numel())
            scores, local = sim_sub.topk(k_eff, dim=-1)
            idx = candidates[local]
        else:
            sim = q @ self.keys.t()
            sim = sim.masked_fill(~self.occupied.unsqueeze(0), -1e4)
            k_eff = min(k, self.size)
            scores, idx = sim.topk(k_eff, dim=-1)
        vals = self.values[idx]

        self.hits.index_add_(
            0, idx.reshape(-1), torch.ones(idx.numel(), device=idx.device)
        )
        self.clock += 1
        self.last_access[idx.reshape(-1)] = self.clock.float()

        if k_eff < k:
            pad = k - k_eff
            vals = torch.cat([vals, vals.new_zeros(b, pad, self.dim)], dim=1)
            scores = torch.cat([scores, scores.new_zeros(b, pad)], dim=1)
            idx = torch.cat([idx, idx.new_full((b, pad), -1)], dim=1)
        return vals.to(query.dtype), scores.to(query.dtype), idx

    @torch.no_grad()
    def novelty(self, query: torch.Tensor) -> torch.Tensor:
        """1 - best cosine match, per batch item.  High means unfamiliar."""
        if self.size == 0:
            return torch.ones(query.shape[0], device=query.device)
        q = F.normalize(query.detach().float(), dim=-1)
        sim = (q @ self.keys.t()).masked_fill(~self.occupied.unsqueeze(0), -1.0)
        return (1.0 - sim.max(dim=-1).values.clamp(-1.0, 1.0)) / 2.0

    @torch.no_grad()
    def decay(self, rate: float = 0.999) -> None:
        """Importance decays; retrieval is what keeps a memory alive."""
        self.importance.mul_(rate)

    @torch.no_grad()
    def merge_duplicates(
        self, threshold: Optional[float] = None, max_compare: int = 4096
    ) -> int:
        """Fuse near-identical entries.  Returns how many slots were freed.

        ``threshold=None`` derives the cut from the store's own similarity
        distribution (``mean + 4 sigma``, capped at 0.999) instead of using a
        fixed constant.  This matters more than it looks: learned summaries are
        strongly collinear, so real duplicates sit at ~0.9998 while genuinely
        different episodes still score ~0.98.  A plausible-sounding fixed 0.97
        therefore merges the entire store into a handful of blobs, and it does
        so silently -- retrieval still returns *something*.

        Merging is destructive, so the estimate is deliberately conservative.
        """
        idx = self.occupied.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return 0
        if idx.numel() > max_compare:
            # O(n^2); sample a window. Sleep runs repeatedly and covers the
            # store over successive cycles.
            pick = torch.randperm(idx.numel(), device=idx.device)[:max_compare]
            idx = idx[pick]

        keys = self.keys[idx]
        sim = keys @ keys.t()
        sim.fill_diagonal_(-1.0)

        if threshold is None:
            off = sim[sim > -1.0]
            threshold = min(
                0.999, float(off.mean() + 4.0 * off.std(unbiased=False))
            )

        freed = 0
        dropped: Set[int] = set()
        pairs = (sim > threshold).nonzero(as_tuple=False)
        for a, b in pairs.tolist():
            if a >= b:
                continue
            ia, ib = int(idx[a]), int(idx[b])
            if ia in dropped or ib in dropped:
                continue
            keep, drop = (ia, ib) if self.importance[ia] >= self.importance[ib] else (ib, ia)
            # Absorb: average the values, keep the stronger importance.
            self.values[keep] = 0.5 * (self.values[keep] + self.values[drop])
            self.importance[keep] = torch.maximum(
                self.importance[keep], self.importance[drop]
            )
            self.hits[keep] += self.hits[drop]
            self.occupied[drop] = False
            self.importance[drop] = 0.0
            dropped.add(drop)
            freed += 1
        if freed and self._indexed:
            self.rebuild_index()
        return freed

    def stats(self) -> Dict[str, float]:
        n = self.size
        return {
            "entries": float(n),
            "capacity": float(self.capacity),
            "utilisation": n / self.capacity if self.capacity else 0.0,
            "mean_importance": float(self.importance[self.occupied].mean()) if n else 0.0,
            "total_hits": float(self.hits.sum()),
        }


class SensoryBuffer:
    """Milliseconds of raw perception, held for temporal integration.

    Plain Python deque: this is genuinely transient and should not be
    checkpointed.
    """

    def __init__(self, span: int) -> None:
        self.span = span
        self._frames: Deque[torch.Tensor] = deque(maxlen=span)

    def push(self, frame: torch.Tensor) -> None:
        if self._frames and self._frames[-1].shape != frame.shape:
            self._frames.clear()   # a different episode shape; don't mix them
        self._frames.append(frame.detach())

    def integrate(self) -> Optional[torch.Tensor]:
        """Recency-weighted mean of the buffer."""
        if not self._frames:
            return None
        frames = list(self._frames)
        weights = torch.linspace(0.5, 1.0, len(frames), device=frames[0].device)
        weights = weights / weights.sum()
        return sum(w * f for w, f in zip(weights, frames))

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


class ShortTermMemory(nn.Module):
    """Spiking-LSTM summary of the recent event stream."""

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.cell = SpikingLSTMCell(cfg.d_model, cfg.d_model, cfg.surrogate_slope)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model)
        self._state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset(self) -> None:
        self._state = None

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """``events``: (B, S, d).  Returns a (B, d) summary of the sequence."""
        state = self._state
        if state is not None and state[0].shape[0] != events.shape[0]:
            # Batch size changed, so the carried state belongs to a different
            # set of episodes.  Without this the LSTM silently broadcasts the
            # old state against the new input and returns the wrong batch size.
            state = None
        out = None
        for t in range(events.shape[1]):
            out, state = self.cell(events[:, t], state)
        self._state = tuple(s.detach() for s in state) if state is not None else None
        return self.norm(self.proj(out))


class WorkingMemory(nn.Module):
    """Addressable slots the conscious pathway reads from and writes to.

    Fully differentiable -- this is the scratchpad reasoning happens on, the
    carry bits in a multiplication, not long-term knowledge.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.slots = cfg.wm_slots
        self.dim = cfg.d_model
        self.init_slots = nn.Parameter(torch.randn(cfg.wm_slots, cfg.d_model) * 0.02)

        self.to_read_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.to_write_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.to_write_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.write_gate = nn.Linear(cfg.d_model, 1)
        self.norm = RMSNorm(cfg.d_model)
        self._content: Optional[torch.Tensor] = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._content = self.init_slots.unsqueeze(0).expand(batch, -1, -1).to(device)

    def content(self, batch: int, device: torch.device) -> torch.Tensor:
        if self._content is None or self._content.shape[0] != batch:
            self.reset(batch, device)
        assert self._content is not None
        return self._content

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """``query``: (B, S, d) -> attention-weighted read of the slots."""
        mem = self.content(query.shape[0], query.device)
        q = self.to_read_q(query)
        attn = F.softmax(q @ mem.transpose(1, 2) / (self.dim ** 0.5), dim=-1)
        return self.norm(attn @ mem)

    def write(self, state: torch.Tensor) -> None:
        """Gated write of a (B, S, d) state into the slots."""
        mem = self.content(state.shape[0], state.device)
        k = self.to_write_k(state)
        v = self.to_write_v(state)
        # Which slot does each token address?  (B, S, slots)
        addr = F.softmax(k @ mem.transpose(1, 2) / (self.dim ** 0.5), dim=-1)
        gate = torch.sigmoid(self.write_gate(state))          # (B, S, 1)
        update = (addr * gate).transpose(1, 2) @ v            # (B, slots, d)
        occupancy = (addr * gate).sum(dim=1).unsqueeze(-1)    # (B, slots, 1)
        # Graph is kept so a read-after-write in the same forward pass trains
        # these projections; BrainAI.forward calls detach() before returning so
        # nothing is carried into the next step.
        self._content = mem * (1.0 - occupancy.clamp(0, 1)) + update

    def detach(self) -> None:
        if self._content is not None:
            self._content = self._content.detach()


class SemanticMemory(VectorStore):
    """Concepts plus the edges between them.

    The graph is what makes this more than a second episodic buffer: retrieving
    "Paris" can pull in "France" without either vector being similar to the
    query.
    """

    def __init__(self, capacity: int, dim: int) -> None:
        super().__init__(capacity, dim, name="semantic")
        self.labels: Dict[int, str] = {}
        self.edges: Dict[int, Set[int]] = {}

    def relate(self, a: int, b: int) -> None:
        self.edges.setdefault(a, set()).add(b)
        self.edges.setdefault(b, set()).add(a)

    def neighbours(self, concept: int) -> Set[int]:
        return self.edges.get(concept, set())

    @torch.no_grad()
    def expand(self, indices: torch.Tensor, hops: int = 1) -> List[int]:
        """Graph-walk out from retrieved concepts."""
        frontier = {int(i) for i in indices.reshape(-1).tolist() if i >= 0}
        seen = set(frontier)
        for _ in range(hops):
            nxt: Set[int] = set()
            for c in frontier:
                nxt |= self.neighbours(c) - seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return sorted(seen)

    def label(self, index: int, text: str) -> None:
        self.labels[index] = text


class MemoryController(nn.Module):
    """The hippocampus: what gets remembered, what gets recalled, how strongly.

    Importance is scored from five signals -- novelty, surprise, reward,
    confidence and salience -- through a small learned head, so the policy for
    what is worth remembering is itself trained rather than hand-tuned.
    """

    FEATURES = ("novelty", "surprise", "reward", "confidence", "salience")

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.episodic = VectorStore(cfg.episodic_capacity, d, name="episodic")
        self.semantic = SemanticMemory(cfg.semantic_capacity, d)
        self.short_term = ShortTermMemory(cfg)
        self.working = WorkingMemory(cfg)
        self.sensory = SensoryBuffer(cfg.sensory_span)

        # Storage projections are FROZEN random projections, not learned.
        # This is not a shortcut: every key already in the store was written
        # with these weights.  If they were trained, each update would silently
        # invalidate the entire store, because old keys and new queries would no
        # longer live in the same space.  A random projection preserves cosine
        # geometry (Johnson-Lindenstrauss), which is all retrieval needs.
        self.to_key = nn.Linear(d, d, bias=False)
        self.to_value = nn.Linear(d, d, bias=False)
        self.to_key.weight.requires_grad_(False)
        self.to_value.weight.requires_grad_(False)

        # Trained during sleep against a prioritised-replay target, not by the
        # language loss - see BrainAI.sleep_step.
        self.scorer = nn.Sequential(
            nn.Linear(len(self.FEATURES), 16), nn.Tanh(), nn.Linear(16, 1)
        )
        # Start eager, learn to be selective.  An untrained scorer sits at 0.5,
        # just under the write threshold, so nothing would ever be stored and
        # the scorer would never see a training example.  Bias it to ~0.7 so an
        # empty memory fills; selectivity is what gets learned, not eagerness.
        nn.init.constant_(self.scorer[-1].bias, 0.85)

        self.retrieval_norm = RMSNorm(d)

        # Running mean of episode summaries, for centering.  Mean-pooled
        # sequence states share a large common component; without removing it
        # every pair of episodes has cosine similarity near 1, retrieval cannot
        # discriminate, and consolidation merges the whole store into one blob.
        self.register_buffer("summary_mean", torch.zeros(d))

    # -- addressing ------------------------------------------------------
    def key_of(self, summary: torch.Tensor, update: bool = False) -> torch.Tensor:
        """The address a summary is stored at and retrieved by.

        Every read and every write must go through this.  Storing
        ``to_key(summary)`` but searching with the raw ``summary`` puts queries
        and keys in different spaces, and cosine similarity between them is
        meaningless.
        """
        if update:
            with torch.no_grad():
                self.summary_mean.mul_(0.99).add_(
                    0.01 * summary.detach().mean(dim=0)
                )
        return self.to_key(summary - self.summary_mean)

    # -- retrieval -------------------------------------------------------
    def recall(self, query: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Pull episodic + semantic context for a (B, d) query.

        Returns a (B, 2k, d) context tensor for the LTM interface to attend to.
        """
        k = self.cfg.retrieval_top_k
        address = self.key_of(query)
        ep_vals, ep_scores, _ = self.episodic.search(address, k)
        se_vals, se_scores, se_idx = self.semantic.search(address, k)

        context = torch.cat([ep_vals, se_vals], dim=1)
        context = self.retrieval_norm(context)

        info = {
            "episodic_best": float(ep_scores[:, 0].mean()),
            "semantic_best": float(se_scores[:, 0].mean()),
            "episodic_entries": float(self.episodic.size),
            "semantic_entries": float(self.semantic.size),
            "graph_expanded": float(len(self.semantic.expand(se_idx))),
        }
        return context, info

    # -- importance ------------------------------------------------------
    def score_importance(
        self,
        summary: torch.Tensor,
        surprise: torch.Tensor,
        reward: torch.Tensor | float = 0.0,
        confidence: torch.Tensor | float = 0.5,
        salience: torch.Tensor | float = 0.5,
    ) -> torch.Tensor:
        """Per-item importance in [0, 1]."""
        b = summary.shape[0]
        dev = summary.device

        def col(x: torch.Tensor | float) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().float().reshape(b, 1).to(dev)
            return torch.full((b, 1), float(x), device=dev)

        novelty = self.episodic.novelty(self.key_of(summary)).reshape(b, 1)
        feats = torch.cat(
            [novelty, col(surprise), col(reward), col(confidence), col(salience)],
            dim=-1,
        )
        return torch.sigmoid(self.scorer(feats)).reshape(b)

    # -- write -----------------------------------------------------------
    @torch.no_grad()
    def remember(
        self,
        summary: torch.Tensor,
        importance: torch.Tensor,
        threshold: float | None = None,
    ) -> List[int]:
        """Commit episodes whose importance clears the threshold."""
        thr = self.cfg.importance_threshold if threshold is None else threshold
        keep = importance >= thr
        if not bool(keep.any()):
            return []
        sel = keep.nonzero(as_tuple=True)[0]
        return self.episodic.write(
            self.key_of(summary[sel], update=True),
            self.to_value(summary[sel]),
            importance[sel],
        )

    @torch.no_grad()
    def learn_concept(
        self, vector: torch.Tensor, label: str = "", related: List[int] | None = None
    ) -> int:
        """Add one semantic concept, optionally wired to existing ones."""
        v = vector.detach().reshape(1, -1)
        slots = self.semantic.write(
            self.key_of(v), self.to_value(v), torch.ones(1, device=v.device)
        )
        if not slots:
            return -1
        idx = slots[0]
        if label:
            self.semantic.label(idx, label)
        for other in related or []:
            self.semantic.relate(idx, other)
        return idx

    def reset_transient(self) -> None:
        """Clear per-episode state; the stores are untouched."""
        self.short_term.reset()
        self.working._content = None
        self.sensory.clear()

    def stats(self) -> Dict[str, Dict[str, float]]:
        return {"episodic": self.episodic.stats(), "semantic": self.semantic.stats()}
