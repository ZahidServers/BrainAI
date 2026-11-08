# BrainAI - built by Mohammed Zahid.
"""The Reality Cortex: deciding what is *true*, not just what is important.

Salience and truth are different questions.  The subconscious answers "is this
worth attending to"; a claim shouted by a thousand people scores high on that
and can still be false.  Everything upstream of here -- episodic writes,
importance scoring, replay -- is truth-agnostic by design.  This module is what
stands between "the robot heard something" and "the robot believes it".

It sits on the *knowledge* path, not the token path::

    episodic memory  ->  Reality Cortex  ->  semantic memory  ->  sleep  ->  weights

Not in the per-token forward pass: contradiction detection over a belief graph
is the wrong thing to run on every token, and would be ruinously slow if it
were.  Beliefs re-enter the forward pass through retrieval, where confidence
weights what gets recalled.

Four mechanisms do the work:

**Log-odds evidence pooling.**  Each piece of evidence contributes
``log(t / (1 - t))`` where ``t`` is its source's reliability -- the standard
naive-Bayes likelihood ratio.  Refutation is the same quantity with a negative
sign, so evidence for and against accumulate on one scale.

**Per-source diminishing returns.**  The nth claim from the *same* source counts
``1/n`` as much as the first.  This is the part that matters most for a robot
that lives somewhere: repetition from one voice saturates (harmonic growth),
while agreement between *independent* sources compounds.  Without it, a system
that learns from its environment simply adopts whatever it hears most, which is
the failure the whole module exists to prevent.

**Rival-relative confidence.**  For functional relations -- ones with a single
correct object, like ``capital_of`` -- confidence is
``sigmoid(logit - best_rival_logit)``.  Two equally supported alternatives sit
at 0.5 and neither gets promoted, which is the honest answer.

**Protected axioms.**  Seeded beliefs carry a large prior log-odds, so
overturning one requires overwhelming independent evidence rather than a
confident stranger.  Adaptable is not the same as credulous.

Beliefs are Python objects, not tensors, and are serialised into the checkpoint
alongside the weights -- what the robot believes must survive a restart.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HYPOTHESIS = "hypothesis"
BELIEVED = "believed"
OBSOLETE = "obsolete"

#: What a claim is *about*.  Triples handle facts well and other things badly,
#: so procedures and causal links carry structured payloads alongside the
#: triple rather than being forced into one.
FACT = "fact"
PROCEDURE = "procedure"
CAUSAL = "causal"

#: Subject domains, for tagging claims in the *single* semantic graph.
#:
#: Deliberately a tag and not a separate store per subject.  Real concepts span
#: domains -- Einstein is physics and history and politics; a titration is
#: chemistry and procedure and safety -- and filing them under one heading is
#: what destroys the cross-domain links that make the graph worth having.
#: Tagging lets you *filter* by domain while keeping one connected graph.
DOMAINS = (
    "mathematics", "physics", "chemistry", "biology", "computing",
    "history", "geography", "arts", "language", "engineering",
    "medicine", "everyday",
)

#: Domains where a claim should not be believed on social consensus alone.
#: A mathematical result is not more true for being popular, and a chemistry
#: fact that is wrong can hurt someone.
RIGOROUS_DOMAINS = frozenset(
    {"mathematics", "physics", "chemistry", "biology", "medicine", "engineering"}
)

#: Relations that admit exactly one object.  A second object contradicts the
#: first rather than adding to it.
#:
#: Arity is a modelling decision, not something to infer, and getting it wrong
#: fails in both directions: mark a many-valued relation functional and real
#: facts suppress each other; leave a single-valued one out and the system
#: happily believes a door is locked *and* unlocked.  Neither default is safe,
#: so declare the relations your deployment uses -- see
#: :meth:`RealityCortex.declare_functional`.
DEFAULT_FUNCTIONAL = frozenset({
    "capital_of", "is_capital_of", "born_in", "equals", "located_in",
    "torque_spec", "is_a", "colour_of", "owner_of",
    "state_of", "status_of", "position_of", "temperature_of", "value_of",
    "mass_of", "shape_of", "material_of", "size_of", "count_of",
})


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Source:
    """A dynamic reliability estimate for one information source.

    Trust is the mean of a Beta posterior over "how often is this source
    right", seeded from ``prior`` with ``strength`` pseudo-observations.  A
    source that is repeatedly wrong loses trust, and every claim it ever
    supported loses confidence with it.
    """

    name: str
    prior: float = 0.5
    strength: float = 4.0
    correct: float = 0.0
    incorrect: float = 0.0

    @property
    def trust(self) -> float:
        a = self.prior * self.strength + self.correct
        b = (1.0 - self.prior) * self.strength + self.incorrect
        return a / (a + b)

    @property
    def observations(self) -> int:
        return int(self.correct + self.incorrect)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "prior": self.prior, "strength": self.strength,
                "correct": self.correct, "incorrect": self.incorrect}


@dataclass
class Evidence:
    source: str
    supports: bool = True
    note: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "supports": self.supports,
                "note": self.note, "at": self.at}


@dataclass
class Claim:
    """One proposition, with its evidence and its history."""

    subject: str
    relation: str
    object: str
    text: str = ""
    status: str = HYPOTHESIS
    prior_logit: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    concept_id: int = -1            # slot in SemanticMemory once promoted
    superseded_by: int = -1
    protected: bool = False
    kind: str = FACT
    #: Subject tag.  One graph, filtered -- not one store per subject.
    domain: str = "everyday"
    #: Structured content for non-fact kinds: ordered ``steps`` for a
    #: procedure, ``strength`` and ``mechanism`` for a causal link.
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.subject, self.relation)

    @property
    def triple(self) -> str:
        return f"{self.subject} --{self.relation}--> {self.object}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject, "relation": self.relation,
            "object": self.object, "text": self.text, "status": self.status,
            "prior_logit": self.prior_logit, "concept_id": self.concept_id,
            "superseded_by": self.superseded_by, "protected": self.protected,
            "kind": self.kind, "domain": self.domain, "payload": self.payload,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Claim":
        d = dict(d)
        ev = [Evidence(**e) for e in d.pop("evidence", [])]
        return cls(evidence=ev, **d)


@dataclass
class ClaimReport:
    """What happened when a claim was submitted."""

    claim_id: int
    triple: str
    confidence: float
    status: str
    promoted: bool = False
    contradicts: List[int] = field(default_factory=list)
    superseded: List[int] = field(default_factory=list)
    reason: str = ""

    def summary(self) -> str:
        bits = [f"{self.triple}  conf={self.confidence:.3f}  {self.status}"]
        if self.contradicts:
            bits.append(f"contradicts {self.contradicts}")
        if self.superseded:
            bits.append(f"superseded {self.superseded}")
        if self.reason:
            bits.append(self.reason)
        return "  ".join(bits)


class RealityCortex:
    """Evidence-weighted belief maintenance."""

    def __init__(
        self,
        semantic_threshold: float = 0.75,
        weight_threshold: float = 0.90,
        min_independent_sources: int = 2,
        functional_relations: Sequence[str] = (),
        protected_prior_logit: float = 12.0,
        credibility_scale: float = 4.0,
        correlation_discount: str = "sqrt",
        cluster_correlated_sources: bool = True,
        cluster_overlap: float = 0.85,
        cluster_min_votes: int = 3,
        cluster_max_pairwise: int = 256,
        cluster_refresh_every: int = 25,
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.weight_threshold = weight_threshold
        self.min_independent_sources = min_independent_sources
        self.functional = set(functional_relations) | set(DEFAULT_FUNCTIONAL)
        self.protected_prior_logit = protected_prior_logit
        self.credibility_scale = credibility_scale
        self.correlation_discount = correlation_discount
        self.cluster_correlated_sources = cluster_correlated_sources
        self.cluster_overlap = cluster_overlap
        self.cluster_min_votes = cluster_min_votes
        self.cluster_max_pairwise = cluster_max_pairwise
        self.cluster_refresh_every = cluster_refresh_every
        self._submits_since_cluster = 0

        self.sources: Dict[str, Source] = {}
        self.claims: List[Claim] = []
        self._index: Dict[Tuple[str, str], List[int]] = {}

        # Voting record per source, for collusion detection.
        self._votes: Dict[str, Dict[int, bool]] = {}
        self._cluster_of: Dict[str, str] = {}
        self._clusters_dirty = True

    # -- sources ---------------------------------------------------------
    def register_source(
        self, name: str, prior: float = 0.5, strength: float = 4.0
    ) -> Source:
        src = self.sources.get(name)
        if src is None:
            src = Source(name=name, prior=prior, strength=strength)
            self.sources[name] = src
        return src

    def trust(self, name: str) -> float:
        return self.register_source(name).trust

    # -- relation arity --------------------------------------------------
    def declare_functional(self, *relations: str) -> "RealityCortex":
        """Mark relations as single-valued, so alternatives compete."""
        self.functional.update(relations)
        return self

    def declare_multivalued(self, *relations: str) -> "RealityCortex":
        """Mark relations as many-valued, so alternatives coexist."""
        self.functional.difference_update(relations)
        return self

    def is_functional(self, relation: str) -> bool:
        return relation in self.functional

    # -- submitting claims -----------------------------------------------
    def submit(
        self,
        subject: str,
        relation: str,
        obj: str,
        source: str,
        supports: bool = True,
        text: str = "",
        note: str = "",
    ) -> ClaimReport:
        """Record one piece of evidence for or against a proposition."""
        self.register_source(source)
        claim_id = self._find_or_create(subject, relation, obj, text)
        claim = self.claims[claim_id]
        claim.evidence.append(Evidence(source=source, supports=supports, note=note))
        self._votes.setdefault(source, {})[claim_id] = supports
        self._submits_since_cluster += 1

        conf = self.confidence(claim_id)
        rivals = [i for i in self._rivals(claim_id)]
        report = ClaimReport(
            claim_id=claim_id,
            triple=claim.triple,
            confidence=conf,
            status=claim.status,
            contradicts=[i for i in rivals if self.confidence(i) > 0.5],
        )

        if claim.status == OBSOLETE and conf > self.semantic_threshold:
            claim.status = HYPOTHESIS
            report.reason = "revived: new evidence for a retired belief"

        report.status = claim.status
        return report

    def assert_axiom(
        self, subject: str, relation: str, obj: str, text: str = ""
    ) -> int:
        """Seed protected core knowledge.

        Overturning one of these needs evidence that outweighs
        ``protected_prior_logit`` -- roughly a dozen agreeing reliable sources,
        not one confident stranger.
        """
        cid = self._find_or_create(subject, relation, obj, text)
        claim = self.claims[cid]
        claim.prior_logit = self.protected_prior_logit
        claim.protected = True
        claim.status = BELIEVED
        return cid

    def _find_or_create(
        self, subject: str, relation: str, obj: str, text: str
    ) -> int:
        for cid in self._index.get((subject, relation), []):
            if self.claims[cid].object == obj:
                if text and not self.claims[cid].text:
                    self.claims[cid].text = text
                return cid
        claim = Claim(subject=subject, relation=relation, object=obj,
                      text=text or f"{subject} {relation} {obj}")
        self.claims.append(claim)
        cid = len(self.claims) - 1
        self._index.setdefault((subject, relation), []).append(cid)
        return cid

    # -- confidence ------------------------------------------------------
    def credibility(self, name: str) -> float:
        """How much a source's *nominal* trust should be believed.

        Trust is a Beta posterior mean, and a mean says nothing about the width
        of the distribution behind it.  A brand-new source asserting a 0.62
        prior has never once been checked against reality, so its likelihood
        ratio is shrunk toward zero until it has a track record.  Without this,
        an attacker mints identities and each one arrives at full strength.
        """
        src = self.register_source(name)
        n_eff = src.strength + src.observations
        return n_eff / (n_eff + self.credibility_scale)

    # -- collusion detection ---------------------------------------------
    def _rebuild_clusters(self) -> None:
        """Group sources that vote on the same claims the same way.

        The correlation discount alone assumes distinct names are distinct
        witnesses, which is exactly the assumption a sybil attack violates.  A
        puppet ring is visible in its *behaviour*: the accounts show up on the
        same claims with the same polarity.  Sources whose voting records
        overlap beyond ``cluster_overlap`` (Jaccard) and who never disagree are
        merged into one cluster and cast one vote between them.

        Two passes, because the exact-match case is both the common one and
        the cheap one:

        1. **Signature grouping**, O(S).  Sources with an identical voting
           record -- same claims, same polarity -- are hashed together.  A
           scripted puppet ring lands here.
        2. **Jaccard refinement**, O(S^2) over at most
           ``cluster_max_pairwise`` sources.  Catches partial overlap from a
           ring with some jitter, at a cost that has to stay bounded.

        Sources below ``cluster_min_votes`` stay singletons -- too little
        record to accuse anyone of collusion.
        """
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        eligible = [
            (name, votes) for name, votes in self._votes.items()
            if len(votes) >= self.cluster_min_votes
        ]
        for name, _ in eligible:
            parent.setdefault(name, name)

        # Pass 1: identical voting records.
        by_signature: Dict[frozenset, List[str]] = {}
        for name, votes in eligible:
            signature = frozenset(votes.items())
            by_signature.setdefault(signature, []).append(name)
        for members in by_signature.values():
            for other in members[1:]:
                union(members[0], other)

        # Pass 2: near-identical records, on a bounded sample.
        representatives = [
            (members[0], dict(self._votes[members[0]]))
            for members in by_signature.values()
        ][: self.cluster_max_pairwise]
        for i, (name_a, votes_a) in enumerate(representatives):
            keys_a = set(votes_a)
            for name_b, votes_b in representatives[i + 1:]:
                keys_b = set(votes_b)
                shared = keys_a & keys_b
                if not shared:
                    continue
                if len(shared) / len(keys_a | keys_b) < self.cluster_overlap:
                    continue
                # Overlapping *and* never disagreeing is the signature.
                if all(votes_a[c] == votes_b[c] for c in shared):
                    union(name_a, name_b)

        self._cluster_of = {name: find(name) for name, _ in eligible}
        self._clusters_dirty = False
        self._submits_since_cluster = 0

    def cluster_of(self, source: str) -> str:
        """The voting bloc a source belongs to, or itself if independent."""
        if not self.cluster_correlated_sources:
            return source
        # Rebuilding on every submit would make submission O(S^2); refresh on a
        # counter instead, so an attack is caught within a bounded number of
        # votes rather than instantly.
        if self._clusters_dirty or (
            self._submits_since_cluster >= self.cluster_refresh_every
        ):
            self._rebuild_clusters()
        return self._cluster_of.get(source, source)

    def clusters(self) -> Dict[str, List[str]]:
        """Detected voting blocs, largest first.  A ring shows up as one entry."""
        self._rebuild_clusters()
        groups: Dict[str, List[str]] = {}
        for name, root in self._cluster_of.items():
            groups.setdefault(root, []).append(name)
        return {k: sorted(v) for k, v in
                sorted(groups.items(), key=lambda kv: -len(kv[1])) if len(v) > 1}

    def own_logit(self, claim_id: int) -> float:
        """Evidence-pooled log-odds for one claim, ignoring rivals.

        Two separate discounts, defending against two different attacks:

        *Within* a source, the nth repeat counts ``1/n`` -- one loud voice
        saturates instead of accumulating.

        *Across* sources, the kth distinct source counts ``1/sqrt(k)``.  Naive
        pooling assumes independence, and 200 unvetted accounts agreeing is far
        better explained by one actor than by 200 witnesses.  The sqrt is the
        usual effective-sample-size correction for correlated observations -- a
        deliberate conservative choice, not a theorem.  Set
        ``correlation_discount="none"`` to pool them as fully independent.
        """
        claim = self.claims[claim_id]

        per_source: Dict[str, float] = {}
        seen: Dict[str, int] = {}
        for ev in claim.evidence:
            n = seen.get(ev.source, 0)
            seen[ev.source] = n + 1
            llr = _logit(self.trust(ev.source)) * self.credibility(ev.source)
            sign = 1.0 if ev.supports else -1.0
            per_source[ev.source] = (
                per_source.get(ev.source, 0.0) + sign * llr / (1.0 + n)
            )

        # A detected voting bloc casts one vote, not one per identity: keep the
        # strongest single voice in each cluster and discard the echoes.
        per_voice: Dict[str, float] = {}
        for name, contribution in per_source.items():
            key = self.cluster_of(name)
            if abs(contribution) > abs(per_voice.get(key, 0.0)):
                per_voice[key] = contribution

        total = claim.prior_logit
        ranked = sorted(per_voice.values(), key=lambda v: -abs(v))
        for rank, contribution in enumerate(ranked, start=1):
            discount = (
                1.0 / math.sqrt(rank)
                if self.correlation_discount == "sqrt" else 1.0
            )
            total += contribution * discount
        return total

    def confidence(self, claim_id: int) -> float:
        """Confidence in [0, 1], relative to the best competing claim."""
        mine = self.own_logit(claim_id)
        rivals = self._rivals(claim_id)
        if not rivals:
            return _sigmoid(mine)
        best_rival = max(self.own_logit(r) for r in rivals)
        return _sigmoid(mine - best_rival)

    def _rivals(self, claim_id: int) -> List[int]:
        """Mutually exclusive alternatives, for functional relations only."""
        claim = self.claims[claim_id]
        if claim.relation not in self.functional:
            return []
        return [
            cid for cid in self._index.get(claim.key, [])
            if cid != claim_id and self.claims[cid].status != OBSOLETE
        ]

    def independent_sources(self, claim_id: int) -> int:
        return len({e.source for e in self.claims[claim_id].evidence if e.supports})

    # -- subject knowledge -------------------------------------------------
    def learn_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        source: str,
        domain: str = "everyday",
        text: str = "",
    ) -> ClaimReport:
        """Submit a domain-tagged fact.

        Rigorous domains -- maths, physics, chemistry, biology, medicine,
        engineering -- get a higher promotion bar.  A mathematical result is
        not more true for being widely repeated, and a wrong chemistry fact can
        hurt someone, so social consensus alone should not promote either.
        """
        report = self.submit(subject, relation, obj, source, text=text)
        self.claims[report.claim_id].domain = domain
        return report

    def domain_bar(self, domain: str) -> Tuple[float, int]:
        """(confidence, independent sources) required to believe in this domain."""
        if domain in RIGOROUS_DOMAINS:
            return max(self.semantic_threshold, 0.90), self.min_independent_sources + 1
        return self.semantic_threshold, self.min_independent_sources

    def knowledge(self, domain: str, min_confidence: float = 0.5):
        """Believed claims in one subject, strongest first.

        A filter over the single graph, not a separate store.  Concepts that
        span domains stay connected to everything they touch.
        """
        found = [
            (cid, claim, self.confidence(cid))
            for cid, claim in enumerate(self.claims)
            if claim.domain == domain and claim.status != OBSOLETE
        ]
        return sorted(
            (f for f in found if f[2] >= min_confidence), key=lambda t: -t[2]
        )

    def cross_domain_links(self) -> Dict[str, Set[str]]:
        """Subjects connected by a shared subject or object.

        The reason for one graph: "Einstein" reaches physics *and* history, and
        that edge is exactly what per-subject databases would have cut.
        """
        by_entity: Dict[str, Set[str]] = {}
        for claim in self.claims:
            if claim.status == OBSOLETE:
                continue
            for entity in (claim.subject, claim.object):
                by_entity.setdefault(entity, set()).add(claim.domain)

        links: Dict[str, Set[str]] = {}
        for domains in by_entity.values():
            if len(domains) < 2:
                continue
            for domain in domains:
                links.setdefault(domain, set()).update(domains - {domain})
        return links

    def domain_report(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for domain in DOMAINS:
            claims = [c for c in self.claims if c.domain == domain]
            if not claims:
                continue
            believed = sum(1 for c in claims if c.status == BELIEVED)
            out[domain] = {
                "claims": float(len(claims)),
                "believed": float(believed),
                "bar": self.domain_bar(domain)[0],
            }
        return out

    # -- structured knowledge --------------------------------------------
    def submit_procedure(
        self, name: str, steps: Sequence[str], source: str, note: str = ""
    ) -> ClaimReport:
        """A procedure is an ordered list, not a triple.

        Triples handle facts well and sequences badly.  The triple form is kept
        so the same evidence, contradiction and promotion machinery applies --
        two conflicting procedures for one task compete like any other rival --
        while the ordered steps live in the payload.
        """
        digest = " > ".join(steps)
        report = self.submit(
            name, "procedure_for", digest, source,
            text=f"procedure for {name}: {digest}", note=note,
        )
        claim = self.claims[report.claim_id]
        claim.kind = PROCEDURE
        claim.payload["steps"] = list(steps)
        return report

    def submit_causal(
        self,
        cause: str,
        effect: str,
        source: str,
        strength: float = 1.0,
        mechanism: str = "",
        note: str = "",
    ) -> ClaimReport:
        """A causal link, with a strength and an optional mechanism.

        ``causes`` is deliberately many-valued: an effect usually has several
        contributing causes, and forcing them to compete would delete real ones.
        """
        report = self.submit(
            cause, "causes", effect, source,
            text=f"{cause} causes {effect}", note=note,
        )
        claim = self.claims[report.claim_id]
        claim.kind = CAUSAL
        claim.payload.update(strength=float(strength), mechanism=mechanism)
        return report

    def procedure(self, name: str, min_confidence: float = 0.5) -> List[str]:
        """The best-supported ordered procedure for a task, or an empty list."""
        best, score = [], min_confidence
        for cid in self._index.get((name, "procedure_for"), []):
            claim = self.claims[cid]
            if claim.status == OBSOLETE:
                continue
            conf = self.confidence(cid)
            if conf >= score:
                best, score = list(claim.payload.get("steps", [])), conf
        return best

    def causes_of(self, effect: str, min_confidence: float = 0.5):
        """Believed causes of an effect, strongest first."""
        found = []
        for cid, claim in enumerate(self.claims):
            if claim.kind != CAUSAL or claim.object != effect:
                continue
            if claim.status == OBSOLETE:
                continue
            conf = self.confidence(cid)
            if conf >= min_confidence:
                found.append((claim.subject, conf,
                              float(claim.payload.get("strength", 1.0))))
        return sorted(found, key=lambda t: -(t[1] * t[2]))

    # -- reality feedback ------------------------------------------------
    def ground(self, verdicts: Sequence[Tuple[int, bool]]) -> Dict[str, float]:
        """Apply a batch of prediction outcomes from :class:`PredictionLedger`.

        This is the call that removes the oracle.  The verdicts come from the
        environment disagreeing with a prediction, not from a human labelling
        anything, so source trust is anchored to whether beliefs actually
        predict the world.
        """
        touched: Dict[str, float] = {}
        for claim_id, was_correct in verdicts:
            if 0 <= claim_id < len(self.claims):
                touched.update(self.observe_outcome(claim_id, was_correct))
        return touched

    def observe_outcome(self, claim_id: int, was_correct: bool) -> Dict[str, float]:
        """Ground a belief against what actually happened.

        The robot predicted the door was unlocked and it was not.  Every source
        that supported that belief becomes a little less trusted, and so does
        every other claim resting on them.  This is what keeps source trust
        anchored to reality instead of to popularity.
        """
        claim = self.claims[claim_id]
        touched: Dict[str, float] = {}
        for name in {e.source for e in claim.evidence if e.supports}:
            src = self.register_source(name)
            if was_correct:
                src.correct += 1.0
            else:
                src.incorrect += 1.0
            touched[name] = src.trust
        claim.evidence.append(
            Evidence(source="observation", supports=was_correct, note="outcome")
        )
        self.register_source("observation", prior=0.95, strength=8.0)
        return touched

    # -- promotion -------------------------------------------------------
    def promote(
        self,
        memory,
        embed: Optional[Callable[[str], Any]] = None,
    ) -> List[ClaimReport]:
        """Move qualifying hypotheses into semantic memory.

        Requires both enough confidence *and* enough independent sources -- one
        very trusted source repeating itself is not corroboration.
        """
        promoted: List[ClaimReport] = []
        for cid, claim in enumerate(self.claims):
            if claim.status == OBSOLETE or claim.concept_id >= 0:
                continue
            # Rigorous domains carry a higher bar; see domain_bar().
            threshold, needed_sources = self.domain_bar(claim.domain)
            conf = self.confidence(cid)
            if conf < threshold:
                continue
            if (not claim.protected
                    and self.independent_sources(cid) < needed_sources):
                continue

            claim.status = BELIEVED
            superseded = self._retire_rivals(cid)

            if embed is not None:
                vec = embed(claim.text)
                claim.concept_id = memory.learn_concept(vec, label=claim.triple)
            reason = (
                "protected axiom"
                if claim.protected
                else f"{self.independent_sources(cid)} independent sources"
            )
            promoted.append(ClaimReport(
                claim_id=cid, triple=claim.triple, confidence=conf,
                status=BELIEVED, promoted=True, superseded=superseded,
                reason=reason,
            ))
        return promoted

    def _retire_rivals(self, claim_id: int) -> List[int]:
        """Mark beaten alternatives obsolete.  History is kept, not deleted."""
        retired = []
        for rid in self._rivals(claim_id):
            rival = self.claims[rid]
            if rival.protected:
                continue
            if self.confidence(rid) < 0.5 and rival.status != OBSOLETE:
                rival.status = OBSOLETE
                rival.superseded_by = claim_id
                retired.append(rid)
        return retired

    def weight_eligible(self) -> List[int]:
        """Claims settled enough to be allowed to change the weights in sleep."""
        return [
            cid for cid, c in enumerate(self.claims)
            if c.status == BELIEVED
            and self.confidence(cid) >= self.weight_threshold
            and (c.protected
                 or self.independent_sources(cid) >= self.min_independent_sources)
        ]

    # -- maintenance -----------------------------------------------------
    def decay(self, rate: float = 0.99) -> None:
        """Unreinforced beliefs drift back toward their prior."""
        for claim in self.claims:
            if not claim.protected:
                claim.prior_logit *= rate

    def beliefs(self, min_confidence: float = 0.5) -> List[Tuple[int, Claim, float]]:
        out = []
        for cid, claim in enumerate(self.claims):
            if claim.status == OBSOLETE:
                continue
            conf = self.confidence(cid)
            if conf >= min_confidence:
                out.append((cid, claim, conf))
        return sorted(out, key=lambda t: -t[2])

    def report(self) -> Dict[str, float]:
        statuses = [c.status for c in self.claims]
        return {
            "claims": float(len(self.claims)),
            "believed": float(statuses.count(BELIEVED)),
            "hypotheses": float(statuses.count(HYPOTHESIS)),
            "obsolete": float(statuses.count(OBSOLETE)),
            "sources": float(len(self.sources)),
            "weight_eligible": float(len(self.weight_eligible())),
        }

    def explain(self, claim_id: int) -> str:
        """Why is this believed?  Auditable, not a black box."""
        claim = self.claims[claim_id]
        lines = [
            f"{claim.triple}",
            f"  status      {claim.status}"
            f"{'  (protected axiom)' if claim.protected else ''}",
            f"  confidence  {self.confidence(claim_id):.4f}"
            f"   (own log-odds {self.own_logit(claim_id):+.3f})",
        ]
        if claim.prior_logit:
            lines.append(f"  prior       {claim.prior_logit:+.3f} log-odds")
        seen: Dict[str, int] = {}
        for ev in claim.evidence:
            n = seen.get(ev.source, 0)
            seen[ev.source] = n + 1
            sign = "+" if ev.supports else "-"
            lines.append(
                f"  {sign} {ev.source:<16} trust={self.trust(ev.source):.3f}"
                f"  weight x{1.0 / (1.0 + n):.2f}"
                + (f"  ({ev.note})" if ev.note else "")
            )
        for rid in self._rivals(claim_id):
            lines.append(
                f"  vs rival    {self.claims[rid].object!r} "
                f"log-odds {self.own_logit(rid):+.3f}"
            )
        return "\n".join(lines)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_threshold": self.semantic_threshold,
            "weight_threshold": self.weight_threshold,
            "min_independent_sources": self.min_independent_sources,
            "protected_prior_logit": self.protected_prior_logit,
            "credibility_scale": self.credibility_scale,
            "correlation_discount": self.correlation_discount,
            "functional": sorted(self.functional),
            "sources": [s.to_dict() for s in self.sources.values()],
            "claims": [c.to_dict() for c in self.claims],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RealityCortex":
        rc = cls(
            semantic_threshold=data.get("semantic_threshold", 0.75),
            weight_threshold=data.get("weight_threshold", 0.90),
            min_independent_sources=data.get("min_independent_sources", 2),
            functional_relations=data.get("functional", ()),
            protected_prior_logit=data.get("protected_prior_logit", 12.0),
            credibility_scale=data.get("credibility_scale", 4.0),
            correlation_discount=data.get("correlation_discount", "sqrt"),
        )
        for s in data.get("sources", []):
            rc.sources[s["name"]] = Source(**s)
        for c in data.get("claims", []):
            rc.claims.append(Claim.from_dict(c))
        for cid, claim in enumerate(rc.claims):
            rc._index.setdefault(claim.key, []).append(cid)
        return rc
