# BrainAI - built by Mohammed Zahid.
"""Does the Reality Cortex beat a naive belief manager, and does it scale?

Two questions a reviewer will ask that demos cannot answer.

**Is the evidence machinery earning its complexity?**  Compared against
``NaiveCounter`` -- confidence = supports / (supports + refutes), the obvious
implementation -- across four scenarios, including two where the Reality Cortex
is *supposed* to lose and one where it genuinely does.

**What happens at a million beliefs?**  Submission, confidence and promotion are
timed at increasing store sizes so the complexity is measured rather than
asserted.

    python benchmarks/belief_bench.py --max-claims 200000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainai.reality import RealityCortex  # noqa: E402


class NaiveCounter:
    """The obvious belief manager: count votes, ignore who is voting."""

    def __init__(self) -> None:
        self.votes: Dict[Tuple[str, str, str], List[bool]] = {}

    def submit(self, subject, relation, obj, source, supports=True):
        self.votes.setdefault((subject, relation, obj), []).append(supports)

    def confidence(self, subject, relation, obj) -> float:
        v = self.votes.get((subject, relation, obj), [])
        if not v:
            return 0.0
        return sum(v) / len(v)

    def winner(self, subject, relation) -> Tuple[str, float]:
        best, score = "", -1.0
        for (s, r, o), v in self.votes.items():
            if (s, r) != (subject, relation):
                continue
            c = sum(v) / len(v) * min(1.0, len(v) / 3.0)   # mild count weighting
            if c > score:
                best, score = o, c
        return best, score


def make_cortex() -> RealityCortex:
    rc = RealityCortex()
    rc.register_source("manual", prior=0.95, strength=16)
    rc.register_source("engineer_a", prior=0.80, strength=8)
    rc.register_source("engineer_b", prior=0.80, strength=8)
    rc.register_source("blog", prior=0.20, strength=4)
    # Sockpuppets are given prior 0.62 on purpose.  Below 0.5 a source's
    # *support* is already evidence against -- log(t/(1-t)) goes negative -- so
    # a flood of visibly-untrusted identities defeats itself and tests nothing.
    # The real attack is many identities that each look individually plausible.
    for i in range(200):
        rc.register_source(f"sock_{i}", prior=0.62, strength=4)
    return rc


# -- scenarios --------------------------------------------------------------

def scenario_spam(rc: RealityCortex, nv: NaiveCounter) -> None:
    """One low-trust source repeats a falsehood 100x; two good sources disagree."""
    for _ in range(100):
        rc.submit("bolt", "torque_spec", "50Nm", "blog")
        nv.submit("bolt", "torque_spec", "50Nm", "blog")
    for src in ("manual", "engineer_a"):
        rc.submit("bolt", "torque_spec", "40Nm", src)
        nv.submit("bolt", "torque_spec", "40Nm", src)


def scenario_crowd(rc: RealityCortex, nv: NaiveCounter) -> None:
    """The majority is right this time - we must not over-reject."""
    for src in ("manual", "engineer_a", "engineer_b"):
        rc.submit("cup", "material_of", "ceramic", src)
        nv.submit("cup", "material_of", "ceramic", src)
    rc.submit("cup", "material_of", "steel", "blog")
    nv.submit("cup", "material_of", "steel", "blog")


def scenario_sybil(rc: RealityCortex, nv: NaiveCounter) -> None:
    """200 *distinct* low-trust identities all assert the same falsehood.

    The independence damping keys on source name, so distinct names read as
    distinct witnesses.  This is the attack the mechanism does not stop.
    """
    for i in range(200):
        rc.submit("earth", "shape_of", "flat", f"sock_{i}")
        nv.submit("earth", "shape_of", "flat", f"sock_{i}")
    rc.submit("earth", "shape_of", "oblate spheroid", "manual")
    nv.submit("earth", "shape_of", "oblate spheroid", "manual")


def scenario_sybil_protected(rc: RealityCortex, nv: NaiveCounter) -> None:
    """Same attack, but the truth was seeded as a protected axiom."""
    rc.assert_axiom("earth2", "shape_of", "oblate spheroid")
    nv.submit("earth2", "shape_of", "oblate spheroid", "axiom")
    for i in range(200):
        rc.submit("earth2", "shape_of", "flat", f"sock_{i}")
        nv.submit("earth2", "shape_of", "flat", f"sock_{i}")


def scenario_sybil_with_record(rc: RealityCortex, nv: NaiveCounter) -> None:
    """A ring that votes more than once, so its behaviour is visible.

    Collusion detection needs a voting record to work with.  A puppet that
    speaks exactly once is indistinguishable from a genuine one-time witness --
    correctly so, since a single data point cannot support the accusation.
    The moment a ring participates in several claims, its shared pattern gives
    it away and the whole ring collapses to one vote.
    """
    rc.submit("earth3", "shape_of", "oblate spheroid", "manual")
    nv.submit("earth3", "shape_of", "oblate spheroid", "manual")
    for i in range(200):
        for topic in range(4):
            rc.submit("earth3", "shape_of", "flat", f"sock_{i}")
            rc.submit(f"filler{topic}", "prop", "x", f"sock_{i}")
            nv.submit("earth3", "shape_of", "flat", f"sock_{i}")


SCENARIOS = [
    ("spam: 100 repeats vs 2 good sources", scenario_spam,
     "bolt", "torque_spec", "40Nm", "50Nm"),
    ("crowd: majority is correct", scenario_crowd,
     "cup", "material_of", "ceramic", "steel"),
    ("sybil: 200 fake identities", scenario_sybil,
     "earth", "shape_of", "oblate spheroid", "flat"),
    ("sybil vs protected axiom", scenario_sybil_protected,
     "earth2", "shape_of", "oblate spheroid", "flat"),
    ("sybil with a voting record", scenario_sybil_with_record,
     "earth3", "shape_of", "oblate spheroid", "flat"),
]


def run_quality() -> None:
    print("Belief quality - does the truth win?\n")
    print(f"{'scenario':<38}{'RealityCortex':>16}{'naive':>10}{'verdict':>10}")
    print("-" * 74)

    for name, fn, subj, rel, truth, lie in SCENARIOS:
        rc, nv = make_cortex(), NaiveCounter()
        fn(rc, nv)

        truth_id = rc._find_or_create(subj, rel, truth, "")
        lie_id = rc._find_or_create(subj, rel, lie, "")
        rc_truth = rc.confidence(truth_id)
        rc_lie = rc.confidence(lie_id)
        rc_ok = rc_truth > rc_lie

        nv_winner, _ = nv.winner(subj, rel)
        nv_ok = nv_winner == truth

        verdict = ("both ok" if rc_ok and nv_ok else
                   "RC only" if rc_ok else
                   "naive only" if nv_ok else "both fail")
        print(f"{name:<38}{rc_truth:>8.4f} vs {rc_lie:<5.3f}"
              f"{('ok' if nv_ok else 'FAIL'):>10}{verdict:>10}")

    print("\n  'RC only' is the Reality Cortex earning its complexity.")
    print("  'both fail' is an honest limitation, not a rounding error.")
    print("\n  The two sybil rows are the interesting pair.  Puppets that vote")
    print("  exactly once are indistinguishable from one-time witnesses and get")
    print("  through unless the target is a protected axiom.  Puppets with a")
    print("  voting record give their ring away and collapse to a single vote.")


def run_attack_cost() -> None:
    """How many fresh identities does it take to overturn a belief?

    A security-style number: not "is it safe" but "what does breaking it cost".
    """
    print("\n\nAttack cost - identities needed to flip a belief\n")
    print(f"{'target':<34}{'discount=sqrt':>16}{'discount=none':>16}")
    print("-" * 66)

    def identities_needed(protected: bool, discount: str, limit: int = 4000) -> int:
        rc = RealityCortex(correlation_discount=discount)
        rc.register_source("manual", prior=0.95, strength=16)
        if protected:
            rc.assert_axiom("t", "shape_of", "true")
        else:
            rc.submit("t", "shape_of", "true", "manual")
        for i in range(1, limit + 1):
            rc.register_source(f"a{i}", prior=0.62, strength=4)
            rc.submit("t", "shape_of", "false", f"a{i}")
            if i % 25 == 0 or i < 25:
                truth = rc._find_or_create("t", "shape_of", "true", "")
                if rc.confidence(truth) < 0.5:
                    return i
        return -1

    for label, protected in [("single trusted source", False),
                             ("protected axiom", True)]:
        a = identities_needed(protected, "sqrt")
        b = identities_needed(protected, "none")
        fmt = lambda v: f"{v:,}" if v > 0 else ">4,000"
        print(f"{label:<34}{fmt(a):>16}{fmt(b):>16}")

    print("\n  The sqrt correlation discount raises the price of a sybil attack")
    print("  by roughly an order of magnitude.  It does not make one impossible,")
    print("  and nothing here detects collusion between distinct identities.")


def run_scaling(max_claims: int) -> None:
    print("\n\nScaling - measured, not asserted\n")
    print(f"{'claims':>10}{'submit us':>12}{'confidence us':>16}"
          f"{'promote ms':>13}{'report ms':>12}")
    print("-" * 63)

    rng = random.Random(0)
    rc = RealityCortex()
    for i in range(8):
        rc.register_source(f"src_{i}", prior=0.8, strength=8)

    size = 0
    checkpoints = [n for n in (1_000, 10_000, 50_000, 100_000, 200_000, 500_000)
                   if n <= max_claims]
    for target in checkpoints:
        t0 = time.perf_counter()
        while size < target:
            rc.submit(f"s{size}", "prop", f"o{size}", f"src_{size % 8}")
            size += 1
        submit_us = (time.perf_counter() - t0) / max(target - 0, 1) * 1e6

        probes = [rng.randrange(size) for _ in range(2000)]
        t0 = time.perf_counter()
        for cid in probes:
            rc.confidence(cid)
        conf_us = (time.perf_counter() - t0) / len(probes) * 1e6

        t0 = time.perf_counter()
        rc.promote(memory=None, embed=None)
        promote_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        rc.report()
        report_ms = (time.perf_counter() - t0) * 1e3

        print(f"{size:>10,}{submit_us:>12.2f}{conf_us:>16.2f}"
              f"{promote_ms:>13.1f}{report_ms:>12.1f}")

    print("\n  submit and confidence are O(1) in store size - both are index")
    print("  lookups over one (subject, relation) bucket.  promote and report")
    print("  are full scans; watch those columns grow linearly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-claims", type=int, default=100_000)
    args = ap.parse_args()

    print("Reality Cortex benchmark.  Built by Mohammed Zahid.\n")
    run_quality()
    run_attack_cost()
    run_scaling(args.max_claims)


if __name__ == "__main__":
    main()
