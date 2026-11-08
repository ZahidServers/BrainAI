# BrainAI - built by Mohammed Zahid.
"""The executive: what to work on, what to drop, and when to stop and sleep.

The old executive was one ``nn.Linear`` mixing the reflex answer with the
deliberate one.  That is an output arbiter, not an executive -- it has no notion
of a goal, cannot be interrupted, and cannot resume anything.  That arbiter
still exists and still does the mixing (:class:`brainai.pathways.
ExecutiveController`); this module is the part that decides what the model is
*for* at any moment.

Four jobs:

**Arbitration.**  Goals are scored by ``priority x urgency x feasibility``.
Urgency rises as a deadline approaches, so a low-priority job that is about to
expire can correctly outrank an important one with time left.

**Interruption.**  A safety event or a reflex trigger preempts immediately
without discarding what was underway.

**Task switching with context.**  The outgoing goal's working-memory contents
are saved into the goal and restored when it resumes.  Without that, switching
away from a task destroys it -- which is the difference between an interruption
and an abort.

**Conflict resolution.**  Goals may declare incompatibility.  The loser is
suspended, not failed, because "not now" and "never" are different outcomes.

Sleep scheduling lives here too, because deciding when to stop acting and
consolidate is an executive decision, not a memory one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import torch

PENDING = "pending"
ACTIVE = "active"
SUSPENDED = "suspended"
ACHIEVED = "achieved"
FAILED = "failed"
ABANDONED = "abandoned"

TERMINAL = frozenset({ACHIEVED, FAILED, ABANDONED})


@dataclass
class Goal:
    """Something the system is trying to bring about."""

    name: str
    priority: float = 0.5
    deadline: Optional[int] = None          # executive tick
    parent: Optional[int] = None
    status: str = PENDING
    progress: float = 0.0
    feasibility: float = 1.0
    interrupt: bool = False
    conflicts_with: Set[str] = field(default_factory=set)
    created_at: int = 0
    resumed_count: int = 0
    #: Working-memory snapshot, saved on switch-out and restored on resume.
    context: Optional[torch.Tensor] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    #: Urgency of a goal with no deadline at all.
    BASELINE_URGENCY = 0.5

    def urgency(self, now: int) -> float:
        """Rises from the baseline toward 1.0 as the deadline approaches.

        Deliberately bounded *below* by the no-deadline baseline.  A naive
        ``1/(1 + k*remaining)`` sends a distant deadline toward zero, which
        means attaching a deadline to a goal makes it *less* likely to run than
        having no deadline -- so an important task scheduled for later loses to
        an unimportant one scheduled for never.
        """
        if self.deadline is None:
            return self.BASELINE_URGENCY
        remaining = self.deadline - now
        if remaining <= 0:
            return 1.0
        proximity = 1.0 / (1.0 + 0.1 * remaining)
        return self.BASELINE_URGENCY + (1.0 - self.BASELINE_URGENCY) * proximity

    def score(self, now: int) -> float:
        if self.finished:
            return -1.0
        if self.interrupt:
            return 1e6 + self.priority        # preempts everything
        return self.priority * self.urgency(now) * max(self.feasibility, 1e-3)

    def describe(self, now: int = 0) -> str:
        due = "" if self.deadline is None else f" due@{self.deadline}"
        return (f"{self.name} [{self.status}] p={self.priority:.2f}"
                f" u={self.urgency(now):.2f} progress={self.progress:.0%}{due}")


class GoalStack:
    """Goals, their relationships, and which one wins right now."""

    def __init__(self) -> None:
        self.goals: List[Goal] = []
        self.tick = 0
        self.active: Optional[int] = None
        self.switches = 0
        self.preemptions = 0

    # -- lifecycle -------------------------------------------------------
    def push(
        self,
        name: str,
        priority: float = 0.5,
        deadline: Optional[int] = None,
        parent: Optional[int] = None,
        conflicts_with: Sequence[str] = (),
        interrupt: bool = False,
        **payload: Any,
    ) -> int:
        goal = Goal(
            name=name, priority=priority, deadline=deadline, parent=parent,
            conflicts_with=set(conflicts_with), interrupt=interrupt,
            created_at=self.tick, payload=payload,
        )
        self.goals.append(goal)
        return len(self.goals) - 1

    def interrupt(self, name: str, priority: float = 1.0, **payload: Any) -> int:
        """Preempt whatever is running.  The incumbent is suspended, not lost."""
        self.preemptions += 1
        return self.push(name, priority=priority, interrupt=True, **payload)

    def complete(self, goal_id: int, achieved: bool = True) -> None:
        goal = self.goals[goal_id]
        goal.status = ACHIEVED if achieved else FAILED
        goal.progress = 1.0 if achieved else goal.progress
        if self.active == goal_id:
            self.active = None
        if not achieved:
            self._abandon_descendants(goal_id)
            # Decomposition is conjunctive: a parent needs all its subgoals, so
            # it cannot survive one of them failing.  The planner may still
            # rescue the situation with a different method, but at the goal
            # level this branch is dead.
            parent = self.goals[goal_id].parent
            while parent is not None and not self.goals[parent].finished:
                self.goals[parent].status = ABANDONED
                if self.active == parent:
                    self.active = None
                parent = self.goals[parent].parent

    def _abandon_descendants(self, goal_id: int) -> None:
        """Subgoals of a dead goal have nothing left to contribute to."""
        for index, other in enumerate(self.goals):
            if other.parent == goal_id and not other.finished:
                other.status = ABANDONED
                if self.active == index:
                    self.active = None
                self._abandon_descendants(index)

    def abandon(self, goal_id: int) -> None:
        self.goals[goal_id].status = ABANDONED
        if self.active == goal_id:
            self.active = None

    def progress(self, goal_id: int, amount: float) -> None:
        goal = self.goals[goal_id]
        goal.progress = min(1.0, goal.progress + amount)
        if goal.progress >= 1.0:
            self.complete(goal_id, achieved=True)

    # -- arbitration -----------------------------------------------------
    def eligible(self) -> List[int]:
        return [i for i, g in enumerate(self.goals) if not g.finished]

    def conflicts(self, goal_id: int) -> List[int]:
        goal = self.goals[goal_id]
        return [
            i for i in self.eligible()
            if i != goal_id and (
                self.goals[i].name in goal.conflicts_with
                or goal.name in self.goals[i].conflicts_with
            )
        ]

    def arbitrate(self) -> Optional[int]:
        """Pick the goal that should be running, suspending anything it blocks."""
        candidates = self.eligible()
        if not candidates:
            self.active = None
            return None

        winner = max(candidates, key=lambda i: self.goals[i].score(self.tick))
        if self.goals[winner].score(self.tick) < 0:
            self.active = None
            return None

        # Suspend, do not fail: "not now" is not "never".
        for loser in self.conflicts(winner):
            if self.goals[loser].status == ACTIVE:
                self.goals[loser].status = SUSPENDED

        if self.active is not None and self.active != winner:
            incumbent = self.goals[self.active]
            if not incumbent.finished:
                incumbent.status = SUSPENDED
            self.switches += 1

        goal = self.goals[winner]
        if goal.status == SUSPENDED:
            goal.resumed_count += 1
        goal.status = ACTIVE
        self.active = winner
        return winner

    def advance(self, ticks: int = 1) -> None:
        self.tick += ticks

    def report(self) -> Dict[str, float]:
        statuses = [g.status for g in self.goals]
        return {
            "goals": float(len(self.goals)),
            "active": float(statuses.count(ACTIVE)),
            "suspended": float(statuses.count(SUSPENDED)),
            "achieved": float(statuses.count(ACHIEVED)),
            "failed": float(statuses.count(FAILED)),
            "abandoned": float(statuses.count(ABANDONED)),
            "switches": float(self.switches),
            "preemptions": float(self.preemptions),
        }

    def render(self) -> str:
        if not self.goals:
            return "  (no goals)"
        rows = sorted(
            ((i, g) for i, g in enumerate(self.goals)),
            key=lambda t: -t[1].score(self.tick),
        )
        return "\n".join(
            f"  {'>' if i == self.active else ' '} {i:>2} {g.describe(self.tick)}"
            for i, g in rows
        )


class Executive:
    """Goal management, interruption, task switching and sleep scheduling.

    Holds no parameters.  Like the safety controller, it is deliberately not
    something online learning can rewrite -- a system that can edit its own
    priorities under gradient pressure is much harder to reason about.
    """

    def __init__(
        self,
        working_memory: Any = None,
        sleep_pressure_threshold: float = 1.0,
    ) -> None:
        self.stack = GoalStack()
        self.working_memory = working_memory
        self.sleep_pressure_threshold = sleep_pressure_threshold
        self.ticks_since_sleep = 0
        self.sleeps = 0

    # -- task switching --------------------------------------------------
    def _save_context(self, goal: Goal) -> None:
        wm = self.working_memory
        if wm is not None and getattr(wm, "_content", None) is not None:
            goal.context = wm._content.detach().clone()

    def _restore_context(self, goal: Goal) -> bool:
        wm = self.working_memory
        if wm is None or goal.context is None:
            return False
        wm._content = goal.context.clone()
        return True

    def step(self) -> Dict[str, Any]:
        """One executive cycle: arbitrate, switch context if the winner changed."""
        previous = self.stack.active
        if previous is not None:
            self._save_context(self.stack.goals[previous])

        winner = self.stack.arbitrate()
        restored = False
        if winner is not None and winner != previous:
            restored = self._restore_context(self.stack.goals[winner])

        self.stack.advance()
        self.ticks_since_sleep += 1

        return {
            "active": winner,
            "goal": None if winner is None else self.stack.goals[winner].name,
            "switched": winner != previous,
            "context_restored": restored,
            "tick": self.stack.tick,
        }

    # -- sleep scheduling ------------------------------------------------
    def sleep_pressure(
        self,
        replay_fill: float = 0.0,
        unconsolidated_claims: int = 0,
        drift: float = 0.0,
        idle: bool = False,
    ) -> Dict[str, float]:
        """How badly does this system need to consolidate?

        Four independent pressures, so no single one can force a sleep on its
        own.  Idleness is a *multiplier*, not a term: the right time to
        consolidate is when nothing is being asked of you, which is the whole
        reason biological sleep is scheduled the way it is.
        """
        buffer_term = min(1.0, replay_fill)
        claim_term = min(1.0, unconsolidated_claims / 32.0)
        drift_term = min(1.0, drift / 0.5)
        time_term = min(1.0, self.ticks_since_sleep / 512.0)

        total = buffer_term + claim_term + drift_term + time_term
        if idle:
            total *= 1.5

        return {
            "pressure": total,
            "buffer": buffer_term,
            "claims": claim_term,
            "drift": drift_term,
            "time": time_term,
            "should_sleep": float(total >= self.sleep_pressure_threshold),
        }

    def sleep_plan(self, pressure: Dict[str, float]) -> Dict[str, int]:
        """How long to sleep and what to prioritise, given the pressure profile.

        Cycles scale with total pressure; the dominant term decides emphasis --
        a full replay buffer wants rehearsal, accumulated drift wants
        re-anchoring.
        """
        total = pressure["pressure"]
        cycles = int(max(2, min(32, round(4 * total))))
        dominant = max(
            ("buffer", "claims", "drift", "time"), key=lambda k: pressure[k]
        )
        return {
            "cycles": cycles,
            "batch_size": 8 if dominant == "buffer" else 4,
            "emphasis": dominant,
        }

    def slept(self) -> None:
        self.ticks_since_sleep = 0
        self.sleeps += 1

    def report(self) -> Dict[str, float]:
        out = self.stack.report()
        out.update(ticks_since_sleep=float(self.ticks_since_sleep),
                   sleeps=float(self.sleeps))
        return out
