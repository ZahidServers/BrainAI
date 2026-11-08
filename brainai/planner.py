# BrainAI - built by Mohammed Zahid.
"""Hierarchical planning: goals into subgoals into actions, checked by imagination.

``SpikingStack("planner")`` is a transformer stack with "planner" in its name.
It maps a state to an action and has no representation of a goal, a subgoal, or
a plan that could fail halfway through.  This module is the actual planner.

Decomposition is HTN-style: a **method library** maps a goal to an ordered list
of subgoals, recursively, until every leaf is a primitive the motor system can
execute.  Methods come from two places -- registered by the operator, or learned
as procedures in the Reality Cortex, which means a procedure the robot picked up
by watching becomes a plan it can execute once enough evidence accumulates.

Candidate plans are scored by rolling them forward through
:class:`brainai.world.WorldModel` rather than by executing them.  That is what
imagination buys: the cost of a bad plan is a few matmuls instead of a broken
gripper.  Rollout confidence compounds, so a long plan through poorly understood
dynamics scores low even when every individual step looks fine.

Replanning is the part that matters in practice. Plans fail; the interesting
question is whether the failure is local -- retry the step, or find another
method for that subgoal -- or fatal to the parent goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: A method turns a goal into ordered subgoals.  Returning () means primitive.
Method = Callable[[str, Dict], Sequence[str]]


@dataclass
class PlanNode:
    """One node of the plan tree."""

    goal: str
    depth: int = 0
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    action: Optional[torch.Tensor] = None
    status: str = PENDING
    attempts: int = 0
    method: str = ""
    expected_value: float = 0.0

    @property
    def primitive(self) -> bool:
        return not self.children


@dataclass
class Plan:
    """A decomposed goal, plus the order its leaves must execute in."""

    root: int
    nodes: List[PlanNode]
    order: List[int]
    confidence: float = 0.0
    imagined_value: float = 0.0

    def leaves(self) -> List[int]:
        return [i for i in self.order if self.nodes[i].primitive]

    def next_action(self) -> Optional[int]:
        for i in self.order:
            node = self.nodes[i]
            if node.primitive and node.status in (PENDING, RUNNING):
                return i
        return None

    @property
    def complete(self) -> bool:
        return all(self.nodes[i].status == DONE for i in self.leaves())

    @property
    def failed(self) -> bool:
        return self.nodes[self.root].status == FAILED

    def render(self) -> str:
        lines = []
        for i in self.order:
            node = self.nodes[i]
            mark = {PENDING: " ", RUNNING: ">", DONE: "x", FAILED: "!"}[node.status]
            method = f"  <{node.method}>" if node.method else ""
            lines.append(f"  [{mark}] {'  ' * node.depth}{node.goal}{method}")
        return "\n".join(lines)


class MethodLibrary:
    """Goal -> subgoals.  Registered by hand, or learned from procedures."""

    def __init__(self) -> None:
        self._methods: Dict[str, List[Tuple[str, Sequence[str]]]] = {}

    def register(self, goal: str, subgoals: Sequence[str], name: str = "") -> None:
        self._methods.setdefault(goal, []).append(
            (name or f"method_{len(self._methods.get(goal, []))}", list(subgoals))
        )

    def learn_from_reality(self, reality, min_confidence: float = 0.6) -> int:
        """Import believed procedures as planning methods.

        A procedure that survived the Reality Cortex -- corroborated, not
        contradicted -- becomes something the robot can actually carry out.
        Evidence and capability meet here.
        """
        added = 0
        for cid, claim in enumerate(reality.claims):
            if claim.kind != "procedure" or claim.status == "obsolete":
                continue
            steps = claim.payload.get("steps") or []
            if not steps or reality.confidence(cid) < min_confidence:
                continue
            if any(name == f"learned:{cid}"
                   for name, _ in self._methods.get(claim.subject, [])):
                continue
            self.register(claim.subject, steps, name=f"learned:{cid}")
            added += 1
        return added

    def methods_for(self, goal: str) -> List[Tuple[str, Sequence[str]]]:
        return self._methods.get(goal, [])

    def is_primitive(self, goal: str) -> bool:
        return goal not in self._methods

    def __len__(self) -> int:
        return sum(len(v) for v in self._methods.values())


class HierarchicalPlanner:
    """Decompose, imagine, commit, replan."""

    def __init__(
        self,
        library: Optional[MethodLibrary] = None,
        max_depth: int = 4,
        max_retries: int = 2,
    ) -> None:
        self.library = library or MethodLibrary()
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.plans_built = 0
        self.replans = 0

    # -- decomposition ---------------------------------------------------
    def decompose(self, goal: str, method_index: int = 0) -> Plan:
        """Expand a goal into a plan tree, depth-first, left to right."""
        nodes: List[PlanNode] = [PlanNode(goal=goal, depth=0)]
        order: List[int] = []

        def expand(node_id: int, choose: int) -> None:
            node = nodes[node_id]
            order.append(node_id)
            if node.depth >= self.max_depth:
                return
            options = self.library.methods_for(node.goal)
            if not options:
                return                                   # primitive
            name, subgoals = options[min(choose, len(options) - 1)]
            node.method = name
            for sub in subgoals:
                nodes.append(PlanNode(goal=sub, depth=node.depth + 1,
                                      parent=node_id))
                child_id = len(nodes) - 1
                node.children.append(child_id)
                expand(child_id, 0)

        expand(0, method_index)
        self.plans_built += 1
        return Plan(root=0, nodes=nodes, order=order)

    # -- imagination -----------------------------------------------------
    @torch.no_grad()
    def score(
        self,
        plan: Plan,
        state: torch.Tensor,
        world,
        action_for: Callable[[str], Optional[torch.Tensor]],
        value_of: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Plan:
        """Roll the plan forward in imagination and record what it is worth.

        ``state``: (1, d).  ``action_for`` maps a primitive goal name to an
        action vector.  ``value_of`` scores an imagined end state; without one,
        plans are ranked by rollout confidence alone.
        """
        leaves = plan.leaves()
        actions = [action_for(plan.nodes[i].goal) for i in leaves]
        if not actions:
            plan.confidence, plan.imagined_value = 0.0, 0.0
            return plan

        trajectory, confidence = world.imagine(state, actions)
        plan.confidence = float(confidence.mean())

        final = trajectory[:, -1]
        value = 0.0 if value_of is None else float(value_of(final).mean())
        # A long plan through uncertain dynamics is worth less than its end
        # state suggests, because it probably will not get there.
        plan.imagined_value = value * plan.confidence
        for node_id, step in zip(leaves, range(trajectory.shape[1])):
            plan.nodes[node_id].action = actions[step]
            plan.nodes[node_id].expected_value = plan.imagined_value
        return plan

    def best_plan(
        self,
        goal: str,
        state: torch.Tensor,
        world,
        action_for: Callable[[str], Optional[torch.Tensor]],
        value_of: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Optional[Plan]:
        """Build one plan per available method and imagine each; return the best."""
        options = self.library.methods_for(goal)
        candidates = [
            self.score(self.decompose(goal, i), state, world, action_for, value_of)
            for i in range(max(1, len(options)))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.imagined_value, p.confidence))

    # -- execution and repair --------------------------------------------
    def report_result(self, plan: Plan, node_id: int, succeeded: bool) -> str:
        """Record a step outcome and say what to do next.

        Returns one of ``continue``, ``retry``, ``remethod``, ``failed``.
        Distinguishing these is the whole point: a step that failed once is not
        the same as a goal that cannot be achieved.
        """
        node = plan.nodes[node_id]
        node.attempts += 1

        if succeeded:
            node.status = DONE
            self._propagate_done(plan, node.parent)
            return "continue"

        if node.attempts <= self.max_retries:
            node.status = PENDING
            return "retry"

        node.status = FAILED

        # Walk *up* looking for somewhere to replan.  Checking only the
        # immediate parent misses the common case: alternatives are usually
        # registered against the high-level task, not the leaf that broke, so
        # a recoverable failure gets reported as fatal.
        if self.remethod_target(plan, node_id) is not None:
            self.replans += 1
            return "remethod"

        self._propagate_failure(plan, node_id)
        return "failed"

    def remethod_target(self, plan: Plan, node_id: int) -> Optional[int]:
        """Nearest ancestor with an unused alternative method, if any."""
        current = plan.nodes[node_id].parent
        while current is not None:
            node = plan.nodes[current]
            if len(self.library.methods_for(node.goal)) > 1:
                return current
            current = node.parent
        return None

    def remethod(self, plan: Plan, node_id: int) -> Optional[Plan]:
        """Rebuild the subtree under the nearest re-methodable ancestor.

        Returns a fresh plan for that ancestor's goal using its next method, or
        None when every alternative has already been tried.
        """
        target = self.remethod_target(plan, node_id)
        if target is None:
            return None
        goal = plan.nodes[target].goal
        tried = {plan.nodes[target].method}
        for index, (name, _) in enumerate(self.library.methods_for(goal)):
            if name not in tried:
                return self.decompose(goal, index)
        return None

    def _propagate_done(self, plan: Plan, node_id: Optional[int]) -> None:
        while node_id is not None:
            node = plan.nodes[node_id]
            if all(plan.nodes[c].status == DONE for c in node.children):
                node.status = DONE
                node_id = node.parent
            else:
                break

    def _propagate_failure(self, plan: Plan, node_id: int) -> None:
        current: Optional[int] = node_id
        while current is not None:
            plan.nodes[current].status = FAILED
            current = plan.nodes[current].parent

    def report(self) -> Dict[str, float]:
        return {
            "methods": float(len(self.library)),
            "plans_built": float(self.plans_built),
            "replans": float(self.replans),
        }
