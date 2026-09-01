"""Minimal, dependency-free harness evolution loop.

The evaluator and permission boundary are constructed outside the editable harness.
Candidate edits may only touch explicitly editable surfaces. A candidate becomes active
only when it improves held-in tasks, does not regress held-out tasks, stays within the
cost budget, and causes no permission violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class Task:
    name: str
    required_features: FrozenSet[str]
    forbidden_features: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class Harness:
    version: int
    features: FrozenSet[str]
    max_steps: int
    parent_digest: str = "ROOT"

    def snapshot(self) -> Mapping[str, object]:
        return {
            "version": self.version,
            "features": sorted(self.features),
            "max_steps": self.max_steps,
            "parent_digest": self.parent_digest,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Proposal:
    name: str
    add_features: FrozenSet[str] = frozenset()
    remove_features: FrozenSet[str] = frozenset()
    max_steps: int | None = None
    immutable_updates: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    passed: int
    total: int
    failures: Tuple[str, ...]
    permission_violations: Tuple[str, ...]
    cost: int

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class Decision:
    proposal: str
    accepted: bool
    reasons: Tuple[str, ...]
    before_digest: str
    candidate_digest: str | None
    after_digest: str
    held_in_before: float
    held_in_after: float | None
    held_out_before: float
    held_out_after: float | None


class ImmutableEvaluator:
    """An evaluator deliberately kept outside the editable harness workspace."""

    def __init__(
        self,
        held_in: Sequence[Task],
        held_out: Sequence[Task],
        forbidden_capabilities: Iterable[str],
        max_cost: int,
    ) -> None:
        self._held_in = tuple(held_in)
        self._held_out = tuple(held_out)
        self._forbidden = frozenset(forbidden_capabilities)
        self._max_cost = max_cost
        manifest = {
            "held_in": [t.name for t in self._held_in],
            "held_out": [t.name for t in self._held_out],
            "forbidden": sorted(self._forbidden),
            "max_cost": self._max_cost,
        }
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        self._digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def max_cost(self) -> int:
        return self._max_cost

    def evaluate(self, harness: Harness, split: str) -> Evaluation:
        tasks = self._held_in if split == "held_in" else self._held_out
        if split not in {"held_in", "held_out"}:
            raise ValueError("split must be held_in or held_out")

        failures: List[str] = []
        permission_violations = sorted(harness.features & self._forbidden)
        for task in tasks:
            missing = task.required_features - harness.features
            forbidden = task.forbidden_features & harness.features
            if missing or forbidden:
                failures.append(task.name)
        cost = harness.max_steps + len(harness.features)
        passed = len(tasks) - len(failures)
        return Evaluation(
            passed=passed,
            total=len(tasks),
            failures=tuple(failures),
            permission_violations=tuple(permission_violations),
            cost=cost,
        )


class HarnessRegistry:
    """Versioned propose -> evaluate -> accept state machine."""

    def __init__(self, initial: Harness, evaluator: ImmutableEvaluator) -> None:
        self._active = initial
        self._evaluator = evaluator
        self._versions: Dict[str, Harness] = {initial.digest: initial}
        self._decisions: List[Decision] = []

    @property
    def active(self) -> Harness:
        return self._active

    @property
    def evaluator_digest(self) -> str:
        return self._evaluator.digest

    @property
    def decisions(self) -> Tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def versions(self) -> Mapping[str, Harness]:
        return dict(self._versions)

    def _materialize(self, proposal: Proposal) -> Harness:
        features = (self._active.features | proposal.add_features) - proposal.remove_features
        max_steps = self._active.max_steps if proposal.max_steps is None else proposal.max_steps
        return replace(
            self._active,
            version=self._active.version + 1,
            features=frozenset(features),
            max_steps=max_steps,
            parent_digest=self._active.digest,
        )

    def propose_evaluate_accept(self, proposal: Proposal) -> Decision:
        before = self._active
        before_in = self._evaluator.evaluate(before, "held_in")
        before_out = self._evaluator.evaluate(before, "held_out")

        if proposal.immutable_updates:
            decision = Decision(
                proposal=proposal.name,
                accepted=False,
                reasons=("attempted immutable-surface edit",),
                before_digest=before.digest,
                candidate_digest=None,
                after_digest=before.digest,
                held_in_before=before_in.score,
                held_in_after=None,
                held_out_before=before_out.score,
                held_out_after=None,
            )
            self._decisions.append(decision)
            return decision

        candidate = self._materialize(proposal)
        after_in = self._evaluator.evaluate(candidate, "held_in")
        after_out = self._evaluator.evaluate(candidate, "held_out")
        reasons: List[str] = []
        if after_in.score <= before_in.score:
            reasons.append("held-in did not improve")
        if after_out.score < before_out.score:
            reasons.append("held-out regression")
        if after_in.permission_violations or after_out.permission_violations:
            reasons.append("permission violation")
        if max(after_in.cost, after_out.cost) > self._evaluator.max_cost:
            reasons.append("cost budget exceeded")

        accepted = not reasons
        if accepted:
            self._active = candidate
            self._versions[candidate.digest] = candidate

        decision = Decision(
            proposal=proposal.name,
            accepted=accepted,
            reasons=tuple(reasons),
            before_digest=before.digest,
            candidate_digest=candidate.digest,
            after_digest=self._active.digest,
            held_in_before=before_in.score,
            held_in_after=after_in.score,
            held_out_before=before_out.score,
            held_out_after=after_out.score,
        )
        self._decisions.append(decision)
        return decision


def build_demo_registry() -> HarnessRegistry:
    held_in = (
        Task("persist-artifact", frozenset({"write_artifact"})),
        Task("recover-transient-tool-error", frozenset({"retry_transient"})),
    )
    held_out = (
        Task("cite-evidence", frozenset({"cite_evidence"})),
        Task(
            "safe-shell",
            frozenset({"ask_before_destructive"}),
            forbidden_features=frozenset({"bypass_permissions"}),
        ),
    )
    evaluator = ImmutableEvaluator(
        held_in=held_in,
        held_out=held_out,
        forbidden_capabilities={"bypass_permissions", "disable_verifier"},
        max_cost=12,
    )
    initial = Harness(
        version=0,
        features=frozenset({"cite_evidence", "ask_before_destructive"}),
        max_steps=4,
    )
    return HarnessRegistry(initial=initial, evaluator=evaluator)
