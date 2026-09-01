"""Dependency-free reference implementation of a bounded harness update gate.

The toy task scorer is intentionally simple. The control semantics are the point:

* the model/harness candidate cannot mutate the evaluator or permission policy;
* edits name their surface, evidence, hypothesis, expected fix, and at-risk behavior;
* held-in and hidden held-out scores are compared against the active version;
* both splits must be non-regressing and at least one must improve;
* permission, cost, and risk gates are external to the editable harness;
* rejected candidates never enter the version registry or change active state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple


class EditSurface(str, Enum):
    CONTEXT = "context"
    WORKFLOW = "workflow"
    TOOL = "tool"
    MEMORY = "memory"


@dataclass(frozen=True)
class Task:
    name: str
    required_context: FrozenSet[str] = frozenset()
    required_workflow: FrozenSet[str] = frozenset()
    required_tools: FrozenSet[str] = frozenset()
    required_memory: FrozenSet[str] = frozenset()
    forbidden_tools: FrozenSet[str] = frozenset()

    def snapshot(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "required_context": sorted(self.required_context),
            "required_workflow": sorted(self.required_workflow),
            "required_tools": sorted(self.required_tools),
            "required_memory": sorted(self.required_memory),
            "forbidden_tools": sorted(self.forbidden_tools),
        }


@dataclass(frozen=True)
class Harness:
    version: int
    context_rules: FrozenSet[str]
    workflow_nodes: FrozenSet[str]
    tool_capabilities: FrozenSet[str]
    memory_rules: FrozenSet[str]
    max_steps: int
    parent_digest: str = "ROOT"

    def snapshot(self) -> Mapping[str, object]:
        return {
            "version": self.version,
            "context_rules": sorted(self.context_rules),
            "workflow_nodes": sorted(self.workflow_nodes),
            "tool_capabilities": sorted(self.tool_capabilities),
            "memory_rules": sorted(self.memory_rules),
            "max_steps": self.max_steps,
            "parent_digest": self.parent_digest,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FailureAttribution:
    failure_type: str
    target_surface: EditSurface
    component: str
    confidence: float
    evidence_refs: Tuple[str, ...]
    counterevidence_refs: Tuple[str, ...] = ()
    replay_fixture: str | None = None

    def validate(self) -> Tuple[str, ...]:
        errors: List[str] = []
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("attribution confidence outside [0, 1]")
        if not self.evidence_refs:
            errors.append("attribution has no evidence")
        if not self.failure_type or not self.component:
            errors.append("attribution is incomplete")
        return tuple(errors)


@dataclass(frozen=True)
class SurfacePatch:
    surface: EditSurface
    add: FrozenSet[str] = frozenset()
    remove: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class Proposal:
    name: str
    parent_digest: str
    attribution: FailureAttribution
    hypothesis: str
    patches: Tuple[SurfacePatch, ...]
    expected_fixes: Tuple[str, ...]
    at_risk: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    max_steps: int | None = None
    attempted_control_plane_updates: FrozenSet[str] = frozenset()

    def validate(self, active: Harness) -> Tuple[str, ...]:
        errors: List[str] = list(self.attribution.validate())
        if self.parent_digest != active.digest:
            errors.append("stale parent digest")
        if not self.hypothesis:
            errors.append("proposal has no hypothesis")
        if not self.patches and self.max_steps is None:
            errors.append("proposal is a no-op")
        if not self.expected_fixes:
            errors.append("proposal has no expected fix")
        if not self.evidence_refs:
            errors.append("proposal has no evidence")
        if len({patch.surface for patch in self.patches}) != len(self.patches):
            errors.append("proposal repeats an edit surface")
        for patch in self.patches:
            if patch.add & patch.remove:
                errors.append(f"same item added and removed on {patch.surface.value}")
        if self.max_steps is not None and self.max_steps <= 0:
            errors.append("max_steps must be positive")
        return tuple(errors)


@dataclass(frozen=True)
class Evaluation:
    split: str
    passed: int
    total: int
    failures: Tuple[str, ...]
    permission_violations: Tuple[str, ...]
    cost: int
    risk: int

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
    control_plane_digest: str


@dataclass(frozen=True)
class ControlPlane:
    """Evaluator, permissions, and budgets outside the editable Harness object."""

    held_in: Tuple[Task, ...]
    held_out: Tuple[Task, ...]
    forbidden_capabilities: FrozenSet[str]
    capability_risk: Tuple[Tuple[str, int], ...]
    max_cost: int
    max_risk: int

    @property
    def digest(self) -> str:
        manifest = {
            "held_in": [task.snapshot() for task in self.held_in],
            "held_out": [task.snapshot() for task in self.held_out],
            "forbidden_capabilities": sorted(self.forbidden_capabilities),
            "capability_risk": sorted(self.capability_risk),
            "max_cost": self.max_cost,
            "max_risk": self.max_risk,
        }
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def evaluate(self, harness: Harness, split: str) -> Evaluation:
        if split == "held_in":
            tasks = self.held_in
        elif split == "held_out":
            tasks = self.held_out
        else:
            raise ValueError("split must be held_in or held_out")

        failures: List[str] = []
        permission_violations = tuple(
            sorted(harness.tool_capabilities & self.forbidden_capabilities)
        )
        for task in tasks:
            failed = (
                bool(task.required_context - harness.context_rules)
                or bool(task.required_workflow - harness.workflow_nodes)
                or bool(task.required_tools - harness.tool_capabilities)
                or bool(task.required_memory - harness.memory_rules)
                or bool(task.forbidden_tools & harness.tool_capabilities)
            )
            if failed:
                failures.append(task.name)

        cost = harness.max_steps + sum(
            len(items)
            for items in (
                harness.context_rules,
                harness.workflow_nodes,
                harness.tool_capabilities,
                harness.memory_rules,
            )
        )
        risk_table = dict(self.capability_risk)
        risk = sum(risk_table.get(capability, 0) for capability in harness.tool_capabilities)
        return Evaluation(
            split=split,
            passed=len(tasks) - len(failures),
            total=len(tasks),
            failures=tuple(failures),
            permission_violations=permission_violations,
            cost=cost,
            risk=risk,
        )


class HarnessRegistry:
    """Versioned propose -> sandbox-evaluate -> gate -> promote state machine."""

    def __init__(self, initial: Harness, control_plane: ControlPlane) -> None:
        self._active = initial
        self._control_plane = control_plane
        self._versions: Dict[str, Harness] = {initial.digest: initial}
        self._decisions: List[Decision] = []

    @property
    def active(self) -> Harness:
        return self._active

    @property
    def control_plane_digest(self) -> str:
        return self._control_plane.digest

    @property
    def decisions(self) -> Tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def versions(self) -> Mapping[str, Harness]:
        return dict(self._versions)

    def _materialize(self, proposal: Proposal) -> Harness:
        values: Dict[EditSurface, FrozenSet[str]] = {
            EditSurface.CONTEXT: self._active.context_rules,
            EditSurface.WORKFLOW: self._active.workflow_nodes,
            EditSurface.TOOL: self._active.tool_capabilities,
            EditSurface.MEMORY: self._active.memory_rules,
        }
        for patch in proposal.patches:
            values[patch.surface] = (values[patch.surface] | patch.add) - patch.remove

        return replace(
            self._active,
            version=self._active.version + 1,
            context_rules=values[EditSurface.CONTEXT],
            workflow_nodes=values[EditSurface.WORKFLOW],
            tool_capabilities=values[EditSurface.TOOL],
            memory_rules=values[EditSurface.MEMORY],
            max_steps=(
                self._active.max_steps
                if proposal.max_steps is None
                else proposal.max_steps
            ),
            parent_digest=self._active.digest,
        )

    def _reject(
        self,
        proposal: Proposal,
        reasons: Iterable[str],
        before_in: Evaluation,
        before_out: Evaluation,
        candidate: Harness | None = None,
        after_in: Evaluation | None = None,
        after_out: Evaluation | None = None,
    ) -> Decision:
        decision = Decision(
            proposal=proposal.name,
            accepted=False,
            reasons=tuple(reasons),
            before_digest=self._active.digest,
            candidate_digest=None if candidate is None else candidate.digest,
            after_digest=self._active.digest,
            held_in_before=before_in.score,
            held_in_after=None if after_in is None else after_in.score,
            held_out_before=before_out.score,
            held_out_after=None if after_out is None else after_out.score,
            control_plane_digest=self._control_plane.digest,
        )
        self._decisions.append(decision)
        return decision

    def propose_evaluate_accept(self, proposal: Proposal) -> Decision:
        before = self._active
        before_in = self._control_plane.evaluate(before, "held_in")
        before_out = self._control_plane.evaluate(before, "held_out")

        if proposal.attempted_control_plane_updates:
            return self._reject(
                proposal,
                ("attempted immutable-control-plane edit",),
                before_in,
                before_out,
            )

        static_errors = proposal.validate(before)
        if static_errors:
            return self._reject(proposal, static_errors, before_in, before_out)

        candidate = self._materialize(proposal)
        after_in = self._control_plane.evaluate(candidate, "held_in")
        after_out = self._control_plane.evaluate(candidate, "held_out")
        delta_in = after_in.score - before_in.score
        delta_out = after_out.score - before_out.score

        reasons: List[str] = []
        if delta_in < 0:
            reasons.append("held-in regression")
        if delta_out < 0:
            reasons.append("held-out regression")
        if max(delta_in, delta_out) <= 0:
            reasons.append("neither split improved")
        if after_in.permission_violations or after_out.permission_violations:
            reasons.append("permission violation")
        if max(after_in.cost, after_out.cost) > self._control_plane.max_cost:
            reasons.append("cost budget exceeded")
        if max(after_in.risk, after_out.risk) > self._control_plane.max_risk:
            reasons.append("risk budget exceeded")

        if reasons:
            return self._reject(
                proposal,
                reasons,
                before_in,
                before_out,
                candidate,
                after_in,
                after_out,
            )

        self._active = candidate
        self._versions[candidate.digest] = candidate
        decision = Decision(
            proposal=proposal.name,
            accepted=True,
            reasons=(),
            before_digest=before.digest,
            candidate_digest=candidate.digest,
            after_digest=candidate.digest,
            held_in_before=before_in.score,
            held_in_after=after_in.score,
            held_out_before=before_out.score,
            held_out_after=after_out.score,
            control_plane_digest=self._control_plane.digest,
        )
        self._decisions.append(decision)
        return decision


def build_demo_registry() -> HarnessRegistry:
    held_in = (
        Task("persist-artifact", required_tools=frozenset({"write_artifact"})),
        Task(
            "recover-transient-tool-error",
            required_workflow=frozenset({"retry_transient"}),
        ),
    )
    held_out = (
        Task("cite-evidence", required_context=frozenset({"cite_evidence"})),
        Task(
            "safe-shell",
            required_workflow=frozenset({"ask_before_destructive"}),
            forbidden_tools=frozenset({"bypass_permissions"}),
        ),
    )
    control_plane = ControlPlane(
        held_in=held_in,
        held_out=held_out,
        forbidden_capabilities=frozenset(
            {"bypass_permissions", "disable_verifier"}
        ),
        capability_risk=(
            ("write_artifact", 2),
            ("network_send", 7),
            ("bypass_permissions", 100),
        ),
        max_cost=12,
        max_risk=5,
    )
    initial = Harness(
        version=0,
        context_rules=frozenset({"cite_evidence"}),
        workflow_nodes=frozenset({"ask_before_destructive"}),
        tool_capabilities=frozenset(),
        memory_rules=frozenset({"retain_provenance"}),
        max_steps=4,
    )
    return HarnessRegistry(initial=initial, control_plane=control_plane)


def attributed_proposal(
    registry: HarnessRegistry,
    *,
    name: str,
    failure_type: str,
    target_surface: EditSurface,
    patches: Sequence[SurfacePatch],
    expected_fixes: Sequence[str],
    at_risk: Sequence[str] = (),
    max_steps: int | None = None,
    attempted_control_plane_updates: Iterable[str] = (),
) -> Proposal:
    """Convenience factory used by the demo while keeping a full manifest."""

    evidence = (f"trace://{failure_type}",)
    attribution = FailureAttribution(
        failure_type=failure_type,
        target_surface=target_surface,
        component=target_surface.value,
        confidence=0.8,
        evidence_refs=evidence,
        replay_fixture=f"fixture://{failure_type}",
    )
    return Proposal(
        name=name,
        parent_digest=registry.active.digest,
        attribution=attribution,
        hypothesis=f"bounded {target_surface.value} edit addresses {failure_type}",
        patches=tuple(patches),
        expected_fixes=tuple(expected_fixes),
        at_risk=tuple(at_risk),
        evidence_refs=evidence,
        max_steps=max_steps,
        attempted_control_plane_updates=frozenset(attempted_control_plane_updates),
    )
