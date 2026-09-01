import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "code"))

from harness_lab import (  # noqa: E402
    ControlPlane,
    EditSurface,
    FailureAttribution,
    Harness,
    HarnessRegistry,
    Proposal,
    SurfacePatch,
    Task,
    attributed_proposal,
    build_demo_registry,
)


class HarnessLabTest(unittest.TestCase):
    def proposal(
        self,
        registry,
        *,
        name="proposal",
        failure="failure",
        surface=EditSurface.WORKFLOW,
        patches=(),
        fixes=("failure",),
        max_steps=None,
        control_updates=(),
    ):
        return attributed_proposal(
            registry,
            name=name,
            failure_type=failure,
            target_surface=surface,
            patches=patches,
            expected_fixes=fixes,
            max_steps=max_steps,
            attempted_control_plane_updates=control_updates,
        )

    def accept_baseline_fix(self, registry):
        return registry.propose_evaluate_accept(
            self.proposal(
                registry,
                name="fix failures",
                patches=(
                    SurfacePatch(
                        EditSurface.WORKFLOW,
                        add=frozenset({"retry_transient"}),
                    ),
                    SurfacePatch(
                        EditSurface.TOOL,
                        add=frozenset({"write_artifact"}),
                    ),
                ),
                fixes=("persist-artifact", "recover-transient-tool-error"),
            )
        )

    def test_useful_no_regression_edit_is_accepted_and_versioned(self):
        registry = build_demo_registry()
        before = registry.active
        decision = self.accept_baseline_fix(registry)
        self.assertTrue(decision.accepted)
        self.assertEqual(registry.active.version, before.version + 1)
        self.assertEqual(registry.active.parent_digest, before.digest)
        self.assertEqual(decision.held_in_before, 0.0)
        self.assertEqual(decision.held_in_after, 1.0)
        self.assertEqual(decision.held_out_before, decision.held_out_after)
        self.assertEqual(len(registry.versions), 2)

    def test_exact_gate_allows_held_out_only_improvement(self):
        initial = Harness(
            version=0,
            context_rules=frozenset(),
            workflow_nodes=frozenset(),
            tool_capabilities=frozenset(),
            memory_rules=frozenset(),
            max_steps=1,
        )
        control = ControlPlane(
            held_in=(Task("already-passes"),),
            held_out=(
                Task("needs-citation", required_context=frozenset({"cite"})),
            ),
            forbidden_capabilities=frozenset(),
            capability_risk=(),
            max_cost=5,
            max_risk=0,
        )
        registry = HarnessRegistry(initial, control)
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                surface=EditSurface.CONTEXT,
                patches=(
                    SurfacePatch(EditSurface.CONTEXT, add=frozenset({"cite"})),
                ),
            )
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.held_in_before, decision.held_in_after)
        self.assertGreater(decision.held_out_after, decision.held_out_before)

    def test_held_out_regression_is_rejected_without_mutation(self):
        registry = build_demo_registry()
        self.accept_baseline_fix(registry)
        active_digest = registry.active.digest
        versions_before = dict(registry.versions)
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                name="unsafe simplification",
                patches=(
                    SurfacePatch(
                        EditSurface.WORKFLOW,
                        remove=frozenset({"ask_before_destructive"}),
                    ),
                ),
                fixes=("latency",),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("held-out regression", decision.reasons)
        self.assertEqual(registry.active.digest, active_digest)
        self.assertEqual(registry.versions, versions_before)

    def test_no_improvement_is_rejected(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                patches=(
                    SurfacePatch(
                        EditSurface.MEMORY,
                        add=frozenset({"decorative_rule"}),
                    ),
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("neither split improved", decision.reasons)

    def test_control_plane_edit_never_materializes(self):
        registry = build_demo_registry()
        before = registry.active.digest
        control_digest = registry.control_plane_digest
        decision = registry.propose_evaluate_accept(
            self.proposal(registry, control_updates=("verifier",))
        )
        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.candidate_digest)
        self.assertIn("attempted immutable-control-plane edit", decision.reasons)
        self.assertEqual(registry.active.digest, before)
        self.assertEqual(registry.control_plane_digest, control_digest)

    def test_forbidden_capability_is_rejected(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                surface=EditSurface.TOOL,
                patches=(
                    SurfacePatch(
                        EditSurface.TOOL,
                        add=frozenset(
                            {
                                "write_artifact",
                                "bypass_permissions",
                            }
                        ),
                    ),
                    SurfacePatch(
                        EditSurface.WORKFLOW,
                        add=frozenset({"retry_transient"}),
                    ),
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("permission violation", decision.reasons)

    def test_risk_budget_is_enforced(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                surface=EditSurface.TOOL,
                patches=(
                    SurfacePatch(
                        EditSurface.TOOL,
                        add=frozenset({"write_artifact", "network_send"}),
                    ),
                    SurfacePatch(
                        EditSurface.WORKFLOW,
                        add=frozenset({"retry_transient"}),
                    ),
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("risk budget exceeded", decision.reasons)

    def test_cost_budget_is_enforced(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                patches=(
                    SurfacePatch(
                        EditSurface.WORKFLOW,
                        add=frozenset({"retry_transient"}),
                    ),
                    SurfacePatch(
                        EditSurface.TOOL,
                        add=frozenset({"write_artifact"}),
                    ),
                ),
                max_steps=100,
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("cost budget exceeded", decision.reasons)

    def test_stale_parent_and_missing_evidence_fail_static_validation(self):
        registry = build_demo_registry()
        attribution = FailureAttribution(
            failure_type="failure",
            target_surface=EditSurface.CONTEXT,
            component="compiler",
            confidence=0.8,
            evidence_refs=(),
        )
        proposal = Proposal(
            name="invalid manifest",
            parent_digest="stale",
            attribution=attribution,
            hypothesis="change context",
            patches=(
                SurfacePatch(EditSurface.CONTEXT, add=frozenset({"rule"})),
            ),
            expected_fixes=("failure",),
            at_risk=(),
            evidence_refs=(),
        )
        decision = registry.propose_evaluate_accept(proposal)
        self.assertFalse(decision.accepted)
        self.assertIn("stale parent digest", decision.reasons)
        self.assertIn("attribution has no evidence", decision.reasons)
        self.assertIn("proposal has no evidence", decision.reasons)
        self.assertIsNone(decision.candidate_digest)

    def test_rejected_candidate_does_not_enter_version_registry(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            self.proposal(
                registry,
                patches=(
                    SurfacePatch(
                        EditSurface.MEMORY,
                        add=frozenset({"decorative_rule"}),
                    ),
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(len(registry.versions), 1)
        self.assertEqual(decision.after_digest, decision.before_digest)
        self.assertIsNotNone(decision.candidate_digest)

    def test_control_plane_digest_covers_task_semantics(self):
        common = dict(
            held_out=(),
            forbidden_capabilities=frozenset(),
            capability_risk=(),
            max_cost=5,
            max_risk=0,
        )
        first = ControlPlane(
            held_in=(Task("task", required_context=frozenset({"a"})),),
            **common,
        )
        second = ControlPlane(
            held_in=(Task("task", required_context=frozenset({"b"})),),
            **common,
        )
        self.assertNotEqual(first.digest, second.digest)

    def test_markdown_math_navigation_and_control_char_contract(self):
        topic_dir = pathlib.Path(__file__).parents[1]
        expected = [
            "README.md",
            "01_harness_vs_model.md",
            "02_agent_runtime_loop.md",
            "03_context_and_persistent_memory.md",
            "04_workflow_and_subagents.md",
            "05_harness_optimization.md",
            "06_self_improving_harness.md",
            "07_observability_evaluation_security.md",
            "08_harness_rl_and_weight_updates.md",
            "papers.md",
        ]
        for name in expected:
            path = topic_dir / name
            self.assertTrue(path.exists(), name)
            raw = path.read_bytes()
            bad = [
                byte
                for byte in raw
                if (byte < 32 and byte not in (9, 10, 13)) or byte == 127
            ]
            self.assertFalse(bad, f"{name} has control characters")
            text = raw.decode("utf-8")
            self.assertIn("<!-- NAVIGATION -->", text, name)
            self.assertNotIn("\\[", text, name)
            self.assertNotIn("\\]", text, name)
            self.assertNotIn("\\operatorname", text, name)
            for line_number, line in enumerate(text.splitlines(), start=1):
                if "$$" in line:
                    self.assertEqual(
                        line.count("$$"),
                        2,
                        f"{name}:{line_number} has split/unpaired display math",
                    )


if __name__ == "__main__":
    unittest.main()
