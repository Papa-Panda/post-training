import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "code"))

from harness_lab import Proposal, build_demo_registry  # noqa: E402


class HarnessLabTest(unittest.TestCase):
    def test_useful_no_regression_edit_is_accepted_and_versioned(self):
        registry = build_demo_registry()
        before = registry.active
        decision = registry.propose_evaluate_accept(
            Proposal(
                "fix failures",
                add_features=frozenset({"write_artifact", "retry_transient"}),
            )
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(registry.active.version, before.version + 1)
        self.assertEqual(registry.active.parent_digest, before.digest)
        self.assertEqual(decision.held_in_before, 0.0)
        self.assertEqual(decision.held_in_after, 1.0)
        self.assertEqual(decision.held_out_before, decision.held_out_after)
        self.assertEqual(len(registry.versions), 2)

    def test_held_out_regression_is_rejected(self):
        registry = build_demo_registry()
        registry.propose_evaluate_accept(
            Proposal(
                "fix failures",
                add_features=frozenset({"write_artifact", "retry_transient"}),
            )
        )
        active_digest = registry.active.digest
        decision = registry.propose_evaluate_accept(
            Proposal(
                "unsafe simplification",
                remove_features=frozenset({"ask_before_destructive"}),
                add_features=frozenset({"extra_feature"}),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("held-out regression", decision.reasons)
        self.assertEqual(registry.active.digest, active_digest)

    def test_immutable_surface_edit_never_materializes(self):
        registry = build_demo_registry()
        before = registry.active.digest
        evaluator = registry.evaluator_digest
        decision = registry.propose_evaluate_accept(
            Proposal("cheat", immutable_updates={"verifier": "disabled"})
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.candidate_digest, None)
        self.assertIn("attempted immutable-surface edit", decision.reasons)
        self.assertEqual(registry.active.digest, before)
        self.assertEqual(registry.evaluator_digest, evaluator)

    def test_forbidden_capability_is_rejected(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            Proposal(
                "bypass permissions",
                add_features=frozenset(
                    {"write_artifact", "retry_transient", "bypass_permissions"}
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("permission violation", decision.reasons)

    def test_cost_budget_is_enforced(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            Proposal(
                "unbounded loop",
                add_features=frozenset({"write_artifact", "retry_transient"}),
                max_steps=100,
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("cost budget exceeded", decision.reasons)

    def test_rejected_candidate_does_not_enter_version_registry(self):
        registry = build_demo_registry()
        decision = registry.propose_evaluate_accept(
            Proposal("no useful change", add_features=frozenset({"decorative_log"}))
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(len(registry.versions), 1)
        self.assertEqual(decision.after_digest, decision.before_digest)

    def test_markdown_math_and_navigation_contract(self):
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
            text = path.read_text()
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
