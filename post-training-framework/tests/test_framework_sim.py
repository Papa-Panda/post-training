import math
import pathlib
import subprocess
import sys
import unittest

TOPIC_ROOT = pathlib.Path(__file__).parents[1]
CODE_DIR = TOPIC_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from framework_sim import (  # noqa: E402
    BatchPlan,
    PostTrainingSimulator,
    SimulationConfig,
    Trajectory,
    clipped_importance_weight,
    filter_by_staleness,
    long_tail_config,
    make_optimizer_batches,
    validate_batch_accounting,
    validate_grpo_groups,
)


def group(group_id, version=0, size=4, prompt_id=None):
    prompt_id = prompt_id or f"prompt-{group_id}"
    return tuple(
        Trajectory(
            trajectory_id=f"{group_id}/{index}",
            group_id=group_id,
            prompt_id=prompt_id,
            sample_index=index,
            policy_version=version,
        )
        for index in range(size)
    )


class BatchAccountingTest(unittest.TestCase):
    def test_batch_hierarchy_identities_and_optimizer_steps(self):
        plan = BatchPlan(
            prompt_batch_size=8,
            grpo_group_size=4,
            per_device_micro_batch_size=2,
            data_parallel_world_size=2,
            gradient_accumulation_steps=2,
            epochs=3,
        )
        accounting = validate_batch_accounting(plan)
        self.assertEqual(accounting.trajectories, 8 * 4)
        self.assertEqual(accounting.global_micro_batch, 2 * 2)
        self.assertEqual(accounting.trajectories_per_optimizer_step, 2 * 2 * 2)
        self.assertEqual(accounting.groups_per_optimizer_step, 2)
        self.assertEqual(accounting.optimizer_steps_per_epoch, 4)
        self.assertEqual(accounting.total_optimizer_steps, 12)
        self.assertEqual(
            accounting.optimizer_steps_per_epoch
            * accounting.trajectories_per_optimizer_step,
            accounting.trajectories,
        )

    def test_invalid_optimizer_batch_that_splits_group_is_rejected(self):
        plan = BatchPlan(
            prompt_batch_size=4,
            grpo_group_size=4,
            per_device_micro_batch_size=3,
            data_parallel_world_size=1,
            gradient_accumulation_steps=1,
        )
        with self.assertRaisesRegex(ValueError, "split a GRPO group"):
            validate_batch_accounting(plan)

    def test_non_integral_epoch_step_count_is_rejected(self):
        plan = BatchPlan(
            prompt_batch_size=3,
            grpo_group_size=4,
            per_device_micro_batch_size=4,
            data_parallel_world_size=2,
            gradient_accumulation_steps=1,
        )
        with self.assertRaisesRegex(ValueError, "integer number"):
            validate_batch_accounting(plan)

    def test_singleton_grpo_group_is_rejected(self):
        plan = BatchPlan(
            prompt_batch_size=8,
            grpo_group_size=1,
            per_device_micro_batch_size=2,
            data_parallel_world_size=2,
            gradient_accumulation_steps=1,
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_batch_accounting(plan)


class GroupIntegrityTest(unittest.TestCase):
    def test_complete_groups_pack_without_splitting(self):
        trajectories = group("a") + group("b") + group("c") + group("d")
        batches = make_optimizer_batches(
            trajectories,
            trajectories_per_batch=8,
            grpo_group_size=4,
        )
        self.assertEqual(len(batches), 2)
        self.assertEqual([{item.group_id for item in batch} for batch in batches], [{"a", "b"}, {"c", "d"}])

    def test_incomplete_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 4"):
            validate_grpo_groups(group("a")[:-1], expected_group_size=4)

    def test_mixed_policy_versions_inside_group_are_rejected(self):
        members = list(group("a"))
        members[-1] = Trajectory("a/3", "a", "prompt-a", 3, 1)
        with self.assertRaisesRegex(ValueError, "mixes policy versions"):
            validate_grpo_groups(members, expected_group_size=4)

    def test_batch_width_that_would_split_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "split a GRPO group"):
            make_optimizer_batches(
                group("a") + group("b"),
                trajectories_per_batch=6,
                grpo_group_size=4,
            )


class OffPolicyTest(unittest.TestCase):
    def test_stale_samples_are_rejected_by_version_lag(self):
        trajectories = group("v0", version=0) + group("v1", version=1) + group("v2", version=2)
        result = filter_by_staleness(
            trajectories,
            current_policy_version=2,
            max_staleness=1,
        )
        self.assertEqual({item.policy_version for item in result.accepted}, {1, 2})
        self.assertEqual({item.policy_version for item in result.rejected}, {0})
        self.assertEqual(len(result.rejected), 4)

    def test_future_policy_stamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "future policy"):
            filter_by_staleness(group("future", version=3), current_policy_version=2, max_staleness=1)

    def test_importance_ratio_and_positive_advantage_clipping(self):
        result = clipped_importance_weight(
            behavior_logprob=0.0,
            target_logprob=math.log(2.0),
            advantage=3.0,
            clip_epsilon=0.2,
        )
        self.assertAlmostEqual(result.ratio, 2.0)
        self.assertAlmostEqual(result.clipped_ratio, 1.2)
        self.assertAlmostEqual(result.unclipped_objective, 6.0)
        self.assertAlmostEqual(result.clipped_objective, 3.6)

    def test_negative_advantage_uses_lower_clipped_surrogate(self):
        result = clipped_importance_weight(
            behavior_logprob=0.0,
            target_logprob=math.log(0.5),
            advantage=-1.0,
            clip_epsilon=0.2,
        )
        self.assertAlmostEqual(result.ratio, 0.5)
        self.assertAlmostEqual(result.clipped_ratio, 0.8)
        self.assertAlmostEqual(result.clipped_objective, -0.8)


class QueueSimulatorTest(unittest.TestCase):
    def test_sync_has_zero_version_lag_and_exact_step_count(self):
        result = PostTrainingSimulator(long_tail_config("sync")).run()
        self.assertEqual(result.max_version_lag, 0)
        self.assertEqual(result.mean_version_lag, 0.0)
        self.assertEqual(result.p95_version_lag, 0.0)
        self.assertEqual(result.effective_sample_fraction, 1.0)
        self.assertEqual(result.optimizer_steps, 4)
        self.assertEqual(result.final_policy_version, 4)
        self.assertEqual(result.trained_groups, 12)
        self.assertTrue(all(item.policy_version >= 0 for item in result.trajectories))

    def test_async_can_have_positive_staleness(self):
        result = PostTrainingSimulator(long_tail_config("async")).run()
        self.assertGreater(result.max_version_lag, 0)
        finished = [event for event in result.events if event.kind == "rollout_finished"]
        self.assertTrue(
            any(event.current_policy_version > event.policy_version for event in finished)
        )

    def test_staleness_metric_is_measured_at_learner_admission(self):
        config = SimulationConfig(
            mode="async",
            rollout_workers=2,
            total_groups=4,
            grpo_group_size=2,
            groups_per_optimizer_step=2,
            train_duration=5.0,
            max_staleness=10,
            rollout_durations=(1.0, 1.0, 1.0, 1.0),
        )
        result = PostTrainingSimulator(config).run()
        # The second wave finishes under learner version zero, then waits in
        # the queue until version one is published and admission runs.
        self.assertEqual(result.max_version_lag, 1)
        self.assertEqual(result.p95_version_lag, 1.0)

    def test_async_long_tail_improves_makespan_and_throughput(self):
        sync = PostTrainingSimulator(long_tail_config("sync")).run()
        asynchronous = PostTrainingSimulator(long_tail_config("async")).run()
        self.assertGreater(sync.barrier_idle, asynchronous.barrier_idle)
        self.assertLess(asynchronous.makespan, sync.makespan)
        self.assertGreater(asynchronous.rollout_utilization, sync.rollout_utilization)
        self.assertGreater(asynchronous.learner_utilization, sync.learner_utilization)
        self.assertGreater(asynchronous.p95_version_lag, 0)
        self.assertGreaterEqual(asynchronous.max_queue_depth, 3)
        self.assertGreater(
            asynchronous.throughput_trajectories_per_time,
            sync.throughput_trajectories_per_time,
        )

    def test_async_simulator_rejects_stale_groups(self):
        config = SimulationConfig(
            mode="async",
            rollout_workers=3,
            total_groups=9,
            grpo_group_size=2,
            groups_per_optimizer_step=1,
            train_duration=0.25,
            max_staleness=0,
            rollout_durations=(5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        result = PostTrainingSimulator(config).run()
        self.assertGreater(result.rejected_groups, 0)
        self.assertEqual(result.dropped_trajectories, result.rejected_groups * 2)
        self.assertLess(result.effective_sample_fraction, 1.0)
        self.assertTrue(any(event.kind == "stale_group_rejected" for event in result.events))

    def test_published_policy_is_used_by_subsequent_rollouts(self):
        config = SimulationConfig(
            mode="async",
            rollout_workers=2,
            total_groups=8,
            grpo_group_size=2,
            groups_per_optimizer_step=1,
            train_duration=0.5,
            rollout_durations=(1.0,) * 8,
        )
        result = PostTrainingSimulator(config).run()
        publications = [event for event in result.events if event.kind == "policy_published"]
        starts = [event for event in result.events if event.kind == "rollout_started"]
        first_publish = publications[0]
        later_starts = [event for event in starts if event.time >= first_publish.time]
        self.assertTrue(later_starts)
        self.assertTrue(
            all(event.policy_version >= first_publish.policy_version for event in later_starts)
        )
        self.assertTrue(any(event.policy_version > 0 for event in starts))

    def test_fixed_seed_is_reproducible(self):
        config = SimulationConfig(
            mode="async",
            rollout_workers=3,
            total_groups=12,
            grpo_group_size=4,
            groups_per_optimizer_step=3,
            train_duration=0.5,
            seed=12345,
        )
        first = PostTrainingSimulator(config).run()
        second = PostTrainingSimulator(config).run()
        self.assertEqual(first, second)

    def test_demo_cli_runs_without_third_party_packages(self):
        completed = subprocess.run(
            [sys.executable, str(CODE_DIR / "demo.py"), "--mode", "both"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('"mode": "sync"', completed.stdout)
        self.assertIn('"mode": "async"', completed.stdout)


class DocumentationContractTest(unittest.TestCase):
    def test_math_and_relative_links(self):
        import re

        for path in TOPIC_ROOT.rglob("*.md"):
            text = path.read_text()
            self.assertNotIn("\\operatorname", text, path)
            self.assertNotIn("\\[", text, path)
            self.assertNotIn("\\]", text, path)
            self.assertEqual(text.count("$$") % 2, 0, path)
            for line in text.splitlines():
                if "$$" in line:
                    self.assertEqual(line.count("$$"), 2, f"display math must be one line: {path}: {line}")
            for target in re.findall(r"\[[^]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)", text):
                self.assertTrue((path.parent / target).resolve().exists(), f"broken link {path}: {target}")

    def test_no_employer_identifier_or_control_characters(self):
        import re

        banned = re.compile(r"\b(?:meta|aai)\b", re.IGNORECASE)
        for path in TOPIC_ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path == pathlib.Path(__file__):
                continue
            text = path.read_text(errors="ignore")
            self.assertIsNone(banned.search(text), path)
            for character in text:
                self.assertTrue(character >= " " or character in "\t\n\r", f"control character in {path}")


if __name__ == "__main__":
    unittest.main()
