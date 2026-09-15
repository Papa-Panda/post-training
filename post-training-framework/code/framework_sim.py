"""Deterministic, dependency-free models for post-training data flow.

The module is deliberately a semantics simulator rather than a performance
predictor.  It makes policy versions, GRPO groups, optimizer batches, stale
samples, and sync/async queue behavior explicit enough to test on any CPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
import math
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BatchPlan:
    """Hierarchy from prompts to trajectories to optimizer steps.

    ``per_device_micro_batch_size`` counts trajectories, not prompts. This toy
    validator models the stricter grouped-packing mode: every optimizer batch
    contains complete GRPO groups, and an epoch contains an integer number of
    optimizer batches. Systems that precompute immutable group advantages may
    use a different post-statistics packing contract.
    """

    prompt_batch_size: int
    grpo_group_size: int
    per_device_micro_batch_size: int
    data_parallel_world_size: int
    gradient_accumulation_steps: int
    epochs: int = 1

    @property
    def trajectories_per_epoch(self) -> int:
        return self.prompt_batch_size * self.grpo_group_size

    @property
    def global_micro_batch_size(self) -> int:
        return self.per_device_micro_batch_size * self.data_parallel_world_size

    @property
    def trajectories_per_optimizer_step(self) -> int:
        return self.global_micro_batch_size * self.gradient_accumulation_steps

    @property
    def groups_per_optimizer_step(self) -> int:
        self.validate()
        return self.trajectories_per_optimizer_step // self.grpo_group_size

    @property
    def optimizer_steps_per_epoch(self) -> int:
        self.validate()
        return self.trajectories_per_epoch // self.trajectories_per_optimizer_step

    @property
    def total_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_epoch * self.epochs

    def validate(self) -> None:
        fields = {
            "prompt_batch_size": self.prompt_batch_size,
            "grpo_group_size": self.grpo_group_size,
            "per_device_micro_batch_size": self.per_device_micro_batch_size,
            "data_parallel_world_size": self.data_parallel_world_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs": self.epochs,
        }
        nonpositive = [name for name, value in fields.items() if value <= 0]
        if nonpositive:
            raise ValueError("batch dimensions must be positive: " + ", ".join(nonpositive))
        if self.grpo_group_size < 2:
            raise ValueError("GRPO requires at least two trajectories per prompt group")
        optimizer_batch = self.trajectories_per_optimizer_step
        if optimizer_batch % self.grpo_group_size:
            raise ValueError("optimizer batch would split a GRPO group")
        if self.trajectories_per_epoch % optimizer_batch:
            raise ValueError("epoch does not contain an integer number of optimizer steps")


@dataclass(frozen=True)
class BatchAccounting:
    prompts: int
    trajectories: int
    global_micro_batch: int
    trajectories_per_optimizer_step: int
    groups_per_optimizer_step: int
    optimizer_steps_per_epoch: int
    epochs: int
    total_optimizer_steps: int


def validate_batch_accounting(plan: BatchPlan) -> BatchAccounting:
    """Validate the hierarchy and return all derived counts."""

    plan.validate()
    return BatchAccounting(
        prompts=plan.prompt_batch_size,
        trajectories=plan.trajectories_per_epoch,
        global_micro_batch=plan.global_micro_batch_size,
        trajectories_per_optimizer_step=plan.trajectories_per_optimizer_step,
        groups_per_optimizer_step=plan.groups_per_optimizer_step,
        optimizer_steps_per_epoch=plan.optimizer_steps_per_epoch,
        epochs=plan.epochs,
        total_optimizer_steps=plan.total_optimizer_steps,
    )


@dataclass(frozen=True)
class Trajectory:
    """One sampled response with an immutable behavior-policy stamp."""

    trajectory_id: str
    group_id: str
    prompt_id: str
    sample_index: int
    policy_version: int
    behavior_logprob: float = 0.0
    target_logprob: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")


def validate_grpo_groups(
    trajectories: Iterable[Trajectory], expected_group_size: int
) -> tuple[str, ...]:
    """Reject incomplete, duplicated, or internally inconsistent GRPO groups."""

    if expected_group_size < 2:
        raise ValueError("GRPO expected_group_size must be at least two")
    groups: dict[str, list[Trajectory]] = {}
    trajectory_ids: set[str] = set()
    for trajectory in trajectories:
        if trajectory.trajectory_id in trajectory_ids:
            raise ValueError(f"duplicate trajectory id: {trajectory.trajectory_id}")
        trajectory_ids.add(trajectory.trajectory_id)
        groups.setdefault(trajectory.group_id, []).append(trajectory)

    for group_id, members in groups.items():
        if len(members) != expected_group_size:
            raise ValueError(
                f"GRPO group {group_id!r} has {len(members)} samples; "
                f"expected {expected_group_size}"
            )
        indices = sorted(member.sample_index for member in members)
        if indices != list(range(expected_group_size)):
            raise ValueError(f"GRPO group {group_id!r} has invalid sample indices")
        if len({member.prompt_id for member in members}) != 1:
            raise ValueError(f"GRPO group {group_id!r} mixes prompts")
        if len({member.policy_version for member in members}) != 1:
            raise ValueError(f"GRPO group {group_id!r} mixes policy versions")
    return tuple(groups)


def make_optimizer_batches(
    trajectories: Sequence[Trajectory],
    *,
    trajectories_per_batch: int,
    grpo_group_size: int,
) -> tuple[tuple[Trajectory, ...], ...]:
    """Pack complete groups into fixed optimizer batches without splitting them."""

    if trajectories_per_batch <= 0:
        raise ValueError("trajectories_per_batch must be positive")
    if trajectories_per_batch % grpo_group_size:
        raise ValueError("optimizer batch would split a GRPO group")
    group_order = validate_grpo_groups(trajectories, grpo_group_size)
    groups: dict[str, list[Trajectory]] = {group_id: [] for group_id in group_order}
    for trajectory in trajectories:
        groups[trajectory.group_id].append(trajectory)
    groups_per_batch = trajectories_per_batch // grpo_group_size
    if len(groups) % groups_per_batch:
        raise ValueError("trajectory set does not fill an integer number of optimizer batches")

    batches: list[tuple[Trajectory, ...]] = []
    ordered_ids = list(group_order)
    for start in range(0, len(ordered_ids), groups_per_batch):
        members: list[Trajectory] = []
        for group_id in ordered_ids[start : start + groups_per_batch]:
            members.extend(sorted(groups[group_id], key=lambda item: item.sample_index))
        batches.append(tuple(members))
    return tuple(batches)


@dataclass(frozen=True)
class StalenessFilterResult:
    accepted: tuple[Trajectory, ...]
    rejected: tuple[Trajectory, ...]


def policy_version_lag(trajectory: Trajectory, current_policy_version: int) -> int:
    if current_policy_version < trajectory.policy_version:
        raise ValueError("trajectory comes from a future policy version")
    return current_policy_version - trajectory.policy_version


def filter_by_staleness(
    trajectories: Iterable[Trajectory],
    *,
    current_policy_version: int,
    max_staleness: int,
) -> StalenessFilterResult:
    """Keep samples whose policy-version lag is at most ``max_staleness``."""

    if current_policy_version < 0:
        raise ValueError("current_policy_version must be non-negative")
    if max_staleness < 0:
        raise ValueError("max_staleness must be non-negative")
    accepted: list[Trajectory] = []
    rejected: list[Trajectory] = []
    for trajectory in trajectories:
        destination = (
            accepted
            if policy_version_lag(trajectory, current_policy_version) <= max_staleness
            else rejected
        )
        destination.append(trajectory)
    return StalenessFilterResult(tuple(accepted), tuple(rejected))


@dataclass(frozen=True)
class ImportanceResult:
    ratio: float
    clipped_ratio: float
    unclipped_objective: float
    clipped_objective: float


def clipped_importance_weight(
    *,
    behavior_logprob: float,
    target_logprob: float,
    advantage: float = 1.0,
    clip_epsilon: float = 0.2,
) -> ImportanceResult:
    """Clip a target/behavior ratio around one for the synchronous toy case.

    This equals PPO's proximal ratio only when the behavior policy is also the
    frozen old/update policy. Async decoupled correction needs separate
    behavior and proximal ratios and is intentionally outside this helper.
    """

    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in [0, 1)")
    ratio = math.exp(target_logprob - behavior_logprob)
    clipped_ratio = min(1.0 + clip_epsilon, max(1.0 - clip_epsilon, ratio))
    raw = ratio * advantage
    clipped = clipped_ratio * advantage
    return ImportanceResult(ratio, clipped_ratio, raw, min(raw, clipped))


@dataclass(frozen=True)
class SimulationConfig:
    mode: str
    rollout_workers: int
    total_groups: int
    grpo_group_size: int
    groups_per_optimizer_step: int
    train_duration: float
    max_staleness: int = 1_000_000
    seed: int = 0
    rollout_durations: tuple[float, ...] | None = None

    def validate(self) -> None:
        if self.mode not in {"sync", "async"}:
            raise ValueError("mode must be 'sync' or 'async'")
        integer_fields = {
            "rollout_workers": self.rollout_workers,
            "total_groups": self.total_groups,
            "grpo_group_size": self.grpo_group_size,
            "groups_per_optimizer_step": self.groups_per_optimizer_step,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError("worker, group, and batch counts must be positive")
        if self.grpo_group_size < 2:
            raise ValueError("GRPO requires at least two trajectories per prompt group")
        if self.total_groups % self.groups_per_optimizer_step:
            raise ValueError("total_groups must fill complete optimizer steps")
        if self.train_duration <= 0:
            raise ValueError("train_duration must be positive")
        if self.max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")
        if self.rollout_durations is not None:
            if len(self.rollout_durations) != self.total_groups:
                raise ValueError("rollout_durations must have one entry per group")
            if any(duration <= 0 for duration in self.rollout_durations):
                raise ValueError("rollout durations must be positive")

    def resolved_durations(self) -> tuple[float, ...]:
        self.validate()
        if self.rollout_durations is not None:
            return tuple(float(item) for item in self.rollout_durations)
        rng = random.Random(self.seed)
        return tuple(float(rng.randint(1, 8)) for _ in range(self.total_groups))


@dataclass(frozen=True)
class SimulationEvent:
    time: float
    kind: str
    group_id: str | None = None
    worker_id: int | None = None
    policy_version: int | None = None
    current_policy_version: int | None = None


@dataclass(frozen=True)
class SimulationResult:
    mode: str
    makespan: float
    generated_groups: int
    accepted_groups: int
    rejected_groups: int
    trained_groups: int
    queued_groups: int
    optimizer_steps: int
    final_policy_version: int
    max_version_lag: int
    mean_version_lag: float
    p95_version_lag: float
    max_queue_depth: int
    dropped_trajectories: int
    rollout_utilization: float
    learner_utilization: float
    effective_sample_fraction: float
    barrier_idle: float
    throughput_trajectories_per_time: float
    trajectories: tuple[Trajectory, ...]
    events: tuple[SimulationEvent, ...]


@dataclass(frozen=True)
class _CompletedGroup:
    job_id: int
    policy_version: int
    trajectories: tuple[Trajectory, ...]


def _make_group(job_id: int, policy_version: int, group_size: int) -> _CompletedGroup:
    group_id = f"group-{job_id:04d}"
    prompt_id = f"prompt-{job_id:04d}"
    members = tuple(
        Trajectory(
            trajectory_id=f"{group_id}/sample-{sample_index}",
            group_id=group_id,
            prompt_id=prompt_id,
            sample_index=sample_index,
            policy_version=policy_version,
            behavior_logprob=-0.1 * (sample_index + 1) - 0.001 * job_id,
            target_logprob=-0.1 * (sample_index + 1) - 0.001 * job_id,
        )
        for sample_index in range(group_size)
    )
    return _CompletedGroup(job_id, policy_version, members)


def _percentile_nearest_rank(values: Sequence[int], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for a non-empty sample."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def _event_sort_key(event: SimulationEvent) -> tuple[float, int, int, str]:
    # At equal times, publication precedes rollout starts so the event log states
    # the same causality used by the simulator.
    priority = {
        "optimizer_finished": 0,
        "policy_published": 1,
        "rollout_finished": 2,
        "stale_group_rejected": 3,
        "optimizer_started": 4,
        "rollout_started": 5,
    }
    return (event.time, priority.get(event.kind, 99), event.worker_id or -1, event.group_id or "")


class PostTrainingSimulator:
    """Event-driven sync/async rollout and training queue model."""

    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config

    def run(self) -> SimulationResult:
        if self.config.mode == "sync":
            return self._run_sync()
        return self._run_async()

    def _run_sync(self) -> SimulationResult:
        config = self.config
        durations = config.resolved_durations()
        events: list[SimulationEvent] = []
        all_trajectories: list[Trajectory] = []
        current_time = 0.0
        policy_version = 0
        barrier_idle = 0.0
        optimizer_steps = 0
        max_lag = 0

        for batch_start in range(0, config.total_groups, config.groups_per_optimizer_step):
            batch_jobs = range(batch_start, batch_start + config.groups_per_optimizer_step)
            worker_free = [current_time] * config.rollout_workers
            completed: list[tuple[float, _CompletedGroup]] = []
            for job_id in batch_jobs:
                worker_id = min(range(config.rollout_workers), key=lambda index: (worker_free[index], index))
                start = worker_free[worker_id]
                finish = start + durations[job_id]
                worker_free[worker_id] = finish
                events.append(
                    SimulationEvent(start, "rollout_started", f"group-{job_id:04d}", worker_id, policy_version, policy_version)
                )
                completed.append((finish, _make_group(job_id, policy_version, config.grpo_group_size)))

            barrier_end = max(worker_free)
            busy_time = sum(durations[job_id] for job_id in batch_jobs)
            barrier_idle += config.rollout_workers * (barrier_end - current_time) - busy_time
            for finish, group in sorted(completed, key=lambda item: (item[0], item[1].job_id)):
                events.append(
                    SimulationEvent(finish, "rollout_finished", f"group-{group.job_id:04d}", None, group.policy_version, policy_version)
                )
                all_trajectories.extend(group.trajectories)
                max_lag = max(max_lag, policy_version - group.policy_version)

            events.append(SimulationEvent(barrier_end, "optimizer_started", policy_version=policy_version, current_policy_version=policy_version))
            current_time = barrier_end + config.train_duration
            optimizer_steps += 1
            events.append(SimulationEvent(current_time, "optimizer_finished", policy_version=policy_version, current_policy_version=policy_version))
            policy_version += 1
            events.append(SimulationEvent(current_time, "policy_published", policy_version=policy_version, current_policy_version=policy_version))

        trained_groups = config.total_groups
        trajectory_count = trained_groups * config.grpo_group_size
        lag_observations = [0] * config.total_groups
        rollout_busy = sum(durations)
        learner_busy = optimizer_steps * config.train_duration
        return SimulationResult(
            mode="sync",
            makespan=current_time,
            generated_groups=config.total_groups,
            accepted_groups=config.total_groups,
            rejected_groups=0,
            trained_groups=trained_groups,
            queued_groups=0,
            optimizer_steps=optimizer_steps,
            final_policy_version=policy_version,
            max_version_lag=max_lag,
            mean_version_lag=sum(lag_observations) / len(lag_observations),
            p95_version_lag=_percentile_nearest_rank(lag_observations, 0.95),
            max_queue_depth=config.groups_per_optimizer_step,
            dropped_trajectories=0,
            rollout_utilization=rollout_busy / (config.rollout_workers * current_time),
            learner_utilization=learner_busy / current_time,
            effective_sample_fraction=1.0,
            barrier_idle=barrier_idle,
            throughput_trajectories_per_time=trajectory_count / current_time,
            trajectories=tuple(all_trajectories),
            events=tuple(sorted(events, key=_event_sort_key)),
        )

    def _run_async(self) -> SimulationResult:
        config = self.config
        durations = config.resolved_durations()
        events: list[SimulationEvent] = []
        all_trajectories: list[Trajectory] = []
        completion_heap: list[tuple[float, int, int, int]] = []
        queue: list[_CompletedGroup] = []
        next_job = 0
        policy_version = 0
        optimizer_steps = 0
        trainer_end: float | None = None
        generated_groups = 0
        rejected_groups = 0
        trained_groups = 0
        max_lag = 0
        lag_observations: list[int] = []
        max_queue_depth = 0
        current_time = 0.0

        def start_rollout(worker_id: int, start: float) -> None:
            nonlocal next_job
            if next_job >= config.total_groups:
                return
            job_id = next_job
            next_job += 1
            version = policy_version
            finish = start + durations[job_id]
            heapq.heappush(completion_heap, (finish, worker_id, job_id, version))
            events.append(
                SimulationEvent(start, "rollout_started", f"group-{job_id:04d}", worker_id, version, policy_version)
            )

        for worker_id in range(min(config.rollout_workers, config.total_groups)):
            start_rollout(worker_id, 0.0)

        while completion_heap or trainer_end is not None:
            next_rollout_time = completion_heap[0][0] if completion_heap else math.inf
            next_train_time = trainer_end if trainer_end is not None else math.inf
            current_time = min(next_rollout_time, next_train_time)

            if trainer_end is not None and trainer_end == current_time:
                events.append(SimulationEvent(current_time, "optimizer_finished", policy_version=policy_version, current_policy_version=policy_version))
                optimizer_steps += 1
                policy_version += 1
                events.append(SimulationEvent(current_time, "policy_published", policy_version=policy_version, current_policy_version=policy_version))
                trainer_end = None

            freed_workers: list[int] = []
            while completion_heap and completion_heap[0][0] == current_time:
                _, worker_id, job_id, sampled_version = heapq.heappop(completion_heap)
                group = _make_group(job_id, sampled_version, config.grpo_group_size)
                queue.append(group)
                generated_groups += 1
                all_trajectories.extend(group.trajectories)
                events.append(
                    SimulationEvent(current_time, "rollout_finished", f"group-{job_id:04d}", worker_id, sampled_version, policy_version)
                )
                freed_workers.append(worker_id)

            max_queue_depth = max(max_queue_depth, len(queue))
            if trainer_end is None:
                fresh_queue: list[_CompletedGroup] = []
                for group in queue:
                    lag = policy_version - group.policy_version
                    if lag > config.max_staleness:
                        lag_observations.append(lag)
                        max_lag = max(max_lag, lag)
                        rejected_groups += 1
                        events.append(
                            SimulationEvent(current_time, "stale_group_rejected", f"group-{group.job_id:04d}", policy_version=group.policy_version, current_policy_version=policy_version)
                        )
                    else:
                        fresh_queue.append(group)
                queue = fresh_queue
                if len(queue) >= config.groups_per_optimizer_step:
                    selected = queue[: config.groups_per_optimizer_step]
                    del queue[: config.groups_per_optimizer_step]
                    for group in selected:
                        lag = policy_version - group.policy_version
                        lag_observations.append(lag)
                        max_lag = max(max_lag, lag)
                    trained_groups += config.groups_per_optimizer_step
                    trainer_end = current_time + config.train_duration
                    events.append(SimulationEvent(current_time, "optimizer_started", policy_version=policy_version, current_policy_version=policy_version))

            for worker_id in sorted(freed_workers):
                start_rollout(worker_id, current_time)

        # A final filter makes the queue accounting truthful even when too few
        # groups remain to form another optimizer batch.
        fresh_queue = []
        for group in queue:
            lag = policy_version - group.policy_version
            lag_observations.append(lag)
            max_lag = max(max_lag, lag)
            if lag > config.max_staleness:
                rejected_groups += 1
                events.append(
                    SimulationEvent(current_time, "stale_group_rejected", f"group-{group.job_id:04d}", policy_version=group.policy_version, current_policy_version=policy_version)
                )
            else:
                fresh_queue.append(group)
        queue = fresh_queue
        accepted_groups = generated_groups - rejected_groups
        trajectory_count = trained_groups * config.grpo_group_size
        generated_trajectories = generated_groups * config.grpo_group_size
        rollout_busy = sum(durations)
        learner_busy = optimizer_steps * config.train_duration
        return SimulationResult(
            mode="async",
            makespan=current_time,
            generated_groups=generated_groups,
            accepted_groups=accepted_groups,
            rejected_groups=rejected_groups,
            trained_groups=trained_groups,
            queued_groups=len(queue),
            optimizer_steps=optimizer_steps,
            final_policy_version=policy_version,
            max_version_lag=max_lag,
            mean_version_lag=(sum(lag_observations) / len(lag_observations) if lag_observations else 0.0),
            p95_version_lag=_percentile_nearest_rank(lag_observations, 0.95),
            max_queue_depth=max_queue_depth,
            dropped_trajectories=rejected_groups * config.grpo_group_size,
            rollout_utilization=(rollout_busy / (config.rollout_workers * current_time) if current_time else 0.0),
            learner_utilization=(learner_busy / current_time if current_time else 0.0),
            effective_sample_fraction=(trajectory_count / generated_trajectories if generated_trajectories else 0.0),
            barrier_idle=0.0,
            throughput_trajectories_per_time=(trajectory_count / current_time if current_time else 0.0),
            trajectories=tuple(all_trajectories),
            events=tuple(sorted(events, key=_event_sort_key)),
        )


def long_tail_config(mode: str, *, seed: int = 7) -> SimulationConfig:
    """Small deterministic scenario where a sync barrier waits on a straggler."""

    return SimulationConfig(
        mode=mode,
        rollout_workers=3,
        total_groups=12,
        grpo_group_size=4,
        groups_per_optimizer_step=3,
        train_duration=0.5,
        max_staleness=100,
        seed=seed,
        rollout_durations=(8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )


def _result_summary(result: SimulationResult) -> dict[str, object]:
    return {
        "mode": result.mode,
        "makespan": result.makespan,
        "throughput_trajectories_per_time": result.throughput_trajectories_per_time,
        "optimizer_steps": result.optimizer_steps,
        "final_policy_version": result.final_policy_version,
        "max_version_lag": result.max_version_lag,
        "mean_version_lag": result.mean_version_lag,
        "p95_version_lag": result.p95_version_lag,
        "max_queue_depth": result.max_queue_depth,
        "dropped_trajectories": result.dropped_trajectories,
        "rollout_utilization": result.rollout_utilization,
        "learner_utilization": result.learner_utilization,
        "effective_sample_fraction": result.effective_sample_fraction,
        "barrier_idle": result.barrier_idle,
        "generated_groups": result.generated_groups,
        "trained_groups": result.trained_groups,
        "rejected_groups": result.rejected_groups,
        "queued_groups": result.queued_groups,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CPU-only post-training queue simulator")
    parser.add_argument("--mode", choices=("sync", "async", "both"), default="both")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    modes = ("sync", "async") if args.mode == "both" else (args.mode,)
    summaries = [_result_summary(PostTrainingSimulator(long_tail_config(mode, seed=args.seed)).run()) for mode in modes]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
