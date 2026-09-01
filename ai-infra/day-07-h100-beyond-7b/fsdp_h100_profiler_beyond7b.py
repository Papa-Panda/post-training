"""Explicit FSDP model-state capacity accounting for large models.

Despite the historical filename, this is an analytical planner, not a profiler.
It intentionally does not instantiate a 7B/13B/70B model and does not emit
synthetic throughput or communication percentages.  Use a real model plus
``torch.cuda.max_memory_allocated`` and a profiler for measured results.
"""

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StatePrecision:
    parameter_bytes: int = 2
    gradient_bytes: int = 2
    optimizer_bytes: int = 8
    master_parameter_bytes: int = 4

    @property
    def bytes_per_parameter(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
            + self.master_parameter_bytes
        )


@dataclass(frozen=True)
class CapacityEstimate:
    parameters: int
    ranks: int
    largest_layer_parameters: int
    bytes_per_parameter: int
    unsharded_model_state_gb: float
    fsdp_resident_model_state_gb_per_rank: float
    largest_layer_materialization_extra_gb: float
    fsdp_model_state_lower_bound_gb_per_rank: float


def estimate_capacity(
    *,
    parameters: int,
    ranks: int,
    largest_layer_parameters: int = 0,
    precision: StatePrecision = StatePrecision(),
) -> CapacityEstimate:
    """Return decimal-GB model-state accounting.

    The FSDP lower bound assumes parameters, gradients, and optimizer state are
    evenly sharded.  It adds the missing parameter shards needed to materialize
    the largest wrapped unit.  Activations, temporary buffers, allocator
    fragmentation, communication workspaces, embeddings outside wrapped units,
    and the runtime itself are intentionally excluded.
    """
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    if not 0 <= largest_layer_parameters <= parameters:
        raise ValueError("largest_layer_parameters must be in [0, parameters]")
    for value in asdict(precision).values():
        if value < 0:
            raise ValueError("precision byte counts must be non-negative")

    total_bytes = parameters * precision.bytes_per_parameter
    resident_bytes = total_bytes / ranks
    materialization_extra = (
        largest_layer_parameters
        * precision.parameter_bytes
        * (ranks - 1)
        / ranks
    )
    return CapacityEstimate(
        parameters=parameters,
        ranks=ranks,
        largest_layer_parameters=largest_layer_parameters,
        bytes_per_parameter=precision.bytes_per_parameter,
        unsharded_model_state_gb=total_bytes / 1e9,
        fsdp_resident_model_state_gb_per_rank=resident_bytes / 1e9,
        largest_layer_materialization_extra_gb=materialization_extra / 1e9,
        fsdp_model_state_lower_bound_gb_per_rank=(
            resident_bytes + materialization_extra
        )
        / 1e9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FSDP model-state capacity lower bound (not a benchmark)"
    )
    parser.add_argument("--params-b", type=float, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument(
        "--largest-layer-m",
        type=float,
        default=0.0,
        help="parameters in the largest FSDP wrapped unit, in millions",
    )
    parser.add_argument("--parameter-bytes", type=int, default=2)
    parser.add_argument("--gradient-bytes", type=int, default=2)
    parser.add_argument("--optimizer-bytes", type=int, default=8)
    parser.add_argument("--master-parameter-bytes", type=int, default=4)
    args = parser.parse_args()

    precision = StatePrecision(
        parameter_bytes=args.parameter_bytes,
        gradient_bytes=args.gradient_bytes,
        optimizer_bytes=args.optimizer_bytes,
        master_parameter_bytes=args.master_parameter_bytes,
    )
    report = estimate_capacity(
        parameters=round(args.params_b * 1e9),
        ranks=args.ranks,
        largest_layer_parameters=round(args.largest_layer_m * 1e6),
        precision=precision,
    )
    output = asdict(report)
    output["excluded_from_lower_bound"] = [
        "activations",
        "temporary buffers",
        "allocator fragmentation",
        "communication workspaces",
        "runtime/context memory",
    ]
    output["measurement_status"] = "analytical model only"
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
