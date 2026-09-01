"""DDP semantic helpers plus an optional PyTorch/Gloo-or-NCCL demo.

The pure-Python helpers let CI test data ownership, gradient averaging, ring
volume, and replica equality without pretending that a simulation benchmarks a
real distributed backend.
"""

import argparse
import math
import os
from typing import Iterable, Sequence


def distributed_indices(
    dataset_size: int,
    world_size: int,
    rank: int,
    *,
    drop_last: bool = False,
) -> list[int]:
    """Model DistributedSampler ownership for shuffle=False.

    Non-divisible datasets are padded by repeating leading indices unless
    ``drop_last`` is true. Therefore disjointness is guaranteed only in the
    divisible case (or after dropping the tail).
    """
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid world_size or rank")
    if drop_last:
        samples_per_rank = dataset_size // world_size
    else:
        samples_per_rank = math.ceil(dataset_size / world_size)
    total_size = samples_per_rank * world_size
    indices = list(range(dataset_size))
    if not drop_last and total_size > dataset_size and dataset_size:
        needed = total_size - dataset_size
        indices.extend(indices[i % dataset_size] for i in range(needed))
    else:
        indices = indices[:total_size]
    return indices[rank:total_size:world_size]


def average_rank_gradients(rank_gradients: Sequence[Sequence[float]]) -> list[float]:
    """Average equally weighted per-rank gradient vectors."""
    if not rank_gradients:
        raise ValueError("at least one rank gradient is required")
    width = len(rank_gradients[0])
    if any(len(gradient) != width for gradient in rank_gradients):
        raise ValueError("gradient vectors must have equal length")
    return [
        sum(gradient[index] for gradient in rank_gradients) / len(rank_gradients)
        for index in range(width)
    ]


def sgd_step(parameters: Sequence[float], gradient: Sequence[float], lr: float) -> list[float]:
    if len(parameters) != len(gradient):
        raise ValueError("parameters and gradient must have equal length")
    return [value - lr * grad for value, grad in zip(parameters, gradient)]


def replicas_equal(replicas: Iterable[Sequence[float]], tolerance: float = 0.0) -> bool:
    replicas = list(replicas)
    if not replicas:
        return True
    reference = replicas[0]
    return all(
        len(replica) == len(reference)
        and all(abs(left - right) <= tolerance for left, right in zip(reference, replica))
        for replica in replicas[1:]
    )


def ring_allreduce_bytes(payload_bytes: int, world_size: int) -> float:
    if payload_bytes < 0 or world_size < 2:
        raise ValueError("payload must be non-negative and world_size at least two")
    return 2 * (world_size - 1) / world_size * payload_bytes


def _run_torch_demo(checkpoint: str) -> None:
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import DataLoader, TensorDataset
        from torch.utils.data.distributed import DistributedSampler
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run test_ddp_semantics.py for the "
            "dependency-free semantic checks."
        ) from exc

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    # Every rank constructs the same logical dataset and same initial module.
    data_generator = torch.Generator().manual_seed(20260901)
    features = torch.randn(64, 8, generator=data_generator)
    labels = torch.randint(0, 2, (64,), generator=data_generator)
    dataset = TensorDataset(features, labels)

    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=31,
            drop_last=False,
        )
        if distributed
        else None
    )
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, shuffle=sampler is None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    def checksum():
        module = model.module if distributed else model
        flat = torch.cat(
            [parameter.detach().reshape(-1).float() for parameter in module.parameters()]
        )
        return torch.stack((flat.sum(), flat.square().sum()))

    def assert_replicas_match(stage: str) -> None:
        if not distributed:
            return
        local = checksum()
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        if not all(
            torch.allclose(gathered[0], item, atol=1e-7, rtol=1e-7)
            for item in gathered[1:]
        ):
            raise RuntimeError(f"replica parameter mismatch {stage}")

    assert_replicas_match("after DDP initialization")
    if sampler is not None:
        sampler.set_epoch(0)

    model.train()
    for features_batch, labels_batch in loader:
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(features_batch), labels_batch)
        loss.backward()
        optimizer.step()
        assert_replicas_match("after optimizer step")

    # Rank-local minibatch losses are not expected to be equal. Parameters are.
    if rank == 0:
        module = model.module if distributed else model
        torch.save(
            {"model": module.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint,
        )
        print(f"checkpoint={checkpoint}; final rank-0 minibatch loss={loss.item():.6f}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/tmp/r2_day04_ddp.pt")
    args = parser.parse_args()
    _run_torch_demo(args.checkpoint)


if __name__ == "__main__":
    main()
