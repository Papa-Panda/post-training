"""
Day 2 - FSDP intro (CPU-safe, 2-rank)

对比 Day1 DDP: DDP 每卡全量模型，FSDP 按 ZeRO-3 思路把参数/梯度/优化器分片，
前向 all-gather 把 shard 聚成全参算，反向 reduce-scatter 再散回去。
Transformer 按 block 包是甜点 - 通信和计算能 overlap，避免 per-layer 太碎的启动开销。

用 torch.distributed._composable.fsdp.fully_shard 实现，CPU gloo 也能跑通，
GPU 上再看 max_memory_allocated 对比。
"""
import os, time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def get_model():
    # 两段当成两个 block，方便按 block fully_shard，类比 transformer block
    block1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
    )
    block2 = nn.Sequential(
        nn.Linear(128, 10)
    )
    return nn.Sequential(block1, block2), (block1, block2)

def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend="gloo")

    torch.manual_seed(42 + rank)

    n_samples = 1000
    X = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model, (block1, block2) = get_model()

    # --- FSDP wrapping (composable) ---
    use_fsdp = False
    if world_size > 1:
        try:
            from torch.distributed._composable.fsdp import fully_shard
            # per-block wrap: sweet spot. per-layer would be too fine, per-model too coarse.
            fully_shard(block1)
            fully_shard(block2)
            # wrap root as well
            fully_shard(model)
            use_fsdp = True
            if rank == 0:
                print("[FSDP] using fully_shard per-block (block1, block2, root)")
        except Exception as e:
            if rank == 0:
                print(f"[FSDP] fully_shard not available, fallback to DDP: {e}")
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model)
    else:
        if rank == 0:
            print("[FSDP] single rank, running without sharding (for API check)")

    # 显存测量（GPU 才有意义，CPU 打印 N/A）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    sampler.set_epoch(0)
    t0 = time.time()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        total = 0.0
        for xb, yb in loader:
            if torch.cuda.is_available():
                xb = xb.cuda()
                yb = yb.cuda()
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        if rank == 0:
            print(f"epoch {epoch} avg_loss {total/len(loader):.3f}")

    elapsed = time.time() - t0

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        if rank == 0:
            print(f"[MEM] peak {peak:.1f} MB | elapsed {elapsed:.1f}s | FSDP={use_fsdp}")
            print(f"[COMPARE] DDP 全量 = P, FSDP sharded ~ P/{world_size} + buffer. 2卡理论省 ~50% 参数显存")
    else:
        if rank == 0:
            print(f"[MEM] CUDA N/A (CPU gloo) | elapsed {elapsed:.1f}s | FSDP api ok={use_fsdp}")
            print(f"[COMPARE] 待上真机：torch.cuda.max_memory_allocated 对比 DDP vs FSDP")

    # rank0 checkpoint
    if rank == 0:
        state = model.state_dict() if not use_fsdp else model.state_dict()
        torch.save(state, "/tmp/fsdp_day2_ckpt.pt")
        print("checkpoint rank0 /tmp/fsdp_day2_ckpt.pt")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
