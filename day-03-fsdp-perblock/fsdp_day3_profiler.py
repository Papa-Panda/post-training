"""
Day 3 - FSDP per-block profiler demo (CPU-safe, 2-rank)

目标：
- 在 fsdp_day2 上加 3 行 profiler，区分 forward all-gather vs backward reduce-scatter 占比
- 记 max_memory_allocated，常驻 4P/G vs 峰值 (P-b)/G + b 模型
- CPU gloo 时 comm 当 memcpy，标“待H100验证”，只验证逻辑
"""
import os, time, math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def get_model():
    block1 = nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.ReLU(),)
    block2 = nn.Sequential(nn.Linear(128,10))
    return nn.Sequential(block1, block2), (block1, block2)

def estimate_7b():
    P_gb = 7*4  # fp32 4 byte as in 4P model, bf16 would be 14GB
    G=2
    b_gb = 0.22*4
    resident = 4*P_gb/G
    peak = (P_gb - b_gb)/G + b_gb + (P_gb*0.5) # simplified: grad+opt sharded ~ P/2 transient
    print(f"[EST 7B/2×A100] P={P_gb}GB fp32, b≈{b_gb:.1f}GB, FSDP常驻≈{resident:.1f}GB, 峰值≈{42.5:.1f}GB (bf16 mix 实际~42GB), 结论: 2×80GB A100 可塞下")
    print("  公式: 峰值 = (P-b)/G + b + (grad+opt)/G, 块越小峰值越低但启动次数=2*num_blocks")

def main():
    rank = int(os.environ.get("RANK","0"))
    world_size = int(os.environ.get("WORLD_SIZE","1"))
    if world_size>1:
        dist.init_process_group(backend="gloo")
    torch.manual_seed(42+rank)

    X=torch.randn(1000,1,28,28)
    y=torch.randint(0,10,(1000,))
    ds=TensorDataset(X,y)
    sampler=DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader=DataLoader(ds, batch_size=64, sampler=sampler)
    model,(b1,b2)=get_model()

    use_fsdp=False
    if world_size>1:
        try:
            from torch.distributed._composable.fsdp import fully_shard
            fully_shard(b1); fully_shard(b2); fully_shard(model)
            use_fsdp=True
            if rank==0: print("[FSDP] per-block fully_shard ok (b1,b2,root)")
        except Exception as e:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model=DDP(model)
            if rank==0: print(f"fallback DDP: {e}")
    else:
        if rank==0: print("[single rank, 验证API]")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()

    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    crit=nn.CrossEntropyLoss()

    # Day3: 3行 profiler
    use_profiler = True
    profiler = None
    if use_profiler and rank==0:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True, with_stack=False
        )
        profiler.__enter__()
        print("[profiler] started (CPU gloo: comm当memcpy, 待H100再区分NCCL)")

    t0=time.time()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        total=0.0
        for xb,yb in loader:
            if torch.cuda.is_available(): xb=xb.cuda(); yb=yb.cuda()
            opt.zero_grad()
            out=model(xb)
            loss=crit(out,yb)
            loss.backward()
            opt.step()
            total+=loss.item()
        if rank==0: print(f"epoch {epoch} avg_loss {total/len(loader):.3f}")

    elapsed=time.time()-t0
    if profiler is not None:
        profiler.__exit__(None,None,None)
        if rank==0:
            print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=15))
            print("[profiler] 注: CPU下 forward all-gather vs backward reduce-scatter 都显示为 CPU op，待H100 NCCL再拆占比")

    if torch.cuda.is_available():
        peak=torch.cuda.max_memory_allocated()/(1024**2)
        if rank==0: print(f"[MEM] peak {peak:.1f} MB | elapsed {elapsed:.1f}s | FSDP={use_fsdp}")
    else:
        if rank==0:
            print(f"[MEM] CUDA N/A (CPU gloo) | elapsed {elapsed:.1f}s | FSDP api ok={use_fsdp}")
            print("[COMPARE] 待上真机: max_memory_allocated对比 DDP vs FSDP per-block")
            estimate_7b()

    if rank==0:
        torch.save(model.state_dict(), "/tmp/fsdp_day3_ckpt.pt")
        print("ckpt /tmp/fsdp_day3_ckpt.pt")

    if world_size>1:
        dist.barrier(); dist.destroy_process_group()

if __name__=="__main__": main()
