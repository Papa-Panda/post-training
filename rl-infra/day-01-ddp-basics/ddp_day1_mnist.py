"""
DDP Day 1 - CPU 2-rank mnist demo
Matches briefing: torchrun 2 ranks, DistributedSampler, set_epoch, rank0 checkpoint, grad 1.0+3.0 -> 2.0验证
"""
import os, time, math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group(backend="gloo")
    
    torch.manual_seed(42 + rank)
    
    # Fake MNIST-like data (avoid download): 1000 samples 28*28=784 dim, 10 classes
    n_samples = 1000
    X = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(X, y)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Tiny model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)
    
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    
    # grad sync demo: rank0 grad 1.0, rank1 grad 3.0 -> averaged 2.0
    if world_size > 1:
        demo_param = torch.tensor([1.0 if rank==0 else 3.0])
        t0 = time.time()
        dist.all_reduce(demo_param, op=dist.ReduceOp.SUM)
        demo_param /= world_size
        allreduce_time = time.time() - t0
        if rank==0:
            print(f"[DEMO] grad sync: 1.0 & 3.0 -> {demo_param.item()} (expected 2.0) | all_reduce time {allreduce_time*1000:.1f}ms")
    
    # Train 2 epochs
    sampler.set_epoch(0)
    t_single = time.time()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if rank==0:
            print(f"epoch {epoch} avg_loss {ep_loss/len(loader):.3f}")
    
    if world_size>1:
        dist.barrier()
    
    # rank0写 checkpoint
    if rank==0:
        torch.save(model.state_dict(), "/tmp/ddp_day1_ckpt.pt")
        print("checkpoint saved rank0 /tmp/ddp_day1_ckpt.pt")
    
    if world_size>1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
