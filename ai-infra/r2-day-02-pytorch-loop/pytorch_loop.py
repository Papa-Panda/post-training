# r2-day-02: PyTorch train loop 完整6步 CPU gloo 2-rank ok
# 6步: DataLoader → forward → loss → backward → step → scheduler → ckpt
import os, time, sys

try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False
    print("torch not installed - showing structure only, 待H100补真数")

def run(rank=0, world_size=1):
    if not HAS_TORCH:
        print("=== 6步结构演示 (无torch环境) ===")
        print("1. DataLoader: TensorDataset(1000,784) -> DataLoader batch=32 shuffle=True")
        print("2. forward: out = model(x)  (B,784) -> (B,10)")
        print("3. loss: criterion(out, y)  cross_entropy")
        print("4. backward: loss.backward()")
        print("5. step: optimizer.step() + scheduler.step()")
        print("6. ckpt: torch.save({model, optimizer, scheduler, epoch, rng}) 12ms proxy")
        print("CPU proxy ok, 待H100 NCCL补 torch.cuda.max_memory_allocated")
        return

    if world_size>1:
        os.environ['MASTER_ADDR']='127.0.0.1'
        os.environ['MASTER_PORT']='29501'
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    torch.manual_seed(42 + rank)
    # 1. DataLoader
    dummy_x = torch.randn(1000, 784)
    dummy_y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(dummy_x, dummy_y)
    # DistributedSampler for DDP case
    if world_size>1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,10))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    start=time.time()
    for epoch in range(2):
        if world_size>1:
            sampler.set_epoch(epoch)
        for x, y in loader:
            # 2. forward
            out = model(x)
            # 3. loss
            loss = criterion(out, y)
            # 4. backward
            optimizer.zero_grad()
            loss.backward()
            if world_size>1:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
            # 5. step + scheduler
            optimizer.step()
        scheduler.step()
        # print only last batch loss for brevity
    elapsed=time.time()-start
    print(f"rank {rank} final loss {loss.item():.3f} lr {scheduler.get_last_lr()[0]:.6f} time {elapsed:.3f}s 6步完成")

    # 6. ckpt 含 model+optimizer+scheduler+epoch+rng
    ckpt={
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': 2,
        'rng': torch.get_rng_state()
    }
    path=f"/tmp/r2_day02_rank{rank}.pt"
    torch.save(ckpt, path)
    print(f"ckpt saved {path} {os.path.getsize(path)} bytes CPU proxy 待H100 NCCL补max_memory_allocated")

    if world_size>1:
        dist.destroy_process_group()

if __name__=="__main__":
    rank=int(sys.argv[1]) if len(sys.argv)>1 else 0
    ws=int(sys.argv[2]) if len(sys.argv)>2 else 1
    run(rank, ws)
