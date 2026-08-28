# r2-day-04 DDP demo - 严格6步对齐Day02 + DDP扩展
# 6步: init_process_group → DataLoader(DistributedSampler) → DDP wrap → forward → loss/backward(DDP hook AllReduce) → step → ckpt
import os, sys

try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import TensorDataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False

def run(rank=0, world_size=1):
    if not HAS_TORCH:
        print("=== r2-Day04 DDP 6步结构 (无torch环境模拟) ===")
        print("[Step1] init_process_group: dist.init_process_group(backend='gloo', rank=rank, world_size=2)")
        print("  -> MASTER_ADDR=127.0.0.1 MASTER_PORT=29502")
        print("[Step2] DataLoader: TensorDataset(1000,784) + DistributedSampler(num_replicas=2, rank=rank) + DataLoader batch=32")
        print("  -> 每卡500样本，不重复，切分对应Day02的DataLoader")
        print("[Step3] DDP wrap: model=Sequential(Linear784-128, ReLU, Linear128-10) -> DDP(model)")
        print("  -> broadcast rank0权重，注册autograd hook，bucket 25MB")
        print("[Step4] forward: out=model(x)  (B,784)->(B,10)")
        print("[Step5] loss+backward: loss=cross_entropy(out,y) -> loss.backward()")
        print("  -> DDP hook自动 AllReduce grad SUM/WS，每bucket一次，overlap backward")
        print("[Step6] step+ckpt: optimizer.step() 各rank一致 -> torch.save({model,opt,epoch,rng})")
        print("  -> comm量≈参数量 101k*4B≈0.4MB per iter, 2-rank Ring 0.2MB")
        print("CPU proxy ok, 待H100 NCCL补 torch.cuda.max_memory_allocated + MFU")
        return

    # Step1: init
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29502'
    if world_size>1:
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        print(f"[Step1] rank{rank} init_process_group gloo world_size={world_size} ok")

    torch.manual_seed(42+rank)
    
    # Step2: DataLoader with DistributedSampler
    dataset=TensorDataset(torch.randn(1000,784), torch.randint(0,10,(1000,)))
    if world_size>1:
        sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader=DataLoader(dataset, batch_size=32, sampler=sampler)
        print(f"[Step2] rank{rank} DistributedSampler 500 samples")
    else:
        loader=DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"[Step2] rank{rank} DataLoader 1000 samples single")

    # Step3: DDP wrap
    model=torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,10))
    if world_size>1:
        model=DDP(model)
        print(f"[Step3] rank{rank} DDP wrap broadcast+hook bucket25MB")
    else:
        print(f"[Step3] rank{rank} single model no DDP")

    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion=torch.nn.CrossEntropyLoss()

    # Step4-6: train loop
    for epoch in range(2):
        if world_size>1:
            sampler.set_epoch(epoch)
        for x,y in loader:
            # Step4 forward
            out=model(x)
            # Step5 loss+backward
            loss=criterion(out, y)
            opt.zero_grad()
            loss.backward()  # DDP hook AllReduce here if world_size>1
            # Step6 step
            opt.step()
    print(f"[Step4-6] rank{rank} final loss {loss.item():.3f} forward+backward+step done")

    # ckpt
    ckpt_path=f"/tmp/r2_day04_rank{rank}.pt"
    torch.save({'model':model.module.state_dict() if world_size>1 else model.state_dict(),
                'optimizer':opt.state_dict(), 'epoch':2, 'rng':torch.get_rng_state()}, ckpt_path)
    print(f"[Step6 ckpt] rank{rank} saved {ckpt_path} {os.path.getsize(ckpt_path)} bytes CPU proxy 待H100 NCCL")

    if world_size>1:
        dist.destroy_process_group()

if __name__=="__main__":
    rank=int(sys.argv[1]) if len(sys.argv)>1 else 0
    ws=int(sys.argv[2]) if len(sys.argv)>2 else 1
    run(rank, ws)
