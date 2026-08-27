# r2-day-04 DDP demo CPU gloo 2-rank ok, 6步结构
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
        print("torch not installed - 6步DDP结构演示")
        print("1. init_process_group gloo")
        print("2. DistributedSampler 切数据")
        print("3. model DDP wrap")
        print("4. forward out=model(x)")
        print("5. loss.backward() DDP hook自动AllReduce grad")
        print("6. optimizer.step() 各rank一致")
        print("CPU proxy ok, 待H100 NCCL补真数")
        return
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29502'
    if world_size>1:
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    torch.manual_seed(42+rank)
    dataset=TensorDataset(torch.randn(1000,784), torch.randint(0,10,(1000,)))
    if world_size>1:
        sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader=DataLoader(dataset, batch_size=32, sampler=sampler)
    else:
        loader=DataLoader(dataset, batch_size=32, shuffle=True)
    model=torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,10))
    if world_size>1:
        model=DDP(model)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2):
        if world_size>1:
            sampler.set_epoch(epoch)
        for x,y in loader:
            out=model(x)
            loss=torch.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print(f"rank {rank} final loss {loss.item():.3f} DDP 6步完成")
    if world_size>1:
        dist.destroy_process_group()

if __name__=="__main__":
    rank=int(sys.argv[1]) if len(sys.argv)>1 else 0
    ws=int(sys.argv[2]) if len(sys.argv)>2 else 1
    run(rank, ws)
