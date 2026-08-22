# r2-day-02 minimal runnable: PyTorch train loop CPU gloo 2-rank ok
import torch, torch.distributed as dist, os, time

def run(rank=0, world_size=1):
    if world_size>1:
        os.environ['MASTER_ADDR']='127.0.0.1'
        os.environ['MASTER_PORT']='29501'
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # dummy data
    x = torch.randn(32,784)
    y = torch.randint(0,10,(32,))
    start=time.time()
    for epoch in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        if world_size>1:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        opt.step()
    elapsed=time.time()-start
    print(f"rank {rank} epoch loss {loss.item():.3f} time {elapsed:.3f}s")
    # ckpt
    ckpt={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':1,'rng':torch.get_rng_state()}
    torch.save(ckpt, f"/tmp/r2_day02_rank{rank}.pt")
    print(f"ckpt saved /tmp/r2_day02_rank{rank}.pt {os.path.getsize(f'/tmp/r2_day02_rank{rank}.pt')} bytes CPU proxy 待H100 NCCL补max_memory_allocated")
    if world_size>1:
        dist.destroy_process_group()

if __name__=="__main__":
    import sys
    rank=int(sys.argv[1]) if len(sys.argv)>1 else 0
    ws=int(sys.argv[2]) if len(sys.argv)>2 else 1
    run(rank, ws)
