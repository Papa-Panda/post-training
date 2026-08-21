"""
Day 15 - Megatron 3D Parallelism decision tree
CPU gloo 2-rank ok, 待H100 NCCL 补 max_memory_allocated + TP AllReduce bandwidth + PP bubble实测

模拟 Megatron DP/TP/PP 决策:
- 计算 per-GPU memory proxy = param_bytes/G_TP/PP + activation + overhead
- 决策树: 何时切 TP, 何时切 PP
- 通信占比 proxy: DP AllReduce ~ P, TP AllGather per layer

Links:
 Day14 PUE 1.2576 -> per-GPU memory决定能否跑，跑不起来PUE无意义
 Day13 Tj 90.5C throttle 2.5% -> TP scatter thermal burst
 Day07 checkpoint per-block -> TP/PP sharded ckpt同源
"""

import os
import math
import argparse

def estimate_mem(param_B, G, tp=1, pp=1, seq=2048, hidden=4096, layers=32, dtype_bytes=2):
    # param bytes
    param_total_GB = param_B * 1e9 * dtype_bytes / (1024**3)  # rough: 7B *2 =14GB
    # activation proxy: seq*hidden*layers*2*4 bytes per micro-batch ~(approx)
    # simple: act per GPU ∝ seq*hidden*layers / TP / PP * batch factor
    act_base_GB = (seq * hidden * layers * 2 * 4) / (1024**3)  # ~1GB per? scaled up later
    # scale act heuristically to match typical: 7B act ~4GB, 13B ~8GB, 70B ~42GB at large seq
    act_scale = {7:4.2, 13:7.8, 70:42.0}
    act = act_scale.get(param_B, 4.2)
    # TP shards act linearly partially
    act_sharded = act / tp / (1 if pp==1 else 1.2)  # PP slightly reduces act per stage due to pipeline staging
    param_sharded = param_total_GB / tp / pp if tp>1 or pp>1 else param_total_GB / G if G>1 else param_total_GB
    # FSDP/DP sharding: if pure DP with G, param / G
    if tp==1 and pp==1 and G>1:
        param_sharded = param_total_GB / G
        act_sharded = act  # DP keeps full act per rank (unless activation checkpoint)
    # overhead: optimizer states if training (Adam 8 bytes per param) sharded by DP/FSDP
    # for inference proxy we ignore optimizer, for RL training include
    # RL training includes optimizer: 8 bytes per param / G
    opt_per_gpu = (param_B*1e9*8)/(1024**3) / G  # Adam states
    # total per GPU approx for training
    per_gpu_infer = param_sharded + act_sharded
    per_gpu_train = param_sharded + act_sharded + opt_per_gpu * 0.5  # ZeRO-1 style half
    return {
        "param_total_GB": param_total_GB,
        "param_sharded_GB": param_sharded,
        "act_GB": act_sharded,
        "opt_shard_GB": opt_per_gpu,
        "infer_GB": per_gpu_infer,
        "train_GB": per_gpu_train,
    }

def comm_proxy(param_B, G, tp, pp):
    # DP AllReduce size ∝ param_B
    dp_comm = (param_B/7.0) * 1.2  # GB proxy per step
    tp_comm_per_layer = 0.02 * tp  # GB per layer proxy for AllGather
    tp_total = tp_comm_per_layer * 32 * (32/32)  # layers placeholder
    pp_bubble = (pp-1)/8.0 if pp>1 else 0  # bubble fraction proxy  micro_batch=8
    total_comm_pct = min(0.30, (dp_comm*0.01 + tp_total*0.05 + pp_bubble*0.2))
    return {
        "dp_GB": dp_comm,
        "tp_GB": tp_total,
        "pp_bubble_frac": pp_bubble,
        "comm_pct_wall": total_comm_pct*100,
    }

def decision(model_B, G):
    mem_dp = estimate_mem(model_B, G, tp=1, pp=1)
    # try TP options
    candidates = []
    for tp in [1,2,4]:
        for pp in [1,2,4]:
            if tp*pp > G:  # need G>=tp*pp for simple mapping, allow remainder DP
                continue
            mem = estimate_mem(model_B, G, tp=tp, pp=pp)
            comm = comm_proxy(model_B, G, tp, pp)
            # OOM threshold 80GB H100
            oom = mem["infer_GB"] > 78 or mem["train_GB"] > 78
            # scoring: prefer low memory + low comm
            score = mem["infer_GB"]*1.0 + comm["comm_pct_wall"]*0.5 + (100 if oom else 0)
            candidates.append((score, tp, pp, mem, comm, oom))
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    # human readable decision
    tp, pp = best[1], best[2]
    if tp==1 and pp==1:
        dec = f"DP/FSDP only (G={G}) sufficient - perGPU infer {best[3]['infer_GB']:.1f}GB"
    elif pp==1:
        dec = f"TP={tp}+DP (G={G}) - scatter GEMM {model_B}B, Tj Scatter, comm {best[4]['comm_pct_wall']:.1f}%"
    else:
        dec = f"TP={tp}+PP={pp}+DP rem (G={G}) - bubble {best[4]['pp_bubble_frac']*100:.0f}%, infer {best[3]['infer_GB']:.1f}GB train {best[3]['train_GB']:.1f}GB"
    return best, dec

def main():
    # support torchrun env
    try:
        import torch.distributed as dist
        import torch
        has_dist = True
    except:
        has_dist = False
    rank = 0
    world = 1
    if has_dist and "RANK" in os.environ:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()

    models = [7,13,70]
    Gs = [1,2,4,8]
    print(f"[Rank {rank}/{world}] Day15 Megatron 3D Parallelism CPU proxy (待H100 NCCL 补 max_memory_allocated)")
    results = []
    for m in models:
        for G in Gs:
            if G > world and world>1:
                # in 2-rank run we simulate G=2,4,8 logically but only G=2 physically runs
                if G>2 and world==2:
                    # still compute decision logically
                    pass
            best, dec = decision(m, G)
            score,tp,pp,mem,comm,oom = best
            results.append((m,G,tp,pp,mem["infer_GB"],mem["train_GB"],comm["comm_pct_wall"],oom,dec))
            if rank==0:
                print(f"Model {m}B G={G} -> {dec} | infer {mem['infer_GB']:.2f}GB train {mem['train_GB']:.2f}GB comm {comm['comm_pct_wall']:.1f}% oom={oom}")

    # 2-rank consensus check via all_reduce of best decisions count
    if has_dist and world>1:
        import torch
        t = torch.tensor([len(results)], dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if rank==0:
            print(f"[gloo ok] all_reduce SUM len(results) = {t.item()} expected {len(results)*world}")
        # consensus of 7B G2 infer_GB
        for m,G,tp,pp,infer,train,comm_pct,oom,dec in results:
            if m==7 and G==2:
                v = torch.tensor([infer], dtype=torch.float32)
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                if rank==0:
                    avg = v.item()/world
                    print(f"[consensus] 7B G=2 infer_GB avg {avg:.2f} GB gloo_ok {abs(avg-infer)<1e-3}")
                break
        dist.destroy_process_group()

    if rank==0:
        # Save summary JSON for NOTES
        import json, pathlib
        pathlib.Path("tmp_summary.json").write_text(json.dumps([r[4:7] for r in results][:3], indent=2))
        print("Done - 待H100 NCCL: torch.cuda.max_memory_allocated + TP AllReduce BW + PP bubble 12-20% + TTFT/TPOT")

if __name__ == "__main__":
    main()
