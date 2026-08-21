#!/usr/bin/env python3
"""
Day 14 — Paper3 PUE拆解 → 训练成本 per token / per useful rollout
CPU gloo 2-rank runnable, 待H100 NCCL 补 max_memory_allocated + nvidia-smi Tj + RAPL

Mapping Paper3 PUE → RL Infra COST:
  Data-center PUE = (P_IT + P_cooling + P_loss) / P_IT
  RL  $$/useful = PUE * (GPU $ * wall-clock + 重试 $ + eval 空转 $) / useful_rollouts

- P_IT: GPU 本体功率 450W base + burst 250W (vLLM 长CoT 500→5000 导致 80% 墙钟功率尖峰)
- P_cooling: Paper2 双线性 IT*外温 + 二次换热 COP + 冷冻机 hyst 0.85/0.35 + 风机立方定律 flow^3  → 外推到机柜风扇 + 冷机
- useful_rollout: verifier 通过且 reward σ 阈值过滤后 (Day12) 且非 5 类 infra 失败 (Day10/13)
- COST: $ / 1k useful rollouts, $/k tokens proxy, idle waste GPU-hr, PUE overhead $

Run:
  python pue_cost_model.py
  torchrun --nproc_per_node=2 pue_cost_model.py
"""
import os
import json
import math
import random
import time
from collections import defaultdict

def simulate_pue(seed=42, n_roll=300, gpu_dollar_per_hr=3.2):
    random.seed(seed)
    # Paper3 PUE-like modeling params (simplified from Paper2 mech SSM, no mortgage analogy)
    # COP ~ 5.2 baseline but degrades quadratic with ΔT
    # cooling power = IT * (1/COP) + fan cubic + hyst overhead
    results = []
    power_it_samples = []
    power_cooling_samples = []
    pue_samples = []
    tj_samples = []
    throttle = 0
    fails = defaultdict(int)
    useful = 0
    filtered_high_uncert = 0  # Day12 connection: σ>0.15 filtered, not infra fail
    total_gpu_sec = 0.0
    retry_gpu_sec = 0.0
    eval_idle_sec = 0.0

    # hyst state for chiller
    chiller_on = False
    for i in range(n_roll):
        is_burst = (i % 35 == 0)  # long CoT burst every 35
        is_tool_call = (i % 7 == 0)  # tool-use rollout longer
        # duration: short 8-15s, long 60-120s scaled to CPU sim 0.08-0.15s / 0.6-1.2s
        dur = random.uniform(0.08,0.15)
        if is_burst:
            dur = random.uniform(0.6,1.2)  # scaled, real 60-120s
        elif is_tool_call:
            dur = random.uniform(0.25,0.45)
        total_gpu_sec += dur

        # 5类失败 + reward uncertainty 过滤 (Day12 OAS calibration offset 视角)
        p_fail = 0.16 if is_burst else 0.06
        # reward uncertainty proxy: ensemble σ 0.045 mean, high-uncert 0-2%
        rew_sigma = random.gauss(0.045, 0.02)
        rew_sigma = max(0.0, rew_sigma)
        is_high_uncert = rew_sigma > 0.15

        if is_high_uncert and random.random() < 0.7:
            filtered_high_uncert += 1
            # still spend GPU but not counted as useful, add small retry cost
            retry_gpu_sec += dur * 0.3
            # no fail bucket, just filtered
        elif random.random() < p_fail:
            r = random.random()
            if r < 0.38:
                fails['timeout'] += 1
            elif r < 0.63:
                fails['tool_retry'] += 1
            elif r < 0.80:
                fails['vcj_parse'] += 1
            elif r < 0.93:
                fails['oom_kv'] += 1
            else:
                fails['nccl'] += 1
            retry_gpu_sec += dur * 0.8  # retry cost
            # eval idle due to failure debug: add 0.1s scaled
            eval_idle_sec += random.uniform(0.05,0.12)
        else:
            useful += 1
            # small eval idle for async verifier (Day08 nowcasting EWMA)
            eval_idle_sec += random.uniform(0.01,0.04) if not is_burst else random.uniform(0.04,0.10)

        # P_IT thermal (Day11 Paper2 SSM → Tj mapping)
        p_it = 460 + (260 if is_burst else 0) + (80 if is_tool_call else 0) + random.uniform(-18,18)
        # Tj proxy
        tj = 54 + p_it*0.038 + (6 if is_burst else 0) + random.uniform(-1.5,1.5)
        if tj > 82:
            throttle += 1
            p_it *= 0.92  # throttling lowers IT but wastes perf

        # outdoor delta proxy for cooling: 12-28°C range
        t_out = 18 + 10*math.sin(i/45) + random.uniform(-2,2)
        delta_t = max(0, (tj - t_out) * 0.6)
        # COP degrad: COP = 5.2 - 0.04*delta_t - 0.002*delta_t^2 (二次项 from Paper3)
        cop = max(1.2, 5.2 - 0.04*delta_t - 0.002*delta_t*delta_t)
        # chiller hyst 0.85 on / 0.35 off load factor
        load_factor = p_it / 1000.0  # normalized
        if not chiller_on and load_factor > 0.85:
            chiller_on = True
        elif chiller_on and load_factor < 0.35:
            chiller_on = False
        # fan flow cubic: flow ∝ load_factor ^0.9
        flow = load_factor**0.9
        p_fan = 28 * (flow**3) + 6  # W per GPU rack amortized
        p_chiller = (p_it / cop) * (1.15 if chiller_on else 0.35)
        p_loss = p_it * 0.03  # PDU loss 3%
        p_cooling = p_fan + p_chiller + p_loss
        pue = (p_it + p_cooling) / p_it if p_it>0 else 1.0

        power_it_samples.append(p_it)
        power_cooling_samples.append(p_cooling)
        pue_samples.append(pue)
        tj_samples.append(tj)

    total_gpu_hour = total_gpu_sec / 3600.0
    retry_gpu_hour = retry_gpu_sec / 3600.0
    eval_idle_hour = eval_idle_sec / 3600.0

    # $ calc
    gpu_cost_total = (total_gpu_sec + retry_gpu_sec + eval_idle_sec) * (gpu_dollar_per_hr/3600.0)
    # PUE overhead $ = (PUE-1)*IT $ 
    pue_mean = sum(pue_samples)/len(pue_samples)
    pue_p50 = sorted(pue_samples)[len(pue_samples)//2]
    pue_p95 = sorted(pue_samples)[int(0.95*len(pue_samples))]
    pue_max = max(pue_samples)
    pue_min = min(pue_samples)

    # facility 总费用带 PUE
    facility_cost = gpu_cost_total * pue_mean  # simplified: multiply IT $ by PUE
    # cost per useful rollout
    cost_per_useful = facility_cost / useful if useful>0 else 0
    cost_per_1k_useful = cost_per_useful * 1000
    # $ / k tokens proxy: assume avg 2k tokens per rollout (mixed short/long), 40% useful have 3k
    avg_tokens_per_rollout = 2100
    total_useful_tokens = useful * avg_tokens_per_rollout
    cost_per_1k_tokens = (facility_cost / total_useful_tokens * 1000) if total_useful_tokens>0 else 0

    # inefficiency breakdown
    power_it_mean = sum(power_it_samples)/len(power_it_samples)
    power_cool_mean = sum(power_cooling_samples)/len(power_cooling_samples)
    tj_max = max(tj_samples)
    tj_avg = sum(tj_samples)/len(tj_samples)
    throttle_rate = throttle / len(tj_samples)

    overhead_ratio = (facility_cost - gpu_cost_total) / gpu_cost_total if gpu_cost_total>0 else 0

    return {
        "n_roll": n_roll,
        "useful": useful,
        "useful_ratio": useful/n_roll,
        "filtered_high_uncert": filtered_high_uncert,
        "fails": dict(fails),
        "fail_rate": sum(fails.values())/n_roll,
        "gpu_sec": total_gpu_sec,
        "retry_sec": retry_gpu_sec,
        "eval_idle_sec": eval_idle_sec,
        "gpu_hour_total": total_gpu_hour+retry_gpu_hour+eval_idle_hour,
        "gpu_cost_it": gpu_cost_total,
        "facility_cost": facility_cost,
        "pue_mean": pue_mean,
        "pue_p50": pue_p50,
        "pue_p95": pue_p95,
        "pue_min": pue_min,
        "pue_max": pue_max,
        "pue_overhead_ratio": overhead_ratio,
        "power_it_mean": power_it_mean,
        "power_cooling_mean": power_cool_mean,
        "cost_per_useful_rollout": cost_per_useful,
        "cost_per_1k_useful": cost_per_1k_useful,
        "cost_per_1k_tokens": cost_per_1k_tokens,
        "tj_max": tj_max,
        "tj_avg": tj_avg,
        "throttle_rate": throttle_rate,
        "gpu_dollar_per_hr": gpu_dollar_per_hr,
    }

def main():
    use_dist = False
    rank = 0
    world = 1
    try:
        import torch.distributed as dist
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="gloo")
            use_dist = True
            rank = dist.get_rank()
            world = dist.get_world_size()
    except Exception:
        pass

    t0 = time.time()
    sim = simulate_pue(seed=42, n_roll=300, gpu_dollar_per_hr=3.2)
    t1 = time.time()
    sim["wall_sec_single"] = t1-t0
    sim["mode"] = "single"
    sim["cpu_true_numbers"] = True

    if use_dist:
        import torch
        # all_reduce facility_cost and useful counts to prove gloo sync
        vals = torch.tensor([sim["facility_cost"], float(sim["useful"]), sim["pue_mean"]], dtype=torch.float32)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        sim["dist_world"] = world
        sim["dist_rank"] = rank
        sim["dist_facility_cost_avg"] = vals[0].item()/world
        sim["dist_useful_avg"] = vals[1].item()/world
        sim["dist_pue_avg"] = vals[2].item()/world
        sim["dist_note"] = "gloo CPU 2-rank ok, 待H100 NCCL + torch.cuda.max_memory_allocated + nvidia-smi Tj + RAPL"
        try:
            dist.barrier()
        except Exception:
            pass
        if rank != 0:
            print(json.dumps({"rank":rank,"world":world,"gloo_ok":True,
                              "pue_mean":sim["pue_mean"],"cost_per_1k_useful":sim["cost_per_1k_useful"]},ensure_ascii=False))
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            return
    else:
        sim["dist_note"] = "single ok, torchrun --nproc_per_node=2 可验证 gloo, 待H100 NCCL 补显存 + Tj"

    # rank0 print
    print(json.dumps(sim, ensure_ascii=False, indent=2))
    print("\n--- 3 CPU真数 (待H100 NCCL 补 max_memory_allocated + nvidia-smi Tj + RAPL) ---")
    print(f"PUE mean {sim['pue_mean']:.4f} p50 {sim['pue_p50']:.4f} p95 {sim['pue_p95']:.4f} min {sim['pue_min']:.3f} max {sim['pue_max']:.3f} overhead {sim['pue_overhead_ratio']:.2%}")
    print(f"$/useful rollout {sim['cost_per_useful_rollout']:.6f} $ /1k useful {sim['cost_per_1k_useful']:.4f} $ /1k tokens proxy {sim['cost_per_1k_tokens']:.5f}")
    print(f"useful {sim['useful']}/{sim['n_roll']}={sim['useful_ratio']:.3f} fail {sim['fail_rate']:.3%} filtered_high_uncert {sim['filtered_high_uncert']} "
          f"gpu_sec {sim['gpu_sec']:.1f}s retry {sim['retry_sec']:.1f}s idle {sim['eval_idle_sec']:.1f}s Tj_avg {sim['tj_avg']:.1f}C Tj_max {sim['tj_max']:.1f}C throttle {sim['throttle_rate']:.2%}")
    print(f"power IT {sim['power_it_mean']:.1f}W cooling {sim['power_cooling_mean']:.1f}W facility_cost ${sim['facility_cost']:.4f} IT_cost ${sim['gpu_cost_it']:.4f}")
    print("\n待H100: torch.cuda.max_memory_allocated() + nvidia-smi Tj + nvml power trace + 真vLLM 500→5000 tok rollout长尾 12-18%失败替换模拟")

    if use_dist:
        try:
            import torch.distributed as dist
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()
