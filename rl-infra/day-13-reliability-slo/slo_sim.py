"""
Day 13 - Reliability/SLO for RL training cluster
CPU gloo 2-rank ok, 待H100 NCCL 补 max_memory_allocated + NVML power trace

Simulates small cluster:
- jobs arrival Poisson, success/fail, queue depth, power jitter linked to vLLM rollout burst

3 SLOs:
  SLO1 job_success_rate >= 0.98 (rollout失败分类: timeout/tool/VCJ/OOM/NCCL)
  SLO2 queue_wait_p95 <= 1.2s scaled (real 120s) - eval异步排队
  SLO3 power_jitter p_std/mean <= 0.15 + Tj throttle_rate <=1%

Links to prev:
  Day12 reward σ 0.045 是SLO1的部分噪声源, Day11 Tj 82.49C throttle 0.83% 是SLO3直接输入
"""
import json
import os
import time
import math
import random
from collections import defaultdict

def simulate(seed=42, n_jobs=200):
    random.seed(seed)
    # arrival Poisson-ish discrete
    queue_depth = []
    wait_times = []
    successes = 0
    fails = defaultdict(int)  # 5类失败 taxonomy from Day10 vLLM
    power_samples = []
    tj_samples = []
    throttle_events = 0

    cur_queue = 0
    base_power = 450.0  # W per GPU base
    # simulate rollout burst影响power
    for i in range(n_jobs):
        # arrival burst every 40 jobs -> mimic Day10 long CoT 500->5000
        is_burst = (i % 40 == 0)
        ar = 1 + (3 if is_burst else 0)  # arrival count proxy
        cur_queue += ar
        # service 1-2 per step
        served = random.randint(1,2) if cur_queue>0 else 0
        cur_queue = max(0, cur_queue-served)
        queue_depth.append(cur_queue)
        wait = cur_queue * 0.15 + random.uniform(0,0.2)  # scaled sec
        if is_burst:
            wait += random.uniform(0.3,0.6)
        wait_times.append(wait)

        # success/fail sim: 5-8%短失败 12-18%长失败 proxy
        p_fail = 0.15 if is_burst else 0.06
        if random.random() < p_fail:
            # 分类
            r = random.random()
            if r < 0.40:
                fails['timeout'] += 1
            elif r < 0.70:
                fails['tool_retry'] += 1
            elif r < 0.85:
                fails['vcj_parse'] += 1
            elif r < 0.95:
                fails['oom_kv'] += 1
            else:
                fails['nccl'] += 1
        else:
            successes += 1

        # power / thermal power模仿Day11两节点SSM简化
        p = base_power + (250 if is_burst else 0) + random.uniform(-20,20) + 80*math.sin(i/10)
        # Tj proxy from power
        tj = 55 + p*0.04 + random.uniform(-2,2)  # 55 + 450*0.04=73 baseline
        if is_burst:
            tj += 6
        power_samples.append(p)
        tj_samples.append(tj)
        if tj > 82:
            throttle_events += 1

    total = successes + sum(fails.values())
    success_rate = successes / total if total>0 else 0
    # p95 queue wait
    sorted_wait = sorted(wait_times)
    p95 = sorted_wait[int(0.95*len(sorted_wait))] if sorted_wait else 0
    p50 = sorted_wait[int(0.5*len(sorted_wait))] if sorted_wait else 0
    avg_q = sum(queue_depth)/len(queue_depth) if queue_depth else 0
    # power jitter
    mean_p = sum(power_samples)/len(power_samples)
    var_p = sum((x-mean_p)**2 for x in power_samples)/len(power_samples)
    p_std = math.sqrt(var_p)
    jitter_ratio = p_std/mean_p if mean_p>0 else 0
    tj_max = max(tj_samples) if tj_samples else 0
    tj_avg = sum(tj_samples)/len(tj_samples) if tj_samples else 0
    throttle_rate = throttle_events/len(tj_samples) if tj_samples else 0

    slo1_pass = success_rate >= 0.98
    # scaled: real SLO 120s -> sim 1.2s
    slo2_pass = p95 <= 1.2
    slo3_pass = (jitter_ratio <= 0.15) and (throttle_rate <= 0.01)

    return {
        "seed": seed,
        "n_jobs": n_jobs,
        "success_rate": success_rate,
        "successes": successes,
        "fails": dict(fails),
        "fail_rate": 1-success_rate,
        "queue_p50": p50,
        "queue_p95": p95,
        "queue_avg_depth": avg_q,
        "queue_max_depth": max(queue_depth) if queue_depth else 0,
        "power_mean": mean_p,
        "power_std": p_std,
        "power_jitter_ratio": jitter_ratio,
        "tj_max": tj_max,
        "tj_avg": tj_avg,
        "throttle_rate": throttle_rate,
        "throttle_events": throttle_events,
        "slo1_success_rate_pass": slo1_pass,
        "slo2_queue_p95_pass": slo2_pass,
        "slo3_power_jitter_pass": slo3_pass,
        "slo_all_pass": slo1_pass and slo2_pass and slo3_pass,
    }

def main():
    # distributed init optional
    use_dist = False
    rank = 0
    world = 1
    try:
        import torch.distributed as dist
        import torch
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="gloo")
            use_dist = True
            rank = dist.get_rank()
            world = dist.get_world_size()
    except Exception:
        pass

    t0 = time.time()
    result = simulate(seed=42, n_jobs=200)
    t1 = time.time()
    result["wall_sec_single"] = t1-t0
    result["mode"] = "single"

    # 2-rank path: rank0模拟后all_reduce成功率均值演示
    if use_dist:
        import torch
        # 把success_rate同步取平均，展示gloo可用
        sr_t = torch.tensor([result["success_rate"]], dtype=torch.float32)
        dist.all_reduce(sr_t, op=dist.ReduceOp.SUM)
        # 这里演示通信 ok
        result["dist_world"] = world
        result["dist_rank"] = rank
        result["dist_success_rate_avg"] = (sr_t.item()/world) if world>0 else result["success_rate"]
        result["mode"] = f"dist_rank{rank}_world{world}"
        if rank == 0:
            result["dist_note"] = "gloo CPU 2-rank ok, 待H100 NCCL + torch.cuda.max_memory_allocated"
        # barrier模拟eval同步阻塞 同构Day08
        try:
            dist.barrier()
        except Exception:
            pass
        if rank != 0:
            # non-zero rank不打印主JSON避免重复
            print(json.dumps({"rank":rank,"world":world,"gloo_ok":True,"sr_avg": float(sr_t.item()/world),"wait_p95":result["queue_p95"]},ensure_ascii=False))
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            return
    else:
        result["dist_note"] = "single进程跑通，torchrun --nproc_per_node=2 可验证gloo，待H100 NCCL补显存"

    # 打印汇总 rank0 only
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("\n--- 3 SLO判定 ---")
    print(f"SLO1 job成功率 {result['success_rate']:.3f} >=0.98 ? {result['slo1_success_rate_pass']}  (fail {result['fail_rate']:.3f} 5类:{result['fails']})")
    print(f"SLO2 queue p95 {result['queue_p95']:.3f}s (scaled 1.2s≈真实120s) p50 {result['queue_p50']:.3f}s avg_depth {result['queue_avg_depth']:.2f} 通过? {result['slo2_queue_p95_pass']}")
    print(f"SLO3 power jitter {result['power_jitter_ratio']:.3f} (σ={result['power_std']:.1f}W mean={result['power_mean']:.1f}W) tj_max {result['tj_max']:.1f}°C throttle {result['throttle_rate']:.3%} 通过? {result['slo3_power_jitter_pass']}")
    print(f"ALL PASS? {result['slo_all_pass']}  wall {result['wall_sec_single']:.3f}s")
    print("\n待H100 NCCL: max_memory_allocated + nvidia-smi Tj + RAPL power trace + 真vLLM 500→5000 tok失败率替换sim")

    if use_dist:
        try:
            import torch.distributed as dist
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()
