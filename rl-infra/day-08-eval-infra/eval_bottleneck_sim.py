"""
Eval infra bottleneck simulation — CPU gloo 2-rank verifiable, H100 NCCL待验证。

Goal: 理解 eval 为什么卡住训练，量化 P50/P95, queue depth, GPU idle。

Design:
- train loop 10 steps, every 3 steps triggers eval
- eval latency simulated: base 0.6-1.4s + 10% flaky retry (+2s)
- rollout占比 80% wall-clock模拟：train step 0.2s, eval 0.6-1.4s
- 2-rank: rank0 = train orchestrator, rank1 = eval worker
  - sync mode: barrier before/after eval → trains blocked
  - async mode: rank0 不等，继续 train，rank1 后台 eval

Metrics output:
- eval_latency_p50 / p95 (sec)
- queue_depth avg / max
- gpu_idle_time (sec) sync vs async
- bottleneck_ratio = eval_time / total_time
- 3 CPU numbers for NOTES.md

Usage:
  torchrun --nproc_per_node=2 eval_bottleneck_sim.py
  torchrun --nproc_per_node=2 eval_bottleneck_sim.py --async-mode
  python eval_bottleneck_sim.py  # single rank fallback

Fail-closed: 不编 H100 数，GPU 有则打印 max_memory_allocated，否则标待H100。

关联 Paper1 nowcasting: EWMA 预测 eval 延迟，queue depth 超阈转异步。
"""

import argparse
import os
import random
import time
import statistics
from collections import deque

import torch
import torch.distributed as dist


def init_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # torchrun sets MASTER_ADDR/PORT
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        return rank, world_size
    else:
        # single process fallback for quick验
        return 0, 1


def simulate_eval_latency(rng, base_low=0.6, base_high=1.4, flaky_prob=0.1, flaky_penalty=2.0):
    base = rng.uniform(base_low, base_high)
    if rng.random() < flaky_prob:
        base += flaky_penalty + rng.uniform(0, 0.5)  # retry
        flaky = True
    else:
        flaky = False
    return base, flaky


def ewma_predict(history, alpha=0.3):
    if not history:
        return 0.0
    pred = history[0]
    for x in history[1:]:
        pred = alpha * x + (1 - alpha) * pred
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--async-mode", action="store_true", help="async eval: rank0 不 barrier，等同 nowcasting 预测后跳过阻塞")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world_size = init_dist()
    rng = random.Random(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
        # tiny dummy alloc to show max_memory if GPU present
        _ = torch.randn(1024, 1024, device=device)
    else:
        device = torch.device("cpu")

    eval_latencies = []
    flaky_count = 0
    queue_depths = []
    eval_queue = deque()

    total_start = time.time()
    gpu_idle_accum = 0.0
    train_time_accum = 0.0

    # for nowcasting demo
    recent_latencies = []

    # train loop
    for step in range(1, args.train_steps + 1):
        # simulate train step 0.2s + dummy compute
        t0 = time.time()
        # dummy compute: 0.1M param matmul similar to Day07
        a = torch.randn(256, 256)
        b = torch.randn(256, 256)
        _ = torch.mm(a, b)
        time.sleep(0.05)  # small sleep to emulate forward/backward cpu

        # trigger eval?
        if step % args.eval_every == 0:
            # simulate eval arrival into queue
            eval_queue.append(step)
            queue_depths.append(len(eval_queue))
            pred = ewma_predict(recent_latencies, alpha=0.3)

            if rank == 0:
                print(f"[rank{rank}] step {step} trigger eval, queue_depth={len(eval_queue)}, nowcasting_pred={pred:.2f}s")

            # eval worker handling (rank1 or single)
            if world_size > 1:
                # barrier sync unless async
                if not args.async_mode:
                    # sync: train blocked waiting eval
                    if rank in (0, 1):
                        # eval latency sim - should be same on both ranks via broadcast of seed? we simulate independently but barrier makes them meet
                        latency, flaky = simulate_eval_latency(rng)
                        # ensure both ranks have similar latency via all_reduce avg? simple: let rank1 decide and broadcast
                        if rank == 1:
                            # rank1 is eval worker, simulate work
                            time.sleep(latency * 0.2)  # scaled down for CPU test speed; real eval 5-10s
                        # sync point - both wait
                        if dist.is_initialized():
                            dist.barrier()
                        eval_latencies.append(latency)
                        if flaky:
                            flaky_count += 1
                        recent_latencies.append(latency)
                        eval_queue.popleft()
                        idle = latency * 0.2
                        gpu_idle_accum += idle if not args.async_mode else 0
                        if rank == 0:
                            print(f"[rank{rank}] eval done step {step} latency {latency:.2f}s flaky={flaky} idle_accum {gpu_idle_accum:.2f}s")
                else:
                    # async: rank0 continues, rank1 does eval in background (simulated as no barrier)
                    if rank == 1:
                        latency, flaky = simulate_eval_latency(rng)
                        time.sleep(latency * 0.2)
                        eval_latencies.append(latency)
                        recent_latencies.append(latency)
                        if flaky:
                            flaky_count += 1
                        if eval_queue:
                            eval_queue.popleft()
                        if rank == 1:
                            print(f"[rank{rank} async] eval step {step} latency {latency:.2f}s")
                    if rank == 0:
                        # rank0 does not wait
                        pass
            else:
                # single rank
                latency, flaky = simulate_eval_latency(rng)
                if not args.async_mode:
                    time.sleep(latency * 0.2)
                    gpu_idle_accum += latency * 0.2
                eval_latencies.append(latency)
                recent_latencies.append(latency)
                if flaky:
                    flaky_count += 1
                eval_queue.popleft()
                print(f"[single] step {step} eval latency {latency:.2f}s flaky={flaky} queue_after {len(eval_queue)}")

        t1 = time.time()
        train_time_accum += (t1 - t0)

    total_time = time.time() - total_start

    # gather metrics across ranks for report (only rank0 prints final summary)
    if world_size > 1 and dist.is_initialized():
        # gather eval_latencies lengths to rank0 via all_gather not trivial for variable sizes -> just rank0 summary from its own perspective
        # barrier before summary
        dist.barrier()

    if rank == 0:
        if len(eval_latencies) == 0:
            # in multi-rank async, rank0 eval_latencies may be empty (rank1 did work). synthesize from shared history for demo.
            # fallback: use recent_latencies seen via train side if any, else note.
            eval_latencies_report = recent_latencies
        else:
            eval_latencies_report = eval_latencies

        if len(eval_latencies_report) > 0:
            p50 = statistics.median(eval_latencies_report)
            # p95 approx: 95th percentile
            sorted_l = sorted(eval_latencies_report)
            idx = int(0.95 * len(sorted_l))
            idx = min(idx, len(sorted_l) - 1)
            p95 = sorted_l[idx]
            avg_q = sum(queue_depths) / len(queue_depths) if queue_depths else 0
            max_q = max(queue_depths) if queue_depths else 0
            bottleneck_ratio = (sum(eval_latencies_report) * 0.2) / total_time if total_time > 0 else 0
        else:
            p50 = p95 = avg_q = max_q = bottleneck_ratio = 0

        print("\n===== Eval Bottleneck Simulation Result =====")
        print(f"mode: {'async' if args.async_mode else 'sync'}  world_size={world_size}  train_steps={args.train_steps} eval_every={args.eval_every}")
        print(f"eval_count: {len(eval_latencies_report)}  flaky_count: {flaky_count}  flaky_rate: {flaky_count/max(1,len(eval_latencies_report)):.1%}")
        print(f"eval_latency_p50: {p50:.3f}s (scaled CPU, real eval 5-10x)")
        print(f"eval_latency_p95: {p95:.3f}s")
        print(f"queue_depth_avg: {avg_q:.2f}  max: {max_q}")
        print(f"gpu_idle_time_accum: {gpu_idle_accum:.3f}s  total_wall {total_time:.3f}s  bottleneck_ratio {bottleneck_ratio:.2%}")
        print(f"recent_latencies EWMA pred next: {ewma_predict(recent_latencies):.3f}s")
        print(f"3 numbers for NOTES: p50={p50:.3f}, queue_avg={avg_q:.2f}, gpu_idle={gpu_idle_accum:.3f}")

        if torch.cuda.is_available():
            print(f"CUDA available: max_memory_allocated={torch.cuda.max_memory_allocated()/1024**2:.1f}MB  max_reserved={torch.cuda.max_memory_reserved()/1024**2:.1f}MB")
        else:
            print("CUDA not available — 待H100 NCCL 补 max_memory_allocated, tokens/sec, P95 eval 8-15min 真值")

        print("\nNowcasting decision demo:")
        for qd, plat in [(2, 1.2), (5, 6.5), (7, 11.0)]:
            decision = "async/skip eval, 转后台" if (qd > 5 or plat > 10) else "sync eval 可接受"
            print(f"  queue_depth={qd}, pred_p95={plat}min -> {decision}")

        print("\nCoding Eval 3 flaky points mitigation:")
        print(" 1. sandbox 冷启动 5-10s → 热池 + 预热容器 + 重用")
        print(" 2. HumanEval 全量 3h block → 采样 20% fast-pass + 分级 (fast/ full)")
        print(" 3. verifier 非确定/超时 → 重试队列 + 超时 30s 熔断 + flaky 标记跳过")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
