"""r2-Day19 复盘周：第二层（分布式训练）端到端大账。

把 r2-Day13/14/15/16/17/18 已手算的公式收拢到一个可跑的总账里：
  - 显存账：Day14 模型状态台账 / Day16 AMP / Day17 FSDP 均摊 / Day18 checkpoint
  - 通信账：Day03/Day15 ring all-reduce / Day15 TP-4-allreduce+PP bubble /
    Day17 ZeRO-3 1.5x DDP / Day18 DCP sharded vs rank0 gather
  - 决策闭环：Day17 框架选型口径 + Day18 Young/Daly 最优间隔

纯 Python，无 torch / CUDA 依赖，CPU 可跑。
GB 口径：十进制 1e9（与 Day14/15/16/17/18 口径一致）。
所有"实测"数字均为 Day14-18 已建立的手算口径复用；
带宽换算时间为 theoretical estimate，已声明。
"""

from __future__ import annotations

import math

GB = 1e9  # 十进制，与 Day14/15/16/17/18 一致


# ---------------------------------------------------------------------------
# Day14：模型状态台账（params + grads + optimizer states）
# ---------------------------------------------------------------------------
def model_state_ledger(P, bytes_param=2, bytes_grad=2, bytes_opt=8):
    """Day14 台账。默认 fp16 参数/梯度（2B）+ Adam fp32 m/v（8B）= 12 B/param。

    7B：14 + 14 + 56 = 84 GB > 80 GB -> 单卡装不下（Day14 结论）。
    """
    params = P * bytes_param
    grads = P * bytes_grad
    opt = P * bytes_opt
    total = params + grads + opt
    return {
        "params_gb": params / GB,
        "grads_gb": grads / GB,
        "opt_gb": opt / GB,
        "total_gb": total / GB,
        "bytes_per_param": bytes_param + bytes_grad + bytes_opt,
    }


def amp_state_ledger(P):
    """Day16：AMP 混合精度台账。

    fp16 参数 2 + fp16 梯度 2 + fp32 master 4 + fp32 Adam m/v 8 = 16 B/param。
    与纯 fp32 训练（4+4+8=16）同账：BF16/FP16 省的是数值范围，不是字节。
    7B：14 + 14 + 28 + 56 = 112 GB。
    """
    params_fp16 = P * 2
    grads_fp16 = P * 2
    master_fp32 = P * 4
    opt_fp32 = P * 8
    total = params_fp16 + grads_fp16 + master_fp32 + opt_fp32
    return {
        "params_fp16_gb": params_fp16 / GB,
        "grads_fp16_gb": grads_fp16 / GB,
        "master_fp32_gb": master_fp32 / GB,
        "opt_fp32_gb": opt_fp32 / GB,
        "total_gb": total / GB,
        "bytes_per_param": 16,
    }


# ---------------------------------------------------------------------------
# Day17：框架/分片（FSDP = ZeRO-3 生产形态）
# ---------------------------------------------------------------------------
def fsdp_per_rank_gb(total_gb, N):
    """Day17：FSDP（ZeRO-3）每 rank 均摊 = total / N。

    7B / N=8：84 / 8 = 10.5 GB。
    """
    return total_gb / N


def zero3_comm_gb(ddp_comm_gb):
    """Day17：ZeRO-3 每步通信 = 1.5 x DDP（forward/backward 各一次 gather）。

    7B / N=8：DDP 24.5 GB/step -> ZeRO-3 36.75 GB/step。
    """
    return 1.5 * ddp_comm_gb


# ---------------------------------------------------------------------------
# Day03/Day15：集合通信
# ---------------------------------------------------------------------------
def ring_allreduce_traffic(msg_bytes, N):
    """ring all-reduce 单 rank 流量 = 2(N-1)/N x M（Day03 公式，Day15 复用）。

    手算锚点（Day15）：M = 67,108,864 B，N = 8 -> ~117.4 MB。
    """
    return 2.0 * (N - 1) / N * msg_bytes


def tp_traffic(b, S, H, T, L, bytes_elem=2, ars_per_layer=4):
    """Day15 例 A：TP 机内 all-reduce 流量。

    payload = b*S*H 个元素（行并行输出）；
    每层 ars_per_layer=4 次（attn out + mlp out，forward 2 + backward 2）；
    TP 域大小 T（ring 公式走 T）。
    返回 (per_ar_bytes, per_layer_bytes, total_bytes)。

    手算锚点（70B 量级：H=8192, bf16, b=1, S=4096, T=8, L=80）：
    per_ar ~117.4 MB，per_layer ~469.8 MB，total ~37.6 GB。
    """
    payload = b * S * H * bytes_elem
    per_ar = ring_allreduce_traffic(payload, T)
    per_layer = ars_per_layer * per_ar
    return per_ar, per_layer, L * per_layer


def pp_bubble(p, m):
    """Day15 例 C：GPipe bubble = (p-1)/(m+p-1)。

    p=4, m=8 -> 3/11 ~ 27.3%；p=4, m=32 -> 3/35 ~ 8.6%。
    """
    return (p - 1) / (m + p - 1)


def comm_time_s(total_bytes, bw_gbps):
    """带宽换算时间下限（theoretical estimate，非实测）。

    Day15 锚点：37.6 GB / 900 GB/s ~ 41.8 ms；/ 50 GB/s ~ 0.75 s（18x）。
    """
    return total_bytes / (bw_gbps * GB)


# ---------------------------------------------------------------------------
# Day18：checkpoint & recovery（DCP）
# ---------------------------------------------------------------------------
def checkpoint_ledger(P, bytes_param=2, bytes_opt=8):
    """Day18 台账：params + opt states，不含 grads（step 内瞬态）= 10 B/param。

    7B：14 + 56 = 70 GB。与 Day14 的 84 GB 差 14 GB，正是 grads。
    """
    total = P * (bytes_param + bytes_opt)
    return {
        "total_gb": total / GB,
        "bytes_per_param": bytes_param + bytes_opt,
    }


def dcp_sharded_vs_gather(P, N, bytes_param=2, bytes_opt=8):
    """Day18：sharded 每 rank = 总量/N（零额外通信）；
    rank0 gather 额外流量 = (N-1)/N x 总量（单向汇聚，没有 ring 的 x2）。

    7B / N=64：per-rank 1.09375 GB；gather 额外 68.90625 GB。
    """
    total = P * (bytes_param + bytes_opt)
    return {
        "per_rank_gb": total / N / GB,
        "gather_extra_gb": (N - 1) / N * total / GB,
    }


def young_daly(delta_s, M_s):
    """Day18：tau* = sqrt(2*delta*M)，W(tau) = delta/tau + tau/(2M)。

    锚点：delta=10s, M=1000s -> tau*~141.4s, W~14.14%；
    delta=60s, M=10h -> tau*~34.6min, W~5.8%；
    async（stall~55ms）, M=10h -> tau*~63s, W~0.17%。
    """
    tau = math.sqrt(2 * delta_s * M_s)
    w = delta_s / tau + tau / (2 * M_s)
    return {"tau_s": tau, "waste": w}


def checkpoint_worth_opening(delta_s, M_s, T_train_s):
    """Day18 讨论题：训练总时长 T_train < tau* -> 一次都轮不到，不赚。"""
    tau = young_daly(delta_s, M_s)["tau_s"]
    return {"tau_s": tau, "worth": T_train_s >= tau}


# ---------------------------------------------------------------------------
# Day13：KV cache 口径（复盘讨论题 d 问：MLA 省的是哪笔账）
# ---------------------------------------------------------------------------
def kv_bytes_per_token_per_layer(scheme):
    """Day13 例 A：每 token 每层 KV 字节（fp16 口径，直接复用 Day13 表）。

    mha 16384 / gqa8 4096 / mla 1152 / mqa 512（B）。
    MLA = (512+64)x2，相对 MHA 14.2x（16384/1152）。
    """
    table = {"mha": 16384, "gqa8": 4096, "mla": 1152, "mqa": 512}
    return table[scheme]


# ---------------------------------------------------------------------------
# 端到端验收：7B fp16+Adam，N=64（TP=8/PP=4/DP=2），delta=60s，M=10h
# ---------------------------------------------------------------------------
def end_to_end_7b():
    """复盘讨论题的完整手算答案（纯公式，无实测）。"""
    P = 7e9
    states = model_state_ledger(P)
    ckpt = checkpoint_ledger(P)
    per_ar, per_layer, total_tp = tp_traffic(1, 4096, 8192, 8, 80)
    yd = young_daly(60, 10 * 3600)
    worth = checkpoint_worth_opening(60, 10 * 3600, 2 * 3600)
    return {
        # (a) 每 rank：FSDP 口径 model-state 与 checkpoint
        "per_rank_state_gb": fsdp_per_rank_gb(states["total_gb"], 64),
        "per_rank_ckpt_gb": ckpt["total_gb"] / 64,
        # (b) TP 走线：80 层总量 + NVLink/IB 时间下限（theoretical）
        "tp_total_gb": total_tp / GB,
        "tp_nvlink_ms": comm_time_s(total_tp, 900) * 1e3,
        "tp_ib_s": comm_time_s(total_tp, 50),
        # (c) Young/Daly + 2h 训练是否值得开 checkpoint
        "tau_star_min": yd["tau_s"] / 60,
        "waste": yd["waste"],
        "worth_2h": worth["worth"],
        # (d) MLA：训练侧台账不变；推理侧 KV（7B 风格 128k 上下文，GiB）
        "mla_kv_gib_vs_mha": (
            kv_bytes_per_token_per_layer("mha") * 32 * 131072 / 2**30,
            kv_bytes_per_token_per_layer("mla") * 32 * 131072 / 2**30,
        ),
    }


if __name__ == "__main__":
    r = end_to_end_7b()
    print("per-rank model-state (FSDP, N=64): %.4f GB" % r["per_rank_state_gb"])
    print("per-rank checkpoint (DCP, N=64):  %.4f GB" % r["per_rank_ckpt_gb"])
    print("TP total (80 layers):             %.2f GB" % r["tp_total_gb"])
    print("  NVLink 900GB/s: %.1f ms | IB 50GB/s: %.2f s (theoretical)"
          % (r["tp_nvlink_ms"], r["tp_ib_s"]))
    print("Young/Daly tau*: %.1f min, W=%.2f%%; 2h worth=%s"
          % (r["tau_star_min"], r["waste"] * 100, r["worth_2h"]))
    print("KV 128k ctx: MHA %.1f GiB -> MLA %.1f GiB (Day13)"
          % r["mla_kv_gib_vs_mha"])
