#!/usr/bin/env python3
"""r2-Day17：Megatron / DeepSpeed / FSDP 选型 —— 一句话区分、显存/通信台账、选型决策表。

纯 Python（仅标准库），不依赖 torch / CUDA / NCCL。
README 里的全部手算数字都在这里建模，断言见 test_framework_selection.py。

符号含义（与 README 第 0 节符号表一致）：
  P：参数量；N：参与训练的 GPU 卡数；
  B_p：每个参数的参数字节（fp16/bf16 = 2，fp32 = 4）；
  B_opt：每个参数的 optimizer 状态字节（Adam fp32 m+v = 8）；
  GB：十进制 1e9（与 r2-Day14/15/16 口径一致）。

口径声明（显式假设，README 例 B 引用）：
  - model-state 台账 = params + grads + opt states，不含 activations
    （activations 是 r2-Day16 checkpoint 的活，两课正交）；
  - DDP：每 rank 存全量 model states，通信 = grad 的 ring all-reduce；
  - FSDP / DeepSpeed ZeRO-3（full shard）：model states 按 N 切分，
    通信 ≈ 1.5 × DDP（forward all-gather + backward all-gather +
    reduce-scatter，r2-Day14 已算）；
  - Megatron TP=t：params 按 t 切（常驻分片），通信 = 每层固定
    4 次 all-reduce（forward 2 + backward 2，r2-Day15 已算），
    payload ∝ b·s·h，只认机内带宽。
"""
from __future__ import annotations

GB = 1e9


# ------------------------------------------------------------ 台账：显存
def bytes_per_param(param_bytes: int = 2, opt_bytes: int = 8) -> int:
    """每参数 model-state 字节 = params + grads + opt states。

    默认 fp16/bf16 参数+梯度（2+2）+ Adam fp32 m/v（8）= 12 B/param。
    """
    return param_bytes + param_bytes + opt_bytes


def ddp_state_per_rank_gb(p: float, param_bytes: int = 2, opt_bytes: int = 8) -> float:
    """DDP：每 rank 存全量 model states（不切分）。"""
    return p * bytes_per_param(param_bytes, opt_bytes) / GB


def fsdp_sharded_state_per_rank_gb(
    p: float, n: int, param_bytes: int = 2, opt_bytes: int = 8
) -> float:
    """FSDP / ZeRO-3 full shard：model states 按 N 卡切分，每 rank 1/N。"""
    return ddp_state_per_rank_gb(p, param_bytes, opt_bytes) / n


def megatron_tp_params_per_rank_gb(p: float, t: int, param_bytes: int = 2) -> float:
    """Megatron TP=t：params 按 t 切分常驻每 rank（只算 params 一项，便于对比）。"""
    return p * param_bytes / t / GB


# ------------------------------------------------------------ 台账：通信
def ring_allreduce_bytes(payload_bytes: float, n: int) -> float:
    """ring all-reduce 通信量 = 2(N-1)/N × payload（r2-Day03 公式）。"""
    return 2.0 * (n - 1) / n * payload_bytes


def ddp_grad_comm_gb(p: float, n: int, param_bytes: int = 2) -> float:
    """DDP 每 step 通信 = grad（P×B_p 字节）的 ring all-reduce。"""
    return ring_allreduce_bytes(p * param_bytes, n) / GB


def fsdp_comm_gb(p: float, n: int, param_bytes: int = 2) -> float:
    """FSDP / ZeRO-3 每 step 通信 ≈ 1.5 × DDP。

    构成：forward 每层 all-gather 参数 + backward 每层 all-gather 参数 +
    reduce-scatter 梯度。r2-Day14 的 toy 已验证 1.5× 系数。
    """
    return 1.5 * ddp_grad_comm_gb(p, n, param_bytes)


def tp_allreduces_per_layer() -> int:
    """Megatron TP：每层固定 4 次 all-reduce（forward 2 + backward 2）。

    attention out-proj（row-parallel）1 次 + MLP down-proj（row-parallel）1 次，
    forward 共 2 次；backward 对称再 2 次。r2-Day15 已用此口径。
    """
    return 4


# ------------------------------------------------------------ 选型决策表
# 返回 (框架名, 一句话理由)。toy 决策辅助，不是生产选型建议；
# 每个分支都被测试覆盖。
def select_framework(
    num_gpus: int,
    params_b: float,
    cross_node: bool,
    hf_trainer: bool,
    low_code_intrusion: bool,
) -> tuple:
    """选型三问的可执行版本。

    参数：
      num_gpus：卡数；params_b：参数量（单位：十亿）；
      cross_node：是否跨机；hf_trainer：是否已在用 HF Trainer；
      low_code_intrusion：是否要求最小代码侵入。
    """
    if num_gpus <= 1 or params_b < 1.0:
        return (
            "none",
            "单卡或 <1B 小模型：Megatron/DeepSpeed/FSDP 三者都 overkill，"
            "单卡 + AMP / DDP 即够，框架迁移成本 > 收益（何时不赚）。",
        )
    if cross_node and params_b >= 20.0:
        return (
            "megatron",
            "跨机 20B+：Megatron-LM 的 TP(机内)+PP(跨机)+DP 3D 样板最成熟，"
            "通信模式固定、可预期；FSDP 跨机也 work，但大模型 3D 调优"
            "经验沉淀在 Megatron 一侧。",
        )
    if hf_trainer:
        return (
            "deepspeed",
            "已在 HF Trainer 生态：DeepSpeed 只需一行 config 开 ZeRO-3，"
            "代码侵入最小，还能顺手开 ZeRO-Offload / 1-bit Adam。",
        )
    if low_code_intrusion:
        return (
            "fsdp",
            "要求最小代码侵入且 PyTorch 原生：FSDP2 的 fully_shard 包裹模型"
            "即得 ZeRO-3 语义（用完即 reshard），与 torch.compile / "
            "DTensor 生态最顺。",
        )
    return (
        "fsdp",
        "默认兜底：无特殊约束时选 PyTorch 原生 FSDP，"
        "10.5GB/rank（7B/N=8 口径）+ 1.5×DDP 通信，跨机单机都 work。",
    )


def comparison_table_7b_n8() -> list:
    """7B、N=8 口径下三框架对照表（数字全部来自本模块函数，可复算）。"""
    p, n = 7e9, 8
    return [
        {
            "framework": "DDP（基线）",
            "state_gb_per_rank": round(ddp_state_per_rank_gb(p), 2),
            "comm_gb_per_step": round(ddp_grad_comm_gb(p, n), 2),
            "code_intrusion": "低（30min 改造）",
            "cross_node_ok": True,
        },
        {
            "framework": "FSDP / DeepSpeed ZeRO-3",
            "state_gb_per_rank": round(fsdp_sharded_state_per_rank_gb(p, n), 2),
            "comm_gb_per_step": round(fsdp_comm_gb(p, n), 2),
            "code_intrusion": "低（fully_shard / 一行 config）",
            "cross_node_ok": True,
        },
        {
            "framework": "Megatron TP=8（params 项）",
            "state_gb_per_rank": round(megatron_tp_params_per_rank_gb(p, 8), 2),
            "comm_gb_per_step": "每层 4 次 all-reduce（见 r2-Day15）",
            "code_intrusion": "高（按 3D 切分组织模型/脚本）",
            "cross_node_ok": "TP 限机内，PP 跨机",
        },
    ]
