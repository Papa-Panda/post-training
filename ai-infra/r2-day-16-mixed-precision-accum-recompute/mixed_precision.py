#!/usr/bin/env python3
"""r2-Day16：BF16 混合精度 / 梯度累积 / Activation Checkpoint 的手算模型。

纯 Python（仅标准库），不依赖 torch / numpy / CUDA。
README / NOTES 里的全部手算数字都在这里建模，
断言见 test_mixed_precision.py。

符号含义（与 README 第 0 节符号表一致）：
  L：层数；A：每层 activation 字节数；
  b：microbatch 大小；m：梯度累积步数；B_eff = b*m：有效 batch；
  s：loss scale 因子；eps：machine epsilon = 2**(-显式尾数位)；
  u：unit roundoff = eps/2；k：checkpoint 间隔（每 k 层存一个）。
"""
from __future__ import annotations

import math
import struct


# ------------------------------------------------------------ DTYPE 位布局
# mantissa_bits 为显式尾数位（不含隐藏的 leading 1）。
DTYPES = {
    "fp32": {"sign_bits": 1, "exp_bits": 8, "mantissa_bits": 23},
    "bf16": {"sign_bits": 1, "exp_bits": 8, "mantissa_bits": 7},
    "fp16": {"sign_bits": 1, "exp_bits": 5, "mantissa_bits": 10},
}

# 十进制 GB（与 r2-Day15 口径一致）。
GB = 1e9


def dtype_bits(name):
    """返回 (符号位, 指数位, 显式尾数位)。"""
    d = DTYPES[name]
    return d["sign_bits"], d["exp_bits"], d["mantissa_bits"]


def machine_epsilon(name):
    """eps = 2**(-显式尾数位)：1.0 处相邻可表示数的间距。"""
    return 2.0 ** (-DTYPES[name]["mantissa_bits"])


def unit_roundoff(name):
    """u = eps/2：舍入到最近（round-to-nearest）时的最大相对误差界。"""
    return machine_epsilon(name) / 2.0


def min_normal(name):
    """最小正规格数 = 2**(2 - 2**(e-1))，IEEE 风格偏置 bias = 2**(e-1) - 1。"""
    e = DTYPES[name]["exp_bits"]
    return 2.0 ** (2 - 2 ** (e - 1))


def dtype_max(name):
    """最大值 = (2 - eps) * 2**(2**(e-1) - 1)。"""
    e = DTYPES[name]["exp_bits"]
    eps = machine_epsilon(name)
    return (2.0 - eps) * 2.0 ** (2 ** (e - 1) - 1)


def in_fp16_range(x):
    """x 是否落在 FP16 可表示的正数范围内（忽略符号与次正规数）。"""
    return min_normal("fp16") <= abs(x) <= dtype_max("fp16")


# ------------------------------------------------------------ 例子 B：下溢与 loss scaling
def fp16_underflow_demo(grad=3.0e-5, scale=2 ** 15):
    """FP16 下溢演示：小梯度在 fp16 下冲到 0，放大 s 倍后落回可表示范围。"""
    scaled = grad * scale
    return {
        "grad": grad,
        "scale": scale,
        "scaled_grad": scaled,
        "orig_in_fp16_range": in_fp16_range(grad),
        "scaled_in_fp16_range": in_fp16_range(scaled),
    }


def dynamic_loss_scale_update(scale, grads, growth=2.0, backoff=0.5):
    """动态 loss scaling 一步：缩放后梯度仍在 fp16 范围内则放大 scale，
    否则收缩 scale 并跳过本步更新（返回新 scale 与是否跳过）。"""
    ok = all(in_fp16_range(g * scale) for g in grads)
    if ok:
        return scale * growth, False
    return scale * backoff, True


# ------------------------------------------------------------ bf16 舍入模拟
def round_to_bf16(x):
    """把 Python float 舍入到 bf16 可表示值（round-half-to-even）。
    原理：bf16 就是 fp32 砍掉低 16 位尾数，对被砍掉的位做偶数舍入。"""
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    sign = bits >> 31
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    low = mant & 0xFFFF        # 被砍掉的低 16 位
    keep = mant >> 16          # 保留的 7 位尾数
    halfway = 0x8000
    if low > halfway or (low == halfway and (keep & 1)):
        keep += 1
        if keep == 0x80:       # 尾数进位，指数加一
            keep = 0
            exp += 1
    new_bits = (sign << 31) | (exp << 23) | (keep << 16)
    return struct.unpack(">f", struct.pack(">I", new_bits))[0]


# ------------------------------------------------------------ 例子 C：梯度累积
def grad_accum(per_sample_grads, m):
    """梯度累积：n 个样本分成 m 个 microbatch，每 micro 求和再累加。
    返回 microbatch 求和、总和，以及 mean-reduction 下累积均值。"""
    n = len(per_sample_grads)
    assert n % m == 0, "样本数必须能被累积步数整除"
    b = n // m
    micro_sums = [sum(per_sample_grads[i * b:(i + 1) * b]) for i in range(m)]
    full_sum = sum(per_sample_grads)
    micro_means = [s / b for s in micro_sums]
    # mean reduction 下：microbatch 内部先平均，累积后必须再除以 m
    # 才等于全 batch 的均值梯度。
    accum_mean = sum(micro_means) / m
    return {
        "b": b,
        "m": m,
        "B_eff": b * m,
        "micro_sums": micro_sums,
        "total": sum(micro_sums),
        "full_sum": full_sum,
        "accum_mean": accum_mean,
        "full_mean": full_sum / n,
    }


def ddp_comm_per_sample(grad_bytes_per_step, batch):
    """DDP 每 optimizer step 做一次 all-reduce：每样本分摊通信量。"""
    return grad_bytes_per_step / batch


# ------------------------------------------------------------ 例子 D：activation checkpoint
def naive_activation_memory_mb(L, A_mb):
    """朴素反向：存下全部 L 层的 activation。"""
    return L * A_mb


def checkpoint_activation_memory_mb(L, A_mb, k):
    """每 k 层存一个 checkpoint；反向时从最近 checkpoint 重算段内 k 层。
    内存 = (checkpoint 数 ceil(L/k) + 重算段长度 k) * A。"""
    n_ckpts = math.ceil(L / k)
    return {
        "checkpoints": n_ckpts,
        "recompute_segment": k,
        "memory_mb": (n_ckpts + k) * A_mb,
    }


def optimal_checkpoint_interval(L):
    """(L/k + k) 在 k = sqrt(L) 处取最小 -> 2*sqrt(L)*A。"""
    return round(math.sqrt(L))


def recompute_extra_compute_frac():
    """重算多做一次 forward；一次训练迭代 forward:backward = 1:2，
    额外计算占比 = 1/(1+2) = 1/3。"""
    return 1.0 / 3.0


# ------------------------------------------------------------ 例子 E：7B AMP 显存台账
def amp_memory_ledger_gb(n_params, low_bytes=2, high_bytes=4, adam_bytes_per_param=8):
    """AMP 台账：bf16 参数 + bf16 梯度 + fp32 master + Adam fp32 m/v。"""
    params = n_params * low_bytes / GB
    grads = n_params * low_bytes / GB
    master = n_params * high_bytes / GB
    adam = n_params * adam_bytes_per_param / GB
    return {
        "params_GB": params,
        "grads_GB": grads,
        "master_GB": master,
        "adam_GB": adam,
        "total_GB": params + grads + master + adam,
    }


def pure_fp32_memory_ledger_gb(n_params):
    """纯 fp32 台账：master 即参数本身，无额外 master copy。"""
    params = n_params * 4 / GB
    grads = n_params * 4 / GB
    adam = n_params * 8 / GB
    return {
        "params_GB": params,
        "grads_GB": grads,
        "adam_GB": adam,
        "total_GB": params + grads + adam,
    }
