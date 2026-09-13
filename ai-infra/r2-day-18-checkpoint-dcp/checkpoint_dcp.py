#!/usr/bin/env python3
"""r2-Day18：Checkpoint & Recovery（DCP 语义）—— 台账、Young/Daly 间隔、
一致快照、两槽轮换、异步写、换 world size 重分片、crash 恢复模拟。

纯 Python（仅标准库），不依赖 torch / CUDA / NCCL。
README 里的全部手算数字都在这里建模，断言见 test_checkpoint_dcp.py。

符号含义（与 README 第 0 节符号表一致）：
  P：参数量；N：参与训练的 GPU 卡数；
  B_p：每个参数的参数字节（fp16/bf16 = 2，fp32 = 4）；
  B_opt：每个参数的 optimizer 状态字节（Adam fp32 m+v = 8）；
  GB：十进制 1e9（与 r2-Day14/15/16/17 口径一致）；
  delta：一次 checkpoint 写盘耗时（s）；M：系统 MTBF（s）；
  tau：checkpoint 间隔（s）。

口径声明（显式假设，README 引用）：
  - checkpoint 存 params + optimizer states + step/epoch + rng，
    不含 grads（grads 是 step 内瞬态，重算即可，存了也白存）；
  - 默认 fp16/bf16 参数（2B）+ Adam fp32 m/v（8B）= 10 B/param；
  - full rank0 gather：除 rank0 外所有 shard 都要搬运到 rank0，
    额外流量 = (N-1)/N × 总量，rank0 需 total 大小的 staging；
  - sharded DCP：每 rank 只写自己手里的 shard，零额外通信，
    代价是 N 个文件 + 换 world size 时重分片。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor

GB = 1e9


# ============================================================ 第 1 节：台账
def ckpt_bytes_per_param(param_bytes: int = 2, opt_bytes: int = 8) -> int:
    """每参数 checkpoint 字节 = params + optimizer states。

    grads 不存：它是 step 内的瞬态量，恢复后第一个 step 就会重算。
    默认 fp16 参数（2B）+ Adam fp32 m/v（8B）= 10 B/param。
    """
    return param_bytes + opt_bytes


def full_ckpt_gb(p: float, param_bytes: int = 2, opt_bytes: int = 8) -> float:
    """完整 checkpoint 大小（GB）：params + opt states。"""
    return p * ckpt_bytes_per_param(param_bytes, opt_bytes) / GB


def sharded_per_rank_gb(
    p: float, n: int, param_bytes: int = 2, opt_bytes: int = 8
) -> float:
    """DCP sharded：每 rank 只写自己手里的 shard，零额外通信。"""
    return full_ckpt_gb(p, param_bytes, opt_bytes) / n


def rank0_gather_extra_gb(total_gb: float, n: int) -> float:
    """full rank0 gather 的额外搬运流量（GB）= (N-1)/N × 总量。

    训练本身不需要这次搬运，它是"为了写一个文件"纯付出的。
    """
    return (n - 1) / n * total_gb


# ================================================== 第 2 节：Young / Daly 间隔
def optimal_interval_s(delta_s: float, mtbf_s: float) -> float:
    """最优 checkpoint 间隔 tau* = sqrt(2 * delta * M)（Young 1974 / Daly 2006）。

    推导：浪费占比 W(tau) = delta/tau（写盘开销）+ tau/(2M)（故障期望返工，
    故障均匀落在间隔内，平均丢 tau/2 的进度，单位时间故障率 1/M）。
    dW/dtau = -delta/tau^2 + 1/(2M) = 0 即得。
    """
    if delta_s <= 0 or mtbf_s <= 0:
        raise ValueError("delta_s and mtbf_s must be positive")
    return math.sqrt(2.0 * delta_s * mtbf_s)


def waste_fraction(delta_s: float, mtbf_s: float, tau_s: float) -> float:
    """给定间隔 tau 下的浪费占比 W(tau) = delta/tau + tau/(2M)。"""
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")
    return delta_s / tau_s + tau_s / (2.0 * mtbf_s)


# ================================================== 第 3 节：一致快照 / 两槽 / 异步 / 重分片
class TinyRng:
    """可 JSON 序列化的确定性 RNG（LCG），代表 checkpoint 里必须存的 rng state。

    真实训练要存 python/numpy/torch-cpu/每 rank torch-cuda 四套；
    这里用一整数状态演示"rng 必须进 checkpoint，否则恢复后轨迹分叉"。
    """

    _A = 1664525
    _C = 1013904223
    _M = 2**32

    def __init__(self, seed: int):
        self.state = seed % self._M

    def next(self) -> float:
        self.state = (self._A * self.state + self._C) % self._M
        return self.state / self._M

    def getstate(self) -> int:
        return self.state

    def setstate(self, s: int) -> None:
        self.state = s % self._M


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_flat(flat: list, world_size: int) -> list:
    """按 rank 做连续切分（DCP planner 的最简语义），允许空 shard。"""
    n = len(flat)
    base, rem = divmod(n, world_size)
    out, start = [], 0
    for r in range(world_size):
        size = base + (1 if r < rem else 0)
        out.append(flat[start : start + size])
        start += size
    return out


class ShardedCheckpointer:
    """模拟 torch.distributed.checkpoint 的核心语义（单机多 shard 版）。

    planner：flat params 按 rank 连续切分；
    writer：每 shard 一个 JSON 文件 + MANIFEST.json（原子 rename 发布）；
    manifest：记录全局形状、world_size、step、每 shard sha256；
    两槽轮换：active 槽写坏（torn write）时回退到另一槽；
    async_save：先 deepcopy 做一致性快照，再后台线程写盘——快照之后
      训练继续改参数，写盘看到的仍是快照时刻的值。
    """

    def __init__(self, root: str, world_size: int):
        self.root = root
        self.world_size = world_size
        os.makedirs(root, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1)

    # -- 路径 --
    def _slot_dir(self, slot: str) -> str:
        return os.path.join(self.root, f"slot_{slot}")

    def _manifest_path(self) -> str:
        return os.path.join(self.root, "MANIFEST.json")

    # -- 写 --
    def save(self, state: dict, slot: str) -> None:
        """同步写：shard 文件 + 原子发布的 manifest。"""
        slot_dir = self._slot_dir(slot)
        os.makedirs(slot_dir, exist_ok=True)
        shards = _split_flat(state["flat_params"], self.world_size)
        opt_shards = _split_flat(state["opt_state"], self.world_size)
        entries = []
        for r, (ps, qs) in enumerate(zip(shards, opt_shards)):
            path = os.path.join(slot_dir, f"shard_{r:04d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"params": ps, "opt": qs}, f)
            entries.append({"file": os.path.basename(path),
                            "sha256": _sha256_of_file(path)})
        manifest = self._read_manifest()
        manifest["slots"][slot] = {
            "world_size": self.world_size,
            "step": state["step"],
            "rng": state["rng"],
            "sched": state.get("sched", {}),
            "num_params": len(state["flat_params"]),
            "shards": entries,
        }
        manifest["active"] = slot
        tmp = self._manifest_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        os.replace(tmp, self._manifest_path())  # 原子发布：torn manifest 不可见

    def async_save(self, state: dict, slot: str) -> Future:
        """异步写：先 deepcopy 快照（一致性点），再后台线程写盘。"""
        snapshot = copy.deepcopy(state)
        return self._pool.submit(self.save, snapshot, slot)

    # -- 读 --
    def _read_manifest(self) -> dict:
        if not os.path.exists(self._manifest_path()):
            return {"slots": {}, "active": None}
        with open(self._manifest_path(), encoding="utf-8") as f:
            return json.load(f)

    def _verify_slot(self, manifest: dict, slot: str) -> bool:
        info = manifest["slots"].get(slot)
        if not info:
            return False
        slot_dir = self._slot_dir(slot)
        for e in info["shards"]:
            path = os.path.join(slot_dir, e["file"])
            if not os.path.exists(path):
                return False
            if _sha256_of_file(path) != e["sha256"]:
                return False  # torn write：内容与发布时不一致
        return True

    def load(self, world_size: int | None = None) -> dict:
        """读 active 槽；校验失败则回退到另一槽；支持换 world size 重分片。"""
        manifest = self._read_manifest()
        order = [manifest.get("active")]
        order += [s for s in manifest["slots"] if s not in order]
        for slot in order:
            if slot and self._verify_slot(manifest, slot):
                return self._materialize(manifest["slots"][slot], slot,
                                         world_size or self.world_size)
        raise RuntimeError("no valid checkpoint slot (all slots torn or missing)")

    def _materialize(self, info: dict, slot: str, world_size: int) -> dict:
        slot_dir = self._slot_dir(slot)
        flat, opt = [], []
        for e in info["shards"]:  # 按 rank 顺序拼回全局
            with open(os.path.join(slot_dir, e["file"]), encoding="utf-8") as f:
                d = json.load(f)
            flat.extend(d["params"])
            opt.extend(d["opt"])
        assert len(flat) == info["num_params"], "shard sizes do not add up"
        # 换 world size：重分片（DCP reshard 语义的最简版）——
        # 先拼回全局，再按新的 world size 切，每 rank 拿到自己那份。
        param_shards = _split_flat(flat, world_size)
        opt_shards = _split_flat(opt, world_size)
        return {"flat_params": flat, "opt_state": opt, "step": info["step"],
                "rng": info["rng"], "sched": info["sched"],
                "saved_world_size": info["world_size"],
                "loaded_world_size": world_size,
                "rank_param_shards": param_shards,
                "rank_opt_shards": opt_shards}


# ================================================== 第 4 节：crash 恢复模拟
class ToyTrainer:
    """确定性 toy 训练器：R^4 上 momentum-SGD 拟合 y = c·x。

    全部可变状态 = flat_params(4) + opt_state momentum(4) + step + rng，
    恰好是 checkpoint 的四件套。数据固定，随机性只来自 rng 决定的采样顺序。
    """

    def __init__(self, seed: int = 7, lr: float = 0.05, mu: float = 0.9):
        self.rng = TinyRng(seed)
        self.flat_params = [0.1 * (i + 1) for i in range(4)]
        self.opt_state = [0.0] * 4
        self.step = 0
        self.lr = lr
        self.mu = mu
        # 固定数据：y = c·x，c = (3, -2, 1, 0.5)，8 个样本
        c = (3.0, -2.0, 1.0, 0.5)
        self.data = [
            ([float((i >> j) & 1) for j in range(4)],
             sum(c[j] * float((i >> j) & 1) for j in range(4)))
            for i in range(1, 9)
        ]

    def train_step(self) -> float:
        x, y = self.data[int(self.rng.next() * len(self.data))]
        pred = sum(w * xi for w, xi in zip(self.flat_params, x))
        err = pred - y
        for j in range(4):
            g = 2.0 * err * x[j]
            self.opt_state[j] = self.mu * self.opt_state[j] + g
            self.flat_params[j] -= self.lr * self.opt_state[j]
        self.step += 1
        return err * err

    def state_dict(self) -> dict:
        return {"flat_params": list(self.flat_params),
                "opt_state": list(self.opt_state),
                "step": self.step, "rng": self.rng.getstate(),
                "sched": {"lr": self.lr, "mu": self.mu}}

    def load_state_dict(self, sd: dict) -> None:
        self.flat_params = list(sd["flat_params"])
        self.opt_state = list(sd["opt_state"])
        self.step = sd["step"]
        self.rng.setstate(sd["rng"])
        self.lr = sd["sched"]["lr"]
        self.mu = sd["sched"]["mu"]
