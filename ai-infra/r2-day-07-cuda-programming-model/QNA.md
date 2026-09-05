# r2-Day07 — CUDA 编程模型：问答归档

用户在当天 ai 课程聊天中提出的问题与总结。问题原文保留，答案按聊天记录整理；
英文术语（SM、block、warp、thread、register、shared memory）保留。

## 2026-09-01（周二）

### Q1：一个sm对应一个或多个block？

**A：** 一个 SM 通常可以**同时驻留多个 block**，但一个 block 只能放在一个 SM 上，不能跨 SM 拆开。

关键规则：

- 一个 block 只能驻留一个 SM，不能跨 SM 拆分；
- 一个 SM 可以同时驻留多个 blocks，并交错执行它们的 warps；
- SM 放不下的新 block 会等待，等已有 block 完成并释放资源后再进入；
- 一个 grid 的 blocks 会动态分配给所有 SM，所以不要假设某个 block
  固定对应某个 SM。

以 H100 每个 SM 最多容纳 **64 个 resident warps** 为例（只看 warp 上限）：

- `blockDim = 256`：每个 block 有 `256/32 = 8` 个 warps，
  单看 warp 上限最多可驻留 `64/8 = 8` 个 blocks；
- `blockDim = 1024`：每个 block 有 32 个 warps，
  单看 warp 上限最多只能驻留 2 个 blocks。

实际数量还受 threads、registers、shared memory、
block 数量硬件上限等约束，取各约束的最小值。

### Q2：一个 warp 是32个threads 对应的呢

**A：** 对，NVIDIA GPU 上一个 warp **固定由 32 个 threads 组成**。
层次关系是：

```text
Grid
  └── Blocks
        └── Warps
              └── 32 Threads
```

例如 `blockDim.x = 256` 的 block 会被分成：

```text
Warp 0: thread 0–31
Warp 1: thread 32–63
...
Warp 7: thread 224–255
```

共 **8 个 warps**。

GPU 实际调度的基本单位是 **warp**，不是单个 thread。
一个 warp 中的 32 个 threads 通常执行同一条指令，
但每个 thread 有自己的 `threadIdx`、自己的 registers、处理不同的数据地址。

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
```

同一 warp 内每个 thread 的 `threadIdx.x` 不同，
所以读写的 `A[i]`、`C[i]` 地址也不同——这就是 Day07 的地址映射。

延伸两个术语：

- **partial warp**：100-thread 的 block 会占用 4 个 warps，
  最后一个 warp 只有 4 个有效 lanes；
- **warp divergence**：同一 warp 内 threads 走不同分支路径，
  不同路径串行执行。

### Q3：上面两个讲的很好 加入repo啊

**处理结果：** 已把两个解释整理进 Day07 的 README/NOTES：
SM/block/warp 对应关系、resident-block 的资源约束公式、
256 vs 1024 threads/block 的对比、warp=32、partial warp 与 warp divergence 例子；
commit `16939f8`，测试 8/8 通过，已推 GitHub。
