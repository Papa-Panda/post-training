# r2-Day08 — Parallel Reduce：问答归档

用户在当天 ai 课程聊天中提出的问题与总结。问题原文保留，答案按聊天记录整理；
英文术语（atomic、shared memory、warp shuffle、block、barrier）保留。

## 2026-09-03（周四）

### Q1：三种 reduce 方式的区别是什么

**A：** 核心不是减少加法次数，而是**让汇合发生在更快、争用更小的层级**。

```text
output = x[0] + x[1] + ... + x[N-1]
```

1. **每元素 global atomic**：每个 thread 直接 `atomicAdd(output, x[i])`；
   `N` 个 threads 争抢同一个 global 地址，共 **N 次 global atomic**，
   代码最简单，contention 最大。
2. **Shared-memory tree**：每个 block 在 shared memory 里做树形归约，
   只由 block leader 做一次 global atomic；原子数约为 `ceil(N/T)`。
3. **Warp shuffle**：先在各 warp 内用 `__shfl_down_sync` 在 registers 间交换，
   再把每个 warp 的一个 partial 写入 shared memory，
   最后由第一个 warp 收尾；仍是一 block 一次 atomic，
   但只需一次 block barrier。

### Q2：如果 N=1024, T=256，共有 4 个 blocks，因此从 1024 次 global atomic 降为 4 次：没看懂

**A：** 这里假设**每个 thread 负责一个元素**，每个 block 有 `T=256` 个 threads。
`N=1024` 个元素分给 `1024/256 = 4` 个 blocks：

```text
Block 0：256 threads，处理 x[0]   到 x[255]
Block 1：256 threads，处理 x[256] 到 x[511]
Block 2：256 threads，处理 x[512] 到 x[767]
Block 3：256 threads，处理 x[768] 到 x[1023]
```

**朴素 atomic 版**：每个 thread 都执行一次 `atomicAdd(output, x[i])`，
共 1024 个 threads，所以是 **1024 次 global atomic**。

**Shared-tree 版**：每个 block 先在 shared memory 中把 256 个元素
加成一个 `block_sum`：

```text
Block 0 → sum(x[0:256])    → atomicAdd 一次
Block 1 → sum(x[256:512])  → atomicAdd 一次
Block 2 → sum(x[512:768])  → atomicAdd 一次
Block 3 → sum(x[768:1024]) → atomicAdd 一次
```

最后只有 4 个 block leaders 更新 global output，
所以从 **每个元素一次 atomic** 变成 **每个 block 一次 atomic**：1024 → 4。

一般公式（"一线程处理一个元素"简化模型）：
`global atomic 数量 = ceil(N/T) = blocks 数量`。
求和总量没变，只是绝大部分加法搬到了更快、没有跨 block 争抢的 shared memory。

### Q3：Shared tree 的问题是：每轮都要读写 shared memory，并让整个 block 同步

**A：** 对。一个 block 归约 8 个数，`shared = [1, 2, 3, 4, 5, 6, 7, 8]`：

```cuda
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
        s[threadIdx.x] += s[threadIdx.x + stride];
    }
    __syncthreads();
}
```

- stride=4：`s[0..3]` 分别加上 `s[4..7]` → `[6, 8, 10, 12, ...]`；
- 必须 `__syncthreads()`，因为下一轮要读取本轮刚写出的结果；
  否则 thread 0 可能已经开始第二轮，而 thread 2 还没写完 `10`，
  读到旧值 `3` 就会算错；
- stride=2：`6+10=16, 8+12=20`；再同步；最后 `16+20=36`。

代价不是加法多，而是**每一层都要 shared-memory round trip，
加上 block-wide barrier**。256 threads 约需 `log₂256 = 8` 轮；
warp shuffle 就是为了把 warp 内这些读写和 block-wide barriers
换成 register 之间的直接交换。

### Q4：Warp shuffle 那个没看懂

**A：** 关键是：**同一 warp 内，thread 可以直接读取另一个 thread 的
register 值，不经过 shared memory。**

缩小版 8-thread warp，每个 thread 的 register `v` 保存一个数：

```text
lane:  0  1  2  3  4  5  6  7
v:     1  2  3  4  5  6  7  8
```

执行 `v += __shfl_down_sync(mask, v, 4)`：每个 lane 读取右边相距 4 个
lanes 的 `v`：

```text
lane 0 读取 lane 4：1 + 5 = 6
lane 1 读取 lane 5：2 + 6 = 8
lane 2 读取 lane 6：3 + 7 = 10
lane 3 读取 lane 7：4 + 8 = 12
```

offsets 依次为 4、2、1：`[6, 8, 10, 12] → [16, 20] → [36]`，
lane 0 得到整个 warp 的和。真实 32-lane warp 的 offsets
依次为 `16, 8, 4, 2, 1`。

与 shared tree 的区别：

```text
Shared tree:  register → shared memory → 同步 → shared memory → register
Warp shuffle: 一个 lane 的 register → 另一个 lane 的 register
```

但 shuffle **不能跨 warp**。256-thread block（8 个 warps）的完整流程：

```text
8 个 warps 各自在 registers 内归约出 1 个 partial
→ 8 个 warp leaders 把 8 个 partials 写入 shared memory
→ __syncthreads() 一次
→ 第一个 warp 把这 8 个 partials 继续归约得到 block_sum
→ global atomic 一次
```

那一次 barrier 是因为必须等 8 个 warp 都写完 partial sum，
第一个 warp 才能读取。
