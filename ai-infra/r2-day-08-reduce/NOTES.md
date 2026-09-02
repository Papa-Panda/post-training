# NOTES — r2-Day08 Parallel Reduce 三连

## 准确术语

- **Reduction / fold**：用 associative binary operator 把一列输入组合成一个 aggregate。整数加法满足结合律；IEEE-754 浮点加法只近似满足，因此并行树改变顺序时结果末位可变。
- **Global atomic update**：对 global-memory location 做不可分割 read-modify-write；正确性不等于无 contention。
- **Block-wide barrier**：`__syncthreads()` 要求 block 中所有未退出线程到达，保证 barrier 前的 shared/global memory accesses 对 block 内线程可见。
- **Warp shuffle**：`__shfl_down_sync` 让 participating lanes 直接读取另一 lane 的 register value；mask 描述参与线程，不能把 inactive source lane 当有效输入。
- **Partial reduction**：每个 block/warp 先产生局部 aggregate，再由更高层归约。
- **Theoretical structure count**：从源码可数出的 atomics/barriers/shared writes；不是硬件 counter，更不是 latency/throughput benchmark。

## Connection to Prev 的实质

Day07 的 coalescing 解决“32 个 lanes 如何读入”；Day08 的 reduction 解决“读入后在哪里汇合”。连续读取仍要保留，但新瓶颈可能移动到：

1. 同一个 global output 的 atomic contention；
2. shared-memory tree 的每轮 barrier；
3. 寄存器与 shared memory 用量导致的 occupancy 变化；
4. 最后一块与 partial warp 的 mask/zero-padding 正确性。

所以优化路线不是“shuffle 永远最快”，而是把同步范围逐层缩小：grid-wide hotspot → block-local tree → warp-local lane exchange。

## 三个版本真实执行什么

### A. `reduce_atomic`

每个有效 thread 执行：

```cuda
atomicAdd(output, input[index]);
```

- 输入读取：$N$ 个 FP32，即理论 payload $4N$ bytes；
- global atomics：$N$；
- shared memory / block barrier：0。

`4N bytes` 只是输入 payload；atomic 是 read-modify-write，其真实 memory traffic 不能只按额外 4 bytes 计算，需 profiler/硬件验证。

### B. `reduce_shared_tree`

每 block 的 $T$ 个 threads 各写一个 shared value（越界写 0），然后做：

$$T/2+T/4+\cdots+1=T-1$$

次有效树加法。源码有一次 load 后 barrier，加上每层一次 barrier，共：

$$1+\log_2 T$$

个 block barriers；最后每 block 一个 global atomic。若 $B=\lceil N/T\rceil$：

$$A_{global}=B$$

### C. `reduce_warp_shuffle`

1. 每 warp 用 offsets $16,8,4,2,1$ 归约；
2. 每 warp 仅 lane 0 写一个 shared partial；
3. 全 block `__syncthreads()` 一次；
4. 第一个 warp 对这些 partials（其余 lanes 补 0）再做一次 warp reduction；
5. lane 0 做每 block 一次 global atomic。

若 $T$ 是 32 的倍数，shared writes/block 为 $T/32$，global atomics 仍为 $B$，block barriers 为 $B$（每 block 一次）。

## 可手算例子 1：1…8 的 width=8 shuffle

CUDA shuffle 的 `width` 可取不超过 32 的 2 的幂。初始 lanes：

$$[1,2,3,4,5,6,7,8]$$

只跟踪最终有用的 lane 0 路径：

- offset 4：lane 0 得 $1+5=6$；lane 2 得 $3+7=10$；
- offset 2：lane 0 得 $6+10=16$；lane 1 的对应 subtree 得 $8+12=20$；
- offset 1：lane 0 得 $16+20=36$。

`reduce_models.py` 的 32-lane版本把缺失的 24 lanes 补 0，实际执行 offsets $16,8,4,2,1$，测试结果同样为 36。

## 可手算例子 2：N=64, T=64

### 每元素 atomic

$$A_{global}=64$$

### Shared tree

一 block；有效加法数：

$$32+16+8+4+2+1=63$$

结构计数：1 个 global atomic、64 个初始 shared writes、$1+6=7$ 个 block barriers。

### Warp shuffle

两个 warps 各自得到一个 partial；所以是 2 个 shared writes、1 个 block barrier、1 个 global atomic。两个 warp 的内部归约和第一个 warp 的 block 收尾分别执行 5 条 warp-level shuffle instructions，合计 15 条 warp-level instructions。这里“15”不是所有 lanes 的动态 scalar instruction 数。

## 为什么结果正确但最后几 bit 可能不同

对实数加法，$(a+b)+c=a+(b+c)$；但浮点 rounding 使它一般不严格成立。例如：

$$a=10^{20},\quad b=-10^{20},\quad c=3.14$$

不同归约顺序可先抵消大数，也可先把小数吸收掉。因此：

- benchmark harness 用全 1 输入，$N=2^{22}$，FP32 可精确表示最终整数；
- 真实训练张量需要按 dtype、规模和误差目标设置 tolerance；
- 需要 bitwise determinism 时，不能把 nondeterministic atomic order 当作可复现 reduction。

## 示例代码每一步的对应关系

### `reduce_models.py`

- `atomic_reduce()` 逐元素累加到同一个 output，结果与 atomic count 都被报告使用；
- `shared_tree_reduce()` 真正构造/补零每个 block tile，并在 `_shared_tree()` 中执行 stride-halving additions；
- `warp_shuffle_reduce()` 真正对每个 32-lane warp 执行 lane-exchange tree，把 warp partials 送入第一个 warp，再累加 block result；
- `build_report()` 调用并返回三种结果，测试断言三者都等于正确 sum。

### `reduce_three_ways.cu`

- `block_size` 同时进入 grid 计算、kernel launch 与 dynamic shared-memory bytes；
- 三种 kernel 的输出都被 copy 回 host 并与 expected 比较；
- 每个计时 case 先 `cudaMemset(output, 0)`，否则重复运行会累加旧结果；
- warmup 后用 CUDA events 测 21 次 kernel-only latency，输出中位数；这段流程只有在 CUDA 环境真正运行后才构成 benchmark。

## 讨论难点

**为什么 warp shuffle 结构更省同步，却不能在没跑 profiler 时断言一定比 shared tree 快？**

回答至少要覆盖：

1. 两者都可能已受输入读取带宽约束，省下 barrier 不一定成为端到端主导；
2. compiler 生成指令、register pressure、occupancy 与 block size 会改变结果；
3. 很小的输入下 launch/固定开销占主导；
4. 新架构的 atomic 实现和 contention pattern 不同；
5. partial tile、非 32 倍 block、非 sum operator 需要更谨慎的 mask/identity；
6. 生产中优先考虑 CUB 等经验证 collective，再用 workload-specific evidence 判断是否手写。

## 真机验证协议（尚未执行）

记录后再比较：

- GPU 型号、SM 数、clock/power policy、driver、CUDA toolkit；
- `N`、dtype、block size、grid size、warmups、repeats；
- 完整编译命令与原始 stdout；
- Nsight Compute 的 DRAM bytes/throughput、atomic 指标、barrier/stall、occupancy；
- 每个 variant 的数值校验与 tolerance。

不要把 theoretical payload $4N$ bytes 直接当 DRAM bytes，也不要把结构计数换算成虚构 speedup。

## 验证状态

- 已执行 Python `py_compile`、7 个 CPU 单元测试和 CPU report。
- 已核对 shape/index：输入是一维 `[N]`；每 thread 对应一个 index；最后 block 越界 lanes 用 additive identity 0。
- 已核对单位：数组元素是 FP32（4 bytes）；CUDA event 输出单位是 ms；atomic/barrier/shared-write 是无量纲次数。
- **execution not validated on CUDA/H100 / 待H100验证**：环境没有 `nvcc`/GPU，CUDA kernel、CUDA events 与 Nsight 均未运行；性能状态保持 `blocked`。

## 原始 / 官方来源

1. NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
2. NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
3. NVIDIA CUB `BlockReduce`: https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html
