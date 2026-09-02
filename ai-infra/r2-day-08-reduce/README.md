# r2-Day08 — Parallel Reduce：原子加 → Shared Tree → Warp Shuffle

## Connection to Prev

r2-Day07 学会把 **warp 的线程编号映射到地址、transaction 和 shared-memory bank**；r2-Day08 把这套映射真正组织成 reduction：把 $N$ 个数压成一个数时，瓶颈不是“会不会相加”，而是**在哪里汇合、同步多少次、让多少线程争用同一地址**。

- **牺牲什么**：shared tree 增加 shared memory 与 block barriers；warp shuffle 增加 warp-level 假设、mask/partial-warp 正确性负担和更复杂的两级归约。
- **换取什么**：从每元素一次 global atomic，降到每 block 一次 global atomic；再把大部分 shared-memory round trips 和 block-wide barriers 换成 register-to-register lane exchange。
- **何时不赚**：输入很小、global atomic contention 已低、算子融合能直接消费局部结果、数据类型/归约操作不支持高效 shuffle，或更复杂实现降低 occupancy。浮点加法不满足结合律，三种树形顺序也可能产生不同的末位结果。

## 1. Reduction 的工作与理论下界

求和：

$$y=\sum_{i=0}^{N-1}x_i$$

无论并行结构怎样，数学上都需要 $N-1$ 次有效加法，并至少读取 $N$ 个输入。优化的核心不是消灭加法，而是改变汇合位置：

1. **global atomic per element**：每个线程直接 `atomicAdd(output, x[i])`；代码最简单，但对同一个地址有 $N$ 次原子更新。
2. **shared-memory tree**：每个 block 先在 shared memory 做树形归约，只由 block leader 做一次 global atomic；原子数约为 $\lceil N/T\rceil$。
3. **warp shuffle**：先在各 warp 内通过 `__shfl_down_sync` 在 registers 间交换，再把每个 warp 的一个 partial 写入 shared memory，最后由第一个 warp 收尾；仍是一 block 一次 atomic，但只需一次 block barrier。

这里 $T$ 是 threads per block。原子数与 barrier 数是结构计数，不是性能测量。

## 2. 一个可手算的小例子：8 lanes 求和 1…8

用 `width=8` 的缩小版 shuffle tree（CUDA `width` 允许 2 的幂且不超过 32）：

```text
初始:      [1, 2, 3, 4, 5, 6, 7, 8]
offset=4: lane0 = 1 + 5 = 6
offset=2: lane0 = 6 + (3 + 7) = 16
offset=1: lane0 = 16 + (2 + 4 + 6 + 8) = 36
```

lane 0 最终得到 $36$。对真实 32-lane warp，offset 依次为 $16,8,4,2,1$，关键路径是 5 轮 shuffle-add。

再看 $N=64,T=64$ 的结构账：

| 版本 | global atomic updates | block barriers | shared writes |
|---|---:|---:|---:|
| 每元素 atomic | 64 | 0 | 0 |
| shared tree | 1 | $1+\log_2 64=7$ | 64 次初始写入 |
| warp shuffle | 1 | 1 | 2 个 warp partials |

这张表只描述本课代码的结构；不能据此编造 latency、带宽或吞吐排序。

## 3. 从 contention 到层级归约

### 3.1 每元素 global atomic

优点是短、通用、无需 block 协作；代价是所有线程都更新同一 global address。atomic 保证更新不可分割，不保证高并发下没有序列化/争用成本。

### 3.2 Shared-memory binary tree

一个 block 的 $T$ 个值写进 shared memory，然后 stride 取 $T/2,T/4,\ldots,1$：

$$s_t \leftarrow s_t+s_{t+\text{stride}},\qquad t<\text{stride}$$

每轮之后必须 block-wide 同步，防止下一轮读取尚未完成的 partial。最后只有 `threadIdx.x==0` 更新 global output。

### 3.3 Warp shuffle + 一次 block barrier

warp 内 lanes 可用 `__shfl_down_sync(mask, value, offset)` 直接读取另一 lane 的 register 值，无需把每一层 partial 写回 shared memory。每个 warp 的 lane 0 写一个 partial；block 同步一次后，第一个 warp 再归约这些 partials。这里仍保留每 block 一次 atomic，便于任意 grid size；进一步的多 kernel 两阶段归约可完全去掉最终热点，但多一次 launch 与中间 buffer。

## 4. 可执行代码

### CPU：语义与结构计数

`reduce_models.py` 真正执行三种数据流，并记录 atomic、barrier、shared write 与 warp-shuffle 指令的结构计数；不是只打印步骤。

```bash
python3 ai-infra/r2-day-08-reduce/reduce_models.py --elements 64 --block-size 64
python3 -m unittest discover -s ai-infra/r2-day-08-reduce -p 'test_*.py' -v
```

### CUDA：三种 kernel 与统一校验 harness

`reduce_three_ways.cu` 中 block size 真正参与 launch 和 shared-memory allocation；三种 kernel 都会 warm up、用 CUDA events 记录 21 次 kernel-only 时间的中位数、拷回结果并检查正确性。

```bash
nvcc -O3 -std=c++17 ai-infra/r2-day-08-reduce/reduce_three_ways.cu -o /tmp/reduce_three_ways
/tmp/reduce_three_ways 4194304 256
```

真机比较时还应固定 GPU、clock/power policy、CUDA/driver、输入规模、block size、warmup/repeat，并保存原始输出；用 Nsight Compute 检查 atomic throughput、DRAM bytes、shared transactions、barriers/stalls。CUDA event 数字只覆盖 kernel，不含 H2D/D2H。

## 5. 与 Attention 的连接

softmax 需要 row-wise max 与 sum reduction；LayerNorm/RMSNorm 也需要统计量归约。Day08 的层级结构因此是 Day10 FlashAttention 中 online softmax、以及后续 fused normalization kernels 的原型：先把局部数据留在更近的层级归约，再减少昂贵的全局汇合。

## 状态

- 已验证：Python 语法、7 个 CPU 单元测试、三种归约语义、partial block zero-padding 和结构计数。
- **execution not validated on CUDA/H100 / 待H100验证**：环境无 `nvcc`/CUDA GPU；`reduce_three_ways.cu` 未编译或运行，Nsight 未执行。
- 本课保持 `blocked`；没有声称 latency、bandwidth、atomic throughput、occupancy、comm%、MFU 或设备拓扑测量。所有计数均为源码结构或 theoretical estimate，不是 benchmark。

## 原始 / 官方来源

- NVIDIA CUDA C++ Programming Guide（SIMT、同步、atomic、shuffle intrinsics）：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA CUDA C++ Best Practices Guide（coalescing、shared memory、profiling）：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA CUB `BlockReduce` API（生产实现与算法选项）：https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html
