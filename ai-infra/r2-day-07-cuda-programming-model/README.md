# r2-Day07 — CUDA 编程模型：Grid / Block / Warp 与访存映射

## Connection to Prev

r2-Day06 用 Roofline 判断 H100 上的 vector add 因 arithmetic intensity 只有 $1/12$ FLOP/byte 而受 HBM 带宽约束；r2-Day07 继续追问：**同样的 12 bytes/element，32 个线程如何发地址，才能少做内存事务？**

- **牺牲什么**：把数据布局和线程布局对齐、给 shared-memory tile padding、以及调 block size，会增加索引逻辑、额外 padding 和调参成本。
- **换取什么**：warp 的 global-memory 请求可合并成更少的 32-byte transactions；shared memory 避免 bank-conflict 序列化；足够 active warps 可隐藏延迟。
- **何时不赚**：算子本来 compute-bound；不规则访问无法重排；padding/转置/同步成本超过节省的流量；或为提高 occupancy 压低寄存器后发生 spill。**高 occupancy 也不等于高性能。**

## 1. 执行层次：thread → warp → block → grid

```text
grid
  ├─ block 0: threads 0 ... blockDim.x-1
  ├─ block 1: threads 0 ... blockDim.x-1
  └─ ...

1D global index = blockIdx.x * blockDim.x + threadIdx.x
```

- `thread` 写一个逻辑元素；
- `warp` 是当前 NVIDIA GPU 上 32 个线程的执行组；
- `block` 内线程可通过 shared memory 和 block barrier 协作；
- `grid` 是一次 kernel launch 的全部 blocks。

### 一个 SM 对应一个还是多个 blocks？

**一个 block 只能在一个 SM 上执行，不能跨 SM 拆分；一个 SM 则通常可以同时驻留多个 blocks。** block 被调度到某个 SM 后，会一直留在那里直到执行完成。grid 中暂时放不下的 blocks 排队等待，等某个 SM 释放资源后再进入，因此 blocks 是分批（waves）执行的。

```text
Grid
  ├─ Block 0 ─┐
  ├─ Block 1 ─┼─> SM 0
  ├─ Block 2 ─┘
  ├─ Block 3 ─┐
  └─ Block 4 ─┴─> SM 1
```

每个 SM 能同时驻留多少 blocks，不只取决于 block 数量上限，而是多种资源约束的最小值：

$$B_{resident}=\min\left(B_{hw},\left\lfloor\frac{T_{SM}}{T_{block}}\right\rfloor,\left\lfloor\frac{W_{SM}}{W_{block}}\right\rfloor,\left\lfloor\frac{R_{SM}}{R_{block}}\right\rfloor,\left\lfloor\frac{S_{SM}}{S_{block}}\right\rfloor\right)$$

其中 $T/W/R/S$ 分别代表 threads、warps、registers 和 shared memory。以 H100 的每 SM 最多 64 resident warps 为例：

- `blockDim=256`：每 block 有 8 warps，只看 warp 上限最多是 $64/8=8$ blocks；
- `blockDim=1024`：每 block 有 32 warps，只看 warp 上限最多是 $64/32=2$ blocks。

实际值还可能被 registers 或 shared memory 进一步压低，所以 block 更大不保证更快。

### 一个 warp 与 32 个 threads

在当前 NVIDIA GPU 上，一个 warp 固定包含 32 threads。GPU 以 warp 为基本执行调度单位，但每个 thread 仍有自己的 `threadIdx`、register state 和数据地址。

```text
blockDim.x = 256
Warp 0: threads   0–31
Warp 1: threads  32–63
...
Warp 7: threads 224–255
```

因此一个 256-thread block 包含 8 warps。若 block 有 100 threads，则需要 $\lceil100/32\rceil=4$ warps；最后一个 warp 只有 4 个有效 threads，但仍占一个 warp 的调度位置。

同一 warp 的 threads 执行相同指令、处理不同元素。如果同一 warp 内的 threads 走不同控制流路径，就会发生 **warp divergence**：硬件分别执行各条路径，并屏蔽当前路径不活跃的 lanes。分支本身不是问题；同一 warp 内路径不同才是问题。

对 $N=1000$、`blockDim.x=256`：

$$\text{gridDim.x}=\left\lceil\frac{1000}{256}\right\rceil=4$$

实际发出 $4\times256=1024$ threads，所以 kernel 必须写 `if (i < N)`；末尾 24 个线程不访问数组。

## 2. Warp coalescing：地址模式比线程数量更关键

在 compute capability 6.0+ 的官方简化模型里，warp 的 global-memory 访问会合并为覆盖请求地址所需的 **32-byte transactions**。对 32 个线程各读一个 FP32（4 bytes）：

| 一个 warp 的地址 | 请求 bytes | 32B segments | 模型效率 |
|---|---:|---:|---:|
| `lane * 4`（对齐、连续） | 128 | 4 | 100% |
| `(lane + 1) * 4`（错 1 个 float） | 128 | 5 | 80% |
| `lane * 2 * 4`（stride=2） | 128 | 8 | 50% |

这里的“效率”只是 $\text{requested bytes}/\text{transaction bytes}$ 的**事务模型**，不是实测带宽。cache line reuse、ECC、实际 transaction replay 等会改变真机结果。

## 3. Shared-memory bank conflict：`[32][32]` 为什么常加一列

compute capability 5.x+ 的 shared memory 有 32 banks，连续 32-bit words 映射到连续 banks：

$$\text{bank}(w)=w\bmod32$$

若一个 warp 逐行访问 `tile[row][lane]`，lane 0..31 落到 32 个不同 banks。若逐列访问未 padding 的 `tile[lane][column]`，word index 为 $32\cdot lane+column$，所有 lanes 都落到同一个 bank，形成 32-way conflict（同地址 broadcast 例外）。改成 `tile[32][33]` 后：

$$\text{bank}(lane)=(33\cdot lane+column)\bmod32=(lane+column)\bmod32$$

于是 32 lanes 分散到 32 banks。代价是每行多一个 float 的 shared memory。

## 4. Occupancy：是延迟隐藏能力，不是得分

$$\text{occupancy}=\frac{\text{active warps per SM}}{\text{maximum warps per SM}}$$

它同时受 threads/block、registers/thread、shared memory/block 和硬件上限约束。block 更大不保证 occupancy 更高；occupancy 超过某个点也不保证更快。正确做法是先用 128–256 threads/block 作为实验起点，再用 compiler resource report、occupancy calculator 与 profiler 验证。

## 5. 可执行小任务

### CPU 可执行：检查真实映射

`cuda_access_model.py` 不是“打印步骤”：它实际构造 32 个 lane 地址，集合化计算 32-byte segments，按 word index `% 32` 计算 shared-memory bank，并输出 launch tail。

```bash
python3 ai-infra/r2-day-07-cuda-programming-model/cuda_access_model.py
python3 -m unittest discover -s ai-infra/r2-day-07-cuda-programming-model -p 'test_*.py' -v
```

### CUDA 待验证：真正执行 vector add

`vector_add.cu` 会交叉运行 `blockSize={128,256,512}` 与 `stride={1,2}`；每个 case 都会：分配/拷贝 device buffers、执行 kernel、用 CUDA events 计时、拷回并逐个验证结果；没有把结构输出冒充运行成功。

```bash
nvcc -O3 ai-infra/r2-day-07-cuda-programming-model/vector_add.cu -o /tmp/vector_add
/tmp/vector_add
```

只有在 CUDA GPU 上运行后，才能记录两种访问的 kernel latency / effective bandwidth，并应配合 Nsight Compute 看 global-memory transactions；单次 event 时间也不应直接当稳定 benchmark。

## 状态

- 已验证：纯 Python 语法、8 个 CPU 单元测试、launch/segment/bank 映射模型。
- **execution not validated on CUDA/H100 / 待H100验证**：`vector_add.cu` 未编译、kernel 未运行；未验证 latency、bandwidth、occupancy、bank-conflict 或任何 profiler 指标。
- 因缺少 CUDA toolkit / GPU，本课保持 `blocked`；文中 transaction count 是 theoretical access model，不是 benchmark。

## 原始 / 官方来源

- NVIDIA CUDA Programming Guide（programming model）：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA CUDA C++ Best Practices Guide（coalescing、shared-memory banks、occupancy）：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA Nsight Compute Occupancy Calculator：https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator
