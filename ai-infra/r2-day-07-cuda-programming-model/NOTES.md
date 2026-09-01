# NOTES — r2-Day07 CUDA 编程模型与访存映射

## 准确术语

- **Grid**：一次 kernel launch 创建的全部 thread blocks。
- **Thread block**：可在同一 SM 上驻留并通过 shared memory / block barrier 协作的一组线程；不同 blocks 不能依赖普通 `__syncthreads()` 互相同步。
- **Warp**：当前 NVIDIA GPU 上由 32 threads 组成的执行组；不是“一个线程”。
- **Coalescing**：硬件把一个 warp 的 global-memory requests 合并成尽可能少的 memory transactions。
- **Shared-memory bank conflict**：同一 warp 对多个不同地址的访问映射到同一 bank 时，请求被拆分/序列化；多个线程读同一地址的 broadcast 是例外。
- **Occupancy**：active warps per SM 与该 SM 最大可能 active warps 的比值；它帮助隐藏延迟，但最大 occupancy 不是优化目标本身。

## 核心心智模型：SM、block、warp、thread

### SM 与 block 不是一一对应

- 一个 block 只能驻留在一个 SM，不能跨 SM 拆分，也不会执行到一半迁移到另一个 SM；
- 一个 SM 通常可以同时驻留多个 blocks，并在这些 blocks 的 ready warps 之间调度；
- 放不下的 blocks 等待已有 blocks 完成，形成一轮一轮的 execution waves；
- 每 SM 的 resident block 数由 thread、warp、register、shared-memory 和硬件 block 上限共同决定。

$$B_{resident}=\min\left(B_{hw},\left\lfloor\frac{T_{SM}}{T_{block}}\right\rfloor,\left\lfloor\frac{W_{SM}}{W_{block}}\right\rfloor,\left\lfloor\frac{R_{SM}}{R_{block}}\right\rfloor,\left\lfloor\frac{S_{SM}}{S_{block}}\right\rfloor\right)$$

H100 每个 SM 最多 64 resident warps。若暂时只看 warp 上限：

| threads/block | warps/block | warp 上限给出的 blocks/SM 上界 |
|---:|---:|---:|
| 128 | 4 | 16 |
| 256 | 8 | 8 |
| 512 | 16 | 4 |
| 1024 | 32 | 2 |

这是资源上界，不是实测 occupancy；registers 和 shared memory 可能让实际 resident blocks 更少。

### Warp 是 32 个 threads 的执行组

`blockDim.x=256` 时：

```text
Warp 0: thread 0–31
Warp 1: thread 32–63
...
Warp 7: thread 224–255
```

所以该 block 有 $256/32=8$ warps。若 block 有 100 threads，则创建 $\lceil100/32\rceil=4$ warps，最后一个 warp 只有 4 个有效 lanes。

warp 是执行调度单位，不意味着 32 个 threads 共享一个逻辑线程：每个 thread 仍有自己的 index、register state 和输入/输出地址。对于：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
```

一个 warp 的 32 个 threads 执行同一条 add 指令，但计算 32 个不同的 `i`。

若同一 warp 内偶数 lanes 走路径 A、奇数 lanes 走路径 B，硬件需要分别执行两条路径并 mask 掉另一半 lanes，这叫 warp divergence。若整个 warp 一起选择同一路径，则没有 intra-warp divergence。

## 可手算小例子 1：launch geometry

$N=1000$，每 block 256 threads：

$$B=\left\lceil\frac{N}{256}\right\rceil=\left\lceil3.90625\right\rceil=4$$

$$T_{launched}=4\times256=1024,\qquad T_{inactive}=1024-1000=24$$

所以 `i = blockIdx.x * blockDim.x + threadIdx.x` 后必须有 `if (i < N)`。CPU 模型实际得到 `blocks=4`、`launched_threads=1024`、`inactive_threads=24`。

## 可手算小例子 2：一个 warp 访问几个 32B segments

假设 base address 32-byte aligned，每 lane 读一个 FP32。

### 对齐连续

lane $\ell$ 访问 byte address $4\ell$，范围 0–127 bytes，覆盖 segment bases：

$$\{0,32,64,96\}$$

共 4 transactions；请求 128 B，模型搬运 128 B，效率 100%。

### 错位一个 float

lane $\ell$ 访问 $4(\ell+1)$，范围 4–128 bytes，覆盖：

$$\{0,32,64,96,128\}$$

共 5 transactions；模型搬运 160 B，效率 $128/160=80\%$。

### stride = 2

lane $\ell$ 访问 $8\ell$，范围 0–248 bytes，覆盖 8 个 32B segments；每个 segment 只用一半的 float words：

$$\text{modeled efficiency}=128/(8\times32)=50\%$$

这些数字与官方 Best Practices Guide 的 CC 6.0+ 简化规则一致，但仍是理论 transaction model。真机 cache reuse 可能减轻错位访问损失。

## 可手算小例子 3：padding 消掉 32-way bank conflict

shared tile 为 row-major FP32 `tile[32][32]`。warp 的 lane $\ell$ 写同一 column 时，word index 为 $32\ell+c$：

$$\text{bank}(\ell)=(32\ell+c)\bmod32=c$$

32 lanes 命中 1 个 bank，即 32-way conflict。改成 `tile[32][33]`：

$$\text{bank}(\ell)=(33\ell+c)\bmod32=(\ell+c)\bmod32$$

32 lanes 命中 32 个不同 banks。CPU 模型已实际枚举 bank IDs 并测试 `distinct_banks: 1 -> 32`。

## 示例代码如何对应每一步

1. `launch_geometry()` 真正计算 blocks、发出线程数和 guarded tail；
2. `warp_float_addresses()` 真正构造 32 个 lane byte addresses；
3. `global_segments()` 真正对地址做 32-byte 对齐并去重；
4. `shared_bank_report()` 真正计算每 lane 的 bank，比较 row stride 32 与 33；
5. `vector_add.cu` 的 block size 真正参与 launch，stride 真正参与 array indexing；每个 case 都执行 kernel、结果校验和 CUDA-event timing，不存在只定义不用的参数。

## 讨论难点

为什么把 `blockDim.x` 从 256 改成 1024，不能推出 kernel 更快？请同时解释：

- warp 数量只是 launch 形状，active warps 还受 registers/thread 与 shared memory/block 限制；
- 大 block 可能减少每 SM 可同时驻留的 blocks；
- coalescing 由一个 warp 内地址模式决定，不由 block 越大自动改善；
- 更高 occupancy 可能隐藏延迟，但若触发 register spill 或增加同步，并不一定提高性能。

## 验证状态

- 已执行 `py_compile`、8 个 CPU 单元测试和纯 Python 报告。
- 已核对 shape/index：1D kernel 的 logical index 与 stride 都参与真实地址计算，tail 有边界保护。
- 已核对单位：global address 与 segment 均为 bytes；shared-memory bank 映射使用 32-bit words；模型效率为无量纲比值。
- **execution not validated on CUDA/H100 / 待H100验证**：未编译 `vector_add.cu`，未运行 CUDA kernel / Nsight；没有生成或声称任何实测 latency、bandwidth、occupancy、bank-conflict、comm% 或 MFU。

## 原始 / 官方来源

1. NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
2. NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
3. NVIDIA Nsight Compute Occupancy Calculator: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator
