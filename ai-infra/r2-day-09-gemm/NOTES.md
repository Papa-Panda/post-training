# NOTES — r2-Day09 GEMM Tiling

## 准确术语

- **GEMM (General Matrix Multiply)**：通常指 $C\leftarrow\alpha\,op(A)op(B)+\beta C$；本课自写 kernel 固定为 row-major、$\alpha=1,\beta=0$、不转置的 $C=AB$。
- **Tile / blocking**：把 $M,N,K$ 三个循环按块组织，使一块输入在更近的 memory hierarchy 中被多次使用。
- **CTA / thread block tile**：一个 CUDA block 协作计算的输出子矩阵；本课是 $16\times16$。
- **Register accumulator**：每个 thread 的局部 `acc`；遍历所有 $K$ tiles 后才写 global $C$。
- **Arithmetic intensity**：算法工作量 FLOPs 除以数据移动 bytes；必须注明在哪一级 memory 与如何计数。本课只给理想 FP32 global payload model，不冒充 DRAM counter。
- **Tensor Core / MMA**：warp 或 warp-group 协作执行 matrix multiply-accumulate 的专用路径；本课 scalar FP32 kernel 没有使用它。
- **Epilogue**：把 accumulator 转换/缩放并写回输出的阶段；生产 GEMM 常在这里融合 bias、activation 等操作。

## Connection to Prev 的实质

Day08 reduction 用“global → shared → registers”缩小汇合范围；Day09 GEMM 用同一层级扩大复用范围：

1. naive kernel 的多个输出会反复读取同一个 $A_{iq}$ 或 $B_{qj}$；
2. tiled kernel 让一个 block 协作加载 $A/B$ tiles；
3. `__syncthreads()` 保证 tile 完整后再消费，也保证下一轮覆盖 shared memory 前上一轮消费已结束；
4. 每个 thread 把自己的 $C_{ij}$ partial sum 留在 register 中跨越全部 $K$ tiles。

**牺牲**：2 个 block barriers/K-tile、shared memory、边界 zero-padding、布局/occupancy 调优复杂度。**换取**：整除模型下，global scalar loads 从 $2MNK$ 降到 $2MNK/T$。**何时不赚**：固定开销主导的小矩阵、极瘦矩阵、低复用/不规则访问、tile 资源压力压低 occupancy，或 cuBLAS/CUTLASS 已覆盖的标准算子。

## 三重循环与 shape

数学定义：

$$A[M,K]\times B[K,N]\rightarrow C[M,N]$$

标量循环：

```text
for i in [0, M):
  for j in [0, N):
    acc = 0
    for q in [0, K):
      acc += A[i,q] * B[q,j]
    C[i,j] = acc
```

- 输出元素：$MN$；
- 每元素：$K$ 次乘法和 $K$ 次累加到 zero-initialized accumulator；
- conventional count：$2MNK$ FLOPs；
- 数学上若把第一个 product 直接赋值，可写 $MNK$ multiplies 与 $MN(K-1)$ adds，但 GEMM 性能口径仍通常用 $2MNK$。

## 可手算例子 1：数值正确性

$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$$

逐项：

- $C_{00}=1\cdot5+2\cdot7=19$
- $C_{01}=1\cdot6+2\cdot8=22$
- $C_{10}=3\cdot5+4\cdot7=43$
- $C_{11}=3\cdot6+4\cdot8=50$

`test_hand_computable_two_by_two` 对 naive、tile=1、tile=2 三条真实执行路径都断言此结果。

## 可手算例子 2：$4\times4,T=2$ 流量

共有 $2\times2=4$ 个 output tiles，每个 output tile 遍历 $K/T=2$ 个 K-tiles。每阶段加载：

$$A_{tile}:2^2=4,\quad B_{tile}:2^2=4$$

因此 tiled 总 loads：

$$4\ \text{output tiles}\times2\ \text{K stages}\times(4+4)=64$$

而 naive 总 loads：

$$MN\times2K=16\times8=128$$

两者都做 64 multiply-adds，按惯例为 128 FLOPs，并写 16 个输出。计入 FP32 payload：

$$I_{naive}=\frac{128}{4(128+16)}\approx0.222\ \text{FLOP/byte}$$

$$I_{tiled}=\frac{128}{4(64+16)}=0.4\ \text{FLOP/byte}$$

这不是硬件 DRAM bytes，因为 cache line、write policy、transactions、编译器与 L1/L2 命中都未建模。

## CUDA 代码每一步确实做了什么

### `gemm_naive`

- `row/col` 由真实 `blockIdx/threadIdx` 计算；
- inner loop 真实读取 `a[row*k+inner]` 与 `b[inner*n+col]`；
- `acc` 被最终写入 `c[row*n+col]`，不会被 dead-code eliminate。

### `gemm_tiled`

每个 $K$-tile：

1. 每 thread 各加载一个 $A$ 与一个 $B$ scalar；越界位置写 0；
2. 第一个 `__syncthreads()` 后，任何 thread 才能读取其他线程发布的 tile 值；
3. inner loop 使用 shared tiles 更新 register `acc`；
4. 第二个 `__syncthreads()` 防止下一阶段覆盖仍被其他 warp 消费的 tile；
5. 所有阶段完成后，in-bounds thread 把真实 accumulator 写回 $C$。

### cuBLAS row-major 映射

cuBLAS 原生按 column-major 解释指针。row-major 存储中的 $A$ 可被看成 column-major 的 $A^T$，所以 harness 交换输入顺序并计算：

$$C^T=B^TA^T$$

具体调用的维度是 `(n, m, k)`，先传 `B`（leading dimension `n`），再传 `A`（leading dimension `k`），输出 leading dimension `n`。harness 显式设置并打印 `CUBLAS_DEFAULT_MATH`；cuBLAS 结果先保存为 reference，再用于检查两个自写 kernel。

## Edge tile 与计数边界

CUDA kernel 对 $M/N/K$ 非 16 整除时，把 out-of-bounds shared entries 置 0，因此数值语义正确；CPU model 只 materialize 有效子块。于是 CPU 的 `modeled_global_loads` 数的是**有效 scalar payload**，不是 CUDA warp 实际发出的 memory transactions。整除公式 $2MNK/T$ 只用于整除尺寸。

## 为什么“tile 越大越好”是错的

- shared footprint 按 $2T^2$ 增长，可能减少每 SM resident blocks；
- 本课一 thread/输出的设计需要 $T^2$ threads，受 1024 threads/block 限制；
- register pressure、bank conflict、coalescing 与 instruction issue 会改变瓶颈；
- 生产 kernel 用 thread tile/warp tile，让每 thread 算多个输出，并通过 MMA/Tensor Cores 提升 instruction-level throughput；
- async copy/double buffering 可覆盖 global→shared latency，但又增加 stage storage。

所以 tile 是 hardware/workload-specific 参数，应由正确性检查与 profiler 共同选择。

## 讨论难点

**为什么把 theoretical global loads 降低 16 倍，仍不能推导 tiled kernel 达到 cuBLAS 50%，甚至不能保证比 naive 快？**

回答应覆盖：

1. cache 可能已消除部分 naive 重读，payload model 不等于 DRAM transactions；
2. 两次 barrier/K-tile、address arithmetic 与边界分支会增加开销；
3. tile 的 shared/register footprint 会改变 occupancy；
4. 本课 scalar FP32 FMA 没用 Tensor Cores，cuBLAS 可能使用架构专用 MMA、pipelining 与 tuned epilogue；
5. 尺寸与 layout 决定 coalescing、reuse 和 launch amortization；
6. 只有同输入、同精度语义、warmup/repeats、CUDA events 与数值校验后的真机测量才可报告 ratio。

## 真机验证协议（尚未执行）

- 记录 GPU 型号、SM、clock/power policy、driver、CUDA/cuBLAS 版本；
- 固定 $M,N,K$、dtype/layout、tile、warmups=5、repeats=21；
- 保存完整编译命令与原始 stdout；
- 分别报告 median kernel ms 与按 $2MNK/t$ 计算的 TFLOP/s；
- 报告相对 cuBLAS ratio，同时保存 `max_abs_error_vs_cublas`；
- 用 Nsight Compute 核对 DRAM bytes、global-load efficiency、shared transactions/bank conflicts、barrier stalls、occupancy、Tensor Core utilization；
- 不把 CPU theoretical payload 或旧机器数字复制成 H100 benchmark。

## 验证状态

- 已执行 `py_compile`、8 个 CPU 单元测试和 CPU report。
- 已核对 shape：$A[M,K]B[K,N]\to C[M,N]$；CUDA 与 CPU 都支持 rectangular/edge shapes。
- 已核对单位：FP32 scalar 为 4 bytes；CUDA event 为 ms；TFLOP/s 公式分母使用 `ms * 1e9`；FLOPs、loads、stores 为无量纲计数。
- **execution not validated on CUDA/H100 / 待H100验证**：无 `nvcc`/GPU/cuBLAS，CUDA 编译、性能、Nsight 与 ROADMAP 50% cuBLAS 目标均保持 blocked/todo。

## 原始 / 官方来源

1. NVIDIA CUDA C++ Best Practices Guide — Shared Memory in Matrix Multiplication: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-best-practices-guide/index.html
2. NVIDIA cuBLAS documentation — GEMM and `cublasSgemm`: https://docs.nvidia.com/cuda/archive/12.1.0/cublas/index.html
3. NVIDIA CUTLASS — Efficient GEMM in CUDA: https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/efficient_gemm.html
