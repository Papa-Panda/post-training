# r2-Day09 — GEMM Tiling：用数据复用把乘法从 HBM 喂给寄存器

## Connection to Prev

r2-Day08 把 reduction 的中间量从 global hotspot 收回到 block/warp；r2-Day09 把同一原则扩展到 $C=AB$：先把 $A/B$ 子块搬进 shared memory，再让一个 tile 内的线程反复复用。**牺牲** shared memory、barrier、边界处理与更复杂布局，**换取**约 $T$ 倍的理想 global-load 降幅和更高算术强度；当矩阵很小/瘦、tile 降低 occupancy、布局导致 bank conflict，或库已能直接调用时，手写 tiled kernel 不赚。

## 1. GEMM 的 shape 与工作量

对 row-major 矩阵

$$A\in\mathbb{R}^{M\times K},\quad B\in\mathbb{R}^{K\times N},\quad C=AB\in\mathbb{R}^{M\times N}$$

每个输出元素为

$$C_{ij}=\sum_{q=0}^{K-1}A_{iq}B_{qj}$$

共有 $MNK$ 次 multiply-add；按 GEMM 常用口径“一乘一加算 2 FLOPs”，理论工作量为

$$F=2MNK\ \text{FLOPs}$$

这里的 FLOPs 是算法计数，不是设备实测吞吐。

## 2. Naive 与 tiled 的流量账

### Naive：一个 thread 算一个 $C_{ij}$

每个输出都独立重读 $K$ 个 $A$ 与 $K$ 个 $B$ 元素。忽略 cache、只数源码语义上的 FP32 payload：

$$L_{naive}=2MNK\ \text{scalar loads},\qquad S=MN\ \text{stores}$$

$$I_{naive}=\frac{2MNK}{4(2MNK+MN)}\ \text{FLOP/byte}$$

### Shared-memory tile：一个 block 算 $T\times T$ 输出

对整除尺寸，每个 output tile 在每个 $K$-tile 阶段只加载 $T^2$ 个 $A$ 和 $T^2$ 个 $B$ 元素，然后在 block 内复用：

$$L_{tiled}=\frac{M}{T}\frac{N}{T}\frac{K}{T}(2T^2)=\frac{2MNK}{T}$$

所以理想 global scalar-load 数下降 $T$ 倍；每 block 的两块 FP32 shared tile 占

$$S_{shared}=2T^2\times 4\ \text{bytes}$$

本课 CUDA 代码取 $T=16$，即 $2\times16^2\times4=2048$ bytes/block。它不是 Tensor Core kernel：每个 thread 仍用 FP32 scalar FMA 累加一个输出。

## 3. 可手算例子

数值先算 $2\times2$：

$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$$

$$C=AB=\begin{bmatrix}1\cdot5+2\cdot7&1\cdot6+2\cdot8\\3\cdot5+4\cdot7&3\cdot6+4\cdot8\end{bmatrix}=\begin{bmatrix}19&22\\43&50\end{bmatrix}$$

再算流量：$M=N=K=4,T=2$。

- 工作量：$2MNK=128$ FLOPs；输出 store 为 $MN=16$ 个 scalar。
- naive：$2MNK=128$ 个 scalar loads，FP32 payload $4(128+16)=576$ bytes，$I=128/576\approx0.222$ FLOP/byte。
- tiled：$2MNK/T=64$ 个 scalar loads，payload $4(64+16)=320$ bytes，$I=128/320=0.4$ FLOP/byte。

这只是**无 cache 的 theoretical payload model**；不能写成实际 DRAM bytes、带宽或 speedup。

## 4. 可执行代码

### CPU：真正执行两种数据流

`gemm_models.py` 的 naive 版按 output element 做完整 inner loop；tiled 版实际 materialize $A/B$ tiles、在 tile 内复用并写回结果。计数来自被执行的数据流，不是打印步骤。

```bash
python3 ai-infra/r2-day-09-gemm/gemm_models.py --size 4 --tile 2
python3 -m unittest discover -s ai-infra/r2-day-09-gemm -p 'test_*.py' -v
```

### CUDA：naive、shared tiled、cuBLAS 同一 harness

`gemm_naive_tiled.cu` 让 `M/N/K` 真正进入分配、grid、索引、循环和 cuBLAS 调用。两个自写 kernel 与 `cublasSgemm` 使用同一输入；先保存 cuBLAS 结果，再逐元素检查最大绝对误差。21 次 CUDA-event kernel 时间取中位数，输出 TFLOP/s。

```bash
nvcc -O3 -std=c++17 ai-infra/r2-day-09-gemm/gemm_naive_tiled.cu \
  -lcublas -o /tmp/gemm_naive_tiled
/tmp/gemm_naive_tiled 1024 1024 1024
```

ROADMAP 的“1024² tiled GEMM 达到 cuBLAS 50%”是**待测目标**，不是本课现状。只有真机输出满足

$$\text{ratio}=\frac{\text{TFLOP/s}_{tiled}}{\text{TFLOP/s}_{cuBLAS}}\ge 0.5$$

才可记录达标；本环境没有 CUDA/H100，不能声称达标。

## 5. 为什么生产 GEMM 远不止一个 shared tile

本课只建立第一层复用。高性能 GEMM 还会把 tiling 映射到 threadblock → warp → MMA instruction，使用 register fragments、Tensor Cores、vectorized/coalesced copies、double buffering/async copy 与专门 epilogue。tile 越大，理论复用越高，但 shared memory/register pressure 也会上升，可能减少 active blocks/warps；因此不能只看 $L_{tiled}$ 公式选 tile。

这也直接连接 Day10 FlashAttention：不 materialize 整个 $S\times S$ attention matrix，而是让 $Q/K/V$ tiles 与 online-softmax 状态在片上分块流动。

## 状态

- 已验证：Python 语法、8 个 CPU 单元测试、naive/tiled 数值一致、矩形与非整除 edge tile、$4^3,T=2$ 的 FLOPs/traffic/算术强度。
- **execution not validated on CUDA/H100 / 待H100验证**：环境无 `nvcc`/CUDA GPU/cuBLAS；CUDA 编译、kernel 正确性、CUDA events、Nsight Compute、实际带宽/occupancy/TFLOP/s、以及相对 cuBLAS 50% 目标均未验证。
- 本课状态保持 `blocked`；没有声称 loss、耗时、带宽、comm%、MFU、设备拓扑或 benchmark speedup。

## 原始 / 官方来源

- NVIDIA CUDA C++ Best Practices Guide，Shared Memory in Matrix Multiplication：<https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-best-practices-guide/index.html>
- NVIDIA cuBLAS 文档，GEMM 定义与 `cublasSgemm`：<https://docs.nvidia.com/cuda/archive/12.1.0/cublas/index.html>
- NVIDIA CUTLASS，Efficient GEMM in CUDA（threadblock/warp/instruction 分层）：<https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/efficient_gemm.html>
