# r2-Day11 — Triton 与 torch.compile：fused 算子的两种工程形态

## Connection to Prev

r2-Day10 在白板上推导了 fused tiling + online softmax：数学上把 $O(N^2)$ 中间量消灭了，但**谁来写这个 fused kernel**？r2-Day07/08 的 CUDA 是 thread-level（一个 thread 处理一个元素），手写 fused kernel 门槛极高。r2-Day11 回答工程形态：**Triton** 用 block-level 的 Python 语义写 fused 算子；**torch.compile** 则反过来——你不写 kernel，编译器从 eager 图里抓 subgraph 自动 fuse。**牺牲**：Triton 放弃 warp/register 级细粒度控制；torch.compile 放弃融合粒度的确定性，并付出编译耗时与 Graph Break 风险；**换取**：Triton 以 Python 级开发速度拿到接近手写 CUDA 的 fused 性能，torch.compile 零改写拿到自动融合。**何时不赚**：$N$ 很小（kernel launch / 编译开销主导）、大 dense GEMM 已有 cuBLAS 极致优化、torch.compile 遇到动态 shape 频繁 recompile 或 graph break 碎片化时。

## 1. 从 thread 到 program：Triton 的 block-level 模型

CUDA（r2-Day07）：`tid = blockIdx.x * blockDim.x + threadIdx.x`，一个 thread 处理**一个**元素，同步/访存全部手排。

Triton：一个 **program** 处理**一个 block**（`BLOCK` 个元素）。一维 softmax kernel 的核心三行：

```python
pid  = tl.program_id(axis=0)                  # 我是第几个 program
offs = pid * BLOCK + tl.arange(0, BLOCK)      # 我负责的 BLOCK 个元素下标
mask = offs < n                               # 尾部谓词，不是分支
x = tl.load(x_ptr + offs, mask=mask)          # 一次搬一个 block
```

$$\\text{programs} = \\lceil N / BLOCK \\rceil,\\qquad \\text{program } p \\text{ 处理元素 } [p\\cdot BLOCK,\\ (p+1)\\cdot BLOCK)$$

编译器接管剩下的事：自动做访存合并（coalescing）、把 `tl.max/tl.sum`  lower 成 block 内归约、做 pipeline 与 unroll。你**声明**"一个 block 的数据流"，编译器**决定**"warp 怎么排"。这正是"牺牲细控、换取生产力"的精确含义——也是为什么 Day08 的三版 reduce（atomic → shared-tree → warp-shuffle）在 Triton 里通常只写一种：编译器替你选了等价实现。

## 2. 可手算例子：fused softmax

$x=[1,2,\\dots,8]$（$N=8$）。Softmax $=\\exp(x-m)/\\sum\\exp(x-m)$，$m=\\max x$。

**Eager（PyTorch 不 fuse 时的 4 遍）**：rowmax → sub+exp → rowsum → div，每遍都扫一遍 $N$ 个元素：

$$L_{eager}=4N,\\quad S_{eager}=2N,\\quad \\text{payload}=6N\\ \\text{elements}$$

**Fused（Triton 单 kernel）**：一次 `tl.load`，max/exp/sum/div 全在片上，一次 `tl.store`：

$$L_{fused}=N,\\quad S_{fused}=N,\\quad \\text{payload}=2N\\ \\text{elements}$$

$N=8$ 手算：$m=8$，$S=\\sum_{k=0}^{7}e^{-k}=1.5814460128059595$，

$$out_0 = e^{-7}/S \\approx 0.000577,\\qquad out_7 = 1/S \\approx 0.632333$$

两条路径最大绝对差 $0.00\\times10^{0}$（CPU 双精度，见测试）——fused 是**精确等价**，不是近似。$N=8$ 时流量 $48$ vs $16$ 个元素，省 $3\\times$；这正是"fuse 消灭的是中间量的 HBM 往返"，与 Day10 消灭 $S=QK^T$ 落地是同一笔账。

Program 映射（$BLOCK=4$）：2 个 programs，`pid=0: offs=[0,1,2,3]`，`pid=1: offs=[4,5,6,7]`，mask 全 True；$N=10$ 时第 3 个 program 的 `mask=[True,True,False,False]`——尾部用谓词吞掉，不引入分支。

## 3. torch.compile：抓图、fuse、Graph Break

torch.compile 的流水线（文档行为，本课未实测）：**Dynamo** 把 Python 字节码转成 FX graph（抓图）→ **AOTAutograd** 处理 autograd → **Inductor** 生成 Triton/C++ 代码。eager 里的 `torch.softmax` 经 Inductor 同样会被 fuse 成接近第 2 节的单 kernel——**你没写 Triton，但最终跑的是 Triton**。

**Graph Break** 是这条路的核心风险：Dynamo 遇到它处理不了的 Python 语义（data-dependent 的 `if`、`.item()` 强制同步、`print` 等）就会把图**切断**，break 之间回落到 eager。fuse 只发生在一段 graph 内部：

```text
[matmul, add] --break(.item())--> [relu, mul] --break(data-dependent if)--> [softmax]
→ 3 个 graphs，fuse 不跨 break
```

本课 `split_graphs` 就是这个切分过程的可执行模型。工程含义：想吃到 compile 红利，**先消灭 graph break**（`torch.compile(..., fullgraph=True)` 会直接报错逼你修），而不是先调超参。

## 4. 可执行代码

### CPU：三组真正执行的语义模型

`triton_models.py` 的 `eager_softmax` 实际跑 4 遍循环并随遍累加流量计数；`fused_softmax` 实际跑单遍片上流水线；`program_grid` 实际算出每个 program 的 offs/mask；`split_graphs` 实际按 break 标记切分 op trace。计数来自被执行的数据流，不是打印步骤。

```bash
python3 ai-infra/r2-day-11-triton--torch.compile/triton_models.py
python3 -m unittest discover -s ai-infra/r2-day-11-triton--torch.compile -p 'test_*.py' -v
```

### Triton kernel：真实源码，guarded 未执行

`softmax_triton_kernel.py` 是完整的真实 Triton kernel（`@triton.jit`、`tl.program_id`、`tl.arange`、`mask`、`BLOCK: tl.constexpr`），与 CPU 模型的语义逐行对应。环境无 triton/torch/CUDA，`launch()` 拒绝执行并抛 `RuntimeError`；本地只做 `ast.parse` 语法检查。kernel 正确性与任何加速比均为 **execution not validated / 待H100验证**。

## 5. 何时不赚（再强调一次）

1. **$N$ 很小**：kernel launch 的固定开销主导，fused 省下的 HBM 往返不值 launch 钱；torch.compile 的编译耗时更要分摊到足够多的调用上。
2. **大 dense GEMM**：cuBLAS/cuDNN 已是手写 SASS 级优化，手写 Triton GEMM 很难赢——Triton 的甜点是**内存 bound 的 fused elementwise/reduction**（softmax、norm、gated MLP 激活），不是算力 bound 的大矩阵乘。
3. **torch.compile 动态 shape**：shape 频繁变化 → guard 失效 → recompile，编译开销吃掉 fuse 收益；对策是 `mark_dynamic` 约束或 padding 到固定 shape。
4. **Graph break 碎片化**：break 太多 = 回到 eager，fuse 无从谈起；先 `fullgraph=True` 修 break，再谈性能。

这也直接连接 Day12（Nsight）："到底省在哪"必须用 profiler 验证，而不是数流量模型——流量模型是理论估计，wall-time 才是答案。

## 状态

- 已验证：Python 语法；15 个 CPU 单元测试（手算 $S=1.5814460128059595$、$out_0=0.000577$、$out_7=0.632333$ 精确到 12 位；eager vs fused 差 $0$；流量计数 $48$ vs $16$；program 网格与尾部 mask；graph 切分 $3$ 段；kernel 文件 `ast.parse` 通过且含 Triton 结构 token；缺 triton 时 `launch` 拒绝执行）。
- **execution not validated on Triton/CUDA/H100 / 待H100验证**：环境无 triton/torch/nvcc/CUDA GPU；`.py` kernel 未编译执行、正确性、真实 HBM 流量、相对 eager 的加速比、torch.compile 的抓图/fuse 行为均未验证。`6N→2N` 是 theoretical estimate，不是测量。
- 本课状态保持 `blocked`；没有声称 loss、耗时、带宽、comm%、MFU、设备拓扑或 benchmark speedup。

## 原始 / 官方来源

- Triton 官方教程（block-level 编程模型、`tl.program_id`/`tl.arange`/`mask` 语义）：<https://triton-lang.org/main/getting-started/tutorials/index.html>
- torch.compile 文档（Dynamo 抓图、Inductor codegen、Graph Break）：<https://pytorch.org/docs/stable/torch.compiler.html>
- Inductor 生成 Triton 的说明：<https://pytorch.org/docs/stable/torch.compiler_inductor.html>
