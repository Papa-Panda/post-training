# NOTES — r2-Day11 Triton 与 torch.compile

## 准确术语

- **Program**：Triton 的执行单元，一个 program 处理一个数据 block（`BLOCK` 个元素）；对应 CUDA 的 thread 处理一个元素。`tl.program_id(axis=0)` 返回当前 program 在 grid 中的编号。
- **Grid**：`kernel[grid](...)` 的 launch 配置，一维 softmax 用 `(triton.cdiv(n, BLOCK),)`；program 总数 $=\lceil N/BLOCK\rceil$。
- **`tl.arange(0, BLOCK)`**：编译期常量向量，生成 `[0..BLOCK)` 的下标；与 `pid*BLOCK` 相加得到本 program 负责的元素下标。
- **Mask（谓词）**：`mask = offs < n` 传给 `tl.load/tl.store`，越界 lane 不访存。这是谓词化的访存，不是 `if` 分支——语义上"这些 lane 不存在"，而不是"这些 thread 提前返回"。
- **`tl.constexpr`**：编译期常量（如 `BLOCK`），kernel 针对每个取值重新特化编译；这也是 Triton kernel 对不同 shape 会 recompile 的根因之一。
- **Eager**：PyTorch 默认执行模式，每行 Python 即 dispatch 一个 kernel；`torch.softmax` 不 fuse 时是 rowmax / sub+exp / rowsum / div 四个 kernel。
- **Dynamo**：torch.compile 的前端，把 Python 字节码转成 FX graph（抓图）；遇到不支持的语义即 **Graph Break**，break 之间回落 eager。
- **Inductor**：torch.compile 的后端，把 graph lower 成 Triton（GPU）/C++（CPU）代码；fuse 发生在 graph 内部。
- **Guard / recompile**：Dynamo 对 tensor 的 shape/stride 等做假设（guard），假设失效即 recompile；动态 shape 频繁变化时编译开销吃掉 fuse 收益。
- **Fusion（融合）**：把多个访存遍合并成一个 kernel，消灭的是中间结果的 HBM 往返，不是 FLOPs。本课 $6N\to2N$ 与 Day10 消灭 $S=QK^T$ 落地是同一类账。

## Connection to Prev 的实质

Day07/08 教 thread-level："一个 thread 干一件事，同步和访存自己排"。Day10 教 fused 的数学："中间量不落地"。Day11 教 fused 的两种工程写法：

1. **手写 fused（Triton）**：你声明 block 级数据流，编译器排 warp。`softmax_triton_kernel.py` 的 5 行核心（program_id → offs/mask → load → 片上 max/exp/sum → store）就是 Day10 online softmax"单行版本"的工程形态——区别只是 softmax 行内不需要跨 block 的 running $(m,\ell)$。
2. **自动 fused（torch.compile）**：你不写 kernel，Dynamo 抓图、Inductor 生成 Triton。代价是**失控感**：graph break、recompile、编译耗时都不在你手里，得用 `fullgraph=True` / profiler 拿回来。

两条路在 Inductor 处汇合：最终跑的都是 Triton。

## Kernel 逐行注释（`softmax_triton_kernel.py`）

```python
pid  = tl.program_id(axis=0)              # 本 program 是 grid 里第几个
offs = pid * BLOCK + tl.arange(0, BLOCK)  # 本 block 的 N 个元素下标（编译期向量）
mask = offs < n                           # 尾部谓词：越界 lane 不访存
x = tl.load(x_ptr + offs, mask=mask)      # 全行唯一的一次 load（HBM→片上）
m = tl.max(x, axis=0)                     # 片上归约：数值稳定的 max
e = tl.exp(x - m)                         # 片上 elementwise
s = tl.sum(e, axis=0)                     # 片上归约：分母
tl.store(out_ptr + offs, e / s, mask=mask)# 全行唯一的一次 store（片上→HBM）
```

对比 CUDA thread-level（Day07 vector-add）：那里每个 thread 自己算 `tid`、自己 `if (tid < n)`、访存合并靠"连续 thread 访问连续地址"的人工保证；这里 program 一次声明一个 block，coalescing/pipeline 由编译器保证。

## Graph Break 实例（`split_graphs` 模型）

```text
输入 trace: [matmul, add, BREAK(tensor.item() 强制同步), relu, mul,
             BREAK(data-dependent python if), softmax]
输出 graphs: [[matmul, add], [relu, mul], [softmax]]
```

工程 checklist：`torch.compile(model, fullgraph=True)` 先跑通（break 会直接抛错逼你修），再用 profiler 看 Inductor 是否真把 softmax 融了——这是 Day12 的活。

## 何时不赚（再强调一次）

1. 小 $N$：launch 开销主导；compile 开销要分摊。
2. 大 dense GEMM：cuBLAS 已是 SASS 级，手写 Triton 不赚；Triton 甜点是内存 bound 的 fused elementwise/reduction。
3. 动态 shape + 频繁 recompile：先定 shape（`mark_dynamic` 约束或 pad）。
4. break 碎片化：fuse 只在 graph 内，先修 break。

## 与 Day12 的连接

Day12（Nsight）回答"怎么证明省下的 HBM 真的变成了 wall-time"：Nsight Systems 看 kernel 个数是否从 4 个变成 1 个、host 端是否有 compile/recompile gap；Nsight Compute 看 fused kernel 的 memory throughput 是否顶到 HBM 屋顶。流量模型 $6N\to2N$ 只是理论估计，profiler 才是尺子。
