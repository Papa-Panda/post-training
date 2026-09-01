# 10 — GPU 架构到 LLM Training / Inference Kernel 的映射

## 1. Transformer 不是一个 kernel

一层包含：

- Q/K/V 与 output projection GEMM；
- attention score、softmax、value aggregation；
- MLP GEMM/activation；
- normalization、residual、dropout；
- training 的 backward、optimizer、activation/parameter communication；
- inference 的 KV-cache read/write、sampling 和调度。

同一层不同算子的瓶颈不同；“模型是 compute-bound”没有足够语义。

## 2. GEMM shape 决定硬件行为

线性层 $Y=XW$：

$$X\in\mathbb R^{M\times K},\quad W\in\mathbb R^{K\times N},\quad F\approx2MKN.$$

- training/大 batch prefill：$M=B\times S$ 大，权重在多个 tokens 间复用，容易提高 arithmetic intensity；
- autoregressive decode：每步 $M$ 很小，反复读取权重，容易 memory/launch-bound；
- tensor parallel 把 N/K 切小，单卡 GEMM 可能更 skinny，kernel efficiency 降低且 collective 频率上升。

因此增加 GPU 数可能同时降低本地 GEMM 效率并增加通信，不保证线性加速。

## 3. Attention 的数据流

标准 attention：

$$O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right)V.$$

若显式物化 $S\times S$ score/probability，HBM 流量与空间为 $O(S^2)$。FlashAttention 类算法按 tiles 在线维护 softmax 统计，使 score blocks 留在 on-chip memory，避免完整中间矩阵写回；它仍执行精确 attention（在浮点重排误差意义下），主要收益来自 I/O-aware tiling，而不是少算掉标准 attention 的全部 $QK^\top$ FLOPs。

在线 softmax 对每行维护 running max $m$ 和 normalizer $l$；合并新 block 时重新缩放旧 partial output。这是算法、数值稳定性和 GPU memory hierarchy 的协同设计。

## 4. KV cache 与 decode

每层 KV cache 容量的简化式：

$$Q_{\mathrm{KV}}=2B S H e,$$

其中 2 表示 K/V，$B$ 是并发序列，$S$ 是缓存长度，$H$ 是 hidden size，$e$ 是每元素 bytes；若使用 grouped-query/multi-query attention，KV heads 数下降，应按实际 KV dimension 替换 $H$。

每生成一个 token 要读取历史 KV 的相关部分，因此 decode latency/throughput 对：

- cache layout/page table；
- HBM bandwidth；
- continuous batching；
- quantized KV；
- tensor/context parallel 通信；
- scheduler fragmentation

高度敏感。容量公式不等于每 token 实际 HBM bytes，cache/paging/kernel fusion 会改变它。

## 5. Training 的算力、内存与通信

粗粒度 step 时间：

$$T_{\mathrm{step}}\approx T_{\mathrm{fwd}}+T_{\mathrm{bwd}}+T_{\mathrm{opt}}+T_{\mathrm{collective}}-T_{\mathrm{overlap}}+T_{\mathrm{bubble}}.$$

- activation checkpointing：减少存储，增加 recompute FLOPs；
- mixed precision：减少 bytes/提高 Tensor Core throughput，但需要数值协议；
- ZeRO/FSDP：shard 参数/梯度/optimizer state，引入 all-gather/reduce-scatter；
- pipeline parallel：减小单卡模型状态，增加 bubbles 与 activation P2P；
- tensor parallel：降低单卡 GEMM 规模，增加每层 collective。

选择并行方案本质是 HBM capacity、local compute shape、link bandwidth/latency 和 schedule 的联合优化。

## 6. Quantization 的性能方程

权重从 $e_1$ bytes 降到 $e_2$ bytes，理论权重流量下降 $e_1/e_2$，但端到端收益取决于：

- 是否有原生低精度 Tensor Core path；
- dequant scale/zero-point 的 FLOPs 与 bytes；
- group size/layout；
- activation/KV/collective 是否仍主导；
- 精度与校准。

decode 的小-M GEMM 更可能受益于减少权重 bytes；大训练 GEMM 可能原本已 compute-bound，量化收益来自更高 compute peak 而非仅带宽。

## 7. Fusion 与 persistent execution

典型融合：RMSNorm+linear、bias+activation、attention block、dequant+GEMM、collective+GEMM overlap。收益：减少中间 HBM 和 launch；风险：register/shared pressure、specialization、数值/正确性和维护成本。

通信计算融合不是把 NCCL call 放进同一函数就完成，而要设计 tile ownership 和 dependency：例如 GEMM 产生一片 output 后立刻 reduce-scatter，同时下一片继续计算。

## 8. 诊断路径

### Training step 慢

1. timeline 拆 fwd/bwd/optimizer/NCCL；
2. 看关键 GEMM shape 与 Tensor Core path；
3. 检查 activation/parameter memory 与 recompute；
4. 看 collective 是否在关键路径、是否真实 overlap；
5. 对长尾/straggler 做 rank-level 分析。

### Decode 慢

1. 分 TTFT 与 TPOT；
2. 按 prefill/decode kernel 分开；
3. 看 batch/token/sequence 分布；
4. 检查 weight/KV bytes、paged-cache 命中/碎片；
5. 看小 GEMM、collective 和 launch gap；
6. 做固定 workload 的并发 sweep，而非只测单请求。

## 9. 与仓库专题相接

- [`vllm-rollout/`](../vllm-rollout/)：把本章的 KV/TTFT/TPOT 机制落到 rollout stress test；
- [`ai-infra/`](../ai-infra/)：把 collective/capacity 模型落到 DDP/FSDP/checkpoint；
- [`grpo-vs-ppo/`](../grpo-vs-ppo/)：RL 算法产生的 rollout/train 比例由该专题负责，本章只解释执行成本；
- [`harness-engineering/`](../harness-engineering/)：agent runtime 的工具/调度逻辑不在本专题，模型调用的 GPU 性能在这里。

## 10. 结论

GPU 性能优化的闭环不是“写更低层代码”，而是：

$$\text{workload shape}\to\text{dataflow}\to\text{resource bottleneck}\to\text{measurement}\to\text{bounded change}\to\text{regression test}.$$

只有当这个链条证明现有 library/compiler 无法实现目标，才值得投入定制 kernel。

## 导航

- 上一篇：[09 Roofline 与 Profiling](09_roofline_profiling_tuning.md)
- 回到：[专题 README](README.md)
- 证据：[sources.md](sources.md) · 代码：[code/](code/) · 测试：[tests/](tests/)
