# r2-Day13 — Attention 变种：MHA→MQA/GQA/MLA（省 KV cache）与 MoE（稀疏换算力）

## Connection to Prev

r2-Day12 给了我们验证的仪器（Nsight Systems / Compute）：Day11 的 fused kernel
到底省没省，要拿测量说话。r2-Day13 往上走一层：**不再是"同样的数学、更快的执行"，
而是"换一套数学、省掉整块显存"**。r2-Day10 的 FlashAttention 省的是**一次**
attention 计算内部的 HBM 往返（ $O(N^2)$ 流量 → $O(N)$ ）；今天的
MHA→MQA/GQA/MLA 省的是 **decode 每一步都要重读的 KV cache**；MoE 走另一条路：
**参数量涨 $E$ 倍，FLOPs/token 不涨**，用稀疏换算力。

**牺牲**：MQA/GQA 把多个 query head 的 KV 绑在一起 → 表达能力下降（MQA 在部分任务上
可测的质量损失）；MLA 低秩压缩 → 训练与并行切分的复杂度上升（RoPE 必须解耦、
吸收技巧与 TP 切分互相作用）；MoE 稀疏 → expert parallelism 的 all-to-all 通信税 +
负载不均（capacity / dropping）。**换取**：decode 的 KV cache 字节数和 HBM 流量
成倍下降——decode 是 memory-bound（Day06/Day12 的 roofline 逻辑），省字节 = 省时间；
MoE 用参数量换"同样 FLOPs 下更强的模型"。**何时不赚**：prefill 阶段（compute-bound，
 $QK^T$ 的 FLOPs 一点没变，见下）；KV 本来就不占大头的短序列/小模型；MoE 在小 batch
下专家权重加载摊不薄（见例 C）。

## 0. 符号表（每个字母的含义）

- $L$ ：transformer 层数
- $h_q$ ：query head 数； $h_{kv}$ ：key/value head 数
- $d$ ：每个 head 的维度； $d_{model}$ ：模型 hidden 维
- $B$ ：每个元素的字节数（fp16 → $B=2$ ）
- $S$ ：序列长度（KV cache 里存的 token 数）
- $G$ ：GQA 的组数（ $G$ 个 KV head，每组 $h_q/G$ 个 query head 共享）
- $d_c$ ：MLA 压缩隐向量维度； $d_r$ ：MLA 解耦 RoPE 维度
- $c_t^{KV}$ ：第 $t$ 个 token 的压缩隐向量， $c_t^{KV} = h_t W^{DKV}$ ，
  $h_t$ 为层输入， $W^{DKV}$ 为下投影矩阵——**cache 里实际存的就是它**
- $E$ ：MoE 专家总数； $k$ ：每个 token 激活的 top-k 专家数
- $d_{ff}$ ：专家 FFN 中间维； $T$ ：一次前向的 token 数

## 1. 可手算例子 A：KV cache 账本（7B 风格： $L=32$ ， $h_q=32$ ， $d=128$ ，fp16， $S=131072$ ）

每 token 每层存 K 和 V 两个 $(h_{kv}, d)$ 张量：

$$bytes = L \times 2 \times h_{kv} \times d \times B \times S$$

| 方案 | $h_{kv}$ | 每 token 每层 | 总量（ $S=128k$ ） | 相对 MHA |
|---|---|---|---|---|
| MHA | 32 | $16384\ \text{B}$ | $68719476736\ \text{B} = 64\ \text{GiB}$ | $1\times$ |
| GQA-8 | 8 | $4096\ \text{B}$ | $17179869184\ \text{B} = 16\ \text{GiB}$ | $4\times$ |
| MLA（V3 维度 $d_c=512, d_r=64$ ） | — | $(512+64)\times2 = 1152\ \text{B}$ | $4831838208\ \text{B} = 4.5\ \text{GiB}$ | $14.2\times$ |
| MQA | 1 | $512\ \text{B}$ | $2147483648\ \text{B} = 2\ \text{GiB}$ | $32\times$ |

（head 口径验算 DeepSeek-V3：MHA $2\times128\times128\times2 = 65536\ \text{B}$ ，
MLA $(512+64)\times2 = 1152\ \text{B}$ ， $65536/1152 \approx 56.9\times$ 。）

decode 每步每个 layer 都要把全量 KV 从 HBM 读一遍，所以上表同时是**每步的 KV 流量**。
换成时间下限（HBM $3.35\ \text{TB/s}$ ，Day06 官方规格，**theoretical estimate**）：
MHA $64\ \text{GiB}$ → $20.5\ \text{ms}$ ；GQA-8 → $5.1\ \text{ms}$ 。
这就是"省字节 = 省时间"的定量含义——注意这只是 KV 流量的下限，真实 decode 还要读权重。

为什么 prefill 不赚：scores 的 FLOPs $= 2 S^2 h_q d$ ，只和 query head 数有关，
GQA/MQA/MLA 都没动 $h_q$ ，prefill（compute-bound）的算量原样不动。

## 2. 可手算例子 B：GQA 分组（ $n_q=4$ ， $G=2$ ， $d=2$ ， $S=2$ ）

Group 1（head 1、2 共用）： $K_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ ，
 $V_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ ；
Group 2（head 3、4 共用）： $K_2 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ ，
 $V_2 = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$ 。

取 head 1 的 $q_1 = [1, 0]$ ：
scores $= q_1 K_1^T = [1, 0]$ ；除以 $\sqrt{2}$ 得 `[0.7071, 0]` ；
softmax： $e^{0.7071} \approx 2.028$ ， $e^0 = 1$ ，和 `3.028` →
$[0.6698,\ 0.3302]$ ；
输出 $= 0.6698 \times [1,2] + 0.3302 \times [3,4] = [1.6604,\ 2.6604]$ 。
head 2 的 $q_2 = [0, 1]$ 用**同一份** $K_1, V_1$ 再算一遍——
KV 只存了一份、读了一份，这就是 GQA 相对 MHA 省掉的那一半流量。
代码里 `test_hand_example_gqa_grouping` 断言了 `[1.6604, 2.6604]` 。

两个退化（代码已数值验证）： $G = n_q$ 时 GQA $\equiv$ MHA；
 $G = 1$ 时 GQA $\equiv$ MQA。GQA 就是 MHA↔MQA 之间的连续插值，
Llama-2-70B 选 GQA-8（64 个 query head，8 个 KV head）是实证折中。

## 3. 可手算例子 C：MoE 路由与账本（ $E=4$ ， $k=1$ ， $d=4$ ， $d_{ff}=8$ ， $T=3$ ）

Router logits（ $T \times E$ ）：
token1 `[2.0, 1.0, 0.5, 0.1]` → expert 0；
token2 `[0.2, 3.0, 0.1, 0.4]` → expert 1；
token3 `[0.1, 0.2, 2.5, 0.3]` → expert 2。

- FLOPs/token：dense $= 4 d d_{ff} = 128$ ；
  MoE $= k \times 128 = 128$ （ $k=1$ 时**一分不涨**）。
- 参数量：dense $= 2 d d_{ff} = 64$ ；
  MoE $= E \times 64 = 256$ （ $4\times$ ，这就是"拿参数换能力"）。
- all-to-all：dispatch 把 token 发往专家所在卡、combine 发回拼，
  $2 \times T \times d \times B = 2\times3\times4\times2 = 48\ \text{B}$ 。
  这是稀疏的通信税，与 Day03 的 collective 主题直接相连。

何时不赚（小 batch 算账）： $E=8$ 、 $d=4096$ 、 $d_{ff}=14336$ 时，
单层 MoE 专家权重 $= 8 \times 2 \times 4096 \times 14336 \times 2\ \text{B} \approx 1.88\ \text{GB}$ ；
batch=1 时每步都要把 8 个专家的权重从 HBM 读一遍，
只为算 1 个 token 的 1 个专家——memory-bound 下这就是纯浪费。
（theoretical estimate：只算了权重流量，未含路由与 all-to-all。）

## 4. 可执行代码

- `attention_variants.py`：`kv_cache_bytes`（账本公式）、
  `mla_cache_bytes_per_token`、`decode_kv_traffic_per_step`、
  `kv_time_lower_bound_ms`（ $3.35\ \text{TB/s}$ 下限换算）、
  `mha_forward` / `gqa_forward`（numpy，可验证 $G=n_q$ 退化与 $G=1$ 共享）、
  `mla_absorb_demo`（验证吸收恒等式，见 NOTES §2）。
- `moe_accounting.py`：`router_topk`、`dense_ffn_flops` / `moe_ffn_flops`、
  `dense_ffn_params` / `moe_ffn_params`、`alltoall_bytes`（dispatch+combine）。

```bash
python3 ai-infra/r2-day-13-mha-mqa-gqa-mla-moe/attention_variants.py
python3 ai-infra/r2-day-13-mha-mqa-gqa-mla-moe/moe_accounting.py
python3 -m unittest discover -s ai-infra/r2-day-13-mha-mqa-gqa-mla-moe -p 'test_*.py' -v
```

## 5. 何时不赚（分方案）

1. **MQA/GQA 在 prefill 不赚**： $QK^T$ 的 FLOPs 只和 $h_q$ 有关，省的是 decode 的
   HBM 流量；短 prompt、prefill 主导的离线任务收益很小。
2. **MQA 走极端有质量代价**：GQA 论文明确报告 MQA 可导致质量下降与训练不稳定，
   且已有 MHA checkpoint 转 MQA 需要 uptraining（约 5% 预训练算量）。
3. **MLA 的复杂度税**：RoPE 必须解耦（ $R_t$ 卡在 $W^{UQ}$ 与 $W^{UK}$ 之间，
   矩阵乘法不可交换，吸收恒等式对 RoPE 部分不成立）；吸收后的投影改变了 TP 切分面，
   与标准 MHA 的按 head 切 TP 需要额外处理——这是 Day15（TP/PP/SP）的议题。
4. **MoE 小 batch 不赚**：专家权重全量加载摊不薄；all-to-all 在带宽不足时成为瓶颈；
   路由抖动带来延迟不确定性；全部专家权重常驻 HBM，单卡推理的显存压力反而更大。

## 状态

- 已验证：Python 语法；18 个 CPU 单元测试（账本 $64/16/4.5/2\ \text{GiB}$ 精确整数；
  GQA $G=n_q$ ≡ MHA、 $G=1$ 共享 KV 数值一致；MLA 吸收恒等式多 seed 通过；
  手算例 B `[1.6604, 2.6604]` ；路由 top-k、FLOPs/参数账本、all-to-all $48\ \text{B}$ ；
  KV 时间下限 $20.5/5.1\ \text{ms}$ ）。
- **execution not validated / 待H100验证**：本环境无 torch/CUDA GPU；
  `mha_forward` / `gqa_forward` 的 numpy 语义不等于 `torch.nn` 算子实测；
  $20.5/5.1\ \text{ms}$ 是 theoretical estimate（KV 流量下限，未含权重与计算）；
  未测量任何真实带宽、延迟、MFU、comm%、设备拓扑。
- 本课状态保持 `blocked`；没有声称 loss、耗时实测、带宽实测、comm%、MFU 或设备拓扑。

## 原始 / 官方来源

- MQA：Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need",
  <https://arxiv.org/abs/1911.02150>
- GQA：Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints", <https://arxiv.org/abs/2305.13245>
- MLA：DeepSeek-AI, "DeepSeek-V2", <https://arxiv.org/abs/2405.04434>；
  "DeepSeek-V3 Technical Report", <https://arxiv.org/abs/2412.19437>
  （ $d_c=512$ ， $d_r=64$ ， $n_h=128$ ， $d_h=128$ ；V2 236B/21B active，
  V3 671B/37B active）
- MoE：Shazeer et al., "Outrageously Large Neural Networks",
  <https://arxiv.org/abs/1701.06538>；
  Fedus et al., "Switch Transformers", <https://arxiv.org/abs/2101.03961>；
  DeepSeekMoE, <https://arxiv.org/abs/2401.06066>；
  Mixtral of Experts, <https://arxiv.org/abs/2401.04088>
- Llama-2-70B GQA-8 配置：Touvron et al., "Llama 2", <https://arxiv.org/abs/2307.09288>
