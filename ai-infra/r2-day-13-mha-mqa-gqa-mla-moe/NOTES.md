# r2-Day13 NOTES — Attention 变种的数学与账本

记号与 README §0 同一份，这里不重定义，只推导。

## 1. MHA → MQA → GQA：KV head 共享是一条连续谱

标准 MHA 的第 $l$ 层： $h_q$ 个 head，每个 head 有独立的
 $W^Q_h, W^K_h, W^V_h \in \mathbb{R}^{d_{model} \times d}$ 。
decode 时第 $t$ 步，新 token 的 $q_{t,h}$ 要和**全部** $S$ 个历史 token 的
 $k_{j,h}, v_{j,h}$ 做 attention，所以每个 head 的 $(K_h, V_h)$ 必须常驻 cache。

KV cache 公式（README 例 A 已手算）：

$$bytes = L \times 2 \times h_{kv} \times d \times B \times S$$

三点观察：

1. 它对 $S$ 线性——长上下文下这一项主导 decode 显存（呼应 Day 20–21 的 KV Cache 课）。
2. 它对 $h_{kv}$ 线性——**这就是全部三种变体下手的地方**。
   MQA（Shazeer 2019）： $h_{kv} = 1$ ，所有 query head 共享一对 KV。
   GQA（Ainslie et al. 2023）： $h_{kv} = G$ ， $h_q$ 个 query head 分成 $G$ 组，
   每组共享一对 KV。 $G = 1$ 退化为 MQA， $G = h_q$ 退化为 MHA。
3. 它**不碰** $h_q$ ——所以 prefill 的 $QK^T$ FLOPs（ $2 S^2 h_q d$ ）原样不动。
   省的是 decode 的 HBM 读流量，不是算量。这是一个常见的误解澄清点：
   GQA/MQA/MLA 是 memory 优化，不是 compute 优化。

GQA 论文的两个实证结论（原文摘要，可查 arXiv:2305.13245）：

- MQA 能大幅加速 decoder 推理，但可导致质量下降与训练不稳定；
- 把已有的 MHA checkpoint 做 mean-pool（组内 K/V 投影矩阵取平均）再 uptraining
  （约 5% 预训练算量），得到的 GQA 质量接近 MHA、速度接近 MQA。

## 2. MLA：低秩联合压缩 + 吸收恒等式

DeepSeek-V2（arXiv:2405.04434）提出的 Multi-Head Latent Attention。
核心假设：**所有 head 的 K、V 联合起来是低秩的**，可以先压缩再重建。

对第 $t$ 个 token（ $h_t \in \mathbb{R}^{1 \times d_{model}}$ 为层输入）：

$$c_t^{KV} = h_t W^{DKV}, \quad W^{DKV} \in \mathbb{R}^{d_{model} \times d_c}$$

$$k_t = c_t^{KV} W^{UK}, \quad v_t = c_t^{KV} W^{UV}, \quad W^{UK}, W^{UV} \in \mathbb{R}^{d_c \times (n_h d_h)}$$

cache 里只存 $c_t^{KV}$ （ $d_c$ 维）。V3 取 $d_c = 512$ ，
对比 MHA 的 $2 \times 128 \times 128 = 32768$ 维 → $56.9\times$ 。

**吸收恒等式**（inference 时不用展开 K 的数学依据）。
记 $W^{DQ}, W^{UQ}$ 为 Q 侧的下/上投影， $q^{full}_t = h_t W^{DQ} W^{UQ}$ ，
 $k^{full}_t = c_t^{KV} W^{UK}$ 。attention logit：

$$s_t = q^{full}_t (k^{full}_t)^T = h_t W^{DQ} W^{UQ} (W^{UK})^T (c_t^{KV})^T$$

定义 $W^{abs} = W^{DQ} W^{UQ} (W^{UK})^T \in \mathbb{R}^{d_{model} \times d_c}$
（离线预计算一次），则：

$$s_t = h_t W^{abs} (c_t^{KV})^T$$

全程只碰 $d_c$ 维的 $c_t^{KV}$ ，**从未 materialize** $n_h d_h$ 维的 $k^{full}_t$ 。
`mla_absorb_demo` 用随机矩阵验证了该恒等式（多 seed， $10^{-10}$ 精度）。

**为什么 RoPE 不能被吸收**：真实 MLA 把位置信息解耦成单独的
 $k_t^R \in \mathbb{R}^{d_r}$ （ $d_r = 64$ ），logit 里有一项
 $q_t^T R_t W^{UK} c_t^{KV}$ ，其中 $R_t$ 是随位置 $t$ 变化的旋转矩阵。
 $R_t$ 卡在 $W^{UQ}$ 与 $W^{UK}$ 之间，矩阵乘法不可交换，
 $W^{abs}$ 的预计算对 RoPE 部分不成立——所以 cache 是
 $(c_t^{KV}, k_t^R)$ 两部分，共 $(d_c + d_r) B$ 字节/token/layer。

**与 TP 的耦合**（Day15 预告）：吸收后 K 的 head 结构不再显式存在，
朴素的"按 head 切 TP"会破坏吸收恒等式；MLA 推理的并行切分需要与标准 MHA
不同的处理。这是"架构决定并行策略"（ai_daily.csv 原话）的实例。

## 3. MoE：FLOPs/token 不涨，参数涨 $E$ 倍

每层 FFN 换成 $E$ 个专家 $E_1 \dots E_E$ + router。
第 $t$ 个 token：router 给出 logits $\ell_t \in \mathbb{R}^E$ ，
取 top-k，下发的权重为 top-k logits 的 softmax 重归一：

$$y_t = \sum_{i \in \text{topk}(\ell_t)} w_{t,i}\, E_i(h_t), \quad w_t = \text{softmax}(\ell_t[\text{topk}])$$

账本（README 例 C 手算，代码 `moe_accounting.py`）：

- FLOPs/token：dense $4 d d_{ff}$ ；MoE $k \times 4 d d_{ff}$ 。
  $k=1$ 时与 dense **完全相同**——MoE 省的不是单 token 算量，
  而是"同样算量下能装下更大的模型容量"。
- 参数量：dense $2 d d_{ff}$ ；MoE $E \times 2 d d_{ff}$ 。
- 通信税（expert parallelism）：dispatch + combine
  $= 2 \times T \times d \times B$ 字节（Day03 collective 的实例）。
- 负载不均：router 可能把太多 token 发给同一个专家；
  实践中用 capacity factor（每个专家最多接 $C \times T/E$ 个 token，
  多余的 drop 掉）或 aux loss / bias 调整（DeepSeek-V3 的
  auxiliary-loss-free bias 策略，arXiv:2412.19437）来处理。
  drop token =  silent 的质量损失，这是 MoE 账本里最容易漏的一行。

## 4. 术语表（准确术语）

- **KV head**：产生 K/V 的 head；MHA 里 $h_{kv} = h_q$ ，MQA 里 $h_{kv} = 1$ 。
- **Group（GQA）**：共享一对 KV 的 query head 集合；Llama-2-70B 的 GQA-8
  指 $h_q = 64$ 、 $h_{kv} = 8$ 。
- **Latent（MLA）**：压缩隐向量 $c_t^{KV}$ ；**decoupled RoPE**：与压缩路径
  分离的位置编码分支 $k_t^R$ 。
- **Weight absorption（吸收）**：把 $W^{UK}$ 乘进 Q 侧、推理时不展开 K 的技巧。
- **Expert parallelism**：按专家切分的并行方式，token 在卡间搬运，
  代价是 all-to-all。
- **Capacity factor / token dropping**：专家容量上限与溢出 token 的丢弃策略。

## 5. 与前后课程的连接

- ← Day10（FlashAttention）：FlashAttention 省的是**一次** attention 的 HBM 流量；
  今天省的是**跨 decode 步**的 KV 流量。两者正交，可叠加。
- ← Day12（Profiling）："GQA-8 把 decode KV 流量从 64 GiB/step 降到 16 GiB/step"
  这类断言，正确验证方式是 nsys 看 HBM 读字节、ncu 看 DRAM throughput，
  而不是只看公式。
- → Day14（FSDP/ZeRO）：MoE 的 $E\times$ 参数直接放大 ZeRO 切分的分母，
  也放大了 checkpoint 体积——参数账和显存账在 Day14 汇合。
- → Day15（TP/PP/SP）：GQA 的 KV head 数决定 TP 切分的粒度；
  MLA 的吸收与 TP 切分互相作用；MoE 引入 expert parallelism（第 4 种并行维）。
- → Day 20–21（KV Cache）：今天的账本是那两课 $32\ \text{GB}$ 手算的前置。
