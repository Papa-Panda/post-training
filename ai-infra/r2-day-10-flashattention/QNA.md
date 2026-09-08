# r2-Day10 — FlashAttention：问答归档

用户在 ai 课程聊天中提出的问题与总结。问题原文保留，答案按聊天记录整理；
英文术语（KV cache、hit rate、prefill、TTFT、block、hash、LRU、ring attention、
sequence parallelism、online softmax、P2P、NVLink、causal mask）保留。

归档说明：以下两问于 2026-09-06（周日）提出。KV cache 的完整课程在 roadmap
第三层"推理"（Day 20–21 TTFT/TPOT & KV Cache）；Ring Attention 的对口位置是
Day 15（TP/PP/SP 中的 SP）。两者都直接建立在 Day10 的 attention / K-V /
blockwise 机制之上，故暂归于本目录 QNA，待对应课程日创建时可迁移。

## 2026-09-06（周日）

### Q1：kv cache hit rate啥意思？为啥能hit？（需要所有token都一致啊！多难啊）

**A：** 你直觉的"难"是对的——如果要求整个请求一字不差，hit rate 会是 0。
关键在于：**它只要求前缀一致，不要求整个请求一致**，而真实负载的前缀高度重合。

**定义。** 一个请求的 prompt 是 token 序列 $x_1, x_2, \dots, x_n$ 。
vLLM 把 KV cache 按 block 存，block 大小 $B = 16$ （默认）。
第 $j$ 个 block 存 token $x_{(j-1)B+1} \dots x_{jB}$ 的 K、V 张量。
每个 block 有一个 hash，hash 内容 =（父 block 的 hash，本 block 的 16 个 token）；
链式 hash 保证位置也对得上——同一个词出现在位置 5 和位置 500，KV 值不一样，不能混用。
新请求进来按顺序查：block 1 的 token 序列跟缓存里某个 block 完全一样 → hit，直接复用，不用算；
第一个 miss 的 block 往后全部重算。
**hit rate** = 从缓存复用的 prompt token 数 / 总 prompt token 数。

所以"hit"的门槛是：**从头开始连续的若干个 block，每个 block 内 16 个 token 完全一致**。
不需要后缀一样。

**为啥现实中真能 hit。** 四个重合来源（以 rollout 场景为例）：

1. **System prompt**：线上服务每个请求都带同一套 system prompt，几百上千 token，前缀天然一致。
2. **多轮对话**：第 $t$ 轮的 prompt = 前 $t-1$ 轮全部内容 + 新问题，天然是前缀关系。
3. **Agentic rollout**：agent 每一步的 prompt = 上一步的完整 trajectory + 新 observation，天然前缀复用。
4. **GRPO 的 $G$ 个采样**：同一 prompt 采样 $G$ 个 response， $G$ 个请求的 prompt 部分**完全相同**，
   prefill 直接省掉 $(G-1)/G$ 。

**手算小例子。** $B = 16$ 。100 个请求，共享 160 token 的 system prompt（10 个 block），
各自后缀 32 token（2 个 block）：

- 总 block 数： $100 \times 12 = 1200$
- 实际计算：第一个请求算 12 个，后面 99 个只算后缀 2 个 → $12 + 99 \times 2 = 210$
- hit rate = $(1200 - 210) / 1200 =$ **82.5%**

**牺牲什么、换取什么、何时不赚。**

- **换取**：prefill 是 compute-bound（每 token 要跟全序列做 attention），
  跳过 prefill 直接省 FLOPs 和 TTFT（首 token 时间）。
- **牺牲**：GPU HBM 被缓存 block 占用，能同时跑的请求就少了；
  还有 hash 查表、LRU 淘汰的管理开销。
- **何时不赚**：prompt 各不相同、毫无共享前缀 → hit rate 趋近 0，白占内存；
  或者工作集远大于缓存，LRU 反复淘汰，缓存了也留不住。

一句话：KV cache 的 hit 不是"两个请求长得一样"，而是"大家从同一起点出发"。
前缀是起点，后缀各走各的。

### Q2：ring attention在我们的roadmap的哪个部分？或者和哪个部分相关？

**A：** 它不在 roadmap 的显式条目里，但它有且只有一个对口的位置：
**Day 15 "TP/PP/SP" 里的 SP（Sequence Parallelism）**。
Ring Attention 就是 sequence parallelism 在 attention 层落地的那个具体算法。

**跟已学内容的关系。**

- **Day10 FlashAttention**：FlashAttention 把 attention 按 block 切分、用 online softmax
  在单卡 SRAM 里逐块算，避免物化 $N \times N$ 的 $QK^T$ 。
  Ring Attention 是**同一套 blockwise 思想的跨卡推广**：序列维切分到 $P$ 张卡上排成环，
  每张卡算自己分到的 query block，同时把 K/V block 在环上传给邻居；
  传的时候手头有活干（算 attention），通信被计算盖住。
- **Day03 通信拓扑**：它依赖的 P2P 带宽就是那天算的 NVLink vs PCIe 的账。

**符号。** $N$ 是序列长度， $P$ 是卡数，每卡分到 $N/P$ 个 query token。
每个 step，每卡把自己手里的 K/V block 发给下一张卡，
同时用刚收到的 K/V block 更新本地的 online softmax 统计量
（ $m$ = row max， $\ell$ = row sum，跟 Day10 一模一样）。
$P-1$ 轮之后，每卡见过全部 K/V，结果精确等价于全量 attention。

**牺牲 / 换取 / 何时不赚。**

- **换取**：上下文长度不再受单卡显存限制， $P$ 张卡拼出 $P$ 倍的序列容量；
  通信被计算掩盖后接近线性扩展。
- **牺牲**：每个 K/V block 要在环上走一整圈，通信量是 $O(N \cdot d)$ 每卡；
  causal mask 下不同 block 的有效计算量不一样，负载不均；实现比 TP 复杂。
- **何时不赚**：序列不够长（block 太小，通信盖不住计算）；跨机 P2P 带宽低；
  batch 维度还有并行度没吃完——能先用 DP 解决就别上 SP。

**在 roadmap 里放哪。** 最合适的是 **Day15 讲 SP 时**，
作为"SP 不是一句切分序列，而是 attention 层要这样具体地算"的算法实例。
（Day15 尚未创建；届时可附 $P = 4$ 、 $N = 4096$ 的走查：
每轮哪个 block 在哪张卡上、传了什么、 $m/\ell$ 怎么更新。）

## 2026-09-07（周一·复习日）

以下 9 问于 2026-09-07 复习日提出，均为对 Day10 attention 机制的复习与追问，
问题原文保留，答案按聊天记录整理。Megatron-SP 相关问答的 roadmap 正式归属是
Day 15（TP/PP/SP 中的 SP），decode / KV cache 问答的正式归属是 Day 20–21；
因都直接建立在 Day10 的 attention / QKV / softmax 机制上，暂归于此，
待对应课程日创建时可迁移。英文术语保留，无雇主标识。

### Q1：帮我复习一下。传统 Megatron-SP：在 Attention 前后需要做 All-Gather 和 Reduce-Scatter，通信的数据量与全量 Activation 挂钩，在超长序列下，单卡要顶住全局的通信压力。

**A：** 三句基本准确，逐句给精确版与数字。记号： $s$ 序列长， $b$ microbatch，
 $h$ hidden 维， $t$ TP 卡数；上标 $s$ 表沿 sequence 切，上标 $h$ 表沿 hidden/head 切。

**纠正（重要）：** 第一版回答说"all-gather 的对象是 K 和 V，不包括 Q"——这是错的，
当场已纠正。按 Megatron-SP 论文 Equation (3)， $g$ （forward 做 all-gather）
作用在 **QKV 的输入 $Y$ **（LayerNorm 的输出）上。所以 Q、K、V 在每张卡上都是
全序列 `[s, b, h/t]` ——"每张卡拿到全量 K/V"效果没错，但机制是 gather 输入 $Y$
再各自做 QKV projection，不是直接 gather K/V。

**1. "Attention 前后 All-Gather / Reduce-Scatter" ✓**

Attention 后 reduce-scatter 发生在 output projection（row-parallel GEMM）之后，
MLP 块同理。原理：`all-reduce = reduce-scatter + all-gather`——纯 TP 里那个
AllReduce 被拆开，总通信量和纯 TP 一模一样。Megatron-SP 省的是 activation 显存
（LayerNorm/Dropout 的输入从每卡复制 `[s,h]` 降到 `[s/t,h]` ），不是通信量。

**2. "通信量与全量 Activation 挂钩" ✓**

数字： $N=10^6$ 、 $d=4096$ 、 $P=8$ 、fp16、MHA——每张卡每层 forward，
K/V all-gather 要收到 $2 \times N \times (d/P) \times 2\text{B} = 2\text{GB}$ ；
式子里 $N$ 是全量， $P$ 只除在 head 维。80 层即每卡每 forward `160GB` ，
反向再来一遍。第三方对比：Megatron-SP 每 worker 总通信量级 `10Nd` ，
DistFlashAttn 类方案 `3Nd` ——量级里都是全量 $N$ 。

**3. "单卡顶住全局通信压力"——基本对，精确化**

真正的痛点不是"通信量不随 $P$ 下降"（它随 $P$ 下降，正比于 $N/P$ ），而是两点：
（1）它是 blocking collective： $P$ 张卡到齐、每张卡收齐 $2Nd/P$ 之后，
attention 才能开始算第一个 block，没法像 Ring 那样边传边算；
 $N$ 增大时同步点和延迟按全局 $N$ 涨，加卡分摊不了这个"先收齐"。
（2）峰值 buffer：每张卡要同时摆下全序列的 K、V（各 `[N, d/P]` ），
 $N \to \infty$ 时是硬上限。

串联：这正是 Ring Attention（2026-09-06 Q2 已归档）要解决的问题——
每张卡每轮只跟邻居做 P2P 传 K/V block，传的时候手里有活干
（Day10 的 online softmax 天然支持流式更新 $m/\ell$ ）。

### Q2：Attention 后 reduce-scatter 发生在 output projection 这一步的数学

**A：** 输入：卡 $j$ 上 attention 本地算完（ $Y$ 已 gather 成全序列），
 $A_j \in \mathbb{R}^{s \times b \times (h/t)}$ ，权重行分片
 $W_j^{out} \in \mathbb{R}^{(h/t) \times h}$ 。本地 GEMM 得部分和
 $O_j = A_j W_j^{out} \in \mathbb{R}^{s \times b \times h}$ ——注意序列维是全量 $s$ ，
只在求和维（head 分片）上是部分的。真实输出应为 $O = \sum_j O_j$ 。

$\bar{g}$ 的 forward = 沿序列维 reduce-scatter：每张卡交出 $O_j$ ，
collective 先求和再把第 $i$ 个序列分片发给第 $i$ 张卡。
正确性（一行）： $O_i^s = \sum_j (A_j W_j^{out})|_{\text{shard } i} = (A W^{out})|_{\text{shard } i}$ ，正好是真实输出在第 $i$ 个分片上的值。

小例子： $t=2, s=4$ 。卡 1 交出 $O_1$ （4 行），卡 2 交出 $O_2$ （4 行）；
卡 1 拿走 `(O_1+O_2)` 的第 0–1 行，卡 2 拿走第 2–3 行。

### Q3：帮我汇一下 all gather reduce scatter

（用户接着说"回忆下"，同一轮。）

**A：** 符号： $t$ TP 卡数； $s$ 、 $b$ 、 $h$ 同上；上标 $s$ / $h$ 表切分维。

- **All-Gather：只拼接，不求和。** 每张卡手里有分片 $X_i$ ，
  之后每张卡都得到完整拼接 $X = [X_1; X_2; \dots; X_t]$ 。
- **Reduce-Scatter：先求和，再切分。** 每张卡手里有局部结果 $X_j$
  （通常是部分和），先跨卡求和 $\sum_j X_j$ ，再沿某个维度切开，
  第 $i$ 张卡只保留第 $i$ 片。
- **All-Reduce = Reduce-Scatter + All-Gather**：ring all-reduce 正好拆成这两步，
  所以这两个算子是 All-Reduce 的两个"半步"。

放回 Megatron-SP（ $t=2, s=4$ ）：初始 LayerNorm 输出按 sequence 分片，
每张卡 2 个 token 行，形状 `(2, b, h)` ；All-Gather 把 $Y$ 拼成全序列
`(4, b, h)` （注意 gather 对象是 $Y$ ，见 Q1 纠正）；各卡用自己的 QKV 权重分片
（column-parallel）做 projection，算自己负责的 heads，得 $A_j$ ；
output projection 权重是 row-parallel 分片，本地 GEMM 得部分和 $O_j$ ；
Reduce-Scatter 求和后按 sequence 切回，下游回到 sequence-sharded 区域。

### Q4：Megatron-SP 是啥？这是TP？

**A：** 不是 TP 本身，是**建在 TP 之上的一层补充**，全称 Sequence Parallelism
（Megatron 团队 2022 年论文），专门解决纯 TP 的一个浪费。

纯 TP 切权重（column-parallel、row-parallel），GEMM 部分每张卡只算一部分。
但 LayerNorm、Dropout、residual add 不依赖权重切分，输入输出形状是
`(s, b, h)` ，纯 TP 里每张卡完整复制一份。SP 把这些操作也沿 sequence 维切开，
每张卡只存 `s/t` 行——**TP 切的是权重/GEMM，SP 切的是 activation 的 sequence 维**，
两者叠加记作 TP+SP。

但 attention 和 MLP 的 GEMM 需要完整输入（attention 要看全序列，
GEMM 的输入 hidden 维要完整），所以 SP 引入两种布局区域，用 collective 转换：

```
sequence-sharded 区域：LayerNorm、residual、Dropout，每张卡 (s/t, b, h)
        │ All-Gather（拼成完整序列）
        ▼
head/tensor-parallel 区域：QKV GEMM、attention、output GEMM，每张卡 (s, b, h/t)
        │ Reduce-Scatter（求和后再按 sequence 切开）
        ▼
sequence-sharded 区域：Dropout、residual
```

一个 transformer layer 的 forward 有 2 次 All-Gather 加 2 次 Reduce-Scatter，
backward 互换，通信量理论上和纯 TP 的 4 次 All-Reduce 相同。
论文的原始动机之一：activation 变小之后，可以少做 activation recomputation。

### Q5：megaton主要切啥维度？

（"megaton"为用户原文，即 Megatron。）

**A：** 主要切 **hidden 维** $h$ ，两种切法：

1. **Column-parallel，切权重的列（输出维）**：QKV projection、MLP 第一层线性层。
   每张卡拿不同的列，算不同的输出通道；在 attention 里体现为每张卡负责一部分
   attention heads（head 维分片）。
2. **Row-parallel，切权重的行（输入维）**：output projection、MLP 第二层线性层。
   每张卡拿不同的行，各自算出全局输出的一个部分和，再 All-Reduce 跨卡求和。

关键是它**不切什么**：sequence 维 $s$ 和 batch 维 $b$ 在纯 TP 里每张卡完整复制。
这正是 Megatron-SP 要补上的。

### Q6：attention layer的数学？

**A：** 符号： $s$ 序列长； $b$ microbatch； $h$ hidden 维；
 $n$ attention head 数； $d = h/n$ 每 head 维；
 $X \in \mathbb{R}^{s \times b \times h}$ 为 attention layer 输入；
 $W^Q, W^K, W^V \in \mathbb{R}^{h \times h}$ 为 QKV 投影权重；
 $W^{out} \in \mathbb{R}^{h \times h}$ 为 output projection 权重。

七步：

1. **QKV projection**： $Q = XW^Q$ ， $K = XW^K$ ， $V = XW^V$ ，形状都是 `(s, b, h)` 。
2. **Split heads**：hidden 维拆成 $n$ 个 head，每个维度 $d$ ，得 `(s, b, n, d)` ；
   记第 $k$ 个 head 为 $Q_k, K_k, V_k \in \mathbb{R}^{s \times b \times d}$ 。
3. **Attention scores**：每 head 独立算 $S_k = Q_k K_k^T / \sqrt{d}$ ，形状 `(s, b, s)` ；
   $S_k[i,j]$ 是第 $i$ 个 token 对第 $j$ 个 token 的原始相关度；
   除以 $\sqrt{d}$ 是让点积方差不随 $d$ 增大而爆炸，否则 softmax 会 saturate。
4. **Softmax**：对 key 维（最后一个 $s$ ）做 softmax， $P_k$ 每行和为 1；
   decoder 加 causal mask： $j > i$ 的位置设为 $-\infty$ ，让 token 看不到未来。
5. **Weighted sum**： $A_k = P_k V_k$ ， $A_k[i] = \sum_j P_k[i,j]\, V_k[j]$ ，
   第 $i$ 个输出 token 是所有 value 向量的凸组合。
6. **Merge heads**： $n$ 个 head 拼回 `(s, b, h)` 。
7. **Output projection**： $O = A W^{out}$ ，形状 `(s, b, h)` 。

为什么 TP 能按 head 切：3–5 步里每个 head 只用自己的 $Q_k, K_k, V_k$ ，
head 之间完全独立，所以 TP 把 $n$ 个 heads 分给 $t$ 张卡后 attention 内部零通信；
通信只发生在最后一步（row-parallel 的部分和合并）。

### Q7：soft Max（Q K^T）之后出来的是一个 1 * 3 的向量？

**A：** 看切的是哪一层。单 head、 $s=3$ 、忽略 batch：
 $Q$ 、 $K$ 是 `(3, 2)` ； $QK^T$ 是 `(3, 3)` ，第 $i$ 行是第 $i$ 个 token
对全部 3 个 token 的 scores；softmax 后 `(3, 3)` 的**每一行**都是一个
`1 x 3` 的概率向量，和为 1。你看到的 `[0.25, 0.25, 0.50]` 就是第 3 行——
第 3 个 token 的 attention 分布：50% 给自己，25% 给 token 1，25% 给 token 2。
这行 `1 x 3` 乘上 $V$ （ `(3, 2)` ）得到 `1 x 2` ，正好是第 3 个 token 的输出行。
每一行独立做一次"拿分布当权重、拌 value"，3 行拼起来就是完整的 `(3, 2)` 输出。
加上 batch 和多 head，完整形状是 `(s, b, n, s)` ：每个 (batch, head) 切片里
都是一张 `s x s` 的表，每行是一个 `1 x s` 的分布。

### Q8：不对不对  告诉我 QK^T 是什么 softmax QK^T怎么算的

**A：** 退回最机械的一层，一个数一个数地算。

**QK^T 是什么。** $Q$ 是 `(3, 2)` ， $K$ 是 `(3, 2)` 。
记 $q_i$ 为 $Q$ 的第 $i$ 行（第 $i$ 个 token 的 query 向量），
 $k_j$ 为 $K$ 的第 $j$ 行（第 $j$ 个 token 的 key 向量）。
$(QK^T)[i,j] = q_i \cdot k_j = \sum_p q_i[p]\, k_j[p]$ ——
第 $i$ 个 token 的 query 和第 $j$ 个 token 的 key 有多"对得上"，数值越大越相关。

例 $Q = K =$ `[[1,0],[0,1],[1,1]]` ：
第 1 行 `[1, 0, 1]` ，第 2 行 `[0, 1, 1]` ，第 3 行 `[1, 1, 2]` ；
再除以 $\sqrt{2} \approx 1.414$ ，得
`S = [[0.707, 0, 0.707],[0, 0.707, 0.707],[0.707, 0.707, 1.414]]` 。

**softmax 怎么算。** 逐行，行与行互不干扰。固定第 $i$ 行：
$P[i,j] = e^{S[i,j]} / \sum_m e^{S[i,m]}$ ，三步：取指数 → 求和 → 相除。
拿第 3 行 `[0.707, 0.707, 1.414]` 演示：
取指数得 `2.028, 2.028, 4.113` ，求和得 `8.169` ，
相除得 `[0.248, 0.248, 0.504]` 。第 1、2 行同理得
`[0.401, 0.198, 0.401]` 和 `[0.198, 0.401, 0.401]` 。

整件事就是：先用点积给每对 token 打分，再把每个 token 那一行分数用
softmax 压成"注意力分配比例"。

### Q9：对哦 实际中Q一般是不是s= 1啊？告诉我下一个token怎么算的

**A：** 对，直觉正确。分两个阶段：

- **Prefill**：第一次处理 prompt，Q 的序列长度 = 整个 prompt 长度，一次性并行算完。
- **Decode**：逐个生成新 token 时，每步的 Q 只有**当前这一个新 token**，
  所以 Q 的序列长度确实是 1。而 K 和 V 从 **KV cache** 里拿，
  包含之前所有 token，外加当前这个新算出来的。

KV cache 就是把之前每步算好的 $k_j, v_j$ 存下来，避免重复计算；
每步只算新 token 的 $q$ 、 $k$ 、 $v$ ，再把 $k$ 、 $v$ 追加进 cache。

**下一个 token 怎么算（接 Q8 的例子）。**
假设已生成 token 1、2，cache 里有 $k_1, k_2, v_1, v_2$ ；
现算第 3 个 token：

1. 新 token 的 embedding 过 QKV projection 得 $q_3, k_3, v_3$ ，各 `(1, 2)` 。
2. $K =$ `[k_1;k_2;k_3]` 是 `(3, 2)` ， $V$ 同理； $k_1, k_2$ 从 cache 直接拿，不重算。
3. scores $= q_3 K^T / \sqrt{d}$ ： `(1, 2)` 乘 `(2, 3)` 得 `(1, 3)` ，**只有一行**——
   正是 Q8 矩阵的第 3 行 `[0.707, 0.707, 1.414]` ，decode 时这一行就是全部。
4. softmax → `[0.248, 0.248, 0.504]` ，一个 `1 x 3` 的分布。
5. 输出 $= 0.248\,v_1 + 0.248\,v_2 + 0.504\,v_3$ ，形状 `(1, 2)` 。

之后拼 head → output projection → residual/LayerNorm/MLP 走完所有 layer，
得 `(1, h)` ，乘 unembedding `(h, vocab)` 得 logits，softmax 后采样出下一个 token；
把它的 $k$ 、 $v$ 存进 cache，循环往复。

所以 decode 的每一步本质上就是：拿 1 个 query 去查整张 $m \times m$ 表里
属于自己的那一行。
