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
