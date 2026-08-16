# 04 Prompt vs Context vs KV — 三条压缩路线对比

Long context 时有三件不同的事，大家都叫 compression，tradeoff 完全不同。

## 1. Prompt Compression (LLMLingua 等)

> Paper: Prompt Compression in the Wild [arxiv:2604.02985]

- **目标**: `20K prompt → 8K` 喂 LLM，减少 prefill FLOPs
- **方法**: Token klasifikátor / entropy filter 删不重要 token，保留 question 相关 span
- **Gain**: 报告 up to 18% end-to-end speedup 当窗口匹配时，memory offload 显著
- **Failure**: Over-compress 把 question 核心删了 → accuracy cliff

公式视角：

$$
\text{speedup} \approx \frac{T_{prefill}(L)}{T_{prefill}(r L)} \approx \frac{1}{r} \text{ for large } L
$$

其中 $r = 0.4$ compression ratio，理论 2.5x，但有 decompression overhead + quality loss，所以实测 1.18x。

**When to use**: Single-turn QA / RAG, short horizon, 高 throughput serving。

## 2. Context Compression (Agent Memory)

> Factory Dec 2025, Hermes compressor

- **目标**: `Session 200K → 20K working memory`，跨多轮存活
- **方法**: Summarization + extractive tool trace keep + structured artifact log
- **Gain**: 使 long-horizon 任务能继续跑，不 OOM
- **Failure**: 丢 instruction / artifact trail / continuity → agent loop

关键 difference：此压缩是 **stateful** 的，多次调用，每次都有损累积误差 `ε_t` :

$$
M_{t+1} = compress(M_t \cup \Delta_t) ;\; \|M_{t+1} - ideal(M)\| \leq \|M_t - ideal\| + \epsilon_{compress}
$$

多次压缩后高方差问题和 GLM-5.2 放弃 GRPO 理由同构：长度方差大、group 不齐。

**When to use**: SWE-agent, multi-turn tool, 需要跨 session 记忆。

**Latency**: summarizer 本身是一次 LLM call ~2-3s，但避免了 OOM 重置 session。

## 3. KV-cache Compression (TurboQuant / LeanKV)

- **目标**: `KV [N*H] → [N/2*H]`，管 GPU HBM
- **方法**: Quantization 4bit, eviction (H2O), low-rank projection
- **Gain**: 能把 `max_num_seqs` 从 32 → 128，大幅提升并发
- **Failure**: 精度掉 → attention drift → long context 后部 answer 逐渐发散

数学：量化误差 $\|K - \hat K\|_F \leq \epsilon$，但 attention softmax 放大：

$$
|\text{Attn}(Q,K) - \text{Attn}(Q,\hat K)| \leq O(\exp(\epsilon))
$$

所以 KV quant 在长 context 下阈值更敏感。

**When to use**: vLLM serving, `gpu_memory_utilization=0.95` 时提升吞吐。

## 对比表

| | Prompt Compres. | Context Compres. | KV-cache Compres. |
|---|---|---|---|
| **Line** | 输入压缩 | 记忆压缩 | 状态压缩 |
| **ratio** | 2-5x | 5-10x | 2-4x |
| **latency save** | 10-18% e2e【2604.02985】 | 避免重新开始 (∞) | 提升 throughput |
| **quality risk** | 删 question 细节 | 丢 artifact/consistency | drift 后段答案 |
| **eval** | Squad EM | 6-dim probe judge (Factory) | perplexity + downstream |
| **Infra** | pre-processor | agent loop middleware | inference kernel |

## 取舍 Guidance (for our eval-compression track)

> Long-horizon agent = 必须用 2 + 可选 1 & 3

- 只做 KV-cache 优化不直接解决 GLM-5.2 长任务 GRPO 崩问题 — 崩在 **sub-trajectory 可比性**
- Prompt compression 侧重 single-turn，和 agent memory 的多次累积误差不是一回事
- Eval 时应 separate: measure **compression ratio**, **decompression / re-ask latency**, **probe recall** 三个数，不要混成一个榜

> 对应 vllm-rollout 理解：当我们 `max_model_len=128K`，开 KV quant 可以多塞并发，但如果 context compressor 已把 history 弄丢，越多并发 = 越多错答并发。
