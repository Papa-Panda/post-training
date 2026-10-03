# EP21 — SGLang v0.2：面向 LLM 和 VLM 的快速、高效通用服务引擎

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP21-sglang-v02.html

> **主旨引文**（论文 arXiv:2312.07104，摘要）：「Experiments show that SGLang achieves up to $6.4\times$ higher throughput compared to state-of-the-art inference systems.」—— 核心机制：把「多调用、结构化」的 LM 程序当成一级公民，用前端语言 + 运行时协同复用 KV 缓存、加速受约束解码。

## 元信息

- 期号：EP21（官网期号 21；青稞Talk 第 21 期）
- 标题：SGLang v0.2：面向 LLM 和 VLM 的快速、高效通用服务引擎
- 讲者：盛颖（Ying Sheng；Databricks Mosaic Research 研究科学家，斯坦福大学博士；节目页时任介绍）
- 讲者来源：[节目页 qingkeai.online/blog/1npUmMSk](https://qingkeai.online/blog/1npUmMSk)（嘉宾介绍为公开署名；论文作者表同名作者 Ying Sheng 列于末位）
- 观看/提炼日期：2026-10-02
- 直播时间：2024-09-03 11:00–12:00（节目页）
- BV：official-only（官网有预告，B站合集无对应视频，台账 `episode-index.csv` 著录）
- 相关论文：[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（arXiv:2312.07104，v1 2023-12-12 提交，v2 2024-06-06；本期于 2024-09-03 直播，数字采 v2 正文）
- 相关代码：[github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)（节目页成果链接）
- 提炼方式：**无视频、无字幕，基于对应论文/技术报告还原，非逐字稿。** 数字、结论均标注论文出处；讲演题名中的「v0.2」是运行时版本沿革，论文只给系统本体，现场讲到的 v0.2 增量无法从本文复原，见文末未确证清单。

## 一句话总结

LLM 的用法已经从「单次聊天」变成「多调用、可分支、带结构化输出的程序」，而现成推理引擎对程序结构一无所知：SGLang 把程序结构暴露给运行时，用 RadixAttention 跨调用复用 KV 缓存、压缩有限状态机加速 JSON 等约束解码，在 LLaMA/Mixtral/LLaVA 等模型上做到最高 $6.4\times$ 吞吐、最高 $3.7\times$ 延迟降低（相对 Guidance、vLLM、LMQL 等基线，论文口径）。

## 核心

1. **背景/问题**：单次生成只是 chat 的形态；真实应用是 LM 程序（Language Model Programs）：agent 控制流、few-shot、self-consistency、tree-of-thought、RAG 流水线，本质都是多次 dependent 的生成调用拼控制流与结构化输入输出。单次视角下可复用的 KV 缓存前缀（如 few-shot 示例、共享 system prompt、agent 模板、多轮历史）在请求边界处被丢弃，受约束解码（JSON schema 等）也只能逐 token 踩刹车。论文的判定：引擎「workload-agnostic」是通用性的代价，但也让每个具体 workload 都在重复付钱。
2. **方法/设计（语言与运行时协同）**：前端是嵌入 Python 的 DSL，提供生成（ `gen` 、带正则约束的 `gen(regex=...)` ）、选择（ `select` ）、并行控制（ `fork` / `join` ）与多模态（ `image` / `video` ）原语，prompt 状态按流处理；解释器把原语异步提交、程序内的并行自然暴露（也可编译为计算图）。同一程序手写 OpenAI 风格接口要多 $2.1\times$ 代码量（论文 §2）。运行时三项关键优化：(a) **RadixAttention**——把所有请求的 KV 缓存放进一棵 radix tree 按前缀索引，LRU 淘汰 + cache-aware scheduling，让跨调用、跨实例、多级共享（含图像 token 的哈希键）都自动命中，命中率在本组 benchmark 上 50%–99%，调度达到理论最优命中率的 96% 左右（论文 §6.2）；无复用场景（ShareGPT）管理开销 <0.3%（论文 §6.3）；(b) **compressed finite state machine**——分析正则/语法约束，把可一次解码的多 token 路径压成单步，JSON 解码吞吐 + $1.6\times$ ，且状态机预处理须跨请求复用，否则反而慢 $2.4\times$ （论文 §6.3）；(c) **API speculative execution**——对 OpenAI 这类 API-only 模型，用小模型/启发式预生成后续可能字段、命中再校验，GPT-3.5 三字段抽取场景输入 token 成本约降为 1/3（论文 §6.2）。
3. **实验/实战**：工作负载包括 5-shot MMLU、20-shot HellaSwag、ReAct agent 与 generative agents 轨迹回放、Tree-of-Thought 解 GSM-8K、Skeleton-of-Thought 提示生成、branch-solve-merge 式 LLM 裁判、JSON 解码、4 轮多轮对话、DSPy 官方 RAG 流水线；模型 Llama-7B/70B、Mixtral 8 $\times$ 7B（tensor parallel）、LLaVA-v1.5-7B（图像）、LLaVA-NeXT-34B（视频），硬件 A10G/A100；基线为 Guidance、vLLM、LMQL。端到端：吞吐最高 $6.4\times$ 、延迟最高 $3.7\times$ （论文 §6.2）；多模态：LLaVA-v1.5-7B 0.18 → 1.15 image/s、LLaVA-NeXT-34B 0.02 → 0.10 frame/s（对比作者原生 HF 实现，论文 Table 2）。同一图像上的多问题共享图像 token 的 KV，是多模态侧的主要命中来源。
4. **生产与结论**（论文口径）：SGLang 已上线 Chatbot Arena 服务开源模型，一个月观测：LLaVA-NeXT-34B 的 RadixAttention 命中率 52.4%、Vicuna-33B 74.1%，Vicuna-33B 首 token 延迟平均降 $1.7\times$ （论文 §6.2）。结论：LM 程序的结构（共享前缀、分支、约束语法）是可系统化榨取的运行时信号，语言与运行时必须 co-design——消融里关掉前端 fork hint 或并行暴露都会掉速（论文 §6.3）。

## 关键数字

| 指标 | 数字 | 来源 |
|---|---|---|
| 端到端吞吐提升（相对 Guidance/vLLM/LMQL 等基线） | 最高 $6.4\times$ | 论文摘要、§6.2 |
| 延迟降低 | 最高 $3.7\times$ | 论文 §6.2 |
| 基准组 KV 命中率 / 调度达成率 | 50%–99% / 约最优 96% | 论文 §6.2 |
| LLaVA-v1.5-7B 吞吐 | 0.18 → 1.15 image/s | 论文 Table 2（基线为作者原生实现） |
| LLaVA-NeXT-34B 吞吐 | 0.02 → 0.10 frame/s | 论文 Table 2 |
| Chatbot Arena 线上命中率 | LLaVA-NeXT-34B 52.4%、Vicuna-33B 74.1% | 论文 §6.2（上线一个月观测） |
| 首 token 延迟降低（Vicuna-33B） | 平均 $1.7\times$ | 论文 §6.2 |
| 压缩状态机对 JSON 解码吞吐 | + $1.6\times$ ；状态机不跨请求复用则慢 $2.4\times$ | 论文 §6.3 |
| RadixAttention 管理开销（无复用时） | <0.3% | 论文 §6.3（ShareGPT） |
| API speculative execution 输入成本 | 约降至 1/3（GPT-3.5 三字段抽取） | 论文 §6.2 |
| 等价程序代码量（vs OpenAI 风格接口） | 1 / $2.1\times$ | 论文 §2 |

## 可迁移

- **把「程序级结构」当运行时信号的通式**：共享前缀、并行分支、输出约束是跨调用的全局事实；落到后训练/推理基建就是三件事——(1) KV/中间结果按「前缀树」而非「单请求」组织与淘汰，rollout 侧的采样树（tree search、best-of-n、beam、agent 多分支）先验天然有共享前缀，按树索引能把 prefill 摊薄；(2) 约束（JSON schema、动作空间、格式校验）在解码器里前置为状态机/掩码预编译，且必须跨请求复用预处理结果（论文给了反例：每请求重建反而慢 $2.4\times$ ）；(3) 调度要感知缓存亲和性（cache-aware scheduling 逼近最优），而不是纯 FIFO——批推理、eval harness、多租户 serving 排队都适用。
- **Infra 视角**：本期给「serving 系统优化」立了量化基线——先测命中率分布（50%–99% 才是可榨空间），再谈结构优化；无命中场景的优化必须证明开销可忽略（<0.3%）才配默认开启。评测口径上，论文用 «programs/s» 而非 tokens/s 衡量 LM 程序吞吐，对 agent/裁判/采样链路的 eval harness 选指标有直接借鉴：端到端任务吞吐比 token 吞吐更接近成本。

## 疑问 / 下一步

- 讲演题名是「v0.2」，而论文止于系统本体：v0.2 运行时相对论文版的增量（例如更完整的 RadixAttention、多模态部署细节、当时路线图）讲了多少，本文无法确认，想补需另找 v0.2 发布说明或当期回放（无 B站视频）。
- RadixAttention 的跨机/分布式 KV 共享在论文里只作展望（related work 提及支持 distributed cases）；放到大批量 rollout 的场景（训练内推理），前缀命中率分布与调度收益需要按真实轨迹重测。

## 原文金句（论文原话，非讲者语录）

> "The core idea is to systematically exploit the multi-call structure in LM programs for efficient execution."

> "This complexity significantly reduces the readability of even simple programs ... Secondly and importantly, executing LM programs is inefficient due to redundant computation and memory usage."

## 未确证项

1. **无视频、无字幕**：本期为 official-only，B站合集无对应视频；全文基于论文 arXiv:2312.07104 v2 与节目页，现场讲解顺序、v0.2 的具体增量、问答内容均未覆盖。
2. **讲者身份**：以节目页嘉宾介绍为准（盛颖，Databricks Mosaic Research 研究科学家、斯坦福博士）；论文作者表末位 Ying Sheng 与之同名、为论文共同作者，但「讲演即论文作者亲述」未拿到节目页外的第三方佐证，按节目页署名著录。
3. **版本差**：论文 v1（2023-12）→ v2（2024-06）之间正文有修订，直播（2024-09-03）时口头数字可能引用 v2 之后的代码版本；文中数字一律按 arXiv v2 标注出处。
4. **节目页时间**：节目页只写直播时段 2024-09-03 11:00–12:00，未见回放链接，未确证。
