# EP123 — DeepSeek V4 模型在 SGLang 中的系统级优化与全栈适配

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP123-deepseek-v4-sglang.html

> 「我觉得今天最重要的两个技术点，首先就是 ShadowRadix……另外一个我可能会挑 CP，这是针对 100 万长上下文的优化。」——讲者在 Q&A 收尾时的自评

## 元信息

- 期号：123
- 标题：DeepSeek V4 模型在 SGLang 中的系统级优化与全栈适配
- BV：BV12YRdBzEM2
- 时长：约 60 分钟（讲授约 31 分钟 + Q&A 约 29 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：张柏洲（SGLang 核心开发者/维护者，DeepSeek V4 day-0 支持成员；就职单位字幕作「REDARK」，疑为识别误差，未确证）
- 相关论文：DeepSeek V4（官方技术报告，讲授以其架构为准）
- 相关代码：SGLang（V4 支持 PR 已于讲授前一日合入主分支）；Cookbook、V4 支持 roadmap 与 SGLang 技术博客在讲授结尾以二维码给出
- B站链接：https://www.bilibili.com/video/BV12YRdBzEM2/
- 官网期号：EP123（官网预告页未能打开，故不附链接）
- 字幕原文存档：本地 `transcripts/EP123.txt`（1215 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（经清洗，专名可能有识别误差）提炼。字幕中的常见识别误差包括：SGLang 作「H浪 / Sg long / edge line」、ShadowRadix 作「shadow red / redis」、HiSparse 作「high bus / high sparts」、DeepSeek 作「deep sick / GIPSKV4」、DeepGEMM 作「deep jam」、DeepEP 作「DPP」、Megatron/Miles 一类训练框架作「max / mouse」（未能确证，见正文标注）、Triton/TileLang/CUDA 作「CHATTEN / TALONG / 枯大」、EAGLE 作「ego」、Medusa 作「MEDA」、SemiAnalysis 作「SEMINAIS」、InferenceX 作「INFANEX」、Marlin 作「marin」，均按上下文还原。讲者非 RL 专家，Q&A 中涉及 RL 细节处他多次明确表示需问社区里做 RL 的同学，以下照实记录。

## 一句话总结

DeepSeek V4 把 V3 的单一 MLA 换成 SWA + CSA + HCA 三路混合注意力，SGLang 的 day-0 适配围绕「三种 KV 布局怎么统一管理」展开：新组件 ShadowRadix 用虚拟位置页表把三路 KV 池映射到同一 RadixTree，再叠加投机采样全图化、FlashCompressor/Lightning Top-K/Mega-MoE 内核优化、CP/EP/PD 分离等并行支持，以及后训练侧的 router/indexer replay 稳定性方案；InferenceX（8K 入 / 1K 出）上开 MTP 可达 180 token/秒/用户，GB300 NVL72 高吞吐可达约 11500 token/秒/GPU。

## 核心

### V4 的适配难点：从一种注意力到三种

V3 只有 MLA 一种注意力，V4 的 attention 变成三部分（字幕 02:44 起）：

- **SWA**（Sliding Window Attention）：每个 token 只看前面 128 个 token；
- **CSA**（Compressed Sparse Attention）：先做 C4 压缩（每 4 个 token 压成 1 个，序列 4000 → KV 长度 1000，字幕 03:09），再像 DSA 那样用 indexer 做 sparse top-k 挑选；
- **HCA**（Highly Compressed Attention）：相邻 128 个 token 压成 1 个，没有 top-k，是 dense 的（字幕 03:50）。

三种注意力的 KV 规模、访问模式完全不同，原有 RadixTree 前缀缓存无法直接套用——这是后面所有系统工作的起点。

### ShadowRadix：用虚拟位置页表统一三路 KV

ShadowRadix 是在 RadixTree 上扩展出的新组件（Q&A 中确认是 SGLang 全新组件、专为 V4 做，其他模型暂不支持，字幕 34:09）。做法类似页表：维护一个 virtual position 到物理 KV 的映射，page size 为 256（字幕 04:59）。三个 shadow 池分别管理：

- **Shadow A 管 SWA**：用 tombstone 机制回收已经滑出窗口的久远 token；
- **Shadow B 管 CSA 池**：物理长度为 virtual 长度 ÷4；
- **Shadow C 管 HCA 池**：物理长度为 virtual 长度 ÷128；
- 另有 ring buffer 存 compressor 的中间状态。

这样前缀匹配、淘汰、多路 KV 的索引都收敛到同一套虚拟地址上。Q&A 里讲者澄清了一个常见误解：SWA 的 KV cache 实际比 128 大，因为前缀匹配希望保留更多 SWA KV，让后来的序列更容易 match 到前面序列的前缀（字幕 33:12）；它能做前缀缓存（字幕 48:34）。三路 KV 的相对大小差异很大：CSA 与 HCA 分别是 4:1 与 128:1 压缩，未压缩的 SWA 反而是其中最大的一个。

### 投机采样：MTP metadata 全图化 + CPU/GPU 全异步 overlap

V4 自带 MTP layer（只含 SWA 一路注意力）。SGLang 侧的优化是把投机采样的 metadata 准备全部放进 CUDA graph，并让 CPU/GPU 异步调度完全 overlap。给出的 benchmark 条件是 TP8、batch=1、MTP accept length 取 3，对比 4K 与 90 万（900K）上下文（字幕 09:14–09:55）：从 4K 到 900K，decode 吞吐下降较小，长上下文下投机采样依然成立。

Q&A 中补了两条判断：其一，投机采样本质是访存优化（读一次 KV 验证多个 token），batch 大到 512/1024 时 GPU 变为计算瓶颈、访存不再是瓶颈，效果明显变差；小 batch（4/8/16）时 MTP 效果仍然很好（字幕 44:16、49:19、54:05）。其二，MTP 只是投机采样的一种算法，EAGLE、DFlash、Medusa 等范式仍有价值：模型自带 MTP 未必在所有数据集上都训练充分，社区为流行模型单独训的 EAGLE head（如 EAGLE-3）可能更快（字幕 34:38、52:26）。

### KV offloading：HiSparse 的 day-0 取舍

HiCache 在稀疏模型上的扩展叫 HiSparse。Day-0 的做法是只 offload CSA 池到 CPU 内存，因为 SWA/HCA 池太小、留在 GPU 即可；未来会支持 SWA/HCA 的 offload。这条与并行部分呼应：PD 分离与 ShadowRadix 结合，按 virtual index 传输 KV，从 P 节点以配置的方式传到 D 节点（字幕 20:37）。

### 内核三件套

1. **FlashCompressor**：把 compressor 里的 softmax、bias、scale 等小算子融合，把 HBM 读写从 5 次降到 2 次（字幕 12:43）。
2. **Lightning Top-K**：把序列分片到多个 CTA 做 histogram，并利用 cluster 内 SM 间通信做 reduce；1M 上下文、batch=1 时 top-k 耗时从约 100μs 降到约 15μs（字幕 13:32、14:27）。这个 kernel 用 CUDA 实现——讲者在 Q&A 中给出口径：Triton 迭代最快但优化力度不够，CUDA 开发最慢但能做最细的优化，TileLang 在中间；开发中三种语言都用了，特殊优化（如 top-k）用 CUDA（字幕 37:44–38:35）。
3. **Mega-MoE**：DeepSeek 官方的大融合 kernel，把 dispatch / compute / combine 全 overlap 进一个 kernel，SGLang 侧调用 DeepGEMM 实现。约束很硬：只支持 FP8 activation × FP4 MoE 权重，当前只支持 SM100（Blackwell），未来考虑 SM90（字幕 16:23–16:50）。Q&A 中讲者明确它不是 DeepGEMM+DeepEP 的平替：Mega-MoE 只针对 V4、只能在 NVLink 相连的 NVIDIA B 系列卡上跑，跨机 RDMA 这类场景仍需 DeepEP 方案；且 Mega-MoE 不再走 DeepEP，它自己管理 all-to-all（字幕 41:22–41:59、50:35）。关于「会不会破坏时序」：token 之间互相独立，batch 可切成多块分别 overlap，不破坏单个 token 的时序关系（字幕 39:30）。

### 并行：DP/TP/CP/EP 与多 stream 注意力

- **CP（context parallel）是这期重点**。长上下文下 CP 对延迟与 TTFT 帮助很大。具体做法朴素：每张卡只持有整个序列的一部分，index 等计算被均摊；做 attention 之前把需要的 KV cache（CSA、HCA、SWA 三路都）gather 过来——cache 本身不大，gather 开销小（字幕 17:52–18:44）。Q&A 中坦承 gather 目前还没有做 overlap，因为 gather 前后数据都有依赖、不好 overlap；未来考虑 ring attention 式的做法，它与 DSA 的 indexer 是两个独立层面的事（字幕 40:10、53:38）。CP 的 OOM 风险真实存在：缓解靠降并发、调小 KV cache 容量；当前 CP 主要针对小并发场景，大并发下的 OOM 还没碰到（字幕 42:22–43:09）。
- **TP 与 FlashMLA**：DeepSeek 官方 FlashMLA 库主要为 DP attention 设计（每卡 head 完整），开 TP 切 head 后先 pad 到所需 head 数——讲者自称这是「比较朴素的实现」，未来有优化空间。也澄清了命名误会：V4 不用 MLA，但 FlashMLA 库里装了很多 MLA 以外的 kernel，V4 用的也在里面（字幕 19:00、40:56）。
- **EP**：与 V3 差别不大，DeepEP all-to-all + DeepGEMM；large-scale EP 主要在 GB200/GB300 NVL72 上做（每卡 NVLink 互连、all-to-all 快），吞吐提升很大。另多了 Mega-MoE 的可选支持。
- **Decode 多 stream 并行**：decode 时 SM 用不满，SWA/CSA/HCA 三路注意力可放到不同 stream 上并行。level1 把 KV projection+store（SWA 的 KV 计算）、compressor（CSA/HCA 的 KV 计算）、indexer（只针对 CSA）拆开并行，main stream 上的 Q 等计算与它们也没有依赖；level2 里 indexer 内部同样有可并行处，用 CUDA event 控制各算子等自己需要的数据（字幕 21:00–22:34）。

选型判据（Q&A）：先看硬件能不能装下（单卡装得下就没必要开 TP），再看场景——高并发用 DP，低延迟用 TP 或 CP；输入不长时 TP 更好（CP 有通信开销），90 万到 100 万这种超长输入 CP 会好很多；64K 卡在中间地带，讲者猜测开 CP 划算但需实测（字幕 36:03、44:54、51:39）。H100 上跨节点与 CP 已有规划：H100 容量小，需要 PD 分离 + 流水线并行配合（字幕 45:44）。

### RL / 后训练的 day-0 支持

SGLang 对 V4 后训练做了 day-0 支持（字幕 22:42 起）：

- **稳定性**：router replay + indexer top-k replay——把路由结果与 indexer 的 top-k 结果记录下来重放，增强训练稳定性；
- **并行齐全**：DP/TP/EP/PP/CP 都有支持（原文「E p p p c p」按上下文还原）；
- **训练框架侧补反向算子**：训练框架比推理框架多一个反向，用 TileLang 支持了很多反向算子（字幕作「max / mouse」的训练框架，未能确证具体所指）；
- **混合精度**：rollout 用 FP8、training 用 BF16，因为 rollout 占训练的大部分时间，这样能加快整体效率（字幕 23:59–24:16）；
- 整个训练流程在 Hopper、Blackwell、Grace Blackwell 等多代 NVIDIA GPU 上都验证过；给出的 RL 训练结果显示随 step 增加 reward 与 AIME 分数有一定提高，证明训练可行（字幕 24:20–24:49）。

对「roll out 与 training 精度不同、是否要在 training 时重算 logits 否则训推不一致」的问题，讲者明说自己不是 RL 这块的专家，请提问者进社区问做 RL 的同学（字幕 43:37）。他给出的总判断是：对 RL 训练而言 rollout 是瓶颈，所以推理框架的优化能很好地提升训练大模型的效率（字幕 49:41）。

### 发布后性能：InferenceX 上的 Pareto 前沿

模型发布后约一两周集中优化的结果用 InferenceX（SemiAnalysis 的公共 benchmark）展示：8K 输入 / 1K 输出，横轴 interactivity（每用户 token/秒）、纵轴吞吐（token/秒/GPU）的 Pareto 曲线，覆盖 B200、B300（开/不开 MTP）、GB300 NVL72（字幕 25:06–26:41）。两个读数：开 MTP 时 interactivity 可达 180 token/秒（在线场景）；GB300 NVL72 用一点 interactivity 换吞吐，可达约 11500 token/秒/GPU（离线场景）（字幕 26:56、27:18）。

进度与计划：讲授前一天，V4 的支持 PR 已合入 SGLang 主分支（字幕 27:44）。未来工作包括 pipeline parallelism 与 PD 分离（在 H20/H200 上更有用）、FP4 indexer（DeepGEMM 官方库的一部分，day-0 时因支持复杂而推后）、HiSparse 扩展到 SWA 池 offload、DeepEP v2，以及更多硬件：SM120 已有 PR 在 review（目标是让 V4 能跑在 RTX 5090 这类更便宜的卡上），还在推进 SM80（字幕 28:13–29:25）。Q&A 补充：H 系列其实已经能跑 FP4 权重——Marlin 的 W4A8 MoE 算子已支持，H200 单机 8 卡可以用 SGLang 起 DeepSeek V4 Pro（字幕 55:33）；SM120 适配完成后，理论上 2–4 张 5090 就能跑 V4 Flash（字幕 52:02）。5090 支持此前较少是人力问题，之前更专注数据中心显卡，后续会逐步加大消费级显卡支持（字幕 59:56）。

### Q&A 其他要点

- **V4 为什么缓存命中率高**？讲者给了三个可能：CSA/HCA 的压缩让 KV 更容易命中；训练时可能对高缓存命中率做过针对性训练；DeepSeek API 侧有多级缓存，能把 KV cache 扩到很大尺寸（字幕 31:55–32:49）。SWA 的 KV 也能做前缀缓存（字幕 48:34）。
- **CSA、HCA 都是 DSA 的变种吗**？CSA 与 DSA 非常像（都有 indexer 结构）；HCA 不是，它没有任何 indexer，只是纯粹的压缩（字幕 37:10）。
- **与 vLLM 在 V4 上的性能对比**？讲者不下结论：两个框架优化侧重点不同，要看具体场景自己测（字幕 36:48）。
- **Agent 场景下并行策略能否动态调整**？单个 server 运行中动态改并行策略很难（CUDA graph 要重新 capture、metadata 要变）；切换到不同服务则是合理的，处理好 KV 传输即可（字幕 56:51–57:21）。启动后动态调整与跨服务切换是两回事。
- **并行设计要避免 all-reduce 吗**？因果反过来：并行策略决定通信算子，必须用 TP 的场景里 all-reduce 无法避免（字幕 54:57）。
- **新手如何参与 SGLang**？先补 Transformer 原理与 AI infra（分布式训练、通信、并行计算），然后去 GitHub 从 bug 和季度 roadmap 找任务，联系对应负责人；急的话去 Slack 找开发者讨论。学习路径上他推荐 Stanford CS336，同时笑言现在「不懂什么就问模型」（字幕 47:45–48:24、58:56）。没有 GPU 也能参与 CPU 可验证的问题，或租 Colab 等便宜 GPU（字幕 55:59）。
- TurboQuant 的支持已有相关 issue，但暂未排期（字幕 59:31）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| SWA 窗口大小 | — | 每 token 只看前 128 个 | 字幕 02:44 |
| CSA 压缩比 | 序列 4000 | KV 长度 1000（C4 压缩） | 字幕 03:09 |
| HCA 压缩比 | — | 相邻 128 token 压 1，无 top-k | 字幕 03:50 |
| ShadowRadix page size | — | 256（virtual position 页表映射） | 字幕 04:59 |
| 投机采样 benchmark | 4K vs 900K 上下文，TP8、batch=1 | accept length=3，4K→900K decode 吞吐下降较小 | 字幕 09:14–09:55 |
| FlashCompressor HBM 读写 | 融合前 | 5 次 → 2 次 | 字幕 12:43 |
| Lightning Top-K 耗时 | 1M 上下文、batch=1 | 约 100μs → 约 15μs | 字幕 13:32、14:27 |
| Mega-MoE 精度/硬件约束 | — | FP8 激活 × FP4 权重，仅 SM100（未来 SM90） | 字幕 16:23–16:50 |
| InferenceX interactivity | B300 开 MTP，8K 入/1K 出 | 180 token/秒/用户 | 字幕 26:56 |
| InferenceX 吞吐 | GB300 NVL72，8K 入/1K 出 | 约 11500 token/秒/GPU | 字幕 27:18 |
| 投机采样失效点 | batch 512/1024 | 大 batch 效果明显变差（计算瓶颈） | 字幕 44:19 |
| MTP 仍有效的小 batch | batch 4/8/16 | 效果仍然很好 | 字幕 49:19 |
| FP4 权重在 H 卡 | Marlin W4A8 MoE 算子 | H200 单机 8 卡可起 V4 Pro | 字幕 55:33–55:49 |
| 消费级卡跑 V4 Flash | SM120 适配完成后（预期） | 2–4 张 RTX 5090 | 字幕 52:02 |

## 可迁移

- **虚拟位置页表是多路异构 KV 的通用解法**：当一个模型同时有滑动窗口、压缩稀疏、高度压缩三类 KV 时，不要为每类各维护一套前缀缓存——用统一的 virtual position + 每池一个压缩比映射，把索引层收敛成一套。这与 verl/SGLang 在 LLM 侧用统一 slot 管理 KV 的思路同源，可迁移到任何多分辨率 KV 的推理系统。
- **投机采样的收益判据是访存/计算瓶颈之分**：batch 小、访存瓶颈时收益大；batch 大到计算瓶颈时收益塌陷。给 RL rollout 配投机采样前，先按 batch size 与序列长度估这笔账——大 batch rollout 场景（512+）不要默认开投机。
- **CP 的朴素实现（attention 前 gather KV）成立的前提是 KV 足够小**：压缩类注意力把 KV 缩到可 gather 的量级，才让「先 gather 再算」比 ring 式分片更划算。做长上下文并行设计时，先算压缩后 KV 的总量，再选 CP 的实现形态。
- **RL 侧稳定性靠 replay 路由与 indexer 结果**：MoE 的 router 与稀疏注意力的 indexer 都是离散选择，训练/推理两次算不一致就会引入噪声；把它们在 rollout 时记录、训练时重放，是低成本的训推一致手段，与 EP111 中「策略版本隔离」属于同一类问题。
- **混合精度 rollout（FP8）+ 高精度训练（BF16）**：rollout 占 RL 时间大头，用 FP8 rollout 提速是直接可试的配置；但训推精度差带来的 logits 不一致问题讲者未给结论，需自己验证。

## 疑问 / 下一步

- ShadowRadix 的 tombstone 回收与三路池的内存配比细节（各池容量如何定、ring buffer 大小）字幕只给了框架，细节在 SGLang 技术博客与源码里，可按图索骥。
- CP 的 gather 在大并发下 OOM 的实际边界未测（当前只针对小并发）；ring attention 式 CP 还在考虑中，长上下文大并发的最终形态未定。
- FP4 indexer 推后的代价没有量化：day-0 版本在 indexer 上用了什么替代实现、对 decode 延迟影响多少，字幕未展开。
- Rollout FP8 与 training BF16 的精度差是否需要训练时重算 logits，讲者明确表示不是他的专长——这是 SGLang 用于 RL 时与 verl/slime 等框架衔接时要单独确认的点。
- Mega-MoE 只支持 SM100 + NVLink 域内，H 系列与跨机场景的等价融合方案（DeepEP v2？）还在计划里。

## 原文金句（1-2句）

> 「投机采样优化的是访存：读一次 KV cache 就能验证好几个 token；但 batch size 大的时候访存其实不是瓶颈，所以它的效果不会那么好。」——Q&A 中讲者对「大 batch 下投机为什么变差」的回答（字幕 54:12–54:43，按干净口径转写）

> 「应该是并行推理策略决定了 all-reduce，而不是反过来——必须要用 TP 的场景里，all-reduce 是没法被避免的。」——讲者谈并行策略与通信算子的因果关系（字幕 55:05–55:20，按干净口径转写）

> 「我觉得最重要的（技术点），首先就是 ShadowRadix——怎么样在现有的 RadixTree 上去支持 V4 复杂的 attention 架构；另外一个我会挑 CP，这是针对 100 万长上下文的优化。」——讲者收尾自评（字幕 58:20–58:44，按干净口径转写）
