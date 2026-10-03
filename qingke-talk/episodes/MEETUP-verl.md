# MEETUP-verl — verl: An Open-Source Large-Scale LLM RL Framework for Agentic Tasks
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/MEETUP-verl.html

> 「最后可能来点面条，然后就是这 verl 。」——讲者自谦本场是线下 Meetup 的收尾分享（字幕 00:48–00:55 转写）

## 元信息

- 活动归属：2025-08-24「LLM RL & RL Infra」线下 Meetup（青稞社区线下分享，本场为收尾场；与 B 站线上合集 sid=6759789 同系列）
- 标题：verl: An Open-Source Large-Scale LLM RL Framework for Agentic Tasks
- BV：BV1wbCFB1E7e
- 时长：约 1372 秒（约 22 分 52 秒，字幕时间戳至 22:50）
- 提炼日期：2026-10-02
- 分享嘉宾：方家瑞（字节跳动，verl 团队；字幕 00:00 自述「我是方家瑞，来自字节跳动」，本场由他代 verl 团队现场分享；具体组内头衔以公开资料为准）
- 相关框架：verl（HybridFlow），https://github.com/verl-project/verl（字幕以「VR / word / 沃尔」等音近词指代，均已按官方拼写还原）
- B 站链接：https://www.bilibili.com/video/BV1wbCFB1E7e/
- 字幕原文存档：本地 `transcripts/MEETUP-VERL.txt`（577 条，带时间戳，来源为 B 站 AI 字幕，生成日期 2026-10-02）
- 关联期号：本仓库 EP049 已收录 verl 的源码导读版（同为 verl / HybridFlow 线的互补一讲），本篇为线下 Meetup 的框架总览 + 最近更新版（讲者与 EP049 不同，具体分工见「疑问 / 下一步」）

> 📝 提炼方式说明：本纪要基于 B 站 AI 字幕原文（已存档）清洗提炼。字幕中「VR / word / 沃尔」为 verl 的 AI 字幕误识、「史莱姆」为 slime、「ROU / rot」为 rollout、「麦克 / mc tron」为 Megatron、「deep sick」为 DeepSeek、「千问」为 Qwen、「BOAD / RT / ATR」等均为误识，已在纪要中按上下文还原为官方拼写；方家瑞、张弛等自述姓名以字幕原文为准，正式信息以公开资料为准。凡数字标「字幕」即为讲授现场口径；EP049 论文口径数字（如 HybridFlow 吞吐）未在此重复，详见 EP049。

## 一句话总结

这场 23 分钟的线下分享把 verl 放回它最原始的定位：一个以「RL = dataflow」为抽象原点的框架。讲者用 Hybrid Controller（单控制器编排 + 节点内 SPMD 执行）与 Hybrid Engine（rollout 与 trainer 同资源异并行、靠 reshard 自动搬运 activations/参数）两个设计，说清 verl 为什么能用十几行代码写 GRPO/PPO、又能扛住 671B MoE 规模；分享的另一半是 2025 年上半年的更新清单：DeepSeek 671B / Qwen 235B 的可跑通规模、server-based 异步 rollout 与 partial rollout、agent loop 与 MCP/tool-use 配方、角色-backend 解耦与 Megatron 强化，以及 Q3 的 partial rollout / 全异步、TRKT-PPO 与 SWE-bench 配方路线。

## 核心

### 引入：两拨人做这个框架

讲者开场先把做 LLM RL 的人分成两拨：一拨是算法背景（他点名前面分享的算法同学），从负载与算法目标出发设计系统；另一拨是系统背景出身（如 slime 的子林、verl 的张弛——后者之前做 FPGA、对 data flow 特别熟），从资源抽象与执行引擎出发做框架。verl 属于后者，它的愿景不是「再发明一个 RL 算法」，而是「让用户只管指定 dataflow 图和算法数学行为，底层 kernel 与并行优化交给框架」。

规模语境也直接给了：字节内部除预训练外，大部分后训练的 RL 计算量已远超 SFT 与持续训练，多模态（图像、视频、音乐）都在用 RL。verl 是在这种内部需求里长大、又恰好踩中开源风口的项目，讲者称它是 R1 之后可能最大的受益者（开源项目）。

### 设计原点：RL 是 dataflow，每个节点都是分布式程序

讲者给 verl 的第一性抽象是：RL 是一个 dataflow。每个阶段（generation/rollout、reward、reference、advantage、update）是一个节点，节点之间有边和依赖；与傳統 RL 不同，每个节点内部已经不是单机程序，而是分布式程序（大模型 + 长 context，rollout 与 trainer 各自都要分布式）。这带来两个结构性难题：

1. **依赖与编排复杂**：rollout 与 reverse/reference 之间有依赖，内存塞不下要换入换出，资源受限时「怎么合理排布不同阶段的模型与任务」本身就是调度问题。
2. **并行策略异构**：rollout 与 trainer 的最优并行度通常不同（讲者举例：训练用 TP4/DP2/PP1，generation 用 TP2/DP4/PP1 的切分组合），中间需要 parameter reshard 与数据重组。

verl 用两个对应设计回应这两点——Hybrid Controller 管编排（用什么编程模型），Hybrid Engine 管执行（不同并行之间怎么高效搬运）。

### Hybrid Controller：单控制器编排、节点内多控制器执行

讲者溯源到 Google Pathways（2022 年初，540B 模型那篇）里的两种范式：

- **Single Controller（单控制器）**：一个中心化控制器管理所有 worker，worker 可执行不同程序，即 MPMD 范式；适合描述 RL 这种节点异构的 dataflow。
- **Multi Controller（多控制器）**：类 MPI 的 SPMD 范式，每个进程跑同一程序的不同数据分片；PyTorch DDP/FSDP、Megatron 3D 并行都是这一类，适合节点内部的训练/推理执行。

verl 的 Hybrid Controller 把两者拼起来：**顶层用 single controller 把 dataflow 编排写成近似单机代码**（用户只操作控制面，讲者称 GRPO/PPO 「可能十几二十行就能实现一个比较复杂的流程」），**每个节点内部切换到 multi controller 去跑成熟的 SPMD 后端**（rollout 的推理引擎本质是 multi controller 优化，trainer 的 FSDP/Megatron 同样）。这样复用了既有推理/训练引擎的并行优化，又保留了上层编排的灵活性。讲者强调，这套编程模型现在「基本所有 LLM RL 框架也都采用了」。

### Hybrid Engine：rollout 与 trainer 同资源、异并行，自动 reshard

Hybrid Engine（HybridFlow 论文的第二大设计，讲者说明这一思路最早由 DeepSpeed 提出同类想法，verl 把它「做得比较极致」）解决的是：rollout 与 trainer 部署在同一批计算资源上，但两者最优并行度不同，于是每次切换都要在不同并行布局之间重组参数与 activations。

verl 的做法是提供 decorator 形式的语法糖：用户只声明每个组件期望的并行方式（例如 3D 并行或 DP），框架在后台自动处理 gather → reshard → scatter 的通信，把「不同并行度之间要传什么、按什么顺序传」的细节封起来。论文里进一步说可针对负载与资源规模搜最优并行方案；讲者现场的评价是「当时大家对它应用性评价非常高」，8 卡/16 卡规模下当时性能领先。这一设计后来也是 verl 被工业界严肃项目选用的原因之一：切换成本低、封装完整。

### verl 为什么成功：几个小而关键的早期决定

讲者明确说，除了 HybridFlow 论文的两大创新，verl 的成功还有一批细节：

- **API 早兼容**：很早就兼容 Megatron、FSDP、vLLM、SGLang 等后端，而不是自成封闭生态。
- **灵活的 device mapping**：既支持分离部署，也支持 hybrid（同资源）部署，不锁死一种拓扑。
- **与 DAPO 的协同出口**：verl 发布时即与 DAPO 相关工作衔接（讲者称「DAPO 通过 verl 做了一个出口，给项目导入了不少流量」；按：DAPO 的论文与开源在 2025 年由字节 Seed 等团队发布，细节以论文为准）。
- **性能锚点**：靠 3D Hybrid Engine 在小规模上打出领先性能，建立早期口碑。

另一个定位判断：verl 把自己做成 researcher（易改、功能全、知名度高，高校发 paper 优先选）与工业界用户（稳定性、规模支持）的二元汇合点。它天然在训练框架、推理框架、硬件厂商、agent 研究者之间形成一个社区——讲者提及团队每周四有公开 Zoom meeting 供各方参与。

### 最近更新（2025 上半年）：大 MoE 支持

讲者说这是最近半年最重要的 updates 之一。团队在 6 月之前花了很大功夫让 DeepSeek 671B 与 Qwen 235B 两类大 MoE 能在 verl 上跑起来、loss 曲线正确。现场给的规模数字：

- **DeepSeek 671B**：可在 **96 张 H20** 上跑起来（字幕口径），但 MFU 仍偏低，后续在持续优化。
- **Qwen 235B**：可在 **32 张 A100** 上跑起来（字幕中「32 张 AHR0」，按上下文还原为 A100；讲者只给「能跑起来」，同样强调 MFU 偏低）。

讲者对这两条的定位很克制：意义不在「跑得快」，而在「显著降低了做 SOTA 模型 RL 的门槛」，让更多团队能对当红开源权重做 RL 实验。

### 最近更新：从同步 rollout 到 server-based 异步与 partial rollout

原始 verl 是同步的：先全量 rollout、同步、再 trainer 更新。讲者承认「如果你是做算法的人设计的，肯定第一时间就不会这样」——同步意味着 GPU 要等最慢的样本，工具调用的 IO 开销也无法与计算重叠。

最近半年的 rollout 升级是把 LLM inference engine 做成 **server-based**：engine 以 server 形态常驻，client 按需发请求，于是可以实现：

- **Async rollout（A-think，字幕音）**：tool calling 等 IO 开销与计算重叠，GPU 不再整批等待。
- **Partial rollout**：长样本不必等全组完成，支持分离式「rollout 不间断、 trainer 攒够 batch 就更新」的全异步方案（讲者在 Q3 路线里点名这一管线正在完善，会很快支持与 AREAL 类似的全异步方式）。

讲者的坐标很清楚：同步是 verl 的出发点，不是它的主张；异步化（尤其在 agent 场景下）是这一代框架的公共必修课，前面的分享（讲者点名 AREAL）在这条线走得靠前，verl 正在补齐。

### 最近更新：agent 生态——tool use、sandbox、MCP、agent loop

今年主题是 agent，verl 的对应更新分三层：

1. **典型 tool-use 场景的 recipe**：code sandbox、search engine、computer use 等都有相应支持。
2. **成熟 agent framework 与协议接入**：引入 LangGraph 一类框架，并支持 MCP，可通过 MCP 接入外部工具。
3. **Debug 工具**：tool use 涉及多轮调用，需要能回看 rollout 与调用链是否正确，verl 提供相应可视化/检查工具；讲者说明这一块与 vLLM/SGLang 同学合作较多（性能、易用性、debug 工具）。

其中 agent loop 是讲者团队最近亲手做的一个小而重要的修复。背景是复现字节的 **ReTool** 工作（字幕作「RETO」；字节 Seed 的代码工具 RL 工作，原团队未开源代码，本复现是与 SGLang、面壁智能同学一起做的）：过程中发现最大坑在 **tokenizer** —— 原实现是 text-in/text-out，中间对文本做一点拼接再 detokenize，token id 就与原序列不一致，导致训练曲线不对。修复方法是把 agent loop 做成 **token-in/token-out** 的接口：用户可定义任意轮数的 LLM generation 与 tool-use 交替（不只是 ReTool 的一轮 generation、一轮 tool use，还可以在中间插入总结或多轮交互），且全程不经过文本重切分。讲者称这是「把 ReTool 的训练曲线都调正确了」之后沉淀下来的抽象。

### 架构解耦：角色与 backend 分离

原先 verl 把角色（trainer/rollout 等）与 backend（Megatron trainer、FSDP trainer）耦合，换后端等于换一整套实现。最近的重构把两者解耦：**角色是角色，backend 是 backend，每个角色可切换不同 backend**。讲者说明当前对 Megatron 的支持力度在加大，原因很直接：「像一些比较大的 MoE 训练，肯定还是 Megatron 这边性能更好。」这条线与大 MoE 支持是同一件事的两面：规模越大，backend 的成熟度越决定上限。

### Q3 路线（讲者现场口径）

讲者给的 Q3 roadmap 有四条 + 一条社区信息：

1. 基于 agent loop 的 **partial rollout 与 full-async training pipeline** 实现（目标：接近 AREAL 式的全异步方式）。
2. **TRKT-PPO**：字节 Seed 算法同学提出的一个方法（字幕作「TRKT」，另有论文），verl 正把对应 recipe 放出。
3. **持续性能优化**：重点是 DeepSeek、Qwen 235B/480B 等大模型的 MFU，与 NVIDIA 同学合作中。
4. **新 agent RL recipe**：最近会放出 **SWE-bench** 的 recipe，讲者提醒「难度比较大，大家复现起来（有挑战）」。

社区入口：verl 有微信群，问题可提 issue 或直接发邮件给团队联系人（字幕作「海滨」，指团队成员 Haiyin？以仓库维护者列表为准）。

## 关键数字总表

| 指标 | 基线 / 口径 | 结果 / 数值 | 来源 |
|---|---|---|---|
| 本场时长 | — | 约 1372 秒（22 分 52 秒） | 字幕时间戳（末尾至 22:50） |
| 字节内部 RL 计算量定位 | SFT / continue training | 大部分后训练 RL 计算量已远超二者（字幕口径） | 字幕（引入） |
| Hybrid Controller 代码量印象 | 复杂 RL 流程 | 十几到二十行可实现 GRPO/PPO | 字幕（讲者印象） |
| DeepSeek 671B 跑通规模 | — | 96 张 H20 可跑起来，MFU 尚低 | 字幕（更新） |
| Qwen 235B 跑通规模 | — | 32 张 A100 可跑起来，MFU 尚低 | 字幕（更新，「AHR0」按上下文还原） |
| DeepSpeed / Pathways 引用 | Hybrid Engine 思想来源 | DeepSpeed 曾提同类想法；Pathways 提出 single/multi controller 二分（2022，540B） | 字幕（设计溯源） |
| Q3 待优化规模 | — | DeepSeek、Qwen 235B、480B 大模型 MFU | 字幕（Q3 路线） |
| 社区同步节奏 | — | 每周四公开 Zoom meeting | 字幕（社区） |

## 可迁移

- **把 RL 系统先写成 dataflow 再谈并行**：讲者反复强调节点与边、依赖与排布，是可迁移到任何 RL/agent 基建讨论的第一性框架；先对齐「每个节点是什么、谁等谁」，再问 TP/DP/PP 怎么切，能避开很多过早优化。
- **Agent loop 的 token-in/token-out 原则**：多轮工具场景下，凡经过 text 拼接再 detokenize 的中间层，都要怀疑 token id 漂移；verl 的修复说明这类 bug 会先以「训练曲线不对」的形式出现，而不是报错。
- **同步→异步的迁移顺序**：verl 的路线是先把推理引擎 server 化（client 请求、engine 常驻），再谈 partial rollout 与全异步 trainer——先改接口形态，再改调度语义，比直接重写训练循环便宜得多。
- **角色/backend 解耦的时机**：当框架被多个团队以不同规模（小规模研究 vs 大 MoE 生产）复用时，耦合实现会先在 backend 选型上卡住；verl 的选择是把 Megatron 这类成熟 backend 做成可切换项，而不是自研一切。
- **对研究者的选型信号**：verl 的成功因素（早兼容、各后端都接得上、论文出口、社区可见）提示，评估一个 RL 框架时，「出口通量」（论文/项目愿意在它上面发布结果）与 API 兼容面，和某个单点性能数字同样重要。

## 疑问 / 下一步

- **本篇与 EP049 的关系**：EP049 是 verl 的 EP 期号分享（源码导读体裁），本篇是 2025-08-24 线下 Meetup 的框架总览 + 更新分享（讲者为方家瑞，EP049 讲者为童雨轩）。两者同属 verl 线，但还缺一环：EP049 以论文（HybridFlow, arXiv:2409.19256）为主要交叉口，本篇的框架细节（尤其 Hybrid Engine 的 reshard 实现与「自动搜最优并行」）与论文的对应关系，讲者只点到为止，具体实现可回 EP049 对照。
- **MFU 数字未给出**：671B/235B 的「能跑起来」没有附 MFU 数值，只有「还比较低」的定性判断；与其他框架的同等规模对比，也不是本场的目标（讲者明确说本场是「给大家简单过一下」）。
- **TRKT-PPO 与 ReTool 的出处**：字幕只给音近词（「TRKT」「RETO」），未给论文链接；ReTool 的官方开源状态需另查（讲者说原团队未开源，本复现为社區合作版）。
- **Partial rollout 的具体语义**：讲者点名与 AREAL 的全异步对标，但 partial 的粒度（sample 级还是 token 级）、off-policy 程度的控制（staleness bound）等实现选择，字幕没有展开；Q3 的实现落地情况需查 verl repo 的 release 与文档。

## 原文金句（1-2句）

> 「RL 是一个 dataflow，我觉得这个是 verl 设计的一个比较重要的核心。」——讲者在背景部分对框架原点的定义（字幕 04:13–04:19，转写后）

> 「最后可能来点面条，然后就是这 verl 。」——讲者开场自谦本场是 Meetup 的收尾分享（字幕 00:48–00:55，转写后）

> 「（同步 rollout）如果你是一个算法的人设计的，肯定第一时间就不会是这样。」——讲者在 rollout 升级部分对原始同步实现的坦承（字幕 18:31–18:35，转写后）
