# EP76 — NeMo RL：让大规模 MoE 模型权重 Refit 加速 10 倍

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP76-nemo-rl.html

> 「在 MOE 这种模型会占比会更高吗……我们要整体把这个模型进行相应的 refit 传输，然后在训练过程中我们是以比较轻量化的方式只 activate 部分，所以对比两者的差异，其实 refit 占据的比重会大大增长。」——李知宇在 Q&A 中对 refit 在 MoE 上相对开销的判断

## 元信息

- 期号：76
- 标题：NeMo RL：让大规模 MoE 模型权重 Refit 加速 10 倍
- BV：BV1qNWyzcEuv
- 时长：01:06:55（4015 秒）
- 提炼日期：2026-10-02
- 分享嘉宾：高文文（NVIDIA NeMo 团队高级产品经理；字幕中亦作「高雯雯」）、李知宇（NVIDIA NeMo 团队高级深度学习算法工程师；字幕中亦作「知愈」）。主持：王过（青稞社区主理人；字幕作「青科社区」）
- 相关代码：NVIDIA NeMo RL（github.com/NVIDIA-NeMo/RL）；NeMo Gym 为独立仓库
- 相关论文：本期为框架工作分享，无单一对应论文；文中提及 FlashRL（FP8 训练 recipe）、Megatron-LM、vLLM 等外部工作
- B站链接：https://www.bilibili.com/video/BV1qNWyzcEuv/
- 官网期号：EP76（预告链接未确认，略）
- 字幕原文存档：本地 `transcripts/EP76.txt`（1255 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP76，已存档）清洗提炼。AI 字幕识别误差较多，专名按公开资料校正后写入纪要：NeMo RL 在字幕中作「NEO2L / 李某RL / NEO jam」，vLLM 作「BLM / BLOM / VRM」，Megatron 作「Mac 创 / 麦克创」，refit 作「REFAT / reflect」。讲者口头给出的约 15 倍 refit 加速与标题的 10 倍并存，下文按字幕口径记录。

## 一句话总结

NVIDIA NeMo 团队介绍了 NeMo RL 框架（双训练后端 + vLLM 生成 + Ray 调度）的整体设计，并重点拆解了大规模 MoE 模型上 refit（训练侧到生成侧的权重同步）的三级优化：张量打包、点对点通信替代全量广播、元数据缓存，叠加后 refit 本身约提速 15 倍；同时给出 FP8 端到端训练、异步 RL 与 NeMo Gym 环境层的路线图。

## 核心

### 背景：为什么 RL 框架要重点优化 refit

高文文先给出框架定位：NeMo RL 是 NVIDIA 的强化学习训练框架，算法层面覆盖当今主流算法（GRPO、DPO、REINFORCE 系），卖点是性能与可扩展性、与 Hugging Face 生态的开箱即用。背景判断有两条：

1. 预训练已把可用数据基本用完，RL 对原始数据依赖低（模型自生成多条思维链再挑好坏），是当前提升模型能力的主要路径；
2. RL 的算力需求已与预训练同量级——讲者引用 xAI Grok 的公开说法：约 20 万张卡做 RL，RL 机器资源与预训练「几乎相等」。

架构上（李知宇部分亦有呼应）：policy 模型训练与生成分处不同进程，训练后端可选 PyTorch 原生后端（易用、新模型第一周可上手）或 Megatron 后端（四维模型并行、面向大模型长序列），生成用 vLLM，Ray 做 single-controller 调度，reward 模型单独放在一个 Ray cluster（通常只需 1 张或非 8 倍数张 GPU，塞进训推 cluster 没有收益）。环境/依赖用 uv 做隔离。

### 精度对齐：先保证训推两侧 log-prob 一致

框架部分最实用的经验是训推精度对齐的 debug 方法：对比训练侧与推理侧的 log probability，若两者一致则其比值（ratio）应为 1；讲者给的经验阈值是超过 1.05 就说明两侧差异过大、需要深挖。实测案例：早期做 Qwen 1.5B 时，两侧 log-prob 在 600 步之后出现显著分叉、reward 曲线同步异常，最终定位到 tied embedding 在训练与推理两侧实现不同。讲者强调 RL 系统里训练、生成各是一套软件（PyTorch/Megatron + vLLM），新模型支持期的大量时间就花在这类精度 debug 上。

PyTorch 后端与 Megatron 后端的取舍给了实测参考：PyTorch 后端约可跑 4B/64K、32B/24K 量级；70B 模型上 Megatron 后端训练比 PyTorch 后端快约 3~4 倍、端到端约 2 倍；8B 上差距不明显。两个后端在 8B 与 70B（Llama 系）上的 reward 曲线几乎重合，精度已验证一致。若用 Hugging Face TRL 或 TorchTitan 这类原生 PyTorch 框架，速度可参考 NeMo RL 的 PyTorch 后端数据。

另一个口径提醒：平均生成长度对耗时的影响不亚于参数量。Qwen-32B 参数量约为 Llama-70B 的一半，但平均生成 token 数约是后者的 17 倍，最终训练与前向耗时反约 3 倍。Qwen-30B（字幕作「宽30比ME」）的耗时讲者自承尚有问题未优化完，数据待更新。

### Refit 三级优化：从逐张量 IPC 到约 15 倍

李知宇接棒讲核心工作。refit 指 on-policy RL 中每个训练步之后，把 policy 模型参数同步到生成模型；难点在于它同时是跨并行方案（两侧并行切分不同）与跨进程（训练与生成独立进程）的同步。NeMo RL 的三步工作流是：

1. 从各分片 all-gather / broadcast 聚合 policy 参数；
2. 通过 CUDA IPC handle 跨进程传递显存引用（传指针而非参数本体，省去整份拷贝）；
3. 生成侧按 handle 恢复 tensor 并更新自身分片。

瓶颈在超大 MoE 上被放大：以 DeepSeek-V3 为例，671B 参数对应约 4.5 万个独立参数张量，逐张量生成 handle、传递、恢复要执行约 4.5 万次。三级优化：

- **张量打包**：把多个参数 pack 进一个大 IPC buffer 一起传、到对侧再解包。单次 IPC 的固定 overhead 不变，但调用次数大幅下降：讲者给出的单步耗时从 120 ms 降到 12 ms（profiling 图中原先 655 个 tensor 逐个走 IPC 的开销被压成一次传输），refit 整体约提速 1.5 倍。
- **点对点分发替代全量广播**：vLLM 的 collective RPC 是 broadcast 语义——同一份信息要发给每一个 executor，信息量是平方级复杂度。改成点对点后，每个 executor 只收自己需要的那份，通信冗余消除，refit 约再提速近 4 倍。
- **元数据缓存与本地重算**：每次 refit 不变的静态元数据直接缓存、不再重复传输；另一部分信息改为在生成侧本地重算。refit 约再提速 2.5 倍。

三级叠加，refit 本身约提速 15 倍（标题的 10 倍应为对外的保守说法）。Q&A 给了这件事的边界条件：

- refit 是纯 GPU idle 事件，耗时大体固定，而训练/生成耗时随序列长度与 batch 变化——短序列下未优化的 refit 占比可达 600 多秒，优化后这部分时间全部还给 GPU 利用率；
- refit 传输量与参数总量近似线性（BF16 每权重 2 字节，gather 与更新两侧合计约 2 倍权重量的传输，讲者原话为「两倍的传输量」），与训练/推理耗时同为粗略线性，所以占比大体固定，但实际强依赖并行方案；
- MoE 上 refit 的相对占比显著更高，因为 refit 必须传全量权重，而训练/推理只激活部分专家——这正是标题点名 MoE 的原因；
- 与 verl 的 hybrid engine 对比：小模型上 NeMo RL 的 refit 可能更快，大模型上 verl 略占优；差距来自设计取舍——verl 没有把训练与推理拆成独立进程，不需要付 IPC 跨进程通信的开销，NeMo RL 为模块化与可扩展性接受了这部分成本（优化后已大幅缩小但仍存在）；
- 本期讲的都是 colocate（训推同卡）路径；训推分离部署用的是另一套 NCCL refit（字幕作「nico refit」），尚未做完，后续单独成文。

未来计划：集成 NeMo Megatron Bridge，借它打通 Megatron 与 Hugging Face 模型生态，预计进一步压缩 refit 时间。

### FP8 路线：先生成侧，后端到端

目标是 RL 全流程端到端 FP8（训练 + 生成都 FP8）。当前进度只完成生成侧：Llama-3-8B 上用 DeepSeek 同款 block-wise FP8（block scaling），配合重要性采样修正，3000 步左右的训练里序长从约 600 涨到约 1500；生成侧 FP8 带来生成部分约 30% 提速、端到端约 10%。端到端收益有限的原因是生成中约 70% 是 GEMM 可 FP8 化，attention 与 KV cache 仍是 BF16——下一步把这两块也 FP8 化，并尝试 QAT。

讲者明确的精度判断：只在生成侧做 FP8、训练仍是 BF16，中间的 recast 会损精度；训练与推理同为 FP8 反而更好，内部测试明显优于当前展示的数据。端到端 FP8 计划随 9 月 release 发布博客。关于 refit 与低精度的交叉：当前 all-gather 仍走 BF16 通信、更新权重时在本地 cast；理想做法是通信本身走 FP8（数据量减半），但会破坏训推两侧的模块独立性，可能结合 Megatron Bridge 再实现。FlashRL 是社区的 FP8 训练 recipe 而非框架，NeMo RL 会在框架层提供验证过可收敛的 recipe。

### 异步 RL 与 partial rollout

高文文梳理了进行中的异步工作。术语坐标：colocate = 训推同资源，split = 训推分资源；完全同步的 colocate 流程里，生成长短不齐必然产生 idle。异步方案是让一组 GPU 专职生成、另一组专职训练，训练用上一轮数据更新（one-step off-policy），可进一步放宽到 two-step off-policy——代价是 off 差距越大精度风险越大。当前重点工作就是在压低精度损失的同时保住异步的利用率收益。partial rollout（把过长的生成切下、留到下一轮继续）在 colocate 与 split 两种部署下都可用，工程上有 caching 等 trick 减少新旧模型数据混杂。这部分同样计划在 9 月 release。

### NeMo Gym：把环境层拆成独立仓库

NeMo Gym 把环境从 RL 框架中解耦：框架侧模型发 prompt 并声明要用的工具，Gym 侧调用工具（Google 搜索、计算器、代码执行等）并返回结果，多轮之后 verifier 给出 reward/signal。两个正交概念被反复强调：工具与 verifier 正交——工具是动作通道，verifier 按领域/数据集定义验证逻辑；Hugging Face 上几乎所有数据集都可做成 verifier。运行方式灵活：工具与 verifier 可 in-memory 直跑、走 HTTP endpoint，甚至把保密的下游任务封进 Docker container 以 plug-and-play 方式接入。已做的 verifier 覆盖数学、coding、工具调用、科学、游戏，后续加 domain-specific 环境。

Q&A 里讲者把 Gym 定位为「环境动物园」：每个领域有多个数据集、每个数据集即是一个环境；sandbox（如 OpenHands 沙盒）是架在环境之上的脚手架（scaffolding），与环境搭配使用。Gym 与 RL 框架是两个独立 GitHub 仓库、只通过接口连接，也可为 verl 等其他框架服务。对「Gym 是否会像 Isaac Sim」：Isaac 本来就有机器人侧的 gym，两者不会合并，NeMo Gym 专注 LLM 的工具调用与验证。

### Q&A 其余要点

- **Agentic RL 对框架的新挑战**：讲者的答案只有两个字——长尾。agent 多轮调外部工具（搜索、视频处理等），单条轨迹时长不可控、超出框架可控范围，这是异步 RL 需求变大的主因；长尾问题目前除异步外没有更好的解法（李知宇附议）。
- **Hugging Face 模型前向实现可信度**：NeMo RL 支持的主流模型大致可信、偶有小 bug；小众模型 bug 较多。更根本的难点是 PyTorch、Megatron、vLLM 三套软件的精度要对齐，训崩时定位是哪一层出问题本身很费时。
- **硬件与 IPC API 细节**：后端对比用 H100；IPC handle 是否支持 CUDA memory VM 分配与跨节点 fabric handle，讲者现场未确认，承诺回去查文档后补进 PPT。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| xAI Grok RL 算力（讲者引用公开说法） | 约 20 万张卡，RL 资源 ≈ 预训练 | 字幕（背景部分） |
| PyTorch 后端可跑规模（实测参考） | 4B/64K、32B/24K | 字幕（后端选型部分） |
| 70B 上 Megatron vs PyTorch 后端 | 训练约 3~4 倍，端到端约 2 倍 | 字幕（后端选型部分） |
| 训推 log-prob ratio 经验阈值 | 1（理想）；> 1.05 需深度 debug | 字幕（精度对齐部分） |
| Qwen 1.5B 精度分叉案例 | 约 600 步后 log-prob 显著分叉（tied embedding 实现差异） | 字幕（精度对齐部分） |
| Qwen-32B 平均生成长度 vs Llama-70B | 约 17 倍，耗时反约 3 倍 | 字幕（多模型实测部分） |
| DeepSeek-V3 参数张量数 | 671B 参数 / 约 4.5 万个张量 | 字幕（refit 部分） |
| 张量打包单步耗时 | 120 ms → 12 ms；refit 约 1.5 倍 | 字幕（refit 部分） |
| 点对点分发替代广播 | refit 约 4 倍 | 字幕（refit 部分） |
| 元数据缓存/本地重算 | refit 约 2.5 倍 | 字幕（refit 部分） |
| 三级叠加 refit 总提速 | 约 15 倍（标题口径 10 倍） | 字幕（refit 总结部分） |
| 未优化 refit 占比（短序列） | 600 多秒 | 字幕（Q&A） |
| 生成侧 FP8（Llama-3-8B） | 生成约 +30%，端到端约 +10% | 字幕（FP8 部分） |
| 生成中可 FP8 化比例 | 约 70%（GEMM）；attention/KV cache 仍 BF16 | 字幕（FP8 部分） |

## 可迁移

- 对 RL infra 工作的直接可试点：训推精度对齐用「两侧 log-prob ratio，> 1.05 报警」做常驻监控，比盯 reward 曲线更早发现 tied embedding 这类实现不一致；refit 优化先数张量个数——张量数（而非参数量）决定逐张量 IPC 的固定开销，打包 + 点对点是与框架无关的通用手法。
- Infra 视角：MoE 场景下 refit 传全量权重、计算只激活部分专家，权重同步的相对成本随稀疏度上升而上升；评估 MoE RL 框架时应把 refit 占空比单列。另外「工具/verifier 与框架解耦成独立仓库 + 接口」的 Gym 式分层，与 verl 等框架互通，值得在自己的环境层设计中照用。

## 疑问 / 下一步

- 讲者承诺补进 PPT 的两个未答问题值得追踪：IPC handle 对 CUDA memory VM 与跨节点 fabric handle 的支持情况；训推分离的 NCCL refit 方案细节（后续博客）。另注意：本 talk 的发布时间据讲者口径 FP8 与异步 RL 将在「9 月 release」落地，需以 NeMo RL 仓库实际版本为准复核。

## 原文金句（1-2句）

> 「所以啊强化学习就是一个不断探索的过程。」——高文文用「小学有标准答案（SFT）→ 读博无标准答案但有参考（预训练）→ 走入社会只能自己试错（RL）」的类比收束 RL 定义

> 「其实我觉得最大的挑战就是长尾。」——高文文答 agentic RL 对框架的新挑战
