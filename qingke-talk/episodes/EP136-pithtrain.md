# EP136 — PithTrain：Agent 时代的 MoE 训练框架设计
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP136-pithtrain.html

> 「我们能不能有一个框架，它是 agent friendly 的，同时不牺牲 state-of-the-art 的性能？」——讲者全场要回答的问题

## 元信息

- 期号：青稞Talk EP136（B站期号 B136）
- 标题：PithTrain：Agent 时代的 MoE 训练框架设计
- BV：BV1vjNM6wE9i
- 时长：01:13:36（讲授约 61 分钟 + Q&A 约 12 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：赖瑞航（CMU 博士生，字幕自述）、方浩（即将入学的 CMU 博士生，字幕自述；后半程主讲评测部分）
- 相关论文：讲授中未给出论文编号（PithTrain 与 ATE Bench 均已开源，以项目主页与论文原文为准）
- 相关代码：讲授末尾给出项目链接 / 二维码（字幕未报具体地址）
- B站链接：https://www.bilibili.com/video/BV1vjNM6wE9i/
- 字幕原文存档：本地 `transcripts/EP136.txt`（1213 条，末条 4411.44 秒，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP136，已存档）清洗提炼，信息来源为 AI 字幕原文（经清洗，专名可能有识别误差）：字幕中框架名作「pet train / peace train / page train / PVE train / peef train」、对标框架作「mea tron / Mac tron / mega 虫」、TorchTitan 作「torch titan / torch titten / torch chen」、DualPipe 作「do 派 V / 独派」、Muse 作「cloud code」、base 模型作「open4.7」，均按上下文还原，正式写法以项目主页与论文原文为准。凡讲授口径数字均标注「字幕」；个别数字单位字幕未说清，已在表中注明。

## 一句话总结

PithTrain 是一个为 MoE 训练打造的极简框架：它提出先给「框架对 coding agent 好不好用」起名字并量化——Agent Task Efficiency（ATE），再围绕紧凑代码库、纯 Python、少隐式间接、任务级 agent skill 四条 agent-native 原则，用约 1.1 万行代码在训练吞吐上追平 Megatron-LM，同时在 ATE Bench 上让 agent 用明显更少的轮数、token 与 GPU 时间完成问答、运维与新特性移植三类任务。

## 核心

### 为什么只做 MoE：它已经是前沿模型的默认架构

讲者开场给出选型理由：近几年各厂商发布的 frontier model 基本都是 MoE，它在固定推理成本下提供更大的模型容量（每个 token 只激活一部分参数）、推理比 dense 模型便宜、在 reasoning 模型时代更有优势，加上 GPU 显存变大能装下更大的模型，工业界已形成继续 scale MoE 的共识。因此这个工作只聚焦 MoE 一个架构方向（字幕）。

### 摩擦的来源：生产框架对人重，对 agent 同样重

当前训 MoE 的默认选择是 Megatron-LM、DeepSpeed 这类成熟生产框架：模型覆盖全、功能全、跑得快、工业级优化多、社区大、背后有大公司支持，甚至已经延伸到 RL 后端（讲者举例 slime 用 Megatron 作 training 后端，字幕）。但日常使用中有三类高频动作会撞上摩擦：想搞清某个技术在框架里到底怎么实现的；把框架部署到一个新 cluster 上跑起来；看到一篇新论文想快速把新架构 / 新特性加进去。做这些事都必须先深入框架内部，负担会一次次累加。

讲者把摩擦的来源总结为：这些框架代码量都超过 10 万行，有大量 C++ extension 和很多层 abstraction。对人类这是 overhead：要读完所有代码、穿过一层层抽象，改 C++ extension 还要反复 build。关键论点是：到了 2026 年开发主力已经换成 coding agent（讲者说他们自己的开发任务已全部转向 agent，自己基本只在打 prompt、甚至用语音转文字说 prompt），但摩擦并不会因为使用者换成 agent 就消失——agent 要读更多文件、做更多分析才知道发生了什么；错误若出在 C++ 里，报错信息不透明，agent 只能反复试探。由此提出全场的问题：能不能有一个 agent friendly、同时不牺牲 SOTA 性能的框架？对 agent friendly 的框架，对人通常也 friendly（字幕）。

### 先命名，再量化：Agent Task Efficiency（ATE）

过去评估训练框架只看 training efficiency（throughput、MFU），因为训得越快越省钱。但当重心转向 agent，这个维度此前连名字都没有。讲者提出 ATE：固定同一个 task，比较一个 agent 在框架 A 与框架 B 上完成它的效率；可测的量包括 agent 的总用时、总输出 token、agent turns / tool calls 轮数等。需要强调的是，目前还没有找到单一指标能概括整体的 agent 效率，必须同时看一组指标（字幕）。本期的评估因此是双维度的：training efficiency 与 ATE，讲者称之为 dual efficiency（字幕作「dudual efficiency」）。

### 系统结构：从第一性原理看，训练框架只需要三层

讲者复盘从零（或从较低层的 PyTorch）搭一个训练框架实际需要的东西只有三层：最下面是 operator library（PyTorch / 算子库 / 自定义算子）；中间是 training engine（building blocks、model、分布式 runtime）；最上面是应用层（具体的 pretrain / SFT 训练 loop）。十几万行的体量并不等于难度本身。PithTrain 即按这三层打造：名字取「柑橘类水果果肉与果皮之间那层白色物质」之意，也取 the essence of something——想抓住 MoE 训练的本质（字幕）。

### 四条 agent-native 设计原则

1. **Compact codebase（紧凑代码库）**。讲者称这可能是作用最大的一条：代码越少，agent 需要搜索、追踪、阅读、修改的代码就越少；而且现在较好的 agent context window 在百万 token 量级，一个足够紧凑的代码库可以整体装进 context window。PithTrain 目前全部代码约 1.1 万行（讲者注：这是几周前的截图，现在可能 1.2–1.3 万行），比十几万行的生产框架小一个量级。代价是模型与特性覆盖不如生产框架；他们的应对是：新特性可以让 agent 现场加上，未必都要 merge 进主仓库（字幕）。
2. **Python native（纯 Python）**。好处非常具体：没有跨语言接口要理解；报错是 Python traceback 而不是看不见内部的 C++ 错误；不用像 C++ / Rust extension 那样每改几行就重新编译。讲者强调 Python native 不等于放弃 GPU kernel 性能：他们不写 CUDA kernel，而是用 Triton、TileLang 这类 Python DSL 写高效 kernel（字幕作「try on、tower line、tel bil」，按上下文还原），同时拿到可读性、单语言与速度。
3. **No implicit indirection（少隐式间接）**。讲者用 Megatron 的模型定义举例：单独看 transformer layer 文件，看不出 `self.mlp` 到底是什么，它的真正定义写在另一个文件里，要跨文件、跨多层抽象去找——对新手和 coding agent 都是很大负担。PithTrain 的模型定义仿 HuggingFace Transformers 风格，全部写在同一个文件里，人或 agent 从头读到尾就知道发生了什么。代价是牺牲一部分代码复用，换取每个模型尽可能 self-contained。这条原则也推广到其他设计：避免递归的抽象类、过深的嵌套 module 定义、以及 PyTorch 的 nn.Module hook 一类隐式机制（字幕）。
4. **Task-specific agent skills（任务级 agent skill）**。前三条是对代码本身的设计，这一条承认：现阶段还不能把所有事都交给 agent 自己摸索，人类手里有大量高质量的开发 / debug recipe 与固定任务流程，直接给 agent 能省去它大量试错时间。讲者调研了已有框架的 agent 配套现状：Megatron 虽有 skill，但大多与 CI、处理 PR 有关，对开发帮助不大；TorchTitan 与 DeepSpeed 只有一个顶层的 CLAUDE.md / AGENTS.md 说明代码库大概长什么样，没有具体 skill。skill 可以互相组合调用，像用自然语言描述的函数。讲者总结了让 skill 真正好用的三件事：**specific scope**（把这个 skill 到底做什么说清楚，最好加触发关键词，让 agent 听到相关描述就自动来跑）；**explicit prerequisites**（所有必需输入要人提供齐，缺信息时 skill 要明确反馈「跑不了」）；**verifiable success**（最重要：提供确定性脚本帮 agent 判断任务到底完成没有——例如一个 compare 脚本，输出 0 表示搞定、输出 1 表示没搞定；这比让 agent 自己判断成败可靠得多，还能防止 agent 的 reward hacking，并省下它自己思考判断的 token，字幕）。

### 正确性：与 Megatron 对齐 loss 曲线

CMU 没有太多卡、训不了 trillion token，他们在 billion token 规模上训 Qwen3-30B-A3B（字幕作「queen three thirty ba three b」）验证正确性：与 Megatron 用同样的 config 各起一个独立 run，比较 pretraining loss curve。配置为 pipeline parallel 开 4、expert parallel 开 8（字幕作「pipeline 是四、EXPREPARALLEL 是八」）。结果两条 loss 曲线基本重合，早期有小幅颠簸，讲者推测是 router 稳定过程中的噪声；之后又存了几个 checkpoint，比较了两个框架训出模型在标准预训练 benchmark 上的 downstream 表现（字幕）。

### 吞吐：追平 Megatron，靠的是已知技术的组合

吞吐测试覆盖 H100 与 B200 两种显卡、跨节点 InfiniBand 通信（字幕作「infinite band」），每个测试场景的并行维度都开大于 1。三个模型（GPT-OSS-20B、Qwen3-30B-A3B、DeepSeek-V2-Lite，字幕作「DP CV two light」）上，PithTrain 的性能与 Megatron 追平；唯一稍落后的是 Qwen3-30B-A3B，当时测到 124K 对 126K 的训练吞吐（字幕；单位未在口播中说清，引用时需核对原文）。一个未测场景是更大的 FSDP 规模（如 NVL72 全互联），讲者说明学校场景是多个 node 跨节点通信，所以他们把 pipeline parallel 打开（字幕）。

背后的优化，讲者说全是大家已经知道的技术、不是新 idea，但组合起来能让一个纯 Python 框架把吞吐提上来：

- **DualPipe 做计算/通信重叠**（最主要的优化）：DualPipe 本是 DeepSeek 开源的方案，他们在其上做了 overlap 的写法。把一个 transformer layer 按执行切成五个阶段——attention 与 router 的专家选择（计算）、dispatch（专家间通信）、各 expert 的 MLP 计算、combine（通信）、以及 combine 后的 residual 处理——scheduler 就能把不同 batch、不同 layer 的阶段交错执行，让专家通信被计算盖住。他们用框架自带的 Nsight profile skill 抓了 profile 验证：default stream 上跑计算、另一条 stream 上跑 EP 的 dispatch / combine 的 forward 与 backward，通信大部分被隐藏起来（字幕）；
- **full-graph torch.compile**：除 MoE 在开 expert parallel 时有 dynamic shape 没法 compile 外，其余阶段（attention / router 与最后的加权 combine 等）都开 compile（字幕作「stage 1 和 stage 5」，具体阶段划分以原文为准）；
- 其他常规项：EP 通信前先对 token 做 dedup、把一些小循环 fuse 到一起（字幕作「speak loop 给他 few 集」，识别含糊）、delayed wgrad、需要 fuse 的地方写 Triton kernels 等（字幕）。

### ATE Bench：固定 agent 与任务，换框架比 effort

ATE Bench 与 SWE-bench、HumanEval 这类 coding benchmark 是反过来的：那些 benchmark 固定代码库与任务、换不同 agent 来给 agent 能力打分排名；ATE Bench 固定 agent 与任务、换 framework，比较 agent 在不同框架上完成同一件事的工作量差异（turns、token、GPU 时间）。它和能力类 benchmark 是互补关系。

任务按 agent 介入深度分三档（来自他们平时真实使用框架的方式）：

1. **QA**：问代码库相关问题，可能只是通过 GitHub API 看某个文件、确认支持不支持某功能、看 context parallel 怎么实现，读完代码就回答，不实际跑代码。共 12 个问题（如某个 RoPE 怎么实现、有没有用到某物、kernel 怎样 dispatch）。指标看 agent turns、每 turn 的 context 与 output tokens；这一档不看 GPU time 与总时长，因为所有任务都在 23 分钟内完成，API 延迟波动会让时间测量不准（字幕）；
2. **Operate & Profile**：agent 要真的把框架跑起来——拿到框架后装环境、配配置、把训练跑起来（getting started）；训练后接 benchmark 评估正确性（train & evaluate）；收集 MoE router 在训练中用了哪些 expert 的 routing trace 并存下来本地 replay 分析；跑训练时抓 Nsight profile、看哪些 kernel 耗时长（可能要改代码加 NVTX marker）。这一档加入 session 时长与 GPU active 时长两个指标（字幕）；
3. **New Feature**：agent 把一篇新论文的架构实现进框架并真正跑起来。讲者选了 4 篇近期论文：两篇是 attention 侧改动（Differential Transformer 与 MoBA，字幕作「mobile」，按上下文还原为 Mixture of Block Attention）、两篇是 FFN 侧改动（字幕作「dine m o e 和 MOE 加加」，具体论文名以原文为准）。这一档 agent 要独立完成编辑、运行、测试、debug 全流程，因此 active GPU time 是非常重要的指标——它等价于把一个特性 port 进来要烧多少 GPU hours（字幕）。

测试设置：直接用 Muse，base model 为 Opus 4.7（字幕作「open4.7」），extra-high effort，每个任务独立跑 3 次、报告各指标的中位数（字幕）。

### ATE Bench 结果：轮数、token、GPU 时间全面更少

- **QA 档**：与 Megatron 相比，PithTrain 上 agent turns 最多减少约 67%（字幕作「少了百分之大概是 67%」）。讲者归因于两条设计：紧凑代码库，以及去掉 implicit indirection——agent 直接读代码就知道运行时会跑什么，不用在文件之间跳来跳去找答案；
- **Operate 档**：以 getting started 为例，Megatron 上 agent 用了 88 轮才把环境装好、训练跑起来，PithTrain 只用 26 轮；session 时长从 40 多分钟压到 7 分钟以内。总体上 agent turns 比 Megatron 少 70%、比 TorchTitan 少 57%，output tokens 分别少 78% 与 65%（字幕）。一个有意思的细节：在 report heavy kernels 任务里，agent 会自动调用框架内置的 Nsight profile skill、按固定流程分析，省了不少 token——这是 skill 在真实任务里起作用的例子（字幕）；
- **New Feature 档**：把同一个新特性 port 进来，PithTrain 的 active GPU time 比 Megatron 少 44%、比 TorchTitan 少 64%（字幕）。讲者的失败模式分析：GPU 时间主要烧在反复 debug 重跑上。Megatron 侧的典型坑是 agent 加的命令行参数与 transformer config 互相冲突、训练还没起来就 crash，要重新读代码 debug；以及 agent 改到 TransformerEngine 的 C++ 代码、遇到 segmentation fault 却读不到 traceback，只能猜错在哪——猜对了修好、猜错了误入歧途。这正是他们认为 Python-only traceback 对 agent 迭代效率帮助大的原因。TorchTitan 侧的主要问题不是代码结构，而是 OOM：attention 相关改动在显存不够时训练起不来，要反复修复、relaunch（字幕）。

### Skill 消融：GPU 时间不变，agent 的轮数与 token 变少

他们还单独量化了 agent skill 到底帮了多少：选两个 skill（validate correctness 与 capture Nsight profiles），针对一个具体改动（delayed wgrad 这个 commit）做开 / 关对照。关的方式是把 `.claude/skills` 文件夹（字幕作「dog dog cloud skills folder」）连同 git worktree 与 git history 全部移除，让 agent 在本地完全访问不到。结果与预期一致：这类任务本身不确定性低（跑几步、抓一个 profile），所以 GPU time 基本接近（如 20.8 对 22.5 分钟、5.6 对 5.55 分钟，字幕）；但 agent 侧的 turns 与 output tokens 明显更少——因为 agent 是按 skill 给的固定计划走，而不是每个任务都从头推导该怎么做（字幕）。

最后是 token 去向分析（以 MoBA 移植任务为例）：editing 部分 PithTrain 用 4.7K token，Megatron 用 13.1K、TorchTitan 用 22.2K；探索代码库部分 PithTrain 2.2K、TorchTitan 3.8K、Megatron 10.2K；三次独立 run 里 PithTrain 的 per-turn context 增长曲线保持又低又平（字幕）。

### 开放问题：讲者自己列出的四点不确定

1. **工业界到底 care 吗**：工业界烧 token 没人算账，甚至有公司鼓励多用；省 token、省时间的框架对工业界的实际意义存疑。讲者同时指出社区确实在动：Megatron 自己也开始有 agentic 的 PR、在做 Megatron-Lite，希望代码更 agent friendly（字幕作「A 站 tic 的 pr、mega 虫 light」，按上下文还原）；
2. **差距不是数量级**：代码量小了一个量级，但 output token 与任务总时长并没有小一个量级；这种程度的差异是否足以驱动迁移，是个问题；
3. **agent 自身在进化**：再过一年 agent 足够强之后，各框架的 ATE 会不会趋同，现在没法预测；
4. **ATE 可以当 CI 用**：把 ATE Bench 像 CI 一样定期跑（每个 PR 或每隔十天），监控 agent 完成任务有没有显著变慢，作为框架设计目标的回归测试。还有一个脑洞：新特性可以先在 PithTrain 里做一个简洁实现当参考，再移植到复杂框架里，而不是直接去读复杂实现。

讲者也明说 ATE 本身还没定义清楚：没有单一指标，四条原则各自贡献多少无法消融（比如没法把代码手动加到十几万行再测一遍，几乎做不了）；ATE Bench 的任务还有个维护约束——一个新特性任务只有在任何被比较框架都没实现过时才能入选，某框架一旦实现了，这个任务就必须退出、要不断补充新任务；PithTrain 的代码量肯定会涨，关键是增长过程始终守住四条原则（字幕）。

### Q&A 要点

- **building block 为什么不直接叫 library**：概念层级不同——operator library 里就是纯接口加底下一个具体的 kernel 或计算 function；building block 在他们的设计里是一个围绕某个 feature 的全部实现集合。它只是概念上的划分，代码里并没有一个叫 building block 的文件夹装所有东西（字幕）；
- **agent-native 会不会牺牲人类可读性**：讲者答 Yes and No。少 implicit indirection 可能带来更多代码重复，人看起来会觉得有点乱；但紧凑代码量与纯 Python 这两条对人同样友好——读 Python 比读 C++ 简单，读一个 1 万行框架的一部分比读十几万行框架的一部分直接得多；
- **要不要为系统软件工程专门训 agent / 做微调**：两位讲者的共识偏向 skill 路线。skill 只是加一些文件、agent 读了就知道流程，发现问题还能当场改 skill 自我更新；微调则要自己收集数据、训练、部署，成本高且更新慢。如果能方便地拿到训练数据微调一把也有价值，但更新速度比不上 skill；
- **代码涨到 50K / 100K 行后优势还在吗**：他们的目标就是不让它涨到那个量级——50K 可能还守得住 compact 的定义，100K 就已经超出；会靠架构设计、重构与对 agent 的 guidance 控制增长；
- **ATE Bench 指标之间是否独立维度**：没有做过严肃的因子分析。讲者感性判断有些指标相关（如 turns 多往往 output token 也多）、有些相关性没那么大（active GPU time 与 session duration 有时相关有时不相关），没有实验证明它们是不同维度；
- **跨模型 / 系统级维护任务**：ATE Bench 暂时没有这类任务。new feature 档目前主要是模型侧改动，他们最初想加系统级优化任务，但候选任务大多已经被某个 baseline 框架实现过、不满足入选条件；operate & profile 档的任务则基本都与系统相关；
- **Prime-RO 一类对比**（字幕作「prime ro」）：讲者表示暂时没做过、需要再去了解该项目；对「R3」所指亦未确认，欢迎评论区或邮件继续讨论。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| 生产框架（Megatron-LM / DeepSpeed）代码量 | 超过 10 万行 | 字幕（问题提出部分） |
| PithTrain 代码量 | 约 1.1 万行（几周前口径，现在约 1.2–1.3 万行） | 字幕（原则一） |
| 正确性验证规模 | billion token 级，训 Qwen3-30B-A3B，PP = 4、EP = 8，与 Megatron 对照 | 字幕（正确性部分） |
| 吞吐对照 | 三个模型与 Megatron 追平；Qwen3-30B-A3B 为 124K 对 126K（单位字幕未说清） | 字幕（吞吐部分） |
| ATE Bench 测试设置 | Muse + Opus 4.7，extra-high effort，每任务 3 次取中位数 | 字幕（ATE Bench 设置） |
| QA 档（12 个问题） | agent turns 相对 Megatron 最多减少约 67% | 字幕（QA 结果） |
| Operate 档 getting started | Megatron 88 轮 / 40 多分钟 vs PithTrain 26 轮 / 7 分钟以内 | 字幕（Operate 结果） |
| Operate 档总体 | turns 比 Megatron 少 70%、比 TorchTitan 少 57%；output tokens 分别少 78%、65% | 字幕（Operate 结果） |
| New Feature 档 | active GPU time 比 Megatron 少 44%、比 TorchTitan 少 64% | 字幕（New Feature 结果） |
| MoBA 移植的 token 去向 | editing：4.7K（PithTrain）/ 13.1K（Megatron）/ 22.2K（TorchTitan）；探索代码库：2.2K / 10.2K / 3.8K | 字幕（token 分析） |
| Skill 消融（delayed wgrad） | GPU time 基本不变（20.8 对 22.5 分钟；5.6 对 5.55 分钟），turns 与 output tokens 明显减少 | 字幕（Skill 消融） |

## 可迁移

- 给「框架对 agent 好不好用」先命名再量化，是可直接借用的方法论：固定 agent 与任务、换代码库，测 turns / output tokens / active GPU time，把框架的 agent 效率纳入 CI 式回归监控，而不是只看 MFU。
- 写任何要长期被 agent 维护的代码库时，四条原则可直接套用：控制代码总量（最好能整仓进 context window）、尽量单语言纯 Python 拿到可读的 traceback、减少跨文件隐式间接让定义自包含、把固定流程写成带明确 scope、prerequisites 与确定性验收脚本的 skill——验收脚本输出 0/1 这一条，对防止 agent 自我误判与 reward hacking 尤其值得抄。
- 从失败模式反推设计：agent 的 GPU 时间主要浪费在「crash 后无 traceback 只能猜」与「参数 / 配置冲突导致训练起不来就重跑」上；给 agent 用的系统应优先保证报错透明、配置单一事实来源。

## 疑问 / 下一步

- ATE 还没有单一指标，四条设计原则各自的贡献无法消融（讲者自述几乎做不了这种实验）；ATE 的提升到底主要来自代码量、还是来自 Python 与少间接，尚无分解证据。
- 优势不是数量级：token 与总时长只省了百分之几十，讲者自问这是否足以驱动工业界迁移；且工业界对 token 成本不敏感，ATE 的实际买家可能主要是学术界与小团队。
- 吞吐只在 H100 / B200 与跨节点场景测过，更大 FSDP 规模（如 NVL72）未测；Qwen3-30B-A3B 上 124K 对 126K 的吞吐单位与口径需查原文确认。
- agent 自身进化会不会抹平框架间的 ATE 差距，是讲者明确列出的未决问题；ATE Bench 的任务池也需要持续补充新特性任务才能维持可比性。

## 原文金句（1-2句）

> 「我们能不能有一个框架，它是 agent friendly 的，然后同时不希望牺牲我们的 state-of-the-art 性能？」（字幕 08:42–08:52，按干净口径转写）

> 「你可以把 skill 理解为一个用人类语言描述的函数。」（字幕 22:43–22:49，按干净口径转写）

> 「如果全部交给 agent 自己去搞的话，它可能之后会有更多的 reward hacking。」——讲者论证 skill 必须带确定性验收脚本时的话（字幕 25:32–25:37，按干净口径转写）
