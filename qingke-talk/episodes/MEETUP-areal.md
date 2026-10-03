# MEETUP — AReaL：可扩展和可定制的面向智能体的强化学习
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/MEETUP-areal.html

> 「RL 框架的 API design 不是一个 bottom-up 的过程，不是从硬件开始设计的，而是一个 top-down 的过程：你有什么样的算法、什么样的需求，就设计什么样的 API。」——讲者在讲 AReaL 设计哲学时的判词（字幕口径）

## 元信息

- 活动归属：2025-08-24「LLM RL & RL Infra」线下 Meetup（青稞社区线下分享，非青稞Talk 编号期）；B站合集 sid=6759789
- 标题：AReaL：可扩展和可定制的面向智能体的强化学习
- BV：BV1HCkoBBEzL
- B站链接：https://www.bilibili.com/video/BV1HCkoBBEzL/
- 时长：约 1878 秒（约 31 分 18 秒；正片讲授约 29 分钟 + 现场 Q&A 约 2 分钟，字幕末条至 31:16）
- 提炼日期：2026-10-02
- 分享嘉宾：傅伟（字幕自述：博士五年级、还有一年毕业，研究方向为 deep reinforcement learning 与 distributed systems；字幕中学校与导师姓名识别不清，未确证，见文末疑问节）
- 讲者自述背景：该工作为其在蚂蚁实习期间所做（字幕口径）
- 相关工作：AReaL —— 讲者口径为「a large-scale asynchronous reinforcement learning system」；其论文版标题偏向 LLM reasoning（字幕原话：paper 名字不叫这个、是 for LLM reasoning，因 agentic 话题热度而在本场改从 agent 角度讲），arXiv 编号字幕未给出
- 相关代码：已开源（讲者在片尾给出 paper link、code 与微信群入口；具体仓库地址字幕未逐字给出，未确证）
- 字幕原文存档：本地 `transcripts/MEETUP-AREAL.txt`（888 条，带时间戳）

> 📝 提炼方式说明：本纪要为**字幕实录版**，基于 B站 AI 字幕原文（已存档）清洗提炼。字幕中框架名、人名多有识别误差（如 AReaL 被识别为「a real / ERROL / ARIO」等、rollout interruption 被识别为「road interruption」、continuous batching 被识别为「CONTINUBATTING」），均按上下文还原；凡数字均标来源为「字幕」，讲者口头自纠处单独注明。未查论文原文核对，故论文口径的精确数字（如 benchmark 分数）一律不引，只记讲者现场口径。

## 一句话总结

傅伟把 AReaL 讲成一个「先把系统跑满、再把代码放开」的答案：针对 agent RL 的两大系统痛点——长短轨迹混杂导致的推理设备空等、decoding 随 GPU 数增加由 compute-bound 转 memory-bound 导致的不可 scale——AReaL 用训推分离 + fully asynchronous 训练（continuous batching、生成与训练 overlap、rollout interruption）把推理与训练两侧利用率同时打满，并以显式 off-policy 控制与解耦 PPO 目标承接由此带来的多版本轨迹；在 API 层它反 trainer 框架而行，用单文件 orchestration + workflow 抽象 + 后端组合（composition over inheritance）让算法开发者只写 user code、系统开发者只写后端，讲者自称这套形态不是「三明治」而是「tapas」——底下一片统一面包，上面放什么都行。

## 核心

### 背景：两条 scaling 曲线的交汇，与 agent workflow 的三维定义

讲者用两条历史曲线开场：一条是参数规模曲线（ResNet → recurrent models → Transformer → GPT 系列），特征是参数变大但 inference compute 长期不显著增加，直到近期才意识到要把更大的推理算力喂给更大的模型；另一条是经典 RL 曲线（Atari DQN、AlphaGo），特征是小模型（十兆到几十兆参数）+ MCTS 式推理算力扩展。两条曲线近期的交汇点，就是今天做 LLM RL 的全部理由，讲者戏称交汇终点是 super artificial intelligence（原话声明「不负责任的预测」）。

落到 agent，讲者把 agentic workflow 定义为三种范式的混合，缺一不可：

1. **Long-context reasoning**：模型在调用 tool 或返回 response 之前先做长时间思考（以 OpenAI o1、DeepSeek-R1 为已知证据）；
2. **Tool interaction**：以 Claude Code 为例，一步用户指令内会混用 read、write、shell、Python 等多种 tool 调用；
3. **Multi-turn user interaction**：同一例子，用户需多轮下达指令任务才完成。

他随即给了一个只测不训的 scaling 证据：用 QwQ 做 search agent，随着 search 轮数增加，其在 GAIA 与 xbench 上的表现呈线性/单调提升——由此推出做 agent RL 必须放开轮数上限，而这正是后面两个系统问题的来源。

### 问题一：长短轨迹混杂，推理设备在等最长的那条

讲者给的真实统计是：search agent 的轨迹长度极端分化，最长一条可达 140K token、最短不足 10K token；简单题搜 10 分钟结束，难题打满 128 轮上限要搜 2 小时。当一个 batch 里混着大量简单题与少量难题时，GPU 的大部分时间在等最长 response 返回——这是他定义的第一问题，inference device under-utilization。

若以轨迹长度记号形式化（笔者按讲者口述整理）：设 batch 内轨迹长度为 $L_i$ ，同步式生成必须等到 $L_{\max}$ 全部生成完毕才能进入训练步，即

$$T_{\mathrm{step}} \propto L_{\max} = \max_i L_i$$

而讲者给的量级是 ， $L_{\max}$ 可达 140K token ，多数 $L_i$ 不足 10K token——同步屏障的成本由最长尾决定，与平均长度无关。这是 fully asynchronous 设计要消灭的第一件事。

### 问题二：decoding 的 compute-bound / memory-bound 分界，决定了能 scale 到几张卡

第二问题是 hard to scale up：集群翻倍后，同样实验能否用一半时间跑完？讲者指出 RL 推理的 decoding 段存在范式切换——每卡 batch 足够大时是 compute-bound，GPU 数增加、每卡 batch 被摊薄后转为 memory-bound，大部分时间耗在 memory IO 上。两者交界被他命名为 critical GPU count（或 critical batch size per GPU）：界之下可 linear scale，界之上再加卡也缩短不了推理时间。

两个关键判断（字幕口径）：

- 该临界值**与模型大小无关**，只由 GPU 的算力/带宽比决定；讲者举例 A100 该比值约 160、H100 约 600（字幕口径）；
- 由此普通 RL 实验很快撞墙：讲者口头先说 A100 约 16 个 data parallel 即到界，随即自纠为「A100 应该是 60 个 GPU、H100 是 16 个 GPU」（字幕口径，含自纠，最终数值未确证，见疑问节）。破界的唯一出路是加大模型本身（tensor parallel / pipeline parallel 吃卡），而不是继续堆 data parallel。

### 解法：fully asynchronous training —— 训推分离 + 两层 asynchronous

AReaL 的系统解法是 fully asynchronous training，讲者强调这不是 AReaL 首创、「大家现在都这么做」，他只讲 AReaL 里怎么落地。结构分两半：

**训推分离**：推理与训练用不同 device，且可独立 scaling——推理 scale 不上去时，训练侧仍可单独扩展。

**Asynchronous 的两层含义**（讲者明确分层）：

1. 第一层是 continuous batching：持续往 inference server 灌 request，不等整 batch 齐（他点名先行分享的 ROLL 作者也讲了同一机制）；
2. 第二层是 overlap：下一步推理与上一步训练重叠执行，目标是 full GPU utilization。

工作流程（讲者以 4 卡生成 4 个 request、batch size 为 4 的玩具例子走了一遍）：凑满一个 batch 即启动第一步训练；训练未结束、模型版本未更新期间，推理侧继续灌新 request 保持满载；训练产出新版本后立即同步到推理 server。此时未生成完的 request 怎么办——**rollout interruption**：打断它，用新参数续生成同一条 request。于是单条轨迹天然由多个模型版本拼成，按讲者口述形式化即：

$$\tau = \tau^{(v_0)} \oplus \tau^{(v_1)} \oplus \cdots \oplus \tau^{(v_K)}$$

其中 ， $v_0, v_1, \dots, v_K$ 为同一轨迹不同片段所属的策略版本 ， $\oplus$ 表示片段拼接。好处是双侧的：推理/训练利用率同时打满，且超长 request 不再充当 straggler 阻塞全场。

### 算法承接：off-policyness 是有代价的，AReaL 做了两件事

讲者不回避代价：fully asynchronous 把设备利用率打满的同时，也改变了 PPO 的算法语义——轨迹由多版本策略生成，天生 off-policy。AReaL 的两个承接手段（细节他推给论文，本场只点名）：

1. **Explicit off-policy control**：显式控制轨迹的 off-policyness 程度；
2. **Decoupled PPO objective**：解耦的 PPO 目标（讲者声明非 AReaL 发明、是应用已有工作）。

验证分两档。小规模：1.5B 模型做 math reasoning，在 off-policyness 取 0（同步）、1、2、4 等档位下 learning curve 基本重合——即算法表现未被异步性吃掉（字幕口径，档位数字按字幕转写）。大规模：基于 QwQ-32B 的端到端 search agent「A-Searcher」，讲者强调其端到端性与已有 agent 的区别——已有做法常在 search 后接一个并非被训模型自产的 summarization 环节，AReaL 是全程端到端训练；收益是行为层面的 truthful：搜不到就承认搜不到，而不是编一个号码出来。成绩口径：GAIA 与 xbench 上均为端到端 RL 训练的 SOTA，且只花了 8000 GPU hours；他给的对照是年初 DeepScaleR（1.5B 模型、4 卡、3600 GPU hours）——32B 模型用约两倍 GPU hours 拿到 search agent SOTA，被他用作异步系统效率的证据（均为字幕口径，benchmark 具体分数未给出）。

### API 设计（上）：orchestration —— 没有 trainer 的单文件训练脚本

后半场的真正主角是 API。讲者的立场先行：RL 框架的 API 设计是 top-down 而非 bottom-up——有什么算法需求、尤其 agentic RL 需求，就长什么 API；AReaL 现有 API 是专门为 agentic RL 做过一版 refactor 的结果。他现场走了一个最小 RLVR 例子（math / coding 类可验证奖励任务）：workflow 的骨架是创建 AReaL 版 OpenAI client → 调 chat completion 收集多条 response → evaluate → export；再串上训练侧（load config、data loader、remote SGLang engine、FSDP PPO actor、rollout → compute log-prob → PPO update → update weights）即是一个能跑的完整 GRPO 流程。

由此提炼出 AReaL 的最高层抽象 **orchestration（算法编排）**，即把一个完整 RL 流程全部暴露在同一个文件里供用户改：

- 数据集处理（不同格式不同处理，加一行结束）；
- 模型初始化（想加 reward model、reference model、critic，各加一行）；
- Training loop 本身（双 data loader 混采、先 SFT 再 RL、权重更新策略，全部可见可改）；
- Agent workflow（与上面同文件）。

讲者自评这是一个 trainer-free 的实现：不像 TRL、OpenRLHF 那样有 trainer 抽象，就是「一坨东西拼在一起」，叫训练脚本亦可。优劣他也讲明：好处是好改，坏处是不能只改 config 完成小功能、必须写代码——对开发者友好，对纯配置用户不友好。

### API 设计（中）：算法实现与后端解耦 —— composition 而非 inheritance

第二层是 algorithm implementation（forward/backward 怎么组织、loss 怎么定义、forward 结果怎么处理）。讲者先立「不应该怎么做」：不能给 Megatron、DeepSpeed 等每个后端各写一套 PPO——逻辑相同、后端不同，维护成本爆炸。AReaL 的做法是定义一个不继承任何东西的 PPO actor 类，把 engine 作为参数传进来，actor 只调 engine 的 `forward` 与 `train_batch` 两个基本接口来实现 compute log-prob、mini-batch update 等算法功能（critic 侧同理，如区分 process supervision reward 与 outcome reward）。接新后端只需写几行胶水：继承 FSDP engine 或 Megatron engine、构造时把 engine 传入 actor——同一套算法代码即支持多后端。讲者自己给了类比：这就是 Rust trait / Go interface 式的组合。收益是算法开发者不需要懂分布式逻辑即可写算法（他对比了在 verl 里实现同类算法时对分布式知识的要求，字幕口径）。

第三层是后端统一接口：训练侧是标准的 forward/backward，推理侧的核心接口是 `generate` 与 `update_weights`，外加一对关键方法 `submit` 与 `wait`（下节展开）。

### API 设计（下）：agent 抽象、sync 到 async 只改一行、与「tapas」隐喻

Agent 在 AReaL 里通过 rollout workflow 定义：一个只规定签名 `run_episode`（入参为推理引擎与 prompt、内部逻辑随便写、返回值甚至可以是 OpenAI chat completion 格式而非 tensor dict）的空壳。用户写好 workflow 后通过推理引擎的 `submit` 提交，框架自动做 batching 与异步执行——search agent、SWE agent、CLI agent 都是同一抽象的不同填充。

由此带出他最想卖的几个用法：

- **Sync → async 只改一行**：把 rollout batch 换成 prepare batch 的写法，本质仍是调 `submit` 与 `wait`，同步版调通后改一行即得异步加速；
- **与 DAPO 式 dynamic filtering 天然统一**：异步逐条返回，过滤也是逐条决定——传一个 `should_accept` 函数即可，讲者给的示例逻辑是轨迹平均 reward 大于 0 则 accept、小于等于 0 则拒绝，即

$$\mathrm{accept}(\tau) = 1 \iff \bar{r}(\tau) > 0$$

其中 ， $\bar{r}(\tau)$ 为轨迹 $\tau$ 的平均 reward（笔者按讲者口述形式化）；

- **Workflow 里可以做生成期协作**：团队实验过让同一 prompt 的多条并行 response 互相协商——先得出确定结果的短 response 可以打断其他 response、把结果共享过去，多条「思考流」之间做 negotiation，这类花活在 workflow 抽象里实现成本很低；
- **Pluggable 使用**：可以只把 AReaL 的 remote SGLang engine import 到框架外，给定 server 地址后独立跑 agentic rollout，拿到的 token id 可 dump 做 SFT、也可直连 RL 训练；workflow 内部 import ROMA、MCP、LangGraph 等现成生态都不影响可训练性，前提是经 AReaL 的 OpenAI 兼容 client 发请求。

最后的隐喻是全场记忆点：多数 RL 系统像**三明治**——上下两片框架面包（distributed launching、batch rollout execution、data flow、后端集成）夹着中间的 user code（算法编排 + 具体 agent workflow），想改中间必须先掀开上面那片面包，典型如 Ray 的 placement group 这类算法同学一开始根本不懂的东西。AReaL 自称是**塔帕斯（tapas）**：下面仍是一片统一面包，上面没有盖子，用户直接把 cheese、三文鱼、任何东西摁上去——所有 orchestration 在同一文件可见可改，而底下的训练/推理后端保持统一。算法开发者在上面换菜，系统开发者在下面换后端，互不阻塞。讲者用前面 32B 的 A-Searcher 实现自证：它本身就是这样一个单文件（开源可查），核心 workflow 不过是多轮地「备 query → 处理结果 → 调 tool」。

### 局限自述与 Q&A

讲者对缺点的交代异常直白：当前开源版只支持 FSDP，Megatron 不支持（在路上），FSDP 内部的 context parallel、tensor parallel 也在路上——「现在想训 70B 模型估计还是没戏，还得等一等，我们最近确实在加班」。其他能力点到为止：支持 Ray 与 Kubernetes（K8s 版当时未开源、之后开源，字幕口径）。

现场 Q&A 只有两问，讲者先立规矩（不接受拉踩其他框架、不接受贬低演讲者本人，「可以去发小红书」）：

- **会考虑联邦学习式的异构/分布式设备训练吗？** 讲者答与 AReaL 正交：当前面向同构 H100/H800 集群，把统一后端扩展到分布式设备做联邦学习是下一步的好方向；
- **下一步最重要的事是训大模型吗？** 答：先服务好用户，用户要什么加什么，「有问题提 issue、微信群 call 我」；
- **Rollout interruption 在代码里怎么体现（提问者没在示例里找到）？** 讲者答在推理引擎的 `generate` 方法内：interruption 由外部（训练侧同步参数时）触发，经 SGLang 实现（非 AReaL 自研），rollout 返回后引擎检查 finish reason 是否为 abort，是则等待后重新提交续生成，直到 EOS 或触及最大 token 上限。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| Search agent 轨迹长度（最长） | 同批最短轨迹不足 10K token | 最长可达 140K token | 字幕（问题一统计） |
| 单条轨迹耗时分化 | 简单题约 10 分钟结束 | 难题打满 128 轮上限约 2 小时 | 字幕（问题一统计） |
| 算力/带宽比（讲者举例） | —— | A100 约 160、H100 约 600 | 字幕（问题二） |
| Critical GPU count（口头自纠后口径） | data parallel 规模 | A100 约 60 卡、H100 约 16 卡即到 scale 上限（与模型大小无关） | 字幕（含讲者自纠，最终值未确证） |
| 异步算法验证（小规模） | 1.5B 模型 math reasoning，off-policyness = 0（同步） | off-policyness 取 1、2、4 时 learning curve 与同步基本重合 | 字幕（实验部分） |
| A-Searcher 成绩 | 基于 QwQ-32B 端到端 RL 训练 | GAIA 与 xbench 均为端到端 RL 的 SOTA（具体分数未给出） | 字幕（实验部分） |
| A-Searcher 训练成本 | 32B 模型 | 8000 GPU hours | 字幕（实验部分） |
| 对照成本（讲者自选参照） | DeepScaleR，1.5B 模型、4 卡 | 3600 GPU hours | 字幕（实验部分） |
| 支持集/超参类经验值 | 本场未涉及 | —— | —— |
| 当前开源后端覆盖 | Megatron 未支持（在路上） | 仅 FSDP；FSDP 内 context/tensor parallel 在路上；70B 训练暂不可行 | 字幕（局限自述） |

## 可迁移

- 对 RL infra 的直接清单一：为 agent RL 选/建系统前先量两个数——轨迹长度的 $L_{\max}$ 与中位长度之比（决定同步屏障的浪费程度，讲者案例里是 140K 对不足 10K），以及目标集群的 critical GPU count（由算力/带宽比决定、与模型大小无关）；前者大或后者小，fully asynchronous 就不是优化项而是必需项。
- 对 RL infra 的直接清单二：评估任何异步 RL 系统时，把「off-policyness 如何显式控制、PPO 目标如何解耦」列为算法侧的对等审查项——利用率打满只是系统侧的一半，轨迹多版本拼接（ $\tau = \tau^{(v_0)} \oplus \tau^{(v_1)} \oplus \cdots$ ）带来的 off-policy 程度必须有可调的闸门，并用同步档位（如 off-policyness = 0）做 learning curve 对照验证后再放大规模。
- 框架 API 的设计判据（讲者的 top-down 主张）：编排层是否单文件可见可改（数据集处理、模型拼装、training loop、agent workflow 同处一文件）、算法层是否与后端以组合而非继承解耦（actor 只依赖 `forward` / `train_batch` 两个接口）、agent 层是否只规定 `run_episode` 签名而不限制内部实现——三条任一不满足，后续每加一个 agentic 玩法都要先掀框架的「上面包」。
- 调试/实现纪律：rollout interruption 这类机制不要自研，优先复用推理引擎（如 SGLang）已有的 abort/续生成能力，框架侧只做 finish reason 检查与重提交——讲者在 Q&A 里明确这条边界。

## 疑问 / 下一步

- 讲者学校、导师与论文正式出处未确证：字幕自述段识别严重失真（「jr新剧院5年级」「导师是5E」），只可确认博士五年级、研究方向为 deep RL 与 distributed systems、该工作为蚂蚁实习期间所做；AReaL 论文的 arXiv 编号、作者列表与正式标题（讲者称论文版偏 LLM reasoning）字幕均未给出，需查论文页补录。
- Critical GPU count 的最终数值未确证：讲者先说 A100 约 16 个 data parallel 到界、随即自纠为 A100 约 60 卡 / H100 约 16 卡，两版口径矛盾，需回看其 slide 或论文核对；A100 约 160、H100 约 600 的算力/带宽比同为字幕口径。
- A-Searcher 的 GAIA / xbench 具体分数、评测配置与数据量未给出，只有「端到端 RL 的 SOTA」定性口径与 8000 GPU hours 成本；DeepScaleR 的 3600 GPU hours 对照亦为讲者自选参照，不能直接当同口径效率结论。
- Explicit off-policy control 与 decoupled PPO objective 的具体形式讲者明确推给论文（「paper 里面都有，这里不细讲」），本纪要只能记其名；off-policyness 取 1、2、4 三档的 learning curve「基本重合」亦无误差范围，需论文图核对。
- 片尾给出的 Q3（至 10 月 31 日）development roadmap、paper link 与代码仓库地址字幕未逐字给出，开源仓库的确切地址未确证。

## 原文金句（1-2句）

> 「我们把一个 RL 系统表示成一个三明治……但 AReaL 不是一个三明治，它是一个塔帕斯：下面还是一片面包，但是上面你可以加任何的东西。」——讲者对 AReaL API 形态的核心隐喻（字幕口径，按干净转写整理）

> 「他缺点特别多。比如说他现在开源的只支持 FSDP，Megatron 也不支持、在路上……所以说现在大家想训一个特别大的模型、想训一个 70B，那估计还是没戏，还得等一等。」——讲者在局限自述中的原话（字幕口径）
