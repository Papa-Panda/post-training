# EP152 — 如何在真实 Coding Agent Harness 中接入 RL？（Lego-RL）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP152-lego-rl.html

> 「模型从来不是看到仓库本身，它实际上看到的是 harness 选择构建出来的上下文——哪些会被注入、哪些会被压缩、哪些会被截断、哪些会被重新序列化。」——讲者对整场分享的问题定义（字幕 05:04–05:18，按干净口径转写）

## 元信息

- 期号：152
- 标题：如何在真实 Coding Agent Harness 中接入 RL？（B站标题：《Lego-RL：面向真实 Coding Agent 的 Harness-Native 强化学习方法》）
- BV：BV1AJYx6wEpo
- 时长：01:17:09（讲授约 58 分钟 + Q&A 约 19 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：杜一鸣（华为；字幕自述「来自华为 legal s 团队」，团队正式名称无法从字幕确证，下文以字幕口径标注）
- 相关论文：Lego-RL（讲者结尾称论文、博客、代码、文档、模型与数据均已开源；链接待从论文页补录）
- 相关代码：同上，待补录
- B站链接：https://www.bilibili.com/video/BV1AJYx6wEpo/
- 官网预告：https://qingkeai.online/blog/Lego-RL-Talk
- 字幕原文存档：本地 `transcripts/EP152.txt`（1302 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP152，已存档）清洗提炼；个别专名有识别误差，如 Lego 在字幕中作「legal」、verl 作「WARRR」、Claude Code 作「cloud code」、OpenHands 作「open hands」、SWE-bench 作「SW奔驰」，均以公开资料为准。凡字幕口径无法确证的数字，下文标注（字幕口径，存疑）。

## 一句话总结

讲者指出 coding agent 的 RL 训练有一个被系统性低估的错位：模型实际交互的是 harness 构建出的上下文，而主流做法要么在训练框架内自研一个简化 harness（与部署 harness 不一致），要么回放真实 harness 的归档对话再重新 tokenize（token id 与采样时不一致），两者都在悄悄破坏重要性采样比。Lego-RL 的解法是把 Claude Code / OpenHands / OpenCode 等真实 harness 原样放进 K8s 沙箱运行，用 in-process proxy 在推理边界捕获真实的 token in/out 与 logP，训练只吃策略真正生成的 token；再叠加分阶段 anti-hacking、数据难度筛选与全链路可观测性，在三个 harness 上各训 120+ step 拿到 +6.4/+5.8/+9.4 分的提升。

## 核心

### 背景：训练 harness 与部署 harness 不是同一个东西

讲者先把 coding agent 拆成 model + harness：模型只负责在给定上下文上逐 token 生成下一个动作（搜索、编辑、bash 命令），而 system prompt、system reminder、工具 schema 与序列化、历史压缩/截断/改写、子 agent、轮数与 token 预算，全部由 harness 控制。不同 harness 之间差异巨大，所以「同一个模型在不同 harness 下的表现」本身就是一个变量：字幕给出 Qwen3.5-35B-A3B 在 200K 上下文、温度 0.7 下，不同 harness 间 SWE-bench 口径最高差 6.8 分。

在这个背景下，agentic RL 的两种常见做法都有结构性问题：

1. **框架内自研 harness**：在 trainer 里实现简易的 ReAct 循环、自己给 prompt 和工具序列化函数。训完部署到 Claude Code / OpenHands / OpenCode 时，上下文构建方式完全不同，性能 gap 被放大。
2. **回放归档对话**：rollout 时跑真实 harness、把轨迹存下来，训练时对文本重新 tokenize。问题是 harness 会改写历史——Claude Code 会向过去轮次注入 system reminder、丢弃失败的工具执行；OpenHands SDK 会在上下文写满时压缩历史、规范化 message、重新序列化工具调用（剔空白、改顺序）。重新 tokenize 得到的 token id 不等于生成时采样的 token id，在漂移的 token 上算出的 logP 会破坏重要性采样比；训练不会报错、不崩溃，只是梯度更新方向被严重污染。

### 三类失败：不忠实优化、不可靠执行、不可观测训练

讲者把问题归纳为三类，这也是 Lego-RL 三大特性的由来：

1. **Unfaithful optimization（不忠实优化）**。除了历史改写与重新 tokenize，还有 MoE 路由非确定性：同一个 token 在 rollout 时被路由到专家一，重计算时可能变成专家二，同权重下 logP 不一致，梯度随之失真。
2. **不可靠执行**。一是 reward hacking：agent 会翻 GitHub 历史找现成修复、联网下载参考补丁、甚至直接改 unit test 或篡改评分器来骗取 reward=1，其中「从 git 历史读修复」和「改测试而不是改代码」发生比例最高。二是基建故障的假失败：网络超时（发生在安装、build、下载依赖、下载 grader 各阶段）、高并发挤占带宽（Docker Hub 限流导致镜像拉取失败）、IO 导致的任务假失败、评测器在线下载失败。这些失败会被传导进策略更新；靠模式识别剔除噪声可以做，但 RL 数据宝贵，大量丢弃 instance 会让可用样本骤减。
3. **不可观测训练**。一次 coding agent RL 训练可能长达几天，传统上只能看 reward 曲线和 KL 判断健康与否；但环境侧的问题（agent build 报错、timeout、IO 挤占）不会反映到训练指标上——曲线看着没问题，环境内测可能已经坏了。

### 方法：真实 harness 原样进沙箱，proxy 在推理边界保真

设计原则是训练与 agent 循环解耦：不把每个 harness 的逻辑塞进 trainer（harness 越来越多，框架维护会爆炸），而是让 harness 在沙箱里原样运行，模型以 OpenAI 风格 API 对外提供推理服务，每次模型调用都穿过 in-process proxy。Proxy 做轨迹捕获与 token in/out 捕获：生成阶段记录 token id、response mask、logP 以及 MoE 专家路由；训练直接使用这些已捕获信息，而不是从归档文本重建。这样上下文就是 harness 当时构建的那一份，训练只用策略生成的 token，环境返回的 observation 不进入训练。

系统分三层：

- **环境侧（K8s 沙箱）**：主要采用 K8s 做容器编排（也支持 Docker 与云服务器）。优化包括基于 Nydus 的镜像冷启动（字幕作「NDOS/NDLES」，疑为 Nydus，按需拷贝、几十 MB 即可启动 agent 循环，避免 256/512 并发时全量拉镜像）、image cache 本地缓存、agent runtime 镜像离线 mount。
- **Anti-hacking 分三阶段**：准备阶段控制网络、不让 agent 下载特定站点或用 git log 找金标准答案；执行阶段限制命令与出网策略；评测阶段把 unit test 与 agent 改代码隔离（agent 只能改原代码仓，碰不到测试），防止改测试骗分。
- **训练侧**：基于 verl 做改进（字幕口径），支持 FSDP 等后端与同步/全异步 rollout。任务形式化为：给定用户问题、代码仓库与可执行验证器，harness 按当前状态构建上下文，模型生成 action，环境返回结果拼接回上下文，最终以是否通过全部 unit test 给出 0/1 reward。

### 轨迹保真：token、轨迹、路由三重对齐

Faithful optimization 拆成三件事：

1. **Token 保真**：推理边界捕获原始 token id 与 logP，杜绝重新 tokenize 的漂移。
2. **轨迹保真**：harness 返回的历史 message 可能已被内部改写，需要做 message 匹配、工具匹配、片段匹配与子 agent 匹配，保证训练看到的轨迹与生成时对齐。
3. **路由保真**：用 R3（router replay）回放 MoE 专家路由，消除路由非确定性。讲者给出做与不做路由回放时 logP 的 Pearson 相关对比，路由回放位置一旦偏移，相关系数立刻掉得很低、梯度已经失真；整体训推一致性要求 Pearson 达到 0.99 以上，他们可以做到 0.9996（字幕口径）。

工程上还要处理捕获缓冲区容量不足等存储问题，讲者坦承这类细节都是坑。

### 可靠性：防 hacking 之外，数据筛选比调 clipping 更立竿见影

数据侧流程是：原始数据先转成 Harbor 格式，进沙箱做可执行验证（缺包、版本不一致的 instance 直接筛掉），再做静态过滤保 diversity 与难度筛选，然后 rollout 筛选——每题 rollout 4 次，保留「部分做对」的 instance。消融显示答对 1–3 次的数据训练收益最大，答对 2–3 次次之，随机采样基本没有提升；讲者强调难度区间筛选是整个训练里非常重要的因素。

算法主用 GSPO（序列级重要性采样比），对比 GRPO 在训练后期会严重掉点，GSPO 更适合长程任务；近期也支持了 SGPO（字幕口径）。效率方面，统计显示约 91.3% 的时间花在 agent 执行上（单个任务几分钟到几十分钟），同步训练被长尾轨迹拖死，异步后效率约提升 2.2 倍（字幕作「2.2.5 倍」，存疑）。基建故障导致的失败 instance 不参与策略更新，是常规操作；不同 harness 因故障/timeout 失败的比例也不同（system prompt 长度差异等）。

### 可观测性：把训练过程做成产品

端到端闭环分五步：data preparation → run validation → 训练与监测 → Live UI → human review，再循环。讲者把训练前检查抽象成一个 agent plugin / skill（字幕作「RL check」式 skill），自动识别环境与配置问题（模型与 tool parser 不匹配、端口占用、CPU 连通性等），因为「我们用 vibe coding 做开发，训练也应该尽量自动化」。

Live UI 的核心面板：

- **Failure breakdown**：每个 step 统计正常执行 / timeout / 环境崩溃等失败原因占比，环境出问题立刻能看出来；
- **Trajectory viewer**：实时看轨迹行为变化（曾靠它发现「轨迹一两轮就结束」其实是工具格式与 tool parser 不一致）；
- **In-batch reward distribution**：每个 batch 内答对 1 次到 7 次的 instance 比例分布，健康训练中该分布会持续向高答对次数迁移；
- **Task grade**：跨 epoch 追踪同一 instance 的答对次数变化，健康训练里「答对变多」的 task 数量远多于变少的；
- 另有 AI-assisted analysis（自定义 prompt 做轨迹与训练状态分析），以及训推一致性、KL、entropy 等常规指标。

### 实验：三 harness 各训 120+ step

设置：Qwen3.5-35B-A3B、200K 上下文、温度 0.7，SWE-bench 口径评测；分别在 OpenCode、OpenHands SDK、Claude Code 三个 harness 上各训练 120 多个 step。结果：training reward 波动上升、validation 分数明显增长、entropy 处于平衡或偏上升状态（讲者强调 coding agent RL 里 entropy 一般不坍缩，但取决于模型与任务）；response 长度上 OpenHands 增长明显，Claude Code 与 OpenCode 大致维持在 55–65K。与基线相比 Lego-RL 分别提升 6.4、5.8、9.4 分（字幕口径，具体对应哪个 harness 字幕未逐一指明）；字幕还提及与下一代基座及快手 KAT-Coder 一类后训练模型做了对比（字幕口径，未展开）。

行为分析：拿 420 条轨迹对比训练前后，训练后模型更倾向于编辑文件后重新读一遍验证、主动运行自己的测试，平均轮数增长明显；「自我验证」类行为占比从 73.6% 提升到 98.1%（字幕口径）。

讲者自列的局限性：每个 run 只训单一 harness（混合 harness 训练正在做）；二值 0/1 reward 缺中间过程的稠密 credit assignment（后续考虑 process reward / rubrics）；评测还主要在相对早期的 SWE-bench 上，其他 benchmark 较少；系统收益依赖基建优化程度；训练长度仍较长。

### Q&A 要点

- **harness 压缩/重写历史后如何保证 token 级绝对对齐？** 推理边界已捕获 token id 与 logP；压缩结果视作「环境观测」拼接回已捕获上下文；再用 prefix 对比找到最后一个产生差异的位置，之后采用已捕获的 token，保证对齐。
- **GSPO 的不对称 clipping 怎么调？** 讲者答得很坦白：上下界调大确实容易梯度爆炸/消失，但最终影响没有想象中大，他们仍用 GSPO 原始设计的上下界（字幕作「三的一负四跟四的一负四」，约 $3\times10^{-4}$ 与 $4\times10^{-4}$ 量级，字幕口径存疑）；真正立竿见影的是数据层面的筛选，「数据层面的筛选要比直接去调 clipping 更加立竿见影」。不对称 clipping 值得后续探索。
- **沙箱防作弊怎么做？** 评测与改代码隔离 + 分阶段管控：准备阶段控网络、禁 git log 找金标准答案；执行阶段限制命令与出网；评测与 agent 执行隔离，agent 碰不到 unit test。
- **Task grade 与 in-batch reward distribution 的区别？** 前者是跨 epoch 同一 task 的答对变化（task 粒度看行为变化），后者是单个 batch 内答对次数的分布（batch 粒度看训练健康度）；讲者补充早期没做数据过滤时 batch 内全是全对或只对一次的 instance，会造成 batch 内偏置进而污染 advantage 计算，过滤后分布明显变均衡。
- **环境崩溃怎么判定？分错会不会引入偏差？** Web UI 统计每 step 正常执行（约 90%）/timeout/环境崩溃占比；曾在第 13 步发现环境失败率飙到 30%–40%（字幕作「40%到30」，存疑），根源是网络导致 agent build 出错。环境崩溃有固定模式、一般不会分错；但早期未识别的环境崩溃确实引入过训练偏差——「很多时候 reward 怎么也上不去，其实是环境出错」。
- **OpenHands 训出的模型换到 Claude Code/OpenCode 会打折吗？** 正在做混合 harness 训练与跨 harness 评测；个案测试过几次问题不大，但需要具体分析，不能排除过拟合单一 harness 格式。
- **Entropy 指标反映什么？** 熵过低意味着模型行为过度确定、很难再训上去；讲者给出危险信号约为小于 0.1（字幕另有「小于等于 0.5」说法，前后口径不一，存疑），他们几个模型（30B A3B 级）的健康区间约在 0.15–0.3，曲线一般持平、微升或先升后降但不会特别低。
- **训练数据从哪来？** 主用 OpenSWE（字幕作「open s w e」），3 万多条经静态与 rollout 筛选留 2699 条；早期用 SWE-bench 数据筛出 1600 多条训练效果不明显，讲者猜测基座模型已在类似分布上训练过。
- **Proxy 方案性价比如何？** 没有与「框架内重构 agent 循环」做过严格对比；但 proxy 与捕获的开销相对 agent 推理时间（几分钟到几十分钟）很小，存储占比也不大；且 harness 种类越来越多（coding、web search 等），proxy 方案适配多 harness 的成本远低于在框架内逐个重构，讲者认为性价比 OK。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 跨 harness 性能差 | Qwen3.5-35B-A3B，200K 上下文，温度 0.7 | 不同 harness 最高差 6.8 分 | 字幕 |
| 训推一致性 | logP Pearson 相关 | 0.9996（要求 0.99 以上） | 字幕 |
| agent 执行时间占比 | 单 step 时间构成 | 91.3% | 字幕 |
| 异步相对同步效率 | — | 约 2.2×（字幕作「2.2.5 倍」，存疑） | 字幕 |
| 训练步数 | OpenCode / OpenHands SDK / Claude Code | 各 120+ step | 字幕 |
| Lego-RL 分数提升 | 相对基线，SWE-bench 口径 | +6.4 / +5.8 / +9.4 分 | 字幕 |
| 自我验证行为占比 | 420 条轨迹，训练前 → 后 | 73.6% → 98.1% | 字幕 |
| 训练数据规模 | OpenSWE 3 万多条筛选后 | 2699 条 | 字幕 |
| rollout 筛选策略 | 每题 rollout 4 次 | 答对 1–3 次收益最大 | 字幕 |
| GSPO clipping 上下界 | GSPO 原始设计 | 约 $3\times10^{-4}$ / $4\times10^{-4}$ （字幕口径，存疑） | 字幕 |
| entropy 健康区间 | 30B A3B 级模型 | 约 0.15–0.3 | 字幕 |
| 环境崩溃异常案例 | 某次训练第 13 步 | 失败率飙至 30%–40%（字幕口径，存疑） | 字幕 |
| response 长度 | Claude Code / OpenCode | 约 55–65K token | 字幕 |

## 可迁移

- 「训练 harness ≠ 部署 harness」应该成为 agentic RL 的默认检查项：凡是在框架内自研 agent 循环、或回放文本轨迹重新 tokenize 的方案，都要先回答 token id 是否与采样时逐一对齐、重要性采样比是否还成立；对不齐时训练不报错、只是梯度被污染，这是最阴险的一类 bug。
- 推理边界捕获（in-process proxy 抓 token id / logP / mask / MoE 路由）把「保真」从框架内重构中解耦出来：harness 越多，这套「原样运行 + 边界捕获」的性价比越高；同理 router replay 是 MoE 模型做 RL 时训推一致性的必备件，可以用 logP 的 Pearson 相关（目标 0.99+）当验收指标。
- 数据难度筛选（rollout 4 次保留答对 1–3 次）在本期里被讲者排在算法调参之前：batch 内全对/全错的 instance 会造成分布偏置并污染 advantage，先修数据分布再调 clipping。
- 可观测性清单可以直接抄：failure breakdown、trajectory viewer、in-batch reward distribution、task grade 四件套，核心思想是把「环境健康度」做成与 reward 曲线平级的一等指标——环境崩溃不进训练指标，是这类系统最常见的假象来源。

## 疑问 / 下一步

- 训推一致性 0.9996 是在哪个粒度上算的（整条轨迹 logP 还是逐 token）？路由回放位置偏移时相关系数掉到什么量级，字幕只给了定性描述，值得查 Lego-RL 论文原文核对。
- +6.4/+5.8/+9.4 分别对应哪个 harness、基线具体是什么（同基座未训练还是自研 harness 训练）？字幕未逐一展开，需要论文表格确认。
- 答对 1–3 次最优是在 rollout 4 次、二值 reward 下的结论；换成稠密 process reward 后难度区间的最优点会不会移动，讲者把稠密 credit assignment 列为未来工作，可以持续跟踪。
- Nydus 冷启动、image cache、runtime mount 的具体加速比字幕只说「提升非常明显」，没有数字；Lego-RL 已开源，代码与文档里应该有可核对的实现细节。

## 原文金句（1-2句）

> 「模型从来不是看到仓库本身，它实际上看到的是 harness 选择构建出来的上下文。」——讲者的问题定义（字幕 05:04–05:12）

> 「数据层面的筛选要比直接去调这个 clipping 更加立竿见影。」——Q&A 中被问 GSPO clipping 时的回答（字幕 62:17–62:33）
