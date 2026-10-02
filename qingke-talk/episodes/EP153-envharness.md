# EP153 — EnvHarness：Agent 与环境如何「左脚踩右脚」实现自进化

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP153-envharness.html

> 「环境就是训练数据。只要环境不撒谎，我们就可以像扩展数学公理一样扩展训练信号。」——讲者自进化观点的点题句（字幕口径归纳）

## 元信息

- 期号：153
- 标题：EnvHarness：Agent 与环境如何「左脚踩右脚」实现自进化
- BV：BV1ZqYX6YE3T
- 时长：01:08:03
- 提炼日期：2026-10-02
- 分享嘉宾：黄承松（Google 实习期间完成本工作；此前做自演化系列 R-Zero / G-Zero；字幕 [43:14] 主持人致谢时点名，姓名口径以论文作者页为准）
- 相关论文：*Environment Harness*（副标题字幕作 "Awaking static words for agent learning"，拼写以论文页为准；代码与 24 点 playground 已开源，链接以论文/官网页为准）
- B站链接：https://www.bilibili.com/video/BV1ZqYX6YE3T/
- 字幕原文存档：本地 `transcripts/EP153.txt`（1506 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP153，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如论文副标题、playground 演示的牌面数字在字幕中多处破碎，均已标注字幕口径或略去）。凡讲授口径与论文版可能不同，以下标注（字幕口径）。

## 一句话总结

黄承松的 Environment Harness 给「静态环境」加一层可编程外壳：用 Stage / Contract / Chain 三类不改 task 与 verifier 的组件调难度、改观测、拼长程，再让 Environment Rigger 这个 agent 读 policy 轨迹、写 harness 代码、以 0.4–0.6 成功率带做闭环 accept/reject/refine——实证仅 50 个 SWE-Lite 基础环境迭代改造后训练效果超过 300 个真实环境，而每个改造环境的成本最低约 \$0.5 、仅为人造环境的四十分之一。

## 核心

### 出发点：训练信号的天花板是环境

讲者的切入点是 agent 训练里一个常被忽视的不对称：policy 在学，环境却是死的。由此有三态失效——任务太难则 policy 永远采不到正确轨迹、无梯度可言；太易则全解、无信号；更麻烦的是难度会随 policy 变强而失效。环境合成每域单独设计 pipeline、成本高且不迁移；人工造环境贵得离谱：Terminal-Bench 每个环境约 15.9 小时（制作 7–8 小时 + 人工验证约 3 小时、可达三轮），约 100 个环境近 1400 小时；Apex Agents 33 个环境每个约 500 小时、共约 16000 小时。在 outcome-only reward 下，这意味着 GRPO 的组内经常全错或全对、没有可用信号。讲者把问题归到：训练用的「数据」是环境，而环境是唯一没有跟着 agent 一起变的变量。

### 环境的定义：只改三件套中的壳，不改任务与验证器

讲者把环境拆为三件套：**task**（要完成什么）、**sandbox**（初始状态、action space、可观测信息、状态转移，即一个马尔可夫式的环境定义）、**verifier**（判定成败）。Environment Harness 的铁律是：**绝不改 task 与 verifier**。理由很硬——改了 verifier，就无法保证 reward 仍然正确，这是所有环境合成方法的通病。Harness 的定位是类比 Agent Harness（skill / memory / context control 之于 LLM 本身）：在环境外面加一层可控、可定制的壳，不动底层。这层壳由三个组件构成：

1. **Stage（开局布置）**：在 policy 开始行动之前，由另一个 agent 用环境原生的合法 action 预先改变初始状态。可以调难：把要洗的杯子藏进抽屉并关上，于是 policy 多学「找—开—取」这段前置行为；也可以调易：预先把杯子拿好，缩短有效 horizon。代码环境同理：删一行、改一行代码即改变开局。
2. **Contract（行为约束）**：对 action / observation / transition 施加 filter。讲者举例：禁掉 `goto` 这类直达动作，只留前后左右移动，逼出导航能力；或把 clean 升格为含「拿起+清洗」的高层 wash 动作、缩短 action 序列；transition 层可以规定「当前结果已注定无法达标就直接报错」；observation 层可以让 agent 只看到操作过程而非真实状态。本质上它是在 model 输出与环境 action 之间插一层 bridge/filter。
3. **Chain（任务拼图）**：把多个环境串联/拼成有向图——完成 A 跳到 B，跨 web / code / 现实环境，带记忆与语义信息一起过去；可以约定全部完成才算过，或完成一半给半分。目的是造出真实工作区里那种 long-horizon 任务。

组件组合**有序、不可交换**：先用 Contract 禁掉加法、再 Stage 就不能靠加法改初始状态；反过来先 Stage 再 Contract 则可以。24 点例子（6666）把三者串了一遍：Stage 先把前两个 6 相乘，初始状态变成 [36, 6, 6]（字幕作「30 666」，按字面理解应为 36、6、6），horizon 立刻变短；Contract 禁掉加法，逼出非平凡解法；Chain 规定做完 6666 跳到 3388 继续。

### Environment Rigger：写 Harness 的代理与闭环判据

光有组件不够，还得有人决定给哪个环境加什么。**Environment Rigger** 是另一个 agent（LLM），工作流是：读 policy 在环境里的轨迹 → 对照模板写出 harness 代码草稿 → 拼回环境让 policy 重跑 → 按客观条件判断是否 accept / reject / refine。它可以一次写 stage+contract 的组合，要求也可以全交 rigger 自定。讲者给了真实工程味很重的例子：SWE 场景里 agent 总在提交前不跑 unit test，harness 就规定「不跑 unit test 提交即判失败」——这是把缺陷诊断直接写成环境约束。

采纳判据讲者讲得很直白：5 次 rollout 里对 2 次或 3 次，即成功率落在 **0.4–0.6** 区间。理由是 GRPO 类算法在正确率约 50% 时训练信号最有用；且所有 benchmark 共用这一个固定区间，不逐个调。他原话是「选五这个数字没有别的原因，就是写了五」。Rigger 的候选接受率约 40%–60%（不同 benchmark 在 30%–80% 之间）；产不出组件时回退原环境继续训。

### 实证：50 个基础环境迭代改造 > 300 个真实环境

主实验是 skill optimization（从轨迹里蒸馏 skill 的 harness-optimization 流程），覆盖 coding、office QA（表格类）、document QA 等域，结论：EnvHarness 改造过的环境能给出比真实环境更好的训练信号，因为改造是针对当前 policy 的能力分布与缺陷定制的。关键的可迁移证据两条：

1. **下游无污染**：SWE-Lite 的 50 个基础环境上多轮迭代 EnvHarness，训练效果超过 300 个真实环境；评测在未加 harness 的 held-out 测试集（SWE-bench Verified 类）上做，不存在对 harness 本身过拟合的问题。Chain 蒸馏出的 skill 多在教 agent「如何适应一个更新更好的环境」，迁移到无 chain 的真实环境仍有效且省训练步数。
2. **增益主要来自诊断而非堆数据/堆难度**（论文实验）：同样的「加更多数据」曲线打不过 harness 曲线——harness 是在具体失败模式上动手脚，不是把分布平移。

还有一条容易被低估的机制论证：难任务上把 rollout 次数从 8 加到 80，弱模型可能照样采不出正样本；harness 给 hint、甚至让 rigger 看到 ground-truth 轨迹都**不产生 off-policy 问题**——因为训练仍在（改过的）新环境上 on-policy 地进行。这与「拼正确轨迹进 GRPO rollout」有本质区别。

### 成本账：把昂贵的人造环境变成同模型自产

成本口径（字幕）：rigger 与 policy 用**同一个 base model**（排除「强模型蒸馏弱模型」这个混杂因素）；用便宜的 Gemini Flash Lite 时，每个改造环境的成本约 \$0.5 ，同一预算可造约 2000 个定制环境；即使用最贵的 Opus 类模型，每个也仅约 \$40 ，是人造环境约 \$1600（按 \$100/小时、Terminal-Bench 约 16 小时）的大约 1/40 。更重要的是它能给已在 Terminal-Bench 上无信号可拿的强模型（如 Opus 5）造出可训练的难度——成本不是唯一卖点，信号覆盖才是。

### 讲者自陈的三条局限

1. **规模成本仍不便宜**：覆盖上万任务时逐实例生成、成本线性（缓解路线是 harness 跨同类环境复用或对环境/harness 聚类）。
2. **只适用于可 reset 的模拟/sandbox 环境**：真实世界不可逆操作（杯子砸碎、钱已花出）不适用；机器人场景仅限 simulator 侧。
3. **语义型 Chain 暂未打通**：rigger 还没法知道另一个环境里装了什么，长上下文塞多环境 rollout 会让模型「胡言乱语」；可行解是给 rigger 配专用 harness、按需读 rollout 文件；当前 rigger 实际只自选 stage 与 contract 两类组件。

### Q&A 精华

- **接口门槛**：只需环境支持 gym 式三操作 reset / step / evaluate，绝大多数 benchmark 都满足（forecasting 这类与时间强相关的除外）。
- **reward 安全**：task 与 verifier 不变、且有 verify 闭环兜底，改造必然留有正确轨迹，信号始终正确。
- **基模门槛**：方法跨模型迁移（Gemini Flash 系、Qwen、Claude Sonnet 4.6 都有提升；Opus 5 在 SWE-bench Verified 已近满分，故图中用 SWE-Pro 的 4.8 代替），唯一要求是基模要能写代码，因为 harness 本身以代码存在。
- **自进化观**（与其 R-Zero 一脉相承）：data-driven 自进化是靠谱的——环境即训练数据，如数学从公理演绎扩展；但存在经验上的饱和上限：50 个 SWE-Lite 环境迭代到某点就上不去，原因未知。
- **额外用途**：harness 可以 block shortcut / reward hacking 提升泛化；整套方法也可反过来用于 benchmark 增强；或针对性上采样某类环境做能力保持（如只留 coding 环境并放大）。

## 可迁移

- 评估 RL 环境/数据 pipeline 的一张问卷：信号是否有随 policy 强度自适应调节的机制？没有的话，天花板就是环境本身。EnvHarness 给出的最小实现是：改初始状态、改可用动作集、改观测、拼任务图，四种变换都不得触碰 verifier。
- 「诊断优先于堆数据」是可直接借用的实验纪律：先看 policy 的失败轨迹再决定加什么训练变换，而不是先把数据量/难度拉满——讲者用对照曲线把增益归因在诊断一侧，这对自己的 RL 数据工作是同样的提醒。
- on-policy 纯净性的一条设计判据：想把 ground truth / hint 注进训练时，先问注入发生在「环境侧」还是「轨迹侧」——前者（EnvHarness 式）保持 on-policy，后者（拼轨迹）会引入 off-policy 偏差。
- 成本结构类比：环境侧自动化的边际成本（\$0.5–\$40/个）比人造（\$1600/个）低一到三个数量级，与「模型自产数据」替代「人工标注」的量级关系一致；定预算时可以按这个量级先算上界。

## 疑问 / 下一步

- 50→300 的对比基线（300 个真实环境的来源与构造方式）与各域的完整数字需回论文核对；字幕只给了结论与 SWE 口径。
- 饱和上限的成因讲者明说未知：是环境信息量上限、rigger 诊断上限还是 policy 容量上限，三者的区分实验值得等后续版本。
- 语义 Chain 一旦打通（rigger 能索引跨环境语义），long-horizon 训练信号的形态会变；当前实现里 chain 还只能靠人工拼或模板，属半成品能力，下结论时勿当已完成功能引用。
- 讲者正在求职 RSI / Agentic RL 方向（字幕自述），后续版本是否继续迭代此线，取决于其去向。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 人造环境成本（Terminal-Bench 2.0） | 约 15.9 小时/个，约 100 个环境近 1400 小时；按 \$100/小时 约 \$1600/个 | — | 字幕（背景+成本部分） |
| 人造环境成本（Apex Agents） | 33 个环境，每个约 500 小时，共约 16000 小时 | 约 \$1.6M（按 \$100/小时） | 字幕（成本部分） |
| EnvHarness 改造单个环境成本 | Gemini Flash Lite | 约 \$0.5/个；同预算约 2000 个 | 字幕（成本部分） |
| EnvHarness 改造单个环境成本（最贵口径） | Opus 类最贵模型 | 约 \$40/个，为人造的 1/40 | 字幕（成本部分） |
| 基础环境效率 | 50 个 SWE-Lite 环境 + 多轮迭代 EnvHarness | 效果超过 300 个真实环境（下游 SWE-bench Verified） | 字幕（主实验） |
| Rigger 成功率判据 | 5 次 rollout 对 2 或 3 次 | 落在 0.4–0.6 区间 | 字幕（Rigger 闭环） |
| Rigger 候选接受率 | 不同 benchmark | 约 40%–60%（区间 30%–80%） | 字幕（Q&A） |
| 接口要求 | Gym 式三操作 | reset / step / evaluate | 字幕（Q&A） |

## 原文金句（1-2句）

> 「这套东西你可以理解成把 data augmentation 扩展到了环境上：环境就是训练数据。」——讲者答「这不就是课程学习吗」（字幕 [57:14–57:32] 口径转写）
