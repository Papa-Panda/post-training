# EP151 — AI4AI at Test-Time：从 Harness 看 AI4AI 如何实现以及未来
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP151-ai4ai-test-time.html

> 「在 AI for AI at test time 当中，其实核心就是一个词，就是 cognitive offload，只不过这个 cognitive offload 是通过 harness 来表现、或者说是通过 harness 来转移的。」——讲者在总结中给全讲下的定义（字幕口径）

## 元信息

- 期号：青稞Talk EP151（B站 151 期）
- 标题：AI4AI at Test-Time：从 Harness 看 AI4AI 如何实现以及未来
- BV：BV1Wet26CEgH
- 时长：01:11:12（讲授约 49 分钟 + Q&A 约 22 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：UIUC 三年级博士生（字幕自述作「虔诚」，主持在 Q&A 开场称「陈博士」；本工作为讲者今年暑假在 Salesforce AI Research 实习期间完成，字幕作「south forth AI research」；全名以论文作者页为准）
- 相关论文：本讲主体工作（AI for AI at test time / harness 分析向论文，讲者未在字幕中给出题名与编号）及前作 User Harness（为小模型设计 harness 系统提升 Theory of Mind 表现）；post-training 侧的对照工作 PostTrainBench 在 Q&A 中被提及；具体题名与编号以论文原文为准
- 相关代码：字幕未给出仓库地址；讲者说给 builder 的完整 instruction 在论文附录里
- B站链接：https://www.bilibili.com/video/BV1Wet26CEgH/
- 字幕原文存档：本地 `transcripts/EP151.txt`（1715 条，带时间戳）
- 编号说明：本期为 B站合集第 151 期，与官网 151 期《Harness-AI4AI》同题不同 talk——官网期与本 B站视频不是同一场分享，勿混淆；本纪要按 B站视频实际内容写。

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP151，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 Theory of Mind 在字幕中多作「series of mind / server of mind」、Gemini 作「CHINI / GEMINA」、Muse 作「cloud code」、Codex 作「Gbt Code dex」，模型名如 GPT-5.4-mini / GPT-5.5 / Gemini 3.5 Flash 均按字幕口径，均以公开资料为准）。凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

不改任何参数、只在 test time 动手：让一个 builder AI 像做 OJ 题一样在 validation set 上反复试错，把它对任务的洞察外化成一套 harness（prompt 套路、状态追踪、格式检查、few-shot 等），再把弱小的 target AI 放进这套 harness 里跑 hidden test——57 个针对小模型的 run 里 harness 全部带来提升，平均分从不到 0.5 涨到 0.76 、最好 0.91 ，逼近人类专家手写的 0.93 ；讲者把机制命名为 cognitive offloading：builder 提前替 target 把「探索任务、总结解法」的认知成本付掉，并提醒 harness 对强模型可能是束缚、模型与 harness 需要协同进化。

## 核心

### 动机：系统能力 = 模型 + harness

出发点是一组反差数字：一个小模型（字幕作 GPT-5.4-mini）在 Theory of Mind 任务上裸跑不到 0.5 ，放进一套 harness 后能到 0.91–0.92（字幕）。这套 harness 来自讲者前作 User Harness：把「面对环境拿到 partial observation、更新 belief、结合目标决定 action、action 再影响环境」这类形式化推理流程写给小模型照着走。由此引出全讲的立场：现在用户面对的从来不是裸模型而是系统，系统能力可拆成模型与其外围 harness 之和；harness 指包裹模型的全部东西——system prompt 与指令模板、routine、工具、code 与 state memory、答案验证与格式控制等。Training-time 的 AI for AI（上层 AI 选数据、调超参、改配置去训下层 AI）大家已经熟悉，这讲问的是 test-time 版：参数冻结时，AI 还能怎样替 AI 变强？答案只能落在 harness 上。

### 范式：builder 造 harness，target 在里面跑

定义一句话：builder AI 构建一套 harness 系统，target AI 在其中运行，以求在下游任务上表现更好——传统上人类坐的「诊断错误、设计解法」的位置，换成 builder AI 来坐。实验流程被讲者类比为做 OJ：给 builder 一份 instruction、一个可调用的 target 模型和 195 道 validation 题（字幕）；builder 提出 harness、在 validation 上测、看对错、再改，轮数不限；最后把 harness「提交」，用它在 builder 看不见的 3900 道 hidden test 上给 target 打分（字幕）。hidden 集必须隔离，否则 builder 把每题答案直接写进 harness 就能 hack 满分；validation 与 hidden 同分布、具代表性，harness 才可迁移。任务选 Theory of Mind（BigToM、HiToM、MMToM-QA、MuToM 等，字幕；只用文本格式）是刻意的：这类题一部分可结构化（状态追踪、符号化 belief 更新）、一部分必须靠模型自身推理，正好能测「什么能被 harness 外化、什么必须留给模型能力」。评测规模：多个模型家族在 Cursor、Muse、Codex 等 coding 平台上以 agent 形式当 builder，target 用两个小模型，每个配置重复 3 次，共 72 个独立 run（字幕）。

### 主结果：harness 全数有效，小模型反超同系大模型裸跑

Headline 数字（均为字幕口径）：target 裸跑平均不到 0.5 ，同系大一号模型裸跑约 0.6 ；builder 造的 harness 平均 0.76 、最好约 0.91 ，人类专家手写的 User Harness 为 0.93 。在针对小模型的 57 个 run 里，每一套 harness 都带来提升，提升率 100%（字幕）；分 benchmark 看也普遍成立，小模型加 harness 能反超同 family 更大模型的裸跑，但离人类专家手写仍有差距。讲者强调这不只是刷分：它说明「builder 对任务的理解」可以经 harness 这种可执行工件转移给 target，而不经过参数。

### 优化动力学：像训练曲线，但涨幅不靠轮数

把 builder 在 validation 上的迭代画成曲线，形状酷似训练曲线：多数 builder 随迭代轮数提升、约第 5–6 轮后饱和（字幕），饱和点即该 builder 对 validation 洞察力的上限。但最终 hidden 分与迭代轮数几乎不相关（字幕给出相关系数 R 约 0.17 ），与 validation 准确率才正相关——说明瓶颈是「看懂任务」的质量而非「多试几轮」的数量。另一条动力学结论：最终成绩与 builder 花了多少 reasoning effort 高度正相关，与它跑在哪个 coding 平台、是否原生平台关系不大；不把 reasoning effort 开大，原生平台也救不回来。讲者因此说这篇分析完全可以反着读成一个 builder 能力的 benchmark：测的其实是大模型「自我改进」的上限，只是改进对象是 harness 而非权重。

### 机制：cognitive offloading，及其证据

Builder 到底造了什么？对 57 个 run 的统计显示，harness 组件以 format enforcement 几乎全覆盖（强制唯一答案格式），外加按题型分 routine、动态 few-shot、状态追踪、答案格式检查、negation / polarity 逻辑等；有无某组件的粗对照显示单个技巧大约值 3–4 个点，negation 逻辑约 0.09（字幕；讲者提醒对照并不严格公平、有的组件无对照组）。讲者的解释是 cognitive offloading：builder 提前在 validation 上替 target 做完探索，把 reasoning 总结成 skill 写进 harness，target 运行时只需照章执行，省下的认知成本就是涨分来源——类比新员工拿到 mentor 写的环境配置手册，或人类穿外骨骼跑得更快但肌肉本身没变强。可外化程度决定上限：早期 benchmark（如 BigToM）状态可符号化追踪、外化空间大、涨分最多；高阶 Theory of Mind（2–4 阶 belief）难以结构化，仍是主要 error 来源。讲者也承认这一观察与既有 Theory of Mind 分析文献互相印证。

### 边界：harness 是双刃剑，强 target 会被束缚

两个反直觉发现。其一，同一套 harness 对弱 target 全数有效，对更强的 target（字幕作 Gemini 3.5 Flash，裸跑本就更高）平均仍涨、但个别 run 出现 performance regression：约束对弱者是扶手、对强者是镣铐，harness 的「剂量」应随 target 能力调整。其二，模型与 harness 需要 coevolve：为某个 target 的 failure pattern 定制的 harness 换到另一个模型未必迁移（Q&A 中讲者以老师按学生弱科编教案类比），同等能力模型间大体可迁移、能力差大则不保证。此外讲者坦承实验有监督把关以防 builder hack validation：观察曲线是否平滑收敛，突兀跳升会被回查，论文呈现的结果未见明显 hacking。

### Q&A 要点

- **builder 必须比 target 强吗**：不必。让同一小模型给自己造 harness 再自己跑，仍有提升（自己先 explore 任务、把结构外化给自己），但上限明显低于强 builder；这也说明涨的不是知识、是把 on-the-fly reasoning 换成预先固化的流程带来的可靠性。
- **怎么防过拟合那 195 题**：核心就是 hidden 隔离；把答案写进 harness 的极端过拟合在 hidden 上天然失效。同理，validation 曲线平滑收敛也可作 hacking 的探针。
- **诊断粒度**：强 builder 会 case-by-case 看失败、先归类（格式错 vs 推理错）再统计、再针对性改 harness；讲者顺带批评当下 coding agent 过度盯小 case、宏观系统改进思考不足。
- **target 能力下限**：没有硬性下限，越弱提升空间越大（涨 100%–200% 都可能，字幕），但 builder 要付的「因材施教」成本越高；为弱模型定的 harness 换个模型，效果取决于 failure pattern 是否相同。
- **同样的 builder 预算直接喂强模型做长 CoT 行不行**：讲者判断不如先造 harness 再做题——每次现想现做无法保证系统可靠性，预先探索任务、固化框架后的系统更可靠，小模型上尤其明显。
- **builder 自己的 harness**：实验直接用各平台原生 harness（Cursor / Muse / Codex），限制极少、给足自由度；测的正是 builder 在给定沙盒里的探索与利用能力。
- **人的角色**：讲者的收尾判断是，AI for AI 时代人类从「经理」退到「CEO」——不再逐题监督，而是给 builder 注入知识、把控宏观方向。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 小模型裸跑平均分（Theory of Mind） | — | 不到 0.5 | 字幕 |
| 小模型 + 前作 User Harness | 裸跑不到 0.5 | 0.91–0.92 ；人类专家版 0.93 | 字幕 |
| 同系大模型裸跑 | 小模型不到 0.5 | 约 0.6 | 字幕 |
| builder 造 harness 的 target 平均分 | 裸跑不到 0.5 | 平均 0.76 ，最好约 0.91 | 字幕 |
| harness 有效比例 | 57 个小模型 target run | 100% 带来提升 | 字幕 |
| 实验规模 | — | validation 195 题，hidden test 3900 题；72 个独立 run（每配置重复 3 次） | 字幕 |
| 迭代收敛轮数 | — | 约第 5–6 轮后饱和 | 字幕 |
| 迭代轮数与最终分的相关 | — | R 约 0.17（基本无关）；与 validation 准确率正相关 | 字幕 |
| 单个 harness 技巧的粗略价值 | 无该组件的 run 对照 | 约 3–4 个点；negation 逻辑约 0.09 | 字幕 |

## 可迁移

- 给小模型 / 便宜模型提分时，先试「强模型离线造 harness」而不是急着微调：把强模型在验证集上试错得到的解题 routine、状态追踪与格式约束固化成 prompt 与代码工件，弱模型照章执行即可涨分，且 harness 可复用、摊薄到后续所有 run 上——这正是 post-training 团队用大模型给小模型做脚手架的现成范式。
- Harness 是有剂量的：同一套约束从弱模型迁到强模型可能倒扣分。上线前应按 target 能力分档测 harness 的净收益（修对了多少、又把原本对的搞错多少），而不是默认越全越好。
- 评测 builder / agent 的一种低成本读法：固定弱 target 与 validation 集，让不同 builder 各造一套 harness 比 hidden 分，既测出其任务洞察与自我迭代上限，又避免直接比裸模型分数时的任务污染问题。

## 疑问 / 下一步

- 论文题名、arXiv 编号与 builder instruction 全文字幕均未给出：过拟合防护细节、每种 harness 组件的严格消融表需回论文附录核对（字幕中的组件级数字为非严格对照的粗估计）。
- 结论目前只在 Theory of Mind 一类「半可结构化」任务上验证：代码生成、长程 agent 任务里可外化的比例与涨幅是否同样成立，讲者未给数据。
- 「模型与 harness coevolve」被列为未来方向，但如何让 target 反过来适配 harness（而不是只调 harness 适配 target）在本工作里没有方法，只有问题。

## 原文金句（1-2句）

> 「在 AI for AI at test time 当中，其实核心就是一个词，就是 cognitive offload，只不过这个 cognitive offload 是通过 harness 来表现、或者说是通过 harness 来转移的。」——讲者总结（字幕约 43:22–43:37 ，干净口径转写）

> 「你对于弱模型来说，你是在辅助他的能力；但是同时你对于强（模型）来说，如果你加了非常多的条条框框，也是一种束缚。」——讲者论 harness 的双刃性（字幕约 41:06–41:20 ，干净口径转写）

> 「未来人类可能在 AI for AI 的时代……我们人类呢就更像 CEO 的角色，负责整体的决策和整个宏观方向上的把控。」——讲者收尾（字幕约 48:03–48:17 ，干净口径转写）
