# EP24 — 使用 CAMEL Agents 构建 GraphRAG 及应用实践

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP24-camel-graphrag.html

> 「智慧的力量来源于我们广泛的多样性，而不是某一个单一的完美的准则。」——讲者转引 Marvin Minsky《心智社会》的判词，作为多智能体的立论起点（字幕 02:09–02:21）

## 元信息

- 期号：24（官网第 24 期）
- 标题：使用 CAMEL Agents 构建 GraphRAG 及应用实践
- BV：BV1feNZeYEbm
- 时长：00:49:57（讲授约 39 分钟 + Q&A 约 10 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：范文栋（字幕开场自述；CAMEL-AI 多智能体开源框架核心贡献者）
- 相关论文：CAMEL role-playing 框架及 CAMEL 数据集相关工作（讲者以框架与项目页为准）
- 相关代码：CAMEL-AI 开源社区（GitHub）；文中另提到 CRAB 跨平台 agent benchmark（开源）
- 字幕原文存档：本地 `transcripts/EP24.txt`（1166 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP24，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 CAMEL 在字幕中多作「cameo / camera / KCAO」，NebulaGraph 作「内部拉 graph / NEBAGGRAPH」，AutoGen 作「auto jin」，Ollama 作「欧拉码」，均已按公开资料校正）。讲授中的演示以屏幕内容为主者，纪要只取讲者口播结论。

## 一句话总结

这期用 CAMEL 回答了两个问题：多智能体 role-playing 不只是「能完成任务」，它还是一个能自动产数据、产结构的引擎——一边生成可微调模型的对话数据，一边用 Knowledge Graph Agent 把文本抽成实体与关系；再把图检索与向量检索拼起来做 GraphRAG，补上传统向量 RAG 切块丢指代、丢关系的短板。

## 核心

### 框架立论：从「心智社会」到 agent scaling law

讲者先铺三段背景：Minsky《心智社会》（1986）里 agent 是「没有思想的进程」，单个简单、成群才涌现智慧；强化学习里的 agent 靠试错改策略但泛化受限（AlphaGo 下不了象棋）；生成式 AI 之后，Lilian Weng 的范式把 LLM agent 拆成 memory / tools / planning + 大脑（LLM），自然语言进出带来泛化。讲者还抛出一个假说式类比：他称之为 agent 的 scaling law——模型参数量对应 agent 数量，训练对应与环境交互，训练过程沉淀为 memory / interaction，暗示多智能体系统可能是模型能力见顶后的下一条扩展曲线（字幕 11:47–12:14）。

### CAMEL role-playing：对话既是执行也是数据

CAMEL 的核心模块是 role-playing：用户给一个 idea / task，先定义 AI user（出需求，如股票交易员）与 AI assistant（解问题，如 Python 程序员）两个角色，经 task agent 把任务具象化后，两边交替对话直到一方说出任务完成标记。现场演示了两段：牛津大学年龄计算（search + math 工具分工）和 Pygame 寻宝游戏（一指令一代码步步推进）。

两个经验证据值得记：一是裁判对比实验——人类裁判认为 CAMEL agents 更好的场景占 76.3%，GPT-4 作裁判时占 73%（字幕 18:03–18:51）。二是**对话数据可以反哺训练**：框架自动生成数据（AI Society 设定：50 种 assistant 角色 × 50 种 user 角色 × 10 类 task；代码设定：20 种编程语言 × 50 个领域 × 50 种 task），微调后模型在讲者演示的数学领域对比中由 8:7 翻成 3:16 压倒性领先（字幕 19:04–20:38）。讲者称 Hugging Face 上有 180+ 个模型用了 CAMEL 数据集，MPT-30B、M4（字幕音译）亦在其列。Q&A 中他明确框架定位「更加专注于 bot、data generation 方向」，与 AutoGen 各有侧重（字幕 46:33–46:54）。

### GraphRAG：为什么需要图，以及图怎么让 agent 自己建

传统向量 RAG 的毛病讲者给得很具体：切块会切断指代——柏林介绍的第二块里「it / the city」到底指谁，块内看不出来（字幕 21:55–22:35）；企业数据里产品间的上下位、替代关系，向量检索同样捞不回来。GraphRAG 把 entity 与 relationship 存进图数据库，检索能拿到全局与关系信息。

但手工建图成本高（专家逐条找实体、关系再入库），CAMEL 的解法是定义一个 **Knowledge Graph Agent**：给它角色、抽取任务（抽所有 node 与 relation）和结构化输出格式加示例，让 agent 直接把文本变成结构化数据。检索侧的 workflow 是双路拼装：向量路按 query embedding 捞语义块，图谱路抽 query 的实体去图里匹配关系，两路结果合并给 agent 回答——讲者口径是「vector retrieval 偏向语义检索，knowledge graph 偏向 relationship 检索」，且两路各有优劣、结合最好（字幕 26:00–28:08）。演示案例是一个端到端任务：给一个人名，agent 自己去搜索、抓网页、写出一篇报告，再从报告自动抽出知识图谱（含被引申出的关联实体信息）。

Q&A 里对数据可信度的回答很务实：agent 生成的图数据当然有幻觉，可以再引一个 critic agent 复核（成本更高）或人类专家复核；框架支持人工删改任意 entity / relationship（字幕 40:44–41:25）。被问到因果关系能否抽，答：改 Knowledge Graph Agent 的 prompt 即可，当前 prompt 只 focus 在 relationship 与 entity（字幕 44:04–44:26）。

### 周边与边界

- 商业化载体 Agent AI 两件产品：Agent Bot（定位 knowledge generation、能调工具跨系统、支持 Slack / Discord / Telegram 等部署）和 Agent Data（用框架生成合成数据做 RAG / 微调）。均属「正在进行」口径。
- CRAB（Cross-platform Agent Benchmark）：用基准测不同 LLM / VLM 的能力差，任务天生跨平台（如打开 Slack 看消息、总结后发给手机联系人），多智能体底座由 CAMEL 提供。
- 工具调用成本判断：工具本身执行成本低，schema 只有在被调用时才读入，讲者认为「成本可以接受」（字幕 44:51–45:14）。工具 / 函数规模化靠在研的 CAMEL Scale 项目（未上线）。
- 本地模型支持 Ollama / vLLM 调用（Q&A 确认）；对 SGLang 直言不了解、CAMEL 暂未支持，欢迎提 issue——这类坦白边界在纪要里如实保留。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| 人类裁判偏好 CAMEL agents 的场景占比 | 76.3% | 字幕（18:19–18:27 实验口径） |
| GPT-4 裁判偏好 CAMEL agents 的场景占比 | 73% | 字幕（18:36–18:46 实验口径） |
| 框架数据设定（AI Society） | 50 assistant 角色 × 50 user 角色 × 10 类 task | 字幕（19:04–19:17） |
| 框架数据设定（代码） | 20 种编程语言 × 50 个领域 × 50 种 task | 字幕（19:17–19:22） |
| 微调效果演示（数学领域模型对比得分） | 8:7 → 3:16（加入对应数据微调后） | 字幕（20:13–20:32） |
| 使用 CAMEL 数据集的 Hugging Face 模型数 | 180+ | 字幕（20:47–20:53 讲者口述） |
| 社区贡献者数（讲者致谢口径） | 51 位 PR 贡献者 | 字幕（39:01–39:06） |

## 可迁移

- **「生成数据 → 微调」闭环是最便宜的 agent 增值路径**：role-playing 副产的对话数据能把开源模型在单一领域从五五开打到压倒性领先。给 coding agent 做轨迹数据采集时，这套「定义角色对 → 自动对话 → 筛轨迹微调」可以直接借。
- **图检索与向量检索是互补不是替代**：语义相似找向量、关系推理找图，两路结果合并喂模型。内部知识库（代码关系、服务依赖、产品上下位）属于典型关系密集型数据，先用 agent 自动抽图再双路检索，比全量向量化更能答全局问题。
- **结构化抽取靠 prompt 三件套**：角色 + 抽取范围（node / relation）+ 输出格式与示例。抽错时先加 critic 或人工复核，不要先怀疑模型。
- **多智能体框架选型先看侧重**：讲者自陈 CAMEL 侧重 bot 与 data generation。选型时按「我要的是协作执行、数据生成、还是流程控制」分流，而不是只看 star 数。

## 疑问 / 下一步

- 裁判实验（76.3% / 73%）的任务集与样本量讲授未展开，这两个百分比的适用边界需回 CAMEL 论文核对。
- GraphRAG 的双路 workflow 讲授以 demo 为主，没有给出与纯向量检索的定量对比（准确率 / 召回）；落地前要自备评测集，别直接采用定性结论。
- 180+ 模型使用 CAMEL 数据集为讲者口述数字，引用时注明出处口径。

## 原文金句

> 「智慧的力量来源于我们广泛的多样性，而不是某一个单一的完美的准则。」——字幕 02:09–02:21（讲者转引《心智社会》）

> 「vector retrieval 更加偏向于语义的一个检索，knowledge graph 更加偏向于 relationship 信息的一个检索。」——字幕 27:52–27:59（双路检索的定位判词）
