# CUA Workshop — 通义 Mobile-Agent-v3.5：多平台基础 GUI Agent 与 GUI-Owl 1.5

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/CUA-mobile-agent-v35.html

> 「不同的端侧之间其实会存在拉扯的问题，所以我们是多个端侧进行交替的这种 RL，这样来缓解不同端侧拉扯的问题。」——讲者给出多端统一模型的关键训练技巧

## 元信息

- 期号：青稞社区 ICLR 2026 CUA Workshop 系列分享（合集 sid=7971490）
- 标题：Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents（B站标题作「通义Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents｜ICLR‘26 CUA Workshop」）
- BV：BV1kKoSBDExS
- 时长：00:20:45
- 提炼日期：2026-10-02
- 分享嘉宾：徐海洋（阿里巴巴通义实验室；字幕中自述「我是来自通义实验室的徐海洋」）
- 相关论文：Mobile-Agent 系列 v1/v2/v3 与 Mobile-Agent-v3.5、GUI-Owl 1.5（字幕作「GYOU / GROL1.5」，以论文与开源仓库正式名 GUI-Owl 为准）；讲者称模型已在 ModelScope 与 Hugging Face 开源，阿里云百炼提供体验 API
- 相关代码：Mobile-Agent 系列 GitHub 仓库（讲者称约 8.3K stars），含 GUI-Owl 等开源工作
- B站链接：https://www.bilibili.com/video/BV1kKoSBDExS/
- 字幕原文存档：本地 `transcripts/CUA-MOBILE-AGENT.txt`（489 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文清洗提炼。专名识别误差较多（CUA 作「COA/酷啊」，GUI-Owl 作「GYOU/GROL」，OSWorld/AndroidWorld 作「os word / 安卓word」，ModelScope 作「摩达」等），均以公开资料为准；性能表述全部为字幕口径。讲者自 v3.5 起改称团队方向为「多模态多端 GUI Agent」，本场即该版本的总览介绍。

## 一句话总结

这一场是通义 Mobile-Agent 系列的版本总览：v3.5 与基础模型 GUI-Owl 1.5 把手机、平板、车机、PC、浏览器五端统一进一个模型（2B/4B/8B/32B 全尺寸，各有 instruct 与 thinking 两版），针对 GUI 领域的四项老问题——界面理解、定位精度、长程规划与反思、操作效率——给出一条完整配方：合成环境打底的 grounding 数据、仿真为主真实为辅的轨迹合成、海量 GUI 世界知识预训练与统一 CoT 合成、多端交替在线 RL（缓解端间拉扯）配 infra 侧修训推不一致；讲者称在 20 多个 GUI benchmark（grounding、UI 知识、在线环境操作、工具调用）上取得显著提升，其中 ScreenSpot-Pro 拿到 SOTA 且高于 Gemini 3（字幕口径）。

## 核心

### 系列脉络：从框架拼装到基础模型

讲者先回顾系列演进，这条线本身就是过去两年 GUI agent 的缩影：

- **v1**：首个把多模态智能体用到手机端的工作。当时大模型工具调用能力不行，靠外挂带定位能力的工具（OCR、SAM 等）拼起系统。
- **v2**：多智能体（multi-agent）框架。
- **-E（字幕作「目标按键的杠E」）**：自主进化方向——把过往成功经验总结成 tips 与 shortcuts，提升后续复杂任务的效率与效果。
- **v3 / GUI-Owl**：基于无影云环境搭一整套 infra（浏览器、computer use、移动端三类沙盒环境），在其中做在线 RL，在 OSWorld、AndroidWorld 等管理类任务上取得 SOTA。
- **v3.5 / GUI-Owl 1.5（本场主角）**：跨平台支持更强（这一版重点补了 browser 能力）、扩充真实 app 数据与在线 RL、release 全尺寸模型，并针对端侧实时要求同时提供 thinking 版（带 CoT）与 instruct 版（输出短、执行快）。

团队口径的其他信息：该系列连续两年获 CCL best demo；覆盖端包括手机、平板、车机、桌面（PC + browser）。

### 四项能力短板与对应解法

讲者把 GUI 领域的核心挑战列为四项，并一一对应到 v3.5 的做法：

1. **界面理解与 grounding**：难点在多窗口、专业 app 这类复杂场景。做法是合成环境造数据——通过代码、图像生成仿真各种软件，让模型在其中做 grounding 数据合成。
2. **轨迹数据**：真实 app 有账号安全与成本问题，采集效率低。路线是「仿真环境为主、真实环境为辅」：用代码与 app 开发合成与现实 app 高度相像的模拟 app 产数据，最后在真实环境数据上做 alignment，才真正做到 scaling。
3. **GUI 世界知识与 agent 能力**：设置在哪、功能在哪这类问题需要海量世界知识。预训练阶段注入百度知道、百度文库、各类软件操作说明书与视频教程，并用 world model 式任务（如预测下一状态分布）提升 GUI knowledge；同时把 CoT 合成流程统一化（讲者发现 CoT pattern 五花八门会损害学习效率）；数据生产 workflow 里除模型合成与人工标注外，还引入多智能体框架合成的数据以提升泛化与适配。
4. **多端训练的拉扯**：统一模型最现实的痛点。训练三阶段——世界知识与 grounding 数据预训练、各端轨迹 SFT 冷启动、多端在线 RL。RL 阶段的两个关键处理是本场最具操作价值的部分：**多端交替 RL**（不同端侧交替训练，缓解端间互相拉扯；讲者展示的曲线显示交替训练确实缓解了单端训练时的此消彼长）与 **infra 侧修训推不一致**（升级 infra 缓解训练/推理不一致对在线 RL 的干扰）。另外，记忆能力与工具调用/API/MCP 能力在这一版被着重增强，因为业界多用 multi-agent 架构，只训单一能力的模型难以适配各种框架。

### 评测与开源（字幕口径）

讲者称共评测了 20 多个 GUI benchmark，覆盖四类：grounding 任务、UI knowledge 任务（ScreenSpot-Pro 上为「绝对的 SOTA」，比 Gemini 3 还高）、在线环境操作（OSWorld、WindowsArena、AndroidWorld、WebArena 等）、工具调用（团队自研的 OSWorld-MCP、MobileWorld 等 benchmark，亦称 SOTA），且全部统一到一个模型里。模型在 ModelScope 与 Hugging Face 开源（2B/4B/8B/32B，instruct 与 thinking 两版），阿里云百炼上有体验 API（字幕念出的接口名以平台实际为准），ModelScope 提供云电脑/云手机/云浏览器的在线 demo。

### 展望：三个未解问题

讲者收尾列出他认为最需要解决的三件事：（1）真正的 agent RL scaling 还没做通——只有真实环境下的 scaling 才能提升自主推理与知识进化，因为那是在线学习；（2）桌面端把 MCP、code、GUI 结合起来自主解决问题是当下主流路径（OpenClaw、Claude Code 都沿此思路）；（3）个性化交互与记忆。

## 关键数字

| 指标 | 数字 | 来源 |
|---|---|---|
| 模型尺寸 | 2B / 4B / 8B / 32B（「两币」按 2B 还原，字幕口径待核） | 字幕 |
| 模型版本 | 每尺寸均有 instruct 与 thinking 两版 | 字幕 |
| 评测覆盖 | 20 多个 GUI benchmark | 字幕 |
| ScreenSpot-Pro | SOTA，高于 Gemini 3 | 字幕 |
| GitHub stars | 约 8.3K | 字幕 |

注：全部为讲者口头口径，未与公开榜单逐项核对。

## 可迁移

- **多端交替 RL 缓解任务间拉扯**：多任务/多域 RL 出现此消彼长时，不要先怀疑算法——先试交替训练的课程安排，并把单任务训练曲线与交替曲线画在一起验证拉扯是否缓解。这招对任何多域混合的 agent 后训练都可直接试。
- **训推不一致要从 infra 修**：在线 RL 效果异常时，先查训练/推理两侧的数值与格式一致性，再调算法超参；讲者团队就是先升 infra 再谈 RL 收益。
- **轨迹数据的「仿真为主、真实对齐」配比**：真实环境采集贵且有账号风险时，用代码合成高仿 app 产轨迹、最后用少量真实数据 alignment，比硬采真实数据更能 scale——这与纯合成环境（WebFactory 路线）形成对照，两条路线可按数据预算组合。
- **CoT pattern 先统一再训练**：合成思维链数据时统一格式模板，避免 pattern 多样性本身成为学习噪声。

## 疑问 / 下一步

- GUI-Owl 1.5 的技术报告细节（数据配比、各 benchmark 具体分数、多端交替的排程参数）值得查论文原文核对；尤其 ScreenSpot-Pro 高于 Gemini 3 的口径（具体分数与评测设置）。
- 模型尺寸的最小档字幕作「两币」，疑为 2B，开源仓库可直接确认。

## 原文金句

> 「如果只用真实 app 的数据采集链路，其实采集的效率还是比较低的。」——讲者解释为何轨迹路线改为仿真为主

> 「现在其实还是没有把这件事情完全做 work——只有在真实环境下的 scaling，它才能真正地提升模型的自主推理和知识进化能力。」——谈 agent RL scaling 的未解状态
