# CUA — The Illusion of Self-Improving Agents（ICLR 2026 CUA Workshop）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/CUA-illusion-self-improving.html

## 元信息

- 专题：ICLR 2026 CUA Workshop（青稞合集 sid=7971490）开场主题分享
- 标题：The Illusion of Self-Improving Agents
- BV：BV1cKoLB9Evh
- 时长：约 32 分钟（1932 秒）
- 观看/提炼日期：2026-10-02
- 讲者：字幕未自报姓名（主持人德涵介绍；讲者自述涉及公司内部工作、不便展开具体技术，此处不猜）
- 主持：德涵（字幕口径）
- B站链接：https://www.bilibili.com/video/BV1cKoLB9Evh/
- 字幕原文存档：本地 `transcripts/CUA-ILLUSION.txt`（826 条，带时间戳）
- 说明：基于 B站 AI 字幕整理；本场为观点分享，无配套论文与代码链接；英文术语按字幕可辨者记。

## 一句话总结

讲者的核心判断是：今天大多数自称 self-improving 的 agent 系统只是在「攒数据」，并不是在「学习」——因为它们的记忆表示没有抽象与结构、更新没有可靠的 refinement 机制、执行没有闭环反馈；真正的自我改进系统必须同时过这三关，外加一个决定「何时学、学什么」的 meta control。

## 核心

1. **背景/问题**：当前模型训练仍是 passive setting——每一步用什么数据、什么 recipe 都由人定好（预训练→mid-training→SFT→RL）。主要瓶颈已经是数据：documented data 终将 saturate，而人机交互、人与软件交互产生的 undocumented interaction data 几乎未被利用。deployment 阶段的 self-improve 被寄予厚望（OpenClaw 靠积累 memory file 越用越熟、Hermes Agent 主打「grows with you」、ClawHub 上下载量前四的 skill 至少两个与 self-improving 相关），但热度之下需要 first principle 的审视。

2. **框架：learning as the evolution of memory**。讲者借神经科学共识把 learning 与 memory 视为一体：学习即记忆载体的改变、再反过来影响未来行为。由此拆出 agent 学习系统的三个耦合问题——(a) memory 如何表示；(b) 如何可靠地更新；(c) 如何用 memory 影响执行。表示决定更新方式（向量空间→梯度下降；文本空间→改写/拼接）。三个 first principles 分别是：
   - 表示：**abstraction + structure**。学习是把经验压缩成 concept（common pattern），抽象—复用—压缩—泛化四位一体；没有结构就无法可靠地处理冲突信息、做 conceptualization（婴儿用颜色区分猫狗、用户偏好从「喜欢冷饮」refine 到「喜欢拿铁」两个例子）。
   - 更新：**reliable refinement**。学习是长期 consolidation：要处理冲突、形成 coherent 理解。symbolic AI 是现成的老师——给 override 加 priority、给每个 claim 加 provenance 与 confidence、做 context management（冲突命题可能只是前置条件不同）；有了 LLM 这个通用 symbol manipulator，不必再强求 formal language，只要有 LLM 看得懂的协议。
   - 执行：**闭环**。memory 不能只作为 reference 注入 prompt：那样对执行没有任何 guarantee，失败也无法 ground 到 memory 的具体一步。memory 若是可执行脚本，报错能精确定位到步、给出 grounded feedback——learning 与 inference 本就不应分家。

3. **对现有方案的逐一体检**（表示×更新×执行）：
   - markdown skill 文件：有一定抽象（agent 反思后 crystallize 成自然语言描述），但几乎无结构；更新靠「发现不好就重写」，不保证与旧版兼容、不能可靠处理冲突，长期 accumulation 无保障。
   - vector database：几乎无抽象，来一条存一条、append 更新，「说难听点就是死记硬背」，把整合全押给在线推理；这也是 RAG 长期只能做好 open-domain QA 的原因。
   - 模型权重：压缩效率高、向量空间天然有几何结构、梯度下降更新可靠；但 continual learning 的 catastrophic forgetting 短期难以解决（除非假设有较大 replay buffer），所以讲者主张短期 prioritize 非参数化路线。

4. **Proactiveness（what & when）**：讲者自认本场主要讲 how，最后推荐了 Meta 一个月前的工作《Why AI Systems Don't Learn and What To Do About It》（与两位认知科学家合著，字幕口径）：类比人类学习的 System A（observation，如读书上课，类监督学习）与 System B（trial-and-error，类强化学习），再加一个 System M（meta control）决定何时切换、用何种数据——与本场三原则可有机结合。

## 关键数字

本场为观点分享，无实验数字。仅有背景量级：讲者设问「要 saturate 一个 10T 甚至 100T 模型，数据从哪来」——答案指向 interaction data（字幕口径）。

## 可迁移

- 给 agent 写 memory/skill 规范时，先定**结构 schema**（字段、维度、前置条件）再让模型填内容，并给每条 claim 附 provenance 与 confidence、给 override 定 priority——直接借用 symbolic AI 的冲突处理协议，比自由重写 markdown 可靠得多。
- 评测一套 self-improving 系统时用三问验收：表示有没有抽象与结构、更新有没有可靠 refinement、执行是否闭环且失败可 ground 到步；三者缺一，「越用越好」就只是攒日志。
- 执行侧优先把沉淀下来的流程做成**可执行脚本/工具**而非 prompt 参考文本：报错定位到步，才有 grounded 的持续改进信号。

## 疑问 / 下一步

- 非参数化 memory 的「结构」具体该长什么样，讲者只给了偏好学习的 toy example；业务逻辑级知识的结构化表示仍是开放问题，值得对照 symbolic AI 的知识表示文献（如 claim 的 context 条件化）细读。

## 原文金句

> "They accumulate data, but may not truly learn."
> （指当前多数 self-improving 系统：数据在攒，但未必在学。）

> 「人的 learning 和 inference 本身就不应该是分家的。」
