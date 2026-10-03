# EP108 — BOAD：引入多臂老虎机，自动化搜索层级结构的多 Agent 架构设计

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP108-boad.html

## 元信息

- 期号：108
- 标题：BOAD：引入多臂老虎机，自动化搜索层级结构的多 Agent 架构设计
- BV：BV1MBPMzPEet（付费预览，本期无字幕轨；台账 episode-index.csv 中该 BV 登记为 B108《MiniMax-M2》，两边编号自 EP93 后分叉，本期按官网 EP108 内容编号，BV 归属待后续枚举合集时再核）
- 时长：未知（官网预告直播时段 2026-02-28 周六 10:00–11:00）
- 提炼日期：2026-10-02
- 分享嘉宾：Iris Xu（MIT 本科生，主修人工智能与数学；论文第一作者；嘉宾身份据官网预告页，非字幕自述）
- 相关论文：Iris Xu, Guangtao Zeng, Zexue He, Charles Jin, Aldo Pareja, Dan Gutfreund, Chuang Gan, Zhang-Wei Hong, *BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization*，arXiv:2512.23631（2025-12-29 提交，v2 2026-01-01），https://arxiv.org/abs/2512.23631
- 相关代码：https://github.com/iamxjy/BOAD-SWE-Agent
- 官网预告：https://qingkeai.online/blog/BOAD

> 📝 提炼方式说明：本期 B站视频为付费预览、无字幕轨可用，本纪要基于对应论文（arXiv:2512.23631）还原，**非逐字稿**；讲授环节的展开顺序、口头补充与 AMA 内容均无法确证，以下内容以论文为准，凡涉及「讲者观点」的表述均指论文作者在文中给出的判断。

## 一句话总结

单智能体在长时序 SWE 任务上要同时做需求理解、代码定位、编辑与验证，上下文被无关信息占据、泛化差；BOAD 把「自动设计层级多智能体系统」建模成多臂老虎机——每个臂是一个候选子智能体、回报是它在团队协作中的 helpfulness（LLM-as-a-judge 事后归因），用 UCB 在有限评测预算下自动搜出 orchestrator–子智能体结构，在 SWE-bench-Live 上以 36B 模型做到当时排行榜第二。

## 核心

1. **背景/问题**：现有 SWE 系统多为单智能体单链推理，定位、编辑、验证挤在一条上下文里，无关上下文引入虚假关联、限制 OOD 泛化；改成 orchestrator 协调子智能体后，新问题是：子智能体越多，层级结构的搜索空间组合爆炸，且团队成败难以归因到单个子智能体（一个子智能体单独存在时甚至无法完成任务，如 localizer 不能改代码）。
2. **方法/设计**：bottom-up 而非搜组合——先逐个识别有希望的子智能体再组队，搜索空间随子智能体数线性（而非指数）增长。贡献信号用轨迹上的 LLM-as-a-judge 估「helpfulness」，比整队成败的二值信号稠密；随后把子智能体选择建模为 MAB，用 UCB 类算法在 exploration/exploitation 间平衡，在有限评测预算下发现有效子智能体集合（UCB 形的乐观估计把未试充分的臂也纳入考虑）：

$$\hat{\mu}_a + c\sqrt{\frac{\ln t}{n_a}}$$

符号说明：候选子智能体记为 ， $a$ ，；其平均 helpfulness 记为 ， $\hat{\mu}_a$ ，；被评测次数记为 ， $n_a$ ，；轮次记为 ， $t$ ，。子智能体在 SWE-agent 框架内实现为 tool（无共享执行历史），档案可用 Chinese Restaurant Process 动态扩展新子智能体，并有 warmup 阶段自动修订子智能体文档以便 orchestrator 调用（据官方代码仓 README）。
3. **实验/实战**：基座为 Seed-OSS-36B-Instruct，在 SWE-bench-Verified（500 题）与 SWE-bench-Live（300 题，更新、OOD）上评测。BOAD 在 Live 上解决 20.0% issue（评测当时排行榜第二，超过 GPT-4o、Claude 3.7 Sonnet 等更大模型在流行 scaffold 上的成绩），比同模型的默认 SWE-agent 提升 63%；Verified 上 53.12%，为较小模型新 SOTA，比默认 SWE-agent 高 13.4%。消融显示恰好两个子智能体时性能峰值（20.0%），单子智能体只有 49/300；定制化 orchestrator（60/300）优于不定制（50/300）。同等评测预算下，进化式搜索（ADAS 式）只有 17.0%，且 Claude API 成本两倍以上。
4. **结论/观点**：人工设计的子智能体角色反而降低性能——人写死的分工与 LLM 的实际行为错位；自动发现的结构不仅提 ID 性能，OOD 泛化更好。作者判断：multi-agent 设计的关键不在堆人手，而在可归因的贡献信号与样本高效的搜索。

## 关键数字

| 指标 | 基线 | 结果 | 来源 |
|---|---|---|---|
| SWE-bench-Live 解决率（Seed-OSS-36B） | 默认 SWE-agent（论文称相对提升 63%，绝对值约 12.2%） | 20.0%（60/300，当时排行榜第二） | arXiv:2512.23631 主结果 |
| SWE-bench-Verified 解决率 | 默认 SWE-agent（相对 +13.4%） | 53.12% | arXiv:2512.23631 主结果 |
| 进化式搜索基线（同评测预算） | 17.0%（Live） | BOAD 20.0%，且其 Claude API 成本 ≥2× | arXiv:2512.23631 |
| 子智能体数量消融 | 单子智能体 49/300 | 两个子智能体峰值 60/300（20.0%） | arXiv:2512.23631 消融 |
| token 用量（Live） | SWE-agent 1.49M | BOAD 1.13M（−23.8%） | arXiv:2512.23631 |

## 可迁移

- 对 coding data / RL infra 工作的直接可试点：agent 架构搜索不必搜组合——先给候选组件一个可归因的单项贡献信号（如 judge 对轨迹片段的 helpfulness 打分），再用 bandit 做样本高效筛选，这套「稠密归因 + UCB 选臂」可直接搬到工具/子任务模块的自动筛选与评测预算分配上。
- Infra 视角：评测预算本身是稀缺资源，把「测哪个配置」形式化为 bandit 问题、并复用已有组件跨轮次摊销成本（进化式搜索每轮重生成组件导致成本翻倍），是评测自动化里可复用的调度思路。

## 疑问 / 下一步

- helpfulness 由 LLM-as-a-judge 事后打分，其与真实边际贡献的相关性、judge 偏差会如何污染 UCB 的臂排序，论文未见系统性校验——值得深挖。

## 原文金句（1-2句）

> （论文原句）"Instead of judging success by whether the entire team solves a problem, we estimate each subagent's helpfulness, measuring how much it contributes to solving the problem when in combination with other subagents."

> 说明：本期无字幕，上引为论文原文，非讲者口述。
