# EP140 — DRIFT：在线自进化后训练框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP140-drift.html

> "Effective self-evolution depends less on the choice between reinforcement learning and self-distillation than on when and where each signal is applied." —— DRIFT 论文 §5 结论：自进化的关键不在 RL 还是蒸馏，而在每种信号何时、用在哪

## 元信息

- 期号：140（官网期，无 B站视频）
- 标题：DRIFT：在线自进化后训练框架——没有外部专家监督时，如何让 OPD 策略学会自我持续进化
- BV：无——B站合集无对应视频（注意编号冲突：B站合集第 140 期是《鸿蒙智能体发展编年史》，与本期无关）
- 直播时间：2026-07-28 20:00–21:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：王颢宁（清华大学数学系博士生）、刘懿玮（巴黎萨克雷高等师范学院 MVA 研究生），均为贝壳认知智能组实习生（官网预告嘉宾介绍）
- 相关论文：Haisen Luo, Yiwei Liu, Haoning Wang 等（Beike / 清华 / ENS Paris-Saclay），*DRIFT: Difficulty Routing Self-DIstillation with Rhythm-Gated Exploration and Success BuFfer Training*，https://arxiv.org/abs/2606.30345（v4, 2026-08-10）
- 相关代码：https://github.com/LianjiaTech/drift ；项目博客 https://lianjiatech.github.io/drift/blog
- 官网预告：https://qingkeai.online/blog/DRIFT

> ⚠️ 提炼方式说明：本期为官网期，B站合集无对应视频，字幕无从获取。本纪要根据该期对应的公开材料还原——官网预告（含讲者与提纲）+ DRIFT 论文原文（arXiv:2606.30345）。Q&A 即兴内容未覆盖，talk 现场对"OPD 策略"的具体表述以论文的 self-distillation 设定为准。

## 一句话总结

DRIFT 回答"没有外部专家时，模型怎么靠自己的产出持续变强且不崩"：核心不是在 GRPO（RL）和 SDPO（自蒸馏）之间二选一，而是按**每道题的历史通过率**做难度路由——难题走自蒸馏（用成功缓冲里存的历史正确轨迹做 token 级纠正）、能力边界题走 RL 探索并用节奏门控放大关键决策点的更新、已掌握题减少更新；再加两阶段课程（先蒸馏热身攒缓冲、后 RL+蒸馏混合）。在五个科学推理/工具使用基准、三个模型上全面超过 GRPO 与 SDPO，Qwen3-8B 平均 79.5%（论文 Table 1）。

## 核心

### 背景/问题：自进化信号不会"因材施教"

论文 §1–§2 的问题诊断很具体：

- **GRPO 的粗粒度**：组内奖励同质（全对或全错）时优势塌陷；sequence-level 优势把更新摊到所有 token 上，定位不到决定对错的关键推理步。
- **SDPO 的一视同仁**：自蒸馏把所有成功轨迹当同样监督，不区分这道题是已掌握、偶然蒙对、还是在能力边界——已掌握题被反复模仿（冗余更新扰动策略），高价值边界题强化不足，难题没有正样本时则完全没信号。
- **共同缺失**：没有**跨 batch 持久的题目级学习状态跟踪**；同期 sample-routing 方法只看当前 batch 的即时结果，且高质量正确解不跨 batch 保存复用。

一句话：信号分配不随模型能力演化而调整，是现有自进化范式不稳的根源。

### 方法/设计：两层信号精修

DRIFT 的三个组件互锁（论文 §3）：

1. **动态难度路由（Difficulty Routing）**：用每道题的**历史通过率**估计掌握程度并分流——难题路由到自蒸馏分支，用成功兄弟轨迹（ $\mathrm{sib}^{+}$ ）提供 token 级纠正；能力边界题路由到 GRPO 分支做探索；稳定做对的题降低更新强度，避免无谓扰动。
2. **节奏门控探索（Rhythm Gating）**：只在难度路由的 GRPO 分支上生效的 token 级调制，利用 teacher–student 不确定性动态，在学生决策仍不稳定的位置**选择性放大已验证的策略创新**，鼓励在成功边界附近变异、同时保住已有推理模式（论文称之为 structure-preserving exploration）。
3. **成功缓冲 + 两阶段课程（Success Buffer + Two-Stage Curriculum）**：跨 batch 保存高质量正确轨迹作为蒸馏参考，让当前 batch 很少做对的难题也能吃到历史成功经验；课程第一阶段以自蒸馏快速热身并积累缓冲，第二阶段切换到蒸馏+RL 混合的稳定优化。

其自蒸馏分支沿用 SDPO 的设定：教师是"条件化在成功兄弟轨迹等反馈 $f$ 上的自身策略"，学生与教师在每个 prefix 上用 Jensen–Shannon 散度对齐（论文 §2.1 采用 JSD，理由是其对称性利于保探索与稳定）。

### 实验/实战（论文 §4）

设置：Qwen3-8B 为主模型，另有 Qwen3-4B 与 OLMo3-7B 验证跨规模/跨家族（论文称"三个模型规模"）；五个基准是 SciKnowEval 的 Biology、Chemistry、Materials、Physics 推理子集加 Tool Use（ToolAlpaca 设定）；指标为 mean@16；8×H200 训练 400+ 步。对照 GRPO、SDPO（均以 SDPO 官方框架复现）、SRPO、DistIL、SC-SDPO、PGPO。

- **主结果（Table 1）**：Qwen3-8B 上 DRIFT 五基准平均 79.5%，比最强基线 SRPO 高 2.1 点，比底座起点 49.5% 高 30.0 点，且每个单项都排进前二——不是靠单项拉平均。Tool Use 提升最大（79.2%）：论文解释是工具调用的正确性由 schema 与参数格式定义，成功示范可跨样本干净迁移；STEM 知识更依赖具体题目，增益相对温和。
- **跨规模（Table 2）**：Qwen3-4B 平均 74.6%、OLMo3-7B 平均 72.8%，均为最高；且**起点越弱增益越大**——OLMo3-7B（底座 30.5%）DRIFT 提升 42.3 点、领先在线 GRPO 约 12 点，因为正样本稀缺时成功缓冲的特权回放恰好补上 GRPO 优势塌陷的空洞。对 SRPO 的领先随规模扩大（4B +0.4 → 8B +2.1），论文解读为"自教能力随规模涌现"。
- **消融（Table 3）**：去掉难度路由平均掉到 74.1%（Biology 与 Tool Use 掉得最明显）；去掉成功缓冲 74.9%；去掉热身阶段 71.7%（早期能力引导与缓冲积累是后续稳定的前提）。
- **训练动态（§4.3）**：actor 熵早期上升、约 200 步后稳定，响应长度有界；缓冲快速增长后饱和；简单题占比渐升、中等题始终占主导——路由决策随能力演化持续调整。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| 五基准平均（Qwen3-8B） | 最强基线 SRPO 77.4% | DRIFT 79.5% | 论文 Table 1 |
| 五基准平均（Qwen3-8B） | 复现 SDPO 72.0%、GRPO 70.0%* | DRIFT 分别 +7.5 / +9.5 点 | 论文 Table 1、Abstract（*为引用值） |
| Tool Use（Qwen3-8B） | 复现 SDPO 67.7% | DRIFT 79.2%（+11.5） | 论文 Table 1 |
| 相对底座提升（Qwen3-8B） | 起点平均 49.5% | +30.0 点至 79.5% | 论文 §4.2 |
| Qwen3-4B / OLMo3-7B 平均 | 各自最强基线 | DRIFT 74.6% / 72.8%（OLMo3-7B 相对底座 +42.3 点） | 论文 Table 2 |
| 消融：去难度路由 / 去成功缓冲 / 去热身 | 完整 DRIFT 79.5% | 74.1% / 74.9% / 71.7% | 论文 Table 3 |
| 训练配置 | — | 8×H200，400+ 步，mean@16 评测 | 论文 §4.1 |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **题目级历史通过率是廉价的课程学习信号**：在现有 GRPO 流水线里维护一个 per-problem pass-rate 状态，按难度分流"蒸馏/探索/跳过"，不需要改奖励函数就能减少已掌握题上的冗余更新——对 coding RL 的题目池同样适用。
  2. **成功缓冲 = 在线数据的跨 batch 回放**：把历史正确轨迹存下来给难题做蒸馏参考，本质是把"这一 batch 没采样到的正样本"补回来；对 agentic 任务里稀疏成功轨迹的复用是直接可借的 infra 件。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. 路由决策依赖持久状态（per-problem 历史、成功缓冲），这要求 RL 训练系统把数据侧状态做成可checkpoint 的一等公民，而不是无状态的 batch 流——状态管理成本要计入框架选型。
  2. 论文特意按训练步数而非墙钟时间报告结果（§4.1），方便跨硬件复现对比；自研框架的实验记录值得照此规范。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：节奏门控利用的 "privileged teacher–student uncertainty" 中，教师是条件化在反馈 $f$ 上的自身策略——在没有任何外部 verifier 的纯自进化设定下， $f$ 的来源与可靠性如何保证不被模型自己的错误成功污染缓冲？想细看论文 §3 与附录的实现。
- 现场内容不可还原：讲者对"RL 与蒸馏何时切换"的现场判据、AMA 讨论无法从论文推知；若 B站后续补录视频，应以视频为准修订本纪要。

## 原文金句（1-2句）

> "training signals are allocated suboptimally: already-mastered problems keep receiving redundant updates, hard problems fail to yield reliable supervision, and high-value boundary cases are under-explored" —— 论文 §1 对现有自进化范式的一句诊断

> "when positive samples are scarce, the privileged replay from the success buffer fills exactly the void left by GRPO's advantage collapse" —— 论文 §4.2，成功缓冲在弱起点模型上增益最大的机制解释
