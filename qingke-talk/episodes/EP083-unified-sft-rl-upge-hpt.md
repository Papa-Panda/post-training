# EP083 — 统一 SFT & RL：迈向大语言模型后训练的统一视角

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP083-unified-sft-rl-upge-hpt.html

## 元信息

- 期号：83
- 标题：统一 SFT & RL：迈向大型语言模型后训练的统一视角
- BV：BV1Gz1MBZEGk（https://www.bilibili.com/video/BV1Gz1MBZEGk/）
- 时长：未确认（2025-10-28 直播安排 20:00–21:00）
- 提炼日期：2026-10-02
- 信息来源：论文还原（B站字幕接口在本环境不可达，未能取得 AI 字幕）
- 讲者：吕兴泰（Xingtai Lv，清华大学二年级博士生，导师周伯文 Bowen Zhou）
- 相关论文：Towards a Unified View of Large Language Model Post-Training（Xingtai Lv、Yuxin Zuo、Youbang Sun 等，清华大学 + 上海人工智能实验室 + WeChat AI），https://arxiv.org/abs/2509.04419（2025-09）
- 相关代码：https://github.com/TsinghuaC3I/Unify-Post-Training

> ⚠️ 提炼方式说明：本期字幕未能取得，本纪要根据公开材料还原——青稞Talk 官网预告文 + 对应论文（arXiv:2509.04419）。官网三段核心提纲（后训练算法概览 / 统一理论框架 UPGE / 动态算法 HPT）与论文结构对应；视频实录与 Q&A 未覆盖。本期官网页另提供直播配套 PPT 下载（知识星球），未纳入本纪要。

## 一句话总结

SFT 与 RL 不是两种对立范式，而是**同一个策略梯度估计器在不同数据分布假设与偏差-方差取舍下的两个实例**。论文把这个估计器（UPGE）拆成四个可互换部件——稳定化掩码、参考策略分母、优势估计、对数似然梯度——并证明 SFT 只是"优势恒为 1、参考策略取当前策略"的特例。在这个统一视角上长出的算法 HPT 做法极简：按每道题的实时 rollout 正确率逐题门控，做对了就继续 RL 探索、全军覆没才切 SFT 喂示范，在 Qwen2.5-Math-7B 上 AIME 2024 比最强基线高出约 7 个点。

## 核心

### 背景/问题：SFT-then-RL 流水线的结构性浪费

后训练有两类数据：离线示范（人或其他模型写的）喂 SFT，在线 rollout（模型自己生成的）喂 RL。标准做法是串行流水线：先 SFT 抬升能力、再 RL 精修，但资源消耗大、阶段衔接要精细调参；而纯 Zero-RL 对弱模型或难题常因探索不到奖励信号而失效，纯 SFT 又压制探索、易过拟合示范分布。近期 LUFFY（固定比例混离线数据）、SRFT（按熵动态调权重）等混合方法经验上有效，却缺一个理论解释：为什么这两个信号能放进同一个优化过程而不打架？这是本期的立论缺口。

### 方法/设计：UPGE 四部件统一 + HPT 逐题门控

1. **统一估计器（UPGE）**：把 SFT、PPO、GRPO、REINFORCE、CISPO、GSPO、LUFFY、SRFT 的梯度全部写成同一形式，差异只来自四个部件的取值：
   - **稳定化掩码（stabilization mask）**：PPO 的 clip、CISPO 的掩码等，决定何时关掉当前梯度；
   - **参考策略分母（reference policy denominator）**：梯度里除以谁的概率——RL 类方法除以 $\pi_{\theta_{\text{old}}}$ ，SFT 除以 $\pi_{\theta}$ 本身，LUFFY/SRFT 的离线项则取 1；
   - **优势估计（advantage estimate）**：SFT 恒为 1（示范无条件全盘接受），GRPO 用组内标准化奖励，PPO 用 GAE；
   - **对数似然梯度（likelihood gradient）**：各方法共享的骨架。
   
   结论：这些梯度并不冲突，而是对同一目标的不同"带噪测量"，各自有不同的偏差-方差特性——SFT 低方差但有分布偏置，RL 无偏但高方差。这就把"要不要混"从信仰问题变成了估计器设计问题。
2. **HPT（Hybrid Post-Training）**：混合损失 $\mathcal{L} = \alpha \mathcal{L}_{\text{RL}} + \beta \mathcal{L}_{\text{SFT}}$ ，系数由**每道题的实时 rollout 表现** $P$ （一组采样里 verifier 给出的正确率）门控： $P \le \gamma$ 时 $(\alpha, \beta) = (0, 1)$ 纯 SFT， $P > \gamma$ 时 $(\alpha, \beta) = (1, 0)$ 纯 RL。实验中 Qwen 家族取 $\gamma = 0$ ——一道题 8 次采样一次都没做对，才切去学示范；做对了就继续自我探索。这个二值化的极简实现就是论文主结果的全部"算法"。
3. **为什么这样切合理**：RL 在"全军覆没"的题目上梯度信号为零（组内优势恒为 0），继续 RL 是空转，此时 SFT 注入的新知识恰好补上探索的前置条件；而模型已能做对的题目，SFT 只会拉低熵、压制探索。门控即"按题难度自适应课程"。

### 实验/实战：数字

设置：Qwen2.5-Math-1.5B/7B、LLaMA3.1-8B；GRPO 为 RL 骨架；每题采样 8 条（温度 1.0），最大生成 8192 token；6 个数学基准（AIME 2024/2025、AMC、MATH-500、Minerva、OlympiadBench）+ 2 个 OOD 套件（ARC-c、GPQA-Diamond）；8×A800 80GB。

- **Qwen2.5-Math-7B（分布内平均）**：HPT 52.7，超过 LUFFY 49.8、SFT→GRPO 46.5、SFT 44.5、GRPO 43.1；AIME 2024 单项 HPT 33.0 vs LUFFY 26.1（+6.9）、vs SRFT 18.4（+14.6）。OOD 平均 HPT 62.3 同样第一（GPQA 42.9 vs LUFFY 39.4）。
- **小模型与跨家族**：Qwen2.5-Math-1.5B 平均 HPT 41.9 vs LUFFY 34.7、SFT 36.9；LLaMA3.1-8B 平均 HPT 18.2 vs LUFFY 13.2、GRPO 9.6——弱底座上差距更大。
- **探索不缩水**：Pass@k（k 到 1024）曲线 HPT 最高，不是介于 SFT 与 GRPO 之间；MATH-500 独占解题分析显示 HPT 新解出的题目随难度升高而增多（对 GRPO 为 +58/-15），同时已会题目基本不丢——缓解灾难性遗忘。
- **消融结论（论文口径）**：离线数据用 SFT 学优于用 off-policy RL 学；阈值并非越松越好， $\gamma = 0$ 的严格口径在 Qwen 上最好；切到 SFT 的步不需要 rollout，反而省算力。

### 结论/观点（区分事实与判断）

- 事实：UPGE 的四部件归约覆盖了主流方法的梯度形式（论文 Table 1 给出逐条写法）；HPT 的主结果与消融有完整表格支撑。
- 判断（讲者立场）：SFT 与 RL 是互补而非冲突的同一过程的两个面；固定混合比例或僵化训练顺序都是次优的，**按实时表现动态切换**才是与统一理论一致的算法形态。这也解释了一个常见现象——SFT 学到的长链推理模式在切回 RL 后不被遗忘，因为两者本就在优化同一目标。

## 关键数字

| 指标 | 基线 | 结果（HPT） |
|---|---|---|
| Qwen2.5-Math-7B AIME 2024（avg@32） | LUFFY 26.1 / SFT→GRPO 25.7 | 33.0 |
| Qwen2.5-Math-7B 分布内平均 | LUFFY 49.8 | 52.7 |
| Qwen2.5-Math-7B OOD 平均（ARC-c + GPQA） | LUFFY 60.1 | 62.3 |
| Qwen2.5-Math-1.5B 六基准平均 | SFT 36.9 / LUFFY 34.7 | 41.9 |
| LLaMA3.1-8B 六基准平均 | LUFFY 13.2 | 18.2 |
| 门控阈值 $\gamma$ （Qwen 家族） | — | 0（8 次采样全错才切 SFT） |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **把 SFT 数据当"救援通道"而不是第一阶段**：训练中按题/任务统计 rollout 通过率，全失败的样本自动路由到示范学习——这对 coding data 尤其直接（单元测试通过率就是现成的 $P$ ）。
  2. 混合训练的旋钮优先级重排：先试逐样本动态门控，最后才考虑全局固定混合比例；UPGE 框架提示，调 mask/分母/优势的取舍本质是偏差-方差工程。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：HPT 的门控信号来自训练内已有的 verifier 结果，不需要额外评测管线；且 SFT 步省 rollout，整体算力低于 SFT→GRPO 串行流水线——"用在线表现调度离线数据"是便宜的自适应。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：UPGE 是梯度形式的归约，但 SFT 与 RL 的数据分布差异（示范分布 vs 当前策略分布）在这套估计器里只体现在分母与优势上——分布偏移本身（如示范质量参差）如何进入同一框架？值得对照 LUFFY 的 policy shaping 再看。
- 本期字幕与配套 PPT 若后续取得，应回补讲者第一段的算法概览与未来研究讨论。

## 原文金句（1-2句）

> "These approaches are not in contradiction, but are instances of a single optimization process."——论文摘要对全期主题的定调：SFT 与 RL 不是对立面，而是同一优化过程的两个实例。
