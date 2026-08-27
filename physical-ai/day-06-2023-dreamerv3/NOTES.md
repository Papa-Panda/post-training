# Day 06 — DreamerV3：Mastering Diverse Domains through World Models

> Day 06 of physical-ai track, following Day05 UniSim. Focus on compact latent dynamics, imagined actor-critic training, scale-robust objectives, and what is still missing for real-robot sim2real.

## 元信息
- Title: Mastering Diverse Domains through World Models
- Authors / Org: Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap / DeepMind, University of Toronto
- Link / arXiv / Nature: https://arxiv.org/abs/2301.04104v2 / https://doi.org/10.1038/s41586-025-08744-2
- First submitted: 2023-01-10; arXiv v2: 2024-04-17; Nature version: 2025
- Date read: 2026-08-27
- Tags: [physical-ai, world-model, dreamerv3, latent-dynamics, model-based-rl, control, actor-critic]
- Thread: physical-ai
- Folder: day-06-2023-dreamerv3
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-06-2023-dreamerv3

## 一句话总结
DreamerV3 把真实 interaction 压进离散 latent RSSM，在模型想象出的 16-step trajectory 上训练 actor-critic，并用 free bits、KL balancing、symlog/two-hot 与 percentile return normalization 消除跨任务量纲差异；同一套超参覆盖 8 个 domain、150+ tasks，并首次从零、无人工数据/课程地在 Minecraft 收集钻石。

## 和之前工作的关系

- **接了哪条线：** 接 Day04 Genie / Day05 UniSim 的 learned world model，但从“生成可观看的像素世界”切到“学习只需服务控制的 compact latent dynamics”，重点是高吞吐 imagination 与 policy optimization。
- **补了哪个短板：** UniSim 把 video diffusion 包成 environment，视觉丰富但每一步昂贵；DreamerV3 不在 actor rollout 中解码像素，直接在 `(h_t, z_t)` 上预测 reward / continuation / value，可用短 latent rollout 高效复用真实交互。
- **替代 / 分叉 / 改进：** 相对 Day02 MuJoCo / Day03 Isaac Lab 的已知物理 solver，DreamerV3 从经验学习 transition；它不要求接触参数和精确动力学，但会引入 model bias，且论文没有证明真实机器人 sim2real。
- **对之前 Day X 的直接对比：** Day05 UniSim 的 state 是 recent video、transition 是 5.6B diffusion、reward 另训；Day06 的 state 是 discrete latent + recurrent memory、transition/reward/continue 同一 RSSM 内联合学习、actor-critic 完全在 latent imagination 中训练。前者重视觉覆盖，后者重控制闭环与样本效率。

## 为什么今天读它

Day06 路线图指定 DreamerV3。它是从“world model 作为可交互生成器”走向“world model 作为 RL 训练内核”的关键节点：尤其适合对比 Physical AI 中 simulator rollout、replay、actor-learner 和 evaluator 的系统接口。

## 今天的 3 问
1. **什么信息值得进入 control state？** RSSM 怎样用 deterministic recurrent state `h_t` 与 stochastic discrete state `z_t` 在记忆、可预测性和重建信息之间取舍？
2. **模型误差为什么没有迅速毁掉 policy？** 16-step latent imagination、reward/continue prediction、λ-return 和 replay start states 如何限制 compounding error；还有哪些 model exploitation 风险未被解决？
3. **一套超参为什么能跨量纲工作？** free bits、KL balancing、1% unimix、symlog/two-hot 和 5–95 percentile return normalization 分别稳定哪一条信号链？

## 核心

1. **Motivation：不是为每个 domain 调一个 RL recipe**
   - 连续/离散动作、像素/向量输入、稠密/稀疏奖励、2D/3D 环境的 signal scale 与 exploration demand 差异巨大，传统做法依赖 domain-specific tuning。
   - DreamerV3 的目标是固定超参下统一这些差异：先学预测环境的 world model，再在模型内大量想象未来，而不是把每个真实 interaction 只用一次。
   - 对 Physical AI，核心价值不是 Minecraft 本身，而是把昂贵环境交互转换成可复用的 latent rollout；真实机器人每一步更贵，这个 amortization 更重要。

2. **System / Method：RSSM world model + imagined actor-critic**
   - Encoder 将 observation `x_t` 编成离散随机表示 `z_t ~ q(z_t | h_t, x_t)`；recurrent sequence model 用 `(h_{t-1}, z_{t-1}, a_{t-1})` 更新 deterministic state `h_t`；prior `p(z_t | h_t)` 负责无新观测时的 latent rollout。
   - `(h_t, z_t)` 同时预测 observation reconstruction、reward `r_t` 与 continuation `c_t`。世界模型损失由 prediction、dynamics KL、representation KL 三部分组成，权重分别为 1、1、0.1。
   - Actor 与 critic 从 replay 中的 posterior state 起步，在 world model 内想象 horizon `T=16`；critic 学 distributional λ-return，actor 用 REINFORCE estimator（连续、离散动作同一形式）最大化 normalized return 并加 entropy。
   - 与在线 MPC 不同，部署交互时 actor 直接从当前 model state 采样动作，不做 test-time lookahead；计算主要移到训练期 imagination。

3. **Training / Data Details：真实 replay 学模型，latent replay 学行为**
   - World model、critic、actor 在 agent 与环境交互时并行训练；训练序列来自 replay buffer。默认 16 个 environment instances，Minecraft 因环境较慢使用 64 个 remote CPU workers。
   - Control Suite（proprio / visual）预算均为 1M environment steps；Atari 200M frames、ProcGen 50M、DMLab 100M、Atari100k 400k frames、Minecraft 100M environment steps。
   - 所有 Dreamer agent 单卡 A100 训练；默认模型约 200M 参数。论文以 5 seeds 为主，BSuite 与 Minecraft 用 10 seeds。
   - 论文测试 12M–400M 的 6 档模型和不同 replay ratio：更大模型与更多 gradient updates 都提高表现和 data efficiency，但这是以额外训练计算换更少环境交互。
   - 机器人证据来自模拟的 Proprio / Visual Control Suite（locomotion + manipulation）；没有真实机器人、contact-rich manipulation 或 sim2real 闭环实验。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — KL balancing + free bits：** dynamics KL 对 posterior stop-gradient，representation KL 对 prior stop-gradient；两项都在 1 nat 以下截断，避免 latent collapse，又不让视觉复杂度迫使每个 domain 重调 regularization。
   - **Trick 2 — 量纲先变换，再共享超参：** observation 用 `symlog(x)=sign(x)log(|x|+1)`；reward 与 value 用 symexp-spaced bins + two-hot cross-entropy，把梯度尺度与目标绝对值解耦。
   - **Trick 3 — Percentile return normalization：** actor advantage 除以 `max(1, EMA(P95(R)-P5(R)))`，既抗 outlier，也不在 sparse reward 近零方差时放大噪声，从而固定 entropy scale。
   - **Trick 4 — 1% uniform mixture：** categorical posterior / prior（以及 actor）混入 1% uniform，防止概率变成确定分布后出现 KL spike。
   - **Trick 5 — 零初始化 reward/value output：** 避免训练早期随机大 reward / value 延迟学习；这是小但很实用的稳定性工程。

5. **Results：广度和样本效率强，但不是 real-robot 证明**
   - 固定超参覆盖 **8 domains、150+ tasks**；在 Atari、ProcGen、DMLab、Atari100k、Proprio Control、Visual Control、BSuite 等综合比较中匹配或超过相应强基线，并跨域超过同一固定配置的 PPO。
   - DMLab 用 100M steps 超过使用 1B steps 的 IMPALA / R2D2+，论文称 data-efficiency gain 超过 1,000%；对照基线并非为样本效率设计，需保留这一边界。
   - Visual Control Suite（20 个像素输入连续控制任务、1M steps）达到 state of the art，超过依赖 augmentation 的 DrQ-v2 / TD-MPC2；Proprio Control Suite 匹配 DMPO / TD-MPC2。
   - Minecraft 10 个训练 run 都在 100M environment steps 内发现 diamond；使用 1 GPU 约 9 天，无 human data / adaptive curriculum。动作空间仍包含 abstract crafting，并加速 block breaking，不应解读为完全原生键鼠控制。
   - Ablation 显示贡献最大的是 KL balancing + free bits，其次是 return normalization 与 symexp two-hot；性能随 12M→400M 参数单调增加，同时减少所需环境交互。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？**
  - 论文证明的是同一算法/超参跨已分别训练的 domains 稳健，而不是一个预训练 world model zero-shot transfer 到新机器人。模型架构与 robustness recipe 的贡献大于共享世界知识。
  - 结果主要来自模拟 benchmark；对真实传感噪声、actuator delay、接触不连续与 morphology shift 的 transfer 仍未验证。

- **对 Infra → Post-training → Physical AI 迁移的直接启发：**
  1. **Imagination batch ≈ synthetic rollout batch：** 必须记录 world-model checkpoint、replay start state、horizon、policy version、reward/continue head version，才能定位 model exploitation 和 stale-policy 问题。
  2. **稳定性来自 scale-invariant interfaces：** symlog、two-hot、percentile normalization 与 free bits 都是在模块边界消除单位/量纲；类似 Agentic RL 中 reward normalization、value target binning 与 KL floors。

- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：**
  - **可扩展性：** latent rollout 省去逐步像素 diffusion，适合把 imagination 作为 GPU 批处理服务；瓶颈变为 replay sampling、sequence unroll、model/policy version skew。
  - **成本：** replay ratio 越高，样本效率越好但训练 FLOPs 越高；要同时报告 environment steps、gradient steps、wall-clock、GPU-hours，不能只报样本效率。
  - **评测自动化：** 除 return 外，应测 multi-step latent calibration、reward/continue error、policy-conditioned rollout error，以及 model-return 与真实环境 return 的差距。
  - **可复现性：** 固定 seeds、环境版本、action repeat、replay ratio、model size、train ratio 与 checkpoint cadence；论文显示相同“算法名”下这些设置足以显著改变结果。

## 疑问 / 下一步

- **想深挖：** 怎样把 epistemic uncertainty 接进 imagination，让 actor 不能专门走 world model 没见过、但被错误预测为高回报的 latent region？可比较 ensemble disagreement、pessimistic reward 与 periodic real/sim grounding。
- **限制提醒：** 16-step imagination 限制短期 compounding error，却把长程 credit 交给 critic；Minecraft 成功不等于真实机器人上的安全探索；reconstruction-dominant representation 也可能保留与控制无关的视觉细节。
- **第一个小实验：** 在 DM Control walker-walk 上跑小配置，记录每个 replay start state 的 1/5/16-step imagined reward 与真实 rollout 差值，再比较加 uncertainty penalty 前后的 policy return 和 OOD visitation。
- **下一步：** Day07 whole-body humanoid control——从通用 latent model-based RL 切到 humanoid 的 contact、balance、tracking 与 whole-body constraint。

## 原文金句 (1-2句)

> “The algorithm is based on the idea of learning a world model that equips the agent with rich perception and the ability to imagine the future.”

> “Notably, larger models not only achieve higher scores but also require less interaction to solve a task, offering practitioners a predictable way to increase performance and data efficiency.”

## 今晚产出

- [ ] 画一页 `real env → replay → RSSM → latent imagination → actor/critic` 数据流图，标出 posterior / prior 与 stop-gradient
- [ ] 手推 16-step λ-return，并解释 critic 如何把 horizon 外回报 bootstrap 回来
- [ ] 用 20 行伪代码写 imagined actor-critic update，标出 world model / actor / critic 的梯度边界
- [ ] 做一张 UniSim vs DreamerV3 表：state、transition、decoder、reward、rollout latency、model exploitation、real-world evidence

## 连接
- 上一篇: Day05 — UniSim: Learning Interactive Real-World Simulators
- 下一篇预告: Day07 — Humanoid Whole-Body Control
- 相关: Day02 MuJoCo（显式动力学）；Day03 Isaac Lab（GPU physics + sim2real）；Day04/05（像素生成式 world model）

## 参考链接
- Paper (arXiv): https://arxiv.org/abs/2301.04104v2
- Nature: https://doi.org/10.1038/s41586-025-08744-2
