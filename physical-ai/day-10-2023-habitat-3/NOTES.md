# Day 10 — Habitat 3.0：把 embodied AI 从“独居 agent”推进到人机共居

## 元信息
- Title: Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots
- Authors / Org: Xavier Puig, Eric Undersander, Andrew Szot et al. / FAIR at Meta, Georgia Tech, Simon Fraser University, UC Berkeley, University of Washington, Stanford University, Carnegie Mellon University
- Link / arXiv / Blog: https://arxiv.org/abs/2310.13724
- Project: https://aihabitat.org/habitat3/
- Official code: https://github.com/facebookresearch/habitat-lab
- Venue: ICLR 2024
- Date read: 2026-08-31
- Tags: [physical-ai, habitat-3, habitat-lab, embodied-ai, human-robot-interaction, social-navigation, social-rearrangement, multi-agent-rl, human-in-the-loop]
- Thread: physical-ai
- Folder: day-10-2023-habitat-3
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-10-2023-habitat-3

## 一句话总结
Habitat 3.0 在 Habitat-Sim / Habitat-Lab 上加入高效多样的 SMPL-X humanoid、可回放的 human-in-the-loop 工具，以及 Social Navigation / Social Rearrangement 两个标准任务，使人机协作策略能在未见家庭、未见伙伴和真实人类控制的 avatar 上训练与评测；关键不只是在模拟器里“加个人”，而是把动态伙伴、协作效率和安全距离变成闭环 RL 问题。

## 和之前工作的关系

- **接了哪条线：** Day02 MuJoCo 与 Day03 Isaac Lab 解决物理和并行模拟；Day04–06 研究可学习的世界模型；Day07–08 解决 humanoid 低层全身控制；Day09 RT-2 / OpenVLA 连接语义感知与动作。Day10 把这些能力放进多人共享、动态变化的家庭环境，转向 embodied navigation、协作和标准化评测。
- **补了哪个短板：** Day09 的 VLA 多数在单机器人、桌面 manipulation、单帧观测上评估，几乎不测人类伙伴会移动、抢占空间或改变任务状态。Habitat 3.0 明确定义 social navigation、zero-shot coordination、collision 与 relative efficiency，补上“和人一起做事”的环境与指标。
- **替代 / 分叉 / 改进：** 它不替代 VLA 或 torque controller，而是提供更高层的训练 / 评测 substrate。低层 locomotion、pick/place、VLA action proposal 都可以成为 Habitat policy 的 skill；Habitat 负责场景、多人状态、任务生成、rollout 和评测。
- **对之前 Day X 的直接对比：** Day09 用 web + robot demonstrations 扩展 semantic generalization；Day10 用伙伴 population 与未见场景扩展 interaction generalization。前者问“看懂指令后做什么”，后者问“另一位 agent 也在行动时，怎样安全、高效地配合”。

## 为什么今天读它

路线图 Day10 从 VLA 切到 Habitat。Physical AI 的可靠性不能只在静态桌面和单 agent 成功率上衡量；真实家庭是部分可观测、多人、动态且安全敏感的系统。Habitat 3.0 的价值在于把 humanoid simulation、HITL 数据 / 评测、multi-agent RL 和可复现 benchmark 接成一条 pipeline，也暴露出高层 policy 依赖 oracle skill 时会被低层误差击穿的典型分层系统问题。

## 今天的 3 问
1. 自动 humanoid population 上的 policy ranking 在多大程度上能预测真实人类协作体验；需要什么 behavior coverage 才不把 simulator partner 过拟合误当作 generalization？
2. 为什么扩大 scripted partner population 能改善 zero-shot coordination，而 8 个随机初始化的 learned partners 仍缺乏足够行为多样性；怎样用数据驱动指标而不是 population size 衡量 partner diversity？
3. 高层策略用 oracle navigation / pick / place 训练、换成 learned skills 后性能骤降，说明层间接口缺了什么：failure state、uncertainty、retry / recovery，还是端到端 joint fine-tuning？

## 核心

1. **Motivation：embodied AI 不能永远是“独居 agent”**
   - 传统 Habitat / embodied benchmarks 通常假设环境只因单个 agent 的动作而变化，但家用辅助机器人必须与会移动、会改动环境、行为偏好各异的人共享空间。
   - 真机 + 真人训练成本高、难规模化且有安全风险，也很难做标准化重复实验；因此需要同时支持 realistic humanoid、real human-in-the-loop 和机器人 policy 的高速仿真平台。
   - Habitat 3.0 的三项核心贡献是：高效 humanoid simulation、HITL 基础设施、Social Navigation / Social Rearrangement 两个协作任务及其 learned / heuristic baselines。

2. **System / Method：物理 skeleton、视觉 skin、分层行为与 HITL client-server**
   - **Humanoid representation：** 用 articulated skeleton 做碰撞与物理，用 skinned surface mesh 做视觉；SMPL-X 参数化 pose / shape。系统离线缓存 12 个基础 body（4 male、4 female、4 neutral）的 rig、mesh 和 blend-shape 结果，运行时主要做 linear blend skinning，以少量视觉保真损失换吞吐。
   - **Motion / behavior：** 高层 planner 或 learned policy 组合 navigation、pick、place 等低层 skill。行走用 AMASS motion clip 循环并沿 waypoint 投影；reach / pick / place 用 VPoser 预计算 pose library，运行时按目标手部位置插值，再以 kinematic attach / detach 处理物体。
   - **Human-in-the-loop：** server 负责仿真、agent inference 与 avatar control，client 负责平台相关的渲染和输入；支持键鼠、浏览器与 VR，并可记录 / 回放高层 action、精确 pose 与物体 trajectory，也能从不同 camera 重新渲染。
   - **Social Navigation：** Spot 在未见场景中寻找并保持距 humanoid 1–2 m，输入 depth、humanoid detector、相对位置 / 朝向，DD-PPO recurrent policy 输出局部 linear / angular velocity；指标覆盖 Finding Success、SPS、Following Rate、Collision Rate。
   - **Social Rearrangement：** robot 与 humanoid 把两个物体搬到目标位置。两层 policy 中，高层 DD-PPO 从预定义 skill library 选择 navigate / pick / place；训练 population 可由单一伙伴、1–4 个 planner partner 或 8 个 jointly learned partner 组成，评测强调未见伙伴的 zero-shot coordination。

3. **Training / Data Details：高吞吐 rollout + partner population + 可验证 reward**
   - 场景来自 HSSD，Social Navigation 使用 37 train / 12 validation / 10 test scenes；两类任务都用 Boston Dynamics Spot 与 humanoid avatar。
   - Social Navigation 用 4×A100、每卡 24 parallel environments、每次 update 收集 128 steps；ResNet-18 + 2-layer LSTM 约 8.5M 参数，DD-PPO 约 200M environment steps（约 4 天）收敛，3 seeds。
   - Navigation reward 按 geodesic distance 塑形：太近（<1 m）奖励远离，1–2 m 内给常量奖励，太远则奖励接近；保持 1–2 m 且朝向 humanoid 400 steps 得 +10，collision 终止，另有 -0.1 slack penalty。
   - Social Rearrangement 同样用 4×A100、96 parallel environments、100M steps、ResNet-18 + 2-layer LSTM；reward 为成功 +10、每个 pick / place subgoal +5、collision -5 并终止、每步 -0.005。所有结果按 3 seeds 汇报。
   - HITL 评测覆盖 30 participants，比较 human solo、Learn-Single 和 Plan-Pop3。回放机制使同一策略 / 任务可保存并重渲染，为 failure analysis 和数据闭环提供可追溯轨迹。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — physics / appearance 解耦并缓存人体形变：** skeleton 管碰撞，skin mesh 管视觉，SMPL-X / VPoser 的昂贵部分离线缓存，动作运行时做 motion projection / pose interpolation；robot + humanoid 在单 GPU 16 environments 下达到约 1191 FPS。
   - **Trick 2 — 用 partner population 训练 coordination，而不是只做 scene randomization：** Plan-Pop3/4 的多种 scripted strategy 让 policy 学会适应“伙伴会做什么”，比单一伙伴或仅靠随机初始化得到的 learned population 更稳健。
   - **Trick 3 — 分层 action space + 可中断 skill：** 高层选择语义 skill，低层执行导航 / 抓取；当 robot 距 humanoid 小于 1.5 m 时终止当前 skill 并重规划，由简单的 safety interrupt 诱导出后退让路、改拿另一件物体等 reactive behavior。

5. **Results：仿真吞吐强、协作泛化可见，但 oracle-to-learned gap 很大**
   - **速度：** 单环境 robot 为 `245±19 FPS`、humanoid 为 `188±2 FPS`；双 agent 时 robot-robot `150±13 FPS`、robot-humanoid `136±8 FPS`；单 GPU 16 environments 时 robot-humanoid 为 `1191±3 FPS`。
   - **Social Navigation：** heuristic expert 的 Finding Success / SPS / Following Rate / Collision Rate 为 `1.00 / 0.97 / 0.51 / 0.52`；无地图 end-to-end RL 为 `0.97 / 0.65 / 0.44 / 0.51`。RL 虽路径效率较低，但学到 anticipation、backing-up 和在窄道让路。
   - **Partner generalization：** 单伙伴 Learn-Single 从 train-partner SR `98.50%` 降到 unseen-partner `50.94%`；Plan-Pop3 在 unseen partner 上达到 `71.79%`，Plan-Pop4 为 `71.32%`，说明“行为多样的伙伴集”比只优化已知搭档更重要。
   - **层间 sim-to-real 类比 gap：** Plan-Pop3 高层策略从 oracle skill 切换到 learned low-level skills、且不重训时，train-pop SR 从 `77.79%` 降到 `41.09%`，ZSC SR 从 `71.79%` 降到 `21.44%`。高层若看不到低层 uncertainty / failure，就会严重 distribution shift。
   - **真人协作：** 30 人 HITL 中，solo 平均 1253.17 steps；Learn-Single 降至 936.60（relative efficiency 133.80），Plan-Pop3 为 1015.05（123.46）。自动 humanoid evaluation 能反映相对排序，但论文并未证明它可完全替代真实用户评测。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？** 论文明确在未见 HSSD 场景、未见 humanoid policies 与 30 位真实人类控制者上评测；partner-population training 改善了未见搭档 SR。不过结果依赖 Habitat 的高速 simulator、任务定义、传感器和 oracle skill 设计，因此这是“框架 + benchmark + policy”共同结果，不应归因于单一 network。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：**
  1. Partner diversity 对 robotics RL 很像 post-training 的 task / opponent / user distribution：不能只数样本或 partner 数，要测 behavioral coverage、held-out collaborator success、worst-bucket failure。
  2. Oracle skill → learned skill 的性能坍塌对应 agentic RL 里 planner 在完美 tool 假设下训练、上线却遭遇 latency / error / partial execution；需要把 tool failure、uncertainty 和 recovery 放进 training loop。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：** 把 simulator FPS、并发 env 数、environment steps、GPU-hours 与 sample efficiency 一起记；evaluation 按 scene × partner × skill backend 分桶，并强制记录 collision、interrupt、replan、subgoal completion 与 HITL replay，避免 aggregate SR 掩盖安全和协作失败。

## 疑问 / 下一步

- **没看懂 / 想深挖：** 自动 avatar population 要达到怎样的行为覆盖，才能可靠预测真实人的长尾反应？仅用 SR / RE 排序不足以识别礼貌、可预测性、个人空间与主观信任之间的差异。
- **如果要复现 / 小规模试，第一个实验做什么？** 用 Habitat-Lab v0.3.0 的 Habitat 3.0 multi-agent config 跑最小 Social Navigation evaluation：固定同一批 scenes / seeds，对比完整 sensor、去掉 humanoid GPS、去掉 arm depth 三组；记录 S / SPS / F / CR、FPS 与 collision replay。先只跑 evaluation / 短 rollout，不复现 200M-step full training。
- **下一步：** 路线图当前表只定义到 Day10；下一日应先在 README 补齐并锁定 Day11–30 映射，再按序进入 VLA / Habitat data scaling，而不是临时选题。

## 原文金句 (1-2句)
> “Today’s embodied AI agents are largely hermits – existing within and navigating through virtual worlds as solitary occupants.” — Habitat 3.0, Introduction

> “We believe it is now time to more comprehensively study and develop social embodied agents that assist and cooperate with humans.” — Habitat 3.0, Introduction

## 今晚产出
- [ ] 画 `HSSD scene → humanoid population → robot policy → SocialNav/SocialRearrange → automated/HITL eval` 数据流图
- [ ] 从 Habitat-Lab 跑通一个最小 multi-agent episode，记录环境版本、FPS、seed 与 replay 路径
- [ ] 复算 Social Navigation 的 S / SPS / F / CR，并检查 collision termination 与 1–2 m safety band
- [ ] 列一张 oracle skill vs learned skill 的 distribution-shift 表：可见状态、失败模式、恢复机制、SR drop
- [ ] 为下一次实验定义 partner-diversity 指标，至少覆盖 task allocation、yielding、no-op / waiting 与 adversarial conflict

## 连接
- 上一篇: Day09 — RT-2 / OpenVLA（语义感知到动作 token 的 VLA）
- 下一篇预告: Day11 — 待 README 路线图补齐后按顺序执行
- 相关: Day02 MuJoCo；Day03 Isaac Lab；Day07 H2O；Day08 Humanoid-Gym

## 参考链接
- Paper: https://arxiv.org/abs/2310.13724
- Project: https://aihabitat.org/habitat3/
- Official code: https://github.com/facebookresearch/habitat-lab
- ICLR 2024: https://iclr.cc/virtual/2024/poster/19442
