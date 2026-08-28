# Day 07 — H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation

## 元信息
- Title: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation (H2O)
- Authors / Org: Tairan He, Zhengyi Luo, Wenli Xiao, Chong Zhang, Kris Kitani, Changliu Liu, Guanya Shi / Carnegie Mellon University
- Link / arXiv / Blog: https://arxiv.org/html/2403.04436v1
- Date read: 2026-08-28
- Tags: [physical-ai, humanoid, whole-body-control, teleoperation, sim2real, reinforcement-learning, motion-retargeting]
- Thread: physical-ai
- Folder: day-07-2024-h2o-whole-body-control
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-07-2024-h2o-whole-body-control

## 一句话总结
H2O 把 AMASS 人体动作先经形态对齐和 privileged policy 做 **sim-to-data 可行性过滤**，再用只依赖真实机器人可观测量的 PPO 全身跟踪策略、动力学随机化与 PD 控制实现 Unitree H1 的零样本 sim-to-real；完整方法在 10k 条未清洗动作上的仿真跟踪成功率为 72.5%，并以单 RGB 相机实时完成走路、踢球、后跳、推车等动作。

## 和之前工作的关系

- 接了哪条线：Day02 MuJoCo 的接触动力学与 Day03 Isaac Lab 的并行仿真 / domain randomization，终于落到一个完整的 humanoid 控制闭环：人体目标 → 动作重定向 → 仿真 RL → 真机 PD 控制。
- 补了哪个短板：Day04–06 的 Genie / UniSim / DreamerV3 主要回答“如何学世界模型并在想象中训练”；H2O 补上真实 humanoid 中“目标动作怎样表示、不可行动作怎样过滤、可部署状态怎样设计、sim2real 怎样落地”。
- 替代 / 分叉 / 改进：相对依赖显式接触状态、力传感器或简化动力学的 model-based whole-body controller，H2O 用 goal-conditioned RL 隐式学习接触与平衡；相对只重放离线动作，它支持 RGB + 3D pose 的实时控制。
- 对之前 Day X 的直接对比：与 Day05 UniSim 的 `action-conditioned world model` 路径互补——UniSim 学一个可交互环境供 policy 学习，H2O 直接在物理模拟器中学真实可部署的 tracking policy；与 Day06 DreamerV3 不同，H2O 不是 latent imagination，而是高吞吐物理仿真 + PPO。

## 为什么今天读它

Day07 路线图进入 humanoid whole-body control。H2O 的价值不只是“动作很酷”，而是把三种 gap 拆开处理：**representation gap**（目标状态）、**embodiment gap**（sim-to-data 过滤）、**sim-to-real gap**（可部署观测 + 随机化）。它把此前 simulator / world model 的抽象能力具体化成可验证的全身控制系统。

## 今天的 3 问
1. 为什么普通 inverse-kinematics retargeting 不够，`sim-to-data` 为什么能让“更少但更可行”的数据反而训练出更强 policy？
2. 如何在不依赖仿真 privileged state / 接触力的条件下，设计既能表达全身目标、又能在真实机器人上实时获得的 observation / goal state？
3. 哪些 domain randomization、reward 与 early termination 设计真正承担了 zero-shot sim-to-real，代价又是什么？

## 核心
1. **Motivation**：传统 whole-body teleoperation 常依赖简化动力学、预设/测量接触状态、外骨骼或力传感器，难以扩展到自由动态动作。图形学里的 humanoid imitation 虽能在仿真中生成复杂动作，却常使用真机不可得状态、过大关节力矩或非物理辅助力。H2O 要用一个 policy，在 full-sized humanoid 上实时跟踪开放式人类全身动作。

2. **System / Method**：
   - **Human → robot retargeting**：先优化 SMPL body shape，使 12 个对应关节贴合 H1 形态；再最小化 12 个关节位置差，重点保持 ankles / elbows / wrists 等末端轨迹。
   - **Sim-to-data cleaning**：对约 10k 条 retargeted AMASS motion，训练可访问 778 维全刚体 privileged state、且无 domain randomization 的 imitation policy；把连这个“仿真能力上界”都跟不住的动作判为 embodiment-infeasible，留下约 8.5k 条 clean motions。
   - **Deployable goal-conditioned policy**：PPO policy 的 proprioception 只用 joint position/velocity、root linear/angular velocity、projected gravity 和上一动作；goal 用 8 个 keypoints（肩、肘、手、踝）的参考位置、tracking error 和参考速度。输出 19 维 joint targets，由 PD controller 转成 torque：$\tau=K_p(a_t-q_t)-K_d\dot q_t$。
   - **Deployment**：1080p RGB webcam + HybrIK 3D pose estimator（30 Hz）产生人类目标；H1 内置传感器以 200 Hz 提供其余 proprioception。实验中 root linear velocity 仍由 50 Hz MoCap 提供，这是“单 RGB”叙事之外的重要系统依赖。

3. **Training / Data Details**：
   - 数据来自 AMASS 的约 13k motion sequences；启发式预过滤与 retargeting 后约 10k，再由 privileged imitator 过滤为约 8.5k feasible sequences。
   - Reward = penalty + regularization + task imitation。虽然 observation 只含 8 个目标 keypoints，训练 reward 对全部 joints / bodies 提供 DoF position/velocity、body position/rotation/linear/angular velocity 六类 dense signal。
   - Sim2Real 随机化覆盖 friction $U(0.2,1.1)$、base CoM offset $U(-0.1,0.1)$ m、link mass $0.7$–$1.3\times$、PD gains $0.75$–$1.25\times$、torque noise、20–60 ms control delay、每 5 s 横向 push 及 flat/rough/low-obstacle terrain。
   - Early termination：base height < 0.3 m、projected gravity 的 x/y 分量 > 0.7，或平均 link tracking distance > 0.5 m。
   - Verifiable signal：仿真中若任一时刻平均 body distance > 0.5 m，则判 imitation failure；同时报告 global / root-relative MPJPE 与 acceleration / velocity error。

4. **Key Tricks**：
   - **用 simulation 做 data quality model**：privileged policy 不是最终 policy，而是 morphology-aware feasibility filter；先把“机器人根本做不到”的目标清掉，再谈 scaling。
   - **训练时 privileged reward，部署时 non-privileged observation**：部署输入保持传感器可得，但 reward 仍可用仿真真值密集监督全部刚体，形成 asymmetric information pipeline。
   - **随机化覆盖完整 control stack**：不仅 randomize mass/friction，也显式覆盖 PD gains、torque noise、control delay、外力和地形；把 actuator / latency / disturbance 一起纳入训练分布。

5. **Results**：
   - 在 10k 条未清洗 retargeted AMASS 序列上，完整 H2O 的 tracking success 为 **72.5%**，高于不做 sim-to-data 的 **67.9%**，也高于 reduced goal state 的 **53.2%**。
   - clean data scaling 从 0.1% / 1% / 10% / 100% 时，成功率为 52.0% / 58.8% / 61.3% / 72.5%，说明数据覆盖仍有效，但少量数据配合强 randomization 已有显著泛化。
   - 真机 Unitree H1 展示 walking、back jumping、kicking、turning、waving、pushing、boxing 等动态动作，并在外力扰动下保持平衡；论文未报告统一的真机任务成功率，因此不能把演示等同于完整 benchmark。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？H2O 在全 10k 条未清洗动作上做仿真评测，并对 noisy real-time pose goals 做 zero-shot 真机演示；但没有跨 humanoid embodiment 或标准化真机 success-rate 证据。现有 ablation 显示，**data cleaning + state design + randomization pipeline** 的贡献比单纯换一个 policy architecture 更明确。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. `sim-to-data` 很像 post-training data selection：用强 verifier / teacher 先筛去不可满足样本，数据可学性比原始规模更重要。
  2. privileged-train / deployable-inference 类似训练期可以用昂贵 judge / dense signal，线上 policy 只保留低延迟、可观测接口；设计重点是信息边界而非只看模型。
- Infra 视角：将 pipeline 拆成 retargeting、feasibility scoring、cleaned-dataset versioning、massively parallel PPO、randomization sweep、sim benchmark、hardware rollout。最值得系统化的是数据 lineage、每种 randomization 的消融、sim / real 指标对应关系，以及自动失败归因。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：privileged policy 失败是否真的等价于动作对某个 humanoid embodiment 不可行？它也可能只是 optimizer / reward / capacity 失败。能否给每段 motion 输出 uncertainty-aware feasibility score，而不是硬过滤？
- 如果要复现 / 小规模试，第一个实验做什么？在 Isaac Lab 或 MuJoCo 上选一个简化 humanoid + 小型 motion subset，比较 `raw retargeted` vs `privileged-policy filtered` 两组 PPO tracking：固定 architecture / budget，测 success、MPJPE、fall rate，并记录过滤阈值与 false rejection。

## 原文金句 (1-2句)
> “To achieve real-time teleoperation of humanoid robots, the state space of RL policy must contain only quantities available in the real world.”

> “By comparing H2O with H2O-w/o-sim2data, we can see that our ‘sim-to-data’ process is effective in obtaining higher success rate, even when the RL policy is trained on less data.”

## 今晚产出
- [ ] 画出 H2O 四段链路：SMPL retarget → privileged feasibility filter → deployable PPO → RGB/H1 rollout
- [ ] 把 observation / reward / action 三张表压成一页，标出 train-only 与 deploy-time 信号
- [ ] 写一个 `sim-to-data` 最小实验设计：过滤阈值、对照组、success / MPJPE / fall-rate 指标
- [ ] 明确限制：root linear velocity 的真机演示使用 50 Hz MoCap；真机结果没有统一 success-rate

## 连接
- 上一篇: Day06 — DreamerV3（latent RSSM + imagined actor-critic）
- 下一篇预告: Day08 — Humanoid Locomotion（robust locomotion / terrain / command tracking）
