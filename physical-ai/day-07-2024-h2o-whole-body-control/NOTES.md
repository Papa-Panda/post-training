# Day 07 — H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation

> 📖 阅读版：https://papa-panda.github.io/post-training/physical-ai/day-07-2024-h2o-whole-body-control/

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

## 大纲
- 问题：full-size humanoid 实时跟踪任意人体全身动作；传统 WBC 依赖显式接触状态与力传感器，不可扩展
- 表示：SMPL 重定向对齐 H1 的 12 个关节；goal 压到 8 个可部署 keypoints
- 数据：778 维特权策略做 sim-to-data 可行性过滤，10k → 8.5k clean motions
- 策略：PPO 全身跟踪，训练用特权稠密奖励、部署只吃可部署观测；19 维关节目标经 PD 转力矩
- 迁移：全栈随机化（摩擦 / 质量 / PD 增益 / 延迟 / 横推 / 地形）实现真机 zero-shot

## 流程图
```mermaid
graph LR
  A[人体动作 AMASS] --> B[SMPL 重定向 12 关节]
  B --> C[特权策略可行性过滤]
  C --> D[清洁动作集 8.5k]
  D --> E[PPO 全身跟踪训练]
  E --> F[随机化 PD 执行]
  F --> G[真机 H1 部署]
```

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
   - **Deployable goal-conditioned policy**：PPO policy 的 proprioception 只用 joint position/velocity、root linear/angular velocity、projected gravity 和上一动作；goal 用 8 个 keypoints（肩、肘、手、踝）的参考位置、tracking error 和参考速度。输出 19 维 joint targets，由 PD controller 转成 torque： $\tau=K_p(a_t-q_t)-K_d\dot q_t$ 。
   - **Deployment**：1080p RGB webcam + HybrIK 3D pose estimator（30 Hz）产生人类目标；H1 内置传感器以 200 Hz 提供其余 proprioception。实验中 root linear velocity 仍由 50 Hz MoCap 提供，这是“单 RGB”叙事之外的重要系统依赖。

3. **Training / Data Details**：
   - 数据来自 AMASS 的约 13k motion sequences；启发式预过滤与 retargeting 后约 10k，再由 privileged imitator 过滤为约 8.5k feasible sequences。
   - Reward = penalty + regularization + task imitation。虽然 observation 只含 8 个目标 keypoints，训练 reward 对全部 joints / bodies 提供 DoF position/velocity、body position/rotation/linear/angular velocity 六类 dense signal。
   - Sim2Real 随机化覆盖 friction $U(0.2,1.1)$ 、base CoM offset $U(-0.1,0.1)$ m、link mass $0.7$– $1.3\times$ 、PD gains $0.75$– $1.25\times$ 、torque noise、20–60 ms control delay、每 5 s 横向 push 及 flat/rough/low-obstacle terrain。
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

<!-- viz:stats: 72.5% tracking 成功率 | 67.9% 不做 sim-to-data | 53.2% reduced goal -->
<!-- viz:flow: 人体目标 → SMPL 重定向 12 关节 → 仿真 RL → 真机 PD 控制 -->
<!-- viz:bars: 数据 0.1% 52.0% | 1% 58.8% | 10% 61.3% | 100% 72.5% -->
<!-- viz:vs: 特权训练信号 | 778 维全刚体状态 + 六类稠密奖励; 仅仿真可用 || 可部署观测 | 本体感觉 + 8 关键点目标; 部署期可测 -->

## 第二轮复习（2026-10-02）

> 本轮做两处事实考证（发表场地 IROS 2024 oral 与 OmniH2O → HOVER 谱系）＋一处叙事修正（"单 RGB"演示中 root 线速度仍由 50 Hz MoCap 提供）。核心收获：把 H2O 从"teleoperation demo"重读为"sim-to-data 数据验算器"——仿真器在数据管道里的第三种坐法（UniSim 当环境、Dreamer 当想象、H2O 当准入考试）；同时，它的特权训练 / 可部署推理信息边界，是这条谱系后来共同的设计底座。

### 元信息修正

1. **发表场地**：H2O 正式发表于 IROS 2024（oral）；NOTES 元信息只记了 arXiv v1 链接（2403.04436，2024-03）。官方代码已开源：LeCAR-Lab/human2humanoid。
2. **谱系补全**：同组 OmniH2O（CoRL 2024，arXiv:2406.08858）把接口泛化为 kinematic-pose 通用接口（VR / 口头指令 / RGB 三种输入 + 灵巧手），并发布 OmniH2O-6 数据集；再之后的 HOVER（ICRA 2025，NVlabs）把多模式跟踪蒸馏成统一 WBC，成为 NVIDIA GR00T WBC 栈的公开前身。H2O 的 retarget → filter → imitate → deploy 是这条线的公共配方，初读时未展开。
3. **"单 RGB"叙事修正**：NOTES 一句话总结写"以单 RGB 相机实时完成"，但正文 §2 已自记——真机演示的 root 线速度仍由 50 Hz MoCap 提供；部署观测的"非特权化"只在 proprioception 上成立。引用时须带这条限定。
4. **定位校准**：初读把 sim-to-data 记成数据清洗的一种；复习校准为**可行性过滤**——过滤标准不是标注质量，而是"这具身体物理上做不做得到"，与质量清洗是两个问题（见核心第 2 点）。

### 一句话总结

35 天后回看，H2O 的真正贡献不是"会跟动作的人形"，而是把 sim2real 拆成三个可分别工程化的问题：表示 gap（goal 只留可部署量）、embodiment gap（用特权策略在数据层先删掉身体做不到的动作）、执行 gap（全栈随机化 + PD 执行）。它给出的可迁移命题是：先验算数据的可学性，再谈 policy 规模——过滤后 8.5k 条训出的策略（72.5%）胜过全量 10k 条（67.9%）。

### 和之前工作的关系

- **vs Day06 DreamerV3（直接对比）**：Dreamer 学一个世界模型，让 policy 在想象里练；H2O 不学模型，物理仿真本身既是训练场，又兼职"数据验算器"。一个把算力花在生成经验，一个花在审经验。对 Day28 的 exploit gap 而言，H2O 的审查跑在物理引擎里，没有可被利用的生成模型——代价是审查本身等于一次完整的策略训练。
- **vs Day03 Isaac Lab**：H2O 是规模层第一个被真正消费的下游——并行仿真的吞吐在这里不只换来"训得快"，而是换来"10k 条候选动作逐一试做"的数据质量。规模层的并行度，就是数据准入考试的考场大小。
- **跨阶段 vs Day19 / 21 / 22**：deployable policy 就是 Day19 的 PPO + Day21 式随机化；但鲁棒性全部买在训练分布里，部署时不做适配——与 Day22 RMA 的"部署时快速辨识"正好互补。分布漂移时先失效的是 H2O 这一路。
- **vs Day15–17（数据专题）**：sim-to-data 是数据 curation 的物理版：过滤标准不是"标得好不好"，而是"这个身体做不做得到"。对应 post-training 里用 verifier 先行过滤不可解样本（见迁移节）。
- **vs Day33 / 34（VLA 阶段）**：H2O 是纯执行层——goal 只有 8 个关键点，没有任务语义。Helix / π0.7 的分层接口最终要落到这类 tracker 上；"gap 被压到执行层"这句话，在 H2O 这里有最具体的形态：goal 表示就是接口合同。

### 核心

1. **三 gap 分工（机制总图）**：表示 gap——goal 只留 8 个关键点（肩 / 肘 / 手 / 踝）的位置、跟踪误差与参考速度，"目标必须写成部署时测得到的量"；embodiment gap——778 维特权策略充当"能力上界探针"，跟不住的动作判为 infeasible（10k → 8.5k）；执行 gap——deployable policy 输出 19 维关节目标，PD 转力矩： $\tau=K_p(a_t-q_t)-K_d\dot q_t$ ，随机化覆盖整条控制链（摩擦 $U(0.2,1.1)$ 、质量 0.7–1.3×、PD 增益 0.75–1.25×、20–60 ms 延迟、周期横推、地形）。三个 gap 的工具完全不同：表示靠接口设计，形态靠数据过滤，执行靠分布覆盖。
2. **sim-to-data 的本质 = 数据准入考试**：特权策略不是最终策略，而是形态感知的可行性判定器。它做的事与 post-training 里"用强 verifier 删不可满足样本"同构：先证明监督信号可满足，再让学生去学。为什么有效：不可行动作给出的梯度互相矛盾（身体物理上到不了那个位姿），删掉它们等于删掉标签噪声——72.5% 与 67.9% 之间那 4.6 个点，就是这部分噪声的定价。
3. **训练 / 部署信息不对称**：训练奖励用全部刚体的六类稠密信号（特权），部署观测只留可部署量。仿真器一身三任：训练场、数据验算器、奖励真值源。这是它与 UniSim（仿真 = 环境）、Dreamer（模型 = 想象）在三层栈里的分工差异——世界模型的第三种住法：住在数据管道里。
4. **数据 scaling 的反直觉形态**：clean 数据从 0.1% 到 100%，成功率 52.0% → 72.5%。随机化先把大部分泛化垫付了，数据覆盖的边际收益递减——与 LLM 的 scaling 直觉相反，因为这里的"泛化"主要由分布覆盖（随机化）而非数据多样性买单。

### 边界

1. 真机只有演示、无统一成功率；72.5% 是仿真数，不可外推真机。
2. Root 线速度真机仍靠 50 Hz MoCap——"部署观测非特权化"最关键的一维没被完全吃掉；严格说真机系统仍含一个实验室仪器量。
3. Privileged filter 假阴性未量化：跟不住 ≠ 物理不可行（可能只是优化器 / 奖励 / 容量失败）；过滤是硬阈值，无 uncertainty 出口——NOTES 疑问区已点名，论文未答。
4. 单 embodiment（H1）、AMASS 实验室动作分布；跨本体迁移无证据（OmniH2O 仍在 H1 上验证）。
5. 随机化只覆盖参数化扰动；传动柔性、磨损、温漂等未参数化的真实 gap 不在分布内——这是它与 RMA / SimOpt 的分工线。

### 迁移到 post-training / Agentic RL Infra

可执行的同构：在 agentic RL 启动前加一道"数据准入考试"——用带完整日志与真实答案的 oracle solver（特权版）对候选任务集预跑，把 oracle 都解不出的任务标为当前不可学并降权（不删：保留 feasibility 分数做课程难度与分层采样），只在可学子集上做 RL。验收判据照 H2O 的实验设计：filtered 组最终成功率不低于 raw 组、样本效率更高（对应 72.5% vs 67.9%）。第二层：privileged-train / deployable-inference 的信息边界——训练期可用昂贵 judge 与稠密信号设计奖励，部署接口只保留可观测、低延迟信号；设计重点从模型转向信息边界。第三层：feasibility 分数跨 solver / 种子的翻转率可作假阴性监控（对应机器人侧的 false rejection），翻转率高的样本优先回炉复核，而不是直接剔除。

### 思考题（综合 Day03 / Day05 / Day06 / Day07 / Day19 / Day21 / Day22 / Day24 / Day33 / Day34）

- **(a) 随机化覆盖 vs 在线辨识的分界**：H2O 把鲁棒性全买在训练分布里（随机化覆盖），部署时零适配。若真机地面摩擦随季节漂移出训练分布，哪类观测量最先报警？请设计一个最小实验，区分"随机化没覆盖"与"需要在线系统辨识（Day24 SimOpt）"：什么指标告诉你该打开 SimOpt，而不是继续加宽随机化？
- **(b) 世界模型的三种坐法**：UniSim 把仿真当环境、Dreamer 把模型当想象空间、H2O 把仿真当数据验算器。若为 Figure Helix 的低层控制器处理家庭场景失败数据，失败样本应回流到哪一截：验算器阈值（改数据准入）、想象模型（改世界）还是真机重采集（改分布）？给出每条回路的单位成本与先失效条件。
- **(c) VLA 与 tracker 的接口合同**：H2O 的 goal 是 8 个关键点的位置目标；Helix / π0.7 的输出是连续 action chunk。若把 VLA 接到 H2O 式 tracker 上，接口合同应放在关键点层还是关节层？分别论证延迟上界、接触鲁棒性、数据量三项支持哪一层，并指出你最不确定的一项。

参考链接（本轮复习）：
- H2O（IROS 2024 oral；arXiv）：https://arxiv.org/abs/2403.04436
- 官方代码（LeCAR-Lab/human2humanoid）：https://github.com/LeCAR-Lab/human2humanoid
- OmniH2O（CoRL 2024）：https://arxiv.org/abs/2406.08858

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/physical-ai/day-07-2024-h2o-whole-body-control/NOTES.md
