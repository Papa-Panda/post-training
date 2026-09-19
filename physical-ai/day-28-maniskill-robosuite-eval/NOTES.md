# Day 28 — ManiSkill3 / robosuite：可复现评测基准（Reproducible Manipulation Benchmarks）

## 元信息
- Title: ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI（主）+ robosuite: A Modular Simulation Framework and Benchmark for Robot Learning（对照）
- Authors / Org: ManiSkill3 — Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse-kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Viswesh Nagaswamy Rajesh, Yong Woo Choi, Yen-Ru Chen, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, Hao Su（UCSD / Hillbot，另有 CMU、TU Dresden、清华、King's College London）；robosuite — Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Kevin Lin, Abhiram Maddukuri, Soroush Nasiriany, Yifeng Zhu（robosuite.ai）
- Link / arXiv: ManiSkill3 — https://arxiv.org/abs/2410.00425（v2；2024-10-01）；robosuite — http://arxiv.org/abs/2009.12293v3（v3 2025-01-18；v1 2020-09-25，v2 2022-11-15）
- 官方项目页：ManiSkill3 — maniskill.ai（文档 maniskill.readthedocs.io，Apache-2.0 开源）；robosuite — robosuite.ai（姐妹项目 robomimic，见 https://arise-initiative.github.io/robomimic-web/）
- Date read: 2026-09-19
- Tags: [physical-ai, benchmark, maniskill, robosuite, gpu-simulation, sim2real, reproducibility, evaluation, sapien, mujoco, physx]
- Thread: physical-ai
- Folder: day-28-maniskill-robosuite-eval
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-28-maniskill-robosuite-eval

## 一句话总结
Day28 是 Physical AGI 块里的**评测基准日**：robosuite（2020, MuJoCo CPU 时代的模块化标准：robot × arena × object × controller 正交组合、标准化任务 + RL 基线结果）定义了"可复现"的合同，ManiSkill3（2024, GPU 时代的开源答案：SAPIEN 并行渲染 + PhysX GPU 仿真 + 异构仿真）把这个合同搬到 GPU 上——30,000+ FPS sim+render、2–3× 更少显存、12 类任务域 / 20+ embodiment / 数百万 demo 帧 / RL 与 LfD 全套基线；它直接回答了 Day27 的第三问（评测协议缺失），并把 Day30 数据飞轮的"gate"落到可执行的统计协议上。

## 和之前工作的关系

- **接了哪条线**：Day15–18 的"数据资产"线（Open X-Embodiment / DROID / BridgeData V2 / RoboCasa）的**对偶**——数据回答"用什么训"，benchmark 回答"怎么比"。没有固定的评测合同，Days 15–18 的 scaling 数字（OXE 的 +50%、DROID 的 +20%、RoboCasa 的 28.8%→47.6%）都不可跨论文比较。Benchmark 是数据资产的镜像面。
- **补了哪个短板**：Day27（Cosmos）留下的第三问——"从视频 realism 到 policy 提升之间，缺一个什么样的评测协议？"ManiSkill3 就是那个协议的载体：统一的任务、资产、传感器、成功判据，加上快到能跑出统计显著性的吞吐。论文 Table I 直接把 RoboCasa、RLBench、OmniGibson、Habitat、AI2-THOR 列成"CPU backend、无并行渲染、无 visual RL 基线"的对照组——这正是 Day18 RoboCasa（MimicGen 放大 demo）的反面：RoboCasa 靠**离线 demo 放大**，ManiSkill3 靠**在线并行吞吐**。
- **替代 / 分叉 / 改进**：
  - vs Day03（Isaac Lab）：ManiSkill3 与 Isaac Lab 同用 PhysX GPU 仿真，但 Isaac Lab 依赖**闭源** Isaac Sim 做 GPU 并行仿真+渲染，ManiSkill3 依赖**开源** SAPIEN；128 个并行 env、640×480 渲染时 ManiSkill3 占 3.5GB 显存 vs Isaac Lab 14.1GB。更重要的是 Isaac Lab 开箱没有 ManiSkill3 这类任务域（论文明说人形/双臂操作等类型"Isaac Lab currently does not have these types of tasks out of the box"），且只有 ManiSkill3 支持**异构仿真**（heterogeneous simulation）。
  - vs Day02（MuJoCo）/ robosuite：robosuite 是 MuJoCo CPU 时代的模块化标准；Brax/MJX（MuJoCo GPU 后端）至今没有并行渲染——这是 ManiSkill3 能用"visual RL 几分钟 vs 几小时"碾压 CPU 栈的直接原因。
  - vs Day20（RLPD）：ManiSkill3 的 demo 生成管线**直接调用 RLPD**（Ball et al. 2023）+ RFCL 做"在线从演示中学习"——Day20 的真机在线 RL 配方，被 ManiSkill3 拿去当**仿真数据放大器**。Infra 互文：数据飞轮的每一个齿轮都复用之前 Day 的零件。
  - vs Day18（RoboCasa / MimicGen）：ManiSkill3 明确对比——MimicGen 假设末端执行器动作空间、几乎无几何变化、工程化阶段指示器；RLPD/RFCL 更灵活。两条 demo 放大路线的假设强度排序：MimicGen（SE(3) 搬运，最强假设）> RLPD/RFCL（在线 IL，中等）> 纯 teleop（无假设，最贵）。
- **对之前 Day X 的直接对比**：Table I（论文 Sec II）的评测维度矩阵本身就是一张"benchmark 设计 checklist"：Parallelized Simulation / Rendering / Heterogeneous Simulation / Large Scale Demonstrations / Realistic Object Physics / Photorealistic Rendering / Room-Scale Scenes / Visual RL Baselines / Vision-based sim2real setups / Trajectory replay / Task Categories / Interactive GUI。以后读任何 benchmark paper，先拿这 12 行打分。

## 为什么今天读它

Roadmap Day28 的官方主题就是"统一任务、资产、传感器和成功判据，建立算法与系统的可复现实验矩阵"。它是 Day27 Q3（评测协议）的正解、Day24（SimOpt release gate）的工程载体、Day30（数据飞轮）的 gate 实现。读完它，"评测"从一句口号变成一个**可计算的统计对象**（见数学视角）。

## 今天的 3 问
1. **评测的统计显著性怎么凑？** 仿真里 N=1000 次 trial 很便宜（GPU 并行几分钟），真机 N=20 次很贵（遥操作/重置成本）。Sim 给出高精度 $\hat{J}_{sim}$ ，real 只有稀疏验证 $\hat{J}_{real}$ ——两者之间用什么判据连接？Day24 的 sim2sim / HIL 探针 / 分桶指标 / release gate 四件套里，哪一个真正回答了"sim 的数字能不能外推到 real"？
2. **异构仿真对 on-policy RL 是帮助还是伤害？** 每 env 不同场景/物体/关节数 ⇒ 一个 batch 内 $\xi$ 直接覆盖整个 $p_\phi$ ——对 PPO 这种 on-policy 算法，这是"分布匹配"的帮助（rollout 分布 = 训练分布），还是"方差爆炸"的伤害（不同 env 的 advantage 尺度不一致）？和 Day19 的 GPU 向量化 rollout（legged_gym 8192 env）在数学上是什么关系？
3. **Benchmark 的覆盖 vs 可比 trade-off**：robosuite 的"模块化"（robot × arena × object × controller 正交组合，组合爆炸）vs ManiSkill3 的"12 类任务模板"（固定协议、模板可扩展）。评测设计上，覆盖广度（能测的组合越多越好）和可比性（所有人测的是同一份试卷）如何取舍？为什么 ManiSkill3 选择"多做模板、少做穷举"（论文明说核心不是每个类别建很多环境，而是建很多用户可扩展的模板）？

## 核心

### 1. Motivation
- 机器人学不像 vision/language，没有好的数据集可训（论文原话："there are still no good datasets for robotic manipulation"）。两条路：真机遥操作采集（贵、慢、成功率低）或真机 RL（setup 贵、reward/reset 难）。
- Baseline 为什么不行：Isaac 等 GPU 并行仿真把 locomotion 的 RL 训到几分钟量级，但**操作任务**仍被窄任务+强状态估计卡住——已有 GPU 仿真器不支持异构仿真、没有快速并行渲染 ⇒ visual RL（从 RGB/pointcloud 学）慢到不实用。CPU 栈（RoboCasa/Habitat/AI2-THOR/RLBench/OmniGibson）只能玩 IL/motion planning，碰不了在线 RL。
- 和 Physical AGI 的关系：Benchmark 是 AGI 路线图的**度量衡**——没有它，"通用操作"的 claim 无法证伪。ManiSkill3 的赌注：把"可复现实验矩阵"做到 GPU 并行量级，让评测本身成为可 scaling 的 infra。

### 2. System / Method
- **robosuite（模块化设计，CPU/MuJoCo 时代）**：
  - 模块 = robot models × arenas × parameterized 3D objects 的程序化组合 API：新环境/新任务 = 选机器人 + 选竞技场 + 参数化物体，正交拼装。
  - Controller 支持：joint-space velocity、IK control、operational space control、3D motion devices（teleop）。
  - 多模态传感器：low-level 物理状态、RGB 相机、depth、proprioception。
  - Human demonstration 工具链：采集 → 回放 → 利用 demo 数据学习（姐妹项目 robomimic 把这条做成离线 RL benchmark）。
  - 产出：标准化任务套件（多样性/复杂度分级）+ RL benchmarking 结果——"reproducible research"的合同文本。
  - 渲染：集成 photorealistic 渲染（含 NVIDIA Isaac Sim 渲染后端）。
- **ManiSkill3（GPU 时代，SAPIEN + PhysX）**：
  - 五大贡献：① SOTA GPU 并行仿真+渲染（PPO visual RL 比别的仿真器快一个量级；sim+render FPS 可达 30,000+；显存通常低 2–3× ⇒ 单卡可跑 visual RL + 更大网络）；② 12 类任务域 / 20+ embodiment 开箱（tabletop、mobile manipulation、room-scale、quadruped/humanoid locomotion、humanoid/bimanual、multi-agent、drawing/cleaning、dextrous、vision-tactile、classic control、digital twins、soft body；机器人含 quadruped、floating gripper、humanoid、灵巧手；sim2real 与 real2sim  setups）；③ **异构仿真**（heterogeneous simulation）：每个并行 env 可仿完全不同的物体/关节/房间场景——data-oriented 设计 + 易用的 GPU 内存管理 API（不同 DoF 的 articulation 共存）；④ 统一简洁 API（面向对象、无复杂 tensor 索引；domain randomization、轨迹回放、controller 动作转换等工具）；⑤ 从少量 demo 出发的可扩展数据生成管线（见下）。
  - 渲染细节：128×128 单相机、仿真频率 120Hz、控制 60Hz ⇒ 每 2 个 sim step 渲染一次；RGB + depth + segmentation **同时**并行渲染。
  - ReplicaCAD/AI2-THOR 场景改造：CoACD 凸分解生成可抓取的非凸碰撞网格；人工标注 kinematic 类别（桌子/电视/钟表/画 = 不可动，苹果/杯子/球棒 = 可动）以优化仿真速度。
  - 人形任务实现细节：腿部固定、只训上身操作（降低难度、训得更快），可换全关节版本（更难）。
  - VR teleoperation 系统：demo 采集的人机接口。
- **关系**：robosuite 定义"模块化可复现"的**语义**（什么叫一个标准任务），ManiSkill3 定义它的**GPU 实现**（怎么快到能跑统计）。两者不是替代，是代际升级。

### 3. Training / Data Details
- **ManiSkill3 demo 生成三档**（按任务难度分流）：
  - 简单任务（reward 易写）：motion planning 脚本 / RL 直接生成 demo。
  - 复杂任务（motion planning 难写、reward 难定）：**在线 IL 算法 RLPD + RFCL**——从少量 teleop/硬编码 demo 出发，学一个泛化的神经网络策略，再 rollout 出大规模数据集。比 MimicGen 假设更弱（不要求末端执行器动作空间/低几何变化/阶段指示器）。
  - 总量：数百万 demo 帧（motion planning + RL + teleop 三源）。
- **基线**：覆盖主流 RL 与 learning-from-demonstrations 算法的完整 baseline 套件（含 visual RL baselines——Table I 里只有 ManiSkill3、Isaac Lab、Habitat、AI2-THOR 有这一行）。
- **Sim2Real**：Koch 机械臂 pickcube 的 vision-based sim2real（论文 Fig 13：grasp/lift/return 三阶段 sim vs real 成功率对照）；Appendix VIII 给了数字孪生的 domain randomization 与 controller 实现细节——"即使没建模阴影和裸露线缆，sim2real policy 依然成功"，这是对 Day21/24 的实证注脚：**评测合同里固定的 randomization 集本身就是 transfer 的载体**。
- **Verifiable signal**：成功率（sim 大 N + real 小 N）、sim+render FPS、显存占用、demo 生成吞吐——全是可复现、可审计的数字。

### 4. Key Tricks（最值得抄的 3 个）
1. **开源并行渲染栈打闭源**：SAPIEN（开源）+ PhysX GPU vs Isaac Lab 依赖闭源 Isaac Sim；128 env、640×480 时 3.5GB vs 14.1GB 显存。教训：benchmark 的"快"必须和"开源可审计"绑定，否则别人的数字你复现不了——可复现性是 benchmark 的第一性，不是附加分。
2. **异构仿真 = 并行 env 里的分布内采样**：传统向量化 env 所有 env 共享同一场景（ $\xi$ 相同），ManiSkill3 允许每 env 不同场景/物体/DoF——一个 step 的 batch 就在经验上覆盖 $p_\phi$ 。对 Day21 DR 的意义：训练分布的 Monte Carlo 估计从"跨 batch 时间平均"变成"单 batch 空间平均"，方差结构完全不同（见数学视角）。
3. **用 RLPD/RFCL 当 demo 放大器**：demo 生成不走"几何搬运"（MimicGen），走"在线 IL 学策略再 rollout"——假设更弱、适用任务更广。这是 Day20 的 RLPD 从"真机在线 RL 配方"转职为"仿真数据 infra"的实例：**好 infra 零件会被下游反复复用**。

### 5. Results
- **速度**：sim+render 最高 30,000+ FPS；显存低 2–3×；"原来几小时的任务现在几分钟"（visual RL，论文原话 "Tasks that used to take hours to train can now take minutes"）。
- **规模**：12 类任务域、20+ embodiment、数百万 demo 帧；ReplicaCAD/AI2-THOR 房间级场景。
- **Sim2Real**：Koch pickcube 三阶段（grasp/lift/return）sim vs real 成功率对照（Fig 13）——vision-based sim2real 可复现 setups 是 Table I 里**只有 ManiSkill3 打勾**的一行。
- **robosuite**：标准化任务 + 发布的 RL benchmarking 结果（可复现研究的基线锚点）；模块化组合 API 被 robomimic 等下游沿用。

### 数学视角（统一框架：评测 = 固定测度下的策略性能估计）

**框架**：把一个 benchmark 形式化为六元组 $\mathcal{B}=(R,E,O,\rho_0,p_\phi,S)$ —— $R$ 为机器人+控制器对（形态与控制接口）， $E$ 为任务集， $O$ 为观测/传感器套件， $\rho_0$ 为初始状态分布， $p_\phi(\xi)$ 为实例化参数（几何/纹理/光照/动力学）的随机化分布， $S$ 为成功判据。策略性能是**固定测度**下的期望：

$$J(\pi)=\mathbb{E}_{\xi\sim p_\phi}\mathbb{E}_{\tau\sim p_\pi(\cdot\mid\xi)}[\mathbf{1}\{S(\tau)=1\}]$$

robosuite 的"模块化"贡献，用这套语言说就是：**把测度 $\mu=(R,E,O,\rho_0,p_\phi,S)$ 写下来、固定住、公开发布**。可复现 = 大家估计的是同一个 $J$ ；此前的论文各用各的 $\mu$ ，数字不可比——这正是 benchmark 作为"合同"的数学本质。

**估计与方差**： $N$ 次 trial 的成功率估计 $\hat{J}=\frac{1}{N}\sum_{i=1}^N s_i$ ， $s_i\in\{0,1\}$ ，是二项比例估计，

$$\mathrm{Var}(\hat{J})=\frac{J(1-J)}{N},\quad 95\%\text{置信半宽}\approx 1.96\sqrt{\frac{J(1-J)}{N}}$$

$J=0.5$ 时要分辨 5 个点的提升需要 $N\approx 1500$ ； $J=0.9$ 时分辨 5 个点需要 $N\approx 140$ 。这就是为什么**仿真吞吐直接决定评测精度**：真机 $N=20$ 时半宽 $\pm 22\%$ （ $J=0.5$ ），任何"真机提升 10%"的 claim 在统计上都是噪声——Day27 Q3 的"评测协议"缺口，数学上就是" $N$ 凑不够"。ManiSkill3 的 30,000+ FPS 不是炫技，是把 $N$ 推到大数定律生效区间的**必要条件**。

**异构仿真 = 空间平均替代时间平均**：标准向量化 env（Day19 legged_gym 式）所有 env 共享同一 $\xi$ ，对 $p_\phi$ 的 Monte Carlo 靠跨 batch 的时间平均；异构仿真允许 $\xi_i\neq\xi_j$ ，单个 batch 的经验分布 $\hat{p}=\frac{1}{B}\sum_i\delta_{\xi_i}$ 直接逼近 $p_\phi$ 。对 PPO 的影响是双重的：rollout 分布与训练目标分布对齐（偏差↓），但 batch 内 advantage 的跨 env 方差增大（方差↑）——这正是 Q2 的答案：帮助还是伤害取决于 $p_\phi$ 的支撑集宽度 vs batch size，ManiSkill3 用"更多 env 数"（显存省 2–3× 换来的）对冲方差。

**与 Day21 DR 的统一**：Day21 的 $J_{\text{DR}}(\theta)=\mathbb{E}_{\xi\sim p_\phi}[J(\theta;\xi)]$ 里 $p_\phi$ 是**训练**分布；Day28 的 $J(\pi)$ 里 $p_\phi$ 是**评测**分布——同一个数学对象，两种角色。Day24 SimOpt 的"有锚辨识"（用真机行为残差更新 $p_\phi$ ）则是在**校准**这个分布：benchmark 固定 $p_\phi$ 当合同，SimOpt 让合同向真机对齐。Day30 的数据飞轮 gate，本质上就是" $p_\phi$ 校准 → 大 $N$ 评测 → 达标放行"的闭环。

**数学模型没覆盖的**：① 成功判据 $S$ 是二值的，掩盖了"怎么成功的"（轨迹质量、力控安全性）——Day29 的 CBF/shield 正是补这一块；② $p_\phi$ 再宽也只是**参数化不确定性**，结构性缺失（没建模的接触模态、线缆、阴影）不在支撑集里——论文 Appendix VIII 的 Koch sim2real 诚实注脚（没建模阴影/线缆仍成功）恰好说明：transfer 的充分条件比"测度覆盖真机"更弱，这是理论空白；③ FPS/显存数字是特定任务+分辨率+GPU 下的点估计，换任务可能不成立（论文 Fig 2 caption 自己声明了这点）。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？ManiSkill3 的 transfer 证据是 Koch pickcube 的 sim2real（Fig 13）与 digital twin setups；robosuite 的 transfer 证据是下游（robomimic 等）沿用其任务/数据协议。模型 vs 框架：这里**框架贡献 >> 单个模型**——benchmark 的价值在于被复用，不在于单次 SOTA。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. **评测是 infra，不是附录**：以后做任何 post-training 实验，先写 $\mathcal{B}$ 六元组（任务/观测/随机化分布/成功判据/ $N$ /置信区间），再跑实验。没有固定测度的 ablation 数字是不可发表的。
  2. ** $N$ 的预算公式**：要 claim $\Delta$ 的提升，先算 $N\approx (1.96/\Delta)^2\cdot 4J(1-J)$ 量级的 trial 数，再决定用 sim 凑还是 real 凑——这是把"统计功效分析"前置到实验设计里，和 infra 的容量规划是同一件事。
- Infra 视角：可扩展性（30k FPS、异构仿真）/ 成本（显存 2–3×↓ ⇒ 单卡可跑）/ 评测自动化（基线套件 + demo 管线开箱）/ 可复现性（Apache-2.0 + 固定 $p_\phi$ + 轨迹回放）——ManiSkill3 是"评测即服务"的开源实现。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：异构仿真下 PPO 的 batch 内跨 env advantage 方差到底多大？论文没有给"同构 vs 异构"的 RL 收敛曲线对照——这是 Q2 的实证缺口，值得自己跑一个对照实验（固定总 env 数，一组同构、一组异构，看 sample efficiency 与最终 $J$ ）。
- 如果要复现 / 小规模试，第一个实验做什么：pip 装 ManiSkill3，跑通 PickCube 的 PPO visual baseline，记录 sim+render FPS 与 128 env 显存占用，复现论文 Fig 2/Fig 4 的数量级；再把 $N=1000$ 的 $\hat{J}$ 置信区间算出来，体会"吞吐→精度"的链条。

## 原文金句 (1-2句)
> "Simulation has enabled unprecedented compute-scalable approaches to robot learning."（摘要首句——整个 Day28 的合法性来源：仿真让机器人学习第一次有了"算力可扩展"的入场券。）
> "Tasks that used to take hours to train can now take minutes."（贡献①——visual RL 的 wall-clock 从小时到分钟，这是 benchmark 能跑出统计显著性的物质基础。）

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节
- 数学视角：评测 = 固定测度 $p_\phi$ 下的 $J(\pi)$ 估计；二项方差公式给出 $N$ 预算；异构仿真 = 空间平均替代时间平均；与 Day21/24/27/30 的统一

## 连接
- 上一篇: day-27-2025-cosmos-world-foundation（Cosmos 世界基础模型平台；Day27 Q3"缺评测协议"→ Day28 正解）
- 下一篇预告: day-29-safe-robot-learning（Safe Robot Learning — 约束 MDP / CBF / shield / runtime monitor： $S$ 只判成功不判安全，Day29 补"怎么成功"的约束）

## 问答补充
（本篇暂无 side chat 问答；跨篇通识问答见 README 问答记录。）
