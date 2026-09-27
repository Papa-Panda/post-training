# Day 02 — MuJoCo: Multi-Joint dynamics with Contact

> 📖 阅读版：https://htmlpreview.github.io/?https://github.com/Papa-Panda/post-training/blob/master/physical-ai/day-02-2024-mujoco/index.html

> Day 02 of ai-physical track, following Day01 ARI/MSL. Focus on fast & accurate contact simulation for robotics.

## 元信息
- Title: MuJoCo — Multi-Joint dynamics with Contact
- Authors / Org: Roboti LLC (Emanuel Todorov) → Google DeepMind (acquired Oct 2021, open-sourced May 2022)
- Link / Docs: https://mujoco.org/ / https://github.com/google-deepmind/mujoco / https://deepmind.google/blog/opening-up-a-physics-simulator-for-robotics/
- Date read: 2026-08-23
- Tags: [physical-ai, mujoco, sim2real, contact-model, humanoid, control, rl-robotics]
- Thread: physical-ai
- Folder: day-02-2024-mujoco
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-physical/day-02-2024-mujoco

## 一句话总结
MuJoCo 是 DeepMind 主力物理引擎，C/C++ 核心 + MJCF XML 建模，以 rich-yet-efficient 的接触模型著称，追求 fast and accurate 的平衡，已从单机 CPU 扩展到 MuJoCo XLA (MJX) 在 TPU/GPU 上每秒百万步，是人形机器人控制与 RL 采样的轻量基座。

## 和之前工作的关系

- **接了哪条线：** 接 Day01 ARI/MSL 的 physical AGI 定义。ARI 要实现 humanoid whole-body control，需要一个能快速迭代 contact dynamics 的仿真器，MuJoCo 就是 DeepMind 内部首选，Meta 内部虽主推 Isaac Lab，但 MuJoCo 是学术 baseline 和快速验证层。
- **补了哪个短板：** Day01 只讲战略，没有讲 sim 层怎么搭。MuJoCo 补上“接触物理怎么算得又快又准”这一层，对应你之前 Isaac / Habitat / MuJoCo 三选型里的轻量选项。
- **替代 / 分叉 / 改进：** vs Isaac Sim (GPU, photoreal, USD, PhysX) — MuJoCo 更轻、更快、接触更准，但渲染弱；vs Habitat (高层导航) — MuJoCo 是底层力控。MuJoCo 3 + MJX 开始加 GPU 加速，试图追上 Isaac 的规模化能力。
- **对你 Infra 迁移的直接对比：** 你 7年做过 data center 预测，习惯算 throughput / latency，MuJoCo 的卖点就是 compute efficiency vs accuracy trade-off，和你 ai-infra 里 FlashAttention / ZeRO 的思维同构。

## 为什么今天读它

你要求 Day2 出 MuJoCo 的 repo。MuJoCo 是 Physical AI sim 层的必备基础，几乎所有 locomotion / humanoid 控制 paper 都会用它当评测或训练环境，Day01 的 ARI humanoid 控制离不开它。

## 今天的 3 问
1. MuJoCo 的 contact model 为什么被称为 rich-yet-efficient？soft contact / optimization-based contact 怎么实现的，和 Isaac PhysX 的硬接触有何区别？
2. MJCF vs URDF：为什么 MuJoCo 坚持自己的 XML 格式？humanoid 建模时 joint / tendon / actuator 怎么定义才高效？
3. MJX (MuJoCo XLA) 如何做到在 TPU/GPU 上百万步/秒？对你 RL for robotics 的大规模采样有什么直接加速？和 Isaac Lab 的 GPU 并行有何选型差异？

## 核心

1. **Motivation**: 机器人与物理世界交互的核心是接触（走路脚触地，写字手指握笔），接触发生在微观尺度，可软可硬可滑可粘，仿真最难。游戏/影视引擎为稳定牺牲准确，MuJoCo 反其道，追求准确且高效的接触仿真，服务于 control synthesis, state estimation, system identification, RL sampling。

2. **System / Method**:
   - **核心数据结构**：C/C++ 库，预分配 low-level data structures，MJCF (MuJoCo XML) 描述场景，人可读可编辑，也支持 URDF 导入。
   - **Contact**：optimization-based contact dynamics，支持 soft contacts and constraints，generalized coordinates，允许穿透 soft 约束解，稳定性好。
   - **Menagerie**：DeepMind 发布的高质量模型库，robot arms / dogs / mobile manipulators / humanoids，开箱即用。
   - **MJX**：MuJoCo 3 新增 XLA 后端，JAX 编写，`pip install mujoco-mjx`，可跑在 TPU/GPU，支持 domain randomization 大规模并行。
   - **Viewer**：native GUI + OpenGL，也有 offscreen rendering 用于 headless 训练。

3. **Training / Data Details**:
   - MuJoCo 本身不产生数据，是环境。配合 dm_control (DeepMind) 或 gymnasium[mujoco] 做 RL。
   - 典型 pipeline：MJCF 定义 humanoid → MuJoCo 计算 forward dynamics + contact → RL policy 输出 torque / position → reward (locomotion 速度 / 平衡 / energy) → 并行采样百万步。
   - 和你 coding data 的 exec-filter 类比：MuJoCo 的 contact solver 就是 filter，保证物理合理性，不产生穿透/抖动脏数据。

4. **Key Tricks**:
   - **Trick 1 - Soft contact optimization**：不像硬接触强行零穿透，允许可控穿透，用 optimization 解接触力，fast + stable，特别适合 humanoid 脚部多接触。
   - **Trick 2 - Generalized coordinates + sparse**：用关节坐标而非笛卡尔，自由度少，计算快，适合 articulated structures。
   - **Trick 3 - MJX + domain randomization**：JAX 写法天然支持 vmap，同一张卡跑上千 env 不同 friction / mass / delay，sim2real 必备，类似你 ai-data 的 diversity 策略。

5. **Results**:
   - DeepMind 内部 robotics team 主力，Anymal / Unitree / Shadow Hand 等 humanoid / quadruped 都用它。
   - 开源后 2022-2024 社区增长最快的物理引擎，GitHub > 10k stars，MuJoCo Menagerie 模型质量被 Isaac Lab 引用对照。
   - 性能：单机 CPU 10k+ steps/sec (humanoid)，MJX TPU 上百万 steps/sec，满足 RL 大规模采样。

## 可迁移 / Transfer

- **对你 Infra → Physical AI 迁移的 1-2 个直接启发：**
  1. **Throughput 算账**：你算过 7B 32GB KV cache，MuJoCo 算的是 1k env * 1k steps = 1M steps 的 wall-clock，直接对应你 eval-bench-efficiency 的 IRT 蒸馏思维 — 如何用最少 sim 覆盖最多 dynamics。
  2. **Data quality gate**：MuJoCo 的 contact solver 稳定性就是数据清洗，类似 FineWeb 5级过滤，物理不合理轨迹直接丢弃，不进 RL。

- **Infra 视角：**
  - 可扩展性：MJX 解决 CPU 瓶颈，TPU 规模化是 Isaac Lab GPU 的对偶。
  - 成本：MuJoCo 免费 Apache 2.0，无 Isaac 的 GPU 成本，适合快速迭代。
  - 评测：MuJoCo 是 control 的 unit test，Isaac 是 integration test，Habitat 是 e2e test。

## 疑问 / 下一步

- **没看懂的**：MuJoCo 的 soft contact 具体 optimization 形式，solver 迭代次数 vs accuracy trade-off。
- **第一个实验**：`pip install mujoco gymnasium[mujoco]` 跑 humanoid stand / walk，调 friction / joint damping，看 contact 变化；再试 MJX 在 colab 跑 1k env 并行。
- **下一步预告**：Day03 Isaac Lab — 对比 MuJoCo，看 GPU photoreal + USD + PhysX 怎么补齐 MuJoCo 的渲染和规模化短板。

## 原文金句

> "The rich-yet-efficient contact model of the MuJoCo physics simulator has made it a leading choice by robotics researchers" — DeepMind Blog 2021

> "MuJoCo stands for Multi-Joint dynamics with Contact. It is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning" — MuJoCo Docs

## 今晚产出

- [x] Day02 MuJoCo NOTES 初版
- [ ] 跑通 gymnasium humanoid demo
- [ ] Day03 Isaac Lab 预习

## 连接
- 上一篇: Day01 ARI/MSL Robotics Studio — Physical AGI 战略
- 下一篇预告: Day03 Isaac Lab / Isaac Sim — GPU 规模化 + Photoreal
- 相关: ai-infra Day01 Transformer 白板 (地基类比), ai-data Day19 Vendi Score (多样性)

## 参考链接
- DeepMind Blog: https://deepmind.google/blog/opening-up-a-physics-simulator-for-robotics/
- GitHub: https://github.com/google-deepmind/mujoco
- Docs: https://mujoco.org/
- Tutorial: https://github.com/tayalmanan28/MuJoCo-Tutorial

## 第二轮复习（2026-09-27）

> 本轮核验：MuJoCo 主线已到 3.10.0（2026-07，mujoco-mjx 同版内部 vendored MJWarp）；3.8.0 changelog（2026-04-24）确认 multiccd 默认开启、flex 软体多单元支持。MuJoCo Warp（DeepMind + NVIDIA，NVIDIA Warp 实现）仍处 Beta，已并入 MJX 发行、无独立 wheel，并被 NVIDIA Newton physics 收为 8 种可插拔 solver 之一（第三方 intel 文档）。初读 NOTES 的技术事实无硬伤，补三处版本/生态演进 + 一处官方表述精确化。

### 元信息修正

- **版本演进**：NOTES 写于 2026-08-23 的 "MuJoCo 3 + MJX" 已是旧快照。当前主线 3.10.0；3.8.0（2026-04-24）把 convex collision detection 的 multiccd（多接触点）改为默认开启、flex 软体支持多单元（multi-cell）。"MuJoCo 不擅长软体"的边界正在被官方推进，但仍在早期。
- **MJX 与 MJWarp 的关系变化**：初读把 MJX（JAX/TPU）和 Isaac GPU 并行对立。2026 年现状：MuJoCo Warp（GPU 版，DeepMind + NVIDIA 联合维护）仍处 Beta、"mostly feature complete"，已直接 vendored 进 mujoco-mjx 3.10.0（无独立 wheel）；同时 NVIDIA 的 Newton physics 框架把它收为 8 种可插拔 solver 之一，Isaac Sim 6.0 用 Newton 补"研究级物理"。Day03 的"竞争"叙事要更新为"吸收"：MuJoCo 的接触数学正在变成 Isaac 生态的一个 solver 选项。
- **接触模型的官方精确表述**：NOTES 的 "optimization-based / soft contact" 是对的，官方完整说法是 "soft, convex and analytically-invertible"——把摩擦接触从通常的 LCP/NCP（NP-hard）化为凸优化；默认 Newton solver 二次收敛，另有 CG 和广义 Projected Gauss-Seidel（可处理 elliptic 真摩擦锥）；统一求解器处理 torsional/rolling friction、关节与肌腱限位、干摩擦、等式约束。`condim` 1/3/4/6 开关无摩擦/滑动/扭转/滚动摩擦分量。
- **初读 3 问第 1 问的答案现在可以收口**：rich-yet-efficient = 凸优化公式（快）+ 软接触允许可控穿透（稳）+ 唯一可逆 analytically invertible（利于控制与数据分析）。和 Isaac PhysX 的区别不在"准 vs 快"一句口号，而在"凸优化统一求解 vs 迭代求解器游戏级近似"。

### 一句话总结

MuJoCo 把摩擦接触从 NP-hard 的互补问题变成凸优化：soft + convex + 可逆的接触动力学是它的第一性原理，C++ 核心给 CPU 10k steps/s 级速度，MJX（JAX）和 MuJoCo Warp 把同一套数学搬上 TPU/GPU 做百万步并行。**34 天后回看：Day02 是整条路线的"物理公理层"——Day21 的随机化、Day24 的系统辨识、Day27 的世界模型、Day30 的飞轮，全部是围绕"这个公理层是近似的"展开的补救工程。**

### 和之前工作的关系

- **vs Day03 Isaac Lab（直接对比：公理层 vs 规模层）**：Day02 回答"接触怎么算对"（generalized coordinates + 凸优化接触），Day03 回答"怎么算得多、看得真"（USD + PhysX + photoreal）。2026 年的新事实是合流而非对立：Newton physics 把 MuJoCo Warp 收为可插拔 solver，Isaac Sim 6.0 承认 PhysX 单独不够研究级。初读"选型差异"的答案变了：不再是二选一，是"MuJoCo 数学 + Isaac 规模"的拼装。
- **vs Day09/10（总览脚手架）**：RT-2 / OpenVLA 的 action 最终要落在扭矩/位置指令上，MuJoCo（经 Day08 Humanoid Gym）是这些 VLA 在 sim 里练手的第一个环境。VLA 是 policy 层，MuJoCo 是物理层——policy 的泛化（Day34 π0.7）不能补偿物理层的错。
- **vs Day11–18（专题扩展：数据）**：DROID / Bridge / Open X-Embodiment 的真实轨迹是"观测"，MuJoCo 是这些数据背后的"物理先验"。数据飞轮里 sim 与 real 对齐的锚点，正是 MuJoCo 的接触参数（friction、solref）——而这正是 Day24 要辨识、Day21 要随机化的东西。
- **vs Day19–24（RL / sim2real：被攻击、被拟合、被随机化的对象）**：Day19 PPO 的 env 大多是 MuJoCo/MJX；Day21 随机化的就是 MuJoCo 的 contact 参数；Day22 RMA 的 adaptation module 隐式编码的也是这些参数；Day24 系统辨识直接拟合 MuJoCo 模型参数。Day19–24 的共同前提是"Day02 是近似的"——随机化/辨识/残差（Day23）是三种不同的"承认近似"的方式。
- **vs Day25–30（physical AGI / eval / safety）**：Day27 Cosmos 是"学习的世界模型"，Day02 是"解析的世界模型"，Day05 UniSim 已走混合路线；Day28 评测的物理真实性轴（penetration / jitter）本质在量"离 MuJoCo 有多远"；Day29 的 CBF 安全证书需要动力学模型，而 MuJoCo 的 analytically invertible（给定轨迹可反推接触力/参数）正是证书与系统辨识能成立的数学前提——游戏引擎没有这个性质，所以做不了严肃机器人学。
- **vs Day31–34（最新进展）**：Day31 Atlas（生成式世界模型）vs Day02（解析物理）是两条世界模型路线的分野，见思考题 (b)；Day33 Figure Helix 2.5 进家庭 56%——接触丰富的家务恰恰是 MuJoCo 最弱的一环（软体、触觉、形变），Day01 复习补的 e-Flesh（传感-智能绑定）是解析物理覆盖不到的地带；Day32 GPT-6 Astra 的 computer-use 是横向对照，提醒"物理公理层"之外还有"数字世界公理层"（OS/API），两者的 sim 哲学完全不同。

### 核心

1. **Motivation 深挖**：接触是机器人与物理世界交互的唯一通道（脚-地、手-物），发生在毫米/毫秒尺度。解析接触的本质困难是互补性：接触或分离、粘滞或滑动——写成 LCP/NCP 是 NP-hard。MuJoCo 的第一性原理选择：不求解精确互补，求解一个"软化后的凸问题"，用可控穿透换可解性。后面 32 天的所有 sim2real 工作，都是在为这个 trade-off 还债。
2. **机制 = 三层**：
   - **数学层**：接触脉冲 $z$ 是以下凸优化的解，摩擦锥 $\mathcal{K}$ 为约束（pyramidal 线性近似求快，或 elliptic 真锥配 PGS 求准）：

     $$z^\* = \arg\min_{z \in \mathcal{K}} \left( \tfrac{1}{2} z^\top A z + b^\top z \right)$$

     其中 $\mathcal{K}$ 是摩擦锥， $A$ 、 $b$ 由当前构型 $q$ 的质量矩阵与接触雅可比决定。soft 约束允许穿透 $\delta$ ， $\delta$ 的"软度"由 solref/solimp（刚度/阻尼的离散化参数）控制。同一套 EFC 求解器统一处理接触、关节限位、肌腱、干摩擦、等式约束——"统一"是 rich 的来源。
   - **表示层**：generalized coordinates（关节坐标，自由度最少，适合 articulated 结构）+ MJCF（tendon/muscle 一等公民，人可读）+ `condim` 维度开关摩擦分量。表示即先验：选关节坐标就是选了"机器人是铰接的"这个归纳偏置。
   - **规模层**：MJX 把前向动力学写成纯函数式 JAX，`vmap` 天然并行 + autodiff 穿过接触求解器——可微性是 Day24 梯度法系统辨识的数学前提；MJWarp 用 NVIDIA Warp 重写，绕过 MJX 的 sharp bits（可微/变结构限制），2026 年已并入 MJX 发行。
3. **前向动力学方程**：policy 输出的关节力矩 $\tau$ 经由下式变成运动——这是 VLA 的 action 与物理世界之间的翻译官：

   $$M(q)\,\ddot{q} + C(q, \dot{q}) = \tau + J_c^\top \lambda$$

   其中 $M(q)$ 是广义质量矩阵， $C$ 含科氏/重力项， $J_c^\top \lambda$ 是接触雅可比转置乘接触脉冲。
4. **analytically invertible 的意义**：接触映射唯一可逆，给定轨迹可反推接触力与参数——这是 control synthesis、state estimation、system identification 三件事能成立的共同根。Day24 之所以可能，前提是 Day02 的这个性质。

### 边界

1. **"准"的是求解器，不是参数**：friction、solref、solimp、margin 没有第一性原理值，全是手调。MuJoCo 保证 sim 内部自洽，不保证 sim = real。Day21（随机化）和 Day24（辨识）存在，恰恰因为引擎把不确定性从求解器推到了参数——这是 Day02 留给全路线最大的诚实声明。
2. **刚体假设**：软体（flex 多单元 3.8 才支持，仍早期）、流体、颗粒、粘附/吸盘、线缆都不在 Day02 的舒适区。Day33 Figure 进家庭的家务（叠衣、擦桌）是软接触密集区——解析物理的盲区正是 Helix 类端到端最想吃掉的地盘。
3. **GPU 路径的确定性打折**：第三方实测（omnisim determinism 文档）显示 CPU `mj_step` 逐位确定（5/5 场景），而 mujoco_warp GPU 路径 24 对冷启动 0 对逐位一致、1000 步后偏差可达米级（`wp.atomic_add` 接触槽竞争）。含义：大规模并行训练的"同一 seed 可复现"在 GPU 路径上不成立，论文的 seed 声明要配 solver/路径披露。见思考题 (c)。
4. **凸近似永远有代价**：pyramidal 锥是线性近似（快但失真），elliptic 真锥要 PGS 迭代（准但慢）；solver 迭代次数 vs 精度是永恒 trade-off。MuJoCo 从未"证明"任何 sim2real 结论——它只证明 sim 内部自洽。
5. **可微不等于可学**：MJX 的 autodiff 能穿过接触求解器，但接触事件的非光滑性让梯度在接触切换处病态——梯度法系统辨识（Day24）在接触丰富任务上仍不稳定，这是数学结构决定的，不是实现 bug。Day02 的可微性是必要非充分条件。

### 迁移到 post-training / Agentic RL Infra

- **可执行的映射**：把 MuJoCo 的"求解器即数据清洗器"翻译成你的 RL infra——在 VLA / world-model 训练数据 pipeline 里加一个 **physics plausibility gate**：用 headless MuJoCo（CPU `mj_step`，逐位确定、可复现）重放 DROID / Bridge 类轨迹，自动剔除深穿透（penetration > margin）、接触抖动（contact chatter）、能量异常的片段，再进训练。这和你 coding data 的 exec-filter 同构：exec 跑不通的代码不进 SFT，物理跑不通的轨迹不进 policy 训练。进一步：Day28 的评测轴（physical realism）直接变成数据 pipeline 的一个 gate；Day30 飞轮的 resimulate 分支就有了第一块可落地的砖——先有 gate，才有"什么值得 resimulate"的判据；gate 拦下的失败聚类方式，就是 Day30 triage 的输入。

### 思考题（综合 Day02 / Day21 / Day24 / Day27 / Day33）

- **(a) 参数的不可辨识性**：Day21 随机化 friction，Day24 辨识 friction——但从纯运动学轨迹看，(friction, solref, mass) 三元组是部分不可辨识的：不同组合可以产生相同轨迹。设计判据实验：在 MJX 里固定"真值"参数，分别用 (i) 纯运动学轨迹、(ii) 运动学 + 接触力（e-Flesh 类触觉，见 Day01 复习）做系统辨识，比较参数恢复误差。你预期 (i) 恢复的是等价类而非真值——这对"sim2real 已解决"的叙事意味着什么？Day22 RMA 的 adaptation module 是在辨识参数，还是在学习等价类的隐式编码？
- **(b) 两条世界模型路线的分工**：Day02 解析物理 vs Day27 Cosmos 生成式世界模型，Day05 UniSim 已是混合（action-conditioned video diffusion + learned simulator RL）。给出分工判据：什么任务必须用解析物理（提示：Day29 的 CBF 安全证书需要什么数学性质？Day02 的哪个词——analytically invertible——是 Cosmos 给不了的？），什么任务生成式更优（提示：Day33 家庭场景的软接触与视觉多样性）？Day31 Atlas 会改变这个分工吗，还是只是把"学习侧"做得更大？
- **(c) 复现性危机**：第三方实测 mujoco_warp GPU 路径 1000 步后偏差米级、24 对冷启动 0 对逐位一致。若一篇 humanoid RL 论文用 MJWarp 训练、只报 "seed=0"，它的结果在多大程度上可复现？设计一个最小披露标准（solver 类型、摩擦锥类型、迭代次数、CPU/GPU 路径、接触参数表），并论证：为什么 Day30 飞轮的 gated deploy 必须把"sim 配置哈希"作为 release gate 的一部分？

参考链接（本轮复习）：
- MuJoCo changelog（3.8.0，2026-04-24）：https://github.com/synreal/mujoco/blob/HEAD/doc/changelog.rst
- MuJoCo overview（soft, convex, analytically-invertible 表述）：https://github.com/haoxiangyou/sdpg/blob/HEAD/externals/mujoco/doc/overview.rst
- MJWarp README（Beta 状态，DeepMind + NVIDIA 维护；fork 镜像，内容与官方一致）：https://github.com/kxqovwcmq-lgtm/mujoco_warp
- Newton / Isaac Sim 6.0 solver 对比（第三方 intel 文档）：https://github.com/redhat-et/physical-ai-platform-intel/blob/HEAD/deliverables/intel/project-comparisons/simulation-engines.md
- GPU 路径确定性实测（第三方测试文档）：https://github.com/omnilink-tech/omnisim/blob/HEAD/docs/developer/simulator-comparison.md

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/physical-ai/day-02-2024-mujoco/NOTES.md

<!-- viz:vs: MuJoCo | 轻、快、接触准; 渲染弱; MJX 加 GPU || Isaac Sim | GPU photoreal USD PhysX; 重、贵、需建模 -->
<!-- viz:flow: MJCF 定义 humanoid → forward dynamics + contact → RL policy 输出 torque → reward → 并行采样百万步 -->
