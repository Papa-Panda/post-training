# Day 02 — MuJoCo: Multi-Joint dynamics with Contact

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
