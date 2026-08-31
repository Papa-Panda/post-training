# physical-ai — Physical AI / 人形机器人 / World Model

> 专门读 **Physical AI** 相关的 paper / 系统 / 开源项目的沉淀区。服务于从 ML for Infra → Post-training / Agentic RL Infra → Physical AI 转型，重点是 **humanoid / world model / sim2real / Isaac Lab / Habitat / 控制 / RL for robotics**。
> Scope：Physical AI 全链路，不谈纯 LLM data curation（那是 ai-data）。
> 命名对齐 `ai-data/day-01-xxx`，`physical-ai/day-01-xxx` ~ `day-30-xxx`，便于 Day N 直连。

## 结构

```
physical-ai/
├── README.md                # 本路线图（10已完成/30总规划）
├── PAPER_TEMPLATE.md
├── reading-log.csv          # 快速索引
└── day-01-xxx/              # 每篇一个文件夹
    ├── NOTES.md             # 必须含「和之前工作的关系」
    └── assets/
```

## 发展路线图 (10/30 - 已起步)

> 总计 **30篇** 即闭环，当前 Day01 ARI/MSL + Day02 MuJoCo + Day03 Isaac Lab + Day04 Genie + Day05 UniSim + Day06 DreamerV3 + Day07 H2O Whole-Body Control + Day08 Humanoid-Gym Locomotion + Day09 RT-2 / OpenVLA + Day10 Habitat 3.0 已完成。

### 图谱总览

```mermaid
graph TD
  A[Day01 ARI/MSL Robotics Studio] --> B[Day02 MuJoCo Contact]
  B --> C[Day03 Isaac Lab / Sim2Real]
  C --> D[Day04-06 World Model: Genie/UniSim/Dreamer]
  D --> E[Day07-10 Humanoid Control / Locomotion]
  E --> F[Day11-14 VLA: RT-2 / OpenVLA / Pi0]
  F --> G[Day15-18 Habitat / Data Scaling]
  G --> H[Day19-24 RL for Robotics / Sim2Real]
  H --> I[Day25-30 Physical AGI / Eval / Safety]
  style A fill:#ffd700
  style B fill:#ffd700
  style C fill:#ffd700
  style D fill:#ffd700
  style E fill:#ffd700
```

### 30天闭环计划

| Day | Folder | 标题 | Tier |
|-----|--------|------|------|
| 01 | day-01-2025-ari-msl-robotics-studio | Meta ARI / MSL / Robotics Studio ✅ 2026-08-23 | S |
| 02 | day-02-2024-mujoco | MuJoCo Contact Model ✅ 2026-08-23 | S |
| 03 | day-03-2025-isaac-lab | Isaac Lab / Isaac Sim — USD + PhysX + Sim2Real ✅ 2026-08-24 | S |
| 04 | day-04-2024-genie-world-model | Genie / Genie 2 / Genie 3 — latent action 可交互生成式世界 | S |
| 05 | day-05-2023-unisim | UniSim — action-conditioned video diffusion + learned simulator RL | S |
| 06 | day-06-2023-dreamerv3 | DreamerV3 — latent RSSM + imagined actor-critic ✅ 2026-08-27 | S |
| 07 | day-07-2024-h2o-whole-body-control | H2O — Human-to-Humanoid Real-Time Whole-Body Teleoperation ✅ 2026-08-28 | S |
| 08 | day-08-2024-humanoid-gym-locomotion | Humanoid-Gym — RL Locomotion + Sim2Sim + Zero-Shot Sim2Real ✅ 2026-08-29 | A |
| 09 | day-09-2024-rt2-openvla | RT-2 / OpenVLA — action tokenization + web knowledge transfer + open VLA scaling ✅ 2026-08-30 | S |
| 10 | day-10-2023-habitat-3 | Habitat 3.0 / Habitat-Lab — humanoid simulation + HITL + social collaboration ✅ 2026-08-31 | A |

### Day N 映射表 (10已完成)

| Day | Folder | 贡献 | Tier |
|-----|--------|------|------|
| 01 | day-01-2025-ari-msl-robotics-studio | Physical AGI 定义，MSL 生态，humanoid scaling 哲学，learning from human experience vs teleop | S |
| 02 | day-02-2024-mujoco | MuJoCo fast accurate contact, MJCF, MJX million steps/s, lightweight baseline for humanoid control | S |
| 03 | day-03-2025-isaac-lab | OpenUSD scene layer + PhysX Direct-GPU + RTX tiled rendering + manager-based MDP + domain randomization, scalable sim2real platform | S |
| 04 | day-04-2024-genie-world-model | Genie foundation world model：无标签视频 → latent action → 可交互生成式世界 | S |
| 05 | day-05-2023-unisim | 多源数据统一为 action-in-video-out；video diffusion simulator + learned reward 支持 VLM / RL 与 zero-shot real-robot transfer | S |
| 06 | day-06-2023-dreamerv3 | 离散 latent RSSM + imagined actor-critic；free bits / symlog / two-hot / percentile normalization 支撑固定超参跨 150+ tasks | S |
| 07 | day-07-2024-h2o-whole-body-control | sim-to-data 筛掉 embodiment-infeasible motions；deployable goal state + PPO + domain randomization 实现 RGB 驱动 H1 全身控制与 zero-shot sim2real | S |
| 08 | day-08-2024-humanoid-gym-locomotion | Isaac Gym 8192-env PPO + 15-frame history + asymmetric critic + gait prior；MuJoCo sim2sim gate 后在 XBot-S/L 展示 zero-shot sim2real locomotion | A |
| 09 | day-09-2024-rt2-openvla | RT-2 把 action 变成 token 并用 web+robot co-finetuning 保留语义；OpenVLA 用 970k OpenX demonstrations、DINOv2+SigLIP+Llama 2 7B 与 LoRA/量化把 VLA 变成开源可适配系统 | S |
| 10 | day-10-2023-habitat-3 | 高速 SMPL-X humanoid + HITL + Social Navigation/Rearrangement；以 partner population 和未见场景评测协作泛化，暴露 oracle skill → learned skill 的层间 distribution shift | A |

---
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai
---
