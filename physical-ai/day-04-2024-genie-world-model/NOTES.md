# Day 04 — Genie: Generative Interactive Environments (World Model)

> Day 04 of physical-ai track, following Day03 Isaac Lab. First World Model entry.

## 元信息
- Title: Genie: Generative Interactive Environments
- Authors / Org: DeepMind — Bruce, Dennis, Edwards, Parker-Holder, Shi et al. (Feb 2024 Genie 1, Dec 2024 Genie 2, Aug 2025 Genie 3)
- Link / Blog: https://deepmind.google/research/publications/60474/ / https://en.wikipedia.org/wiki/Genie_(world_model) / https://techcrunch.com/2025/08/05/deepmind-thinks-genie-3-world-model-presents-stepping-stone-towards-agi/
- Date read: 2026-08-25
- Tags: [physical-ai, world-model, genie, sim2real, interactive-env, latent-action]
- Thread: physical-ai
- Folder: day-04-2024-genie-world-model
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-04-2024-genie-world-model

## 一句话总结
Genie 是 DeepMind 的 foundation world model，从无标签 Internet videos 无监督训出，通过 spatiotemporal video tokenizer + autoregressive dynamics + latent action model，用单图/文本 prompt 生成可帧级交互的虚拟世界，11B 参数起点，Genie 2 扩展到 3D 10-20s 360p，Genie 3 到 720p 24fps 实时交互 1-2 分钟记忆。

## 和之前工作的关系

- **接了哪条线：** 接 Day01 ARI/MSL 的 physical AGI 定义（learning from human experience）和 Day02 MuJoCo / Day03 Isaac Lab 的仿真基座线。MuJoCo/Isaac 是 physics-grounded sim，Genie 是 generative world model，两条 sim 路线分叉。
- **补了哪个短板：** 物理仿真器需要手工建 MJCF/USD + 材质，Genie 直接从视频学 dynamics，无需 action labels，用 latent action 隐式控制，补“如何从海量无标签视频 scale 出可交互环境”的短板。
- **替代 / 分叉 / 改进：** vs Isaac Lab：Isaac 准确但贵、需建模；Genie 便宜可无限生成但物理一致性弱，gamey artifacts。两者可互补：Genie 生成 diverse edge cases，Isaac 做 physics-correct filter。
- **对你 Infra 迁移的直接对比：** 你 ai-data 的合成数据 flywheel (Self-Instruct → Evol-Instruct) 类比到 Genie 的无标签视频 → latent action → endless envs，infra 挑战都是如何评一致性和防止漂移。

## 为什么今天读它

World Model 是 Physical AI 的另一半，Meta MSL 要做 personal superintelligence in physical world，需要能在想象中 rollout 的 model，Genie 是最经典的 foundation world model 基线，理解它才能看懂后续 UniSim / DreamerV3 / Waymo World Model。

## 今天的 3 问
1. Genie 如何在无 ground-truth action 情况下学出 latent action model？ST tokenizer 如何把视频切成离散 tokens 供 autoregressive dynamics 学？
2. Genie 1 → 2 → 3 的记忆从 1s → 10-20s → 1min + 720p 24fps 的关键技术跃迁是什么？自回归误差累积怎么缓解？
3. 对比 Isaac Lab 的 physics-correct sim，Genie 的 generative sim 在训练 humanoid VLA / RL 时 pros/cons？能否用 Genie 生成数据 + Isaac 验证形成闭环？

## 核心

1. **Motivation**: 传统 world model 需要 domain-specific + action labels，难以 scale。Genie 目标是从无标签 Internet videos 无监督训出可交互环境生成器，promptable via text / synthetic image / photo / sketch，endless variety，为 generalist agent 提供无限训练场。

2. **System / Method**:
   - **架构三件套 (11B Genie 1)**：spatiotemporal video tokenizer (视频 → 离散 tokens，类似 VQ-VAE 时空版) + autoregressive dynamics model (Transformer 预测下一帧 tokens，条件是历史帧 + latent action) + simple scalable latent action model (从相邻帧差分无监督学出离散 action codes，无需人工标)。
   - **交互**：用户键盘输入映射到 latent action space，帧级控制，尽管训练时无 action labels。
   - **Genie 2**：diffusion-based 3D，Imagen 3 生成首帧，支持 first-person / isometric / third-person，10-20s 一致性，记忆被遮挡后重现优于 Oasis。
   - **Genie 3**：实时交互，720p 24fps，promptable world events (改天气/加物体/调相机)，视觉记忆 1 分钟，自回归生成但保持物理一致性，Waymo 用其变体训 robotaxi edge cases。
   - **Project Genie (2026)**：Google AI Ultra 订阅 Web UI，World Sketching / Exploration / Remixing。

3. **Training / Data Details**:
   - 数据：大规模无标签 Internet videos + platformer 游戏视频，约 200k+ hours (?)，无需 action。
   - Tokenizer 训：重建视频帧，时空压缩。
   - Dynamics 训：给定历史 tokens + latent action 预测未来 tokens，cross-entropy。
   - Latent Action 训：看两帧差，学出可解释的离散 action codebook (8 actions in Genie 1 demo)，类似 VQ。
   - 无需 reward，可用于 imitation：从 unseen video 推断 latent actions 训 agent。

4. **Key Tricks**:
   - **Trick 1 - Latent Action from Video Only**：不靠人工标，用帧间变化无监督聚类出 action，避免 teleop 成本，和 Day01 ARI 的 human experience scaling 哲学同构。
   - **Trick 2 - Spatiotemporal Tokenizer + AR Dynamics**：把视频当语言，tokenizer 压到离散，Transformer 像 LLM 一样 next-token prediction 学世界 dynamics，复用 LLM infra。
   - **Trick 3 - Memory via History Conditioning**：Genie 2/3 通过 conditioning 过去 1 分钟帧来保持一致性，解决自回归漂移，类似 LLM 的 long context，物理一致性涌现而非显式编程。

5. **Results**:
   - Genie 1：2D platformer，1 fps，endless，11B 可控。
   - Genie 2：3D，360p，10-20s，multi-view，physics 推断水/烟但 gamey。
   - Genie 3：720p 24fps，1-2min，real-time interactive，promptable events，Waymo World Model 变体训 edge cases。
   - 对比：优于 Oasis 的记忆，Waymo 用其生成 street envs  via Street View。

## 可迁移 / Transfer

- **对你 Infra → Physical AI 迁移的 1-2 个直接启发：**
  1. **Data flywheel**：你 ai-data 的 15T → 5级过滤类比到 Internet video → ST tokenizer → latent action filter，如何从海量视频里筛出可交互片段是关键，质量门设计可复用。
  2. **Eval**：Genie 的一致性评测类似你 eval-bench-efficiency 的 IRT，如何自动评生成世界是否物理一致、记忆是否保持，可借鉴 VBench / physics benchmark。

- **Infra 视角：**
  - 可扩展性：Genie 像 LLM infra，tokenizer + AR 训练可 scale 到 100B，但 inference 24fps 实时要求高，需 speculative / distillation。
  - 成本：无标签视频便宜，比 Isaac 建模便宜，但训练 11B+ Transformer 贵。
  - 评测自动化：需 physics consistency / controllability 自动评，否则 human eval 贵。

## 疑问 / 下一步

- **没看懂的**：latent action codebook 具体大小和可解释性，8 个 action 如何覆盖 platformer 的复杂控制？
- **第一个实验**：跑 Genie 1 开源复现 (Oasis 300M) 对比，看 latent action 控制感；再试 Project Genie Web UI 生成一个 street scene 测记忆。
- **下一步预告**：Day05 UniSim — 真实世界交互模拟，real-world video + action 条件生成，和 Genie 的 game world 互补。

## 原文金句

> "We introduce Genie, the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos. The model can be prompted to generate an endless variety of action-controllable virtual worlds" — DeepMind Abstract

> "Genie 3 is the first real-time interactive general-purpose world model... It can generate both photo-realistic and imaginary worlds, and everything in between." — Shlomi Fruchter, DeepMind Research Director

## 今晚产出

- [x] Day04 Genie NOTES 初版
- [ ] Genie 2/3 vs Oasis 记忆对比视频
- [ ] Day05 UniSim 预习

## 连接
- 上一篇: Day03 Isaac Lab — GPU 规模化仿真
- 下一篇预告: Day05 UniSim — Real-world Interaction Simulator
- 相关: ai-data Day21 Self-Instruct (无标签合成类比), Day06 Phi-1 (合成数据)

## 参考链接
- DeepMind Genie: https://deepmind.google/research/publications/60474/
- Wiki: https://en.wikipedia.org/wiki/Genie_(world_model)
- Genie 3 TC: https://techcrunch.com/2025/08/05/deepmind-thinks-genie-3-world-model-presents-stepping-stone-towards-agi/
- Engadget Genie 2: https://www.engadget.com/ai/google-deepminds-genie-2-can-generate-interactive-3d-worlds-200708207.html
