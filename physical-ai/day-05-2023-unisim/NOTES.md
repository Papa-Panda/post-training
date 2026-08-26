# Day 05 — UniSim：Learning Interactive Real-World Simulators

> Day 05 of physical-ai track, following Day04 Genie. Focus on action-conditioned video diffusion as a learned simulator for real-world interaction and policy training.

## 元信息
- Title: Learning Interactive Real-World Simulators
- Authors / Org: Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, Pieter Abbeel / UC Berkeley, Google DeepMind, MIT, University of Alberta
- Link / arXiv / Project: https://arxiv.org/abs/2310.06114 / https://universal-simulator.github.io
- First submitted: 2023-10-09
- Date read: 2026-08-26
- Tags: [physical-ai, world-model, unisim, video-diffusion, sim2real, vla, model-based-rl]
- Thread: physical-ai
- Folder: day-05-2023-unisim
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-05-2023-unisim

## 一句话总结
UniSim 把互联网图像/视频、人类活动、全景扫描、仿真和真实机器人数据统一成 `action in → video out`：用 5.6B action-conditioned video diffusion 预测下一段观察，再把生成式世界包成 RL environment；在 Language Table 上，纯模拟 rollout 的 RL 把成功率从 BC 的 0.58 提到 0.81，并展示了 zero-shot real-robot transfer。

## 和之前工作的关系

- **接了哪条线：** 接 Day04 Genie 的 learned world model 路线，但从“无标签视频中发现 latent action”转向“显式把语言、机器人控制、相机运动统一成 action condition”，更直接服务 policy training。
- **补了哪个短板：** Genie 擅长开放式、可交互世界生成，但 action grounding 偏隐式；UniSim 把 high-level instruction 和 low-level control 都纳入统一接口，并展示 VLM hindsight data、model-based RL、real-robot zero-shot transfer 三种下游用途。
- **替代 / 分叉 / 改进：** 相对 Day02 MuJoCo / Day03 Isaac Lab，UniSim 不显式求解接触和刚体动力学，而从像素经验学习视觉后果；它能吸收真实世界长尾外观，却失去精确 state、force/contact observability 和 hard physical constraints。
- **对之前 Day X 的直接对比：** Day03 的 transition 是 PhysX solver，Day04 的 action 是 learned latent code，Day05 的 transition 是 video diffusion，action 是 T5 language embedding + discretized controls。三者分别代表显式物理、隐式交互和显式条件生成三种 simulator abstraction。

## 为什么今天读它

Day05 路线图指定 UniSim。它把 world model 从“好看的交互视频”推进到“可被 agent 反复 step、可接 reward、可做 policy optimization 的环境”，与 Agentic RL Infra 的 rollout server / actor-learner / learned verifier 结构高度同构。

## 今天的 3 问
1. **异构数据怎样进入同一个 action space？** 文本指令、连续机器人控制、相机位姿和静态图像分别如何对齐到可条件化的视频 transition？
2. **生成模型怎样成为 RL environment？** `p(o_t | h_{t-1}, a_{t-1})` 如何 autoregressive rollout，reward 从哪里来，simulator bias 又会怎样被 policy exploit？
3. **“视觉逼真”是否足够支撑 sim2real？** UniSim 的 zero-shot 展示证明了什么，又没有证明什么；如何加入 contact、force、uncertainty 和真实闭环校准？

## 核心

1. **Motivation：世界数据各自只覆盖一条轴**
   - 互联网图像覆盖丰富对象/场景但缺动作；human activity video 有高层动作但少机械控制；robotics data 有稠密 control 却规模小；panorama 有空间变化但无真实交互。
   - 单一数据集无法同时学到 object diversity、action granularity、embodiment 和 navigation。UniSim 的核心命题不是“一个数据集包打天下”，而是把互补数据编排进统一 `action-in-video-out` 接口。
   - 真正目标是可交互 observation prediction，而非一次性 text-to-video：给定近期观察与动作，生成动作的视觉后果，并跨 video segment 自回归展开。

2. **System / Method：统一 action-conditioned observation prediction**
   - 定义 transition：`p(o_t | h_{t-1}, a_{t-1})`。`o_t` 是下一段可变长度视频，`h_{t-1}` 是有限近期帧，`a_{t-1}` 可为语言、相机运动或低层 motor control。
   - 文本先经 T5 得到连续 embedding；连续 control 先 normalize，再离散到 4096 bins，并与 language embedding 拼接。静态 text-image 被视为 single-frame video；panorama 由 camera pose 构造 turn/move action。
   - 生成器是 5.6B 3D video U-Net：base model 在 `[16,24,40]` 时空分辨率预测，两个 spatial super-resolution stages 依次放大到 `[48,80]` 与 `[192,320]`；时空 attention / convolution 交错。
   - 历史条件取上一 segment 的 4 帧，沿 channel 维与未来帧 noise 拼接；action 通过 classifier-free guidance 注入。下一 segment 再条件于刚生成的帧，形成 autoregressive rollout。
   - 训练规模：512 TPU-v3、20 天、1M steps、batch 256、256 diffusion sampling steps。模型规模从 500M → 1.6B → 5.6B 时 FVD 277.85 → 224.61 → 211.30，但作者指出收益开始平台化。

3. **Training / Data Details：多源 mixture + 生成式 rollout + learned reward**
   - **数据组成：** Habitat HM3D 710、Language Table sim 160k、Bridge Data 2k、RT-1 70k、Language Table real 440k、Ego4D 3.5M、Something-Something V2 160k、EPIC-KITCHENS 25k、Matterport R2R scans 3.5M、LAION-400M 400M、ALIGN 400M、互联网视频 13M，外加未公开的 robot/human video collections。
   - **Mixture：** 各域权重仅用 0.05 或 0.1，未精调。低数据域可在 action 前加 dataset identifier 提升 in-domain generation，但会伤害 out-of-domain generalization。
   - **长程 VLM data：** 在 simulator 中每条轨迹 rollout 3–5 次 scripted instruction，合成 10k long-horizon trajectories；以最终帧为 goal 做 hindsight relabeling，再训练 image-goal-conditioned PaLM-E policy。
   - **RL loop：** PaLI 3B 先做 BC、steps-to-success prediction、instruction prediction；把 video generation 暴露为 RPC，并用 DM Env API 包成 `step()`。64 actor processes 在 UniSim 中 rollout，冻结的 steps-to-success model 产生 progress reward，REINFORCE 更新 policy。
   - **Reward：** `r_t = -[d(o_{t+1},g)-d(o_t,g)]·C`，其中 `d` 预测距离成功还剩多少步，`C=5e-2`。这是 learned visual progress verifier，不是 simulator 自带 ground-truth state reward。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — Action-space normalization, not raw dataset merging：** 先把不同数据集统一成 temporally extended action + video segment，再做 mixture；数据 schema 比“多收数据”更关键。
   - **Trick 2 — Finite recent history as practical state：** 4 个 recent frames 把 Ego4D FVD 从 315.69 降到 211.30，优于单帧和 distant history；不是记忆越长越好，而是要找足够 Markov 的最小窗口。
   - **Trick 3 — Hindsight relabeling turns stochastic generation into supervision：** 先 rollout，再把真正到达的末帧当 goal，避免要求生成器严格命中预设目标；相当于把 model error 吸收进 label construction。
   - **Trick 4 — Simulator 与 reward 解耦：** transition model 只预测观察，reward 单独学习；同一 simulator 可以复用到多任务，但也必须防 reward model 与 simulator 共同偏差被 policy exploit。
   - **Trick 5 — RPC environment boundary：** 把昂贵 video generation 包在远端 environment service 后面，actor-learner 不依赖模型内部实现；这正是可横向扩容、限流、版本化和 shadow evaluation 的 infra seam。

5. **Results：有效，但证据边界要说清**
   - **视频预测：** Ego4D 上 4 recent-frame conditioning 达到 FID 34.63、FVD 211.30、IS 3.52、CLIP 22.63；单帧条件为 FID 59.47、FVD 315.69。
   - **长程 policy：** 10k UniSim hindsight trajectories 训练的 VLM，在 5 次模拟评估中 RDG(moved/all) 为 0.34/0.34；短程 BC 为 0.11/0.07，约 3–4× 提升。
   - **RL：** Language Table 的 48 个模拟任务中，Simulator-RL overall success 0.81 vs VLA-BC 0.58；pointing tasks 为 0.71 vs 0.12。
   - **Real robot：** 论文给出 simulator-only training 后的 zero-shot Language Table 成功案例，但主要是定性展示，没有报告与表 3 同等级的大样本真实机器人成功率；不能把 0.81 当作真实机器人 success rate。
   - **跨任务生成数据：** PaLI-X 仅用 UniSim 生成视频微调后，ActivityNet CIDEr 从 15.2 升到 46.23，达到真实数据微调 54.90 的约 84%；并在 MSR-VTT / VATEX / SMIT 上超过只用 ActivityNet 真实数据微调。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？**
  - UniSim 展示了跨数据类型与少量 real-robot zero-shot transfer，但作者明确指出：训练主要覆盖 4 种 robot morphology，未见 embodiment 上泛化有限。这里的贡献更像“统一数据接口 + 足够大的条件视频模型”，不是一个已解决通用物理规律的 simulator。
  - 数据 ablation 中，internet-only FVD 219.62、without-internet 307.80、完整 mixture 211.30，说明 broad prior 与 action-rich domain data 缺一不可；不是单纯扩大模型就能替代数据编排。

- **对 Infra → Post-training → Physical AI 迁移的直接启发：**
  1. **World-model rollout server ≈ RL rollout engine：** model version、environment seed、action schema、history window、sampling config、reward version都必须进入 trajectory metadata，否则无法复现和定位 reward hacking。
  2. **Learned simulator 需要 uncertainty-aware routing：** 高置信常规段走生成 simulator，contact-heavy / OOD 段回退显式物理或真机数据；像 cascade evaluator，而不是让单一模型裁决全部 rollout。

- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：**
  - **可扩展性：** 256-step diffusion 极慢，actor 会被 environment latency 主导；需要 batching、异步 actor、rate limiting、cache，未来更适合 latent video model / consistency distillation。
  - **成本：** 512 TPU-v3 × 20 天约 245,760 chip-hours，只是 simulator pretraining；policy rollout 还持续支付生成成本。
  - **评测自动化：** FVD/CLIP 只能测感知质量，必须加 action compliance、object permanence、contact consistency、counterfactual consistency、closed-loop policy regret 和 real-robot calibration。
  - **可复现性：** 记录 dataset mixture、domain identifiers、CFG strength、history frames、diffusion seed、simulator checkpoint、reward checkpoint 与 RPC version；否则同一 action 的 stochastic outcome 无法审计。

## 疑问 / 下一步

- **想深挖：** 如何定义 `model exploitation gap`：同一 policy 在 UniSim、显式 simulator、real robot 三个环境中的 return / state visitation divergence？仅比较生成视频质量会漏掉最危险的 policy-induced distribution shift。
- **限制提醒：** unrealistic action 会触发 hallucination；近期 4 帧无法保存长期 object permanence；未见 morphology 泛化弱；只模拟视觉，不适合 force/contact 变化但像素近似不变的任务。
- **第一个小实验：** 不复现 5.6B 训练，先实现一个 toy RPC Gym env：用小型 action-conditioned video predictor 作为 `step()`，另训 progress reward；比较 BC 与 model-based rollout fine-tuning，并用 held-out ground-truth env 统计 exploit gap。
- **下一步：** Day06 DreamerV3——从像素级 diffusion simulator 切到 compact latent dynamics，比较 rollout throughput、reward grounding、uncertainty 与真实世界可迁移性。

## 原文金句 (1-2句)

> “We define a simulator of the real world as a model that, given some state of the world (e.g., an image frame), can take in some action as input, and produce the visual consequence of the action (in the form of a video) as output.”

> “We formulate the action-in-video-out framework as an observation prediction model conditioned on finite history and parametrized by a video diffusion model.”

## 今晚产出

- [ ] 画一页 `actor → RPC video env → learned reward → replay/learner` 数据流图，标清版本与 trajectory metadata
- [ ] 用表格对比 Isaac Lab / Genie / UniSim：state、action、transition、reward、throughput、可验证性、主要 failure mode
- [ ] 写出 5 个 learned-simulator eval：action compliance、object permanence、contact consistency、OOD detection、policy exploit gap
- [ ] 用 10 行伪代码写 UniSim 的 autoregressive `step()` 与 progress reward

## 连接
- 上一篇: Day04 — Genie: Generative Interactive Environments
- 下一篇预告: Day06 — DreamerV3: Mastering Diverse Domains through World Models
- 相关: Day02 MuJoCo（显式 dynamics）；Day03 Isaac Lab（GPU physics + sim2real）；后续 VLA / RL for Robotics

## 参考链接
- Paper (arXiv): https://arxiv.org/abs/2310.06114
- Project / demos: https://universal-simulator.github.io
