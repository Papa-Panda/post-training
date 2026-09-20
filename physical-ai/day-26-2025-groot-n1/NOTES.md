# Day 26 — GR00T N1：双系统人形基础模型 + 数据金字塔

## 元信息
- Title: GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- Authors / Org: NVIDIA（App. A 完整作者列表；GTC 2025 发布）
- Link / arXiv: https://arxiv.org/abs/2503.14734（v1 2025-03-18；v2 含修订）
- 官方页面：https://research.nvidia.com/publication/2025-03_nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots
- Official code: https://github.com/NVIDIA/Isaac-GR00T（Apache 2.0，模型 / 代码 / 数据 / 评测基准全部开源）
- 模型权重：https://huggingface.co/nvidia/GR00T-N1-2B（GR00T-N1-2B：2.2B 总参数，其中 VLM 1.34B）
- Date read: 2026-09-17
- Tags: [physical-ai, groot-n1, humanoid, vla, dual-system, flow-matching, diffusion-transformer, data-pyramid, sim2real, cross-embodiment]
- Thread: physical-ai
- Folder: day-26-2025-groot-n1
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-26-2025-groot-n1

## 一句话总结
GR00T N1 是 NVIDIA GTC 2025 开源的**首个人形机器人 foundation model**：Kahneman 双系统架构——慢思考的预训练 VLM（Eagle-2，System 2，10Hz）负责理解场景和语言目标，快思考的 Diffusion Transformer（System 1，120Hz）用 flow matching 生成高频闭环动作 chunk；训练上最大的创新不是模型而是**数据金字塔**（web/人类视频垫底 → 仿真+神经生成数据在中 → 真机轨迹封顶，780K 条仿真轨迹 11 小时生成，等价 9 个月人工遥操作），真机 GR-1 双臂语言条件任务上以高数据效率超越 SOTA 模仿学习基线——它把"数据从哪来"这个 Physical AI 最大的 infra 问题，做成了一个可工程化的回答。

## 和之前工作的关系

- **接了哪条线**：Day25–30 "Physical AGI / Eval / Safety" 块的第二篇。Day25 Gato 定义了 generalist 的想象上限，GR00T N1 给出一个**完全开源、可复现、可 fine-tune**的工程实现——Gato 无官方代码，GR00T N1 模型+代码+数据+评测全开（Apache 2.0）。
- **补了哪个短板**：Day25 Gato 用离散 token 做动作（精度瓶颈）；GR00T N1 继承 Gato 的 generalist 精神，但动作头改成**连续的 flow matching DiT**（Day11 π₀ / Day12 Diffusion Policy 路线的修正方向），同时保留 VLM 的语义泛化——是"Gato 的 generalist + π₀ 的连续动作"的一次合流。
- **替代 / 分叉 / 改进**：
  - vs Day11（π₀）：都是 flow matching 动作生成。差别有三：① π₀ 单系统（一个网络从观测直达动作），GR00T N1 双系统（VLM 推理与 DiT 控制解耦、异步频率）；② π₀ 闭源，GR00T N1 全开；③ GR00T N1 的数据金字塔（human video + neural traj + sim）是 π₀ 公开材料之外的**数据 infra 方法论**。
  - vs Day12（Diffusion Policy）：DiT + flow matching 本质上是 DP 的 DDPM action diffusion 的**工程替代**（ODE 积分，K=4 步即可推理，比 DDPM 的多步去噪快；Day11 已讲过 rectified flow 的数学）。
  - vs Day13（Octo）："统一 backbone + embodiment 可插拔 readout"思想的**直接继承和放大**——Octo 的 diffusion readout head → GR00T N1 的 embodiment-specific state/action encoder-decoder MLP；Octo 只在机器人域内，GR00T N1 把人类视频（latent action / LAPA）也当成一种 embodiment 混进来。
  - vs Day09（RT-2 / OpenVLA）：RT-2 用离散 action token，OpenVLA 同理。GR00T N1 的选择是：**VLM 只负责推理（System 2），动作一律连续生成**——承认离散化精度损失是接触任务的硬伤，不在 System 1 里妥协。
  - vs Day15（OXE）/ Day18（RoboCasa）：OXE 真实多 embodiment 数据是数据金字塔的塔尖原料之一；DexMimicGen（RoboCasa 的 MimicGen 人形版）是塔中仿真数据的生成引擎——Day18 的"程序化放大演示"在这里变成 780K 轨迹的工业管线。
  - vs Day19–24（RL 块）：GR00T N1 是**纯 BC / flow-matching，没有 RL**。它回答的是"演示者从哪来、怎么放大"，RL 块回答"如何超越演示者"——数据资产 infra 和在线优化 infra 的两条腿，合起来才是 Day30 数据飞轮。

## 为什么今天读它

Roadmap Day26–30 是"Physical AGI 落地工程"：Day26 的 GR00T N1 是这个块里**唯一全开源的、可复现的人形 VLA**——对你而言它是"如果自己搭一个 generalist humanoid policy stack，参考实现长什么样"的活答案。它的双系统架构给出了一个**可部署的两时间尺度控制**设计（10Hz 推理 vs 120Hz 控制），它的数据金字塔给出了"真机数据永远不够"这个根本约束下的**数据 infra 配方**（web 视频 + VQ-VAE latent action + 视频生成模型 + DexMimicGen 仿真），训练 infra（OSMO + Ray + 1024 H100，5 万 H100 小时）则是你熟悉的 infra 语言。这是把 Day01–25 的点连成"一条可交付的管线"的一天。

## 今天的 3 问
1. 双系统解耦：System 2（VLM，10Hz）输出的 vision-language token $\phi_t$ 是 System 1（DiT，120Hz）唯一的语义条件。推理侧频率差 12 倍意味着 DiT 在连续 12 个控制步里复用同一套语义 token——**语义 stale**的代价是什么？论文用"中层 LLM embedding（第 12 层）而非末层"来同时提速和提分，这暗示了什么（表征的哪一层对动作生成最有用）？
2. Flow-matching 损失（式 1）里，LAPA latent action 和真实机器人动作是**同一套 $V_\theta$ 参数、按 embodiment 分头训练**的。VQ-VAE 从视频帧对 $(x_t, x_{t+H})$ 提取的连续 latent embedding，真的是"动作"吗？它在物理上对应什么（跨 embodiment 的运动方向先验），又在什么情况下会误导 DiT？
3. 神经轨迹（neural trajectories）用视频生成模型把 88 小时真机遥操作放大到 827 小时（~10x），IDM 标注伪动作后 1:1 co-train——**视频模型的幻觉（物体消失/物理不自洽）会不会污染 policy？** 论文用 MLLM 当 judge 过滤+重打标，这个"模型审模型"的数据闭环的失败模式是什么？

相关讨论（Gemini 网页版，2026-09-19）：https://gemini.google.com/app/ee2b6d0b878b1530

## 核心

### 1. Motivation
- 人形机器人是 physical AGI 最自然的硬件载体，但"没有人形机器人的互联网级数据集"——单台硬件的数据量小好几个数量级；OXE 式的跨 embodiment 混合又变成"数据孤岛群岛"（控制模式、自由度、传感器各异）。
- Baseline（单 embodiment IL、RT-2 式离散 VLA、纯仿真 DP）各自缺一块：泛化语义、连续精细控制、数据规模。GR00T N1 的目标是**一次给出"模型+数据+开源"的完整答案**，让社区能 fine-tune 而不是从零造轮子。
- 和 Physical AGI 的关系：它不宣称 AGI，只宣称"foundation model for humanoid robots"——**语义理解（System 2）+ 实时运动生成（System 1）+ 跨 embodiment** 三个 generalist 的必要条件，全部给出可训练的实现。

### 2. System / Method
- **双系统架构**（Kahneman, 2011 的快慢思考）：
  - System 2：Eagle-2 VLM（SigLIP-2 视觉编码器 + 微调自 SmolLM2 的 LLM；224×224 图像经 pixel shuffle 得每帧 64 token），10Hz 运行在 L40 上，把图像+语言指令变成环境/任务 token $\phi_t$ 。
  - System 1：Diffusion Transformer $V_\theta$ （AdaLN 步数条件），在 proprioceptive state $q_t$ 、加噪动作 chunk $A_t^\tau$ 上做 self-attention，对 $\phi_t$ 做 cross-attention（Flamingo/VIMA 式交替块），120Hz 输出闭环电机动作。
  - 两系统**紧耦合、端到端联合训练**，不是 pipeline 拼接。
- **跨 embodiment 处理**：每个 embodiment 一对 MLP（state/action encoder 把变维度 $s/a$ 投到共享 embedding 维；末端 embodiment-specific action decoder），"不同机器人 = 不同 embodiment 分头"——包括人类视频的 latent action（LAPA embodiment）。
- **动作分块**： $A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}]$ ， $H = 16$ （Zhao et al., 2023 的 ACT 路线）；推理时 $K=4$ 步欧拉积分去噪，16 个动作 chunk 在 L40 bf16 下 63.9ms。
- **关键工程细节**：取 LLM **中间层（第 12 层）**embedding 而非末层——更快、policy 成功率反而更高（末层过拟合语言任务，中间层保留更多视觉 grounding）。

### 3. Training / Data Details
- **数据金字塔**（从底到顶：数据量递减、embodiment 特异性递增）：
  - 底层：web 数据（VLM 预训练已含）+ 人类第一视角视频 → VQ-VAE 从帧对 $(x_t, x_{t+H})$ 提取 latent action $z_t$ （编码器→codebook 最近邻量化，解码器重建 $x_{t+H}$ ），连续 pre-quantized embedding 当伪动作标签，按 LAPA embodiment 训练——**把"看人类视频"变成"多一个 embodiment 的 BC"**。
  - 中层：① **神经轨迹**：在 88 小时自有遥操作数据上微调 img2video 模型，用新语言 prompt 生成反事实视频，88h → 827h（~10x）；MLLM 当 judge 过滤不跟随指令的视频并重打标；② **仿真轨迹**：DexMimicGen（Day18 MimicGen 的人形双臂版）从少量人类演示自动扩增，**780K 条轨迹 = 6500 小时 = 9 个月连续人工演示，11 小时生成完毕**。
  - 顶层：真机数据——自有 GR-1 遥操作（VIVE 追踪手腕 + Xsens 手套捕捉手指，IK 重定向，20Hz，头戴相机）、OXE（RT-1/Bridge-v2/DROID 等）、AgiBot-Alpha（14 万轨迹）。
- **训练配方**：预训练全程 flow-matching（式 1），跨 embodiment+跨数据源混合采样；post-training 阶段冻结 VL backbone 的语言部分，只 fine-tune 其余（适配器层+DiT 可在**单张 A6000** 上训，batch 200）——**低成本适配是开源策略的核心**。
- **训练 infra**：NVIDIA OSMO 编排 + 自研 Ray 库容错多节点训练，单模型最多 1024 H100，GR00T-N1-2B 预训练约 **50,000 H100 GPU 小时**。
- Verifiable signal：纯 BC/flow-matching，**没有 RL、没有 reward**——成功信号只来自演示数据的筛选（DexMimicGen 只保留成功的仿真 rollout；神经轨迹靠 MLLM judge 过滤）。

### 4. Key Tricks（最值得抄的 3 个）
1. **latent action 把 action-less 视频变成可训练数据**：VQ-VAE $(x_t, x_{t+H}) \to z_t$ 统一了所有数据源的"动作"接口，人类视频/神经视频/机器人轨迹共享一套 latent action 空间（图 4 跨 8 种 embodiment 检索到一致的"右臂左移"）。这是数据金字塔能成立的**数学粘合剂**。
2. **视频生成模型当数据放大器**：88h → 827h 的神经轨迹管线（微调 img2video + 新 prompt 反事实 + MLLM 过滤/重打标 + IDM 伪动作 + 1:1 co-train），是"真机数据永远不够"约束下的**可复制配方**——Day27 Cosmos 就是把这条线做成 foundation model。
3. **双系统异步 + 中层特征**：10Hz 语义 / 120Hz 控制的频率解耦让 VLM 的延迟不进控制环；取 LLM 第 12 层而非末层 embedding（更快+更准）——**"最好的表征层不一定是最后一层"**，对任何 VLM-as-backbone 的 policy 都是可直接抄的消融结论。

### 5. Results
- 仿真标准 benchmark（多 embodiment）上**超越 SOTA 模仿学习基线**（论文摘要原话；具体 benchmark 含 RoboCasa/DexMimicGen 系任务）。
- 真机：Fourier GR-1 人形机器人，**语言条件双臂操作任务**，高数据效率（post-training 少量演示即达强性能）。
- 开源交付：GR00T-N1-2B 权重（HF）、训练数据、仿真评测基准、Isaac-GR00T 代码库，Apache 2.0——这是它和 Gato/π₀/RT-2 最大的**生态位差异**：别人发论文，它发 infra。
- 续作注记：GR00T N1.5（2025-05）后续发布，主要改进合成数据质量和指令跟随；路线未变。

### 数学视角（统一框架：两时间尺度级联控制 + 条件流匹配）

**框架**：把 GR00T N1 看成一个**部分可观测马尔可夫决策过程（POMDP）上的两时间尺度级联策略**。

- **记号**：时刻 $t$ ，真实状态 $s_t$ 不可观测；观测 $o_t = (I_t, q_t, \ell)$ ，其中 $I_t$ 为多视角 RGB 图像（每帧经 SigLIP-2 + pixel shuffle 压缩为 64 token）， $q_t \in \mathbb{R}^{d_q}$ 为本体感知（关节角/速度， $d_q$ 随 embodiment 变）， $\ell$ 为语言指令。动作 $a_t \in \mathbb{R}^{d_a}$ （关节/末端指令， $d_a$ 随 embodiment 变）。动作为块 $A_t = [a_t, \dots, a_{t+H-1}] \in \mathbb{R}^{H \times d_a}$ ， $H = 16$ 。
- **System 2（慢系统，10Hz）**：VLM 编码器 $\Phi$ 把 $(I_t, \ell)$ 映射为语义 token $\phi_t = \Phi(I_t, \ell) \in \mathbb{R}^{L \times d}$ （实际取 LLM 第 12 层隐状态）。它不输出动作，只输出**任务的充分统计量**——相当于把 POMDP 的 belief $b_t$ 压缩成一个低频更新的语义先验。
- **System 1（快系统，120Hz）**：DiT 向量场 $V_\theta(\phi_t, A_t^\tau, q_t)$ ，在加噪动作块 $A_t^\tau$ 上做流匹配去噪。

核心训练目标（论文式 1，flow matching）：

$$\mathcal{L}_{fm}(\theta) = \mathbb{E}_{\tau}\left[\lVert V_{\theta}(\phi_t, A_t^{\tau}, q_t) - (\epsilon - A_t)\rVert^{2}\right]$$

- **每个符号**： $\tau \in [0,1]$ 为流时间（注意它**不是**机器人物理时间 $t$ ，是去噪过程的虚拟时间）； $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 为高斯噪声； $A_t^\tau = \tau A_t + (1-\tau)\epsilon$ 为插值路径（ $\tau=0$ 纯噪声， $\tau=1$ 真实动作块）；目标向量场 $\epsilon - A_t$ 是从噪声指向数据的**直线方向**（rectified flow，Day11 π₀ 已讲：比 DDPM 的弯曲 SDE 路径更短、更少步数可解）； $\tau$ 的采样分布 $p(\tau) = \text{Beta}(\frac{s-\tau}{s}; 1.5, 1)$ ， $s = 0.999$ ——Beta(1.5,1) 偏向大 $\tau$ ，即**训练时更关注接近真实动作的精细去噪阶段**，这正是接触任务需要精度的区域。
- **推理**：从 $A_t^0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发， $K=4$ 步前向欧拉 $A_t^{\tau+1/K} = A_t^{\tau} + \frac{1}{K}V_\theta(\phi_t, A_t^\tau, q_t)$ 。 $K=4$ 之所以够用，正因为目标向量场近似直线——这是 flow matching 相对 DDPM（Day12）的**核心数学红利**：直线路径的截断误差是 $O(1/K^2)$ 量级，弯曲路径需要更多步。
- **跨 embodiment 的数学处理**：encoder MLP $E_{\text{emb}}: \mathbb{R}^{d_a^{(\text{emb})}} \to \mathbb{R}^{d}$ 把不同维度动作投影到共享空间——相当于为每个 embodiment 学一个**线性（+非线性）坐标卡**，DiT 永远在共享坐标里做流匹配，decoder 再投影回去。LAPA latent action 只是多了一张"人类视频"坐标卡。
- **两时间尺度的控制含义**： $\phi_t$ 每 100ms 更新一次，DiT 每 ~8ms 用同一 $\phi_t$ 生成新动作块——这是一个**慢变参数的快变控制器**（slow-fast cascade）。数学上要求任务语义在 100ms 内近似恒定（ $\lVert \phi_{t+\delta} - \phi_t \rVert$ 小），快速变化的场景（物体被碰倒）会有最多 100ms 的语义延迟——这是双系统架构**没有覆盖的真实部署误差之一**。
- **数学没有覆盖的**：① flow matching 只建模**动作分布**，不建模世界动力学 $p(s_{t+1}|s_t,a_t)$ ——和 Day25 Gato 一样，GR00T N1 不能做 planning/imagination（对比 Day06 Dreamer）；② 损失是 BC，没有**超越演示者**的机制（对比 Day19–24 RL）；③ VQ-VAE latent action 的"动作"语义只在**视觉变化可逆**时成立——力/触觉不可见的接触（打滑、软体变形）无法从 $(x_t, x_{t+H})$ 恢复，latent 监督在这些任务上是**有偏的**。

## 可迁移 / Transfer

- 模型 vs 框架哪个贡献大：论文的消融指向**数据框架**（金字塔 + 神经轨迹 + 1:1 co-train）贡献大于单点模型 trick——模型本身是"VLM + DiT + flow matching"的标准件组合，没有私有算子。**开源的真正资产是数据配方和评测基准**，不是 2.2B 权重。
- 对 Infra → Post-training → Physical AI 的启发：
  1. **数据 infra 是 Physical AI 的第一性原理约束**：780K 轨迹 / 11 小时 / 50K H100 小时这组数字说明，robotics foundation model 的 scaling law 目前瓶颈不在模型而在**数据生成管线**（仿真并行 + 视频生成 + 自动标注）。这正是你 ML-for-infra 背景（autoscaling、数据管线）可以直接变现的地方。
  2. **双系统 = 推理与控制的解耦部署**：10Hz 大模型 + 120Hz 小快模型的异步级联，是把"大模型延迟"挡在控制环之外的通用模式——和 LLM serving 里 prefill/decode 分离、speculative decoding 的思想同构。
- Infra 视角：训练用 OSMO + Ray 容错（1024 H100 fat-tree），post-training 可压到**单 A6000**（只训 adapter + DiT，batch 200）——**预训练重、适配轻**的分层算力设计，是 foundation model 落地的标准 infra 形状；评测基准开源则解决了 robotics 最大的可复现性痛点（对比 Day28 要讲的评测主题）。

## 疑问 / 下一步

- 没看懂的 1 个问题：LAPA latent action 和真实动作**共享同一套 DiT 参数**训练时，两种监督信号的梯度会不会打架（latent action 是视觉运动先验，真实动作是精确电机指令）？论文按 embodiment 分头但共享主干——这个共享到底是迁移还是干扰，需要看消融。
- 如果要复现 / 小规模试：第一个实验——拿 Isaac-GR00T 仓库 + GR00T-N1-2B 权重，在**单 A6000** 上按官方脚本只 fine-tune adapter（state/action encoder-decoder + DiT），用 50–100 条自己 teleop 的单臂任务验证"冻结 VLM 语言部分 + 小数据适配"的 data-efficiency 宣称是否成立。

## 原文金句
> However, unlike the digital realms of words and pixels, no Internet of humanoid robot datasets exists for large-scale pre-training.
> （不像文字和像素的数字世界，不存在可供大规模预训练的人形机器人数据集互联网——这句话是整篇论文、也是数据金字塔的全部动机。）

## 今晚产出
- [x] 按模板补齐 System / Training / Key Tricks / Results / 可迁移（含数学视角：两时间尺度级联 + 条件流匹配）
- [x] 保留并完善「和之前工作的关系」小节（vs Day25 Gato / Day11 π₀ / Day12 DP / Day13 Octo / Day09 RT-2 / Day15 OXE / Day18 RoboCasa / Day19–24 RL）
- [x] 更新 reading-log.csv、README.md 进度（26/30），commit + push

## 连接
- 上一篇: Day 25 — Gato（https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-25-2022-gato）
- 下一篇预告: Day 27 — Cosmos 世界基础模型（用世界模型生成/筛选 Physical AI 训练数据——正是今天"神经轨迹"这条线的 foundation-model 版本）
