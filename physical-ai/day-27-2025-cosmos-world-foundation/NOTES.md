# Day 27 — Cosmos：世界基础模型平台（World Foundation Models for Physical AI）

## 元信息
- Title: Cosmos World Foundation Model Platform for Physical AI
- Authors / Org: NVIDIA（Niket Agarwal, Arslan Ali, …, Sanja Fidler, Dieter Fox, Ming-Yu Liu 等；完整贡献名单见论文附录 A）
- Link / arXiv: http://arxiv.org/abs/2501.03575（v1 2025-01-07；v3 2025-07-09）
- 官方项目页：https://research.nvidia.com/labs/cosmos-lab/cosmos-predict1/
- Official code: https://github.com/nvidia-cosmos（代码 Apache 2.0；模型权重 NVIDIA Open Model License）
  - Cosmos-Transfer1（world-to-world）：https://github.com/nvidia-cosmos/cosmos-transfer1
  - Cosmos-Reason1（物理推理 VLM）：https://github.com/nvidia-cosmos/cosmos-reason1（论文 http://arxiv.org/abs/2503.15558）
  - Cosmos-Drive-Dreams（合成驾驶数据管线）：https://github.com/nv-tlabs/Cosmos-Drive-Dreams
- Date read: 2026-09-18
- Tags: [physical-ai, cosmos, world-foundation-model, video-generation, world-model, synthetic-data, sim2real, tokenizer, diffusion, autoregressive]
- Thread: physical-ai
- Folder: day-27-2025-cosmos-world-foundation
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-27-2025-cosmos-world-foundation

## 一句话总结
Cosmos 是 NVIDIA 在 CES 2025 发布的**世界基础模型（WFM）平台**：核心主张是"Physical AI 需要先在数字世界里训练"——一个 policy model（自己的数字孪生）配一个 world model（世界的数字孪生）；平台把互联网 20M 小时视频切成约 100M 个 clip，用因果视频 tokenizer（CV 连续 / DV 离散）压缩成 token，在上面训出 diffusion 与 autoregressive 两条 WFM 路线，再用 post-training（相机控制 / 机器人操作 / 自动驾驶）定制到具体 Physical AI 场景——它是把 Day04–06 的"世界模型"从单点方法升级成**可复用的数据-模型-配方 infra**的工业答案；但论文自己诚实声明：五个宣称用途（policy 评估/初始化/训练/planning/合成数据）**全部没有实证结果**——视频 realism 和 action-grounded usefulness 之间的差距，正是今天要解剖的。

## 和之前工作的关系

- **接了哪条线**：Day04 Genie / Day05 UniSim / Day06 DreamerV3 的"世界模型"线。Genie 是从无标签视频学 latent action 的可交互生成式世界（学术起点）；UniSim 是 action-in-video-out 的 learned simulator（5.6B video diffusion，Language Table 模拟评估 BC 0.58→0.81）；DreamerV3 是 latent RSSM 用来在隐空间里"想"（planning/imagination）。Cosmos 是这一脉的**工业基础模型平台版**：video curator 数据管线 + tokenizer 家族 + 双路线预训练 WFM + post-training 配方 + guardrail，全套开源——Genie/UniSim/Dreamer 各自是单点方法，Cosmos 回答的是"别人怎么低成本训出自己的定制 WFM"。
- **补了哪个短板**：Day04–06 都没有给出"数据从哪来、怎么处理、怎么规模化"的工程答案；Cosmos 的 video curator（Ray 编排 + GPU H.264 硬解 + VLM 标注 + 语义去重）正是那个缺失的 infra 层。"Data determines the ceiling of an AI model"——它把数据 curation 写成平台的一等公民。
- **替代 / 分叉 / 改进**：
  - vs Day26（GR00T N1，昨天）：GR00T N1 的"神经轨迹"（88h 真机遥操作 → 827h 生成视频 → IDM 伪动作 → 1:1 co-train）就是 **WFM 当数据放大器**的一个具体实例；Cosmos 是把"视频生成模型"本身做成 foundation model。神经轨迹 = Cosmos 思想的特化版；Cosmos-Drive-Dreams（arXiv:2506.09042）则是这条线在自动驾驶上第一次交出实证：合成数据确实提升 3D lane detection / 3D object detection / policy learning，尤其长尾场景，即使叠加在大量真实数据上仍有增益。
  - vs Day06（DreamerV3）：Dreamer 是 latent dynamics model，用来"想"；Cosmos 是 generative world model，用来"造"。数学上 Cosmos 的 $\mathcal{W}$ 是 Dreamer 的 RSSM $p(s_{t+1}\mid s_t,a_t)$ 的**像素空间对应物**——不学隐状态，直接在视频观测上 rollout（见数学视角）。
  - vs Day05（UniSim）：UniSim 已经是"action-conditioned video diffusion 当 RL 环境"，Cosmos 的 post-training（Sec 6.2，video-action 序列微调）走的是同一条路，但 Cosmos 多了**预训练-后训练范式**：先在 20M 小时通用视频上学生成式物理先验，再用小得多的 prompt-video 对定制。
  - vs Day25（Gato）：Gato 把世界和动作统一进一个 token 流做 BC（"在世界里行动"）；Cosmos 只建模世界 $p(x_{t+1}\mid x_{0:t},c_t)$ ，不直接输出动作（"造世界"）——两者是分工关系，不是替代关系。
  - **Atlas 对照（World Labs，2026-09-01 发布）**：Atlas 是 Cosmos 这条线的"新一代对照组"（见 README 问答归档 2026-09-12）。它是个 omni-model，第一次把"像素重建"和"像素生成"统一到以 viewpoint 为锚的模型里：2–3 张照片重建真实场景、最长 1 分钟 1440p 视频、输出显式 3D（point clouds、3D Gaussian splats）、几何持久性；官方定位之一正是 real-to-sim for robotics（手机拍 24 帧视频 → 重建出"仿真机器人移动时会看到的 RGB 和 depth 观测"）。对比 Cosmos：Atlas 比 Cosmos 更接近 useful（显式 3D、有 depth、能进仿真管线），但硬标准仍未达到——需要的是"可交互、有碰撞、有物理和动力学"的 3D 环境。好看、可走进去 $\neq$ 有物理 $\neq$ 策略在里面训完能 transfer。这个 realism → usefulness 的断层，就是 Day27 的主题句。
  - Cosmos-Reason1 是 Cosmos 家族的"推理侧"：7B/56B MLLM，physical common sense 层级本体 + embodied reasoning 二维本体，Physical AI SFT + RL——对应 Day26 双系统架构里的 System 2（语义推理），而 Cosmos Predict/Transfer 是"世界动力学"侧。两者合起来才是完整的"理解 + 生成"栈。

## 为什么今天读它

Roadmap Day27 的官方主题就是"以世界基础模型生成/筛选 Physical AI 训练数据，评估视频 realism 与 action-grounded usefulness 的差距"——Cosmos 是这一主题的工业级主角（20M 小时视频、开源平台、post-training 配方、transfer 到 sim2real 的具体模型）。它直接衔接昨天 Day26 的"神经轨迹"数据放大：从特例走向平台。同时它是 Day30 数据飞轮里"世界模型"一角的 canonical 参考实现，读完它，Day28（评测基准）和 Day29（安全）才有东西可评、可约束。

## 今天的 3 问
1. WFM 的数学定义 $\hat{x}_{t+1}=\mathcal{W}(x_{0:t},c_t)$ 里 $c_t$ 可以是动作、文本描述或随机扰动——但预训练数据（20M 小时互联网视频）里**几乎没有 $(x_t,a_t)$ 交错的动作监督**。没有 action-grounded 监督的 WFM，学到的到底是"物理"还是"视频统计相关性"？Sec 6.2 的 video-action 序列 post-training 能补上多少因果性？（这是 realism vs usefulness 差距的数学根源，见数学视角末尾。）
2. Cosmos-Transfer1 的自适应时空控制：同一视频里 segmentation / depth / edge / vis 可以按区域、按时刻加不同权重——这对 sim2real 意味着什么？Day21 的 ADR 是"手工随机化物理参数分布"，Transfer 是"保留模拟几何、用真实感重绘纹理"。控制权重 $w_m(x,t)$ 目前是手工旋钮，能不能从真机数据里学出来（呼应 Day24 SimOpt 的"有锚辨识"思想）？
3. 论文 Sec 2.1 末尾诚实声明：五个用途（policy 评估/初始化/训练/planning-MPC/合成数据）"没有实证结果"。从视频 realism（FVD/PSNR 好看）到 policy 提升之间，缺一个什么样的评测协议？action-conditioned rollout 的 counterfactual 一致性该怎么量？（直接引出 Day28 评测主题。）

## 核心

### 1. Motivation
- Physical AI 的数据 scaling 比 LLM 难一个量级：需要的是**观测-动作交错序列**，而探索动作会扰动物理世界、可能损坏系统——"infancy 时期的探索尤其危险"。WFM（世界的数字孪生）是"a long-sought remedy to the data scaling problem"。
- Baseline 为什么不行：纯仿真缺真实感（Day21/24 的 sim2real 鸿沟），纯真机数据贵且危险（Day16 DROID 的 76k 轨迹已是极限采集），单点视频生成模型（Genie/UniSim）没有可复用的训练 infra。
- 和 Physical AGI 的关系：WFM 是 Physical AGI 的"数字练功房"——policy 先在 WFM 里试错，再上真机。论文把用途列成五条：policy 评估、policy 初始化、policy 训练（+reward 当 RL 环境）、planning/MPC、合成数据生成。

### 2. System / Method
- **WFM 的数学定义**（论文 Sec 2，图 3）： $\hat{x}_{t+1}=\mathcal{W}(x_{0:t},c_t)$ ， $x_{0:t}$ 为 RGB 视频观测， $c_t$ 为对世界的扰动（可以是 Physical AI 的动作、随机扰动、文本描述等）。平台五件套：
  1. **Video curator**：视频按 shot 切分（无场景切换）→ 质量/动态信息量过滤 → VLM 每 256 帧打 caption → 语义去重；Ray 编排异构吞吐模型，GPU H.264 硬件编解码。20M 小时 → 约 100M 个 2–60 秒 clip。
  2. **Video tokenizer（Cosmos Tokenizer）**：attention encoder-decoder，同时做连续（CV）与离散（DV）token；**因果设计**——当前帧的 token 不依赖未来观测，好处有二：训练侧可做图像+视频联合训练（因果视频 tokenizer 在单帧输入时就是图像 tokenizer），应用侧对齐 Physical AI 的因果世界。压缩率：图像 CI/DI 为 $8\times8$ / $16\times16$ ，视频 CV/DV 为 $4\times8\times8$ / $8\times8\times8$ / $8\times16\times16$ （ $T\times H\times W$ ）。编码器 $\mathcal{E}$ ： $z_{0:T'}\in\mathbb{R}^{(1+T')\times H'\times W'\times C}$ ，空间压缩 $s_{HW}=H/H'$ ，时间压缩 $s_T=T/T'$ ；首个时间 token 表第一帧，实现图像-视频共享 latent 空间。微调阶段加对抗 loss 保大压缩率下的细节。
  3. **Pre-trained WFM（双路线）**：① Diffusion 路线（连续 token）：Text2World 预训练 → Video2World 微调（过去视频 + 文本 prompt → 未来视频），DiT 架构，T5 文本 embedding 经 cross-attention 注入；② Autoregressive 路线（离散 token）：vanilla next-token 预训练（foresight）→ text-conditioned Video2World。用 Cosmos-Tokenize1-DV8×16×16-720p（离散整数，压缩激进）时 AR 输出易模糊——解法是**两段式**：AR 先 rollout 离散 token，再用一个微调出的 diffusion decoder（Cosmos-Predict1-7B-Decoder-DV8×16×16ToCV8×8×8-720p）把离散 token 视频"翻译"回连续 token 视频（CV8×8×8）再解码——"粗生成 + 精修复"，承认离散化的精度税。另有基于 LLM 的 prompt upsampler 增强可控性。
  4. **Post-training（三示例）**：① 相机位姿控制（Plücker embedding，时间压缩 $8\times$ 下每 8 帧取第 4 帧位姿拼到 latent 上）→ 可导航虚拟世界；② 机器人操作（video-action 序列微调，instruction following）→ "better predict the future state of the world based on the action taken by the robot"；③ 自动驾驶多任务。范式：预训练给通用物理先验，post-training 只需小得多的 prompt-video 对。
  5. **Guardrail**：pre-Guard 挡有害输入，post-Guard 挡有害输出。
- **Cosmos-Transfer1（world-to-world）**：在模拟渲染的 segmentation / depth / canny edge / blur visual（Sample-AV 版：LiDAR / HDMap）上做条件生成——"bridges the perceptual divide between simulated and real-world environments"。多 ControlNet 分支 + **自适应时空控制图**：不同模态在不同区域、不同时刻可加不同权重（结构保真 vs 视觉多样性的旋钮）；单模态退化为 ControlNet；另有 4K upscaler。7B DiT，预训练于约 20M 小时视频，可直接零微调用。
- **Cosmos-Reason1**：7B/56B MLLM，physical common sense（空间/时间/物理的层级本体）+ embodied reasoning（跨 embodiment 的二维本体），Physical AI SFT + Physical AI RL，长 CoT 输出具身决策（如下一步动作）——家族里的"推理侧"。
- **Cosmos-Drive-Dreams**：Cosmos 在 AV 上的 post-training 套件 + 合成数据管线（RDS 3.6M 个 20 秒 6 视角 clip ≈ 20,000 小时；RDS-HQ 750 小时带 HDMap/3D cuboid/LiDAR），含 Annotate（给野外视频打 HDMap/LiDAR 标注）、Single2Multiview、LiDAR-GEN 模型——"数据飞轮"的 AV 实例。

### 3. Training / Data Details
- 数据：20M 小时原始视频 → curator → 约 100M clip（2–60s）；VLM 每 256 帧一个 caption；语义去重保多样性。
- Tokenizer：重建 loss + 对抗微调；DAVIS 与 TokenBench 上压缩-质量 trade-off 达 SOTA（ $8\times16\times16$ 下仍超前人）。
- Diffusion WFM：Text2World 预训练 → Video2World 微调；AR WFM：next-token 预训练 → text-conditioned Video2World。
- Post-training 数据：目标 Physical AI 场景的 prompt-video 对（action 命令 / 轨迹 / 指令），量级远小于预训练。
- Verifiable signal：生成质量（3D 一致性、物理准确性；官网视频定性）+ tokenizer 重建指标（PSNR）+ Drive-Dreams 的下游任务增益；**主论文五个用途无实证**（作者原话，见金句）。

### 4. Key Tricks（最值得抄的 3 个）
1. **因果 tokenizer 即图像 tokenizer**： $z_{0:T'}$ 首 token 表第一帧 → 图像和视频共享 latent 空间，图像数据集（外观多样性远超视频）可直接参与 WFM 训练。任何视频世界模型的标准起手式：别为图像和视频各训一套 tokenizer。
2. **离散太激进就用 diffusion decoder 救**：AR 用 DV8×16×16 快速 rollout（token 少、便宜），再用条件去噪器升回 CV8×8×8 解码——把"生成"和"高保真重建"解耦。教训：压缩率是留给生成侧的预算，不是免费午餐。
3. **Transfer 的自适应时空控制图**：seg/depth/edge 权重可按空间位置与时刻调节——sim2real 从此有了"几何保真、纹理重绘"的可微旋钮：结构关键区域（交通参与者位置）权重拉满保语义，路面/光照区域放开让模型发挥。这是从 Day21 手工 DR 到"可控重绘"的升级。

### 5. Results
- Tokenizer：在 DAVIS / TokenBench 上各压缩率全面 SOTA（ $4\times8\times8$ 基准， $8\times8\times8$ / $8\times16\times16$ 下仍超前人）。
- 生成：高 3D 一致性、物理准确的视频（官网大量示例）；post-training 得到相机可控世界、指令跟随的机器人预测、AV 场景生成。
- Cosmos-Drive-Dreams：合成数据提升 3D lane detection / 3D object detection / policy learning，尤其挑战场景；即使叠加在大量真实数据上仍有可测增益——**WFM 当数据放大器第一次交出下游实证**。
- 诚实声明：主论文 Sec 2.1 末尾——五个用途"this paper does not include empirical results… We are eager to verify the claims in future work." 平台先行、实证待补，这是读 Cosmos 必须记住的诚实刻度。

### 数学视角（统一框架：条件视频动力学 —— Dreamer RSSM 的像素空间对应物）

**框架**：把 Cosmos 的 WFM 看成 POMDP 观测动力学 $p(x_{t+1}\mid x_{0:t},c_t)$ 的**生成式估计**。Day06 DreamerV3 学的是隐状态动力学 $p(s_{t+1}\mid s_t,a_t)$ 再在隐空间里 planning（"想"）；Cosmos 不学显式隐状态，直接在**视频观测空间**建模未来（"造"）。两者是同一张 POMDP 的两面：Dreamer 压缩世界再想，Cosmos 生成世界再看。

- **记号**： $x_{0:t}$ 为 RGB 视频（ $x_k\in\mathbb{R}^{H\times W\times 3}$ ，30fps 下 1 秒 = 30 帧）； $c_t$ 为扰动——可以是机器人动作 $a_t$ 、文本描述、相机位姿或随机噪声（论文刻意保持 $c_t$ 的开放性：预训练时它大多是文本，post-training 时才变成动作）； $\mathcal{W}$ 为世界基础模型； $\mathcal{E}$ / $\mathcal{D}$ 为因果视频编码器/解码器。
- **Token 化**： $z_{0:T'}=\mathcal{E}(x_{0:T})\in\mathbb{R}^{(1+T')\times H'\times W'\times C}$ ，空间压缩 $s_{HW}=H/H'=W/W'$ ，时间压缩 $s_T=T/T'$ 。取 $s_T=8$ 、720p 输入：一秒 30 帧 → 约 4 个 latent 时间步——**WFM 的"一步"是视频块级（~0.25 秒），不是控制级（8ms）**。这意味着 WFM 天然不适合直接进 100Hz+ 控制环；它的正确位置是 planning（秒级前瞻）或离线数据生成。这是时间尺度上必须记住的错位：Day11/12/26 的 policy 在动作块上做 diffusion/flow（毫秒级），Cosmos 在视频块上做 diffusion（秒级）。
- **Diffusion 路线**（连续 latent $z$ ，DiT backbone）：

$$\mathcal{L}_{\text{diff}}=\mathbb{E}_{z_0,\epsilon,\tau}\left[\lVert\epsilon-\epsilon_\theta(z^\tau,\tau,c)\rVert^2\right]$$

$z^\tau$ 为加噪 latent 视频， $\tau$ 为扩散时间， $c$ 经 T5 embedding + cross-attention 注入。直觉：和 Day12 Diffusion Policy 是同一个去噪数学，只是去噪对象从**动作轨迹**换成了**未来视频**——DP 学 $p(A\mid O)$ （"看到观测，动作该怎么走"），Cosmos 学 $p(x_{t+1}\mid x_{0:t},c_t)$ （"给定扰动，世界会变成什么样"）。一个是对自己身体的生成模型，一个是对世界的生成模型。

- **Autoregressive 路线**（离散 token $y_i\in\mathbb{Z}$ ，DV tokenizer）：

$$\mathcal{L}_{\text{AR}}=-\sum_{i=1}^{N}\log p_\theta(y_i\mid y_{<i},c)$$

直觉：把视频当语言来"说"，和 Day09 RT-2 把动作离散成 token 是同一个思想——离散化买可扩展性，付精度税。Cosmos 为这笔税付了两段式方案：AR 只管"说什么内容"（DV8×16×16，token 少、rollout 快），diffusion decoder 再把离散 token 翻译回连续 latent（CV8×8×8）做高保真解码——**内容规划与像素渲染解耦**。

- **Transfer 路线**（world-to-world，sim2real 的数学形态）：

$$\hat{x}=G(x_{\text{sim}};C),\quad C(x,t)=\sum_{m}w_m(x,t)\,C_m$$

$x_{\text{sim}}$ 为模拟渲染， $C_m$ 为各控制模态（seg/depth/edge/vis）， $w_m(x,t)$ 为自适应时空权重。直觉：Day21 ADR 的目标是 $J_{\text{DR}}(\theta)=\mathbb{E}_{\xi\sim p_\phi}[J(\theta;\xi)]$ ——在**参数分布**里随机化世界；Transfer 的目标是把**同一个几何**重绘成真实纹理——随机化的是外观流形，不是物理参数。两者正交：ADR 保"物理覆盖"，Transfer 保"视觉保真"，可以叠加。

- **数学没有覆盖的**（realism vs usefulness 差距的根源）：
  1. **无因果监督**：预训练的 $(x_{0:t},c_t)$ 对里 $c_t$ 几乎不是真实动作——模型学的是 $p(x_{t+1}\mid x_{0:t},\text{文本})$ 的相关性，不是干预分布 $p(x_{t+1}\mid x_{0:t},\text{do}(a_t))$ 。"如果我推一下杯子会怎样"这种 counterfactual，像素 loss 从不惩罚答错——只要生成的视频平滑、合理，loss 就满意。这是论文五个用途无实证的**数学原因**：好看 $\neq$ 可用来 planning。
  2. **像素重建不编码物理**：物体凭空消失、穿透、违反动量——只要像素连续， $\lVert\epsilon-\epsilon_\theta\rVert^2$ 照样小。物理正确性在 loss 里是**隐式**的（靠数据分布），不是显式的（没有接触/动力学约束项）——对比 Atlas 朝显式 3D 迈了一步，但仍无碰撞与动力学。
  3. **时间粒度错位**：latent 时间压缩 $8\times$ → WFM 的一步 ≈ 0.25 秒物理时间；接触动力学（碰撞、打滑）发生在毫秒级——WFM 的视频块根本**看不见**接触事件，做不了精细操作的 MPC。这也是它更适合 AV（秒级决策）而非灵巧手（毫秒级接触）的原因。

## 可迁移 / Transfer

- **模型 vs 框架哪个贡献大**：DiT / AR Transformer 都是标准件，真正的资产是**框架**：video curator 管线（Ray + GPU H.264 + VLM 标注 + 语义去重）、因果 tokenizer 家族、pre-train 大 + post-train 小的配方、Transfer 的可控生成接口。开源（代码 Apache 2.0，权重 NVIDIA Open Model License）的也是这一整套 infra，不是某个 checkpoint。
- 对 Infra → Post-training → Physical AI 的启发：
  1. **"Data determines the ceiling" 是 infra 人的母语**：20M 小时 → 100M clip 的 curation 管线，本质上是数据 autoscaling + 质量门禁——你在 ML-for-infra 里做过的数据管线、采样策略，在这里是第一性原理。WFM 的 scaling law 目前瓶颈在数据侧，不在模型侧。
  2. **pre-train/post-train 范式是通用形状**：大预训练（通用物理先验）+ 小 post-training（prompt-video 对定制）= LLM 的 pretrain/SFT 在 Physical AI 的翻版。你熟悉的 post-training 语言（SFT、RL、评测门禁）可以直接平移过来——Cosmos-Reason1 甚至就叫 "Physical AI SFT + Physical AI RL"。
- Infra 视角：Ray 编排异构吞吐（VLM 标注慢、转码快，按算子配资源）+ GPU 硬件编解码 + 开源权重，三者把"训自己的 WFM"从大厂专利变成可复现工程；guardrail（pre/post-Guard）是 WFM 进生产管线的必备件，Day29 安全主题会再展开。

## 疑问 / 下一步

- 没看懂的 1 个问题：Transfer1 的自适应时空权重 $w_m(x,t)$ 到底是手工设的还是学出来的？论文说 "adaptive"，但控制图的学习信号是什么（重建 loss？下游任务 reward？）——这决定了它是"可微旋钮"还是"超参"。
- 如果要复现 / 小规模试：第一个实验——用 Cosmos-Transfer1-7B 的开源权重，拿 Isaac Sim 渲染的 segmentation + depth 做一次 sim→real 纹理重绘，对比同一任务上 ADR（Day21）训出的 policy 的真机迁移成功率——验证"可控重绘 vs 参数随机化"哪条 sim2real 路线更有效、能否叠加。

## 原文金句
> Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model.
> （Physical AI 需要先在数字世界里训练：它需要自己的数字孪生——policy model，也需要世界的数字孪生——world model。整篇论文、整个 Cosmos 平台的立论句。）

> While we list the possibilities, this paper does not include empirical results in applying Cosmos WFMs to them. We are eager to verify the claims in future work.
> （五个用途全部没有实证结果——难得的诚实，也是 Day27 主题"realism vs usefulness 差距"的官方注脚。）

## 今晚产出
- [x] 按模板补齐 System / Training / Key Tricks / Results / 可迁移（含数学视角：条件视频动力学 = Dreamer RSSM 的像素空间对应物；diffusion/AR/Transfer 三条路线的统一 loss 形态；时间粒度错位与无因果监督的差距分析）
- [x] 保留并完善「和之前工作的关系」小节（vs Day04 Genie / Day05 UniSim / Day06 DreamerV3 / Day26 GR00T N1 神经轨迹 / Day25 Gato / Atlas 对照 / Cosmos-Reason1 推理侧）
- [x] 更新 reading-log.csv、README.md 进度（27/30），commit + push

## 连接
- 上一篇: Day 26 — GR00T N1（https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-26-2025-groot-n1）
- 下一篇预告: Day 28 — ManiSkill / robosuite 评测基准（统一任务、资产、传感器和成功判据，建立算法与系统的可复现实验矩阵——正好回答今天第 3 问：WFM 的"usefulness"该怎么量）
