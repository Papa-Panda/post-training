# Day 31 — World Labs Atlas：首个从零训练的多模态世界模型（最新进展 1/4）

## 元信息
- Title: Atlas: A World Model for Spatial Intelligence
- Authors / Org: World Labs 团队（创始人 Fei-Fei Li；公司 2024-02 创立）
- Link / Blog: https://www.worldlabs.ai/blog/atlas（官方博客，2026-09-01 发布）
- 官方 X 公告：https://x.com/theworldlabs/status/2094839756329041984（2026-09-01）
- 发布视频：https://www.youtube.com/watch?v=hzvXRHBInx0
- Date read: 2026-09-23
- Tags: [physical-ai, world-model, atlas, spatial-intelligence, camera-control, real-to-sim, 3d-reconstruction, video-generation]
- Thread: physical-ai
- Folder: day-31-2026-worldlabs-atlas
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-31-2026-worldlabs-atlas
- 前传：本路线 README「World Labs Atlas 与 roadmap 的关系（2026-09-12）」问答——当时结论是 Atlas 落在 Day27（Cosmos）射程内、不必单独开 Day；2026-09-22 用户采纳李昊建议把路线扩到 34 天，Atlas 成为 Day31 最新进展第一篇
- 非技术背景（仅记录，不作技术证据）：公司累计融资约 \$1.23B（含 Autodesk \$200M 战略投资，NVIDIA / AMD / Fidelity 参投），报道估值约 \$5B（来源：ainvest / bestaitoolfinder 转述，PR 口径，未独立核实）；产品线 Marble 2025-11 发布、World API 2026-01 上线、2026-07 收购 SceniX（机器人空间智能方向）（来源：cryptobriefing 整理）

## 一句话总结
Atlas 是 World Labs 2026-09-01 发布的 omni 世界模型：从零预训练、原生处理 text / image / video / 3D，以相机位姿为一等输入，把「像素重建」与「像素生成」统一到 viewpoint 锚定的 spatial context 里；能输出最长 1 分钟 1440p 相机可控视频、2–3 张照片的显式 3D 重建（point cloud / 3D Gaussian splat）、以及手机视频驱动的 real-to-sim 机器人仿真数据。但它目前是面向 select partners 的 early access 公告：无公开论文、无权重、无公开 API、无独立第三方复现——所有性能数字均为公司自报，这是今天最硬的边界。

## 和之前工作的关系

- **接了哪条线**：Day04 Genie → Day27 Cosmos → Atlas，「生成式世界模型」线（负责「造」世界的模型）。Genie（2024）从无标签视频学 latent action、做可交互生成；Cosmos（Day27）是 NVIDIA 的世界基础模型工业版（diffusion / AR 双路线，做可控 world-to-world 生成）；Atlas 是 2026-09 的最新一代——多了**相机位姿原生输入**和**显式 3D 输出**（point cloud / splat），以及官方反复强调的几何持久性（"preserve the place behind the clip"：回到同一位置，几何和光照还在）。
- **补了哪个短板**：Day27 留下的核心问题是「视频 realism 与 action-grounded usefulness 的差距」。Atlas 把差距收窄了一步：输出的不再只是好看的视频，而是带显式几何（depth / point cloud / splat）+ 像素级相机可控的「可走进去的世界」，并直接对接 real-to-sim（手机 24 帧视频 → 机器人 RGB + depth 观测仿真，官方 blog 数据）。但 Day27 的硬标准仍未达到：Atlas 的世界**没有可交互的物理**——无碰撞、无接触动力学。realism → usefulness 的断层还在，只是断层变窄了。
- **替代 / 分叉 / 改进**：
  - vs Day27 Cosmos：Cosmos 是「视频生成当数据引擎」（Transfer 可控重绘失败场景）；Atlas 是「3D 一致的世界当数据引擎」——几何持久性是 Cosmos 没有的。但 Cosmos 开源了 WFM 平台，Atlas 目前 closed + early access——在可复现性这条路线价值观上是倒退。
  - vs Day06 DreamerV3：**另一条线，别混**。Dreamer 是 latent dynamics model，在隐空间里「想」（planning / imagination，给策略做梦）；Atlas 是 generative world model，用来「造」（数据 / 仿真，给训练管线供料）。前传问答（2026-09-12）已明确此区分，今天依然成立。
  - vs Day11 π₀ / Day12 Diffusion Policy（VLA 策略线）：Atlas 不输出 action、不做策略——它是 VLA 和仿真管线**上游**的世界生成器。Day33（Figure Helix 2.5）和 Day34（π0.7）会回到「策略」本身；Atlas 回答的是「策略在哪练」。
- **跨阶段连接**：Day30 数据飞轮的 resimulate 分支（GPU 小时替代真机小时）——Atlas 的 real-to-sim 正是这个分支的 2026 年最新实例；Day28 评测合同（固定测度 $\mu$ 下的二项估计）——Atlas 官方自己承认「no single benchmark fully captures its generality」，恰恰暴露了 omni 世界模型评测协议的缺失，这是 Day28 问题的下一站。

## 今天的 3 问
1. **相机位姿当原生输入 vs 写进 text prompt，差距到底在哪？** 官方评测里，基线模型只能用文字描述相机路径（"pan left"），Atlas 用精确几何输入——这到底是架构胜利还是输入格式胜利？如果给基线同样结构化的相机编码，差距会缩小多少？官方自己都承认 "more sophisticated prompt engineering…could improve camera following for some models"（官方 blog Benchmarks 小节）。这是一个**评测设计**问题，也是 post-training 里 conditioning 设计的通用问题。
2. **「重建」和「生成」的统一，在数学上到底统一了什么？** 同一个条件分布 $p_\theta(\cdot\mid\mathrm{context})$ ，输入 1 张图时先验主导（生成/想象）、输入 100+ 张时似然主导（重建/忠实）——"the more it sees, the less it imagines" 有没有可计算的形态？条件熵 $H(\hat{W}\mid I_{1:N})$ 随 $N$ 如何衰减？这是今天数学视角的核心。
3. **real-to-sim 的证据链缺了哪一环？** 官方展示了「手机视频 → 3D 重建 → 机器人视角 RGB + depth 渲染」（官方 blog Robotics Simulation 小节，24 帧手机视频重建大场景），但**没有展示**：在这个重建世界里训出的策略，放到真机上成功率多少（sim2real transfer 数字）。没有这一环，Atlas 只是 Day27 差距的「更接近」，不是「跨过」。Day33 / Day34 的策略侧数字会是很好的对照。

## 核心

### 1. Motivation：为什么相机几何必须是一等输入
- 经典 CV 把「重建」（reconstruction：多视角精确恢复几何）和「生成」（generation：从无到有创造像素）当成两个学科、两套博士论文。视频生成模型（Sora 类）用 text prompt 控制相机——"pan left" 这种粗糙指令无法精确指定视角，更无法保证多视角几何一致。
- Atlas 的动机：**把相机几何变成原生输入类型**（"going beyond coarse text-based instructions for camera control"，官方 blog），让「导演摆机位」替代「prompt 抽奖」——"you are staging the scene, not pulling the lever of a slot machine"。
- 和 Physical AGI 的关系：Day01 定义的 Physical AGI 需要「理解物理世界」的 substrate。Atlas 的定位是空间智能的 substrate——不是策略本身，而是策略训练/评测用的**可控世界供应方**（real-to-sim）。Fei-Fei Li 在 a16z 访谈中的表述（媒体转述）："It's the first time we have a unification of pixel generation and pixel reconstruction in the world of computer vision… by anchoring on viewpoints."
- 为什么从零训练：官方强调 pretrained from scratch——多模态（text / image / video / 3D）在统一 spatial context 下联合训练，而不是拿视频模型后期嫁接 3D 头。这是 omni 路线的架构宣言。

### 2. 机制：multimodal autoregressive diffusion transformer
官方技术描述拆成四个词：
- **Multimodal**：原生处理 text、images、camera poses、3D depth maps；video 表示为 image 序列。每张 image / depth map 都带显式相机位姿——「每张图被锚定在 3D 空间的一个位置」，组成 **spatial context**（空间上下文）。"Similar to an LLM, Atlas first encodes its inputs into a context, then generates outputs conditioned on the context. However, Atlas is unique because each image is grounded at a 3D position in space."
- **Autoregressive**：像 LLM 一样逐个生成序列元素；**每个任务只是不同的序列形态**（inputs 后跟 outputs）。这让它能复用 LLM 侧的 serving 优化：KV-caching、cache-aware routing、disaggregated serving——这是 infra 视角最值得抄的一句：世界模型和 LLM 可以**共享 serving 栈**。
- **Diffusion**：rectified flow 模型，渐进式去噪生成高维连续数据（图像/视频）；推理时用 denoising steps 数 trade-off 速度/质量；可用 diffusion distillation、classifier-free guidance、shifted noise schedules、VAE 改进。
- **Transformer**：大矩阵乘主导，适配现代硬件；"a blend of ideas from modern LLMs and video models"。
- **Scale 信仰**：训练了一系列增大规模/算力的模型，"each new level of compute unlocked new model capabilities"——**无曲线公开**，当口号听。

### 3. 四个能力
1. **Camera-Controlled Generation**（官方 blog）：1–6 张参考图 + 手工设计的相机路径 → 最长 1 分钟 1440p 视频，几何一致、可从任意角度看。
2. **Spatial Reconstruction**（官方 blog）：1 到几十张输入；2–3 张即达 faithful 重建（自称超过专为 3D 重建训练的 SOTA）；spatial context 可容纳 100+ 张。demo：单张地面照片 → 空中视角；Stanford Main Quad 用 2–25 张地面图生成校园空中飞行路径。
3. **Space-Time Simulation**（官方 blog）：3–5 个普通手机拍的视频 → "bullet time" 多视角定格（reframing）；**Robotics Simulation**：手机 24 帧视频重建两个大场景 → 模拟不同机器人在其中导航，Atlas 生成机载相机视角的 RGB + depth；manipulation 侧：少量随意录制 → 重建含刚体/铰接/可变形物体交互的仿真，可换物体/位置/光照/背景批量造数据——"diverse training data and testing environments for robotics at scale"。
4. **Image Generation**（官方 blog）：text → 360° 全景、复杂 prompt、文字渲染（非主业，顺带能力）。

### 4. 官方评测 vs 独立视角
- Camera-conditioned generation：单张输入图 + 1–3 个电影级相机运动（pan / truck / crane）；基线用 text 描述相机路径（Atlas 用原生相机输入）；第三方 human rater 判谁更贴合相机路径 → Atlas 胜，且轨迹越复杂优势越大。
- 3D reconstruction：稀疏视角输入 + 相机位姿 → 逐像素预测 3D 点；在多个 SOTA benchmark 上复现基线、统一协议 → 自称超过最佳开源专用重建模型。
- 相机控制盲测胜率 75%–94%（对手 MiniMax H3、Gemini Omni Flash、FLUX 3、Seedance 2.5；来源：BigGo Finance / adityas-tech-report 转述）；稀疏视角重建误差 25.3‰（对手 Pi3X、Depth Anything 3；来源：BigGo Finance 转述）。**这些数字没有独立第三方复现**，按"独立评测"标准只能算半独立（rater 第三方、协议公司设计）。
- SiliconANGLE：真正的考验在更广泛可用之后；公司未公布 GA 时间，early access 仅面向 select enterprises。Startup Fortune：真正的 claim 不是"又一个视频 demo"，而是"可重访、可使用的 3D 世界"——要 Atlas 成为 video / 设计 / 机器人训练之下的空间层。thebrief.news《World-Model Startups Are Raising Billions While Saying Little》：信息披露不足——数据、传感器、物理约束、测试方法、算力预算全未公开；数据供应商 Physicl CEO 称不清楚自己的数据被如何使用。

## 数学视角

记底层 3D 世界为 $W$ （几何 + 外观 + 随时间的演化）。Atlas 不直接建模 $W$ 的物理动力学，它建模的是**从 viewpoint 锚定的观测中生成一致世界**的条件分布。

- **Observation（输入）**： $I=\{(I_i,P_i)\}_{i=1}^{N}$ ，其中图像 $I_i\in\mathbb{R}^{H\times W\times 3}$ ， $P_i=(R_i,t_i,K_i)$ 为显式相机位姿（ $R_i\in SO(3)$ 为旋转， $t_i\in\mathbb{R}^3$ 为平移， $K_i$ 为内参）；depth 图 $D_i\in\mathbb{R}^{H\times W}$ 同样带位姿。其中 $N=1$ 到 $100+$ 可变—— $N$ 是今天最重要的超参数。
- **Action / 干预（输入侧）**：相机轨迹 $C=(P_1,\dots,P_T)$ 。这是 Atlas 与 Sora 类模型的本质区别：相机是**原生几何输入**，不是 text prompt 里的模糊动词。real-to-sim 里，机器人运动指令 $a_t$ 决定机载相机位姿 $P_t$ ，Atlas 渲染传感器观测 $o_t=(I_t,D_t)$ ——注意这里的"action→observation"映射是**渲染**，不是物理仿真。
- **State（隐式）**：spatial context $S=\{(e_i,P_i)\}$ ，其中 $e_i$ 为编码后 token，全部被 $P_i$ 锚定在 3D 空间。生成即 $p_\theta(x_{k+1}\mid x_{\le k},S)$ ，并要求 3D 一致性：对已观测位姿 $P$ ，投影约束 $\pi(\hat{x};P)\approx I$ 。
- **输出**：新视角帧 $\{\hat{I}_j\}$ ；显式 3D：点云 $\mathcal{P}\subset\mathbb{R}^3$ ，或 3D Gaussian splat $\mathcal{G}=\{(\mu_j,\Sigma_j,c_j,\alpha_j)\}_{j=1}^{M}$ （ $\mu$ 位置、 $\Sigma$ 协方差、 $c$ 颜色、 $\alpha$ 不透明度），可直接进 Unity / Unreal / Blender / Spark。
- **生成动力学**：rectified flow，即 $dx_t=v_\theta(x_t,t)\,dt$ ， $t\in[0,1]$ 从噪声流向数据；推理步数 $K$ 控制速度–质量 trade-off（ $K$ 越大质量越高、延迟越大）。
- **重建–生成的统一**：记 $\hat{W}$ 为模型"心中"的世界。重建与生成是同一个条件分布的两种采样区间：

$$p_\theta(\hat{W}\mid I_{1:N}),\qquad N=1,2,\dots,100+$$

$N$ 很小时先验 $p(W)$ 主导——生成、想象； $N$ 增大时似然主导——重建、忠实。"the more it sees, the less it imagines" 即条件熵 $H(\hat{W}\mid I_{1:N})$ 随 $N$ 单调递减。这是 Atlas 最漂亮的数学陈述，也是 Day27「realism vs usefulness」差距的量化入口：usefulness 要求的是 $N$ 很大时的**忠实**（sim 忠于 real），而 Atlas 的 demo 大多停在 $N$ 很小的**想象**区。
- **时间尺度**：输出视频最长 1 分钟（1440p，官方 blog）；real-to-sim 用 24 帧手机视频做离线重建——注意这是**离线重建**，不是实时闭环渲染（latency 未公布，无法评估 Day30 的 $T_{99}$ 约束）。
- **缺失的方程**：Atlas 不建模接触动力学 $s_{t+1}=f(s_t,a_t)$ 与碰撞约束——这正是它与「可交互仿真器」（Day02 MuJoCo / Day03 Isaac Lab）的本质区别，也是 Day27 差距的数学形态： $\mathrm{render}(W,P)$ 有了， $f$ 还没有。

## 可迁移到 post-training

1. **结构化 conditioning > prompt 工程**：Atlas 评测里最诚实的一课——基线用文字描述相机路径，Atlas 用原生几何输入，差距巨大。平移到 post-training：凡是能结构化的控制信号（工具 schema、相机/位姿、reward 结构），都不要塞进自然语言 prompt 里赌模型"听懂"。conditioning 的信息论位置决定上限。
2. **"每个任务只是不同的序列形态"**：Atlas 把重建/生成/仿真统一成"inputs 后跟 outputs"的自回归序列——这正是 LLM post-training（SFT / RL）的序列 framing。启发：多模态 post-training 的 infra 可以和 LLM 共享（KV-cache、disaggregated serving，官方明确点名），世界模型不是另一套 infra。
3. **评测设计的诚实模板**：官方承认「no single benchmark fully captures its generality」，于是只报两个可测的任务（camera-conditioned generation、3D reconstruction），且公开承认基线可能因 prompt 工程不足而吃亏。这种"报什么、没报什么、为什么没报"的三段式，是 omni / generalist 模型（Day25 Gato 同理）评测写作的范本。
4. **Day30 飞轮的 resimulate 实例**：Atlas 的 real-to-sim（换物体/位置/光照/背景批量造数据）就是 Day30「GPU 小时替代真机小时」的 2026 年最新形态。post-training 数据飞轮里同理：合成数据引擎的价值不在"好看"，在**失败场景的可控复现**。
5. **第一性原理的对齐**：Fei-Fei Li 的"anchoring on viewpoints"统一重建与生成——对应 post-training 里反复出现的主题：**找到对的锚**（reward 的可验证信号、评测的固定测度 $\mu$ 、这里的相机位姿），问题就从"玄学"变成"工程"。

## 疑问

- 没看懂的 1 个问题：spatial context 里 $100+$ 张图的 attention 怎么做—— $N$ 增大时 context 长度爆炸，官方未提任何稀疏化/压缩机制（对比 LLM 的 context compression）。"the more it sees" 的**算力账**没算。
- 演示选择偏差：1 分钟 1440p 视频用的是"hand-designed camera path"（官方 blog 原话）——路径是手工挑的。相机轨迹越复杂优势越大（官方评测结论），那**最难的轨迹**是什么？没报。
- Real-to-sim 的 transfer 数字缺失（见 Q3）：这是今天唯一真正重要、却没人回答的问题。

## 下一步

- 等 early access / Marble 集成后，用真实场景测一次「手机视频 → 重建 → 视角渲染」的几何保真度（对照 Day28 的评测合同思想：固定测度 $\mu$ ，二项估计）。
- Day32（GPT-6 Astra，computer-use）是横向对照：Atlas 是"空间智能"的世界模型，Astra 是"agentic"的世界操作者——两个"世界"定义的碰撞。
- Day33（Figure Helix 2.5）/ Day34（π0.7）回到策略侧：届时用"策略在哪练"（Atlas）vs"策略本身"（Helix / π0.7）做一次上下游串联。

## 速查表

| # | 条目 | 内容 |
|---|------|------|
| 1 | 发布 | 2026-09-01，World Labs 官方 blog + X（@theworldlabs） |
| 2 | 一句话 | 首个从零训练的多模态世界模型：text / image / video / 3D，相机位姿为一等输入 |
| 3 | 架构 | multimodal autoregressive diffusion transformer（rectified flow 去噪） |
| 4 | 核心创新 | 像素重建 × 像素生成统一，以 viewpoint 为锚（spatial context） |
| 5 | 视频 | 最长 1 分钟，1440p，像素级相机控制（官方 blog） |
| 6 | 重建 | 2–3 张照片即 faithful；context 可容纳 100+ 张（官方 blog） |
| 7 | 显式 3D | point cloud / 3D Gaussian splat，进 Unity / Unreal / Blender / Spark |
| 8 | Bullet time | 3–5 部手机视频 → 多视角定格重构（官方 blog） |
| 9 | Real-to-sim | 24 帧手机视频重建大场景 → 机器人 RGB + depth 观测仿真（官方 blog） |
| 10 | 公司评测 | 相机路径跟随盲测胜 75%–94%（对手 MiniMax H3 / Gemini Omni Flash 等，媒体转述） |
| 11 | 公司评测 | 稀疏视角重建误差 25.3‰（对手 Pi3X / Depth Anything 3，媒体转述） |
| 12 | 可用性 | select partners early access；无 GA 时间；无公开论文/权重/API |
| 13 | 路线位置 | Day04 Genie → Day27 Cosmos → Atlas（生成式世界模型线） |
| 14 | 对照 | Day06 DreamerV3 是另一条线（隐空间 planning），别混 |
| 15 | 断层 | 有 render、无物理（无碰撞/接触动力学）——Day27 差距仍在 |
| 16 | 数学 | $p_\theta(\hat{W}\mid I_{1:N})$ ； $H(\hat{W}\mid I_{1:N})$ 随 $N$ 递减 |
| 17 | 迁移 | 结构化 conditioning > prompt；与 LLM 共享 serving 栈 |

## 连接
- 上一篇：Day 30 — Physical AI Eval + Data Flywheel（https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-30-physical-ai-eval-data-flywheel）
- 前传：README「World Labs Atlas 与 roadmap 的关系（2026-09-12）」问答——Atlas 落在 Day27 射程内的"新一代对照组"
- 下一篇预告：Day 32 — OpenAI GPT-6 Astra（2026-09-03 发布，computer-use 旗舰，"AGI era"；2026-09-24）
