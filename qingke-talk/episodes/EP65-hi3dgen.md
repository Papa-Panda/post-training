# EP65 — Hi3DGen：法线为桥，为高清三维几何生成另辟蹊径

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP65-hi3dgen.html

> 「我们当时还是希望说能把 SDS 再往前做一做，能不能 make SDS great again——但没做神。」——讲者复盘 DreamVerse 数据尝试时的判词

## 元信息

- 期号：65
- 标题：Hi3DGen：法线为桥，为高清三维几何生成另辟蹊径
- BV：BV1qk4QzfEv4
- 时长：00:56:45（讲授约 45 分钟 + Q&A 约 8 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：叶崇杰（香港中文大学（深圳）；主持人介绍作叶崇杰，讲者是 Stable Normal、Hi3DGen 等工作的作者；字幕中其姓名多处作「叶纯洁/叶文」等，均系识别误差）
- 相关论文：Hi3DGen（ICCV 2025；讲者口头提及，论文编号以公开资料为准，本纪要不挂 arXiv 链接）
- 相关代码：已开源（讲者 Q&A 中确认可本地部署，GitHub 搜 Hi3DGen 可得部署包；带纹理版本讲者称「下个月会放出来」）
- B站链接：https://www.bilibili.com/video/BV1qk4QzfEv4/
- 字幕原文存档：本地 `transcripts/EP65.txt`（1188 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP65，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 TripoSG 字幕作「区块SG」、TRELLIS 字幕作「区列斯/序列斯」、Clay 字幕作「CLAY/接阅」、Spar3D 字幕作「spark」，均以公开资料为准）。凡涉及具体数字，均为讲者口头口径，未与论文版逐项核对，以下关键数字总表统一标注来源「字幕」。

## 一句话总结

Hi3DGen 不直接从图片生成 3D，而是以法线图（normal）为桥梁分而治之：先把 image→normal 做好，再做 normal enhancement，最后用 normal→3D 把几何生成出来；配套的数据工作 DetailVerse 通过蒸馏已有 3D 生成模型 + 人工打标 + 分类器筛出高质量资产。讲者更坦白地复盘了这条工作线之前的两次「梭哈」失败：Stable Normal 的 scale-up 尝试因为没有做对比实验而「没有学到任何 lesson」，以及 SDS 造数据（DreamVerse）被判定「已经不是 SDS 的时代」。

## 核心

### 领域判断：3D 生成的主线就是 scaling，人工设计退居其次

讲者先给领域全景。工业界（混元、TripoSG、StepFun 一类）在 vecset 路线上堆数据、堆 MoE 等在 LLM 下验证有效的架构、做 3D 大模型 scale up；学术界在探索表征与训练范式的上限。法线估计方向今年从 image→normal 往 video diffusion 靠，追求多视角一致性，并讨论 normal 作为中间表征对 relighting 等下游任务的应用。

他的核心判断是：**数据量与参数量是最重要的，人工设计反而没那么重要**——人工设计的价值在于「怎么用好大量数据、怎么分阶段分步骤训好大模型、怎么做对齐让多视角一致性更好」。同时他点出开源数据的天花板：接月（StepFun 系工作）之后，混元 2.5、Spar3D 的出现让他明显感到「一个坎」——分辨率不够高、开源数据越来越成为整个领域往前推的瓶颈。他特别提到 TRELLIS 团队把清洗数据与全套清洗/渲染流程开源，「使得我们后面再做很多 research 工作都受到很大的利好」。

对分辨率的判断：现在 sparse voxel（Spar3D 一系）表征还有大量冗余，「这样的冗余一定是能通过某种方式解决掉的」，分辨率从 1536 往 3000、甚至 8K 推是可能的。

### 失败复盘一：Stable Normal 的 scale-up，与「一把梭哈」的教训

Stable Normal 在室内场景不错，但在室外、动态数据（如 Sora 的视频）上效果不好，且在 OOD 数据上有 over-sharp 的假细节（artifacts）。讲者的归因是训练与选 checkpoint 全都在室内数据上做。

他的改进计划本来很合理：收集网上所有 normal 数据集（室内、室外、物体、人物，特意补了强反光/透明物体等 corner case），训一个能像 Depth Anything 一样打伪标签、再 scale up 的法线模型。但训到一半发现模型虽然稳定、多视角一致性好、泛化不差，却非常 over-smooth，细节不如 regression-based 方法。

最值得记的一段是他对这段经历的复盘（字幕口径）：

> 「我们一把梭哈了……因为我们没有办法做对比实验，所以我们并不知道里面做错了什么，所以我们没有学到任何 lesson。」

他猜测问题可能出在训练数据质量、训练时长，或多域分布需要更精妙的多阶段训练设计，但因为当时资源不足以支撑对比实验，这个问题至今没有答案。结论性判断有两个：一是这件事「整个的范式没问题」，scaling 本身是对的；二是 normal 作为给下游重建用的 prior model，「作为一种弱约束、一种正则，其实也够了」，更好的法线模型在下游的边际价值没有那么大——这直接导致团队转向 3D 生成。

### 失败复盘二：DreamVerse——用 SDS 造数据，然后承认时代过了

转做 3D 生成时的出发点是数据清洗成本：高质量 3D 数据靠人工清理太贵、学校负担不起。他们的想法是用生成代替清理：通过 SDS（2D diffusion 蒸馏到 3D）造一批高细节数据 DreamVerse。动机有二：2D 的语义分布比 Objaverse（源自 Sketchfab，集中在游戏/影视资产）更接近「人类想象空间」；高细节数据即使质量不如人工数据，也可以作为 pretraining 阶段的补充。

跑了几十万个 case 之后的结论很硬：

- 与 text 的对应性差：大概只有 10%–20% 的 case 能跟 text 对上（字幕口径），其余要么只对上 text 里的某一个词（讲「宇航员猫」只生成猫、没有宇航员），要么直接是 artifacts；
- 细节质量也不如预期。

对 CraftMan 的训练有一定效果，但总体不理想。讲者的判词是「SDS 这个事情确实已经被淘汰了，已经不是 SDS 的时代了」——而且这批数据又是一次「把所有的卡都拿来造数据」的梭哈，那段时间基本没有产出。

### 中间路线：3D enhancement 三条路，卡在「不可控」

第三次尝试是把问题降级：3D enhancement 是 low-level 任务，不需要那么多数据（他们在 normal enhancement 上用约 5 万条数据就训出了很好的模型）。只要把精细度做高、不要求与输入完全对齐，就能探 3D 生成的上限。三条路径都试过：

1. 3D render 成 normal → normal 超分/enhancement → normal→3D；
2. 端到端学一个 flow matching，从输入 3D 直接到增强后的 3D（backbone 选 TRELLIS 一系）；
3. Retrieval/局部增强：proposal net 给出 bounding box + 语义（如「盔甲」），从图片/3D 数据库 retrieve 参考（normal map、精细几何），学 style transfer 式的局部增强。

三条路共同的死因是**不可控**：增强过头或不到位，增强出来的细节很多是 artifacts 而非真实细节。要做可控增强又需要大量 paired 数据——绕回数据问题。讲者说在这个事情上「卡了很久」。

### Hi3DGen：三件没做好的事，缝成一个完整故事

Hi3DGen 就是把三条线拼起来：更好的 normal estimation 模型 + 3D 数据 scale up + 以 normal 为 bridge 的细粒度几何生成。讲者非常坦白：「它其实并没有带来什么开创性的贡献」，「内部一直戏称它是缝了三篇 paper」。方法论本身只有一句：**不直接从图片到 3D，用 normal 分而治之**——把 image→normal 做好、normal 训练时加 online regularization、normal→3D 收敛更好，再配一个数据集把两段都喂好。

数据侧是 DetailVerse。做 Hi3DGen 时团队卡在数据上：用开源数据训不出高细节生成模型。他们的做法是系统化工程：text/image 一侧清洗（从公开数据筛单物体、做 prompt engineering 让 FLUX 一类模型生成更 CG 风格的图片），关键经验是**输入图片是 CG 风格、等距（isometric）视角时，喂给 TRELLIS 的生成成功率更高**，因为 isometric 提供的信息量足够多。然后对生成结果做人工打标 + 训练 classifier，筛出高质量资产，用于训 image→normal 与 normal→3D。

全场最有信息量的一个 insight 是：**为什么用 TRELLIS 蒸馏出的数据训出的模型能比 TRELLIS 本身更好？**讲者的解释是：image→3D 是一对多问题，TRELLIS 有把细节表达到纹理而非几何上的倾向（其 VAE 用 Gaussian Splatting 做 pretraining、先训 GS 再冻结 encoder 训后面的 decoder，于是更倾向用高纹理细节表示高频信号）。而 normal 与 3D 是一一对应关系，用 normal→3D 训练时监督信号直接落在几何上，等于通过 normal 这个 bridge 把 TRELLIS 潜在的几何生成能力「激活」出来，让它把参数用在预测几何上。效果对比上（讲者口径），在精细度上优于混元、Clay、TRELLIS 等当时 SOTA。

### 应用：生成模型当 CAD 用，做 6D 位姿估计

Hi3DGen 之后的一个应用是 6D pose estimation：传统做法需要 CAD model 做 reference，现在单张图片输入、用生成模型现场生成一个 reference CAD 替代。主要难点在 scale 恢复（生成资产不是 metric scale），以及生成资产与图片不完全对齐时怎么把位姿做准。

### 宏观思考：三个未收敛的问题

讲者最后给了他对 3D 几何生成的框架性思考，围绕三点：

1. **Information gap**：稀疏输入下，可见部分要服从输入（他称之为「重建」），不可见部分要补全且与可见部分一致。解法方向是「大力出奇迹」：更多数据 + 更强的 inductive bias（edge map、VGGT 点云、MoGe 的 normal 都可以作为先验输入），以及更大参数量——「换个角度，卡非常重要，卡和钱非常重要」。
2. **表征的 trade-off**：保真度、灵活性（scalable 地表示粗细结构）、压缩率三者兼得的表征尚不存在标准答案；Gaussian Splatting、sparse cube、函数式表征、异形几何结构都未收敛。
3. **结构与细节的 trade-off**：本质可能不是 multi-scale conflict，而是 multi-quality 数据混训的 trade-off——细节追多了出 artifacts，结构保全了细节变差。Coarse-to-fine 已成共识之一；「粗结构 mesh + normal map 表示细节」也 feasible，可能成为下一个共识。

### Q&A 要点

- **高斯表示怎么获得**：底座用 TRELLIS，本身有 3D Gaussian head，把 decoder 从 mesh decode 换成 3D Gaussian decode 即可；当前训练数据无纹理，出来的是白模高斯。新版本把 image 与 normal 都作为 diffusion 输入，能出带颜色的 3D 高斯。
- **生成过程加正则会不会显著拖慢训练**：会慢两三倍，但这一步定位是 continuous training / SFT 量级——约 3 万步、8 卡、约一天训完，整体可接受。讲者顺带给出他的 3D 生成训练四阶段观：pretraining → continuous training → SFT → 后训练。
- **本地部署**：推理约 6 GB 显存即可，NVIDIA GPU，大部分消费级显卡可用；社区已贡献 Windows 部署包。
- **为什么不用 depth 当 bridge**：试过。depth 对表面细节的方差整体偏小，作为 condition 时模型对细节的理解不到位；normal 更适合出细节，depth 更适合出结构（可作为第一阶段表征）。
- **真实与合成数据配比**：讲者的经验认知是仿真：真实约 10:1 较好（字幕口径），但强调要服从任务、靠经验试出来。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| SDS 生成数据与 text 的对应率 | DreamVerse，跑了几十万个 case | 仅约 10%–20% 能对上 | 字幕 |
| normal enhancement 训练数据量 | 低数据量 low-level 任务设定 | 约 5 万条即可训出很好的模型 | 字幕 |
| 加正则的训练成本 | 8 卡 | 约 3 万步、约 1 天（比原版慢 2–3 倍） | 字幕 |
| Hi3DGen 推理显存 | 本地部署 | 约 6 GB（NVIDIA GPU） | 字幕 |
| 仿真与真实数据配比 | 讲者经验认知 | 仿真：真实约 10:1 较好，需按任务调参 | 字幕 |
| 3D 生成分辨率趋势 | 当前 1536 量级（讲者口径） | 往 3000、甚至 8K 推是可能的 | 字幕 |

## 可迁移

- **Bridge 表征做能力迁移**：normal↔几何的一一对应监督，把教师（TRELLIS）藏在纹理里的几何能力逼出来。这个思路可迁移到数据蒸馏：蒸馏效果不取决于教师多强，而取决于监督通道是否与目标能力一一对齐——选错通道，教师的容量会花在无关的自由度上。
- **「梭哈」实验纪律**：两次失败的共同点是资源全压、没留对比实验预算，结果是连「做错了什么」都不知道。对 RL infra 实验同样成立：先用小规模 ablation 把因果钉住，再 scale；没有对照组的 scale-up 不产生知识。
- **低数据量的 low-level 降级**：数据不够时把问题降级成 enhancement 这类 low-level 任务（5 万条量级），是绕开数据天花板的一条真实存在的路，但要提前想好「可控性」怎么做，否则会卡在不可控上。

## 疑问 / 下一步

- Hi3DGen 的 normal→3D 在 0.5 版做了数据 scale up，但讲者自述因同期混元/Spar3D 效果不够理想而未放出——纯几何数据（3D 打印数据等）scale up 这条路的上限仍待验证。
- 「粗结构 mesh + normal map 表示细节」会不会成为与 coarse-to-fine 并列的共识表征，值得跟踪后续工作。

## 原文金句（1-2句）

> 「因为我们没有办法做对比实验，所以我们并不知道里面做错了什么，所以我们没有学到任何 lesson。」

> 「换个角度，卡非常重要，卡和钱非常重要。」
