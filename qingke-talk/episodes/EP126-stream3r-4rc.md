# EP126 — STream3R & 4RC：面向几何与运动理解的流式前馈 3D/4D 重建

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP126-stream3r-4rc.html

> 「只有把这个网格的 N 乘 N 个网格全部都给重建出来，我们才可以把它叫做一个完整的 4D 重建。」——讲者对 4D 重建的定义（字幕 33:40–33:47）

## 元信息

- 期号：126
- 标题：STream3R & 4RC：面向几何与运动理解的流式前馈 3D/4D 重建
- BV：BV11gGt6pEKw
- 时长：01:31:20（讲授 STream3R 约 21 分钟 + 中场 Q&A 约 10 分钟 + 4RC 讲授约 26 分钟 + 未来方向与 Q&A 约 30 分钟；直播约 27–30 分钟处有一次推流中断，字幕在该段有缺损）
- 提炼日期：2026-10-02
- 分享嘉宾：罗一航（南洋理工大学 NTU MMLab 博士生，导师 Chen Change Loy、Xingang Pan；姓名据字幕开场自述订正，论文作者页英文名为 Yihang Luo）
- 相关论文：STream3R https://arxiv.org/abs/2508.10893 ；4RC https://arxiv.org/abs/2602.10094
- 相关代码：STream3R https://nirvanalan.github.io/projects/stream3r ；4RC https://yihangluo.com/projects/4RC/
- B站链接：https://www.bilibili.com/video/BV11gGt6pEKw/
- 官网预告：https://qingkeai.online/blog/STream3R%264RC
- 字幕原文存档：本地 `transcripts/EP126.txt`（1633 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP126，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 CUT3R 在字幕中作「cutter/puter」、VGGT 作「VGT/VHT」、Waymo 作「VIVO」、Kubric 作「Could break」，均以论文与公开资料为准）。两篇论文（arXiv:2508.10893、arXiv:2602.10094）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。本期官网与 B站同为 126 期，无编号冲突（此前版本误标为官网-only，特此订正）。

## 一句话总结

3D/4D 重建正从"全局优化"转向"前馈预测"：STream3R 把多视图 pointmap 预测重构成 decoder-only Transformer 的问题，用因果注意力像处理语言流一样流式处理图像序列；4RC 再进一步，把整段视频一次编码为紧凑的时空潜变量，之后可在任意时刻、任意视角条件查询 3D 几何与运动——"encode-once, query-anywhere and anytime"（论文口径概括，见文末金句）。

## 核心

### 背景：从 COLMAP 的全局优化到 feed-forward 重建

讲者先回顾 3D 重建的定义（多视图图片或单目视频 → 稠密几何，输出可为 point map、深度图、相机位姿），并对比两条路线：

- **传统路线（COLMAP 为代表）**：先找 correspondence，再对整个系统做 bundle adjustment 式循环优化求最优解。缺陷讲者列了三条：多 stage 很耗时、对每个场景单独优化不 scalable、系统复杂。
- **feed-forward 路线**：文本、图像、音频都已被统一为「encoder 出 tokens → Transformer 处理」的范式，3D 重建也一样——VGGT（CVPR 2025 best paper）把多视图图片直接送 Transformer 输出 point map / depth map。但 VGGT 只能离线处理一组固定长度图片，处理不了 streaming。

### STream3R：因果注意力的流式 3D 重建

**Motivation**：具身智能（机器人头戴相机逐帧感知）、AR/VR 实时生成都需要一帧一帧输入、一帧一帧输出的 streaming 重建。已有的 streaming 方案 CUT3R 走 RNN 路线，用 fixed-size latent 存全部历史，scalable 不如 Transformer 且只能逐帧训练。

**方法**（字幕口径）：每张图片过 ViT encoder 得到 tokens 存入 KV cache；新帧 tokenize 后用 causal attention 去 attend 之前的 KV cache，逐帧输出点云、深度、camera pose。直接借用 LLM 的一整套技术（causal attention、KV cache）。训练时用 causal attention mask 做并行训练（不必像推理一样逐帧回传梯度），收敛速度与每步训练速度都优于 CUT3R，尤其是带 camera 的 global branch（字幕口径）。

**变体与实验**（均为字幕口径，讲者未在 talk 中逐项给数值表）：

- 两个版本： $\alpha$ 版在 DUSt3R 上 fine-tune、 $\beta$ 版在 VGGT 上 fine-tune，后者结果明显更好； $\alpha$ 版支持 metric scale（深度带真实单位）。
- **Window attention 变体（W5）**：纯 causal attention 要 attend 全部历史帧，内存随帧数增长；window 版每帧只 attend 最近 5 帧 + 永远保留第一帧作 anchor。讲者的 insight 是 window 版反而去掉了冗余 token（相邻帧 RGB 大量 overlap），在很多指标上与纯 streaming 版相当甚至更好。
- 速度：window 版快于 CUT3R，讲者给出 FPS 32.93（分辨率约 518×360，字幕口径）；纯 causal 的 $\beta$ 版速度不及 RNN 方案。
- 在 video depth、3D reconstruction、camera pose 三类任务上，STream3R 在 streaming 方法中达到 SOTA（字幕口径）；动态场景只需在训练集加入动态数据即可支持，无需改网络结构。
- 训练规模（字幕口径）：STream3R 用 20 多个 3D 数据集（保存量级十几 T 到 20 TB）； $\alpha$ 版在 DUSt3R 上 fine-tune 约 7 天、 $\beta$ 版在 VGGT 上约 3–4 天，均为 8 卡级别。

**局限**（讲者自陈）：streaming 到后面发现早期某帧的 pointmap 或 camera pose 估错了，没法回头纠正——这是这一系列 streaming 方法的共同缺陷。可行的补法是把 streaming 重建作前端、接 SLAM 类 optimization-based 后端在线修正。

### 中场 Q&A 要点（字幕口径）

- **为什么用 causal attention**：full attention 解决不了 streaming；causal attention 本就是 masked attention 的一种（把未来帧全部 mask 掉），且天然可退化为 window attention。讲者特别说明 window 版没有做任何 fine-tune，只是推理时改了 attention mask。
- **window size 怎么选**：推理时可调，试过 3/5/7/9；在常见的 100–200 帧输入下 5 最合适；理想值应与输入长度和帧间 gap 相关。纯 streaming 在 100 帧左右累积误差尚可控。
- **camera token 与 register token**：点云输出需要坐标系原点，本工作以第一帧相机位置为原点，register token 的作用就是把第一帧「区别出来」告诉网络坐标系在哪；camera token 与每帧 feature 一起做 attention，最后经 camera head 解出相机参数（9 个数字：旋转、平移、FOV）。

### 4RC：什么是「完整的」4D 重建

讲者用一个 N×N 网格定义三种任务：纵轴是静态 3D 重建（第 $i$ 帧输出第 $i$ 帧点云）、对角线是动态 3D 重建（每帧输出其对应时刻的点云，STream3R 做的就是这条线）、横轴是 3D tracking（第 $i$ 帧的点在所有时刻的位置）。只有把整个 N×N 网格填满——同时恢复几何与每个点的运动（tracking / scene flow）——才叫完整的 4D 重建。此前方法的不足：VGGT/STream3R 一系只出几何不出运动；专门做 3D tracking 的方法多是 multi-stage（correspondence → cost volume → iterative update），inductive bias 重且不是单个 Transformer 能 feed-forward 解决的；双视图 scene flow 方法看不到多视图全貌、做不了长序列 motion 建模。

### 4RC 方法：一次编码 + 轻量 motion decoder

- **编码**（字幕口径）：沿用 VGGT 范式，multi-view 图片送 ViT encoder（全帧 full attention）得到每帧一组高维 latent。讲者强调主要计算都在 encoder 完成——每一帧在编码时已能看到所有其他帧，先学到对物体运动的 implicit 理解。
- **几何解码**：与 STream3R 一样用 depth head、camera head 解出每帧几何。
- **motion 解码**（字幕口径）：motion 需要 query 帧与 target 帧的 pair 信息，故设计了一个非常轻量化的 decoder，由 self-attention 与 cross-attention 两部分组成——cross-attention 让 query 帧 feature 去 attend target 帧 feature（显式看到目标帧长什么样）；self-attention 中每帧带一个 learnable 的 time token 表示该帧的时间信息，target 时间经 adaptive layer normalization（沿用 diffusion 的注入范式）传入。
- **输出表示的极简分解**：motion 不直接输出点云，而是输出相对位移（displacement / delta）——几何 head 已给出 query 时刻的 base 点云，只需再学「这个点相对 query 时刻移动了多少」，base 点云 + 位移就是 tracking 结果。这种 base geometry + relative motion 的分解把 motion 与几何解耦：背景静止点的位移天然为零，网络更好学。消融（字幕口径）显示：去掉 cross-attention，轨迹只剩粗轮廓（如跳跃的人腿部位置跟踪不到）；把 motion 直接输出成 points，背景会出现大量 floating points 和杂乱轨迹。
- 4RC 同样可以只跑几何 head 做静态 3D 重建；讲者还展示了在 STream3R 上 fine-tune 出的 streaming 4RC 版本，可接收更多帧做流式 4D 重建。

### 4RC 的训练与 loss（字幕口径）

- **数据**：共 7 个数据集，以带 motion ground truth 的动态数据为主——PointOdyssey、Dynamic Replica（游戏引擎渲染；两者场景数合计仅四位数级别）、Kubric（Blender 渲染物体碰撞，约 1 万条）、Waymo（自驾真实数据，利用车辆刚体运动由旋转/平移推得每个点的 motion GT），再加静态数据集（motion 为零）保泛化。训练时从长序列中采样 2–18 帧；8 卡 fine-tune 约 2–3 天。
- **Loss**：depth 用 L1 + spatial gradient loss（对深度图求空间梯度作监督，输出边缘更 sharp）；motion 在 displacement 上算 L1，另加 temporal gradient loss——相邻时刻位移之差即速度，等于对速度再加一层监督，轨迹更顺。
- **Confidence loss**（讲者花了较多篇幅）：网络同时输出与 loss map 同尺寸的 confidence map（表示对每个点估计的置信度），总 loss 形如 confidence 与 L1 的乘积再减去一个 confidence 相关的常数项——估计差值大的点会被压低置信度、天空等远点自动被放弃，网络优先学好近处易学的点。confidence 没有 ground truth、完全自监督学出，下游可用它 filter 高置信度点。讲者也说明这不是他们首创，前作多有使用。
- **ray 与 camera 双监督**：相机有两种表示——camera token 经 MLP 输出 9 个参数（3 个平移 + 4 个旋转四元数 + 2 个 FOV），或表示成 H×W 的 ray map（每像素一条射线）；ray map 更好学、精度更高但需后处理换算回 9 参数，本工作两个都输出、双监督。

### 实验结论（字幕口径）

在 dense tracking（每像素都输出 motion）与 sparse point tracking 两套评测上，4RC 在同期方法中表现最好；传统 sparse query 方法（如 CoTracker 一系）每次最多处理几百个点、做不了 dense tracking。可视化对比中，双视图方法（如 St4RTrek，DUSt3R 的 4D 版本）在遮挡场景（如摩托车被花盆遮挡）会困惑，多视图方法在跟踪精度与连续性上都更好。

### 近期工作与未来方向（字幕口径，均为讲者口头提及）

- **OmniVGGT**（字幕作「OMINAVHGT」）：把 camera、depth 等已知量也可以作为输入注入网络（如球场 20 个固定机位已知位姿，只需学深度）。
- **S-FoRD 数据集**（字幕作「SFORD」）：用虚幻引擎渲染、带 motion 与几何 ground truth 且每个时刻多视角拍摄的数据，支撑把数据量 scale 上去。
- 领域关注点：token compression（新版 VGGT Omega 用少量 context token 表示一帧特征，大幅减少 attention 的 token 数）、test-time training 处理大场景长序列、non-pixel-aligned 输出（不再逐像素对齐输出点云，可补全被遮挡信息）、geometric grounded world modeling（把几何/三维一致性带进 world model）。

### 结尾 Q&A 要点（字幕口径）

- **time token 为什么用 adaptive layer norm 注入**：time token 是 learnable 的 implicit token，在 self-attention 中学习该帧处于哪个时间点，并非显式 positional embedding；注入方式沿用 diffusion 范式，cross-attention 则专职看 target 帧 latent，两者分工。
- **query 帧与 target 帧可以隔多远**：无硬限制。动态物体若已不在 target 帧中，motion 本身有歧义，不在讨论范围；只要还在，cross-attention 就能取到信息，且 encoder 阶段已全局交互，decoder 只是把已有信息轻量解出。同帧作 query 与 target 时输出位移为零，训练中也包含这种 pair。
- **loss 项多会不会更难收敛**：讲者给了两层回答——多任务输出之间有信息重叠（point map 本就含 depth 信息），互相帮助；gradient loss、confidence loss 这类附加项目的是让目标更准，经验上反而更好收敛。系数不必精调，几个 loss 系数都设为 1 都能收敛到不错的结果，只要 scale 别相差几个数量级。
- **没有 cross-attention 为什么还能部分 work**：主要计算在前面的大 Transformer encoder 里已完成，每帧 feature 在编码时就已学到该点大概如何运动，decoder 只做轻量补充。
- **基础点云从哪来**：由 4RC 自己的 geometric head 输出，motion decoder 只出位移，两者相加才是最终位置。
- **微小场景（如工业电路板）是否适用**：与训练分布 domain gap 太大时泛化不到；范式的好处是统一框架下有足够数据就能 scale 过去。
- **少算力方向**：training-free 的 token compression（按 attention weight 保留重要 token，已有工作以 STream3R/VGGT 为 backbone 做）、拿现成几何 backbone 做结构微调扩展到 streaming/更长场景——不必都从大规模预训练做起。

## 关键数字

以下均为字幕口径（讲者在 talk 中口头给出，论文表格数值以论文为准）。

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| STream3R window 版推理速度 | CUT3R 对比 | FPS 32.93（分辨率约 518×360） | 字幕（实验部分） |
| Window attention 窗口大小 | 试过 3/5/7/9 | 100–200 帧输入下 5 最合适 | 字幕（Q&A） |
| STream3R 训练数据规模 | — | 20 多个数据集，十几 T 到 20 TB | 字幕（Q&A） |
| STream3R fine-tune 时长 | 8 卡级别 | $\alpha$ 版（DUSt3R 基底）约 7 天； $\beta$ 版（VGGT 基底）约 3–4 天 | 字幕（Q&A） |
| 4RC 训练数据 | 7 个数据集 | Kubric 约 1 万条；PointOdyssey + Dynamic Replica 合计四位数场景 | 字幕（训练部分） |
| 4RC 训练设置 | 采样 2–18 帧 | 8 卡 fine-tune 约 2–3 天 | 字幕（Q&A） |
| 相机参数表示 | — | 9 个数字（3 平移 + 4 四元数 + 2 FOV） | 字幕（Q&A） |

论文口径补充：两篇论文的摘要未给出可直接引用的基准分数，本文不转引二手数字；具体指标（重建误差、轨迹精度等）以论文实验表格为准。

| 工作 | 核心机制 | 输出 |
|---|---|---|
| STream3R | 因果注意力流式配准（decoder-only） | 在线 pointmap / 相机与几何 |
| 4RC | 一次编码 + 条件查询（时空潜空间） | 任意时刻稠密几何 + 运动 |

## 可迁移

- Infra 视角：「把视觉序列当语言流处理」是可迁移的系统范式——causal attention + KV 缓存式增量推理让在线 3D 感知复用 LLM 的整套 serving 基础设施（与具身/机器人感知的实时性需求直接相关）；window 版「删冗余 token 反而更好」与 LLM 侧的 KV cache 压缩/驱逐是同一类问题。
- 对 post-training 的旁注：4RC 的「base + relative motion」因子分解与 RL 里的 baseline/advantage 分解同构——先学静态结构、再学时间残差，比端到端硬学联合分布更容易优化。

## 疑问 / 下一步

- STream3R 在超长序列下因果注意力的误差累积（drift）如何控制？讲者给的答案是 window 版在 100–200 帧量级可用、更长序列可接 SLAM 后端修正；官网提纲第 4 点「未来方向」与字幕提到的 token compression、test-time training 是同一问题的后续，值得回论文与后续工作确认。
- 4RC 的条件查询在任意时间戳的插值质量与训练帧率的依赖关系，值得回论文实验节确认。
- 直播约 27–30 分钟处推流中断，该段字幕缺损严重（疑似错过了 STream3R 评测的一部分讲解），本纪要该段以讲者后文复述为准，细节待回论文核对。

## 原文金句（1-2句）

> 「只有把这个网格的 N 乘 N 个网格全部都给重建出来，我们才可以把它叫做一个完整的 4D 重建。」——讲者对 4RC 任务的定义（字幕 33:40–33:47）

> "encode-once, query-anywhere and anytime" —— 4RC 论文对自身范式的一句概括（摘要原文，非视频原话）。
