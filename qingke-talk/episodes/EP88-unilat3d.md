# EP88 — UniLat3D：几何–外观统一VAE的单阶段 3D 生成框架
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP88-unilat3d.html

> 「我们能不能去做一个单阶段的 3D 生成？」——讲者提出 UniLat3D 时的核心设问（字幕 09:47–09:56 口径）

## 元信息

- 期号：青稞Talk EP88
- 标题：UniLat3D：几何–外观统一VAE的单阶段 3D 生成框架
- BV：BV1qDC7BWEBo
- 时长：00:49:31（总时长 2971 秒；讲授约 37 分钟 + Q&A 约 12 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：吴冠俊（华中科技大学博士生；主持人口径为华中科技大学与华为联培，字幕自述以华中科技大学为准）
- 相关论文：UniLat3D（讲者口径：代码与推理权重已开源，含 3DGS / mesh 的 encoder 与 decoder；论文与项目页地址字幕未给出，待补录）
- 相关代码：已开源（讲者口径：HuggingFace 上有可试用的生成 demo，mesh viewer 当时尚未就绪）
- B站链接：https://www.bilibili.com/video/BV1qDC7BWEBo/
- 字幕原文存档：本地 `transcripts/EP88.txt`（1165 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP88，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名有识别误差：TRELLIS 在字幕中作「CHINESE / CHARLES / CHIS」，UniLat3D 作「unit lix」，DINOv2 作「dao v two」，3DGS 作「3DJS」，mesh 作「match」，occupancy 作「ACCUPANCY」，均以公开资料为准）。凡讲授口径与论文版可能不同，以下数字表统一标注来源为「字幕」。

## 一句话总结

UniLat3D 把 3D 资产的几何与外观塞进同一个稠密 latent：先借 TRELLIS 式稀疏体素特征，再做稠密化并用第二个 VAE 压到 $16^3$ 的紧凑空间，最后只用一个 flow transformer 从噪声单阶段生成、一个 decoder 解出 3DGS 或带纹理 mesh，绕开了主流「先几何、后上色」的两阶段管线。

## 核心

### 背景：3D 生成为什么一直是两阶段

讲者先回顾路线演进：早期靠 2D/2.5D 蒸馏（DreamFusion 的 SDS、Zero123 的多视角生成），之后进入数据驱动的 feedforward 时代（LGM 一类单次推理），再到 3D diffusion 的两条主流路线——Clay 用稀疏体素表征，混元 3D / Step3D 走 shape latent 先出 mesh 再单独上色。

TRELLIS（字幕作「CHINESE」）是转折点：它提出 structured latent，用稀疏空间体素同时承载几何与纹理信息，一个 decoder 就能解出 3DGS、NeRF 或 mesh。但讲者指出，TRELLIS 的**生成**仍是两步：先生成几何（稀疏结构），再训第二个 transformer 生成对应的 structured latent。混元 3D 同理，第一个模型只生成 shape，第二个模型负责上色。两阶段的问题在讲者看来是结构性的：几何与纹理分开训，latent 层面交互很少；他最初想做 4D 生成时更明显——每帧都要独立走一遍「生成几何 → 生成纹理」，偏差会被放大。

由此提出本期的设问：能不能像 2D 图像生成那样，只用一个稠密 latent、一个生成模型、单阶段完成 3D 生成。

### 方法：一个 VAE 同时装下几何与外观

UniLat3D 的 VAE 分编码与解码两侧，关键动作只有两个：**densification（稠密化）** 与 **downsample（再压缩）**。

编码侧（复用 TRELLIS 的数据预处理管线）：

1. 对 3D 资产渲染多视角图片（承载纹理信息），同时把资产体素化到 $64^3$ 分辨率。
2. 用 DINOv2 提取图像特征（维度一千多维），投影到体素上，得到 voxel feature。
3. 先过一个 sparse VAE 把通道压到几十维，得到稀疏的 3D feature（空间结构不变）。
4. **稠密化**：初始化一个 $64^3$ 的全零张量，把稀疏 latent 逐位置填进去，空位置即全零——稀疏表示由此变成稠密张量。
5. **再压缩**：第二个在稠密空间操作的 VAE 把 $64^3$ 压到 $16^3$ ，得到最终的 UniLat，换算成 token 是 4096 个。讲者强调这个分辨率「非常小、非常紧凑」，token 数与 TRELLIS 大致相当。

解码侧是编码的逆过程：UniLat 先经两次上采样回到 $64^3$ ，并行预测 latent 与 occupancy（ $1 \times 64^3$ ）；对 occupancy 设阈值，筛出非空位置索引 latent，还原出与 TRELLIS 同构的 sparse feature；最后接 3DGS decoder 或 mesh decoder 输出最终资产。一个 latent、一个 decoder 入口，同时出几何与带纹理外观——这是「几何–外观统一」的落点。

VAE 的 loss 讲者称「比较简单」：前几项是基于图像的 loss（含 mesh / 3DGS 监督），加 KL loss 约束分布，再加一项专门监督预测 occupancy 与真值一致性的 loss。

### 工程硬骨头：512 分辨率 mesh decoder

讲者坦言这部分踩坑最多。他们把 mesh decoder 的分辨率从 256 提到 512（TRELLIS 是 256）：

- 256 → 512 时顶点数呈立方增长，生成 mesh 时频繁爆显存。
- 解法一：加 self-pruning layer，每次上采样后先过滤一遍，用 GT loss 监督，把无用顶点尽早剪掉。
- 解法二：所用稀疏卷积库原本是 int32 索引，训 512 时会崩，改到 int64 后才跑通。

讲者特别点出对照：很多能做 1024 / 2048 高分辨率的方法不出纹理，训练开销因此低一截；UniLat3D 要求完整出带纹理的 mesh，管线必须全程扛住。

### 生成模型：稠密 latent 让训练回归图像式管线

有了稠密 latent，生成模型就是标准配方：一个 dense flow transformer，给定条件图片与稠密噪声，直接优化生成 UniLat。讲者选 flow matching 而非 diffusion 的理由与 TRELLIS 相同——效果更好、结果更稳定、部署步数更少。结构稠密的好处是能直接用 FlashAttention-3 加速，训练开销相对小、效率高。选 flow matching 而非两阶段拼接，也是「统一」主张的一部分：整个训练范式与 2D / 2.5D 生成对齐，后续才好与 VLM / LLM 结合。

### 训练策略：站在 TRELLIS checkpoint 上

- 数据：TRELLIS 的 500K 数据集（Objaverse 全集为主，含游戏资产与扫描数据），处理时损失约 6 万，最终可用约 42 万；**没有做任何数据清洗**，讲者原话是整个 pipeline「都比较简单」。
- VAE：稀疏部分权重直接继承 TRELLIS checkpoint 并冻结，只训新增的 dense VAE，约 200K 步后 PSNR 到 34.2；再解冻做全量微调（不用 LoRA），又用 1024 分辨率渲染图微调 decoder，总计不到 400K 步（约 360–370K）。
- Flow transformer：先以 batch size 256 训 500K 步，再把 batch 开到 1024 全量微调 160K 步探性能上限；讲者补了一句，全量微调其实十几 K 步效果就差不多了，长训只是为了看收敛极限。

### 实验与消融：16 是甜点，32 反而学不动

- 评测：Toys4K 全集 3218 个样本（每样本渲染前后左右四张 512 分辨率图）；另自建 1000 个复杂物体样本做评估（UID 将开源）。3DGS 生成质量明显优于对比的开源方法（TRELLIS、混元 3D、TripoSG 一类），mesh 指标与混元 3D 大致持平或略好；几十人、20 个样本的用户研究投票也更高。讲者的经验判断是：同等条件下 3DGS 的生成质量就是比 mesh 好。
- 速度与规模：A100 上测（无 FlashAttention-3）时间略吃亏，开 FA3 后与 TRELLIS 相当、比其他方法快很多；参数量比 TRELLIS 略多，同数量级。
- 编码器消融：项目中途把 DINOv2 换成 DINOv3（只换 flow 侧编码器，预处理仍用 DINOv2），复杂物体指标更好。
- **latent 分辨率消融是本期最有信息量的负结果**：压到 $8^3$ （512 个 token）时重建指标怎么训都到不了 34（卡在 33 点几），放弃； $16^3$ 是训得稳的甜点；升到 $32^3$ 后重建指标确实涨了，但 flow transformer「怎么都上不去」——讲者推测是 32 分辨率 latent 的分布对 flow 来说学习难度太大。结论： $16^3$ 的稠密 latent 已经足够表征一个 3D 资产，分辨率不是越高越好，生成模型学得动才是上限。

### Q&A 要点

- **实时 3D 生成的瓶颈**：讲者判断 decoder 不是问题（3DGS decode 只要 0.0 几秒）；网络规模与噪声调度都有人在做，他们另一篇工作可把 TRELLIS 式生成从 3 秒蒸馏加速到零点几秒（十倍以上）。本地实测约 3 秒，但 HuggingFace demo 要四五十秒——他认为更大的问题在部署优化。
- **本地部署资源**：3DGS 推理约 6–6.7 GB 显存，8 GB 应该够，3090 / 4090 肯定够；mesh 看最终面片数。
- **为什么统一训更好**：分开训时几何与纹理在 latent 层面交互少；统一后在更低分辨率下反而促进信息融合。他也承认在当前数据量下，方法的性能上限基本就到这个水平，UniLat 的收益更多在结构而非堆料。
- **与 TRELLIS 的一句话区别**：TRELLIS 的 VAE 只管 appearance、另有模型管 geometry，生成分两步；UniLat3D 是「二合一」——一个 VAE、一个 flow。
- **灵感来源**：最初是想做 4D 生成，发现两阶段管线在逐帧场景下偏差难控；加上「统一」本就是各领域的趋势，才有了几何纹理二合一、单个 flow 的设计。
- **未来方向**：更高分辨率 mesh 生成；多物体与 3D 场景生成、单目视频输入的 4D 生成；以及把稠密 latent token 交给 LLM / VLM 去赋能——讲者说这正是做 dense latent 的初衷，希望 3D 开源模型「用单模型解决更多 task」。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 体素分辨率 | — | $64^3$ | 字幕 |
| UniLat 分辨率 | TRELLIS 稀疏 structured latent（token 数大致相当） | $16^3$ ，即 4096 个 token | 字幕 |
| mesh decoder 分辨率 | TRELLIS 256 | 512 | 字幕 |
| VAE 重建 PSNR | 只训 dense VAE 约 200K 步 | 34.2 | 字幕 |
| VAE 总训练量 | — | 不到 400K 步（约 360–370K） | 字幕 |
| Flow transformer 训练 | batch 256 | 500K 步；后 batch 1024 微调 160K 步 | 字幕 |
| 训练数据 | TRELLIS 500K 数据集 | 处理后约 42 万可用，未做数据清洗 | 字幕 |
| Toys4K 评测规模 | — | 3218 个样本 × 4 视角 × 512 分辨率 | 字幕 |
| 3DGS 推理显存 | — | 约 6–6.7 GB | 字幕（Q&A） |
| 生成耗时 | HuggingFace demo 约 40–50 秒 | 本地约 3 秒 | 字幕（Q&A） |
| latent 分辨率消融下限 | $8^3$ = 512 token | 重建指标卡在 33 点几，不可用 | 字幕 |
| latent 分辨率消融上限 | $32^3$ | 重建更好，但 flow 学不动、生成指标上不去 | 字幕 |

## 可迁移

- 「分辨率甜点」纪律：latent 分辨率往上加之前，先验证下游生成模型学得动该分布——重建指标涨而生成指标不动，是分布难度先于表达能力成为瓶颈的信号。这条对任何 latent diffusion / flow 管线都成立，先做小规模分辨率消融再定版。
- 稠密化换兼容性：把稀疏表示填进稠密张量（空位补零）看似浪费，却换来了与图像管线完全一致的训练/推理栈（FlashAttention 等现成加速直接可用）。做跨表示迁移时，「先对齐张量形态、再谈效率」往往比维护稀疏专用算子更省总工时。

## 疑问 / 下一步

- 论文与项目页地址字幕未给出，UniLat3D 的正式发表信息（会议/arXiv）与讲者口播的「48 左右」的 3DGS 指标名（字幕作「FAD」）需查论文原文核对，本纪要未展开该数字。
- $32^3$ latent「flow 学不动」只有推测（分布难度），没有给出分布层面的证据（如 latent 统计量对比）；若后续工作要复用这个结论，需要论文消融支撑。
- 自建 1000 样本复杂物体评测集的构成与开源时间字幕未细说，复现对比时需等其 UID 释放。

## 原文金句（1-2句）

> 「我们能不能去做一个啊单阶段的这个 3D 生成？」——讲者提出 UniLat3D 的设问（字幕 09:47–09:56，按干净口径转写）

> 「我能肯定是这个 decoder 不是问题……我们现在做 3DGS 这个 decoder，他的 decode 时间基本都是 0.0 几秒。」——Q&A 中对实时生成瓶颈的判断（字幕 38:30–38:42 口径）
