# EP43 — VideoGrain：基于扩散模型的多粒度视频编辑的探索与应用

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP43-videograin.html

> 「现在所有的视频编辑方法都忽略了对视频粒度（instance level 乃至更细）的感知。」——讲者对既有方法失效原因的判词

## 元信息

- 期号：青稞Talk EP43
- 标题：VideoGrain：基于扩散模型的多粒度视频编辑的探索与应用
- BV：BV1REpYzZEWy
- 时长：01:00:55（讲授约 47 分钟 + Q&A 约 14 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：VideoGrain 论文一作（悉尼科技大学 ReLER Lab；合作导师包括朱林潮、杨毅等，杨毅已回浙江大学；讲者姓名未在字幕中清晰自述，以论文作者页为准）
- 相关论文：VideoGrain: Modulating Space-Time Attention for Multi-Grained Video Editing（讲授中提到发表于 ACCV 一类会议，字幕作「ACCLEAR」，以论文原文为准）
- 相关代码：已开源（GitHub 仓库含 class/instance/part 三个粒度的全部 config 与数据，可一比一复现；数据在 HuggingFace 与 Google Drive）
- B站链接：https://www.bilibili.com/video/BV1REpYzZEWy/
- 字幕原文存档：本地 `transcripts/EP43.txt`（1392 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP43，已存档）清洗提炼，信息来源为 AI 字幕原文（经清洗；个别专名可能有识别误差，如 VideoGrain 在字幕中作「video gram」、Video-P2P 作「V6P6P」、latent blending 作「雷电blanch」，均以论文与公开资料为准）。论文仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

讲者指出既有视频编辑方法做不了「同类多实例分别编辑」的原因是扩散模型自身缺乏实例级感知（self-attention 特征分不开同类实例、cross-attention 权重定位错误且泄漏），提出 VideoGrain：对时空注意力做正负对调制（spatiotemporal layout attention），在一次去噪过程中同时实现 class、instance、part 三个粒度的文本到区域控制，并做到 16 帧视频不到 4 分钟完成编辑。

## 核心

### 背景：视频编辑的任务谱系与三大挑战

讲者先梳理谱系：风格编辑（SD 1.5 时代起）、单物体前景编辑、motion transfer、motion editing（NeurIPS ReVideo 一系，用 trajectory 改动作）、inpainting/outpainting/inserting——本质都是对某个区域的动作与纹理修改。三大挑战：

1. **时空一致性**：编辑结果不能抖动，光照与运动轨迹要连贯，长视频还要处理镜头转换；
2. **语义理解与保持**：多物体场景下要准确定位追踪目标、理解目标间关系、符合场景语义约束；
3. **效率**：主流模型推理一个视频 50 步约 20 分钟，蒸馏可到 1–2 步；理想目标是 5 秒出预览、单张 3090 一两分钟处理几百帧。

代表性前作：TokenFlow（只编辑关键帧，用扩散特征的最近邻对应把编辑传播到全帧保一致性）、VideoSwap（CVPR 2024，用 semantic point correspondence 做大形变物体的编辑）、trajectory 控制的 motion editing（三阶段渐进预训练引入轨迹控制）。

### 问题定义：多粒度编辑与既有方法的系统性失效

VideoGrain 定义三个粒度：**class level**（同类整体编辑，如两个人→蜘蛛侠、地面→雪地）、**instance level**（同类内每个实例分别编辑，如左猫→萨摩耶、右猫→老虎）、**part level**（物体局部，如给超人加帽子/墨镜、只改衬衫颜色）。既有方法在这个任务上系统性失效：TokenFlow 与 Grounded Video 把两个人都混成蜘蛛侠；给了左右 bounding box 仍有 leakage；用视频模型编辑的 DMT 分不出左右；工业界的 Pika 把左边编辑成蜘蛛侠却给下半身安上狗熊的腿、右边的人根本没编辑。

讲者用两个诊断实验定位原因：

1. **self-attention 特征没有 instance awareness**：对 DDIM inversion 过程中的特征做 k-means 聚类，聚类数从 3 加到 5 都分不开左右两个人（只能把两人的上半身一起聚出来）——扩散模型预训练时就没有学到同类不同实例的区分；
2. **cross-attention 权重定位错误且泄漏**：全局 prompt 里 IronMan 与 SpiderMan 的权重都重叠在左边的人身上，cherry blossom 的权重又漏到人身上，所以编辑结果必然错位。

### 方法：时空布局注意力（正负对调制）

统一思路：在 attention 的 query-key 打分图（condition map）上做调制——增大正对（positive pair）的权重、减小负对（negative pair）的权重，self-attention 与 cross-attention 分别定义正负对：

- **文本到区域控制（cross-attention）**：用 SAM 得到目标区域 mask 后，增大目标词（如 SpiderMan）到其目标区域的权重、减小它到其余区域的权重；每个词只许控制自己的区域。调制量需做归一化处理（以该词打分的最大值等为参照构造正负增量），保证修改后的值落在原始 cross-attention 分布范围内；
- **实例级特征解耦（self-attention）**：diffusion feature 本身带有跨帧对应关系（点一个帧中人的中心点，其他帧的最大响应也落在人的中心、最小响应在地面），但多实例时响应会同时落在左右两个人身上。利用已有的区域 mask 约束：让左人的正对只加在左人区域、区域外置负值，self-attention 调制后左人只关注左人，实现同类实例的特征分离。

方法还可输入 ControlNet 条件进一步稳定运动相似度，整体在一次 denoising 过程中完成多区域编辑。

### 实验：指标、效率与三组关键消融

- **量化指标**：CLIP-F（帧间一致性）、CLIP-T（文本-视频对齐）、warp error（像素级时序一致性）、综合分 $Q_{edit}$ 由 CLIP-T 与 warp error 组合（字幕口径），另有 user study；VideoGrain 各项均为最好（具体数值讲授未口播，需查论文）。
- **效率**：编辑 16 帧视频不到 4 分钟，显存占用也是对比方法中最低档。
- **消融一（正负对的帧范围）**：正负对只取每帧自身 → 蜘蛛侠与钢铁侠的纹理仍有耦合；取第一帧+上一帧 → 好转但仍有；取全部帧的最大响应 → 纹理分离最好。时空范围越大，一致性调制越有效。
- **消融二（不用 SAM 的粗糙 mask）**：用聚类得到的粗糙 layout mask（与人体形态时合时不合）作条件，编辑效果依然成立，方法对 mask 质量鲁棒。
- **消融三（给基线同样的 mask 公平吗）**：给 Video-P2P 三个区域的 mask 让它一次编辑，它仍把 SpiderMan 与北极熊的权重混在左边人身上；分三次顺序编辑也做不到——原因是多轮 denoising 误差累积、latent 逐渐偏离原始视频分布。讲者结论是失效根源在基线自身的 cross-attention 权重分布就不准，给 mask 救不回来。

讲者还指出 VideoGrain 能定位概念（左人/右人分别定位再编辑），做到了 2025 年 2 月走红的 concept attention 类工作的事，且更早。

### 讲者的方向判断（结尾展望）

1. **motion editing 从 trajectory 走向文本**：不用用户输入轨迹，用带动作先验的强视频模型直接按文本 prompt 改动作；
2. **长视频编辑**：现在方法多在 64 帧（万相 81 帧），真实一分钟视频约 600–700 帧，全塞进 GPU 不可行；关键帧传播路线有边界模糊问题待解；
3. **instruction-based editing**：GPT-4o 式强多模态 backbone 让模型直接理解用户意图（「把女人变成男人」），不再靠后端算法设计 prompt 替换；
4. **生成侧**：同场景多视角切换生成、multi-shot 长上下文生成、音视频统一生成（foley sound 与 human voice 的统一）。

### Q&A 要点

- **inpainting 与 editing 的关系**：inpainting 是 editing 的子类；多区域 inpainting/inserting（同时抹除或插入多个有交互的物体）是现有 inpainting 方法做不到、而多粒度框架能覆盖的问题；inserting 之后多物体的运动交互控制仍是难点。
- **背景保持**：zero-shot 路线可用 latent blending（区域外 latent 保持原噪声 latent）；但 backbone 若在预训练时做过 mask 类训练，可能根本不需要 blending——是否还需要取决于 backbone 强度。
- **两个角色位置交叉时还能跟踪吗**：讲者的 trick 是保证给模型的 condition mask 本身 fluent（时序流畅），可把编辑后的 mask video 放回原视频检查流畅度；GitHub 里两人/两动物编辑的例子可复现。
- **数据与复现**：三个粒度的 demo config 全部 release，视频与 mask 数据在 HuggingFace/Google Drive，聚类 mask 功能在 config 里打开分类 inversion 特征开关即可。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 编辑效率 | 对比方法（16 帧） | VideoGrain 不到 4 分钟，显存最低档 | 字幕 |
| 主流扩散模型推理耗时 | 50 步 | 约 20 分钟/视频（讲者举例口径） | 字幕 |
| k-means 聚类诊断 | 聚类数 3 / 4 / 5 | 均无法分离同类左右实例 | 字幕 |
| 量化指标 | CLIP-F、CLIP-T、warp error、综合分与 user study | 各项最好（数值未口播，查论文） | 字幕 |
| 长视频规模对照 | 现方法 64 帧（万相 81 帧） | 一分钟真实视频约 600–700 帧 | 字幕 |

## 可迁移

- 「attention 打分图上直接做正负对调制」是一个免训练的控制注入范式：先用外部信号（mask、区域、对应关系）定义哪些 query-key 对该增强、哪些该抑制，再对打分做保分布的归一化增减。它可迁移到任何需要「把某段文本/条件精确绑定到某个区域或某条轨迹」的注意力系统，包括多模态模型的区域 grounding 与 agent 的工具调用路由。
- 失效诊断方法值得借鉴：先用无监督聚类探针（k-means 探特征）验证「模型内部到底有没有这个概念」，再看 cross-attention 权重图定位错误来源——先证明表征缺失、再决定是加调制还是加训练，避免盲目堆方法。
- 多粒度（class/instance/part）的任务定义方式可迁移到评测设计：凡是「整体能做对、个体分不开」的任务（多智能体分别控制、多目标分别编辑、多文档分别改写），都应显式分粒度设评测集，否则平均指标会掩盖实例级失效。

## 疑问 / 下一步

- 讲授只报了指标名与「各项最好」，CLIP-F/CLIP-T/warp error 的具体数值和基线差距需查论文原文核对。
- 方法依赖 SAM 或聚类提供区域 mask：mask 本身出错（如严重遮挡、实例粘连）时编辑如何退化，讲授只给了「mask 需 fluent」的经验 trick，没有失败案例分析。
- 正负对调制的增量幅度、归一化细节只给了构造思路，超参如何选、是否分粒度不同需查论文方法节。

## 原文金句（1-2句）

> 「（对特征做聚类）聚类数加到 5 也没有办法区分左边人和右边人——这说明扩散模型在预训练过程中就没有这种对实例的感知。」——讲者用诊断实验给出的失效根源（字幕口径）

> 「inpainting 其实只是 editing 的一个子类。」——Q&A 中讲者对两条路线关系的定性（字幕口径）
