# EP29 — VILA^2：视觉语言模型能力的自我提升（Self-Augmenting VLM）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP29-vila2.html

> 「VILA^2 是让资本越来越好的一种方式，且让模型在更好的 cap 上进行训练——是一个 self-augmenting 的 cycle。」——讲者总结

## 元信息

- 期号：29
- 标题：视觉语言模型能力的自我提升（VILA^2）
- BV：BV1cHYNzJE77
- 时长：01:11:06（总时长4266 秒，含 Q&A）
- 提炼日期：2026-10-02
- 分享嘉宾：方云浩（Yunhao Fang；讲者自述：浙大本科、UCSD 硕士（导师苏浩老师）、当时在 NVIDIA VILA 团队实习；以字幕自述为准）
- 相关论文：VILA^2（讲者未在字幕中给出 arXiv 号，待从论文页补录）
- 相关代码：VILA codebase 开源（讲者在讲授中说明）
- B站链接：https://www.bilibili.com/video/BV1cHYNzJE77/
- 字幕原文存档：本地 `transcripts/EP29.txt`（1570 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP29，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 VILA² 在字幕中作「维拉斯科尔/BILASQUARE/blast square」、S² 作「s two」，均以论文与公开资料为准）。凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

讲者把「已经训好的 VLM」当作自标注工具，提出两个 AI-in-the-loop 闭环：一是 self-augmentation（让 SFT 后的模型回过头去对预训练数据做稠密 recaption、再从头重训），二是 specialist augmentation（用 OCR、grounding、空间关系三个 specialist 的结构化标注继续喂回预训练）；实验证据显示 self 循环在约三轮时饱和，叠加 specialist 标注后还能再上一个台阶。

## 核心

### 背景：VLM 不是缺图片，是缺好 caption

讲者先把范围定在视觉智能：从 CLIP/ALIGN 式对比学习、Flamingo/LLaVA 式生成式 VLM，走到今天的伪问题是：互联网图的 caption 质量太差（COCO 那类短且图不对文的 alt-text），而 SFT 阶段用的是极高质量的数据。于是悖论出现：用高质量 SFT 训好的模型，知识没办法回灌到 pretrain——VILA^2 的问题就是能不能把训后模型的 re-caption 注入回预训练，帮模型平滑起步。

VILA 本体讲者过了一遍：CLIP ViT + 简单 MLP connector（试过 perceiver/resampler，但 OCR 类任务 token 剪太多、掉分，于是用 MLP）；训练走三阶段：projector-only 对齐→交错图图文 pretrain（in-context learning 能力主要来自这里）→多样 QA 做 SFT。以及 token 压缩方案 S²：原图插值放大 2x2、分区过 ViT 拼 feature map（类似 FPN 的多尺度），再用 pixel-shuffle 风格的 2x2 downsample 把 visual token 压到约 200 个/图（SigLIP 最大配置本需 729）。

### 第一条循环：self-augmenting VLM（self-augmentation）

做法是把 V0→V1→V2 的 recaption 拿回 pretrain 从头再训：Round 0 是原始数据训的 V0，Round 1 让 V0 回头去 re-caption 整个 pretrain 集，再从头训，依此迭代。每轮 caption 平均长度增加、细节增多，但 hallucation 并不随之增加（用 closed model 当 judge、VSR 当幻觉探针、组里 15 个博士手动评 win rate，三套证据交叉验证，到 Round 3 前 caption 越来越好，Round 3 后饱和、Round 4 掉点）。

讲者强调：SFT 数据仍保留人类高质量标注；recaption 只发生在 pretraining 这个阶段，因为 pretrain 才是两个模态真正 alignment 的过程。视频案例（self-aug video）归结为：视频只训了短帧（8 帧）时，长帧数训练受限于当前帧数，SFT 数据无法放大同样效果。

### 第二条循环：specialist augmentation

第一步把三个视觉密集型 specialist（OCR、grounding、空间关系）当作高质量标注源：OCR 认图上文字（table/chart 数据），grounding 让模型输出 bounding box，空间关系 specialist 从 3D 公共数据集里 filter bounding box、再从 relation pool 里采一条关系构造 Yes/No 的 QA。关键设计点：bounding box 坐标被量化为 000-999、每格一个 special token（共约 1000 个新增 token），预测时用 soft cross-entropy（因为 token 序列本身是顺序敏感的，123 vs 120 和 123 vs 999 距离不同，裸文本 CE 不适用；采用类似五档权重 0.1/0.2/0.3/0.4/0.3 分布式的 soft loss，讲者口径，五档权重具体数值以字幕为准）。三类 specialist 标注拼到同一张图的多轮 QA 上训练，从头训出新一代。

实证对比：recaption 数据从 10% subset（2.5M）放到 25M 全量后性能明显好；SFT 只 1.8M 子集也能拿到强结果；8B specialist 标的数据训 34B 仍然涨分（weak-to-strong generalization）。

但同理把 specialist 的源数据直接混进 pretrain 不行，因为数据量太小（点明：specialist 的价值是把领域专业知识高频、大量地表征出来再蒸回模型，不是加几条数据）。

### 天花板与后续方向

饱和之后讲者的推断是两个：第一未来应利用 specialist 的 crop verification、visual chain-of-thought 这类更强的标注工具（讲者自评只是 proof-of-concept，没 scale 下去）；第二是把这个循环和 video pretraining、world model 结合（372 帧视频输入 generalization 时视频模型的改善会被当前 SFT 数据集压回 8-16 帧上限，所以 pretrain 越好、SFT 压回的损失越大）；第三，沿着这条路他看到的 broader 图景是 GPT-式 self-evolving loop（exploration→diverse data→verification），以及 VILLu（understanding+generation 统一）、LongVILA（视频打长）等团队后续工作。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| self-augmentation 有效轮数 | Round 0（V0） | 约 3 轮后饱和（到 Round 4 性能回落） | 字幕（讲授口径） |
| visual token 压缩（S² + pixel shuffle） | SigLIP 最大 729 token/图 | 约 200 token/图 | 字幕（讲授口径） |
| recaption 数据规模（实验子集） | 2.5M（25M 的 10%） | 全量 25M 时性能更好 | 字幕（讲授口径） |
| SFT 数据规模（实验子集） | — | 1.8M（快速实验子集） | 字幕（讲授口径） |
| bounding box 坐标量化 | 连续坐标 | 000–999，共约 1000 个 special token | 字幕（讲授口径） |
| weak-to-strong 结果 | 8B 模型标注 | 训 34B 仍能涨分 | 字幕（讲授口径） |
| VSR 空间幻觉探针 | 基线幻觉率 | 经 VILA^2 训练后幻觉下降（无具体数值） | 字幕（讲授口径） |
| 论文与具体基准逐题涨幅 | — | talk 未逐项给出 | 字幕（未展开） |

## 可迁移

- 「训后模型回灌 pretrain」是后训练数据工作的一个通用模式：SFT 阶段的高质量分布可以蒸回预训练阶段，dense recaption + 保留真实标注数据做 mix 是讲者建议的 recipe（尤其别把 SFT 数据也换成自标注，讲者自述试过换了会训崩）。
- Specialist 的工程要点是 coordinate tokenization：把连续回归任务（bbox）改造成有限词表分类 + soft CE，比硬当 text 去预测 CE 更靠谱；这个做法对 agent 里 bounding-box 类输出（如 GUI 操控）可直接迁移。
- 评估方法可借鉴：self-aug 这类数据循环的质量检查可以由「closed-model judge + 定向幻觉探针 benchmark（如 VSR 空间关系）+ 小规模专家 win rate」三件事拼起来，避免只看 benchmark 总分的假阳性。

## 疑问 / 下一步

- 从 self-augmentation 饱和（约 3 轮）到 specialist augmentation 的收益，能持续多少个 specialist？讲者未做 specialist 组合消融，只示范了 OCR/grounding/spatial 三个。
- 把 specialist 标注直接混进 pretrain 为什么没涨（讲者一句归因为量太小），量做到多大才涨、以及 grounding tokenize + soft-CE 这种改造脱离 VILA 管线还成立否，需要单独实验。
- Video pretraining 数据（32 帧级 specialization）与 SFT 数据 8-16 帧上限的错位，讲者说是当前视频应用瓶颈——这条和现有 video SFT 的数据 pipeline 如何衔接，未展开。

## 原文金句（1-2句）

> 「VLA square 是一个让资本越来越好的方式」——讲者对 self-augmenting cycle 的概括（字幕 22:19-22:28，按干净口径转写；注：VLA/VILA 字幕识别有混淆）

> 「他们实际上不是 invariants scale 的问题，他们更像是我们把模型变得更聪明一点」——讲者解释更多 FLOPs 训 baseline 并不会让它变好（字幕 22:20-22:28，按干净口径转写）
