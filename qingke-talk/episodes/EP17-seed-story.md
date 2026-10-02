# EP17 — SEED-Story：生成长篇图文故事的多模态大型语言模型

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP17-seed-story.html

> 「我们就想，能不能直接通过这个大语言模型来生成长的故事序列——训练的时候最多只见过长度为 10 的故事，靠多模态 attention sink，推理时能稳定生成到 25 章。」——讲者对本工作雄心的概括（字幕口径）

## 元信息

- 期号：17（官网期号）
- 标题：《SEED-Story：生成长篇图文故事的多模态大型语言模型》
- BV：BV1MDaYzgEt1
- 时长：00:57:30（讲授与弹幕答疑交替，Q&A 分布于全程）
- 提炼日期：2026-10-02
- 分享嘉宾：杨帅（香港科技大学（广州）博士二年级，师从陈颖聪教授；研究方向：图像生成、多模态大模型与 efficient learning；字幕自述，头衔以官网预告为准）
- 相关论文：SEED-Story（与腾讯 PCG Arc Lab 合作；字幕中作「SEED组」系列工作）
- 相关代码：已开源（GitHub，含模型、代码与数据集；讲者提及发布当天 GitHub 约 605 star）
- B站链接：https://www.bilibili.com/video/BV1MDaYzgEt1/
- 字幕原文存档：本地 `transcripts/EP17.txt`（1545 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP17，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 SEED 作「c story」、Q-Former 作「q phone」、GPT-4V 作「GBT」，均以论文与公开资料为准）。

## 一句话总结

讲者把「多模态故事生成」拆成输入理解与输出生成两端同时变难的任务，并给出一条完整解法：用 SEED 的视觉 tokenizer 把图片压成连续特征 token，让 LLaMA-2 以自回归方式同时生成下一段文字与下一张图；三阶段训练（视觉编解码对齐 → 指令微调 → 冻结 LLM 只调 SDXL detokenizer 补风格 gap）解决图文生成，再用一个从 attention map 里重新发现的「多模态 attention sink」（滑窗推理时除首 token 外，额外保留每张图的 begin/end-of-image 附近 token）实现 train-short-test-long，把有用长度从训练见过的 10 推进到 25。配套的数据集用 video-to-dataset 流水线（视频抽帧 + GPT-4V 逐帧 caption + GPT-4 织成 30 长度的故事）把规模做到约 60 万图、高分辨率、叙事性文本。

## 核心

### 任务定义：为什么图文故事生成比 caption 难一个量级

讲者先把任务边界划清：用户给第一张图与一段文字，模型自回归地往后生成图文穿插的长故事。这不是 story visualization（只按给定文字出下一张图），而是图、文同时生成。难点在输入、输出两端各三条：

1. **图复杂**。漫画/绘本图高清、细节多（角落的人物、衣着纹理），要求像素级理解；
2. **文复杂**。故事文字是叙述性的（时间状语、情感描写），与图的对应远弱于 caption；
3. **穿插复杂**。输入可能是「10 张图 + 10 段文 → 第 11 图 + 第 11 段文」的多图多文上下文。

输出端同理：图要风格一致、文要连贯有吸引力、图文还要对应。GAN/diffusion 路线做不了这个，核心原因是 diffusion 的 text encoder 理解不了叙述性文本；StoryGPT-V 一类工作要靠逐人物检测框这类额外监督。MLLM 路线的好处是把图、文都压成 token 在同一个自回归序列里处理。

### 三阶段训练：先让 LLM「看见」，再教它讲，最后补画质

模型基于 SEED 系列：视觉 tokenizer（冻结的 ViT encoder + SDXL decoder，类 VAE 结构）把图片 encode 成与 LLaMA token 同维（4096）的连续特征，decode 回图能做到近 pixel-wise 重建——讲者强调这比早期 Emu/CLIP 式只能语义对齐的方案保真得多。

- **阶段 1（视觉理解）**：直接用 SEED-X 早期预训练模型做 visual tokenization / detokenization，在大量图文对（CC3M 一类）上训，使图片能无损进出 LLM 的特征空间；
- **阶段 2（指令微调）**：把图文序列拼成一个长序列（BOS → 文 token → 图的连续 token → 可选用户 prompt → …），每个样本随机截取前缀、只 predict 最后一对（图+文），类似多轮对话；监督分两路——文字位置用正常 LM 的 CE loss（图像位置 label 置 -100 不算 LM loss），图像位置则取 LLM 最后一层 hidden states，经 learnable query / Q-Former 与目标 image feature 交互，算回归 loss（MSE 或 cosine 都试过）；
- **阶段 3（补风格 gap）**：只做阶段 2 的模型语义对、画质歪。讲者归因于 MLLM 输出特征与真实 image feature 之间的 space misalignment（loss 始终降不下去）——既然对不齐，就冻结第二阶段的 LLM、只微调 SDXL detokenizer，让解码器去适配 MLLM 的输出分布，风格问题随之消失。

单图在 LLM 中的 token 数：原始 image feature 是 $256 \times 4096$ ，经 Q-Former 压到 64 个 token / 图。底座 LLM 是 LLaMA-2 7B（讲者 Q&A 补充：SEED-X 有 13B 版，换上应该只会更好，未实测；SEED tokenizer/detokenizer 只有 7B 配套版本）。

### 长故事生成：把 StreamingLLM 的 attention sink 推广到多模态

训练数据最长 30、实际只训到长度 10，但推理想出更长——train short, test long。讲者实证 naive 做法的两种死法：直接全长推理到后期直接出灰图；用 sliding window 一滑窗就崩，这和纯语言里的 StreamingLLM 现象一致（滑掉第 0 号 token 那一列大 attention，性能崩溃）。但语言版 attention sink（只保留开头 token）套过来也不够，长了仍出灰图。

他们重新可视化了长故事生成时的 attention map，发现高 attention 的 token 有四类：① 第 0 号 token；② 标点符号；③ 每张图的 begin-of-image 附近 token；④ end-of-image 附近 token。标点符号虽然 QK 值大但 V 值很小（讲者引既有结论），故不保留；**Multimodal Attention Sink** = 滑窗时保留开头 token + 每一张图的 BOI/EOI 附近 token。效果是 dense attention 与 plain sliding window 在后期全部崩溃时，它推理到第 35、39、44 章仍能出有意义的图，且相比 dense attention 又快、显存又低（窗口有界、只多保留少量 token）。量化实验中该推理方式的 FID 最低、CLIP score 最好。

长度边界讲者交代得很明确：理论上无上限（算力除外），但训练长度 10 时，超过 25 章开始出现单章重复或周期循环重复（每 3–5 章一循环），所以 claim 的可用长度是 25。数据其实最长到 30，训 30 理论上能推得更远，只是没算力验证。

### 数据集：video-to-dataset 是真正的工程主体

现有 Flintstones / Pororo 分辨率只有 $128 \times 128$ 、文本近乎「谁在房间里站着」的图片描述，撑不起大模型。SEED-Story 的数据流水线：从动画视频抽 I 帧 → DINOv2 算相邻帧相似度去重（太像的不像故事）→ GPT-4V（或千问 VL）逐帧打纯描述性 caption（讲者注：GPT-4V 一次吃不了 30 张图，所以逐帧走）→ 把 30 个 caption + 原视频字幕一起给 GPT-4，让它织成连贯的叙事文本。成品特性：规模从首版（Curious George）约 257K 图扩到三个动画（Curious George、Rabbit Invasion、The Land Before Time）约 600K 图；每个故事足足 30 个图文对（旧数据只有 5）；文本平均长度约为旧数据的 2 倍。数据集、prompt 与构建代码均已开源。

### 实验与评估

- **Story visualization**（给前文出单图）：与 LDM（SDXL 逐图生成）和支持前文角色的工作对比，SEED-Story 一致性最好；量化上 FID 最低且与纯 LDM 相当（讲者解读：FID 主要衡量画质）、CLIP score 最高。讲者点出竞品的典型失败：前三张图是角色 A、第四张该换角色时它换不过去。
- **Multimodal story generation**（开放式续写）：没有 GT，只能用 GPT-4 当裁判，三个维度——image style consistency、story engagement、text-image coherence；与 Emu-Interleave 一类（字幕作「m interleave」）的相对胜率比较中 SEED-Story 风格一致性大体打平、engagement 略胜、图文 coherence 胜出更多；另有 0–10 分绝对评分。他们把这套评估作为 benchmark 发布，方便后人同口径比较。
- **Multimodal Attention Sink 消融**：FID 最低、CLIP/IP-adapter 指标最好，inference time 与显存都低于 dense attention，只比什么都不保留的 plain window 慢。

### Q&A 要点

- **视觉 tokenizer 怎么实现**：冻结 ViT 编码 + SDXL 解码，训 SDXL 实现 cycle-consistency 重建；
- **图文是什么关系**：有交集但不完全重合——文本负责情感描写（安慰、担忧），图负责背景细节（山、雪花、风向），像漫画一样各管一部分、不可互替；一致性主要靠训练数据本身的图文对齐 + LLM 自己学，没有专门融合模块（对比 diffusion 工作里的 MFMM / cross-attention 专门设计，LLM 里天然成序、丢进去就行）；
- **生成新 IP / 全新形象行不行**：不行。这是「一个 IP 内自由发挥」的模型——没见过的角色（钢铁侠、新照片里的人）做不了高保真保持；但没见过的普通角色描述（穿蓝外套戴帽子的小男孩）能靠语言 prompt 泛化出来。讲者自评这是与 StoryDiffusion 那类 personalization 工作的取舍：后者任意输入保真强，但假设主角每张都在场；故事生成图与图之间变化大得多，任务更难；
- **长度限制**：见上，训练长度决定可用长度，25 章 claim；
- **防遗忘**：短序列全量塞入、attention 自己学分配；滑窗后远端信息确实会丢，但讲者认为这合理——新图与很早的图关系本就弱，何况 BOI/EOI token 还留着些长期依赖；
- **应用形态**：生成的图文序列接 image-to-video 模型 + AI 旁白就是可给孩子的动态绘本（demo 里开源 I2V 效果「有点鬼畜」，换可灵这类好模型即可）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 单图在 LLM 中的 token 数 | 原始 image feature $256 \times 4096$ | 64 token / 图（Q-Former 压缩） | 字幕（Q&A，49:22–49:47） |
| 底座 LLM | — | LLaMA-2 7B（13B 未实测 SEED-Story） | 字幕（Q&A，55:55–57:15） |
| 训练故事长度 | 数据最长 30 | 实际训到 10 | 字幕（21:38–22:42、50:52–51:12） |
| 可生成的故事长度 | 训练长度 10 | 有意义生成 up to 25 章；超 25 出现单章/周期重复 | 字幕（07:23–07:37、48:50–50:50） |
| Multimodal Attention Sink 有效长度 | dense / plain window 后期出灰图 | 第 35/39/44 章仍出有意义的图 | 字幕（27:37–27:46） |
| 数据集规模 | Flintstones/Pororo 约 5 图/故事， $128 \times 128$  | 约 60 万图（首版 257K）、每故事 30 图文对、文本平均长度约 2× 旧数据 | 字幕（34:17–37:27） |
| Story visualization 指标 | LDM（SDXL）与前文角色方法 | FID 最低、CLIP score 最高（FID 约与 LDM 相当） | 字幕（48:04–48:38） |
| GPT-4 评估维度 | 与 Emu-Interleave 类工作相对比较 | engagement 略胜、text-image coherence 胜更多、风格一致性打平 | 字幕（52:22–54:03） |
| 开源情况 | — | 模型/代码/数据集全开源；GitHub 约 605 star | 字幕（55:35–56:12） |

## 可迁移

- **训推长度外推先查 attention sink**：长上下文推理崩溃，未必是位置编码问题——先可视化 attention map 看哪几类 token 在「承重」。SEED-Story 的经验是：除首 token 外，每个结构边界 token（这里的 BOI/EOI）都可能是一根承重柱，长上下文/滑动窗口推理时应按结构保留，而非只认第 0 号。对 agentic RL 的长轨迹推理同样适用：system prompt、工具调用边界 token 的 KV 保留策略值得单独做实验。
- **特征空间对不齐时，冻结主干去适配解码器**：阶段 3 的思路是「output feature 与 target feature 之间 gap 降不下去时，别硬逼主干，训一个便宜的适配端去吸收分布偏移」。这与 post-training 里常见的「训 adapter / 重标定输出头吸收 SFT 偏移」是同一招，且成本低一个数量级。
- **数据流水线里 GPT-4V 逐帧打标 + GPT-4 全局织文的分工**：局部理解用多模态模型、全局连贯用纯文本模型，是一个可复用的「感知贵、编排便宜」分工；约束是多模态模型一次吃不下 30 张图——先降维成文本再做长程编排。对 coding data / 轨迹数据的合成（逐帧 caption ≈ 逐 step 标注，织文 ≈ 轨迹级重写）有直接对应。
- **开放式生成任务的评估模板**：无 GT 时用强模型当裁判 + 相对胜率（pairwise）与绝对分（0–10）双轨，并把 prompt 与裁判模型版本固定下来发布——比只报一个 FID 可复现得多。

## 疑问 / 下一步

- Multimodal Attention Sink 里 BOI/EOI 附近到底保留几个 token、保留的 KV 随故事推移线性增长到什么程度，讲者未给数值；44 章之后的退化形态（是重复还是崩坏）也未展示。
- 阶段 3 只训 detokenizer 吸收偏移，会不会把 MLLM 输出里本应纠正的错误固化进解码器？讲者未做「阶段 2.5 交替训」的对照。
- 25 章可用长度是 LLaMA-2 7B + 训练长度 10 的口径；换更强底座或训满 30 后，重复出现的临界点会怎么移动，讲者明确说因算力未验证——这是论文最容易被后续工作刷新的数字。
- 与 StoryGPT-V 的检测框监督路线相比，「无额外监督」的上限在哪、两种监督能否叠加，talk 未讨论。

## 原文金句

> 「MLLM 有一个很大的好处，就是它能把图、文都压缩成一个一个 token，高效地实现自回归。那我们能不能直接通过这个大语言模型来生成长的故事序列？」（字幕 05:56–06:10）

> 「（图文）是一个有交集但又不完全重合的关系……文字和图各负责了一部分功能，它都没有一种替换关系。」（字幕 30:07–30:25）

> 「如果你真的想去实现一个 open world storytelling，在这么难的一个任务前提下，我觉得你有非常大量的数据才能做到这一点。」（字幕 31:17–31:28）
