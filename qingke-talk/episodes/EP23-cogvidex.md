# EP23 — CogVideoX 视频生成开源模型上手实践

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP23-cogvidex.html

> 「始终坚持的就是做社区想要的、开发者想要的一个模型。」——讲者谈 CogVideoX 开源取舍时的原话

## 元信息

- 期号：23
- 标题：CogVideoX 视频生成开源模型上手实践
- BV：BV1PbYVz2E6d
- 时长：01:01:00（讲授约 48 分钟 + Q&A 约 12 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：张雨轩（智谱 AI，CogVideoX 团队；本人负责 SFT 阶段与开源模型重构。字幕把其单位识别为「知乎」、团队名多处作「质谱/智谱混写」，因 CogVideoX 为智谱出品且字幕后文亦有「我们作为智谱」的口径，单位按智谱记录）
- 相关论文：CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer（arXiv:2408.06072）
- 相关代码：GitHub `THUDM/CogVideo`；HuggingFace diffusers 实现（推理代码以 diffusers 库为准）
- B站链接：https://www.bilibili.com/video/BV1PbYVz2E6d/
- 官网期号：EP23（官网预告链接未确证，省略）
- 字幕原文存档：本地 `transcripts/EP23.txt`（1579 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP23，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 CogVideoX 在字幕中多作「code video x / COVIX」、diffusers 多作「defuses/defer」，清影作「轻影」，均以官方仓库为准）。讲授中反复出现的口径（如 5B 生成耗时在 A100/H100 两处说法）按字幕原样保留并注明。

## 一句话总结

这是一场面向开发者的 CogVideoX 开源实操讲：前半讲智谱视频生成线的开源履历与模型侧（3D Transformer + 3D VAE + 专用视频 caption 数据线），后半逐项拆 diffusers 落地细节——显存优化（Tiled VAE、fake context parallel、CPU offload）如何把 5B 模型塞进 4–5GB 显存、推理参数（帧数公式、步数、scheduler/精度配对）与 SAT/diffusers 两套微调路径（LoRA 25 条、SFT 100 条起步），并给出数据格式与超参的完整清单。

## 核心

### 谱系与开源节奏

团队自述的谱系：2021 年 5 月 CogView，2022 年 CogView2 与第一代 CogVideo 开源（当时开源视频生成尚冷）。Sora 演示后社区需求暴涨，团队决定把 CogVideoX 全量开源，节奏为：2024-08-06 开源 2B，08-27 开源 5B 文生视频，09-19 开源 5B 图生视频（I2V），三版均登顶 HuggingFace 热门。开源前先把实现转成 diffusers 规范（"让社区更好用"），社区反馈想要 I2V 就立刻派人做适配——讲者把这种节奏概括为"做社区想要的模型"。讲者本人负责的是 SFT 阶段与开源模型重构。

### 模型侧：3D Transformer + 3D VAE + 数据线

全系采用 3D Transformer（文本与视频 token 在同一注意力内联合建模，解决 2D 方案难以捕捉快速时序运动的问题），配 3D VAE（压缩率高于 2D VAE，时间维做 context parallel，使推理成本只比 2D 高一点、流畅度高得多）。训练走低分辨率 → 高分辨率 → SFT（高质量数据）三段式，视频统一切成 ≤6 秒片段（有 4/5/6 秒多档）。

数据标注线值得单独记一笔：先用 CogVLM 逐帧描述、汇总结为视频级长描述，再把描述压回 `≤226` token（即 prompt 上限）；为一步到位得到这种训练 prompt，团队另训了专用 **CogVLM2-Video caption 模型**（基于已开源 CogVLM2-Video 微调，同样在 09-19 开源）——它不是通用理解模型，提示词只有"描述这个视频"一句，专为造训练数据。图生视频模型基于文生视频全量微调：图片作为额外条件输入、训练时加大噪声，输入通道 16 → 32；另加可学习位置嵌入，但讲者实测关掉它肉眼差距不大，称之为探索性组件。

### 显存工程：瓶颈在 VAE decode

讲者把推理优化讲得很具体：最初 2B 推理需 12–24GB 且有巨大峰值，根因在 VAE decoder 的显存峰值。三件套：(1) 3D 卷积按张量 2GB 阈值分块（单张量可达 10GB+）；(2) **Tiled VAE**——把输入切成小块分别编解码再拼接，单这一步把峰值从 36G 拉到 24G；(3) **Fake Context Parallel**——49 帧切成多个 13 帧块串行传，每块保留最后 2 帧给下一块 continuity，encoder/decoder 都已合入（与 HuggingFace diffusers 团队合作完成并 PR 进主仓库）。加密度优化叠加后，完整 VAE 的 encode+decode 峰值从约 71G 降到约 30G（这也决定了微调的起步线）。再加 slicing 与 CPU offload（model offload / sequential offload）三行代码，即可 5B BF16 在 4–5GB 显存推理；代价讲者明说：所有显存优化都是"拿时间换空间"，且成倍数变慢。

模块结构上的配对细节（用错很常见）：2B 用 FP16 训练与推理 + DDIM scheduler + sin/cos 位置编码；5B 用 BF16 + DPM scheduler + RoPE。2B 选 FP16 是刻意照顾 1080Ti/2080Ti 这类不支持 BF16 的老卡用户；反过来 5B 用 FP16 推理效果明显变差。

### 快速上手：推理参数清单（字幕口径）

- **提示词润色是必经工序，不是可选项**：训练 prompt 是 200+ 词的详细长描述（如 "a girl is riding a bike" 这类短句"没办法达到很好的效果"）。官方提供了用 GPT-4/GPT-4o 类大模型 few-shot 润色的代码；中文输入可由润色模型自动翻成英文，最终喂给模型的必须是英文（文本编码器为 T5）。
- **不传 negative prompt**：模型没有对 negative prompt 做特殊训练，"效果应该也不会好"。
- 帧数随秒数：每秒 8 帧，满足 $n = 8s + 1$ ，即 6 秒 49 帧、5 秒 41 帧、4 秒 33 帧。
- 采样步数推荐 30–50 步（降到 25 步效果明显变差），guidance scale 2B 用 6、5B 可放到 7；固定种子（如 README 样例的 42 号）可复现官方视频。
- 生成 6 秒视频耗时参考（A100/H100、关闭显存优化）：字幕口径约 2B 90 秒级、5B 45–90 秒级（讲者现场表述略有交叠，按原样记录）；3060 开齐全部显存优化约 20 分钟一个视频，讲者直言消费卡只适合体验、不建议工业场景。

### 微调：SAT 现行、diffusers 将至，数据格式两者不同

当前官方路径是 SAT（Swiss Army Transformer）框架：4 卡以上 A100、ZeRO-2、纯 DP（每卡载完整模型，无 TP/PP），gradient checkpointing 必开（不开必 OOM），batch size 默认 1。数据量建议：LoRA 最低 25 条、推荐 69 条量级即可起步（会上用迪士尼早期动画样例数据集演示），SFT 建议 100 条。超参：LoRA 学习率约 `1e-3` 量级，SFT 约 `1e-5`（字幕口径）；save 每 500 步、eval 每 100 步；训练/验证集手动按 8:2 或 9:1 切（模板默认同集，不建议照抄）。两个实操坑：SAT 默认每次保存**完整权重**（单个 checkpoint 20GB 量级），需用官方 tools 脚本把 LoRA adapter 单独导出成 diffusers 的 `pytorch_lora_weights.safetensors` 才能在推理侧挂载；SAT 版推理本身显存也高（2B 16.8G、5B 26.5G），已被 diffusers 版取代为推荐路径。
全量微调 5B/I2V 按 SAT 路径要 16–32 张 A100（每卡 batch 1）才能不 OOM，讲者承认这是当前局限；diffusers 版微调代码已上线主仓库测试（需装 diffusers 0.31.0-dev 主分支），成本远低于 SAT、batch 可到 4，LoRA 可用但尚不能做 SFT，正式版随 diffusers 发版在一两周内跟进。

数据格式差异（实操最容易踩）：SAT 版是 `videos/` + `labels/` 两目录，每条视频 ≤6 秒、8fps 采到 49 帧，不足则复制最后一帧 pad 满；label 必须完整英文描述，T5 上限 226 token，保守建议约 200 词（512 token 未测）；diffusers 版改为单个 `prompt.txt`，每行与 `videos/` 下编号一一对应。I2V 微调无需单独准备图片：训练代码自动取视频第一帧作条件，因此**第一帧必须清晰**。

### 讲者自认边界与下一步

ControlNet 式可控生成还没做到——"目前微调只能做到与目标相似，达不到 ControlNet 那样的效果"，在 LoRA/diffusers 微调补齐后会作为首要目标；VAE 没有单独开源微调代码（团队内 VAE 与 Transformer 也是分开训的）；社区贡献方向讲者列了四项：微调示例与公开数据集（现有都只是 demo 级）、数据处理加速、推理加速方案、ComfyUI 类周边工具。开源协议提醒：2B 及 VAE 为 Apache，5B 与 5B-I2V 为 CogVideoX 自有协议。商用清影与开源版结构相同、原理一致，差异主要是原生分辨率（1440×960 vs 720×480，后者需超分追平）与训练数据，讲者称开源版加超分插帧等工程优化可做到与清影接近的效果。

## 关键数字总表（含来源）

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 训练 prompt 长度上限 | T5 编码 | 226 tokens（微调建议约 200 词） | 字幕 |
| VAE 峰值显存（优化后） | encode+decode 约 71G | 约 30G | 字幕 |
| Tiled VAE 单项收益 | 峰值 36G | 24G | 字幕 |
| 5B 推理最低显存 | 顺序 CPU offload + VAE 优化全开 | 4–5GB（BF16） | 字幕 |
| SAT 版推理显存 | 固定占用 | 2B 16.8G / 5B 26.5G | 字幕 |
| 帧数公式 | 8 fps | 6 秒 49 / 5 秒 41 / 4 秒 33 帧 | 字幕 |
| 推荐采样步数 | 低于 25 步效果明显变差 | 30–50 步 | 字幕 |
| guidance scale | — | 2B 取 6；5B 可取 7 | 字幕 |
| LoRA 起步数据量 | 最低 25 条 | 推荐 69 条量级；SFT 约 100 条 | 字幕 |
| LoRA / SFT 学习率 | — | 约 $10^{-3}$ / $10^{-5}$ 量级 | 字幕 |
| 训练硬件（SAT 微调） | ZeRO-2、batch=1 | 4+×A100；5B 全量微调 16–32×A100 | 字幕 |
| SAT 单个 checkpoint 体积 | 默认存完整权重 | 约 20GB（LoRA 需另导出 adapter） | 字幕 |

## 可迁移

- "给开源模型的显存预算"应作为设计约束而非上线后补丁：VAE decode 峰值这类非主干瓶颈决定了模型实际能被多少人跑起来；逐个模块测峰值、用 tiling/context parallel 把峰值砍下来，比整体量化更优先。
- 显存优化的会计学很实用：offload、tiling、slicing 每开一项都"拿时间换空间"且成倍叠加——给用户分档配置（全关最快、显存够用 .to(cuda)；全开最低显存），而不是一个默认打天下。
- 微调可复现性纪律：固定种子复现样例、明确 scheduler 与精度的配对（2B/DDIM/FP16 vs 5B/DPM/BF16）、数据格式与帧数公式写成机械checklist——这类"上手参数表"对任何要被社区微调的模型都适用，且都应标注来源版本，避免把旧口径当默认。

## 疑问 / 下一步

- Fake context parallel 的块间 2 帧重叠对长视频一致性的量化影响字幕未给（只给了显存收益），重叠帧数与运动剧烈程度的关系值得查 diffusers PR。
- 专用 caption 模型"一步到位"替代逐帧描述后，数据质量相对 GPT-4o 汇总管线的对照数字未在讲授中出现，可查 CogVLM2-Video 的发布说明。
- diffusers 版微调（尤其 SFT）当时尚未发布，讲者称一两周内跟进；今天复核时应先确认该路径的现状再决定微调方案。

## 原文金句（1-2句）

> 「（短提示词）不仅仅是很重要，这是一个必须的过程……大家如果使用了这种短提示词，我们的效果是非常差的。」——讲者谈提示词润色（字幕 25:43–25:53，按干净口径转写）

> 「（这些显存优化）都是一个拿时间换空间的行为，所以它的时间一定是在增加的，而且成倍数级的增加。」——讲者在介绍三行优化代码时（字幕 20:21–20:27 转写）
