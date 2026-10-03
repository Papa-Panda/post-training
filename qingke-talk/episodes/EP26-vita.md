# EP26 — VITA：开源交互式多模态基础大模型

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP26-vita.html

> **主旨引文**（论文 arXiv:2408.05211，摘要）：「VITA, the first-ever open-source Multimodal Large Language Model (MLLM) adept at simultaneous processing and analysis of Video, Image, Text, and Audio modalities, and meanwhile has an advanced multimodal interactive experience.」

## 元信息

- 期号：EP26（官网期号 26；青稞Talk 第 26 期）
- 标题：VITA：开源交互式多模态基础大模型
- 讲者：傅朝友（Chaoyou Fu；南京大学智能科学与技术学院研究员、助理教授、博导；VITA 第一作者）
- 讲者来源：[节目页 qingkeai.online/blog/MVlAV8qB](https://qingkeai.online/blog/MVlAV8qB)（嘉宾介绍为公开署名，明示「VITA 第一作者」；论文作者表 Chaoyou Fu 居首位）
- 观看/提炼日期：2026-10-02
- 直播时间：2024-10-14 19:00–20:00（节目页）
- BV：official-only（官网有预告，B站合集无对应视频，台账 `episode-index.csv` 著录）
- 相关论文：[VITA: Towards Open-Source Interactive Omni Multimodal LLM](https://arxiv.org/abs/2408.05211)（arXiv:2408.05211，v1 2024-08-09 提交，v3 2025-05-30 修订；本期于 2024-10-14 直播，讲解对应的应为 v1 一代，数字采 v1 正文）
- 相关代码：[github.com/VITA-MLLM/VITA](https://github.com/VITA-MLLM/VITA)（节目页成果链接），项目页 https://vita-home.github.io/
- 提炼方式：**无视频、无字幕，基于对应论文还原，非逐字稿。** 数字、结论均标注论文出处；未确证者见文末清单。

## 一句话总结

开源社区第一次同时把「四模态端到端理解」和「GPT-4o 式自然交互」塞进一个模型：VITA 以 Mixtral 8 $\times$ 7B 为语言底座扩中文词表、三阶段训练灌入图像/视频/音频能力，再用状态 token 教模型端到端区分「查询音频 / 噪声音频 / 文本查询」，配合双工（duplex）双模型部署实现免唤醒交互与语音打断——理解面做到开源前列，但论文自陈与闭源仍有差距，尤其视频理解与端到端语音输出。

## 核心

1. **背景/问题**：GPT-4o 把标杆立在两件事上——端到端多模态理解 + 自然的人机交互；当时开源 MLLM 多数只有图-文两种模态，且交互仍靠唤醒词或按键，生成期间不能被打断（论文 §1 列为两条硬限制）。VITA 的命题是开源侧同时补这两课：同时处理视频、图像、文本、音频，并让交互像真人对话一样不需要唤醒、可随时插话。
2. **方法/设计（三阶段训练）**：(a) **语言底座双语化**——Mixtral 8 $\times$ 7B 中文能力弱，先把词表从 32000 扩到 51747，再用 500 万条双语合成语料做纯文本指令微调（论文 §3.1）；词表扩容同时减少同文本 token 数，推理更省。(b) **多模态对齐**——视觉侧 InternViT-300M-448px 编码器 + 两层 MLP connector（448 $\times$ 448 图输出 256 token，高清图用动态切分，视频按时长 4/16 帧采样）；音频侧 Mel Filter Bank + 4 $\times$ CNN 下采样 + 24 层 Transformer（341M 参数），每 2 秒音频编码为 25 token（论文 §3.2）。(c) **多模态指令微调**——约 275 万条拼接样本（Table 1 口径，含图文、视频、OCR 与合成数据），约一半问题用 TTS 合成语音版本，让模型同时学文本指令与语音指令；关键创新是**状态 token**：<1> 查询音频、<2> 噪声音频、<3> 文本查询，训练时插在回答开头让模型自己判别输入类型（论文 §3.3）。噪声负样本的构造很讲究：从既有 QA 的回答里随机取 474K 句（非问题、无需回应的话）合成音频；且不能直接训 EOS（会伤性能），改为让 LLM 对噪声文本生成回复作训练目标、部署时把 <2> 当作另一个 EOS（论文 §3.3.2）。
3. **交互实现（部署层）**：免唤醒靠两段配合——SileroVAD 实时追踪环境声音先判「有人声」，状态 token 再判「是不是问我的」（论文 §3.4.1）。打断靠**双工（duplex）方案**：两个 VITA 模型同时部署，一个生成回答、一个监视环境；监视方一旦识别到有效查询音频，生成方暂停、监视方接管历史上下文回答新问题，两个模型随机换位（论文 §3.4.2）。工程上适配了多模态 vLLM 承载推理。注意 TTS 仍是外挂工具把文本转语音，非端到端语音生成——论文自陈这是限制（论文 §5）。讲演提纲在官网预告里把这部分列为「非唤醒交互和语音打断交互的实现」。
4. **实验/实战（论文口径）**：语言——扩词表 + 双语指令微调后，中文评测 C-EVAL 53.30 → 56.68、AGIEVAL 41.72 → 46.17，MMLU 基本持平（70.35 → 70.98），GSM8K 反升 63.99 → 75.66（论文 Table 3）；语音 ASR——WenetSpeech CER test_net 12.15 / test_meeting 16.53，Librispeech WER test_clean 8.14（论文 Table 4）；图像理解在 MME、OCRBench、HallusionBench 上压过同期开源 LLaVA-NeXT、接近 Gemini 1.5 Pro，Video-MME 上强于 Video-CCAM 但仍让于 LLaVA-NeXT-Video（论文 §4 与 Fig. 5 定性结论；图内具体分数未能从论文文本层提取，见未确证项）。

## 关键数字

| 指标 | 数字 | 来源 |
|---|---|---|
| 语言底座 | Mixtral 8 $\times$ 7B（SMoE）；词表 32000 → 51747 | 论文 §3.1 |
| 语言微调语料 | 500 万条双语合成语料 | 论文 §3.1 |
| 语言基准 C-EVAL / AGIEVAL（Mixtral 官方 → VITA 底座） | 53.30 → 56.68 / 41.72 → 46.17 | 论文 Table 3 |
| 语言基准 MMLU / GSM8K | 70.35 → 70.98 / 63.99 → 75.66 | 论文 Table 3 |
| 视觉编码器 | InternViT-300M-448px，单图 256 token | 论文 §3.2 |
| 音频编码器 | 341M 参数；每 2 s 音频 → 25 token | 论文 §3.2 |
| 多模态指令数据 | 约 275 万条拼接样本（Table 1 总计 2749.9K） | 论文 Table 1 |
| 噪声音频负样本 | 474K 句（取自既有 QA 回答） | 论文 §3.3 |
| ASR：WenetSpeech CER（test_net / test_meeting） | 12.15 / 16.53 | 论文 Table 4 |
| ASR：Librispeech WER（test_clean） | 8.14 | 论文 Table 4 |
| 图像理解（定性） | 优于 LLaVA-NeXT、接近 Gemini 1.5 Pro | 论文 §4 / Fig. 5 |
| 视频理解（定性） | 优于 Video-CCAM，逊于 LLaVA-NeXT-Video | 论文 §4 / Fig. 5 |

## 可迁移

- **「交互原语进训练目标」这条思路可搬到 agent 产品化**：VITA 没在推理链路上写规则，而是在 SFT 目标里加状态 token、让模型端到端判别「要不要回应」——同类问题（该不该打断、该不该追问、该不该澄清）在 agent 产品里通常被后处理掉，把判别做成一等公民的训练信号比规则编排更可扩展；噪声负样本构造也给了模板：负样本从「真实会出现的非请求内容」采样（已有 QA 回答）、且标签不能粗暴截断（训练里改监督目标、部署时改控制逻辑）。
- **双工部署是产品层模式，不只是 VITA 的技巧**：生成进程与监视进程互换身份，本质是把「可中断生成」做成部署拓扑而非模型能力；在 streaming agent、实时语音/视频助手、代码 copilot 的长生成场景里可借鉴：监视方持同一上下文、可随时接管。成本侧要记一笔：双工意味着常驻两份模型权重。
- **多模态 token 预算意识**：每 2 秒音频 25 token、单图 256 token、视频帧数按时长分档——交互系统对每种模态的 token 成本都要有显式预算，视频数据不参与拼接训练也是同一逻辑（论文 §3.2）。这套「预算先行」对多模态数据的长度统一与打包（6K 拼接）是直接可抄的工程细节。

## 疑问 / 下一步

- 视频理解只给定性结论（强于 Video-CCAM、逊于 LLaVA-NeXT-Video、与闭源差距大），Fig. 5 的具体分数文本层取不到；想核数字需看论文 PDF 图表原图或官方项目页。
- 讲演时间（2024-10-14）在 v1（2024-08）之后、v3（2025-05）修订之前，v1 表述与直播口径大概率一致，但「v0.2 式增量」无法排除——本文的 v1 归属是时间推断，见未确证项。
- VITA 的语音输出仍是外挂 TTS；论文称端到端语音生成是后续工作。开源节奏上，VITA 之后的多模态交互模型（端到端语音输出方向）值得一条跟进口径。

## 原文金句（论文原话，非讲者语录）

> "By contrast ... VITA supports these modalities end-to-end. ... Non-awakening Interaction: VITA automatically filters background noise like non-query human voices ... Audio Interrupt Interaction: If the user interrupts with another question, the generation process is paused, and the model immediately responds to the latest query."（论文 §1，概述）

> "While there is still lots of work to be done on VITA to get close to close-source counterparts, we hope that its role as a pioneer can serve as a cornerstone for subsequent research."（论文摘要）

## 未确证项

1. **无视频、无字幕**：本期为 official-only（官网有预告，B站合集无对应视频）；全文基于论文 arXiv:2408.05211（v1）与节目页，讲演现场内容、问答、演示细节均未覆盖。
2. **版本时序推断**：论文 v3 修订于 2025-05-30（讲演之后近 7 个月），本文按直播时间归属 v1 一代；若讲演当天已有更新口径，无法从公开材料复原。
3. **讲者身份**：节目页明示讲者为「VITA 第一作者」傅朝友（南京大学），论文作者表 Chaoyou Fu 居首位，两者互洽；未另行独立核对个人主页，以节目页为准。
4. **Fig. 5 分数**：MME / OCRBench / HallusionBench / Video-MME 的具体数值在论文 HTML 文本层无法读取到（图中），纪要只记论文正文的定性表述；数值核验需查 PDF 或官方项目页。
5. **扩容后的推理效率**：论文称扩词表可减少同文本 token 数并提效，但未给量化数字，未采录。
