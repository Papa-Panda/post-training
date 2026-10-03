# EP96 — Fast-dLLM v2：高效训练推理的块扩散大语言模型框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP96-fast-dllm-v2.html

## 元信息

- 期号：EP96（官网台账登记为 96 期；官网预告页正文自称「青稞Talk 第95期」，与台账编号有一位之差，本篇以 episode-index.csv 登记的 96 为准，差异待复核）
- 标题：Fast-dLLM v2：高效训练推理的块扩散大语言模型框架
- 讲者：吴成岳（香港大学 MMLab 博士生；据官网预告嘉宾介绍，为 Fast-dLLM 项目核心成员，且为论文第一作者 Chengyue Wu）
- BV：无（official-only，官网有预告、B站合集无对应视频）
- 时长：未知（官网预告 2025-12-09 20:00–21:00，未见回放存档）
- 提炼日期：2026-10-02
- 相关论文：Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, Enze Xie, *Fast-dLLM v2: Efficient Block-Diffusion LLM*，arXiv:2509.26328（2025-09-30 提交；机构：港大、NVIDIA、MIT），https://arxiv.org/abs/2509.26328
- 相关代码：https://github.com/NVlabs/Fast-dLLM
- 官网预告：https://qingkeai.online/blog/Fast-dLLM-v2

> 📝 提炼方式说明：本期为 official-only，B站合集无对应视频、无字幕可用，本纪要基于对应论文（arXiv:2509.26328）还原，**非逐字稿**；讲授环节的展开顺序、口头补充与 AMA 内容均无法确证，以下内容以论文为准，凡涉及「讲者观点」的表述均指论文作者在文中给出的判断。

## 一句话总结

Fast-dLLM v2 把预训练 AR 模型改造成块扩散 LLM：块间保持因果、块内并行去噪，只花约 10 亿 token 微调（全注意扩散模型 Dream 要 5800 亿 token，少约 500 倍）就无损保留 AR 质量，再配分层缓存实现相对标准 AR 解码最高 2.5 倍加速——讲者一方的定位是给大模型推理降本的一条务实路径（来源：arXiv:2509.26328）。

## 核心

1. **背景/问题**：AR 模型逐 token 串行解码、延迟下不来；扩散 LLM（dLLM）能并行预测多 token，但全注意双向结构难以用 KV cache、且从 AR 适配需要海量数据（如 Dream 从 Qwen2.5-7B 适配用 580B token 量级），落地性差。块扩散（block diffusion）是中间路线：块间自回归、块内扩散，可变长生成且块间能缓存，但此前只在小模型上验证过，没人证明能扩到现代 LLM 规模（来源：arXiv:2509.26328，§1、§2.2）。
2. **方法/设计**：在 Qwen2.5-Instruct（1.5B 与 7B）上 SFT 适配。训练侧三件套：块大小固定为 ， $D=32$ ，（子块大小 8），块内随机掩码 + 互补掩码（同一 batch 放掩码与其补两份视图，保证每个 token 在掩码/未掩码两种上下文都受监督）；token shift——被掩位置的预测用前一位的 logit，保持与 AR 的 next-token 表示兼容；注意力用「块内双向 + 块间因果」的混合掩码，并拼接带噪序列与干净序列让两份视图互相监督。推理侧：分层缓存——已解码块作只读前缀的 block-level KV cache，块内用 Fast-dLLM v1 的 DualCache（prefix+suffix 双缓存）支撑部分解码块的增量复用；块内用置信度感知并行解码，置信超阈值（默认 0.9）的 token 并行定版、其余留着继续 refine（来源：arXiv:2509.26328，§3.2、§3.3）。
3. **实验/实战**：训练用 LLaMA-Nemotron 后训练数据，64 张 A100，7B 模型 2500 步、约 12 小时。质量（论文 Table 1 综合平均分）：7B 版 60.3，高于同数据同类微调的 Qwen2.5-7B-Nemo-FT（59.6）与 Dream（57.6），多个单项（HumanEval 63.4、GSM8K 83.7）超原始 AR 基线；1.5B 版平均 45.0 同为同档最优。速度：GSM8K 上置信阈值取 0.9 时吞吐从 39.1 升到 101.7 token/s（2.6 倍）而精度仅微降；批量解码在 A100（batch 64）达 1.5 倍、H100 达 1.8 倍；官方口径总结为相对标准 AR 解码最高约 ， $2.5\times$ ，的吞吐提升且 GSM8K 精度相对优化过的 LLaDA 基线（Fast-dLLM-LLaDA）高 +5.2 个百分点（来源：arXiv:2509.26328，§4.1、§4.2、图 1）。
4. **结论/观点**（论文口径）：作者强调这条路线的卖点不是峰值速度而是「适配成本」——后训练式地把现成 AR 权重转成可并行解码的块扩散模型，数据成本只有全注意扩散路线的约 1/500，质量无损，是 dLLM 走向实际部署的关键一步。

## 关键数字

| 指标 | 基线 | 结果 | 来源 |
|---|---|---|---|
| 适配训练数据量 | 全注意扩散 LLM Dream：约 580B token | Fast-dLLM v2：约 1B token（少约 500 倍），质量无损 | arXiv:2509.26328，摘要与 §1 |
| 端到端解码加速 | 标准 AR 解码 | 最高 ， $2.5\times$ | arXiv:2509.26328，摘要 |
| GSM8K 吞吐（置信阈值 0.9） | 39.1 token/s | 101.7 token/s（2.6 倍），精度仅微降 | arXiv:2509.26328，§4.2 图 4 |
| 批量解码吞吐加速 | AR 基线（同批大小） | A100 batch 64：1.5 倍；H100：1.8 倍 | arXiv:2509.26328，§4.2 图 5 |
| 综合平均分（7B） | Qwen2.5-7B-Nemo-FT 59.6；Dream 57.6 | Fast-dLLM v2 7B：60.3 | arXiv:2509.26328，Table 1 |
| 训练成本 | — | 64×A100；7B：2500 步约 12 小时（1.5B 约 8 小时） | arXiv:2509.26328，§4.1 |

## 可迁移

- 「低成本适配现成检查点」是 post-training 的一种通用算式：v2 只用约 1B token 后训练就把 AR 模型改造成新解码范式而不掉点——和 RL/SFT 管线里把基座模型改造成新行为（工具调用、结构化输出）是同一件事，关键是新旧范式之间的接口兼容设计（这里是 token shift + 块间因果，让 AR 表示不被破坏）。在 RL rollout 侧，块内并行 + 置信阈值提前定版、与 speculative decoding 是同族的速度/质量旋钮，可按任务难度调阈值换吞吐。
- 分层缓存（块级只读前缀 KV + 块内双缓存）对长 rollout 服务有直接启发：agent 轨迹里「已定版的历史」与「正在生成的当前块」计算特征不同，分开缓存、只读化历史，既降重复计算又避免近似缓存与原计算不等价这类隐患——论文明确指出 v1 式近似 KV cache 不等于原计算、是他们要绕开的坑（来源：arXiv:2509.26328，§1）。

## 疑问 / 下一步

- 论文只验证到 7B；块扩散适配在更大模型（70B+）和长链推理（如长 CoT 数学/代码）上的质量-速度曲线、置信阈值对不同任务的最优取值（论文消融显示最优子块大小任务相关），尚无系统结论——若用于生产推理栈，还需与连续批处理（continuous batching）等服务层机制的兼容性验证。

## 原文金句（1-2句）

> （论文原句，arXiv:2509.26328 §1）"transforms pretrained autoregressive (AR) models into diffusion-style decoders for parallel text generation ... with only about 1B tokens of fine-tuning—achieving lossless adaptation without retraining from scratch"

> 说明：本期无字幕，上引为论文原文，非讲者口述。
