# Paper 模板 - Day 17

> 复用 PAPER_TEMPLATE.md 骨架，自动生成

## 元信息
- Title: LIMO: Less is More for Reasoning
- Authors / Org: Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu / SJTU, SII, GAIR
- Link / arXiv: https://arxiv.org/abs/2502.03387
- Date read: 2026-08-17
- Tags: [rl-data, sft, reasoning, data-selection, less-is-more, coding-data, curation, quality]

## 一句话总结
用仅 817 条精挑的复杂推理样本 SFT，Qwen2-32B-Instruct 在 AIME 57.1%→63.3% (v3) 和 MATH 94.8%→95.6%，打赢 10 万+样本训的模型，提出 LIMO 假说：预训练已编码知识时，极少但精的“认知模板”即可唤醒复杂推理，挑战 SFT=记忆的常识。

## 和之前工作的关系

> 知识图谱位置：post-train / SFT selection / reasoning / 少即是多 分支的 SFT 极点，和 LIMR (Day 11 RL 极点) 互为姐妹篇，共同收束 Phi-1→Llama3→Qwen2.5→DeepSeek-R1→Qwen2.5-Coder 的合成/执行过滤主线

- 接了哪条线：
  - selection 线：Influence (Day02) → TracIn (Day03) → LESS (Day04 Adam感知梯度选5%) → DataInf (Day05 LoRA闭式) → SuperFiltering (Day12 弱模型IFD) → LIMR (Day11 RL轨迹对齐选1389难例) → **LIMO (本篇 817 SFT难例)**
  - synthetic/pretrain 线：Phi-1 (Day06 教科书合成) → Llama3 15T瀑布 (Day07) → DeepSeek-V3 MoE 14.8T 30%code (Day08) → Qwen2.5 18T flywheel (Day09) → DeepSeek-R1 <10k冷启动+可验证RL涌现 (Day15) → Qwen2.5-Coder 三级执行过滤 (Day16)
  - SFT vs RL 对比线：Phi-1 天然指令 → Llama3.1-3.2 多轮RS+DPO (Day10) → LIMR RL少即是多 → R1 SFT memorizing vs RL generalizing → **LIMO SFT generalizing 反例**

- 补了哪个短板：
  - LIMR 只证 RL 少即是多，LIMO 补 SFT 少即是多，且 817 < 1389 更极端
  - R1 说 SFT 记忆、RL 泛化，LIMO 说精心挑的 SFT 也能泛化（OOD +40.5% 跨10基准），补 R1 的 SFT 刻板印象
  - SuperFiltering 弱到强选但没讲“认知模板”设计，LIMO 补了模板四原则：难度分层、过程完备性、去重多样性、长链可验证

- 替代/分叉/改进：
  - 对 LESS/SuperFiltering/LIMR/DPO-Gap 是 **收敛与提纯**：从梯度/IFD/轨迹/gap 各种代理 → 回归到“难+多样+全链路”的可解释启发式，7 维过滤规则可直接抄
  - 对 Phi-1/Qwen 合成是 **替代**：不是合成更多，而是从大量合成候选中精选 1%，1% > 100% 的实证
  - 对 R1 冷启动是 **互补**：R1 冷启动 <10k 为 RL 稳格式，LIMO 817 为 SFT 极致，证明两阶段都可极少样本

- 对之前 Day X 的直接对比：
  - vs Day11 LIMR：同门同机构（SJTU/SII/GAIR），同 Less-Is-More 标题，LIMR 8.5k→1.3k RL 选难例 AIME +16.7%，LIMO 100k→817 SFT 选难例 AIME 6.5%→57.1%/63.3%，RL 选 vs SFT 选的镜像实验，样本都来自同一 MATH/AIME 池但 LIMO 更强调去 leak + 过程模板
  - vs Day12 SuperFiltering：都弱到强，但 SuperFiltering 125M 模型算 IFD 自动选，LIMO 用强模型+规则（难度/多样/去重）人工+自动化，SuperFiltering 省算力，LIMO 省样本但费筛选，成本-质量曲线两端
  - vs Day13 DPO-Reward-Gap：都挑难，DPO-Gap 留 gap 小的 10% 偏好对，LIMO 留最难的 817 且要求解题过程完整，gap 是隐式奖励，LIMO 是显式长链，RLHF vs SFT 殊途同归
  - vs Day15 DeepSeek-R1：R1 证明纯 RL 可涌现长链，LIMO 证明纯 SFT 也可涌现长链 (SFT≠记忆)，但 LIMO 依赖 Qwen2-32B-Instruct 强预训练基座，验证基座Completeness×模板有效性 二因子假说
  - vs Day06 Phi-1 / Day16 Qwen2.5-Coder：Phi-1 书本合成 1B+6B，Qwen2.5-Coder 执行三级瀑布 5.5T，LIMO 反向：不扩量只提质，1% > 100%

## 为什么今天读它

- 跟 coding data / SFT / RL data 的连接：LIMO 的 817 虽是数学推理，但其 4 原则（难例、多样、去重、完整推理链）可直接平移到 coding data：挑 hard coding problems（AIME 难度的 Codeforces / SWE-Bench Hard）、完整 solution trace、去 repo 重复、执行可验证，复用 Qwen2.5-Coder 的 parser+exec 过滤作为第一级，再用 LIMO 难度+多样做第二级，做出 1k 级的 coding RL 冷启动集，替代当前动辄 100k SFT 的做法

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？：业界默认复杂推理需 100k+ SFT，SFT 被认为只会记忆不泛化，R1 之后大家转纯 RL。作者挑战两点：大量数据是否必要？SFT 是否只能记忆？
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：来源 NuminaMath/AIME/MATH等 10万池 → 7 步过滤：去 leak(AIME/MATH测集n-gram去重)、难度分层(只留最难)、多样性(领域/技能去重)、长链完整性(需含完整reasoning)、去模板化(拒绝过度结构化)、质量复核(强LLM judge)、817定版 → 直接 SFT Qwen2-32B-Instruct (lr 1e-5, 15 epoch, no RL) → 评 AIME/MATH + 10 OOD 基准
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
   - 难度优先：只留模型 4 次采样全错或 1/4 对的最难，抛弃简单，阈值：最难 817 / 100k ≈ 0.8%
   - 多样性去重：领域去重（代数/几何/组合/数论）+ 技能去重（n-gram embedding cos <0.8）+ 去测集 leak（13-gram 命中即删）
   - 认知模板完整性：要求每条必须包含 problem → plan → stepwise reasoning → final verification 四段，缺一段即丢，类似 Qwen2.5-Coder 的执行完整性检查
4.  **Results**: 对 downstream 有多大提升？用什么评的？：Qwen2-32B-Instruct 基 6.5% AIME / 59.2% MATH → LIMO 57.1%/63.3% AIME(v1/v3) / 94.8%/95.6% MATH，超 NuminaMath 100k SFT 模型，OOD 10 基准 +40.5%~45.8% 绝对提升，1% 数据打赢 100x 数据

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 用 LIMR 轨迹对齐 + LIMO 难度/多样/完整性 4 原则，做一个 1k coding 冷启动集：SWE-Bench Hard + Codeforces 2500+，经 Qwen2.5-Coder 执行过滤后，用 Qwen2.5-72B judge 挑 817 条最难+多样
  - 把 DPO-Gap 的“小 gap=难例”与 LIMO 的“全错=难例”做 ensemble，RL 前先用 LIMR/LIMO 双过滤，再进 DAPO/GRPO，验证 coding 上 SFT 1k 是否也能涌现长链
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：LIMO 流水线极轻：无需训练选模型，只需规则+去重+强 judge，成本是 LESS 的 1/100，适合 nightly 小批量精选；评测用 AIME/MATH + Exec 可验证作为 OOD 探针，可嵌入 ai-data sheet 自动跑分

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：LIMO 假说二因子中“预训练知识完备性”阈值到底是多少？32B Qwen-Instruct 已够，7B 是否可复现？若换成 7B Llama3.1，817 是否仍有效，还是需要 3k？这对 coding 7B 冷启动的可迁移性至关重要

## 原文金句 (1-2句)
> In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but precisely orchestrated demonstrations of cognitive processes.
> SFT does not necessarily memorize — when curated as cognitive templates, 1% can beat 100%.

## 今晚产出
- NOTES.md 按模板已填（含和之前工作的关系小节）
- reading-log.csv 待追加
- GitHub folder day-17-2025-limo

> 自动化：reading-log 已追加 / commit 待推 / ai data sheet 待同步

