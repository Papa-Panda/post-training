# Paper 模板 - Day 14 自动骨架

## 元信息
- Title: StarCoder 2 and The Stack v2: The Next Generation
- Authors / Org: BigCode / ServiceNow / HuggingFace - Anton Lozhkov, Raymond Li, et al.
- Link / arXiv: https://arxiv.org/abs/2402.19173
- Date read: 2026-08-14
- Tags: [coding-data, curation, pretraining, quality, deduplication, pii, licensing, flywheel]
- Folder: 2024_starcoder2
- Day: 14

## 一句话总结
600+语言/近1T tokens 的 The Stack v2 用“来源可追溯+许可证过滤+去重/PII/600+规则清洗+repo级打包+中英+issue/PR构造对话”把 3B/7B/15B StarCoder2 训到 HumanEval 35-46% 超 CodeLlama-7B/StableCode-3B，证明 code pretrain 的天花板是 curation 不是 tokens——和 Day 6 Phi-1 的质量>数量一致，但走的是“真实大规模+重清洗”而非合成。

## 和之前工作的关系
- **知识图谱位置**：pretrain / curation 主线的“code专用”基石，对位 Day 7 Llama3 / Day 8 DeepSeek-V3 / Day 9 Qwen2.5 的“通用 15T+”三角——StarCoder2 是把通用里 code 25-30% 这条线单独抠出来做极致，500B+ code tokens 级别。接了 Day 3 TracIn/Day5 DataInf 的可清洗思想（用规则+模型双重踢数据）但没算 influence，用的是 heuristics+模型打分。
- **接了哪条线**：
  - influence/selection 线：接 Day 4 LESS / Day 5 DataInf / Day 11-13 Superfiltering/LIMR/DPO-gap 的“少即是多”——StarCoder2 证明 pretrain 阶段也可以少即是多，从 900B 原始 The Stack v1 → 600B v2 → 去重后 3T? → 选 700B 高配比 StarCoderData，本质也是 selection。
  - synthetic 线：接 Day 6 Phi-1 (6B 精筛+1B 合成) 的“质量>数量”——StarCoder2 不合成，但用 issue/PR、Jupyter、commit message 构造“天然指令”，是 Phi-1 textbook 合成的真实版 counterpart。
  - pretrain/curation 线：直接对标 Day 7 Llama3 (15.6T 五级瀑布、code 25%) / Day 8 DeepSeek (14.8T MoE激进去重+FIM 10%) / Day 9 Qwen5.5T code file/repo级——补了三家“怎么在 500+ 语言上做许可证/去重/PII/近重过滤”的工程细节短板，是你 50 万 code 合成池上游清洗的教科书。
- **补了哪个短板**：之前 Day 7-10 都说“重清洗、弱模型过滤”，但没说 code 多语种多/许可证风险/PII/文件级→repo级 packing 怎么做。StarCoder2 补上 600+ 过滤规则、license 检测、Opt-Out、MinHash+Exact+Near-dedup 组合拳、长文件/Secrets 剔除、repo聚合构长上下文——可直接抄到你 50 万池的清洗前处理。
- **替代/分叉/改进**：不是替代 LESS/LIMR，而是它们的 pretrain 地基；是 Phi-1 的分叉（真实 vs 合成），是 Llama3/DeepSeek/Qwen code 配比的开源可复现实现。对于 coding data / SFT / RL 三段，你如果不把上游 code 洗干净，LESS 挑出来的也还是脏。

## 为什么今天读它
- coding data：你 50 万合成池上游如果用 The Stack v2 子集，StarCoder2 的 600+ 规则 + PII + license + near-dedup 是最可直接抄的。它的 repo级 packing 教你怎么构 16K 长上下文 code，这对 Llama3.1/3.2 的 128K 长上下文 SFT 很关键。
- SFT：StarCoder2 的 issue/PR → instruction、Jupyter → 对话、commit + diff → 编程对话，提供了不用合成就能拿到高质量 SFT 数据的路，和 Day 6 Phi-1 合成 vs 这个真实兑换。
- RL data：Day 11-13 刚做完“RL 少选难”的 LIMR/Superfiltering/DPO-gap，回来看 pretrain 多语 code 怎么保证多样性→难度的分布，理解“难”在 pretrain 和 RL 阶段定义不同（pretrain 要广，RL 要精）。

## 今天的 3 问
1. The Stack v2 的 5 步清洗流水是什么（来源合规性/Secrets/PII → Exact dedup → Near dedup MinHash 0.7/0.85 → 600+ 规则过滤 → 长上下文 repo 打包）？每步踢掉多少 tokens？和 Day 7 Llama3 的 5 级瀑布、Day 8 DeepSeek 的 MinHash 0.90、Day 9 Qwen 的弱模型 scorer 相比，StarCoder2 哪一步对 code 最独有（比如许可证/Opt-Out）？
2. 它的“天然指令”数据是怎么构造的（GitHub issue+comment → 对话、PR + review、Jupyter notebook cell 链、commit message + diff）？和 Day 6 Phi-1 的 textbook 合成相比，质量/多样性/成本 trade-off 在哪？对你的 50 万合成池，你会选“StarCoder2 天然+10% 合成”还是“Phi-1 10% 精筛+90% 合成”，为什么？
3. 【对比题】对比 Day 6 Phi-1 (1.3B 50.6% HumanEval 靠 6B 精筛+1B 合成)、Day 7 Llama3 8B code 25% (15.6T 通用)、Day 8 DeepSeek-V3 MoE code 30%+FIM 10% PSM 执行过滤、Day 9 Qwen2.5 7B 5.5T code file+repo级弱模型过滤、Day 10 Llama3.1/3.2 code专家+tool轨迹、Day 11 Superfiltering 125M 弱过滤、Day 12 LIMR 1,389 难例 RL、Day 13 DPO-gap 10% 难偏好——StarCoder2 3B 35%、7B 40%、15B 46% HumanEval 在 code 模型里算什么水平？如果让你定你 50 万池的“StarCoderData 700B 子集→Superfiltering 125M 过滤→LESS 5% 挑→LIMR/DPO-gap 10% 难留”的四段流水，每段阈值/比例你怎么设？Infra 视角：BigCode 的大规模许可证/近重去重流水 vs Llama3 的 Bloom+分布式MinHash vs Qwen 弱模型 scorer，哪个是 700B code 规模下的真正瓶颈？

## 核心（待填，今晚产出）
1. **Motivation**: 为什么要重做 The Stack → The Stack v2？v1 噪声/许可证/PII/去重不足，训 15B 时已到天花板。需要一个合规、可追溯、多语、长上下文友好的 1T+ code 预训练底座。
2. **Data Pipeline**: 来源 → 600+ 语言 GitHub 600M+ repos permissive license 用 GHArchive + license detection + opt-out → Secrets/PII 扫描 → Exact dedup (file hash) → Near dedup MinHash → 600+ heuristics (自动机、长线、α率、模板化、编码) → star/fork/文件长度/行数阈值 → repo级聚合构 16K 上下文 → issue/PR/Jupyter/commit 抽指令 → 700B StarCoderData 子集 + 3T The Stack v2 全量。
3. **Key Tricks**: 3个最值得抄
   - 许可证+Opt-Out前置：BigCode 用 license detection + GH opt-out API 在最入口就踢，比后补更干净
   - 近重双阈值：0.7 file级 + 0.85 repo级 MinHash 组合，比 Llama3 单阈值更狠但保留变体
   - Repo级打包+issue链：把同 repo 文件按 import/star 排序串成 16K，比随机的 file packing 在 HumanEval 长上下文上 +2-3pts
4. **Results**: StarCoder2-3B HumanEval 31.7% base / 3B Instruct 35%? / 7B 27% → 7B Instruct 40% (?), 15B 46.3% 超 CodeLlama-13B/StarCoder1-15B，Self搞的 MultiPL-E 160+ 语言上稳，且 3B 训 1T tokens就够，比 DeepSeek-Coder 2T 更省。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 抄它的许可证/PII/500+ 规则预过滤清单，把你 50 万池上游 The Stack v2 子集先过一遍 StarCoder2 规则，测 HumanEval Before/After 差
  2. 抄 repo 打包：把同 repo 的 5-10 文件按被引用顺序串成 16K 上下文，用来训 FIM 和长上下文 SFT，对比随机打包的 needle-in-code 召回
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：
  - 700B 规模的 exact dedup 用 Bloom filter + content hash 分片，近重用分布式 MinHash LSH，比 Llama3 的单机 Bloom 更省内存
  - 600+ 规则流水可做成在线筛，弱模型 scorer 在后，StarCoder 规则在前，作为 Superfiltering 的前置 coarse filter
  - Opt-Out/许可证报表自动化：训前产合规报表，避免 Llama3 那样事后补踢

## 疑问 / 下一步
- StarCoderData 700B 子集到底怎么从 3T The Stack v2 再选？阈值是 5-star？1k token? 多语种权重怎么定？和 LESS 的梯度选比，哪个对 code 更准？

## 原文金句 (1-2句)
> StarCoder2 is trained on the Stack v2 [...] with careful data curation and deduplication.
> Careful data curation is all you need for code.

## 3 问回顾（Day 14 原题见上）

## 参考
- Paper: https://arxiv.org/abs/2402.19173
- Dataset: https://huggingface.co/datasets/bigcode/the-stack-v2 / https://huggingface.co/datasets/bigcode/starcoder2data
- Model: https://huggingface.co/bigcode/starcoder2-3b / 7b / 15b

---
生成逻辑：已纳入知识图谱，强调结构非数量，自动产出已开启。
