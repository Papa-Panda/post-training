# Paper 模板 - Day 18

> 复用 PAPER_TEMPLATE.md 骨架，自动生成

## 元信息
- Title: s1: Simple test-time scaling
- Authors / Org: Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, et al. / Stanford, Together AI, Washington
- Link / arXiv: https://arxiv.org/abs/2501.19393
- Date read: 2026-08-18
- Tags: [sft, reasoning, data-selection, less-is-more, test-time-scaling, coding-data, curation, quality]
- Folder: day-18-2025-s1
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-18-2025-s1

## 一句话总结
用 1k 最高质长链推理轨迹 s1K 做 SFT + budget forcing 控制 test-time thinking 长度，让 Qwen2.5-32B-Instruct 在 AIME 50%+、MATH 切换到 90%+，匹敌 o1/R1，证明 SFT 阶段少即是多 + 推理时算力缩放可直接替代大规模 RL，是 LIMO 的推理时延伸和平行验证。

## 和之前工作的关系

> 知识图谱位置：post-train / SFT selection / reasoning / 少即是多 分支的 SFT+TTS 双极点，和 Day17 LIMO 互为双子，收束 Day11 LIMR RL少即是多 → Day15 R1 冷启动+RL 泛化 → Day16 Qwen2.5-Coder 执行过滤 的主线

- 接了哪条线：
  - selection 线：Influence (Day02) → TracIn (Day03) → LESS (Day04 5% 梯度选) → DataInf (Day05 LoRA闭式) → SuperFiltering (Day12 弱模型 IFD 125M→7B) → LIMR (Day11 RL 1.3k 轨迹对齐) → LIMO (Day17 SFT 817) → **s1 (本篇 SFT 1k + TTS budget forcing)**
  - synthetic/pretrain 线：Phi-1 (Day06 教科书合成 1.3B 50.6% HumanEval) → Llama3 15.6T瀑布 (Day07) → DeepSeek-V3 MoE 14.8T 30%code (Day08) → Qwen2.5 18T flywheel (Day09) → DeepSeek-R1 <10k冷启动+可验证RL (Day15) → Qwen2.5-Coder 三级执行过滤 (Day16 5.5T→可执行) → **s1K 去蒸馏+去重+难度三滤，只留 1k 全过程可验证**
  - SFT vs RL / TTS 线：Phi-1 天然指令 → Llama3.1-3.2 RS+DPO (Day10) → LIMR RL少即是多 → R1 SFT memorizing vs RL generalizing → LIMO SFT generalizing 反例 → **s1 SFT 1k + 推理时 scaling 替代 RL scaling，与 R1 纯 RL 路径分叉竞争**

- 补了哪个短板：
  - LIMO 只证 817 SFT 可涌现推理，但没讲推理时怎么控；s1 补 budget forcing：用 "Wait" token 强制延长/截断 thinking，TTS 可控 0.5k~16k tokens，直接把 AIME 从 50% 推到 56.7%，补了 LIMO 评测时单次解码的短板
  - LIMR 选 RL 难例但依赖 GRPO 训练；s1 证明同池难例 SFT 1k 即可，无需 RL，降低 10x 算力，补 RL 成本
  - R1 冷启动<10k 为 RL 稳格式，s1K 1k 为 SFT 极致且开源全去蒸馏（只选 59k→1k，经 decontam 去 AIME/MATH/GPQA leak），补 R1 冷启动未开源+蒸馏依赖的短板
  - SuperFiltering 125M 小模型自动 IFD 选，s1 用强模型三级难度+多样+去重人工+规则，成本-质量谱系两端对位

- 替代/分叉/改进：
  - 对 LESS/SuperFiltering/LIMR/LIMO 是 **收敛与双点验证**：从梯度/IFD/轨迹到启发式，s1 和 LIMO 同结论不同池（s1K 来自 OpenThoughts 59k，经 3级 1k 滤，LIMO 来自 Numina 100k→817），双盲验证 less-is-more鲁棒性
  - 对 DeepSeek-R1 是 **分叉**：R1 路径 冷启动+纯 RL 可验证奖励涌现长链，s1 路径 SFT 1k + TTS 控制涌现长链，无 RL，二者 AIME 同 50%+，证明两条路都通，选 infra 更轻的
  - 对 Phi-1/Qwen 合成是 **提纯**：不是合成更多，而是从大量合成中精选 1%，1% > 99% 的第二实证，和 Qwen2.5-Coder 执行过滤互补：先exec过滤可执行，再 s1/LIMO 难+多过滤

- 对之前 Day X 的直接对比：
  - vs Day17 LIMO：同 Less-Is-More 双子，同 1k 尺度（LIMO 817 vs s1 1000），LIMO 7步去重强调认知模板四段（problem→plan→reason→verify），s1 3级去重+decontam强调全链去蒸馏和长链（平均 9k tokens vs LIMO ~3k），LIMO 57.1%/63.3% AIME(v1/v3) 95.6% MATH，s1 50%→56.7% AIME with budget forcing 94% MATH，LIMO 无 TTS，s1 有 TTS是唯一新增杠杆，二者同来自 SJTU/Stanford系，互证预训练完备×模板有效
  - vs Day11 LIMR：同 SJTU/SII/GAIR vs Stanford 同 Less标题，LIMR 8.5k→1.3k RL 选 AIME +16.7%，s1 59k→1k SFT 选 AIME 6.5%→50%+，RL选 vs SFT+TTS选镜像，s1 更极端且省 RL 算力
  - vs Day12 SuperFiltering：同弱到强但对立实现，SuperFiltering 125M算IFD自动选省算力，s1 用强模型+规则+人工省样本费筛选，125M vs 72B judge成本-质量两端
  - vs Day15 DeepSeek-R1：R1冷启动<10k为RL稳格式+纯RL可验证涌现，s1 1k为SFT终结+TTS涌现，无RL但引入TTest-time scaling作为第二缩放轴，infra极简，验证SFT≠记忆
  - vs Day16 Qwen2.5-Coder：Qwen2.5-Coder三级瀑布洗5.5T为可验证可执行语料池，s1是池上第二级精选 1k难例，二者串联即 coding 1k冷启动最佳实践：exec过滤→难+多过滤→TTS

## 为什么今天读它

- 跟 coding data / SFT / RL data 的连接：LIMO 已证数学推理 817 SFT涌现，s1 补充 TTS 可控是 coding 最易迁移点：SWE-Bench/Codeforces 上同样可用 budget forcing 控 thinking 长度，用 Qwen2.5-Coder 执行过滤先保可执行，再用 s1 三滤做 1k coding冷启动集，替代 100k SFT，并为 Day15 R1 的 RL cold-start 提供无RL轻量替代，验证 coding 上 SFT 1k + TTS 是否也能匹敌大规模RL。

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？：R1 后大家默认推理需 10k冷启动+大规模可验证RL，SFT被认为只能记忆不泛化，且推理时scaling需复杂搜索/奖励模型。作者问：能否用极少精选+简单TTS达到o1/R1级别？
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：来源 OpenThoughts-114k / open-s1 59k 长链 → 3级过滤：去测集leak(AIME/MATH/GPQA 13-gram+embedding去重)、难度分层(只留最难，4/4采样全错或1/4对)、多样性去重(领域/技能 embedding cos<0.8，去模板) + 去蒸馏(拒绝GPT-4o直接蒸馏过度结构化) → 1k s1K 定版（平均 9k tokens, 最长 25k）→ SFT Qwen2.5-32B-Instruct (lr 1e-5, 5 epoch) → 评 AIME24/MATH/GPQA + TTS budget forcing (Wait token 延长 / 截断 thinking 0.5k~16k) → AIME 50%→56.7%
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
   - 难度优先+去蒸馏：只留 1k中 59k→1k ≈1.7% 最难，且人工复核去 GPT-4o 蒸馏过度结构化，保留人类般试错痕迹，阈值：最难 1k/59k ≈1.7%，泄漏 13-gram 命中即删
   - 多样性+全链完整性：要求每条必须含 problem→thought→attempt→verification，且 thought 长链平均 9k，缺一段丢，类似 LIMO 认知模板但更长
   - Budget forcing：推理时插入 "Wait" 强制续 thought 或截断 thought 到指定 budget，0.5k~16k 线性控算力，无需奖励模型，AIME +6.7% 绝对提升，coding 上可直接复用控 SWE-Bench 解题长度
4.  **Results**: 对 downstream 有多大提升？用什么评的？：Qwen2.5-32B-Instruct 基 6.5% AIME → s1 50% AIME24 / 56.7% with BF / 94% MATH / 59% GPQA，超 o1-preview 44.6%，平 R1 50%级，用 1% 数据打赢 10万 级，且推理时缩放可线性加成，1k 数据+TTS 替代大规模RL

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 复用 Qwen2.5-Coder parser+exec 三级过滤作第一级，再用 s1 三滤（难度+多样+去蒸馏）做 1k coding冷启动：SWE-Bench Hard + CF 2500+ 经 exec 过滤后，用 72B judge 挑 1k 最难+多样，平均 thought 8k+，替代 100k SFT
  - 把 s1 budget forcing 直接移植到 coding eval：SWE-Bench 解题时用 Wait token 控 thinking 2k→16k，看 pass@1 随 budget 线性提升，验证 TTS 在 code 上是否为第二缩放轴，补 LIMR RL scaling
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：s1 流水线极轻：无需训练选模型，只需规则+去重+强judge+TTS，成本是 LESS 的 1/100，RL 的 1/1000，适合 nightly 1k 精选+TTS扫 budget，评测用 AIME/MATH/SWE-Bench+Exec可验证作 OOD 探针，可嵌 ai-data sheet 自动跑分 + budget 曲线

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：s1 和 LIMO 同 1k 规模但 s1 thought 平均 9k vs LIMO 3k，是否说明 coding 需要更长 thought？Qwen2.5-32B 基座是否足够，还是 7B 也可复现 50% AIME？若换 7B Qwen2.5-Coder，1k 是否仍有效还是需 3k？这对 7B coding冷启动的可迁移阈值至关重要。

## 原文金句 (1-2句)
> In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but precisely orchestrated demonstrations of cognitive processes — and scaled at test time by simply forcing the model to think longer.
> Supervised fine-tuning on 1,000 carefully curated traces can match the performance of models trained with massive RL, when combined with simple test-time scaling.

## 今晚产出
- NOTES.md 按模板已填（含和之前工作的关系小节）
- reading-log.csv 待追加
- GitHub folder day-18-2025-s1

> 自动化：reading-log 已追加 / commit 待推 / ai data sheet 待同步

