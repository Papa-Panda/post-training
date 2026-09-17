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

---

## 第二轮复习（2026-09-17）

> 本轮复核：arXiv:2502.03387 v3（7/29 修订）摘要 + export.arxiv 全文 + Hugging Face GAIR/LIMO 模型卡逐项核验。修正初读 NOTES 四处：① 基座模型不是 "Qwen2-32B-Instruct"，是 **Qwen2.5-32B-Instruct**（HF Model Zoo 明确 backbone: Qwen2.5-32B-Instruct）。② "来源 NuminaMath/AIME/MATH 等 10 万池" 不准确：论文初始候选池是 **tens of millions of problems**（NuminaMath-CoT、AIME 历史题、MATH 等多源组池）；"100k" 是对比基线 NuminaMath-100k 的规模，不是候选池。③ "7 步过滤（4 次采样全错或 1/4 对）" 过度具体化：论文实际管线是多源组池 → Qwen2.5-Math-7B-Instruct baseline 难度粗筛 → R1 / R1-Distill-Qwen-32B 等强模型多采样、成功率低于阈值才保留 → 领域/复杂度均衡的战略采样；没有披露 "4 采样" 的具体阈值数字，也没有 "problem→plan→reason→verify 四段" 的硬性格式要求——论文对 chain 的要求是 detailed、logically coherent、verify intermediate conclusions（见 RQ1 的 L5/L1 分级实验），四段是复核者的转述。④ 数字口径：57.1% AIME / 94.8% MATH 是 **v1**，63.3% / 95.6% 是 **v3**（7/29 修订摘要），OOD 绝对提升 40.5%（v1）→ 45.8%（v3）同样是版本差，不是两个实验。

### 1. 核心命题

Day 17 真正解决的 data 问题，不是"推理数据不够多"，而是：**SFT 阶段的数据到底在"教"什么**。当预训练已经把领域知识编码完备时，复杂推理不是从 SFT 数据里"学"出来的，而是被"唤醒"的——数据的作用从"知识载体"降维成"认知过程模板"。

R1 建立了一个刻板印象：SFT = memorization（dense-but-capped，把分布压到示范的几个 mode 上），RL = generalization（稀疏但无上界的 verifiable reward）。LIMO 的反命题是：**SFT 也可以泛化，前提是数据被当作"过程模板"而非"知识条目"来策展**。LIMO 假说的两因子——(1) 预训练知识完备性 × (2) 模板有效性——把"少即是多"的成立条件写成了条件命题：因子 (1) 是门槛，因子 (2) 是杠杆。

### 2. 图谱位置

- **直接前驱 Day23 LIMA（对齐极简 → 推理极简，重点直接对比）**：LIMA 用 1k 人工策展 SFT 证明"对齐可以很薄"（Superficial Alignment Hypothesis）；LIMO 用 817 证明"推理 elicitation 也可以很薄"。两者同 1k 尺度、同纯 SFT、同强基座（LLaMA-65B vs Qwen2.5-32B），但**选择器哲学完全对立**：LIMA 信人的品味——人工三标准（来源质量/风格统一/任务多样性，StackExchange/wikiHow/Reddit），选"输入好"；LIMO 信模型的失败率——cheap-to-expensive 难度级联（7B 粗筛 + R1 级多采样阈值），选"模型做不出来"。一个向外看（数据本身好不好），一个向内看（模型缺什么）。两者共享同一个未被测量的软肋：两因子里的"预训练完备性"都没度量——LIMA 的 65B 和 LIMO 的 32B 都是强基座，换 7B 会怎样，两篇都没回答（呼应初读疑问）。
- **镜像姐妹 Day11 LIMR（RL 选 vs SFT 选）**：同门 SJTU/SII/GAIR，同 Less-Is-More 标题。**关键分野是时序**：LIMR 是"先付全款再提纯"——8,523 题全量 RL 跑完，记轨迹，用 LIM 对齐度留 1,389（第二轮提纯法）；LIMO 是"先提纯再训练"——tens of millions 离线过滤到 817 再 SFT（第一轮预选法）。LIMO 便宜 100x（选数据不需要训练），但 LIMR 的评估器是真实 rollout 信号，LIMO 的评估器是"强模型失败率"这个代理信号。代理 vs 真实，是便宜和可信的 trade-off。
- **上游替代 Day04 LESS / Day12 SuperFiltering（选择哲学的"返祖"）**：LESS 用目标梯度相似（任务参照、算法化），SuperFiltering 用 125M 弱模型算 IFD（自动化、weak-to-strong）；LIMO 退回可解释启发式（难度 × 多样 × 去分布偏）。这不是退步：9/9 Q&A 情形三已证明，梯度代理只在正确切空间里有效——推理任务的"影响力"用梯度算，参照系本身可疑。**当算法代理不可靠时，诚实的启发式 > 精致的伪算法**，这是选择线的一条元教训。
- **后继/双子 Day18 s1（双盲互证）**：同 1k 尺度（817 vs 1000），LIMO 不控 TTS，s1 用 budget forcing 把 TTS 变成第二缩放轴。两者同结论（1% 打赢 100x）、不同池（tens of millions→817 vs OpenThoughts 59k→1k）、不同机构——这是 less-is-more 最强的证据形态：**独立复现**。

### 3. 机制深挖

**(a) 三条选择标准的拆解。** 论文的选择标准不是"难+多"两词能概括的，第三条最被低估：

1. **Complexity（难度 = 强模型的失败率）**：Qwen2.5-Math-7B-Instruct 先做 baseline 粗筛（便宜地砍掉"显然太简单"的），再用 R1 / R1-Distill-Qwen-32B 等最强模型多采样、成功率低于阈值的才保留。形式化： $d(x) = 1 - \mathrm{pass@k}(x; \theta_{\mathrm{strong}})$ ，留 $d(x)$ 最高的那一档。这是 cheap-to-expensive 级联，和 Day16 的"AST 全量 + sandbox 分诊"同构，只是分诊依据从"可执行性"换成了"失败率"。注意它和 DPO-Gap（Day13）的亲缘：都是"用模型行为定义难度"，不是用人工标签。
2. **Knowledge diversity（战略配额，不是后过滤）**：领域/技能均衡采样，避免概念冗余。对应 Day20 DEITA 的多样性因子和 Day24 D4 的原型剪枝，但 LIMO 的多样性是**采样时的配额**，不是训后的去重——先定"代数/几何/组合/数论各占多少"，再往里填最难的题。
3. **Deviation from training distribution（反记忆设计）**：选偏离模型训练分布的问题。这条直接解释了下面 (c) 的"毒数据"现象：和预训练分布太近的题，SFT 训的不是推理，是"背答案"——SFT 的 cross-entropy 会把"回忆"也当成正确行为强化。选"没见过的"，逼模型用推理而非回忆。这是 LIMO 对"去重"概念的升级：Day24 的去重是"训练集内部别重复"，LIMO 的去分布偏是"**别和预训练分布重复**"。

**(b) 推理链质量的实验（RQ1，§6.3.1）——"模板有效性"是有实验支撑的。** 同一题、不同质量的 chain 做对照：最高质 L5 vs 最低质 L1，AIME24 差 **15pp**，MATH500 差 **12pp**。这个数字的量级值得细品：选"题"的 817-vs-100k 对比带来几十个点的提升，但**选"链"在同一题上就能差出 15pp**——链的质量是独立的一维杠杆。这是 LIMO 对 LIMA "风格统一" 的实质升级：LIMA 要的是回答风格统一（表面），LIMO 要的是 chain 详细、逻辑连贯、中间结论被验证（过程）。对 coding 的直接启示：SWE-Gym 的轨迹数据（Day29）里，"修对了"的轨迹和"修对且过程干净"的轨迹，可能是 15pp 级的差距——**Day29 的"轨迹"概念需要 LIMO 式的质量分级**。

**(c) "1% > 100%" 的机制：uncurated 数据是负资产。** 论文的对比实验：NuminaMath-100k 训练后 **32.3% vs 基座 49.9%**——100k 数据把模型训**退化**了；OpenThoughts-114k 58.3% vs LIMO 平均 78.1%（"unfocused problem selection"）。机制：低质 chain 里的错误推理步骤是"有毒示范"，SFT 的 dense 信号不分青红皂白地把错误过程也压进分布。所以 LIMO 的真正命题不是"少即是多"，是"**毒即是少**"：先保证无毒，再谈多少。这和 Day16 的执行门禁同构：Qwen 用 sandbox 拦"跑不通的"，LIMO 用难度+链质拦"想错了的"——都是**否定式选择**（先定义什么不能进），而不是肯定式选择。

**(d) OOD 测量的诚实度。** 10 个 benchmark，v1 40.5% → v3 45.8% 绝对提升（摘要口径），AMC23 +51.4、CHMath +64.2 很漂亮；但 HF 表格里 Minerva 44.9 vs 47.1（**-2.2**）、GPQA 66.7 vs 73.3（**-6.6**）——OOD 增益**不均匀**，科学知识类反而降。这不是瑕疵，是证据：见 §4 边界第 2 条。

### 4. 边界与反例

1. **"预训练完备性"不可观测**：两因子里因子 (1) 没有度量、没有阈值。817 在 Qwen2.5-32B 上有效，换 7B 基座需要 3k 还是 10k？论文没做这个消融。**LIMO 假说是个条件命题，不是普适命题**——"完备性"这个前提本身是黑箱，假说在前提不满足时静默失效，这是它最脆弱的环节。
2. **选难 = 选窄**：只留最难 → 池子向竞赛数学极端倾斜；Minerva（-2.2）、GPQA（-6.6）低于 previous SOTA 就是分布偏置的收据。coding 迁移时若只留 Codeforces 2500+，仓库胶水代码、并发、测试代码的能力可能同样被牺牲——**难度过滤的幸存者偏差**。
3. **817 的"便宜"是样本数便宜，不是成本便宜**：chain 从哪来、L5/L1 谁打的分、R1 级多采样的算力账——论文没拆"筛选成本" vs "标注成本"。若每条 chain 都是强模型多采样 + 强 judge 精修，817 的真成本可能并不低。**样本数少 ≠ 成本低**，这是 less-is-more 叙事里最常被偷换的概念。
4. **去 leak 只有 surface-level**：训练池含 AIME 历史题 + MATH，论文的去重是 n-gram 级；Day30 告诉我们 semantic-level 近义改写泄漏是另一回事。AIME 63.3% 里多少是"模板迁移"、多少是"见过同构题"，不可分——OOD 数字里混着未知比例的分布内增益。
5. **数学独占**：817 全是数学；coding 是否同理未证。数学有天然优势：答案唯一、chain 对错可判（RQ1 的 L5/L1 分级才做得下去）。code 的"对"需要执行——LIMO 的难度定义（强模型失败率）在 code 上必须先过 Day16 的 sandbox 门禁，否则"失败"可能只是"环境没配好"。

### 5. 迁移到 coding / post-training data

**可执行的映射："LIMO 式 1k coding 冷启动集"（2 周可跑通）**

1. **组大池（抄 tens of millions 思路：宁滥勿缺）**：SWE-Gym issue 池 + Codeforces 1800+ + 内部仓库真实 bug 单，三源混合，先大再筛。
2. **Cheap-to-expensive 难度级联（抄两级过滤）**：7B-Coder 先粗筛（pass@1 高的直接丢，便宜）；32B/72B 强模型 4 采样、成功率 <25% 的才留（贵但准）。难度定义： $d(x) = 1 - \mathrm{pass@4}(x; \theta_{\mathrm{strong}})$ 。
3. **三标准转译**：complexity → 强模型失败率；knowledge diversity → 语言/题型/仓库域配额采样（先定配额再填题，抄战略采样）；deviation-from-distribution → 基座 perplexity 最高的 slice 优先（反记忆，抄第三条标准）。
4. **Chain 质量门（LIMO 过程完备性的 code 版 + Day16 前置）**：每条 trace 必须含 problem → repro → fix → verify 四段，缺一段即丢；且必须通过 sandbox 执行——数学天然可验，code 必须先验，这是 LIMO 移植到 code 时不可省略的一步。
5. **复刻 RQ1**：同题高低质 trace 对比，看 15pp 级 gap 在 code 上是否成立。若成立，"选链 > 选量"就是 coding 冷启动的第一性原理，Day29 的轨迹数据要按 LIMO 标准重分级。
6. **验收**：基座 Qwen2.5-Coder-32B，对照组 = 随机 1k / 100k 全量，看 SWE-bench-lite delta。可证伪：若 1k 精选打不赢 100k 全量，说明 code 的"完备性"前提不满足（数学知识预训练已编码，仓库知识没有）——这本身就是对 LIMO 假说边界的一次测量。

### 6. 今天的一道思考题

> 综合 **Day17（LIMO）、Day18（s1）、Day04（LESS）**：
>
> (a) **"模板 vs 长度"的归因实验**。LIMO 817（短链，~3k tokens）AIME 57.1%（v1）；s1 1k（长链，平均 9k tokens）AIME 50% → 56.7%（+budget forcing）。到底是"选得准"还是"想得长"？设计 2×2 消融：固定同一题集，{短链, 长链} × {LIMO 式难度精选, s1 式三滤精选}，训同基座（Qwen2.5-32B-Instruct）比 AIME/MATH delta。可证伪判据：若"长链"主效应显著而"精选方式"不显著 → s1 的 9k 链长才是真杠杆，LIMO 的难度级联可被"堆长度"替代（TTS 即数据）；若反之 → 认知模板的选择标准才是本质，长度只是 s1 的混杂。追问：budget forcing 的 +6.7% 和"精选"的增益是否可加？若不可加，说明两者在抢同一个自由度——"把推理做对"只有一种方式。
>
> (b) **"启发式 vs 梯度"的 head-to-head**。同一候选池，三臂各选 817：A = LIMO 难度 × 多样启发式；B = Day04 LESS 式目标梯度相似（以 AIME/MATH 为目标任务， $\cos(\Gamma(z), \bar{\Gamma}_{tgt})$ ）；C = 随机 817。同基座训练比 delta。判据：若 A > B → 推理任务上"强模型失败率"是比"梯度相似"更便宜有效的代理（呼应 9/9 Q&A 情形三：梯度代理的参照系在弱/错切空间里整体跑偏）；若 B > A → LIMO 的启发式仍有算法化空间，LESS 值得在 reasoning 上重做。关键控制：LESS 的目标梯度**来自哪个模型**？若用 7B 算梯度、选给 32B 训，是否复现"错误切空间"跑偏——这正是 9/9 情形三的实验版，顺手把那个思想实验落地了。

论文原文：https://arxiv.org/abs/2502.03387

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-17-2025-limo/NOTES.md

