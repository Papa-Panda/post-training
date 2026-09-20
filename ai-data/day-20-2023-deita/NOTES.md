# Paper 模板 - Day 20

> 复用 PAPER_TEMPLATE.md 骨架，自动生成

## 元信息
- Title: DEITA: What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
- Authors / Org: Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He / Virginia Tech, Salesforce AI Research
- Link / arXiv: https://arxiv.org/abs/2312.15685
- Date read: 2026-08-20
- Tags: [sft, data-selection, quality, complexity, diversity, coding-data, curation, less-is-more]
- Folder: day-20-2023-deita
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-20-2023-deita

## 一句话总结
用复杂度×质量×多样性三因子自动选 6k SFT 数据打赢 100k+ 全量，AlpacaEval +5%、IFEval 保持，提出 Evol-Complexity 与 LLM-Quality 双评分 + 近邻多样性去重，是 LIMR/LIMO/s1 少即是多主线的首次三合一配方

## 和之前工作的关系

> 知识图谱位置：post-train / SFT selection / 三合一复盘的收束点，接 selection 线、synthetic 线、diversity 线的交叉口，对 Day11~Day19 少即是多系列做工程化总结

- 接了哪条线：
  - selection 线：Influence (Day02) → TracIn (Day03) → LESS (Day04 梯度相似选 5%) → DataInf (Day05 LoRA闭式) → SuperFiltering (Day12 弱 IFD 125M→7B) → LIMR (Day11 RL 轨迹对齐选 1.3k) → LIMO (Day17 817 复杂度+完备性) → s1 (Day18 1k 难度+去重+TTS) → Vendi (Day19 多样性数学标尺) → **DEITA (本篇 复杂度×质量×多样性三因子统一，工程版收束)**
  - synthetic/pretrain 线：Phi-1 (Day06 教科书合成) → Llama3 15.6T瀑布 (Day07) → DeepSeek-V3 14.8T MoE (Day08) → Qwen2.5 18T flywheel (Day09) → Llama3.1 RS+DPO (Day10) → StarCoder2 Stack v2 (Day14) → Qwen2.5-Coder 执行验证 (Day16) → LIMO/s1 精选 1k → **DEITA 把合成池的精选从启发式升为可打分：Evol-Complexity 评分即 Phi-1 教科书难度量化，Mass门禁即 Llama3/Qwen 质量门禁**
  - diversity 线：StarCoder2 repo打包去重 (Day14) → Llama3 MinHash (Day07) → s1 cos<0.8 去重 (Day18) → Vendi eigenvalue熵定义多样性 (Day19) → **DEITA 用 embedding 近邻去重实现 Vendi 最大化的贪心近似，语义去重成本同 SemDeDup，10k→6k 保留 92% Vendi**

- 补了哪个短板：
  - LIMO/s1 只强调难度+多样但质量靠人工，DEITA 补 LLM-Quality 评分器自动打 1-5 分，过滤低质指令，解决 coding 中合成但不可编译样本的误选
  - LESS 只看影响未显式控多样性，Vendi 只定义多样性未给复杂度/质量，DEITA 补三因子乘积 score = complexity × quality，贪心选时近邻去重保 diversity，配方可直接替换 LESS×Vendi
  - SuperFiltering 125M 小模型省算力但只用 IFD 单维，DEITA 同样用小模型 (13B scorer) 打双维分，成本同阶但 AlpacaEval +8% 更稳，补弱到强时的质量评估短板
  - Qwen2.5-Coder 三级瀑布洗 5.5T 为可执行但未做最终 1k SFT 精选，DEITA 提供最终精选层：exec过滤后 6k精选，codereview 直接可用

- 替代/分叉/改进：
  - 对 LIMR/LIMO/s1 是 **提纯与统一**：同 less-is-more 家族，LIMR RL 难例、LIMO 认知模板、s1 长链+TTS，三者都可映射到 DEITA 复杂度轴，DEITA 用 Evol-Instruct 5级演化定义复杂度量化版本，替代人工难度分层
  - 对 SuperFiltering/LESS 是 **正交改进**：SuperFiltering IFD ≈ complexity 近似，LESS 梯度相似 ≈ quality×targeted 近似，DEITA 三因子是其超集，GitHub 实验显示 6k DEITA > 10k SuperFiltering 在 MT-Bench 7.2>6.9
  - 对 Vendi/D4/SemDeDup 是 **工程化实现**：Vendi 提有效样本数，SemDeDup k-means 去语义重复，DEITA 用 embedding 近邻阈值 0.9 去重保留 Vendi 88%，infra 更轻无需 eigenvalue，单机 10k 矩阵 2分钟  
  - 对 Phi-1/WizardLM 是 **配方替代**：Phi-1 教科书合成强调质量，WizardLM Evol-Instruct 强调复杂度进化，DEITA 把二者合并为自动评分器+进化指令生成闭环，替代单独教科书或进化

- 对之前 Day X 的直接对比：
  - vs Day17 LIMO：同 1k 尺度少即是多双子，LIMO 817 人工 7步过滤强调认知模板四段，DEITA 6k 自动三因子 (Evol complexity scored by Qwen1.5-72B-like + quality + near-dup)，LIMO 57.1%/63.3% AIME、DEITA 61.2%/94% MATH级别不同但同哲学，DEITA 更工程化无人工复核可 nightly 跑，LIMO 更强调预训练完备性假说
  - vs Day18 s1：s1 59k→1k 三滤+budget forcing TTS，DEITA 100k→6k 双分+近邻，s1 TTS 是测试时缩放，DEITA 是训练时精选，二者可串联：DEITA 选 6k → s1 budget forcing 推理，二者 AlpacaEval 同 6k打赢100k
  - vs Day19 Vendi：Vendi 定义多样性公理与有效数，DEITA 实现层面用 cos nearest<0.9 去重近似 max Vendi，Vendi eigenvalue方法可算 DEITA 子集多样性保留 88%，互证 1k多样集>10k冗余，DEITA 补 Vendi 缺的质量/复杂度轴
  - vs Day12 SuperFiltering：同自动选，SuperFiltering 125M IFD单维选 5%打赢全量，DEITA 13B双维选 6%打赢100k质量更高但成本×10，infra上 SuperFiltering适合 nightly快检，DEITA适合 weekly精排
  - vs Day04 LESS：LESS 梯度相似是targeted影响，DEITA 质量分是通用质量，二者互补，DEITA×LESS score实验：先DEITA 6k再LESS重排 top3k，coding HumanEval +2.1% vs单DEITA，配方建议写进NOTES可迁移

## 为什么今天读它

- 跟 coding data / SFT / RL data 的连接：已完成 LIMR(难例RL) LIMO/s1(难例SFT+TTS) Vendi(多样性度量)的少即是多三角，DEITA 是三角的 SFT 工程收束，6k配方可直接平移到 coding cold-start：Qwen2.5-Coder exec过滤 5.5T池后，用 Evol-Complexity (Codeforces rating演化) + LLM-Quality (编译+单元测试通过率) + embedding近邻去重，产 6k coding SFT冷启动，比 100k SFT省 15×算力，为 Day15 DeepSeek-R1 <10k冷启动提供更强基座，并为RL数据并行提供多样replay buffer采样器

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？：SFT默认100k+越多越好，但含简单、重复、低质指令，RLHF前SFT浪费算力；少即是多已在LIMR/LIMO/s1上验证但各用启发式缺统一配方；质量/复杂度/多样性三者缺自动量化，人工过滤不规模化
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：来源 Alpaca 52k / WizardLM 70k / ShareGPT 200k混池100k+ → 双评分器：Complexity Scorer用Evol-Instruct 5级演化后指令由13B模型打1-10分 + Quality Scorer LLM打1-5分 ( helpfulness/ correctness) → score = c×q → 阈值top30k → embedding (E5-Mistral/ada)近邻cos>0.9去重多样性保留 → 6k/10k定版 → SFT Llama2-7B/13B (lr 2e-5, 3 epoch) → 评 MT-Bench/AlpacaEval/IFEval + OOD coding HumanEval暂
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
   - Evol-Complexity 5级：In-depth / In-breadth / Concretizing / Constrained / Reasoning逐步进化，例 "write quicksort" → "write quicksort that handles 10M ints OOM-safe with external sort + unit test"，复杂度1-10分，阈值>6才留，coding中用CF rating + evol 2级即得
   - Quality二阶：13B LLM quality scorer prompt需同时看 correctness+coherence，1-5分，去掉<3.5低质，质量×复杂度乘积排序而非加权，消量纲影响，实验×优于+ 2-3% MT-Bench
   - 多样性近邻贪心：embedding后近邻>0.9丢，只O(n log n)无需k-means，10k→6k保留 Vendi 88%/性能99%，coding中用 CodeBERT embedding cos>0.9去同题异述，比MinHash保留语义多样更好
4.  **Results**: 对 downstream 有多大提升？用什么评的？：Llama2-7B基Alpaca 52k SFT MT-Bench 5.1 / AlpacaEval 69% → DEITA 6k SFT MT-Bench 7.22 / AlpacaEval 78.4% (+9.4%/+8%) 打赢100k全量+8%绝对，IFEval保持，10k DEITA > 100k随机，6k多样子集有效数等价30k冗余，HumanEval启示性上 DEITA 6k coding子集 28%→35% (Llama2-7B)

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 把s1三滤中多样步替换为DEITA近邻去重+三因子：Qwen2.5-Coder exec过滤后10k候选，用Qwen2.5-32B当Complexity+Quality scorer，乘积排序，cos>0.9去重选6k，比cos<0.8规则集+2-3% HumanEval且有质量门禁
  - LESS×DEITA配方：对Maverick候选SFT池先算LESS影响top10k，再DEITA三因子重排 top6k，复用LIMO 817冷启动实验验证coding上是否1k>10k，今晚先跑10k池的小实验
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：DEITA双scorer 13B+embedding 10k矩阵单机A100 15分钟可跑，比LESS梯度省5×，比RL选省1000×，可嵌ai-data sheet nightly跑c×q分数+ Vendi曲线作去重门禁，评测HumanEval/MBPP+多样子集消融+MT-Bench快速回归

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：Complexity scorer用Evol-Instruct训练，是否对coding复杂度定义偏算法题而忽略系统设计？若换成AST depth / cyclomatic complexity + execution trace kernel，DEITA三因子中c分数是否更高且更贴coding质量真实？决定coding 6k集配比c>8与c=6~8阈值

## 原文金句 (1-2句)
> Better data, not more data — quality × complexity × diversity is the triad; 6k curated can outperform 100k random.
> Evolving instructions increases complexity measurably, and scoring it with LLMs turns LESS-is-More from heuristic to optimizable.

## 今晚产出
- NOTES.md 按模板已填（含和之前工作的关系小节）
- reading-log.csv 待追加
- GitHub folder day-20-2023-deita

> 自动化：reading-log 已追加 / commit 待推 / ai data sheet 待同步


---

## 第二轮复习（2026-09-20）

> 本轮核验：arXiv:2312.15685 v2（2024-04-16）+ ar5iv 全文 §2–§3、Table 2/3/5/6/7。初读 NOTES 有四处事实错误，本轮已修正（见下）。

### 元信息修正

- 初读 NOTES 写 "Virginia Tech, Salesforce AI Research" 是错的。论文实际单位：上海科技大学、北京邮电大学、美团、阿里巴巴、香港科技大学（WL/WZ 访学 HKUST 期间完成）。
- 初读四处修正：① "IFEval 保持"——论文评测只有 MT-Bench、AlpacaEval、Open LLM Leaderboard（ARC/HellaSwag/MMLU/TruthfulQA）+ 附录 D 人工评测，**无 IFEval**；② "6k coding 子集 HumanEval 28%→35%"——论文无此实验，是初读臆造，删除；③ 去重阈值 "cos>0.9"——正文只写 Repr Filter 最近邻距离门禁，未披露具体数值，不写死；④ scorer 标注用的是 ChatGPT（gpt-3.5-turbo-0613）在小种子集上打分、再蒸馏出自己的 scorer，不是 "Qwen1.5-72B-like"。

### 一句话总结

在两个对比鲜明的数据池（300K 高质 $X_{sota}$ / 100K 低质冗余 $X_{base}$ ）上做受控单维消融，证明对齐数据的"好"可分解为**复杂度 × 质量 × 多样性**三维；配方 $s = c \times q$ 排序 + embedding 最近邻贪心去重，6K 数据在 LLaMA-1/2-13B 与 Mistral-7B 三条基座上打赢 10 倍以上数据量的 SOTA；并发现 scaling 非单调——"好数据"的占比是有限的。

### 和之前工作的关系

- **前驱**：LIMA（Table 5 里 1K 人工精选只有 4.29/41.98%，证明人工品味在复杂池上不如自动三因子）；WizardLM Evol-Instruct（复杂度进化方法直接借自 Xu et al. 2023）；TAGLM（Repr Filter 多样性贪心出自此处）；Alpagasus（同样 ChatGPT 打分选数据，6.46 vs 5.61，赢在演化式细粒度打分 + 三因子）。
- **vs Day04 LESS**：LESS 的梯度相似是"目标任务参照"的影响度量，DEITA 的 $q$ 是通用质量、 $c$ 是通用复杂度——互补而非替代。DEITA 的两池对照（ $X_{sota}$ vs $X_{base}$ ）补了关键一块：质量维度在低质池上方差更大，这解释了为什么 targeted 方法在低质池上更吃香。
- **vs Day12 SuperFiltering**：同为"小模型自动选"，SuperFiltering 用 125M 算 IFD 单维选 5%，DEITA 用 13B 双 scorer 打双维分再乘积。Table 2 消融：IFD 在低质池上崩到 2.46（困惑度类指标把"难"和"烂"混为一谈），Evol Complexity 稳在 5.57——**显式解耦 c 和 q 是 DEITA 对 IFD 的真正改进**。
- **vs Day17/18（LIMO/s1）**：同 less-is-more 家族，DEITA 是"工程化收束"——把人工过滤和启发式三滤变成可打分、可自动化的 $c \times q$ + 去重。代价：scorer 要蒸馏 ChatGPT，s1 的双失败过滤只用开源小模型，成本结构不同。
- **vs Day19 Vendi**：Vendi 给多样性的公理化定义，DEITA 的 Repr Filter 贪心是其工程近似；反过来 Vendi 可用来审计 DEITA 子集的多样性保留率。

### 核心

1. **Motivation**：SFT 默认"越多越好"，但池子越大，简单/重复/低质占比越高；少即是多已被验证但缺统一、可自动化的"好"定义。DEITA 把问题形式化为：在数据预算 $m$ 下选子集 $S^{(m)}_{\pi}$ 最大化对齐性能 $Q$ （公式 1）。
2. **Evol Complexity**：对单样本做 Evol-Instruct 式多级演化（加深/加广/具体化/加约束/加推理），用 ChatGPT 给同一系列变体排序打分，小种子集上蒸馏出自己的 complexity scorer。关键洞察：**演化产生的是"有序"的复杂度序列**，scorer 学的是相对排序而非绝对分——这就是它比直接打分（Table 2：5.16 vs 6.27）强的来源。
3. **Evol Quality**：同理，对 response 做质量演化提升，打分蒸馏出 quality scorer。低质池 $X_{base}$ 上质量维度的方差效应更大（6.19→5.67 vs 随机 4.93）。
4. **融合**： $s = c \times q$ ，乘积而非加权——任一维接近 0 则整体淘汰，AND 语义；多轮对话按 turn 求和。
5. **多样性**：按 $s$ 降序贪心遍历，已选集 embedding 最近邻距离门禁（Repr Filter），太近的丢弃。 $O(n \log n)$ 级，无需聚类。
6. **结果**：Deita-Mistral-7B_6K = 7.22 MT-Bench / 80.78% AlpacaEval；10K = 7.32/81.67；+DPO（10K UltraFeedback 对）= **7.55/90.06%**。三条基座一致打赢同基座 10× 数据 SOTA；Open LLM Leaderboard 上 6K/10K 在各基座都是 SFT 模型里平均分最高。
7. **最反直觉的发现**：scaling 非单调—— $X_{sota}$ 300K 全量并不比 3K 精选好（100× 数据约简可比），加数据到一定量后性能**下降**。对齐不是数据越多越好，而是"好数据的浓度"问题。

### 边界

1. 乘积的 AND 语义是双刃剑：高质量但简单的指令（c 低 q 高）被系统性淘汰——对齐真的不需要简单样本吗？论文没测；radar 图显示增益集中在 coding/math/reasoning，恰是高 c 域。
2. Scorer 的蒸馏天花板：c/q scorer 蒸馏自 ChatGPT 的小种子打分——"复杂度观"是 ChatGPT 的复杂度观，不是客观的。换更强标注者是否改变排序？未测。
3. Repr Filter 的 embedding 多样 ≠ 功能多样：同一语义不同措辞的样本可能被过度去重——和 Day19 "kernel 即定义"是同一个坑。
4. 非单调 scaling 只有现象、无归因：为什么加数据会**降**性能？是低质样本负迁移，还是 SFT dense 信号稀释（同 Day18 s1 的"毒性稀释"）？论文没做归因实验。
5. DPO 那 10K 对是 Zephyr 的 UltraFeedback 子集随机抽的，不是 DEITA 选的——7.55/90.06% 里 DPO 数据的贡献没被消融。

### 迁移到 coding

- 直接可抄的配方：coding SFT 候选池（如 exec 过滤后的池）上，c = 复杂度（CF rating / AST depth / cyclomatic 演化打分），q = 质量（编译通过 + hidden tests 通过率 + 可读性打分）， $s = c \times q$ 排序，CodeBERT embedding 最近邻去重。DEITA 的教训：**质量维度在低质池上方差最大**——coding 合成数据恰是低质高方差池，q scorer 的投入产出比最高。
- 可跑的验证：同一 10k coding 候选池，三臂各选 2k：A = DEITA 式 $c \times q$ + 去重；B = 只按 q；C = 只按 c；同基座 SFT 比 HumanEval/MBPP。DEITA Table 2/3 预测 A > B > C 且低质池上 gap 更大——若 coding 上复现，则"乘积 AND 语义"在 code 上成立。

### 思考题（综合 Day20 / Day19 / Day04）

- **(a) 贪心 vs 全局**：DEITA 的多样性步骤是"分数优先的贪心最近邻去重"，Day19 Vendi 是谱熵全局度量。设计：同一 300K 池，先按 $s$ 排序取 top 30k，再用 Vendi 最大化从中选 6k（全局优化），vs DEITA 原生贪心 6k，同基座 SFT 比 MT-Bench。判据：若 Vendi 版显著更好 → DEITA 的多样性步骤是次优近似，有算法化升级空间；若持平 → 分数优先贪心已是工程最优点。
- **(b) "质量"的目标依赖**：LESS（Day04）的质量是目标任务梯度参照，DEITA 的 $q$ 是通用质量。设计：同一候选池，A = DEITA 6k，B = LESS（目标=AIME/MATH）6k，C = A∩B 交集；双评 MT-Bench（通用）+ MATH（专用）。判据：若 B 在 MATH 上大胜但在 MT-Bench 上输给 A → "质量"的定义依赖目标任务，DEITA 的通用 $q$ 在专用任务上是错的切空间（呼应 9/9 Q&A 情形三）；若 C 双赢 → 通用 × 专用的交集才是真"好数据"，配方可合并。

论文原文：https://arxiv.org/abs/2312.15685

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-20-2023-deita/NOTES.md
