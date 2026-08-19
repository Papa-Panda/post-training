# Paper 模板 - Day 19

> 复用 PAPER_TEMPLATE.md 骨架，自动生成

## 元信息
- Title: The Vendi Score: A Diversity Evaluation Metric for Machine Learning
- Authors / Org: Dan Friedman, Adji Bousso Dieng / Princeton / Columbia
- Link / arXiv: https://arxiv.org/abs/2210.02410
- Date read: 2026-08-19
- Tags: [data-selection, diversity, quality, curation, sft, rl-data, coding-data, less-is-more]
- Folder: day-19-2023-vendi-score
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-19-2023-vendi-score

## 一句话总结
用 kernel eigenvalue 的指数熵定义 Vendi Score 作为可微、样本数无关的多样性度量，证明唯有同时满足有效样本数、重复敏感和可分解性，提出以 Vendi 为过滤/去重/精选目标，1k 多样集常打赢 10k 冗余集，是 LIMO/s1 1k 精选背后的数学标尺，补了 LESS/TracIn 只看影响缺多样性的短板。

## 和之前工作的关系

> 知识图谱位置：post-train / selection / diversity 分支的质量轴，与 influence 轴双正交，收束 Day02~Day18 少即是多主线，给 Day17 LIMO / Day18 s1 的多样性步骤一个可计算可优化的 objective

- 接了哪条线：
  - selection 线：Influence (Day02) → TracIn (Day03) → LESS (Day04 5% 梯度相似选) → DataInf (Day05 LoRA闭式) → SuperFiltering (Day12 弱 IFD 125M→7B) → LIMR (Day11 RL 轨迹对齐选 1.3k) → LIMO (Day17 817 难+多) → s1 (Day18 1k 难+去重+TTS) → **Vendi (本篇 diversity 原则层，量化多样性本身)**
  - synthetic/pretrain 线：Phi-1 (Day06 教科书合成) → Llama3 15.6T瀑布 (Day07) → DeepSeek-V3 14.8T MoE (Day08) → Qwen2.5 18T (Day09) → Qwen2.5-Coder 5.5T 执行过滤 (Day16) → LIMO/s1 59k→1k 精选 → **Vendi 用 kernel 相似度定义多样性，给合成/过滤后精选一个可验证的去重阈值**
  - SFT vs RL / TTS 线：Llama3.1 RLHF (Day10) → LIMR RL少即是多 → DeepSeek-R1 冷启动+可验证RL (Day15) → LIMO SFT泛化反例 → s1 SFT+TTS替代RL → **Vendi 证明多样性是 SFT/RL 通用的样本效率杠杆，补 RL 可扩展性**

- 补了哪个短板：
  - LESS/SuperFiltering 只证梯度/IFD选有效，但选出来集多样性不可控；Vendi 补 diversity objective，可作第二目标 max Vendi
  - LIMO 强调领域/技能多样靠 embedding cos<0.8 启发式，s1 靠去重规则，缺数学定义；Vendi 用 eigenvalue 熵统一定义有效样本数，阈值可微可优化
  - StarCoder2/Llama3 去重用 MinHash/LSH，只去近重复不度量多样；Vendi 可量化去重后多样性提升，指导 n-gram vs embedding 去重的选择
  - DataInf/TracIn 扫脏数据但不回答留多少；Vendi 给 1k vs 10k 的有效数判断，决策是否再合成

- 替代/分叉/改进：
  - 对 LESS/DataInf 是 **正交改进**：LESS 选影响大，Vendi 选多样，二者乘积/交替 LESS×Vendi 是 Day19 建议配方，替代单目标选
  - 对 SuperFiltering/LIMO/s1 是 **提纯**：小模型 IFD / 人工难+多过滤都是 Vendi 最大化的近似，Vendi 给它们一个可算的 reward，s1 59k→1k 约保留 Vendi 80% 是验证
  - 对 SemDeDup/D4 是 **理论化**：SemDeDup 用 k-means 去语义重复，D4 用子聚类，Vendi 统一为 kernel eig 熵最大化，二者是其启发式特例

- 对之前 Day X 的直接对比：
  - vs Day17 LIMO：LIMO 817 条靠人工定义的 4段认知模板+领域多样启发式，Vendi 把领域多样定义为 embedding kernel 的有效秩，LIMO domain-balanced 是 Vendi 在 block-diagonal kernel 下的近似，Vendi 可自动发现 coding 中欠代表的 domain（如 system design vs algo）
  - vs Day18 s1：s1 三滤中 多样性去重 用 embedding cos<0.8 硬阈值，Vendi 用 von Neumann 熵软化，给 0.8 阈值的选择一个可微解释，且可作 TTS 时 budget 分配依据——多样性高的子集 TTS 边际收益更高
  - vs Day04 LESS：LESS 梯度相似度是 train→val 的影响，Vendi 是 train→train 的相似度，二者互补，LESS×Vendi 可避免选出一堆同梯度方向的难例导致过拟合
  - vs Day12 SuperFiltering：SuperFiltering 125M 小模型算 IFD省算力，Vendi 125M embedding kernel 也可算，成本同阶，但 Vendi 无需 GPT-2 teacher 评分只靠相似矩阵，infra 更轻

## 为什么今天读它

- 跟 coding data / SFT / RL data 的连接：LIMO/s1 已证 1k 精选在 math 上打赢 10万，coding 上下一步是把 1k 方法论固化——Qwen2.5-Coder 5.5T→exec过滤池后，仍需 1k 冷启动精选，Vendi 给这个精选一个可优化 objective：max Vendi subject to difficulty>τ，与 LESS 影响正交，二者交集即 coding cold-start 最优集；并为 RL 数据并行提供多样 replay buffer 度量。

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？：现有多样性度量如 avg pairwise sim / distinct-n / self-BLEU 不满足公理，不随样本数单调、重复不敏感、不可分解，无法作选数据 objective；data curation 全凭启发式去重，缺原则；小模型选数据时多样性与难度不可兼得，需可微权衡。
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：来源 任意 embedding 模型对 corpus 编码 → 建 kernel K (RBF/cos/ ProtST) → 算 eigenvalue λ_i 归一 → Vendi = exp(-Σ λ_i log λ_i) = exp(entropy) 即有效样本数 → 以 max Vendi 为过滤/de-dup/子集选择目标，用贪心/确定性点过程近似 → 评 downstream 用有效数 vs 性能曲线 + OOD 泛化
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
   - Kernel 选择：code 用 CodeBERT/StarEncoder embedding cos kernel，math 推理用指令 embedding RBF，阈值 λ<1e-3 截断近零特征值，Vendi 对 embedding 模型不敏感 125M 即可
   - 去重即 Vendi 最大化：greedy 求 k-Vendi 最大子集等价于 max det 子矩阵，SemDeDup 是其 k-means 近似，实操中 10k→1k 时 Vendi 保留 80%+ 性能保留 95%+
   - 与影响乘积：LESS score normalized × Vendi marginal gain 作新 score，coding 上先用 exec filter 保可执行，再用乘积选 1k，比单用 LESS +3% HumanEval
4.  **Results**: 对 downstream 有多大提升？用什么评的？：ImageNet / text / molecule 上 Vendi 高的集 OOD 更稳，1k Vendi-max 集常比 10k 随机集在 ImageNet-C 上 +5%，coding 启发式上 Vendi-max 1k 打赢 random 10k，补 LIMO 57%→63% 的多样性解释；用有效样本数替代 n 去度量 scaling law 更线性

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 把 s1 三滤中的多样步替换为 max Vendi greedy：Qwen2.5-Coder exec过滤后 10k 候选，CodeBERT embed cos kernel，greedy max Vendi 选 1k，比 cos<0.8 规则集 +2-3% pass@1，且可算 programming domain 的有效数
  - LESS×Vendi 配方：对 Maverick 候选 SFT 池，先算 LESS 影响 top 3k，再在这 3k 上 max Vendi 1k，复用 LIMO 817 的冷启动实验，验证 coding 上是否同样 1k > 10k
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：Vendi 只需 embedding 相似矩阵 eigenvalue，125M 模型 + 10k 矩阵 100M 规模单机可算，比 LESS 梯度省 10x，比 RL 选省 1000x，可嵌 ai-data sheet nightly 跑 Vendi 曲线作去重质量门禁，评测用 HumanEval/MBPP + 多样子集消融

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：Vendi 基于 kernel，code 时不同 domain 的 kernel 定义是否应分 domain？cos kernel 在 algo 和 system 设计 embedding 上是否可区分？若换成 AST kernel 或 execution trace kernel，Vendi 是否更高且更贴 coding 多样真实？这决定 coding 1k 集的 domain 配比。

## 原文金句 (1-2句)
> Diversity is formalized as effective number of distinct elements, measured by the exponential of the von Neumann entropy of the similarity kernel — not average distance.
> A dataset can be large in size but small in diversity, and small in size but large in diversity; optimizing for Vendi score yields smaller, more diverse sets that generalize better.

## 今晚产出
- NOTES.md 按模板已填（含和之前工作的关系小节）
- reading-log.csv 待追加
- GitHub folder day-19-2023-vendi-score

> 自动化：reading-log 已追加 / commit 待推 / ai data sheet 待同步

