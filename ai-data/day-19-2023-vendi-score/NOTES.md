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


## 第二轮复习（2026-09-19）

> 本轮复核：arXiv:2210.02410（v2，2023-07-02 修订；ICML 2023）摘要逐项核验 + 开源评审版 Theorem 3.1（Vendi Score 四条性质）全文核对。修正初读 NOTES 六处：① "重复敏感"方向写反：定理 3.1(2) identical elements 说的是**合并全同样本不改变 VS**（probability-weighted 形式）——Vendi 对重复是"不膨胀计数"，不是"对重复敏感"；② "证明唯有同时满足…"：论文只证明 VS 满足四条性质，没有唯一性 claim，"唯有"二字论文没有；③ 下游数字虚构：原文是生成模型/数据集诊断论文，**零训练实验**，"ImageNet-C +5%"、"coding Vendi-max 1k 打赢 random 10k"、"LESS×Vendi +3% HumanEval"、"s1 59k→1k 保留 Vendi 80%"、"DEITA 保留 Vendi 88%"、"10k→1k 性能保留 95%+"在原文中都不存在，一律划掉；④ "greedy max Vendi 等价于 max det"：DPP 的 $\log\det$ 与 Vendi 特征值熵是两个不同的谱目标，论文没提 greedy 算法，更无等价性；⑤ 遗漏 q 阶推广 $VS_q$ （q=0→rank、q=1→Vendi、q=∞→ $1/\lambda_{\max}$ ），这是对数据工作最有用的旋钮；⑥ "Vendi 对 embedding 不敏感 125M 即可"、" $\lambda<10^{-3}$ 截断"等工程数字原文没有。初读疑问中"cos kernel 在 algo 与 system design 上是否可区分"是个真问题，保留并深化为 §4.1。

### 1. 核心命题

Vendi 真正解决的 data 问题：**多样性没有公理化的度量**。baseline 们各坏一种：avg pairwise similarity 被样本数绑架（加一堆近重复，平均值几乎不动，n 却涨了）；distinct-n / self-BLEU 对长度敏感、不是真度量；IS/FID 把质量和多样性混在一起；分子领域的 uniqueness/novelty 只数"是否见过"，不度量分布形状。Vendi 的动作是把生态学（effective number of species / Hill numbers）与量子统计（von Neumann 熵）的现成答案搬进 ML：**多样性 = 有效不同元素数 = 相似 kernel 归一化特征值谱的 Shannon 熵的指数**。

关键定性：这是一篇**评估/诊断**论文，不是选数据论文。它的实验是：分子生成（修 GuacaMol 的 uniqueness/novelty 短板）、图像生成（StackGAN）、文本解码算法（证实 sampling > greedy/beam 的已知结论）、GAN mode collapse（连"抓住所有 mode"的 GAN 都比原数据集更多样性缺失——揭示的是 mode *内*多样性）、benchmark 数据集诊断。**"1k 多样集打赢 10k 冗余集"不是论文的 claim**，是我们在 Day17/18 语境下给它的外推定位（"少即是多背后的数学标尺"这个定性可以保留，但必须注明是外推，不是原文结果）。

### 2. 图谱位置

- **前驱（图外）**：生态学 Hill numbers / Jost 2006（有效物种数；论文摘要明说 connects ideas from ecology）；量子统计 von Neumann 熵（把归一化 kernel 当 density matrix，特征值熵即 von Neumann 熵）；ML 内被它替代的度量：avg pairwise sim、distinct-n、self-BLEU、GuacaMol uniqueness/novelty。
- **直接对比 Day24 D4（重点）**：**标尺 vs 流程**。Vendi 只给度量： $O(n^2)$ 建 kernel + $O(n^3)$ 特征分解，不告诉你删哪些样本，论文里没有任何选择算法——"会量多样性，但不会规模化选数"（Day24 NOTES 原话）。D4 给两步可执行流程：SemDeDup（k-means 去语义近重复）+ prototype-based diversification（砍过密原型），能跑预训练规模，精确告诉你删谁——但全程没有写下任何标量目标函数。两者互补的精确表述：D4 第一步 ≈ 在优化 identical-elements 那一侧（把重复合并掉），第二步 ≈ 把特征谱压平；但 D4 从没证明自己在优化什么。可执行的组合：**用 Vendi 当 D4 的审计器**——diversification 前后各算一次 VS，看有效多样性真涨了还是只换了簇的计数；再用 q 阶 VS 看 D4 有没有顺手砍掉稀有尾部（见 §6b 实验）。
- **vs Day18 s1**：s1 的 embedding cos<0.8 是"去重规则"，Vendi 是"多样性定义"。硬阈值的两个毛病 Vendi 恰好点名：① 0.8 拍脑袋，无公理支撑；② pairwise 阈值不满足 partitioning——删一个样本会改变其他样本对的去留判定（非单调），而 Vendi 的 partitioning 性质（ $VS(S_1,\dots,S_m)=\exp(H(p_1,\dots,p_m))\prod_{i=1}^m VS(S_i)^{p_i}$ ）恰好回答"子集多样性能否分块独立算再合并"。
- **vs Day20 DEITA**：DEITA 的 embedding 近邻去重是 max-VS 的工程贪心近似（无特征分解， $O(n\log n)$ ）；但 DEITA 把 diversity 只列为三因子之一——因为 VS 不管质量不管难度（见 §4.4），"多样性好"≠"数据好"。
- **后继（图外，待读）**：DPP 家族是谱表亲（ $\log\det$ vs 特征值熵，同"谱散度"不同目标）；2026 年搜索到的 "How Much Is a Dataset Worth? Scaling Laws, the Vendi Score, and Matrix Spectral Functions" 把 Vendi 接到 scaling law 与数据集估值、做了 ImageNet 子集实验——列为后继，未细读不展开。

### 3. 机制深挖

**(a) 定义与谱直觉。** 样本 $x_1,\dots,x_n$ ，相似函数 $k$ 满足 $k(x,x)=1$ 且 kernel 矩阵 PSD。定义归一化 kernel $K_{ij}=k(x_i,x_j)/n$ ，则 $\mathrm{tr}(K)=\sum_i K_{ii}=1$ ，特征值 $\lambda_1,\dots,\lambda_n$ 自动构成概率分布。 $VS_k=\exp(-\sum_i\lambda_i\log\lambda_i)$ 。

最有用的特例：取单位范数 embedding + cosine 相似，则 $K=XX^T/n$ ，其非零特征值等于数据协方差 $X^TX/n$ 的 PCA 谱。于是 **Vendi = 数据集协方差"有效秩"的 Shannon 版本**：谱平 → $VS\approx n$ ；谱塌到单个特征值 → $VS=1$ 。"多样性"在这里被翻译成"数据在表示空间里占了几个有效维度"——这句话是后面所有边界讨论的起点。

**(b) 四条性质的机制含义（Theorem 3.1）。**

1. Effective number：全异（ $k(x_i,x_j)=0$ ， $i\ne j$ ）→ $VS=n$ ；全同 → $VS=1$ 。给出解释标尺：" $VS=m$ ≈ 和 $m$ 个全异元素一样多样"。
2. Identical elements：probability-weighted 形式下，把两个全同样本合并（ $p'_i=p_i+p_j$ ， $p'_j=0$ ）VS 不变。机制含义：**Vendi 是分布的度量，不是样本的度量**——加 1000 个副本不增加多样性。这正是它修 avg-pairwise-sim 的地方：pairwise 平均会被重复样本的数量绑架，Vendi 不会。
3. Partitioning：跨子集零相似时， $VS(S_1,\dots,S_m)=\exp(H(p_1,\dots,p_m))\prod_{i=1}^m VS(S_i)^{p_i}$ ， $p_i=|S_i|/\sum_j|S_j|$ 。不是简单加权平均——多出来的 $\exp(H(p))$ 项是"子集间分布"的贡献。实操含义：分域配额式 curation（s1 的 MSC 50 域、LIMO 的领域均衡）可以用这个公式做**分块审计**：先各算各域的 VS，再用公式合并，定位是哪个域在拖累整体。
4. Symmetry：与样本顺序无关——度量的是集合，不是序列。

**(c) q 阶旋钮（初读遗漏，论文有）。** $VS_q=\exp(\frac{1}{1-q}\log\sum_i\bar\lambda_i^q)$ ：q=0 → rank（数非零特征值，对稀有特征最敏感——长尾猎手）；q=1 → 标准 Vendi；q=∞ → $1/\lambda_{\max}$ （只看最大簇——塌缩检测器，专抓"一个 dominant mode 吃掉一切"）。数据工作的直接用法：选数据保长尾用低 q 审计，查头部失衡用高 q。Pasarkar & Dieng 2024 的解读：q 控制对稀有 vs 常见特征的敏感度。

**(d) "kernel 即定义"。** 摘要原话：VS takes a similarity function as input, enabling the user to specify any desired form of diversity。**多样性没有无条件的定义，k 就是定义本身。** 机制推论：换 k = 换问题。用 CodeBERT-cos 算出的 VS 高，只说明"表示散"，不说明"功能散"——这是 §4.1 和 §5 双 kernel 设计的理论依据。

**(e) 与 DPP 的谱亲缘（初读"等价于 max det"划掉）。** DPP 优化 $\log\det(K_S)$ ，Vendi 优化特征值熵——都是"让谱散开"的谱目标，但**目标函数不同**，论文没提任何 greedy 算法，更无等价性。只能说：DEITA/D4 的启发式去重是这一谱家族的工程近似。

### 4. 边界与反例

1. **kernel 盲区**：cos-embedding 下，同一算法题的 100 种正确解法可能彼此很近（VS 低估功能多样），100 个同模板换数字的题可能很散（VS 高估）。反例：两个共享大量 boilerplate 但功能迥异的程序会被判相似——"表示多样 ≠ 功能多样"，而 Vendi 本身不负责区分。
2. **PSD 约束**：相似矩阵非 PSD → 负特征值 → 熵无定义 → VS 失效。连对称矩阵都不保证 PSD（后续文献明确讨论过这点）。实操要么用真 kernel（RBF、归一化 embedding 的 cos），要么 clip 负特征值——后者扭曲度量且无论文内误差界。
3. **size-coupled**： $VS\le n$ ，不同大小集合的 VS 不可直接比。"1k 打赢 10k"若拿 VS 当论据是范畴错误； $VS/n$ 可比，但论文主要用 raw VS。
4. **只管散，不管好**：100 条各不相同但全错的解，VS 很高——多样性 ⊥ 质量 ⊥ 难度。这是 DEITA 必须三因子的原因，也是 Vendi 不能单独当 selection objective 的根本理由。
5. **成本**： $O(n^2)$ kernel + $O(n^3)$ 特征分解；10k 单机可算，100k+ 得 Nyström/采样近似，近似误差传到 VS 上没有论文内保证。
6. **证据没证明什么**：原文零训练实验——没证明"VS 高的训练集 → 下游性能好"；greedy max-VS 当选择目标是外推，无收敛/泛化保证。初读的六个下游数字（见本节 §头）全部划掉。

### 5. 迁移到 coding / post-training data

**可执行的映射："双 kernel Vendi 审计 + 行为 kernel 贪心选 1k coding 冷启动"（2 周可跑通）**

1. 大池：exec 过滤后的 10k coding SFT 候选（抄 Day16：parser + 执行验证门）。
2. 双 kernel： $k_{\text{text}}$ = CodeBERT/StarEncoder 归一化 embedding 的 cosine（表示相似）； $k_{\text{exec}}$ = 固定 hidden-test 电池上的通过/失败向量做 RBF（功能相似）。两个 kernel 下各算 $VS_q$ （q=0/1/∞）——两个 VS 的 gap 就是"表示多样但功能单一"的量化值。
3. 选择：先过 exec 质量门（hidden tests 全过），再在 $k_{\text{exec}}$ 下贪心 max 边际 VS 选 1k——**功能多样优先**，不是文本多样。
4. 审计：报告 VS、 $VS/n$ 、q 谱的选前/选后；硬性门禁：q=0 的 VS 不得下降（稀有技能保住，防 D4 式砍尾）。
5. 验收：与 s1 式 cos<0.8 规则、DEITA 近邻去重同基座对比 HumanEval/MBPP + 仓库任务 slice。可证伪预测： $k_{\text{exec}}$ 版在 OOD 仓库任务上 ≥ $k_{\text{text}}$ 版；若不成立，则"kernel 即定义"在 coding 上不敏感，cos 够用——这本身也是个有价值的否定结果。

### 6. 今天的一道思考题

> 综合 **Day19（Vendi）、Day24（D4）、Day20（DEITA）**：
>
> (a) **"kernel 即定义"的证伪实验**。同一 10k coding 候选池（先过 exec 质量门）， $k_{\text{text}}$ （CodeBERT cos）与 $k_{\text{exec}}$ （hidden-test 通过向量 RBF）各 greedy 选 1k。同基座 SFT，比 HumanEval/MBPP（分布内）与 SWE-bench-lite slice（分布外）。判据：若两者打平 → 表示多样 ≈ 功能多样，cos kernel 够用，Vendi 的 kernel 选择不敏感；若 $k_{\text{exec}}$ 版在分布外显著更好 → "什么算多样"的定义权在 kernel，Vendi 只负责度量不负责定义——那所有用 embedding cos 算多样性的工作（s1 的去重、DEITA 的近邻、D4 的 SemDeDup）都在度量一个可能错的"多样"。追问：q 阶怎么选？q=0（rank，长尾敏感）选出的 1k vs q=∞（ $1/\lambda_{\max}$ ，头部敏感）选出的 1k，哪个在长尾仓库任务上更好？把 Vendi 的 q 旋钮和 D4 "保长尾"的目标放进同一个实验对质。
>
> (b) **Vendi 审计 D4**。取 D4 管线（SemDeDup → prototype diversification）的前/中/后三个快照，算 VS 三件套（VS、 $VS/n$ 、q=0/1/∞ 谱）。可证伪判据：若 diversification 后 VS 上升但 q=0 的 VS 下降 → D4 在压平头部的同时砍了稀有尾部，"保覆盖"不成立，prototype 步的阈值要回退；若 q=0 不降反升 → D4 的"去重+保多样"自洽，Vendi 从此可作 D4 的 nightly 质量门禁。追问：partitioning 公式允许分域独立算 VS 再精确合并——能否用它给 D4 的每个语义簇设"簇内 VS 下限"，把全局启发式变成逐簇可验证的约束？

论文原文：https://arxiv.org/abs/2210.02410

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-19-2023-vendi-score/NOTES.md
