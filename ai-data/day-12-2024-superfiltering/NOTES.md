# Paper 模板

## 元信息
- Title: Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
- Authors / Org: Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, et al. - UMD / Ping An
- Link / arXiv: https://arxiv.org/abs/2402.00530
- Date read: 2026-08-13
- Tags: [sft, data-selection, coding-data, weak-to-strong, instruction-tuning]

## 一句话总结
不用大模型当过滤器，用小 125M 的 GPT-2 算 IFD 难度分去筛指令，能筛出给 7B 训后效果反而更好的数据，证明选数据的能力在小模型上就有了。

## 核心
1.  **Motivation**: 指令微调数据又烂又冗余，用 GPT-4 去筛太贵，SFT 全量训又浪费
2.  **Data Pipeline**: 小模型算每条指令的 Instruction-Following Difficulty -> 排难度 -> 中高难度留 -> 给 LLaMA-7B 训。发现弱强模型选数据的结果高度一致【4921601271219063014†L16-L19】。
3.  **Key Tricks**: 弱到强过滤一致性：GPT-2 125M 挑的和 7B 自己挑的高度一致【4921601271219063014†L16-L19】；IFD 分比 perplexity/diversity 都好【4921601271219063014†L38-L42】；不训也行，plug-and-play无须额外hold-out【4921601271219063014†L54-L57】
4.  **Results**: 超过全量训，GPT-4 判赢比更高，LLaMA2-7B 用自己 IFD 还能再涨一点【4921601271219063014†L48-L51】

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 用 350M CodeT5 去筛你 50万 coding SFT，全量别训，IFD 高的留 10% 先训，省钱
  - 把 CodeSift 这种不跑的验证和 Superfiltering 的 IFD 套一起，一遍弱模型算分一遍 LLM 判，coding 上更准
- Infra 视角：弱模型过滤器可以常驻做在线流入筛选，1/500 成本筛 GPT-4 級质量

## 和之前工作的关系
是 LESS 的便宜版替代。LESS 要建梯度库，算力重；Superfiltering 只算一次前向 IFD，弱模型就行。是 Day 11 memory 里说的 ai_daily Tab 45天计划里 foundation 阶段的快速打法，可放在 Day 4 LESS 之前做粗筛。

## 疑问 / 下一步
- IFD 在 coding 上会不会把太难的 FIM 题都砍掉？需要结合执行过滤？

## 原文金句
> Can we use a smaller and weaker model to select data for finetuning a larger and stronger model?【4921601271219063014†L16-L18】

> This enables us to use a much smaller and more efficient model to filter【4921601271219063014†L18-L20】

## 第二轮复习（2026-09-12）

> 本轮已对照原文全文核对：arXiv 2402.00530（ar5iv 全文，§1–§3 逐段核对，含 Table 1 / Figure 1–3 数字）。初读 NOTES 只有 34 行、IFD 公式和弱强一致性的数字证据都缺失，本轮补齐，并修正两处初读表述（GPT-2 是 124M 不是 125M；"高度一致"需加限定见 §4）。

### 1. 核心命题

Day 12 真正解决的 data 问题：**数据筛选本身的成本死结**。筛选数据是为了省训练算力，但已有筛法（ChatGPT 当判官、学生模型自己筛、训 reward model 再筛）用的过滤器和被训的模型一样大甚至更大——筛数据的成本常常吃掉省下的训练成本。而便宜的启发式（长度 / PPL / diversity）又筛不出对 instruction tuning 真正有用的样本。

SuperFiltering 的命题是：**"感知指令难度"这个能力，在弱模型上就已经成立了**。GPT-2（124M）和 LLaMA2-7B 给同一批指令的难度排序高度相关（Spearman ρ 在 0.68–0.80），因此可以用弱模型当过滤器，把筛选成本降到原来的 1/20（Figure 1e 的 ~20× 提速），同时 5% 筛出的数据训出的 7B 在 GPT-4 盲评里打赢 100% 全量。

它回答的不是"哪条数据质量高"（Day20 DEITA 的问题）、不是"哪条数据对目标任务有用"（Day04 LESS 的问题），而是：**哪条指令是真正需要"被 follow"的指令**——即答案不能靠语言模型自由生成、必须依赖指令约束才能产出的样本。这正是 IFD（Instruction Following Difficulty）测量的对象。注意它的单位：不是"数据质量"，而是"指令的信息含量"。

### 2. 图谱位置

- **直接前驱：同组 Li et al. 2023b（Cherry / self-guided）**：IFD 分数是该组前一篇工作提出的（原文 §2.1 明说 "Li et al. 2023b firstly proposes"）。Cherry 用"学生模型自己算 IFD"筛数据；SuperFiltering 的唯一增量就是把过滤器从学生模型弱化到 GPT-2（124M）——证明感知难度的能力不需要学生级模型。第一作者都是 Ming Li，通讯 Tianyi Zhou，同一组人的续作。
- **重点直接对比 Day04 LESS（同月双生子）**：SuperFiltering arXiv 2402.00530（2026-02-01 提交），LESS 2402.04333（晚约两周）。同属选择主线，走了相反路线：
  - 信号性质：LESS 是**关系型**（target 梯度相似度，需要 few-shot 锚点）；SuperFiltering 是**内在型**（难度，需要 nothing，无监督、plug-and-play、无 hold-out target、无 warmup）。
  - 成本：LESS 建梯度库（5% warmup + LoRA 梯度 + JL 投影到 8192 维）；SuperFiltering 只算两次前向 PPL，124M 模型，成本低 1–2 个数量级。
  - 盲区互补：IFD 只看难度不看目标相关——筛出的难指令可能和你的目标任务完全正交；LESS 只看目标对齐不看难度——target 简单时选的全是简单样本。DEITA NOTES 的串联实验证明了 DEITA→LESS 重排在 HumanEval +2.1%，SuperFiltering→LESS 同理成立：先用 IFD 粗筛难例（便宜），再用 LESS 定向排 top（准）。
  - 可迁移性同构：LESS 证明"7B 选的数据给 13B/Mistral 用同样赢"（模型间 transfer）；SuperFiltering 证明"124M 选的给 7B/13B 用同样赢"（弱→强 transfer）——两篇证明的是同一个现象的两个切面：**选择信号对模型尺度鲁棒**。这是整条选择线 infra 友好的根基。
- **后继/关联**：
  - vs Day20 DEITA：DEITA NOTES 明确写 "SuperFiltering IFD ≈ complexity 近似"，DEITA 补了 quality 维（IFD 不看答案对错，DEITA 的 LLM-Quality scorer 补这个洞）+ diversity 维（IFD top% 无去重）。
  - vs Day11 LIMR：LIMR 的轨迹证据（§3.3）是 SuperFiltering 最大的挑战者——静态难 ≠ 可学。IFD 高尾部混着"有信息量的难"和"永远学不会的难"。
  - vs Day23 LIMA：LIMA 用 1k 人工高质量证明"少即是多"，SuperFiltering 是它的自动化、弱模型版本。

### 3. 机制深挖

IFD 的定义（原文 Eq. 2，本轮从 ar5iv 逐符号核对）：

$$\text{IFD}(y_i|x_i)=\frac{\text{PPL}(y_i|x_i)}{\text{PPL}(y_i)}$$

其中 $x_i$ 是完整指令（含 input 段的拼接），$y_i$ 是答案，$\text{PPL}(y_i|x_i)$ 是给定指令时答案的条件困惑度，$\text{PPL}(y_i)$ 是答案的直接困惑度。逐项拆解：

- **分母 PPL(y)**：答案文本自身的生成难度（词频、长度、格式复杂度）。一条很长的答案 PPL 自然高。
- **分子 PPL(y|x)**：知道指令之后答案的难度。
- **比值**：剥离"答案本身难"，留下"指令为答案提供的约束信息"。IFD 越高 = 指令帮的忙越少 = 跟随这条指令越难。原文原话："A higher IFD score, indicating less instructional help, suggests a greater difficulty."
- **为什么不用 PPL(y|x) 直接排序**：PPL(y|x) 混入了答案长度和词频噪声；比值做归一化后，**IFD 的尺度在不同模型间是一致的**（Figure 3：IFD 分布跨模型尺度稳定，而 PPL 的绝对尺度跨模型剧烈漂移）。这就是 IFD 比 PPL 更适合做弱强迁移分数的数学原因——Table 1 里 IFD 的 rank 相关在 Wizard 70k 上达到 0.802。
- **124M 为什么够**：IFD 只依赖**相对排序**（取 top 5%/10%/15%），不依赖绝对 PPL 值。弱模型和强模型的 PPL 绝对值差很多，但"哪条指令更难"的排序一致。排序是序数信号，对模型能力的绝对值不敏感——弱到强成立的直觉就这么简单。代价是：排序相关 ≠ 选出的子集相同（见 §4）。
- **Pipeline**：Alpaca 52k / Alpaca-GPT4 / WizardLM 70k → GPT-2（124M）算每条 IFD（两次前向）→ 排序 → top 5%（Alpaca）→ LLaMA2-7B/13B 全参 SFT → GPT-4 当裁判在 WizardLM 测试集上做 win-tie-lose 盲评（Figure 2），5% 超 100% 全量。
- **Plug-and-play 三无**：无 target 锚点（对比 LESS）、无 warmup（对比 LESS 的 5% 热机）、无 teacher 蒸馏。一次前向扫描完事，适合做 nightly 常驻过滤器。

### 4. 边界与反例

- **"高度一致"被初读夸大了**：Table 1 的 overlap ratio（GPT-2 的 top% 和 LLaMA2-7B 的 top% 的交集比例）在 5% 预算下只有 **0.28（Alpaca）/ 0.24（Alpaca-GPT4）/ 0.42（Wizard 70k）**。排序相关（Spearman 0.68–0.80）≠ 选出同一个子集。论文真正证明的是"各自筛出的 5% 都打赢全量"，而不是"存在唯一的弱强共享最优子集"。一个诚实的解读：IFD 高尾部是宽而冗余的——具体筛出哪 5% 不那么要紧，要紧的是"落在难区间"。这对 infra 是好消息（鲁棒），对方法论是提醒（别神化排序）。
- **静态难 ≠ 可学**（Day11 LIMR 的直接挑战）：LIMR §3.3 证明 RL 里长期 near-zero 的硬题是纯噪声。IFD 是训练前的一次性静态分，不知道模型当前学不学得动。SFT 里 IFD 最高尾部同样混着"永远学不会的难"——论文只报告了 top 5% 打赢全量，没有做"top 1% vs top 5–10%"的消融，无法排除"再难一点就崩"的可能。
- **难 ≠ 对目标有用**：IFD 是 task-agnostic 的。目标是代码推理时，它照样选"写十四行诗"。论文只在通用 instruction tuning 上验证，没有跨领域定向证据——这是它和 LESS 最本质的分工线。
- **难 ≠ 答案对**：IFD 完全不检查 response 质量。一条指令很难但答案是幻觉/错的，IFD 照样高。coding 上这是致命的：难算法题配错解在 IFD 排序里排很高，训进去是负收益。DEITA 的 quality 维、Qwen2.5-Coder 的 exec 过滤正是补这个洞。初读 NOTES 留的疑问（"IFD 会不会把太难的 FIM 题都砍掉"）方向反了：真正的风险不是砍掉太难的，而是**留下难但错的**。
- **无多样性控制**：top 5% 难指令可能全是同一类（复杂推理），语义重复。DEITA / s1 / Vendi 线补去重。
- **弱强验证的上限**：论文验证到 124M → 13B（~105× 参数差），学生只到 LLaMA2-13B。124M 选给 70B 用的效果没有证据——GPT-2 自身没见过的知识维度（如长推理），它感知的"难度"可能系统性失真。引用时只说"论文验证到 13B"，不要外推。
- **弱模型的先验偏差**：GPT-2 从没被 instruction-tuned，它感知的难度和真正被调过的模型感知的可能有系统差。论文用"7B 自己算 IFD 还能再涨一点"间接承认了这一点——强模型自筛仍是上限，弱筛是成本换精度的 trade-off。

### 5. 迁移到 coding / post-training data

**可执行的映射：coding SFT 冷启动三阶段筛**（把三篇的盲区拼成闭环）：

1. **Stage 1 — SuperFiltering 粗筛（省 10×）**：用 350M 级 code 小模型（CodeT5 / StarCoder2 小 proxy）在 coding SFT 池（如 50 万合成样本）上算 $\text{IFD}_{\text{code}}=\text{PPL}(\text{code}|\text{prompt})/\text{PPL}(\text{code})$，留 top 10% 难例。纯前向，单卡一晚上扫完。这是把"124M 筛通用指令"平移到"350M 筛代码指令"。
2. **Stage 2 — 执行过滤（补 IFD 的"难但错"盲区）**：Stage 1 的难例跑单元测试，砍掉"难但错"的。这是初读 NOTES 里"CodeSift + IFD 套一起"的想法的正式版——IFD 不看答案对错，exec 是唯一的答案质量门禁。
3. **Stage 3 — LESS 定向重排（补 IFD 的"难但无关"盲区）**：用 10 条目标难例（如 HumanEval+/LiveCodeBench 难例）当锚点，对 Stage 2 的干净难例做梯度相似度重排，留 top 5% 进 SFT。

**为什么这个串联对**：IFD 解决"算力"（小模型前向），exec 解决"正确性"（IFD 的最大盲区），LESS 解决"目标对齐"（IFD 的 task-agnostic 盲区）。三者的成本也是递增的（前向 < 跑测试 < 建梯度库），正好构成漏斗：每一阶段只把最贵的操作用在最小的子集上。Infra 视角：Stage 1 可以常驻做在线流入筛选（1/500 成本筛 GPT-4 级质量是初读的估计，本轮修正为论文实测的 ~20× 提速）。

### 6. 今天的一道思考题

> **综合 Day12（SuperFiltering）、Day04（LESS）、Day11（LIMR）**：
> 
> (a) SuperFiltering 证明"124M 弱模型选的数据能让 7B 训得更好"，LESS 证明"7B 选的数据能给 13B/Mistral 用"。两篇都宣称"选择信号可迁移"。但 LIMR §3.3 的实验显示：LIMO/s1 精选的 1k 难例直接 SFT 到 7B 上会拉胯（AIME24 15.8，甚至低于 base）。请回答："选择信号可迁移"在什么条件下成立、什么条件下失效？（提示：区分"谁选的"和"选的什么范式"——LIMO/s1 的 1k 是 32B teacher 蒸馏的长 CoT，SFT 是在背诵 teacher 路径；SuperFiltering/LESS 选的是"训什么"，不是"怎么训"。）
> 
> (b) 如果把 SuperFiltering 的 IFD top 5% 选集直接拿去做 Day15 R1 范式的 RL cold-start，你预测会发生什么？用 LIMR 的"轨迹对齐"语言解释：IFD 高尾部里哪一类样本的 $r_i^k$ 轨迹会是长期 near-zero，哪一类会是快涨？为什么静态 IFD 分辨不出这两类？
> 
> (c) 设计一个实验，区分"IFD 高但 RL 里学不动"和"IFD 高且 RL 里学得动"的两类样本：只允许用一个小 proxy 模型跑一轮短 RL（不能跑全量），你会记录什么信号？这个信号和 LIM 的 $r_i^k$ 序列是什么关系？
