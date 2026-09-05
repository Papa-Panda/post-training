## 元信息
- Title: DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models
- Authors / Org: Yongchan Kwon, Eric Wu, Kevin Wu, James Zou / Columbia & Stanford
- Link / arXiv: https://arxiv.org/abs/2310.00902
- Code: https://github.com/YeonwooSung/DataInf (official), plus https://github.com/UKPLab/datainf community impls
- Date read: 2026-08-07
- Tags: [data-attribution, influence-functions, lora, efficiency, curation, quality]

## 一句话总结
把 Influence Functions 中 $H^{-1}$ 的迭代求解，改成 LoRA 参数上的闭式近似 $(1/n \sum g_i g_i^T + \lambda I)^{-1}$，比 LiSSA/CG 快 1000倍，1秒算一条 influence，专门为 LLM LoRA 微调设计，可直接用于扫脏数据/高影响样本挖掘。

## 核心
1.  **Motivation**: Influence 很好但算不动，LLM 上算 $H^{-1}v$ 要 LiSSA 迭代几百次，每次全量 HVP。大模型 + LoRA 场景急需快版。DataInf 盯的就是 LoRA 微调这个常见设定。
2.  **Data Pipeline**: 
    - 在 LoRA 微调模型上，对每个训练点算 LoRA 梯度 $g_i$
    - 用经验 Fisher 近似 $H \approx (1/n)\sum g_i g_i^T$，然后 influence Closed-form：$I(z_j, z_{test}) \approx - g_{test}^T (G^T G / n + \lambda I)^{-1} g_j$
    - 只在低秩 LoRA 维度上求逆，维度几十k不是几十亿，可闭式解
    - 拿 test 点（或 few-shot target 池）批量算，排序找 high influence / mislabeled
3.  **Key Tricks**: 
    - LoRA 是关键：全参上 $H$ 奇异且不可逆，LoRA 上参数少、满秩，闭式才稳
    - 不用二阶反传，只用一阶梯度外积，内存/计算都 $O(d_{LoRA})$
    - 对 diffusion 也适用，同理算 UNet LoRA 梯度
4.  **Results**: RoBERTa-large / Llama-2-13B-chat / Stable-Diffusion 上，近似误差 < 10% vs LiSSA 真值，速度提升 2-3 数量级；mislabel detection AUC 明显高于 TracIn, Representer。论文 ICLR 2024。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 在你 SFT 的 LoRA checkpoint 上跑 DataInf，给 50w 合成 code 打 influence 分，top negative 就是拖累 HumanEval 的脏数据
  2. 把 LESS 的 datastore 换成 DataInf 的闭式分数，做 5% 筛选对比，看速度/精度 trade-off
- Infra 视角：LoRA 上求逆可用 batched Cholesky，单 A100 可并行算 10k 条影响分，适合 nightly data flywheel 的自动清洗。

## 疑问 / 下一步
- LoRA 的 Fisher 近似在 RLHF / GRPO 的 policy gradient 上还成立吗，还是需要修正？
- $\lambda$ 如何自适应选，coding data 上 Fisher 奇异时如何稳住逆？

## 原文金句 (1-2句)
> DataInf is particularly well-suited for parameter-efficient fine-tuning techniques such as LoRA, with an easy-to-compute closed-form expression.

> DataInf is orders of magnitude faster than existing influence methods while accurately approximating influence scores.


## 第二轮复习（2026-09-05）

### 1. 核心命题

DataInf 真正解决的 data 问题：**influence 是最有原则的数据归因单位，但在 LLM 设定下算不动**——Day 02 的 $$H^{-1}$$ 需要 LiSSA 几百次 HVP 迭代，Day 03 的 TracIn 干脆放弃了 Hessian，Day 04 的 LESS 用 Adam 预条件替换了曲率。三条路都绕开了"精确的二阶反事实"。

DataInf 的核心命题是：**在 LoRA 这个实际最常见的微调设定下，精确 influence 可以闭式算**。因为 (a) LoRA 参数维度只有几万（不是几十亿），(b) 在收敛的 adapter 上经验 Fisher $$(1/n)\sum_i g_i g_i^\top + \lambda I$$ 是良态可逆的、且对负对数似然损失它是 Hessian 的合法近似。于是每条训练样本对测试点的影响
$$I(z_j, z_{test}) = -g_{test}^\top (S + \lambda I)^{-1} g_j$$
可以在**一次 Cholesky 分解之后，摊销为每条样本一次点积**。

所以 DataInf 的真正产品不是"更快的 influence"，而是**把归因从研究仪器变成数据管线的一个常规步骤**：对 50 万条 coding 池逐条打 influence 分成为 nightly job 可负担的事。它的输出是**校准过的、有符号的影响分数**（论文在可算真值的设定下报告近似误差 <10%，且 orders of magnitude 快于 LiSSA/CG），这正是"扫脏数据"任务需要的：你不仅要知道哪条有害，还要知道**有害多少**、以及**删掉它值不值**。

一句话：Day 02 定义了影响，Day 03 让影响可算，Day 04 让影响可定向选，Day 05 让影响**可常规化、可校准**——它是把归因变成数据清洗基础设施的那一步。

### 2. 图谱位置

- **直接前驱 Day 02 Influence (2017)**：DataInf 是 influence 谱系里最忠实的一支。它保留了 $$H^{-1}$$（用经验 Fisher 近似），计算的是端点式重优化反事实 $$-g_*^\top H^{-1} g_z$$，而不是 TracIn 的路径记账或 LESS 的优化器近似。Day 02 复习里推过 $$H^{-1}\approx\eta\sum_k(I-\eta H)^k$$（迭代路线）；DataInf 走了另一条路——**统计近似**（Fisher）代替**迭代求解**（LiSSA）。同一个逆矩阵，两种到达方式。
- **平行 / 互补 Day 04 LESS（重点直接对比）**：同一块 LoRA 梯度 substrate，走了相反的方向。
  - DataInf 保留**曲率**（Fisher ≈ H），放弃**优化器**（假设无穷小重加权的一步牛顿式反事实）；LESS 放弃曲率，保留**优化器**（Adam 预条件的真实更新方向 $$\Gamma(z)$$）。
  - DataInf 的分数是 **per-sample 校准归因**：$$I(z_j, z_{test})$$ 有符号、有量级，回答"这条样本让测试 loss 动了多少"。LESS 的分数是 **target-conditioned 排名**：$$\cos(\bar\Gamma_{target}, \Gamma(z))$$ 只保序不保量，回答"哪些数据该进 top 5%"。
  - 代价结构不同：DataInf 对每个 test 锚点付一次 $$O(d^3)$$ 求逆 + $$O(Nd)$$ 点积（test-dependent，不可跨目标复用）；LESS 建一次 gradient datastore，之后每次选择是 $$O(N)$$ cosine（target-independent，可复用）。
  - **分工因此是天然的**：DataInf 是**清道夫**——找负影响样本（误标、坏测试、记忆陷阱），删掉；LESS 是**星探**——找对目标能力有用的样本，选入。Day 04 复习第 2 节已经画出这个分工，本轮复核：正确，且可以更精确——DataInf 的符号/量级校准是扫脏的必要条件（见第 6 节思考题），LESS 的排名鲁棒性是定向选择的必要条件，两者不可互换。
  - 一个漂亮的统一视角（第 3 节 (b) 展开）：DataInf 在 $$\lambda \to \infty$$ 时退化为一阶点积（TracIn 式）；LESS 在"只用方向"这一点上等价于对 magnitude 做了无限正则——两篇论文在"要不要 magnitude"上做了相反但各自自洽的选择。
- **对比 Day 03 TracIn**：TracIn 用 checkpoint 路径累加绕开 $$H^{-1}$$（path-wise），DataInf 用 LoRA 闭式 Fisher 正面拿下 $$H^{-1}$$（endpoint-wise）。TracIn 的 self-influence 扫"需强记的异常样本"，DataInf 的 mislabel detection 扫"标签错的样本"——同一"扫脏"任务，TracIn 靠**记忆强度**（self 点积大 = 模型靠死记学会），DataInf 靠**对验证集的负向因果**（删掉它验证 loss 下降）。前者不需要验证锚点，后者必须有锚点。这是"无监督异常检测" vs "有监督因果归因"的区别。
- **后继呼应**：Day 11 LIMR / Day 28 ORZ 的"困难尾部挖掘"在做的事，可以看作 DataInf 式"找高影响样本"的 RL 版本——只不过 RL 里"影响"通过"训了之后验证集涨没涨"来度量，而不是解析算。解析的 influence 在 RL 的 policy gradient 下是否还成立，正是初读疑问里留下的开放问题。

### 3. 机制深挖

**(a) 闭式推导：两次近似，一次重排。**
从经典 influence 出发：
$$I(z_j, z_{test}) = -\nabla_\theta L(z_{test}; \hat\theta)^\top H_{\hat\theta}^{-1} \nabla_\theta L(z_j; \hat\theta)$$
近似一：$$H \approx S + \lambda I$$，其中 $$S = (1/n)\sum_i g_i g_i^\top$$ 为经验 Fisher。合法性来自两点：① 在收敛点附近，对负对数似然损失有信息矩阵等式（Fisher ≈ Hessian）；② **LoRA 是关键**——全参空间下 H 奇异、维度灾难，这个近似是垃圾；但在几万维的 adapter 空间、且 adapter 已训至收敛时，Fisher 是满秩良态的。LoRA 在这里不只是省内存的技巧，**它是让数学成立的前提**。
近似二（重排，真正的工程洞察）：不要对每个训练点都算 $$(S+\lambda I)^{-1} g_j$$。先算一次
$$v := (S + \lambda I)^{-1} g_{test}$$
（Cholesky，$$O(d^3)$$，d 为 LoRA 参数量，只做一次），然后每条训练点的影响就是一次点积：
$$I(z_j) = -v^\top g_j \qquad O(d)$$
LiSSA 对每个 test 点要跑几百次 HVP（每次两次反传、$$O(\text{params})$$）；DataInf 把"每个 test 点"的代价从"几百次全量反传"降到"一次小矩阵分解"。这就是"快 2-3 个数量级"的来源——不是常数优化，是**把迭代变成了摊销**。

**(b) $$\lambda$$ 的双重身份：正则项 + 插值旋钮。**
Fisher 来自有限样本，在数据没探索过的方向上秩亏；$$\lambda I$$ 是 Tikhonov 正则 = "没探索过的方向曲率为 $$\lambda$$"的先验。但更有意思的是它的极限行为：
- $$\lambda \to 0$$：精确（但噪声大）的 influence——Fisher 零空间方向的逆会爆炸，分数被噪声主导。
- $$\lambda \to \infty$$：$$(S+\lambda I)^{-1} \approx I/\lambda$$，于是
$$I(z_j, z_{test}) \approx -(1/\lambda)\, g_{test}^\top g_j$$
**退化为 TracIn 式一阶点积**。DataInf 以 TracIn 为其高阻尼极限——这是 Day 03 与 Day 05 在数学上最干净的连接：TracIn 不是另一套理论，它是 DataInf 在"完全不信任曲率估计"时的特例。
工程含义：$$\lambda$$ 选小了，分数是噪声；选大了，曲率修正被洗掉、退化成点积，论文"近似误差 <10%"的前提就没了。初读疑问"$$\lambda$$ 如何自适应选"至今没有理论处方——实践中只能在"误标检测 AUC"这类下游代理指标上调参。这是 DataInf 最大的工程软肋。

**(c) 只看到 adapter 空间：influence 是 (数据, 模型, 配方) 的三元属性。**
分数在 LoRA 参数空间度量——那些"作用在冻结 base 方向"上的样本效应是不可见的。对扫脏这没问题（我们问的是"在这套 LoRA 训练下哪条数据有害"），但它意味着：**同一份数据，换 rank / 换 target module / 换 warmup，分数会变**。Influence 不是数据的内在属性（DEITA 的 complexity/quality 那种），而是关系属性——这一点上它和 LESS 一致（Day 04 复习第 1 节的核心命题），只是关系的对象不同：LESS 的关系对象是"目标能力 + 优化器"，DataInf 的是"验证锚点 + 曲率"。

**(d) Test-point 依赖与锚点均值。**
$$v$$ 对每个 test 点都要重算：M 个锚点 → M 次求逆 + M×N 次点积。论文的实用模式是取 50~100 个验证样本做锚点（尤其**当前做错的**），平均 influence。这和 LESS 的"few-shot target 锚点"是**独立收敛到同一个设计**：两边都发现"全验证集太贵、单点太噪，一小撮代表性锚点最合适"。区别在于 LESS 对锚点梯度取均值再做 cosine（方向），DataInf 对每个锚点分别算校准分数再平均（量级）。

**(e) 为什么"闭式"对 flywheel 是质变。**
Per-point 代价是一次点积 → 整个池子的分数就是一次 GEMM：$$G\,v$$，其中 $$G$$ 为 $$N \times d$$ 梯度矩阵。50 万条 coding 样本 × 几万维 LoRA 梯度，一次矩阵乘就出全部分数。LiSSA 做不到 nightly 重算，DataInf 可以——**归因从"离线研究"变成"数据管线的一个 stage"**。这正是第 5 节方案可行的计算基础。

### 4. 边界与反例

1. **Fisher ≈ Hessian 只在收敛点成立**：论文所有实验都用训至收敛的 LoRA。在 warmup 中途 / early checkpoint 上跑 DataInf，Fisher 与 Hessian 差距大，"负影响"可能是"还没学会"而不是"数据有害"。反例：在 SFT 第 1 个 epoch 的 checkpoint 上扫脏，会把所有高 loss 难例标成有害——而难例恰恰是 Day 11/28 要保留的困难尾部。
2. **$$\lambda$$ 无理论处方**（第 3 节 (b)）：coding 池模板化严重时梯度共线，Fisher 接近奇异；$$\lambda$$ 太小分数爆炸，太大退化成 TracIn。论文在固定 $$\lambda$$ 下调参，没有自适应方案——这是部署时必须自己啃的硬骨头。
3. **证据没证明什么**：mislabel detection 的漂亮 AUC 是在**人工翻转标签**的合成脏数据上测的（GLUE 任务随机翻标签）。真实脏数据——错的单元测试、过时 API 用法、能跑但逻辑错的代码、prompt 与解法错位——**不是"标签翻转"分布**。论文没有在真实 noisy code 数据上验证扫脏效果，"扫脏"目前仍是外推，不是实证。
4. **LoRA 子空间限制**（与 Day 04 第 4 节第 6 条同源）：若最终做全参 SFT，LoRA 空间的 influence 排名可能错位。论文自洽（下游验证也在 LoRA 设定下），迁移到全参训练前需做小规模 ablation 验证排名保持。
5. **"<10% 近似误差"不可外推到大模型**：这个数是在**能算出 LiSSA 真值**的小设定下量的；Llama-2-13B 上的"准确"是通过 mislabel AUC 间接论证的——13B 的真值根本算不出来，误差上界未知。引用时必须加这个限定。
6. **Influence 选择的经典陷阱：偏向简单样本**。$$I(z_j)$$ 按"边际上减少验证 loss 的量"排序——简单、高频、易拟合的样本天然排前面（loss 下降空间大、梯度方向与验证集一致）。拿 DataInf 做"选 top-k 训"会选出一池**简单题**，与 Day 11/17/18 的"少即是多要难例"背道而驰。**DataInf 是扫脏工具，不是选择工具**——用它做选择是范畴错误。LESS 的 target 锚点设计部分对冲了这个（锚点是难例时，选的是"对难例有用"的数据）。
7. **计算边界**：$$O(d^3)$$ Cholesky 在大 rank（64+）或 target 模块多时，d 可达几十万维，闭式优势缩小。论文的"约 1 秒一条"是摊销口径（一次求逆摊到 N 条），N 很小时反而可能比 TracIn 慢——小池子别用 DataInf。

### 5. 迁移到 coding / post-training data

**可执行方案：DataInf nightly 扫脏 + LESS 定向选的两段管线（针对 500k 合成 coding 池）**

1. **锚点构造**（对冲"无锚点"误用）：取当前 LoRA SFT checkpoint 在 HumanEval+ / LiveCodeBench 上**做错的 50~100 题**做失败锚点，另取 50 道做对的做成功锚点。对每条训练样本算净有害分
$$\Delta I(z_j) = \bar I_{\text{fail}}(z_j) - \bar I_{\text{succ}}(z_j)$$
即"对失败集有害、对成功集无害"的部分。符号约定：$$I>0$$ 表示加权该样本会**增加**测试 loss（有害）。
2. **一次求逆，多次复用**：对失败锚点梯度取均值 $$\bar g_{\text{fail}}$$（Day 04 式的均值锚点思想），只做**一次** Cholesky 得 $$v$$，全池分数 = 一次 GEMM $$Gv$$。500k × 30k 维 bf16 矩阵乘，单卡分钟级。
3. **可证伪的删除闭环**：取净有害分 top 1%（约 5k 条）→ 抽 200 条做 LLM-judge + 人工抽查，确认真是坏测试/错解/过时 API → 删掉重训 LoRA → 看 HumanEval+ delta。**先小规模验证"删了真涨分"，再放大到全量删除**——不要一上来就删 5k。
4. **与 LESS 的 cascade**（两篇的分工落地）：先 DataInf 扫脏（删有害的），再在清洗后的池子上跑 LESS 双锚点选 5%（选有用的）。顺序不能反：LESS 的 cosine 对"有害但方向对"的样本无抵抗力（方向对 + magnitude 大 = 高 cosine，但 DataInf 会告诉你它是负影响）。
5. **Infra 沉淀**：把每条样本的 $$I(z_j)$$、IFD（Day 12）、complexity（Day 20 式）都写进数据 registry 的同一行，变成多信号选择的基础设施——DEITA 三因子里 influence 轴的现成供给。
6. **诚实边界**：最终若做全参 SFT，先用 1B proxy 做"删 top-1% 有害 vs 随机删 1%"的 ablation，确认 HumanEval 提升可迁移，再在全参管线里用。

### 6. 今天的一道思考题

综合 **Day 05 DataInf、Day 04 LESS、Day 03 TracIn**：

DataInf 的分数为 $$I(z_j, z_{test}) = -g_{test}^\top (S + \lambda I)^{-1} g_j$$，在 $$\lambda \to \infty$$ 时退化为 $$-(1/\lambda)\, g_{test}^\top g_j$$（一阶点积，即 TracIn 单 checkpoint 步的形式）；LESS 的分数为 $$\cos(\bar\Gamma_{target}, \Gamma(z))$$，**主动丢掉了 magnitude**。

(a) 从数学上解释：为什么 DataInf **必须保留 magnitude**——即"扫脏"任务为什么不能只用方向？（提示：删一条样本的决策是"值不值"的二值决策，需要把"有害程度"与"重训成本 / 误删好数据的风险"比较；方向只能给序，序不能回答"前 1% 和前 10% 的危害差几个数量级"。再想想：两条同样"方向有害"的样本，一条让验证 loss 涨 0.5，一条涨 0.001，cosine 会怎么排？排出来对"删不删"有指导意义吗？）

(b) 为什么 LESS **必须丢掉 magnitude**——即"定向选 5%"为什么不能用 dot？（提示：梯度 norm 被 token 数和"惊讶度"主导；Day 04 复习第 3 节 (c) 已证实 dot 会被长序列/高 loss 异常值劫持。用 (a) 的框架反过来想：定向选择要回答的是"推哪个方向"，magnitude 在这里是噪声还是信号？）

(c) 构造具体例子，分两个 framing：
- **扫脏 framing**（有害 = 与失败锚点梯度夹角为钝角）：样本 A：800 token 长代码，高 loss，梯度 norm 为 B 的 10 倍，与锚点夹角 100°（轻度有害方向）；样本 B：50 token 短代码，低 loss，与锚点夹角 170°（重度有害方向）。分别用"DataInf 式（含曲率的）dot"和"cosine"给 A、B 的"有害度"排序：谁排第一？哪个排序对"删脏数据"正确，为什么？（注意：删数据的决策依据是**总因果效应** = 方向 × 量级。）
- **定向选 framing**（有用 = 与能力锚点夹角为锐角）：样本 A'：800 token 高 loss，norm 为 B' 的 10 倍，与目标锚点夹角 30°；样本 B'：50 token 低 loss，与锚点夹角 10°。分别用 dot 和 cosine 排序：谁排第一？哪个排序对"选 5% 训推理"正确，为什么？
- 然后回答：**两道任务的最优排序是相反的**——这正是 DataInf（清道夫）与 LESS（星探）分工的数学根源。用一句话概括这个根源。

(d) 论证"影响分数的符号/量级校准"与"排名鲁棒性"不可兼得：前者要求保留 magnitude（从而继承 norm 噪声），后者要求丢掉 magnitude（从而失去校准）。然后设计一个同时需要两者的 pipeline：**先用哪篇的方法做粗筛、用哪篇做精排，阈值/比例怎么定，以及为什么这个顺序不能反**（提示：Day 04 复习第 5 节的 cascade 顺序 + 本轮第 5 节第 4 条的论证）。

答案不许只复述 DataInf 原文；必须实质用到至少两篇对比论文的机制，数学推导要写全。

## 原文链接
- Paper: https://arxiv.org/abs/2310.00902
- GitHub NOTES: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-05-2024-datainf
