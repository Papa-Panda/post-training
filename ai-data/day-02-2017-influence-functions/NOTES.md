# 元信息
- Title: Understanding Black-box Predictions via Influence Functions
- Authors / Org: Pang Wei Koh, Percy Liang / Stanford
- Link / arXiv: https://arxiv.org/abs/1703.04730 / https://proceedings.mlr.press/v70/koh17a.html
- Date read: 2026-08-05
- Review date: 2026-09-02
- Tags: [data-attribution, influence-functions, data-curation, debugging, quality, classic] ## 一句话总结
用鲁棒统计里的 Influence Function，把黑盒模型的单条预测“回溯”到哪些训练样本最负责，不用重训就能估计删掉/改动某条训练数据对测试 loss 的影响 —— 是数据归因（data attribution）和数据清洗的鼻祖工作。 ## 核心
### 1. Motivation
黑盒模型（深度网络）预测难解释。以往解释都围绕“固定模型对输入多敏感”，但作者问：模型本身是从哪里来的？能不能从训练数据层面解释。暴力做法是逐条删掉重训，太贵。需要一个便宜的、闭式的近似。 ### 2. Data Pipeline
- 不是直接讲 data pipeline，但提供了一个通用工具来 **评价训练数据的价值**： `训练集 z_i = (x_i, y_i)` → 训练得到 `θ_hat` → 对任一测试点 `z_test`，计算所有训练点的影响分数
- 公式： - 参数影响：`I_up,params(z) = - H^{-1} ∇L(z, θ_hat)`，H 是平均 Hessian - 去掉一点的近似：`θ_{-z} - θ ≈ -1/n I_up,params` - 对测试 loss 的影响：`I_up,loss(z, z_test) = -∇L(z_test)^T H^{-1} ∇L(z)`
- 这就把“数据→模型→预测”链路可微分了。 ### 3. Key Tricks（怎么算得动）
1. **不需要显式求逆 H**：只解 `H^{-1} v`，即 Hessian-vector product (HVP)。用 Pearlmutter trick + 共轭梯度 / 随机估计 LiSSA，复杂度 O(np) 近似线性，支持上十万维模型。
2. **非凸/不可微也管用**：理论要求凸+二阶可微，但实验在 CNN / 非凸上近似依然有效（加 damping `H+λI` 保证可逆）。
3. **Input 扰动版 `I_pert,loss`**：把训练点 `x → x+δ`，算 `∇_x∇_θ L`，找到对测试点最不利的微小扰动——反向构造“投毒”样本，证明模型脆弱性。 ### 4. Results / 用途验证
在 MNIST Logistic / CNN、Spam 等上：
- **理解模型**：找对某测试最 helpful / harmful 的训练样本，比欧氏最近邻准得多（图1：同标签也会 harmful，如果长得不像）
- **Debug / 找错标**：按 `I_up,loss` 排序检查训练集，优先发现 mislabeled / 噪声样本，比随机/LOO 快得多
- **数据集攻击**：对单张训练图做人眼不可见扰动，能翻转几十个测试点的预测（adversarial training example）
- 理论上相关度与实际 leave-one-out 重训的相关性 >0.9 (logistic)，CNN 也有高相关。 ## 可迁移
- **对你现在 coding data 工作的 1-2 个直接可试的点：** 1. **Coding data 质量过滤器**：对你已有的 small-scale SFT/评测集，训一个小 proxy model (如 1B)，对高-loss 的 validation 码样本算 influence，筛出最 harmful 的训练代码（错的 API 用法、过时语言、抄答案的样本）。这是比“规则过滤”更模型的清洗法。 2. **合成数据价值评估**：你正在做合成 code data，别只看 pass@k。用 influence 对真实评测集打分：`I_up,loss(合成样本, 评测集)` 平均是否为负（降低 loss）。为负才留，省 GPU。 3. **Redundancy prune**：高 helpful 但彼此冗余的样本 cluster 只留 1 个，呼应你 Infra cost-saving 思维：用最少数据达到同样效果。 - **Infra 视角：** - LiSSA 这类随机二阶估计本身就是 infra 题：大规模 Hessian 逆向量积如何分片、checkpoint、容错。 - 这是 modern data attribution (TracIn, DataInf, LESS, MoE data routing) 的起点，后续都可接入你以后想做的 RL Data Flywheel 评估闭环。 ## 疑问 / 下一步
- 在 LLM (7B+) 上 H 巨大且非凸，damping + LiSSA 的误差到底多大？看后续：Grosse et al. 2023 (LoGRA/TracIn), DataInf, LESS (Xia et al. 2024) 怎么把 influence 做到 LLM instruction data selection 的。
- 今晚想：能否用 cheap proxy (Code Llama 1B) 算 influence，然后迁移到大模型？迁移性验证。 ## 原文金句 (1-2句)
> Influence functions give us a way to “differentiate through the training” to trace a model's prediction back to its training data.
> Even on non-convex and non-differentiable models where the theory breaks down, approximations to influence functions can still provide valuable information. ## 复现链接
- Official TF1 code: https://github.com/kohpangwei/influence-release
- PyTorch reimpl: https://github.com/PRAISE-Lab-Repository/pytorch_influence_functions

## 第二轮复习（2026-09-02）

### 1. 核心命题

这篇真正解决的 data 问题是：**不为每条训练数据重新训练一次模型，能否估计“某条数据若被加权、删除或微调，会怎样改变某个目标样本/目标集合的 loss”？** 它把数据质量从静态属性（长度、格式、重复度）改写成一个带模型和目标分布条件的反事实量：

$$I_{up,loss}(z,z_t)=-\nabla_\theta L(z_t,\hat\theta)^\top H_{\hat\theta}^{-1}\nabla_\theta L(z,\hat\theta).$$

其中正值表示**上调**训练点 $z$ 的权重会提高目标 loss，因而该点对 $z_t$ 是 harmful；删除一个样本等价于令权重变化 $\epsilon=-1/n$，所以删除后的目标 loss 变化近似为 $-I_{up,loss}/n$。核心不是给数据贴上全局“好/坏”标签，而是得到关系型价值：同一条数据对目标 A 可能 helpful，对目标 B 可能 harmful。

### 2. 图谱位置

- **前驱**：Cook / Weisberg 的鲁棒统计 influence function 与 infinitesimal jackknife，原本用于线性/广义线性模型中的异常点诊断；本文把它扩展成“训练数据 → 模型参数 → 单个预测”的可微归因。
- **后继**：Day 03 TracIn 去掉 $H^{-1}$，沿训练 checkpoint 累积梯度点积；Day 04 LESS 把目标梯度匹配、LoRA 梯度和随机投影变成定向 SFT 选数；Day 05 DataInf 则保留 influence 的曲率思想，用 LoRA/经验 Fisher 近似降低求逆成本。
- **直接对比 Day 01 StarCoder2**：Day 01 的 license、规则过滤、MinHash 去重、PII 与 decontamination 是 **model-agnostic gates**，便宜、可审计、适合全量粗筛，但不知道一条通过规则的数据是否真正帮助目标能力；Day 02 是 **model- and target-conditioned attribution**，能发现格式正常却损害目标 loss 的数据，但昂贵且依赖模型、checkpoint 与目标集。最合理关系是级联互补：先用 Day 01 缩池，再用 influence 做定向审计，不应互相替代。
- **与 Day 03 TracIn 的关键差别**：Influence Functions 是终点局部最优附近的反事实，并用 $H^{-1}$ 校正曲率；TracIn 是训练路径上的累积贡献，绕过 Hessian、扩展性更好，但结果依赖 checkpoint 与优化轨迹。一个回答“在当前解附近删点会怎样”，另一个回答“训练途中这条样本把模型往哪里推过”。

### 3. 机制深挖

#### 3.1 从加权到删除

把单条训练点 $z$ 的权重增加 $\epsilon$：

$$\hat\theta_{\epsilon,z}=\arg\min_\theta \frac{1}{n}\sum_{i=1}^{n}L(z_i,\theta)+\epsilon L(z,\theta).$$

对一阶最优条件关于 $\epsilon$ 隐式求导，可得：

$$I_{up,params}(z)=\left.\frac{d\hat\theta_{\epsilon,z}}{d\epsilon}\right|_{0}=-H_{\hat\theta}^{-1}\nabla_\theta L(z,\hat\theta).$$

再与目标梯度做内积就得到 $I_{up,loss}$。删除一点只是把无穷小加权近似外推到 $\epsilon=-1/n$。因此它本质上是**局部线性反事实**，不是精确 leave-one-out。

#### 3.2 为什么不能只看 loss 或梯度相似度

分数有三部分：训练点梯度 $g_z$、目标梯度 $g_t$、曲率预条件器 $H^{-1}$。高训练 loss 让 $\|g_z\|$ 变大，但“不常见方向”也会被 $H^{-1}$ 放大：若其他训练数据在某个方向曲率小，模型对该点施加的扰动缺少整体数据的“阻力”，影响就可能很大。因此高 influence 可能是错标，也可能是稀有但关键的长尾；直接删除 top-influence 会把异常和有价值覆盖混在一起。

#### 3.3 怎么算得动

不显式形成或求逆 $p\times p$ Hessian。对一个目标先解：

$$(H+\lambda I)s_t=g_t,$$

再对所有训练点打分 $I_i=-s_t^\top g_i$。论文给出两条路线：共轭梯度只需 Hessian-vector product；随机逆-HVP 用 Taylor/Neumann 递推并对多次估计取平均。这样每个目标只求一次 $s_t$，之后扫描训练点只需梯度点积。论文在 55,000 个 MNIST 训练点上使用随机估计 $r=10,t=5{,}000$；非凸 CNN 中加入 $\lambda=0.01$ damping。

#### 3.4 证据到底支持什么

- 10-class MNIST logistic regression 中，预测的 leave-one-out loss 变化与真实删点重训贴合；非收敛、非凸 CNN 上相关系数仍为 $R=0.86$。
- 对不可微 hinge loss 直接算不准；换成 smooth hinge 后，$t=0.001$ 时与真实重训的 Pearson $R=0.95$，$t=0.1$ 时为 $0.91$。
- Enron spam 人为翻转 10% 标签后，按 self-influence 安排人工检查，比按训练 loss 或随机检查更快修复数据与恢复测试准确率。
- 攻击实验说明影响高度集中也是风险信号：扰动 1/2/10 张训练图，可分别翻转 57%/77%/几乎全部被单独攻击的正确测试预测。

这些结果验证的是“局部归因近似有诊断价值”，不是“按分数删数据一定提升现代 LLM”。

### 4. 边界与反例

1. **局部性失效**：删一个小权重点时一阶近似合理；批量删除 5%–20%、改变数据分布或跨越 basin 后，影响不能简单相加，重训可能到另一个解。
2. **Hessian 与 checkpoint 敏感**：深网非凸、未收敛、存在负特征值和大量近零方向；damping 虽让求解稳定，却会改排名。若排名随 $\lambda$ 或随机种子大幅变化，就不该据此自动删数。
3. **目标集偏置**：分数只相对 $z_t$ 或目标集合成立。用 HumanEval 当 target 会偏向短函数题，并可能误删对 repo repair、SQL、测试生成有益的数据；被污染的 target 还会主动选出与 benchmark 相似的数据。
4. **高 influence 不等于坏数据**：错标、冲突样本会高；罕见 API、长尾语言、边界条件也会高。只按绝对值删 top 1% 会系统性损害覆盖，必须区分 harmful sign、执行正确性与 slice coverage。
5. **LLM loss 的长度/位置混杂**：按 token sum 算梯度会让长回答天然更大；按 token mean 又可能稀释关键错误。prompt、reasoning、final answer、tests 的 attribution 还可能互相抵消。
6. **论文没有证明**：没有全参数 LLM、生成式 coding、LoRA proxy→大模型迁移、固定 token budget 的数据选择实验，也没有证明单点 influence 能可靠预测成组数据交互。它展示了小模型上的诊断、错标修复和攻击，不应把后来的 LESS/DataInf 能力倒灌成本文结论。

### 5. 迁移到 coding / post-training data

做一个可执行的 **20k 候选 coding SFT 数据审计实验**，目标不是先删数，而是验证 influence 是否能提升人工/执行审查的命中率：

1. 先按 Day 01 做 license、secret/PII、exact/MinHash、benchmark contamination 粗筛；保留 repo、commit、language、task-type provenance。
2. 建一个去污染的 200 题 target set：code generation / bug repair / test generation / long-tail language 各 50 题；分别算 slice target gradient，不把所有能力压成一个平均分。
3. 用 0.5B–1B proxy 的固定 adapter 参数或最后若干层训练到稳定 checkpoint；对 sequence loss 做统一长度归一化。用 HVP+CG 解 $(H+\lambda I)s_k=g_{target,k}$，在 $\lambda\in\{10^{-3},10^{-2},10^{-1}\}$ 下检查排名稳定性。
4. 对每条候选记录 $I_{i,k}=-s_k^\top g_i$。只把多个 target slice 上稳定为 harmful 的 top 1% 放进 quarantine；同时抽样 high-helpful、near-zero 和随机样本作对照。
5. 人工 + sandbox execution 标注：错误答案、过时 API、测试伪通过、依赖不可复现、benchmark 泄漏、真正长尾难例。核心指标是 harmful queue 的坏样本 precision / recall，而不是分数本身。
6. 最后在固定 token budget 下训练三组：随机、Day 01 规则门、规则门 + influence quarantine；报告 clean pass@1、repo-level repair、长尾 slice、污染命中率和每 1k GPU-hour 的收益。若 influence 排名对 damping/seed 不稳，或删数收益不超过随机置信区间，就停止自动化，只保留人工审计优先级功能。

### 6. 今天的一道思考题

综合 **Day 01 StarCoder2、Day 02 Influence Functions、Day 03 TracIn**：一个 coding SFT 池中，某条 Rust bug-fix 样本通过所有规则和执行测试，但对 HumanEval target 的 Influence score 为 harmful，对 repo-repair target 为 helpful；TracIn 又显示它在训练早期 helpful、后期 harmful。你会保留、降权、隔离还是删除它？

请设计决策协议并说明：
- 如何区分 benchmark 偏置、长度效应、曲率/damping 误差与真实负迁移；
- Day 01 的 provenance / coverage 约束如何防止模型分数吞掉 Rust 长尾；
- 用哪些 slice、checkpoint、反事实重训和置信区间证据，才足以推翻“保留这条样本”的默认决定。

## 原文链接
- Paper: https://proceedings.mlr.press/v70/koh17a.html
- arXiv: https://arxiv.org/abs/1703.04730
- GitHub NOTES: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-02-2017-influence-functions


## 数学补充：为什么 Influence 比 TracIn 多一个 $H^{-1}$

统一记号，避免把 target 和训练 step 都写成下标 $t$：

$$g_z=\nabla_\theta L(z,\theta),\qquad g_*=\nabla_\theta L(z_{\mathrm{target}},\theta),$$

其中 $z$ 是候选训练样本，$z_{\mathrm{target}}$ 是目标/验证样本。

### 1. TracIn：一次训练更新的即时作用

若训练样本 $z$ 产生一步 SGD 更新：

$$\theta'=\theta-\eta g_z,$$

对目标 loss 做一阶展开：

$$L_*(\theta')\approx L_*(\theta)-\eta g_*^\top g_z.$$

所以目标 loss 的下降量为：

$$S_{\mathrm{TracIn}}(z,z_*)\approx \eta g_*^\top g_z.$$

它测的是：**在当前模型状态下，这条训练数据的即时更新是否与目标梯度同向。** TracInCP 只是沿多个 checkpoint 累加这个量：

$$S_{\mathrm{TracInCP}}(z,z_*)=\sum_k\eta_k(g_*^k)^\top g_z^k.$$

### 2. Influence：改变数据权重并重新达到最优点

Influence 不是只走一步。它把 $z$ 的训练权重永久增加 $\epsilon$，然后让模型重新优化：

$$\theta_\epsilon=\arg\min_\theta\left[L_{\mathrm{train}}(\theta)+\epsilon L(z,\theta)\right].$$

新的最优点仍满足一阶条件。对该条件做隐式求导：

$$\frac{d\theta_\epsilon}{d\epsilon}\Big|_{\epsilon=0}=-H^{-1}g_z.$$

因此目标 loss 的变化是：

$$\frac{dL_*}{d\epsilon}=-g_*^\top H^{-1}g_z.$$

若把“降低目标 loss”定义为正的 helpful score：

$$S_{\mathrm{IF}}(z,z_*)=g_*^\top H^{-1}g_z.$$

所以 $H^{-1}$ 来自**重新优化后的全局参数响应**：改变一个样本的权重后，其余训练数据会通过局部曲率抵消或放大该扰动。

### 3. Hessian 特征方向上的差别

若

$$H=Q\,\mathrm{diag}(\lambda_i)Q^\top,$$

并把 $g_*,g_z$ 在这些方向上的分量分别记作 $a_i,b_i$，那么：

$$S_{\mathrm{TracIn}}\propto\sum_i a_i b_i,\qquad S_{\mathrm{IF}}=\sum_i\frac{a_i b_i}{\lambda_i}.$$

因此 TracIn 对各方向做普通梯度点积；Influence 用 $1/\lambda_i$ 校正曲率：平坦方向被放大，高曲率方向被压低。高 loss 或大 gradient norm 并不自动等于高 Influence，关键还包括扰动所在方向是否容易被整个训练集“拉回来”。

### 4. 两者为什么又有联系

在局部二次损失、小步长 gradient descent 且动力学稳定时：

$$H^{-1}\approx\eta\sum_{k=0}^{\infty}(I-\eta H)^k.$$

这说明 $H^{-1}$ 可以理解为：一次扰动经过未来许多步局部优化后残留作用的总和。于是：

- **TracIn** 是 path-wise 的即时梯度记账；
- **Influence** 是 endpoint 的重优化反事实；
- checkpoint 轨迹会隐式包含曲率动力学，但 TracInCP 没有显式、精确地应用同一个 $H^{-1}$。

若 optimizer 是 Adam 或带 momentum 的方法，一步真实作用更接近 $g_*^\top\Delta\theta_z$，而不只是 $\eta g_*^\top g_z$；这也是 LESS 加入 optimizer-aware 表示的原因。
