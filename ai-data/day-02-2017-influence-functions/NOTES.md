# 元信息
- Title: Understanding Black-box Predictions via Influence Functions
- Authors / Org: Pang Wei Koh, Percy Liang / Stanford
- Link / arXiv: https://arxiv.org/abs/1703.04730 / https://proceedings.mlr.press/v70/koh17a.html
- Date read: 2026-08-05
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
