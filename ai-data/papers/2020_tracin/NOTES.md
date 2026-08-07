# 元信息
- Title: Estimating Training Data Influence by Tracing Gradient Descent (TracIn)
- Authors / Org: Garima Pruthi, Frederick Liu, Mukund Sundararajan, et al. / Google
- Link / arXiv: https://arxiv.org/abs/2002.08484
- PDF: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
- Date read: 2026-08-05
- Tags: [data-attribution, tracin, influence-functions, coding-data, data-cleaning, self-influence]
- Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/
- Code: https://github.com/frederick0329/TracIn

## 一句话总结
不用 Hessian，把训练过程上每个 checkpoint 的梯度点积加起来估计影响 —— TracIn，工程上把 Influence Functions 做到了大模型可用的版本，是现在 LLM 数据清洗最实用的基线。

## 核心
### 1. Motivation
Influence Functions 要 $H^{-1}$，贵、不稳、非凸不成立。作者问：既然模型是 SGD 一步步走出来的，能不能直接看路上谁推了谁一把？理想的 influence 应该是整个轨迹上损失下降的累积。

### 2. Data Pipeline
- 训练时存 $K$ 个 checkpoint $\theta_{t_1}... \theta_{t_K}$，学习率 $\eta_t$
- 对任一训练点 $z$ 和测试点 $z'$：
  $$ TracIn(z,z') = \sum_{t} \eta_t \nabla L(z',\theta_t)^T \nabla L(z,\theta_t) $$
- 实际用 TracInCP：只用 checkpoint，忽略同一 checkpoint 内不同 step 的差异，batch 内近似
- Self-influence：$z'=z$ 时，分数越高，模型越靠死记这条点才能记住它

### 3. Key Tricks (3个最值得抄的)
1. **不要Hessian，只要点积**：$\eta \nabla_{test}\cdot\nabla_{train}$ 就是 influence。实现上就是两次 backward，算 cosine/dot，比 Influence Functions 快 10-100x
2. **Checkpoint 选择**：论文用最后几个 + 均匀采样，3-5个就够。实践：早期 checkpoint 抓语法/去重噪声，后期 checkpoint 抓语义难例。对 code 建议：epoch 1,2,末尾 各1个，共3个起步
3. **Self-influence = 脏数据探测器**：把训练集按 self-influence 排序，top 1% 拿去人工看，基本是 mislabeled / 爬到的孤岛代码 / 极长尾 API / 重复但标签矛盾的数据。Google 用这个清 10% 数据不掉点

### 4. Results / 用途验证
- CIFAR/MNIST 上：和真 LOO 相关性 >0.8，Influence Functions LiSSA 只有 0.6 且慢 10x
- 找 mislabeled：按 self-influence 排序，查 20% 数据就能找到 80% 错误标签，比 Influence Functions 快
- 工程上已用到 BERT/ResNet，checkpoint 存 3-10 个就行，内存可接受

## 可迁移
- **对你现在 coding data 工作的 1-2 个直接可试的点：**
  1. **脏数据过滤器 (今晚就能跑)**：用你 1B proxy 训 3 epoch，存 3 个 checkpoint，算所有训练 code 的 self-influence。Top 2% 导出，看是不是过时语言/错的API/从 StackOverflow 拷的带问号的代码。清掉再训 7B，看 HumanEval 有没有稳
  2. **合成数据价值评估**：合成的 code 不是全留。用 TracIn：对 LiveCodeBench 难例算 $TracIn(合成样本, 难例)$，平均为正才留。比“过得了单元测试就留”更贴近真实 eval
  3. **冗余 prune**：Helpful 的样本彼此梯度余弦相似度高，只留 1 个，呼应你 infra 省钱思维

- **Infra 视角：**
  - 不用二阶，天然 DDPer-friendly，checkpoint 存量 $K \times P$，可以只存 LoRA 分支梯度来把成本压到 1/100
  - 可扩展到 RL data：RLHF 偏好数据哪条最有用，用 TracIn 对 reward model 难例打分

## 疑问 / 下一步
- 3个 checkpoint 的采样策略对 code 任务是否最优？试早期密集 vs 均匀
- TracInCP 有偏，TracIn 论文里 checkpoint 内用一阶近似，误差多大？看后续 2023 TracIn++ / D-TracIn
- 今晚实验：跑通 `tracin_demo.py` 20 行版，对你手头小 code 集算 self-influence 分布，是否长尾

## 原文金句
> The influence of a training example can be estimated by tracing the loss reduction it contributes during training.

> Self-influence is a strong signal for mislabeled and outlier examples.

## 复现链接
- Official TF: https://github.com/frederick0329/TracIn
- PyTorch toy: `tracin_demo.py` (本文件夹)
- Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/
