# 09 — SPICE：Fisher 覆盖之后，如何选出协调的集合

## 元信息

- 内容类型：单篇论文精读 + 跨论文几何辨析
- Paper: **SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training**
- Venue: ICLR 2026
- Sources: [arXiv v2](https://arxiv.org/pdf/2601.23155v2) · [OpenReview](https://openreview.net/attachment?id=9rCRy58TPF&name=pdf) · [official code](https://github.com/Chang-pw/SPICE)
- 适用范围：论文研究 SFT 数据选择；RL 和多模态只被列为未来方向。

## 1. 先给定位：创新主要在 set-level selection，不在 gradient representation

SPICE 仍从每个样本的 loss gradient 出发：

$$
g_i=\nabla_\theta \ell(z_i;\theta).
$$

per-sample gradient、proxy model、随机投影和 Fisher information 都已有清晰前序工作。SPICE 新增的关键不是一种新的 gradient embedding，而是一个逐步构造集合的规则：

1. 用 Fisher/log-det 找仍能增加 information volume 的候选；
2. 用 sign-sensitive penalty 避免候选抵消已选集合的平均更新；
3. 当边际信息增益衰减到阈值时提前停止。

因此最准确的归类是：

> **SPICE 是 information coverage + selected-set optimization coherence，不是通用的数据价值函数，也不是 retention safety 方法。**

## 2. Fisher/log-det：覆盖新的梯度方向

对已选集合 $S$，定义经验 Fisher 矩阵：

$$
F_S=\sum_{i\in S}g_i g_i^\top.
$$

信息效用为：

$$
U(S)=\log\det(I+\alpha F_S).
$$

加入候选 $x$ 的边际增益可由 matrix determinant lemma 写为：

$$
\Delta_x(S)
=
\log\left(
1+\alpha g_x^\top
(I+\alpha F_S)^{-1}g_x
\right).
$$

如果 $g_x$ 主要落在当前 Fisher 尚未覆盖的方向，逆矩阵项不会把它压小，$\Delta_x(S)$ 较大；如果它与已选子空间高度重复，增益就会下降。

纯 $U(S)$ 是 normalized、monotone、submodular。固定预算 $k$ 下，纯 Fisher greedy 有经典保证：

$$
U(S_{\mathrm{greedy}})
\ge
\left(1-\frac1e\right)U(S^\star).
$$

这部分是 D-optimal design / Fisher selection 的共同数学骨架，不是 SPICE 独有的 gradient representation。

## 3. 第一条关键边界：Fisher coverage 对符号不敏感

对任意梯度 $g$：

$$
gg^\top=(-g)(-g)^\top.
$$

所以给定同一个已选集合，$g$ 和 $-g$ 的 Fisher marginal gain 完全相同：

$$
\Delta_g(S)=\Delta_{-g}(S).
$$

这意味着 Fisher/log-det 能判断“是否带来新的轴或新的 information volume”，却不能判断“这个更新沿该轴向前还是向后”。

例如，若当前集合平均梯度是 $e_1$，候选分别是 $e_1$ 与 $-e_1$：

- 两者对 Fisher 都贡献 $e_1e_1^\top$；
- 但前者强化当前更新，后者抵消当前更新。

只靠 log-det 无法区分它们。

## 4. SPICE 的实际选择规则：给负向抵消加罚项

令已选集合平均梯度为：

$$
\bar g_S=\frac1{|S|}\sum_{i\in S}g_i.
$$

SPICE 定义候选相对于已选集合的冲突：

$$
\mathrm{conflict}(x\mid S)
=
\max\left\{
0,
-\frac{g_x^\top\bar g_S}
{\lVert g_x\rVert_2\lVert\bar g_S\rVert_2+\eta}
\right\}.
$$

实际 greedy score 是：

$$
\mathrm{score}(x\mid S)
=
\Delta_x(S)
-
\lambda\,\mathrm{conflict}(x\mid S).
$$

因此：

- $g_x$ 与 $\bar g_S$ 同向：penalty 为 0；
- 二者正交：penalty 为 0；
- 二者反向：按负 cosine 的强度扣分。

论文默认 $\lambda=0.1$，并报告 $\lambda\in[0.1,0.5]$ 时表现较稳定。SPICE+ 还可用信息增益而非总 score 提前停止：

$$
\Delta_{x_t}(S_{t-1})
\le
\omega\Delta_{x_1}(\varnothing),
$$

默认 $\omega=0.5$。

这解决的是**训练集合内部的更新协调性**：在仍获得信息覆盖的同时，减少下一条数据对当前 aggregate update 的抵消。

## 5. 理论解释在哪里，保证又止于哪里

论文定义空集合上的单点基线：

$$
b_x=\Delta_x(\varnothing)
=\log(1+\alpha\lVert g_x\rVert_2^2),
$$

以及边际增益衰减：

$$
\epsilon_x(S)=\Delta_x(S)-b_x\le 0.
$$

其近似分析把衰减幅度连接到 pairwise gradient interaction：

$$
|\epsilon_x(S)|
\le
C
\frac{
\alpha^2\sum_{y\in S}(g_x^\top g_y)^2
}{
1+\alpha\lVert g_x\rVert_2^2
}.
$$

也可用 total curvature：

$$
c=
1-\min_x
\frac{\Delta_x(D\setminus\{x\})}
{\Delta_x(\varnothing)}
$$

写出更细的数据依赖保证：

$$
U(S_{\mathrm{greedy}})
\ge
\frac{1-e^{-c}}{c}U(S^\star).
$$

但这里有一个不能略过的逻辑边界：理论中的 interaction 是

$$
(g_x^\top g_y)^2,
$$

它对正负号不敏感；实际 conflict penalty 却只惩罚

$$
g_x^\top\bar g_S<0.
$$

所以二者控制的不是同一个量：

- 理论 bound 控制 interaction magnitude 与 Fisher marginal-gain decay；
- 实际 penalty 控制 negative alignment 与更新抵消；
- 经典 submodular / curvature guarantee 直接覆盖的是纯 $U(S)$ greedy；
- 加入依赖当前集合均值的 sign-sensitive penalty 后，论文没有完整证明新 score 仍是 monotone submodular，因而不能把原保证原封不动地搬给 SPICE score。

工程动机与实验结果可以成立，但不能把它们写成“理论已经证明负冲突 penalty 最优”。

## 6. 四种不能混成一个 cosine 的梯度几何

### 6.1 Diversity / coverage：集合还缺哪个方向？

对象是整个已选集合的 span、kernel spectrum 或 information volume：

$$
\Delta_x^{\mathrm{cover}}(S)
=
\log\det(I+\alpha F_{S\cup\{x\}})
-
\log\det(I+\alpha F_S).
$$

参照物是**已覆盖子空间**，输出是 set-dependent marginal gain。它不需要一个单独的 target direction，而且 Fisher 外积本身 sign-blind。

### 6.2 Conflict / coherence：会不会抵消当前训练集合？

SPICE 比较：

$$
g_x\quad\text{vs.}\quad\bar g_S.
$$

参照物是**由 selector 自己构造、不断变化的已选集合均值**。它回答的是 optimizer coherence，而不是业务目标价值或安全保留。

### 6.3 Retention：会不会损害明确要保护的能力？

真正的 retention anchor 来自 held-out general/safety/replay set：

$$
g_{\mathrm{protected}}
=
\frac1{|P|}\sum_{p\in P}\nabla_\theta\ell(p;\theta),
$$

$$
\mathrm{risk}_{\mathrm{retain}}(x)
=
\max\{0,-\cos(g_x,g_{\mathrm{protected}})\}.
$$

参照物是**外部定义的 protected objective**。即使 $g_x$ 与 $\bar g_S$ 同向、SPICE conflict 为 0，它也可能与 $g_{\mathrm{protected}}$ 反向并导致遗忘。因此 selected-set conflict 不等于 retention safety。

### 6.4 Unlearnability：当前模型是否有可复用表征把它学进去？

梯度孤立只是一个静态症状：

$$
\mathrm{Isolation}(x)
=
1-
\max_{y\in D\setminus\{x\}}
|\cos(g_x,g_y)|.
$$

高 isolation 可能表示：

1. 真正稀缺而有用的新能力；
2. 标签错误或低质量 reasoning；
3. 当前表示空间缺乏可复用 feature，导致 RL/SFT 难以学会；
4. proxy gradient 噪声。

所以 unlearnability 需要时间维度和训练证据，例如 reward trend、重复正确 rollout 后的参数响应、跨 checkpoint gradient similarity 或 mid-training 后可学性变化。它不能由一次 cosine 或“离群 = 多样”直接推出。

## 7. 同一个二维例子，看清四者不等价

设当前已选均值、保护梯度和目标梯度分别为：

$$
\bar g_S=e_1,
\qquad
g_{\mathrm{protected}}=-e_1,
\qquad
g_{\mathrm{target}}=e_1.
$$

候选 $x$ 的梯度为 $g_x=e_1$：

- selected-set conflict $=0$，因为它与 $\bar g_S$ 同向；
- retention risk $=1$，因为它与 $g_{\mathrm{protected}}$ 完全反向；
- target alignment $=1$；
- Fisher coverage marginal gain 取决于当前在 $e_1$ 上已积累多少信息，而不由前三个 cosine 决定。

再看孤立候选 $g_u=e_2$：

- 对只覆盖 $e_1$ 的集合，它提供很高的方向 novelty；
- 但对 $g_{\mathrm{target}}=e_1$，target alignment 为 0；
- 若它来自错误标签或当前模型无法形成稳定 credit assignment，则“高 diversity”也不等于“可学且有用”。

因此四个信号应保留成分开的 feature、constraint 和 dashboard，而不是压成一个没有语义边界的 cosine score。

## 8. 实验结果与 proxy 边界

论文从约 97.5K 条、覆盖 math、code、ShareGPT 和 Alpaca 的池中固定选择 10%，在 Qwen2-7B 与 LLaMA2-7B 上评估 8 个 benchmark：

- Qwen2-7B：SPICE 平均 58.0，full-data 为 56.4；
- LLaMA2-7B：SPICE 平均 31.1，full-data 为 30.8；
- Qwen2-7B 的 IFEval：SPICE 38.6，full-data 为 33.5。

同 family 的小 proxy 到大 target transfer 较稳定；LLaMA proxy 到 Qwen2-7B 较弱，说明 gradient geometry 不能假定跨 architecture 不变。

数字核查边界：Table 2 把 LLaMA2 平均提升写成 `+1.8`，但表中平均分是 $31.1-30.8=0.3$；`1.8` 是八项 benchmark 差值之和，不是平均提升。

## 9. 可运行最小实现

[`code/model_aware_curation.py`](code/model_aware_curation.py) 新增：

- `fisher_logdet` 与 `fisher_marginal_gain`；
- `selected_set_conflict` 与 `spice_score`；
- `spice_greedy_select`；
- `gradient_isolation_scores`。

[`tests/test_model_aware_curation.py`](tests/test_model_aware_curation.py) 明确验证：

1. Fisher/log-det 对 $g$ 与 $-g$ 完全等价；
2. sign-sensitive selected-set conflict 能区分二者；
3. selected-set conflict 与 protected-gradient retention 可以给出相反判断；
4. 孤立梯度可以有高 novelty、同时没有 target value，因此不能自动标成 useful diversity。

这个 toy implementation 用完整小矩阵强调数学语义；生产实现需要低秩 sketch、Cholesky/rank-one update、候选预筛和 proxy-transfer 监控。
