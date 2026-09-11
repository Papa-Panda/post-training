# Q&A 归档（2026-09-10）：Rust bug-fix 样本的多目标 / 多时间归因与处置

- 日期：2026-09-10
- 关联论文：Day 01 StarCoder2、Day 02 Influence Functions（Koh & Liang 2017）、Day 03 TracIn（Pruthi et al. 2020）
- 来源：Day 02 NOTES 思考题的延伸讨论（用户主动提问）
- 状态说明：用户的判断与下面的分析结论一致；其中"隔离 + 时间衰减权重"为综合三篇论文得到的工程建议，可证伪，非论文原结论

## Q1：保留、降权、隔离还是删除？

**问题**：coding SFT 池中某条 Rust bug-fix 样本通过所有规则和执行测试，但对 HumanEval target 的 Influence score 为 harmful，对 repo-repair target 为 helpful；TracIn 显示它在训练早期 helpful、后期 harmful。你会保留、降权、隔离还是删除它？

**用户的判断**：

> 对 HumanEval target 的 Influence score 为 harmful，对 repo-repair target 为 helpful 完全很有可能。TracIn 又显示它在训练早期 helpful、后期 harmful，应该是 overfit 了。尝试降低权重。

**分析**：

三个信号各管一维，先分开读。

**Day 02 的 Influence：价值是目标的函数。**

$$I_{up,loss}(z,z_t) = -\nabla_\theta L(z_t,\hat\theta)^\top H_{\hat\theta}^{-1} \nabla_\theta L(z,\hat\theta)$$

这个分数以 $z_t$ 为参数——"harmful" 永远是"对谁 harmful"。对 HumanEval 有害、对 repo-repair 有益不仅可能，而是预期内：两个 target 的 $\nabla_\theta L(z_t)$ 在 $H^{-1}$ 度量下指向不同方向。机制上：Rust bug-fix 的梯度混着 Rust 特异成分（所有权模式、Result/Option 处理、diff 格式），与 HumanEval 的 Python docstring→code 分布异面甚至相斥，但与 repo-repair 的 issue→patch 分布同构。Day 02 的信号翻译成操作语言就是：**这不是样本的问题，是路由的问题**。

**Day 03 的 TracIn：价值是时间的函数。**

$$TracIn(z,z') = \sum_t \eta_t \langle \nabla\ell(w_t,z), \nabla\ell(w_t,z') \rangle$$

早期为正、后期为负：早期 checkpoint 上样本梯度与 test 梯度内积为正——模型在学"修 bug"这个共享方向；后期转负——共享方向学完，剩余梯度残差拟合样本特异成分（Rust 惯用法、这条 bug 的 idiosyncrasy），把参数往 test 目标反方向推。用户说的"overfit"精确表述就是：**先学特征、后记噪声**。这是 Day 02 给不出的信息：Influence 只给收敛点的总效应，TracIn 给的是时间路径。

**Day 01 的 StarCoder2：有效性门槛，不是价值判断。** 通过规则和执行测试只说明"这是合法的 Rust 修补"，没说"它对我的 target 有无边际价值"。validity gate 与 value estimation 是两层。

**决策矩阵**：

- 不删除：Day 01 有效 + Day 02 对 repo-repair 有益，删掉是扔真信号。
- 不原样保留：Day 02 对 HumanEval 有害 + Day 03 后期有害，全权重等于让有害成分在训练后期主导。
- 降权（用户的答案）：标量干预，同时按比例压小有益和有害成分。池子主目标是 HumanEval 类通用 coding 时，这是正确的第一步。
- 隔离（更 surgical）：有害是目标特异的，按目标路由——全权重进 repo-repair 专家 slice（Day 07/10 专家分支先例），通用池剔除或降权。标量降权是"目标结构"的最粗近似。
- 时间衰减（更 surgical）：有害是时间特异的，权重是时间的函数——在 TracIn 符号翻转的 checkpoint $t^*$ 之前全权重，之后衰减。把 TracIn 诊断直接编译成 curriculum。

**结论**：最精确的答案是**隔离 + 时间衰减权重**；用户的"尝试降低权重"是它的一阶近似，方向正确。可执行：先降权到 0.3–0.5，看 HumanEval delta 是否回正且 repo-repair 不掉；进一步可扫 checkpoint 找 TracIn 变号点 $t^*$ ， $t^*$ 之后线性衰减到 0——一次实验同时验证 overfit 假说和时间结构。

## Q2：对 coding data，这些归因技巧到底能做什么？预测？归因？curriculum？

**问题**：取一个合理大小的模型（如 20B 量级），在各 checkpoint 上算每个 (task, trajectory) 的梯度——Day 02 / Day 03 的方法能用来做什么？

**Setup（定义先行）**： $w \in \mathbb{R}^d$ （ $d \approx 2\times 10^{10}$ ），checkpoint 序列 $w_1,\dots,w_T$ ，学习率 $\eta_t$ 。样本 $z = (\text{task}, \text{trajectory})$ ， $g_t(z) = \nabla_w \ell(w_t,z)$ 。目标集总梯度 $G_t = \sum_{z'} g_t(z')$ 。TracIn 要的是被分析的那次训练自己的 checkpoint（TracInCP 标准做法是 SFT run 的 checkpoint）。

**归因**：反事实问题——"测试点 $z_t$ 的 loss 下降，有多少是样本 $z$ 的功劳？"（或反过来：这道题做错了，哪条训练 trajectory 把它推歪的？）公式即上面的 TracIn；Influence 给收敛点的牛顿修正版， $H^{-1}$ 在 Hessian 特征方向上按 $1/\lambda_j$ 重加权。能做 failure attribution（某题挂了 → top-k 负责样本 → 人工看是数据 bug 还是真难）、capability attribution（repo-repair 能力从哪批 trajectory 长出来）。边界：公式输出的是数值，不是人类可读的原因——"Rust 惯用法干扰"是我们对梯度对齐模式的解释，不是公式本身说的话。

**预测**：反事实问题——"这条刚合成的 trajectory（还没进过训练）加进去，HumanEval 会涨还是掉？"不用重训：

$$\Delta L(z') \approx \varepsilon \cdot I_{up,loss}(z_{new}, z')$$

其中 $z_{new}$ 的梯度只需要在已有的 $\hat w$ 上做一次前向 + 反向。用途有二：(a) 合成数据的价值预筛——Day 10 的 execution feedback 只管 validity，influence 管 value，拼起来就是"先筛选" pipeline 的完整形态；(b) self-influence $I_{up,loss}(z,z)$ 抓 outlier：通过了执行测试但行为怪异的 trajectory（自影响极大 ≈ 模型靠死记才拟合它）。边界有三层：(a) 一阶 Taylor 只对无穷小 $\varepsilon$ 精确， $\varepsilon=1$ （整条加入/删除）是外推；(b) 新样本的梯度算在实际走过的轨迹上，而"当初带上它训练"走的是另一条轨迹，小改动可忽略、大改动失效；(c) 群体效应不可加——top-k 各自有害的样本一起删，效果不等于分数求和，因为 $H$ 本身变了，这是近似不是恒等式。

**Curriculum**：反事实问题——" $z$ 在第 $t$ 步还值不值得给模型看？"定义瞬时价值 $v_t(z) = \eta_t \langle g_t(z), G_t \rangle$ ，TracIn 总分是它对时间的积分，而 curriculum 要的是被积函数本身。每阶段 upweight 当前 $v_t > 0$ 的样本——Q1 的"时间衰减权重"就是它的特例。它比手拍的 easy→hard 多一个关键性质：**可证伪**—— $v_t$ 的符号翻转是可测量的，schedule 错了能被数据打脸。Day 04 LESS 是"预测式选择"的实例化： $\arg\max_{|S|=k} \sum_{z \in S} \langle g(z), G_{val} \rangle$ ，用梯度内积当 influence 的便宜一阶代理。边界：拿历史 checkpoint 的 $v_t$ 排未来的 schedule，隐含假设"价值的时间结构会重演"——这是工程 heuristic，不是恒等式。

**工程现实检查**： $N$ 个 (task, trajectory) × $T$ 个 checkpoint × $d$ 维的梯度矩阵存不下。两条路：(a) 随机投影到 $k \sim 10^4$ 维，JL 引理保内积——有误差界的近似；(b) 只存 last-layer 梯度——有偏近似，偏多少未知，引用结论时要声明。20B 上的 $H^{-1}$ ：LiSSA 的 HVP 迭代次数与条件数挂钩，极贵；Day 05 DataInf 的对角近似是便宜替代。

**一句话串联**：Day 01 回答"能不能进池子"（validity），Day 02/03 回答"进了之后值多少、对谁值、何时值"（value × target × time）。Pipeline：StarCoder2 式过滤 → influence 预筛合成数据 → TracIn 时间结构定 curriculum → 按 target 隔离进专家 slice。Q1 的"降权"是这条 pipeline 里 curriculum 那一段的手动版。
