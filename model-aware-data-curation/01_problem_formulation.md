# 01 — 统一问题：Value × Coverage × Safety

## 元信息
- 内容类型：跨论文方法论综合，不对应单篇论文
- 核心来源：[Influence Functions](https://arxiv.org/abs/1703.04730) · [TracIn](https://arxiv.org/abs/2002.08484) · [LESS](https://arxiv.org/abs/2402.04333) · [Prismatic Synthesis](https://arxiv.org/abs/2505.20161)
- 完整论文索引：[papers.md](papers.md)


## 1. 从静态过滤到模型闭环

这章不是在讲一篇论文，而是在建立一个**统一数学框架**。核心是：数据价值不是固定的，而取决于“当前模型缺什么”。

传统 data curation 给每条样本一个与模型弱相关的静态分数：规则质量、长度、语言、去重距离。Model-aware curation 则显式依赖当前模型状态 $\theta_t$：

$$\mathcal D_{t+1}=\mathrm{Curate}(\mathcal P_t,\theta_t,V_t,B), \qquad \theta_{t+1}=\mathrm{Train}(\theta_t,\mathcal D_{t+1}),$$

其中 $\mathcal P_t$ 是候选池，$V_t$ 是目标/保护验证集，$B$ 是训练预算。因为 $\theta$ 在变，数据价值也会变；因此这不是一次性 ETL，而是控制回路：从候选池中，结合当前模型、验证目标和预算，选出下一轮训练集；训练后得到新模型，再重新选数据，形成闭环。

## 2. 三个不可互相替代的目标

对样本 $z$ 计算低维梯度表征 $\tilde g_z\in\mathbb R^d$，定义：

### Target value：能否推动目标能力

$$v(z)=\cos(\tilde g_z,\bar g_{V^+}), \qquad \bar g_{V^+}=\frac1{|V^+|}\sum_{v\in V^+}\tilde g_v.$$

如果样本 $z$ 的梯度与目标验证集梯度同向，训练它大概率能改善目标能力。

### Coverage gain：是否补了新方向

若已选集合为 $S$，可用谱熵增益：

$$c(z\mid S)=\log \mathrm{GV}(S\cup\{z\})- \log \mathrm{GV}(S).$$

它问的不是“样本新不新”，而是“是否带来了当前数据集没有覆盖的新梯度方向”。

### Safety / retention：是否与保护能力冲突

$$r(z)=\max\left(0,-\cos(\tilde g_z,\bar g_{V^-})\right),$$

$V^-$ 代表要保留的一般能力、格式约束或安全行为。负余弦表示一步更新可能增加保护集损失。如果新样本梯度与旧能力保护集反向，训练它可能导致遗忘，因此施加惩罚。

## 3. 不是简单加权平均

一个可部署的选择形式是带约束的预算优化：在预算内最大化 **目标收益 + 梯度覆盖**，同时要求平均遗忘风险不超过 $\epsilon$，并且每条样本必须通过正确性、执行和污染检查：

$$\max_{S\subseteq\mathcal P,\ |S|\le B} \sum_{z\in S}v(z)+\lambda\log\mathrm{GV}(S) \quad\text{s.t.}\quad \frac1{|S|}\sum_{z\in S}r(z)\le\epsilon, \quad q(z)=1.$$

$q(z)$ 是 correctness / execution / contamination gate。它必须在梯度目标之外：一个错误答案可能有很大梯度、很高新颖度，却仍然不该进训练集。

拿 coding data 举例：$V^+$ 是模型经常失败的动态规划题，$V^-$ 是原本会做的通用代码题；我们要选出既能修复动态规划短板、又覆盖新算法方向、还不会破坏已有能力的正确样本。整章本质上是：**选有用的、补缺的、且不会添乱的数据。**

## 4. Pareto front 比单一总分更稳

保留三维向量：

$$\mathbf s(z)=(v(z),\ c(z\mid S),\ -r(z)).$$

先剔除被 Pareto 支配的候选，再按当前阶段调度：

- 冷启动：coverage 权重大；
- 定向修能力：target value 权重大；
- 连续更新：retention 约束更严；
- RL 中后期：必须随 $\theta_t$ 重算，避免旧分数过期。

## 5. 假设与失效模式

1. **代理模型一致性**：小模型的梯度几何需能迁移到目标模型；必须抽样做 rank-correlation 校准。
2. **验证集代表性**：target alignment 只能优化 $V^+$ 表示的目标，窄验证集会产生“朝 benchmark 过拟合”。
3. **梯度可比性**：长度、loss reduction、layer choice、optimizer preconditioning 都会改变方向；需固定协议。
4. **相关不等于因果**：G-Vendi 与 OOD 的高相关来自受控实验，不替代 train-on-subset 验证。
5. **非平稳性**：RL policy 或持续学习状态变化后，旧 datastore 需刷新。

下一步：[`02_attribution_to_targeting.md`](02_attribution_to_targeting.md) 将这些量连接到具体方法；[`06_system_architecture.md`](06_system_architecture.md) 给出工程实现。

<!-- NAVIGATION -->
## 导航

- 上一篇：[目录](README.md)
- 下一篇：[02 归因到目标化](02_attribution_to_targeting.md)
- 回到：[目录 README](README.md) | [论文证据](papers.md) | [路线图](README.md#路线图)

> 串联：01 统一框架 → 02 归因/目标化 → 03 覆盖 → 04 生成 → 05 安全 → 06 系统 → 07 Coding 落地 → 08 边界 → 09 SPICE → 论文证据

