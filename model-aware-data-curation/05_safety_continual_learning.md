# 05 — 安全与持续学习：GrADS / OGS

## 元信息
- 内容类型：双论文对照 + 持续学习综合；用 SPICE 反例澄清 selected-set conflict 不等于 retention
- Paper 1: **Learn More, Forget Less: A Gradient-Aware Data Selection Approach for LLM (GrADS)** — https://arxiv.org/abs/2511.08620
- Paper 2: **Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation (OGS)** — https://arxiv.org/abs/2602.06359
- 论文状态：均为预印本；本章把实验结论与工程推演分开标注。


## 1. 稳定—可塑性不是附加指标

目标域数据梯度为 $g_d$，保护集梯度为 $g_p$。一步 SGD 后：

$$L_p(\theta-\eta g_d) \approx L_p(\theta)-\eta g_p^\top g_d.$$

若 $g_p^\top g_d<0$，目标域更新会一阶增加保护集损失，即梯度冲突。高 target relevance 并不排除这种冲突。

## 2. GrADS：用梯度统计找“模型需要”的数据

**Learn More, Forget Less: A Gradient-Aware Data Selection Approach for LLM**（2025 预印本）在一次 preliminary SFT 中提取 embedding layer 和 LM-head layer 的样本梯度，再按梯度 magnitude / statistical distribution 的自适应准则选子集。

系统角色：

```text
full domain pool -> 1-epoch probe -> gradient statistics
                 -> self-guided subset -> standard SFT
```

论文摘要报告 5% selected data 已超过 full-data fine-tuning，并同时缓解 catastrophic forgetting；正文的 robustness 表述更保守：多数情况下 2.5%–5% 与全量相当。这个结果应视为其给定模型/医疗、法律、金融数据设置的实证，不应解释为“梯度越大越好”或通用 5% 法则。

局限：梯度 norm 没有方向信息；中等 norm 的样本仍可能与保护梯度反向。

## 3. OGS：把 gradient surgery 前移到 data selection

**Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation**（OGS，2026 预印本）定义要保护的一般能力 anchor，并用轻量 Navigator model 估计候选梯度几何，再选择更安全的样本给 Target model 标准训练。

两个基本量：

$$\text{conflict}(z)=\max(0,-\cos(g_z,g_p)),$$

$$\text{orthogonality}(z)=1-|\cos(g_z,g_p)|.$$

但“完全正交”只表示不干扰，也可能没有目标学习价值；所以实际目标应同时包含 domain gain：

$$\max_{S,|S|\le B}\ \mathrm{Gain}_{\mathrm{domain}}(S) \quad\text{s.t.}\quad \mathbb E_{z\in S}[\text{conflict}(z)]\le\epsilon.$$

OGS 将 optimizer 每步投影的几何思想变成离线/阶段性 data surgery，减少 target training 的 runtime overhead；论文还使用 Navigator–Target 架构与 RL-driven selection policy。它是较新的预印本，需独立复现后再判断跨模型迁移可靠性。

## 4. SPICE conflict 为什么不是 retention safety

SPICE 的参照物是已选集合的平均梯度：

$$\bar g_S=\frac1{|S|}\sum_{i\in S}g_i,$$

$$\mathrm{conflict}_{\mathrm{SPICE}}(x\mid S) = \max\{0,-\cos(g_x,\bar g_S)\}.$$

Retention 的参照物则必须来自明确的 protected set：

$$\mathrm{risk}_{\mathrm{retain}}(x) = \max\{0,-\cos(g_x,g_{\mathrm{protected}})\}.$$

两者虽然都用了负 cosine，语义却完全不同：

- $\bar g_S$ 由 selector 内生决定，问“会不会抵消当前训练集合”；
- $g_{\mathrm{protected}}$ 由安全、通用能力或旧任务目标外生定义，问“会不会损害必须保留的能力”。

若 $g_x=\bar g_S=-g_{\mathrm{protected}}$，则 SPICE conflict 为 0，但 retention risk 为 1。集合内部高度协调，完全可能一致地朝损害旧能力的方向更新。因此 SPICE 可作为 optimization-coherence 项，不能代替 held-out retention anchor、回放或安全评估。更完整的 Fisher/sign 分析见 [`09_spice_information_conflict.md`](09_spice_information_conflict.md)。

## 5. 与 GradAlign 的区别

| 方法 | 被保护/优化对象 | 信号 | 更新频率 |
|---|---|---|---|
| GrADS | 域能力 + 一般能力 | embedding/LM-head 梯度统计 | preliminary pass 后选数 |
| OGS | 域能力，同时保护 general anchor | 正交/冲突几何 + selector policy | 选择阶段动态决策 |
| SPICE | 已选训练集合内部协调 | candidate 与 selected-set mean 的负 cosine | 每次 greedy 加点时更新 |
| GradAlign | RL downstream validation | 当前 policy gradient 对齐 | 周期性重算 curriculum |

## 6. 持续学习控制面

每轮 $t$ 保存四类指标：

- `target_gain_t`：目标验证集变化；
- `retention_delta_t`：保护集相对初始/上轮变化；
- `conflict_rate_t`：候选中负余弦比例；
- `coverage_t`：G-Vendi 与簇 occupancy。

触发器：

```text
if retention_delta < -budget:
    tighten conflict threshold
    add replay/protection samples
if target_gain plateaus and coverage plateaus:
    generate toward sparse target-aligned clusters
if proxy/target rank correlation drops:
    refresh proxy or sample gradients on target model
```

## 7. 安全边界

梯度不冲突只是一阶局部条件，不保证：

- 长训练轨迹后不遗忘；
- 行为安全不退化；
- 代理模型与目标模型方向一致；
- 低冲突样本没有数据污染。

必须保留独立 held-out retention suite、红队/安全 eval、全量周期回归和 rollback checkpoint。

> 相关：SPICE conflict ≠ retention 的完整几何辨析见 [09 SPICE §6](09_spice_information_conflict.md#6-四种不能混成一个-cosine-的梯度几何)。

<!-- NAVIGATION -->
## 导航

- 上一篇：[04 Prismatic 主动生成](04_prismatic_synthesis.md)
- 下一篇：[06 系统架构](06_system_architecture.md)
- 回到：[目录 README](README.md) | [论文证据](papers.md) | [路线图](README.md#路线图)

> 串联：01 统一框架 → 02 归因/目标化 → 03 覆盖 → 04 生成 → 05 安全 → 06 系统 → 07 Coding 落地 → 08 边界 → 09 SPICE → 论文证据

