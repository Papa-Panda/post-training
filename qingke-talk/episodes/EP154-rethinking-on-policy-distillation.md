# EP154 — Rethinking On-Policy Distillation of Large Language Models：现象学、机制与 Recipe

## 元信息

- 期号：154
- 标题：Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
- BV：BV1WbeS6KEFf
- 时长：01:24:51
- 提炼日期：2026-09-22
- 相关论文：Yaxuan Li, Yuxin Zuo, Bingxiang He 等（THUNLP），https://arxiv.org/abs/2604.13016（2026-04-14，30 页 23 图）
- 相关代码：https://github.com/thunlp/OPD

> ⚠️ 提炼方式说明：B站反自动化，视频字幕未能直接获取。本纪要根据该期 talk 对应的公开材料还原——论文原文（arXiv:2604.13016）+ 官方代码仓库 thunlp/OPD。Q&A 即兴内容未覆盖。
>
> 关联：本仓库 125 期《重探 On-Policy Distillation（OPD）：三类典型失败以及修复路径》是同作者群的早期 talk；`~/workspace/opd-reading-list/papers/2026_revisiting-opd/` 的《Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes》(arXiv:2603.25562) 也被本论文引用。

## 一句话总结

OPD（on-policy distillation：学生自己生成 rollout，老师用每 token 的 log-prob 做 dense reward）在工业界已成标配（Qwen3、MiMo、GLM-5、Thinking Machines Lab 低成本复现），但实践中脆弱——**更强的老师可能蒸馏不动学生，而弱老师反而行**。这篇论文系统回答了"OPD 什么时候成、什么时候败"：成败被两个条件支配（① 师生 thinking-pattern 一致；② 老师必须带学生没见过的新知识，光是分数高不够）；机制上成功 OPD 就是师生在学生访问过的状态上、对共享的高概率 token 集合（占 97–99% 概率质量）的渐进对齐；并给出两个修复失败配置的 recipe（off-policy cold start、teacher-aligned prompt 选择），最后讨论 dense reward 的代价——reward 质量随轨迹深度退化，长程/agentic OPD 有天然天花板。

## 核心

### 背景/问题：OPD 的诱惑与脆弱

- **OPD 的设定**：区别于 off-policy distillation（在固定的老师生成序列上训，有 exposure bias），OPD 让当前学生 $\pi_{\theta}$ 自己采样 rollout $\hat{y}\sim\pi_{\theta}(\cdot\mid x)$ ，老师在学生走过的每个 prefix $\hat{y}_{<t}$ 上给出 next-token 分布 $q_t(v)=\pi_T(v\mid x,\hat{y}_{<t})$ ，以 per-token 的 KL / log-ratio 做 dense reward。目标是 sequence-level reverse KL，可精确分解为 token 级 KL 之和（论文 §2 Eq.1→2）。
- **三种实现粒度**：① sampled-token OPD——只看学生采样的那一个 token， $\ell_t^{\text{sample}}=\log p_t(\hat{y}_t)-\log q_t(\hat{y}_t)$ ，是 token-level reverse KL 的无偏单样本估计，工业常用；② full-vocab OPD——整词表算 KL，梯度最密但 $O(BTM)$ 内存；③ top-k OPD——取 student top-k 子集做 KL，中间路线。
- **痛点**：OPD 的"成功经验"不少，但"失败条件"无人系统研究。论文的出发观察：一个更强的老师可以**完全**蒸馏不动学生，而一个弱老师从更低的初始对齐出发反而成功。

### 现象学（§3）：支配成败的两个条件

**条件 1——thinking-pattern 一致性**（§3.1）。学生固定为 Qwen3-1.7B-Base，对比两个老师：Qwen3-4B（Non-thinking）vs Qwen3-4B-Base-GRPO（同一底座做 zero-RL 训出来的）。两个老师 benchmark 分数大体相当，但 GRPO 老师的**初始 overlap ratio**（师生 top-k 交集占比， $\mathcal{M}_{\text{overlap}}=\mathbb{E}_t[|S_t^{(p)}\cap S_t^{(q)}|/k]$ ）更高，蒸馏效果明显更好。关键：后期两条 overlap 曲线会收敛，但性能 gap 一直存在——**早期的 pattern mismatch 造成的损失，后期补不回来**。

**条件 2——新知识，不只是高分**（§3.2）。学生 R1-Distill-1.5B，两个老师：R1-Distill-7B（同管线放大，分数更高）vs Skywork-OR1-Math-7B（7B 又做了一轮 RL 后训）。同管线 7B 老师提升有限；RL 后训过的老师（带来新能力）提升显著，且 teacher-student **gap recovery rate**（ $(\text{Acc}_{\text{after}}-\text{Acc}_{\text{before}})/(\text{Acc}_{\text{teacher}}-\text{Acc}_{\text{before}})$ ）高得多。分数高可能只是"对同一份数据的拟合程度不同"，不是新知识。

**反向蒸馏验证**（§3.3，最漂亮的一组实验）。学生 JustRL-1.5B（从 R1-Distill-1.5B 做 RL 训出来的）：

1. 老师用它自己的 pre-RL checkpoint（R1-Distill-1.5B，**更弱**）→ 学生几乎精确退化回 pre-RL 水平，RL 挣来的全部吐回去——说明 **OPD 学的是 thinking pattern，会覆盖学生自己的**。
2. 老师换成 R1-Distill-7B（同家族、**分数更高**）→ 训练轨迹几乎不可区分，退化到同一水平。

结论三条：OPD 本质在学 thinking pattern；**benchmark 分数完全预测不了 OPD 结果**（甚至反向）；同家族 1.5B 和 7B 老师在学生访问状态上的局部分布几乎不可区分——这就是为什么大老师有时"白给"。

### 机制（§4）：成功 OPD 的 token 级签名

固定学生 R1-Distill-1.5B，对比成功老师 JustRL-1.5B vs 失败老师 R1-Distill-7B（分数相当、略强）：

- 成功 run：overlap ratio 从约 72% 稳步升到 91% 以上；overlap-token advantage 趋近 0（交集内学生不再过度自信）；**entropy gap 收窄**（学生在自己走过的状态上，学会了老师的不确定性 profile）。失败 run：三个指标全部停滞。
- **overlap 的 token 集合承载 97–99% 的总概率质量**——这不是集合层面的巧合，而是概率主导区域的对齐。
- 消融（§4.2）：top-k 支持拆成 overlap 部分 vs non-overlap 部分单独训。只优化 overlap 交集（k=16）几乎复现标准 Student Top-k 的全部收益；non-overlap 部分始终弱。且优化是 **self-reinforcing** 的：token 一旦进入共享高概率区被老师青睐，reverse KL 的 mode-seeking 就把更多质量压上去，把竞争者挤出学生 top-k——正循环。
- 统一机制表述：OPD 的主要效应就是**在学生访问的状态上，渐进式精炼学生分布对老师支持的高概率 token 的对齐**。这是成功 OPD 的签名，也是梯度的主战场。

### Recipe（§5）：两个修复失败配置的方法

**Recipe 1——off-policy cold start**（§5.1）：pattern gap 太大时，先用老师生成的 200K rollouts 对学生做 SFT（OpenThoughts3-1.2M 的 math 子集），再做 OPD（剩余 prompt 去重后约 30K）。SFT 初始化的学生初始 overlap 高、轨迹稳定、entropy gap 小，且**性能上限本身也更高**——cold start 改善的不只是早期优化。

**Recipe 2——teacher-aligned prompt 选择**（§5.2）：老师的策略被它后训练时见过的 prompt 塑造，所以 OPD 时用老师的 prompt 有好处，分两个粒度验证：

- **template 对齐**：同样 DAPO-Math-17K 的题，换成老师后训练用的 prompt 格式，三个 benchmark 全涨，overlap 起点和终点都更高——连 template 这种小事都显著影响 OPD。
- **content 对齐**：用和老师 RL 训练集一致的 prompt（vs 同域但不同的 DeepMath 子集），下游更强，学生概率质量更集中在共享 token 上。**代价**：学生熵显著降低——只在老师见过的 prompt 上做 OPD 会过度压熵。实践建议：**teacher-aligned prompts 混入 OOD prompts**，保熵、保探索。

### 代价（§6）：dense reward 的免费午餐并不免费

- **Reward 质量随轨迹深度退化**：R1-Distill-1.5B 对 JustRL-1.5B 做 OPD，max length 扫 0.5K→15K。0.5K/1K 太短没信号；3K/7K 最强；10K/15K 后期 overlap 崩塌。位置分析显示不稳定**从 response 尾部开始、向前传播**——老师在越来越陌生的学生 prefix 上给出越来越 noisy 的 reward。
- **老师 continuation 优势随 prefix 深度衰减**：从 1K prefix 的 +0.37 单调掉到 16K prefix 的 +0.02。dense reward 在长程上不可靠。
- **全局信息性 ≠ 局部可利用**：失败配置（R1-Distill-7B 老师）下，sequence mean reward 区分正确/错误 rollout 的 AUROC 为 0.75，和成功配置的 0.73 相当——信号质量没问题。问题在局部优化几何：大老师的 per-token advantage 虽大但**各向异性**（不同位置方向不一致），聚合成梯度时互相抵消，gradient norm 反而小。论文明确这是 hypothesis，未直接验证，留作 open question。
- **k 不必大**：sampled-token OPD 和 top-k（k≥4）效果相当；Top-1 明显差且不稳定——不是"token 太少"，而是 argmax 选择有偏、mode 集中，小的策略变化就会翻转 rank-1 token，reward 信号不稳定。k 超过 4 收益可忽略，只增加 teacher query 成本。

## 关键数字

| 指标 | 基线/对照 | 结果 |
|---|---|---|
| 成功 OPD 的 overlap ratio | 初始约 72% | 升至 91% 以上 |
| Overlap token 集合的概率质量 | — | 97–99%（师生双方全程）|
| JustRL-1.5B → R1-Distill-1.5B 的 gap recovery | — | >80% |
| 老师 continuation 优势（prefix 1K → 16K）| +0.37 | +0.02（单调衰减）|
| OPD 最优轨迹长度 | 0.5K/1K 太短，10K/15K 后期崩塌 | 3K/7K 最强 |
| Top-k 的 k | sampled-token ≈ k≥4 | Top-1 明显差且不稳定 |

评测设置：DAPO-Math-17K 训练；AIME 2024 / AIME 2025 / AMC 2023，avg@16（每题 16 采样，temperature 0.7）。

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **选蒸馏老师别只看 benchmark 分**：优先"同底座 + 额外 RL 后训"带来的新能力（Skywork-OR1-Math-7B 型），而不是同管线放大（R1-Distill-7B 型）。thinking-pattern 不一致是前置杀手，且早期损失不可逆——动手前先测师生初始 overlap ratio。
  2. **sampled-token OPD 已经够用**：k≥4 的 top-k 收益可忽略，省 teacher query 成本；轨迹长度是 sweet spot 超参（3K–7K），不要无脑拉长。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. OPD 训练监控看 **overlap ratio / entropy gap 曲线**而不是只看 loss——这是论文给出的成功签名，失败 run 早期就停滞，可做早停信号。
  2. 长程 agentic OPD 的天花板是结构性的：dense token reward 随深度退化，论文建议 hybrid（短 segment 用 dense token 监督 + 长程用稀疏 outcome reward）或 curriculum 逐步拉长监督 horizon——这和当前 agentic RL infra 的 reward 设计直接相关。
  3. 本 repo 的 opd-reading-list 有 10 篇 OPD 论文（含 2603.25562，被本论文引用），可对照看"三类典型失败"（125 期）和本篇的"两个条件"是否同构。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：§6.2 的 anisotropy 假设（per-token advantage 方向不一致→梯度抵消）论文明确没验证；如果成立，什么样的 objective 能利用各向异性的 reward 结构？这是 open question。
- 续作：同作者群的 Part II《Rethinking OPD II: One Training Example》(arXiv:2609.04172) 发现单 query 训练几百步仍持续提升、16 个语义不同 query 即达全数据效果——"OPD 是 data-overfed but algorithm-starved"。这和本篇的 recipe（补数据/对齐 prompt）放在一起看，值得深挖：瓶颈到底在数据还是在算法效率？

## 原文金句（1-2句）

> "a globally informative reward does not guarantee a locally exploitable one." —— 全局信息性的 reward，不保证局部可利用。这是 §6 的核心判词，也是选老师、设 k、定轨迹长度所有工程决策背后的第一性原理。
