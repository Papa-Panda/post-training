# EP99 — TRPO重生：大模型时代的信任域策略优化
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP99-trpo-rebirth.html

> 「我们没有办法拿到 $\pi_\theta$ 的采样，我们只有 $\pi_{roll}$ 的采样。」——讲者对全场问题的出发点概括（字幕口径，约 32 分钟处）

## 元信息

- 期号：青稞Talk EP99
- 标题：TRPO重生：大模型时代的信任域策略优化
- BV：BV1UgrKBNEJc
- 时长：01:02:29（总时长 3749 秒；讲授约 51 分钟 + Q&A 约 11 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：李英儒（字幕开场自述「我叫李英儒」；主持人致谢作「映入博士」，单位与头衔字幕未自述，以公开资料为准）
- 相关论文/文献（均为讲授中点名，编号以原文为准）：
  - Kakade & Langford（2002）：surrogate objective 与 performance difference lemma 的源头（讲者反复强调「一阶近似 2002 年就提出来了」）
  - TRPO（2015）：信任域策略优化，讲者称其为「影响了深度强化学习十年发展最重要的算法基础」
  - PPO（2017）：对 TRPO 的非精确近似（token 级 clipping）
  - 讲者 2019 年工作：把 action 级 conditional divergence 改为 state-action joint distribution divergence（字幕口径，会议名识别不清）
  - 讲者团队 2025 年 9 月博客：训推不一致导致 RL 训练崩溃的分析（分 part 1/2/3，字幕口径；本 talk 的内容「beyond 这个 blog 很多」，配套论文讲者称近期会放出）
- 相关代码：verl 与 slime 中均已实现讲者所讲的 rollout correction / sequence 级 masking（讲者口径；verl 文档有 rollout correction 一节与推荐参数）
- 字幕原文存档：本地 `transcripts/EP99.txt`（1202 条，末条 3745.16 秒）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP99，已存档）清洗提炼。AI 字幕识别误差较多：rollout policy 在字幕中作「row/roll/派柔」等，surrogate objective 作「三炮/sarrogate」等，K1 estimator 作「k one s mor」等，均按上下文还原为通用术语；个别专名（如讲者 2019 年工作的会议名、其团队一篇相关论文名、采用其方法的公司名）字幕识别不清，正文只记脉络、不写死。凡讲授口径的数字与事件均标注「字幕」来源。

## 一句话总结

讲者从大模型 RL 的三个 off-policy 来源（训推不一致、MoE 路由不连续、异步训推陈旧）出发，回到 2002 年 Kakade & Langford 的 surrogate objective 与 performance difference lemma、2015 年 TRPO 的 sequence 级信任域这条理论主线，论证 PPO 的 token 级 clipping 在长 horizon 下既挡不住负 advantage 大 ratio 的梯度、又管不了被污染的后续 context；解法是回到 sequence 级：用 K1 估计（平均 log ratio、长度无关）度量整条序列的漂移，超阈值就整条 mask/reject，从而近似恢复 TRPO 式的单调改进保证，该实现已进入 verl 与 slime 并解决了公开的训练崩溃 issue。

## 核心

### 1. 大模型时代的三个 off-policy 来源

讲者先把问题摆出来：采样分布与更新策略不一致这件事在经典 RL 里就有，但大模型给了它三种新形态。

1. **训练/推理引擎不一致**。训练与推理用两个引擎，即便参数相同，算出的 log-prob 也不同。根源是浮点加法没有结合律，加上训练与推理的 kernel 优化方式不一致；在自回归范式下每步误差被逐步放大，长 context 下更严重。讲者团队 2025 年 9 月写过博客专门 demystify 这件事导致的训练崩溃（字幕口径）。
2. **MoE 路由的不连续性**。Router 的 top-k 选择是非连续函数，logit 的微小扰动就可能让选中的 expert 组合突变，同一 token 在不同 expert 下的概率可以从 0.9 跳到 0.001（字幕口径示例）。这不是随机噪声，而是由非连续性决定的跳变，会造出非常奇怪的 importance ratio。
3. **异步训推的策略陈旧**。Agentic RL 要求训推分离加速，rollout 生成时所用的策略可能已经离最新更新的参数很远，rollout 分布与当前策略之间天然有 gap。

三者的共同数学本质：**在分布偏移下做策略优化**——给定任意两个 off-policy 的策略，如何稳定地提升性能。

### 2. 理论回溯：从 2002 年的 surrogate objective 到 TRPO

讲者用自回归生成的记号重述了这条主线（不用 MDP 术语，用 LM 的记号）：prompt $x$ 采样后生成长度为 $T$ 的 response，reward 一般是 terminal reward；优化目标 $J(\pi_\theta)$ 是期望回报。问题在于拿不到 $\pi_\theta$ 本人的采样，只有 $\pi_{roll}$ 的采样。

- **为什么 sequence 级 importance sampling 不行**：sequence 级 ratio 是每个 token 级 ratio 的连乘，两个分布稍远方差就爆炸。能用的只有 token 级修正。
- **Surrogate objective（2002，Kakade & Langford）**：从 $\pi_{roll}$ 采样轨迹，只对单个 token 做 importance weighting，配合 $\pi_{roll}$ 分布下的 advantage function，得到替代目标 $L$ 。Schulman 等人后续沿用了这套目标。
- **一阶近似的含义**：在 $\pi_\theta = \pi_{roll}$ 的点上， $L$ 与原始目标 $J$ 的函数值和梯度都一致（是 $J$ 在该点的切平面）。所以在足够小的范围内提升 $L$ 就能提升 $J$ ——问题只剩下：这个「足够小」到底是多大。
- **Performance difference lemma（同为 2002 年）给出了误差的量法**： $J$ 与 $L$ 的差主要来自 context 分布（即状态分布）的偏移。把偏移累积起来有两种界：按 token 级 divergence 累积会得到 $T^2$ 阶的误差（每步一个 $T$ 、再对 $T$ 步求和）；按 sequence 级 divergence 界定则省掉一个 $T$ 因子。今天的 sequence 长度已经到几万甚至几十万 token（字幕口径），token 级的界要求单步漂移小到不现实，否则优化替代目标对原始目标毫无保证——讲者直言，没有保证就「不能期待它一直往上涨」，崩溃是迟早的事。
- **单调改进的机制**：只要保证 $L$ 减去误差界这个下界函数非负且被提升，就能保证 $J(\pi_{new}) \ge J(\pi_{roll})$ 。这正是优化里的 majorization-minimization 套路：每步构造新的下界函数、最大化它、再在新点重构。讲者强调 TRPO（2015）就是这个思想在深度 RL 里的落地：在 sequence 级 KL 定义的信任域内最大化 surrogate objective。注意他特别点出 TRPO 用的是 sequence 级（joint state-action 分布）的散度——他自己 2019 年的工作也论证过 joint divergence 优于 action 级 conditional divergence（字幕口径）。

讲者还埋了一条关键线索：可以把散度改写成**与 $T$ 无关的量**——平均到每步的 divergence。这个量后面成为其方法能处理长 context 的支点。

### 3. PPO 为什么在大模型上不够用

TRPO 要算信任域需要共轭梯度、海森矩阵与线搜索，计算复杂，于是 PPO 用 token 级 clipping 做近似，其隐含假设是：**token 级 clipping 能起到 sequence 级信任域的作用**。讲者指出这个假设在大模型时代有两处失效：

1. **负 advantage + 大 ratio 不被截断**。PPO 的 clip 只在 advantage 为正且 ratio 超上界、或 advantage 为负且 ratio 超下界时归零梯度；但 advantage 为负而 ratio 很大的 token 完全可能出现（MoE 路由突变就会造出来），这类 token 不被 clip、带着巨大的梯度进更新，造成很不稳定的训练。讲者说他们自己的相关论文讲过这件「一个梯度弄炸训练」的事（字幕口径，论文名识别不清）。
2. **Context contamination（上下文污染）**。就算在第 $k$ 个 token 处把漂移过大的 token mask 掉，它后续生成的整段 context 都已经是受这个偏移点影响的 off-policy 产物；只 mask 单点、不处理后续轨迹，等于在已被污染的 context 上继续优化。本质仍是单步漂移沿时间累积的老问题。

### 4. 解法：sequence 级 masking 把信任域硬执行回来

讲者的方法回到 TRPO 的精神，但换上便宜的执行方式：

- **度量**：需要高效估计 sequence 级 KL。方案是用 K1 estimator——本质是平均 log ratio（sequence ratio 连乘后取几何平均），天然长度无关（length-invariant）：长短序列都可比，且计算几乎没有额外开销，不需要 TRPO 式的二阶计算。K2、K3 estimator 也可考虑，各有取舍（字幕口径，未展开）。
- **执行**：若一条序列的 K1 估计超出信任域阈值 $\delta$ ，就把整条序列 mask/reject 出梯度——「超出 trust region 就不要把它算进 gradient」。只在漂移被 bound 住的样本上优化替代目标，就能近似恢复下界非负、单调改进的保证；这是 PPO 给不了的。
- **来历与验证**：这个方案最早出现在其团队博客发出后第二周（2025 年 9 月，字幕口径），公开层面解决了 verl 里 Megatron 训练崩溃的 issue（9 月底，字幕口径），对 MoE 与 dense 模型都用过，主治训推不一致类的 off-policy。实现已进入 verl 与 slime；讲者称某做 Devin 的 agent 公司（字幕作「DAVIN」，应指 Cognition）采用并验证了 BF16 下的稳定性，英伟达的博客也提到过这个工作（均为字幕口径）。
- **与 GSPO 的关系**：K1 的几何平均形式与 GSPO 的 sequence 级 ratio 有表面相似，但讲者强调 motivation 非常不一样——他这条线是从信任域理论推出来的。

### 5. Q&A 要点

- **MDP 里 context 算什么**：在大模型自回归建模下，整个 context 就是当前状态，下一个 token 是 action，状态转移是确定的。
- **阈值怎么选**：讲者建议可试 0.99 到 1.01 之间（字幕口径），并指路 verl 文档 rollout correction 一节有推荐值。
- **一阶近似能否升二阶**：原理上可以，但要高效地算曲率才 make sense；目前他们的做法是先用 sequence 级 masking 做快速且有效的硬信任域执行。
- **奖励设计类问题**（多目标按难易拆奖励模型等）：讲者明确划界——那与今天讲的训练稳定性正交，不在此讨论。
- **内部实验结果**：被问到某些内部对比，讲者表示试过但暂不方便分享，原则上「答案是 OK 的」（字幕口径）。

### 6. 讲者的阅读建议

他最后给的历史文献清单很明确：不要只从 PPO 那篇文章开始学 RL——「应该去把 02 年和 15 年这两篇 paper 都看一遍」，加上 17 年的 PPO。一阶近似、surrogate objective、performance difference lemma 全是 2002 年就有的东西，今天大模型 RL 的诸多「新问题」是这些老工具在长 horizon 下的重新显形。

## 关键数字总表

| 指标/事项 | 数值/口径 | 来源 |
|---|---|---|
| Surrogate objective 与 performance difference lemma 提出年份 | 2002 年（Kakade & Langford） | 字幕 |
| TRPO / PPO 提出年份 | 2015 年 / 2017 年 | 字幕 |
| 讲者 joint divergence 相关工作年份 | 2019 年 | 字幕 |
| Token 级误差界随 horizon 的累积阶 | $T^2$ 阶（sequence 级界省一个 $T$ 因子） | 字幕 |
| 单步 token 漂移在长序列上的复合 | 每步偏 $\epsilon$ ，整体偏 $\epsilon$ 的 $T$ 次方量级 | 字幕 |
| 现代任务 sequence 长度量级 | 几万至几十万 token | 字幕 |
| MoE 路由突变的概率跳变示例 | 同一 token 概率 0.9 → 0.001 | 字幕 |
| 训推不一致博客发布时间 | 2025 年 9 月（分 part 1/2/3） | 字幕 |
| Sequence 级 masking 方案出现时间 | 博客发出后第二周；verl 崩溃 issue 于 9 月底解决 | 字幕 |
| K1 阈值经验建议 | 0.99–1.01 之间试（具体见 verl 文档） | 字幕（Q&A） |
| 本期字幕规模 | 1202 条，末条 3745.16 秒 / 总时长 3749 秒 | 字幕存档实测 |

## 可迁移

- **把「信任域」当成 RL infra 的一等公民**：训推不一致、MoE 路由跳变、异步陈旧本质都是同一个分布漂移量在变大。工程上与其逐个打补丁，不如在框架里常驻一个 sequence 级漂移度量（平均 log ratio，几乎零成本）+ 超阈整条 reject 的硬执行，并监控被 reject 的比例作为训练健康度指标。
- **Review PPO/GRPO 类实现时先查两个口子**：负 advantage 且 ratio 极大的 token 是否仍在回传梯度；被 mask 的 token 之后同轨迹的 token 是否还在被优化。这两点是 token 级近似与 sequence 级信任域之间最实际的差。
- **长 horizon 场景优先 sequence 级量**：任何随 $T^2$ 涨的误差界在几万 token 的 agentic 任务上都会失效；做算法选型时先问一句「这个量的界与序列长度是什么关系」。
- 本期与本仓库的 verl/slime 相关期互为上下文：讲者明言该 rollout correction 已在 verl 与 slime 落地，做框架对比时可直接对照实现。

## 疑问 / 下一步

- K1、K2、K3 三种 KL estimator 的取舍讲者没有展开（各自的偏差/方差特性、什么场景换哪个），他只说「大家都可以去考虑 connection to theory」；其配套论文放出后值得对照补齐。
- Sequence 级整条 reject 会丢掉长轨迹里后段仍可用的样本，样本效率代价有多大、被 reject 比例的典型量级是多少，talk 中没有给数字；verl 文档的推荐配置与 issue 讨论是下一步核查点。
- 讲者 2019 年 joint divergence 工作与本方法的完整谱系、以及他提到的「把整套方法统一起来的一个命名」（字幕作「how precision making」，识别不清）要等论文放出后才能写准。
- 异步框架下 rollout 陈旧度如果远超阈值，整条 reject 与 partial 更新（如只用前若干 token）的边界在哪里，talk 未涉及。

## 原文金句

> 「我非常希望大家去了解一下这个事情……应该去把 02 年和 15 年这两篇 paper 都看一遍。」——讲者收尾的阅读建议（字幕约 47 分钟处，按干净口径转写）

> 「它不会往上涨的话，也就意味着它是有可能会崩溃的……因为你没有做到一个有效的纠正。」——讲者解释无保证的替代目标优化为何必然不稳定（字幕约 57 分钟处）

> 「如果我超出了这个 trust region，我就不要把它算进 gradient 里面了。」——讲者对 sequence 级 masking 的一句话定义（字幕约 45 分钟处）
