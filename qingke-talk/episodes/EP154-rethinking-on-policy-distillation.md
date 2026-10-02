# EP154 — Rethinking On-Policy Distillation of Large Language Models：现象学、机制与 Recipe

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP154-rethinking-on-policy-distillation.html

> 「OPD 它是一个数据撑死、但是算法饿死的东西。」——讲者对第二篇工作的总结判词（字幕 40:31–40:37）

## 元信息

- 期号：154
- 标题：Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
- BV：BV1WbeS6KEFf
- 时长：01:24:51（讲授约 50 分钟 + Q&A 约 35 分钟；本场实际讲了 Rethinking 系列两篇工作，第二篇详见下文）
- 提炼日期：2026-10-02
- 分享嘉宾：何炳祥（清华大学 THUNLP 2024 级直博生，导师刘志远；据字幕开场自述订正——原论文还原版误将论文作者列整体作讲者栏）
- 相关论文：*Rethinking On-Policy Distillation of Large Language Models*（Yaxuan Li, Yuxin Zuo, Bingxiang He 等，THUNLP），https://arxiv.org/abs/2604.13016 （2026-04-14，30 页 23 图）；续作 *Rethinking OPD II: One Training Example*，https://arxiv.org/abs/2609.04172 （讲者字幕中未给出编号，沿用本仓库既有记录）
- 相关代码：https://github.com/thunlp/OPD （讲者提及论文、代码、模型、数据集均已开源，GitHub 已破 1000 star，字幕口径）
- B站链接：https://www.bilibili.com/video/BV1WbeS6KEFf/
- 字幕原文存档：本地 `transcripts/EP154.txt`（1859 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP154，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 thinking pattern 在字幕中作「singing pattern」、entropy 作「安全比」、verl 作「word」、JustRL 作「驾驶RL」，均以论文与公开资料为准）。论文（arXiv:2604.13016）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。注意：本场 talk 覆盖 Rethinking 系列**两篇**工作——第一篇（现象学、机制与 Recipe）与第二篇《One Training Example》（one-shot OPD），原论文还原版只覆盖第一篇，现已按字幕补全第二篇并新增 Q&A。
>
> 关联：本仓库 125 期《重探 On-Policy Distillation（OPD）：三类典型失败以及修复路径》是另一团队的同主题 talk；`~/workspace/opd-reading-list/papers/2026_revisiting-opd/` 的《Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes》(arXiv:2603.25562) 被本论文引用。

## 一句话总结

OPD（on-policy distillation：学生自己生成 rollout，老师用每 token 的 log-prob 做 dense reward）在工业界已成标配（Qwen3、MiMo、GLM-5、DeepSeek V4），但实践中脆弱——**更强的老师可能蒸馏不动学生，而弱老师反而行**。第一篇工作系统回答了"OPD 什么时候成、什么时候败"：成败被两个条件支配（① 师生 thinking-pattern 一致；② 老师必须带学生没见过的新知识，光是分数高不够）；机制上成功 OPD 就是师生在学生访问过的状态上、对共享的高概率 token 集合（占 97% 以上概率质量）的渐进对齐；并给出两个修复失败配置的 recipe（off-policy cold start、teacher-aligned prompt 选择）。第二篇工作再进一步：一条训练样本就能拿回全量数据的大部分收益，因为单条样本已能覆盖约 70% 的可监督 state 空间，真正的瓶颈不在数据而在算法每步约 2% 的吸收率——OPD 是「数据撑死、算法饿死」。

## 核心

### 背景/问题：OPD 的诱惑与脆弱

- **OPD 的设定**：区别于 off-policy distillation（在固定的老师生成序列上训，有 exposure bias），OPD 让当前学生 $\pi_{\theta}$ 自己采样 rollout $\hat{y}\sim\pi_{\theta}(\cdot\mid x)$ ，老师在学生走过的每个 prefix $\hat{y}_{<t}$ 上给出 next-token 分布 $q_t(v)=\pi_T(v\mid x,\hat{y}_{<t})$ ，以 per-token 的 KL / log-ratio 做 dense reward。目标是 sequence-level reverse KL，可精确分解为 token 级 KL 之和（论文 §2 Eq.1→2）。讲者开场补充了业界采用面（字幕口径）：Qwen3 在大小模型蒸馏中已用 OPD，小米 MiMo 用 MoPD 做模型合并，GLM-5、DeepSeek V4 也已采用（DS V4 甚至直接用 MoPD 管线替换了部分原有流程）。
- **三种实现粒度**：① sampled-token OPD——只看学生采样的那一个 token， $\ell_t^{\text{sample}}=\log p_t(\hat{y}_t)-\log q_t(\hat{y}_t)$ ，是 token-level reverse KL 的无偏单样本估计，工业常用；② full-vocab OPD——整词表算 KL，梯度最密但 $O(BTM)$ 内存；③ top-k OPD——取 student top-k 子集做 KL，中间路线。
- **痛点**：OPD 的"成功经验"不少，但"失败条件"无人系统研究。论文的出发观察：一个更强的老师可以**完全**蒸馏不动学生，而一个弱老师从更低的初始对齐出发反而成功。

### 现象学（第一篇 §3）：支配成败的两个条件

**出发实验**（字幕口径，讲者先讲了这组）：学生 R1-Distill-1.5B，对比两个老师——JustRL-1.5B（由该学生经大规模 RL 得到，讲者给出其 RL 规模为 32 卡跑了半个月）vs R1-Distill-7B（同 R1 轨迹刷出、自评更强）。结果：弱老师让学生在 200 step 时已拿回老师收益的七八十 %、300 step 时 90% 以上；强老师反而让学生基本学不动、在原地打转。

**条件 1——thinking-pattern 一致性**（§3.1）。学生固定为 Qwen3-1.7B-Base，对比两个老师：Qwen3-4B（Non-thinking）vs Qwen3-4B-Base-GRPO（同一底座做 zero-RL 训出来的）。两个老师 benchmark 分数大体相当，但 GRPO 老师的**初始 overlap ratio**（师生 top-k 交集占比， $\mathcal{M}_{\text{overlap}}=\mathbb{E}_t[|S_t^{(p)}\cap S_t^{(q)}|/k]$ ）更高，蒸馏效果明显更好。直觉（讲者字幕口径）：OPD 的奖励是老师在学生前缀上给的，老师若对学生前缀不熟悉、自己给出的 next-token 分布 entropy 很高，这个信号本身就不可靠。关键：后期两条 overlap 曲线会收敛，但性能 gap 一直存在——**早期的 pattern mismatch 造成的损失，后期补不回来**。

**条件 2——新知识，不只是高分**（§3.2）。学生 R1-Distill-1.5B，两个老师：R1-Distill-7B（同管线放大，分数更高）vs Skywork-OR1-Math-7B（7B 又做了一轮 RL 后训）。同管线 7B 老师提升有限；RL 后训过的老师（带来新能力）提升显著，且 teacher-student **gap recovery rate**（ $(\text{Acc}_{\text{after}}-\text{Acc}_{\text{before}})/(\text{Acc}_{\text{teacher}}-\text{Acc}_{\text{before}})$ ）高得多。分数高可能只是"对同一份数据的拟合程度不同"，不是新知识。讲者补充同一现象在 Qwen 家族上也成立（字幕口径）。

**反向蒸馏验证**（§3.3，最漂亮的一组实验）。学生 JustRL-1.5B（从 R1-Distill-1.5B 做 RL 训出来的）：

1. 老师用它自己的 pre-RL checkpoint（R1-Distill-1.5B，**更弱**）→ 学生几乎精确退化回 pre-RL 水平，RL 挣来的全部吐回去——说明 **OPD 学的是 thinking pattern，会覆盖学生自己的**。
2. 老师换成 R1-Distill-7B（同家族、**分数更高**）→ 训练轨迹几乎不可区分，退化到同一水平。讲者在 talk 中把这一组描述为：更强的老师反而让学生越学越差、甚至退回 RL 之前的版本（字幕口径）。

结论三条：OPD 本质在学 thinking pattern；**benchmark 分数完全预测不了 OPD 结果**（甚至反向）；同家族 1.5B 和 7B 老师在学生访问状态上的局部分布几乎不可区分——这就是为什么大老师有时"白给"。讲者的判词（字幕口径）：「更强，不代表更可学习。」

### 机制（第一篇 §4）：成功 OPD 的 token 级签名

固定学生 R1-Distill-1.5B，对比成功老师 JustRL-1.5B vs 失败老师 R1-Distill-7B（分数相当、略强）。讲者先介绍了三个观测指标（overlap ratio、overlap-token advantage、entropy gap；据字幕口径，这组指标已合入 verl 强化学习框架作为 OPD 的标准观测指标）：

- 成功 run：overlap ratio 稳步上升到 90% 以上（讲者字幕口径称甚至接近 100%；论文报告从约 72% 升至 91% 以上，为论文口径）；overlap-token advantage 趋近 0（交集内学生不再过度自信）；**entropy gap 收窄**（学生在自己走过的状态上，学会了老师的不确定性 profile）。失败 run：三个指标全部停滞。
- **overlap 的 token 集合承载 97% 以上的总概率质量**（讲者字幕口径；论文作 97–99%）——这不是集合层面的巧合，而是概率主导区域的对齐。
- 消融（§4.2）：top-k 支持拆成 overlap 部分 vs non-overlap 部分单独训。只优化 overlap 交集（k=16）几乎复现标准 Student Top-k 的全部收益；non-overlap 部分始终弱。且优化是 **self-reinforcing** 的：token 一旦进入共享高概率区被老师青睐，reverse KL 的 mode-seeking 就把更多质量压上去，把竞争者挤出学生 top-k——正循环。
- 统一机制表述：OPD 的主要效应就是**在学生访问的状态上，渐进式精炼学生分布对老师支持的高概率 token 的对齐**。这是成功 OPD 的签名，也是梯度的主战场。

### Recipe（第一篇 §5）：两个修复失败配置的方法

**Recipe 1——off-policy cold start**（§5.1）：pattern gap 太大时，先用老师生成的 200K rollouts 对学生做 SFT（OpenThoughts3-1.2M 的 math 子集），再做 OPD（剩余 prompt 去重后约 30K）。讲者强调（字幕口径）：cold start 的收益不只是初始 overlap 高，它的**蒸馏上限本身也明显更高**，不只是起点问题。

**Recipe 2——teacher-aligned prompt 选择**（§5.2）：老师的策略被它后训练时见过的 prompt 塑造，所以 OPD 时用老师的 prompt 有好处，分两个粒度验证：

- **template 对齐**：同样 DAPO-Math-17K 的题，换成老师后训练用的 prompt 格式，三个 benchmark 全涨，overlap 起点和终点都更高——连 template 这种小事都显著影响 OPD（讲者字幕口径：初始 overlap 会高特别多）。
- **content 对齐**：用和老师 RL 训练集一致的 prompt（vs 同域但不同的 DeepMath 子集），下游更强，学生概率质量更集中在共享 token 上。**代价**：学生熵显著降低（讲者字幕口径：熵后期基本趋近于零，diversity 丧失）。实践建议（论文口径）：**teacher-aligned prompts 混入 OOD prompts**，保熵、保探索。

### 代价（第一篇 §6）：dense reward 的免费午餐并不免费

讲者在这一节的立场很明确（字幕口径）：「稠密监督是有代价的，它并不是一个免费的午餐。」

- **Reward 质量随轨迹深度退化**：R1-Distill-1.5B 对 JustRL-1.5B 做 OPD，max length 做消融。讲者字幕口径称 7K/10K 效果还不错、再往后变差；论文口径为 3K/7K 最强、10K/15K 后期 overlap 崩塌（两口径不一致，并列标注）。位置分析显示不稳定**从 response 尾部开始、向前传播**——讲者给了实操建议（字幕口径）：训练出现 entropy 异常时先看尾部 token 是否已不稳，把后面 mask 掉也许就能缓解。
- **老师 continuation 随 prefix 深度失效**：让老师在学生已写 1000/4000/8000/16000 token 的前缀后续写，讲者的定性结论（字幕口径）是「学生说的话越多，老师也无力回天」。论文给出的定量口径是 continuation 优势从 1K prefix 的 +0.37 单调掉到 16K prefix 的 +0.02（论文口径，talk 未给数值）。
- **全局信息性 ≠ 局部可利用**：失败配置（R1-Distill-7B 老师）与成功配置下，sequence mean reward 区分正确/错误 rollout 的 AUROC 都在 70 多（讲者字幕口径；论文为 0.75 vs 0.73）——信号质量没问题。问题在局部优化几何：大老师的 per-token advantage 虽大但**各向异性**（不同位置方向不一致），聚合成梯度时互相抵消，gradient norm 反而小。讲者明确这是比较粗糙的讨论/假设，未直接验证，留作 open question。
- **k 与词表粒度**：论文口径为 sampled-token OPD 与 top-k（k≥4）效果相当、Top-1 明显差且不稳定。讲者在 talk 中的口径不同（字幕口径）：top-1 与 top-4 都出现过不稳定（指标 spike），K 增大后训练更稳；并指出 DeepSeek V4/V4.1 明确使用全词表、MoPD 多老师合并也用全词表，全词表对 long-horizon 与多老师场景的稳定性更好。两口径并列，结论方向一致（太小的支持集不稳），幅度以论文表格为准。
- 第一篇工作的边界（讲者自陈，字幕口径）：主要在较短任务、数学单 domain 上验证，long-horizon 任务探索不多。

### 第二篇（字幕口径）：One Training Example——OPD 是数据撑死、算法饿死

第二篇工作（讲者称 9 月初放出）问的是另一个问题：OPD 到底需要多少训练数据？

**现象**：只用一条训练样本做 OPD（one-shot OPD），在数学上就能拿回全量数据约 70% 多的收益（讲者先给的数字），多处口径下数学可达 80% 多；MoPD 场景下 16-shot 已与 full data comparable。讲者团队怕这是 cherry-pick，做了系统鲁棒性验证（字幕口径）：4 个 domain（数学、代码、指令遵循、工具调用）× 3 个模型家族（Qwen、Llama、OLMo），结论一致；样本难度（学生采样 8 次全对 / 4 对 4 错 / 全错）与 rollout 参数（3K/5K/7K 输出长度、采样温度）对结果影响都不大。把训练从 300 步拉到 1000 步，前 200–300 步就已到位，后面基本原地踏步。

**解释一：state coverage**。OPD 的监督发生在学生 rollout 到达的每个 state（token 位置）上——一条样本在成百上千次采样中也能产生上万个可监督 state，问题数量低估了监督总量。度量方法（字幕口径）：把 full-data 实验的全部 rollout 打 embedding、K-means 聚成 K=200 个桶，再看一条样本的 rollout 能落进多少桶。结果：全量定义为 100% 时，单条样本已覆盖约 70% 的 state 空间（约 100 step 时就覆盖到较高水平）；单条 response 本身约覆盖 60%；从 query 数看，1 条约 66%、16 条的 coverage 已达约 99%——与 performance 上 16 条即接近全量相互印证。讲者也给了跨 cluster 选样的变体（每个 cluster 挑一条 vs 一个 cluster 挑 16 条），coverage 与 performance 都略好但不显著。

**解释二：吸收率**。定义师生分布距离 distance 及其每步下降率（吸收率）后发现（字幕口径）：给 1 条、4 条、16 条还是上万条数据，每步吸收率都固定在约 2%；学习率调大只是步子迈得大，按学习率归一化后曲线重合。更极端的控制实验：off-policy OPD（固定用最初采出的一条 response 反复训，state 空间完全不变）也要一两百步才把收益拿回——state 都固定好了，吸收信息仍需几百步。讲者的总结（判词见文首金句）：数据一侧一条样本就撑满监督信号，算法一侧每步只吸收约 2%，**OPD 是数据撑死、算法饿死**；它比 RLVR 高效得多，但远称不上高效算法。

**极端实验：prompt 甚至可以没内容**（字幕口径）。既然关键是把学生带到某些 state 上受监督，讲者试了三个极端设定：base template（只有 BOS 与 think 标记、无题面）、system template（一句无信息量的话）、WildChat（与数学代码无关的真实闲聊 query）。结果是 base/system template 居然也能与一万多条真实数学 query 掰手腕；换算到同等 rollout token 口径，只用约 1/3 到 1/2 的 rollout token 就能达到 comparable 的效果（讲者同时坦白：这个 base template 是 cherry-pick 出来的，不同 template 效果差异很大）。与 one-shot RLVR 的对比（字幕口径）：RLVR 需要 outcome reward，问题必须处在学生「有时能解、有时不能解」的中间难度才有 advantage 信号；OPD 对此不敏感——学生全做对或全做不对的题照样能学，因为它不需要结果奖励，只在中间 state 上学老师分布。代价是对称的：OPD 学的是老师分布，天花板就是老师本身。

**第二篇的局限**（讲者自陈，字幕口径）：state coverage 只是一个 proxy 指标（K 的取值在附录里调过，未做太多消融）；MoPD 只做了 3 个 domain、序列不长，工业级 MoPD（十几个老师、12K 到一两百 K 上下文）没有 scale 验证；吸收率为什么这么低、能不能提上去，是明确留下的开放问题。

### Q&A 要点（字幕口径）

- **OPD 对数据多样性的真实需求**：coverage 的分母本身只是 full-data rollout 的估计，真实场景中学生采不到的状态永远存在；多样性的价值在于覆盖学生自己 on-policy 采不到、但老师擅长的状态。MoPD 中多 domain 数据对同一 state 的重复学习是否带来不均衡，尚无定论。
- **K=200 怎么选**：试过 240、500 等；K 越细单条覆盖率数字越低（90% 到 50% 都可能），但这是相对量、不影响结论。
- **算法瓶颈在哪**：讲者认为这不只是 OPD 的问题，而是 on-policy policy gradient 一类的共性——每步能容纳的 rollout 量有限，一个 prompt 学一轮学不干净。采样时其实已经拿到全词表分布，目前只对 sampled token 算梯度，「能不能直接对采样树做监督」是他提的方向之一。
- **pattern 不兼容时 OPD 是否有效**：基本无效——老师对学生前缀不熟、给的分布噪声大，长轨迹上容易不涨点甚至训崩。能力差距只是 benchmark 表象，关键看老师在学生前缀上打的 reward 准不准。工业界学生多不开 thinking 模式做 OPD，也是因为长轨迹上信号不准（字幕口径）。
- **OPD 学的到底是知识还是 reasoning manifold**：讲者明确倾向后者——蒸馏本质是模仿目标分布，OPD 学的是「老师怎么说话」；数学上学好能泛化到代码也支持这一点。学完与下游准确率的相关性则不保证。
- **OPD-then-RL**：讲者认可这是可行方向（已看到先 OPD 再 RL 的工作），OPD 负责快速拿分布收益，真正的探索与新状态发现还得靠带 ground-truth 环境奖励的 RL。
- **长程 MoPD 业界怎么做**：讲者举 Cursor Composer 2.5 的 OPS D 做法（字幕口径）：长 agent 轨迹采出来后不整条学，只学其中一轮（turn），逐轮累积；另一路是识别值得学的 token——老师 entropy 高的位置直接 mask 掉，不作局部 reward shaping。
- **top-k 是否归一化**：第一篇里做过对比，归一化后效果更稳；MoPD 场景下未深入。
- **资源充足时还要不要管 one-shot**：不需要——one-shot/few-shot 更多是学术问题；已有海量数据的团队直接堆数据即可。但「少量样本覆盖大量 state」这件事本身在各设定下都成立。讲者也提醒 few-shot 的「few」随规模可能要涨到几百量级（如 512-shot）才能拿回全量收益，这同样可能。
- **下一篇最值得解决的问题**：极致蒸馏——跨尺寸、跨分布仍是大坑；「什么样才算蒸馏得好、蒸馏的极限在哪」还没有好定义。
- **黑盒蒸馏怎么看**：拿不到 logits 时，目前性价比最高的仍是 SFT。OPSD（把 context/环境信息内化进参数）与 OPD 的两个主场景（极致蒸馏、多能力合并即 MoPD）不完全是一回事。
- **MoPD 为何缓解多任务数据冲突**（讲者提及小米 6–7 月的 MoPD vs Mix RL/Cascade RL 工作）：他的猜想是 dense signal 每步迈得平缓、更容易挪到 Pareto 前沿——明说只是猜想，未做实验验证。学生能否超过老师：单 domain 难，MoPD 多老师互促时单个 domain 有可能。
- **科研方法论**（被问「由点到面做文章最重要的特质」）：现象驱动——先确认现象是 general 的（不是某个模型/某条样本的偶然），再抽练出科学问题、针对问题做解释；他们实际的实验顺序与论文叙事顺序并不一致（第二篇实际先做了无数据的 template 实验，才扩展到 one-shot）。

## 关键数字

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 成功 OPD 的 overlap ratio | 失败 run 停滞 | 升至 90% 以上（接近 100%）；论文版 72%→91% 以上 | 字幕（机制部分）；论文 §4 |
| Overlap token 集合的概率质量 | — | 97% 以上（论文版 97–99%） | 字幕（机制部分）；论文 §4 |
| 弱老师蒸馏进度（JustRL-1.5B） | R1-Distill-7B 强老师学不动 | 200 step 拿回七八十 %，300 step 90% 以上 | 字幕（现象部分） |
| 反向蒸馏 gap recovery（JustRL-1.5B 学生） | — | >80% | 论文 §3（字幕未给此数值） |
| 老师 continuation 优势（prefix 1K → 16K） | +0.37 | +0.02（单调衰减） | 论文 §6（talk 仅定性：学生写得越多老师越无力回天） |
| OPD 轨迹长度口径 | 太短没信号 | 字幕口径 7K/10K 还不错；论文口径 3K/7K 最强、10K/15K 崩塌 | 字幕（代价部分）；论文 §6 |
| Top-k 的 k | sampled-token | 论文口径 k≥4 相当、Top-1 差；字幕口径 top-1/top-4 均见不稳、K 越大越稳、全词表更稳 | 论文 §6；字幕（代价部分、Q&A） |
| Reward 区分正确/错误 rollout 的 AUROC | 成功 vs 失败配置 | 都在 70 多（论文版 0.73 vs 0.75），区分不开 | 字幕（代价部分）；论文 §6 |
| One-shot OPD 收益 | full data 为 100% | 一条样本约 70% 多（数学可达 80% 多）；16-shot ≈ full data | 字幕（第二篇） |
| State coverage（K=200） | full data = 100% | 一条样本约 70%；16 条约 99% | 字幕（第二篇） |
| 算法每步吸收率 | 数据量 1 条到上万条 | 约 2%/step，与数据量无关 | 字幕（第二篇） |
| 无 prompt 模板的 rollout 成本 | 真实 query full data | 约 1/3 到 1/2 的 rollout token 达 comparable 效果 | 字幕（第二篇） |

评测设置（论文口径）：DAPO-Math-17K 训练；AIME 2024 / AIME 2025 / AMC 2023，avg@16（每题 16 采样，temperature 0.7）。

## 可迁移

- 对 coding data / RL infra 工作的直接可试的点：
  1. **选蒸馏老师别只看 benchmark 分**：优先"同底座 + 额外 RL 后训"带来的新能力（Skywork-OR1-Math-7B 型），而不是同管线放大（R1-Distill-7B 型）。thinking-pattern 不一致是前置杀手，且早期损失不可逆——动手前先测师生初始 overlap ratio。
  2. **数据量不是 OPD 的瓶颈，吸收率才是**：扩数据前先想清楚每步监督的 state 覆盖与梯度利用率；少量高覆盖样本 + 更多训练步数，可能比堆全量数据更划算。监控侧用 overlap ratio / entropy gap 曲线作成功签名与早停信号，失败 run 早期就停滞。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. 长程 agentic OPD 的天花板是结构性的：dense token reward 随深度退化。讲者在 Q&A 给的工业界现行解法值得抄：只学长轨迹中的单轮（turn-level）、把老师高 entropy 位置 mask 掉；论文建议 hybrid（短 segment 用 dense token 监督 + 长程用稀疏 outcome reward）。
  2. 轨迹长度是 sweet spot 超参且 domain 相关（讲者 Q&A 口径：数学上 10000 token 左右差不多），不要无脑拉长；缺的正是「token 级信号质量」的在线评价指标。
  3. 本 repo 的 opd-reading-list 有 10 篇 OPD 论文（含 2603.25562，被本论文引用），可对照看"三类典型失败"（125 期）和本篇的"两个条件"是否同构。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：§6.2 的 anisotropy 假设（per-token advantage 方向不一致→梯度抵消）论文明确没验证；如果成立，什么样的 objective 能利用各向异性的 reward 结构？这是 open question。
- 第二篇留下的开放问题（讲者自陈，字幕口径）：吸收率为什么只有约 2%、能不能提上去（如对采样树而非采样链做监督）；state coverage 目前只能后验测量，能否先验地（不 rollout、只看 prompt embedding）判断一条样本值不值得训。
- 续作关系：第二篇《One Training Example》(arXiv:2609.04172) 的结论「OPD 是 data-overfed but algorithm-starved」与第一篇的 recipe（补数据/对齐 prompt）放在一起看有张力——第一篇说对齐数据有用，第二篇说数据早就过剩。这和讲者 Q&A 的说法可以对齐：对齐提升的是 state 覆盖的质量（覆盖到老师擅长的 state），而非数量。值得深挖。

## 原文金句（1-2句）

> 「OPD 它是一个数据撑死、但是算法饿死的东西：一条样本就已经提供了 70% 的监督信号，16 条样本就已经打满了，但算法每个 step 的吸收率仍然非常有限。」——讲者对第二篇工作的总结（字幕 40:31–40:48，按干净口径转写）

> "a globally informative reward does not guarantee a locally exploitable one." ——论文 §6 的核心判词：全局信息性的 reward，不保证局部可利用。这是选老师、设 k、定轨迹长度所有工程决策背后的第一性原理。
