# Reasoning Trace Analysis

> 一条 reasoning trace 有两个身份：它是 RL 优化出的**行为**（只按最终答案给 reward 的副产品），也是人类拿来审计和 debug 的**证据**。本专题研究前者如何被解剖、被压缩、被打分；后者的可靠性问题（faithfulness）暂缓，只留入口。
>
> A reasoning trace has two identities: it is **behavior** optimized by RL (a byproduct of rewarding only final answers), and **evidence** humans use to audit and debug. This topic dissects, compresses, and scores the former; the reliability of the latter (faithfulness) is deferred, with entry points only.

## 核心框架：trace 的双重身份

- **Trace as behavior**：策略 $\pi_{\theta}$ 的采样轨迹。R1-Zero 的 reward 只看最终预测正确性、不约束推理过程本身——trace 的长度、hedging、中英混杂都是优化的副产品。从这个视角看 trace，问题是工程问题：预算怎么分、空转怎么压、credit 怎么分。
- **Trace as evidence**：模型"如何想"的记录。人类拿它审计、debug、教小模型、给过程打分。从这个视角看 trace，问题是认识论问题：这段 verify 是真验算吗？答案对但步骤错的解藏在哪里？

两者的 tension 是结构性的：优化行为的 reward 不奖励"写出可信证据"。本专题的三条线全部站在"行为"一侧，把 trace 当成可测量、可压缩、可打分的对象来处理。

## What this topic answers

1. **结构**：trace 里面装了什么？R1 论文到底观察到了哪些行为，机制视角（deep-thinking token）又量到了什么？
2. **效率**：哪些 token 是浪费？熵的 U 型、self-doubt、hedging 如何被检测，又如何被 batch 结构压缩？
3. **监督**：怎么给中间步骤打分？PRM 相对 ORM 解决了什么，credit 又该如何分到每个 token？

## 阅读路线：数学 → 系统 → 可运行代码

| 步骤 | 章节 | 问题 |
|---|---|---|
| 0 | [Trace as an Object](00_trace_as_object.md) | trace 的数学对象是什么？step 如何切分？ |
| 1 | [Cognitive Behaviors](01_cognitive_behaviors.md) | R1 论文原文到底写了什么行为？（只引原文，不发明分类） |
| 2 | [Deep-Thinking Tokens](02_deep_thinking_tokens.md) | 不数长度，数"想了几层"：settling depth 与 DTR |
| 3 | [Entropy and DiffAdapt](03_entropy_and_diffadapt.md) | 熵为什么是 U 型的？三档路由如何省 token？ |
| 4 | [Self-Doubt and Hedging](04_self_doubt_and_hedging.md) | 已经做对了为何还在检查？HVR 与六信号 |
| 5 | [Batch Prompting](05_batch_prompting.md) | 为什么打包问题比喊"想短点"更有效？ |
| 6 | [Process Supervision](06_prm.md) | PRM 如何给每步打分？credit 该分到多细？ |
| 7 | [Reading a Trace](07_reading_a_trace.md) | 拿到一条 trace 的三遍读法与双重身份收束 |
| — | [Sources](sources.md) | 每个数字与结论的一手来源 |

## 范围边界

**本专题展开的三条线**：trace 结构解剖（00–02）、长度/效率与 overthinking（03–05）、过程监督 PRM（06–07）。

**暂缓：faithfulness**。Trace 作为证据的可信度（模型写下的推理是否真是它推理的过程）是独立的大议题，本专题不展开，只留三个后续入口，一行一个：

- https://arxiv.org/abs/2505.05410
- https://arxiv.org/abs/2510.04040v1
- https://arxiv.org/abs/2509.13334v1

## 与相邻专题的边界

| 相邻专题 | 它的职责 | 本专题只覆盖 |
|---|---|---|
| [`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) | 目标函数推导与 PPO/GRPO 对比 | GRPO 的标量 advantage 在 trace 上的 credit 缺陷（06 §6） |
| [`post-training-framework/`](../post-training-framework/README.md) | rollout/learner 数据流与框架 contract | trace 作为被分析、被打分的对象，不碰框架本身 |
| [`ai-data/`](../ai-data/README.md) | 数据选择与价值度量 | PRM800K 作为过程监督数据的实例（06 §3） |

## 可运行的 CPU 模型

`code/trace_lab.py` 是纯 NumPy（无 torch）语义仿真器，四组玩具实验对应三条线：

- (a) deep-thinking ratio：settling depth → DTR，复现"DTR 预测准确率优于长度"；
- (b) U 型熵 + 三档路由：uniform 基线 vs DiffAdapt 式路由的 token 节省；
- (c) hedging/self-doubt 检测：正则扫描五类 marker，HVR，长 trace vs batched 压缩对比；
- (d) PRM vs ORM：false positive 玩具数据上的 best-of-N 对比，GRPO 标量 vs segment 级 credit。

```bash
cd reasoning-trace-analysis
python3 -m pytest tests/ -q
python3 code/trace_lab.py
```

它是**语义仿真器，不是性能预测器**：隔离的是"在真实实现中也该成立"的正确性关系（DTR 单调性、HVR 公式、PRM 打分单调性、路由省 token），不是 GPU 上的速度。

## 不可协商的不变量

- 引用的论文没有官方"认知行为分类表"：R1 的行为描述是现象举例，不是分类学。
- Figure/Table 里的具体数字（如 $r=0.828$ 、 $96.1\%$ ）是特定实验设定的产物；可迁移的是序关系和机制，不是常数。
- 候选解释必须标注为假设：batch prompting 的三个机制解释在论文中是 conjectures，不是定论。
- PRM 的 step 边界是数据与系统的 contract（换行格式），不是语义真理；换切分，打分语义就变。
- HVR=0 的高正确率是筛选器，不是安全保证。
- 找不到一手来源的数字写"未核验"，不编造。

## 来源政策

技术主张只由实际打开过的一手论文页面支撑，见 [sources.md](sources.md) 。每个条目注明核验时亲眼看到的事实；faithfulness 三条只确认存在，未核验内容，不引用其结论。
