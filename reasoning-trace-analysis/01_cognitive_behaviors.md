# 01 — Cognitive Behaviors: R1 论文里到底写了什么

> "结构解剖"的起点。DeepSeek-R1 论文确实描述了 trace 里的"认知行为"，但它的用词比二手转述更克制。本章只引用论文原文，不发明分类名。
>
> The starting point of structural dissection. The DeepSeek-R1 paper does describe "cognitive behaviors" in traces, but its wording is more careful than second-hand retellings. This chapter quotes only the paper's own words.

## 1. 论文原文：三处关键表述

**（1）Abstract 的行为清单。** 论文摘要里对 RL 涌现出的推理模式的完整原话是：

> "The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as **self-reflection, verification, and dynamic strategy adaptation**."

注意三点：① 主语是 "advanced reasoning patterns"，不是 "cognitive behaviors"（"认知行为"是后人转述时的叫法）；② 三个例子是 self-reflection（自我反思）、verification（验证）、dynamic strategy adaptation（动态策略调整）；③ 用的是 "such as"，是举例不是穷举分类。

**（2）§2 开头的行为描述。** 在介绍 R1-Zero 时，论文写道，模型"naturally developed diverse and sophisticated reasoning behaviors"，解题时表现出：

> "incorporating **verification, reflection, and the exploration of alternative approaches** within each response"

这里的三个成分是：验证、反思、探索替代解法。措辞和摘要不完全一样——论文自己也没有固定成术语表。

**（3）"backtrack" 一词的真实出处。** "backtracking" 确实出现在论文里，但不是在摘要的行为清单中，而是在讨论涌现行为的段落里：

> "the model learned to dynamically scale computation by generating more thinking tokens to **verify or correct its reasoning steps, or to backtrack and explore alternative approaches** when initial attempts proved unsuccessful."

以及在对比非推理模型时：

> "rarely demonstrate advanced problem-solving techniques like **self-reflection, backtracking, or exploring alternative approaches**."

结论：如果你要引用 R1 论文，准确的说法是——论文观察到涌现的 reasoning patterns 包括 self-reflection、verification、dynamic strategy adaptation，并在正文中用 backtrack / explore alternative approaches 描述回溯行为。不存在一个官方的"认知行为分类表"。

## 2. "aha moment"：论文里最出圈的一句话

Table 2 的标题原话：

> 'An interesting **"aha moment"** of an intermediate version of DeepSeek-R1-Zero. The model learns to **rethink using an anthropomorphic tone**.'

配的例子是模型在解方程时突然停下："Wait, wait. Wait. That's an aha moment I can flag here." 然后 "Let's reevaluate this step-by-step…"。

"aha moment" 在论文里是一个**轶事级观察**（anecdote），不是度量。它之所以重要，是因为它标志着：模型在没有任何人类示范的情况下，自己学会了"停下来、承认可能错了、换路重来"——这正是纯 RL（无 SFT 冷启动）最让人意外的涌现。

## 3. 行为视角 vs 机制视角

R1 论文给的是**行为视角**：读 trace 文本，给看到的现象起名字。这是解剖的第一步，但它有两个局限：

1. **没有操作化定义**：什么是"一次 verification"？从哪个 token 到哪个 token 算？论文没定义，两个标注者可能切出不同的段。
2. **停留在文本表面**："模型写了 'let me verify'" 和"模型真的在验证"是两回事（这正是 faithfulness 线要处理的问题，本专题暂缓）。

**机制视角**（下一章 02）走另一条路：不看文本写了什么，看生成每个 token 时模型内部发生了什么。Deep-thinking token 的定义完全不依赖文本内容——它只看深层 layer 的预测分布收敛得有多慢。两种视角对照：

| | 行为视角（本章） | 机制视角（02） |
|---|---|---|
| 输入 | trace 文本 | 每层的 hidden state 投影到词表 |
| 单位 | 语义段（planning / verify / backtrack） | 单个 token 的 settling depth |
| 代表 | R1 论文的现象描述 | Think Deep 的 DTR |
| 优点 | 人类可读、可审计 | 可计算、不依赖文本措辞 |
| 缺点 | 定义模糊、易被措辞欺骗 | 需要白盒访问 hidden state |

两条路不是竞争关系：PRM（06）要给 step 打分，先得知道 step 的边界和类型——行为视角提供语义，机制视角提供可计算的代理信号。

## 4. 一个常被忽略的细节：trace 的"行为"是训练方式的函数

R1-Zero 用 GRPO 纯 RL 训练，reward 只看最终答案正确性（§2.2），不约束推理过程。论文明确承认的代价：

> "poor readability and language mixing, occasionally **combining English and Chinese within a single chain-of-thought response**"

这就是 00 章说的 tension 的实证：把 trace 当**行为**优化（只奖答案），trace 就会长成"对 reward 有效但对人类读者不友好"的样子——中英混杂、可读性差。人类想把 trace 当**证据**读（审计、debug），就必须额外做功：SFT 对齐格式、PRM 打分、行为分类。整个专题后面三章，都是在给"把行为变成可用证据"这件事提供工具。

## 5. 代码对应

`code/trace_lab.py` 的 (a) 部分是机制视角的玩具实现：`settling_depth` 计算每个 token 的收敛层数，`deep_thinking_ratio` 统计深层收敛 token 的比例。行为视角的文本分类器没有玩具实现——因为那需要真实标注数据；但 (c) 部分的 hedging 检测器展示了"从文本表面提取行为信号"的正则表达式做法，正好可以体会行为视角的脆弱性（换个措辞就检测不到）。
