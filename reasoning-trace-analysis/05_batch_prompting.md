# 05 — Batch Prompting: 把问题打包，模型自己就想短了

> 把多个问题打包进一次请求（batch prompting），推理 token 从平均 2,950 降到 710（降 $76\%$ ），准确率持平或提升。"Reasoning Under Constraint"（Saurabh Srivastava、Janit Bidhan 等，2511.04108）在 DeepSeek-R1 与 OpenAI-o1、13 个 benchmark 上报告了这个效应。
>
> Packing multiple questions into one request (batch prompting) cuts mean reasoning tokens from 2,950 to 710 (a $76\%$ drop) while accuracy holds or improves. "Reasoning Under Constraint" (Saurabh Srivastava, Janit Bidhan, et al., 2511.04108) reports this on DeepSeek-R1 and OpenAI-o1 across 13 benchmarks.

## 1. 现象：batch 越大，想得越短

主对比是 batch size 1 → 15（论文 Figure 1）：

- 平均 reasoning tokens：2,950 → 710，下降 $76\%$ 。
- 准确率：保持或提升，没有为简短付出代价。
- 输出 token 同样下降 83–88% （注意这是输出侧，不是推理侧）。

这不是"模型被要求简短"——batch prompting 没有加任何长度指令。变短是**涌现的**。

## 2. 四种 overthinking 模式都被压缩

论文 Figure 4 比较 batch size 1 → 5 时四种模式的频率（频率按 reasoning trace length 归一化）：

| 模式 | BS=1 | BS=5 | 降幅 |
|---|---|---|---|
| hedging（"wait"、"hold on"、"let me double-check"…） | 0.42 | 0.18 | $57\%$ |
| rechecking | 0.55 | 0.22 | $60\%$ |
| re-derivation | 0.38 | 0.16 | $58\%$ |
| irrelevant tangents | 0.31 | 0.12 | $61\%$ |
| 平均降幅 | | | $59\%$ |

四种模式无一例外被压缩，平均降 $59\%$ 。这说明 batch 压掉的不是某一种措辞习惯，而是整类元认知空转。

## 3. 反直觉的对照：直接喊"想短点"没用

论文做了一个关键对照：显式约束如 "Use no more than 100 tokens in thinking"，模型**要么忽略，要么牺牲准确率**。指令式压缩失败，结构式压缩（batch）成功。

这个对照的工程含义：overthinking 不是模型"不听话"，而是单问题、长上下文的默认工作模式给了它空转的空间。压缩要改变任务结构，而不是加指令。

## 4. 候选解释（注意：是假设，不是定论）

论文提出了三个候选解释，并**明确标注为 candidate hypotheses / conjectures**，不是已证明的机制：

1. **Shared-context pressure**：多个问题共享上下文，模型被迫分配注意力，没空在单个问题上打转。
2. **Sequential anchoring / in-context pattern induction**：前面的短答为后面的生成定了"简洁"的上下文锚。
3. **Implicit difficulty calibration**：看到一堆问题，模型隐式校准了难度预期，不再对每题都"如临大敌"。

引用时必须保留这个限定：论文没有证明是哪一个（或哪几个）在起作用。

## 5. 代价侧：self-doubt loop 会耗尽预算

论文还记录了反面：self-doubt loop 可能耗尽 token budget，导致 API timeout 和 accuracy 下降。这是对 04 章的呼应——空转不只是浪费钱，在有硬预算的 serving 场景下会直接变成失败。

## 6. 与 04 的关系：两种干预路线

| | Peng 的两步法（04） | Batch prompting（本章） |
|---|---|---|
| 干预点 | prompt 指令：先验输入，够信息就短答 | 任务结构：打包多个问题 |
| 假设 | 模型听指令 | 模型受上下文结构影响 |
| 优点 | 单问题可用 | 不用改 prompt，大幅压缩 |
| 缺点 | 指令可能被忽略（本章 §3 已证） | 需要攒 batch，有延迟代价 |

## 7. 代码对应

`code/trace_lab.py` 的 (c) 部分用 `LONG_TRACE` vs `BATCHED_TRACE` 复现了 Figure 4 的定性结论：同一道题，长 trace 里四种模式频率显著高于 batched 短 trace；`pattern_frequencies` 的"按每 100 token 归一化"与论文的归一化口径一致。`tests/test_trace_lab.py` 断言 batched 版每种模式都不高于 long 版，且至少一种严格更低。
