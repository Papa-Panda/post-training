# 07 — Reading a Trace: 拿到一条 trace，先问什么

> 前面六章给了三套工具。本章把它们合成一个读 trace 的 checklist：拿到一条 trace，按什么顺序看、每个信号回答什么问题、看完之后能做什么动作。
>
> The previous six chapters gave three toolkits. This chapter synthesizes them into a checklist for reading a trace: in what order to look, what question each signal answers, and what actions follow.

## 1. 三遍读法

**第一遍：结构（01、02）——"里面装了什么？"**

- 行为视角：标出 planning / compute / verify / backtrack 段。问：verify 段是真验算还是复述？backtrack 是换路还是原地打转？
- 机制视角：如果有白盒访问，算 DTR。问：高 DTR 段和答案正确性对得上吗？长 trace 的长度是 deep-thinking token 撑起来的，还是 filler 撑起来的？

**第二遍：效率（03、04、05）——"哪些部分是浪费？"**

- 熵视角：这题落在哪一档难度？Easy 题的高熵是发散还是困惑？
- 行为视角：数 hedging / recheck / re-derive / tangent。算 HVR：光怀疑不验证的段落占比多少？有没有 abandonment（S3，最强失败信号）？
- 结构视角：如果这条 trace 是 batch 里的一条，对比单问时的版本——被压掉的是空转还是真推理？

**第三遍：监督（06）——"如果给每步打分，哪里该扣分？"**

- 标出 false positive 风险点：答案对，但哪一步的推导站不住？
- 想清楚 step 切分用的是哪种 contract（换行 / 语义 / checkpoint），因为打分只在该 contract 下有意义。
- 如果这条 trace 要进训练：标量 advantage 会误伤哪一段？要不要 segment 级 credit？

## 2. 信号速查表

| 信号 | 章节 | 回答的问题 | 需要什么访问 |
|---|---|---|---|
| 行为标签（verify/backtrack…） | 01 | trace 在干什么 | 文本即可 |
| DTR | 02 | 多少 token 经历了深层修正 | 白盒 hidden state |
| Token 熵 $\bar{H}$ | 03 | 当前难度档位 | 白盒 logits |
| HVR | 04 | 光怀疑不验证的比例 | 文本即可 |
| 六信号（S1–S6） | 04 | 哪种空转模式 | 文本即可 |
| 四模式频率 | 05 | batch 压掉了什么 | 文本即可 |
| PRM step 分数 | 06 | 每步对错 | PRM 模型 |

## 3. 收束：trace 的双重身份

读完一条 trace，你同时在回答两个层面的问题——这正是 00 章开篇的框架，也是本专题的收束：

**Trace 作为行为 (behavior)。** 它是策略 $\pi_{\theta}$ 在 reward 下优化出的采样轨迹。R1-Zero 只按最终正确性给 reward，trace 长成什么样——多长、多少 hedging、中英混杂——都是优化过程的副产品。从这个视角读 trace，问题是工程问题：预算怎么分（03 的路由）、空转怎么压（04、05 的检测与干预）、credit 怎么分（06 的 segment advantage）。目标是让下一条 trace 更便宜、更有效。

**Trace 作为证据 (evidence)。** 它是人类审计、debug、给过程打分的依据。从这个视角读 trace，问题是认识论问题：这段 verify 是真验算吗（01 的行为 vs 机制之分）？答案对但步骤错的 false positive 藏在哪里（06 的 PRM）？HVR=0 的 trace 能直接信任吗（04 的 gate 只是筛选器）？目标是让 trace 成为可信的记录。

两者的 tension 是结构性的：**优化行为的 reward（只看答案）不奖励"写出可信证据"**。R1-Zero 的中英混杂 trace 就是证明——对 reward 有效，对人类读者不友好。所有"把行为变成可用证据"的工作——SFT 对齐格式、PRM 打分、行为分类、DTR 筛选——都是在替 reward 没做的事补课。

**Faithfulness（暂缓）就是这个 tension 的正面战场**：trace 作为证据的可靠性，到底能在多大程度上被信任、被测量、被改进。入口见 README 的 deferred 部分。

## 4. 从读到改：三类动作

| 读完发现 | 动作 | 章节 |
|---|---|---|
| 空转占比高（高 HVR、多 abandonment） | 干预：Peng 两步法 prompt，或 batch 打包 | 04、05 |
| 预算错配（easy 题花满预算、hard 题空转） | 路由：DiffAdapt 三档 | 03 |
| 步骤质量参差（false positive 风险） | 监督：PRM 打分 / segment credit | 06 |
| 不知道长短由什么驱动 | 诊断：先算 DTR，再看行为标签 | 01、02 |

## 5. 代码对应

`code/trace_lab.py` 的 `run_all()` 把四个玩具实验串起来：DTR 相关性、路由节省、hedging 压缩、PRM vs ORM。把它当成"读 trace 之前先校准直觉"的沙盒：每个数字都是可复现的（固定 seed），每个断言都在 `tests/test_trace_lab.py` 里。
