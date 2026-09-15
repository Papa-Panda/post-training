# 00 — Trace as an Object: 把 trace 写成数学对象

> 先把"一条推理轨迹"形式化，再谈解剖、效率和监督。没有这一步，后面所有的"结构""熵""打分"都无从谈起。
>
> First formalize "a reasoning trace", then dissect it. Without this step, every later claim about structure, entropy, or scoring has no object to attach to.

## 1. 三个基本对象

一次问答产生三个东西：

- prompt $x$ ：用户问题（含系统指令、上下文）。
- trace $z=(z_{1},\ldots,z_{T})$ ：模型在给出答案之前生成的 $T$ 个 token。 $z$ 是我们要分析的**主体**。
- 答案 $y$ ：最终输出（可能是答案文本，也可能是从 trace 里解析出来的）。

联合分布记为

$$p_{\theta}(z,y\mid x)=p_{\theta}(z\mid x)\,p_{\theta}(y\mid x,z).$$

注意 $y$ 在给定 $z$ 下常常是近似确定性的（比如从 `<answer>...</answer>` 里抽出来），所以分析重心在 $z$ 上。

同一个对象的两种读法：

| 读法 | trace 是什么 | 关心的问题 |
|---|---|---|
| **行为 (behavior)** | 策略 $\pi_{\theta}$ 的采样轨迹，RL 优化的副产品 | 它如何影响最终 reward？怎么让它更短、更便宜？ |
| **证据 (evidence)** | 模型"如何想"的记录，人类拿来审计和 debug | 它说了真话吗？哪一步错了？ |

RL 训练把 trace 当**行为**：DeepSeek-R1-Zero 的 reward "solely based on the correctness of final predictions against ground-truth answers, without imposing constraints on the reasoning process itself"（R1 论文 §2.2）。trace 长成什么样——包括中英混杂、可读性差——都是"只奖答案"的副产品。而人类读者把 trace 当**证据**：审计、debug、教小模型。两者的 gap，就是整个 faithfulness 议题的根源（本专题暂缓深挖，见 README 的 deferred 部分）。

## 2. Step 切分：trace 不是 token 的均匀流

几乎所有分析都要先把 $z$ 切成 step（一"步"）：

$$z = s_{1}\oplus s_{2}\oplus\cdots\oplus s_{K},\qquad s_{k}\ \text{为第 }k\text{ 步的 token 段}.$$

切分本身就是个建模选择，常见做法：

1. **换行切分**：OpenAI "Let's Verify Step by Step" 让 generator 输出 "newline delimited step-by-step format"，每行即一步。简单、可复现，但语义 step 可能跨行。
2. **语义切分**：按"一次计算 / 一次验证 / 一次回溯"切。需要分类器或 LLM-as-judge，是线 1（结构解剖）的研究对象。
3. **答案检查点切分**："Know When to Stop" 用 trace 里的中间答案承诺（如 `\boxed{...}` 、 `the answer is X` ）做边界，把 trace 切成 segment，再做 segment 级 credit assignment。

没有标准答案：切分粒度决定了你能看到什么。PRM 要求 step 级标注（线 3），熵分析工作在 token 级（线 2），行为分类工作在语义段级（线 1）。

## 3. 四个可计算的投影

同一个 $z$ ，可以投影到四个不同的测量空间——正好对应本专题的三条线加一个暂缓项：

| 投影 | 映射 | 回答的问题 | 章节 |
|---|---|---|---|
| 结构投影 | $z \mapsto$ 行为标签序列（planning / compute / verify / backtrack …） | 里面装了什么 | 01、02 |
| 信息投影 | $z \mapsto H_{t}=-\sum_{j}p_{t,j}\log p_{t,j}$ | 哪些位置是真不确定 | 03 |
| 行为投影 | $z \mapsto$ 四种 overthinking 模式计数 | 哪些部分是元认知空转 | 04、05 |
| 监督投影 | $z \mapsto (r_{1},\ldots,r_{K})$ ， $r_{k}\in\{+,-\}$ | 每一步该打几分 | 06 |

结构投影回答"里面装了什么"，信息投影和行为投影回答"哪些部分是浪费"，监督投影回答"怎么给中间过程打分"。

## 4. 本专题的符号约定

| 符号 | 含义 |
|---|---|
| $x$ | prompt |
| $z=(z_{1},\ldots,z_{T})$ | 推理 trace， $T$ 个 token |
| $y$ | 最终答案 |
| $s_{k}$ | 第 $k$ 个 step / segment |
| $H_{t}$ | 位置 $t$ 的 token 分布熵 |
| $c_{t}$ | token $t$ 的 settling depth（02） |
| $r_{k}$ | step $k$ 的过程监督标签（06） |

## 5. 代码对应

`code/trace_lab.py` 的四个仿真正好实现这四个投影的玩具版本：(a) settling depth → 结构投影的机制版；(b) 熵 + 路由 → 信息投影；(c) hedging 检测器 → 行为投影；(d) PRM vs ORM → 监督投影。每个投影的语义不变量都在 `tests/test_trace_lab.py` 里断言。
