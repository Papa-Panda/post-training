## 元信息
- Title: Textbooks Are All You Need (Phi-1)
- Authors / Org: Suriya Gunasekar et al. / Microsoft Research
- Link / arXiv: https://arxiv.org/abs/2306.11644
- Code/Data: https://huggingface.co/microsoft/phi-1 , synthetic CodeTextbook via GPT-3.5
- Date read: 2026-08-09
- Tags: [coding-data, synthetic-data, quality, curation, pretraining, sft, SOTA]

## 一句话总结
不用堆 100B web code，用 GPT-3.5 合成“教科书质量”的 6B 精筛 web + 1B 合成练习，1.3B 模型训 4 天就 50.6% HumanEval，证明高质量合成数据 >> 大量低质数据。

## 核心
1.  **Motivation**: scaling law 让人以为堆数据就好，但 web code 又臭又长又重复。能不能像教课本一样，把知识蒸馏成干净、渐进、带解释的教材，让小模型也学会？
2.  **Data Pipeline**: 
    - 从 The Stack 挑 textbook-like：用 classifier 选出“解释性强、教育性” 的 Python 文件，6B tokens
    - 合成 CodeTextbook：用 GPT-3.5 按主题生成章节式解释 + 代码示例，1B tokens
    - 合成 CodeExercises：再让 GPT-3.5 生成小练习题 + 解法，微调用，10k级
    - 训练：1.3B Transformer，4天 8 A100，先预训 6B+1B，再在练习上微调
3.  **Key Tricks**: 
    - 不是随机合成，是按“教科书章节”结构化合成：概念 → 例子 → 练习，利于模型建立因果链
    - 小模型也能涌现：350M 同流程仍 45% HumanEval，说明数据带的有推理模式
    - 过滤狠：原来 Stack 100B 级，只留 6B 精的，massive deduplication + decontamination
4.  **Results**: Phi-1 1.3B HumanEval pass@1 50.6% / MBPP 55.5%，超过很多 7B-15B 直接在 web code 上训的；Phi-1-small 350M 仍 45%，证明 pipeline 可迁移到更小 budget。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 直接抄 Phi-1 的 CodeExercises 合成模板，让你的合成器把“题目+逐步解+边界测试”一起出，做 RL 的 rollout 验货数据
  2. 把你的 50万 合成池用 textbook-quality classifier 重筛一遍，留 5% 精的试训，对比现在全量训的效果
- Infra 视角：合成数据 flewheel 成本主要在 GPT-3.5/4 调用，得搭 cache + de-dup + contaminated check，否则 HumanEval 泄漏假高。

## 疑问 / 下一步
- 合成教科书的 diversity 如何保证不坍缩成几种模板，coding 上不同算法范式（DP/图/贪心）是否都被覆盖？
- Phi-1 说 web 数据是 noise，但如果我的下游是长尾库函数调用，纯教科书会不会掉 recall？

## 原文金句 (1-2句)
>  We introduce phi-1, with 1.3B parameters, trained on textbook quality data, attaining 50.6% HumanEval — despite this small scale.

> Textbooks Are All You Need — high-quality curated data can be just as useful as enormous unfocused piles.

## 第二轮复习（2026-09-06）

### 1. 核心命题

Phi-1 真正解决的 data 问题，不是"数据不够多"，而是**训练数据的知识组织形式错了**。Web code（The Stack 级别）的问题不只是噪声和重复，更深的是它**没有教学结构**：概念、例子、练习被打散在成千上万个文件里，一个 1.3B 的小模型没有容量去从 100B 噪声里自己归纳出"推理模式"。

Phi-1 的核心命题是：**把数据做成教科书——用"密度 × 组织结构"替代"堆 token 量"**。6B 精筛 web（classifier 选 textbook-like）提供真实 grounding（真实 API、长尾库调用），1B GPT-3.5 合成的 CodeTextbook 提供"概念 → 例子 → 练习"的 pedagogical 结构，CodeExercises 做格式对齐。结果是 1.3B、4 天、8 张 A100 就 50.6% HumanEval，超过当时很多 7B–15B 直接在 web code 上训的模型。

用 scaling law 的语言说：数据质量移动的是 $$L = A\cdot N^{-\alpha} + B\cdot D^{-\beta} + E$$ 里的**数据项系数/指数**——高质量结构化数据等价于有效数据量 D 被放大。Phi-1 是这个命题在 2023 年最干净的一次实证：它不是"规模论文"，是"配方论文"。

### 2. 图谱位置

- **前驱（合成线）**：Day21 Self-Instruct（合成范式起点，2022）→ Day22 Evol-Instruct（复杂度演化）→ Day06 Phi-1。Phi-1 是 synthetic 线上第一个"**合成不为量、为结构**"的证明：Self-Instruct 解决"有没有指令"，Evol-Instruct 解决"难不难"，Phi-1 解决"**好不好学**"。
- **直接对比 Day27 OSS-Instruct（重点）**：同期 2023、同样做 code 合成、回答的是同一个问题（web code 太烂），但赌的是数据价值的**相反维度**。
  - Phi-1：价值在 **density / pedagogy**——用 GPT-3.5 按章节生成教科书，狠删噪声，6B+1B 极度提纯。代价是**现实性**：合成分布坍缩到教师模型的先验，长尾库、真实 repo 上下文覆盖不足。
  - OSS-Instruct：价值在 **coverage / realism**——用 80K 真实开源代码片段锚定生成，甚至**有意保留不完整 solution** 保现实性。代价是干净度。
  - 一句话分工：Phi-1 回答"怎样让小模型**学会**"，OSS-Instruct 回答"怎样让合成数据**像真的**"。两者正交，可串联：用 OSS-Instruct 的真实片段做 seed，再用 Phi-1 的教科书模板做结构化改写——这正是初读"可迁移"第 2 条的升级版。
- **后继（预训练瀑布）**：Day07 Llama3 是 Phi-1 假设的**工业化版本**——15.6T 上的 5 级过滤、fastText→RoBERTa→Llama-2-70B 三级质量打分器（"educational value"评分正是 Phi-1 classifier 思想的放大）、code 17%→25% 上采样、多轮合成/回译 annealing。图谱 mermaid 里 `I[Day06 Phi-1 教科书] --> J[Day07 Llama3 15.6T]` 这条边，实质是"配方"→"工厂"。
- **互补 Day16 Qwen2.5-Coder**：补上 Phi-1 最大的缺口——**正确性验证**。Phi-1 的合成代码从未经执行验证（见第 4 节第 1 条）；Qwen2.5-Coder 的 parser+exec+LLM 三级过滤正是"教科书"缺的那道"答案对不对"的门。Day27（题从哪来）+ Day16（答案对不对）+ Day06（好不好学）三者合起来才是 coding 合成数据的完整闭环。
- **平行 Day24 SemDeDup / Day25 FineWeb**：Phi-1 做了 massive dedup + decontamination，但用的是朴素 n-gram；Day24 给出语义版升级，Day25 把"classifier 选高质量"做成可复现的开源工厂。Phi-1 的过滤 classifier 是 FineWeb 式管线的精神前驱。

### 3. 机制深挖

**(a) "教科书结构"到底在机制上干了什么。** 概念→worked example→练习的三件套，把**推理模式**和它的**验证闭环**压缩进同一个短上下文。对小模型这意味着：梯度信号里"怎么做对"和"为什么对"是相邻 co-occur 的，模型不需要跨 100B token 去自己发现因果链。换句话说，数据替模型做了一部分**归纳偏置**的工作。Phi-1-small（350M）仍有 45% HumanEval 是最有力的证据：350M 根本记不住多少事实，它学会的是**模式**——而模式只能从结构化数据里高效提取。

**(b) 双 classifier 架构：判别选 vs 生成造。** 6B web 精筛用的是**判别**（classifier 给 The Stack 文件打"教育价值"分，分布选择）；1B CodeTextbook 用的是**生成**（GPT-3.5 按主题章节创造，分布创造）。两者分工不同：判别选保留真实世界 grounding（长尾 API、真实库调用习惯），生成造提供 web 上不存在的 pedagogical 结构。纯合成会坍缩成教师先验（第 4 节第 3 条），纯筛选则受限于 web 已有分布——6B:1B 这个配比是"**现实锚定 + 结构注入**"的折中。

**(c) CodeExercises 是格式对齐，不是知识。** 预训练 6B+1B 学的是"知识和模式"，微调的 CodeExercises（10k 级小练习）学的是"**HumanEval 要的答题格式**"。这是后来 Day15 R1 cold-start 的早期影子：**格式对齐数据**和**能力数据**是两种不同的数据，Phi-1 无意中把两者分开了。读 Phi-1 的 50.6% 时要记住：里面有一部分是"格式分"，不是纯能力分。

**(d) 去重 + decontamination 是分数可信的前提。** 100B 级 Stack 只留 6B，massive dedup 保证模型不是在背诵；benchmark decontamination 保证 HumanEval 分数不是泄漏。但注意第 4 节第 2 条：这两道门都**防不住教师泄漏**。

**(e) 为什么 4 天 8 A100 是 feature 不是 bug。** Compute 极小意味着**复现门槛极低**——任何人都能验证这个配方。Phi-1 的影响力恰恰来自"小预算可复现"：它把"数据质量 > 数据规模"从口号变成了人人可跑的实验。这也是它进 S-tier 的原因：范式定义 + 可复现。

### 4. 边界与反例

1. **合成代码未经执行验证**：GPT-3.5 生成的 CodeTextbook/CodeExercises 里必然有幻觉的错误解法，这些错误成为 ground truth 被蒸馏进模型。初读"可迁移"第 1 条说"把题目+逐步解+边界测试一起出，做 RL 的 rollout 验货"——方向对了，但 Phi-1 原文自己**没做这一步**。Day16 的 parser+exec 三级过滤正是补这一刀：没有 execution filter 的合成教科书，教的是"看起来对的"代码。
2. **教师泄漏（decontamination 防不住的）**：Phi-1 做了 benchmark n-gram 去重，但 GPT-3.5 本人训练时见过 HumanEval——教师记忆会通过"出题风格/解题套路"间接渗入合成数据。50.6% 里有多少是"教师记住答案"，论文**无法回答**，也没有设计实验区分。这是引用 Phi-1 数字时必须加的限定。
3. **模板坍缩**：章节式生成 → 分布坍缩到几种讲解模板；初读疑问已提到——DP/图/贪心等算法范式的覆盖是否均匀，论文没有多样性度量。用 Day19 Vendi 的语言：Phi-1 优化了"干净度"，但没有报告 kernel 熵；"干净"和"多样"在这里是** trade-off**，不是免费午餐。
4. **证据边界：只在短函数级验证**。HumanEval/MBPP 都是单函数短题；repo 级、多文件协作、长尾库调用（初读疑问 #2：纯教科书会不会掉长尾 recall）**没有证据**。Day29 SWE-Gym 的 repo 级可验证任务正是这个缺口的后来者。
5. **"教科书"假设的适用域**："知识可压缩成讲解"是隐含假设。对推理密集、封闭域（算法题）成立；对**不可压缩的长尾事实**（API 签名、版本号、冷门库行为）不成立——教科书教不会你 `libfoo 2.3.1` 的诡异 bug，只能靠见过。Phi-1 的 6B web 精筛部分保留了这部分 grounding，但论文没有量化两部分各自的贡献。
6. **"6B 精筛"本身的选择偏置**：classifier 学的是"像教科书"的**风格**（注释多、解释性强），不是"正确"或"有用"。风格 classifier 会系统性偏好某种写作风格的代码（如教程式、verbose），而生产级简洁代码可能被筛掉——"教育价值"打分器的定义本身就是价值判断，Day07 Llama3 用 Llama-2-70B 当打分器时继承了同样的问题。

### 5. 迁移到 coding / post-training data

**可执行实验：500k 合成 coding 池的"教科书重排"ablation（直接检验 Phi-1 命题在你数据上的可迁移性）**

1. **配对重写**：从池中抽同一批（题，解）对。A 组保持原格式；B 组用强模型按 Phi-1 的 CodeTextbook/CodeExercises 模板重写成三件套——概念讲解（这题考什么模式）→ worked example（带注释的解法）→ 练习 + 边界测试。Prompt 模板抄 Phi-1 原文的章节式约束。调用走 cache + dedup，控制成本。
2. **同预算对照**：两个 1B 级 proxy 模型，同超参、同 token 预算，各训一遍（A 格式 vs B 格式）。看三样东西：HumanEval+/MBPP+ delta、loss curve 前期斜率（教科书格式应该让早期 loss 下降更快——"好学"的操作定义）、以及长尾库调用题（如 BigCodeBench 子集）的 recall 变化（检验第 4 节第 5 条的担忧）。
3. **可证伪标准**：若 B 在 HumanEval+ 上显著胜出（>2-3 个点）且长尾 recall 不掉 → 在池级推广"教科书重排"；若 B 只赢格式分（MBPP 涨、BigCodeBench 掉）→ 说明你的池子瓶颈在 grounding 不在结构，回头补 Day27 式真实代码锚定。
4. **第二步（若第一步成立）**：抄 Day07 做法，训一个小的"教育价值"classifier（用 B/A 人工偏好或 LLM-judge 打标签），对 500k 全池打分，留 top 5% 试训——这就是初读"可迁移"第 2 条的落地版，但先有 ablation 再放大。
5. **诚实边界**：B 组的"练习+边界测试"必须过 execution filter（Day16），否则你在复现第 4 节第 1 条的错误——"格式好但答案错"的数据进 SFT 是毒药（见第 6 节思考题 (a)）。

### 6. 今天的一道思考题

综合 **Day06 Phi-1、Day16 Qwen2.5-Coder、Day27 OSS-Instruct**：

**(a) 格式 vs 正确性：2×2 消融。** Phi-1 证明了"教科书格式"（结构化、带讲解）有用，但它的合成代码**未经执行验证**；Qwen2.5-Coder 证明了 parser+exec 三级过滤有用，但它没碰"格式"这个变量。设计实验区分两者对 HumanEval 的各自贡献：$$\text{格式} \in \{\text{raw}, \text{textbook}\} \times \text{验证} \in \{\text{未验证}, \text{exec验证}\}$$ 四个格子各训一个同预算 proxy 模型。

- 预测四个格子的 HumanEval 排序并给出理由（提示：想想"格式好但答案错" vs "格式糙但答案对"，哪个更糟？）。
- 再回答：在 **SFT（teacher-forcing 模仿）** vs **RL（verifiable reward 过滤）** 两种设定下，"格式好但答案错"的数据危害是否相同？为什么？（提示：SFT 的 loss 是对 token 分布的模仿——错答案的每个 token 都在被鼓励；而 RL 的 reward 只认最终执行结果。把"模仿"和"验证"两种学习信号的数学形式写出来对比。）

**(b) 干净 vs 现实：谁更容易坍缩。** OSS-Instruct **有意保留不完整 solution** 保现实性，Phi-1 狠删只留干净教科书。从 Day24 D4（语义去重+原型剪枝）和 Day19 Vendi（kernel 熵多样性度量）的视角看：哪种策略更容易导致训练分布坍缩？给出在你自己 500k 池上**可操作**的检测方案：用什么 embedding、算什么指标（Vendi score？簇内/簇间距离？模板 n-gram 熵？）、阈值怎么定，才能在"教科书重排"推广前发现"教科书式坍缩"？

---

**论文原文**：https://arxiv.org/abs/2306.11644
**GitHub NOTES**：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-06-2023-phi-1/NOTES.md
