# Paper 模板

> 复制这个模板到 `{name}/NOTES.md`（现在直接在 ai-data 下平铺，不再有 papers/ 中间层）

## 元信息
- Title: Qwen2.5-Coder Technical Report
- Authors / Org: Alibaba Cloud Qwen Team / Qwen Code Team
- Link / arXiv: https://arxiv.org/abs/2409.12186
- Date read: 2026-08-16
- Tags: [coding-data, execution-filter, curation, quality, synthetic-data, sft, rl-data]

## 一句话总结
Qwen2.5-Coder 解决了 code data 灌水与不可执行噪音问题，用 parser 语法过滤 + 执行验证过滤 + LLM质量过滤 + 去重三级瀑布，把5.5T code tokens洗成可验证可执行的高质code/推理语料，让7B/32B在HumanEval/MBPP/Aider上超同级并反哺Qwen2.5生成可验证RL数据。

## 和之前工作的关系
- **知识图谱位置**：post-train coding data主线的「执行过滤」分支，承接 pretrain瀑布过滤（Llama3 5级 / Qwen2.5 file→repo）与合成数据（Phi-1教科书 / DeepSeek-V3高质量合成annealing），补之前 Day 09 Llama3、Day 10 DeepSeek-V3、Day 11 Qwen2.5只讲预训练过滤、Day 15 DeepSeek-R1只讲verifiable reward但没讲filter具体的短板
- **接了哪条线**：influence/selection线（Day 05 LESS梯度选、Day 12 SuperFiltering 125M IFD选、Day 13 DPO-Gap选难例、Day 13 LIMR选难RL）之后，execution是另一种hard filter：不是选influential而是选executable/verifiable；也接 synthetic线（Phi-1、StarCoder2合成教科书）的执行校验闭环
- **对比 Day X**： vs Day 14 StarCoder2（600+语言规则+Near-dedup，规则级但无执行）→ 本篇加执行级验证； vs Day 10 DeepSeek-V3（14.8T code 30%去重更狠FIM 10% PSM）→ 本篇更小但更精、执行去伪； vs Day 15 DeepSeek-R1（<10k冷启动+纯RL涌现推理）→ 本篇是RL前的数据底座，提供可验证reward的clean pool； vs Day 05-06 Influence（TracIn/DataInf扫脏）→ 执行过滤是白盒可验证扫描，成本更低可扩展
- **替代/分叉/改进**：不是替代influence选，而是分叉出execution-verified子图：pretrain流水（Llama/Qwen/DeepSeek）→ influence/selection → execution filter → SFT/RL（LIMR/Difficulty）；是DeepScaleR/PrimeRL可验证RL的前置数据工程化改进，站在Qwen2.5 18T和DeepSeek-V2/V3管线肩上把code data从“多而杂”转向“少而可执行”

## 为什么今天读它（和 coding data / SFT / RL data 的连接）
- **与之前工作的关系** 必写小节已在上方展开
- coding data工作：你每天都在做code curation，执行过滤是可直接抄的三级瀑布（parser→exec→LLM judge），门槛低见效快，infra可做sandbox并发池
- SFT连接：Qwen2.5 1M+ SFT里code/text/code-reasoning配比+decontamination 10-gram，execution过滤后的高质code可直接做SFT seed，比Phi-1合成教科书更可信
- RL data连接：DeepSeek-R1冷启动<10k+可验证reward依赖clean executable data，Qwen2.5-Coder的执行池就是R1/PrimeRL/DeepScaleR的燃料，LIMR选难+执行过滤保真两步互补

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？Code LLM pretrain被大量不可编译、不可执行、重复、低质code拖累，HumanEval高分但真实仓库任务（Aider/SWE）掉点；合成数据幻觉多，需execution作为ground truth。
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：The Stack v2 / GitHub permissive + 高质合成code/text/code-reasoning → ①Parser AST可解析性 → ②去重（repo级/文件级MinHash/精确）→ ③执行过滤（单元测试/沙盒exec pass率）→ ④LLM质量打分过滤（Qwen2.5打分） → ⑤去污染（10-gram）→ 调整sampling ratio进Qwen2.5-Coder SFT+RL。
3.  **Key Tricks**: 3个最值得抄的细节：Sandbox大规模并发执行池（Python/多语言exec timeout 5-10s）作为filter，pass/fail二值最稳；File→Repo聚合后再exec，比file级更保上下文，Qwen2.5 file→repo升级的同款思路；LLM-as-judge二阶段：小Qwen 7B粗筛，大72B精筛，cheap-to-expensive级联，类似SuperFiltering weak-to-strong但用于quality而非IFD。
4.  **Results**: 7B/32B Qwen2.5-Coder在HumanEval 88.4/92.7、MBPP、LiveCodeBench、Aider-edit、McEval上超DeepSeek-Coder-V2/StarCoder2 15B，code reasoning能力反哺Qwen2.5 72B，证明执行过滤>参数扩大。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：立刻搭一个Python多语言exec sandbox池（超aider最小可执行单元），对现有code pool跑pass率过滤，阈值可先0/1硬过滤；借Qwen2.5思路把现有file级pool做repo级聚合+去重再exec，可提Aider/SWE-bench有用性。
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：执行过滤比influence梯度（LESS/DataInf）便宜3个量级，embarrassingly parallel，gloo/CPU可先原型；eval侧直接复用exec结果做verifiable reward生成器，对接DeepScaleR/R1的RL flywheel，成本与评测合并。

## 疑问 / 下一步
- Qwen2.5-Coder的exec沙盒对多文件仓库级代码（非self-contained函数）如何判positive？是否用repo级编译+测试覆盖率而非单函数exec？这点对你做SWE-Gym数据最关键。

## 原文金句 (1-2句)
> Execution is the only scalable ground truth for code data quality; syntactic correctness is necessary but not sufficient.

## 今天的 3 问
1. 执行过滤的pass阈值设计：Qwen2.5-Coder用二值pass/fail硬过滤vs DeepSeek-V3用更狠的去重+10% FIM，你认为对你当前code pool哪种更能提HumanEval vs Aider等仓库级任务的gap？如何用LIMR的难度分层校准exec的hardness？
2. 对比LESS（Day 06 gradient similarity选5%打赢全量）和SuperFiltering（Day 12 125M IFD弱选强），execution signal在选数据效率/迁移性上是否本质优于influence/IFD？什么场景下influence仍不可替代？
3. 如果要把Qwen2.5-Coder的file→repo→exec流水线搬到你现在的SFT/RL数据工厂，最小的可验证闭环（sandbox + MinHash + LLM judge级联）需要多少机器/时间？如何与Day 15 DeepSeek-R1的<10k冷启动合成接起来做RL前的clean seed？

---
先看：https://arxiv.org/abs/2409.12186
今晚产出：ai-data/2024_qwen2.5-coder/NOTES.md 按模板填，NOTES里必须有「和之前工作的关系」小节

> 自动化：reading-log 已追加 / {commit_id} 已推 / ai data sheet 已同步


## 第二轮复习（2026-09-16）

> 本轮复核：论文原文（arXiv:2409.12186 v3，HTML 版）逐节核验。修正初读 NOTES 两处：① "5.5T code tokens" 不精确——摘要的 5.5T 是全 corpus headline（"over 5.5 trillion tokens"），§3.1.2 明确 final training dataset = 5.2T tokens，配比 70:20:10（Code:Text:Math）。5.5T 是训练 token 预算口径，5.2T 是数据配方口径，复现时以后者为准。② "小 Qwen 7B 粗筛、大 72B 精筛的 cheap-to-expensive 级联"——原文无此细节。论文只说"用 LLM 生成多候选、用 LLM 打分取最优"（§4.2 coarse-to-fine），以及 checklist 9 项加权打分；没有披露 judge 模型规模或级联设计。级联是合理的工程推断，但必须标注为推断，不是论文主张。

### 1. 核心命题

Day 16 真正解决的 data 问题，不是"code 数据不够多"，而是：**可验证性在 code 数据管线里的位置问题——把 execution 从"训完之后的评测工具"前移到"进训练之前的数据门禁"**。

当时的 baseline（StarCoder2 式规则过滤：license 门禁、near-dedup、600+ 启发式规则）能删掉"格式坏的"，但删不掉"语法对、跑不通、幻觉的"。Qwen2.5-Coder 的答案是**分诊式执行门禁**：

- self-contained 的算法类代码 → multilingual sandbox 跑单元测试（白盒、客观、可扩展的二值信号）；
- 非自包含的复杂代码片段 → LLM-as-judge（checklist 9 项加权打分，黑盒信号）；
- 全部 snippet 先过 tree-sitter AST 静态检查（语法错的直接出局）。

这和 Day09 Qwen2.5 §4.2 的"白盒信号能评的走 offline"是同一条分诊哲学，只是落点从"训练阶段选择"变成了"数据清洗门禁"。形式化：数据质量函数从 $Q(x) = f_{rule}(x)$ 升级为 $Q(x) = f_{rule}(x) \land f_{exec}(x) \land f_{judge}(x)$ ，其中 $f_{exec}$ 只在"可执行子集"上有定义——**承认 verifier 的定义域边界，而不是假装一个信号能评一切**，是这篇最值得抄的设计观。

### 2. 图谱位置

- **直接前驱 Day14 StarCoder2（规则门禁 → 执行门禁，互补叠加）**：StarCoder2 是"规则级门禁"（92+ 语言同样被 Qwen 沿用、license gate、near-dedup）；Qwen 在规则之上加了"执行级门禁"（AST 静态检查 + 仅 self-contained 进 sandbox + checklist LLM 打分）。两者正交：规则删"格式坏的"，执行删"跑不通的"。**重点直接对比**：同源 GitHub（Feb 2024 前）数据，StarCoder2-7B base HumanEval 35.4 vs Qwen2.5-Coder-7B base 61.6（论文 Table 5）。这个差距不能全归因执行过滤（token 量 5.2T vs 1T、配比、架构都不同），但论文给了"弱模型做质量门禁"的干净消融：Text-Code Grounding 数据经 4-stage fastText 过滤后，1.5B 上 HumanEval+MBPP 均分 41.6% → 46.8%（§3.1.1）。fastText 只看 surface feature、故意不用大模型——这正是 Day12 SuperFiltering "weak-to-strong" 哲学在预训练数据上的实例：**便宜评估器 + 外部锚点验证（下游分数），不需要评估器本身很强**。
- **后继 Day29 SWE-Gym（执行信号的粒度升级）**：Qwen 的 sandbox 验证的是"代码片段能跑"（样本级可执行性）；SWE-Gym 把执行推进为"issue 被修好"（任务级可验证性：repo snapshot + unit tests + agent trajectory）。数据链：Qwen exec（离线门禁）→ SWE-Gym（任务自带 verifier）。Qwen 回答"进训练前拦掉什么"，SWE-Gym 回答"训练时拿什么当 reward"。
- **互补 Day15 R1（verifier 资产化）**：R1 的 code accuracy reward 需要 $(x, \text{tests})$ 对——Qwen 的 Unit Test Generator + Code Execution Engine 就是生产这种对的工厂。Day15 复习已钉死数据链：Qwen exec 产 verifier → R1 用 verifier 做 reward。反过来 R1 证明了这类数据的终极用法不是当 SFT，而是直接当 RL 的 reward——Qwen 把执行用在了"清洗"，R1 把执行用在了"训练信号"，同一资产两种用法。
- **上游 Day27 OSS-Instruct（合成幻觉的两种解法）**：OSS-Instruct 用真实开源代码片段锚定合成约 75k 指令（锚定真实性）；Qwen 用 CodeQwen1.5 生成合成数据 + executor 校验只留可执行的（验证正确性）。两者可串行：OSS-Instruct 锚定生成 → Qwen 式 exec 校验，形成"真实来源 + 可验证"的闭环。
- **替代/互补 vs Day04 LESS、Day12 SuperFiltering**：execution 是"白盒硬门禁"（二值 pass/fail），influence/IFD 是"软选择"（排序）。execution 便宜（embarrassingly parallel）、与模型无关、可解释；但它只能回答"能不能跑"，回答不了"对目标任务有没有用"——influence 在"目标任务定向选择"上不可替代。

### 3. 机制深挖

**(a) Checklist 加权打分（§4.1，9 项、权重先验）。** 评分点：Q&A 一致性、Q&A 相关性、难度、code 存在性、正确性、命名/缩进/最佳实践、清晰度、注释、易学性；总分 $s = w_1 s_1 + \dots + w_9 s_9$ ， $w_i$ 预定义。**张力**：9 项里 5 项是风格项（命名、缩进、注释、清晰、易学），风格分可能主导总分；且"难度"（ $s_3$ ）与"正确性"（ $s_5$ ）天然负相关——难的题更容易错，加权会系统性惩罚难例。这与 Day13 DPO-Gap "留难例"的方向直接相反：**Qwen 的 checklist 在"用难度换正确性"，DPO-Gap 在"用正确性换难度"**，两者都是人为先验，没有数据能同时证明。

**(b) Sandbox 分诊（§4.1，五模块、只收 self-contained）。** 全部 snippet 先 AST 静态检查（tree-sitter，多语言，解析错的出局）；只有 self-contained（算法题类）进五模块 sandbox：语言支持模块、样本代码库、单元测试生成器、执行引擎（隔离环境、并行、timeout）、结果分析器。**关键设计决策**："执行只验证它能验证的"——仓库级/多文件代码不强行编测试，而是分流给 LLM-as-judge。这是诚实的设计，也是 Day29 要补的缺口（Qwen 止步的地方正是 SWE-Gym 的起点）。

**(c) Coarse-to-fine SFT（§4.2，多样性先、质量后）。** 第一阶段：tens of millions 低质但多样的合成指令样本先训（买覆盖）；第二阶段：millions 高质样本（rejection sampling + 同一 query 多候选、LLM 打分取最优）再训（买精度）。与 LIMA（直接 1k 高质）是相反的时序策略：LIMA 赌"预训练已编码一切"，Qwen 赌"先铺量再提纯"。代价是两阶段都要付训练费；收益是 7B-Instruct HumanEval 88.4、32B-Instruct 92.7（Table 16），32B 超 Claude-3.5-Sonnet（92.1）。

**(d) DPO 的执行反馈分诊（§4.2，白盒偏好信号）。** 算法类自包含片段：sandbox 生成测试用例，执行结果当偏好信号；复杂片段：LLM-as-judge 判优劣。这是 Day26 UltraFeedback（GPT-4 打分）+ Day13（选难对）的上游替代：**偏好对的"胜负"不需要人类/GPT-4 判，测试用例判**——偏好数据第一次有了不依赖大模型的 ground truth。但注意：只适用于"测试用例可判定"的那部分代码，复杂代码仍回退到 LLM judge（= UltraFeedback 路线）。

**(e) 配比实验（§3.1.2，70:20:10）。** Code:Text:Math 三档对比 100:0:0 / 85:15:5 / 70:20:10，7B 上 70:20:10 最优（coding 48.3、综合 55.0），甚至超过纯 code 组。论文的解释是 Math/Text 要达到浓度阈值才正向贡献。数据视角：**跨域数据不是"稀释"，是"催化剂"**——但这个结论只在 7B 上测过，0.5B/32B 是否同配比最优没有证据（呼应 §4.3 的规模外推缺口）。

### 4. 边界与反例

1. **执行过滤的幸存者偏差**：只留"可执行的"，系统性偏向"容易写出测试的简单算法题"，惩罚"难但有价值"的代码（并发、分布式、胶水代码、UI）。Sandbox 的"self-contained"准入条件本身就是一种选择偏差——真实仓库代码大多进不了 sandbox。HumanEval 涨了，不代表仓库任务涨了。
2. **测试用例是模型生成的**："代码通过测试"可能是"测试太弱"而不是"代码对"。论文没有报告测试覆盖率、突变分数——pass/fail 二值的可信度没有被校准。这是执行过滤版的 Goodhart：**优化"通过自生成测试"可能选出"测试友好型"代码**，而不是正确的代码。
3. **5.5T vs 5.2T**：headline 与配方口径不一致；且配比实验只在 7B 上做过。复现者按 5.2T + 70:20:10 配 32B，是在做无证据的规模外推。
4. **Decontamination 只有 surface 一半**：10-gram word-level overlap（§5），对代码，变量重命名就能绕过；HumanEval 问题的改写版（语义级泄漏）不在门禁内。Day30 的 surface + semantic 双重匹配正是补这个缺口——Qwen 的防漏只有一半。
5. **Checklist 权重无学习**： $w_i$ 预定义，"难度 vs 正确性"的 trade-off 被人为固定；风格项占 5/9，可能把"丑但正确且难"的代码筛掉。论文没有消融"去掉风格项会怎样"。
6. **证据没证明什么**：全文没有"执行过滤 vs 不执行"的直接消融——论文的消融是 grounding 数据的 4-stage fastText（§3.1.1），不是 sandbox 本身。HumanEval 88.4/92.7 是全套管线（5.2T 预训练 + coarse-to-fine SFT + DPO）的联合产物，**不能归因到执行过滤单步**。初读 NOTES 的"执行过滤 > 参数扩大"是个漂亮口号，但论文没有给"同参数、有/无 exec"的对照。

### 5. 迁移到 coding / post-training data

**可执行的映射："双门禁 + 分诊"最小闭环（2–3 周，不依赖大模型）**

1. **AST 第一道门（零成本）**：对现有 code pool 跑 tree-sitter 解析（Python/Java/JS/Go），解析失败的标 dirty。复刻论文"静态检查 for all snippets"——这是 StarCoder2 规则门禁里最便宜的一档，先拿到。
2. **Self-contained 分诊进 sandbox**：识别无外部 import、单文件可运行的样本；有测试的跑测试，无测试的用强模型生成测试——但必须记录 $\text{test\_source} \in \{\text{human}, \text{generated}\}$ 字段，因为生成测试的可信度低一档（补 §4.2 的缺口：论文没区分这两类测试的可信度）。
3. **非自包含走 checklist**：9 项清单直接抄论文， $w_i$ 先设均匀；离线统计各分项与下游 HumanEval delta 的相关系数，再学权重——把"预定义权重"升级为"数据驱动权重"，这是论文没做、但可以直接做的改进。
4. **产出物是四元组，不是干净池**：每样本输出 $(x, \text{verifier}, \text{pass\_rate}, \text{test\_source})$ ——直接对接 Day15 R1 式的 RL 数据格式（ $(x, \text{verifier})$ 范式）。让"清洗"和"RL 数据资产化"是一次工作，而不是两次。

### 6. 今天的一道思考题

> 综合 **Day16（Qwen2.5-Coder）、Day14（StarCoder2）、Day29（SWE-Gym）**：
>
> (a) **"规则门禁 vs 执行门禁"的归因实验**。固定同一批 GitHub 数据（Feb 2024 前），三臂：A 只跑 StarCoder2 式规则过滤（license + near-dedup + AST 可解析）；B 在 A 基础上加 Qwen 式 sandbox 执行过滤（仅 self-contained 进测试）；C 在 B 基础上对非自包含样本加 LLM checklist 打分过滤。三臂各自训同规模小模型（如 1.5B），在 HumanEval（函数级）、BigCodeBench（工具/指令）、SWE-bench-style 仓库任务三档 benchmark 上看 delta。写出可证伪判据：① 若 B−A 在 HumanEval 上显著为正、但在仓库任务上 $\approx 0$ ，则执行门禁的收益局限在"可测试分布"内，证实 §4.1 的幸存者偏差假说；② 若 C−B 在 BigCodeBench 上为正而 HumanEval 上为负，则 checklist 风格项确实在"用 HumanEval 分数换指令跟随"，量化 §4.5 的风格–正确 trade-off。追问：如何控制"三臂数据量不同"的混杂？（提示：下采样到同 token 数再比；或者把"过滤"重 framing 为"给定 token 预算下的数据选择"，与 Day04 LESS 的 5% 预算视角统一——过滤和选择是同一枚硬币。）
>
> (b) **从"样本可执行"到"任务可验证"的升级路线**。Qwen 的 sandbox 验证"代码片段能跑"，SWE-Gym 验证"issue 被修好"。设计一条数据管线，把 Qwen 式 exec 过滤的输出（带 $\text{pass\_rate}$ 的 self-contained 样本池）升级为 SWE-Gym 式任务池：① 哪些 Qwen 样本天然可升级（有明确输入输出、可构造 FAIL_TO_PASS）？② 对不可升级的样本，能否用"issue 合成"（从 commit/PR 反推任务描述——Day14 的 PR/Commit 数据正好是原料）批量制造仓库级任务？③ 写出质量门禁：合成任务的 verifier 可信度如何评估（测试翻转率？人工抽检率？），以及当 verifier 不可信时 RL 会学到什么（呼应 Day15 §4.5：verifier 的盲区 = RL 的精确攻击面；执行过滤的 Goodhart 在这里会以"刷合成测试"的形式复活）。

---

论文原文：https://arxiv.org/abs/2409.12186

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-16-2024-qwen2.5-coder/NOTES.md
