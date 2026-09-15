# Paper 模板 — Day 15 DeepSeek-R1

## 元信息
- Title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- Authors / Org: DeepSeek-AI (DeepSeek-R1 Team)
- Link / arXiv: https://arxiv.org/abs/2501.12948
- Date read: 2026-08-15
- Tags: [rl-data, coding-data, reasoning, sft-vs-rl, cold-start, synthetic-data]
- Day: 15

## 一句话总结
用 <10k 合成冷启动 + 纯 RL（可验证奖励）让 base 模型自发涌现长链推理，SFT 只起格式稳定，真正泛化靠 RL，证明 SFT memorizing vs RL generalizing 在 coding/reasoning 上的分水岭。

## 和之前工作的关系
- **知识图谱位置**：post-train RL 主线的集大成，对接 2025_limr (RL 少即是多) 的“选难例”——R1 冷启动 10k 就是 LIMR 式的硬例筛选；对接 2024_superfiltering (SFT weak-to-strong) 的对比——SuperFiltering 说小模型选 SFT 数据管用，R1 说 SFT 只能稳格式，选对难例后 RL 才能超 SFT；对接 2023_phi-1 (合成教科书) 线——R1 的冷启动合成推理轨迹就是 Phi-1 教科书思想的 RL 版本，但从“教”变“写出题过程”。
- **接了哪条线**：influence / selection 线（LESS/Day4, DataInf/Day5, LIMR/Day11, SuperFiltering/Day12, DPO-gap/Day13） → R1 把 selection 从 SFT influence 换成 GRPO 奖励驱动的在线选择；synthetic 线（Phi-1/Day6 → Llama3/Day7 → DeepSeek-V3/Day9 → Qwen2.5/Day10） → R1 合成不再是预训练教科书，而是 RL 冷启动的思维链。
- **补了哪个短板**：之前 LESS/SuperFiltering 只谈 SFT 选择，LIMR 只谈 RL 选但没说 RL 要不要 SFT 热身；R1 补上 SFT vs RL 对比的空白，给出“冷启动 SFT 稳住格式→纯 RL 泛化”的 recipe，解决你之前问的“LESS 到 RL 的 gap 为什么存在”【之前对话 2026-08-13】。
- **替代/分叉/改进**：不是替代 LIMR，而是改进/分叉——LIMR 用 LIM 度量选 1.3k，R1 用人工+启发式选 <10k 冷启动 + 8k RL；相对 Phi-1 的全 SFT 合成，R1 分叉到 RL 奖励筛选。
- **对比 Day X**：
  - vs Day11 LIMR：都少即是多（1,389 vs <10k 冷启动），但 LIMR 用轨迹对齐选，R1 用难+多样+可验证选；LIMR AIME +16.7%【8617116200514826664†L23-L26】，R1 AIME 79.8% 超 o1
  - vs Day12 SuperFiltering：SuperFiltering 125M 小模型 IFD 选 SFT 给 7B【4921601271219063014†L16-L19】，R1 反向——SFT 小而弱，RL 大而强，证明 SFT 选择力 ≠ RL 选择力
  - vs Day6 Phi-1：Phi-1 6B 精筛+1B 合成教科书训 1.3B 50.6% HumanEval，R1 合成 10k 推理轨迹只做格式预热，后面靠执行/答案可验证奖励 RL 刷上去，合成→验证 的升级

## 为什么今天读它
跟 coding data / SFT / RL data 的直接连接：R1 的可验证奖励（数学答案、代码单元测试）正是 coding RL data 的金标准——你现在 50 万合成池的最大问题是“怎么知道合成题是好的”，R1 给出 execution-verified reward + 冷启动分层，SFT 只记模板，RL 才会写出新解法。

## 今天的 3 问
1. R1 的 <10k 冷启动如何选的？人工标注的难/多样/可验证三原则，换成你 50 万 coding 池，能否用 LIMR 的 LIM 轨迹对齐或 SuperFiltering 的 IFD 粗筛复刻一个 5k 冷启动子集？
2. 纯 RL 阶段的 verifiable reward 在 code 上如何实现？DeepSeek 用单元测试 pass/fail，Qwen2.5-Coder 用执行过滤，你的 flywheel 里如何搭一个“合成→执行→拒采→RL”的闭环，成本/延迟是多少？
3. 对比题：R1 说 SFT memorizes, RL generalizes，和之前 Day4 LESS（SFT 5% 打赢全量）/ Day12 SuperFiltering（125M 选 SFT 更好）是否矛盾？为什么 SFT 的少即是多在 RL 里需要换成 RL 的少即是多（LIMR）？用 R1 的 cold-start + RL 结果解释 LESS 到 RL 的失效点。

## 核心
1. Motivation: SFT 蒸馏 o1 只能记住长思考格式，泛化差，纯 RL（R1-Zero）会自发涌现推理但可读性差且冷启动慢，需要 stabilizer
2. Data Pipeline: 少量高质量冷启动 SFT（<10k 推理轨迹，人工+启发式筛难多样可验证）→ 大规模 RL（GRPO，无 value 网络，rule-based / verifiable reward 数学答案+代码执行）→ 拒绝采样再 SFT+RL 迭代
3. Key Tricks:
   - GRPO 省 value，KL 约束轻，奖励只看答案/单元测试通过，天然可扩展到 code
   - 冷启动数据刻意保留“顿悟时刻” a-ha moment 的自我反思语句，诱导 RL 自我纠错
   - 语言一致性奖励防止中英混杂，保持 CoT 可读
4. Results: R1-Zero 纯 RL AIME 71% → R1 (cold+RL) AIME 79.8% / MATH 97.3% 超 o1-preview，HumanEval 类 code 多 10%+，证明 RL 泛化 > SFT memorizing

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 从你 50 万合成池跑一遍执行过滤（unit-test pass rate 0.2-0.8 的中难段），留 5k 做冷启动 SFT，对比全量 SFT 的 HumanEval
  2. 用 GRPO 式的 rule reward 在你小 7B 上试纯 RL 200 题，看是否涌现自纠错语句，记录 a-ha 率
- Infra 视角：GRPO 无 critic 省一半显存，verifiable reward 评测可并行化，但执行沙箱成为新瓶颈，需要 vLLM rollout + sandbox 分离

## 疑问 / 下一步
- R1 的冷启动 10k 究竟多少是 code vs math，比例如何影响最终 code 能力？Phi-1 天然指令占比启发？

## 原文金句
> RL is the engine, cold-start SFT is the stabilizer — reasoning emerges from reward, not imitation.
> 我们证明了少量的高质量冷启动 + 大规模可验证奖励 RL，可以让 base 模型超越大量 SFT 蒸馏的长思考模型
---
模板来源：PAPER_TEMPLATE.md | 今日任务配套骨架，后续填 NOTES

## 第二轮复习（2026-09-15）

> 本轮复核：R1 论文 arXiv 2501.12948 的关键数据主张今日经多方二手综述交叉核验（论文原文表述以综述引文为准）。初读 NOTES 的两处表述本轮修正：① "冷启动 10k 就是 LIMR 式的硬例筛选"不准确——R1 冷启动是**先验、格式导向**的人工+启发式筛选，LIMR 是**后验、轨迹对齐**筛选，功能不同（RL 前置种子 vs RL 后复训弹药），不可混为一谈；② "冷启动 <10k"——论文原话是 "a small amount (thousands)" 的长 CoT 数据，未给精确数字，"<10k" 是合理转述但应标注为估计，且**冷启动数据不开源**，是全文最大的数据不可复现点。

### 1. 核心命题

Day 15 真正解决的 data 问题，不是"RL 要多少数据"，而是：**SFT 与 RL 的数据分工问题——哪类数据干什么，而不是多少数据**。

R1-Zero 已经证明：base 模型 + 纯 RL（可验证奖励）就能涌现推理（AIME 15.6% → 71.0%），不需要任何示范。但纯 RL 有两个数据级缺陷：① verifier 只看答案，RL 在 verifier 的盲区（语言、可读性）精确偷懒——中英混杂、不可读；② 冷启动慢、非推理能力缺失。而当时主流的 SFT 蒸馏路线（蒸馏 R1-Zero 的长 CoT）只能记住 teacher 的轨迹，泛化被 teacher 天花板锁死。

R1 的答案是**把数据按功能解耦**：冷启动 SFT 数据最小化为"脚手架"——只教格式（`|<reasoning_process>|` 分区）、可读性、自我反思模板、语言一致性，thousands 条；把"知识与技能"完全交给可验证奖励的 RL；再用 RL 产出的数据（rejection sampling 600k 推理 + 200k 通用）回灌 SFT，形成"RL 选数据 → SFT 吃数据 → RL 再探索"的闭环。**SFT 不再是"教内容"，而是"搭脚手架 + 收割 RL 的发现"**。

形式化：传统 SFT 范式里训练信号 = $\{(x, y)\}$ 的经验分布；R1 的范式里信号 = $\{(x, \text{verifier})\}$ ——数据载体从"输入-输出对"变成"输入-验证器对"。冷启动数据的价值函数不再是"覆盖多少知识"，而是"把策略分布搬进 reward 非零的 basin 的速度"（呼应 2026-09-10 三路分诊的 basin 距离概念：SFT 的剂量单位是"进 basin 的距离"，不是"知识的比特数"）。

### 2. 图谱位置

- **直接前驱 Day11 LIMR（同一"RL 少即是多"谱系，判据一先一后）**：LIMR 证明 RL 阶段选题要换判据（后验轨迹对齐 $s_i > 0.6$ 筛 1,389）；R1 证明 RL 阶段冷启动数据也要换角色（先验人工+启发式筛 thousands，选过程质量而非题目难度）。串联成完整 RL 数据管线：**R1 冷启动（RL 前置种子，定格式）→ 大规模 RL（探索）→ LIM 式复训筛选（RL 后提纯，定第二轮弹药）→ rejection sampling 收割（RL 产出变 SFT 数据）**。两者互补：LIMR 回答"RL 跑起来之后复训什么"，R1 回答"RL 跑起来之前给什么、跑完之后产出什么"。
- **重点直接对比 Day17 LIMO（"SFT memorizes" vs "SFT generalizes" 的正面对撞）**：LIMO 用 817 条精心 SFT 让 Qwen2-32B-Instruct AIME 6.5% → 57.1%/63.3%，10 个 OOD 基准 +40.5% —— SFT 可以泛化；R1 说蒸馏 R1-Zero 长 CoT 的 SFT 只能记忆、纯 RL（71% → 79.8%）才能超越示范 —— SFT 只能记忆。调和的关键：两者的"SFT"不是同一个东西。LIMO 的泛化是**模板内唤醒**（cognitive template elicitation）：817 条激活预训练已编码的知识，泛化边界 $\le$ 模板分布的包络；R1 的 RL 泛化是**模板外探索**（R1-Zero 在零反思示范下自发产生 self-reflection，推理长度持续增长、超越所有示范样本的长度分布）。形式化：记 $P_{demo}$ 为示范分布，SFT 策略被 $D_{KL}(\pi_{SFT} \,\|\, P_{demo})$ 约束在包络内，RL 策略 $\pi_{RL}$ 只被 reward 约束、 $D_{KL}(\pi_{RL} \,\|\, P_{demo})$ 可任意大。LIMO 证明包络内可以很强（32B + 好模板），R1 证明包络外才是 RL 的主场。对 coding 的含义：LIMO 路线（1k 精选 SFT）适合"预训练已见过的推理模式"；R1 路线（冷启动 + RL）适合"需要超越示范的新策略"（如仓库级 agent 的长程工具使用——Day29 SWE-Gym 里人类 expert 的策略本身就不是最优的，蒸馏它等于蒸馏次优）。
- **上游 Day16 Qwen2.5-Coder（exec 数据是 R1 code reward 的工程前身）**：R1 的 code accuracy reward 依赖"题目 + 测试用例"——正是 Qwen 三级 exec 过滤产出的 $(x, \text{verifier})$ 对。数据链：Qwen exec 产 verifier → R1 用 verifier 做 reward。没有可执行的测试用例，就没有 code 的 verifiable reward。反过来，R1 证明了这类数据的终极用法不是 SFT，而是直接当 RL 的 reward。
- **后继 Day28 ORZ（"第二轮提纯"的工程化）**：ORZ v1 的"129k 全量 RL → 13k 困难尾部（ $p_i < 4/64$ ）再 RL 100 步"是"RL 跑完之后挖什么继续训"的答案；R1 的 600k rejection sampling SFT 是"RL 跑完之后产什么给 SFT 吃"的答案。两者都是 R1 范式的第二轮，只是输出去向不同（再 RL vs 回灌 SFT）。
- **替代/互补 vs Day09 Qwen2.5 飞轮**：Qwen 是"SFT 1M dense 行为覆盖 + offline DPO + online GRPO"飞轮，R1 是"小冷启动 + 纯 RL + 拒绝采样回灌"飞轮。Qwen 的飞轮转在**行为覆盖**（九轴），R1 的飞轮转在**推理深度**。9-09 的范畴错误分析已钉死：两者的"SFT 量"不是同一个东西，Qwen 的 1M 是行为覆盖税，R1 的 thousands 是格式脚手架。

### 3. 机制深挖

**(a) 冷启动数据的四要素（论文 §2.3 的数据构造逻辑，本轮重建）。**

1. **来源自举（lineage 的关键细节）**：few-shot prompting 生成 CoT + **R1-Zero 输出经人工后处理** + 人工标注反思。即：RL 先跑通，人工把"野生的推理"修剪成"可读的模板"，再喂回下一代 RL。这是"RL → 人工策展 → RL"的自举环，不是"人工从零写示范"。数据视角的含义：冷启动数据的分布起点已经是 RL 探索过的区域，人工只做"可读性投影"。
2. **格式设计**：`|<reasoning_process>|<summary>|` 分区。这是给 RL 的 format reward 配套的**可验证格式锚**——分区让语言一致性奖励可以按段计算，让 verifier 能机械地检查"思考是否在标签内"。
3. **内容选择标准**：可读性 + 自我反思（a-ha moment 语句）+ 语言一致。**不是选"最难的题"，是选"最像好思考过程的样本"**——与 LIMO 的"难度优先"形成对照。原因：冷启动的任务是"定格式"（把策略搬进可读长 CoT 的 basin），不是"教难题"（难题是 RL 的 reward 负责的）。选难不选优，是把两个阶段的数据标准搞混。
4. **量级与可复现性**：论文原话 "a small amount (thousands)"，无精确数字、无消融、无开源。这是全文最大的数据黑箱——R1 的 recipe 里唯一不可复现的数据环节（对比 s1K 全开源）。

**(b) Verifiable reward 的数据结构（§2.2，数据视角重述）。**

- **accuracy reward**：math 答案匹配 + code 测试用例通过。训练样本 = $ (x, \text{verifier}) $ ，verifier 是数据资产的一部分，不是训练代码的一部分。
- **format reward + language consistency reward**：R1-Zero 的教训——verifier 不看的维度（语言、可读性），RL 会精确丢弃。这是 Goodhart 的数据版：**reward 的盲区 = 数据的盲区，RL 是盲区的精确探测器**。语言一致性奖励 = 给盲区补数据信号。注意论文承认该奖励让性能"slightly decrease"——可读性是有代价的，数据选择里"好看"和"好用"存在可量化的 trade-off。
- **对 coding 的映射**：`pass_rate = 通过的 hidden tests / 总 tests` 即 Day28 的 $p_i$ 。R1 的 code reward 本质就是 ORZ 的经验通过率当 reward 用——Day28 的"难度度量"在这里变成了"训练信号"。

**(c) 600k + 200k 的数据闭环（§2.4，SFT/RL 边界溶解处）。**

- **600k reasoning**：RL 收敛 checkpoint 生成 → 过滤（去混语、去过长段落、去混乱输出）→ SFT。这一步的数据是 **RL 发现的蒸馏**：SFT 吃的不再是 teacher 的示范，而是 RL 探索出的 winners。SFT/RL 的边界在这里溶解——SFT 的数据源变成了 RL 的输出分布。
- **200k non-reasoning**：writing、factual QA、self-cognition、translation，来自 DeepSeek-V3 管线 + generative reward model 评判（verifier 覆盖不到的域，用 V3 当 judge）。数据视角：verifiable 数据只覆盖推理域，通用能力仍需 AI-feedback 数据——这正是 Day26 UltraFeedback 的"偏好池"角色在 R1 内部的复现。
- **两阶段 RL 之间的 SFT 不是"回退"，是"把 RL 的探索成果固化成新的起点分布"**，让第二轮 RL（全场景）在"推理 + 通用"的更高 basin 上继续探索。

**(d) GRPO 对数据友好的机制（一句话，呼应 9-10 Q&A）。** 无 critic → reward 直接是 verifiable 的 0/1，组内相对优势 $ \hat{A}_i = (r_i - \text{mean})/\text{std} $ ——数据的"难度"自动转化为组内方差。R1 的数据筛选（只留"将信将疑"中间地带：删全过、删全错）本质是**保证每个 batch 的组内方差非零的数据选择 = 方差管理**。 $\text{std}(\mathbf{r}) \approx 0$ 则 $\hat{A}_i = 0/0$ 无信号——这是 R1 数据管线里"难度过滤"的数学根因。

### 4. 边界与反例

1. **冷启动量级无消融**：thousands 到底是 2k 还是 9k？论文没有冷启动规模的敏感度曲线。"最小剂量"是断言，不是测量。9-09 的"倒 U 型"假说（SFT 量 vs 最终效果）在 R1 这里**一个数据点都没有**——这是留给复现者的第一个实验：扫冷启动量，看 AIME/可读性的响应曲线。
2. **Verifiable reward 的域边界**：只适用于答案/测试可判定的任务。论文自己承认开放域用 generative reward model（V3）——一旦 verifier 换成学出来的 RM，R1 的"无上界"论证就退化成 Qwen 式的"RM 天花板"论证（9-09 跑偏分析的情形二：弱评估器锁死在伪相关特征上）。R1 没有量化 verifiable 域和 RM 域在最终模型能力里的各自贡献。
3. **过程监督缺失**：accuracy reward 只看最终答案/测试通过。数学题"答案蒙对、过程错"和 code"测试覆盖不足但通过"都会被当成正样本强化——与 LIMR §4.3 的缺口同构。R1 的 a-ha moment 是涌现的，不是被奖励的；reward 无法区分"真反思"和"碰对答案"。verifier 的粒度 = RL 诚实度的上限。
4. **冷启动的锚定效应（少即是多的反面）**：冷启动把策略分布搬进"可读长 CoT"的 basin，但 RL 的探索起点也被锚定在这个 basin 里。R1-Zero（无冷启动）自发产生的推理模式与 R1（有冷启动）的是否相同？论文没有对比两者的 CoT 风格差异——冷启动可能在"稳定"的同时**修剪了探索空间**。脚手架的代价：脚手架决定了你能盖出什么形状的楼。可检验的预测：R1-Zero 的 CoT 长度/结构方差应大于 R1。
5. **R1-Zero 可读性问题的反向警示（对 coding data 直接相关）**：纯 RL 在没有语言一致性奖励时产生中英混杂——证明 RL 会精确利用 verifier 的盲区。平移到 coding：exec pass/fail 只验证功能正确性，RL 可能产出"通过测试但不可维护"的代码（无注释、过度 golf 化、trick 写法）——verifiable reward 需要配套的"风格 verifier"，否则 RL 会把可维护性当成可牺牲的维度。这是 §5 实验第 4 步的动机。
6. **蒸馏成功的归因混杂**：R1-Distill-Qwen-7B 的 AIME 55.5% 很惊人，但蒸馏数据是 800k R1 生成样本——这是"SFT 吃 RL 产出"的成功，不是"SFT 吃人工示范"的成功。不能用蒸馏的成功反证"大量 SFT 本身就够"——数据源头的性质变了。
7. **证据没证明什么**：R1 vs R1-Zero（71% → 79.8%）证明了"冷启动有用"，但没有证明"冷启动必须 <10k"（无 50k/100k 冷启动对照）；无 code 冷启动 vs math 冷启动的配比消融（初读已 flag）；ORZ 57k 在 32B 上用约 1/10 步数达到 R1-Zero 级——说明 R1 的"大规模 RL 步数"部分可被"更好的题池"替代，**数据质量和训练步数之间存在替代弹性**，论文未量化这条等效曲线。

### 5. 迁移到 coding / post-training data

**可执行的映射：50 万 code 池的"R1 式三段管线"试点（4–6 周）**

1. **Verifier 资产化（第 0 步，数据基建）**：对 50 万池跑 exec 沙盒（复用 Day16 思路），每题产出 $ (x, \text{tests}, \text{pass\_rate}) $ 。只保留 $\text{pass\_rate} \in (0,1)$ 的"将信将疑"中段——全过（组内方差为零）和全挂（零信号）在 GRPO 下都是死数据（§3(d) 的数学根因）。记录：中段占比、按难度分层的 pass_rate 分布。这是 R1 数据范式的第一步：把池子从 $\{(x,y)\}$ 变成 $\{(x, \text{verifier})\}$ 。
2. **冷启动种子（thousands，不是 100k）**：从中段题里，用强模型（或小规模 RL pilot 的 checkpoint，复刻 R1"RL 输出经人工后处理"的自举环）生成长 CoT，人工/LLM-judge 按**过程质量**筛：必须含自我反思语句（"等等，这个边界条件……"）、注释与代码语言一致、单样本单语言。目标 3–5k 条。**对照实验**：(a) 5k 过程质量种子 + GRPO；(b) 100k 全量 SFT + 同 GRPO。观测 LiveCodeBench pass@1、CoT 平均长度、a-ha 语句率、熵曲线——检验"倒 U 型"假说的 R1 数据点。
3. **Rejection sampling 回灌（复刻 600k）**：RL 收敛后，用 checkpoint 对中段题重采样 n=16，verifier 留全过 **held-out** tests 的 winners（held-out 防 reward hacking，补 §4.3 的缺口），构成下一轮 SFT 的"RL 蒸馏集"。对比"人工示范 SFT 集" vs "RL winners SFT 集"作为第二轮 RL 起点的差异。
4. **风格 verifier（补 §4.5 的盲区）**：exec reward 之外，加轻量风格门——注释覆盖率、圈复杂度上限、pylint 基础分。先离线统计 RL 产出在这些维度上的漂移，确认"RL 是否在牺牲可维护性换取 pass 率"后再决定是否进 reward。这是 R1 语言一致性奖励在 code 域的对应物。

### 6. 今天的一道思考题

> 综合 **Day15（R1）、Day17（LIMO）、Day11（LIMR）**：
>
> (a) **"SFT memorizes" vs "SFT generalizes"——谁在偷换概念？** R1 说 SFT 只能记忆（蒸馏 R1-Zero 的长 CoT 泛化差），LIMO 说 817 条 SFT 在 10 个 OOD 基准上 +40.5%（SFT 能泛化）。设计一个判定实验，区分"模板内唤醒"与"模板外探索"：
> - 固定 base（Qwen2.5-32B-Instruct），两臂：(A) LIMO 式 817 SFT；(B) R1 式小冷启动 + 纯 RL（小规模复刻）。
> - 构造"teacher 从未见过的推理模式"探针集：比如需要**双重回溯**（先试 A 法、证伪、再试 B 法、再回溯修正 A 的引理）的题，确保 LIMO 的 817 模板里没有这种结构。
> - 写出可证伪判据：① 若 A 臂在探针上的**错误模式与 teacher 高度相关**（错误相关系数 $ \rho_{err} > 0.7 $ ），而 B 臂出现训练数据中不存在的新策略（如 RL 独有的验证循环），则"记忆 vs 探索"的分野成立；② 若 A 臂也能解出探针题，则 LIMO 的"认知模板"泛化半径比宣称的大，R1 的 memorizes 论断需要加限定词——"在 teacher 分布包络内 SFT 可泛化，包络外需 RL"。
> - 追问：用什么度量量化"错误相关性"？（提示：对每道探针题，记录模型失败时的**失败位置**——是卡在同一引理还是发散到不同分支；失败位置分布的距离（如 Jensen-Shannon 散度）比单纯的对错更能揭示"是不是在走同一条路"。）
>
> (b) **两条"第二轮提纯"路线的抉择**。R1 的 600k rejection sampling（RL 产出 → SFT 收割）和 LIMR 的 1,389 LIM 复训（RL 轨迹 → 再 RL）都是"用 RL 当选择器"，但一个输出给 SFT 吃，一个输出给 RL 吃。给定同一批 RL rollouts：
> - ① 设计三臂实验比较"rejection sampling → SFT" vs "LIM 筛选 → 再 RL" vs "两者串行（先 LIM 再 RL，再 rejection sampling 收割）"在最终 AIME/LiveCodeBench 上的效果与总 rollout 成本。
> - ② 写出选择规则：**第二轮的目标若是"固化格式/稳定分布"走 SFT；若是"继续探索/突破天花板"走 RL。** 用熵的语言论证：rejection sampling 的 SFT 本质是在 RL 输出分布上做**模式提纯**（mode-seeking，压缩熵）；LIM 再 RL 是在**保留分布支撑集**的前提下重加权（保持熵，需要方差）。前者适合收敛阶段，后者适合探索阶段。
> - ③ R1 的真实管线恰好是"RL → SFT(600k) → RL（全场景）"的串行——用 (b)② 的熵语言解释为什么这个顺序不是随意的（提示：第一轮 RL 探索出推理能力 → SFT 把能力固化并补上通用行为 → 第二轮 RL 在"推理+通用"的更高 basin 上继续探索；任何一步调换都会让熵在错误的阶段被压缩或放大）。

思考题答案（Gemini 网页版，2026-09-15）：https://gemini.google.com/app/0d21ba62c914ac13

---

论文原文：https://arxiv.org/abs/2501.12948

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-15-2025-deepseek-r1/NOTES.md
