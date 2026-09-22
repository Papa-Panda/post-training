# Paper 模板 - Day 22

> 自动生成骨架 2026-08-22，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: WizardLM: Empowering Large Language Models to Follow Complex Instructions
- Authors / Org: Can Xu et al. / Microsoft Research Asia, Peking University
- Link / arXiv: https://arxiv.org/abs/2304.12244
- Date read: 2026-08-22
- Tags: [synthetic-data, sft, coding-data, instruction-evolution, complexity, curation, quality]
- Folder: day-22-2023-evol-instruct
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-22-2023-evol-instruct

## 一句话总结
用 Evol-Instruct 的 In-depth / In-breadth 演化算子把简单指令递归改写成约 70k 条更复杂、多样的 SFT 数据，补上 Self-Instruct 会自举但容易停留在简单任务分布的短板。

## 和之前工作的关系

> 知识图谱位置：合成指令主线第二站，Day21 Self-Instruct（从少量种子扩规模）→ Day22 Evol-Instruct（显式提升复杂度）→ Day27 OSS-Instruct（迁移到 code 数据）。

- **接了哪条线：**直接接 Day21 的 synthetic/bootstrap 线。Self-Instruct 解决“没有指令池时怎么从种子造池”，Evol-Instruct 解决“造出的池为什么仍太简单”。
- **补了哪个短板：**把复杂度从事后评分变成生成阶段的可控维度；同时为 Day20 DEITA 的 Evol-Complexity scorer 提供来源，形成“先演化复杂度，再按复杂度×质量×多样性筛选”的流水线。
- **替代 / 分叉 / 改进：**不是替代 Self-Instruct，而是其复杂度升级；相对 Day06 Phi-1 的“合成教科书/代码内容”，本篇改造的是 instruction/task 分布；Day27 OSS-Instruct 再把这套演化迁移到 coding 场景。
- **对之前 Day X 的直接对比：**vs Day21，Self-Instruct 主要靠新任务自举与 ROUGE-L 去重扩大覆盖，Evol-Instruct 通过约束增加、具体化、推理步骤增加与 breadth mutation 主动改变难度；今天重点检查这种“更复杂”是否真的带来新能力，而不只是更长、更啰嗦。

## 为什么今天读它

Self-Instruct 给了 coding SFT / RL 冷启动的“造池”起点，但真实 coding data 需要可控难度梯度：从单函数题演化到边界条件、复杂约束、多文件上下文和可执行验证。Evol-Instruct 正好补“复杂度如何生成”的数据方法，并可在进入 SFT 或可验证 RL 数据池前，串上 Day16 parser/exec 过滤与 Day20 DEITA 三因子筛选。

## 今天的 3 问
1. In-depth 与 In-breadth 的演化算子里，哪些真的增加任务约束或推理结构，哪些只是拉长表述？应如何用数据指标审计“复杂度上升但质量不降”？
2. 对比 Day21 Self-Instruct 的 ROUGE-L 去重与分类/非分类分池，递归演化后还需要哪些门禁来拦截语义漂移、不可解任务、重复约束和答案幻觉？
3. 把 Evol-Instruct 迁移到 coding data 时，如何把“增加约束/具体化/增加推理步骤”改写成边界条件、多文件依赖、性能约束与测试用例，并在进 SFT / RL 数据池前接 Day16 的 parser + execution filter？

## 核心
1. **Motivation**: [待读后填写] 为什么普通 instruction-tuning 数据偏简单？复杂指令覆盖有什么缺口？
2. **Data Pipeline**: [待读后填写] 种子从哪来 → In-depth / In-breadth 怎么演化 → 怎么过滤失败样本 → 如何形成约 70k 数据集 → 如何评估。
3. **Key Tricks**: [待读后填写] 记录演化算子、停止条件、失败过滤、去重和复杂度判断；只记数据构造，不展开模型训练算法。
4. **Results**: [待读后填写] 只记录数据规模、质量/复杂度评估和 downstream 对照，不展开 optimizer / RLHF 等算法细节。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：[待读后填写]
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：[待读后填写]

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：[待读后填写]

## 原文金句 (1-2句)
> [阅读后补原文，勿凭记忆引用]

## 今晚产出
- 按模板补齐 Data Pipeline / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节
- 全程只写数据：synthetic / complexity / curation / quality / diversity / execution-filter；算法只一句带过

> 自动化：reading-log 已追加 / commit 由本次自动化推送 / ai data sheet 由本次自动化同步
## 第二轮复习（2026-09-22）

> 本轮核验：arXiv:2304.12244 v2/v3 全文（§2 Evol-Instruct、§4.2 实验细节）+ 多方独立转录交叉核对。初读 NOTES 基本是骨架（多处"待读后填写"），关键数字有一处实质性误读，本轮已修正并补齐数据事实。

### 元信息修正

- **70K 误读修正**：初读写"约 70K 条复杂 SFT 数据"是错的。论文是 52K Alpaca 种子经 4 轮演化得到 **250K** 指令；70K 是为了与 Vicuna 的 70K 真实用户数据（ShareGPT）公平对比而从中采样的子集。reading-log 与 README 映射表中的"约70K"表述同步修正。
- **6 个演化 prompt**：每轮每条指令从 5 个 in-depth（增加约束 / 深化 / 具体化 / 增加推理步骤 / 复杂化输入）+ 1 个 in-breadth 中等概率随机选一个，不是初读猜的"5 级固定流水线"。
- **复杂度增量约束**：每次演化只"a bit harder"，新增词数限制在 10–20 词——这是防止一步跳到不可解任务的显式护栏。
- **回应质量混杂已控制**：基线 Alpaca 的 Davinci-003 回答被替换为 ChatGPT 回答后再对比，说明论文意识到了"指令变难"与"回答变好"的混杂，并做了对照。
- **训练配置**：v2 预印本 LLaMA 7B，Adam lr $2\times10^{-5}$ ，8×V100 + DeepSpeed Zero-3，70 小时，3 epoch；ICLR 2024 正式版（v3）换 LLaMA-13B，8×V100，140 小时；API 总调用量 $52 \times 4 \times 3 = 624\text{K}$ 次（演化 / 淘汰 / 生成回答各一次）。
- Venue：ICLR 2024。

### 一句话总结

把"复杂度"从事后打分变成生成阶段的可控算子：52K Alpaca 种子（本身已是 Self-Instruct 从 175 条人工种子自举而来）经 4 轮 In-depth / In-breadth 演化 + Elimination Evolving 淘汰失败样本，得到 250K 难度梯度可控的指令；70K 子集训出的 WizardLM 在多项评测上超越同等规模的人类数据模型 Vicuna——**证明 SFT 效果的第一变量是指令复杂度分布，而非"是否人类写"**。这是"造"侧从"扩规模"到"控难度"的转折点。

### 和之前工作的关系

- **vs Day21 Self-Instruct（直接前驱 / 被改进）**：链条是 175 人工种子 → Self-Instruct 自举 → 52K Alpaca → Evol-Instruct 演化 → 250K，三级远离人工标注。Self-Instruct 解决"无中生有"，Evol-Instruct 解决"有而不难"——自举数据的复杂度被生成器能力封顶（指令短、任务简单），演化是给这个天花板打的补丁。但注意代价：每级都继承并放大上一级的分布偏置。
- **vs Day20 DEITA（造↔量闭环）**：Evol-Instruct 把复杂度"做"出来，DEITA 把 Evol-Instruct 的演化序列反向蒸馏成 Evol-Complexity scorer 把复杂度"量"出来。"造→量→选"闭环：175→52K（造）→250K（控难造）→ DEITA 三因子→6K（选）。DEITA Table 2 的教训（IFD 把"难"和"烂"混为一谈，Evol-Complexity 稳住）反过来证明：演化产生的有序复杂度序列是比直接打分更可靠的复杂度定义。
- **vs Day27 OSS-Instruct（正交轴）**：合成数据分布 = 来源分布 $P(\text{source})$ × 条件难度分布 $P(\text{difficulty}\mid\text{source})$ 。Evol-Instruct 只动后者（固定 Alpaca 种子，改难度），OSS-Instruct 只动前者（换 80K 真实代码片段当种子，改来源）。两条轴正交，可串联：先用 OSS-Instruct 定来源，再用 Evol-Instruct 拉难度。
- **vs Day24 D4/SemDeDup（缺失的去重）**：4 轮演化每条种子长出一条演化链，近重复必然累积；论文只有 Elimination Evolving（淘汰失败样本），没有语义去重。250K 里"真多样 vs 啰嗦近重复"的比例论文没回答——Day24 的语义去重是这条线的天然补丁。
- **vs Day12 SuperFiltering（弱模型能否审进化数据）**：Evol-Instruct 用强模型（ChatGPT）既当生成器又当淘汰裁判；SuperFiltering 证明 124M 弱模型算 IFD 就能筛 7B 的数据。开放问题：弱模型能否审出"演化失败"（不可解 / 伪复杂）？若能，演化管线的成本可降一个量级。

### 核心

1. **Motivation**：2023 年初的指令数据两极分化——Alpaca（52K，自举，偏简单）vs Vicuna（70K，真实用户，难但有人工参与）。论文要回答：去掉人工后，能否用纯机器方法造出比人类数据更难的指令？更深层的命题是 SFT 的 scaling law 到底 scale 的是什么：是条数，还是复杂度？
2. **Data Pipeline**：52K Alpaca 种子 → 4 轮演化（每轮每条指令等概率抽 1 个演化 prompt；in-depth 五选一改写加难，in-breadth 基于原指令造新任务扩覆盖）→ Elimination Evolving（LLM 裁判淘汰演化失败/不可解样本）→ ChatGPT 重生成全部回答（temp 1，top-p 0.9，max 2048 tokens）→ 250K → 采样 70K 子集训 LLaMA → WizardEval（人工构造的难度均衡新测试集）评测。
3. **关键机制深挖**：
   - **复杂度即算子**：5 个 in-depth prompt 是零样本的（无需 in-context 示例），说明"加难"本身可被 prompt 编程化——这是后来 DEITA 能把演化蒸馏成 scorer 的前提。
   - **渐进约束的数学意义**：每轮只允许 +10–20 词、"a bit harder"，是在做复杂度空间里的**小步随机游走**而非跳跃；大步长会直接掉进不可解区域（Elimination 也救不回来，因为裁判和生成器是同一个模型、共享盲区）。
   - **无课程表**：6 个 prompt 等概率随机选，4 轮下来是固定混合分布，不是 easy→hard 的课程。论文证明了"难样本的存在"重要，但没证明"难度的编排"重要——课程学习这块是留白。
   - **回答重生成的双重作用**：ChatGPT 重写回答既是质量统一，也是把"指令-回答"对齐到同一模型的风格分布——这正是后来 LIMA 强调的"风格统一"的机器版。
4. **Results（数据口径，ICLR 2024 正式版）**：250K 演化指令；70K 子集训 LLaMA-13B，在代码、数学、GPT-4 评测与人工评测上显著超 Alpaca 与 Vicuna；核心结论是"指令复杂度对 SFT 效果至关重要"（preliminary investigation 级别，论文自称初步探索）。

### 边界

1. **裁判与运动员同一人**：演化、淘汰、回答生成全是 ChatGPT。Elimination Evolving 淘汰的是"ChatGPT 认为失败"的样本，系统性盲区（比如某类推理它自己就不擅长）会被完整继承——这是合成数据自举的通用原罪，Day21 的"输出正确性无验证"在这里换了个马甲。
2. **70K 采样的选择偏置**：为公平对比 Vicuna 而采样 70K，但采样策略论文未细说；若采样偏向高复杂度，结论"演化数据 > 人类数据"部分来自采样策略而非演化本身。
3. **复杂度 = 词数代理的污染**：+10–20 词的硬约束把"难"和"长"绑在一起。啰嗦的简单题可能被误判为复杂——5 个算子里"具体化/复杂化输入"最容易产生这种伪复杂。
4. **没做多轮迭代稳定性**：4 轮演化后停，没有回答"第 8 轮会不会 collapse"。与 Day21"只做了一轮自举"的警告同构，只是阈值更高。
5. **论文没证明的**：复杂度提升的边际收益曲线（250K 是否过饱和？）、in-breadth 对多样性的真实贡献（vs 简单重采样）、演化数据在 RL 阶段是否同样有效（全文只在 SFT 验证）。

### 迁移到 coding / post-training data

- **可执行的 coding 演化管线**（今晚可开工）：取 OSS-Instruct 75K（或 Code Alpaca）当种子池 → 定义 code 版 in-depth 四算子：加边界条件 / 加多文件依赖 / 加性能约束 / 要求测试用例 → 2 轮演化（强模型生成）→ **淘汰裁判换成客观门禁**：parser + 编译 + 单元测试三级（Day16），不可编译/测试全挂的直接淘汰，替代论文的 LLM 自裁判——这是对"裁判运动员同一人"最直接的修复 → DEITA 式 $s = c \times q$ 打分（ $c$ 用 Evol-Complexity code 版， $q$ 用测试通过率）→ 按难度分层采样定版。关键设计决策：**coding 域的 Elimination 不该用 LLM，而该用执行器**——执行器没有"觉得难"的偏置，只有"跑不跑得过"的事实。
- **给 post-training 的教训**：SFT 数据设计的第一变量是难度分布的形状，不是条数；先做难度审计（Evol-Complexity 打分看分布），再决定是"演化加难"还是"换源重造"。

### 思考题（综合 Day22 / Day20 / Day24 / Day16）

- **(a) 演化复杂度 vs 有效多样性**：取 Evol-Instruct 式 250K 演化池，三臂各取 70K：A = 随机采样；B = DEITA 式 $s = c \times q$ 取 top；C = 先 SemDeDup 语义去重（cos>0.9）再随机取。评：Evol-Complexity 分布、Vendi 多样性、下游分数。判据：若 B≈C 且都显著>A → 复杂度选择与去重殊途同归，演化的"难" mostly 是真金；若 C>B → 演化制造了大量啰嗦近重复，"复杂度上升"部分是词数幻觉，DEITA 的 $c$ 分被长度污染了。
- **(b) 客观淘汰 vs LLM 淘汰**：同一批 code in-depth 演化样本，A = ChatGPT 当 Elimination 裁判；B = parser+编译+单元测试三级客观淘汰。看两点：① 下游 HumanEval/SWE-bench 分数；② B 误杀 / A 漏杀的样本里，"测试难写但任务有价值"的难样本占比。判据：若 B 下游赢但误杀率高 → 客观门禁系统性偏向"易验证任务"，coding 数据会被执行器反向选择成"短算法题"，这正是 Day27 疑问的定量版；此时正确做法是淘汰分级：编译失败直接杀，测试难写但语义合理的进人工/强模型复审池。

论文原文：https://arxiv.org/abs/2304.12244

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-22-2023-evol-instruct/NOTES.md

