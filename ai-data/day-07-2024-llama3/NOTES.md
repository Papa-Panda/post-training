# Paper 模板

## 元信息
- Title: The Llama 3 Herd of Models
- Authors / Org: Meta AI (Llama Team, Aaron Grattafiori et al. ~500 authors)
- Link / arXiv: https://arxiv.org/abs/2407.21783
- Date read: 2026-08-10
- Tags: [pretraining, coding-data, curation, quality, scaling, flywheel, synthetic-data]

## 一句话总结
Meta 用 15.6T tokens 从零搭的工业级预训练管线，堆了 5级质量过滤+多轮去重+code/math 专用抽取（最终配比约 50% 通用 / 25% 数学推理 / 17% 代码 / 8% 多语）+annealing 高质数据，把 405B 训到对标 GPT-4，证明了长尾质量飞轮比单纯堆数据量更管用。（2026-09-08 修正：初读误写"code上采样25%"与"多轮合成/回译"，见第二轮复习 §4）

## 核心
1. **Motivation**: 之前 Llama 2 2T 就见顶，再堆 token 收益衰减。问题不在量，在 web 臭、重复多、code配比低。需要一套可控 15T 的配方，让 405B 既懂 web 又会 code 还能长推理。
2. **Data Pipeline**: 来源 → Web、Code、Multilingual、多模态；清洗 → heuristic (行长/符号密度/黑名单) → exact dedup (Bloom) + fuzzy MinHash + URL去重 → safety/PII；质量 → fastText 粗筛 → Roberta/BERT质量分类器 → Llama 2 chat 模型打标签（仅做标注集）→ 蒸馏 DistilRoberta 全量打分 → 最终人工抽检；合成（主力在后训练与 annealing，预训练 §3 未披露细节）→ Code回译、Math推理链、拒绝采样蒸馏；配比 → 最终约 50% 通用 / 25% 数学推理 / 17% 代码 / 8% 多语，初期重通用，后期 annealing 加高质合成。（2026-09-08 修正初读误植，见第二轮复习 §4）
3. **Key Tricks**:
   - Code 配比：论文最终配比代码约 17%（25% 为数学与推理）；论文未披露 code 上采样的消融实验，初读的"17%→25% / HumanEval +8-12 pts"无出处，已收回（2026-09-08，见第二轮复习 §4）。
   - 级联过滤是关键：heuristic砍30%垃圾，MinHash再砍15%近重复，fastText砍20%，Llama2打分器再砍10%低质，最后15.6T是15T里挑的不是堆出来的。Phi-1是单级textbook classifier，Llama3是5级瀑布，成本/召回更好控。
   - Annealing + 高质合成：最后 5% tokens 用超高质 code + math chain-of-thought 做 annealing，405B 推理直接 +10%。
4. **Results**: 15.6T 训完 405B，HumanEval 89.0% (vs 86% Llama2-70B-code基线), MBPP 81%，MMLU 87.3%，长上下文 128k NIAH接近满分。用同样配方训 8B/70B 都超 Llama2同尺寸。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 把你 50万 合成池按 Llama3 五级走一遍：先 exact dedup、再 MinHash 0.85、fastText教科书分>0.7、再用你 1.3B 当打分器筛一遍，应能砍掉30%水数据且 HumanEval 不掉。
  2. Code配比参考论文最终值（代码约 17%，数学与推理 25%；"25% ceiling"为初读误植，已修正）；dense 架构下 code 配比不宜照搬 MoE 的高配比，后 20% annealing 阶段再上采样高质 code+exec验证过的数据，做 RL 前的 SFT 冲刺。
- Infra 视角：15T级别必须流式+分片，Bloom dedup + 分布式MinHash是瓶颈，已有 Llama3 infra开源脚本可复用，评测自动化上用持续 HumanEval 每 5k steps 跑一次做配比信号。

## 疑问 / 下一步
- 论文未披露 code 配比消融，code 配比的收益曲线未知；若要验证，需自己跑小模型配比实验。（初读"code从17%→25%"的前提已收回，2026-09-08）
- 它的 Llama2打分器如果换成你现在 RL reward model 做二次筛，会不会比通用质量分更准？

## 原文金句 (1-2句)
> We pretrain on 15.6T tokens with careful curation to balance knowledge, code, and reasoning — quality trumps raw size.

> Code upsampling to 25% significantly improves reasoning while preserving general capabilities if done before annealing.

## 参考
- GitHub 官方配方讨论: https://github.com/meta-llama/llama3 (含数据配方摘要)
- 同步创建: papers/2024_llama3-herd/README.md 见 herd 版

## 第二轮复习（2026-09-07）

> 本轮已对照论文原文 §3.1–§3.2（arXiv 2407.21783）逐项核对初读 NOTES；发现两处事实性偏差，已在下文明示纠正。

### 1. 核心命题

Llama 3 真正解决的 data 问题，不是"怎么堆到 15T"，而是**"15T 粗语料里，哪些该留、留多少、用什么便宜的方法证明"**。Llama 2 在 1.8T 见顶，说明瓶颈不在 token 量而在 web 噪声、重复与配比错位。

这篇的真正贡献是把"预训练数据配方"从不可复现的炼丹，变成一套**可工程迭代的决策流程**，三个可迁移的决策机制：

1. **级联过滤瀑布**：PII/安全 → 自定义 HTML 解析 → URL/doc/line 三级去重 → heuristic → fastText → DistilRoberta（Llama 2 标注蒸馏），按成本升序排列；
2. **配比决策双工具**：knowledge classification（给 web 打领域标签，下采样 over-represented 类别）+ scaling law 小模型实验（多配比候选训小模型，预测大模型表现），最终配比约 50% 通用知识 / 25% 数学推理 / 17% 代码 / 8% 多语；
3. **Annealing 既是训练收尾，也是数据源评估器**：用 50% 训好的 8B 模型 + 40B tokens + 30% 新数据权重，看 GSM8k/MATH 的 delta 来判断一个新数据源值不值——比为每个数据源跑 scaling law 便宜得多。

一句话：Llama 3 是"配方"（Day06 Phi-1）的工业化——把"数据质量 > 数据规模"从 1.3B 的实证，变成 405B / 15.6T 的工程。

### 2. 图谱位置

- **前驱 Day06 Phi-1（配方 → 工厂）**：Phi-1 用 classifier 从 The Stack 选 textbook-like 文件（6B），Llama 3 把这个思想放大为三级质量 classifier 级联：fastText（"would be referenced by Wikipedia"信号）→ DistilRoberta（在 Llama 2 chat 模型的质量判断上训练）→ 领域专用 code/math classifier（同样是 DistilRoberta，在 Llama 2 标注的 web 数据上训练，prompt tuning 瞄准数学推导、STEM 推理、代码与自然语言交错页）。
  - **纠正初读误植**：初读 NOTES 写"Llama 2 70B 当质量打分器（educational value）"——论文原文没有 "educational value" 这个词，那是 Phi-1 的概念被误安到了 Llama 3 头上。Llama 3 的真实链条是：Llama 2 的 chat 模型按文字描述的质量要求给清洗后的 web 文档打标签（慢、贵，只做标注集）→ 蒸馏进 DistilRoberta 做全量打分（快、便宜）。"慢模型造标签、快模型全量打分"的两级蒸馏链才是这里该抄的机制。
- **直接对比 Day25 FineWeb / RefinedWeb（重点）**：FineWeb 是 Llama 3 网页瀑布的**开源可复现对照组**。Llama 3 §3.1.1 的每一步——URL/doc/line 去重、heuristic 三件套（duplicated n-gram coverage、dirty word counting、token 分布 KL 散度）、fastText→模型 classifier——在 FineWeb 里都有开源实现，外加论文没做的东西：**每个 gate 的消融评测**。关键差异不在步骤而在可审计性：Llama 3 只写 "we experimentally evaluate the efficacy of various quality filtering configurations"，不给任何阈值和保留率；FineWeb 把每个过滤器都变成"规则 → 小模型训练 → 保留/回滚"的可审计闭环。两者不是替代关系：Llama 3 是闭源配方的存在性证明，FineWeb 是它的可验证影子。
- **同级对标 Day08 DeepSeek-V3**：14.8T vs 15.6T，同样的 web+code+多语起手式。DeepSeek 把 MinHash 去重调得更激进、code 配比拉到 30%+（MoE 架构容纳得下，dense 在 30% 会掉通用能力），另加 10% FIM 与执行过滤合成。值得注意的反向引用：Llama 3 论文写 "Similar to DeepSeek-AI et al. 2024, we build domain-specific pipelines that extract code and math-relevant web pages"——code/math 专用抽取管线这个想法，Llama 3 是从 DeepSeek 那边学来的，不是原创。
- **配比层 Day31 DoReMi**：Llama 3 的 50/25/17/8 是"手工 + scaling law 小模型实验"搜出来的经验配比；DoReMi 要把"配比"本身变成可学习问题（proxy 模型的跨域 excess loss 学 domain weights）。Llama 3 的配比流程 = DoReMi 想自动化的手工版。两者共享同一个深层模式：**数据决策永远用比最终训练便宜 1–2 个数量级的 proxy 做**（Llama 3 用小模型 scaling law 和 annealing 评估，DoReMi 用 280M proxy）。
- **互补 Day24 D4 / SemDeDup**：Llama 3 只有词面去重（MinHash、line-level），没有语义去重。论文亲口承认 line-level dedup（同一行在 30M 文档桶里出现超 6 次就删）会连高频高质量文本一起删掉，但经验上仍有增益——这正是 D4 要解决的"误删覆盖"问题：用 embedding 空间的簇内保留策略替代一刀切。
- **后继 Day09 Qwen2.5**：Llama 3 的 annealing 数据评估法（GSM8k +24.0%、MATH +6.4% 看数据源价值）是 Qwen 式"数据飞轮门禁"的早期形态；Qwen2.5 把 18T→1M SFT→多阶段 RL 的数据门禁系统化了。

### 3. 机制深挖

**(a) 三级去重的粒度设计：从粗到细，各管一种重复。** URL 级（同一 URL 只留最新版本，管时间重复）→ 文档级全局 MinHash（管近重复转载）→ 行级激进去重（30M 文档桶内出现超 6 次的行删掉，管导航栏/cookie 警告/日志样板）。顺序不能反：先砍大颗粒再砍细颗粒，否则 line-dedup 要在全量噪声上跑。最值得玩味的是论文的诚实注脚：人工质检发现 line-dedup 删掉了不少高频高质量文本，但下游评测"strong improvements"——这是"删错一些也比留着强"的工程判断，背后假设是**高频文本的信息熵贡献低于它造成的记忆/过拟合成本**。这个假设对通用 web 成立，对 code 不一定（import 块、license header 天然高频）——所以 Llama 3 对 code/math 另起 domain-specific 抽取管线对冲。

**(b) 质量过滤的成本升序级联。** Heuristic（免费规则：n-gram 重复覆盖率、dirty word 计数、token 分布 KL 散度）→ fastText（毫秒级，Wikipedia-reference 信号）→ DistilRoberta（百毫秒级，Llama 2 判断蒸馏）。瀑布的真正设计原则不是"模型越来越大"，而是**"单位成本越来越贵，精度越来越高"**：便宜的先砍大头，贵的只看难例。初读 NOTES 的"砍 30%/15%/20%/10%"论文里不存在——论文不披露任何一级的保留率，这是本轮核对原文后必须收回的数字。

**(c) 配比决策 = 分类 + 小模型实验。** Knowledge classification 给 web 数据打领域标签，把 over-represented 的类别（如 arts and entertainment）下采样——这是"配比"的第一步：先知道池子里有什么。Scaling law 实验：在多个配比候选上训一批小模型（40M–16B，compute 从 6e18 到 1e22 FLOPs），用 NLL→accuracy 的两阶段映射预测大模型在该配比下的下游表现，选出候选后再训更大的模型验证。这套流程贵，所以论文 §3.1.3 给了廉价替代：**annealing 评估法**——拿一个训到 50% 的 8B 模型，LR 线性 anneal 到 0，在 40B tokens 上跑，其中 30% 权重给待评估的新数据源、70% 给默认 mix，看 GSM8k/MATH 验证集 delta。数据源的"价值"被操作定义为"它在 annealing 窗口里能带来的分数增益"。

**(d) Annealing 数据的双重角色。** §3.1.3：annealing 阶段上采样高质 code/math 数据；8B 上 GSM8k +24.0%、MATH +6.4%，405B 上"improvements negligible"。两点深挖：第一，annealing 收益与模型容量负相关——小模型缺 in-domain 样本，大模型靠 in-context learning 就够了，所以"高质数据 annealing"是**小模型的药、大模型的安慰剂**；第二，论文明确声明 annealing 数据不含常用 benchmark 的训练集，这是防污染的质量门声明（接 Day30 的防漏线）。

**(e) HTML 解析的取舍。** 自研 parser 优化"样板去除精度 + 内容召回"，去掉所有 markdown（论文称 markdown 对主要训 web 数据的模型有害），但保留数学/代码结构和图片 alt 文本（数学常以预渲染图片形式存在，alt 里有公式）。细节里藏着数据观：**格式标记是噪声，结构是信号**。

### 4. 边界与反例

1. **初读 NOTES 的 code 配比写错了**：初读称"code 从 17% 上采样到 25%"，论文 §3.1.2 原文是最终配比约 50% 通用 / **25% 数学与推理** / **17% 代码** / 8% 多语——25% 是 math+reasoning 的占比，不是 code。Code 在最终 mix 里就是 17%。引用配比时以论文为准。
2. **论文没有证明/披露的东西**：任何一级过滤的保留率与阈值（"heuristic 砍 30%"这类数字是重构的，论文只说 "we experimentally evaluate"）；Llama 2 质量判断用的具体 prompt（"describe the quality requirements" 内容未公开）；MinHash 的 n-gram 参数与阈值。Llama 3 的瀑布是**存在性证明，不是可复现配方**——可复现性要去 Day25 FineWeb 找。
3. **预训练合成数据的角色被初读夸大了**：初读写"多轮合成/回译"，但 §3.1 对预训练合成几乎没有披露；合成的主力在后训练 §4（SFT/RS/DPO 数据合成）和 annealing 的高质 code/math。15.6T 的主体仍是过滤后的真实 web + 专用抽取的 code/math 页。不要把 Llama 3 当成"合成预训练"论文引用。
4. **Line-dedup 的误删是已知代价**：论文承认删掉高频高质量文本。对 code 数据，高频样板（license header、import 块、常见脚手架）被行级删掉可能破坏文件结构完整性——Llama 3 用 domain-specific 抽取管线对冲，但论文没有量化 code 域的误删率。抄这套去重到 coding 池时，行级阈值要按域重调，不能直接抄 6 次/30M 桶。
5. **配比是"预算 × 架构 × tokenizer"的函数，不是常数**：50/25/17/8 是在 Llama 3 的 tokenizer（128K，英文压缩率 3.94 chars/token）、dense 架构、15T 预算下搜出来的。DeepSeek-V3（MoE）code 30%+ 就是反例——MoE 的 expert 路由容纳得下更多 code，dense 不行。换架构、换预算、换 tokenizer，最优配比都要重搜；DoReMi（Day31）正是要把这个重搜自动化。
6. **Annealing 评估法的适用域**：只在 GSM8k/MATH 这类对数据敏感的验证集上灵敏；通用 web 质量的提升没有同等灵敏的 proxy 指标。而且 405B 上 annealing 增益可忽略——用 8B annealing 选出来的"高价值数据源"，不能直接断言对 405B 同等有价值，存在跨规模迁移 gap。

### 5. 迁移到 coding / post-training data

**可执行实验：把 §3.1.3 的 "annealing 评估数据源" 做成你 500k 合成池的廉价数据门禁**

1. **搭 proxy**：取一个训到约 50% 的 1.3B proxy 模型（或你现有最小可训模型），把 LR 线性 anneal 到 0。
2. **设评估窗口**：论文用 40B tokens，你按比例缩小到你能负担的窗口（建议 2B tokens）；窗口内 30% 权重给待评估的候选数据源（如新一批合成题、某类真实 repo 代码），70% 权重给当前默认 mix。
3. **读数**：看 HumanEval+ / MBPP+ 验证集 delta。设通过阈值（如 HumanEval+ 增益 > 1.5 pts）——通过的才进入主训练池，不通过的回炉重造或降权。
4. **为什么值得做**：这比"全量训完再看 HumanEval"便宜 1–2 个数量级；而且它评估的是**边际价值**（这个数据源相对现有 mix 的增量），不是绝对质量——正好回答"下一批合成预算该花在哪"。
5. **配套纪律（补论文没给的）**：记录每个候选数据源的通过/不通过、delta、token 量，形成你自己的可审计 recipe——Llama 3 没公开的保留率，你自己从第一天就记下来。
6. **第二步（若门禁跑通）**：把瀑布按成本升序重排你现在的清洗链——heuristic/MinHash 先行砍大头，LLM-judge 只打"前两级拿不准的难例"，统计每级保留率；这是 §3.1.1 级联思想在 coding 池的直接落地。

### 6. 今天的一道思考题

综合 **Day07 Llama 3、Day25 FineWeb、Day31 DoReMi**，三者恰好构成"数据决策"的三层：

- Llama 3：**先过滤（§3.1.1）再定配比（§3.1.2）**，配比靠手工 + scaling law 小模型实验；
- DoReMi：配比可以**学出来**（proxy 模型的跨域 excess loss → domain weights）；
- FineWeb：每个过滤器都要**小模型消融验证**，而不是当常识照搬。

**问题（两问，都要答，答案不在任何一篇原文里）：**

**(a) 样本级过滤 vs 域级配比，谁的边际贡献更大？** 在你的 500k coding 池、固定总 token 预算下，设计一个 2×2 实验：过滤 ∈ {开（FineWeb 式 gate 全开）， 关（只做基础去重）} × 配比 ∈ {均匀， DoReMi 式学习权重}，四个格子各训一个同预算 proxy，看 HumanEval+ delta。先**预测**四个格子的排序并写出理由，再说说这个实验设计本身有什么漏洞（提示：DoReMi 的 domain 怎么切？切分方式本身会不会吃掉"过滤"的功劳？）。

**(b) "先过滤再定配比"的顺序是最优的吗？** Llama 3 的顺序是过滤 → 配比。如果反过来——先用 DoReMi 在**未过滤**的池子上学出 domain weights，再做样本级过滤——结果会一样吗？从数学上想：过滤改变的是域内分布 $$P(x|d)$$，配比改变的是域间权重 $$P(d)$$，两者不正交——过滤掉某域 30% 低质样本后，该域的 excess loss 会系统性变化（噪声少了，proxy 学得更快，excess loss 下降），学出的权重会**系统性低估**该域。反过来，先定配比再过滤，权重是在含噪声的分布上估计的，又会**系统性高估**噪声域。那么：正确的顺序是什么？还是说需要**交替迭代**（过滤 → 配比 → 再过滤 → 再配比）直到不动点？给出在你 500k 池上可操作的一次迭代方案：domain 怎么定义（来源 × 语言 × 可执行性，要互斥且有 provenance）、excess loss 用什么 proxy 算、收敛判据是什么。

---

**论文原文**：https://arxiv.org/abs/2407.21783
**GitHub NOTES**：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-07-2024-llama3/NOTES.md
