# Paper 模板 - Day 24

> 自动生成骨架 2026-08-24，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: D4: Improving LLM Pretraining via Document De-Duplication and Diversification
- Authors / Org: Tirumala et al. / Meta AI
- Link / arXiv: https://arxiv.org/abs/2308.12284
- Date read: 2026-08-24
- Tags: [pretraining, data-curation, deduplication, diversity, quality, semantic-dedup, coding-data]
- Folder: day-24-2023-semdedup-d4
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-24-2023-semdedup-d4

## 一句话总结
D4 把语义近重复去除（SemDeDup）与表示空间中的原型式多样化剪枝串成预训练数据管线，在压缩冗余语料的同时保留更广覆盖，补上 Day19 Vendi Score“会量多样性但还不会规模化选数”的工程短板。

## 和之前工作的关系

> 知识图谱位置：预训练去重源头；Day07 Llama3 / Day14 StarCoder2 的大规模过滤与去重实践，向上追溯到可复用的语义去重 + 多样化选择方法；并把 Day19 Vendi Score 的多样性概念落成实际数据剪枝。

- **接了哪条线：**接 Day19 Vendi Score → Day20 DEITA 的 diversity / redundancy 线，也回接 Day07 Llama3、Day14 StarCoder2、Day16 Qwen2.5-Coder 的预训练与 code data 清洗线。
- **补了哪个短板：**此前的管线多写 MinHash、规则过滤或一句“去重”，缺少对语义近重复、簇内保留策略和覆盖损失的系统拆解；D4 把“删重复”和“保多样”明确分成两步。
- **替代 / 分叉 / 改进：**它不是替代 MinHash / exact dedup，而是在词面去重后增加 semantic dedup，再用 diversity-aware pruning 避免只保留高密度主流样本；这是质量过滤之外的正交改进。
- **对之前 Day X 的直接对比：**vs Day19 Vendi Score，Vendi 给出数据集多样性的度量标尺，D4 给出可执行的删样本流程；vs Day20 DEITA，DEITA 在 SFT 池中按质量×复杂度×多样性选 6k，D4 面向预训练规模，重点是近重复消除与覆盖保持。

## 为什么今天读它

Day21→23 已完成“合成扩量 → 复杂度演化 → 少量高质策展”，今天转到数据池进入训练前的第一道规模化门禁：先把重复信息删掉，又不能把长尾覆盖一起删掉。对 coding data，可用文件/函数表示做语义近重复聚类，并在每簇中保留质量更高、许可与测试更完整的代表；对 SFT / RL data，也可把同模板改写和等价题目视为语义簇，控制有效样本数与任务覆盖。本文只研究数据去重、选择与覆盖，不展开训练算法。

## 今天的 3 问
1. D4 如何定义并串联 lexical dedup、SemDeDup 和 prototype-based diversification？每一步删除哪类冗余，簇内代表样本按什么数据标准保留？
2. 对比 Day19 Vendi Score 与 Day20 DEITA：D4 的“多样化剪枝”是在优化一个可量化的 diversity 指标，还是依赖表示空间启发式？三者如何组合成“质量门禁 → 去重 → 覆盖审计”的流水线？
3. 迁移到 coding / SFT / RL data 时，embedding 选型、相似度阈值和簇粒度如何避免误删“表面相似但边界条件不同”的样本？该用哪些覆盖与下游评测验证去重没有伤到长尾？

## 核心
1. **Motivation**: [待读后填写] 大规模预训练语料中的重复与高密度主题如何浪费计算、放大记忆，并挤压长尾覆盖？
2. **Data Pipeline**: [待读后填写] 文档表示 → 聚类/近邻 → 语义近重复删除 → 原型式多样化剪枝 → 数据规模与覆盖审计 → 进入预训练。
3. **Key Tricks**: [待读后填写] 记录 embedding、聚类粒度、相似度阈值、簇内保留规则、删除比例，以及不同语料域是否使用不同门槛；不展开训练算法。
4. **Results**: [待读后填写] 只记录数据压缩率、重复率/多样性变化、token/compute 节省和 downstream 对照，不展开 optimizer 等算法细节。

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
- 全程只写数据：curation / selection / quality / diversity / complexity / synthetic / execution-filter；算法只一句带过

> 自动化：reading-log 已追加 / commit 由本次自动化推送 / ai data sheet 由本次自动化同步

## 第二轮复习（2026-09-24）

### 元信息修正
- D4 全称是 **Document De-Duplication and Diversification**（Tirumala et al. / Meta AI，arXiv 2308.12284，2023-08-23），不是 data-diet 式的四个 D。
- 两个组件各有出处：**SemDeDup** 来自 Abbas et al. 2023（另一团队），**SSL Prototypes** 的 prototypicality 概念来自 Sorscher et al. 2022 "Beyond neural scaling laws" 的数据剪枝工作；D4 的贡献是"先去重再重聚类"的串联顺序。
- 文本实验的 embedding 是 **OPT-125M 最后一层表示**（跨数据集的现成模型表示可直接用，Abbas et al. 已验证换 CLIP 也几乎不掉点）；k-means 用球面 k-means（Faiss，单 GPU 可跑）。
- 评测协议：消融用 1.3B OPT / 40B tokens，规模验证到 6.7B；基线数据是 CCdedup；选择率固定 $R_{dedup} = 0.75$ ，只扫 $R_{proto}$ 。
- 初读 3 问之"簇内按什么标准保留代表"：SemDeDup 默认从每组重复对里保留**与簇质心余弦相似度最低**的那一个；三策略（留远/随机/留近）下游差异可忽略（附录 E.3）。

### 一句话总结
D4 的真正洞察不是"去重+多样化"的字面拼接，而是**重复会劫持聚类**：原始数据的 k-means 簇是 duplicate-driven 的，质心被重复模板钉死，所以必须先 SemDeDup 把重复逐出、re-cluster 让质心回到主题中心，再用 SSL Prototypes 剪掉新质心周围的密集区—— ， $R = R_{dedup} \times R_{proto}$ ，在 6.7B / 16 任务上换来约 20% 训练效率增益与最高 2% 的下游平均精度提升。

### 和之前工作的关系
- **vs Day19 Vendi Score（直接对比）**：Vendi 是"标尺"——kernel 特征值熵 $VS_k(D) = \exp(H(\bar{\lambda}))$ ，有有效样本数/重复敏感/可分解三公理的可微多样性度量；D4 是"流程"——可执行的删样本两步走。但关键欠账：D4 全程**从不显式优化任何多样性标量**，"diversification" 只是"剪质心近邻"的密度启发式，原文没有任何"精选前后 Vendi 分数对比"的消融。也就是说，D4 借用了多样性的修辞，却没有用 Day19 的尺子量过自己。
- **vs Day25 FineWeb（直接对比）**：FineWeb 的去重在词面层（MinHash / exact / URL 去重 + 启发式瀑布），D4 在嵌入语义层；两者正交可串联——FineWeb 式"词面瀑布"在前，D4 式"语义去重+覆盖审计"在后。Day07 Llama3 闭源五级过滤里语义去重只是一笔带过，D4 是它的方法学注脚。
- **vs Day20 DEITA（机制同构）**：DEITA 的 Repr Filter 在 SFT 池里做近邻去重（多样性 = 别选近邻），D4 的 SSL Prototypes 在预训练池里剪质心近邻（多样性 = 稀释模式、保长尾）。同一个几何直觉，两个规模级。
- **接线**：Day19（度量）→ Day24（工程流程）→ Day25（端到端瀑布）；回接 Day07 / Day14 的预训练去重管线；前接 Day30 防污染——同一套 embedding 近邻机器，D4 用来删训练集内重复，Day30 用来检 train–eval 泄漏。

### 核心（动机 + 机制深挖）
- **动机**："单 epoch 扫尽可能多 web token" 的收益随规模递减；更深的问题是重复**污染后续一切基于簇的数据操作**——质心落在重复模板堆里，任何"按簇剪枝"剪的都是重复的形状而非主题的分布。D4 把"先驱逐重复"变成多样化的前置条件。
- **Step 1 — SemDeDup（删语义近重复）**：文档 embedding → 球面 k-means（簇数 5 万量级）→ 簇内两两余弦相似度 $> 1 - \epsilon$ 判为重复组，每组只留 1 个（默认留离质心最远者）； $\epsilon$ 不手调，由目标保留率 $R_{dedup}$ 反推。这一步把"模板式重复"从表示空间里连根拔掉。
- **Step 2 — re-cluster（关键顺序）**：在去重后的数据集 $D^{\prime}$ 上重做 k-means。原文 Figure 7 的证据：跳过这步直接在旧簇上做原型剪枝，效果显著变差——因为旧质心仍被残余重复牵引。
- **Step 3 — SSL Prototypes（稀释密集区）**：prototypicality 定义为到新质心的距离 $d(x) = \|e_x - c_k\|$ （或等价余弦距离）；按 $d(x)$ 从小到大剪掉 $1 - R_{proto}$ 。附录 A.4 的定性检查：质心近邻多为语义冗余的模板（如固定格式的 SEO 文本），k-means 质心天然落在稠密区，所以"剪近邻" = "剪模板"。注意这与 Sorscher 原文"保留原型（简单样本）最优"的结论方向相反——D4 要的是覆盖，不是拟合。
- **对比协议（方法论上最值得抄的一点）**：固定 token 预算—— $R$ 每降一档，源数据集等比放大（选 1/4 就用 4 倍源数据），1.3B / 40B tokens 下比较。这样"精选 vs 随机"比的纯粹是**选择质量**，而不是"谁见过更多 token"。很多数据论文的增益其实混杂了数据量，D4 把这个混杂钉死了。
- **反直觉结果**：精选子集多 epoch 重复训练**稳定打赢**基线，而随机子集重复则打输。重复的边际价值 = 分布质量 × 重复次数——"不要重复数据"的惯例只在"重复的是随机 web  sludge"时成立。

### 边界（何时失效 / 证据没证明什么）
- **embedding 即偏见**：语义相似度完全继承 OPT-125M 表示的偏见。搬到 coding：同模板、不同边界条件的代码（如换变量名的等价解 vs 边界条件不同的两个实现）在通用文本 embedding 下可能挤进同一个 $\epsilon$ 球——**误删的方向正是高价值的长尾变体**。阈值全局一刀切，不同学科/代码域密度不同。
- **"模板 ≠ 低质"没有消融**：SSL Prototypes 剪掉的质心近邻里，混有多少高质量规范样本（如标准库文档、canonical 解法模板）？原文只给了定性"多为冗余模板"，没有"被剪掉的近邻里好样本占比"的量化。这是 D4 证据链里最大的暗角。
- **平均数掩盖方差**：16 任务平均 +2%，但任务间方差大（附录 Figure A3），某些任务零增益甚至为负——"多样化"对所有下游一视同仁的假设不成立。
- **先付费后精选**：embedding + 两次全量 k-means 是离线一次性成本；池子太小（< 目标选择量的 4 倍）时， $R$ 降档协议玩不起来，小团队更多是"有多少用多少"。
- **Vendi 欠账**（见上）：没有精选前后的多样性标量对比，"diversification" 是宣称不是测量。

### 迁移到 coding / post-training data（一个具体可执行的映射）
- **场景**：你的 coding SFT 池（如 OSS-Instruct 式合成的 ~75k 条），模板改写泛滥（同一题换变量名/换语言的等价样本）。
- **映射（照抄 D4 三步，但改两处）**：
  1. 表示换成 code embedding（UnixCoder / CodeSage 类；退而求其次用 AST 归一化后的 token embedding），球面 k-means 聚类；簇内余弦 $> 0.95$ 判同模板组，每组只留 1 条——**保留规则改成"留测试覆盖最全者"**，不用 D4 的"留离质心最远"，因为 code 场景质心近邻可能是 canonical 解法模板，盲目剪会伤规范解。
  2. 去重后重聚类，剪新质心近邻 10–20% 作为"模板稀释"对照组。
  3. **验收 = 补上 D4 欠的消融**：固定 token 预算，精选子集 vs 随机子集训 1B 小模型，看 HumanEval / MBPP delta，**同时报告两边精选前后的 Vendi 分数**——一次实验同时验证 D4 的宣称和 Day19 的尺子。
- **红线**： $\epsilon$ 按"域"分档（算法题/系统代码/配置文件密度不同），不要全局一个阈值；被剪样本落盘留档，每月抽 50 条人工复核"模板 vs 规范解"的误伤率。

### 思考题
- **(a) 综合 Day24 + Day19**：D4 声称 diversification 却从未报告 Vendi。设计一个 200 GPU-hour 内可跑的验证：固定 $R = 0.25$ ，比较 D4 精选子集 vs 同比例随机子集——用同一 kernel 算双方的 Vendi Score，再各训一个 1B 模型（10B tokens）看下游 delta。若出现"Vendi 更高但下游不涨"，说明了多样性度量与下游之间的什么关系？进一步：你能构造一个 Vendi 极高但训练有害的子集吗？（提示：往池子里掺与任务正交的噪声域。）
- **(b) 综合 Day24 + Day30**：同一套"embedding 近邻"机器，D4 用来删训练集内重复，Day30 用来检 train–eval 泄漏。假设你的语义去重阈值 $\epsilon_{dedup}$ 设得比防污染阈值 $\epsilon_{decon}$ 更激进，会观测到什么后果？——与 HumanEval 语义相近但合法的训练样本被成片删掉，表现为"去污染分数虚高"：防漏指标变好不是因为模型更干净，而是训练分布被掏空了一块。请设计一个对照实验，区分"真干净"与"分布被掏空"。（提示：固定评测集，比较激进/保守两档 $\epsilon_{dedup}$ 下，被删样本中"与 benchmark 近邻" vs "与 benchmark 远但高质量"的比例，以及两档的下游 delta。）

相关讨论（Gemini 网页版，2026-09-24）：https://gemini.google.com/app/b10432c02e770db5
