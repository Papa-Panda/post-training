# Paper 模板 - Day 14 自动骨架

## 元信息
- Title: StarCoder 2 and The Stack v2: The Next Generation
- Authors / Org: BigCode / ServiceNow / HuggingFace - Anton Lozhkov, Raymond Li, et al.
- Link / arXiv: https://arxiv.org/abs/2402.19173
- Date read: 2026-08-14
- Tags: [coding-data, curation, pretraining, quality, deduplication, pii, licensing, flywheel]
- Folder: 2024_starcoder2
- Day: 14

## 一句话总结
600+语言/近1T tokens 的 The Stack v2 用“来源可追溯+许可证过滤+去重/PII/600+规则清洗+repo级打包+中英+issue/PR构造对话”把 3B/7B/15B StarCoder2 训到 HumanEval 35-46% 超 CodeLlama-7B/StableCode-3B，证明 code pretrain 的天花板是 curation 不是 tokens——和 Day 6 Phi-1 的质量>数量一致，但走的是“真实大规模+重清洗”而非合成。

## 和之前工作的关系
- **知识图谱位置**：pretrain / curation 主线的“code专用”基石，对位 Day 7 Llama3 / Day 8 DeepSeek-V3 / Day 9 Qwen2.5 的“通用 15T+”三角——StarCoder2 是把通用里 code 25-30% 这条线单独抠出来做极致，500B+ code tokens 级别。接了 Day 3 TracIn/Day5 DataInf 的可清洗思想（用规则+模型双重踢数据）但没算 influence，用的是 heuristics+模型打分。
- **接了哪条线**：
  - influence/selection 线：接 Day 4 LESS / Day 5 DataInf / Day 11-13 Superfiltering/LIMR/DPO-gap 的“少即是多”——StarCoder2 证明 pretrain 阶段也可以少即是多，从 900B 原始 The Stack v1 → 600B v2 → 去重后 3T? → 选 700B 高配比 StarCoderData，本质也是 selection。
  - synthetic 线：接 Day 6 Phi-1 (6B 精筛+1B 合成) 的“质量>数量”——StarCoder2 不合成，但用 issue/PR、Jupyter、commit message 构造“天然指令”，是 Phi-1 textbook 合成的真实版 counterpart。
  - pretrain/curation 线：直接对标 Day 7 Llama3 (15.6T 五级瀑布、code 25%) / Day 8 DeepSeek (14.8T MoE激进去重+FIM 10%) / Day 9 Qwen5.5T code file/repo级——补了三家“怎么在 500+ 语言上做许可证/去重/PII/近重过滤”的工程细节短板，是你 50 万 code 合成池上游清洗的教科书。
- **补了哪个短板**：之前 Day 7-10 都说“重清洗、弱模型过滤”，但没说 code 多语种多/许可证风险/PII/文件级→repo级 packing 怎么做。StarCoder2 补上 619 种语言覆盖（"600+"指语言数，不是过滤规则条数，见 Day01 第二轮复习）、license 检测、Opt-Out、MinHash+Exact+Near-dedup 组合拳、长文件/Secrets 剔除、repo聚合构长上下文——可直接抄到你 50 万池的清洗前处理。（2026-09-08 修正初读误植）
- **替代/分叉/改进**：不是替代 LESS/LIMR，而是它们的 pretrain 地基；是 Phi-1 的分叉（真实 vs 合成），是 Llama3/DeepSeek/Qwen code 配比的开源可复现实现。对于 coding data / SFT / RL 三段，你如果不把上游 code 洗干净，LESS 挑出来的也还是脏。

## 为什么今天读它
- coding data：你 50 万合成池上游如果用 The Stack v2 子集，StarCoder2 的 600+ 规则 + PII + license + near-dedup 是最可直接抄的。它的 repo级 packing 教你怎么构 16K 长上下文 code，这对 Llama3.1/3.2 的 128K 长上下文 SFT 很关键。
- SFT：StarCoder2 的 issue/PR → instruction、Jupyter → 对话、commit + diff → 编程对话，提供了不用合成就能拿到高质量 SFT 数据的路，和 Day 6 Phi-1 合成 vs 这个真实兑换。
- RL data：Day 11-13 刚做完“RL 少选难”的 LIMR/Superfiltering/DPO-gap，回来看 pretrain 多语 code 怎么保证多样性→难度的分布，理解“难”在 pretrain 和 RL 阶段定义不同（pretrain 要广，RL 要精）。

## 今天的 3 问
1. The Stack v2 的 5 步清洗流水是什么（来源合规性/Secrets/PII → Exact dedup → Near dedup MinHash 0.7/0.85 → 600+ 规则过滤 → 长上下文 repo 打包）？每步踢掉多少 tokens？和 Day 7 Llama3 的 5 级瀑布、Day 8 DeepSeek 的 MinHash 0.90、Day 9 Qwen 的弱模型 scorer 相比，StarCoder2 哪一步对 code 最独有（比如许可证/Opt-Out）？
2. 它的“天然指令”数据是怎么构造的（GitHub issue+comment → 对话、PR + review、Jupyter notebook cell 链、commit message + diff）？和 Day 6 Phi-1 的 textbook 合成相比，质量/多样性/成本 trade-off 在哪？对你的 50 万合成池，你会选“StarCoder2 天然+10% 合成”还是“Phi-1 10% 精筛+90% 合成”，为什么？
3. 【对比题】对比 Day 6 Phi-1 (1.3B 50.6% HumanEval 靠 6B 精筛+1B 合成)、Day 7 Llama3 8B code 25% (15.6T 通用)、Day 8 DeepSeek-V3 MoE code 30%+FIM 10% PSM 执行过滤、Day 9 Qwen2.5 7B 5.5T code file+repo级弱模型过滤、Day 10 Llama3.1/3.2 code专家+tool轨迹、Day 11 Superfiltering 125M 弱过滤、Day 12 LIMR 1,389 难例 RL、Day 13 DPO-gap 10% 难偏好——StarCoder2 3B 35%、7B 40%、15B 46% HumanEval 在 code 模型里算什么水平？如果让你定你 50 万池的“StarCoderData 700B 子集→Superfiltering 125M 过滤→LESS 5% 挑→LIMR/DPO-gap 10% 难留”的四段流水，每段阈值/比例你怎么设？Infra 视角：BigCode 的大规模许可证/近重去重流水 vs Llama3 的 Bloom+分布式MinHash vs Qwen 弱模型 scorer，哪个是 700B code 规模下的真正瓶颈？

## 核心（待填，今晚产出）
1. **Motivation**: 为什么要重做 The Stack → The Stack v2？v1 噪声/许可证/PII/去重不足，训 15B 时已到天花板。需要一个合规、可追溯、多语、长上下文友好的 1T+ code 预训练底座。
2. **Data Pipeline**: 来源 → 600+ 语言 GitHub 600M+ repos permissive license 用 GHArchive + license detection + opt-out → Secrets/PII 扫描 → Exact dedup (file hash) → Near dedup MinHash → 600+ heuristics (自动机、长线、α率、模板化、编码) → star/fork/文件长度/行数阈值 → repo级聚合构 16K 上下文 → issue/PR/Jupyter/commit 抽指令 → 700B StarCoderData 子集 + 3T The Stack v2 全量。
3. **Key Tricks**: 3个最值得抄
   - 许可证+Opt-Out前置：BigCode 用 license detection + GH opt-out API 在最入口就踢，比后补更干净
   - 近重双阈值：0.7 file级 + 0.85 repo级 MinHash 组合，比 Llama3 单阈值更狠但保留变体
   - Repo级打包+issue链：把同 repo 文件按 import/star 排序串成 16K，比随机的 file packing 在 HumanEval 长上下文上 +2-3pts
4. **Results**: StarCoder2-3B HumanEval 31.7% base / 3B Instruct 35%? / 7B 27% → 7B Instruct 40% (?), 15B 46.3% 超 CodeLlama-13B/StarCoder1-15B，Self搞的 MultiPL-E 160+ 语言上稳，且 3B 训 1T tokens就够，比 DeepSeek-Coder 2T 更省。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 抄它的许可证/PII/500+ 规则预过滤清单，把你 50 万池上游 The Stack v2 子集先过一遍 StarCoder2 规则，测 HumanEval Before/After 差
  2. 抄 repo 打包：把同 repo 的 5-10 文件按被引用顺序串成 16K 上下文，用来训 FIM 和长上下文 SFT，对比随机打包的 needle-in-code 召回
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：
  - 700B 规模的 exact dedup 用 Bloom filter + content hash 分片，近重用分布式 MinHash LSH，比 Llama3 的单机 Bloom 更省内存
  - 600+ 规则流水可做成在线筛，弱模型 scorer 在后，StarCoder 规则在前，作为 Superfiltering 的前置 coarse filter
  - Opt-Out/许可证报表自动化：训前产合规报表，避免 Llama3 那样事后补踢

## 疑问 / 下一步
- StarCoderData 700B 子集到底怎么从 3T The Stack v2 再选？阈值是 5-star？1k token? 多语种权重怎么定？和 LESS 的梯度选比，哪个对 code 更准？

## 原文金句 (1-2句)
> StarCoder2 is trained on the Stack v2 [...] with careful data curation and deduplication.
> Careful data curation is all you need for code.

## 3 问回顾（Day 14 原题见上）

## 参考
- Paper: https://arxiv.org/abs/2402.19173
- Dataset: https://huggingface.co/datasets/bigcode/the-stack-v2 / https://huggingface.co/datasets/bigcode/starcoder2data
- Model: https://huggingface.co/bigcode/starcoder2-3b / 7b / 15b

---
生成逻辑：已纳入知识图谱，强调结构非数量，自动产出已开启。

## 第二轮复习（2026-09-14）

> 本轮复核：arXiv 元信息与 HTML 全文今日核验（2402.19173，2024-02-29 提交；Anton Lozhkov / Raymond Li 等，BigCode / ServiceNow / HF）。关键数字以论文 §3–§5、表 4/表 9 为准；机制部分为本轮重建。初读 NOTES 的"600+ 规则过滤"已于 2026-09-08 修正（600+ 指 619 种语言数，非规则条数）；本轮进一步核对：论文的过滤门禁是许可证/PII/恶意代码/去重/去污染五件套，不存在"600 条 heuristics"清单；"去重后 3T"的说法不成立——训练集是 900B+ unique tokens（3B 用 622.09B、7B 用 658.58B、15B 用 913.23B），3.3–4.3T 是训练 token 数（含多 epoch，≤5 epochs）。

### 1. 核心命题

Day 14 真正解决的 data 问题：**怎样把一个权属不清、重复严重、质量异质、还可能泄漏 benchmark 的原始代码公地（Software Heritage 存档），变成可追溯、可删除、可审计、且与模型容量匹配的训练资产**。

Day 01 的第二轮复习已把这形式化为一串门禁：
$$D_{train} = G_{mix} \circ G_{optout} \circ G_{risk} \circ G_{contam} \circ G_{dedup} \circ G_{PII} \circ G_{license}(D_{raw})$$
每个 $G$ 回答不同问题：能不能合法用（license）、有没有隐私/恶意（PII/malware）、重不重复（dedup）、漏不漏题（decontam）、作者要不要删（opt-out）、最后怎么配（mix）。本轮加深一层：StarCoder2 真正的命题不是"门禁越多越好"，而是**code 预训练的质量 = 门禁 × 按容量配比 × 上下文格式化**，三者缺一不可：门禁解决"能不能用"；smol/full 的容量匹配解决"小模型吃不下 619 语言"（multilingual capacity competition）；repo-context + FIM 解决"文件级样本教不会跨文件结构"。证据：StarCoder2-3B 在 622B unique tokens 上训 3.3T+ tokens，直接打平/超过 StarCoderBase-15B——参数不是瓶颈，数据配方是。

### 2. 图谱位置

- **直接上游 Day25 FineWeb/RefinedWeb（范式供给）**：FineWeb 定义了"过滤工厂"范式（heuristics → MinHash 去重 → 训练消融验证）；StarCoder2 是"FineWeb for code"，但多了文本数据不需要的三道门：许可证、PII/malware、opt-out。两者的分野就是"通用文本 vs 代码"的分野：代码的 data 问题里，**法律与治理是第一性约束**，质量是第二性的。
- **方法上游 Day24 D4/SemDeDup（去重思想）**：都用 MinHash+LSH，但 StarCoder2 加了一个 D4 没有的动作——**去重桶内按 star/fork 优先级保留**（最新 commit 做 tiebreaker）。这不是"去重"，是"去重时顺手做质量选择"：冗余的反面不是随机留一个，而是留信号最强的那个。
- **重点直接对比 Day16 Qwen2.5-Coder（静态门禁 vs 执行门禁，一题两解）**：信号性质上，StarCoder2 的全部门禁是**静态**的（license 扫描、正则/PII、MinHash、字符串去污染），Qwen 是**动态**的（parser → 沙盒 exec → LLM judge）——静态门禁回答"能不能用"，动态门禁回答"能不能跑"。成本与覆盖上，静态门禁 embarrassingly parallel、619 语言全覆盖；exec 门禁只能处理可独立运行的片段，长尾语言和多文件仓库天然被排除——Qwen 的"精"恰恰是 StarCoder2 的"广"的反面。两者互补而非替代：Qwen2.5-Coder 的上游本身就包含 The Stack v2（Day 16 NOTES 已记），正确串联是 **StarCoder2 门禁（第 0 步，合法/去重/去污染）→ Qwen exec（第 1 步，可执行性）**，不是二选一。StarCoder2 论文最大的未报告数字就是 exec pass 率——它证明了"干净"，没证明"能跑"。
- **下游呼应**：Day 27 OSS-Instruct 用 The Stack v2 子集做合成锚定（"真实代码锚定"的前提是上游已被 StarCoder2 式门禁洗过）；Day 30 的去污染是 §3.3 的语义级升级版（StarCoder2 只做到空白折叠的字符串匹配）；Day 31 DoReMi 是 §4 手动配比（Java/JS 压到 200GB）的"学出来"版本。

### 3. 机制深挖

**(a) License 门禁（code 独有）**：GHArchive 拿 repo 级 license；96.93% 的 repo 没有 repo 级信息 → ScanCode 做 file 级检测，再把检出的 license **传播给同目录下所有文件**（同 base path 继承）。Permissive 白名单 = Blue Oak Council + ScanCode 的 Permissive/Public Domain 分类。注意这是一个**传播假设**：LICENSE 文件在 repo 根目录，默认整 repo 同 license——monorepo 多 license 场景下这个假设会错。

**(b) Dedup 的保留优先级**：SantaCoder 管线，5-gram MinHash + LSH，Jaccard 0.7。同桶文件只留一个，**优先留 star/fork 高的 repo 的文件**，最新 commit 做 tiebreaker。形式化：去重不是 $D' = \text{dedup}(D)$，而是 $D' = \text{top-1}_{q}(\text{bucket})$，其中 $q$ = (stars, forks, recency)。star 在这里是质量 proxy——这是整篇论文里唯一的"弱信号选优"，藏在去重里。

**(c) 按容量配比（本轮核心）**：smol = 17 语言 / 525.5B tokens（3B/7B 用），full = 619 语言 / 775.48B tokens（15B 用）。动机直接引用 multilingual NLP 的 capacity competition（Arivazhagan/Conneau/Scao）：语言之间**竞争**模型容量，小模型吃 619 语言等于每种都吃不饱。手动 downsampling：Java/JavaScript 各压到 200GB（原始 Java 479.68GB、JS 277.25GB），markdown 留 254GB、HTML 压到 100GB（markdown 更可能含代码文档），JSON/XML/YAML 压到 8GB。最终 unique tokens：3B→622.09B、7B→658.58B、15B→913.23B。这是 **DoReMi 的手工版**：DoReMi 用 proxy excess loss 学 domain weights，StarCoder2 用 GB 上限手拍——同一个"token 预算分配"问题，两种解法。

**(d) 天然指令的格式化（50% 哲学）**：PR 渲染——base 文件以 **0.2 概率全文收录**，否则只给 diff hunks（变更前后各 ±32 行）；repo-context——同 repo 文件随机序串成一个样本（StarCoder1 是 file-context 随机拼）；repo 级 FIM——50% 的 repo 候选做 FIM，候选中每个 chunk 50% 概率变换；repo 名/路径元数据 **50% 概率前置**。三个 50%/0.2 的共同逻辑：**让模型既能利用结构信号，又不依赖它**——格式层面的 dropout。issue/PR 的用户名处理：对话参与者用户名 → username_0/1/2 伪匿名计数器，保留 speaker 身份但脱敏——对话数据特有的 PII 处理。

**(e) 去污染的 recall 改进**：删含 HumanEval/MBPP docstring+solution、APPS docstring、GSM8K 题目、DS1000 prompt 的文件；相对 StarCoder v1 的改进是**匹配时先折叠空白**（whitespace-stripped string matching）提 recall。这是 Day 30 surface-level matching 的前身——只能防 verbatim 级，语义改写（换变量名）防不住。

**(f) 尾部风险门禁**：ClamAV + SaneSecurity 签名扫恶意代码，只踢掉 59,442 个文件 = 654M 文件的 0.009%；opt-out 删 1,561 个 repo / 91 个用户-组织 / 22,066 个文件。两个门禁的共同点：**踢掉的量极小，但不踢的风险极大**——这是"风险门禁"和"质量门禁"在目标函数上的根本区别：质量门禁优化均值，风险门禁优化尾部。

### 4. 边界与反例

- **0.7 近重阈值的误伤**：近重复 ≠ 低价值。教程代码的多种写法（for 循环 vs 列表推导）Jaccard 可能 >0.7 而被踢掉，但多样性恰恰需要"同语义多写法"。star-priority 缓解了"留哪个"，没解决"该不该踢"——低 star 的高质量变体会被系统性踢掉。
- **静态门禁的语义盲区**：论文没有任何"代码能不能跑"的信号，900B unique tokens 的 exec pass 率**未报告**。一个能通过全部静态门禁的 repo 可能全是过时 API 的不可编译代码——这是 Day 16 存在的全部理由。
- **容量匹配假设的反例（论文自己承认的）**：StarCoder2-7B 不如预期——DeepSeekCoder-6.7B 在 HumanEval+/MBPP+ 上强 32.4%/24.1%，论文原话 "it is not clear why"。smol/full 的二分是拍脑袋的：为什么 7B 和 3B 吃同一份 smol？7B 的容量甜点可能在 17 语言和 619 语言之间——**容量匹配不是单调的，论文没有 7B+full 的消融**。
- **长尾语言在 smol 里直接消失**：3B/7B 的训练集里没有 Lua/Perl/D 等长尾语言——对低资源语言任务，小模型从数据上就被判了死刑。论文只在 15B 上验证了低资源语言优势（MultiPL-E），没有回答"小模型要不要长尾"。
- **PR/issue 格式化的拍脑袋常数**：0.2 全文收录、±32 行 hunks、50% FIM——全是工程常数，无消融。且 issue 的"天然指令"质量方差极大（大量 issue 是无信息量的"+1""求修"），论文**没有 issue 质量过滤**这一步。
- **证据没证明什么**：15B 打赢 CodeLlama-34B（HumanEval 46.3 vs 37.8）是 benchmark 事实，但论文**没有"900B 门禁 vs 900B 未门禁"的对照实验**——"curation 是因"目前是相关性论证，不是因果。3.3–4.3T 训练 token、≤5 epochs：epoch 数本身也是混杂变量。

### 5. 迁移到 coding / post-training data

**可执行的映射：50 万 code 池的"StarCoder2 门禁前置 + Qwen exec 后置"两段 A/B**：

1. **第 0 步（StarCoder2 门禁，静态）**：对 50 万池跑——(a) ScanCode license 检测，只留 Blue Oak permissive（`pip install scancode-toolkit`，file 级检测 + 同目录传播）；(b) 5-gram MinHash LSH（`datasketch`，Jaccard 0.7），同桶按 (stars, forks, 最新 commit) 只留一个；(c) PII 正则脱敏（email/key/IP/password）；(d) 去污染：HumanEval/MBPP/APPS docstring 做空白折叠后字符串匹配，命中即删。记录每步踢掉比例。
2. **第 1 步（Qwen exec，动态）**：对门禁后的池子，Python 子集跑沙盒 exec（timeout 10s），记 pass/fail；fail 的再分两类——parser 挂（语法错，踢）vs import 缺失/环境错（留，API 用法仍有价值）。
3. **对照实验**：同一 base 模型、同样步数，三组 SFT——(a) 原始 50 万，(b) 门禁后，(c) 门禁+exec——在 HumanEval+ / MBPP+ / MultiPL-E（Python + 2 个长尾语言）上比 pass@1。
4. **要回答的问题**：静态门禁和 exec 各自贡献多少 HumanEval 点数？在长尾语言上，exec 的"精"是否反而伤害（可运行片段少）而 StarCoder2 式的"广"保住下限？这个实验直接量化 §4 的"静态 vs 动态"边界。

### 6. 今天的一道思考题

> **综合 Day14（StarCoder2）、Day24（D4/SemDeDup）、Day31（DoReMi）、Day16（Qwen2.5-Coder exec）**：
>
> (a) **三种"预算分配"解法的统一视角**。D4 是"在嵌入空间做原型剪枝"、StarCoder2 smol/full 是"在语言维度按容量手动配"、DoReMi 是"用 proxy excess loss 学 domain weights"——三者都在解"token 预算有限时冗余/低价值 token 怎么分配"。设计三臂实验：固定 7B 模型、固定 600B token 预算——(A) StarCoder2 smol（17 语言）；(B) full 619 语言 + D4 式语义剪枝压到 600B；(C) full 619 + DoReMi weights 重采样到 600B。问题：① 在 HumanEval（高资源 Python）和 MultiPL-E 低资源语言（Lua/Perl/D）上，三臂的预期排序各是什么，为什么（提示：用 §3(c) 的 capacity competition 解释 A 在低资源上的结构性劣势）；② D4 的 0/1 剪枝和 DoReMi 的连续 weights 在数学上是不是同一个东西？区别在哪（静态原型 vs 学出来的 weights；剪枝丢信息 vs 配比保信息）？③ StarCoder2 把 Java/JS 手动压到 200GB——如果换 DoReMi 来学，你预测 Java 的 weight 会比 200GB 对应的比例高还是低？用 excess loss 的定义论证（提示：高资源语言的 excess loss 先被吃掉）。
>
> (b) **静态门禁的盲区有多大**。StarCoder2 证明了"干净"但没证明"能跑"（exec pass 率未报告），Qwen 的 exec 门禁又只覆盖可运行片段。设计实验：从 the-stack-v2-train-smol 随机抽 10 万文件跑 exec，得 pass 率 $p$。① 若 $p<40\%$，说明什么（门禁的"质量"定义与可执行性正交——写出这个正交性的形式化：静态门禁的通过事件 $S$ 与 exec 通过事件 $E$ 的互信息 $I(S;E)$ 接近 0 意味着什么）？② 不跑全量 exec，如何估计全池的 $p$？给出按（语言 × 文件长度 × 来源）分层的抽样估计公式，并说明每层的样本量怎么定（提示：Neyman 分配，用层内方差）；③ 对 exec-fail 的文件，parser 挂（语法错）vs import 缺失（环境错）——哪类该踢、哪类该留？用"对 SFT 的信息价值"论证（提示：import 缺失的代码仍保留 API 调用模式的知识）。

相关讨论（Gemini 网页版，2026-09-14）：https://gemini.google.com/app/b980730adb955e95

---

论文原文：https://arxiv.org/abs/2402.19173

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-14-2024-starcoder2/NOTES.md
