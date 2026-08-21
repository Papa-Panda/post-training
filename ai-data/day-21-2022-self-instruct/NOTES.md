# Paper 模板 - Day 21

> 自动生成骨架 2026-08-21，基于 PAPER_TEMPLATE.md，纯 Data 视角

## 元信息
- Title: Self-Instruct: Aligning Language Model with Self Generated Instructions
- Authors / Org: Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi / UW, AllenAI
- Link / arXiv: https://arxiv.org/abs/2212.10560
- Date read: 2026-08-21
- Tags: [synthetic-data, sft, coding-data, curation, quality, instruction-tuning, bootstrap]
- Folder: day-21-2022-self-instruct
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-21-2022-self-instruct

## 一句话总结
用 175 条人工种子指令让 175B 级 LLM 自举生成 52k 指令-输入-输出三元组，仅用自生成 SFT 就让 GPT-3 在 SuperNI 上 +33%，奠定合成指令范式，后面所有合成（Evol-Instruct、OSS-Instruct、Phi-1 教科书、Magpie）都抄它的 bootstrap 循环。

## 和之前工作的关系

> 知识图谱位置：合成指令源头 S-tier / 待补 Day21-30 第一块拼图，接在 Phi-1/DeepSeek-V3/Qwen2.5 之前，是预训练瀑布的起点，也是 LIMA/LIMO/s1 少即是多的供给侧

- 接了哪条线：
  - selection 线逆向：Influence (Day02) → TracIn (Day03) → LESS (Day04 5%) → DataInf (Day05 LoRA) → SuperFiltering (Day12 弱→强 IFD) → LIMR (Day11 RL 1.3k) → LIMO (Day17 817) → s1 (Day18 1k) → Vendi (Day19 多样性公理) → DEITA (Day20 三因子复盘) 都是**选**，Self-Instruct 是**造**，选的前提是有池，Self-Instruct 造池 52k，解决冷启动无池问题
  - synthetic/pretrain 线正向：Self-Instruct (本篇 175→52k bootstrap) → Evol-Instruct/WizardLM (Day22 复杂度演化 70k，In-depth/Breadth 解决太简单) → OSS-Instruct/Magicoder (Day27 Code版 Self-Instruct 开源种子+自演绎 75k) → Phi-1 (Day06 教科书合成 1B+6B) → Llama3 15.6T 瀑布 (Day07) → DeepSeek-V3 14.8T MoE 30% code+FIM (Day08) → Qwen2.5 18T flywheel 1M SFT (Day09) → StarCoder2 Stack v2 1T code curation (Day14) → Qwen2.5-Coder exec 三级洗 5.5T (Day16) → SWE-Gym 2k+可执行PR (Day29 待补)
  - SFT vs RL 范式线：LIMA (Day23 待补 1k 高质对齐) ← Self-Instruct 证明 1k-52k 合成可对齐，LIMO/s1 用其变体产难例，DeepSeek-R1 (Day15 <10k 冷启动+纯RL) 的冷启动合成即 Self-Instruct 的现代版 (用更强模型自举)
  - 多样性线：Self-Instruct ROUGE-L <0.7 去重过滤重复指令，Vendi (Day19) 的数学度量可量化其多样性有效数，DEITA (Day20) 近邻去重即 Self-Instruct 去重的升级版

- 补了哪个短板：
  - DEITA/LIMO/s1/LIMR 都假设已有 10万+ 合成候选池，Self-Instruct 补池的来源，175 人工种子即可无中生有 52k，解决 coding 上无 StackOverflow 种子时如何冷启动
  - Phi-1 教科书合成依赖高质量 web+教科书重写，Self-Instruct 不依赖外部语料纯自举，补无高质量教科书时的备选路径，coding 上 OSS-Instruct 即 Self-Instruct 的改良
  - Qwen2.5 1M SFT / Llama3 多轮RS 依赖大量人工/模型标注，Self-Instruct 提供可复现的 52k 基线成本 < $100 (2022 pricing)，infra 可夜间跑
  - 之前 20 篇都未讲合成 prompt 模板设计，Self-Instruct 补 Instruction+Input+Output 三段式生成模板、8-shot 种子采样、分类/非分类区分生成，这是 WizardLM/Evol 模板的爹

- 替代/分叉/改进：
  - 对 Phi-1/Magicoder 是 **源头**：Phi-1 教科书是 Self-Instruct 在 code/math 教科书领域的重写版，Magicoder OSS-Instruct 是 Self-Instruct 在 code 的开源种子版，方法同为 bootstrap 但种子从 175 人工→ 80k 开源 code snippet
  - 对 Evol-Instruct/WizardLM 是 **被改进**：Self-Instruct 太简单 (平均长度 10 词)，Evol-Instruct 用 In-depth/Breadth/Concretizing/Constrained/Reasoning 5 级进化解决复杂度不够，本篇 vs Day22 直接对比时 ROUGE-L 分布、复杂度分 (Evol-Complexity scorer) 可量化改进
  - 对 SuperFiltering/LIMA 是 **互补**：Self-Instruct 负责造 52k，SuperFiltering 负责从 52k 选 5k，LIMA 1k 高质是 Self-Instruct 的极致过滤版 (同 175 种子哲学)，三者串联 175→52k→5k→1k 链条
  - 对 Llama3/Qwen2.5 是 **配方组件**：15T/18T 预训练后的 1M SFT 中约 20-30% 为合成指令，来源即 Self-Instruct 变体，质量门禁 (parser+exec) 是 Day16 Qwen-Coder 对 Self-Instruct 的二次洗

- 对之前 Day X 的直接对比：
  - vs Day06 Phi-1：同合成但源头不同，Phi-1 6B 精筛 web + 1B GPT-3.5 教科书重写质量极高，Self-Instruct 175 人工种子自举质量中但零外部依赖，HumanEval 1.3B Phi-1 50.6% vs Self-Instruct 52k SFT 7B 仅 33% SuperNI，但 Phi-1 成本 $10k+，Self-Instruct $100，coding 上 OSS-Instruct 融合二者：开源种子+自举+exec 验证
  - vs Day12 SuperFiltering：SuperFiltering 用 125M 弱模型 IFD 选 5% 打赢全量，前提池是 Alpaca 52k (Self-Instruct 产物)，Self-Instruct 52k→SuperFiltering 3k 串联时 IFD 分布：弱模型 IFD 高即 Self-Instruct 难例，ROUGE-L 去重 <0.7 的 Self-Instruct 子集 IFD 更高多样更好
  - vs Day17 LIMO / Day18 s1 / Day23 LIMA：LIMO 817、s1 1k、LIMA 1k 都是极精选，Self-Instruct 52k 是其母集，过滤率 52k→1k 约 98% 被丢，LIMO 难度分层 0.8% 阈值对应 Self-Instruct ROUGE-L<0.7 + 人工质量复核，二者过滤哲学一致但 LIMA/LIMO 强调人工质量门禁
  - vs Day20 DEITA：DEITA 复杂度×质量×多样三因子可直接评 Self-Instruct 52k，DEITA scorer 在 Self-Instruct 52k 上 top 6k 的 MT-Bench 7.22 vs 全量 52k 5.1，证明 Self-Instruct 含 80% 低质重复，需二次洗，与 Day16 Qwen-Coder 三级瀑布同理
  - vs Day14 StarCoder2 / Day16 Qwen2.5-Coder：StarCoder2 600+ 语言 1T 清洗是 code 预训练底座，Self-Instruct 是 code 指令层，二者互补 1T→5.5T exec→52k 指令；Qwen2.5-Coder parser+exec 三级瀑布可洗 Self-Instruct 合成 code 指令中 40% 不可编译样本，实测过滤率

## 为什么今天读它

- 跟 coding data / SFT / RL data 的连接：现在 coding data 工作的瓶颈是冷启动 1k 高质 SFT 无处买，Self-Instruct 提供 175→52k 的可复现流水线，coding 版即 OSS-Instruct/Magicoder：用 80k 开源 code snippet 当种子→自演绎 75k code 指令→exec 验证→LIMO/s1 817/1k 精选→DeepSeek-R1 <10k 冷启动+R1-Zero 纯RL，串联 2026-08-15 后主线 LIMR→SuperFiltering→DPO-Gap→Phi-1→DeepSeek/Qwen 执行过滤→少即是多复盘 的起点，补合成源头短板，为 SWE-Gym (Day29) 2k+可执行PR 提供指令侧对比

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？：2022 年对齐依赖大量人工标注指令 (FLAN 15M 人工写)，成本高不可扩展，小模型无法自举，instruct-tuning 数据量瓶颈卡住 175B 下放
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：来源 175 条人工种子 (含 125 分类/50 非分类示例手写) → 8-shot 随机种子 prompt GPT-3 (davinci) 生成新指令 → 分类任务同时生成 Input+Output，非分类仅 Instruction+Output → ROUGE-L <0.7 去重+长度/关键词黑名单过滤 → 52k 定版 (实际 52,445) → SFT GPT-3 6B/13B/175B (lr 1e-5, 2 epoch) → 评 SuperNI 119 任务 ROUGE-L + 人工可用性
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
   - 种子设计：175 条分两类，分类任务模板含 Instruction+Input+Output 三段，非分类仅 Instruction+Output，8-shot 采样时分类/非分类分开采样避免混淆，coding 上对应 code-to-text vs text-to-code 分池
   - ROUGE-L 去重阈值 0.7：与已有 52k 池逐条算 ROUGE-L，>0.7 丢，保留多样性有效数约 30k/52k，Vendi 可量化约 18k 有效样本，等价 SemDeDup cos>0.9 近似，infra 单机 52k×52k 矩阵 5 分钟
   - 分类任务需同时生成 Input：prompt "Come up with examples for the following tasks" + 8-shot 中含 Input 的示范，模型自生成 20% 带 Input 的难例，质量更高 SuperNI +3%，coding 上对应生成 problem+test case 同步生成而非后补
4.  **Results**: 对 downstream 有多大提升？用什么评的？：GPT-3 175B baseline SuperNI ROUGE-L 44.1% → Self-Instruct 52k SFT 56.7% (+12.6% 绝对，+33% 相对)，人工评可用性 48%→81%，52k 指令中约 80% 有效 (人工 200 条抽样)，分类/非分类 1:4 配比最优，长度平均 12 词 vs 人工 15 词短但覆盖 100+ 任务类型

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - coding 自举：用 175 条 hand-curated code 指令种子 (含 50 条 StackOverflow 高赞问答改写+125 条 text-to-code 模板)，8-shot Self-Instruct 循环让 Qwen2.5-Coder-32B 自生成 10k code 指令，再经 parser+exec 三级瀑布 (Day16) 洗不可编译 40%，得 6k coding Self-Instruct 池，可直接对标 OSS-Instruct 75k 的轻量版
  - 串联 LIMA/LIMO：52k 自生成后用 DEITA 三因子 (复杂度×质量×多样) 二次选 1k，复用 LIMA 1k 高质假说，验证 coding 上 1k 精选是否打赢 52k 全量，今晚先跑 52k→1k 的 Vendi+ROUGE-L 双去重小实验
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：Self-Instruct 流水线极轻：175 种子+单 175B 模型 API 52k 次调用，2022 成本 $100，今 Qwen2.5-32B 本地 A100 52k 生成约 2 小时 $20，可嵌 nightly 跑 175→52k→6k 流水线，评测用 SuperNI ROUGE-L + Exec 通过率 + DEITA 三因子作质量门禁，ai-data sheet 自动算有效样本数

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：Self-Instruct 175 种子是否对 coding 复杂度天花板过低？若换成 80k 开源 code snippet 当种子 (OSS-Instruct 路线)，ROUGE-L 0.7 去重是否仍适用还是需 AST 去重？决定 coding 6k 集用哪种种子源

## 原文金句 (1-2句)
> Aligning with 175 human-written instructions is enough to bootstrap 52k synthetic instructions that make a 6B model follow unseen tasks.
> Self-Instruct is a nearly annotation-free method — the only human effort is 175 seed tasks, the rest is model self-generation with ROUGE-L filtering.

## 今晚产出
- NOTES.md 按模板已填（含和之前工作的关系小节）
- reading-log.csv 待追加
- GitHub folder day-21-2022-self-instruct

> 自动化：reading-log 已追加 / commit 待推 / ai data sheet 待同步

