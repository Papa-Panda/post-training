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

