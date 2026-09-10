# DeepSeek-V3 Data - MoE 685B Open Recipe

## 元信息
- Title: DeepSeek-V3 Technical Report
- Authors / Org: DeepSeek-AI
- Link / arXiv: https://arxiv.org/abs/2412.19437
- Date read: 2026-08-11
- Tags: [coding-data, pretraining, curation, moe, quality, scaling, synthetic-data, flywheel]

## 一句话总结
用 14.8T MoE 专用管线（提高数学与代码样本占比——论文未给数字，FIM 10% PSM，去冗余保多样——论文未披露方法），训 671B MoE（37B激活）对标 Llama 3 405B；论文未做数据消融，"更激进、更干净"是存在性层面的解读，不可引为因果结论。（2026-09-08 修正初读推断："code 30%+"等非论文事实，见第二轮复习 §4）

## 和之前工作的关系
这篇在知识结构里是 Llama 3 Day 7 的同级对标（pretrain/scaling 线），不是延续。

- **接 Llama 3 15T 的坑**：同样起于 Web+Code+多语种；DeepSeek-V3 强调去冗余保多样（论文未披露去重方法与阈值）、提高数学与代码样本占比（论文未给数字），pack 不做 cross-sample attention 保 document integrity【3762324244574975666†L64-L67】。（2026-09-08 修正："MinHash 0.85→0.90 / code 30%+"为初读推断，非论文事实；另 Llama 3 code 配比为 17%，25% 为数学与推理，见 Day07 第二轮复习）
- **对 LESS Day 4 的呼应**：MoE 容量大对数据质量更敏感；论文强调"去冗余保多样"（未披露具体方法）。（2026-09-08 修正："20T 粗筛到 14.8T / 高影响子集选择"为初读推断，论文无此说法，已收回）
- **对 DataInf Day 5 的呼应**：MoE 的 expert 稀疏激活，某条烂 code 可能只毒一个 expert。DataInf 教你在 LoRA 低秩子空间里闭式算 influence 来踢脏数据；DeepSeek 同理可以把 influence 算到 expert 级，定位到是哪条数据搞脏了哪个 expert，做专家维度的清洗，这是 dense 模型不需要的。
- **对 Phi-1 Day 6 的呼应**：Phi-1 证明合成教科书质量>数量；DeepSeek-V3 把合成题全上执行验证（run-and-filter），比 textbook 多了一级“可验”，是 Phi-1 的升级版。14.8T 里 10% 用了 FIM PSM 框架 `<|fim_begin|>pre ... |fim_hole| suf |fim_end| mid` 来保代码续写能力【3762324244574975666†L70-L76】。

## 核心
1. **Motivation**: MoE 总参 671B 但激活 37B，想训得稳、推理省，必须数据更干净、code 更重。DeepSeek-V2 已验证 DSA + MoE，可沿用，但 V2 的数据偏通用，数学/编程比不够，冗余多，需要重配比。
2. **Data Pipeline**: 来源 →  Web/多语种 + Math/Code 上采样 → 清洗 →  heurisitic 去毒/PII → 去重：文档 dedup + 近重复去除（论文未披露算法与阈值，"MinHash"为初读推断）→ document packing（Ding et al. 2024方法）但不做 cross-sample attention【3762324244574975666†L64-L67】 → 质量过滤：类似 Llama 3 的多级，但 multilingual 扩大、英文/中文外也保多样性 → 合成：Code FIM 10% PSM【3762324244574975666†L70-L76】 + 数学推理链 → 训练：14.8T high-quality diverse tokens【3762324244574975666†L59-L63】 + MLA + Aux-loss-free 负载均衡 + Multi-token prediction。
3. **Key Tricks**:
   - 数学与代码样本占比提高（论文未给数字；"30%+"为初读推断，已收回，2026-09-08）。MoE 的 expert 路由被认为能容纳更多编程模式，但论文未做配比消融，此为解读、非结论。
   - 论文强调 minimize redundancy while maintaining diversity【3762324244574975666†L62-L66】，但未披露去重算法与阈值（"MinHash 0.90 / 比 Llama 3 更狠"为初读推断，已收回，2026-09-08）；文档 pack 保 integrity（不做 cross-sample attention）是论文明确的设计。
   - FIM 0.1 PSM 格式为 `<|fim_begin|>f_pre<|fim_hole|>f_suf<|fim_end|>f_middle<|eos_token|>`【3762324244574975666†L72-L76】（论文格式以 eos 结尾），专门保 code infill；配合 Tokenizer 128K + 合并标点/换行、训练时随机拆分抗 token boundary bias，这些代码向细节是 Phi-1 没有的。
4. **Results**: 671B/37B MoE，2.788M H800 GPU 时，训程零不可恢复 spikes；评测上开源模型里 SOTA，对标闭源 GPT-4 级；14.8T 训完后 SFT+RL 阶段仍稳。Long context 128K、tool use、code HumanEval/MBPP 都超 Llama 3 405B 路线，推理激活参数小 10 倍。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. **code 配比实验**：论文未给 code 配比数字（"30%"为初读推断，已收回，2026-09-08）；若要探索高 code 配比，自己跑消融：把你 50 万合成池的 code 占比设 17% / 25% / 35% 三档，加一道二次去重，用 1.3B dense 看 HumanEval 涨不涨、MMLU 掉不掉。
  2. **FIM + 执行过滤**：抄它的 10% FIM PSM 合成，把 Phi-1 的 textbook 题改成可执行的 fill-in-middle 题，solver 跑单元测试，过不了的题直接扔，留下的题自带 hidden tests，比 Phi-1 多一级可验。
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：
  - MoE 路由的负载均衡如果没 aux loss（DeepSeek 创新），对数据偏斜更敏感，数据必须提前做 expert 分布均衡打点，否则训练抖动。用 DataInf 思路预估每条数据会进哪个 expert，做均衡。
  - 14.8T 级别 pack但不做 cross-doc attention省显存，值得抄到你 vLLM rollout 的 pack上。

## 疑问 / 下一步
- DeepSeek-V3 的 Tokenizer 128K 里合并且随机拆分标点/换行的 trick，在 MoE 上对 code FIM 的收益有多大？能否用你 1.3B 跑个小消融？

## 原文金句 (1-2句)
> We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens【3762324244574975666†L12-L14】

> our data processing pipeline is refined to minimize redundancy while maintaining corpus diversity【3762324244574975666†L62-L66】

> <|fim_begin|>f_pre<|fim_hole|>f_suf<|fim_end|>f_middle【3762324244574975666†L72-L76】

## 3 问回顾（Day 8原题）
1. （2026-09-08 收回重写：原题干"为什么 MoE 要把 MinHash 阈值和 code 占比调得更高"的前提非论文事实，论文未披露这两项数字。）修正版：论文 §4.1 只说了"提高数学与代码占比、去冗余保多样"，如果你是数据负责人，你会设计怎样的可验证实验来补上这些缺失的披露？
2. 它的合成数据是怎么做执行过滤的？和 Day 6 Phi-1 的 textbook 合成比，多了哪一级“可验”？
3. 如果把 DeepSeek 这套搬到你 1.3B dense 上，有哪一条不能抄？为什么 MoE 能抗 30% code，dense 不行？

---
生成逻辑：已纳入知识图谱，强调结构非数量，自动产出已开启。

## 第二轮复习（2026-09-08）

> 本轮已对照论文原文 §4.1–§4.2（arXiv 2412.19437 v2，2025-02-18 修订版）逐项核对初读 NOTES；发现四处把"推断"写成"论文事实"的偏差，已在 §4 开头明示纠正。核对方法：全文检索 "30%" / "MinHash" / "deduplication" / "20T" 四词在论文中均零命中。

### 1. 核心命题

DeepSeek-V3 真正解决的 data 问题不是"怎么训 MoE"，而是：**MoE 架构下，预训练数据的"表示形式"与"混合比例"必须按架构重设计，不能照搬 dense 的配方**。论文 §4.1 只有一段话，却藏了三个"架构-数据耦合"决策：

1. **文档级 FIM（rate 0.1，PSM 框架）**：code 能力的训练信号不靠堆更多 code tokens，而靠改变 code tokens 的呈现方式——把文档重组为 `<|fim_begin|> pre <|fim_hole|> suf <|fim_end|> middle <|eos_token|>`，把"续写"任务变成"在前后文约束下补全"任务；
2. **document packing 但不做 cross-sample attention masking**（Ding et al. 2024）：数据完整性（document integrity）优先于 packing 效率，且与 FIM 的文档级应用耦合——FIM 样本的 suf 不能跨文档泄漏，否则补全任务被污染；
3. **tokenizer 本身就是数据管线**：128K byte-level BPE，pretokenizer 合并标点+换行（code 里换行多，直接决定 code 的有效 token 预算），但引入 token boundary bias（Lundberg 2023），修复方法是训练时随机拆分一部分合并 token——tokenization 从确定性预处理变成了训练时的随机增强。

一句话：DeepSeek-V3 是"架构-数据协同设计"的存在性证明——MoE 的数据配方不是 Llama 3 配方的超参微调，而是表示层面的重写；但论文只给了存在性证明，没有给任何因果证据（零数据消融）。

### 2. 图谱位置

- **同级对标 Day07 Llama 3（存在性证明 vs 存在性证明）**：Llama 3 公开了"瀑布"的存在（5 级过滤）但没给阈值；DeepSeek-V3 连瀑布的存在都没给（§4.1 一段话带过），但给了表示层面的精确规格（FIM PSM 格式、packing 策略、tokenizer 修改细节）。两者互补：Llama 3 告诉你"过滤要有层级"，DeepSeek 告诉你"表示要有设计"。注意 Day07 复习已纠正 Llama 3 的 code 配比是 17%（25% 是 math+reasoning），而 DeepSeek-V3 的"enhancing the ratio of mathematical and programming samples"连数字都没给——"30%+" 是初读推断（见 §4 纠正）。
- **直接对比 Day25 FineWeb（重点）**：FineWeb 是 Llama 3 网页瀑布的开源可复现对照组（每个 gate 都有消融）；DeepSeek-V3 的"minimize redundancy while maintaining corpus diversity"在论文里零方法披露——FineWeb 式的可审计 gate 正是 DeepSeek 没给的那块拼图。引用 DeepSeek 的去重结论时，可复现性要去 Day25 找，不要引用 DeepSeek。
- **互补 Day24 D4 / SemDeDup**：DeepSeek 只说"去冗余保多样"，D4 给了语义去重 + 原型式多样化剪枝的工程实现。D4 是 DeepSeek-V3 §4.1 那句话的操作化版本。
- **配比层 Day31 DoReMi**：DeepSeek 的"enhanced ratio"（无数字、无消融、无 proxy）= DoReMi 想自动化的手工配比的最极端反例——连数字都不披露。两者共享深层模式：数据决策应该用比最终训练便宜 1–2 个数量级的 proxy 做；DeepSeek 没做，DoReMi 做了。
- **后继 Day09 Qwen2.5**：Qwen2.5（18T → 1M SFT → 多阶段 RL 飞轮）把"预训练之后"的数据故事讲全了；DeepSeek-V3 把"预训练之中"的架构-数据耦合讲深了。Qwen 的数据门禁（RM 迭代筛）与 DeepSeek 的表示设计是正交的两条线，可以叠加。
- **选择线 Day04 LESS / Day05 DataInf**：MoE 的 expert 路由让"数据影响"天然有了 expert 维度——某条脏数据可能只毒某几个 expert（初读 NOTES 这个直觉是对的，但论文没做）。DataInf 的 LoRA 子空间闭式 influence 可以平移到"expert 子空间"：对 routed expert 的 FFN 参数算 influence，定位哪条数据搞脏了哪个 expert，这是 dense 模型不需要的数据诊断。
- **合成线 Day06 Phi-1**：Phi-1 用 textbook 改变"数据的文本分布"；DeepSeek-V3 用 FIM 改变"数据的呈现结构"。两者是同构思想：不增加 token 数，靠改变 token 的组织方式来教能力。DeepSeekCoder-V2 的观察（FIM 不损害 next-token 能力）是这条路线的经验证据。

### 3. 机制深挖

**(a) FIM 的三层设计：rate、格式、应用粒度。** Rate 0.1（约每 10 个文档 1 个做 FIM）；格式为 PSM 的 `<|fim_begin|> pre <|fim_hole|> suf <|fim_end|> middle <|eos_token|>`（注意论文格式以 eos 结尾，初读 NOTES 漏了它）；应用粒度是 document level，在 pre-packing 阶段完成。深挖点：FIM 把训练信号从 $$P(\text{middle} \mid \text{pre})$$ 变成 $$P(\text{middle} \mid \text{pre}, \text{suf})$$——后者是信息约束更强的任务；且因为作用于 document level 而非 packed-sequence level，它与"不做 cross-sample attention"的 packing 选择是耦合设计：若 FIM 样本的 suf 来自另一个文档，补全任务就被污染了。这就是论文说的 "document packing method for data integrity" 的真正含义——integrity 不是效率词，是正确性词。

**(b) "minimize redundancy while maintaining corpus diversity" 的操作化缺失。** 论文没给任何去重算法、阈值、保留率。但可以从两个披露的数字反推约束：第一，14.8T 是 "in our tokenizer" 的计数——DeepSeek 128K tokenizer 对中英压缩率高，14.8T tokens 对应的原始字符量与 Llama 3 的 15.6T（英文为主、3.94 chars/token）不可直接比较，跨 tokenizer 比 token 数是范畴错误；第二，训练成本 180K H800 GPU-hours 每 1T tokens——去重每多砍 1T，就省约 180K 卡时，去重的经济价值可以直接换算成卡时，这是"去重强度"的正确度量单位（而不是 MinHash 阈值小数点后两位）。

**(c) Tokenizer 即数据管线。** 三个披露点串成一条因果链：128K byte-level BPE → pretokenizer 合并标点+换行（省 tokens；code 文档换行密集，code 的有效 token 预算直接被这个设计放大）→ 引入 token boundary bias（few-shot prompt 没有 terminal line break 时行为漂移，Lundberg 2023）→ 训练时随机拆分一部分合并 token 来对冲。这是论文里罕见的"数据表示非确定性"被公开承认的案例：tokenization 不再是确定性预处理，而是训练时的随机增强。对 coding 数据工作的启示：分词器改动 = 数据配比改动，两者要在同一张 recipe 表里记。

**(d) Aux-loss-free 的数据含义（§4.2 最值得挖的一句）。** Expert bias 更新速度：前 14.3T tokens 为 0.001，最后 500B tokens 为 0.0（冻结）；balance loss 权重仅 0.0001，"just to avoid extreme imbalance within any single sequence"。数据视角的翻译：前 14.3T 让路由 bias 跟着数据分布学；最后 500B 路由冻结——如果 annealing 窗口换入高质 code/math 数据导致域分布变化，路由无法再适应，expert 负载失衡只能靠数据本身的域间均衡来保证。**数据管线在这里接管了架构的负载均衡职责**，而 loss 里没有惩罚项把它拉回来。这是 aux-loss-free 区别于 aux-loss 模型最锋利的数据侧含义。

**(e) MTP 的数据税。** MTP depth 1，loss weight 0.3（前 10T）→ 0.1（后 4.8T）。每个 token 产生约 1.3 / 1.1 个预测损失——14.8T 的"有效训练信号"是 $$14.8\text{T} \times (1 + w)$$ 量级。比较不同论文的 token 数时，要意识到 MTP 通胀了 per-token 的梯度信号；"14.8T tokens" 是输入计数，不是信号计数。

### 4. 边界与反例

1. **纠正初读的四个"论文事实"（本轮核对原文后的收回）**："code 30%+"、"MinHash 0.85→0.90"、"20T 粗筛到 14.8T"、"MoE 更吃噪声故做高影响子集选择"——"30%"、"MinHash"、"deduplication"、"20T" 四词在论文全文零命中。§4.1 原文只有 "enhancing the ratio of mathematical and programming samples"（无数字、无基线）和 "minimize redundancy while maintaining corpus diversity"（无方法）。这些数字是初读的合理推断，不是论文事实；引用时必须降级为推断，Day08 的"3 问回顾"第 1 题的前提（"为什么 MoE 要把 MinHash 阈值和 code 占比调得更高"）本身就不是论文可验证的。
2. **论文没有做任何数据消融**：没有 FIM 0.1 vs 0 的对比、没有 math/code 配比对比、没有去重强度对比、没有 tokenizer 修改的对比。所有数据决策都是"存在性证明 + SOTA 结果"的捆绑销售，不能从中读出因果。不要引用 V3 证明"30% code 对 MoE 最优"或"FIM 提升 code 能力"——论文没做这些实验。
3. **跨 tokenizer 的 token 数不可比**（见 §3(b)）：14.8T vs 15.6T 的直接比较忽略了分词器压缩率差异。更诚实的比较单位是原始字符量或训练卡时（180K H800-hours/T）。
4. **FIM 的适用域**：论文中 FIM 有效性的直接证据来自 DeepSeekCoder-V2（code 专用模型）；V3 把它搬到通用 14.8T 语料上，0.1 的 rate 无消融。对非 code 文档做随机切分的 FIM（pre/suf/middle 任意切）是否引入噪声、是否稀释了通用文本的 next-token 信号，论文没有讨论。
5. **Aux-loss-free + 末期 bias 冻结的风险**（§3(d) 的反面）：最后 500B tokens LR 已极小（2.2e-5 → 7.3e-6）且 bias 更新为 0——若 annealing 阶段换入分布不同的高质数据，域分布变化 → token 路由分布变化 → expert 负载失衡，而 loss 里没有 aux loss 把它拉回来。这是"训练末期换数据"在 aux-loss-free 下的特有风险，dense 模型和 aux-loss MoE 都没有这个问题。
6. **训练稳定性不能归因到数据**：论文说 "we did not experience any irrecoverable loss spikes or perform any rollbacks"，但把它归功于算法-框架-硬件的整体 co-design，没有把稳定性归因到数据管线。不能引用 V3 说"更干净的数据带来训练稳定"——论文没这么说。

### 5. 迁移到 coding / post-training data

**可执行实验 A — FIM-for-code（两周可跑，检验 §3(a)）**

1. 从你 500k 合成池取 code 子集，按 PSM 格式构造 10% FIM 样本：pre = 函数签名 + docstring，suf = 单元测试，middle = 函数体；组织为 `<|fim_begin|>pre<|fim_hole|>suf<|fim_end|>middle<|eos|>`，在 document 级别、packing 之前完成重组。
2. 对照组：同样的 10% tokens 保持原 next-token 格式。两组各训一个 1.3B proxy，同 token 预算。
3. 评测双指标：HumanEval（next-token 能力是否受损，验证 DeepSeekCoder-V2 的"不损害"观察）+ 自建 infill 评测（随机挖掉函数体、给定签名与测试，测补全通过率）。
4. 通过标准：infill 通过率提升且 HumanEval 不掉 → 把 FIM rate 写进你的 recipe；若 HumanEval 掉 → 说明 0.1 的 rate 在小模型/小数据上不成立，先降到 0.05 重测。

**可执行实验 B — annealing 窗口数据门禁（抄 §4.2 的 LR schedule）**

1. V3 最后 500B / 14.8T ≈ 3.4% tokens 用 2.2e-5 → 7.3e-6 的极小 LR。按比例：你的 1.3B 跑 X tokens，最后 ~3% 用 cosine 尾巴 LR，只喂 execution-verified 的高质 code；对照组同样尾巴但喂默认 mix。
2. 看 HumanEval+ / MBPP+ delta——这是 Day07 复习 §5 的"Llama 3 annealing 评估法"在你池子上的 DeepSeek 版。
3. 纪律（论文没给、但 §3(d) 要求的）：从今天起记录每个数据源"进入训练时的 LR 阶段"——V3 证明了数据价值是 LR-schedule-dependent 的（同样的高质数据，放在 2.2e-4 阶段和 7.3e-6 阶段价值不同）。你的 recipe 表要加一列"LR 窗口"。

### 6. 今天的一道思考题

综合 **Day08 DeepSeek-V3、Day07 Llama 3、Day24 D4、Day31 DoReMi**（答案不在任何一篇原文里）：

**(a) 拼图与缺口。** 四者恰好拼出"预训练数据决策"的完整拼图：Llama 3 给了过滤瀑布的存在性证明，FineWeb（Day25）给了可复现配方，D4 给了语义去重工程，DoReMi 给了可学习的配比——而 DeepSeek-V3 的 §4.1 只有一段话：去重无方法、配比无数字、质量门无披露。假设你是 DeepSeek 的数据负责人，要在内部复现并超越 V3 的 14.8T 配方：设计一个三层可审计流程——L1 去重（D4 式语义去重）、L2 配比（DoReMi 式可学习 domain weights）、L3 门禁（Llama 3 式 annealing 评估）——明确每一层的输入、输出信号、proxy 模型规格、"通过 / 回滚"判据。关键难点：L2 的 domain 怎么切，才能不吃掉 L1 的功劳？（提示：Day07 复习思考题 (b) 的"过滤改变域内分布、配比改变域间权重、两者不正交"问题，在这里同样成立；先想清楚你的 domain 定义要不要包含"是否被 L1 删过"这个 provenance 维度。）

**(b) 冻结的路由与换挡的数据。** V3 用 aux-loss-free 做负载均衡，且最后 500B tokens 把 expert bias 更新速度设为 0（§4.2）。从数据视角论证：这个选择对"训练末期换入高质 annealing 数据"是利还是弊？给出完整机制链条（数据域分布变化 → token 路由分布变化 → expert 负载失衡 → loss 里无惩罚项 → ？），并设计一个**不许改架构**的数据管线侧对冲方案：annealing 窗口的数据域分布要满足什么约束，才能保证冻结的路由不失衡？把这个约束写成在你 500k 池子上可执行的检查规则（输入：候选 annealing 数据集的域直方图；输出：通过 / 降权 / 拒绝；并说明阈值怎么定——不能拍脑袋，要给出定阈值用的 proxy 实验）。

---

**论文原文**：https://arxiv.org/abs/2412.19437
**GitHub NOTES**：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-08-2024-deepseek-v3/NOTES.md

## 问答补充（2026-09-08）

> 来自 `ai data` side chat 用户主动问答（2026-09-08 10:26 PDT），归档时精简整理；答案保留原符号定义与结论。

**问题（用户）**：Aux-loss-free 啥意思？

**核心答案**：

先定义符号。MoE 有 $E$ 个 expert，一批 $T$ 个 token。门控网络给 token $t$ 对 expert $i$ 打亲和分 $a_{t,i}$ ，每个 token 选 top- $K$ 个 expert。

**问题**：训练初期路由会坍缩——少数几个 expert 被所有 token 选中，其他 expert 饿死。负载记 $f_i$ = 分到 expert $i$ 的 token 比例，理想是 $f_i \approx K/E$ 。

**传统解法（aux loss）**：在总 loss 里加一项负载均衡惩罚，最常见的是：

$$L_{\mathrm{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i$$

其中 $P_i$ 是 expert $i$ 的平均路由概率。 $f_i \cdot P_i$ 在均匀分布时最小，失衡时变大，所以最小化它能把负载拉平。代价是：**这项 loss 的梯度和主 LM loss 的梯度打架**——模型要在"预测准"和"负载平"之间妥协， $\alpha$ 选不好两头受损。

**Aux-loss-free（DeepSeek-V2 首创，V3 沿用）**：loss 里**完全没有**这一项。做法是给每个 expert 加一个偏置 $b_i$ ，加在亲和分上再做 top- $K$ ：

$$\mathrm{选中的\ expert} = \mathrm{topK}_i(a_{t,i} + b_i)$$

$b_i$ **不走梯度**，而是走一个控制回路（像恒温器）：统计最近 expert $i$ 的实际负载，超载就调小 $b_i$ ，空闲就调大 $b_i$ 。负载均衡从"优化问题"变成了"反馈控制问题"，主 loss 的梯度干干净净。

V3 论文 §4.2 给的具体数字：前 14.3T tokens， $b_i$ 更新速度 0.001；最后 500B tokens 更新速度设为 **0（冻结）**。另外留了一个权重仅 0.0001 的 sequence-level balance loss 做兜底——"just to avoid extreme imbalance within any single sequence"，防止单个序列内出现极端失衡。

为什么这事值得在数据视角下单拎出来：bias 控制回路是**跟着数据分布学**的——数据里 code 多，code 擅长的 expert 的 $b_i$ 就被调到合适位置。一旦最后 500B 把 $b_i$ 冻结，而此时换入分布不同的 annealing 数据（比如突然加了很多数学），token 的路由分布变了，expert 负载失衡，但**没有任何机制能把它拉回来**（loss 里没有 aux loss，bias 也不更新了）。所以 aux-loss-free 把"负载均衡"这个职责在训练末期**移交给了数据管线**：annealing 窗口的数据域分布必须和之前保持兼容，否则 expert 直接失衡。这就是"架构-数据耦合"最锋利的一个实例。

**关联**：Day 08 DeepSeek-V3 §4.2（aux-loss-free 负载均衡）；与第二轮复习（2026-09-08）思考题 (b)"冻结的路由与换挡的数据"直接呼应——本问答是该思考题的前置机制解释。
