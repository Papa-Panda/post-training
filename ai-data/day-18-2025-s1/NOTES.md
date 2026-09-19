# Paper 模板 - Day 18

> 复用 PAPER_TEMPLATE.md 骨架，自动生成

## 元信息
- Title: s1: Simple test-time scaling
- Authors / Org: Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, et al. / Stanford, Together AI, Washington
- Link / arXiv: https://arxiv.org/abs/2501.19393
- Date read: 2026-08-18
- Tags: [sft, reasoning, data-selection, less-is-more, test-time-scaling, coding-data, curation, quality]
- Folder: day-18-2025-s1
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-18-2025-s1

## 一句话总结
用 1k 最高质长链推理轨迹 s1K 做 SFT + budget forcing 控制 test-time thinking 长度，让 Qwen2.5-32B-Instruct 在 AIME 50%+、MATH 切换到 90%+，匹敌 o1/R1，证明 SFT 阶段少即是多 + 推理时算力缩放可直接替代大规模 RL，是 LIMO 的推理时延伸和平行验证。

## 和之前工作的关系

> 知识图谱位置：post-train / SFT selection / reasoning / 少即是多 分支的 SFT+TTS 双极点，和 Day17 LIMO 互为双子，收束 Day11 LIMR RL少即是多 → Day15 R1 冷启动+RL 泛化 → Day16 Qwen2.5-Coder 执行过滤 的主线

- 接了哪条线：
  - selection 线：Influence (Day02) → TracIn (Day03) → LESS (Day04 5% 梯度选) → DataInf (Day05 LoRA闭式) → SuperFiltering (Day12 弱模型 IFD 125M→7B) → LIMR (Day11 RL 1.3k 轨迹对齐) → LIMO (Day17 SFT 817) → **s1 (本篇 SFT 1k + TTS budget forcing)**
  - synthetic/pretrain 线：Phi-1 (Day06 教科书合成 1.3B 50.6% HumanEval) → Llama3 15.6T瀑布 (Day07) → DeepSeek-V3 MoE 14.8T 30%code (Day08) → Qwen2.5 18T flywheel (Day09) → DeepSeek-R1 <10k冷启动+可验证RL (Day15) → Qwen2.5-Coder 三级执行过滤 (Day16 5.5T→可执行) → **s1K 去蒸馏+去重+难度三滤，只留 1k 全过程可验证**
  - SFT vs RL / TTS 线：Phi-1 天然指令 → Llama3.1-3.2 RS+DPO (Day10) → LIMR RL少即是多 → R1 SFT memorizing vs RL generalizing → LIMO SFT generalizing 反例 → **s1 SFT 1k + 推理时 scaling 替代 RL scaling，与 R1 纯 RL 路径分叉竞争**

- 补了哪个短板：
  - LIMO 只证 817 SFT 可涌现推理，但没讲推理时怎么控；s1 补 budget forcing：用 "Wait" token 强制延长/截断 thinking，TTS 可控 0.5k~16k tokens，直接把 AIME 从 50% 推到 56.7%，补了 LIMO 评测时单次解码的短板
  - LIMR 选 RL 难例但依赖 GRPO 训练；s1 证明同池难例 SFT 1k 即可，无需 RL，降低 10x 算力，补 RL 成本
  - R1 冷启动<10k 为 RL 稳格式，s1K 1k 为 SFT 极致且开源全去蒸馏（只选 59k→1k，经 decontam 去 AIME/MATH/GPQA leak），补 R1 冷启动未开源+蒸馏依赖的短板
  - SuperFiltering 125M 小模型自动 IFD 选，s1 用强模型三级难度+多样+去重人工+规则，成本-质量谱系两端对位

- 替代/分叉/改进：
  - 对 LESS/SuperFiltering/LIMR/LIMO 是 **收敛与双点验证**：从梯度/IFD/轨迹到启发式，s1 和 LIMO 同结论不同池（s1K 来自 OpenThoughts 59k，经 3级 1k 滤，LIMO 来自 Numina 100k→817），双盲验证 less-is-more鲁棒性
  - 对 DeepSeek-R1 是 **分叉**：R1 路径 冷启动+纯 RL 可验证奖励涌现长链，s1 路径 SFT 1k + TTS 控制涌现长链，无 RL，二者 AIME 同 50%+，证明两条路都通，选 infra 更轻的
  - 对 Phi-1/Qwen 合成是 **提纯**：不是合成更多，而是从大量合成中精选 1%，1% > 99% 的第二实证，和 Qwen2.5-Coder 执行过滤互补：先exec过滤可执行，再 s1/LIMO 难+多过滤

- 对之前 Day X 的直接对比：
  - vs Day17 LIMO：同 Less-Is-More 双子，同 1k 尺度（LIMO 817 vs s1 1000），LIMO 7步去重强调认知模板四段（problem→plan→reason→verify），s1 3级去重+decontam强调全链去蒸馏和长链（平均 9k tokens vs LIMO ~3k），LIMO 57.1%/63.3% AIME(v1/v3) 95.6% MATH，s1 50%→56.7% AIME with budget forcing 94% MATH，LIMO 无 TTS，s1 有 TTS是唯一新增杠杆，二者同来自 SJTU/Stanford系，互证预训练完备×模板有效
  - vs Day11 LIMR：同 SJTU/SII/GAIR vs Stanford 同 Less标题，LIMR 8.5k→1.3k RL 选 AIME +16.7%，s1 59k→1k SFT 选 AIME 6.5%→50%+，RL选 vs SFT+TTS选镜像，s1 更极端且省 RL 算力
  - vs Day12 SuperFiltering：同弱到强但对立实现，SuperFiltering 125M算IFD自动选省算力，s1 用强模型+规则+人工省样本费筛选，125M vs 72B judge成本-质量两端
  - vs Day15 DeepSeek-R1：R1冷启动<10k为RL稳格式+纯RL可验证涌现，s1 1k为SFT终结+TTS涌现，无RL但引入TTest-time scaling作为第二缩放轴，infra极简，验证SFT≠记忆
  - vs Day16 Qwen2.5-Coder：Qwen2.5-Coder三级瀑布洗5.5T为可验证可执行语料池，s1是池上第二级精选 1k难例，二者串联即 coding 1k冷启动最佳实践：exec过滤→难+多过滤→TTS

## 为什么今天读它

- 跟 coding data / SFT / RL data 的连接：LIMO 已证数学推理 817 SFT涌现，s1 补充 TTS 可控是 coding 最易迁移点：SWE-Bench/Codeforces 上同样可用 budget forcing 控 thinking 长度，用 Qwen2.5-Coder 执行过滤先保可执行，再用 s1 三滤做 1k coding冷启动集，替代 100k SFT，并为 Day15 R1 的 RL cold-start 提供无RL轻量替代，验证 coding 上 SFT 1k + TTS 是否也能匹敌大规模RL。

## 核心
1.  **Motivation**: 为什么要做这个 data 工作？baseline 痛点？：R1 后大家默认推理需 10k冷启动+大规模可验证RL，SFT被认为只能记忆不泛化，且推理时scaling需复杂搜索/奖励模型。作者问：能否用极少精选+简单TTS达到o1/R1级别？
2.  **Data Pipeline**: 数据从哪来 → 怎么洗/合成/过滤 → 怎么评 → 怎么进训练：来源 OpenThoughts-114k / open-s1 59k 长链 → 3级过滤：去测集leak(AIME/MATH/GPQA 13-gram+embedding去重)、难度分层(只留最难，4/4采样全错或1/4对)、多样性去重(领域/技能 embedding cos<0.8，去模板) + 去蒸馏(拒绝GPT-4o直接蒸馏过度结构化) → 1k s1K 定版（平均 9k tokens, 最长 25k）→ SFT Qwen2.5-32B-Instruct (lr 1e-5, 5 epoch) → 评 AIME24/MATH/GPQA + TTS budget forcing (Wait token 延长 / 截断 thinking 0.5k~16k) → AIME 50%→56.7%
3.  **Key Tricks**: 3个最值得抄的细节（阈值、模型、规则、去重、合成 prompt）
   - 难度优先+去蒸馏：只留 1k中 59k→1k ≈1.7% 最难，且人工复核去 GPT-4o 蒸馏过度结构化，保留人类般试错痕迹，阈值：最难 1k/59k ≈1.7%，泄漏 13-gram 命中即删
   - 多样性+全链完整性：要求每条必须含 problem→thought→attempt→verification，且 thought 长链平均 9k，缺一段丢，类似 LIMO 认知模板但更长
   - Budget forcing：推理时插入 "Wait" 强制续 thought 或截断 thought 到指定 budget，0.5k~16k 线性控算力，无需奖励模型，AIME +6.7% 绝对提升，coding 上可直接复用控 SWE-Bench 解题长度
4.  **Results**: 对 downstream 有多大提升？用什么评的？：Qwen2.5-32B-Instruct 基 6.5% AIME → s1 50% AIME24 / 56.7% with BF / 94% MATH / 59% GPQA，超 o1-preview 44.6%，平 R1 50%级，用 1% 数据打赢 10万 级，且推理时缩放可线性加成，1k 数据+TTS 替代大规模RL

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 复用 Qwen2.5-Coder parser+exec 三级过滤作第一级，再用 s1 三滤（难度+多样+去蒸馏）做 1k coding冷启动：SWE-Bench Hard + CF 2500+ 经 exec 过滤后，用 72B judge 挑 1k 最难+多样，平均 thought 8k+，替代 100k SFT
  - 把 s1 budget forcing 直接移植到 coding eval：SWE-Bench 解题时用 Wait token 控 thinking 2k→16k，看 pass@1 随 budget 线性提升，验证 TTS 在 code 上是否为第二缩放轴，补 LIMR RL scaling
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：s1 流水线极轻：无需训练选模型，只需规则+去重+强judge+TTS，成本是 LESS 的 1/100，RL 的 1/1000，适合 nightly 1k 精选+TTS扫 budget，评测用 AIME/MATH/SWE-Bench+Exec可验证作 OOD 探针，可嵌 ai-data sheet 自动跑分 + budget 曲线

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：s1 和 LIMO 同 1k 规模但 s1 thought 平均 9k vs LIMO 3k，是否说明 coding 需要更长 thought？Qwen2.5-32B 基座是否足够，还是 7B 也可复现 50% AIME？若换 7B Qwen2.5-Coder，1k 是否仍有效还是需 3k？这对 7B coding冷启动的可迁移阈值至关重要。

## 原文金句 (1-2句)
> In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but precisely orchestrated demonstrations of cognitive processes — and scaled at test time by simply forcing the model to think longer.
> Supervised fine-tuning on 1,000 carefully curated traces can match the performance of models trained with massive RL, when combined with simple test-time scaling.

## 今晚产出
- NOTES.md 按模板已填（含和之前工作的关系小节）
- reading-log.csv 待追加
- GitHub folder day-18-2025-s1

> 自动化：reading-log 已追加 / commit 待推 / ai data sheet 待同步


## 第二轮复习（2026-09-18）

> 本轮复核：arXiv:2501.19393 v3（3/1 修订）ar5iv 全文 §1–§4、Table 1、§C.4、§5.1 消融 + arXiv 摘要逐项核验。修正初读 NOTES 四处：① "全去蒸馏/拒绝 GPT-4o 蒸馏"错误：全部 59k 轨迹来自 **Google Gemini Flash Thinking API**（§2.1、引言）——s1K 是蒸馏数据，只是**策展过的蒸馏**；② 去污染是 **8-gram**（§2.1）不是 13-gram；③ 难度过滤是"Qwen2.5-7B 与 32B 都解错才留"（Claude 3.5 Sonnet 对参考答案判分）+ 轨迹长度作难度代理 + 多样性阶段按域内长度偏好采样——初读"4/4 全错或 1/4 对"属过度具体化，论文未披露该阈值数字；④ 结果口径：s1-32B = AIME24 **56.7** / MATH500 **93.0** / GPQA **59.6**（Table 1）；摘要"50%→57%"是 budget forcing 前后（50%→56.7%）的四舍五入；训练 5 epochs / batch 16 / 共 315 步 / lr 1e-5 余弦 / AdamW / bfloat16，16×H100 仅 26 分钟。

### 1. 核心命题

s1 真正解决的 data 问题，不是"推理数据要多少"，而是把"推理能力"拆成两个正交的数据问题：**(a) 过程行为的安装**（s1K：用 1k 条轨迹把"长思考 + 自我修正"的行为模式装进模型），**(b) 推理算力的弹性**（budget forcing：把 thinking 长度变成解码时的可控旋钮）。R1/o1 把两件事绑在一起——用大规模 RL 同时解决"会想"和"想多久"；s1 证明二者可解耦：SFT 装行为，TTS 给算力。

最反直觉的发现藏在 §2.2 末尾一句话："Some distilled generations are incorrect, which we allow in our data as we focus on capturing the reasoning process rather than entirely correct solutions." 论文自己的 grader 判定 s1K 里只有 **53.6%** 的最终答案正确。也就是说：**SFT 的推理数据里，"过程示范价值"与"答案正确性"是可分离的**——近一半答案错的轨迹，照样教出了 56.7% 的 AIME。这是对"数据必须对"的直接证伪，也是对 R1 "SFT memorizes" 刻板印象的精确反驳：SFT 记忆的如果是*行为模式*（自我检查、回溯、修正的循环），这种 memorization 恰恰是有益的。

### 2. 图谱位置

- **直接前驱 Day23 LIMA（对齐极简 → 推理极简，重点直接对比）**：LIMA 用 1k 人工策展证明"对齐可以很薄"，s1 用 1k 证明"推理 elicitation 也可以很薄 + 可伸缩"。两者同 1k 尺度、同 32B 级基座（LLaMA-65B vs Qwen2.5-32B）、同纯 SFT，但**选择器哲学对立**：LIMA 信人的品味——来源质量/风格统一/任务多样性，选"输入好"，教的是*助手口吻*；s1 信模型的行为——7B/32B 双失败率 + MSC 领域配额 + 轨迹长度，选"模型缺的"，教的是*思考行为*。LIMA 的"风格统一"是表面约束，s1 的"过程完备 + 自我修正"是行为约束。s1 还是 LIMA 的延伸：LIMA 之后 SFT 被认为只能做薄对齐，s1 证明 SFT + TTS 这条线能摸到 o1-preview——**SFT 的天花板不在 SFT 本身，而在解码时有没有第二缩放轴**。
- **双子 Day17 LIMO（双盲互证 + 一处张力）**：同 1k、同 Qwen2.5-32B、同"难 × 多 × 质"三滤、同结论（1% 打赢 100x），不同池、不同机构——less-is-more 最强的证据形态是独立复现。但两篇在"答案正确性"上立场相反：LIMO RQ1 证明 L5 vs L1 链质差 15pp（强调链的正确性），s1 保留 46% 错误答案仍有效。**调和**：LIMO 的"质量"是链的*过程质量*（详细、逻辑连贯、验证中间结论），不是答案对错——s1 的"答案错但过程好"轨迹，在 LIMO 的 ladder 上恰恰是 L5 级的。两篇真正一致：SFT 学的是*过程分布*，不是*答案记忆*。
- **镜像 Day11 LIMR（7B 的判决）**：LIMR §3.3 亲手处决了"s1/LIMO 无脑搬到 7B"——7B 上 SFT LIMO-817/s1-1k 的 AIME 只有 15.8，RL 路线 32.5。s1 的 1k 模板在 32B 上是行为安装，在 7B 上是"背不下来的长 CoT"（容量不够，蒸馏天花板）。**判据跟着容量走**：32B 靠模板唤醒（s1/LIMO），7B 靠自己探索（LIMR）——"少即是多"的成立条件里必须写进*容量*这一项。
- **上游 Day04 LESS / Day12 SuperFiltering**：s1 的三滤是选择哲学的"返祖"——从 LESS 的梯度代理、SuperFiltering 的 IFD 代理，退回"失败率 × 领域配额 × 长度"可解释启发式。§5.1 消融给出硬数字：随机选 / 只选最长 / 只选最多样，AIME24 都掉**约 30%**；而训全量 59k 池**不比 1k 好**。三标准是乘法关系，不是加法——缺一即崩，多了白给。
- **后继意义**：s1 把 scaling 的轴从"数据/训练"搬到了"推理时算力"，这是它 S-tier 的真正原因——**数据的单位价值被重新定义**：数据只负责"装上 thinking 模式"，算力缩放交给解码。之后所有 TTS 工作都在这个坐标系里。

### 3. 机制深挖

**(a) 三滤的精确管线（§2.1–§2.2）。** 59,029 题来自 16 源（NuminaMATH 30,660、AIME 1983–2021 历史题、OlympicArena 4,250、OmniMath 4,238、AGIEval 2,385，另自建 s1-prob 182 道斯坦福统计 PhD 资格考概率题、s1-teasers 23 道 quant 面试脑筋急转弯）→ 轨迹全部由 **Gemini Flash Thinking API** 生成 → 8-gram 对 MATH500 / GPQA-Diamond / AIME24 去污染 + 去重。

- **Quality**：API 错误丢弃（→54,116）→ 格式问题（ASCII 图、坏图引用、编号错乱 →51,581）→ 384 条来自可信源直接锁定（§C.4）。注意：质量滤的是*格式*，不是*答案对错*。
- **Difficulty**：双信号。① 行为信号：Qwen2.5-7B 与 Qwen2.5-32B 各解一遍，Claude 3.5 Sonnet 对参考答案判分，**两模型都错才留**（→24,496）——双模型交集防"一模型偶然做错放走简单题"。② 分布信号：轨迹 token 长度作难度代理（"难题需要更多 thinking tokens"的假设），多样性阶段按域内长度偏好采样。
- **Diversity**：Claude 3.5 按 MSC 分类法分域 → 50 个域 → **先均匀抽域、再在域内按长度偏好抽题**，直到 1,000。这是"配额式多样性"（LIMO 的战略采样同构），不是训后去重。

**(b) "46% 答案错但有效"的机制解释。** SFT 的 cross-entropy 在推理轨迹上学的是*行为分布* $p(\text{下一步思考} \mid \text{历史思考})$ ，不是*事实记忆*。一条"过程严谨但末步算错"的轨迹，贡献的是"检查—回溯—修正"的转移模式——这正是 budget forcing 续写 "Wait" 时模型要调用的模式。形式化一点：记轨迹为状态序列，SFT 拟合的是**状态转移算子**，答案只是终止状态的标签；终止标签错了不影响转移算子学得对。这解释了为什么 §5.1 里"只选最长轨迹"反而掉约 30%：长度是难度的*相关*信号不是*因果*信号，长但水（灌水、重复）的轨迹教的是坏的转移算子。

**(c) Budget forcing 的双向旋钮（§3.1）。** Max 侧：直接 append 终止 thinking 的 delimiter（可选再加 "Final Answer:"），强制提前交卷；Min 侧：**压住 delimiter 不让生成**，append "Wait" 逼模型继续反思，常能自我修正错误步骤。对比基线：token / step / class 三种 prompt 条件长度控制（可控性差）、rejection sampling（oracle 后验）。BF 的"完美可控 + 清晰正斜率"来自一个前提：**模型已经装了 thinking 模式**——s1K 和 BF 是耦合的，BF 在没经过推理 SFT 的模型上不会产生 scaling。这就是 s1K 存在的全部理由：1k 数据 = 安装 delimiter 行为的最小剂量。

**(d) "59k 不比 1k 好"的含义。** 全量池训练无实质增益（§1）+ LIMO 的"100k 训退化（32.3% < 49.9% 基座）"→ 推理 SFT 里未策展数据不是中性，是**稀释剂**：稀释"过程模板"信号的浓度。1% > 99% 的准确表述是"**毒性稀释**"：SFT 的 dense 信号不分青红皂白，错误/灌水过程也被压进分布。

### 4. 边界与反例

1. **"去蒸馏"是误读，成本被低估**："16×H100 训 26 分钟"是训练成本，不是数据成本。真成本 = 59k 条 Gemini Flash Thinking 轨迹 + Claude 3.5 全量判分 + MSC 分类 + 人工格式复核——"样本数便宜 ≠ 成本便宜"（同 Day17 复习 §4.3）。s1 的极简是*训练*极简，不是*数据生产*极简。
2. **过程/答案分离的剂量曲线没测**：53.6% 正确有效，那 30% 呢？10% 呢？论文没扫"答案正确率"这个剂量——"过程 > 答案"的命题在哪个正确率阈值下崩，没人知道。更细：s1 的"错"是"末步答案错"，若*中间推理逻辑*全错（坏的转移算子），是否还有效？论文没分层。
3. **难度定义的自我参照**：用 Qwen2.5-7B/32B 双失败定义"难"，再用 Qwen2.5-32B-Instruct 微调——评估器和被评估对象同家族（呼应 9/9 Q&A 情形二：自我参照评估器锁死在家族盲区）。"两模型都错"筛出的可能是 Qwen 家族的*共享盲区*，不是客观难度。LIMO 用 R1 级异家族多采样，自我参照更轻。
4. **去污染只有 surface-level**：8-gram（不是 13-gram）对 AIME/MATH/GPQA；池子里有 AIME 1983–2021 历史题——Day30 告诉我们 semantic-level 近义改写泄漏是另一回事。56.7% 里多少是模板迁移、多少是"见过同构题"，不可分。
5. **"超 o1-preview 27%" 的口径**：o1-preview 不是 o1；"up to 27%" 是 MATH/AIME24 上的最高相对值。且只在 32B 单一规模验证——7B 上 s1 模板已证失效（LIMR §3.3），"SFT+TTS 替代 RL"的结论外推边界是 32B+。
6. **证据没证明什么**：没测"1k 精选 + BF" vs "1k 精选 + RL"的 head-to-head（s1 只和 o1-preview/R1 比了终点，没比路线成本）；BF 的 scaling 只在竞赛数学上画出曲线，coding/科学问答上"Wait 是否同样带来正斜率"未证——这正是初读疑问的延续。

### 5. 迁移到 coding / post-training data

**可执行的映射："s1 式 1k coding 冷启动 + budget forcing 移植"（2–3 周可跑通）**

1. **组大池（抄 59k→16 源思路）**：Codeforces 1800+、SWE-Gym issue 池、内部仓库真实 bug 单三源混合；轨迹用强 thinking 模型（R1-distill 系或 Gemini）生成，**保留可执行性**：每条 trace 必须附带能跑的 repro。
2. **三滤转译**：Quality → 格式门（能解析、能跑通 repro）；Difficulty → 7B-Coder 与 32B-Coder 双失败（hidden tests 全挂）+ trace 长度代理；Diversity → 题型/语言/仓库域配额（先均匀抽域、再域内按长度偏好抽，抄 s1 的 MSC 配额）。
3. **关键实验（s1 §2.2 末尾的 coding 版）**：按"答案对错 × 过程好坏" 2×2 分四组各 250 条——"测试全过 + 过程干净" / "测试挂但含 repro→fix→verify 完整调试过程" / "测试过但过程灌水" / "双差"。同基座 SFT 比 SWE-bench-lite delta。**可证伪预测**：若"挂但过程好" ≈ "过且好" >> "过但灌水"，则 s1 的"过程 > 答案"在 code 上成立——这将改写 coding SFT 的数据标准：从"只收 AC 解"变成"收调试过程好的轨迹"。
4. **Budget forcing 移植**：SWE-bench 解题时压住 thinking 终止符、append "Wait"，扫 thinking budget 2k→16k，看 pass@1 是否线性提升。若正斜率成立，TTS 就是 coding 的第二缩放轴——eval 时的算力可以直接换分数，RL 之前先榨干 TTS。
5. **验收判据**：1k 精选 + BF 的 SWE-bench 分数 vs 100k 全量 SFT；若前者 ≥ 后者，coding 冷启动的"1k + TTS"路线即成立，可替代当前 100k SFT 做法。

### 6. 今天的一道思考题

> 综合 **Day18（s1）、Day17（LIMO）、Day04（LESS）**：
>
> (a) **"过程 vs 答案"的归因实验**。s1 保留 46% 答案错的轨迹仍有效（过程 > 答案）；LIMO RQ1 证明 L5 vs L1 链质差 15pp（链质重要）。设计 2×2 消融：同一题集，{答案对，答案错} × {过程好（含检查/回溯/修正），过程坏（灌水/跳步）}，四组各 250 条，同基座（Qwen2.5-32B-Instruct）SFT，比 AIME/MATH delta。可证伪判据：若"过程好 + 答案错" ≈ "过程好 + 答案对" >> "过程坏 + 答案对" → s1 机制得证：SFT 拟合的是*状态转移算子*（思考行为），不是答案记忆；R1 的"SFT memorizes"刻板印象需要修正为"SFT memorizes 行为模式，好的过程 memorization 是有益的"。若"答案错"组显著差 → s1 的 53.6% 只是"错得不够离谱"的幸存者偏差，过程/答案分离不成立。追问：把"过程好坏"换成 LIMO 的 L5/L1 人工分级，15pp 的 gap 在"答案错"组里是否还存在？
>
> (b) **"难"的自我参照实验**。s1 用 Qwen2.5-7B/32B 双失败定义难（同家族、自参照）；LIMO 用 7B 粗筛 + R1 级异家族多采样（跨家族）；LESS 用目标梯度（任务参照）。同一 59k 候选池，三臂各筛 1k：A = s1 式同家族双失败；B = LIMO 式异家族失败率；C = LESS 式目标梯度相似（目标任务 = AIME/MATH）。同基座 SFT 比 delta。判据：若 B > A → "难"需要跨家族验证，同家族双失败筛出的是家族共享盲区（呼应 9/9 Q&A 情形二：自我参照评估器的系统性跑偏）；若 C 最优 → 推理任务上梯度代理仍有效，s1/LIMO 的启发式有算法化空间。关键控制：LESS 的目标梯度来自 7B 还是 32B？若用 7B 梯度选给 32B 训，是否复现"错误切空间"跑偏——把 9/9 情形三的思想实验和 s1 的难度定义放在同一个实验里对质。

思考题答案（Gemini 网页版，2026-09-19）：https://gemini.google.com/app/d7b79aa9541e966a

论文原文：https://arxiv.org/abs/2501.19393

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-18-2025-s1/NOTES.md
