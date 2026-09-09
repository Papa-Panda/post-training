# Paper 模板

## 元信息
- Title: Qwen2.5 Technical Report
- Authors / Org: Alibaba Cloud Qwen Team (Qwen2.5 / Qwen2.5-Coder / Qwen2.5-Math series)
- Link / arXiv: https://arxiv.org/abs/2412.15115 (main), companion Coder: https://arxiv.org/abs/2409.12186
- Date read: 2026-08-12
- Tags: [pretraining, coding-data, curation, synthetic-data, sft, rl-data, scaling, multilingual, moe]
- Folder: 2024_qwen2.5
- Day: 9

## 一句话总结
从 7T 拉到 18T 预训练（2026-09-09 修正：初读的"其中 code ~5.5T 来自 Qwen2.5-Coder 线"表述不准确——5.5T 语料（最终训练集 5.2T）是 Qwen2.5-Coder 自身的预训练规模，主报告只说 18T 并入了 Math/Coder 专用数据，未披露 code 占比，见第二轮复习 §4），文件级+仓库级混合、弱模型分类器过滤、版图更大且多语，1M+ SFT + 多阶段 RL（offline DPO → online GRPO；RM↔SFT 迭代进化是 Qwen2.5-Math 报告的 self-improvement 设计，见 2026-09-09 澄清与第二轮复习 §4）+ MoE Turbo/Plus，72B 与 Llama-3-405B-Instruct 打到 competitive（2026-09-09 修正：初读的"超"为夸大，论文原文用词是 competitive，见第二轮复习 §4），证明 data recipe / flywheel > 单纯参数堆砌。

## 和之前工作的关系
- **知识图谱位置**：pretrain / scaling / curation 主线的第三极，对齐 Day 7 Llama 3 (15.6T dense) 和 Day 8 DeepSeek-V3 (14.8T MoE) —— 三大 2024 开源 frontier 配方三角在此闭合。从 influence/selection 线（Day 2-5）到 synthetic 线（Day 6 Phi-1）再到 pretrain 线（Day 7-9），Qwen 是 pretrain 向 SFT/RL 跨线的桥梁。
- **接了哪篇的哪条线**：
  - pretrain/curation：接 Llama 3 的 5 级瀑布过滤 + code 专用抽取；接 DeepSeek-V3 的文档级 FIM 与"去冗余保多样"思路（2026-09-09 修正：初读写的"MinHash 0.85→0.90 / code 30%+"已在 Day08 第二轮复习中收回，非论文事实，见本轮 §4）。Qwen 把去重思路换成 file-level + repo-level 覆盖（Coder 线），code recall 用弱模型 scorer，规模拉到 18T。
  - synthetic：接 Phi-1 的 textbook 合成理念（Day 6），Qwen2.5-Math/Coder 展示大规模合成 math/code + self-improvement (Qwen2-Math-Instruct 生成 → RM 采样 → SFT → RM 迭代 → RL) 的工业化版本。
  - selection / influence：接 LESS Day 4（gradient相似度选 5%）和 DataInf Day 5（LoRA闭式 influence 1秒/条），Qwen 用 RM 打分迭代替代梯度影响——目标导向选择 vs 梯度导向选择，计算成本更低但更贴偏好。
- **补了哪个短板**：Llama3/DeepSeek 多讲 pretrain，不管 SFT/RL 怎么挑；Phi-1 只讲小模型合成；LESS/DataInf 只讲怎么选。Qwen 补上“1M+ 精细 SFT 配比 + 多阶段 RL + 长文本/结构化数据/指令跟随”如何搭，以及 MoE Turbo/Plus 如何用相同 data 做到成本/性能权衡。
- **替代/分叉/改进**：不是替代，是分叉后汇合：证明 scaling 之后 data flywheel 比 param 更重要（72B > 405B）。对 Day 7 的直接对比：Llama3 405B 需要 15.6T + annealing 才 GPT-4 级，Qwen2.5 72B 用 18T + 更好的多语/code/合成就超它；MoE 版更进一步说明推理成本可通过 data 控。
- **数量 vs 结构**：强调关系 > 数量，把新点挂到已有图上：pre-train 去重阈值、code 占比、FIM/RM 选择三轴已有，Qwen 加上第四轴“post-training 迭代”。

## 为什么今天读它
- coding data：5.5T code 专用数据集构造来自 Qwen2.5-Coder：GitHub 公有库 + web 爬的 code-related texts，file-level + repo-level pretraining，弱模型分类器/ scorer 去低质，FIM 风格、execution 过滤类比 DeepSeek 的 PSM 可直接抄。
- SFT：1M+ 样本，real-world + synthetic（code-focused LLM 生成），覆盖生成/补全/推理/修复广度，配比 balancing coding/general/math 防止 30% code 掉 MMLU。
- RL data：多阶段 RL 提升偏好对齐、长文本、多轮 agent/tool use，RM 指导采样/过滤，类似 RLHF 的 reward flywheel，为 Agentic RL Infra 的 \$/useful-rollout 提供 data 侧信号。

## 3 问回顾（Day 9 原题）
1. Qwen2.5 把 pre-training 从 7T 拉到 18T，具体怎么做 file-level + repo-level code recall 和弱模型分类器过滤低质？和 Llama 3 的 5 级瀑布 + DeepSeek 的"去冗余保多样"比，去重/质量栅栏有何不同，哪个更省算力？（2026-09-09 注：本题前提中的"DeepSeek MinHash 0.90"已在 Day08 第二轮复习中收回，非论文事实；比较时以"存在性"而非"阈值数字"为准）
2. 它的 1M+ SFT + 多阶段 RL 是怎么迭代进化（RM → SFT → 新 RM → 下轮 SFT → 最终 RM 做 RL，见 Qwen2.5-Math 的 self-improvement），和 Day 4 LESS 用 Adam 感知的 low-rank 梯度相似度选 5% 相比，Trade-off 在哪？为什么 Qwen 选 RM-based 而非 gradient influence？对你 50 万合成池，哪个更可落地？
3. 对比 Day 7 Llama 3 405B(15.6T) 和 Day 8 DeepSeek-V3 671B MoE(14.8T code 30%+ FIM 10%)，Qwen2.5 72B 用 18T 就超 Llama-3-405B-Instruct，三家的 code 占比 / 合成策略 / 多语配比 / 评测差异如何解释参效比差异？若你把 50 万合成池按 Qwen 配方做 repo 级 packing + 长上下文合成，会比 Llama 3 annealing 更省还是更贵？Infra 视角：Bloom + MinHash vs 弱模型 scorer，哪个是 18T 瓶颈？

## 核心（待填，今晚产出）
1. **Motivation**: 
2. **Data Pipeline**: 来源 → 清洗 → 合成 → 配比 → 训练
3. **Key Tricks**: 
4. **Results**:

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
- Infra 视角：

## 疑问 / 下一步

## 原文金句

## 参考
- Qwen2.5 Main: https://arxiv.org/abs/2412.15115
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122 (self-improvement RM↔SFT 迭代)

---

## 第二轮复习（2026-09-09）

> 本轮已对照三篇论文原文逐项核对初读 NOTES：主报告 arXiv 2412.15115 v1（§1/§3/§4/§5）、Coder 报告 2409.12186（§1/§3）、Math 报告 2409.12122（abstract/§1–§3）。发现三处把"推断/夸大/张冠李戴"写成"论文事实"的偏差，已在 §4 开头明示纠正；另发现初读"3 问回顾"的前提继承了 Day08 已收回的数字，已在上文加注。

### 1. 核心命题

Qwen2.5 真正解决的 data 问题，不是"怎么把预训练堆到 18T"，而是：**预训练做完之后，数据故事怎么继续**——把数据工作从一次性的"预训练配方"，拉长为贯穿全生命周期的 **data flywheel（数据飞轮）**：每个阶段的数据产出（合成数据、RM、执行反馈信号）都变成下一阶段的数据输入。

核对原文后，飞轮有三层（主报告 §3–§4），外加 Math 线的一套"递归版"：

1. **预训练层（7T→18T，§3.1 四个可操作杠杆）**：① 用 Qwen2-Instruct 模型当多维质量过滤器给 18T 打分——"上一代模型当下一代的筛子"，代际递归；② 直接并入 Qwen2.5-Math / Qwen2.5-Coder 的专用数据——专家数据反哺通用模型；③ 合成数据由 Qwen2-72B-Instruct + Qwen2-Math-72B-Instruct 生成，再用自有 general RM + Qwen2-Math-RM-72B 过滤——"RM 即质量门"；④ 用 Qwen2-Instruct 做域分类定配比：降采样 overrepresented 的 e-commerce / social media / entertainment，升采样 underrepresented 的 technology / science / academic——配比的第一步是"先知道池子里有什么"（与 Day07 Llama 3 的 knowledge classification 同构，但 Qwen 写明了哪几类）。另：预训练分阶段切换 mixture（§1 "staged to allow transitions among different mixtures"）。
2. **SFT 层（§4.1，1M+ 九轴能力数据）**：long-gen（从预训练语料 back-translation 造长文本 query + 长度约束 + Qwen2 过滤，把输出从 <2K 拉到 8K）、math（Qwen2.5-Math CoT + rejection sampling + RM + 标注答案）、coding（Qwen2.5-Coder 指令数据，~40 种语言，多智能体协作生成 + 多语言 sandbox 静态检查与单元测试）、instruction-following（**code-based validation**：LLM 同时生成 instruction + verification code + unit tests，execution feedback 做 rejection sampling——把主观能力客观化）、structured、logical reasoning（70k 新 query）、cross-lingual（高资源→低资源翻译 + 语义对齐检查）、system prompts（数百个）、response filtering（critic + 多智能体打分，**全票通过才留**）。最终 1M+ examples，2 epochs，seq 32,768，LR $7\times 10^{-6}\to 7\times 10^{-7}$ ，weight decay 0.1，grad clip 1.0。
3. **RL 层（§4.2–§4.3，两阶段）**：offline RL = DPO，约 150K pairs——专攻 **RM 难评**的能力（reasoning / factuality / instruction-following），复用"执行反馈 + 答案匹配"的白盒管线：SFT 模型对新 query 重采样，pass=正例、fail=负例，人工+自动双审；online RL = GRPO——RM 训在"多阶段 checkpoint（SFT/DPO/RL 各阶段）× 多温度"采样的响应上（响应来源刻意多样化，DPO 数据也并入），query 按 RM 打分的**方差降序**排列（高方差优先），每 query 采样 8 个响应，global batch 2048。
4. **Math 线的递归版飞轮**（Math 报告 abstract，非主报告 §4——见 §4 纠正 3）：RM（从 Qwen2-Math-Instruct 大规模采样训练）→ 迭代进化 SFT 数据 → 更强的 SFT 模型 → 迭代更新 RM → 下一轮 SFT → 最终 SFT 模型 + 终极 RM 做 RL。预训练侧同样递归：Qwen2-Math-72B-Instruct 合成数学数据 → Math Corpus v1 700B → v2 超 1T。

一句话：Qwen2.5 是"数据飞轮"的工业化存在性证明——**数据的消费者同时是数据的生产者**（Instruct 模型过滤预训练语料、生成合成数据、训练 RM 筛 SFT）；但和 Day07/08 一样，论文**零数据消融**，飞轮每个环节的边际贡献都不可归因。

### 2. 图谱位置

- **直接对比 Day08 DeepSeek-V3（重点）**：DeepSeek 讲的是"预训练**之中**"——表示层重写（文档级 FIM PSM、packing 不做 cross-sample attention 保 integrity、tokenizer 随机拆分当增强）；Qwen 讲的是"预训练**之后**"——飞轮（每个阶段产出下一阶段的数据信号）。两者正交、可叠加：DeepSeek 的表示设计 × Qwen 的飞轮门禁 = 完整配方。共同点：都不给消融、不给阈值，都是存在性证明。引用 DeepSeek 的去重结论要去 Day25 找可复现版；引用 Qwen 的飞轮环节（如"RM 迭代 SFT 有效"）则**没有任何可复现对照组**——这是 Qwen 比 DeepSeek 更难抄的一点。
- **前驱 Day07 Llama 3（两种"便宜评估器"的哲学分野）**：Llama 3 用 annealing（50% 训好的 8B + 40B tokens + 30% 新数据权重，看 GSM8k/MATH delta）评估数据源价值——评估器是**下游分数**（外部锚点）；Qwen 用 RM 迭代进化 SFT 数据——评估器是**模型自己的偏好**（自我参照）。前者的风险是跨规模迁移 gap（405B 上 annealing 增益可忽略）；后者的风险是自举偏差（RM 的偏见被迭代放大，见 §4.5）。"用便宜信号指导昂贵数据决策"是同一深层模式，参照系完全相反。
- **选择线 Day04 LESS（RM-based vs gradient-based 选择）**：LESS 用 Adam 感知的梯度相似度选目标任务 5% 数据——目标导向、精确，但要目标集梯度、贵；Qwen 用 RM 打分筛 SFT/合成数据——偏好导向、无需显式目标集，但 RM 本身要训、且有偏。Trade-off：LESS 是"贵而准"，Qwen 是"便宜而自举"。对 500k 池的含义：有明确目标任务（如 HumanEval）时 LESS 更对症；做通用 SFT 配比时 RM 更可落地——但要配 Llama 3 式的外部锚点防跑偏。
- **同门 Day16 Qwen2.5-Coder（总论 vs code 分论）**：Day09 是飞轮总论，Day16 是 code 域的实例化——5.5T 语料（最终训练集 5.2T，配比 **70% Code / 20% Text / 10% Math**）、file-level（8K seq，NTP+FIM）→ repo-level（32K，RoPE base 10K→1M）两级预训练、fastText 4 级过滤（HumanEval+MBPP 41.6%→46.8%）、合成数据 executor 验证（只留可执行的）。Coder Table 3 的配比实验是全图谱里罕见的**反单调证据**：100:0:0（纯 code）最差，70:20:10 最好——"code 越多越好"被证伪，math/text 到达阈值浓度后反哺 code。
- **后继/对立 Day15 R1（飞轮 vs 冷启动）**：Qwen 路线 = 1M SFT + 多阶段 RL（DPO 150K + GRPO）堆对齐；R1 路线 = <10k 冷启动 + 纯 RL（可验证奖励）。两者构成"后训练到底需要多少 SFT 数据"的两极对照。但注意：R1 的纯 RL 仍依赖**白盒可验证奖励**——而这正是 Qwen offline RL（§4.2）的执行反馈管线生产的东西。R1 不是 Qwen 的替代，是 Qwen 飞轮里"白盒信号"那条腿的极端化。
- **偏好线 Day26 UltraFeedback**：Qwen online RL 的偏好对来自"多阶段 checkpoint × 多温度"采样 + 人工/自动标注；UltraFeedback 是"64k prompts × 4 模型回答" + GPT-4 细粒度打分。Qwen 把**响应多样性做进采样设计**（不同训练阶段的 checkpoint 是天然的多样性来源），UltraFeedback 把多样性做进模型来源。两者互补：Qwen 教你"评估器的数据也要多样"，UltraFeedback 给你现成的多样偏好池。
- **防漏线 Day30**：Qwen §5 把去污染写进**评测章节**而非数据章节（LCS ≥ 13 tokens 且 ≥ 0.6×min 长度则删）——"防漏"是评测可信度的前置条件，不是数据质量的后置补丁。这个章节位置本身就是个观点。

### 3. 机制深挖

**(a) "上一代模型当下一代的筛子"——代际递归的数据生产关系。** §3.1 的四个杠杆里有三个是"模型生产数据"：Qwen2-Instruct 过滤 18T、Qwen2-72B-Instruct 生成合成、RM 过滤合成。这是 Day06 Phi-1（"textbook 合成"）的工业化：合成不再是"造文本"，而是"造文本 + 用 RM 做质量门"。深挖点：RM 在 Qwen 体系里同时是三个角色——合成过滤器、SFT 数据迭代器、online RL 奖励源。**一个 RM 吃三份工资，偏见也吃三份**（见 §4.5）。Phi-1 只用 classifier 选数据，Qwen 让模型既当运动员又当裁判——这是飞轮的效率来源，也是它的结构性风险来源。

**(b) 配比决策的 Qwen 版 = "先知道池子里有什么"。** Qwen2-Instruct 做域分类 → 降采样 e-commerce/social/entertainment（模板化、机器生成内容多）→ 升采样 tech/science/academic。与 Llama 3 的 knowledge classification 同构，但 Qwen 明确写了哪几类 over/underrepresented——这是配比决策里最诚实的一句话。深挖：这是 DoReMi（Day31）想自动化的手工版；Qwen 没给 domain weights 数字，DoReMi 给了学习方法。两者共享"数据决策用便宜 proxy 做"的模式：Qwen 用上一代 Instruct 模型当分类器（相对 18T 训练，便宜 2 个数量级）。

**(c) SFT 九轴的"可验证性梯度"。** 九个能力按验证方式排成谱：纯合成（long-gen back-translation）→ RM 过滤（math RS）→ 执行验证（coder sandbox、instruction-following 的 code-based validation）。最锋利的是 §4.1(4)：LLM 同时生成 instruction + verification code + unit tests，用 execution feedback 做 rejection sampling——**把"指令跟随"这个主观能力转化成了可执行验证的客观任务**。这是 Day16 execution filter 在 SFT 域的翻版，也是对你 500k 池最直接可抄的一招：凡是能写成"可执行检查"的能力，都不该用 RM/人工去评。

**(d) Offline RL 的设计哲学（§4.2，全报告最值得抄的数据架构思想）。** 按"**评估器可靠性**"切分 RL 阶段：RM 难评的能力（reasoning / factuality / instruction-following）走 offline——因为 online 的 RM 在这些能力上不可靠，所以离线阶段用"执行反馈 + 答案匹配"的白盒信号造 DPO pairs（SFT 模型 resample 新 query，pass=正例 fail=负例，~150K pairs，人工+自动双审，训 1 epoch，LR $7\times 10^{-7}$ ）。数据视角的翻译：**白盒信号能评的走 offline DPO，黑盒偏好才走 online GRPO**。这不是训练技巧，是数据管线的分诊台。

**(e) Online RL 的数据课程（§4.3）。** 两处细节：① query 按 RM 打分方差降序排列，高方差优先——高方差 = RM 最不确定的 query = 信息增益最大的 query，这是 uncertainty sampling 的 RL 版；② RM 的训练响应来自"多阶段 checkpoint（SFT/DPO/RL 各阶段）× 多温度"采样——防止 RM 过拟合到单一策略分布。深挖：这是"**评估器的数据也要多样性**"的思想——Day19 Vendi / Day24 D4 讲训练数据的多样性，Qwen 把它用在了 RM 的训练数据上。每 query 采样 8 个响应、global batch 2048、每 episode 2048 samples——这些数字定义了 GRPO 的数据吞吐形状。

**(f) 去污染门的阈值写法（§5）。** 训练序列 $\mathbf{s}_t$ 若与任一测试序列 $\mathbf{s}_e$ 满足 $|\text{LCS}|\geq 13$ 且 $|\text{LCS}|\geq 0.6\times\min(|\mathbf{s}_t|,|\mathbf{s}_e|)$ 则删除。双条件设计：13 tokens 防短串误杀，0.6×min 防长文档局部重合。注意这是 token 级 LCS，对改写式污染（paraphrase）无效——Day30 的 semantic-level 匹配正是补这块。

### 4. 边界与反例

1. **纠正初读的三个"论文事实"（本轮核对原文后的收回）**：
   - "18T 中 code ~5.5T 来自 Qwen2.5-Coder 线"——**收回**。5.5T（最终训练集 5.2T，70/20/10）是 Qwen2.5-Coder **自身**的预训练语料规模（Coder 报告 abstract + §3.1.2）；主报告只说 18T "incorporate training data from Qwen2.5-Math and Qwen2.5-Coder"（§3.1），**未披露 code 占比数字**。"18T 里 code 占多少"是推断，不是论文事实。
   - "72B 超 Llama-3-405B-Instruct"——**收回，降级为 competitive**。论文原文（abstract + §1）："demonstrates competitive performance to the state-of-the-art open-weight model, Llama-3-405B-Instruct, which is around 5 times larger"。"超"是初读的夸大；参效比故事（72B vs 5× 参数打平）成立，但动词要用对。
   - "多阶段 RL（RM↔SFT 迭代）"——**归因纠正**。主报告的后训练链条是 SFT → offline DPO → online GRPO（§4）；RM→SFT→更新RM→下轮SFT→最终RM做RL 的迭代进化是 **Qwen2.5-Math 报告**的 self-improvement 设计（Math abstract L15–19）。两者都是飞轮，但别张冠李戴。
2. **初读"3 问回顾"的前提已部分失效**：Q1/Q3 拿"DeepSeek MinHash 0.90 / code 30%+"当比较基准——这组数字已在 Day08 第二轮复习 §4 中收回（论文零命中）。上文已加注；以后引用 Day08 的去重结论，只引用"去冗余保多样"的存在性，不引用阈值。
3. **零数据消融**：18T 的四个杠杆、SFT 九轴、DPO→GRPO 两阶段——论文**没有任何一处**的消融。不能引用 Qwen2.5 证明"RM 迭代 SFT 有效""合成数据有用""两阶段 RL 优于单阶段"。Coder 的配比实验（Table 3，70:20:10 > 85:15:5 > 100:0:0）是 Coder 自己的小规模实验，且只在 code 域、只在 Coder 架构上——外推到 18T 通用语料无依据。
4. **70:20:10 的适用域**："code 越多越好"被证伪（100:0:0 最差）是坚实的；但 70/20/10 是否对 dense 72B、是否对 18T 通用语料最优，论文没做。这是"预算 × 架构 × tokenizer"的函数（呼应 Day07 复习 §4.5），不是常数。
5. **RM 的三重角色 = 三重偏见放大**：合成过滤（§3.1.3）+ SFT 数据迭代（Math 线）+ online RL 奖励（§4.3）都用 RM 家族。RM 的系统性偏见（如偏好长回答、偏好某种推理风格）会在三轮中被迭代放大，而论文**没有做 RM 偏见的审计**。这是"自举飞轮"的结构性风险；R1 的"纯 RL + 可验证奖励"路线恰好是对它的反例——用白盒奖励替代 RM，代价是只适用于可验证任务。
6. **多语跨语言迁移的"投影"代价**（§4.1.7）：高资源→低资源的翻译 + 语义对齐检查，本质是把低资源语言的 SFT 数据变成高资源语言的**投影**——低资源语言的本土表达、文化特定指令会被系统性抹平。论文没评估低资源语言的本土性损失；"版图更大"不等于"每种语言都好"。
7. **去污染阈值的武断性**：LCS ≥ 13 这个数无消融；且 token 级 LCS 对改写式污染无效（见 §3(f)）。引用 Qwen 的 decontamination 时要说明它只防原文复制。

### 5. 迁移到 coding / post-training data

**可执行实验 A — "按评估器可靠性切分"你的 SFT/RL 数据管线（抄 §4.2 的 offline/online 分诊哲学）**

1. 把你 500k 池按"有没有白盒验证器"切两类：A 类（有单元测试 / 可执行 / 有标准答案：code、math、instruction-following 的可验证子集）→ 走"执行反馈 + 答案匹配"造 DPO pairs（Qwen offline RL 版：用你当前的 SFT 模型对新 query 重采样，pass=正例、fail=负例，按 150K/1M 的比例缩放到你的量级）；B 类（开放问答、风格、偏好：无白盒信号）→ 走 RM 打分 + 人工抽检（online RL 版）。
2. 纪律：A 类**不许**用 RM 当唯一门禁（RM 在 code 上幻觉打分是已知坑）；B 类**不许**用执行信号硬套。先在 1.3B proxy 上跑通两条管线，再谈合并。
3. 通过标准：A 类 DPO 训完后 HumanEval+/MBPP+ 不掉且 Aider 类任务涨 → 分诊成立；若 B 类 RM 打分与人工抽检一致率 < 80% → 你的 RM 还没到 Qwen §4.3 的可用线，先回炉 RM 数据（见实验 B）。

**可执行实验 B — RM 训练数据的"多阶段 checkpoint × 多温度"多样性（抄 §4.3）**

1. 检查你现在的 RM（或 LLM-judge）训练数据：若只来自单一模型 / 单一温度，按 Qwen §4.3 重构——收集 SFT 前 / SFT 后 / DPO 后三个阶段 checkpoint 的输出 × 高低两种采样温度，混成 RM 训练集。
2. 评测：RM 在 held-out 的"新阶段"输出上的 pairwise 准确率是否提升——这是检验"评估器泛化"的直接指标，不是下游分数的间接推测。
3. 若提升：把"响应来源多样性"写进你的 RM 数据 recipe——这是 Day19 Vendi 的多样性思想在**评估器数据**上的落地。

**可执行实验 C — instruction-following 的 code-based validation（抄 §4.1.4，两周可跑）**

1. 对你池子里 instruction-following 子集：让 LLM 同时生成 verification code + unit tests，用 execution feedback 做 rejection sampling（Qwen 原话：generate both instructions and corresponding verification code, along with comprehensive unit tests for cross-validation）。
2. 这是把主观能力客观化的最便宜路径，比人工标注便宜 1–2 个数量级；通过标准：RS 后的子集训出的模型在 IFEval 类评测上显著涨，且与人工抽检的"指令跟随"判断一致率 > 85%。

### 6. 今天的一道思考题

综合 **Day09 Qwen2.5、Day07 Llama 3、Day04 LESS、Day15 R1**（答案不在任何一篇原文里）：

**(a) 三种"便宜评估器"的哲学对决。** Llama 3 用 annealing（8B proxy + 40B tokens + 30% 新数据权重，看 GSM8k/MATH delta）评估数据源价值——评估器是**下游分数**（外部锚点）；Qwen2.5 用 RM 迭代进化 SFT 数据——评估器是**模型自己的偏好**（自我参照）；LESS 用梯度相似度选目标任务数据——评估器是**目标梯度**（任务参照）。三者都是"用便宜信号指导昂贵的数据决策"，但便宜信号的参照系完全不同。问题：在你的 500k coding 池、固定预算下，你要决定"下一批 50k 合成预算花在哪类数据上"——设计一个三臂实验，每臂用一种评估器做决策：明确每臂的输入信号、决策规则、proxy 验证方式（不许用"训完看 HumanEval"当验证，那是作弊）；并论证：在什么条件下"自我参照"的 RM 会系统性跑偏（提示：§4.5 的三重偏见放大 + R1 用白盒奖励替代 RM 的反例），以及你怎么用 Llama 3 式的外部锚点给 RM"上刹车"（给出刹车的具体触发条件，不能只说"人工抽检"四个字）。

**(b) SFT-vs-RL 的数据量之争。** Qwen2.5 路线：1M+ SFT + 多阶段 RL（DPO ~150K pairs + GRPO）；R1 路线：<10k 冷启动 + 纯 RL（可验证奖励）。两篇合在一起，恰好构成"后训练到底需要多少 SFT 数据"的对照实验——但注意两个不对齐点：① 两者优化的目标能力不同（Qwen 是通用 instruct 能力：长文本/结构化/指令跟随/多语；R1 是数学/代码推理）；② R1 的纯 RL 依赖"可验证奖励"这一白盒信号。从数据视角回答：(1) 画出"能力类型 × 评估器可靠性"二维图，把 Qwen 的九轴 SFT 能力和 R1 的推理能力放进去，标出哪类能力适合"大 SFT + RM"，哪类适合"小冷启动 + 纯 RL"，并说明 Qwen §4.2 的 offline/online 切分正好落在这张图的哪条分界线上；(2) 对你自己的 coding 数据工作：如果目标是"仓库级 agent 能力"（SWE-bench 类），你会选 Qwen 式还是 R1 式？给出数据管线设计（冷启动数据从哪来、可验证奖励怎么做、SFT 占多少 token 预算），并论证为什么"中间路线"（比如 100k SFT + RL）可能两头不靠（提示：想想 R1 冷启动数据的"格式"作用 vs Qwen SFT 数据的"能力覆盖"作用，两者缺一不可时中间量意味着什么）。

---

**论文原文**：https://arxiv.org/abs/2412.15115
**GitHub NOTES**：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-09-2024-qwen2.5/NOTES.md

