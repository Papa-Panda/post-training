# Paper 模板 - Day 23

> 自动生成骨架 2026-08-23，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: LIMA: Less Is More for Alignment
- Authors / Org: Zhou et al. / Meta AI, Carnegie Mellon University, University of Southern California, Tel Aviv University
- Link / arXiv: https://arxiv.org/abs/2305.11206
- Date read: 2026-08-23
- Tags: [sft, alignment-data, data-quality, diversity, curation, less-is-more, coding-data]
- Folder: day-23-2023-lima
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-23-2023-lima

## 一句话总结
用仅 1,000 条经过来源、风格与多样性精心策展的 SFT 样本验证“对齐数据质量与覆盖比数量更关键”，把 Day17 LIMO / Day18 s1 的少即是多现象追溯到通用对齐数据的早期起点。

## 和之前工作的关系

> 知识图谱位置：对齐极简主线起点，Day21 Self-Instruct（合成扩量）→ Day22 Evol-Instruct（提升复杂度）与 Day23 LIMA（人工精选、质量优先）形成分叉；随后汇入 Day17 LIMO / Day18 s1 的少量高质推理数据。

- **接了哪条线：**接 Day17 LIMO、Day18 s1 的“少即是多”线，但时间上是它们的前身；同时和 Day21/22 的 synthetic scale 线形成“扩量 vs 精选”的正面对照。
- **补了哪个短板：**此前知道如何自举更多指令、如何演化复杂度，却缺少“极少量数据到底需要满足什么质量与覆盖条件”的通用对齐基线。LIMA 把来源策展、回答风格、任务多样性与近重复控制放到中心。
- **替代 / 分叉 / 改进：**它不替代 Self-Instruct / Evol-Instruct，而是分叉出 quality-first 路线；可把合成池先扩量，再用 LIMA 式门槛和 Day20 DEITA 的质量×复杂度×多样性筛成小而强的数据集。
- **对之前 Day X 的直接对比：**vs Day20 DEITA，LIMA 主要依靠人工来源与策展原则构造 1k 高质集，DEITA 则把复杂度、质量与多样性评分自动化后选 6k；今天重点判断 LIMA 的人工标准哪些能转成可规模化的数据门禁。

## 为什么今天读它

前两天沿 Day21→Day22 学了“从少量种子合成更多、更复杂指令”，今天需要补相反但关键的一半：什么时候不该继续加量。对 coding SFT，可把高质答案、任务覆盖、风格一致性和去重做成小型 gold set；对 RL data，可让它作为候选题/轨迹进入可验证池之前的质量锚点。本文只研究数据选择与策展，不展开训练算法。

## 今天的 3 问
1. LIMA 的 1,000 条数据具体由哪些来源和策展标准组成？质量、任务覆盖、回答风格与去重各自如何定义，哪些标准最可能贡献主要增益？
2. 对比 Day21 Self-Instruct / Day22 Evol-Instruct 的“合成扩量与复杂度演化”，LIMA 的 quality-first 路线在哪些任务上更强，在哪些长尾覆盖上会吃亏？能否组合成“先扩池、再精选”的数据流水线？
3. 对比 Day20 DEITA 的自动三因子选数，哪些 LIMA 人工标准能自动化为 coding data 的门禁（可执行性、边界条件覆盖、答案简洁度、近重复），并作为 SFT / RL data 的小型 gold set？

## 核心
1. **Motivation**: [待读后填写] 为什么大规模 instruction-tuning 数据未必必要？高质量少样本对齐要解决什么数据问题？
2. **Data Pipeline**: [待读后填写] 来源选择 → 人工/社区答案策展 → 质量与风格门槛 → 多样性和去重 → 形成 1k 数据集 → 如何评估。
3. **Key Tricks**: [待读后填写] 记录数据来源、筛选准则、任务分布、回答风格、去重与质量审计；不展开训练算法。
4. **Results**: [待读后填写] 只记录数据规模、数据消融、质量/覆盖评估和 downstream 对照，不展开 optimizer / RLHF 等算法细节。

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

## 第二轮复习（2026-09-23）

> 本轮复核：arXiv:2305.11206 PDF 全文（§2 数据、§3 训练、§4 人评、§5 消融、Table 1）逐项核验。初读 NOTES 的"核心"小节全是 [待读后填写] 骨架，本轮补齐数据事实并修正两处笼统表述：① "1k 人工精选"精确化为社区 750 + 手写 250 的配比（Table 1）；② 评测口径不是"打赢 GPT-4"，是"43% 等同或更优"。

### 元信息修正

- **数据构成（Table 1，精确到来源）**：训练集 1,000 条 ≈750,000 tokens —— StackExchange(STEM) 200 + StackExchange(Other) 200 + wikiHow 200 + r/WritingPrompts 150 + Natural Instructions 50 + 作者组 Group A 手写 200；dev 50（Group A 手写）；test 300（r/AskReddit 70 + Group B 手写 230）。
- **风格过滤器（§2.1，具体规则）**：回答限 1200–4096 字符窗口；去第一人称（I/my）；去自引用（as mentioned / stack exchange）；去链接、图片、HTML，只保留代码块与列表。StackExchange 按 $\tau=3$ 温度跨 75 个 STEM + 99 个其他社区均匀采样，问题要求顶分 ≥10 且标题自包含。
- **手写回答的格式假设（§2.2）**：统一"先承认问题、再给答案"（acknowledgment + answer）结构；论文明说这被假设能帮模型形成 chain of thought（类比 let's think step by step）——风格统一不只是审美，是生成过程的脚手架。
- **训练配置（§3）**：LLaMA-65B，15 epochs，AdamW，lr $1\times10^{-5}$ 线性衰减到 $1\times10^{-6}$ ，batch 32，EOT 说话人分隔 token，residual dropout 0→0.3。**perplexity 与生成质量不相关**，checkpoint 在 50 条 dev 上人工选 5–10 epoch —— 便宜代理在小数据对齐上不可靠的原文证据。
- **评测口径（§4）**：人工偏好 LIMA vs GPT-4 有 43% 等同或更优、vs Bard 58%、vs DaVinci003 65%；打赢 52K 训的 Alpaca-65B；绝对打分 88% 满足需求、50% 评为 excellent；GPT-4 当裁判复现结论。
- **消融（§5，7B + ChatGPT 1–6 Likert）**：多样性（过滤后 SE 2k 显著 > 同质 "how to" 的 wikiHow 2k）；质量（filtered vs unfiltered SE 差 0.5 分）；数量（2K→32K，16 倍数据量，ChatGPT 打分 plateau，doubling 无增益）。脚注 5：**7B 上 1,000 不稳定，至少 2,000 才稳** —— 剂量下限与容量相关。
- **多轮对话（§6）**：零对话样本也能多轮（但 6/10 在 3 轮内崩）；+30 条手工对话链 → excellent 率 45.2%→76.1%、失败 15/42→1/46。
- **13 条毒性 prompt**：配拒答式回答，safety 行为用少量 SFT 示范直接"打补丁"装进去。
- **Superficial Alignment Hypothesis 原文（§2）**：模型的知识与能力几乎全在预训练习得，对齐只是教"与用户交互时该用哪个格式子分布"；推论是小样本即可。注意 9/10 问答已定调："Superficial"修饰的是对齐，不是能力 —— 证明"对齐可以很薄"，不是"学习可以很薄"。

### 一句话总结

用 1,000 条"社区 750 + 手写 250"、风格统一、任务多样的 SFT 数据在 65B 上验证 Superficial Alignment Hypothesis —— 对齐只是"格式子分布选择"，消融证明多样性与质量是有效轴、纯数量 16 倍无增益；连多轮对话能力都能被 30 条手工对话链"点亮"。这是"少即是多"在通用对齐域的原点，也是 LIMO / s1 的推理极简的前身。

### 和之前工作的关系

- **vs Day21 Self-Instruct（论文自己做的 head-to-head）**：175 种子→52K 自举（造）vs 1k 人工策展（选）。§4 直接对比：LIMA(1k) 打赢 Alpaca-65B(52K) —— 精选人工 > 自举机器，是 quantity 路线第一次被正式记败绩。两条路线的分叉点：Self-Instruct 赌"生成器够强"，LIMA 赌"人的品味够准"。
- **vs Day17 LIMO / Day18 s1（三姐妹，同 1k 尺度同纯 SFT，教的东西不同）**：LIMA 教*助手口吻*（向外看，信人的品味：来源质量/风格统一/任务多样）；LIMO 教*认知模板*（向内看，信模型的失败率：难度级联）；s1 教*思考行为*（信过程分布：双失败率+长度+配额）。三者的选择器哲学构成"人工品味 → 模型行为 → 过程分布"的光谱；共享同一个未被测量的软肋：结论都绑定强基座（65B/32B），"完备性/容量"前提都没度量 —— LIMA 脚注 5 的"7B 需 ≥2k"正是 LIMR §3.3 处决 7B 模板的先声。
- **vs Day20 DEITA（人工原型 → 自动化）**：LIMA 的三个旋钮（多样性 $\tau=3$ 分层采样、质量 1200–4096 字符窗+风格滤、数量 plateau）是 DEITA 的 $d/c/q$ 三因子的"人工原型"；DEITA 把策展自动化。但 DEITA Table 2 的教训反噬 LIMA：人工"品味"在复杂度轴上没有显式定义 —— LIMA 选的是"好回答"，不是"难问题"。
- **vs Day11 LIMR（选择器的两极）**：LIMA 选输入质量（信人，curation），LIMR 选对学习过程的贡献（信 rollout 轨迹，valuation）。"少即是多"成立的条件从来不是"数据少"，而是**选择标准与学习阶段对齐** —— LIMA 的阶段是"对齐"（格式），LIMR 的阶段是 RL（轨迹）。
- **vs Day26 UltraFeedback / Day13 DPO-Gap（偏好线的对照）**：LIMA 证明偏好建模不是必需的 —— 无 RL、无人类偏好建模，纯 SFT 打赢 RLHF 的 DaVinci003。偏好数据底座是另一条路，但 LIMA 的 13 条拒答示范说明 safety 行为可用少量 SFT 补丁直接装 —— 这是后来所有 safety SFT 的雏形。

### 核心

1. **Motivation**：2023 年的默认答案是"对齐 = 百万级指令 + RLHF"。论文问：给定强预训练基座，对齐阶段到底在学什么？假说回答：只学"该用哪个格式子分布"，知识已在预训练 —— 于是问题从"堆多少数据"变成"多样性×质量两个轴各要多少"。
2. **Data Pipeline**：三源社区挖掘（SE 高分问答 + 风格滤 400 条；wikiHow 先分 19 类再抽文章 200 条；r/WritingPrompts 人工精选创意写作 150 条）→ 作者手写 200（Group A，统一助手口吻）+ Natural Instructions 50（增多样性）→ 13 条毒性 prompt 配拒答回答（safety 补丁）→ 1k 定版（≈750k tokens）。训练只一句：65B 纯 SFT，15 epochs，AdamW，EOT 分隔，dev 人工选 checkpoint（ppl 不可用）。
3. **关键机制深挖**：
   - **多样性的操作化**： $\tau=3$ 温度采样是"均匀覆盖"的可执行定义；wikiHow"先抽类、再抽文章"是分层采样；SE 按 STEM/Other 先分层。这三招是后来配额式多样性（LIMO 战略采样、s1 的 MSC 50 域配额）的鼻祖 —— LIMA 把"多样性"从形容词变成了采样协议。
   - **质量过滤器的双层语义**：长度窗（1200–4096）是"信息量下限 + 注意力上限"的工程近似；去第一人称/自引用是"助手身份一致性"约束。注意它筛的是*角色*，不是*难度* —— 和 DPO-Gap 的"难度"、DEITA 的"复杂度"正交。
   - **对齐 scaling law 的形状**： $Q \approx f(\text{diversity}, \text{quality})$ ， $\partial Q / \partial N \approx 0$ （ $N \ge 2\text{K}$ 后）。这不是"数据无用"，是"对齐阶段的边际信息在多样性维度，不在重复采样维度" —— 重复采样只是在同一格式子分布里加密度。
   - **30 条对话链"点亮"多轮的含义**：零对话样本的模型已经能多轮（预训练里有对话格式），30 条链只是把"该用哪个子分布"的开关拨过去 —— 这是 Superficial Alignment Hypothesis 最干净的行为证据：能力在预训练，对齐只做选择。
   - **ppl 脱钩的警示**：§3 明说 perplexity 不 correlate 生成质量 —— 小数据对齐上，训练 loss 是坏的早停信号。这和 9/9 问答"自我参照评估器跑偏"是同一个病：便宜代理在分布外不可靠。

### 边界

1. **容量前提未测**：主结果 65B，消融 7B；脚注 5 承认 7B 上 1,000 不稳定需 ≥2,000。"1k 足够"绑定强基座 —— 和 LIMO/s1 的"预训练完备性未度量"是同一个软肋，只是 LIMA 把它写进了脚注。
2. **去污染只有声明没有方法**：§2.2 脚注承认组间有接触、存在 shared priors；test 300 里 230 来自 Group B 手写。"泛化到未见任务"的声明打折扣（Day30 视角：surface-level 去重都没做）。
3. **评测域局限**：43% vs GPT-4 是"等同或更优"，等同占大头；评的是通用助手问答，不是 coding/数学等可验证任务 —— LIMA 的结论不能直接搬到 coding SFT（code 的"对"需要执行，不是品味）。
4. **风格统一的代价**：统一口吻 = 分布收窄；对需要多风格/多角色的任务可能是负迁移。论文没测"风格统一"本身的消融（§2.2 只说 preliminary experiments 显示有提升）。
5. **30 条链是单点证据**：10 次 live 对话的小样本；0→30 的跳变，5/10/20 条的剂量曲线没测 —— "点亮"的阈值形状未知。

### 迁移到 coding / post-training data

**可执行的映射："LIMA 式 1k coding 对齐 gold set + 30 条工具对话链点亮"（2–3 周可跑通）**

1. **组候选池**：OSS-Instruct 75K（或内部 code SFT 池），宁滥勿缺。
2. **LIMA 三门禁的 code 转译**：① 风格统一门禁 —— 回答结构强制 problem-restatement → solution → explanation 三段（对应 LIMA 的 acknowledgment+answer 脚手架），去第一人称/自引用；② 多样性配额 —— 语言 × 题型（算法/调试/补全/解释）分层， $\tau$ 温度均匀采样（抄 §2.1）；③ 质量门 —— parser + 编译 + 隐藏测试三级（Day16），替代 LIMA 的字符窗：code 的"质量"必须可执行定义，不能靠品味。
3. **验收 head-to-head（复刻论文打赢 Alpaca-65B）**：同基座（如 Qwen2.5-Coder-32B），1k 精选 vs 52K 全量 SFT，比 HumanEval / MBPP / SWE-bench-lite delta。**可证伪**：若 1k 打不赢，说明 code 对齐的"完备性前提"不满足（通用问答知识预训练已编码，仓库/工具知识没有）—— 这本身就是对 Superficial Alignment Hypothesis 在 code 域边界的一次测量。
4. **30 条对话链移植**：人工写 30 条"issue → 读代码 → 写复现 → 修 → 跑测试"的多轮工具交互链（抄 §6），测能否把单轮 coding SFT 模型的多轮工具对话失败率打下来 —— 对标论文 15/42→1/46。若成立，仓库级 agent 的 Route A 前置（9/10 问答）就有了"LIMA 式 30 条"的最小剂量依据。

### 思考题（综合 Day23 / Day17 / Day18 / Day11）

- **(a) 风格统一 vs 过程质量的归因**：LIMA 说"回答风格统一"是关键（表面），LIMO/s1 说"过程完备/自我修正"是关键（行为）。设计 2×2：同一题池，{风格统一，风格杂} × {过程好（含检查/回溯/修正），过程灌水}，四组各 250 条，同基座 SFT，比下游 delta。可证伪判据：若"风格杂 + 过程好" ≈ "双好" >> "风格统一 + 过程差" → s1/LIMO 赢，LIMA 的风格统一只是"过程脚手架"的廉价代理；若"风格统一"主效应显著 → 对齐阶段格式信号独立有效，LIMA 的假说在过程质量之外还有残差解释力。追问：论文 §2.2 声称 acknowledgment+answer 结构"helps form chain of thought"，但从没消融过它 —— 把这一结构单独拿掉看 delta，是补上论文自己欠的实验。
- **(b) 对齐数据的最小剂量 × 容量**：LIMA 脚注 5（7B 上 1k 不稳定、至少 2k）× LIMR §3.3（7B 上 s1/LIMO 1k 模板 AIME 15.8 失效）× 主结果都在 32B/65B。实验：基座 {7B, 32B, 65B} × 数据量 {1k, 2k, 4k, 8k} × {LIMA 式策展，随机}，画对齐质量（人工偏好 / ChatGPT Likert）曲线。判据：若"最小稳定剂量"随容量单调下降 → "少即是多"必须写成容量函数 $N_{min}(C)$ ，1k 不是常数；若某容量下策展 8k 仍不敌随机 8k → 策展标准与该容量错配（呼应 LIMR 的"选择标准与阶段对齐"，这里是与容量对齐）。这是把三篇的容量 caveat 合成一个可测函数的实验。

论文原文：https://arxiv.org/abs/2305.11206

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/ai-data/day-23-2023-lima/NOTES.md
