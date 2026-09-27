# Day35 Let's Verify Step by Step — NOTES

## 元信息
- Title: "Let's Verify Step by Step"
- Authors / Org: Lightman et al. / OpenAI
- Link / arXiv: https://arxiv.org/abs/2305.20050
- Date read: 2026-09-27
- Tags: [rl-data, verifier-data, process-supervision, prm, reasoning-data, quality]

## 一句话总结
PRM 训练数据的开山之作：800K 条 step-level 人工标签（75K 条解法 / 12K 道题）训出的 process-supervised reward model，在 best-of-N 搜索下以 $78.2\%$ 打赢 outcome supervision 的 $72.4\%$ ；核心数据洞见是"首错截断标注 + 诱人错解主动学习"，把昂贵的人工标注预算换成 $2.6\times$ 的数据效率，还顺带拿了个"负对齐税"。

## 核心
1. **Motivation**: 复杂多步推理里，一个逻辑错就足以让整道题报废——而 SOTA 模型恰好经常犯这种错。训练可靠的 reward model 有两条路：outcome supervision（只看最终答案对错）和 process supervision（每一步都给反馈）。Uesato et al. 2022 在小学数学上做过对比，结论是两者打平；但那次基模型弱、人工反馈少、题太简单。本篇用更强的 GPT-4 基座、800K step-level 人工标签、在更难的 MATH 数据集上重做头对头对比。另有一层对齐动机：process 监督奖励的是"人类认可的推理链本身"，而不是把"结果对"当对齐的代理指标。
2. **Data Pipeline**: 生成器 → 人工逐步标注 → PRM 训练 → best-of-N 搜索评测：
   - **生成器**：GPT-4 基座在 MATH 上做 1 epoch 微调，目的只有一个——学会"换行分隔的 step-by-step 输出格式"（few-shot 生成解法 → 只保留最终答案正确的 → 微调），不教新能力；
   - **人工标注**：labeler 对每一步打 positive / negative / neutral；**只监督到第一个错步就停**（first-error-stop），之后的步不再标——这样 outcome 和 process 的信息差恰好等于"错步的位置"，也让人工成本可比（判整解对错 ≈ 找出第一个错步）；neutral 标签是给"说不清"的出口，模糊数据的处理后置到测试时；
   - **PRM800K 规模**：800K step-level 标签，覆盖 75K 条解法、12K 道题（含 4.5K 道 MATH 测试题混入训练集防过拟合；评测只用剩下的 500 道代表性子集）；
   - **主动学习选样**（最关键的数据工程）：不均匀送标。策略是 "convincing wrong-answer"——把当前最优 PRM 打高分、但最终答案错的解法优先送人工标注；采集中途多次用新数据重训 PRM 再选样。小规模复刻实验用 $80\%$ 最"诱人"错解 + $20\%$ 最"诱人"其余解的配比，估计出 $2.6\times$ 数据效率（拟合斜率对比均匀送标）；
   - **PRM 训练**：在每步末 token 后预测该步正确性，标准 LM 训练流程；推理时解法分数 = 各步正确概率的**乘积**（"每一步都对"的联合概率），一次前向得到全部 step 分数；
   - **ORM 对照**：每题 100 条均匀采样、训整解对错分类器（最终答案自动判分，混有"推理错但答案对"的误判噪声）；训练集比 PRM800K 大一个量级、且无交集——是"各自形态的最强版本"对比；
   - **混杂因素剥离**（小规模实验）：用 PRM_large 当标注 oracle 训小模型，三组对照——process（PRM_large 逐步监督）/ outcome（PRM_large 整解监督）/ outcome（最终答案判分），**process 在所有数据规模下都赢**，排除了"数据集不可比"和"答案判分误判"两种解释。
3. **Key Tricks**:
   - **首错截断标注**：只标到第一个错步——信息差精确可控、人工成本可比；错步之后的"错误连锁"信息被主动丢掉（后续工作如自动标注补全正是从这里切入）；
   - **convincing wrong-answer 选样 + 迭代重训**：标注预算只花在"当前模型最被骗"的样本上； $80\%$ / $20\%$ 配比防止数据集过度偏向错解（作者试过把 ORM 训在这种偏置数据上，性能反而更差）；
   - **neutral 标签**：允许标注员对模糊步"弃权"，测试时可灵活当正或负——把"模糊"的裁决从标注时推迟到使用时；
   - **乘积式解法打分**：PRM 分数 = 各步正确概率连乘； $N$ 越大，PRM 相对 ORM 的优势越拉开——搜索越深，信用分配（credit assignment）越重要，这正是 step-level 信号的用武之地。
4. **Results**:
   - Best-of-1860（500 道 MATH 代表性子集）：PRM $78.2\%$ vs ORM $72.4\%$ vs majority voting $69.6\%$ ；
   - OOD 泛化（234 道最新 AP/AMC STEM 真题，预训练后发布、防污染）：PRM aggregate $72.9\%$ vs ORM $63.8\%$ ；
   - 主动学习：约 $2.6\times$ 数据效率提升；
   - 开源 **PRM800K**：完整的 800K step-level 人工标签数据集；
   - 对齐意义：process supervision 是"负对齐税"——更安全的监督方式反而性能更高，降低了采用对齐方法的阻力。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：coding 天然有 step 结构（行/hunk），而单测/CI 只是 outcome 信号——"单测通过但推理胡扯"的错解正是 ORM 的盲区，也是 Day33 STaR 思考题里悬而未决的筛法问题。PRM 式 recipe：用 convincing wrong-answer 选样——模型高置信但单测失败的 patch 优先送审，对"第一个引入 bug 的 hunk"做首错截断标注；rerank 时候选 patch 分数取各 hunk 正确概率乘积。neutral 标签对应 code review 里"这行可疑但不确定"的后置处理。
- Infra 视角：主动学习把最贵的人工标注预算定向花在"模型最被骗"的样本上（ $2.6\times$ 效率）；但论文 4.2 末尾实证过一个坑——采集中途**迭代重训 selector 出现不稳定、没有涨点**，数据飞轮里别盲目加"在线重训选样"环节，先离线验证稳定性。

## 疑问 / 下一步
- 首错截断丢掉了错步之后的全部信息：错误连锁本身是否也有训练信号？后续自动 step 标注（如 MATH-SHEPHERD 类方法）是不是把这块补上了？
- 800K 标签是纯人工（数十人标注团队量级的工作量）：coding 场景里，哪些 step-level 负标签能用"单测失败定位到 hunk"自动构造，哪些必须人工？PRM800K 的人工 recipe 里哪个环节最先被自动化替代？
- PRM 只做了 best-of-N rerank，明确没进 RL 训练 generator（论文 scope 外）——把 PRM 信号接进 PPO/GRPO 会发生什么？Day37 DAPO / Day39 Kimi k1.5 会部分回答。

## 原文金句 (1-2句)
> Process supervision makes credit assignment easier, and we believe that this explains its strong performance.

> Our results show that process supervision in fact incurs a negative alignment tax.

## 思考题
1. Day33 的 STaR 只用"答案对错"做数据阀——PRM 的 step-level 标签是不是它缺的那块拼图？coding 里"单测通过但推理胡扯"的样本，用 convincing wrong-answer 选样 + 首错截断标注能筛出来吗？
2. PRM800K 花 800K 人工 step 标签换 $2.6\times$ 数据效率 + 负对齐税：这笔账在 coding data 上划算吗？哪些 step-level 负标签可以用"单测失败定位"自动造，哪些必须人工？
