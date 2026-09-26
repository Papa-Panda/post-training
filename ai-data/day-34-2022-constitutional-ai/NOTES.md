# Day34 Constitutional AI — NOTES

## 元信息
- Title: "Constitutional AI: Harmlessness from AI Feedback"
- Authors / Org: Bai et al. / Anthropic（51 作者）
- Link / arXiv: https://arxiv.org/abs/2212.08073
- Date read: 2026-09-26
- Tags: [rl-data, preference-data, rlaif, constitution, critique-revision, curation]

## 一句话总结
RLAIF 的源头：人类监督从"逐条标有害/无害"收缩为"写一张原则清单（constitution）"——SL 阶段模型按随机抽到的原则自我 critique+revision、自产对齐数据微调；RL 阶段模型按原则对两条回复做 AI 偏好标注、训练 PM 再 PPO，全程零人类有害标注，human-judged harmlessness 显著超过当时的人类 RLHF 基线（HH-RLHF），是把"规则"编译成"数据"的第一份完整 recipe。

## 核心
1. **Motivation**: RLHF 的 harmlessness 标注扩展性差：要让人逐条判断"哪条回复更有害"，贵、慢、且标注员接触大量有害内容本身是负担。问题是：能不能把人类监督"上移一层"——只写原则（自然语言），让模型自己把原则翻译成标注？这就是 constitution 的由来：对齐目标可审计、可版本化，而不是藏在几十万条人类标注的隐式分布里。
2. **Data Pipeline**: 两段式，全部数据自产：
   - **SL 阶段（self-critique → revision）**：取 helpful RLHF 后的初始模型 + 一批 red-team prompts（人类写的 + 模型 few-shot 扩写的）。每条 prompt：模型先生成初始回复 → 随机抽一条 critique 原则（16 条里抽 1，如"这条回复是否助长了非法行为？"）→ few-shot 格式生成 critique → 再按 revision 指令生成修订版。**只保留最终修订版作为 SL 目标，丢掉原始回复**，在 (prompt, revision) 上微调。数据阀：过滤掉修订版退化成模板式拒绝/重复套话的样本——拒绝容易学，修订难，要的是"不回避但讲清立场"的修订质量。
   - **RL 阶段（AI 偏好标注 → PM → PPO，即 RLAIF）**：对每条 prompt 从 SL 模型采样 2 条回复 → 随机抽一条偏好原则（16 条，如"哪条回复是更明智、有道德、礼貌友好的人会说的？"）→ 模型做选择题式判断（"A 好还是 B 好？"，带 CoT 推理再取 (A)/(B) token 概率得偏好标签）→ 训练 preference model → PPO。偏好数据里的人类成分是零，原则是唯一的注入点。
   - 关键点：critique/revision 和偏好判断都用 few-shot + CoT——CoT 不只涨标注质量，还让 AI 的"判决理由"可读，这是"可审计对齐"的雏形。
3. **Key Tricks**:
   - **原则即数据规格**：16+16 条自然语言原则是唯一的"人工制品"，每条 prompt 随机抽 1 条用——相当于用采样把原则分布注入数据分布，改原则 = 改数据分布，不用重标；
   - **SL 只训 revision、丢弃原回复**：训练信号是"按原则改完之后的样子"，模型学到的是"自我修正的终态分布"而不是"先犯错再改"的过程；
   - **AI 标注用选择题 + CoT 提取 token 概率**：不是让模型直接打分（打分不稳定），而是二选一 + 读 (A)/(B) 概率，偏好标签变成可校准的概率信号；
   - **non-evasive 是刻意的数据目标**：修订/偏好原则明确奖励"正面回应并解释反对理由"而非"一拒了之"，RLAIF 训练出的助手对有害请求会解释立场而不是模板拒绝——这是数据设计直接决定产品行为的例子。
4. **Results**:
   - 人评 Elo：harmlessness 上 CAI 73 vs HH-RLHF（人类标注 RLHF）63；helpfulness 上 HH-RLHF 62 vs CAI 56——无害性大胜、helpfulness 小输，tradeoff 明确；
   - crowdworker 在 red-team prompts 上显著更偏好 CAI 回复；AI 标注训练的 PM 性能接近人类标注训练的 PM（CoT 对一致性提升明显）；
   - 数据意义：证明了 harmlessness 这条维度可以完全不依赖人类有害标注达到 SOTA——RLHF 的"人"不是不可替代的，至少在这一维上。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：把代码规范/安全 checklist 写成"constitution"（如"不引入 SQL 注入面""错误处理不吞异常"），让模型对自生成的代码做 critique→revision，自产 (bad code, revised code) 对做 SFT 或偏好数据——等于把 code review 规则编译成数据，比逐条人工 review 可扩展；Day33 STaR 是"答案对错"做阀，CAI 是"原则符合度"做阀，coding 里两者可以叠：单测做答案阀，constitution 做风格/安全阀。
- Infra 视角：CAI 把最贵的人力环节（有害标注）换成了最便宜的可版本化资产（一页原则）+ 模型 inference——数据成本结构从 O(标注条数) 降到 O(原则条数)；代价是"回音室"风险（标注模型的偏见被放大进 policy），需要独立的红队/人评做外部校验，这正是 Day35 PRM step-level 监督和 Day36 RLAIF-vs-RLHF 对比要回答的。

## 疑问 / 下一步
- 16 条原则是作者手写的——原则的"覆盖度"和"冲突"怎么度量？两条原则打架时（helpful vs harmless）采样机制实际学到的是什么分布？
- AI 偏好标注与人类标注的一致性在 harmlessness 之外（helpfulness、honesty）是否同样成立？Day36 的 RLAIF vs RLHF 头对头对比会给出答案。

## 原文金句 (1-2句)
> We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'.

> These methods make it possible to control AI behavior more precisely and with far fewer human labels.

## 思考题
1. CAI 把"逐条人工标注"换成"一页自然语言原则 + AI 自产数据"——这套"规则前置"范式能搬到 coding data 吗？比如把代码规范写成 constitution，让模型对自生成的代码做 critique→revision，和 Day33 STaR 的"单测答案阀"叠起来用？
2. RLAIF 全程零人类有害标注，但 AI 标注的偏见会被 PPO 放大成"回音室"——如果原则本身有盲区，数据管线里哪个环节能发现？Day36 的 RLAIF vs RLHF 对比、Day35 的 step-level 监督，哪个更可能是解法？
