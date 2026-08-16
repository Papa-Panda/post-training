# Paper 模板 — Day 15 DeepSeek-R1

## 元信息
- Title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- Authors / Org: DeepSeek-AI (DeepSeek-R1 Team)
- Link / arXiv: https://arxiv.org/abs/2501.12948
- Date read: 2026-08-15
- Tags: [rl-data, coding-data, reasoning, sft-vs-rl, cold-start, synthetic-data]
- Day: 15

## 一句话总结
用 <10k 合成冷启动 + 纯 RL（可验证奖励）让 base 模型自发涌现长链推理，SFT 只起格式稳定，真正泛化靠 RL，证明 SFT memorizing vs RL generalizing 在 coding/reasoning 上的分水岭。

## 和之前工作的关系
- **知识图谱位置**：post-train RL 主线的集大成，对接 2025_limr (RL 少即是多) 的“选难例”——R1 冷启动 10k 就是 LIMR 式的硬例筛选；对接 2024_superfiltering (SFT weak-to-strong) 的对比——SuperFiltering 说小模型选 SFT 数据管用，R1 说 SFT 只能稳格式，选对难例后 RL 才能超 SFT；对接 2023_phi-1 (合成教科书) 线——R1 的冷启动合成推理轨迹就是 Phi-1 教科书思想的 RL 版本，但从“教”变“写出题过程”。
- **接了哪条线**：influence / selection 线（LESS/Day4, DataInf/Day5, LIMR/Day11, SuperFiltering/Day12, DPO-gap/Day13） → R1 把 selection 从 SFT influence 换成 GRPO 奖励驱动的在线选择；synthetic 线（Phi-1/Day6 → Llama3/Day7 → DeepSeek-V3/Day9 → Qwen2.5/Day10） → R1 合成不再是预训练教科书，而是 RL 冷启动的思维链。
- **补了哪个短板**：之前 LESS/SuperFiltering 只谈 SFT 选择，LIMR 只谈 RL 选但没说 RL 要不要 SFT 热身；R1 补上 SFT vs RL 对比的空白，给出“冷启动 SFT 稳住格式→纯 RL 泛化”的 recipe，解决你之前问的“LESS 到 RL 的 gap 为什么存在”【之前对话 2026-08-13】。
- **替代/分叉/改进**：不是替代 LIMR，而是改进/分叉——LIMR 用 LIM 度量选 1.3k，R1 用人工+启发式选 <10k 冷启动 + 8k RL；相对 Phi-1 的全 SFT 合成，R1 分叉到 RL 奖励筛选。
- **对比 Day X**：
  - vs Day11 LIMR：都少即是多（1,389 vs <10k 冷启动），但 LIMR 用轨迹对齐选，R1 用难+多样+可验证选；LIMR AIME +16.7%【8617116200514826664†L23-L26】，R1 AIME 79.8% 超 o1
  - vs Day12 SuperFiltering：SuperFiltering 125M 小模型 IFD 选 SFT 给 7B【4921601271219063014†L16-L19】，R1 反向——SFT 小而弱，RL 大而强，证明 SFT 选择力 ≠ RL 选择力
  - vs Day6 Phi-1：Phi-1 6B 精筛+1B 合成教科书训 1.3B 50.6% HumanEval，R1 合成 10k 推理轨迹只做格式预热，后面靠执行/答案可验证奖励 RL 刷上去，合成→验证 的升级

## 为什么今天读它
跟 coding data / SFT / RL data 的直接连接：R1 的可验证奖励（数学答案、代码单元测试）正是 coding RL data 的金标准——你现在 50 万合成池的最大问题是“怎么知道合成题是好的”，R1 给出 execution-verified reward + 冷启动分层，SFT 只记模板，RL 才会写出新解法。

## 今天的 3 问
1. R1 的 <10k 冷启动如何选的？人工标注的难/多样/可验证三原则，换成你 50 万 coding 池，能否用 LIMR 的 LIM 轨迹对齐或 SuperFiltering 的 IFD 粗筛复刻一个 5k 冷启动子集？
2. 纯 RL 阶段的 verifiable reward 在 code 上如何实现？DeepSeek 用单元测试 pass/fail，Qwen2.5-Coder 用执行过滤，你的 flywheel 里如何搭一个“合成→执行→拒采→RL”的闭环，成本/延迟是多少？
3. 对比题：R1 说 SFT memorizes, RL generalizes，和之前 Day4 LESS（SFT 5% 打赢全量）/ Day12 SuperFiltering（125M 选 SFT 更好）是否矛盾？为什么 SFT 的少即是多在 RL 里需要换成 RL 的少即是多（LIMR）？用 R1 的 cold-start + RL 结果解释 LESS 到 RL 的失效点。

## 核心
1. Motivation: SFT 蒸馏 o1 只能记住长思考格式，泛化差，纯 RL（R1-Zero）会自发涌现推理但可读性差且冷启动慢，需要 stabilizer
2. Data Pipeline: 少量高质量冷启动 SFT（<10k 推理轨迹，人工+启发式筛难多样可验证）→ 大规模 RL（GRPO，无 value 网络，rule-based / verifiable reward 数学答案+代码执行）→ 拒绝采样再 SFT+RL 迭代
3. Key Tricks:
   - GRPO 省 value，KL 约束轻，奖励只看答案/单元测试通过，天然可扩展到 code
   - 冷启动数据刻意保留“顿悟时刻” a-ha moment 的自我反思语句，诱导 RL 自我纠错
   - 语言一致性奖励防止中英混杂，保持 CoT 可读
4. Results: R1-Zero 纯 RL AIME 71% → R1 (cold+RL) AIME 79.8% / MATH 97.3% 超 o1-preview，HumanEval 类 code 多 10%+，证明 RL 泛化 > SFT memorizing

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 从你 50 万合成池跑一遍执行过滤（unit-test pass rate 0.2-0.8 的中难段），留 5k 做冷启动 SFT，对比全量 SFT 的 HumanEval
  2. 用 GRPO 式的 rule reward 在你小 7B 上试纯 RL 200 题，看是否涌现自纠错语句，记录 a-ha 率
- Infra 视角：GRPO 无 critic 省一半显存，verifiable reward 评测可并行化，但执行沙箱成为新瓶颈，需要 vLLM rollout + sandbox 分离

## 疑问 / 下一步
- R1 的冷启动 10k 究竟多少是 code vs math，比例如何影响最终 code 能力？Phi-1 天然指令占比启发？

## 原文金句
> RL is the engine, cold-start SFT is the stabilizer — reasoning emerges from reward, not imitation.
> 我们证明了少量的高质量冷启动 + 大规模可验证奖励 RL，可以让 base 模型超越大量 SFT 蒸馏的长思考模型
---
模板来源：PAPER_TEMPLATE.md | 今日任务配套骨架，后续填 NOTES
