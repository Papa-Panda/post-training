## 元信息
- Title: LESS: Selecting Influential Data for Targeted Instruction Tuning
- Authors / Org: Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen / Princeton NLP
- Link / arXiv: https://arxiv.org/abs/2402.04333
- Code: https://github.com/princeton-nlp/LESS
- Date read: 2026-08-06
- Tags: [sft, data-selection, influence, data-attribution, instruction-tuning, curation, targeted]

## 一句话总结
为想定向提升的能力（推理/BBH/MMLU）准备几条 few-shot 锚点，把大池子 27万 条指令的 low-rank 梯度跟锚点做相似度搜索，只训 top 5% 的数据，经常比训全量还好，且用 7B 选的数据能直接给 13B/Mistral 用。

## 核心
1.  **Motivation**: 全量指令微调混了太多水数据，想提升某个专项能力时，大部分数据是 noise。传统 BM25 / embedding 选的是表面像的，训了没用。需要按“对目标 loss 的实际影响”来选。
2.  **Data Pipeline**: 
    - Warmup: 随机抽 5% 数据把 base (Llama-2) 热一下，让梯度不要是纯噪
    - Gradient Datastore: 只取 LoRA adapter 的梯度，随机投影到 8192 维 (JL引理保点积)，建一次库可复用
    - Scoring: `score(z)=cos(mean_g_target, g_low(z))`，优化目标是 Adam 修正后的 influence `η * grad_target^T Gamma(z)`，其中 `Gamma = Adam_precond(grad)`
    - Select: 取分数最高的 5% (≈13k) 去做 instruction tuning
3.  **Key Tricks**: 
    - 优化器感知: 不是 SGD 点积，用 Adam 的 m/v 修正后的梯度
    - LoRA + 随机投影: 全参梯度 7B 维存不下，LoRA 降到几百万再 JL 到 8192
    - Warmup: 不 warmup 梯度全是噪，相似度排名失效
4.  **Results**: 5% LESS 选的数据训 Llama-2-7B，在 MMLU/BBH/TyDiQA 上常打赢 27万 全量；随机 5% / BM25 / RDS 都输。Transfer：7B 选的数据给 13B / Mistral-7B 用同样赢。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 拿 10 条 LiveCodeBench 难例当 target，用你 1B 小 proxy 建 gradient 库，筛 5% 合成 code 数据试 pass@k，对比随机 5%
  2. 用 self-influence 思路把 coding SFT 池子里 high high self-influence 的脏/需强记样本清掉，留推理型
- Infra 视角：gradient datastore 建一次多任务复用，成本 O(N) 建库后每次选数据都是 O(N) cosine，适合 fly-wheels；可扩展到 RLHF 数据筛选。

## 疑问 / 下一步
- 如果 target 是 code generation 而不是推理，few-shot 靶子要怎么写才能让梯度更准？是不是要用 execution trace 而不是最终答案？
- 小 proxy 太弱时 transfer 失效的临界点在哪？对 coding 1B proxy 够吗？

## 原文金句 (1-2句)
> Instruction tuning on a LESS-selected 5% of the data can often outperform training on the full dataset. — and the selected data is highly transferable across models.

> Our method goes beyond surface form cues to identify data that exemplifies the necessary reasoning skills.

## 官方 Repo
- GitHub: https://github.com/princeton-nlp/LESS — 包含 warmup / datastore / selection / train / eval 全流程代码
