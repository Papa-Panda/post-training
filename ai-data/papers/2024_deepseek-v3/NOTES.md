# DeepSeek-V3 Data - MoE 685B Open Recipe

## 元信息
- Title: DeepSeek-V3 Technical Report
- Authors / Org: DeepSeek-AI
- Link / arXiv: https://arxiv.org/abs/2412.19437
- Date read: 2026-08-11
- Tags: [coding-data, pretraining, curation, moe, quality, scaling, synthetic-data, flywheel]

## 一句话总结
用 14.8T MoE专用管线（code 30%+，重度去重，FIM 10%，执行过滤合成），在更干净更偏码的数据上训 671B MoE（37B激活），对标 Llama 3 405B，证明 MoE 的数据配方要比 dense 更激进、更干净。

## 和之前工作的关系
这篇在知识结构里是 Llama 3 Day 7 的同级对标（pretrain/scaling 线），不是延续。

- **接 Llama 3 15T 的坑**：同样起于 Web+Code+多语种，但 Llama 3 是 dense 15.6T + 5级瀑布 + 25% code；DeepSeek-V3 把去重阈值提得更狠（MinHash 0.85→0.90，强调 corpus diversity），code/math 比重继续拉高到 30%+，把 pack 不做 cross-sample attention 的思路也保留下来【3762324244574975666†L64-L67】。
- **对 LESS Day 4 的呼应**：MoE 容量大更吃噪声，所以 pretrain 就做“只留高影响”的子集选择，定时钟从 20T 粗筛到 14.8T。
- **对 DataInf Day 5 的呼应**：MoE 的 expert 稀疏激活，某条烂 code 可能只毒一个 expert。DataInf 教你在 LoRA 低秩子空间里闭式算 influence 来踢脏数据；DeepSeek 同理可以把 influence 算到 expert 级，定位到是哪条数据搞脏了哪个 expert，做专家维度的清洗，这是 dense 模型不需要的。
- **对 Phi-1 Day 6 的呼应**：Phi-1 证明合成教科书质量>数量；DeepSeek-V3 把合成题全上执行验证（run-and-filter），比 textbook 多了一级“可验”，是 Phi-1 的升级版。14.8T 里 10% 用了 FIM PSM 框架 `<|fim_begin|>pre ... |fim_hole| suf |fim_end| mid` 来保代码续写能力【3762324244574975666†L70-L76】。

## 核心
1. **Motivation**: MoE 总参 671B 但激活 37B，想训得稳、推理省，必须数据更干净、code 更重。DeepSeek-V2 已验证 DSA + MoE，可沿用，但 V2 的数据偏通用，数学/编程比不够，冗余多，需要重配比。
2. **Data Pipeline**: 来源 →  Web/多语种 + Math/Code 上采样 → 清洗 →  heurisitic 去毒/PII → 去重：文档 dedup + MinHash 近重复（更激进）→ document packing（Ding et al. 2024方法）但不做 cross-sample attention【3762324244574975666†L64-L67】 → 质量过滤：类似 Llama 3 的多级，但 multilingual 扩大、英文/中文外也保多样性 → 合成：Code FIM 10% PSM【3762324244574975666†L70-L76】 + 数学推理链 → 训练：14.8T high-quality diverse tokens【3762324244574975666†L59-L63】 + MLA + Aux-loss-free 负载均衡 + Multi-token prediction。
3. **Key Tricks**:
   - code/math 比重 30%+（Llama 3 25% 基础上继续加），因为 MoE expert 路由能容纳更多编程模式，dense 30%会掉 MMLU，MoE不会。
   - MinHash 0.90 级去重 + 文档 pack 保 integrity，DeepSeek 强调 minimize redundancy while maintaining diversity【3762324244574975666†L62-L66】，比 Llama 3 的 0.85 更狠，省的算力给 MoE 路由。
   - FIM 0.1 PSM 写成 `<|fim_begin|>f_pre<|fim_hole|>f_suf<|fim_end|>f_middle`【3762324244574975666†L72-L76】，专门保 code infill，配合 Tokenizer 128K + 合并标点/换行 token被随机拆分抗 bias，这些代码向细节是 Phi-1 没有的。
4. **Results**: 671B/37B MoE，2.788M H800 GPU 时，训程零不可恢复 spikes；评测上开源模型里 SOTA，对标闭源 GPT-4 级；14.8T 训完后 SFT+RL 阶段仍稳。Long context 128K、tool use、code HumanEval/MBPP 都超 Llama 3 405B 路线，推理激活参数小 10 倍。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. **30% code 实验**：把你 50万 合成池按 DeepSeek 配方，把 code 从 25% 提到 30%，同时加一道 MinHash 0.90 二次去重，用你现有的 1.3B dense 对比 MoE 小模型（如果换成 8x1.3B MoE），看 HumanEval 涨不涨，MMLU 掉不掉。
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
1. DeepSeek 的 14.8T 和 Llama 3 的 15.6T，同样的源，为什么 MoE 要把 MinHash 阈值和 code 占比调得更高？
2. 它的合成数据是怎么做执行过滤的？和 Day 6 Phi-1 的 textbook 合成比，多了哪一级“可验”？
3. 如果把 DeepSeek 这套搬到你 1.3B dense 上，有哪一条不能抄？为什么 MoE 能抗 30% code，dense 不行？

---
生成逻辑：已纳入知识图谱，强调结构非数量，自动产出已开启。
