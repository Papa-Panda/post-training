## 元信息
- Title: Textbooks Are All You Need (Phi-1)
- Authors / Org: Suriya Gunasekar et al. / Microsoft Research
- Link / arXiv: https://arxiv.org/abs/2306.11644
- Code/Data: https://huggingface.co/microsoft/phi-1 , synthetic CodeTextbook via GPT-3.5
- Date read: 2026-08-09
- Tags: [coding-data, synthetic-data, quality, curation, pretraining, sft, SOTA]

## 一句话总结
不用堆 100B web code，用 GPT-3.5 合成“教科书质量”的 6B 精筛 web + 1B 合成练习，1.3B 模型训 4 天就 50.6% HumanEval，证明高质量合成数据 >> 大量低质数据。

## 核心
1.  **Motivation**: scaling law 让人以为堆数据就好，但 web code 又臭又长又重复。能不能像教课本一样，把知识蒸馏成干净、渐进、带解释的教材，让小模型也学会？
2.  **Data Pipeline**: 
    - 从 The Stack 挑 textbook-like：用 classifier 选出“解释性强、教育性” 的 Python 文件，6B tokens
    - 合成 CodeTextbook：用 GPT-3.5 按主题生成章节式解释 + 代码示例，1B tokens
    - 合成 CodeExercises：再让 GPT-3.5 生成小练习题 + 解法，微调用，10k级
    - 训练：1.3B Transformer，4天 8 A100，先预训 6B+1B，再在练习上微调
3.  **Key Tricks**: 
    - 不是随机合成，是按“教科书章节”结构化合成：概念 → 例子 → 练习，利于模型建立因果链
    - 小模型也能涌现：350M 同流程仍 45% HumanEval，说明数据带的有推理模式
    - 过滤狠：原来 Stack 100B 级，只留 6B 精的，massive deduplication + decontamination
4.  **Results**: Phi-1 1.3B HumanEval pass@1 50.6% / MBPP 55.5%，超过很多 7B-15B 直接在 web code 上训的；Phi-1-small 350M 仍 45%，证明 pipeline 可迁移到更小 budget。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 直接抄 Phi-1 的 CodeExercises 合成模板，让你的合成器把“题目+逐步解+边界测试”一起出，做 RL 的 rollout 验货数据
  2. 把你的 50万 合成池用 textbook-quality classifier 重筛一遍，留 5% 精的试训，对比现在全量训的效果
- Infra 视角：合成数据 flewheel 成本主要在 GPT-3.5/4 调用，得搭 cache + de-dup + contaminated check，否则 HumanEval 泄漏假高。

## 疑问 / 下一步
- 合成教科书的 diversity 如何保证不坍缩成几种模板，coding 上不同算法范式（DP/图/贪心）是否都被覆盖？
- Phi-1 说 web 数据是 noise，但如果我的下游是长尾库函数调用，纯教科书会不会掉 recall？

## 原文金句 (1-2句)
>  We introduce phi-1, with 1.3B parameters, trained on textbook quality data, attaining 50.6% HumanEval — despite this small scale.

> Textbooks Are All You Need — high-quality curated data can be just as useful as enormous unfocused piles.
