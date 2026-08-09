## 元信息
- Title: DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models
- Authors / Org: Yongchan Kwon, Eric Wu, Kevin Wu, James Zou / Columbia & Stanford
- Link / arXiv: https://arxiv.org/abs/2310.00902
- Code: https://github.com/YeonwooSung/DataInf (official), plus https://github.com/UKPLab/datainf community impls
- Date read: 2026-08-07
- Tags: [data-attribution, influence-functions, lora, efficiency, curation, quality]

## 一句话总结
把 Influence Functions 中 $H^{-1}$ 的迭代求解，改成 LoRA 参数上的闭式近似 $(1/n \sum g_i g_i^T + \lambda I)^{-1}$，比 LiSSA/CG 快 1000倍，1秒算一条 influence，专门为 LLM LoRA 微调设计，可直接用于扫脏数据/高影响样本挖掘。

## 核心
1.  **Motivation**: Influence 很好但算不动，LLM 上算 $H^{-1}v$ 要 LiSSA 迭代几百次，每次全量 HVP。大模型 + LoRA 场景急需快版。DataInf 盯的就是 LoRA 微调这个常见设定。
2.  **Data Pipeline**: 
    - 在 LoRA 微调模型上，对每个训练点算 LoRA 梯度 $g_i$
    - 用经验 Fisher 近似 $H \approx (1/n)\sum g_i g_i^T$，然后 influence Closed-form：$I(z_j, z_{test}) \approx - g_{test}^T (G^T G / n + \lambda I)^{-1} g_j$
    - 只在低秩 LoRA 维度上求逆，维度几十k不是几十亿，可闭式解
    - 拿 test 点（或 few-shot target 池）批量算，排序找 high influence / mislabeled
3.  **Key Tricks**: 
    - LoRA 是关键：全参上 $H$ 奇异且不可逆，LoRA 上参数少、满秩，闭式才稳
    - 不用二阶反传，只用一阶梯度外积，内存/计算都 $O(d_{LoRA})$
    - 对 diffusion 也适用，同理算 UNet LoRA 梯度
4.  **Results**: RoBERTa-large / Llama-2-13B-chat / Stable-Diffusion 上，近似误差 < 10% vs LiSSA 真值，速度提升 2-3 数量级；mislabel detection AUC 明显高于 TracIn, Representer。论文 ICLR 2024。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  1. 在你 SFT 的 LoRA checkpoint 上跑 DataInf，给 50w 合成 code 打 influence 分，top negative 就是拖累 HumanEval 的脏数据
  2. 把 LESS 的 datastore 换成 DataInf 的闭式分数，做 5% 筛选对比，看速度/精度 trade-off
- Infra 视角：LoRA 上求逆可用 batched Cholesky，单 A100 可并行算 10k 条影响分，适合 nightly data flywheel 的自动清洗。

## 疑问 / 下一步
- LoRA 的 Fisher 近似在 RLHF / GRPO 的 policy gradient 上还成立吗，还是需要修正？
- $\lambda$ 如何自适应选，coding data 上 Fisher 奇异时如何稳住逆？

## 原文金句 (1-2句)
> DataInf is particularly well-suited for parameter-efficient fine-tuning techniques such as LoRA, with an easy-to-compute closed-form expression.

> DataInf is orders of magnitude faster than existing influence methods while accurately approximating influence scores.
