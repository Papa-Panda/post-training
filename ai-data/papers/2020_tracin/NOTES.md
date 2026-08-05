# Paper 模板 ## 元信息
- Title: Estimating Training Data Influence by Tracing Gradient Descent (TracIn)
- Authors / Org: Garima Pruthi, Frederick Liu, Mukund Sundararajan, et al. / Google
- Link / arXiv: https://arxiv.org/abs/2002.08484 / https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
- Date read: 2026-08-06 (预定 Day 2)
- Tags: [data-attribution, influence, coding-data, data-cleaning, classic]
- Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/
- Code: https://github.com/frederick0329/TracIn ## 一句话总结
（明日填）不用 Hessian，用训练过程 checkpoint 上的梯度点积来估计训练样本对测试样本的影响，工程上可扩展到大模型。 ## 核心
1. **Motivation**: Influence Functions 要 H⁻¹，贵且不准。能否直接看训练过程？
2. **Data Pipeline**: ideal influence = sum_t η_t * ∇L_test(θ_t) · ∇L_train(θ_t)
3. **Key Tricks**: 4. **Results**: ## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
- Infra 视角：checkpoint 存储 / HVP 免除的成本优势 ## 疑问 / 下一步
- 预习：1) 为什么只要点积就行 2) checkpoint 怎么选 3) self-influence = 脏数据？ ## 原文金句 (1-2句)
> ## 预习链接
- arXiv: https://arxiv.org/abs/2002.08484
- PDF: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
- Blog: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/
- Code: https://github.com/frederick0329/TracIn
