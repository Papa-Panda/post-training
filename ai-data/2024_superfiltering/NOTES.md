# Paper 模板

## 元信息
- Title: Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
- Authors / Org: Ming Li, Yong Zhang, Shwai He, Zhitao Li, Hongyu Zhao, et al. - UMD / Ping An
- Link / arXiv: https://arxiv.org/abs/2402.00530
- Date read: 2026-08-13
- Tags: [sft, data-selection, coding-data, weak-to-strong, instruction-tuning]

## 一句话总结
不用大模型当过滤器，用小 125M 的 GPT-2 算 IFD 难度分去筛指令，能筛出给 7B 训后效果反而更好的数据，证明选数据的能力在小模型上就有了。

## 核心
1.  **Motivation**: 指令微调数据又烂又冗余，用 GPT-4 去筛太贵，SFT 全量训又浪费
2.  **Data Pipeline**: 小模型算每条指令的 Instruction-Following Difficulty -> 排难度 -> 中高难度留 -> 给 LLaMA-7B 训。发现弱强模型选数据的结果高度一致【4921601271219063014†L16-L19】。
3.  **Key Tricks**: 弱到强过滤一致性：GPT-2 125M 挑的和 7B 自己挑的高度一致【4921601271219063014†L16-L19】；IFD 分比 perplexity/diversity 都好【4921601271219063014†L38-L42】；不训也行，plug-and-play无须额外hold-out【4921601271219063014†L54-L57】
4.  **Results**: 超过全量训，GPT-4 判赢比更高，LLaMA2-7B 用自己 IFD 还能再涨一点【4921601271219063014†L48-L51】

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - 用 350M CodeT5 去筛你 50万 coding SFT，全量别训，IFD 高的留 10% 先训，省钱
  - 把 CodeSift 这种不跑的验证和 Superfiltering 的 IFD 套一起，一遍弱模型算分一遍 LLM 判，coding 上更准
- Infra 视角：弱模型过滤器可以常驻做在线流入筛选，1/500 成本筛 GPT-4 級质量

## 和之前工作的关系
是 LESS 的便宜版替代。LESS 要建梯度库，算力重；Superfiltering 只算一次前向 IFD，弱模型就行。是 Day 11 memory 里说的 ai_daily Tab 45天计划里 foundation 阶段的快速打法，可放在 Day 4 LESS 之前做粗筛。

## 疑问 / 下一步
- IFD 在 coding 上会不会把太难的 FIM 题都砍掉？需要结合执行过滤？

## 原文金句
> Can we use a smaller and weaker model to select data for finetuning a larger and stronger model?【4921601271219063014†L16-L18】

> This enables us to use a much smaller and more efficient model to filter【4921601271219063014†L18-L20】
