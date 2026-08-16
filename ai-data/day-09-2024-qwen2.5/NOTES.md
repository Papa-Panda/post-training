# Paper 模板

## 元信息
- Title: Qwen2.5 Technical Report
- Authors / Org: Alibaba Cloud Qwen Team (Qwen2.5 / Qwen2.5-Coder / Qwen2.5-Math series)
- Link / arXiv: https://arxiv.org/abs/2412.15115 (main), companion Coder: https://arxiv.org/abs/2409.12186
- Date read: 2026-08-12
- Tags: [pretraining, coding-data, curation, synthetic-data, sft, rl-data, scaling, multilingual, moe]
- Folder: 2024_qwen2.5
- Day: 9

## 一句话总结
从 7T 拉到 18T 预训练（其中 code ~5.5T 来自 Qwen2.5-Coder 线），文件级+仓库级混合、弱模型分类器过滤、版图更大且多语，1M+ SFT + 多阶段 RL（RM↔SFT 迭代）+ MoE Turbo/Plus，72B 就超 Llama-3-405B-Instruct，证明 data recipe / flywheel > 单纯参数堆砌。

## 和之前工作的关系
- **知识图谱位置**：pretrain / scaling / curation 主线的第三极，对齐 Day 7 Llama 3 (15.6T dense) 和 Day 8 DeepSeek-V3 (14.8T MoE) —— 三大 2024 开源 frontier 配方三角在此闭合。从 influence/selection 线（Day 2-5）到 synthetic 线（Day 6 Phi-1）再到 pretrain 线（Day 7-9），Qwen 是 pretrain 向 SFT/RL 跨线的桥梁。
- **接了哪篇的哪条线**：
  - pretrain/curation：接 Llama 3 的 5 级瀑布过滤 + code 25% 上采样；接 DeepSeek-V3 的激进去重（MinHash 0.85→0.90）和 code 30%+ / FIM 10% PSM。Qwen 把 去重思路换成 file-level + repo-level 覆盖，code recall 用弱模型 scorer，规模拉到 18T。
  - synthetic：接 Phi-1 的 textbook 合成理念（Day 6），Qwen2.5-Math/Coder 展示大规模合成 math/code + self-improvement (Qwen2-Math-Instruct 生成 → RM 采样 → SFT → RM 迭代 → RL) 的工业化版本。
  - selection / influence：接 LESS Day 4（gradient相似度选 5%）和 DataInf Day 5（LoRA闭式 influence 1秒/条），Qwen 用 RM 打分迭代替代梯度影响——目标导向选择 vs 梯度导向选择，计算成本更低但更贴偏好。
- **补了哪个短板**：Llama3/DeepSeek 多讲 pretrain，不管 SFT/RL 怎么挑；Phi-1 只讲小模型合成；LESS/DataInf 只讲怎么选。Qwen 补上“1M+ 精细 SFT 配比 + 多阶段 RL + 长文本/结构化数据/指令跟随”如何搭，以及 MoE Turbo/Plus 如何用相同 data 做到成本/性能权衡。
- **替代/分叉/改进**：不是替代，是分叉后汇合：证明 scaling 之后 data flywheel 比 param 更重要（72B > 405B）。对 Day 7 的直接对比：Llama3 405B 需要 15.6T + annealing 才 GPT-4 级，Qwen2.5 72B 用 18T + 更好的多语/code/合成就超它；MoE 版更进一步说明推理成本可通过 data 控。
- **数量 vs 结构**：强调关系 > 数量，把新点挂到已有图上：pre-train 去重阈值、code 占比、FIM/RM 选择三轴已有，Qwen 加上第四轴“post-training 迭代”。

## 为什么今天读它
- coding data：5.5T code 专用数据集构造来自 Qwen2.5-Coder：GitHub 公有库 + web 爬的 code-related texts，file-level + repo-level pretraining，弱模型分类器/ scorer 去低质，FIM 风格、execution 过滤类比 DeepSeek 的 PSM 可直接抄。
- SFT：1M+ 样本，real-world + synthetic（code-focused LLM 生成），覆盖生成/补全/推理/修复广度，配比 balancing coding/general/math 防止 30% code 掉 MMLU。
- RL data：多阶段 RL 提升偏好对齐、长文本、多轮 agent/tool use，RM 指导采样/过滤，类似 RLHF 的 reward flywheel，为 Agentic RL Infra 的 $/useful-rollout 提供 data 侧信号。

## 3 问回顾（Day 9 原题）
1. Qwen2.5 把 pre-training 从 7T 拉到 18T，具体怎么做 file-level + repo-level code recall 和弱模型分类器过滤低质？和 Llama 3 的 5 级瀑布 + DeepSeek 的 0.90 MinHash 比，去重/质量栅栏有何不同，哪个更省算力？
2. 它的 1M+ SFT + 多阶段 RL 是怎么迭代进化（RM → SFT → 新 RM → 下轮 SFT → 最终 RM 做 RL，见 Qwen2.5-Math 的 self-improvement），和 Day 4 LESS 用 Adam 感知的 low-rank 梯度相似度选 5% 相比，Trade-off 在哪？为什么 Qwen 选 RM-based 而非 gradient influence？对你 50 万合成池，哪个更可落地？
3. 对比 Day 7 Llama 3 405B(15.6T) 和 Day 8 DeepSeek-V3 671B MoE(14.8T code 30%+ FIM 10%)，Qwen2.5 72B 用 18T 就超 Llama-3-405B-Instruct，三家的 code 占比 / 合成策略 / 多语配比 / 评测差异如何解释参效比差异？若你把 50 万合成池按 Qwen 配方做 repo 级 packing + 长上下文合成，会比 Llama 3 annealing 更省还是更贵？Infra 视角：Bloom + MinHash vs 弱模型 scorer，哪个是 18T 瓶颈？

## 核心（待填，今晚产出）
1. **Motivation**: 
2. **Data Pipeline**: 来源 → 清洗 → 合成 → 配比 → 训练
3. **Key Tricks**: 
4. **Results**:

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
- Infra 视角：

## 疑问 / 下一步

## 原文金句

## 参考
- Qwen2.5 Main: https://arxiv.org/abs/2412.15115
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122 (self-improvement RM↔SFT 迭代)

