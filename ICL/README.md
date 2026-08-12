# ICL — In-Context Learning 知识脉络与数学

> 冻结权重 θ，给一段 `[(x₁,y₁)...(x_k,y_k), x_q]`，模型直接 `pθ(y|prompt)` 就能干活。  
> 这个文件夹是把之前聊过的三条主线、数学比分、trajectory-error prompting、以及 code 版的迁移，沉淀成 repo 里的可引用笔记。

## 结构

```
ICL/
├── README.md                      # 你现在看的
├── 01_definition_timeline.md       # 定义 + 时间线
├── 02_line_I_bayesian.md           # 线 I：隐式贝叶斯
├── 03_line_II_gd.md                # 线 II：前向梯度下降（最硬核证明）
├── 04_line_III_circuit.md          # 线 III：回路 / Induction Head
├── 05_comparison.md                # 三线比分 + 统一视角
├── 06_trajectory_error_prompt.md   # trajectory error → 通用 prompt
└── 07_coding_data.md               # code 专用版（Socratic-SWE / SWE-Gym / CYCLE）
```

## 一句话总览

- GPT-2 已有 zero-shot 苗头，GPT-3 2020 年《Language Models are Few-Shot Learners》正式命名 ICL 为能力。
- 三条解释线：**Bayes（目标是什么）→ GD（算法怎么算）→ Circuit（硬件谁在搬）**。三者投影同一现象。
- 衍生：CoT = 链上连续触发 induction；many-shot = log k 持续提升但需 LayerNorm；in-context RL = 把 reward 放进 prompt 做 policy improvement。

## 怎么用到你的 post-training / agentic RL

- **数据侧**：按潜在概念 c 平衡 few-shot 分布（线 I）
- **格式侧**：让 demo 结构利于 $V K^T Q$ 累加，KV 对齐梯度形式（线 II）
- **评测侧**：单测 induction 分数，比整体 loss 更早预示 ICL 是否起飞（线 III）
- **数据飞轮**：把 trajectory 失败聚类成 3-5 条通用军规，做法有四种（见 06）：
  1. system 常驻军规
  2. 反洗成 SFT/RL 数据（最值钱）
  3. 当 reward model verifier
  4. 人工写 agent SOP

对应论文卡在 06/07，交互版可视化在之前的 artifact `ts-spaces/icl/index.html`。

> 维护：这个 ICL 路径跟 `ai-data/papers/` 并行，paper 级别的沉淀仍走 `ai-data/`，这里做主线的数学笔记 + coding data 落地。
