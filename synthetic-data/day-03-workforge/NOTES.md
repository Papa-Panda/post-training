# Scaling Verifiable Environments for Long-horizon Work Agents

> 📖 阅读版：https://papa-panda.github.io/post-training/synthetic-data/day-03-workforge/

## 元信息

- Title: Scaling Verifiable Environments for Long-horizon Work Agents
- Authors: Zhang 等
- Link / arXiv: https://arxiv.org/abs/2610.04906
- Date of publication: 2026-10-04
- Tags: [work-agents, verifiable-environments, realism]
- 与前篇的关系：Day 01 SPADE 是合成环境（Designer 造题），Day 02 是真实失败（从 rollout 里挖）；Day 03 把“真实”推到环境本身——不造题、不挖失败，直接把真实工作空间搬进来，再从真实专家工作流导出任务与 verifier。

## 一句话总结

长程工作 agent 的瓶颈不是模型，而是环境太假：直接从真实资源（网络检索 + 真实文件）建工作空间，再从真实专家工作流里导出任务指令、解题计划和“程序化 + 语义”双 verifier，一共 16.7K 个可验证环境，覆盖 40 个专业领域、60 种文件类型。Qwen3.5-35B-A3B-Base 训上 WorkForge 后 GDPval 从 45.5 升到 73.6；27B dense 版 GDPval 82.4，超过 Qwen3.5-397B-A17B 的 79.7。

## 3 个问题

1. 真实感的三个成分（文件数量 5.4×、类型多样性 2.4×、大小 21.5KB→279KB）到底哪个在起作用？设计一个消融实验：固定任务与 verifier，三臂——A 完整真实工作空间；B 真实文件但随机打乱目录结构（数量/类型不变，结构破坏）；C LLM 合成文件但按真实分布补足数量与类型。评 GDPval。判据：若 A≈B≫C → 数量/类型是主因；若 A≫B≈C → 结构本身（文件间的真实关联）才是关键。再说漏洞：B 的“打乱结构”会不会同时破坏 verifier 的可验证性（控制变量不干净）？
2. “程序化 + 语义”双 verifier：设计一个 3×2：verifier ∈ {只程序化，只语义，双} × 任务类型 ∈ {有唯一客观答案，无唯一答案（如开放式写作/分析）}。先预测六个格子的 pass 率与“误判率”（verifier 说对、实际错）排序并写理由；再问：怎么度量语义 verifier 自己的错误率——它本身就是 LLM judge，谁来 verifier the verifier？
3. 27B dense 82.4 超过 397B 的 79.7。设计一个检验，区分“WorkForge 数据赢了规模”与“GDPval 偏向 WorkForge 的任务分布”：固定 27B dense 模型，A 在 WorkForge 上训、GDPval 上评；B 在同等 token 量的通用 agentic 数据上训、GDPval 上评；C 把 A 的模型拿到非 WorkForge 分布的评测（如 BFCL/ACEBench）上评。判据：若 A 在 C 上也赢 B → 数据真赢了规模；若 A 只在 GDPval 赢、B 在 C 上反超 → GDPval 与 WorkForge 同分布，分数有水分。

## 核心机制

1. **Motivation**：长程工作 agent 需要在真实工作空间里做长链条任务，但现有环境大多是 LLM 合成的：文件少、类型单一、规模小，agent 在上面练出来的本事搬不到真实工作里。环境不真实，探索和长程规划都无从练起。
2. **方法**：不合成、直接搬真实：从真实资源（网络检索 + 真实文件）建工作空间；再从真实专家工作流里导出任务指令、解题计划和“程序化 + 语义”双 verifier——程序化 verifier 验可判定的结果，语义 verifier 验开放性的质量。环境、任务、验证三件套都锚定在真实工作上。
3. **关键数字**（论文自报，经本系列 papers.md 注释核实）：16.7K 个可验证环境，覆盖 40 个专业领域、60 种文件类型；真实工作空间比 LLM 合成的多 5.4 倍文件、2.4 倍文件类型，平均大小从 21.5KB 跳到 279KB；Qwen3.5-35B-A3B-Base + WorkForge，GDPval 从 45.5 升到 73.6（+28.1 个点）；27B dense 版 GDPval 82.4，超过 Qwen3.5-397B-A17B 的 79.7（+2.7 个点）。

## 边界

- “真实”不等于“无偏”：16.7K 环境来自可检索到的真实资源与专家工作流，覆盖 40 个领域——检索不到、没有现成工作流的真实工作进不来，分布仍有取舍。
- 双 verifier 的语义一侧挂在 judge 身上：程序化 verifier 是硬的，语义 verifier 是模型判的，judge 的标准漂移会直接污染训练信号。
- GDPval 上的胜利要打问号：27B 超 397B 的 +2.7 个点，是 WorkForge 数据赢了规模，还是 GDPval 的任务分布恰好贴着 WorkForge 的 40 个领域？跨评测的泛化没有在这里证明。

## 迁移

- “直接搬真实工作空间”可搬到 coding agent / 办公 agent：不用先造题，先把真实仓库、真实文档、真实工作流搬进来，任务和 verifier 从工作流里长出来。
- 与 Day 01 SPADE 构成对照：SPADE 是合成环境、靠 hint-regret 卡在能力边界；WorkForge 是真实环境、靠真实感让探索有东西可学——“环境从哪来”的两条路线。
- 与 Day 02 AED 构成对照：AED 从真实失败里挖数据，WorkForge 从真实工作里建环境——一个挖失败、一个搬现场，都是“真实”路线的不同取法。
- 如果复现，第一步先定一个可程序化验证的真实任务来源（真实仓库 / 真实文档），再谈语义 verifier——没有硬 verifier 兜底，语义 judge 会把标准越判越松。

## 讨论题

- 这篇的结论里，哪一条最可能被后来的工作推翻？
- 数据 pipeline 中哪个环节最容易藏 bug？（真实工作流导出任务指令时，专家步骤的隐性知识有没有丢；语义 verifier 的 judge 标准是否稳定）
- 换成今天的模型规模，这个方法还成立吗？
