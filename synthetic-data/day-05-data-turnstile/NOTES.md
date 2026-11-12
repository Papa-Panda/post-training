# Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation

> 📖 阅读版：https://papa-panda.github.io/post-training/synthetic-data/day-05-data-turnstile/

## 元信息

- Title: Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation
- Authors: Ramakrishnan 等
- Link / arXiv: https://arxiv.org/abs/2607.29250
- Date of publication: 2026-07-31
- Tags: [function-calling, dag, small-models]
- 与前篇的关系：Day 04 ToolLoop 是“三段分解 + 每步 verifier + 反馈再生”；Day 05 是另一种分解——把一次多轮工具交互拆成 DAG，按 role 分节点（用户 query / 推理 / API 调用 / 执行输出 / 助手回复），每个 role 独立生成、独立验证、带错误反馈重试。ToolLoop 按阶段拆（采样组合 → 反向 query → 正向调用），Turnstile 按角色拆；两篇都回答同一个问题：合成数据不经过“验”，单次生成会怎样？Day 04 的答案是无反馈版掉到 base 之下，Day 05 的答案更狠：raw 数据直接训 0.6B 小模型，BFCL 掉到 55.1%，Irrelevance 检测崩到 35.7%。

## 一句话总结

小模型（SLM）做工具调用最缺的不是参数，是干净数据：容量小，扛不住低质量监督，噪声直接变成毒药。Data Turnstile 把多轮工具交互拆成一个 DAG，按 role 分节点——用户 query、推理、API 调用、执行输出、助手回复——每个 role 独立生成、独立验证，验不过就带着错误反馈重试，避免单次多轮合成的级联错误；教师只用本地跑的 Qwen2.5-32B-Instruct，不依赖闭源 API。结果：Qwen3-0.6B 的 BFCL（no-think）从 58.2% 升到 75.9%，τ²-bench Telecom 的 pass^1 从 3.5 升到 23.0；而 raw xLAM+Glaive 直接训同一个 0.6B，BFCL 掉到 55.1%，Irrelevance 检测崩到 35.7%。

## 3 个问题

1. DAG 分角色验证的收益，到底来自“拆”还是来自“验”？设计一个实验：固定同一批 API 规格与同一个本地教师（Qwen2.5-32B-Instruct），三臂——A 完整 Turnstile（5 个 role 节点、每 role 独立验证、带错误反馈重试）；B 保留 DAG 结构、但只做端到端验证（整条交互一次验完，不过就整条重生成）；C 单次多轮合成（不拆 DAG，一次生成整段对话）。评 BFCL（单轮）与 τ²-bench Telecom pass^1（多轮），同时记每个臂“每条有效样本的教师 token 成本”。判据：若 A 在 τ² 上 ≫B、但在 BFCL 上 A≈B → 分角色验证的收益主要在多轮；若 A 每条有效样本的成本超过 B 的 2 倍、而分数只高不到 2 个点 → 这笔买卖不划算。再预测一个 2×2 格子：{单轮 BFCL，多轮 τ²} × {分角色验证开，关}，哪个格子收益最大并写理由。漏洞：各 role 的 validator 共用同一个教师，错误可能相关——validator 判错的方向一致时，“独立验证”只是把同一个偏见验了 5 遍；另外三臂的重试次数上限不同，控制不干净。
2. Raw 数据为什么对小模型是毒药——毒在“错样本”还是“分布”？Raw xLAM+Glaive 直接训 0.6B，BFCL 掉到 55.1%（比 58.2% 的 base 还低 3.1 个点），Irrelevance 检测崩到 35.7%。设计一个实验：三臂——A raw 混合直接训；B raw 混合 + Turnstile 的分角色验证做过滤（只筛、不重生成）；C 完整 Turnstile（验证 + 带着错误反馈重生成）。再把模型尺寸做成第二维 {0.6B，4B}，预测一个 2×2 格子：{0.6B，4B} × {raw，Turnstile 清洗后} 在 Irrelevance 上的排序，写理由（提示：容量能不能“扛”住噪声？）。判据：若 B 基本恢复到 C 的水平 → 毒药主要是无效样本，筛掉就够了；若只有 C 恢复 → 重生成本身在补分布，不只是筛。漏洞：raw 混合与 Turnstile 数据的分布本来就不同（API 覆盖、对话长度、query 风格），“毒药”可能来自分布错位而非质量；另外 Irrelevance 崩到 35.7% 可能另有机制——raw 数据是不是在教模型“永远要调用点什么”、把“拒绝调用”这个行为本身训没了？
3. CoT 在单轮与多轮上为什么完全相反？单轮 BFCL 上，高质量 SFT 之后 no-think 反而更好；多轮 τ²-bench 上，关掉 CoT 全线崩 11–22 个点。设计一个实验：三臂——A 开 CoT；B 关 CoT（no-think）；C 短 CoT（只写状态追踪：当前目标 / 已知信息 / 下一步调用，限长）。同时在 BFCL 单轮与 τ² 多轮上评。判据：若 C 在 τ² 上基本恢复 A 的水平 → 多轮 CoT 的作用主要是状态追踪，不是深度推理；若 C 仍崩 → 多轮需要显式的规划推理，短状态不够。再问漏洞：短 CoT 的长度上限是拍脑袋定的，换个上限结论可能翻盘；教师生成的 CoT 可能本身就泄露了答案结构（推理里藏了下一步该调哪个 API）；而且三臂的 token 成本 / 延迟没进评测——多轮场景下 A 的 CoT 是不是贵到没法部署？

## 核心机制

1. **Motivation**：小模型部署 agent 有吸引力（低延迟、低成本、端侧隐私），但工具调用任务上数据又少又脏；大模型能靠容量“扛”住低质量监督，小模型扛不住——数据质量成了关键瓶颈。单次多轮合成更糟：一步错、步步错，错误级联进整条对话。
2. **方法**：DAG + 分角色验证 + 错误反馈重试。把一次多轮工具交互拆成一个 DAG，节点按 role 分：用户 query、推理、API 调用、执行输出、助手回复。每个 role 独立生成、独立验证，验不过就带着错误反馈重试，而不是整条丢掉或把错误留进数据。教师只用本地跑的 Qwen2.5-32B-Instruct，不依赖闭源 API，框架本身开源可复现。
3. **关键数字**（论文自报，经本系列 papers.md 注释核实）：Qwen3-0.6B 的 BFCL（no-think）从 58.2% 升到 75.9%（+17.7 个点）；τ²-bench Telecom 的 pass^1 从 3.5 升到 23.0（约 6.6 倍）；raw xLAM+Glaive 直接训同一个 0.6B，BFCL 掉到 55.1%，比 base 低 3.1 个点，Irrelevance 检测崩到 35.7%；CoT 分化：单轮 BFCL 上高质量 SFT 后 no-think 反而更好，但多轮 τ²-bench 上关掉 CoT 全线崩 11–22 个点。

$$75.9 - 58.2 = 17.7 \quad\text{；}\quad 23.0 / 3.5 \approx 6.6$$

## 边界

- 分角色验证的代价：5 个 role 各验各的，教师调用量与延迟都上去；“验证”本身也是模型判的，validator 判错会把错误固化进“已验证”的数据——DAG 拆得越细，validator 的偏见被放大的节点越多。
- 教师天花板：教师是本地 Qwen2.5-32B-Instruct，数据质量上界就是这个教师；闭源大模型教师没用，省了钱也封了顶。
- 评测只有两个点：BFCL 单轮 + τ²-bench Telecom 多轮；Telecom 一个 domain 的 pass^1 从 3.5 到 23.0 很亮眼，但换个 domain、换个多轮评测，DAG 的收益是不是一样大，没证据。
- Irrelevance 崩到 35.7% 只出现在 raw 混合的消融里：它证明的是“raw 数据有毒”，不是“Turnstile 治好了 Irrelevance”——Turnstile 训完的 Irrelevance 分数本身没在注释里给出来。

## 迁移

- DAG 分角色验证可搬到任何多轮合成数据：多轮对话、办公 agent、数学证明步骤——关键是把“一次生成整段”改成“按角色拆、每段都验、错了带着原因重生”。
- 与 Day 04 ToolLoop 构成对照：ToolLoop 按阶段拆（采样组合 → 反向 query → 正向调用），Turnstile 按角色拆（query / 推理 / 调用 / 执行输出 / 回复）——两种分解维度，哪个更适合你的任务，取决于错误主要出在哪一层。
- 与 Day 02 AED 构成对照：AED 把真实失败当主资产（事后挖），Turnstile 在合成时就拦错（事中验）——失败数据的两种用法，Turnstile 的版本更进一步：错了不丢，带着反馈重生成。
- 如果复现，第一步先搭一个最小的 per-role validator（query 是否可答、推理是否自洽、调用是否可执行、执行输出是否与调用匹配、回复是否基于执行结果），再谈规模；没有 validator 的 DAG 只是把一次大错拆成五次小错。

## 讨论题

- 这篇的结论里，哪一条最可能被后来的工作推翻？
- 数据 pipeline 中哪个环节最容易藏 bug？（某个 role 的 validator 标准最松；带着错误反馈重试时，错误原因本身有没有被模型“编”出来；执行输出节点是不是在拿真实执行结果，还是模型自己“演”的执行结果）
- 换成今天的模型规模，这个方法还成立吗？
