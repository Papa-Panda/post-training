# 注释版篇目：Synthetic Data / Agentic 数据合成与清洗系列

每篇的注释只写核实过的事实：方法、数据规模、论文自报的关键数字。
思考问题是读的时候要带着的，不需要现在回答。

来源：Lei Tian 专栏《Agentic视角下的数据链路》，
知乎慢读清单 id 259（2026-11-02 刊，Day 01–07）与 id 320（2026-11-09 刊，Day 08–15）。

## 01 SPADE：环境自己长出来

- 论文：SPADE: Self-Play in Adaptive Synthetic Executable Environments，Liu 等，2026，arXiv:2608.19197。
- 注释：同一个 LLM 分饰两角。Environment Designer 生成完整可执行
  Gym 风格环境（reset()/step()、状态转移、奖励、verifier），
  Reasoning Agent 在里面学。Designer 的奖励不是通过率，而是
  hint 前后的 regret 差：r = E[R | 有提示] − E[R | 无提示]。
  Designer 必须从大规模预训练语料采样文档做 grounding，
  并维护环境记忆库做增量演化。30B 一轮 400 步生成 3,310 个独立环境。
  Qwen3-30B-A3B 在 8 个 held-out 评测上 50.2 → 58.3。
  消融：去掉 Designer 训练和记忆直接崩到 40.5，比不训还低 9.7 个点。
- 思考：hint-regret 和“学习潜力”度量有什么区别？为什么无锚定的 Designer 会主动误导 Agent？

## 02 AED：失败样本第一次当成主资产

- 论文：Agent Error Dataset: Scaling 50,000 Error–Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training，Zhu 等，2026，arXiv:2609.40111。
- 注释：50,228 条 error–diagnosis 对，来自 9,961 个任务，
  33 个环境、19 个 harness、23 个策略模型。五步 AET 流水线：
  收集真实 rollout 里的自然失败 → citation-first 的 judge 出诊断与修复建议
  （每条诊断必须引用原始 trace 里的 token）→ 对可复现任务做
  matched replay 验证修复动作 → 分别建成“诊断 SFT 视图”和
  “actor 恢复 SFT 视图”。Qwen3-8B 做诊断，teacher-label 的
  exact-step 一致率从 47.19% 升到 63.56%。
  警告：只训“出错后怎么修动作”，在 WebShop-lite 上 +6.67 个点，
  在 TextQuest 上反而 −8.33 个点。
- 思考：诊断为什么是可密集学习的“结构化判断”任务？修复监督为什么高度依赖环境？

## 03 WorkForge：真实工作空间才是瓶颈

- 论文：Scaling Verifiable Environments for Long-horizon Work Agents，Zhang 等，2026，arXiv:2610.04906。专栏称之为 WorkForge。
- 注释：直接从真实资源（网络检索 + 真实文件）建工作空间，
  再从真实专家工作流里导出任务指令、解题计划和“程序化 + 语义”双 verifier。
  一共 16.7K 个可验证环境，覆盖 40 个专业领域、60 种文件类型。
  真实工作空间比 LLM 合成的多 5.4 倍文件、2.4 倍文件类型，
  平均大小从 21.5KB 跳到 279KB。Qwen3.5-35B-A3B-Base + WorkForge，
  GDPval 从 45.5 → 73.6；27B dense 版 GDPval 82.4，
  超过 Qwen3.5-397B-A17B 的 79.7。
- 思考：真实感的哪一部分最关键——文件数量、类型多样性，还是工作流本身？

## 04 ToolLoop：闭环比数据量重要

- 论文：ToolLoop: Closed-Loop Tool-Use Data Synthesis via Decomposed Generation and Dynamic Self-Feedback，Zeng 等，2026，arXiv:2609.09072。
- 注释：三段分解 + 动态自验证 + 反馈驱动再生：先把 5,281 个 API
  聚成 26 组采样组合 → 反向生成强关联的用户 query → 正向生成工具调用。
  每步都有 verifier，出错就带着错误原因重生成，而不是整条丢掉。
  Qwen3-4B 只用 11K 样本，BFCL 从 82.14% → 86.40%，
  超过 60K 的 APIGen（83.11%）和 55K 的 ToolMind（83.53%）。
  最反直觉的消融：去掉反馈的版本掉到 79.97%，比没训过的 base 还低 2.17 个点。
- 思考：反馈修正的收益来自“修错”还是“筛掉难样本”？反向生成 query 会不会引入分布偏差？

## 05 Data Turnstile：小模型的工具数据有毒

- 论文：Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation，Ramakrishnan 等，2026，arXiv:2607.29250。
- 注释：把多轮工具交互拆成一个 DAG，按 role 分节点
  （用户 query / 推理 / API 调用 / 执行输出 / 助手回复），
  每个 role 独立生成、独立验证、带错误反馈重试，
  避免单次多轮合成的级联错误。教师用本地跑的 Qwen2.5-32B-Instruct，
  不依赖闭源 API。Qwen3-0.6B 的 BFCL 从 no-think 58.2% → 75.9%；
  τ²-bench Telecom 的 pass^1 从 3.5 → 23.0。
  而 raw xLAM+Glaive 直接训 0.6B 会掉到 55.1%，Irrelevance 检测崩到 35.7%。
  CoT 分化：单轮 BFCL 上高质量 SFT 后 no-think 反而更好，
  但多轮 τ²-bench 上关掉 CoT 全线崩 11–22 个点。
- 思考：DAG 分角色验证的代价是什么？多轮任务里 CoT 为什么不可替代？

## 06 ToolRACER：专门练“不合作用户”

- 论文：ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation，Chakraborty 等，2026，arXiv:2610.09163。
- 注释：三个模拟模型配合（用户模拟器、助手、工具模拟器），
  显式生成三种轨迹：Happy（合作 + 可满足）、Unhappy（摇摆、漂移、含糊）、
  Impossible（工具满足不了，Agent 应该识别并拒绝）。
  失败轨迹喂回模拟器再生，救回了 65.4% 一开始失败的轨迹。
  最后 5.6K 条多轮对话，约 66% 是容易失败的场景。
  Qwen3-4B 在 τ²-bench 上 32.20% → 39.20%，和 APIGen-MT 叠加能到 41.25%。
  只训 Impossible 子集能把 BFCL 的 Irrelevance 检测从 85.83% 拉到 95.50%。
  但不同路径会互相干扰：H+I 合起来反而比单路径差，不能简单拼接。
- 思考：三种轨迹为什么不能简单拼接？Impossible 训练的泛化边界在哪里？

## 07 Spotify：合成数据 + 自改进闭环直接上生产

- 论文：Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops，Palumbo 等，2026，arXiv:2609.30297。
- 注释：本系列唯一的工业端到端案例，没有微调模型。
  先把单轮 prompt 规划成对话计划，再由 UserLLM 实现成多轮对话，
  用 LLM-as-judge 评估，替代真实用户做测试集；
  再做自我改进循环，对同一问题多次采样算方差，把失败拆成
  可靠性失败（高方差，同 prompt 结果不稳定）和能力缺口
  （低方差但系统性错），前者走方差对比优化，
  后者交给 coding agent 自动修 prompt / 工具描述 / planner 配置。
  合成对话的人工评审四维都在 90% 以上。
  线上 A/B 两周、约 1,500 万用户：收听时长 +14%，WAU +5%，跳过率 −5%。
  但长对话还是难：7 turn 比 2 turn 的 pass@1 低 11.9 个点。
- 思考：方差拆分在什么情况下会误判能力缺口为可靠性问题？合成测试集的天花板在哪里？

## 08 Harness-S：兄弟 harness 救活 GRPO 有效组

- 论文：Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training，Zhang 等，2026，arXiv:2611.04277。
- 注释：痛点是原 harness 全组失败时 GRPO 组内无奖励方差。
  方法是 edit agent 按失败原因生成兄弟 harness；仅当整组失败才路由
  多兄弟重跑，救援组替换失败组；按行为画像路由并随策略更新重校准。
  Qwen3.5-4B AppWorld Avg@3 48.98% → 56.87%；
  有效组比 64.69% → 79.38%；代价 +18.44% rollouts。
- 思考：兄弟 harness 的“兄弟”关系怎么定义才不引入捷径？救援组的奖励方差是真信号还是人工抬起来的？

## 09 Agent-R²：真实附件 + 交付级 rubric 的办公 agent 数据

- 论文：Agent-R²: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis，Pan 等，2026，arXiv:2611.08219。
- 注释：痛点是办公 agent 常不读附件、不验算就交文件。
  方法是 O*NET 活动播种、真实附件检索验证、沙箱校验、
  专家执行 + verifier 核对交付物；SFT 后按交付 rubric 做 RL。
  GDPval Pass@1 32.4% → 45.3% → 51.0%；
  SFT 后步骤暴涨但完成率跌，RL 后回升——探索与可靠完成要分开量。
- 思考：SFT 跌完成率、RL 恢复的模式，在别的 agent 任务里也出现吗？

## 10 ToolSpace：以可执行工具依赖为合成单元

- 论文：ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents，Dong 等，2026，arXiv:2611.05036。
- 注释：痛点是先造库再让 LLM 编工具，调用关系脱离真实执行。
  方法是从真实 MCP 调用挖掘数据 / 状态 / 重试依赖成加权工具图；
  随机游走 + 状态检查回滚；三重验证 V_state / V_trace / V_out。
  35B 七基准均值 23.0 → 41.0 → 43.2；
  真实执行边胜 LLM 判断边（45.8 vs 37.8）。
- 思考：合成单元应该是“工具”还是“可执行依赖”？这对工具库设计有什么启示？

## 11 Axion 1.0：把“变难”与“真需工具”拆成两条数据流

- 论文：Axion 1.0: Open Agent Foundation Models, Simplified，Jia 等，2026，arXiv:2611.03233。
- 注释：Stage 1 针对基座已会解的题做组合 / 参数化 / 分情形变换再验算；
  Stage 2 在真实结构化数据上做确定性计算、屏蔽建库、防闭卷可答；
  RL 用前缀进度奖励。35B HLE-130 47.7% → 70.0%。
  但 BioMysteryBench SFT-I 先跌后涨，FrontierScience RL 后微跌——
  最后阶段并非处处有效。
- 思考：“变难”和“真需工具”为什么要拆开？合在一起会互相污染什么？

## 12 CUA-Master：从外部教程 import 工作流 + 失败快照回滚

- 论文：CUA-Master: Scaling Computer-use Agents through Open-World Recursive Improvements，Yu 等，2026，arXiv:2611.03126。
- 注释：方法是从教程视频触发点重建轨迹；任务抽象为原语组合；
  失败定位最早偏离点、恢复快照、只保留成功且更高效分支；
  外层循环按能力缺口再采集。长任务同重试预算下
  全重启成功率 14.6% vs 快照回滚 49.8%。
- 思考：快照回滚的收益里，有多少是“少走回头路”，有多少是“避开了难状态”？

## 13 WebHarbor：把真实网站“靠岸”成可离线复现的环境

- 论文：WebHarbor: Docking Real Websites for Evolving GUI Agent Environments，Wang 等，2026，arXiv:2611.03634。
- 注释：方法是按目标任务定范围重建前后端 + SQLite 状态；
  逐步执行修补缺失页 / 状态转移；人工评审视觉、功能、任务有效性。
  离线重建环境迁移到真实网站 WebVoyager 81.2%；
  但 WebArena 迁移有限（28.8%），环境建设成本高（57 人约 5 个月）。
- 思考：离线复现的保真度天花板在哪里？成本这么高，什么任务值得建？

## 14 VidUEU-Agent：视频数据的证据组织 + 分布反馈外循环

- 论文：VidUEU-Agent: A Data-Curation Agent for Multimodal Understanding, Editing, and Unified Tasks，Ru 等，2026，arXiv:2611.04418。
- 注释：方法是三级摘要 + 镜头过滤；按内容规划 OCR / ASR / 检测 / 光流工具；
  样本必须引用所用信号 ID；每轮 30K 数据、按评估调分布后扩到 100K。
  InternVL3-8B MMT-Bench 0.546 → 0.596。
  注意反馈用的是测试集聚合分，held-out VSI-Bench 才是迁移的可靠读数。
- 思考：用测试集聚合分调分布，和直接在测试集上训，边界在哪里？

## 15 PyRUA-DataFarm：造数据与批数据分离的机器人数据

- 论文：PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data，Zhang 等，2026，arXiv:2611.05320。
- 注释：方法是编码 agent 重建场景写专家控制器；
  独立角色先冻结成功判定；视频模型只按真实参考图渲染外观；
  每集机器检查 + 独立审片。5 个双臂任务原场景 93/100，
  胜真人遥操 84/100；未见桌布 / 灯光下 87 vs 46；单集摊薄约 $1.06。
- 思考：造数据与批数据分离，为什么是机器人数据特有的关键设计？
