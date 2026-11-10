# Synthetic Data / Agentic 数据合成与清洗系列

Agentic 方式合成 / 清洗训练数据（SFT・mid-training・RL）系列，
与 AI4Math / RLVR（A 系列）和 RL 环境与合成数据（B 系列）并行，
是 daily ai data 阅读的第三条线（C 系列）。

来源：Lei Tian 专栏《Agentic视角下的数据链路》，
知乎慢读清单 id 259（2026-11-02 刊）与 id 320（2026-11-09 刊），
两期共 15 篇论文，正好一天一篇、15 天读完。

## 核心判断

单轮“生成再过滤”过时了。现在的主流是分阶段生成-验证-修正的
多角色闭环。数据合成的竞争点已经从“能不能生成”变成
“能不能验证、能不能修、能不能长出来”。

本系列沿着这条线读 15 篇：

1. **环境自生成**：环境跟着 Agent 的能力边界一起长（SPADE）。
2. **失败即资产**：失败轨迹的诊断与修复是第一手训练信号（AED）。
3. **真实感缩放**：真实工作空间与真实工作流才是瓶颈（WorkForge）。
4. **闭环合成**：没有反馈控制的合成数据会主动污染模型（ToolLoop、Data Turnstile）。
5. **失败场景**：不合作用户与不可满足意图要显式练（ToolRACER）。
6. **可验证信号**：把真实交互变成下一轮可验证的训练信号（Harness-S、Agent-R²、ToolSpace、Axion）。
7. **工业闭环**：合成数据 + 自改进循环直接换成线上指标（Spotify）。

## 路线表（15 篇，每天一篇）

| 天 | 论文 | 作者 | 年份 | 出处 | 一句话 |
| --- | --- | --- | --- | --- | --- |
| 01 | [SPADE: Self-Play in Adaptive Synthetic Executable Environments](https://arxiv.org/abs/2608.19197) | Liu 等 | 2026 | arXiv | hint-regret 让环境卡在 Agent 能力边界上，30B 一轮 400 步生成 3,310 个环境 |
| 02 | [Agent Error Dataset: Scaling 50,000 Error–Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training](https://arxiv.org/abs/2609.40111) | Zhu 等 | 2026 | arXiv | 50,228 条 error–diagnosis 对，失败样本第一次当成主资产 |
| 03 | [Scaling Verifiable Environments for Long-horizon Work Agents](https://arxiv.org/abs/2610.04906) | Zhang 等 | 2026 | arXiv | WorkForge：16.7K 真实工作空间，真实感是学会探索的原因 |
| 04 | [ToolLoop: Closed-Loop Tool-Use Data Synthesis via Decomposed Generation and Dynamic Self-Feedback](https://arxiv.org/abs/2609.09072) | Zeng 等 | 2026 | arXiv | 闭环比数据量重要：无反馈合成数据比不训还差 2.17 个点 |
| 05 | [Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation](https://arxiv.org/abs/2607.29250) | Ramakrishnan 等 | 2026 | arXiv | 多轮工具交互拆成 DAG 分角色验证；小模型上未加工合成数据有毒 |
| 06 | [ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation](https://arxiv.org/abs/2610.09163) | Chakraborty 等 | 2026 | arXiv | Happy / Unhappy / Impossible 三种轨迹，专练不合作用户 |
| 07 | [Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops](https://arxiv.org/abs/2609.30297) | Palumbo 等 | 2026 | arXiv | 合成对话当测试集 + 方差拆分自改进，线上 A/B 收听时长 +14% |
| 08 | [Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training](https://arxiv.org/abs/2611.04277) | Zhang 等 | 2026 | arXiv | 兄弟 harness 救活 GRPO 有效组 |
| 09 | [Agent-R²: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis](https://arxiv.org/abs/2611.08219) | Pan 等 | 2026 | arXiv | 真实附件 + 交付级 rubric 的办公 agent 数据 |
| 10 | [ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents](https://arxiv.org/abs/2611.05036) | Dong 等 | 2026 | arXiv | 以可执行工具依赖为合成单元 |
| 11 | [Axion 1.0: Open Agent Foundation Models, Simplified](https://arxiv.org/abs/2611.03233) | Jia 等 | 2026 | arXiv | 把“变难”与“真需工具”拆成两条数据流 |
| 12 | [CUA-Master: Scaling Computer-use Agents through Open-World Recursive Improvements](https://arxiv.org/abs/2611.03126) | Yu 等 | 2026 | arXiv | 从外部教程 import 工作流 + 失败快照回滚 |
| 13 | [WebHarbor: Docking Real Websites for Evolving GUI Agent Environments](https://arxiv.org/abs/2611.03634) | Wang 等 | 2026 | arXiv | 把真实网站“靠岸”成可离线复现的环境 |
| 14 | [VidUEU-Agent: A Data-Curation Agent for Multimodal Understanding, Editing, and Unified Tasks](https://arxiv.org/abs/2611.04418) | Ru 等 | 2026 | arXiv | 视频数据的证据组织 + 分布反馈外循环 |
| 15 | [PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data](https://arxiv.org/abs/2611.05320) | Zhang 等 | 2026 | arXiv | 造数据与批数据分离的机器人数据 |

## 推送节奏

- 本系列每天一篇，与 A、B 系列并行。
- 每篇笔记按 [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) 写，进度记在 [reading-log.csv](reading-log.csv)。
- 详细注释见 [papers.md](papers.md)。

## 核实说明

- Day 01–07 已于 2026-11-09 经 arXiv API 逐篇核实：真实标题、作者、arXiv ID 与可用链接。
  其中 Day 03 在 arXiv 上的标题是 Scaling Verifiable Environments for Long-horizon Work Agents，
  专栏称之为 WorkForge，已如实标注。
- Day 08–15 已于 2026-11-09 通读知乎原文
  https://zhuanlan.zhihu.com/p/2103100374513082636 核对 arXiv ID，
  再经 arXiv API 逐篇核实标题与作者：Harness-S 2611.04277、
  Agent-R² 2611.08219、ToolSpace 2611.05036、Axion 1.0 2611.03233、
  CUA-Master 2611.03126、WebHarbor 2611.03634、VidUEU-Agent 2611.04418、
  PyRUA-DataFarm 2611.05320。专栏原文只给机构出处、不给个人作者名，
  作者栏以 arXiv 为准。
- 本仓库不含任何私人信息；作者栏只列论文作者人名，不列机构。
- 与 A、B 系列的关系：A 读数学推理数据，B 读环境与数据系统本身，
  C 读 agentic 合成 / 清洗数据管线本身。三边篇目原则上不重叠。
