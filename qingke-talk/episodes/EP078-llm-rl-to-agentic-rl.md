# EP078 — 从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP078-llm-rl-to-agentic-rl.html

## 元信息

- 期号：78
- 标题：从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体
- BV：BV1sxWgzUEZo（https://www.bilibili.com/video/BV1sxWgzUEZo/；另有同题重复上传 BV1sBsFz9EVw）
- 时长：未确认（2025-09-18 直播安排 20:00–21:00）
- 提炼日期：2026-10-02
- 信息来源：论文还原（B站字幕接口在本环境不可达，未能取得 AI 字幕）
- 讲者：张桂彬（Guibin Zhang，新加坡国立大学计算学院博士生，导师颜水成 Shuicheng Yan；G-Designer、G-Safeguard、MaAS、AgentPrune 等工作作者）
- 相关论文：The Landscape of Agentic Reinforcement Learning for LLMs: A Survey（Guibin Zhang、Hejia Geng 等，arXiv:2509.02547，2025-09；后发表于 TMLR），https://arxiv.org/abs/2509.02547
- 相关代码：综述配套汇编（开源环境 / benchmark / 框架清单）见论文项目页

> ⚠️ 提炼方式说明：本期字幕未能取得，本纪要根据公开材料还原——青稞Talk 官网预告文 + 讲者一作的 Agentic RL 综述（arXiv:2509.02547）。官网四段提纲（为什么需要 Agentic RL / POMDP 统一框架 / 动态交互过程 / 应用与未来）与综述骨架对应；视频实录与 Q&A 未覆盖。

## 一句话总结

这期是 Agentic RL 的"概念奠基课"：传统 LLM-RL 本质是**单步、完全可观测的退化 MDP**（给 prompt、出答案、拿奖励，一锤子买卖）；而 agent 要在动态环境里多步行动、只能看到局部观测，必须升级为 **POMDP** 形式化。讲者用这个统一框架把规划、工具使用、记忆、推理、自我改进、感知六类能力与各应用域组织成一张版图（综述综合 500+ 篇工作），核心论点是：RL 是把这些原本靠提示词和启发式拼出来的模块，变成可端到端优化的适应性行为的机制。

## 核心

### 背景/问题：LLM-RL 的形式化天花板

主流 LLM-RL（RLHF/RLVR）的交互结构是退化的：horizon $T = 1$ 、折扣 $\gamma = 1$ 、状态完全可观测。策略只需学好"一次生成"，不需要处理行动后果、环境反馈与长期信用分配。但自主智能体的任务形态完全不同：多轮工具调用、环境状态不可见、奖励稀疏且延迟。继续用单步 MDP 的语言描述 agent，会把最难的问题（长程规划、部分可观测下的信息获取、与环境共同演化）藏进提示词工程里，无法优化、也无法比较。这就是"为什么需要 Agentic RL"的答案：不是多一个应用场景，而是换一套决策形式化。

### 方法/设计：POMDP 统一框架 + 双轴分类法

1. **形式化对比**：传统偏好式 RL 微调可写成 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, T=1, \gamma=1 \rangle$ 的退化 MDP；Agentic RL 则是 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{O} \rangle$ ，其中观测 $o_t = \mathcal{O}(s_t)$ 只是真实状态的部分投影， $T > 1$ 、 $0 < \gamma < 1$ 。LLM 在这个框架里就是策略本身：以观测（上下文）为输入、以文本/工具调用为行动、在环境中滚动。
2. **双轴分类法**：轴一按核心能力组织——planning、tool use、memory、reasoning、self-improvement、perception，问的是"RL 在优化 agent 的哪种能力"；轴二按应用域组织（代码、搜索、GUI、科学等），问的是"在哪类环境里优化"。两轴交叉给出研究版图，综述据此综合 500 余篇工作，并汇编开源环境、benchmark 与框架清单。
3. **动态交互过程**：在 POMDP 下，agent 与环境的交互是"观测 → 行动 → 环境转移 → 新观测"的滚动过程；RL 的作用是把这个循环里的启发式模块（固定提示模板、规则式重试、静态记忆）逐个替换为可学习的策略组件——这是"从模块拼装到行为优化"的转变。

### 实验/实战：综述口径的代表性证据

综述汇总的实证结论形态是：Agentic RL 在多步任务上系统性地优于单步偏好优化与纯提示工程。作为被引用的代表例，以 outcome-based reward 训练的 DeepCoder-14B 在 LiveCodeBench 上取得约 +8 个百分点的 Pass@1 提升（综述所引工作口径）。讲者本人的一串工作（G-Designer 用图结构设计多智能体拓扑、MaAS 做多智能体架构搜索、AgentPrune 做通信剪枝）则展示了"结构层面的决策"同样可以纳入学/搜的范畴——agent 的拓扑不只是行动空间，也可以是被优化对象。

### 结论/观点（区分事实与判断）

- 事实：单步 MDP 与 POMDP 的形式化差异、双轴分类、500+ 工作的汇编是综述可核查的内容。
- 判断（讲者立场）：Agentic RL 的关键命题是把 LLM 从"被动的序列生成器"重构为"嵌入动态世界的自主决策者"；RL 是实现这一重构的关键机制，未来的瓶颈在环境供给、长程信用分配与可扩展评测，而非再多几个偏好优化变体。

## 关键数字

| 指标 | 基线/对照 | 结果 |
|---|---|---|
| 综述综合工作量 | — | 500+ 篇 |
| 形式化 horizon | 传统 LLM-RL $T = 1$ 、 $\gamma = 1$ | Agentic RL $T > 1$ 、 $0 < \gamma < 1$ 、部分可观测 |
| 代表实证（综述所引） | DeepCoder-14B 训练前 | LiveCodeBench Pass@1 约 +8pt（outcome reward） |
| 能力轴分类 | — | 6 类：planning / tool use / memory / reasoning / self-improvement / perception |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **给 agent 训练任务先写 POMDP 五元组**：状态是什么、观测是什么（上下文窗口里到底有什么）、行动空间、奖励何时到账、折扣怎么取。写不出来的任务，多半会在 rollout 阶段以不可调试的形式爆出来。
  2. 评测选型按双轴定位：先确认要测的是哪种能力（规划/工具/记忆…）、哪类环境，再去综述的 benchmark 汇编里挑，而不是按热度挑。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：POMDP 化之后，infra 的核心矛盾从"生成吞吐"变成"环境吞吐与状态管理"：环境实例的并发、状态快照/回滚、观测组装（token 记账）成为 rollout 成本的主导项——这与本仓库 skyrl-agent / harness-engineering 主题直接接续。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：POMDP 框架下长程信用分配目前主要靠 outcome reward + 组内相对优势（如 GRPO 系），何时需要真正的过程奖励或价值模型？综述的开放问题值得对照 EP155 的实战口径再看。
- 本期字幕若后续取得，应回补讲者在第 4 段对复杂环境应用与未来研究的展开。

## 原文金句（1-2句）

> "The emergence of agentic reinforcement learning marks a paradigm shift… reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds."——综述摘要：Agentic RL 的纲领是把 LLM 从被动生成器重构为动态世界中的自主决策者。
