# 06 — GLM-5.2 PPO Comeback: Task-Dependent RL 的分水岭

> 2026.06 Z.ai (Zhipu AI) GLM-5.2 在 long-horizon 阶段悄悄把 GRPO 换回 PPO，引发 open-source RL 选型再审视。核心不是 "GRPO dead"，而是 RL 算法开始按任务形态分化。

## 背景 GLM-5.2 规格

- 架构：744B MoE，40B active，1M 稳定上下文，MIT license 真开权重
- 平台：HuggingFace + Z.ai API + 20+ coding harness，$1.40/1M in + $4.40/1M out
- 成绩 long-horizon：
- FrontierSWE Dominance 74.4% → 接近 Claude Opus 4.8 75.1%，超 GPT-5.5 72.6%
- SWE-bench Pro 62.1% beating GPT-5.5 58.6%
- PostTrainBench 多小时工程 34.3% vs GPT-5.5 25.0%

## 为什么从 GRPO 回到 PPO Why switch back

**GRPO 假设：** 对同 prompt 采样 $G$ 个完整解 $y_{1..G}$，用 group mean/std 做相对分：

$$\hat A_i = \frac{r_i - \mu_G}{\sigma_G},\quad \mu_G=\frac1G\sum r_j$$

短验可证任务（math 单答案、unit test pass/fail）好用：省一个 $V(s)$，显存 −30-50%

**长程 agent 破裂点：** 真实 agent 轨迹是几百步工具调用 chain：读代码→搜文档→改文件→跑测试→修错。压缩后 sub-trajectory 数 $k$ 与长度 $L_i$ 高度不均，50 步成功的解 vs 500 步成功的解都对，强行同 prompt 组队对比无法公平分组，大量数据不可用

> Because of this, comparing complete solutions, as GRPO does, becomes much less useful.

PPO 回归逻辑：不只看 final correctness，问
- Is agent moving toward goal? Did this action improve future? Did it make task harder?
→ 每个小 action 都给反馈，dense credit。

## Zhipu 的解法

把 value network 请回来，从 group-relative 转向 critic-based PPO，token-level advantage：

$$A_t = Q(s_t,a_t)-V(s_t) \approx \sum_{l\ge0}(γλ)^l δ_{t+l}$$

兼容变长 sub-trajectory，不再依赖 peer 组，重训一个能独立打分任意片段的 evaluator

infra & 工程战术（可抄）：

1. **slime 框架** — 把训练与 inference rollout 分离，统一成 `(state, action, tool_call, feedback)` 四元组格式，支持多 env 并发，并行蒸馏合并 10+ expert，2 天跑完
2. **数据 & 失效过滤** — 双边 clip trust region $[1-ε_ℓ,1+ε_h]$ + token-level mask，记录策略版本序列 $w_0<...<w_k$，若 $w'-w_0>τ$ 丢弃过旧样本；环境崩溃而非模型错导致的失败直接排除；GRPO 里若组半数以上有效则重复填充否则整组丢弃 — 降噪
3. **两段式拦截防 reward hacking** — 规则先滤 + LLM judge 识别可疑 tool call，命中时返回 dummy 信息让轨迹继续而非硬终止
4. **IndexShare 长上下文** — 1 个轻量 indexer 选重要 tokens，邻近 4 层 sparse attention 复用，1M 上下文下 per-token FLOPs 降 2.9×，MTP speculative decode 接受长度 +20%。

## 数学&统计视角

- GRPO 方差：$\text{Var}(\hat A) \propto 1/G$，但要求同 prompt 同分布；长轨迹 $L_i$ 异质时 $σ_G$ 估计有偏，数据利用率 $\propto \frac{\text{可组完整组}}{\text{全量}}$ 崩塌。
- PPO token-level $V(s_t)$ 虽需多训练一个 $P$-参数网络，内存 $O(P)$，但给出 dense 且 length-invariant 的 $A_t$，适用不等长 sub-task 学习分段信号：*Each segment provides useful training signals. Model learns from every important decision instead of waiting until task finishes*。
- 学界佐证：论文 *Learning Without Critics? Revisiting GRPO in Classical RL* 显示无早期终止的长 horizon任务上去 critic 始终落后于有 critic 的 PPO，仅 CartPole 类短 horizon 可持平。

> 判断句：RL 算法选型正变为 task-dependent，不再有 one-size-fits-all 默认

## 影响 & 可迁移点

- **何时仍用 GRPO**：短、可验、答案形态统一的任务（math、单函数 code unit test）。省 critic 稳定便宜，仍是默认。
- **何时转 PPO-critic**：多轮 tool-use、小时级 agent、轨迹压缩后长度分散极大的任务。收益是 dense credit，但要重回 critic 调参与显存成本。
- 对你：code agent 6-10 步可先 GRPO，若上到跨仓库重构/多轮 debug 50-500 步两级分化明显，上 critic/PPO token-level advantage；infra 上复用 slime 式 rollout-training 解耦 +版本追踪丢弃老轨 + 规则+LLM judge 两段拦截。
- DeepSeek 自家也在做类似分工：V4 用 GRPO 训 math/code/agent/instruction 专家，合并回统一模型时切 on-policy distillation→ 说明 2026 共识是分阶段混用。

## 小结

GLM-5.2 意义超出刷榜：它是第一个把 *任务形态决定 RL 算法* 写进公开技术博客并开源复现的样本，给社区的可信分界线：短验=GRPO 便宜稳，长程 agent=PPO+ciritc token-level。2026 后再谈 PPO vs GRPO，不再问谁更好，只问你的轨迹长什么样。

---
Sources: Z.ai blog & medium summary on why traditional training not enoughswitch rationaletoken-level questionslong-horizon evalslime/common formatuneven sub-trajectory & can't group fairlycritic return2-day 10-expert merge & 2-stage interceptionFLOPs 2.9x & IndexSharePricingDeepSeek V4 keeping GRPO for experts switching to distillationtimelines & cost chasm narrativelearning without critics paper
