# EP155 — SkyRL：模块化 RL 后训练框架设计，与 397B Office Work Agent 的 RL 训练实战

## 元信息

- 期号：155
- 标题：SkyRL：模块化 RL 后训练框架设计，与 397B Office Work Agent 的 RL 训练实战
- BV：BV1creh6LEu6
- 时长：01:28:18
- 提炼日期：2026-09-20
- 相关论文：SkyRL-Agent https://arxiv.org/abs/2511.16108
- 相关代码：https://github.com/NovaSky-AI/SkyRL ；实战 recipe https://github.com/mercor-intelligence/apexagents-skyrl-recipe
- 实战博客：https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/

> ⚠️ 提炼方式说明：B站反自动化，视频字幕未能直接获取。本纪要根据该期 talk 对应的公开材料还原——SkyRL 官方论文/代码 + Mercor 2026-09-01 发布的 397B 训练博客（SkyRL 官方 README 明确标注为双方合作的 E2E recipe，即 talk 下半场的底稿）。Q&A 即兴内容未覆盖。

## 一句话总结

上半场讲 SkyRL 的模块化 RL 训练栈（Tinker-first、fully-async）；下半场是 Mercor 用 SkyRL 在公开栈上把 Qwen3.5-397B-A17B 纯 RL 训成 office work agent 的完整实战手册——**de-risking 流程比算法选择更重要**。

## 核心

### 上半场：SkyRL 模块化设计（Berkeley Sky Computing Lab + Anyscale）

四个可独立使用的组件：

1. **skyrl-train**：模块化训练框架本体；fully-async 训练 + in-flight NCCL 权重同步；vLLM 推理 + Megatron 训练后端
2. **skyrl-tx**：Tinker API 的开源后端实现。"Tinker-first" 设计——同一套训练循环可换算力后端，这是与 veRL 最大的差异化
3. **skyrl-agent**：多轮长程 agent 训练层；核心是优化过的异步 pipeline dispatcher（比 naive 异步 batching 快 1.55x）+ 轻量工具接入
4. **skyrl-gym**：Gymnasium 接口的环境库（math / code / search / SQL）

代表作（SkyRL-Agent 论文）：纯 RL 把 Qwen3-32B 从 24.4% 训到 **SWE-Bench Verified 39.4% Pass@1**，成本比同水平 prior work 低 2 倍以上；只在 SWE 上训，泛化到 Terminal-Bench、BrowseComp-Plus、WebArena。

### 下半场：397B Office Work Agent RL 实战（Mercor）

- 基座 **Qwen3.5-397B-A17B（MoE）**，**不做 SFT warmup、纯 RL**，1928 个专家标注任务
- APEX-Agents（480 个长程知识工作任务）Pass@1：16.11% → **27.29%**（相对 +70%）；35B 版本（Qwen3.6-35B-A3B）直接超过 Opus 4.5
- 方法论是 6 步，其中 Step 1–3 是 de-risking，**Step 4 之前不烧大算力**

**Step 1 — 环境 / harness / token记账**：先别开训。harness 修 bug 零训练就把 35B 从 22.74% 干到 28.69%（约等于白赚一个 epoch）。关键概念 **TITO**（token-in-token-out）：推理引擎输出转给 trainer 时不能 re-tokenize，否则静默 off-policy。Mercor 用的是改 harness 走 `/completions` 的方案。

**Step 2 — 系统调优**：fully-async 是长程任务默认选项。调优顺序：Megatron 并行参数（TP/EP/PP/CP、offload、dynamic micro-batch）→ rollout/train 卡数切分（以 trainer 不饿死为准；35B 用 12:4，397B 用 12:8）→ rollout 并发度取 min（KV cache 上限，staleness 上限 `(3+1)×16×16=1024`；实际 35B 用 550，397B 用 300）。用 trainer/inference **logprob 差（<0.03 健康）** 抓 train-inference 不一致——他们真抓到过 vLLM CPU offload + GDN + in-flight 更新组合的 correctness bug。

**Step 3 — overfit 跑**：32 个任务同步训，能 overfit 才有资格开 hero run；在这里抓到了 grading 逻辑 bug（文件 diff 判分 fidelity 不够，换第三方 diff 工具后任务才变 learnable）。

**Step 4 — 35B 上做算法消融**（结论可平移到大模型）：
- `prompt_mean`（按 rollout group 平均）比 `token_mean` 高 **+3.9pt**——轨迹长度 2k~128k，token 平均会被长轨迹绑架梯度
- DPPO vs GLM-5 loss 分数打平，但 DPPO 让行为更"干练"（turn 数 21→32，每 turn token 834→588）
- context 剩 20% 时 nudge 模型收尾：**+3.0pt** 纯训练时效应
- overlong filtering 反而 **-1.5pt**；adaptive length penalty 中性偏负
- Hero run 配置：**DPPO + prompt_mean + nudge**

**Step 5 — 397B hero run**：35B 和 397B 唯一的差别就是系统工程，算法 knob 直接平移。

**Step 6 — 泛化**：换掉训练用的 MCP harness、用纯代码的 OpenCode 测，提升大体保留——**RL 训出的能力跟着模型走，不是焊死在 harness 上的**；35B 泛化比 397B 好（它学会了更依赖 code execution 而非 MCP）。Terminal-Bench 2.1 同样泛化；HLE/GPQA 无退化。

## 关键数字

| 指标 | 基线 | 结果 |
|---|---|---|
| SA-SWE-32B SWE-Bench Verified Pass@1 | 24.4% | 39.4%（纯RL，成本<1/2）|
| 397B APEX-Agents Pass@1 | 16.11% | 27.29%（相对+70%）|
| harness 修 bug（零训练）35B mean reward | 22.74% | 28.69% |
| prompt_mean vs token_mean | baseline | +3.9pt |
| context nudge | baseline | +3.0pt（训练时）|
| async dispatcher vs naive | 1x | 1.55x |

## 可迁移

- **Agentic RL infra 必做清单**：TITO token 对齐检查、trainer/inference logprob 差监控（<0.03）、rollout 并发度取 min（KV cache 上限，staleness 上限）——这三项是 correctness 地基
- **开大 run 前的 de-risk 顺序**：环境鲁棒性（error 率压到 ~0）→ harness 修 bug（读 trace，用 coding agent 批量分析 per-tool 失败率）→ overfit 小子集验证 learnable → 小模型上做消融 → hero run
- 对 45-day 计划的直接输入：`post-training-framework/` 的 batch semantics / staleness / rollout-weight-sync 章节可用这套实战数字做注脚

## 疑问 / 下一步

- SkyRL 的 TITO proxy（issue 提到 coming soon）落地后，harness 侧还需要自己做 `/completions` 改造吗？
- 397B 的 inference:train=12:8，这个比例随模型变大怎么 scale？KV cache 上限公式里 trajectory 长度分布是关键变量

## 原文金句

> "Steps 1 to 3 are de-risking. We do not spend significant compute until Step 4."
> "Algorithm choices mattered less than the data: the best of five knobs gave +3.9 points, while post-training as a whole moved both models 10 to 12 points."
