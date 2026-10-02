# EP075 — FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP075-flashrl-tis.html

## 元信息

- 期号：75
- 标题：FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案
- BV：BV1J6WyzTEJP（https://www.bilibili.com/video/BV1J6WyzTEJP/）
- 时长：未确认（2025-08-30 直播安排 11:00–12:00）
- 提炼日期：2026-10-02
- 信息来源：论文还原（B站字幕接口在本环境不可达，未能取得 AI 字幕）
- 讲者：姚峰（Feng Yao，UCSD 二年级博士生，导师商静波 Jingbo Shang；曾在微软研究院实习）
- 相关论文（技术博客形态）：Your Efficient RL Framework Secretly Brings You Off-Policy RL Training（Feng Yao、Liyuan Liu、Dinghuai Zhang、Chengyu Dong、Jingbo Shang、Jianfeng Gao，2025-08），https://fengyao.notion.site/off-policy-rl ；FlashRL 项目页 https://fengyao.notion.site/flash-rl
- 相关代码：https://github.com/yaof20/Flash-RL（`pip install flash-llm-rl`）

> ⚠️ 提炼方式说明：本期字幕未能取得，本纪要根据公开材料还原——青稞Talk 官网预告文 + 讲者团队的 FlashRL 项目页与 off-policy 技术博客。官网提纲（mismatch 问题 → TIS 方案 → 8-bit rollout → 未来方向）与博客内容对应；视频实录与 Q&A 未覆盖。

## 一句话总结

现代 RL 框架为了吞吐，把 rollout 交给推理引擎（vLLM/SGLang）、训练交给训练后端（FSDP/Megatron）——但两个引擎对**同一组参数**算出的 token 概率可以显著不同，甚至一个给 1、另一个给 0。于是名义上的 on-policy 训练被静默变成了 off-policy，梯度被系统性污染。FlashRL 的解法是截断重要性采样（TIS）：用推理/训练两侧概率比构造 token 级修正因子、截断封顶后乘到 policy loss 上；在此之上再把 rollout 权重降到 INT8/FP8，**8-bit 生成、16-bit 效果**，rollout 吞吐（约占 DAPO-32B 总训练时间的七成）被显著压缩。

## 核心

### 背景/问题：rollout-training mismatch——同参数、不同分布

RL 训练里 rollout 生成是主要瓶颈（讲者给出 DAPO-32B 口径下约占总训练时间 70%），所以主流框架（以 verl 为代表）普遍采用"推理引擎采样 + 训练后端更新"的混合架构。问题在于：推理引擎 $\pi_{\text{sampler}}$ 与训练后端 $\pi_{\text{learner}}$ 加载同一份参数 $\theta$ ，数值路径却不同（kernel、并行布局、量化），同一 token 的概率可以差异巨大，个别 token 甚至出现 $\pi_{\text{vllm}}(a)=1$ 而 $\pi_{\text{fsdp}}(a)=0$ 的矛盾预测。REINFORCE/PPO 的更新公式默认采样分布就是 $\pi_{\text{learner}}$ ，这个前提被打破后，训练在无人察觉的情况下变成 off-policy——这是本期的核心诊断：**错位不是精度洁癖，而是梯度估计的系统性偏置**。

### 方法/设计：TIS 修正 + 在线量化 rollout

1. **截断重要性采样（TIS）**：逐 token 计算训练侧与推理侧的概率比 $w = \pi_{\text{train}}(\theta_{\text{old}}) / \pi_{\text{rollout}}(\theta_{\text{old}})$ ，截断为 $\min(w, C)$ 后作为修正因子乘到 policy loss 上。与 PPO clip 的区别是关键：TIS 作用在**更新前**的分布错位上、用乘性权重衰减异常 token，而不是裁剪**更新步**的 surrogate 目标；两个 ratio 是两个独立的量（SkyRL 的 FlashRL 集成示例中 `TIS_IMP_RATIO_CAP = 8.0`，verl 的 FP8 实践用 token-level TIS 且 $C = 2$ ，阈值口径随框架而定）。
2. **在线量化支持**：这补上了第二个工程障碍——vLLM 为 serving 优化，原生不支持"训练中带参数更新的量化推理"。FlashRL 提供 `flash-llm-rl` 包为 vLLM 打补丁，使权重同步时能按 INT8/FP8 在线量化并加载（INT8 需要校准 profile 把 BF16 权重映射到量化格式）。
3. **为什么量化必须配 TIS**：量化把 rollout 分布 $\pi_{\text{int8}}$ 进一步推离训练分布 $\pi_{\text{bf16}}$ ，放大 mismatch。不加 TIS 时 INT8/FP8 rollout 相比 BF16 有显著精度掉点；加上 TIS 后，量化 rollout 的下游精度与 BF16 rollout + TIS 持平、甚至超过不加 TIS 的朴素 BF16 训练。测得的 KL 口径也印证：INT8 rollout 与训练侧的 KL 大于 FP8。

### 实验/实战（项目页口径）

- 场景：DAPO recipe，Qwen2.5-32B，BF16 FSDP 训练后端；FP8 在 H100、INT8 在 H100 与 A100 上测。
- 结论形态：带 TIS 的 INT8/FP8 rollout 在 AIME、GSM8K 上达到与 BF16 rollout 一致的精度曲线；不带 TIS 的量化 rollout（图中灰色虚线）明显掉点。
- FlashRL 自称是第一个开源且可用的量化 rollout RL recipe，后被 SkyRL 原生集成（仅支持单轮训练），OpenRLHF 也集成了 TIS。

### 结论/观点（区分事实与判断）

- 事实：train/inference 两侧概率不一致是实测现象，TIS 修正与量化 rollout 的精度结论有项目页图表与多家框架（SkyRL、OpenRLHF、TRL 文档）采纳背书。
- 判断（讲者立场）：凡是"推理引擎 + 训练后端"的混合架构，都应默认自己在做 off-policy 训练，并把 mismatch 当作需要显式监控与修正的一等公民，而不是等训练崩溃了再回头找。

## 关键数字

| 指标 | 基线/对照 | 结果 |
|---|---|---|
| Rollout 占总训练时间（DAPO-32B 口径） | — | 约 70% |
| 量化 rollout 不加 TIS | BF16 rollout | 显著精度掉点（AIME/GSM8K） |
| 量化 rollout + TIS | BF16 + TIS | 精度持平，且超过不加 TIS 的朴素 BF16 |
| TIS 截断阈值 $C$ | — | SkyRL 集成示例 8.0；verl FP8 实践 2 |
| 两侧概率矛盾样本 | 假设一致 | 实测存在 $\pi_{\text{vllm}}(a)=1$ vs $\pi_{\text{fsdp}}(a)=0$ 的 token |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **给 rollout/训练一致性上监控**：同一批 token 上两侧 log-prob 的差（或 rollout correction KL）应该是仪表盘常驻指标；数值实现一换（引擎版本、并行布局、量化）就可能悄悄漂移。
  2. **rollout 想上低精度，先问修正跟不跟得上**：量化加速的前提是 TIS 类修正已接入且阈值调过；裸上 FP8/INT8 rollout 是用精度换吞吐的隐性交易。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：mismatch 是"引擎选型"这一 infra 决策泄漏到算法层的典型案例；框架层把 $\pi_{\text{sampler}}$ 与 $\pi_{\text{learner}}$ 显式区分为三个策略（采样/近端/目标）来记账，比事后调 clip 更治本（本仓库 `post-training-framework/` 的 rollout 章节可与本期对照）。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：token-level TIS 被后续工作质疑为有偏梯度估计（有文献提出 sequence-level masked IS 作为无偏替代），两者在长序列上的偏差-方差取舍如何量化？值得对照 MIS 类工作再深挖。
- 本期字幕若后续取得，应回补讲者对"未来研究方向"的具体表述。

## 原文金句（1-2句）

> "This unexpected behavior implicitly breaks the on-policy assumption, secretly making the RL training become off-policy."——off-policy 技术博客对 mismatch 的定性：你的高效框架，正在悄悄把 on-policy 训练变成 off-policy。
