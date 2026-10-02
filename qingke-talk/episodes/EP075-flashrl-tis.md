# EP075 — FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP075-flashrl-tis.html

> 「当我们使用混合的训练引擎以及推理引擎去做 RL 训练的时候，它可能会带来，就是在更高效的同时，可能会带来一些潜在的 off-policy 的问题，即便是他们使用的是相同的 model 参数，也会有这个问题。」——讲者总结（字幕 42:47–43:08）

## 元信息

- 期号：75
- 标题：FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案
- BV：BV1J6WyzTEJP（https://www.bilibili.com/video/BV1J6WyzTEJP/）
- 时长：00:55:52（讲授约 44 分钟 + Q&A 约 12 分钟；视频总时长 3352 秒）
- 提炼日期：2026-10-02
- 分享嘉宾：姚峰（Feng Yao；字幕中主持人称「姚峰」，43:43；其 UCSD 二年级博士生、导师商静波 Jingbo Shang、曾在微软研究院实习等背景出自官网预告与项目页，字幕未自述，此处作补充）
- 相关论文（技术博客形态，论文补充）：Your Efficient RL Framework Secretly Brings You Off-Policy RL Training（Feng Yao、Liyuan Liu、Dinghuai Zhang、Chengyu Dong、Jingbo Shang、Jianfeng Gao，2025-08），https://fengyao.notion.site/off-policy-rl ；FlashRL 项目页 https://fengyao.notion.site/flash-rl
- 相关代码（论文补充）：https://github.com/yaof20/Flash-RL（`pip install flash-llm-rl`）
- 字幕原文存档：本地 `transcripts/EP75.txt`（1194 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（已存档）清洗提炼（经人工清洗；个别专名可能有识别误差，如 vLLM 在字幕中作「VOM」、TIS 作「TRS/T2S」、FlashRL 作「flash IO」、truncation 作「TRACATION」，均已按公开资料校正）。本期原为论文还原版，现已据字幕升级为字幕实录版：正文以讲授口径为准，凡仅出自技术博客/项目页而讲授未展开的材料，单独标注（论文补充）。

## 一句话总结

主流 RL 框架用推理引擎（vLLM/SGLang）做 rollout、用训练后端（FSDP/Megatron）做更新，两个引擎对同一组参数算出的 token 概率可以显著不同（极端个例一侧给 1、另一侧给 0），名义上的 on-policy 训练被静默变成 off-policy。讲者的解法是截断重要性采样（TIS）：在 verl/OpenRLHF 既有的 recompute 基础上，再乘一个训练侧/推理侧概率比并截断封顶；在此之上把 rollout 权重降到 INT8/FP8 做量化加速，靠 TIS 保持与 BF16 rollout 一致的下游精度。

## 核心

### 背景/问题：rollout–training mismatch——同参数、不同分布

讲者先给出诊断口径：混合引擎架构下，采样分布来自 vLLM，而梯度计算在 FSDP，两者加载同一份参数，同一段文本算出的概率却不一致。其在 DAPO-32B 设定下实测：token 概率差的最大值可达 1.0，即 vLLM 预测概率为 1.0、FSDP 预测为 0.0 的 token 真实存在（字幕 03:25–04:14；这不意味着每个 token 都差这么多，平均差有正有负会抵消到 0.1 量级甚至更小，关键是最大值——若差 1.0 的 token 恰好是 forking token，整条序列的 advantage 都会受影响，字幕 47:51–49:29）。后果是 policy gradient 的 on-policy 前提被打破，训练在无人察觉时变成 off-policy。

讲者归纳 mismatch 的三个来源（字幕 07:12–09:07）：

1. **拿不到真正的采样概率**。vLLM v1 引擎返回的是未经后处理的原始概率（before any post-processing），而真正采样前还要经过 temperature scaling、Gumbel/top-k/top-p 之类的采样处理（字幕作「钢爆 sampling」，疑为 Gumbel 采样），返回的概率不是实际采样分布的概率。
2. **推理引擎的优化带来数值差异**。vLLM 为速度做了大量优化，与 FSDP/HuggingFace 实现之间存在精度差，且「普通人很难去修复」这类系统层面的差异。
3. **并行配置不一致**。推理与训练两侧可能采用不同的并行化配置（如 sequence parallel、tensor parallel 取值不同），进一步放大差异。

### 方法/设计：从系统级修复的失败，到重要性采样修正

**系统级修复走不通**（字幕 09:10–11:23）：一条直观路线是给 vLLM 打补丁拿真采样概率、并提高推理精度。讲者援引 MiniMax 技术报告的做法——把 language model head 改成 FP32 后 train/inference 概率的 gap 明显缩小——但他们自己复现发现：即便用 FP32 LM head，gap 仍然存在（最大差仍稳定在 1），且整个训练速度变得特别慢，没能跑完。结论：系统级精度对齐不能根除问题。

**TIS 的推导**（字幕 11:39–15:45）：从最朴素的 importance sampling 出发——想求目标分布（FSDP）下的期望，但从它采样太慢，于是换成 vLLM 分布采样，再乘概率比做补偿，两边期望一致。落到 REINFORCE/PPO 上，框架现状是：verl、OpenRLHF 在混合引擎训练时会做一次 **recompute**，即用训练引擎重算 PPO 比率分母中的旧策略概率（理论上那里应是 vLLM 的概率，因 gap 太大而重算）。TIS 不动 recompute 的部分，只在整个梯度前面再乘一个截断后的重要性比：

$$w = \min\!\left(\frac{\pi_{\text{FSDP}}(a)}{\pi_{\text{vLLM}}(a)},\, C\right)$$

截断的理由是：概率差最大可达 1.0 时，未截断的比值可能极大，直接乘上会让梯度数值爆掉、训练崩溃（字幕 13:37–13:52）。

**为什么不是其他变体**（字幕 15:50–18:01、32:46–39:52）——这是讲授里论证最细的一段：

- **PPO-IS**（直接拿 vLLM 真概率当 PPO 比率的分母）：PPO 的 clip 是为 trust region 设计的，time step 0 时新旧策略本应相等、比率应为 1、不该触发任何 clip；但 FSDP 与 vLLM 数值不一致会让该比率远超或远低于 1，引发大量不该发生的 clip。且 PPO-IS 与真正的 PPO 梯度并不等价，是一个有偏的形式。实验上它在 INT8 场景基本直接归零/崩溃。
- **Vanilla IS**（不截断的重要性采样）：比值大（如 16、25）时，梯度方差按其平方放大（16²、25²），gradient noise 失控导致不稳定。实验上同样基本归零。
- **TIS 的截断 ≠ PPO 的 clip**：clip 超出区间后梯度变常数（无梯度），影响 token 利用率；TIS 只是把乘性权重封顶为 $C$ ，右侧项仍然有梯度，不改变 token 利用率。

### 实验/实战（全部为讲授字幕口径）

- **32B（DAPO 设定）**：同数据同设置下，加 TIS 后下游性能显著优于 baseline（字幕 18:21–18:38、26:41–27:33）。
- **小模型天然 gap 小**：0.5B 上 vLLM 与 FSDP 的最大概率差只有约 0.4；讲者刻意用 INT8 rollout 把差放大到 1.0 做压力测试——裸 INT8 rollout 性能明显掉，加 TIS 后回到 BF16 rollout + BF16 训练的效果（字幕 18:38–20:02）。
- **TIS 不是万灵药**：1.5B 规模、概率差较小时，TIS 不带来稳定的性能增益，但也不会伤害性能（字幕 20:04–20:49）。社区反馈与此一致。讲者强调 mismatch 真正咬人的场景是大模型 + 难任务：DAPO-32B 上 INT8 rollout 不加 TIS 时 entropy 会 collapse（越来越小），加 TIS 后 entropy 不再塌缩、慢慢涨上去（字幕 21:04–21:29）；而 GSM8K 这类简单任务加不加都无所谓，他猜测部分原因可能是 Qwen 模型在 GSM8K/MATH 上已有 overfit 或 contamination（字幕 21:29–22:01，原话为猜想）。
- **变体对比实验**：PPO + GSM8K + 0.5B，recompute 为虚线基线、TIS 为实线，PPO-IS 与 vanilla IS 在 INT8/FP8 场景基本归零，TIS 明显更好（字幕 33:52–35:02）。
- **recompute 为什么也会坏——熵塌缩的过惩罚机制**（字幕 35:02–37:10）：某个 token 的 advantage 为负时，FSDP 更新后其概率应已变小；但 INT8 的 vLLM 重算时体现不出这个变小（算出的值跟原来差不多），下一轮梯度又会指示「进一步减少」，于是该 token 被 over-penalized，entropy 越来越小。讲者的比喻：老师以为学生没学会、不停地教，最后「学坏掉了」。
- **社区采纳**（字幕 22:04–22:58）：OpenRLHF 在他们发布后不久主动集成了该 feature；verl 已支持（指定参数与 truncation 值即可）；slime 在分享前两天也已支持；Dr. GRPO 作者等社区研究者与 REINFORCE++ 的博客也都提到/试用了 TIS。讲者据此判断「TIS 这种 fix 确实是有效的」。

### 应用：量化 rollout——8-bit 生成、BF16 效果

动机数字：DAPO-32B 每步时间里 rollout 约占 70%，是真正的瓶颈（字幕 23:24–23:52）。裸换 INT8/FP8 推理最快可达 1.7 倍提速，但性能同步掉下来——因为量化把本就存在 gap 进一步放大（字幕 24:05–25:29）。FlashRL 的 recipe 就是「INT8/FP8 rollout + TIS 修正」：

- 实现形态（字幕 25:29–26:41）：对 vLLM 做 patch 后以包的形式发布，`pip install` 后在环境变量里配置用 INT8 还是 FP8，训练侧代码完全不用改（讲授中口播的包名以字幕为准，实际包名见元信息代码链接）。
- 效果（字幕 26:41–31:00）：INT8 rollout + TIS 在 AIME 的 average@32、majority vote、best 三个指标上均可 match BF16 rollout；0.5B GSM8K 上 INT8 与 FP8 两个场景结论一致。讲者另注意到 INT8 与训练侧的概率差比 FP8 更大，后续会以 INT8 为主做压力测试。
- 提速口径（字幕 28:36–30:40）：小模型（7B/14B）量化提速不明显（1.1–1.2 倍，量化 overhead 本身挡路），推荐 32B 及以上再做；脱离 RL 的纯推理基准在不同 GPU/并行配置/响应长度下可达 2–7 倍。端到端对比因生成长度不同难以完全公平，他们用「单位时间内模型 update 次数」统计，INT8 明显更快。
- **INT8 校准只做一次**（字幕 31:00–32:44）：FP8 自带指数位可在线直接量化；INT8 需要 calibration。FlashRL 的做法是只在训练开始时做一次 calibration，后续每次在线更新都沿用最初的投影。依据是一个对比：同一 32B 模型，SFT 后的模型沿用 base 模型的 calibration 结果，性能掉得明显；而 RL 后的模型沿用 base calibration 性能掉得小得多——即 RL 对参数的改变比 SFT「less aggressive」，这是一次校准可复用的原因。

### 结论/观点（区分事实与判断）

- 事实：train/inference 两侧概率不一致是实测现象；TIS 与量化 rollout 的精度结论有讲者实验与多家框架（OpenRLHF、verl、slime）采纳背书；讲授中确认他们 32B 例子里截断常数 $C = 2$ （字幕 54:38–54:52）。
- 判断（讲者立场）：凡是「推理引擎 + 训练后端」的混合架构，都应默认自己在做 off-policy 训练；TIS 修的是 system level mismatch 带来的实现细节问题，「并不是算法本身的问题」（字幕 42:34–42:47）。他还提出两个猜想：① MoE 下 mismatch 会更严重（动态 routing 使 rollout 与 training 激活的专家可能不同，加上引擎为 MoE 做的特殊 kernel 与训练实现不一致；这也是 GSPO 论文提到的问题之一）；② TIS 与 GSPO/GMPO/GFPO 等「各种 PO」正交可叠加，且 GSPO 可能在某种程度上做了 TIS 在做的事，只是 motivation 与修正程度不同（字幕 40:03–42:34）。

### Q&A 要点

- **true sampling probability 是什么**：采样实际用的分布经过 temperature scaling 等后处理，与引擎返回的原始概率不同；FlashRL 的 patch 就是要返回真正采样那一步的概率（字幕 50:26–51:38、53:52–54:12）。
- **「差别大」指最大值不是平均值**：同一 batch 内逐 token 看差值，平均有正有负抵消后只有 0.1 量级，但最大值能到 1.0；关键 token 上的大差足以毁掉整条序列的 advantage（字幕 47:51–49:29）。
- **引擎/后端覆盖**：SGLang 正在推进支持中（与 SGLang、vLLM 团队都在聊）；Megatron 侧目前没处理，自有实现只支持 vLLM + FSDP（字幕 49:29–50:13）。
- **MoE 效果**：还在做，受算力限制属 work in progress（字幕 51:38–52:04）。
- **异步训练同样适用**：异步本质也有 off-policy 问题，可参 AReaL 的 decoupled PPO，其操作与 TIS 相像（字幕 52:04–52:35）；agent RL、多轮场景同样适用——只要用了混合引擎就存在该差异（字幕 53:31–53:52）。
- **INT8 口径**：W8A8，权重与激活都是 INT8（字幕 52:35–52:49）。
- **使用方式**：训练侧若用 verl/slime/OpenRLHF 无需手动改，只需开启 feature——两个变量：返回 vLLM 概率、指定截断常数（32B 例中为 2）（字幕 53:52–55:33）。TIS 的计算就是 $\pi_{\text{FSDP}} / \pi_{\text{vLLM}}$ ，超过 $C$ 取 $C$ 、小于取真实值（字幕 55:05–55:33）。
- 个别问题讲者未展开：如「截断 PPO」一类工作他表示还没仔细看、不做评价（字幕 50:26–50:35）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| DAPO-32B token 概率差最大值 | 假设两侧一致 | 达 1.0（vLLM 给 1、FSDP 给 0） | 字幕（03:25–04:14） |
| 0.5B 天然最大概率差 | — | 约 0.4 | 字幕（18:38–19:00） |
| Rollout 占总训练时间（DAPO-32B） | — | 约 70% | 字幕（23:24–23:52） |
| 裸 INT8/FP8 rollout 提速 | BF16 rollout | 最快约 1.7 倍（但精度掉点） | 字幕（24:05–24:39） |
| INT8 rollout + TIS | BF16 rollout + TIS | AIME 三指标（avg@32/majority/best）match | 字幕（26:41–31:00） |
| 小模型量化提速（7B/14B） | — | 仅 1.1–1.2 倍，推荐 ≥32B 再量化 | 字幕（28:36–29:18） |
| 纯推理基准量化提速 | — | 2–7 倍（视配置与响应长度） | 字幕（29:18–29:50） |
| TIS 截断常数 $C$ | — | 32B 例子中取 2 | 字幕（54:38–54:52） |
| FP32 LM head 系统级修复 | MiniMax 技术报告做法 | gap 缩小但仍在，训练过慢未跑完 | 字幕（09:58–11:23） |
| 旧纪要引用的 SkyRL 集成阈值 | — | `TIS_IMP_RATIO_CAP = 8.0`（集成示例口径，非讲授内容） | 论文补充（项目页/集成文档） |

## 可迁移

- 对 coding data / RL infra 工作的直接可试点：
  1. **给 rollout/训练一致性上监控**：同一批 token 上两侧概率差的**最大值**（而非均值）应是仪表盘常驻指标；讲授口径下平均差会正负抵消骗人，最大值到 1.0 的 token 才决定序列级后果。数值实现一换（引擎版本、并行布局、量化）就可能悄悄漂移。
  2. **rollout 想上低精度，先确认修正接入**：量化放大 mismatch 是可预期掉点，TIS 类修正接入且截断常数调过之后，才谈 INT8/FP8 的吞吐收益；且优先在 ≥32B 规模上做，小模型量化 overhead 吃掉收益。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：mismatch 是「引擎选型」这一 infra 决策泄漏到算法层的典型案例。讲者的处理顺序值得借鉴——先试图在系统层根除（FP32 LM head），证明走不通后才在算法层做显式记账与修正；且修正点选在框架既有 recompute 之上、只加两个开关变量，落地成本极低。本仓库 `post-training-framework/` 的 rollout 章节可与本期对照。

## 疑问 / 下一步

- token-level TIS 在长序列上的偏差–方差取舍：讲授证明了截断的必要性与 PPO-IS/vanilla IS 的失败，但截断引入的偏差随序列长度如何累积，讲授未展开；值得与 sequence-level masked IS 类后续工作对照。
- MoE 场景的 mismatch 量化：讲者只给了机制猜想（动态 routing + 特殊 kernel），其团队实验尚在进行；GSPO 的序列级比率与 TIS 的 token 级截断在 MoE 上到底谁更对症，需要等数据。
- 字幕中小模型（0.5B/1.5B）结论的迁移边界：讲者自己提示 Qwen 模型在 GSM8K/MATH 上可能存在 overfit/contamination，掩盖了 mismatch 的影响；换未污染基准后结论是否保持，待验证。

## 原文金句（1-2句）

> 「当我们使用混合的训练引擎以及推理引擎去做 RL 训练的时候，它可能会带来，就是在更高效的同时，可能会带来一些潜在的 off-policy 的问题，即便是他们使用的是相同的 model 参数，也会有这个问题。」——讲者总结（字幕 42:47–43:08）

> 「有点像老师教一个学生，但这个学生已经学会了，但这个老师以为这个学生没有学会，所以他就不停地教这个学生，可能最后学学就学坏掉了。」——讲者解释 recompute 路径下熵塌缩的过惩罚机制（字幕 36:57–37:10）
