# EP100 嘉年华 · RL 专题圆桌（字幕实录版）

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP100-rl-panel.html

## 元信息

- 期号：EP100 嘉年华专题（青稞Talk 第 100 期特辑分会场实录）
- 标题：RL 专题｜2025 “青稞” AI 嘉年华
- BV：BV1EUi4BwEGT
- 时长：约 89 分钟（字幕末条 5241.6 秒）
- 提炼日期：2026-10-02
- 提炼方式：B站 AI 字幕实录（2156 条，`transcripts/EP100-RL.txt`）。圆桌无逐字稿分工，发言人归属以字幕自我介绍 + EP100 官网嘉宾名单互证；字幕识别误差（如人名误转）按名单订正，无法确定归属处写「嘉宾」
- 主持人：吕兴泰（清华大学博士生，导师周伯文；ZEDA 作者）
- 嘉宾（官网名单 + 字幕自述互证）：崔淦渠（上海人工智能实验室青年科学家，PRIME / 隐式过程奖励）、胡健（OpenRLHF 发起者，现 NVIDIA）、郑楚杰（通义千问研究员，GSPO 作者）、李英儒（CUHK 博士，TRPO→SAPO、训推不一致系列作者）
- 活动综述见 [EP100-ai-carnival.md](EP100-ai-carnival.md)

## 一句话总结

四位 RL 一线研究者（PRIME、OpenRLHF、GSPO、SAPO 作者们）的共识出奇一致：2025 年 RL 真正的进展不在算法公式，而在 infra、数据与稳定性工程；好算法今天长得像「一套简单可解释的 recipe」，2026 年的主战场是 agentic 长程任务、全异步训练与 environment scaling。

## 核心（按议题）

### 议题一：2025 年最重要的技术进展——不是算法，是 infra 与稳定性

- 崔淦渠：这一年最有影响力的 paper/blog 大多不是关于算法的。训推 mismatch、MoE 训练稳定性、更难更新的数据，收益都高于纯算法改动。原因也现实：大厂与学术界的实验尺度、任务难度完全不在一个水平，很难有一个 universal 的算法结论。
- 胡健：算法本体还在用 2014 年的 PPO 乃至更早的 policy gradient；真正的突破在「如何让大模型在现代 GPU 上高效稳定地训练」——truncated importance sampling、MoE 稳定化这些想法都来自对分布式系统的观察，再反哺算法。未来 RL 必然是算法与 infra 联合优化，不存在纯系统工程师或纯算法工程师。
- 郑楚杰：今年更像一场「文艺复兴」——把经典 RL 理论适配到大模型特有场景：sequence-level 信号、自回归生成、MoE 与 MTP/投机采样带来的新约束。目前共识主要在 MoE 稳定训练上形成，投机采样等还要等。
- 李英儒：上半年很多工作可能要 revisit——下半年大家把 training infrastructure 与实现从头审了一遍，发现不少与理论不匹配之处。他特别点出被忽略的 exploration 问题：现在的大模型 RL 几乎没有真正的 exploration，只是在利用基座先验。

### 议题二：好算法长什么样——从「公式」到「recipe」

- 崔淦渠（PRIME 的出发点）：只用 outcome reward 会有 credit assignment 问题；受 DPO 作者工作启发，用 implicit 方式训 reward model 自动做 credit assignment，本质是给 PPO 的 value model 找一种更自然的建模方式（仍是语言模型、不用扔掉 embedding 接 linear head）。预判：未来 feedback 变丰富后，value-based / actor-critic 会重新被重视；PRIME 与 on-policy distillation 在形式上同为最小化 policy 与 teacher/reward 之间的 KL ，是一脉的。
- 胡健（REINFORCE++ 的出发点）：回归经典——把 PPO 去掉 critic、保留 clip 与 advantage norm 等久经推敲的细节，实验上确实稳。他判断现在的「算法」其实是一组 recipe 的组合（truncated IS、on-policy learning、各种 norm），再配一个好 infra 实现。
- 郑楚杰（GSPO 的出发点）：训 Qwen3 时 GRPO 后期总有稳定性问题，怀疑优化目标本身：sequence-level 的目标与 token-level 目标不好直接关联，于是回到第一性原理直接做 sequence-level 优化，顺带把 routing replay 去掉（235B 上开销很大），稳定且训得好，后迭代出 GSPO v2 修新训推不匹配类 bug。他对「好算法」的定义：不存在完美算法，只有好 recipe——每个 trick 简单、直觉、能验出正向收益，在各尺寸模型与任务上都能稳定训。
- 李英儒：9 月那篇 demystify training collapse 的 blog 之后，他的立场是必须回到第一性原理：采样明明来自 inference engine，很多框架却要在 recompute 处「假装」采样——这类实现与理论的错位才是训崩的源头。他强调理论的价值不是复杂化，而是最终能解释清楚每个 trick 在什么条件下 work、什么条件下不 work，并回归到 Schulman 2015 年 TRPO 的单调提升 insight 上。

### 议题三：模型结构要不要为 RL 适配——先有结构，RL 后适配

- 崔淦渠：架构更多与 infra/硬件相关，算法在后端；但 hybrid 架构上做 MTP 涉及 state 更改与回滚，复杂度远高于 full attention，这类新架构会给 RL 与 infra 带来更高挑战，需要和架构同学共同设计。
- 胡健：对 RL 友好的架构就是对推理友好的架构——RL 的算力瓶颈在 generation 一侧。Mamba/线性注意力、DeepSeek 的 MTP 都是推理友好型创新。若 diffusion LLM 起来，RL 的优化重心会从推理侧翻转到训练侧。
- 郑楚杰：「RL 训不长到底甩锅基模还是 RL」是基模团队的日常。实践中大家倾向「RL 是大模型的辅助」：先定架构（MoE、MTP 都为降本），再做 RL 适配；每一代模型有每一代的问题，两者不能分开看。
- 李英儒：MoE 难训的本质是离散选择架构对优化天然不友好——routing replay 等 trick 都是在补这个洞；这不是 RL 独有问题，预训练训 MoE 早遇到过。希望未来有兼顾稀疏激活与优化友好性的新稀疏架构。

### 议题四：理想的训练框架——易用性、可 debug、社区

- 崔淦渠（资深用户视角，用过 TRL → OpenRLHF → veRL → slime）：关键转折是 OpenRLHF 把推理引擎与训练引擎解耦再拼合，与 TRL 不是一个时代的产物。框架的生命力最终取决于社区与主力 contributor，技术选型决定上限、社区运营决定影响力。
- 胡健（OpenRLHF 作者自述）：出发点就是「没有框架让 RL 工程师用得爽」——当时选 Ray + vLLM + DeepSpeed 的组合（Megatron-Core 当时对 RL 工程师太难用）。今天 veRL 越来越臃肿、改调度流程非常难受，slime 会好一些。OpenRLHF 下一步：迁移 FSDP2、用 auto TP、参考新 agentic API 做一套干净轻量的新 API，定位开源研究用的中小模型模板（30B–100B），600B 级的定制优化该由商业公司自己承担。
- 郑楚杰：框架第一性是简单、data flow 清晰、每一步（rollout / 训练 / recompute / advantage）都能打出来存下来——debug 能力是正确性的保障，他用过「正确性不满足又找不到问题」的框架，深受其苦。其次是借力开源引擎（SGLang/vLLM + Megatron 等），中小团队自搓框架迭代太慢。
- 李英儒：异步、replay buffer、训推分离的根源是 RL 本质的 sample inefficiency——需要海量 trajectory。走向 agentic RL 后瓶颈会从 rollout 转移到环境交互本身（等编译、等反馈），coding agent 与 sandbox 交互几十小时已是现实，框架必须支持这种长程交互。

### 议题五：稳定性与指标——entropy 是结果不是原因，mismatch 是动态现象

- 崔淦渠：认识已更新——小模型时代怕 entropy collapse，大规模 MoE 上反而要防 entropy explode，它常伴随训练失败。实践中 routing replay、R3 等只能推迟问题；目前较有效的是冻结 router（不训 router，轻量但 work）+ 把 entropy 稳定在相对稳定值。
- 胡健：必看的仪表盘是 entropy、PPO KL 、训推 mismatch 的 KL 、与 reference 的距离——任一指标异常抖动，模型基本就被「砍了一刀」。他更认可 on-policy learning 路线：纯在线更新时 IS 恒为 1，expert 选择的剧烈变化问题被绕开；再用 TIS + mask 或 sequence-level filter 扔掉偏离过大的轨迹，比 routing replay 简单轻便。
- 郑楚杰：entropy 是学习的结果不是原因——熵异常升高往往对应 language mixing 这类不稳定表象。理想曲线应平稳甚至缓慢下降；让它别降太快的正道是放大探索空间（batch、query 集、生成长度）。mismatch 不可能完全消除，目标是控制在不影响训练的范围内并防突变。
- 李英儒：从优化角度，policy 在词表单纯形上优化，entropy 本就是结果变量（单纯形内部熵大、收敛到解熵降）；他们解决 mismatch 之后，entropy 曲线自然变稳，反过来证明它是指标而非杠杆。mismatch 最有意思的是它是动态现象：训到后面才突然变大，说明与优化过程耦合——精度噪声在训练中累积、把参数推向 mismatch 易放大的区域，可结合 variance reduction 让 policy 别走太远。

### 议题六：2026 展望

- 崔淦渠：更难的任务、更丰富的 feedback、更长程 + memory、自进化。RL 是 goal-oriented 方法而非 behavior cloning，2026 会走向更开放场景。
- 胡健：两条 scaling 主线——infra scaling（稳定大规模长训练）与数据/environment scaling（更多真实与合成环境）。算法本身大改动不会多，小创新在 process-based value model 等方向。
- 郑楚杰：工业界核心矛盾仍是计算效率 vs 算法性能；明年全异步 RL 会成主流，但一段回复由多个模型版本生成天然引入 off-policy 问题，配方的 off-policy 容忍度与异步框架效率还需要几个月打磨。先把计算效率打上去，才有资格谈 scaling。
- 李英儒：agentic 场景下瓶颈从 rollout 转向交互本身（异构算力、CPU 都会用上）；长程下 effective horizon 其实很短（关键分叉节点可能只占 20%），outcome reward 会让 credit assignment 与 variance 累积更难，trajectory-level advantage 够不够用他存疑——更细粒度 credit assignment 是长程场景的必修课。

## 关键数字（圆桌口径，均为字幕）

| 数字 | 出处语境 |
|---|---|
| PPO 是 2014 年的算法、policy gradient 已有几十年 | 胡健：算法本体进展有限 |
| 235B 模型上去掉 routing replay 后训练明显变快 | 郑楚杰谈 GSPO 的收益 |
| 开源框架定位 30B–100B 研究用模型，600B 级靠商业公司自研 | 胡健谈 OpenRLHF 路线 |
| coding 任务可与 sandbox 交互几十小时 | 李英儒谈长程交互对框架的要求 |
| 长程轨迹中关键分叉节点可能只占约 20% | 李英儒谈 effective horizon 与 credit assignment |

说明：本场为观点型圆桌，硬数字很少且多为经验口径，未做外部核对。

## 共识与分歧

共识：① 2025 的主线是 infra/数据/稳定性而非算法公式；② 「好算法」= 简单可解释、可验证收益的 recipe 组合；③ entropy 是监控指标不是控制杠杆，mismatch 要控范围防突变；④ 框架易用性与可 debug 性优先于极致性能，借力开源引擎；⑤ 2026 主战场是 agentic 长程、全异步、环境与数据 scaling。

分歧（温和）：MoE 稳定路线上，崔淦渠主推冻结 router + 稳 entropy，胡健主推纯 on-policy + TIS/mask 绕开 replay；对 value model 的回归预期，崔淦渠明确看好，李英儒则从统计角度指出 token-level value 的采样复杂度是根本限制（多项式级于序列长度），value 难训是共识、解法未定。

## 可迁移

- 训推 mismatch 先查实现层：采样来源与 logprob 计算路径是否与算法假设一致（李英儒的「recompute 假装采样」批评），再谈算法补丁。
- 仪表盘四件套：entropy、PPO KL 、训推 KL 、对 reference 的距离；异常抖动先停训查因，别指望单个 trick 救场。
- MoE RL 稳定先试两个轻量招：冻结 router、纯 on-policy 更新 + TIS/mask 过滤偏离轨迹；routing replay 是重武器，放最后。
- 选框架按「改调度流程的痛苦程度」评估，不只看吞吐榜；研究团队优先社区活跃、data flow 可全程落盘的框架。

## 疑问 / 下一步

- 郑楚杰预判「全异步成主流」与李英儒「瓶颈转向环境交互」叠加后，off-policy 容忍度的量化标尺在哪——会上没人给出可操作的判据。
- token-level value 的统计根本限制（李英儒）与 dense/process reward 的需求（崔淦渠）之间，有没有采样复杂度可接受的折中建模，值得跟踪后续工作。

## 原文金句

> 「（算法）更多的是一套 recipe，是一堆 trick 的组合……所谓的好的算法，应该是一套 recipe 的组合。」—— 郑楚杰

> 「为什么需要在 recompute 那个地方能产 rollout ？因为其实你真正产生采样是从 inference engine 来。」—— 李英儒（字幕大意，个别词识别有误）

> 「entropy 它是一种模型学习的结果，而不是原因。」—— 郑楚杰
