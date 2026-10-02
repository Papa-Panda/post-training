# EP133 — VeRL-Omni：基于 VeRL 及 vLLM-Omni 构建的面向多模态生成模型的开源 RL 后训练框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP133-verl-omni.html

> 「我们是希望能够去做一个兼顾易用性和性能的一个多模态统一，生成理解生成的这块强化学习的一个训练框架。」——讲者开场定位（字幕 [00:37]）

## 元信息

- 期号：133
- 标题：VeRL-Omni：基于 VeRL 及 vLLM-Omni 构建的面向多模态生成模型的开源 RL 后训练框架
- BV：BV1BAj96ZE9f
- 时长：01:11:20（字幕末条 [71:16]；讲授约 53 分钟 + Q&A 约 18 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：黄永祥（字幕 [00:07] 自述「我是黄永祥」；字幕 [00:08] 识别其所属为「华为莱布尼茨研究所」——字幕识别结果，待核；官网预告页记其为香港科技大学博士、华为莱布尼茨研究所 Research Scientist、vLLM-Omni 和 VeRL-Omni 开源社区核心贡献者）
- 相关代码：https://github.com/verl-project/verl-omni （官网预告页给出；字幕 [52:38] 提到代码仓入口，未口播 URL）
- 官网预告：https://qingkeai.online/blog/verl-omni
- B站链接：https://www.bilibili.com/video/BV1BAj96ZE9f/
- 字幕原文存档：本地 `transcripts/EP133.txt`（1102 条，带时间戳）

> 📝 提炼方式说明：本纪要为字幕实录版——基于 B站 AI 字幕原文（已存档于 `transcripts/EP133.txt`）清洗提炼，所有数字均标注字幕时间戳出处，找不到字幕出处的不写。AI 字幕识别误差在存档中原样保留，纪要正文使用规范专名：VeRL-Omni 在字幕中作「VERALONI / VERONI」，VeRL 作「VERAL」，vLLM-Omni 作「VRONI / VIONI / VLOV2ONI」，Diffusers 作「DIFFUSERS / DIFFUSES」，FSDP 作「FSDP / FST p two」等，均以官网预告页与公开资料为准。Qwen-Image 字幕作「轻微 image / 切问 image / 请问 image」，Bagel 字幕作「贝狗 / BO / BGO / BGO」，Qwen3-Omni 字幕作「切文 3OMMY / 千分 3OMMY / 清瘟 3OMMY」。

> 关联：本期是 RL infra 线的「多模态生成 RL」一讲，与 EP049《verl 源码解读与 HybridFlow 编程范式讲解》同源——VeRL-Omni 直接基于 VeRL（字幕作 VERAL）的 single-controller / hybrid-flow 框架扩展，对照 EP049 可看清从文本 RL 到扩散生成 RL 的增量改动面。

## 一句话总结

VeRL-Omni 在 VeRL 分布式训练框架与 vLLM-Omni 多模态推理引擎之上，为扩散生成与统一理解-生成模型补齐一套 RL 后训练系统：它把三处与 LLM RL 本质不同的负载——多步去噪的 rollout、无法用规则打分的模型化 reward、去噪步维度的 log-prob——分别做成可插拔的 rollout 引擎、异步 multi-reward serving 与 rollout log-prob 校正，相对 Diffusers 基线实现约 20% 的端到端吞吐提升（字幕 [03:57]），并在 Qwen-Image 上跑通 FlowGRPO、DiffusionNFT、DPO 三条算法 recipe。

## 核心

### 引入：多模态生成 RL 为什么需要专门的框架

讲者把当前多模态模型分成三类（字幕 [07:18] 起）：以理解为主的全模态模型（如 Qwen3-Omni，AR + DiT 架构，模态 encoder 出 embedding 后由 AR 做理解与文本生成、talk 部分生成 audio token）、扩散生成器（condition encoder + DiT 多步去噪 + VAE 解码，典型如 Qwen-Image、Wan2.2）、统一理解-生成模型（如 Bagel，双 transformer：一个自回归文本生成、一个扩散生成，understanding expert 的 KV 直接作为 generation 的 condition，涉及 KV 迁移）。

RL 链路本身与 LLM RL 同构——prompt 进、group 生成、reward 打分、算相对优势、actor 更新、权重同步回 rollout（字幕 [10:49] 起）——但每个环节的负载特征都变了。讲者明确判断：链路可复用 VeRL 现成框架（字幕 [19:26]「整个链路跟 LLN 是相似的」），所以 VeRL-Omni 的做法是吃下 VeRL 在分布式编排、SPMD 执行、Ray placement 与权重同步上的存量优势，聚焦优化 diffusion RL 特有的 rollout、reward 等模块（字幕 [20:56]）。

### 系统架构：一个 single controller 编排，三块可插拔引擎

VeRL-Omni 的整体架构（字幕 [00:50] 起）：底层是 VeRL 分布式训练框架 + vLLM-Omni 推理引擎组成的训推一体系统；中间 actor engine 基于 VeRL，支持 Diffusers 的 FSDP 训练引擎与 VeOmni（字幕中亦作 VONI）多模态训练引擎；训练 loop 按 GRPO 场景组织——多样本 rollout 生成 → reward engine 打分 → trajectory 与 reward 送 actor engine 做策略更新（字幕 [01:28]）。上层三类 trainer：FlowGRPO 及变种（FlowDPO、DiffusionNFT）、统一模型的 PPO trainer、偏好对齐 DPO trainer（字幕 [01:48]）。

三处结构性设计被讲者反复强调：

1. **模块全解耦、可插拔**。trainer、rollout、reward module 彼此解耦，actor engine 可在 Diffusers FSDP、vLLM-Omni 与 Megatron 之间切换（字幕 [02:49]）；训练后端通过 hydra config 把前缀从 FSDP config 换成 VeOmni config 即可切换，并行 mesh（DP/SP/TP）与 rollout 独立、互不干扰（字幕 [24:26]）。
2. **Reward 单独成 serving**。扩散生成的图像质量无法用物理规则或数学公式描述，必须用人类偏好模型或理解模型打分（如 Qwen2.5-VL 做图文匹配、OCR 模型做文本渲染打分，字幕 [11:28]、[15:55]），负载远高于 LLM RL 的 reward 计算——这是把 reward 从函数调用升级为独立 serving 引擎的根本原因，也是它与原生 VeRL 最本质的变化（字幕 [65:01]「最本质的变化就是 reward 这一块」）。
3. **权重同步走句柄而非文件**。trainer 更新好的权重通过 ZMQ IPC 同步到 rollout 侧，只传 tensor 分块的句柄、不传权重本体（字幕 [23:10]），并针对 vLLM-Omni engine 做了 checkpoint engine 适配（含 LoRA 权重同步），避开原 VeRL 文件方式的低速同步（字幕 [65:35]）。

顶层编排仍是 VeRL 的 single controller 负责大 training loop（生成调度、优势/reward 评估、梯度更新、权重回同步），每个设备上拉起 rollout engine 与 trainer engine 做训推共卡（字幕 [21:22]、[34:49]）。为什么不直接在 VeRL 主仓里做而要独立子仓，Q&A 给了明确答复：多模态生成迭代快、模型与算法多、负载与 LLM 明显不同，独立成仓避免对主仓代码的过多耦合（字幕 [53:17]）。

### 扩散 RL 与 LLM RL 的五点差异：轨迹、log-prob 与 reward 全部换维度

这是全场最有方法论价值的一节（字幕 [13:12] 起「扩散模型的 RL 跟传统的 LLM 的 RL 到底有什么区别」）：

- **轨迹的原子单位不同**。LLM RL 里一条 trajectory 是一个个 token 的自回归预测；扩散 RL 里一个转移是「一步去噪」——从初始 latent ， $Z_T$ ， 经过多步去噪得到干净 latent ， $Z_0$ ， ，整条去噪链构成一条 trajectory（字幕 [13:59]、[14:39]）。讲者特别指出：flow matching 的 ODE 采样每步是确定的，不满足 GRPO 需要的输出多样性，FlowGRPO 的做法是改用随机微分方程（SDE）、在每步转移中注入高斯随机性，使转移概率可算、相对优势可评（字幕 [18:08]）。
- **输出长度固定**。扩散生成在输入时即指定分辨率/帧数，避免了自回归 RL 常见的长尾序列 bubble 问题（字幕 [15:02]）；文本 rollout 是不定长 + KV cache 加速，扩散 rollout 无法用 KV cache，相邻去噪步的 latent 相似性只能靠 cache 类近似跳步（字幕 [57:42]）。
- **推理引擎不同**。文本侧是 vLLM/SGLang 一类引擎；扩散侧目前主要是 vLLM-Omni，相对较新（字幕 [15:28]）。
- **算法对应关系**。GRPO 对应 FlowGRPO，DPO 对应 DiffusionDPO（字幕 [15:40]）。
- **log-prob 的计算维度不同**。LLM 在 token 维度算 policy gradient loss，拿到完整 token 轨迹即可算每个 token 的 log-prob；扩散 RL 必须在去噪 step 维度算 log-prob，且步数由 ODE schedule 决定——这是后面 rollout correction 优化存在的根源（字幕 [17:03]、[17:34]）。

### 四项性能优化：把收益拆到 rollout、log-prob、reward、算子四层

讲者先给定调：性能优化的前提是训练稳定性与可收敛性（字幕 [05:28]「不只是做这个性能的优化，性能优化的前提还是保证整体训练的一个稳定性」），随后把端到端收益（相对 Diffusers 共卡训推基线整体吞吐 +20% ，字幕 [03:57]；rollout 侧联合优化同样 +20% 左右，字幕 [28:37]）拆成四个来源：

1. **Diffusion-aware routing + prompt embedding caching**（字幕 [25:43]）。新 request 先判断是否与已见 prompt 相同：相同则路由到同一张卡复用已有推理结果，新 request 按 prompt id 路由；目的是让同一 prompt 的多个生成（GRPO 同 prompt 多采样是常态）落在同一 rollout engine 上，batch 内长度一致、消除 padding 计算，并省去 prompt embedding 的重算。
2. **Stepwise continuous batching**（字幕 [27:43]）。类比 LLM 的 continuous batching，但在去噪 step 维度做动态 batching，把不同 request 按步拼批，提升 rollout 吞吐；与 routing 联合带来上述约 20% 的 rollout 侧收益。
3. **Rollout log-prob correction**（字幕 [28:45]）。标准流程里有三路 log-prob：rollout 时算出的转移对数概率、trainer 侧因计算图不同（Diffusers vs vLLM-Omni）必须重算的 old log-prob、策略更新时算的 new log-prob。VeRL-Omni 直接复用 rollout engine 的 log-prob 作为 old policy 的 log-prob，并用 importance sampling 做校准——当 trainer 当前 log-prob 与 rollout log-prob 漂移过大时，对 PPO ratio 的 clip 做更严格限制、直接 reject 掉漂移过大的样本。收益是跳过中间一路重计算，节省大概 50% 的单步时间（字幕 [31:53]「能够节省大概 50% 的一个单步时间」）。
4. **异步 multi-reward serving**（字幕 [31:59]、[32:49]）。支持的 reward 来源包括 VLM 模型（经 vLLM 部署）、偏好模型（UnifiedReward、HPSv3 等小模型）与外部 HTTP 打分（PaddleOCR、GenEval 类模型），多 reward 可 aggregation 联合打分。流水线层面：一个 batch 被切成多个 micro-batch 后，某个 micro-batch 的 rollout 一完成就立刻送 reward engine 打分，与后续 micro-batch 的生成 overlap，省掉整批等待；切分粒度可按实际情况调，最极致可到单 request 级（字幕 [33:57]）。Q&A 补充了部署形态的实证：单独部署一张卡做异步 reward 计算，单卡吞吐高于把 reward 混在训练卡上（字幕 [62:39]）。

算子层面补充支持 FA3 与针对昇腾的 attention 算子优化，硬件上除英伟达 GPU 外已对昇腾做适配（字幕 [06:03]、[04:02]）。分布式侧：64 卡上 VeOmni 训练引擎相对 Diffusers FSDP 约有接近 5% 的提升、趋势吻合（字幕 [34:50]）。

### 模型与算法覆盖：三类模型、五条 recipe

模型支持三类（字幕 [04:12]）：扩散生成模型（Qwen-Image、Wan2.2、Stable Diffusion 系列）；统一理解-生成模型（Bagel 已支持，Janus 类 3B 模型适配中）；Omni 模型（Qwen3-Omni 已支持）。算法侧除主流 FlowGRPO 外，跟进其稳定性/收敛性变种（字幕作 MIGRPO、GRPO Guard），并支持离线与在线 DPO（字幕 [05:14]）。项目开源不到两个月（字幕 [06:31]），当前对齐 VeRL 0.8.0 版本，继承其 fully-async 等能力（字幕 [67:42]）。

五个端到端 recipe 是下半场的主体：

- **Qwen-Image × FlowGRPO**（字幕 [36:36]）。目标是提升文本渲染能力：prompt → group 生图 → Qwen2.5-VL 对文本渲染质量打分（亦可用 PaddleOCR 抽字后与 ground truth 对比，字幕 [40:23]）→ 算优势 → clipped FlowGRPO loss 更新。W&B 曲线显示约 30 步为初始状态、训到 120 步左右 reward 基本收敛、文本渲染明显变强，之后因过拟合略有下降（字幕 [38:42]、[39:22]）。
- **DiffusionNFT**（字幕 [39:51]）。与 DPO 相似的思路：新旧两个权重（即字幕所说的两个 DiT）对同一图像去噪、各自预测 velocity，加权求和构造更强/更弱的参照策略，再以 reward 加权预测误差构造 loss——reward 越高的样本权重越大。收敛显著快于 FlowGRPO：约 40 步 reward 达 0.95 ，而 FlowGRPO 在 60 步时还没到 0.9 （字幕 [43:26]）。Q&A 亦确认 DiffusionNFT 收敛更快、训练速度接近或略快（字幕 [68:46]）。
- **在线 DPO**（字幕 [44:12]）。一个 prompt 生成 K 张图、reward 在线打分后取最好与最差组成 pair 算 DPO loss；近 60 步训练后文本渲染能力明显提升（字幕 [45:03]）。
- **Qwen3-Omni × GSPO**（字幕 [45:12]）。目前只训 think 部分的文本理解与推理，用数学数据集训练；因基座本身能力较强，reward 提升有限，后续将加入多模态输入场景验证。Q&A 中讲者坦言该模型尚未做深入性能优化（Q3 规划），其 think 部分瓶颈与 LLM 相似，主要多出 audio/video encoder（字幕 [54:36]）。
- **Bagel 适配**（字幕 [46:14]）。Bagel 在 Diffusers 上不受支持，此 recipe 意在证明框架对非 Diffusers 生成模型的快速适配：需改四处——数据输入格式（Bagel 原生 token 格式，不同于 Qwen-Image 的 chat template）、training adaptor（基于 diffusion model base 写 BagelForTraining 类，返回中间 latent 与 timestep）、rollout 侧基于 vLLM-Omni 的推理生成、把 ODE schedule 接入 rollout 以拿到多步去噪 trajectory；RL training loop 与 FlowGRPO 相同。

### 路线图与 Q&A 中的未决问题

路线图分四块（字幕 [48:06]）：（1）训练稳定性——先在 VeRL 上做到 fully deterministic 使两次训练 loss bitwise 对齐，目前已在 VeRL 支持（讲者称「这一周」刚合入，本周口径待核），使 VLM reward serving 至少具备完全确定性，Q3 再把 rollout 与 actor 侧的确定性做齐，最终目标是 FlowGRPO / Qwen3-Omni 训练完全可复现；同时进一步提升 trainer 与 rollout 两侧的数值一致性（涉及计算图与 kernel 差异）。（2）模型与算法——补 Qwen-Image-Edit、LTX-2.3（字幕识别，待核）等模型，统一模型继续支持字节系与混元系新模型，Omni 系列补 DPO 后续链路与 audio/video 混合输入的端到端理解训练。（3）效率——大规模部署验证、AR + DiT 架构的全异步探索、与 vLLM-Omni 社区联合优化 rollout 吞吐。（4）可用性——CI 收敛监控标准化（目前 SD3.5 小模型约 1 小时可验证一遍生图 RL 效果，字幕 [51:44]）、NPU 等硬件适配、教程加强。

Q&A 里几个边界值得单独记下：PD 分离目前不支持（只有 AR 类模型才需要，当前主要支持 Qwen3-Omni 的 thinker，字幕 [64:36]）；弹性扩容未验证（架构上可经 VeRL + Ray 支持，字幕 [67:12]）；异构/fully-disaggregated RL 架构上支持但需算法适配——扩散 RL 目前主流仍是 on-policy 算法，off-policy 验证待与算法联合推进（字幕 [68:06]）；diffusion RL 的 reward hacking 被 FlowGRPO 作者亦提及，缓解手段是 early stopping 与多 reward 混合训练（字幕 [55:48]）；多模态联合训练的 credit assignment（视频 vs 图像如何分配资源）尚无实证，讲者只给了按负载给视频生成多分卡的朴素构思（字幕 [59:08]、[60:19]）；世界模型（world action model）可类比视频生成链路接入（条件多加动作方向/位移），团队已有同事在做世界模型推理优化（字幕 [63:09]）。被问 reward model 与 judge model 哪条路线更重要时，讲者明确回避框架层面的推荐（字幕 [56:30]）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源（字幕时间戳） |
|---|---|---|---|
| 端到端训练吞吐 | Diffusers 共卡训推方式 | 整体吞吐 +20% | [03:57]「整体吞吐是有20%的一个提升」 |
| rollout 侧吞吐 | 原 baseline（routing + embedding caching + stepwise continuous batching 联合） | 提升到 20% 左右 | [28:37] |
| 单步时间 | 含 old log-prob 重计算 | rollout correction 后节省大概 50% | [31:53] |
| 64 卡训练引擎对比 | Diffusers FSDP | VeOmni 约接近 5% 提升 | [34:57] |
| Qwen-Image FlowGRPO 收敛步数 | 30 步为初始状态展示 | 约 120 步 reward 基本收敛 | [38:45]、[39:34] |
| DiffusionNFT 收敛 | FlowGRPO 60 步 reward 未到 0.9 | 约 40 步达 0.95 | [43:31]、[43:38] |
| 在线 DPO 效果步数 | — | 近 60 步后文本渲染明显提升 | [45:03] |
| 小模型 CI 验证成本 | SD3.5 | 约 1 小时跑通生图 RL 验证 | [51:44] |
| 开源时长 | — | 不到两个月 | [06:31] |
| 对齐的 VeRL 版本 | — | 0.8.0 | [67:42] |
| 视频总时长 | — | 末条字幕 [71:16]（from 4276.75 秒） | transcripts/EP133.txt |

## 可迁移

- **Reward 必须按负载特征决定形态**：当 reward 从「规则/轻量打分」变成「多模型推理」时，把它从训练 loop 里的函数调用升级为独立 serving + micro-batch 级异步 overlap，是通用的 infra 判断——同样的推理适用于 LLM RL 里 reward model 变重、或 verifier 需要外部工具调用的场景。判定信号很直接：reward 计算能否被规则/公式表达、单样本成本是否与生成同量级。
- **训推 log-prob 不一致时，先校准再重算**：rollout correction 的本质是「复用 rollout 侧 log-prob + importance sampling 漂移检测 + 超限样本 reject」，以一路重计算的删除换约 50% 单步时间。这个模式不限于扩散 RL——任何 rollout/training 双引擎数值不一致的 RL 系统（如推理引擎与训练引擎 kernel 不同）都可先评估漂移分布，再决定是否值得付重算成本。
- **同 prompt 多采样场景优先做 prompt 级路由 + embedding 缓存**：GRPO 系算法天然同 prompt 多采样，把同 prompt 请求钉在同一 engine 上同时消掉 padding 与 embedding 重算，是零算法改动的吞吐收益；对文本 GRPO 的 prompt 级亲和调度同样成立。

## 疑问 / 下一步

- Rollout correction 中「漂移过大即 reject」的阈值如何定：字幕只给了机制描述（对 clip 做更严格限制、丢弃偏移过大的样本），未给阈值取值、reject 比例与对收敛性的消融；这是复现时最先要补的实验。
- 50% 单步时间节省与 20% 端到端提升之间的口径关系：log-prob 重计算占单步约一半，但端到端只兑现 20% ，中间差额花在 rollout/reward/同步哪一段，字幕未给分项 profiling；64 卡下 VeOmni 仅 +5% 也提示训练引擎并非当前主瓶颈，瓶颈占比问题讲者在 Q&A 中亦未正面给出（字幕 [54:36] 称 Qwen3-Omni 尚未深入优化）。
- Bitwise 可复现的时间点：讲者称 VeRL 侧 fully deterministic「这一周」刚支持、Q3 补齐 rollout/actor 确定性——需回 VeRL 主仓核对该特性的实际合入版本，再判断 VeRL-Omni 的复现承诺何时可验。
- DiffusionNFT 中新旧权重加权系数的设定（字幕 [41:40] 提到 $1 + \beta$ 一类加权，未给 $\beta$ 取值）与 reward 加权方式的完整公式，字幕为口述示意，需对论文/代码原文核对后再引用。

## 原文金句（1-2句）

> 「（扩散 RL 与 LLM RL）整个链路跟 LLN 是相似的，只是各部分的计算负载和特征跟传统的 LLNRL 不一样，所以其实我们是可以复用现在 VeRL 的一个整个框架去做这个事情。」——讲者对「为什么基于 VeRL 构建」的正面回答（字幕 [19:17]，按字幕原话转写，LLN 为字幕对 LLM 的识别）

> 「我们其实不只是做这个性能的优化，我们性能优化的前提还是保证整体训练的一个稳定性、可收敛性。」——讲者给全部性能数字定的前提（字幕 [05:28]）

> 「（与 VeRL 主仓比）最本质的变化就是 reward 这一块……Verb 本身它只是只支持这个 reward function 的一个计算，然后我们其实新增了一个 reward engine 去做这个 reward 的 serving。」——Q&A 中对 VeRL-Omni 与 VeRL 差异的一句话概括（字幕 [65:10]，Verb 为字幕对 VeRL 的识别）
