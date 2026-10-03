# EP105 — InternVLA-A1，虚实贯通：理解、生成、执行一体化的具身操作大模型

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP105-internvla-a1.html

> **主旨引文**（论文 arXiv:2601.02456，Conclusion）：「we compromised the fidelity of image prediction, limiting the granularity of generated future frames」—— 本期架构的核心取舍：为实时性牺牲未来帧的像素级保真度。

## 元信息

- 期号：EP105（官网期号 105；青稞Talk 第 105 期）
- 标题：InternVLA-A1，虚实贯通：理解、生成、执行一体化的具身操作大模型
- 讲者：曾嘉（上海人工智能实验室具身智能中心青年科学家；InternVLA-A1 项目负责人及核心贡献者、InternData-A1 通讯作者）
- 讲者来源：[节目页 qingkeai.online/blog/InternVLA-A1](https://qingkeai.online/blog/InternVLA-A1)（嘉宾介绍为公开署名，未查到讲者在视频中的自述，以节目页为准）
- 观看/提炼日期：2026-10-02
- 直播时间：2026-01-27 20:00–21:00（节目页）
- BV：公开台账（`episode-index.csv`）记录「official-only（官网有预告，B站合集无对应视频）」；合集内 BV1zF67BmEHD 实际为另一场 SDAR（纯 RL 训练的扩散语言模型），主题与本期完全不同，不归入本期
- 相关论文：[InternVLA-A1: Unifying Understanding, Generation, and Action for Robotic Manipulation](https://arxiv.org/abs/2601.02456)（arXiv:2601.02456，v1 2026-01-05 提交，v2 2026-02-13；本文数字一律采 v2 正文，未特别说明者除外）
- 相关代码：[github.com/InternRobotics/InternVLA-A1](https://github.com/InternRobotics/InternVLA-A1)
- 提炼方式：**无字幕，基于对应论文 arXiv:2601.02456（v2）还原，非逐字稿。** 所有数字、结论、局限均直接标注论文出处，未确证者见文末清单。

## 一句话总结

把理解、视觉预见、动作执行三件事塞进一个 Mixture-of-Transformers，用「语义上下文 + 轻量未来帧预测」共同条件化 flow matching 动作头，在 3B 参数量下真机静态任务平均成功率 75.1%（vs π₀.₅ 70.7%），动态任务优势更大（Express Sorting 80.0% vs 53.3%），数据与架构两头都在治当前 VLA 的两个老毛病：不懂物理动态、真机数据喂不饱。

## 核心

1. **背景/问题**：主流 VLA 建立在多模态大模型（MLLM）上，语义理解强，但文本 token 并不擅长描述物理规律，模型只能做反应式「看→动」映射，在传送带分拣这类高动态场景里跟不上；同时真机示范数据昂贵且难扩展，纯仿真又有 sim-to-real gap。视频预测型世界模型（VPP、Genie Envisioner 等）虽能给出未来观测，但语义锚定弱、对预测误差脆弱，且推理频率撑不住实时控制（论文给出的参照：SANA-Sprint 在 RTX 4090 上单次生成约 0.16 s、至多约 6 Hz；DreamZero 经过 38 倍工程加速后在 GB200 上也只有 7 Hz）。
2. **方法/设计**：三专家 MoT（Mixture-of-Transformers），统一 masked self-attention，信息流严格按「理解 → 生成 → 执行」单向流动，后面块的 token 可以 attend 前面所有块，反之不行。理解专家直接复用现成的 MLLM（InternVL3 或 Qwen3-VL），产出共享上下文 $h_{\mathrm{und}}$ ；生成专家只做「够用」的未来预测：输入取三个视角（头 + 双腕）的 $t-15$ 与 $t$ 两帧共 6 张 $256\times 256$ 图像，经 COSMOS CI8×8 VAE 编码后用 $8\times 8$ 卷积压缩（每张图 1024 token → 16 token，6 张图共 96 token），沿时间维平均池化到 48 token 再反卷积上采样，单次前向并行解码出 $t+15$ 的未来帧 latent，不用自回归逐 token 生成；动作专家在这个「语义 + 预见」的条件上用 flow matching 预测动作 chunk ， $\hat{a}_{t:t+k}$ 。训练两阶段（大规模预训练 + 任务后训练）共用同一组目标，总损失是生成损失加权 0.01 再加动作损失， $\mathcal{L}_{\mathrm{total}}=\lambda\cdot\mathcal{L}_{\mathrm{gen}}+\mathcal{L}_{\mathrm{act}}$ ，其中未来帧预测步幅 $m=15$ ，动作插值时刻 $\tau\sim\mathrm{Beta}(1.5,1.0)$ 。
3. **实验/实战**：两个实例：2B 模型（InternVL3-1B 作理解专家，生成/动作专家派生自 Qwen2.5）与 3B 模型（Qwen3-VL-2B 作理解专家，生成/动作专家派生自 Qwen3），两者在 RTX 4090（torch.compile）上推理频率都约 13 Hz（2B 的 InternVL3 输入分辨率 $448\times 448$ 高于 3B 的 $224\times 224$ ，所以并未更快）。预训练数据 692M+ 帧，配方带采样权重：InternData-A1 纯合成仿真 396M 帧（权重 0.64）、AgiBot-World（Beta）真机 206M 帧（0.18）、RoboTwin 17M 帧（0.08）、EgoDex 人类第一视角视频 68M 帧（0.08、不使用其动作标签）、RoboMind 5M 帧（0.02）；工程上用 Load-balanced Parallel Training（LPT）把异构数据集贪心分配到 worker，解决 LeRobot 式全量实例化的显存与 I/O 压力。预训练 700K steps（batch 512，常数学习率 $5\times 10^{-5}$ ），后训练 60K steps（batch 128，学习率 $5\times 10^{-5}\rightarrow 5\times 10^{-6}$ ）。真机评测覆盖 Agibot Genie-1、ARX Lift-2、ARX AC One 三个本体，每任务 30 次 rollout：3B 模型静态任务平均 75.1%（π₀.₅ 70.7%、π₀ 60.6%），2B 模型 64.7%；动态任务两项合计 86.7%，其中 Express Sorting 80.0%（vs π₀.₅ 53.3%），In-motion Ingredient Picking 93.3%（vs 66.7%）。RoboTwin 2.0 仿真基准上 3B 模型 Easy/Hard 均领先 π₀.₅ 2.6 个百分点（该基准每任务用 50/500 条示范、共 27,500 条微调、每次评测跑 100 次取平均）。消融：去掉预训练，平均成功率 77.0% → 25.4%（论文称 pre-training acts as a crucial inductive prior）；去掉生成专家，77.0% → 57.6%，12 个任务中 11 个变差，动态任务跌幅最大；纯仿真预训练在 RoboTwin 上已经很强（88.3%/88.5%），但真机 Place Flower / Sort Parts 只有 53.3%/33.3%，加上真机与人类视频后提升到 60.0%/53.3%——异构混合就是在真机侧补这一截。
4. **结论/观点**（论文口径，非讲者原话）：VLA 的泛化瓶颈同时来自架构（语义与物理动态脱节）与数据（真机示范难扩展），两者要一起解；生成专家可以刻意「糊」——牺牲高频像素细节换实时性，预见只需要给出对动作有指导作用的粗粒度未来趋势；作者自陈两个主要局限：理解专家未与大规模多模态 VQA 数据联合训练，复杂指令跟随与语义推理仍弱；为实时性妥协了未来帧预测保真度。

## 关键数字

| 指标 | 数字 | 来源 |
|---|---|---|
| 模型规模 | 2B 实例总参数约 1.8B / 3B 实例约 3.2B | 论文 arXiv:2601.02456 Table 1 |
| 推理频率 | 两实例均约 13 Hz（RTX 4090，torch.compile） | 论文 §3.4、Table 1 |
| 生成专家关键压缩 | 6 张图、每图 1024 token 压到 16 token，共 96 token；池化 48 token 后并行解码 | 论文 §3.2 |
| 生成未来步幅 | 预测 $t+15$ 未来帧（ $m=15$ ） | 论文 §3.4 |
| 预训练数据总量 | 692M+ 帧（692M 为五项表内之和，论文正文写作 over 692M frames） | 论文 Abstract、Table 3 |
| 数据配方 (帧数 / 采样权重) | InternData-A1 396M / 0.64；AgiBot-World(Beta) 206M / 0.18；RoboTwin 17M / 0.08；EgoDex 68M / 0.08；RoboMind 5M / 0.02 | 论文 Table 3 |
| InternData-A1 规模 | >63 万条轨迹、7,433 小时、4 本体、18 技能、70 任务、227 场景 | 论文 §2、§4.2 |
| 训练步数 | 预训练 700K steps（batch 512）；后训练 60K steps（batch 128） | 论文 Table 2 |
| 真机静态任务平均成功率 | InternVLA-A1(3B) 75.1%、(2B) 64.7%；π₀.₅ 70.7%、π₀ 60.6% | 论文 §5.2、Table 4 |
| 动态任务 (Express Sorting / Ingredient Picking) | 3B：80.0% / 93.3%；π₀.₅：53.3% / 66.7%；两项合计 3B 86.7% | 论文 §5.3 |
| RoboTwin 2.0 (Easy/Hard) | 3B 领先 π₀.₅ 各 2.6 pcts；领先 π₀ 9.4%/10.1% | 论文 §5.4 |
| 消融：去掉生成专家 | 12 任务平均 77.0% → 57.6% | 论文 §5.5 |
| 消融：去掉预训练 | 平均 77.0% → 25.4% | 论文 §5.5 |
| 帧级压缩比 | 每图 1024 → 16 token（64 倍压缩） | 论文 §3.2 |

注：arXiv:2601.02456 的 **v1（2026-01-05）与 v2（2026-02-13）数字不完全一致**（v1 摘要写预训练数据 533M+ 帧、动态任务提升 40%–73.3%；v2 写 692M+ 帧、相关真机任务按每项单独与 π₀.₅ 对比），上表均按 v2 正文录入；论文自身未标注两版差异，跨版比较时请留意。

## 可迁移（放到 RL 训练 / 数据构造 / 评测框架）

- **统一条件化 + flow matching 做多模态（文本/动作）联合分布**：动作多模态性不靠离散 token，而是对连续动作 chunk 做噪声→示范的路径学习，Bernoulli-free；可迁移到任何「连续控制/动作 expert 与语言模块联合微调」的场景——在 coding-data 的 RL 里，动作也可以是结构化编辑（edit program），用条件-连续速度场建模重构分布，或许比离散 token 化更抗多解。
- **粗粒度预见信号够好就不必像素级生成**：预见模块刻意压到每张图 16 个 token、单次并行解码，论文明确 argue 高频细节对动作决策不必要。迁移到后训练：任何「先预测未来再决策」的环节（agent 的计划/想象、rollout 预演），都可以先试低保真代理而非真生成，先确认信号量够再谈清晰度。
- **异构数据负载均衡并行训练 (LPT)**：把异构规模的数据集按规模从大到小贪心分派给 worker 使单 worker 装载大致均匀，避免大 worker 反复遍历小数据集造成的隐式重加权——这套工程细节在多源 mixing（合成轨迹 + 真机 + 人类视频混合 SFT/RL）时可直接照搬，并对控制隐式采样分布有任何多源训练通用。

## 疑问 / 下一步

- 真正想拿视频回放核实的是：曾嘉现场怎么拆 InternData-A1 合成管线（合成质量、灵巧操作标注质量、与真机数据的混配比例消融细节），以及在 B站无字幕视频条件下讲者对 RoboTwin 2.0 上 +2.6 pcts 这一相对单薄增益如何解读——更像全面小胜还是接近上限？这部分只能等录像/字幕可得后回填。
- 动态任务（传送带）为什么生成专家的边际收益远大于静态任务、未来帧预测的时间跨度 $m=15$ 如何调、并行单次前向预测的「动态预见」上限（比如是否只适用于短时平稳运动）仍缺细粒度的解释，论文只给了结果未给机理解剖。
- 未与大规模 VQA 数据联合训练的理解专家在长指令/组合指令上的上限未测，下一步若有 follow-up（如 InternVLA-A1.5，arXiv:2607.04988，已有公开线索指向该标题）可继续跟踪。

## 原文金句（论文原话，非讲者语录）

> "we compromised the fidelity of image prediction, limiting the granularity of generated future frames."
> （arXiv:2601.02456， §6 Limitations）

## 未确证项

1. **无字幕/未直接观看**：B站合集无本期对应视频字幕可提，本文全部内容基于论文 arXiv:2601.02456（v2）；讲者曾嘉来自节目页嘉宾介绍与论文团队署名交叉（论文作者表最后一位为 Jia Zeng，与「曾嘉」拼音一致，但未直接核实为同一人的官方对应关系）。
2. **v1/v2 数字冲突**：v1 摘要的 533M+ 帧、动态提升 40%–73.3% 与 v2 的 692M+ 帧、动态项逐个与 π₀.₅ 对比为不同口径，上表采 v2；直播当日（2026-01-27）可能引用的是更接近 v1 的数据，具体以录像为准。
3. **台账 BV 冲突**：交底称 BV1zF67BmEHD 为本期付费预览视频，但 `episode-index.csv` 显示该 BV 是 B105 SDAR（主题不同），且官网 EP105 登记为「official-only（B站合集无对应视频）」；本文按台账实际记录著录：本期不附 BV。
4. **关联 follow-up**：arXiv:2607.04988（InternVLA-A1.5）为 GitHub 仓库 `internvla-a-series` README 中 BibTeX 引用条目所见的后续报告标题，未在本文展开，是否对应本期后续未核实。
