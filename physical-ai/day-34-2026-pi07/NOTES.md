# Day 34 — Physical Intelligence π0.7：组合泛化，做没教过的任务（最新进展 4/4）

## 元信息
- Title: π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities
- Authors / Org: Physical Intelligence（约 100 人，含 Sergey Levine、Chelsea Finn、Karol Hausman 等）
- 官方 Blog：[A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)（2026-04-16 发布；联系邮箱 research@physicalintelligence.company）
- 官方技术报告 PDF：https://www.pi.website/download/pi07.pdf （blog 页"Paper"链接）
- 独立报道：TechCrunch（Brian Heater，2026-04-16；"if the findings hold up" 保留）| The Decoder（数据污染质疑 + "不具备 reasoning" 视角）| TechDogs（发布日期/组合泛化口径确认）
- 抓取盲区：官方 X 帖子 URL、官方 YouTube π0.7 专属视频 URL 未抓取到（demo 视频嵌在 blog 页内）；Fig. 6/9 部分柱状图精确数值未提取，只确认文字结论；π0.7 **权重未开源**（openpi 只有 π0 / π0-FAST / π0.5）
- Date read: 2026-09-26
- Tags: [physical-ai, vla, pi07, compositional-generalization, steerable, metadata, flow-matching, cross-embodiment, instruction-following, world-model]
- Thread: physical-ai
- Folder: day-34-2026-pi07
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-34-2026-pi07
- 范围说明：π0.7 是路线 VLA 主线的最新进展，Day11 π₀ / Day14 π₀.₅ 的直接续集。以下所有数字均为**公司自报、公司自己跑的评测**，无独立复核——这是今天最硬的边界。官方承认"难以确切判定哪些任务真正 unseen"（Discussion 原文口径）。

## 一句话总结
π0.7（2026-04-16，约 5B 参数）：在 π₀.₅ 的 co-training + knowledge insulation 配方上加了一套**可转向（steerable）条件机制**——subtask 语言 + subgoal 图像 + episode metadata（速度/质量/失误标签）+ control mode，训练时 15% 丢弃 metadata、subgoal 只加 25% batch——让单个通用模型**追平甚至超过为每个任务单独 RL 训练的 specialist**（洗衣折叠/盒子组装吞吐量归一化 >1.0），在从未见过的双臂 UR5e 上折 T 恤达到专家人类 teleop 的水平（成功率 80% vs 80.6%），并通过语言 coaching 让模型学会训练中没出现过的任务组合（air fryer 煮红薯），coaching 数据再微调 high-level 策略即全自主执行、零 teleop 数据。官方自称"emergent capabilities"，The Decoder 反驳"不具备 reasoning，只是 hint"——把"组合泛化"读成"大规模 remix"是今天的读法选择。

## 和之前工作的关系

- **接了哪条线**：Day09 VLA 总览 → Day11 π₀（flow-matching 动作专家）→ Day14 π₀.₅（异构 co-training + knowledge insulation + 开放世界泛化）→ Day34 π0.7。官方报告 Sec III 自述：builds on π0.6 VLA + MEM 视频历史编码器，KI 训练 recipe 沿用 π0.5。π*0.6（RECAP）是另一条 specialist 线——π0.7 的定位就是"**一个通用模型取代多个 RL specialist**"。
- **补了哪个短板**：Day14 的 π₀.₅ 把泛化推到"新环境"（4 个未见家庭），但任务仍是见过的清洁类；π0.7 把泛化推到"**没教过的任务**"（unseen 任务组合）和"**没见过的机器人**"（cross-embodiment）。Day33 的预告"泛化三轴"在这里兑现：**环境泛化（Helix 2.5，30 间陌生家庭）/ 开放世界泛化（π₀.₅，新家庭做长程任务）/ 任务组合泛化（π0.7，没教过的任务）**——三个"泛化"定义正好互补，测的是不同的分布外维度。
- **替代 / 分叉 / 改进**：
  - vs π*0.6（RECAP specialist）：π0.7 用**训练时的 strategy metadata 蒸馏**把 Recap 经验装进单模型，达到/超过各任务 specialist——"通用 + 可转向"对"专用 + RL"，官方结论是前者赢了。这是路线里第一次有公司用自家数据证明 generalist ≥ specialist。
  - vs Day27 Cosmos（世界模型造数据）：π0.7 也有世界模型 $g_\psi$ （BAGEL 14B 混合 Transformer 做图像生成），但它的角色是**在线产出 subgoal 图像来 steer 策略**，不是离线造数据。这是"世界模型"的两种职业：Day27 是数据工厂，Day34 是领航员。
  - vs Day12 Diffusion Policy / Day11 flow matching：动作专家仍是 flow matching（860M 参数），路线没变；变的是**条件侧**——π0.7 把一半创新预算花在了"怎么给模型下指令"（prompt 工程的机器人版）。
  - vs Day32 Astra：Astra 是数字身体的指令跟随（OSWorld 2.0），π0.7 是物理身体的指令跟随（14 场景 × 3–6 步开放指令，4 个未见厨房 + 2 个未见卧室）。同一套"听懂人话"问题，Astra 有 DOM 树免费语义，π0.7 只有像素——物理侧更难，但 π0.7 > π0.6 > π0.5 的代际递进和 Astra 的版本递进是同一个形状。
- **跨阶段连接**：
  - Day30 数据飞轮：coaching 数据（人类语言纠错）→ 微调 high-level 策略 → 全自主，正是 Day30「failure → triage → recollect → retrain → gated deploy」的"人类语言版"实例；metadata（quality/mistake 标签）是飞轮里的 triage 信号被训练时直接吃掉。
  - Day28 评测合同：UR5e 折衣 80% vs 专家人类 80.6% 的对照设计是 Day28 精神的实例（人类基线 + 零数据声明），但评测方 = 被评测方、任务"真正 unseen"不可判定——两条都不满足 Day28 的独立性要求。
  - Day31 Atlas：Atlas 是"造世界"（从零训练的多模态世界模型），π0.7 的 $g_\psi$ 是"用世界模型做 subgoal"——Day31–34 四篇最新进展里，世界模型出现了两次，职业不同。
  - Day29 安全：metadata 里的 mistake 标签是把"犯过的错"变成训练信号；但 steerable 的反面是**可被 steer 去做错事**——prompt 注入的物理版，报告未讨论。

## 今天的 3 问
1. **"Remix 即组合泛化的本质"——那什么算"新任务"？** 官方承认数据集太大太杂，无法判定哪些任务真正 unseen；The Decoder 指出 DROID 里 Franka 臂"开 air fryer 抽屉放瓶子"的视频和红薯任务结构高度接近，公司只说"quite different"。这是 LLM 世界的 data contamination 争议第一次进机器人。真问题是：任务空间的"原子"是什么（物体？技能？物体×技能的笛卡尔积？）——不先定义原子，"组合"就是修辞。
2. **Metadata 为什么能"解毒"低质量数据？** 消融（Fig. 18，官方）：带 metadata 时，加再多低质量数据性能继续升；去掉 metadata，加低质量数据反而掉。直觉： $p(A\mid o)$ 混合了好坏示范是污染， $p(A\mid o,m=\text{bad})$ 把污染**条件化**成可识别的 mode——模型学会"坏示范长这样，别学它"。这和 LLM 里"数据质量标注 > 数据清洗"的争论是同一个数学：**标注是信息，清洗是删除**。
3. **Steerability 的代价谁付？** speed/quality/mistake 标签是额外的人工标注（类似 RLHF 的标注成本），subgoal 图像要训一个 14B 世界模型。π0.7 的"通用"是**用标注换来的**——和 Day16 DROID"裸 teleop 数据"的哲学正好相反。问题：如果标注预算砍掉一半，60–80% 的 unseen 成功率掉多少？官方没给这条消融。

## 核心

### 1. Motivation：通用模型打不过 specialist 的尴尬
- 背景：π*0.6（RECAP）证明了每个任务单独做 RL 能训出很强的 specialist，但"一任务一模型"不是产品。π0.5 的 open-world 泛化解决的是"去哪做"，没解决"做什么"——没教过的任务还是不会。
- 动机的数学形状：行为克隆学的是 $p_{\mathcal D}(A\mid o,\ell)$ ，当指令 $\ell_{\text{new}}$ 不在训练分布里，条件分布是外推。π0.7 的赌注：**把外推拆成" steering 信号的条件化"**——只要 steering 信号（subtask 语言、subgoal 图像、metadata）本身可组合，动作分布就能跟着组合。
- 和 Physical AGI 的关系：Day01 定义 Physical AGI 是"理解物理世界"的智能。π0.7 回答的是"理解"的操作定义：**能被语言转向到没做过的任务上**。The Decoder 的反驳（"不具备 reasoning，只是 hint"）恰好划出了今天的边界：steerable ≠ reasoning。

### 2. 机制：三件套系统 + steerable 条件
- **系统三件套**（官方报告 Fig. 2 + Sec IV）：(1) high-level 语义策略——用 coaching 数据微调的 π0.7，产出语言 subtask；(2) 轻量世界模型 $g_\psi$ ——BAGEL 14B 混合 Transformer，产出 subgoal 图像；(3) 期望 metadata。推理时是"语言子任务 → 图像 subgoal → 动作"的级联。
- **模型规模**（官方）：约 5B 参数 = 4B VLM backbone（Gemma 3 4B，含 400M vision encoder）+ 860M action expert（flow matching）。KI 沿用 π0.5：action expert 可 attend VLM，梯度不回流 VLM。
- **输入**（官方）：最多 4 个相机视角、最多 6 帧历史（压缩到与单帧相同的 token 数，MEM 编码器）、最多 3 张 subgoal 图，448×448。
- **Steerable 条件**（官方）：subtask 指令 + subgoal images（只加在 25% 的 batch 样本）+ episode metadata（speed 以 500 step 分桶如 "Speed: 8000"、quality 1–5 分、mistake 标签；整体 15% 丢弃、每子项 5% 丢弃）+ control mode（joint/ee，不 dropout）。Dropout 的设计意图：推理时缺某个条件也能工作。
- **数据**（官方）：in-house demo（实验室/类家居/野外家庭）+ 旧模型评测的 autonomous 数据（含 π0.6 RL 训练数据）+ 开源机器人数据集（含 DROID）+ 第一视角人类视频 + web 非机器人数据（物体定位、VQA、纯文本）。
- **训练配方**：π0.5 的异构 co-training + knowledge insulation 继续沿用；新增的是 metadata 条件化和 coaching 数据微调 high-level 策略。

### 3. 结果：官方记分卡 vs 独立视角
- **vs RL specialist**（官方 Fig. 6）：洗衣、espresso、盒子折叠三个 π*0.6/RECAP 任务，单个 π0.7 **追平 specialist 成功率**；多样洗衣折叠和盒子组装的**归一化吞吐量 >1.0**（更快）。"通用 ≥ 专用"是官方最想让你记住的数字。
- **跨 embodiment**（官方 Fig. 12）：双臂 UR5e（2× UR5e + Robotiq 夹爪，重、惯量大、不精确）折 T 恤，**零 UR5e 折衣数据**：任务进度 85.6%、成功率 80%；对照组是专家 teleop 人类（平均 375 小时操作经验，同样 zero-shot 上 UR5e）：进度 90.9%、成功率 80.6%。官方称"comparable"——注意这是**和人类比**，不是和别的模型比。
- **指令跟随**（官方 Fig. 9/10/11）：14 个场景 × 3–6 步开放指令序列，4 个未见厨房 + 2 个未见卧室；π0.7 **显著优于 π0.5 和 π0.6**（绝对成功率数字在柱状图中，未提取到）。
- **组合泛化**（官方）：air fryer 煮红薯——纯 zero-shot 指令 "load a sweet potato into the air fryer"，数次误启动后**只完成部分**；step-by-step 语言 coaching 后成功；把 coaching 数据微调 high-level 策略 → **全自主执行，零 teleop 数据**。另两个未见多阶段任务：清空 air fryer、烤 bagel。
- **数据扩展消融**（官方 Fig. 18）：带 metadata，加低质量数据性能继续提升；去掉 metadata，加低质量数据反而掉；去掉 20% 最高 task-diversity 数据，泛化大幅下降（随机去掉 20% 则不）——**多样性 > 数量**。
- **官方承认的边界**（Discussion）：seen 任务成功率常 >90%，**unseen 任务/任务-机器人组合 60–80%**；"难以确切判定哪些任务真正 unseen"；公司主张"remix 即组合泛化的本质"。
- **独立视角**：
  - TechCrunch（Heater）：报道框架是"研究人员自己都被能力惊到了"，引用 Levine "once it crosses that threshold…capabilities are going up more than linearly with the amount of data"（LLM 拐点类比）；但加了 **"if the findings hold up"** 的保留——无独立评测。
  - The Decoder：三点质疑——(1) 数据污染：air fryer 任务看似新，DROID 里结构接近的视频早有，公司只说"quite different"；(2) 引用数字确认 UR5e 折衣 80% 追平专家人类；(3) 机制确认 metadata 消融是真实贡献，但模型**不具备 reasoning/"think through"**，报告结尾只是 hint。
  - TechDogs（次级）：确认发布日期 2026-04-16、组合泛化口径、air fryer 与跨硬件折衣两个 demo。

## 数学视角

把 π0.7 写成**可转向的分层 POMDP**：Day14 的双层结构（慢语义层 + 快动作层）上，再加一层"图像 subgoal"和"元数据条件"。

- **Observation（观测）**： $o_t=[I^{(1:4)}_t,q_t]$ ，最多 4 路相机 RGB（448×448）+ 本体感知 $q_t$ （关节角/力矩）。历史用 MEM 编码：6 帧压缩到与单帧相同的 token 数——记忆不是堆帧数，是**信息瓶颈**。
- **Action（动作）**：动作块 $A_t\in\mathbb R^{H\times d_a}$ ，由 860M flow-matching action expert 生成（Day11 π₀的同一机制）。 $d_a$ 随 embodiment 变（UR5e 双臂 vs 移动操作平台），control mode（joint/ee）作为条件告诉模型当前是哪个执行空间——**跨 embodiment 的数学接口就是"条件里写明身体"**。
- **State（隐状态）**： $s_t$ = 场景物体位姿 + 任务进度。π0.7 不显式估计 $s_t$ ，而是用两个可观测的代理：语言 subtask（离散语义）和 subgoal 图像（连续视觉未来）——都是"未来的压缩表示"。
- **三层级联**：

$$\ell \xrightarrow{\;\pi^H\;} s \xrightarrow{\;g_\psi\;} \hat{I}_{\text{goal}} \xrightarrow{\;\pi^L\;} A_t,$$

即总指令 $\ell$ → high-level 策略 $\pi^H(s\mid o,\ell)$ 产出语言子任务 $s$ → 世界模型 $g_\psi(\hat{I}_{\text{goal}}\mid o,s)$ 产出 subgoal 图像 → 低层策略 $\pi^L(A_t\mid o,s,\hat{I}_{\text{goal}},m,u)$ 产出动作块。 $m$ 是 episode metadata（speed/quality/mistake）， $u$ 是 control mode。

- **Steering 的数学**：训练时学的是联合条件 $p(A\mid o,s,\hat{I},m,u)$ ，metadata dropout（15% 整体、每子项 5%）保证边缘分布 $p(A\mid o,s,\hat{I})$ 也可用。Coaching = 在测试时换掉 $s$ 的来源（人类给 step-by-step），模型行为跟着变——**可转向性就是"条件变量可替换"**。
- **组合泛化的边界公式**：unseen 指令 $\ell_{\text{new}}$ 的成功率取决于它能否被分解成见过的 $(s,\hat{I})$ 对。官方 60–80% 的 unseen 成功率 vs >90% 的 seen 成功率，差的就是"分解失败"的概率。DROID 污染争议的数学版： $\ell_{\text{new}}$ 和训练集的"技能原子"交集越大，越不算泛化——原子没定义，数字就没法审计。
- **Metadata 解毒的直觉**：无 metadata 时模型学 $p_{\text{mix}}(A\mid o)$ ，坏示范污染好示范；有 metadata 时学 $p(A\mid o,m)$ ， $m=\text{bad}$ 的 mode 被隔离成"可识别的坏"。这是**条件化 vs 混合**的区别，也是"标注 > 清洗"的信息论论据。
- **时间尺度**：快层动作块几十 Hz（沿用 π₀量级）；慢层 subtask 秒级；subgoal 图像是"段"级（一个子任务一张）。三层时间尺度：毫秒 → 秒 → 数十秒，和 Day14 的双层闭环是一脉相承的加法。
- **和系统实现的对应**：KI（action expert 梯度不回流 VLM）保证 web 语义不被动作数据洗掉——Day14 的绝缘子原样保留；新增的 14B 世界模型只做 subgoal 生成，不进动作梯度回路。

## 可迁移到 post-training

1. **Metadata-as-steering = 数据注解的机器人版**：speed/quality/mistake 标签让低质量数据从"污染"变成"信息"。映射到 post-training：SFT/RL 数据混合时，**给每条数据打质量/难度/来源标签并作为条件喂给模型**，而不是只做二值过滤——π0.7 的消融（去 metadata 则加烂数据掉分）是这条路线的第一个机器人证据。
2. **"一个通用 + 可转向" vs "多个专用"**：π0.7 追平 RL specialist 的组织含义——post-training 里"一个大模型 + prompt/adapter" vs "多任务多模型"的争论，机器人侧已经用自家数据投了一票给前者。代价是标注（steering 信号的采集成本），不是模型尺寸。
3. **Coaching → 微调的闭环是 RLHF 的物理版**：人类语言纠错（coaching）→ 收集成数据 → 微调 high-level 策略 → 全自主。post-training 的 agent 训练可以直接抄这个飞轮：**部署时的纠错交互本身就是最便宜的高质量训练数据**（Day30 数据飞轮的"人类语言版"）。
4. **Contamination 警示同样适用 LLM**：数据集大到无法判定"真正 unseen"时，"zero-shot"声明需要任务原子的定义。post-training 做 benchmark 时照抄：先定义技能原子，再谈泛化。
5. **Dropout 条件 = 鲁棒性的廉价来源**：metadata 15% 丢弃、subgoal 25% batch 才加，保证推理时缺条件也能工作。post-training 的启发：训练时随机丢弃 system prompt / 工具描述 / few-shot，换部署时的条件缺失鲁棒性。
6. **评测设计抄作业**：UR5e 折衣的对照组是"专家人类同样 zero-shot"——人类基线 + 零数据声明，这是 Day28 评测合同精神的好实例。post-training 的 agent 评测可以抄：**人类专家在同样信息下的表现**是最诚实的基线之一。
7. **世界模型的两种职业**：Day27 Cosmos（造数据）vs π0.7 $g_\psi$ （产 subgoal 做在线 steering）。post-training 里"世界模型/模拟器"也有两份工作：离线生成训练数据 vs 在线做规划/验证——立项时先想清楚雇它干哪份。

## 疑问

- 没看懂的 1 个问题：subgoal 图像 $g_\psi$ 对最终成功率的**独立贡献**是多少？报告有三件套系统，但没提取到"去掉世界模型只留语言 subtask"的消融——不知道 14B 图像生成是雪中送炭还是锦上添花。
- Coaching 微调 high-level 策略后，模型在**其他未见任务**上的泛化是变好还是变差（灾难性遗忘）？官方只报了被 coaching 的任务本身。
- "Speed: 8000"这类 metadata 在推理时由谁指定？如果是人类手动指定速度桶，那"全自主"里藏了一个人工程度参数。

## 下一步

- 2026-09-27 起第二轮复习（02/34）：Day34 的"泛化三轴"（环境/开放世界/任务组合）是复习期的现成对照框架，复习 Day11/14/33 时回填。
- 跟踪 openpi 是否放出 π0.7 权重（目前只有 π0/π0-FAST/π0.5）——权重放出后可做最小复现：关掉 subgoal 条件，对比组合泛化成功率，验证 $g_\psi$ 的独立贡献。
- 跟踪 The Decoder 提出的数据污染争议是否有后续（PI 回应 / 第三方复现）——"任务原子"的定义是组合泛化能否被审计的关键。
- Day31–34 四篇最新进展收官：复习期把 Atlas（造世界）/ Astra（数字身体）/ Helix 2.5（环境泛化）/ π0.7（任务组合泛化）四张卡摆在一起，写一篇"2026 年 Physical AI 的四种答案"。

## 速查表

| # | 条目 | 内容 |
|---|------|------|
| 1 | 发布 | 2026-04-16，PI 官方 blog "A Steerable Model with Emergent Capabilities" |
| 2 | 报告 | 官方 PDF《π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities》 |
| 3 | 一句话 | 可转向通用 VLA：追平 RL specialist，做没教过的任务组合 |
| 4 | 规模 | 约 5B：4B VLM（Gemma 3 4B，400M vision encoder）+ 860M flow-matching action expert（官方） |
| 5 | 继承 | π0.6 VLA 架构 + MEM 视频历史编码器；KI 训练 recipe 沿用 π0.5（官方 Sec III） |
| 6 | 三件套 | high-level 语义策略（coaching 微调，产语言 subtask）→ 世界模型 $g_\psi$ （BAGEL 14B，产 subgoal 图）→ metadata |
| 7 | 输入 | 最多 4 相机视角、6 帧历史（压缩至单帧 token 数）、最多 3 张 subgoal 图，448×448（官方） |
| 8 | 条件 | subtask 语言 + subgoal 图（25% batch）+ metadata（15% 丢弃/子项 5%）+ control mode（joint/ee，不丢弃） |
| 9 | metadata | speed（500 step 分桶，如 "Speed: 8000"）、quality 1–5、mistake 标签（官方） |
| 10 | 数据 | in-house demo + 旧模型 autonomous 数据（含 π0.6 RL 数据）+ 开源机器人集（含 DROID）+ 第一视角人类视频 + web 数据（官方） |
| 11 | vs specialist | 洗衣/espresso/盒子折叠追平 π*0.6；洗衣折叠+盒子组装归一化吞吐 >1.0（官方 Fig. 6） |
| 12 | 跨 embodiment | 双臂 UR5e 零折衣数据：进度 85.6%、成功率 80%；专家人类 zero-shot：90.9%/80.6%（官方 Fig. 12） |
| 13 | 指令跟随 | 14 场景 × 3–6 步开放指令，4 未见厨房 + 2 未见卧室；π0.7 > π0.6 > π0.5（官方 Fig. 9–11） |
| 14 | 组合泛化 | air fryer 煮红薯：纯 zero-shot 部分完成 → 语言 coaching 成功 → coaching 微调后全自主、零 teleop 数据（官方） |
| 15 | 另两个 | 清空 air fryer、烤 bagel（未见多阶段任务，官方） |
| 16 | 消融 | 带 metadata 时加烂数据继续涨；去 metadata 则掉；去 20% 最高多样性数据泛化大跌（官方 Fig. 18） |
| 17 | 边界 | seen 常 >90%；unseen 任务/机器人组合 60–80%；"难判定真正 unseen"（官方 Discussion） |
| 18 | 权重 | 未开源（openpi 只有 π0/π0-FAST/π0.5） |
| 19 | TechCrunch | "研究人员自己都被惊到"；Levine 拐点类比 LLM；"if the findings hold up" 保留（2026-04-16） |
| 20 | The Decoder | 数据污染质疑（DROID air fryer 视频结构接近）；"不具备 reasoning，只是 hint"；确认 metadata 消融真实 |
| 21 | 路线位置 | Day11 π₀ → Day14 π₀.₅ → Day34 π0.7；"一个通用取代多个 RL specialist" |
| 22 | 泛化三轴 | Helix 2.5 环境泛化 / π₀.₅ 开放世界泛化 / π0.7 任务组合泛化 |
| 23 | 对照 | Day27 Cosmos（世界模型造数据）vs π0.7 $g_\psi$ （世界模型产 subgoal 做 steering） |
| 24 | 对照 | Day32 Astra（数字身体指令跟随）vs π0.7（物理身体指令跟随） |
| 25 | 数学 | 三层级联 $\ell \to s \to \hat{I}_{\text{goal}} \to A_t$ ；毫秒/秒/数十秒三时间尺度 |
| 26 | 数学 | Steering = 条件变量可替换；metadata 解毒 = 条件化 vs 混合 |
| 27 | 迁移 | 数据打质量标签当条件 > 二值过滤；coaching 纠错是最便宜的训练数据 |
| 28 | 明天 | 2026-09-27 第二轮复习恢复（02/34），只更新已有 NOTES，不新增论文 |

## 连接
- 上一篇：Day 33 — Figure AI Helix 2.5（https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-33-2026-figure-ai-helix25）
- 下一篇预告：第二轮复习 02/34（2026-09-27 起，一天一篇，只更新已有 NOTES）
