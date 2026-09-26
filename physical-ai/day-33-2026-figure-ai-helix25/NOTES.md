# Day 33 — Figure AI Helix 2.5：30 间陌生家庭的 56%（最新进展 3/4）

## 元信息
- Title: Helix 2.5: Zero-Shot 30-Home Generalization
- Authors / Org: Figure AI（创始人兼 CEO Brett Adcock）
- 官方公告：Figure, "Helix 2.5: Zero-Shot 30-Home Generalization"（2026-09-17 发布；官方声明经 roboticfirms 转录核验——官方原文页未直接抓取，口径以多家媒体一致转述为准）
- Adcock X（2026-09-17）："We rented 30 homes in the Bay Area and are doing tasks without any new training"（StartupFortune 转述）
- 配套发布 A：Figure Index 公开（2026-08-25；此前 4 个月 stealth，代号 Project Go-Big）
- 配套发布 B：Figure × Nscale 战略合作（2026-09-03；Nscale 官方新闻稿，经 Reuters / Bloomberg 同日 corroborated——Techtimes 转述）
- 独立报道：TechRepublic（2026-09-21）、Sebertech（2026-09-18）、Biped.News、Unite.AI、Humanoids Daily（Adcock 9-17 RoboStrategy livestream 访谈）、TechTimes、The Weighted Average（算力协议限定词分析）
- Date read: 2026-09-25
- Tags: [physical-ai, vla, humanoid, helix, index, data-flywheel, generalization, scaling-law, compute, eval]
- Thread: physical-ai
- Folder: day-33-2026-figure-ai-helix25
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-33-2026-figure-ai-helix25
- 范围说明：Helix 2.5 是路线 VLA / humanoid 主线的最新进展；Index 数据众包与 Nscale 算力协议是 Day30 数据飞轮的产业实例，一并记录。但**融资/估值/算力协议不算技术证据**——本 NOTES 把公司 PR 口径与独立评测分开标注，每个关键数字标注来源。

## 一句话总结
Figure Helix 2.5（2026-09-17）：一台 Figure 03 人形机器人，用**同一份冻结 checkpoint**走进 30 间从未采集过数据的 Bay Area 陌生家庭，做三件长程家务（铺床、叠毛巾、收拾客厅玩具），420 次试验成功 237 次（56%）；同一架构、同一任务数据、从零训练的对照策略只有 9%——唯一的变量是是否经过 Index 人类行为数据预训练。**全部数字是公司自报、公司自己跑的评测，没有独立复核**，这是今天最硬的边界。

## 和之前工作的关系

- **接了哪条线**：Day09 RT-2 → Day11 π₀ → Day26 GR00T N1 的 VLA 线。Helix 是 Figure 的自家 VLA（Helix 02 早先已演示双机协作整理卧室、200 小时物流任务——Unite.AI），Helix 2.5 是这条线的"泛化证据"版本。Day26 GR00T N1 走的是 dual-system（reasoning + control）foundation model 路线；Helix 2.5 证明了另一张配方：**人类行为数据预训练 + 冻结 checkpoint**，不靠更大的推理系统。
- **补了哪个短板**：Day16 DROID / Day17 BridgeData / Day18 RoboCasa 都是 teleop 或 sim 里的人类遥操作 corpus；Index 是**纯人类视频**（手机/头显记录人做事，不是机器人遥操作），模型从 random weights 预训练。补的是路线里一直缺的一块证据：**数据与 robot embodiment 脱钩之后，迁移到底发不发生**。Helix 2.5 的回答是：发生，且是 9%→56% 的六倍。
- **替代 / 分叉 / 改进**：
  - vs Day27 Cosmos：Cosmos 是"**造数据**"（world model 生成合成数据喂策略）；Index 是"**收数据**"（众包真人视频）。两种数据飞轮哲学正面交锋：Figure 押注真人数据的可迁移性，World Labs / Cosmos 押注生成数据的可扩展性。Day31 的 Atlas 如果成熟，会成为第三极。
  - vs Day12 Diffusion Policy：三件家务都是长程 whole-body 任务（铺床 = locomotion + 双臂 + 柔性体），Day12 的 receding-horizon / 闭环重规划思想在这里的对应物是官方演示里的**自我纠错**：犯错后退后重站位、换 stance、绕到床另一边纠正折叠（homecrux 转述官方演示）——定性证据，不是量化指标。
  - vs Day32 Astra（昨天）：Day32 NOTES 预告的双线串联。Astra 是**数字身体**（观测=屏幕像素+DOM，动作=键鼠坐标，OSWorld 2.0 离线子集 72.6%）；Helix 是**物理身体**（观测=相机+本体感知，动作=全身关节，陌生家庭 56%）。同一套 POMDP，两套身体，同一个瓶颈期：**都停在"一半多一点成功"**。数字身体有 DOM 树这种免费语义标注，物理身体没有——这是两条线难度差的结构原因。
- **跨阶段连接**：
  - Day30 数据飞轮：Index → Helix 预训练 → 部署 → 回流，正是 Day30「failure → triage → recollect / resimulate → retrain → gated deploy」的产业实例；"35 分钟/秒"（Figure 官方口径）是飞轮输入侧的数字。
  - Day28 评测合同：30 间陌生家庭 = 固定测度 $\mu$ 下的二项估计 $\hat{J}(\pi)=237/420$ 。但 $\mu$ 是 Figure 自己租的 30 间 Bay Area 房子、Figure 自己跑的评测——Day28 要求公开测度与评测方独立，这里两条都不满足。
  - Day29 安全：Figure 把每次 safety intervention 直接记为**失败**，不是灰色地带（StartupFortune）——这是 Day29 RTA / runtime monitor 思想在产品评测里的影子：安全干预是成本，要进分母。
  - Day14 π₀.₅ → Day34 预告：π₀.₅ 是开放世界 VLA 泛化（任务/场景级），Helix 2.5 是**环境级** zero-shot（environment generalization），Day34 π0.7 是**任务组合**泛化（compositional）——三个"泛化"定义正好互补，明天 Day34 做"泛化三轴"对照。

## 今天的 3 问
1. **Figure 的 "zero-shot" 到底是什么？** 官方定义的 zero-shot = 评测家庭和操纵对象 unseen，但三个任务类别是别处 fine-tune 过的（TechRepublic、Sebertech 一致指出）。按严格定义这更接近**环境泛化**而非任务 zero-shot。真问题是：Index 预训练到底转移了什么——低层运动原语、物体先验，还是高层任务结构？如果换一个从未 fine-tune 过的任务类别（比如叠衣服），成功率会是多少？官方没给。
2. **Scaling law 的 $L$ 和下游成功率之间差了几步？** Figure 展示的是 held-out action-prediction loss 随 Index 数据翻倍的光滑下降，且能用小 run 预测大 run 的 loss 到小数点后四位（Techtimes、roboticfirms 转述官方）。但 loss→task success 的映射是非线性的：56% 是二值门，中间隔着误差累积 $(1-\epsilon)^T$ （Day12 / Day32 的同一数学）。Figure 没给出 $L\mapsto p_{\mathrm{success}}$ 的外推——56%→90% 需要多少数据仍是未知数，而这正是 Index"未来 12 个月采集量扩大 100x"计划（WOWTALE 转述官方）要回答的算术。
3. **Baseline 的 8.3% vs 9%，以及 30 间房的代表性。** Humanoids Daily 指出 Figure 官方图表显示 baseline 是 35/420 ≈ 8.3%，而官方文字写 9%（Sebertech）——小口径出入，但说明"公司自报"的颗粒度。再往深：30 间全是 Bay Area 租的房子，户型、家具风格、光照的分布 $\mu$ 有多窄？一个东京/上海老破小里的机器人会怎样？这是 Day28"测度透明度"问题的活实例。

## 核心

### 1. Motivation：四章路线里的"扩智能"
- Adcock 在 9-17 RoboStrategy livestream 提出四章路线：造机器 → 做自主 → 扩智能 → 扩生产；前两章靠工程突破，后两章靠资本加速；Figure 自认处在"扩智能"章的开头（Humanoids Daily 转述——这是他自己的定位，不是独立里程碑）。
- 动机就是 Day30 的数据飞轮论点：瓶颈是数据和算力（Adcock 原话，Intelligent CIO 转述官方）。Helix 2.5 是"扩智能"章的第一个有界证据，官方声明（roboticfirms 转录）："The point is not that general humanoid robotics is solved. But Helix 2.5 is the first evidence that whole-body intelligence can be learned from human experience and transferred to new scenarios, rather than rebuilt each time."
- Adcock 自己承认：本年度 best-case 目标（一个机器人在家里做长程自主工作）这次还没达到（Humanoids Daily）——56% 和"产品"之间隔着可靠性鸿沟。他把 Helix 2.5 称为 "The most important project we've ever taken on at Figure"（X，StartupFortune 转述），把 zero-shot generalization 称为行业圣杯——措辞是 PR，数字是证据，要分开读。
- 和 Physical AGI 的关系：Day01 定义 Physical AGI 是"理解物理世界"的智能。Helix 2.5 回答的是"理解能否从人类视频里学"——**embodiment 脱钩的数据假设**，这是它和 Day16/17/18 teleop 数据路线的根本分歧。

### 2. 机制：Index 预训练 + 冻结 checkpoint + 少量任务适配
- **数据（Index）**：2026-08-25 公开，此前 4 个月 stealth（代号 Project Go-Big）。公开时数字（Figure 官方口径，经 WOWTALE / ARC Advisory）：264,000 下载、108 国家、44,000+ 周活贡献者、1,600 万+ 视频、已向 Creators 支付 \$15M。速率：发布时 30 分钟视频/秒（≈ 每天 4.9 年人类活动）；到 9-17 Helix 2.5 发布时 ≈ 35 分钟/秒（≈ 每天 5.7 年；Techtimes）。两个都是公司自报，涨幅 16.7%（The Weighted Average 按官方数字算）。
- **多样性**：每 1,000 小时 373 个 unique tasks、1,146 个 unique objects、116 个 unique environments（Figure 官方口径，经 cryptobriefing）。五阶段管线：自动过滤 → fraud review → 去重 → task-quota rebalancing → hierarchical captioning（WOWTALE 转述官方）。
- **采集方式**：Creators 戴 sensor headset 第一视角记录家务/工作（Humanoids Daily），或在 app 里雇 gig worker 上门做家务并拍摄（Humanoids Daily）。动机：Figure 先试过从外部供应商买数据，Adcock 原话 "We went out and bought a bunch of stuff and it was just crap"（Humanoids Daily livestream 转述）——有用数据需要传感器对齐（和机器人观测/动作空间匹配）、清洗和反欺诈，不只是小时数。
- **训练配方**（Techtimes / bytevyte 转述官方）：从 **random weights 初始化，完全在 Index 上预训练**（没有 LLM base、没有环境特异的演示），然后用少量任务数据适配出三个 whole-body 行为。受控对照：同一架构、同一任务数据、同一评测设置，唯一变量是有无 Index 预训练 → 56% vs 9%。Techtimes 评价：这个单变量对照才是公告的技术心脏，不是 56% 本身。
- **硬件不变**：Figure 03（173cm，61kg），纯软件更新（Techtimes）。30 间房用同一份冻结 checkpoint，"one fixed checkpoint, the same weights, in every single home"（StartupFortune 转述官方）。
- **算力配套（Nscale，2026-09-03）**：初始 \$3.5B，意向扩到 \$6B+；最多 100,000 NVIDIA Vera Rubin GPUs；首批 H2 2027 上线，Barstow, Texas；Nscale 成为 Figure 优先算力供应商 + 战略入股 Figure；探索用人形机器人做 Nscale 供应链（Reuters 通稿，经 SRN News / Intelligent CIO）。Jensen Huang 口径："physical AI flywheel"——Vera Rubin 训练 → Isaac Sim 验证 → Figure 机器人端侧 NVIDIA GPU 部署。**注意限定词**：全是 "potential / intent"，第一批 GPU 要到 2027 下半年（The Weighted Average）——这是算力/融资新闻，不是技术证据，不进技术结论的分母。

### 3. 结果：官方记分卡 vs 独立视角
- **官方**：420 试验，237 成功 = 56%（Figure 官方；TechRepublic / Humanoids Daily 转述）。分任务：铺床 94/140（67%），叠毛巾 87/140（62%），收拾客厅玩具 56/140（40%）（Sebertech / StartupFortune 转述官方图表）。判分严格：必须完整完成；收拾玩具要求 13–15 个玩具全部进篮子；safety intervention 记为失败（StartupFortune / Sebertech）。
- **对照**：同任务数据从零训练 9%（官方文字）；但 Humanoids Daily 指出官方图表显示 35/420 ≈ 8.3%——小口径出入（Sebertech）。
- **效率**：达到 Helix 02 同等成功率只用了 50% 的任务适配数据（Figure 官方，经 roboticfirms / bytevyte）。单次评测任务占预训练总数据不到 1.90%（bytevyte 转述官方，单源）。
- **Scaling law**（Figure 官方，经 Techtimes / roboticfirms）：四档 Index 数据量（每次翻倍）训练同一模型，held-out action-prediction loss 光滑下降；只用小 run 预测大 run 的 test loss 到小数点后四位。号称"人形平台上第一个测得的人到机器人迁移 scaling law"（roboticfirms 转述官方）——公司自报曲线，无独立复现。
- **独立视角**：
  - Sebertech：56% 是进步，但"可靠的家用机器人仍在路上"；评测由 Figure 自己执行，非独立机构；44% 试验未完整完成。
  - Biped.News：这正是 Index 发布时缺的证据（human video 是否真提升迁移）——Helix 2.5 补上了这个对照，但结果仍是内部的、任务是别处 fine-tune 过的、56% 意味着大量试验没走完。
  - TechRepublic：zero-shot 的限定——家庭/布局/物体 unseen，但任务类别是训练过的。
  - Humanoids Daily（Adcock 访谈）：Adcock 把预期锚定在三件已测行为上，不肯外推到"能泛化多少任务"；Index 视频里的人类活动多样性 ≠ 机器人已演示的能力。
  - The Weighted Average：\$3.5B 的限定词——potential capacity ≠ installed capacity；数据采集和算力交付是两条不同的时钟。

## 数学视角

把 Helix 2.5 写成 POMDP，与昨天 Astra 的数字身体逐项对照——"一套数学、两套身体"的物理侧实例。

- **Observation（观测）**： $o_t=(\mathrm{headcam}_t,\mathrm{proprio}_t)$ 。headcam 为头载相机 RGB（类比 VLA 的视觉输入），proprio 为本体感知（关节角/力矩/足底接触）。对照 Day32：Astra 的观测有 DOM 树这种**免费语义标注**，Helix 的观测是纯物理像素+本体，没有语义捷径——这是物理身体难一个数量级的结构原因之一。
- **Action（动作）**：whole-body 动作 $a_t=(a^{\mathrm{loco}}_t,a^{\mathrm{arms}}_t)$ ， $a^{\mathrm{loco}}$ 管移动（足步/基座速度）， $a^{\mathrm{arms}}$ 管双臂关节/末端位姿。连续高维——这正是 VLA 需要 action chunking / flow matching（Day11 π₀）的原因。官方未公开 Helix 2.5 的 DoF 数和控制频率，本 NOTES 不编数字。
- **State（隐状态）**： $s_t$ = 房间布局 + 家具位姿 + 可变形物体构型（毛巾/被褥的布料状态近乎无限维，实际部分可观测）。分任务成功率的直觉解释：铺床 67% > 叠毛巾 62% > 收拾玩具 40%——被褥虽可变形但**结构先验强**（床是矩形、被子要铺平），玩具是刚性小物体但**位姿/类别完全随机**。泛化的难度不在刚/柔，而在**状态分布的熵**。
- **评测的二项模型**：每次试验 $X_i\sim\mathrm{Bernoulli}(p)$ ， $\hat{p}=237/420\approx0.56$ ；分任务 $\hat{p}_{\mathrm{bed}}=94/140\approx0.67$ ， $\hat{p}_{\mathrm{towel}}=87/140\approx0.62$ ， $\hat{p}_{\mathrm{tidy}}=56/140=0.40$ 。这是 Day28"固定测度 $\mu$ 下的二项估计"的实例——但 $\mu$ （30 间 Bay Area 租房的分布）不透明，且评测方 = 被评测方。
- **Scaling law 的缺口**：held-out next-action loss $L(D)$ 随 Index 数据量 $D$ 翻倍光滑下降，Figure 声称可用小 $D$ 拟合外推大 $D$ 的 $L$ 到 $10^{-4}$ 精度。关键缺失是 $L\mapsto p_{\mathrm{success}}$ 的映射： $L$ 是 token 级的，56% 是任务级的，中间隔着误差累积 $(1-\epsilon)^T$ （Day12 / Day32 的同一数学）。没有这条映射，"100x 数据→可靠性"的算术就是外推信仰。
- **数据速率**：35 分钟/秒 $=2100$ 小时/小时 $\approx 50{,}400$ 小时/天。Figure 计划未来 12 个月采集量扩大 100x（WOWTALE 转述官方）——如果 scaling law 斜率成立，这是把"可靠性鸿沟"翻译成"资源分配问题"的算术（Techtimes 的解读）；如果不成立，100x 只是 100x 的成本。
- **时间尺度**：whole-body locomanipulation 是秒~分钟级任务（铺一张床几分钟），控制频率在几十 Hz 量级（VLA 典型；官方未公开具体数字）——介于 Day11 π₀（50 步动作块、几十 Hz）和 Day32 Astra（分钟级任务、约 0.1–1 Hz 有效决策）之间。物理身体的"快"，恰好是数字身体做不到的奢侈；反过来，Astra 每步可做的长 CoT，Helix 也做不到。

## 可迁移到 post-training

1. **人类视频预训练 = 机器人版的"互联网预训练"**：LLM 用互联网文本做 pretrain，Helix 用 Index 人类视频做 pretrain——都是"便宜、脱钩、可扩展"的数据。映射到 post-training：agent 训练的"Index"是什么？是人类操作电脑/手机的录屏（Astra 的数据飞轮）、客服对话、代码 diff——**先找到脱离目标 embodiment 也能采集的便宜数据，再做少量任务适配**，这个配方是通用的。
2. **受控消融是技术可信度的硬通货**：Helix 2.5 公告的技术心脏不是 56%，而是"同一架构同一数据、唯一变量是 Index 预训练"的 9%→56% 对照（Techtimes 的评价）。post-training 写实验报告照抄：single-variable ablation 比绝对数字更有说服力。
3. **报 loss 曲线不如报 loss→success 映射**：Figure 给了漂亮的 scaling law（loss 外推到小数点后四位），但没给 $L\mapsto p_{\mathrm{success}}$ 的外推——56%→90% 需要多少数据仍是未知数。post-training 做 scaling law 时要同时外推**下游指标**，不然曲线只是装饰。
4. **评测的"公司自评"边界要写在最前面**：30 间房 Figure 租的、Figure 跑的、baseline 8.3%/9% 口径小出入——Day28 的评测合同要求公开测度 $\mu$ 和评测方独立性。post-training 的内部评测报告照抄这个诚实模板：谁跑的、分布是什么、有没有独立复核。
5. **Safety intervention 记为失败**：Figure 把安全干预直接计入失败而不是灰色地带——这是把 Day29 的安全栈折进评测指标。post-training 的 agent 评测可以抄：guardrail 触发 = 任务失败，而不是"成功但被拦下"。
6. **数据飞轮的"两条时钟"**：The Weighted Average 的提醒——Index 采集速率涨 16.7%（公司自报）和 Nscale 第一批 GPU 2027 下半年才上线，数据和算力是两条时钟。post-training 规划 infra 时也要分开建模：数据管线 throughput 和算力交付时间表是独立的约束。
7. **"买数据 crap"的教训**：Adcock 说外部供应商的数据质量不行、被迫自建（Humanoids Daily）。映射：post-training 的数据采购——vendor 数据的"传感器对齐"（格式/分布/标注质量与你的模型匹配度）比"小时数"重要；先小批量验证迁移效果再 scale 采购。

## 思考题

1. **语义标注值多少？**（综合 Day33 / Day32）Astra 的观测有 DOM 树这种免费语义标注，Helix 只有纯像素 + 本体感知——两条线都停在"一半多一点"（72.6% vs 56%），但难度结构不同。设计对照实验：如果给 Helix 加上预先建好的房间语义地图 / 物体 6D 位姿（数字身体的"免费午餐"），56% 能涨多少？反过来，把 Astra 的 DOM 拿掉只给纯像素，72.6% 会掉多少？这个双向消融能量化"语义标注"这个结构差异到底值多少个点——也是给 Day31 Atlas 这类"造世界供料"路线的定价问题。
2. **从 56% 到 90% 的算术**（综合 Day33 / Day12 / Day28）：Figure 只给了 loss 的 scaling law，没给 $L\mapsto p_{\mathrm{success}}$ 的映射。设单步错误率 $\epsilon$ ， $T$ 步任务成功率约 $(1-\epsilon)^T$ 。若 Index 数据翻倍把 $\epsilon$ 降低 10%，铺床（67%）和收拾玩具（40%）的成功率分别涨多少？反推两类任务的"有效步数" $T$ ——收拾玩具的 $T$ 为什么更大（13–15 个玩具全部进篮子 = 串行长程）？这条算术能判断"100x 数据"到底是可靠性方案还是成本黑洞。
3. **安全分级会不会成为新的藏数字的地方**（综合 Day33 / Day29 / Day32）：Figure 把 safety intervention 直接记为失败——诚实。但 Day32 的 Astra 把能力按身份分级（Daybreak），Day29 的安全栈是同一模型加装证书。设想同一台 Figure 03：房主身份给全能力、访客身份给受限能力——这时报的 56% 还有意义吗？ $\pi_{\mathrm{deploy}}(a\mid o,\mathrm{identity})$ 让"成功率"变成身份的条件分布。post-training 的 agent 评测如果也引入能力分级，评测合同（Day28）要怎么改写才能不被分级口径稀释？

## 疑问

- 没看懂的 1 个问题：Index 预训练的**目标函数**到底是什么？官方 scaling law 只提了 action-prediction loss，但人类视频没有机器人 action label——"action-prediction"的监督信号从哪来（hand pose 估计？latent action？视频生成辅助目标？）？这是整个路线的技术黑箱，官方没给。
- 冻结 checkpoint 在 30 间房里"零适应"——那部署时遇到分布外到什么程度会触发 fallback？Figure 把 safety intervention 记为失败，但没公开干预率和触发条件。
- "50% task data 达到 Helix 02 同等成功率"的对照里，Helix 02 的 in-environment 成功率绝对数字是多少？官方只说了"同等"，没给分母。

## 下一步

- Day34（π0.7，2026-09-26）：把三个"泛化"定义摆在一起——Helix 2.5 的环境泛化（environment-level zero-shot）vs π0.7 的任务组合泛化（task-level compositional）vs π₀.₅ 的开放世界泛化，写一篇"泛化三轴"对照。
- 跟踪 Index 的 100x 采集计划和 Nscale 2027 下半年首批 GPU 上线：看 Figure 是否公布 $L\mapsto p_{\mathrm{success}}$ 的外推曲线——那是验证 scaling law 有没有兑现的关键。
- Figure 04 硬件发布时（Adcock 称"iPhone one moment"，设计已锁定、无发布时间——Humanoids Daily），回看 Helix 2.5"软件定义泛化"的论点：硬件换代会不会改写数据配方的结论。

## 速查表

| # | 条目 | 内容 |
|---|------|------|
| 1 | 发布 | 2026-09-17，Figure 官方公告 "Helix 2.5: Zero-Shot 30-Home Generalization" |
| 2 | 一句话 | 同一冻结 checkpoint 的 Figure 03，进 30 间陌生家庭做 3 件家务，56% |
| 3 | 试验 | 420 次试验，237 成功 = 56%（官方口径，公司自评，无独立复核） |
| 4 | 分任务 | 铺床 94/140（67%）；叠毛巾 87/140（62%）；收拾玩具 56/140（40%） |
| 5 | 对照 | 同架构同数据从零训练：9%（官方文字；图表显示 35/420≈8.3%，小出入） |
| 6 | zero-shot 定义 | 家庭/布局/物体 unseen；任务类别是别处 fine-tune 过的（环境泛化） |
| 7 | 判分 | 必须完整完成；safety intervention 记为失败（StartupFortune） |
| 8 | 硬件 | Figure 03 不变（173cm / 61kg）；纯软件更新（Techtimes） |
| 9 | 效率 | 达 Helix 02 同等成功率只用 50% 任务适配数据（官方口径） |
| 10 | Scaling law | Index 数据翻倍 → held-out action loss 光滑下降；小 run 预测大 run loss 到 1e-4（官方口径） |
| 11 | 缺口 | 无 $L\mapsto p_{\mathrm{success}}$ 外推；56%→90% 需多少数据未知 |
| 12 | Index 发布 | 2026-08-25 公开；此前 4 个月 stealth（代号 Project Go-Big） |
| 13 | Index 规模 | 264k 下载 / 108 国家 / 44k+ 周活 / 1600 万+ 视频 / 已付 \$15M（官方口径） |
| 14 | Index 速率 | 发布 30 分钟/秒 → 9-17 约 35 分钟/秒（≈5.7 年人类活动/天；官方口径） |
| 15 | Index 多样性 | 每 1k 小时：373 tasks / 1146 objects / 116 envs（官方口径） |
| 16 | Index 管线 | 自动过滤 → fraud review → 去重 → task-quota 再平衡 → hierarchical captioning |
| 17 | Index 动机 | 外部供应商数据 "it was just crap"（Adcock）；未来 12 个月投 \$1B+ 数据+算力，采集扩 100x |
| 18 | Nscale 协议 | 2026-09-03；初始 \$3.5B，意向 \$6B+；最多 100k Vera Rubin GPU（Reuters/Bloomberg corroborated） |
| 19 | Nscale 条款 | 首批 H2 2027，Barstow, Texas；Nscale 优先算力商 + 战略入股；探索人形机器人进其供应链 |
| 20 | Nscale 限定词 | 全是 potential/intent；第一批 GPU 2027 下半年才上线（非技术证据） |
| 21 | Adcock 路线 | 四章：造机器 → 做自主 → 扩智能 → 扩生产；自认处在扩智能章开头 |
| 22 | Adcock 承认 | 年度 best-case（家里长程自主工作）这次没达到（Humanoids Daily） |
| 23 | 路线位置 | Day11 π₀ / Day26 GR00T N1 的 VLA 线；人类视频预训练是另一张配方 |
| 24 | 对照 | Day27 Cosmos（造数据）vs Index（收数据）；Day31 Atlas（造世界）vs Index（录世界） |
| 25 | 对照 | Day32 Astra：数字身体 vs 物理身体；同一年，同一个"一半多一点"瓶颈期 |
| 26 | 数学 | POMDP： $o_t$ 相机+本体， $a_t$ 全身连续动作， $s_t$ 含可变形体构型 |
| 27 | 数学 | 二项估计 $\hat{p}=237/420$ ；难度在状态分布的熵，不在刚/柔 |
| 28 | 迁移 | 人类视频=机器人版互联网预训练；单变量消融是硬通货；guardrail 触发记为失败 |
| 29 | 明天 | Day34 π0.7：环境泛化 vs 任务组合泛化 vs 开放世界泛化，"泛化三轴"对照 |

## 连接
- 上一篇：Day 32 — OpenAI GPT-6 Astra（https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-32-2026-gpt6-astra）
- 下一篇预告：Day 34 — Physical Intelligence π0.7（组合泛化；2026-09-26）
