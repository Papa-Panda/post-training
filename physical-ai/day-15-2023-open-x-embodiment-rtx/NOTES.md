# Day15 — Open X-Embodiment / RT-X: 跨机器人数据规模化

> Open X-Embodiment Collaboration（Google DeepMind 牵头，21 institutions），arXiv 2310.08864（2023-10-13，ICRA 2024）。机器人学的"ImageNet 时刻"宣言：把 34 个实验室的 60 个数据集、22 种机器人、100 万+ 轨迹统一成 RLDS schema，用一个 7 维末端执行器动作接口做粗对齐，训练跨机器人通用策略 RT-X。

## 元信息
- Title: Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- Authors / Org: Open X-Embodiment Collaboration（Quan Vuong, Brian Ichter, Chelsea Finn, Sergey Levine 等 290+ 作者）/ Google DeepMind 牵头 + 21 institutions（Stanford、UC Berkeley、CMU、Meta AI、TRI、NVIDIA、KAIST、东京大学、清华、上海交大等）
- Link / arXiv / Blog: https://arxiv.org/abs/2310.08864 ；项目页 https://robotics-transformer-x.github.io/ ；代码/数据 https://github.com/google-deepmind/open_x_embodiment
- Date read: 2026-09-06
- Tags: [physical-ai, open-x-embodiment, rtx, cross-embodiment, robot-data, data-mixture, behavior-cloning, emergent-skills, rlds]
- Thread: physical-ai
- Folder: day-15-2023-open-x-embodiment-rtx
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-15-2023-open-x-embodiment-rtx

## 一句话总结
OXE 把机器人学的"数据孤岛"问题变成"格式统一"问题：60 个数据集、22 种 embodiment、100 万+ 轨迹、527 个技能（160,266 个任务）全部转成 RLDS 格式，动作统一到 7 维末端执行器接口（$x,y,z$, roll, pitch, yaw, gripper），然后**不做任何显式的 embodiment-gap 对齐机制**，直接在混合数据上训练 RT-1-X（35M）与 RT-2-X（55B PaLI-X）；3600 次真实机器人评测证明跨机器人**正向迁移真实存在**：小数据域平均 +50%，RT-2-X 在别的机器人数据里的技能上相对 RT-2 提升约 3 倍（emergent skills 27.3% → 75.8%）。

## 和之前工作的关系

- 接了哪条线：Day15–18 "机器人数据规模化"专题的**源头**。Day11–14（π₀ / Diffusion Policy / Octo / π₀.₅）讨论的都是"给定数据怎么训出好策略"，OXE 回答的是更底层的问题："数据从哪来、怎么拼成预训练底座"。
- 补了哪个短板：Day09–14 的训练数据都隐含"某一个实验室的几台机器人"；OXE 把规模化从单机房扩展到全社区，是 Day14 π₀.₅"异构 co-training 配方"的历史前传——π₀.₅的配方思想第一次被系统验证的地方就是这里。
- 替代 / 分叉 / 改进：
  - 替代了 per-robot / per-task / per-lab 各自训各自的范式：统一 schema（RLDS + 7 维动作接口）替代各自为政的数据格式。
  - 相对 Day13 Octo：Octo 用的正是 OXE 的 25 个数据集（约 80 万轨迹）——Octo 回答"怎么用混合数据"（block-masked Transformer + diffusion readout + 可插拔接口），OXE 回答"为什么值得聚合"（正向迁移的实验论证）。Octo 是 OXE 的下游消费者。
  - 相对 Day09 RT-2：RT-2 训在 Google Robot 自己的数据上；RT-2-X 把同样的"动作即文本 token"机制扩展到 9 种机械臂的混合数据——路线相同，数据混合是唯一变量。
- 对之前 Day X 的直接对比：Day14 π₀.₅的 ablation 证明"去掉某个数据成分泛化就掉"；OXE 的"去掉 Bridge 数据集"消融（emergent skills 75.8% → 42.8%）是同一个思想的更早、更干净的版本：把跨机器人迁移做成了**可归因的因果实验**。

## 为什么今天读它
路线图 Day15–18 是"Robot Data / Benchmark 扩展"阶段，OXE 是这一切的起点：BridgeData V2（Day17）、DROID（Day16）、RoboCasa（Day18）都是沿着"聚合→统一→规模化"这条路往下走的数据工作。不读 OXE，后面三天的数据论文就只剩"又一个数据集"的表面印象。同时它也是理解 π₀（Day11）"10k+ 小时跨 embodiment 预训练"为什么有效的理论地基。

## 今天的 3 问
1. 动作空间的对齐到底对齐了什么？论文只做了 7 维末端 delta + per-dataset 归一化 + 256-bin 离散化，**刻意没有**统一坐标系、没有区分绝对/相对/速度控制——为什么这么粗的对齐还能产生正向迁移？迁移到底发生在哪个表示层？
2. RT-1-X 在小数据域平均 +50%，但在大域（Bridge、Google Robot）反而不如单机房训的 RT-1（underfit：Bridge 上 27% vs 40%，RT-1 6 技能上 73% vs 92%）——容量 $C$ 与混合数据熵 $H(\mathcal D_{mix})$ 之间的 scaling 关系是什么？这对"先堆数据还是先堆模型"有什么启示？
3. Emergent skills 实验里，去掉 Bridge 数据后 RT-2-X 从 75.8% 跌到 42.8%——这个"移除式消融"如何把"跨机器人迁移"从相关性变成因果证据？它和 Day14 π₀.₅的成分消融在方法论上是什么关系？

## 数学视角：多源混合行为克隆 + 共享动作接口的迁移

把 OXE 理解成一个**跨分布的行为克隆混合问题**：$K$ 个数据源（数据集/机器人），每个有自己的观测分布、动作标定和任务集，但共享一个粗粒度的动作接口。统一的数学框架是——

### 1) State / observation / action / objective

- **Observation**：$o=(I,\ell)$，$I$ 是每个数据集选定的**规范视角**单张 RGB（统一缩放到公共分辨率）+ 语言指令 $\ell$；RT-1 架构用 15 帧历史 $I_{t-14:t}$，RT-2-X 用短历史（2 帧）或单帧。
- **Action**：规范化的 7 维向量 $a=(dx,dy,dz,droll,dpitch,dyaw,g)\in\mathbb R^7$——末端执行器位姿增量或速率 + 夹爪开合。关键：论文**没有**跨数据集统一坐标系，也**没有**统一"绝对位置 vs 相对增量 vs 速度"的控制语义，每个数据集按自己原来的控制方式解释同一个向量。
- **对齐算子**（per-dataset $k$）：先归一化 $a^{(k)}_{norm}=(a-\mu_k)/\sigma_k$，再均匀离散化到 256 档：
$$b_j = \Big\lfloor 255\cdot \mathrm{clip}\!\left(\frac{a^{(k)}_{norm,j}-lo}{hi-lo},\,0,\,1\right)\Big\rfloor\in\{0,\dots,255\},\quad j=1..7,$$
外加第 8 维 episode-termination token。模型在 $8\times 256$ 的离散空间上做分类（RT-1 是 256-way softmax，RT-2 是把 "1 128 91 241 5 101 127" 当文本 token 做 LM loss）。
- **混合目标**：
$$\mathcal L(\theta)=\sum_{k=1}^{K} w_k\;\mathbb E_{(o,\ell,b)\sim\mathcal D_k}\big[\mathrm{CE}(\pi_\theta(b\mid o,\ell))\big],$$
RT-2-X 再加一层 co-fine-tuning：$\mathcal L=\mathcal L_{VLM}(\text{web})+\mathcal L_{robot}$，web 数据与机器人数据比例约 1:1（沿用 RT-2 配方）。

### 2) 为什么粗对齐还能迁移：task-space 共享结构

直觉：不同机器人的逆运动学 $f_i^{-1}:\text{task-space}\to\text{joint-space}$ 各不相同，但"看到香蕉在碗左边 → 末端向左上方移动"这个**任务空间**的映射 $p^*(b\mid o,\ell)$ 在各 embodiment 间有重叠支撑。粗对齐的有效性说明迁移发生在**语义-任务空间层**，而不是关节动力学层——模型学的是"图像+语言 → 末端运动意图"，各数据集自己的 de-normalize 再把意图翻译成具体关节命令。

符号化一点：设真实条件分布可分解为 $p_k(b\mid o,\ell)=p_{shared}(b\mid o,\ell)\cdot p_{k,calib}(b\mid o,\ell)$，其中共享部分是任务语义、calibration 部分是各机器人的标定/坐标系差异。混合训练让大容量模型先拟合 $p_{shared}$（数据量大、跨域一致），小容量或单域模型则被 $p_{k,calib}$ 的噪声淹没。这就解释了 Q2：

### 3) 容量 vs 混合熵：RT-1-X 的 underfit

小数据域 $k$（如 NYU Door Opening、Kitchen Manipulation）：单域样本 $n_k$ 小，估计方差 $\propto 1/n_k$；混合训练用共享表示做正则，等效样本量放大 → 平均 +50%（5 个小域中 4 个打败原作者的专用方法）。

大域（Bridge、Google Robot 数据）：混合分布的熵 $H(\mathcal D_{mix})\gg H(\mathcal D_k)$，35M 的 RT-1 容量 $C$ 成为瓶颈——bias-variance 分解里 bias 项下不去，表现为 underfit（Bridge：RT-1-X 27% < 单域 RT-1 40%；RT-1 6 技能：73% < 92%）。换 55B 的 RT-2-X 后容量够了，Bridge 上 50% 反超单域 RT-1 的 40%。

**Scaling 启示**：数据混合的收益是**容量条件**的——先有足够的 $C$ 吸收 $H(\mathcal D_{mix})$，混合才从"噪声"变成"正则"。这也是为什么 RT-X 论文的标题把 Datasets 放在 Models 前面： infra（数据统一）先行，但模型容量必须跟上。

### 4) Emergent skills：联合分布外推 + 移除式因果归因

定义迁移量 $\Delta = \mathrm{succ}(\text{RT-2-X}) - \mathrm{succ}(\text{RT-2})$，在 Google Robot 上评估、但任务来自 Bridge/WidowX 数据集（RT-2 自己的训练数据里没有这些技能/物体）。这是**联合分布外推**：测试点 $(o,\ell)_{test}$ 在机器人 $i$ 的数据支撑之外、在机器人 $j$ 的数据支撑之内。

- $\Delta$：27.3% → 75.8%，约 **3×**。
- 反事实消融：训练混合去掉 Bridge 后 $\Delta$ 坍缩到 42.8%——直接把"WidowX 数据 → Google Robot 技能"这条因果链钉死。这是 Q3 的答案：移除式消融 = do-calculus 的工程近似，$P(\text{skill}\mid do(\text{remove Bridge}))$ 显著下降 ⇒ Bridge 数据是涌现技能的因。
- 配套消融（Table II）：55B vs 5B（75.8% vs 44.4%）说明迁移量随容量单调增；有/无图像历史（44.4% vs 14.5%）说明时序上下文是迁移的载体之一；无 web 预训练从零训（0%）说明 VLM 语义底座不可或缺——和 Day09 RT-2 的结论一致。

### 5) 和系统实现的对应

- **数据 infra**：RLDS 格式（tfrecord 序列化），容纳不同相机数、深度/点云；TFDS 加载；每个数据集选一个规范视角 + 统一分辨率。60 个数据集的元数据和 citation 维护在一个公开 spreadsheet——社区级数据 infra。
- **模型**：RT-1-X = RT-1 架构原样（35M，EfficientNet + USE + FiLM + decoder-only Transformer），唯一变量是数据；RT-2-X = PaLI-X 55B，动作 token 当文本训。
- **部署**：推理 3–10 Hz（按机器人要求），RT-1 本地跑，RT-2 云端服务网络查询——和 Day09 的部署形态一致。
- **评测**：3600 次真实机器人试验、6 种机器人；分"小域"（5 个）和"大域"（Bridge + RT-1）两套口径。

### 6) 假设与数学没有覆盖的真实误差

- 混合权重 $w_k$ 的设计是经验性的（论文只说"robotics data mixture"，未公开逐数据集权重）——配方不可复现，这是 OXE 作为 infra 最大的短板。
- 相机位姿差异没有对齐：模型可能学到"看相机视角 = 认出机器人"的捷径，用 embodiment 指纹代替真正的跨域泛化；论文承认观测差异大（Fig.1）但没有量化这个 confounder。
- RLDS 统一了**格式**，没有统一**质量**：demo 质量、标注噪声、任务难度在 60 个数据集间异构，混合目标把它们当同等可信。
- 量级诚实性：100 万轨迹 vs NLP 的 15–45 亿 token、CV 的 5–18M 图像——论文自己承认机器人数据仍小 2–3 个数量级，"ImageNet 时刻"是方向宣言不是已达状态。
- 评测全是短时程桌面操作（short-horizon manipulation），长程/移动操作/全身控制不在 OXE 的验证范围内——Day14 π₀.₅的"未见家庭长程任务"正是 OXE 没覆盖的下一棒。

## 核心
1. **Motivation**: 每个实验室各自为政训 per-robot 策略，数据孤岛；NLP/CV 靠 web-scale 数据 consolidation 出通用底座，机器人没有 web 可爬。问题：22 种机器人、60 个数据集能不能拼成一个可用的跨机器人预训练底座？和 Physical AGI 的关系：这是"foundation model for robots"的数据地基宣言。
2. **System / Method**: RLDS schema 统一 60 数据集（tfrecord + 规范视角 + 统一分辨率 + 7 维末端动作 + per-dataset 归一化 + 256-bin 离散化）；RT-1-X（35M，15 帧历史，纯机器人数据混合）与 RT-2-X（55B PaLI-X，web:robot ≈ 1:1 co-fine-tuning）；推理 3–10 Hz，RT-1 本地 / RT-2 云端。
3. **Training / Data Details**: 训练混合 = 9 种机械臂数据（RT-1、QT-Opt、Bridge、Task Agnostic Robot Play、Jaco Play、Cable Routing、RoboTurk、NYU VINN、Austin VIOLA、Berkeley Autolab UR5、TOTO、Language Table）；全量 22 embodiment 数据集已公开发布（实验时只有 9 种可用）；Sim 数据：无；Reward：纯行为克隆，无显式 reward。
4. **Key Tricks**（3个最值得抄的）:
   - **粗对齐就够用**：7 维末端动作 + per-dataset 归一化，不统一坐标系、不区分绝对/相对/速度——把 embodiment 差异留给模型吸收，证明迁移的载体是 task-space 共享结构。做多源数据聚合时，先定义统一的 action/observation contract，再谈精细对齐。
   - **小域放大器**：RT-1-X 在 5 个小数据域中的 4 个上打败原作者的专用方法（平均 +50%）——数据混合是数据穷人的免费午餐，单域训不动时先想混合。
   - **Emergent skills + 移除式消融的评估设计**：测"别的机器人数据里的技能"在本机器人上的表现，再做一次去掉该数据源的反事实——把"泛化"从形容词变成可归因的实验。这个方法论 Day14 π₀.₅又用了一次。
5. **Results**: 3600 次真机试验、6 种机器人；小域 RT-1-X 平均 +50%（4/5 域胜过 original method）；大域 RT-1-X 欠拟合（Bridge 27% vs 单域 RT-1 40%），55B RT-2-X 在 Bridge 上 50% 反超；emergent skills：RT-2 27.3% → RT-2-X 75.8%（~3×），去 Bridge 掉到 42.8%，55B vs 5B 为 75.8% vs 44.4%，无历史帧掉到 14.5%，无 web 预训练从零训 0%。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？是——emergent skills 本来就是 held-out 技能/物体的跨机器人评测。模型 vs 框架贡献：**框架贡献占绝对主导**（RT-1-X 架构与 RT-1 完全相同，增益全来自数据混合）。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. Post-training 数据配方的"混合权重设计"优先于模型改动：OXE 证明同样架构换混合就能 +50% / 3×；做 agent/RL infra 时，SFT/RL 数据的来源混合比例是比改 loss 更早锁定的变量。但注意 OXE 没公开 $w_k$——抄作业时要自己设计权重搜索/消融流程，不能抄数字。
  2. "接口统一先于细节对齐"：7 维动作 contract 让 22 种机器人进同一条训练管线；做 agent harness 时，先定义统一的 action/observation schema（tool call 格式、观测序列化），再谈各环境的精细适配——schema 是规模化的前提。
- Infra 视角：RLDS/tfrecord + TFDS 加载 + 数据集 spreadsheet（citation/元数据）是社区级数据 infra 的范本；3600 次真机评测是 eval infra 的成本标杆（仿真评测再多，真机 gate 省不掉）；"数据集先行、模型跟进"的节奏值得学——OXE 发布时实验只用了 9/22 种 embodiment，但 infra 一次到位，后续 Octo、π₀、DROID 全是它的下游。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：训练混合中各数据集的采样权重 $w_k$ 到底怎么定的（按轨迹数？按技能多样性？手动调？）？论文 IV-C 只给了成分列表没给权重——这决定了混合配方的可复现性，想挖的话要去翻 open_x_embodiment 仓库的训练脚本或 RT-1-X colab。
- 如果要复现 / 小规模试，第一个实验做什么？用 Day13 Octo 的开源代码，在 OXE 的 2–3 个小数据集（如 NYU Door Opening + Cable Routing + 一个 Franka 数据集）上做最小混合：单数据集训 vs 混合训对比小域成功率，验证"+50% 小域放大器"在小规模下是否成立；再做一次"去掉某数据集"的移除消融，复现因果归因的方法论。

## 原文金句 (1-2句)
> "We assemble a dataset from 22 different robots collected through a collaboration between 21 institutions, demonstrating 527 skills (160266 tasks)."（摘要原文）
> "RT-2-X outperforms RT-2 by ~3×, suggesting that incorporating data from other robots into the training improves the range of tasks that can be performed even by a robot that already has large amounts of data available."（V-B 原文）

## 今晚产出
- [x] 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- [x] 保留并完善「和之前工作的关系」小节（含 Day09/11/13/14 对比：RT-2→RT-2-X、Octo 作为下游、π₀.₅配方的历史前传）
- [x] 数学视角：多源混合 BC 目标 + per-dataset 归一化/离散化对齐算子 + task-space 共享结构 + 容量-混合熵 underfit + emergent skills 联合分布外推与移除式因果归因
- [ ] 深挖训练混合权重 $w_k$ 的具体设计（疑问/下一步：翻仓库训练脚本）

## 连接
- 上一篇: day-14-2025-pi05-open-world — π₀.₅的异构 co-training 配方，其"移除式成分消融"的方法论源头正是 OXE 的去 Bridge 实验
- 下一篇预告: day-16-2024-droid — DROID：多机构真实家庭/办公场景的 Franka 数据采集体系，看 OXE 之后社区如何继续补"数据多样性"这块短板
