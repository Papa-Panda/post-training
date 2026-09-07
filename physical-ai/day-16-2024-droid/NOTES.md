# Day16 — DROID: in-the-wild 大规模机器人操作数据集

> Alexander Khazatsky*, Karl Pertsch*（project co-leads）等 / Stanford、UC Berkeley 等 13 所机构，arXiv 2403.12945（2024-03-19；RSS 2024）。机器人数据论文的"田野调查"路线：不追求实验室内的轨迹条数，而追求**场景覆盖**——18 台统一 Franka 硬件栈、50 个采集员、12 个月、在 52 栋建筑的 564 个真实场景（家庭/办公室/实验室）里采集 76k 条轨迹 / 350 小时，全部 CC-BY 4.0 开源。

## 元信息
- Title: DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset
- Authors / Org: Alexander Khazatsky*, Karl Pertsch*（co-leads，对应 alexkhaz@stanford.edu / pertsch@berkeley.edu），Jitendra Malik、Roberto Martín-Martín、Subramanian Ramamoorthy、Dorsa Sadigh、Shuran Song、Jiajun Wu、Yuke Zhu、Thomas Kollar、Sergey Levine、Chelsea Finn 等 / Stanford、UC Berkeley、CMU 等 13 所机构（北美/亚洲/欧洲）
- Link / arXiv / Blog: https://export.arxiv.org/abs/2403.12945 ；项目页 https://droid-dataset.github.io （含数据集可视化、训练代码、预训练 checkpoint、硬件复现指南）
- Date read: 2026-09-07
- Tags: [physical-ai, droid, robot-data, in-the-wild, diffusion-policy, data-diversity, co-training, scene-coverage, franka]
- Thread: physical-ai
- Folder: day-16-2024-droid
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-16-2024-droid

## 一句话总结
DROID 把"机器人数据规模化"从 OXE 的**跨身体聚合**转向**跨场景覆盖**：18 台统一 Franka 硬件、50 个采集员在 52 栋建筑的 564 个真实场景采集 76k 条轨迹（350 小时、86 个任务），每条带 3 路同步 RGB + 深度 + 相机标定 + 语言标注；用 diffusion policy 做 co-training（少量本域数据 + DROID），在 6 个任务 × 4 个地点（实验室→办公室→真实家庭）的评测里，相对已有大规模机器人数据集的 SOTA 做法平均提升约 **20%**——证明**场景多样性本身就是可量化的泛化增益**。

## 和之前工作的关系

- 接了哪条线：Day15–18 "Robot Data / Benchmark 扩展"专题的第二棒。OXE（Day15）回答"怎么拼跨机器人的数据底座"，DROID 回答"拼完之后还缺什么"——答案是**真实场景的分布覆盖**。
- 补了哪个短板：OXE 的 60 个数据集大多仍是实验室桌面；RT-2 / OpenVLA（Day09）、π₀（Day11）、Octo（Day13）的"泛化"评测基本没离开过实验室。DROID 把采集点搬进真实家庭和办公室，直接攻击"lab-only 分布"的盲区。
- 替代 / 分叉 / 改进：
  - 相对 Day15 OXE：**分叉**——OXE 是"固定场景多样性、扩展 embodiment 多样性"（22 种身体），DROID 反其道而行之"固定身体（全 Franka）、扩展场景多样性"（564 场景）。两种多样性正交，DROID 证明后者独立可增益。
  - 相对 Day13 Octo：Octo 是 DROID 精神的上游消费者——Octo 的 25 个 OXE 数据集里 lab 数据占主导；DROID 这类 in-the-wild 数据正是 Octo 式多域混合下一步最缺的成分。
  - 相对 Day12 Diffusion Policy：DROID 的全部策略实验**没有发明新算法**，直接用 Day12 的 diffusion policy——这是"数据论文"的诚实姿态：算法固定，唯一变量是数据，+20% 全部归因于数据。
- 对之前 Day X 的直接对比：Day14 π₀.₅的"异构 co-training + 移除式消融"方法论在 DROID 这里有平行版本——DROID 的评测口径是"本域小数据 vs 本域小数据 + DROID"，同样是**加法式归因**（多加一个数据源，测增量），只是方向相反（π₀.₅是做减法钉因果，DROID 是做加法证增益）。

## 为什么今天读它
路线图 Day16–18 是数据专题的纵深：Day15 OXE 给了"聚合"的全景，Day16 DROID 给了"多样性"的第一性原理——**泛化误差 ≈ 测试分布到训练混合支撑的距离**，场景覆盖就是在直接缩小这个距离。读完 DROID 再读 Day17 BridgeData V2（廉价采集扩展）和 Day18 RoboCasa（仿真补数据），三者正好构成"真实场景采集 / 低成本采集 / 仿真合成"的数据三角。不读 DROID，这个三角就缺了"真实世界长尾"那条边。

## 今天的 3 问
1. DROID 固定身体（全 Franka）换来干净统一的动作接口，OXE 则聚合 22 种身体——"固定身体扩场景" vs "固定场景扩身体"，哪种多样性对泛化更值钱？DROID 的实验设计能回答这个问题吗？（不能直接回答：DROID 只在单 embodiment 内证明场景多样性增益；两种多样性的相对价值需要双因素消融实验。）
2. 564 个场景、平均每场景约 135 条轨迹的长尾结构：per-scene 样本这么少，策略到底从"每个场景只看一百多遍"里学到了什么可迁移的东西？这和 Day15 的"小域 +50% 放大器"是同一种机制吗？
3. 采集协议是 free-form、50 个采集员风格各异 → 同一任务的行为分布天然多峰。为什么这恰好是 diffusion policy（Day12）的舒适区、而不是单峰回归的舒适区？从"数据分布形状 → 算法归纳偏置匹配"的角度怎么形式化？

## 数学视角：分布覆盖 × 混合权重——in-the-wild 数据为什么能提高泛化

DROID 是数据论文，没有新模型；它的数学就是**数据混合与分布覆盖的数学**。统一框架：行为克隆的数据混合目标 + 场景覆盖的泛化界直觉。

### 1) State / observation / action / objective

- **Observation**： $O=(I^{(1:3)}_{t-T_o+1:t},\,q_t)$ ——3 路同步 RGB（两路外部立体相机 + 一路腕部相机，带外参标定）共 $T_o$ 帧历史 + 本体感知 $q_t$ （Franka 关节角/末端位姿）+ 语言指令 $\ell$ 。
- **Action**：Franka 7 自由度关节或末端位姿命令的动作块 $A=(a_t,\dots,a_{t+T_p-1})$ ——因为**硬件栈全网统一**，DROID 不需要 OXE 式的 per-dataset 归一化/离散化对齐算子，动作接口天然对齐。
- **训练目标**（co-training 混合，沿用 Day12 diffusion loss，算法固定）：
$$\mathcal{L}(\theta)=\alpha\,\mathbb{E}_{(O,A)\sim\mathcal{D}_{in}}\!\big[\|\epsilon-\epsilon_\theta(O,A^k,k)\|_2^2\big]+(1-\alpha)\,\mathbb{E}_{(O,A)\sim\mathcal{D}_{droid}}\!\big[\|\epsilon-\epsilon_\theta(O,A^k,k)\|_2^2\big].$$
符号： $\mathcal{D}_{in}$ 是评测任务的少量本域数据（task-relevant 但场景窄）； $\mathcal{D}_{droid}$ 是 76k 条 in-the-wild 数据（task-irrelevant 居多但场景宽）； $\alpha\in(0,1)$ 是混合权重； $k$ 是扩散时间步； $\epsilon_\theta$ 是 Day12 的噪声预测器。对应到实现：训练 batch 按比例从两个数据源采样，网络结构与 Day12 完全相同——**数据是唯一的自变量**。

### 2) 场景覆盖的泛化直觉：混合支撑越宽，测试点离得越近

把每个场景 $s$ 看成一个观测-动作联合分布 $P_s(o,a)$ 。DROID 的训练分布是 564 个场景的混合：
$$P_{train}=\sum_{s=1}^{564} w_s\,P_s,\qquad w_s\propto n_s\;(\text{该场景轨迹数}).$$
部署场景 $s'$ 的泛化误差可以直觉地拆成：
$$\mathrm{err}(s')\;\lesssim\;\underbrace{d\big(P_{s'},\,\mathrm{supp}(P_{train})\big)}_{\text{分布外推距离}}+\underbrace{\lambda_{s'}}_{\text{场景内在难度}}+\text{有限样本方差}}.$$
DROID 的赌注是第一项主导：实验室数据 $P_{train}^{lab}$ 的支撑是几个桌面，真实家庭场景 $s'$ 离这个支撑很远；而 564 场景的混合把支撑铺开（52 栋建筑、家庭/办公室/实验室），让未见场景 $s'$ 更可能落在"某个已见场景的邻域"里。实验验证：在 4 个地点（lab → office → 真实家庭）的 6 个任务上，本域小数据 + DROID 相对已有大机器人数据集的 SOTA 做法平均 **+20%**——增益随部署场景与 lab 的距离单调变大，正是"外推距离缩小"的预测。

**数量 vs 多样性的取舍**：76k 条 / 564 场景 ⇒ 平均每场景约 135 条轨迹——DROID 选择了**支撑扩张**而非**单点加密**。直觉：把第 136 条轨迹加进已见场景，只能降低该场景的估计方差（ $\propto 1/n_s$ ）；而把这条轨迹放到一个新场景，是在支撑上新增一个点，直接缩小最坏情况下的 $d(P_{s'},\mathrm{supp})$ 。这是 Q2 的答案：策略从每个场景学到的不是该场景的精细动力学，而是"场景可变"的**不变性**——背景、光照、桌面纹理在 564 个场景里被随机化掉了。

### 3) 去伪相关：场景随机化打破背景捷径

单场景数据里，背景特征 $B$ （桌面颜色、光照）与正确动作 $A$ 高度相关，策略会学捷径 $B\to A$ 。564 个场景下，背景与任务的联合分布被随机化：
$$\mathrm{Cov}_{s\sim\{1..564\}}(B_s,\,A)\approx 0,$$
捷径失效，模型被迫用真正因果的视觉特征（物体位姿、抓取点）。这是 DROID "robustness" 增益的数学实质，也是 Day21 domain randomization（仿真端随机化）的真实数据版本——**DROID = 物理世界里的 domain randomization**，只是随机化的是真实场景而不是仿真参数。

### 4) 采集员方差与多峰行为分布：为什么配 diffusion policy

50 个采集员、free-form 任务 ⇒ 同一 $(O,\ell)$ 下的行为分布 $p(A\mid O,\ell)$ 天然多峰（不同人的抓取习惯、路径偏好）。单峰回归（MSE）会学到多峰的均值——均值动作往往是非法动作（比如两条绕行路径的平均穿过障碍物）；而 Day12 的 diffusion head 直接拟合完整条件分布 $p_\theta(A\mid O)$ ，多峰是它的舒适区。这是 Q3 的答案：**数据分布的形状决定了算法的归纳偏置需求**——DROID 的采集协议（多样性优先）与 diffusion policy（多峰表达）是互相成就的一对；反过来，如果数据是单采集员精修的，MSE 回归可能更 sample-efficient。

### 5) 和系统实现的对应

- **硬件 infra**：18 台 Franka Panda 全网统一硬件栈 + 开源复现指南——用"固定身体"换"数据可比性"，这是和 OXE 最根本的 infra 哲学分歧。
- **数据 infra**：3 路同步 RGB + 深度 + 相机外参标定 + 语言标注；全量 CC-BY 4.0 开源；带交互式可视化浏览器；跨 52 栋建筑的**自动相机标定**流程（附录 G，质量评估 + 自动重标定）——分布式采集的标定 infra 是论文最被低估的工程贡献。
- **采集协议**：50 个采集员、12 个月、free-form 任务 prompt + 语言标注；场景类型分类（附录 C）与去重（附录 D）保证 564 场景的"场景"是真实不同的物理地点。
- **评测**：6 个任务 × 4 个地点，diffusion policy，batch 按 $\alpha$ 混合采样（附录 F-B）；口径是"本域小数据 + DROID" vs "本域小数据 + 其他大机器人数据集"——**加法式归因**。

### 6) 假设与数学没有覆盖的真实误差

- 50 个采集员的**风格方差**是双刃剑：带来多峰覆盖，也带来质量长尾——demo 里混入次优/失败片段，行为克隆会原样复制（BC 没有 reward 过滤）；论文未量化采集员间的质量分布。
- 跨 52 栋建筑的自动标定（附录 G）有残差：外参误差直接污染"图像 → 动作"的几何对应，而 diffusion policy 没有显式的几何一致性约束来吸收它。
- 564 场景的"多样性"用的是元数据分类（场景类型、建筑 ID），不是视觉分布散度的度量——两个"不同建筑"的厨房可能视觉上很接近，真实的有效覆盖可能小于 564。
- +20% 是 6 个任务的平均：全是短时程桌面操作；embodiment 仍是单一 Franka，**跨身体泛化不在验证范围内**（OXE 的地盘）；长程/移动操作未覆盖。
- 长尾的另一面：平均每场景 135 条轨迹里，86 个任务的分布也不均匀——稀有任务的 per-task 样本可能不足以支撑可靠的条件分布估计。

## 核心
1. **Motivation**: 最通用的机器人策略仍训在少数实验室环境里，场景/任务多样性不足；采集真实场景数据有物流、安全、人力成本三重门槛。问题：能不能组织一次跨机构的分布式采集，把"真实世界"本身变成数据集？和 Physical AGI 的关系：这是"数据飞轮"里最贵的一环——真实世界交互数据的规模化采集方法论。
2. **System / Method**: 统一 Franka 硬件栈 + 开源复现指南；50 采集员 × 18 台机器人 × 13 机构 × 12 个月；每条轨迹 3 路同步 RGB + 深度 + 相机外参 + 语言标注；场景去重与类型分类保证 564 场景真实不同；策略端零创新——全部实验用 Day12 diffusion policy，数据是唯一变量。
3. **Training / Data Details**: 76k 轨迹 / 350 小时 / 564 场景 / 86 任务 / 52 栋建筑；CC-BY 4.0 全开源 + 可视化 + 训练代码 + 预训练 checkpoint；训练 = 本域小数据 + DROID 按权重 $\alpha$ 混合采 batch；Sim 数据：无；Reward：纯行为克隆。
4. **Key Tricks**（3个最值得抄的）:
   - **固定身体、扩张场景**：和 OXE 反向的取舍——18 台统一 Franka 让动作接口天然对齐，省掉跨 embodiment 对齐算子，把全部复杂度预算花在场景覆盖上。做数据 infra 时先想清楚：你要的多样性是哪个轴的？一次只扩张一个轴。
   - **分布式采集的标定 infra**：52 栋建筑的自动相机-机器人外参标定 + 质量评估（附录 G）——真实世界数据规模化的瓶颈不在采集员，在标定和质检的自动化。
   - **加法式归因的评测设计**：固定算法（diffusion policy）、固定本域小数据，唯一变量是"加不加 DROID"、"加的是 DROID 还是别的大数据集"——把数据论文的 claim 做成干净的对照实验。
5. **Results**: 6 个任务 × 4 个地点（lab → office → 真实家庭）；本域小数据 + DROID 相对"本域小数据 + 已有大规模机器人数据集"的 SOTA 做法，性能/鲁棒性/泛化性平均 **+20%**；增益随部署场景与实验室的距离增大而变大。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？是——4 个评测地点里有办公室和真实家庭，本来就是 held-out 场景；+20% 是跨场景泛化的直接证据。模型 vs 框架贡献：**框架（数据）贡献 100%**——算法与 baseline 完全相同，增益全部来自数据混合。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. Post-training 数据配方的"覆盖轴"思维：DROID 证明**支撑扩张 > 单点加密**——做 agent RL 数据时，与其把同一任务刷 10k 遍，不如把任务×环境的笛卡尔积铺开；评测也要按"与训练分布的距离"分桶，而不是只报平均成功率。
  2. "数据分布形状 → 算法归纳偏置"的匹配意识：多采集员 free-form 数据天然多峰，配 diffusion/flow 这类分布拟合器；单采集员精修数据配 MSE 回归更省样本。选算法前先看数据的模态结构——这是 Day12 数学在数据侧的回声。
- Infra 视角：统一硬件栈 + 开源复现指南 + 自动标定 + 场景去重 = 分布式真实数据采集的 infra 四件套；CC-BY 4.0 + 可视化浏览器 + 预训练 checkpoint 是数据集发布的黄金标准（对比 OXE 没公开混合权重 $w_k$ 的短板）；评测成本锚点：6 任务 × 4 地点的真机评测是"数据论文"的 eval 预算下限。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：混合权重 $\alpha$ （本域小数据 vs DROID 的 batch 比例）到底怎么定的？附录 F-B 给了 batch 构造但没给 $\alpha$ 的搜索过程—— $\alpha$ 太大 DROID 被稀释、太小本域任务信号被淹没，这个 trade-off 的 scaling 规律值得挖（和 Day15 的 $w_k$ 问题是同一个：数据混合论文都不爱公开权重）。
- 如果要复现 / 小规模试，第一个实验做什么？拿 DROID 开源数据里 2–3 个场景子集 + 一个本地 Franka（或仿真）小任务：固定 diffusion policy，对比"本域数据 only" vs "本域 + DROID 子集（按场景数递增：10 / 50 / 200 场景）"的成功率曲线——验证"增益随场景数单调增、随单场景条数快速饱和"的预测，复现"支撑扩张 > 单点加密"。

## 原文金句 (1-2句)
> "We introduce DROID (Distributed Robot Interaction Dataset), an "in-the-wild" robot manipulation dataset with 76k trajectories or 350 hours of interaction data, collected across 564 scenes, 86 tasks, and 52 buildings over the course of 12 months."（arXiv 页 Fig.1 caption 原文）
> "we find that DROID boosts policy performance, robustness and generalizability by 20% on average over state-of-the-art approaches that leverage existing large-scale robot manipulation datasets"（arXiv HTML v2 实验结论原文）

## 今晚产出
- [x] 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- [x] 保留并完善「和之前工作的关系」小节（含 Day09/12/13/14/15 对比：OXE 反向取舍、diffusion policy 算法固定、π₀.₅加法/减法归因对照）
- [x] 数学视角：co-training 混合目标 + 场景混合支撑的泛化界直觉 + 支撑扩张vs单点加密 + 去伪相关（物理世界版 domain randomization）+ 采集员方差与多峰分布-算法匹配
- [ ] 深挖混合权重 $\alpha$ 的搜索过程（疑问/下一步：翻附录 F-B 与开源训练代码）

## 连接
- 上一篇: day-15-2023-open-x-embodiment-rtx — OXE 的"跨身体聚合" vs DROID 的"跨场景覆盖"：两种多样性正交，DROID 是 OXE 之后社区补"真实场景分布"那块短板的第一块砖
- 下一篇预告: day-17-2023-bridgedata-v2 — BridgeData V2：廉价遥操作与异构场景扩展 imitation data，看"低成本采集"这条轴怎么走

## 问答补充
（本篇暂无用户主动提问；跨篇通识问答见 README「问答记录」。）
