# Day 20 — RLPD: Efficient Online Reinforcement Learning with Offline Data

## 元信息
- Title: Efficient Online Reinforcement Learning with Offline Data (RLPD — Reinforcement Learning with Prior Data)
- Authors / Org: Philip J. Ball*, Laura Smith*, Ilya Kostrikov* (共同一作), Sergey Levine / UC Berkeley
- Link / arXiv / Blog: https://arxiv.org/abs/2302.02948 (ICML 2023, v4 2023-05-31)
- Date read: 2026-09-11
- Tags: [physical-ai, rlpd, offline-to-online, reinforcement-learning, actor-critic, sac, sample-efficiency, real-world-rl]
- Thread: physical-ai
- Folder: day-20-2023-rlpd
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-20-2023-rlpd
- Official code: https://github.com/ikostrikov/rlpd

## 一句话总结
真机上每一步交互都贵（时间/安全/硬件磨损），而离线机器人数据相对便宜——RLPD 问：**能不能不碰复杂的离线 RL 算法，直接用现成的 off-policy 方法（SAC），把离线数据灌进 replay buffer 里在线训？** 答案是"可以，但需要三个最小却关键的改动"：**50/50 对称采样（离线/在线各半）+ critic 加 LayerNorm + 高 UTD（20）配合 Q 集成**；这套 recipe 在 D4RL / Adroit / AntMaze / V-D4RL 上平均 **2.5×** 领先现有 offline-to-online 方法，且零额外计算开销。它是 Day15–18 那些离线机器人数据集（OXE / DROID / BridgeData）通向**真机在线 RL** 的 canonical 桥梁。

## 和之前工作的关系

- **接了哪条线**：接 Day19（PPO，on-policy RL 的算法基座），同时承接 Day15–18（机器人离线数据：OpenX / DROID / BridgeData V2 / RoboCasa）——Day19 给的是"在仿真里无限交互"的方案，Day20 给的是"交互昂贵时怎么把离线数据榨干"的方案，两条线合起来才覆盖 RL for Robotics 的完整光谱。
- **补了哪个短板**：Day15–18 的数据全是在"离线模仿"（BC / offline RL）语义下读的；但 Day07/08 暴露的问题是——**纯 BC/离线只能复制示范，不能超越示范**。RLPD 补的是"如何从离线数据起步、用在线交互超越它"，而且是**最小改动**的配方，不需要 CQL/IQL 那类保守离线算法做预训练。
- **替代 / 分叉 / 改进**：改进并部分替代"先 IQL/CQL 离线预训练、再在线微调"的两阶段范式。论文的核心实验发现：**悲观（pessimistic）的离线 RL 初始化，在转入在线阶段反而会坍缩性能**——离线阶段学到的保守 Q 函数和在线阶段的数据分布不匹配，初期更新会把好不容易学到的行为洗掉。RLPD 干脆**从零开始训 plain SAC**，离线数据只作为 replay buffer 里的一部分存在，全程在线+离线混合更新。这是一个"少即是多"的论证：离线 RL 的复杂性不是必需品。
- **对之前 Day X 的直接对比**：
  - **vs Day19（PPO）**：PPO 是 on-policy——数据必须来自当前策略，用完即弃，样本效率低但稳定，靠 GPU 并行 rollout（legged_gym 分钟级）解决"数据贵"问题；RLPD 是 off-policy（SAC 基座）——replay buffer 里的数据可以反复用，配合 UTD=20 把每一步交互榨出 20 次梯度更新，样本效率高。**同一条轴（RL），两端点：PPO 赌"仿真无限便宜"，RLPD 赌"真机交互极贵"**。真机 locomotion（Day07/08 的 H1）走 PPO+sim2real 路线；真机灵巧操作（奖励稀疏、仿真难建模接触）更适合 RLPD+离线数据路线。
  - **vs Day06（DreamerV3）**：两者都是 actor-critic off-policy，但 DreamerV3 用**学到的世界模型**做 imagined rollout 来省真实交互；RLPD 用**离线数据集**来省真实交互——"模型"和"数据"是省交互的两条路。Day20 的视角：当你有大量离线机器人数据（Days 15–18）时，数据这条路更直接。
  - **vs Day09–14（VLA，BC-MLE）**：那四篇学的是示范分布 $p_\theta(A\mid O,c)$（固定数据集上的最大似然）；RLPD 学的是**奖励最优策略**（Bellman 备份 + 策略改进），离线数据只提供初始覆盖。数学上：BC 是"拟合 $p_{\text{demo}}$"，RLPD 是"在 $D_{\text{offline}}\cup D_{\text{online}}$ 上解 Bellman 最优性"。这也是为什么 RL 能超越示范上限，而 BC 不能。
  - **vs Day17（BridgeData V2）**：Day17 的结论是"同一套离线数据能跑通 6 种 offline 方法"；Day20 把结论推进一步——**你甚至不需要专门的 offline 方法**，plain SAC + 配方就能把这套数据用起来，并且在线超越。

## 为什么今天读它

Roadmap 把 Day20 放在"RL for Robotics / Sim2Real"块的入口，标题明确是"RLPD — sample-efficient real-world robot RL with prior data"。它是 Day19 PPO 的**对偶**：Day19 解决"仿真便宜时怎么训"，Day20 解决"真机交互贵时怎么训"。对 Jun 的 Physical AI 路线而言，它的战略意义在于：**它把 Days 15–18 的离线数据资产（OXE、DROID、Bridge）变成了一种可直接消费的在线 RL 配方**——不需要再训一个专门的 offline RL 模型，数据飞轮的"recollect → retrain"环节（Day30 预告）可以直接用这套方法闭环。

## 今天的 3 问
1. 朴素地把离线数据倒进 SAC 的 replay buffer 再在线训，为什么会失败？（提示：Q 函数在 OOD $(s,a)$ 上的外推误差 + Bellman 备份的反复放大 + 高 UTD 下的价值发散。）
2. 三个 recipe（50/50 对称采样、critic 加 LayerNorm、高 UTD + Q 集成）各自在数学上解决了什么问题？为什么论文说 ensemble 是稀疏奖励任务上最强的正则化形式（Figure 9，强于 dropout 和 weight decay）？
3. RLPD 和"先 IQL/CQL 离线预训练、再在线微调"相比，少了什么、多出了什么？为什么它敢说"不需要复杂的离线 RL 算法"？（提示：悲观初始化的在线坍缩。）

## 核心
1. **Motivation**: Online RL 的两大痛点是**样本效率**和**探索**——在真机上，这两个痛点直接翻译成"钱和时间"。缓解办法之一是利用离线数据（人类专家轨迹、次优探索策略的旧数据）。此前的 offline-to-online 方法（AWAC、IQL+finetuning、Cal-QL、CQL+finetuning、JSRL 等）都引入了大量额外复杂性：保守正则、两阶段切换、行为约束……RLPD 问了一个工程上最诚实的问题：**能不能直接用现成的 off-policy 算法（SAC），只做最小改动？** 摘要原话："can we simply apply existing off-policy methods to leverage offline data when learning online?" 与 Physical AGI 的关系：它是"让真机从试错里学"的**最低门槛工程路径**——不需要先训一个保守的 offline 模型，离线数据（正是 Days 15–18 攒下的资产）直接变成在线训练的燃料。
2. **System / Method**: 基座 = **Soft Actor-Critic（SAC）**，从零开始训练（无离线预训练阶段）。三个最小改动构成完整 recipe：
   - **Symmetric sampling（对称采样）**：离线数据在训练开始前就预装进 replay buffer；之后每个 minibatch 一半采自离线数据、一半采自在线数据（50/50）。关键细节：**不是**把离线数据一次性倒进 buffer 就不管了（buffer 初始化派会随训练被在线数据稀释），而是**全程**保持 50/50 混合——离线数据是"常驻燃料"。
   - **LayerNorm in critic**：Q 网络的每个隐藏层后加 LayerNorm。作用是**钳住 Q 函数在分布外 $(s,a)$ 上的外推**——没有它，高 UTD 下价值会发散（见数学视角）。
   - **高 UTD + critic ensemble**：UTD（update-to-data ratio，每采集一步做多少次梯度更新）= 20（官方 config：locomotion / AntMaze / Adroit 均为 `--utd_ratio=20`），配合 REDQ 风格的 Q 集成（多个 critic，target 用随机子集取 min）做正则化。高 UTD 把每步昂贵交互的梯度利用率放大 20 倍；ensemble + LayerNorm 保证这 20 倍更新不炸。
   - 没有 CQL 式的保守惩罚、没有 IQL 的 expectile、没有行为克隆约束——**悲观主义被有意删掉了**，因为在线数据会自然纠正外推误差（在线交互是"免费"的真相来源）。
3. **Training / Data Details**: Sim 数据 = D4RL locomotion（halfcheetah/walker2d 等 expert/medium 数据集）、Adroit（稀疏奖励灵巧手：pen-binary 等）、AntMaze（稀疏导航，6 个任务平均）、V-D4RL（像素输入，64px）。Real 数据 = 论文本体实验**没有**真机结果（诚实标注）——但同作者的后续工作（RSS 2023 "Learning to Walk in 20 Minutes"，Smith/Kostrikov/Levine）把同类配方搬上了真机四足，20 分钟学会走路，证明了这条路线的真机可行性。Sim2Real = **不适用**（RLPD 本来就是为"直接在真机/昂贵环境上在线学"设计的；仿真在这里的角色是 benchmark，不是训练场）。Reward = 各 benchmark 自带（Adroit/AntMaze 稀疏二值、locomotion 密集）；**verifiable signal**：论文强调方法对"少量专家演示"和"大量次优轨迹"两种离线数据都鲁棒——离线数据的质量轴被显式消融过。
4. **Key Tricks**: ① **50/50 全程混合而非一次性预装**：消融（Figure 12）显示采样比例不敏感，但 50% 是"方差/收敛速度/渐近性能"的最佳折中；25% 离线能略微提升渐近性能但方差大。工程直觉：离线数据提供**覆盖**（coverage，见过好的 $(s,a)$），在线数据提供**纠偏**（在线 Bellman 备份修正外推），两者缺一不可。② **LayerNorm 放在 critic 而不是 actor**：价值发散是 critic 的病（Bellman 备份反复放大外推误差），actor 只是跟着 Q 走；LN 钳住的是 Q 的外推幅度。这解释了为什么 dropout/weight decay 不如 ensemble+LN——后者直接约束了**函数在 OOD 区域的输出尺度**。③ **UTD=20 的"交互杠杆"**：真机上 1 步交互 ≈ 1 次梯度更新（UTD=1）是浪费；UTD=20 意味着 300k 环境步 = 600 万次梯度更新——**把"数据贵"转化成"计算换数据"**。这是 Infra 视角最值得抄的一点：当 env step 是瓶颈时，优化目标从 wall-clock 吞吐转向"每步交互的梯度利用率"。
5. **Results**: 官方 claim（摘要原文）："correct application of these simple recommendations can provide a **2.5× improvement** over existing approaches across a diverse set of competitive benchmarks, with **no additional computational overhead**." 具体：D4RL AntMaze 6 任务平均（Figure 1，10 seeds），RLPD 在 300k 步内收敛，显著超过 IQL+finetuning 和"SAC+离线数据"朴素基线；Adroit 稀疏任务上 ensemble 正则化是关键；V-D4RL 像素任务上同样成立。预算：locomotion 250k 步、AntMaze 300k 步、Adroit 1M 步（官方 README 命令行原文）。**社区复现注**：GitHub 上有多个独立复现（如 `karan-anchan/rlpd-offline-to-online-rl` 把 RLPD 扩展到 Humanoid-v5，3 seeds × 3 方法 245k 步，确认了 symmetric ratio / LayerNorm / ensemble / UTD 的消融结论）——配方是可复现的，不是论文运气。

## 数学视角

> 符号统一（全文通用）：$s$ 状态，$a$ 动作，$r(s,a)$ 标量奖励，$\gamma\in(0,1)$ 折扣因子，$\pi_\theta(a\mid s)$ 随机策略，$Q_\phi(s,a)$ critic（价值网络），$\bar Q$ 为 target 网络（慢更新），$\mathcal{D}_{\text{off}}$ 离线数据集（来自行为策略 $\mu_1,\dots,\mu_k$，可能是专家也可能是次优），$\mathcal{D}_{\text{on}}$ 在线 replay buffer（当前及历史 $\pi_\theta$ 采集），$\alpha>0$ 为 SAC 的熵温度系数，$B$ 为 minibatch 大小，$u$ 为 UTD（update-to-data ratio）。

### 1. 基座：SAC 的熵正则 Bellman 备份

SAC 最小化 soft Bellman 残差（critic）并用熵正则策略改进（actor）：

$$y = r + \gamma\,\mathbb{E}_{s'\sim p,\,a'\sim\pi_\theta(\cdot\mid s')}\big[\bar Q(s',a') - \alpha\log\pi_\theta(a'\mid s')\big],$$

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{B}}\big[\big(Q_\phi(s,a)-y\big)^2\big],\qquad \mathcal{L}_\pi(\theta)=\mathbb{E}_{s\sim\mathcal{B},\,a\sim\pi_\theta}\big[\alpha\log\pi_\theta(a\mid s)-Q_\phi(s,a)\big].$$

直觉：critic 学"从 $(s,a)$ 出发未来能拿多少（含探索奖励 $-\alpha\log\pi$）"，actor 朝"高 Q + 高熵"的方向挪。$\mathcal{B}$ 是从哪采的 batch——**这就是 RLPD 全部的创新所在**。

### 2. 对称采样：把"覆盖"写进经验分布（回答 Q1 的一半）

朴素做法（"SAC + 离线数据"基线）：把 $\mathcal{D}_{\text{off}}$ 一次性倒进 replay buffer，之后按均匀采样训练。问题：随着在线数据涌入，离线数据的采样占比被稀释到 $\to 0$——**离线数据只在训练初期有用**，而恰恰在初期策略最差、最需要示范覆盖的时候，Q 的外推误差最大。

RLPD 的对称采样把 batch 的经验分布固定为混合分布：

$$\mathcal{B} = \mathcal{B}_{\text{off}}\cup\mathcal{B}_{\text{on}},\quad \mathcal{B}_{\text{off}}\sim\mathcal{D}_{\text{off}},\ \mathcal{B}_{\text{on}}\sim\mathcal{D}_{\text{on}},\quad |\mathcal{B}_{\text{off}}|=|\mathcal{B}_{\text{on}}|=B/2,$$

即 critic loss 的期望拆成两项：

$$\mathcal{L}_Q = \tfrac{1}{2}\underbrace{\mathbb{E}_{\mathcal{D}_{\text{off}}}[(Q-y)^2]}_{\text{离线：好的 }(s,a)\text{ 上把 Q 钉住}}+\tfrac{1}{2}\underbrace{\mathbb{E}_{\mathcal{D}_{\text{on}}}[(Q-y)^2]}_{\text{在线：当前策略分布上纠偏}}.$$

直觉：离线项是"锚"——在示范覆盖过的 $(s,a)$ 上，Bellman 备份有真实 $r$ 监督，Q 学得准；在线项是"探针"——在当前策略实际到达的 $(s,a)$ 上，用真实环境回报纠正外推。两者全程 50/50，Q 函数在"见过的好区域 $\cup$ 正在探索的区域"的并集上始终有监督——这就是**覆盖即泛化**（呼应 Day16 DROID 的"场景覆盖即泛化增益"，同一思想在 RL 里的版本）。

### 3. 为什么需要 LayerNorm + ensemble：高 UTD 下的价值发散（回答 Q1 的另一半 + Q2）

UTD $u$ = 每采集 1 步环境交互，做 $u$ 次梯度更新。$u=20$ 意味着同一批 $(s,a,r,s')$ 被 Bellman 备份"咀嚼" 20 遍。危险在于：**Bellman 备份是压缩映射只在"有数据支撑"的区域成立**；在 OOD $(s,a)$ 上，$Q_\phi$ 的输出是网络外推——一次备份把外推误差写进 target，下一次备份再放大，形成正反馈：

$$Q^{(k+1)} \leftarrow \mathcal{T}Q^{(k)},\qquad \text{误差}_{k+1}\approx \gamma\cdot\text{误差}_k + \text{外推噪声}_k,$$

$u$ 越大，单位交互内的备份次数越多，发散越快（这就是 Day19  noted 的"高 UTD 下价值发散"现象，RLC 2024 的 dissecting UTD 论文有系统分析）。

LayerNorm 的作用：$Q_\phi$ 每个隐藏层后做 $\text{LN}(h)=\frac{h-\mu}{\sigma}\odot g + b$，把激活的尺度钳住——**外推时网络输出的幅度被归一化结构天然压住**，$Q(s,a_{\text{OOD}})$ 不会飙到荒谬的值，target $y$ 的噪声项被压住。Ensemble（$E$ 个 critic，target 取随机子集的 min）的作用：$\min$ 操作是**悲观的方向**但只用在 target 计算里、不污染表示——多个独立初始化的 critic 在 OOD 区域分歧大，取 min 相当于"用分歧做不确定性惩罚"，且论文消融（Figure 9）显示它在稀疏奖励任务上强于 dropout/weight decay：因为后两者是"全局收缩"，而 ensemble-min 是"**只在不确定的地方收缩**"——稀疏奖励下大部分 $(s,a)$ 的 $r=0$，Q 全靠备份传播，全局收缩会把好不容易传过去的价值信号也压没。

一句话：**LN 钳住外推的"幅度"，ensemble-min 钳住外推的"方向"（往悲观走），两者合起来才撑得住 UTD=20 的反复备份。**

### 4. 为什么"不需要复杂离线 RL"（回答 Q3）

IQL/CQL 类方法的核心是**悲观**：在离线数据上学一个"对 OOD 动作打压"的 Q（CQL 显式加 $\log\sum_a\exp Q - \mathbb{E}_{\text{data}}[Q]$ 惩罚）。转在线微调时，策略已经能到达当初被打压的区域，但 Q 还记着"那里很危险"——**悲观先验和在线数据分布不匹配**，初期在线更新会先把 Q 的悲观洗掉，性能**先坍缩再回升**（论文称之为 fine-tuning 的不稳定性，也是 Cal-QL 等后续工作试图修的）。

RLPD 的论证是反直觉但干净的：**悲观是离线 setting 的拐杖，在线 setting 里"真实环境"本身就是最好的纠偏器**。从零开始的 SAC + 对称采样，Q 从一开始就在混合分布上被 Bellman 备份塑造，没有"先悲观、再洗掉"的两阶段撕裂。数学上：RLPD 优化的始终是同一个目标（混合分布上的 soft Bellman 残差），而"预训练+微调"优化的是两个不一致的目标（离线悲观目标 → 在线目标）——**目标一致性**是 RLPD 稳定的深层原因。

### 5. 统一框架：BC-MLE vs on-policy-PG vs off-policy-AC 的三轴

三天连起来（Day12/19/20）正好是机器人策略学习的三条正交轴，统一在"**数据从哪来、目标是什么**"：

| | Day12 Diffusion Policy | Day19 PPO | Day20 RLPD |
|---|---|---|---|
| 数据 | 固定示范 $\mathcal{D}$（死的） | 当前策略 rollout（活的，用完弃） | $\mathcal{D}_{\text{off}}\cup\mathcal{D}_{\text{on}}$（混合，可复用） |
| 目标 | $\max\mathbb{E}_{\mathcal{D}}[\log p_\theta(A\mid O)]$（拟合示范） | $\max\mathbb{E}_{\pi_\theta}[\sum\gamma^t r_t]$（on-policy PG） | 解 soft Bellman 最优性（off-policy AC） |
| 动作表示 | 连续多峰（扩散） | 高斯随机策略（探索用） | 高斯随机策略 + 熵正则 |
| 能超越示范吗 | 不能 | 能（但需要 reward + 大量交互） | 能（离线起步 + 在线超越） |
| 瓶颈 | 数据质量 | 交互成本（仿真便宜则无碍） | 价值发散（用 LN+ensemble+UTD 治） |

直觉：**BC 回答"人怎么做"，PPO 回答"在无限试错里怎么做最好"，RLPD 回答"只有少量试错机会、但有前人数据时怎么做最好"**——第三个问题正是真机人的问题。

### 6. 数学没有覆盖的东西（诚实标注）

- **奖励从哪来**：Adroit/AntMaze 的稀疏奖励是 benchmark 设计好的；真机上 reward engineering（稠密 shaping vs 稀疏 0/1）是另一个大学问，RLPD 的数学对此沉默——稀疏奖励下 ensemble 再强也需要"偶然成功"来点燃第一把火。
- **探索的安全性**：在线交互假设"试错无代价"，真机上一次摔倒可能损坏硬件——论文的数学框架里没有约束/安全项（那是 Day29 safe robot learning 的主题）。
- **跨 embodiment 的离线数据**：对称采样假设 $\mathcal{D}_{\text{off}}$ 和当前机器人的 $(s,a)$ 空间一致；Day15（OXE 22 种机器人）的数据直接灌进某个具体机器人的 buffer，在数学上是未定义的——跨身体的 offline-to-online 仍是开放问题。
- **UTD=20 的计算账**：600 万次梯度更新 vs 30 万步交互——"计算换数据"在 GPU 上划算，但 target 网络、ensemble 的前向/反向开销是真实存在的；论文说"no additional computational overhead"是相对"其他 offline-to-online 方法"而言，不是相对 plain SAC。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？**框架贡献更大**：RLPD 的卖点恰恰是"不依赖特定模型结构"——三个改动（采样/LN/UTD）是**算法层面的 recipe**，可以装到任何 off-policy AC 上。社区复现（Humanoid-v5 扩展）证实了可迁移性。Transfer 的边界：稀疏奖励任务最吃 ensemble，密集奖励 locomotion 主要吃 UTD——recipe 的各组分在不同任务上的权重不同，论文的消融表就是这张"transfer 地图"。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. **"离线数据是燃料，不是预训练"**：做 VLA/策略的 RL 后训练时，别急着搞复杂的 offline RL 预训练管线——先试试把高质量离线轨迹常驻 replay（或 SFT 数据常驻混合），用在线信号纠偏。RLPD 的"目标一致性"论证对 LLM 的 RLHF/RLVR 同样成立：SFT-then-RL 的两阶段撕裂 vs 混合训练，值得对照思考。
  2. **UTD 思维 = "每单位昂贵资源的梯度利用率"**：在 post-training 里昂贵的是"高质量 rollout/人类反馈"，RLPD 把 $u$ 从 1 拉到 20 的思路，对应"同一批 rollout 做多次更新"——PPO 的多 epoch 复用（Day19）是同一思想在 on-policy 下的版本。三天连起来：**昂贵资源（交互/反馈）的复用率，是 RL 工程的第一性原理**。
- Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：① UTD=20 把 env-step 瓶颈转化为 GPU 计算瓶颈——Infra 优化目标从"提高仿真吞吐"转向"提高每步交互的梯度吞吐"，profiling 的重点变了；② 官方 JAX 实现 + 社区 PyTorch 复现双轨并存，reproducibility 好（config 即 recipe：`--utd_ratio=20 --config.num_min_qs=1` 这类命令行就是论文的"可执行摘要"）；③ 评测自动化：D4RL/AntMaze/Adroit/V-D4RL 四套 benchmark 全是标准 API，offline-to-online 的评测矩阵是现成的——对比 Day28（ManiSkill/robosuite）要讲的"可复现实验矩阵"，RL 社区的 benchmark 基础设施是 Physical AI 该抄的作业。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：对称采样的 50/50 是"全程固定"的——但在训练后期，在线数据质量已经超过离线数据（策略超越了示范），此时还坚持 50% 离线采样会不会拖慢收敛？论文 Figure 12 说 50% 是最佳折中，但有没有**自适应退火**（如 $\alpha_t: 0.5\to 0.1$）的后续工作？这直接关系到 Day30 数据飞轮的"recollect → retrain"环节怎么配混合比。
- 如果要复现 / 小规模试，第一个实验做什么？用官方 repo 跑 `halfcheetah-medium-v0` 的最小命令（`--utd_ratio=20 --max_steps 250000`），然后做**单变量消融**：关掉 LayerNorm 跑一遍，观察 Q 值曲线是否发散——这是整篇论文最便宜的"啊哈时刻"（一个 flag 的 difference）。

## 原文金句 (1-2句)
> "can we simply apply existing off-policy methods to leverage offline data when learning online? … we demonstrate that the answer is yes; however, a set of minimal but important changes to existing off-policy RL algorithms are required to achieve reliable performance."（摘要）
> "correct application of these simple recommendations can provide a 2.5× improvement over existing approaches across a diverse set of competitive benchmarks, with no additional computational overhead."（摘要）

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移 ✅
- 保留并完善「和之前工作的关系」小节 ✅
- 数学视角：SAC 基座公式、对称采样的混合分布、LayerNorm+ensemble 治价值发散、"目标一致性"论证、三轴对比表 ✅

## 连接
- 上一篇: day-19-2017-ppo-robotics — PPO：on-policy 端，"仿真无限便宜"时的 RL 解法；Day20 是其对偶——"交互极贵"时的 off-policy 解法
- 下一篇预告: day-21-2019-domain-randomization — 视觉/动力学/延迟随机化：PPO+sim2real 路线（Day07/08）的另一块地基，回答"仿真和真机的 gap 怎么在训练时抹掉"

## 问答补充
（本日暂无用户追问；Day19 的 follow-up 问答已归档于 day-19 NOTES / README 问答记录。）

## 问答补充（2026-09-12）

> 以下 4 组问答来自 2026-09-11 当晚（PDT）用户对 Day20 RLPD 卡片的追问：SAC 基座是什么、"离线数据 + 高 UTD" 为什么必然发散、三个 recipe 各自解决什么（LayerNorm 的数学）、"先 IQL/CQL 离线预训练再在线微调" vs RLPD（悲观初始化的在线坍缩）。用户原文按大意精简归档，核心答案保留公式与符号定义。

### Q1：SAC 是啥？

**问题大意**：用户问"看不懂？SAC 是啥？"——需要从零讲清 RLPD 的基座算法。

**核心答案**：SAC = **Soft Actor-Critic**，连续动作控制（机器人关节力矩、轮速这类连续量）的标准 off-policy 算法。三个词拆开：

- **Actor-Critic**：两个网络分工。actor 是策略 $ \pi_\theta(a \mid s) $ ，输入状态、输出动作（比如四足 12 个关节的目标角度）；critic 是价值 $ Q_\phi(s, a) $ ，评价"在这个状态做这个动作，未来总共能拿多少分"。critic 教 actor：actor 朝 critic 说"分高"的方向挪。
- **Soft（核心）**：目标不只是"拿最多奖励"，还要"保持随机性"。普通 RL 最大化 $ \mathbb{E}[\sum_t r_t] $ ；SAC 最大化 $ \mathbb{E}[\sum_t (r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)))] $ ，其中 $ \mathcal{H} $ 是策略的熵（越随机越大）， $ \alpha > 0 $ 是温度系数（管"探索奖励"占多大比重）。直觉：**拿奖励的同时别把路走窄**——熵项逼着策略保留多种走法，避免过早坍缩到一个局部最优的步态上。
- **Off-policy**：和 Day19 PPO 的关键区别。PPO 是 on-policy——数据必须来自当前策略，用完即弃；SAC 把所有状态-动作-奖励-下一状态四元组存进 replay buffer，**旧数据可以反复拿出来训练**。真机上每步交互都贵，反复用旧数据 = 省钱。

**训练循环（三步）**：

1. **Critic 更新**（学打分）： $ Q_\phi(s, a) $ 去拟合 Bellman 目标 $ y = r + \gamma[\bar Q(s_{\text{next}}, a') - \alpha \log \pi(a' \mid s_{\text{next}})] $ ——"这个动作的价值 = 现在拿到的奖励 $ r $ + 折扣 $ \gamma $ 后的未来价值（含探索奖励 $ -\alpha \log \pi $ ）"。 $ \bar Q $ 是慢更新的 target 网络（稳定用的影子）。
2. **Actor 更新**（学动作）：朝"高 $ Q $ + 高熵"的方向挪策略。
3. **Target 慢更新**： $ \bar Q $ 缓慢跟随 $ Q_\phi $ ，防止目标乱跳。

**回到 trick ①（50/50 对称采样）为什么需要它**：朴素做法是把离线数据一次性倒进 buffer，之后均匀采样，两个死因：

- **被稀释**：在线数据越攒越多，离线数据在 buffer 里的占比 $ \to 0 $ 。训练初期策略最烂、最需要示范数据"带路"的时候，离线数据反而没被用够。
- **critic 瞎猜被放大**：critic 在没见过的状态动作对上会外推出离谱的 Q 值，而 Bellman 备份是"用下一步的 Q 教这一步的 Q"——瞎猜被反复备份、反复放大，Q 直接发散（这就是 Q2 的链条）。

50/50 全程混合的解法：每个 batch 永远一半离线、一半在线。离线那一半是"**锚**"——示范轨迹里的状态动作配的是真实奖励 $ r $ ，Bellman 备份有真凭实据，Q 在好区域被钉住；在线那一半是"**探针**"——当前策略实际到达的地方，用真实环境回报纠正 critic 的外推。两者缺一不可，所以是全程 50/50，不是开局倒一次。

一句话串起来：SAC 是"靠旧数据反复练、靠熵保持探索"的连续控制算法；RLPD 的 50/50 就是保证"旧数据里永远有一半是靠谱的示范"，让 critic 的 Bellman 备份始终有真实奖励可依。

### Q2："离线数据 + 高 UTD" 为什么必然发散？

**问题大意**：用户说"当然会失败——离线数据不是 online 的，会脱离分布；Bellman 备份反复放大 + 高 UTD 下价值发散，没看懂"——前半句的直觉已经抓住核心，后半句的机制需要用四足机器人一步步拆开。

**字母表**：

- $ Q_\phi(s, a) $ ：critic 网络，给"状态下做动作"打分；
- $ \bar Q $ ：它的慢更新影子（target 网络，训练目标用它算，稳一点）；
- $ \pi $ ：actor 策略；
- $ y = r + \gamma \bar Q(s_{\text{next}}, a') $ ：Bellman 目标（"现在奖励 $ r $ + 折扣 $ \gamma $ 后的未来价值"）；
- UTD：每采 1 步环境数据做多少次梯度更新。

**第 1 步：OOD 上神经网络会瞎猜**：离线数据是"站稳了往前走"的轨迹，critic 在这些状态动作上学得很准。但在线策略一探索，踩到一个离线数据里从没出现过的腿部角度——神经网络在没训练过的地方输出是任意的。真实价值可能是 -100（摔倒），网络却输出 +500。**这不是噪声，是系统性的"在没见过的地方瞎给高分"。**

**第 2 步：actor 专门去挑瞎猜**：actor 的工作就是"找 $ Q $ 最高分的动作"。它看到那个怪异动作有 +500，立刻把策略往那挪。最毒的一点：**误差不是随机抵消的，是被 actor 主动狩猎的**——actor 永远去找网络高估得最狠的那个点。

**第 3 步：Bellman 备份把瞎猜"转正"**：critic 的训练目标 $ y $ 里含着 $ \bar Q(s_{\text{next}}, a') $ ——也就是**用网络自己上一轮的输出当老师**。怪异动作的 +500 被写进 $ y $ ，变成"上一步动作"的监督信号；上一步的 $ Q $ 被这个虚高目标拉上去，又成为上上步的老师。误差沿时间轴倒着传播、每步都被 actor 的"挑最高分"再放大一次。**正反馈循环：瞎猜 → 被选中 → 变成教材 → 教出更大的瞎猜。**

**第 4 步：高 UTD 让自循环转得比纠偏快**：正常情况下，真实环境数据是"免费的真相来源"——每采一步，真实的 $ r $ 就把 $ Q $ 往回拽一点。UTD=20 意味着：**每 1 份新真相，要配 20 轮"学生给自己改卷、再照着改过的卷子复习"的自循环**。误差放大转 20 圈，纠偏只来 1 次—— $ Q $ 直接发散到无穷（或垃圾值），策略彻底学废。

一句话链条：**OOD 瞎猜 → actor 专挑瞎猜最大的 → Bellman 把瞎猜当老师 → 高 UTD 让自循环快过真实纠偏 20 倍 → 发散。**

这也解释了 RLPD 三个 recipe 为什么正好对症：50/50 离线数据是"每 batch 一半目标是真实奖励 $ r $ 的锚"（拽回地面）；LayerNorm 是"给 critic 的瞎猜幅度上个限幅器"；ensemble 取 min 是"一个 critic 瞎猜不算数，得所有人都瞎猜才算"——在不确定的地方集体悲观。

### Q3：三个 recipe 各自解决什么？LayerNorm 的数学

**问题大意**：用户问"3 个 recipe：50/50 采样解决了离线数据质量不足、在线数据数量不足；critic 加 LayerNorm 解决——不太懂，给我一点数学；高 UTD 解决数据量过少训练量不足"——三条直觉基本都对，需要校准 50/50 并补上 LayerNorm 的数学。

**① 50/50：对了一半，校准一个词**：不是"离线数据质量不足"——论文的消融显示 RLPD 对离线数据质量是鲁棒的（expert 和 medium 都行）。真正的问题是"**存在感**"：朴素做法里离线数据被在线数据稀释到趋近于 0，训练后期 Bellman 备份只剩在线数据自己跟自己玩。50/50 保证的是**两种监督全程都在场**：离线给"见过的好状态动作配真实奖励 $ r $ "（锚，防漂移），在线给"当前策略真实到过的地方"（探针，纠外推）。一个管"别忘"，一个管"别瞎"。

**② LayerNorm：数学来了**：

先看病根。critic $ Q_\phi(s, a) $ 是个 ReLU MLP。ReLU 网络在训练数据之外是**分段线性、向外无限延伸**的——输入越离谱，输出可以越大，**无界**。OOD 点上 $ Q $ 输出正负 $ 10^9 $ 量级都是可能的，而 Bellman 目标 $ y = r + \gamma \bar Q(s_{\text{next}}, a') $ 把这个量级当"老师"，MSE 一拟合，下一步更大——这就是发散的数学形态。

LayerNorm 对隐藏层激活 $ h \in \mathbb{R}^d $ 做：

$$ \mathrm{LN}(h)_i = \gamma_i \frac{h_i - \mu}{\sigma + \epsilon} + \beta_i, \quad \mu = \frac{1}{d}\sum_j h_j, \ \sigma^2 = \frac{1}{d}\sum_j (h_j - \mu)^2 $$

$ \mu, \sigma $ 是这层自身的均值/标准差（不是 batch 的）， $ \gamma_i, \beta_i $ 是可学习的逐维缩放/平移。

关键性质：归一化后（忽略可学习的 $ \gamma, \beta $ ），输出向量的模被**钉死**在 $ \sqrt{d} $ 附近——**跟输入多大无关**。于是最后一层 $ Q(s, a) = w^\top z + b $ 满足：

$$ |Q(s, a)| \le \|w\| \cdot \|z\| + |b| \le \|w\| \cdot C\sqrt{d} + |b| $$

右边是个**只跟网络参数有关、跟输入无关的常数**。翻译：加了 LN 的 critic 是**全局有界函数**，在 OOD 点上再也瞎猜不出 $ 10^9 $ 量级，最多瞎猜到一个有限的上限。

于是 Bellman 目标 $ y = r + \gamma \bar Q(s_{\text{next}}, a') - \gamma \alpha \log \pi $ 也被钳住（ $ r $ 有界、 $ \bar Q $ 有界）→ MSE 目标不再爆炸 → "瞎猜→当老师→更大瞎猜"的正反馈**在第一步就被掐断**。

为什么放 critic 不放 actor：病灶在 critic——只有 critic 的输出会被 Bellman 备份拿去当下一轮的监督信号。actor 只是跟着 $ Q $ 走，给 actor 加 LN 是治标不治本。这也解释了论文 Figure 9：dropout（随机置零，不约束输出尺度）、weight decay（慢慢缩权重，挡不住 OOD 方向的线性增长）都不如 LN+ensemble——后两者**直接约束函数在 OOD 区域的输出尺度**，ensemble 取 min 则是"一个 critic 瞎猜不算数"。

**③ 高 UTD：对，但补一句警告**：理解没错——真机交互贵（数据少），UTD=20 是"用计算换数据"，把每步交互的梯度利用率放大 20 倍。但 UTD 是个**放大器**：它放大梯度利用率，也放大误差。单独开高 UTD = 让上面的发散循环转 20 倍速，自杀。所以三件套是绑定的：**50/50 和 LN+ensemble 是"刹车"，UTD 是"油门"**——论文的 recipe 本质是"先装好刹车，再敢踩油门"。

### Q4：和"IQL/CQL 离线预训练再在线微调"相比，RLPD 少了什么、多出了什么？

**问题大意**：用户问'和"先 IQL/CQL 离线预训练、再在线微调"相比少了什么、多出了什么？为什么敢说"不需要复杂的离线 RL 算法"？（提示：悲观初始化的在线坍缩）这个真不太理解'——这是 Day20 最反直觉的结论，值得慢拆。

**第一步：IQL/CQL 在干什么**：纯离线 RL 有个绝症——训练时**不能试错**，critic 在没见过的状态动作上瞎猜高分，actor 去捡，部署就翻车。离线 RL 的通用解法叫**悲观主义**——"没见过的，一律先判低分"：

- **CQL**：在 Bellman loss 上加一项惩罚， $ \min_Q \alpha \big( \mathbb{E}_{a \sim \pi}[Q(s, a)] - \mathbb{E}_{a \sim \mathcal{D}}[Q(s, a)] \big) $ ——把策略想去的 OOD 动作的 $ Q $ 往下压，把数据里见过的动作的 $ Q $ 往上抬。 $ Q $ 地形被人工挖出"洼地"：见过的高，没见过的低。
- **IQL**：更鸡贼，干脆**不让 critic 看 OOD 动作**——用 expectile 回归只学数据内动作的价值，actor 只在数据支撑的范围内改进。

两阶段范式的直觉很顺："先用悲观离线法学个安全好用的初始策略，再上线微调超越它。"听起来像"先站稳再学走"。

**第二步：坍缩是怎么发生的**：问题在于，CQL/IQL 学出的 $ Q $ **不是真实价值地图，是"被悲观整形过"的地图**。上线后三件事连锁发生：

1. **洼地是假的**：在线策略稍微探索一下，到了一个离线数据没覆盖、但其实很好的抓取姿势。真实奖励 $ r $ 说"这很好"，但悲观 $ Q $ 说"-1000，没见过就是雷区"。Bellman 目标 $ y = r + \gamma \bar Q(s_{\text{next}}, a') $ 里，真实的 $ r $ 和虚假的 -1000 打架 → 巨大的 TD 误差。
2. **地图被迫重画**：在线微调（通常会关掉或减弱保守惩罚，不然学不动）开始用真实回报修正这些洼地。 $ Q $ 函数经历剧烈改写——洼地被填平，之前 actor 依赖的"哪里能去、哪里不能去"的全部知识**一次洗掉**。
3. **actor 被甩下车**：actor 当初是跟着悲观 $ Q $ 学会的"谨慎好行为"； $ Q $ 一重画，actor 的梯度方向全乱，策略 performance 先**断崖下跌**，再慢慢爬回来——论文的实验里，这个坑深到**不如从零开始训**。

打个比方：CQL 预训练像教开车时说"所有没见过的路都有地雷"，学员只敢在熟路上开；转在线等于告诉他"其实大部分路没雷"，他得先**亲手把地雷地图擦掉**——擦地图的那几分钟，开得比零基础学员还烂。而 RLPD 是：直接上路，旁边坐个教练（在线真实回报），车上装好护栏（LayerNorm+ensemble）——从没画过假地图，也就不用擦。

**第三步：RLPD 少了什么、多出了什么**：

- **少了**：CQL 的保守惩罚项、IQL 的 expectile、BC 约束、"离线阶段/在线阶段"两套切换逻辑——整个悲观主义工具箱都没要。Loss 就是朴素的 $ \frac{1}{2}\mathbb{E}_{\text{off}}[(Q - y)^2] + \frac{1}{2}\mathbb{E}_{\text{on}}[(Q - y)^2] $ ，干干净净。
- **多了**：从第 0 步就存在的在线数据流。悲观主义当年要解决的问题是"**不能收集纠偏数据**时 $ Q $ 会炸"；可一旦你本来就要在线训，纠偏数据是**免费且持续**的——用一个制造假地图的复杂机制，去解决一个已经不存在的问题，还附赠"擦地图"的坍缩代价，这买卖不划算。

**所以为什么敢说"不需要复杂的离线 RL 算法"**：不是说它数学上证明了悲观主义无用，而是实验事实——在 offline-to-online 这个设定下，**在线数据本身就是最好的"保守主义"**：真实回报对 $ Q $ 外推的纠正，又及时又准确，比任何人工惩罚项都强。论文的论点是工程性的"少即是多"：既然 50/50 + LayerNorm + ensemble 这三块"护栏"已经够让 plain SAC 在在线设定下不翻车，离线 RL 的复杂性就是**不必要的**——留着它，你付的是"坍缩税"，得到的是零。

一句话：悲观初始化是给"永远不能上路的人"准备的拐杖；RLPD 发现你本来就能上路，拐杖可以扔了，护栏留下。

**关联**：Day19（PPO：on-policy 端，"仿真无限便宜"时的 RL 解法；Day20 是其对偶）；本 NOTES §数学视角（SAC 基座公式、对称采样、LayerNorm+ensemble 治价值发散）。
