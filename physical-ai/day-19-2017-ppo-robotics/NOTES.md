# Day 19 — PPO for Robotics: clipped policy optimization 与 GPU 向量化 rollout 系统

## 元信息
- Title: Proximal Policy Optimization Algorithms（主论文）+ Legged-Gym / RSL-RL（robotics rollout 系统）
- Authors / Org: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov — OpenAI（2017，主论文）；Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter — Robotic Systems Lab, ETH Zurich & NVIDIA（legged_gym, CoRL 2021）；rsl_rl 维护 Mayank Mittal / Clemens Schwarke — ETH Zurich RSL & NVIDIA
- Link / arXiv / Blog: https://arxiv.org/abs/1707.06347 ｜ https://leggedrobotics.github.io/legged_gym/ ｜ https://arxiv.org/abs/2109.11978
- Date read: 2026-09-10
- Tags: [physical-ai, ppo, reinforcement-learning, actor-critic, rollout, legged-gym, rsl-rl, sim2real]
- Thread: physical-ai
- Folder: day-19-2017-ppo-robotics
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-19-2017-ppo-robotics

## 一句话总结
PPO 用 **clipped surrogate objective + GAE** 把 policy gradient 的更新锁进信任域内，实现"采一次数据、做多个 epoch minibatch 更新"的稳定 on-policy 训练；而 legged_gym / rsl_rl 把 **环境搬进 GPU**（数千并行 env、数据不出显存），让这套算法在单卡上分钟级训出可零样本迁移的腿式机器人行走策略——它是 Day07/08 背后一直在用的算法，今天才正式讲透。

## 和之前工作的关系

- **接了哪条线**：Day19–24 "RL for Robotics / Sim2Real" 专题块的开篇；也是 Day03（Isaac Lab）→ Day07/08（H2O、Humanoid-Gym）这条"仿真里训策略、真机上跑"线的**算法地基**。Isaac Lab 官方配套的 RL 训练器正是 rsl_rl 的 PPO。
- **补了哪个短板**：Day07 H2O 和 Day08 Humanoid-Gym 都写了"用 PPO 训练"，但 PPO 本身是什么、为什么是它、它的数学长什么样，一直是个黑盒。今天把这个黑盒打开——它是整个 30 天路线里**唯一一篇讲"怎么从 reward 里学策略"而不是"怎么从示范里学策略"的**。
- **替代 / 分叉 / 改进**：PPO 本身是对 TRPO 的改进（TRPO 用 KL 硬约束 + 二阶优化，贵且难实现；PPO 用 clip 一阶近似，简单且实证更好）。在机器人 rollout 系统里，它是 SAC/DDPG（off-policy）之外的 on-policy 主流选择；legged_gym 论文明确对比过：在大规模并行 regime 下 PPO 的稳定性胜出。
- **对之前 Day X 的直接对比**：
  - **vs Day11–14（VLA，behavior cloning）**：那四篇都在示范数据上做最大似然 $-\mathbb{E}_{(O,A,c)\sim\mathcal{D}}\log p_\theta(A\mid O,c)$，数据是**死的**（fixed dataset）；PPO 的数据是**活的**（on-policy rollout），目标是 reward 的期望，更新是 advantage 加权的。Action 表示（离散 token / flow / diffusion）和训练原则（BC-MLE / RL-PG）是**两条正交的轴**——π₀ 的 flow 头理论上也可以用 PPO 的 surrogate 去训（这正是 RL 微调 VLA 的方向）。
  - **vs Day06（DreamerV3）**：DreamerV3 也是 actor-critic，但 actor 在**想象的 latent rollout** 里用 value 梯度更新；PPO 是 model-free，在**真实（仿真）环境**里 on-policy 更新。两者都估计 advantage、都更新策略，差别在"数据从哪来"。
  - **vs Day12（Diffusion Policy）**：DP 的去噪头是确定性的 BC；PPO 的策略是**随机高斯策略** $\pi(a\mid s)=\mathcal{N}(\mu_\theta(s),\mathrm{diag}(\sigma^2))$——随机性不是 bug，是**探索的来源**；entropy bonus 防止 σ 过早塌缩到确定性。

## 为什么今天读它

Roadmap 明确写了 Day19 是"PPO for Robotics — clipped policy optimization and rollout systems"，且它是 Day19–24 RL 专题块的入口。更重要的是：**我们之前读的 locomotion 论文（Day07/08）都在用 PPO 却没讲 PPO**，就像读了四天的"怎么做菜"没讲"火是什么"。今天补上这块地基，后面 Day20（RLPD）、Day21（domain randomization）、Day22（RMA）才能站在坚实的算法基础上读。

## 今天的 3 问
1. 为什么 PPO 敢在**同一个 batch** 上做多个 epoch 的 minibatch 更新，而普通 policy gradient（VPG）不行？clip 在其中到底扮演什么角色？
2. **Asymmetric actor-critic**（critic 看 privileged state，actor 只看可部署观测）为什么在 sim2real locomotion 里几乎是标配？它和 PPO 的哪部分设计天然契合？
3. Day11–14 的 VLA 全是 behavior cloning（在示范上做 MLE），PPO 是 on-policy RL（数据来自当前策略）。为什么过去机器人领域 **BC 占主导、PPO 主要用在 locomotion**？这条分界线的本质是什么？

## 核心
1. **Motivation**: Policy gradient 的核心痛点是**步长**：步子太小训得慢，步子太大策略崩——一次激进更新就能让智能体"忘记"之前学到的，永远回不来。TRPO（Schulman et al. 2015）用 KL 硬约束 + 二阶优化保证单调改进，但 Hessian 贵、实现难调。PPO 问：能不能**只用一阶方法、一个 min()**，达到同样的"别走太远"效果？答案是把约束**写进目标函数本身**（clip），而不是写进优化器。与 Physical AGI 的关系：它是"让智能体自己从试错里学"的最可靠工程实现——不需要人类示范（对比 Day15–18 的数据论文），只需要 reward + reset，而仿真恰好能无限提供这两样。
2. **System / Method**: 算法 = **collect → optimize 交替**（论文摘要原话："alternate between sampling data through interaction with the environment, and optimizing a 'surrogate' objective function"）。Collect：用 $\pi_{\theta_{old}}$ 跑 $N$ 个并行 env、每 env $T$ 步，存 $(s_t,a_t,r_t,\log\pi_{\theta_{old}}(a_t\mid s_t),V(s_t))$。Optimize：在这批固定数据上做 $K$ 个 epoch 的 minibatch SGD，最大化 clipped surrogate（见数学视角）。Robotics 系统层（rsl_rl）：这套循环**全在 GPU 上**——env（Isaac Gym/Lab 物理）、策略网络、PPO buffer 都在显存里，数据不出卡；另有 Student-Teacher Distillation 分支（teacher 看 privileged、student 只看部署观测蒸馏），以及 RND 好奇心奖励、对称性增强等插件。
3. **Training / Data Details**: Sim 数据 = 并行 env 交互产生（legged_gym：数千 ANYmal 并行跑，单卡 >100k steps/s 量级吞吐——此数为社区复述）；Real 数据 = **零**（locomotion 经典路线是纯 sim 训练 + zero-shot 真机，Day08 同理）。Sim2Real = domain randomization（摩擦、质量、电机强度、PD 增益、观测噪声、随机推力）+ reward shaping；Reward = 速度跟踪项 − 能量/加加速度惩罚 − 摔倒大惩罚（verifiable signal 就是这些可计算的物理量，不需要人类标注）。Curriculum：legged_gym 的 game-inspired 自动课程——每个 env 按自己的成功情况升/降地形难度（走过边界升级，走不远降级）。
4. **Key Tricks**: ① **Asymmetric actor-critic**：critic 的输入是 privileged state（真实速度、地形高度图），actor 只看部署可观测（IMU、关节、历史）；PPO 的 actor/critic 本来就是两个独立网络，这成了零成本插件——value 估得准（方差小），策略学得稳（可部署）。Day08 的 Humanoid-Gym 用的正是这招。② **GPU-resident rollout**：rsl_rl 从 Isaac Gym 的 rl-pytorch 演化而来，设计哲学是"把环境搬进加速器"——env step 成本趋近于零后，算法选择的约束从 sample efficiency 转向 wall-clock throughput；legged_gym 官方结论：平地策略**不到 4 分钟**、崎岖地形 **20 分钟**（单张 workstation GPU，论文/项目页原文）。③ **Clip 作为稳定契约**：$\epsilon=0.2$ 的 ratio clip + 多 epoch 复用 + 自适应学习率（按 target KL 调 lr），三者合起来让 PPO 在强 reward shaping 和重度 domain randomization 下也不炸——这正是它成为机器人默认 RL 算法的工程原因，而非理论最优性。
5. **Results**: 论文原文实验：MuJoCo 连续控制 locomotion 任务集 + 49 个 Atari 游戏，结论（摘要原文）："PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time." Robotics 侧：legged_gym 在 ANYmal 四足上实现分钟级训练并**零样本迁移到真机 ANYmal C**；该框架成为 Isaac Lab / legged_gym 生态的默认 PPO 实现（rsl_rl），被 50+ 篇腿式 locomotion 论文采用（社区统计）。

## 数学视角

> 先统一符号（全文通用）：$s_t$ 真实状态，$o_t$ 观测（机器人能看到的），$a_t$ 动作，$r_t$ 标量奖励，$\theta$ 策略网络参数，$\phi$ value 网络参数，$\pi_\theta(a\mid s)$ 策略（给定状态输出动作分布），$V_\phi(s)$ 状态价值估计，$\gamma\in(0,1)$ 折扣因子，$\tau=(s_0,a_0,r_0,\dots)$ 一条轨迹。

### 1. 目标：从"学示范"到"学 reward"

Day09–14 的 VLA 学的是**示范分布**：
$$p_\theta(A\mid O,c)\approx p_{\text{demo}}(A\mid O,c),\qquad \mathcal{L}_{\text{BC}}=-\mathbb{E}_{(O,A,c)\sim\mathcal{D}}\log p_\theta(A\mid O,c).$$

PPO 学的是**最优策略**——最大化折扣回报期望：
$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].$$

Policy gradient 定理给出梯度方向（REINFORCE 形式）：
$$\nabla_\theta J(\theta)=\mathbb{E}_{s,a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot A^{\pi}(s,a)\right],$$
其中 $A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$ 是 **advantage**：这个动作比平均水平好多少。直觉：**好的动作（$A>0$）增加其对数概率，坏的动作（$A<0$）减少**——梯度即"加权投票"。

### 2. 核心公式：clipped surrogate（回答 Q1）

普通 PG 每采一批数据只能做**一次**梯度更新——因为更新后数据就"过期"了（off-policy）。PPO 用重要性采样 ratio 把旧数据"续命"：
$$r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}\quad\text{（新策略 vs 采样策略的似然比；}=1\text{ 表示没动过）}.$$

Clipped surrogate objective（论文式 7，本节核心）：
$$L^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_t\left[\min\Big(r_t(\theta)\hat{A}_t,\ \text{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\hat{A}_t\Big)\right].$$

逐符号解释：
- $\hat{\mathbb{E}}_t$：在**当前 batch**（$N$ envs × $T$ steps 个样本）上的经验平均；
- $\hat{A}_t$：advantage 估计（见下 GAE），标量；$>0$ 表示"好动作"，$<0$ 表示"坏动作"；
- $r_t(\theta)\hat{A}_t$：无约束的 surrogate——$r_t$ 偏离 1 越远，说明策略改动越大；
- $\text{clip}(r_t,1-\epsilon,1+\epsilon)$：把 ratio 钳在 $[1-\epsilon,1+\epsilon]$ 内，$\epsilon=0.2$ 是机器人常用值（"信任域半径"）；
- $\min(\cdot,\cdot)$：取悲观的那个——**只允许"保守"的改进**。

为什么 $\min$ 是关键（两情况）：
- $\hat{A}_t>0$（好动作）：目标是增大 $r_t$（让好动作更可能）。但 $\min$ 取 $\text{clip}$ 后的值——$r_t$ 超过 $1+\epsilon$ 后目标不再增长，梯度归零：**"够好了，别再推了"**，防止一次更新把策略推飞。
- $\hat{A}_t<0$（坏动作）：目标是减小 $r_t$（让坏动作更不可能）。$r_t$ 低于 $1-\epsilon$ 后同样被钳住：**"惩罚到此为止"**。

这就是对 Q1 的回答：**clip 让目标函数在策略偏离采样策略太远时自动"罢工"（梯度为零），于是同一批数据可以安全地复用 $K$ 个 epoch**——约束从优化器（TRPO 的二阶 KL 约束）搬进了目标函数（一阶 min-clip），代价是零，实现是三行代码。

### 3. GAE：advantage 怎么估计（bias–variance 的旋钮）

$$\hat{A}_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).$$

- $\delta_t$：**TD error**（时序差分误差）——"实际得到的（即时奖励 + 折扣后的未来价值估计）比预期多多少"；$V_\phi$ 是 critic 网络。
- $\gamma=0.99$：折扣，决定"多远的未来算数"（$\gamma^{100}\approx 0.37$，约 100 步 ≈ 2 秒@50Hz 的有效视野）。
- $\lambda=0.95$：GAE 的 bias–variance 旋钮。$\lambda=1$ 是 Monte Carlo（无偏、高方差：整条轨迹的噪声全进来）；$\lambda=0$ 是单步 TD（有偏、低方差：全靠 $V_\phi$ 的估计）。$\lambda=0.95$ 是"用指数衰减混合多步回看"的折中——**机器人常用值**。
- 直觉：$\hat{A}_t$ 是"未来所有惊喜的折扣加权和"，$\lambda$ 控制你愿意相信多远的惊喜。

### 4. 完整训练目标与张量维度

$$L(\theta,\phi)=\hat{\mathbb{E}}_t\left[L^{\text{CLIP}}_t(\theta)\ -\ c_1\underbrace{(V_\phi(s_t)-V^{\text{targ}}_t)^2}_{\text{value loss}}\ +\ c_2\underbrace{S[\pi_\theta](\cdot\mid s_t)}_{\text{entropy bonus}}\right].$$

- $c_1\approx 0.5\text{–}1.0$：value loss 系数（critic 拟合回报目标 $V^{\text{targ}}_t=\hat{A}_t+V_{\phi_{\text{old}}}(s_t)$）；
- $c_2$（小量，如 0.01）：熵奖励，**防止高斯策略的 $\sigma$ 过早塌缩**——塌缩 = 停止探索；
- $S$：策略熵，$\pi_\theta(a\mid s)=\mathcal{N}(\mu_\theta(s),\mathrm{diag}(\sigma^2))$ 的熵。

**维度表**（以 legged_gym ANYmal 为例的数量级）：
| 符号 | 维度 | 含义 |
|---|---|---|
| $s_t$（privileged） | $\mathbb{R}^{\sim 200+}$ | 真实速度、地形高度采样、本体感知全量 |
| $o_t$（actor 输入） | $\mathbb{R}^{\sim 48}$ | IMU、关节角/速度、命令速度、历史（可部署） |
| $a_t$ | $\mathbb{R}^{12}$ | 12 个关节的目标位置（经 PD 控制器转力矩） |
| $\mu_\theta(s)$ | $\mathbb{R}^{12}$ | 策略网络输出的均值；$\sigma\in\mathbb{R}^{12}$ 可学习或固定 |
| batch | $(N\cdot T)$ 样本 | 如 $N=4096$ envs × $T=24$ steps ≈ 10 万样本/iteration |
| minibatch | batch/$M$ | $M$ 个 minibatch × $K$ epochs（如 $K=5$）复用同一批数据 |

**时间尺度**：仿真 dt≈1–2ms → 控制 50Hz（每 20ms 输出一次 $a_t$）→ 每次 collect 约 $T=24$ 步 ≈ 0.5 秒 sim 时间/env → 1000–2000 次 PPO iteration ≈ 20–40M env steps → **wall-clock 分钟级**（单卡；吞吐 >100k steps/s 为社区复述数，论文/项目页确认的是"平地 <4 分钟、崎岖地形 20 分钟"）。

### 5. 统一框架：一张图看懂 BC / Actor-Critic / PPO

```
数据从哪来                    学什么                          代表
─────────────────────────────────────────────────────────────────
固定示范 D（死的）   →  最大似然拟合 p(A|O,c)        Day09–14 VLA（BC）
想象的 latent       →  value 梯度更新 actor          Day06 DreamerV3
on-policy rollout   →  advantage 加权 PG + clip     Day19 PPO（本篇）
                      （活的，当前策略自己采）
```

三条正交轴：**① 数据来源**（示范 / 想象 / 在线交互）、**② 动作表示**（离散 token / flow 场 / diffusion 去噪器 / 高斯 $\mu,\sigma$）、**③ 训练原则**（MLE / value-gradient / clipped-PG）。Day11–14 固定了①=示范、换②；Day06 换①=想象；今天换①=在线交互 + ③=clipped surrogate。**PPO 的数学（ratio-clip）与动作表示无关**——它既可以训高斯策略（locomotion），也可以训 diffusion/flow 头（RL 微调 VLA 的新兴方向）。

**Asymmetric actor-critic 的数学位置**（回答 Q2）：把 $V_\phi$ 的输入从 $o_t$ 换成 privileged $s_t\supset o_t$，即 $V_\phi(s_t)$ 而 $\pi_\theta(a_t\mid o_t)$。PPO 的 actor 和 critic 是**两个独立网络、只通过 $\hat{A}_t$ 耦合**，所以 critic 看什么不影响 actor 的部署约束——这种解耦是 PPO 设计里天然留出的"后门"，locomotion 社区把它变成了标配。

### 6. 数学没覆盖的东西（诚实清单）

- **接触物理**：PPO 的 MDP 假设转移 $p(s'\mid s,a)$ 光滑可微可采样；真实接触是非光滑、刚性的（Day02 MuJoCo 的 contact solver 才处理这个），reward shaping 里的"摔倒大惩罚"是对接触失败的粗糙代理。
- **延迟与带宽**：公式里的 $a_t$ 假设瞬时执行；真机有 10–30ms 控制延迟 + PD 跟踪误差，Day07/08 靠 domain randomization 硬扛，数学上无显式建模。
- **Reward 即真理的幻觉**：PPO 只保证"最大化你写的 reward"，不保证 reward = 你想要的步态——"reward hacking"（如原地抖动刷速度奖励）是 locomotion 论文里反复出现的工程坑，clip 保的是**更新稳定**，不保**目标正确**。
- **On-policy 的代价**：数据用完即弃（无 replay buffer），sample efficiency 天生低于 off-policy（SAC）；机器人能接受纯粹因为**仿真把采样成本打到了零**——这是 rsl_rl 这类系统的存在意义，也是 sim2real  pipeline 的前提假设。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？** PPO 算法本身的贡献是**更新契约**（clip-surrogate 的稳定性），跨任务通用；但让它在机器人落地的真正钥匙是**框架**（rsl_rl/legged_gym 的 GPU 向量化 rollout + reward/curriculum 工程）。论文的 MuJoCo/Atari 结论 transfer 到了腿式机器人，但"分钟级训练"这个数量级变化 100% 来自系统侧。这是"算法论文 + 系统实现"双轮的一个标准案例。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发**：
  1. **RLHF 的 PPO/GRPO 是同一条血统**：InstructGPT/ChatGPT 的 RLHF 用的就是 PPO-clip（你 ai-infra 里 grpo-vs-ppo 文件夹的老熟人）。GRPO = PPO 去掉 value、换成 group 内 baseline——数学上都是"ratio-clip 的多步复用"。今天把机器人侧的 PPO 读透，RLHF 侧的那套公式就自动对齐了：$\hat{A}_t$（GAE）↔ GRPO 的 group-normalized reward，$r_t(\theta)$ clip ↔ 同一个 clip。
  2. **Rollout 系统思维可平移**：rsl_rl 的"vectorized collect → minibatch optimize → 分布式同步"三段式，就是 RLHF rollout/verifier 流水线的机器人版；"把环境搬进加速器"（Isaac GPU 物理）和"把推理搬进加速器"（vLLM rollout）是同一个 Infra 直觉——**先把采样成本打下来，再谈算法**。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性**：rsl_rl 证明了 on-policy RL 的扩展瓶颈不在算法而在**吞吐**：单卡数千 env、数据不出显存、reward/curriculum 全向量化。评测侧：legged_gym 的"地形难度自动课程"本质是**按 env 粒度的自适应 eval**——每个 env 既是训练样本也是难度探针，这对设计机器人 eval harness（Day28/30 的主题）是现成模板。可复现性上，PPO 对随机种子相对鲁棒（clip 的功劳），但 reward shaping 的每一项权重都是隐性超参——复现 legged_gym 结果时，reward 文件比算法文件更重要。

## 疑问 / 下一步

- **没看懂的 / 想深挖的 1 个问题**：论文其实提了两个变体——PPO-Clip 和 PPO-Penalty（KL 惩罚 + 自适应系数 $\beta$，理论上更接近 TRPO）。为什么实证上 PPO-Clip 赢了？是 $\beta$ 自适应调参没调好，还是"硬 clip 的悲观 min"在结构上就更适合深度网络的噪声梯度？（这关系到 RLHF 里为什么大家也都用 clip 版。）
- **如果要复现 / 小规模试，第一个实验做什么？**：`pip install rsl-rl-lib` + legged_gym（或 Isaac Lab 的官方 locomotion 例），跑 ANYmal 平地行走 500 iterations，画三条曲线：episode reward、clip fraction（被 clip 的样本比例，rsl_rl 会 log）、mean KL——观察 clip fraction 是否随训练收敛到一个稳定带，如果它长期 =0 或 =1，说明 $\epsilon$ 或 lr 选错了。这是判断一次 PPO 训练是否健康的**第一性指标**。

## 原文金句 (1-2句)
> "Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time." —— PPO 论文摘要（arXiv:1707.06347，逐字引用）

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移 ✅
- 保留并完善「和之前工作的关系」小节 ✅（新增：BC vs RL 两条正交轴、Day07/08 的算法黑盒补完、与 DreamerV3 的 actor-critic 对比）
- 数学视角：clipped surrogate + GAE + 维度/时间尺度表 + 统一框架图 + 诚实清单 ✅

## 连接
- 上一篇: [day-18-2024-robocasa](../day-18-2024-robocasa/NOTES.md) — RoboCasa：仿真数据规模化的"数据端"
- 下一篇预告: [day-20-2023-rlpd](../day-20-2023-rlpd) — RLPD：把离线先验数据混进在线 RL，回答"PPO 的 on-policy 数据能不能和示范数据一起吃"

## 问答补充（2026-09-10）

> 以下 4 组问答来自 2026-09-10 当晚用户对 Day19 PPO 卡片（batch 复用 / asymmetric actor-critic / locomotion-manipulation 边界 / PPO 跨域）的追问。用户原文按大意精简归档，核心答案保留公式与符号定义。

### Q1：PPO 把同一批 rollout 复用多个 epoch 训，为什么概率比不会失控？

**问题大意**：普通 policy gradient 如果同一批数据训好几轮，新/旧动作概率比 $r_t(\theta)$ 会一路跑偏、把训练搞崩；用户指出 PPO 靠 clip 把这个兜住了。

**核心答案**：PPO 的目标是 clipped surrogate：

$$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]$$

**字母表**：

- $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$：新策略相对旧策略的动作概率比（importance-sampling ratio）；旧策略 $\pi_{\theta_{\text{old}}}$ 是采这批数据的那个策略。
- $\hat{A}_t$：GAE 估计的 advantage（本 NOTES §数学视角）。
- $\epsilon$：clip 半径，常取 $0.1$–$0.3$；$\text{clip}(r,1-\epsilon,1+\epsilon)$ 把 $r$ 钳在 $[1-\epsilon,1+\epsilon]$ 里。
- **多 epoch 复用**：同一批 rollout 数据上做 $K$ 个 epoch 的 minibatch 更新；每一轮 $\theta$ 都在变，所以 $r_t(\theta)$ 逐渐偏离 1。

**为什么普通 PG 会崩**：无 clip 时目标是 $\hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t]$。重复 epoch 里 $\theta$ 每轮都朝 advantage 方向走，$r_t(\theta)$ 可以单调变大（$\hat{A}_t>0$ 的样本被反复加码），策略一步迈太大 → 训练发散。

**asymmetric clip 的机制**（非对称体现在 $\min$ 操作里）：

- $\hat{A}_t>0$（好动作）：$r_t$ 超过 $1+\epsilon$ 后，$\min$ 取的是被 clip 的那一项，梯度归零——"已经够好了，别再加码"；但如果更新方向错了（$r_t<1$，好动作概率反而下降），$\min$ 取 $r_t\hat{A}_t$，梯度保留，把概率纠回来。
- $\hat{A}_t<0$（坏动作）：对称地，$r_t$ 降到 $1-\epsilon$ 以下后梯度归零；但如果 $r_t>1$（坏动作概率反而上升），$\min$ 取 $r_t\hat{A}_t$，梯度保留，压回去。

一句话：clip 让"已经充分的好/坏更新自己停下来"，但"走错方向的纠正梯度"永远保留——这就是 asymmetric 的含义，也是 batch 多 epoch 复用不崩的数学原因。

**和之前工作的关系**：这是本 NOTES §数学视角 clipped surrogate 的展开；和 ai-infra `grpo-vs-ppo` 里 RLHF 的 PPO-clip 是同一个公式——GRPO 只是把 $\hat{A}_t$ 换成组内归一化 reward，clip 的思想不变。

### Q2：asymmetric actor-critic 是因为环境噪声大、actor 需要稳定吗？

**问题大意**：用户最初猜测 asymmetric actor-critic 的动机是"环境更嘈杂、actor 需要稳定"。

**核心答案**：不是。环境噪声不是重点，关键是**部署约束的不对称**：

- critic 可以用仿真器里的 privileged state $s_t$（真实关节力矩、接触状态、地面摩擦系数……），actor 只能用真机上实际可观测的 $o_t$（IMU、关节编码器）。
- **数学位置**：把 $V_\phi$ 的输入从 $o_t$ 换成 $s_t\supset o_t$，即 $V_\phi(s_t)$ 而 $\pi_\theta(a_t\mid o_t)$（本 NOTES §数学视角已有这句）。
- 为什么有效：critic 的任务是估计 value/advantage $\hat{A}_t$，输入越全、方差越小，actor 的策略梯度就越稳。PPO 的 actor 和 critic 是**两个独立网络、只通过 $\hat{A}_t$ 耦合**，所以 critic 看 privileged 信息不会污染 actor 的部署约束——这种解耦是 PPO 设计里天然留出的"后门"，locomotion 社区把它变成了标配。
- 部署时 critic 直接扔掉，只留 actor。

**和之前工作的关系**：呼应 Day07 (H2O) / Day08 (Humanoid-Gym) 的工程实践——"训练用仿真特权信息、部署只用本体感知"。

### Q3：manipulation 的探索空间比 locomotion 大/贵得多，所以 RL 做不动？

**问题大意**：用户指出 manipulation 的 rollout 和探索空间比 locomotion 大得多、也贵得多。

**核心答案**：对。这正是实践中 **RL vs BC 的边界**：

- **locomotion**：reward 稠密可算（速度跟踪、姿态、能耗项），仿真 reset 便宜（Isaac/MuJoCo 并行几千 env），随机探索能踩出 reward 信号 → on-policy RL (PPO) 占优。
- **manipulation**：reward 稀疏难写（"把杯子放进柜子"的成功信号只有 0/1），接触丰富的任务随机探索成功概率接近 0（$p_{\text{random success}}\approx 0$）→ RL 采不到梯度，**behavior cloning**（拟合演示分布）占主导。Day11–14 的 VLA 路线全是 BC / Diffusion / Flow 头，正是这个原因。
- 工程直觉：探索空间"小/便宜"→ 纯在线 RL；"大/贵" → 先 BC 起步（或 Day20 RLPD 那种离线+在线混合）。

**和之前工作的关系**：这是本 NOTES §统一框架图"BC vs RL 两条正交轴"的实例化，也是 Day12 Diffusion Policy 路线选择的底层理由。

### Q4：机器人 PPO 和 LLM post-training 的 PPO/GRPO，是同一套东西吗？

**问题大意**：用户观察到这场讨论说明 RL 技术在多个领域通用，问机器人 PPO 和 LLM 侧 PPO/GRPO 的关系。

**核心答案**：核心数学同一套，工程约束不同。

- **相同**：advantage $\hat{A}_t$（GAE）↔ GRPO 的组内归一化 reward；ratio $r_t(\theta)$ + clip 的 trust-region 式控制；多 epoch batch 复用。
- **不同**：① rollout 来源——机器人是仿真器 env step，LLM 是模型自采样 token；② reward 构造——机器人是 hand-shaped dense reward + curriculum，LLM 是 verifier/RM 稀疏 reward；③ 失败成本——机器人真机摔了是硬件钱，LLM 采样错了是算力钱。

**和之前工作的关系**：呼应本 NOTES §可迁移 "RLHF 的 PPO/GRPO 是同一条血统"。
