# Day 21 — Domain Randomization / ADR：Solving Rubik's Cube with a Robot Hand

## 元信息
- Title: Solving Rubik's Cube with a Robot Hand（Automatic Domain Randomization，ADR）
- Authors / Org: OpenAI（Ilge Akkaya*, Marcin Andrychowicz*, Maciek Chociej*, Mateusz Litwin*, Bob McGrew*, Arthur Petron*, Alex Paino*, Matthias Plappert*, Glenn Powell*, Raphael Ribas*, Jonas Schneider*, Nikolas Tezak*, Jerry Tworek*, Peter Welinder*, Lilian Weng*, Qiming Yuan*, Wojciech Zaremba*, Lei Zhang*；作者按字母序，引用请用 OpenAI et al.）
- Link / arXiv / Blog: https://arxiv.org/abs/1910.07113（2019-10-17 preprint）/ 官方博客 http://openai.com/index/solving-rubiks-cube/
- Date read: 2026-09-12
- Tags: [physical-ai, domain-randomization, adr, sim2real, dexterous-manipulation, ppo, meta-learning, rubiks-cube]
- Thread: physical-ai
- Folder: day-21-2019-domain-randomization
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-21-2019-domain-randomization
- Official code: 训练系统**无官方开源**（同 2018 Dactyl 一样）；环境层面的 Shadow hand 可参考 openai/robogym，但不含 ADR 训练管线——卡片里只给论文 + 官方博客链接，诚实标注。

## 一句话总结
真机数据太贵、接触动力学（摩擦/弹性/延迟）又测不准——DR 的赌注是"**别试图把仿真做准，把仿真做宽**"：在随机化的环境参数分布 $p_\phi(\xi)$ 上训策略，只要真实世界落在分布的支撑集里，transfer 自然发生。Tobin 2017 证明了视觉 DR 可行（纯仿真 RGB 训物体检测器转真机抓取）；OpenAI 把它推到动力学 DR（2018 Dactyl 方块重定向），再用 **ADR（Automatic Domain Randomization）** 解决"手工调随机化范围"这个工程瓶颈——随机化分布的边界按策略在边界上的表现**自动扩张**，熵单调递增，形成"难度永远递增"的自动 curriculum。最终：**纯仿真训练的五指灵巧手在真机上解魔方，完整打乱 60% 成功率、极限打乱 20%**，且戴橡胶手套、绑住手指、被长颈鹿玩偶捅都不崩——鲁棒性不是调出来的，是分布"吃"出来的。

## 和之前工作的关系

- **接了哪条线**：接 Day02（MuJoCo 接触物理是仿真器）、Day03（Isaac Lab 的 randomization manager 正是 DR 的工程载体）、Day07/08（locomotion 的 sim2real 全靠动力学 DR）——DR 是这三天"为什么能 transfer"的**数学地基**。同时接 Day19（PPO）：ADR 论文用的就是 OpenAI Five 那套分布式 PPO，Day19 的算法黑盒在这里是"被随机化喂大的"版本。
- **补了哪个短板**：Day07/08 讲了"PPO + DR 能 zero-shot sim2real"，但**没讲 DR 本身**——随机化什么、范围怎么定、为什么有效、边界在哪。Day21 把这块地基补上：DR 的数学是"在世界分布上做期望/鲁棒优化"，transfer 的充要直觉是"真机参数落在随机化支撑集里"。
- **替代 / 分叉 / 改进**：三条 sim2real 路线的分叉点就在今天：
  - **DR/ADR 路线**（今天）：不碰真机数据，把仿真做宽。代价是算力（数千年仿真经验）和峰值性能（hedging 吃精度）。
  - **RLPD 路线**（Day20）：直接在真机/贵环境上在线学，用离线数据当燃料。代价是真机交互和安全风险。
  - **显式自适应路线**（Day22 RMA 预告）：学一个 adaptation module 在线辨识动力学。代价是需要 privileged 训练。
  三者不是谁替代谁——**交互成本 × 仿真可信度 × 接触复杂度**决定选哪条（见数学视角 §6 的决策表）。
- **对之前 Day X 的直接对比**：
  - **vs Day19（PPO）**：同一套 PPO，但在 ADR 下优化的不再是"一个 MDP"，而是"一个 MDP 分布"——Day19 的 trust region 管的是"策略别走太远"，ADR 的边界扩张管的是"世界别太窄"。两个"别"合起来才是真机可用的策略。
  - **vs Day20（RLPD）**：数学上的对偶。RLPD：$ \mathcal{L}_Q $ 在"离线+在线"混合经验分布上做 Bellman 备份——**用数据覆盖不确定性**；ADR：$ J(\theta) $ 在"随机化世界分布"上做期望——**用世界覆盖不确定性**。一个说"真机交互贵，所以复用数据"，一个说"真机交互贵，所以别碰真机"。Day20 诚实标注了"论文无真机结果"，ADR 则是真机魔方的端到端 demo——两条路各有各的证据。
  - **vs Day12（Diffusion Policy）**：DP 完全不做 sim2real——它假设你有真机示范数据；ADR 假设你**没有**真机数据、只有仿真器。数据假设相反，数学工具也相反（DP 是"拟合示范分布"，ADR 是"在世界分布上做 RL"）。

## 为什么今天读它

Roadmap 把 Day21 放在"RL for Robotics / Sim2Real"块的第二天，标题是"Domain Randomization — visual/dynamics randomization for sim2real"。它是 Day07/08/19/20 共同依赖却从未正面拆解的地基：**sim2real 的"为什么能 transfer"**。对 Jun 的路线而言，它的战略意义是给你一个**可计算的 transfer 判据**——"真机参数是否落在随机化支撑集里"，以及一个**自动化的答案**（ADR）来回答"范围怎么定"这个最折磨人的工程问题。同时它和 Day22（RMA）的关系是"隐式 vs 显式自适应"的经典对照：ADR+LSTM 的 emergent meta-learning vs RMA 的 adaptation module——明天读 RMA 时直接对比。

## 今天的 3 问
1. DR 的数学到底是"鲁棒优化"（max-min）还是"贝叶斯平均"（期望）？transfer 发生的精确条件是什么？（提示：$\xi_{\text{real}} \in \text{support}(p_\phi)$；期望 vs 最坏情况，两种解读对应两种失败模式。）
2. ADR 的"边界扩张"规则为什么是一个 curriculum？它和固定 DR 的本质区别是什么？为什么论文说 memory-augmented（LSTM）策略在 ADR 下会涌现 meta-learning？（提示：随机化分布的熵 $H[p_\phi]$ 单调递增；LSTM hidden state 是对 $\xi$ 的在线后验。）
3. DR 和 RLPD 是两条相反的路——什么时候该"把仿真做宽"（DR），什么时候该"直接在真机上学"（RLPD）？（提示：交互成本 × 仿真可信度 × 接触复杂度，三轴决策。）

## 核心
1. **Motivation**: sim2real 的核心障碍是 **reality gap**：仿真器的接触动力学（摩擦、弹性、延迟、柔性）和真实世界永远对不准，而真机数据贵（时间/安全/硬件磨损）。两条经典出路——system identification（把仿真调准，Day24 主题）和 domain adaptation（用真机数据对齐）——都要真机数据。DR 问了一个反直觉的问题：**能不能不做准、只做宽？** Tobin 2017（IROS）先证明了视觉侧可行：纯仿真随机纹理/光照/相机位姿训出的物体检测器，直接转真机抓取。OpenAI 2018（Dactyl 方块重定向）证明了动力学侧可行：摩擦/质量/外观/延迟全随机化，Shadow hand 真机 vision-based 重定向成功。但手工 DR 有个致命工程痛点：**随机化范围是人拍的**——太窄，真机掉出去；太宽，策略学不出来（hedging 吃掉所有性能）。ADR 就是为杀掉这个"人拍参数"而生的。
2. **System / Method**: 系统 = **ADR（环境分布生成器）+ 分布式 PPO（OpenAI Five 同款栈）+ LSTM memory policy + vision pose estimator + Kociemba 子目标分解 + 为 ML 定制的硬件平台**。
   - **ADR**：对每个随机化维度 $i$（摩擦、质量、cube 尺寸、视觉材质、延迟……）维护区间 $[lo_i, hi_i]$，每个 episode 从均匀分布 $\xi_i \sim U(lo_i, hi_i)$ 采样环境参数。训练中定期在**区间边界**上评估策略表现：表现高于阈值 → 向外扩张边界；低于阈值 → 收缩。所有维度独立进行。效果：分布的熵只增不减（除收缩外），策略永远在"刚好能学会"的最难分布上训练——**自动 curriculum**。
   - **Policy**：LSTM（memory-augmented）+ PPO。输入是**指尖位置 + 视觉估计的 cube 位姿**（不是全关节本体感知——论文发现指尖位置就够了，这是个反直觉的输入选择）；输出 20 维连续动作（Shadow hand 24 自由度，腕部固定）。
   - **Vision estimator**：CNN 从 RGB 图像估计 cube 位姿 + sticker 角度。训练数据：先用训好的 policy 在仿真里 rollout 收集 1M 个 state，再用 Unity 渲染合成图像（label 来自仿真器 ground truth，免费）。真机 ground truth 用 PhaseSpace 动作捕捉给。
   - **任务分解**：魔方求解本身用 Kociemba 算法算出子目标序列（rotate top face / flip 等），RL 只学"执行子目标"的灵巧操作——**规划和控制解耦**，RL 不负责想步骤。
   - **硬件**：定制版 Shadow hand + RGB 相机阵列，为 ML 训练加固（能扛住 RL 训练初期的暴力探索）。
3. **Training / Data Details**: Sim 数据 = **全部**：policy 的 rollout 100% 来自随机化 MuJoCo 仿真（量级：block 任务用了约 100 年仿真经验 / 50 小时 wall-clock，8 V100 + 6144 CPU cores；魔方任务量级更大，论文称"数千年"人类等效经验）。Real 数据 = **零**（训练期不用任何真机数据——这是 DR 路线的定义性特征；真机只做评估）。Sim2Real = **ADR 本体**：vision estimator 和 control policy 都在 ADR 分布上训练，真机是"分布内的一个采样点"。Reward = 稀疏子目标奖励（完成一个 rotate/flip 给分）+ 密集 shaping；**verifiable signal**：子目标完成是二值的、可验证的——和 Day20 的 Adroit 稀疏奖励同类，但这里用 ADR + 大算力硬扛稀疏性。
4. **Key Tricks**: ① **ADR 边界规则只看"边界表现"**：不评估分布内部（内部迟早会被边界扩张覆盖），只在每个维度的 $lo_i/hi_i$ 处测——$O(d)$ 次评估管 $d$ 个维度，维度灾难被"边界即最难"这个单调性假设消掉。这是 ADR 工程上最聪明的一刀。② **LSTM 不是为了"记忆轨迹"，是为了"在线辨识"**：policy 的 hidden state 在 episode 内累积观测证据，隐式估计当前 $\xi$——论文明确报告了 emergent meta-learning：同一个 frozen policy，在测试时对未见过的扰动（绑手指、戴手套）表现出适应性行为。**这是 Day22 RMA 的"隐式版"**——RMA 把这件事显式化（adaptation module），ADR 让它从 LSTM 里长出来。③ **输入做减法**：只用指尖位置 + 视觉位姿，不用全关节角度——仿真里"测得准"的量才值得进 policy，测不准的量进了就是 reality gap 的入口。**DR 的哲学延伸到输入设计**：只信任那些"随机化后仍然可辨识"的信号。
5. **Results**: 真机完整打乱魔方 **60% 成功率**，极限难度打乱 20%（论文/博客原文）。鲁棒性 demo：戴橡胶手套、绑住几根手指、被毛绒长颈鹿捅——全是训练分布外的扰动，策略不崩。Vision estimator 真机位姿估计误差在可操作范围内（PhaseSpace ground truth 对照）。消融：无 ADR（固定窄随机化）的策略真机直接失败；无 memory（feedforward）的策略在宽分布上学不出来——**ADR 和 LSTM 是绑定的**，缺一不可。预算：OpenAI Five 级别的分布式 infra（数千 CPU + 数十 GPU，数周）——这是 DR 路线的"明码标价"。

## 数学视角

> 符号统一：$\xi \in \Xi$ 环境参数向量（摩擦系数、质量、延迟、纹理、光照……每个维度一个物理/视觉量）；$p_\phi(\xi)$ 参数为 $\phi$ 的随机化分布（ADR 里 $\phi = \{(lo_i, hi_i)\}_{i=1}^d$）；$p_\xi(s'\mid s,a)$、$p_\xi(o\mid s)$ 为参数 $\xi$ 下的转移/观测模型；$\pi_\theta(a\mid o, m)$ 为带 memory $m$ 的策略；$\xi_{\text{real}}$ 真实世界的（未知）参数；$J(\theta; \xi)$ 为固定 $\xi$ 下的期望回报。

### 1. DR 的目标函数：从"一个 MDP"到"MDP 分布"

标准 RL（Day19 的 PPO）：

$$J(\theta; \xi_0) = \mathbb{E}_{\tau \sim p_{\xi_0}(\cdot\mid\pi_\theta)}\left[\sum_t \gamma^t r_t\right],$$

$\xi_0$ 是仿真器的**一组**参数。DR 把它换成：

$$J_{\text{DR}}(\theta; \phi) = \mathbb{E}_{\xi \sim p_\phi(\xi)}\left[\,J(\theta; \xi)\,\right] = \mathbb{E}_{\xi \sim p_\phi}\mathbb{E}_{\tau \sim p_\xi(\cdot\mid\pi_\theta)}\left[\sum_t \gamma^t r_t\right].$$

直觉：**不再赌仿真器准，赌"真世界是分布里的一个采样"**。优化器被迫学一个"在所有 $\xi$ 上都还行"的策略——hedging 是内建的。

### 2. 两种解读：贝叶斯平均 vs 鲁棒优化（回答 Q1）

- **贝叶斯平均解读**：把 $p_\phi(\xi)$ 看成对 $\xi_{\text{real}}$ 的先验（epistemic uncertainty），$J_{\text{DR}}$ 是先验下的期望回报。transfer 条件：$\xi_{\text{real}} \in \text{support}(p_\phi)$ 且 $p_\phi$ 在那附近有足够密度——**先验盖住真相**。失败模式：先验拍错了（真机摩擦超出 $hi_{\text{friction}}$），策略在没见过的 $\xi$ 上行为未定义。
- **鲁棒优化解读**：当 $p_\phi$ 很宽时，期望近似于 $\min_{\xi \in \text{supp}}$ 的软版本——策略在学"最坏情况也别太烂"。失败模式：分布太宽 → 策略过度保守，**在简单 $\xi$ 上的峰值性能被 hedging 吃掉**（论文里魔方 60% 而不是 99%，部分原因在此）。

一句话：**DR 的 transfer 定理只有一句话——"真机参数落在随机化支撑集里"。** 所有 DR 工程（Tobin 的视觉随机化、Isaac Lab 的 randomization manager、ADR 的自动扩张）都是在**扩大支撑集**，同时**不让期望回报塌掉**。

### 3. ADR：熵单调递增的自动 curriculum（回答 Q2）

ADR 维护 $\phi = \{(lo_i, hi_i)\}_{i=1}^d$，每个 episode $\xi_i \sim U(lo_i, hi_i)$。更新规则（第 $i$ 维）：

$$\text{评估 } \bar J_i^{\text{hi}} = \mathbb{E}[J(\theta; \xi)\mid \xi_i = hi_i], \quad \bar J_i^{\text{lo}} = \mathbb{E}[J(\theta; \xi)\mid \xi_i = lo_i],$$

$$\text{若 } \bar J_i^{\text{hi}} > t_H \Rightarrow hi_i \leftarrow hi_i + \Delta_i \quad (\text{扩张}), \qquad \text{若 } \bar J_i^{\text{hi}} < t_L \Rightarrow hi_i \leftarrow hi_i - \Delta_i \quad (\text{收缩}),$$

$lo_i$ 对称处理。$t_H > t_L$ 是表现阈值。

为什么这是 curriculum：均匀分布的熵 $H[p_\phi] = \sum_i \log(hi_i - lo_i)$ ——每次扩张都增加熵，**环境分布的"难度"（熵）单调递增**，而策略的表现被阈值 $t_H$ 钉在"刚好能学会"的边界上。对比固定 DR：$\phi$ 是人拍的常数，策略要么太简单（学不到鲁棒性）要么太难（学不出来）；ADR 让 $\phi$ 成为**策略能力的函数** $\phi(\theta)$，难度永远比能力高一格。

为什么只测边界：若策略在 $\xi_i = hi_i$（该维度最极端）表现达标，则内部 $\xi_i \in (lo_i, hi_i)$ 大概率也达标（单调性假设）——$d$ 个维度只需 $2d$ 次边界评估，绕开维度灾难。

### 4. LSTM 的 emergent meta-learning：隐式的在线系统辨识（回答 Q2 后半）

Memory policy：$m_t = f_\theta(m_{t-1}, o_t, a_{t-1})$，$\pi_\theta(a_t \mid o_t, m_t)$。在 ADR 分布上训练时，最优策略的形式是**贝叶斯自适应策略**：

$$m_t \approx \text{belief}(\xi \mid o_{1:t}, a_{1:t-1}), \qquad \pi^*(a_t \mid o_t, m_t) \approx \pi^*(a_t \mid o_t, \xi \text{ 的后验}).$$

直觉：episode 开始时 $m_0$ 是均匀先验（"不知道摩擦多大"）；手一摸 cube，观测到的滑动/阻力就是关于 $\xi$ 的证据，$m_t$ 收敛到"当前世界"的点估计；策略据此调整发力。**这就是论文说的 emergent meta-learning**——不是训了个 meta-learner，而是"宽分布 + memory + RL"三者逼出了在线辨识。数学上它和 Day22（RMA）的 adaptation module 是同一个东西：RMA 把 $m_t \to \hat z_t$（latent dynamics）显式化、两阶段训；ADR 让它隐式地长在 LSTM 里。**明天读 RMA 时，这个对照是第一问。**

### 5. DR 的代价方程：鲁棒性不是免费的

设 $\xi_0$ 为"标称"仿真参数，$\theta^*_{\xi_0} = \arg\max J(\theta; \xi_0)$ 为单环境最优。DR 策略 $\theta^*_{\text{DR}} = \arg\max J_{\text{DR}}(\theta; \phi)$ 满足：

$$\underbrace{J(\theta^*_{\xi_0}; \xi_0) - J(\theta^*_{\text{DR}}; \xi_0)}_{\text{hedging 代价：在标称环境上变差}} \quad \text{vs} \quad \underbrace{\mathbb{E}_{\xi\sim p_\phi}[J(\theta^*_{\text{DR}}; \xi)] - \mathbb{E}_{\xi\sim p_\phi}[J(\theta^*_{\xi_0}; \xi)]}_{\text{鲁棒性收益：在分布上变好}}.$$

$p_\phi$ 越宽，左边越大（峰值性能越低），右边越大（transfer 越稳）。**ADR 的阈值 $t_H/t_L$ 本质上是在自动走这条 trade-off 曲线**——表现不够就不扩张（保住左边的性能），表现够了就扩张（去赚右边的鲁棒性）。这也是为什么 ADR 论文的魔方只有 60%：它优化的是"宽分布上的期望"，不是"标称环境上的峰值"。

### 6. 三条 sim2real 路线的决策表（回答 Q3：DR vs RLPD vs RMA）

| | DR/ADR（今天） | RLPD（Day20） | RMA（Day22 预告） |
|---|---|---|---|
| 真机数据 | 0（训练期不用） | 大量（在线交互） | 少量（ adaptation module 部署时在线用） |
| 仿真器要求 | 高（接触/动力学要"可随机化"，即参数化） | 无（可直接真机） | 高（privileged $\xi$ 训练） |
| 数学本质 | $\max_\theta \mathbb{E}_{\xi\sim p_\phi}[J(\theta;\xi)]$（世界分布上期望） | 混合经验分布上解 Bellman 最优（数据复用） | $\pi(a\mid o, \hat z), \hat z = g(\text{history})$（显式辨识） |
| 算力账 | 巨大（数千年仿真经验） | 中（真机步数 × UTD 20 的梯度） | 中（仿真 + adaptation 训练） |
| 失败模式 | $\xi_{\text{real}} \notin \text{support}$（先验拍错） | 探索不安全 / 奖励稀疏点不燃 | sim 里 privileged 信息真机不可测 |
| 适合 | 接触复杂、真机贵、仿真"结构对参数错"（灵巧手） | 仿真建不出来、真机可试错（部分操作任务） | 地形/载荷变化快、需要毫秒级适应（足式） |

决策直觉：**仿真器"结构对、参数错" → DR**（魔方、抓取）；**仿真器"结构都不对" → 别仿真，直接上真机 RLPD**；**环境变化比 episode 还快 → RMA 式在线辨识**。三者可以叠：Isaac Lab 里训 locomotion 就是"DR 先做宽 + 真机 RLPD 微调"的组合拳（Day07/08 的 H1/Humanoid-Gym 都是这个配方）。

### 7. 数学没有覆盖的东西（诚实标注）

- **单调性假设**：ADR"只测边界"依赖"边界最难"的单调性——真实物理里参数耦合（摩擦×质量×延迟联合效应）可能让最难的点不在边界上，ADR 对此沉默。
- **分布族的限制**：$p_\phi$ 是各维度独立均匀分布——真实世界的参数是**相关的**（重的东西往往摩擦也大），独立均匀是错的先验，只是"够宽所以够用"。
- **视觉 vs 动力学的不同数学**：Tobin 2017 的视觉 DR（纹理/光照随机化）本质是"让 CNN 学不变量"，动力学 DR 是"让 policy 学鲁棒反馈"——两者都叫 DR，但一个作用在表示、一个作用在控制，论文把它们混在一个框架里讲，数学上是两回事。
- **60% 的天花板**：hedging 代价（§5）+ 稀疏奖励 + LSTM 辨识的滞后——ADR 没有回答"宽分布上的期望最优"离"真机最优"还差多远。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？**框架贡献远大于模型**：ADR 的 policy 就是普通 LSTM+PPO，没有结构创新——transfer 的功劳全在"环境分布生成器"这个**训练框架**。证据：同一套 ADR 管线从 2018 block 重定向搬到 2019 魔方（任务复杂度跃升一个量级），只换了任务和随机化维度，框架不动。这正是"把泛化归因于训练配方"的又一个案例（呼应 README 问答里 Day13 vs Day14 的"接口即泛化 vs 配方即泛化"——ADR 是配方派最极端的形态：**连任务都不用换，换分布就行**）。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. **"把仿真做宽" vs "把数据做宽"**：ADR 的 $\xi \sim p_\phi$ 和 LLM post-training 的"prompt/环境多样化"是同一数学——都是在输入分布上做期望来换鲁棒性。做 VLA 的 RL 后训练时，"随机化仿真参数"和"随机化指令/场景"是同一枚硬币的两面，ADR 的边界扩张规则可以直接抄去做"课程式数据混合"。
  2. **LSTM 隐式辨识的教训**：memory + 宽分布 ⇒ 在线自适应免费出现——这对具身智能的 policy 架构有直接含义：**别急着把 adaptation 做成显式模块，先问宽分布 + memory 能不能长出来**。RMA（明天）就是"长不出来/不够快时才显式化"的答案。
- Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：① ADR 的算力账是"数千年仿真经验"——Infra 视角这是**用算力买数据**，和 Day20 RLPD"用梯度买数据"（UTD=20）是同一笔账的不同记法；② ADR 的边界评估是天然的**自动化评测门**：每个维度的 $\bar J_i^{\text{hi}}$ 就是 sim2real readiness 的分项指标——这正是 Day24（system identification + sim2real evaluation）要讲的"release gate"的雏形；③ 可复现性差：无官方代码，随机化维度全是论文附录里的表——这是 DR 路线至今的工程短板，也是 Isaac Lab 的 randomization manager（Day03）要解决的"把 DR 做成可复用算子"的问题。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：ADR 的阈值 $t_H/t_L$ 和扩张步长 $\Delta_i$ 是怎么定的？论文说"performance above a threshold"，但阈值本身是不是又成了新的"人拍参数"？——ADR 号称杀掉手工调参，但可能只是把"调随机化范围"变成了"调阈值"。有没有后续工作把阈值也自动化（比如按分位数自适应）？
- 如果要复现 / 小规模试，第一个实验做什么？在 Isaac Lab（Day03）里拿一个单维度做最小 ADR：固定其他参数，只随机化**地面摩擦**，实现"边界评估→扩张/收缩"循环，观察 $H[p_\phi]$ 曲线和策略成功率曲线的关系——这是整套 ADR 最便宜的"啊哈时刻"（一个维度的 entropy-performance 曲线）。

## 原文金句 (1-2句)
> "We demonstrate that models trained only in simulation can be used to solve a manipulation problem of unprecedented complexity on a real robot. This is made possible by two key components: a novel algorithm, which we call automatic domain randomization (ADR) and a robot platform built for machine learning."（摘要）
> "ADR automatically generates a distribution over randomized environments of ever-increasing difficulty."（摘要）

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移 ✅
- 保留并完善「和之前工作的关系」小节 ✅
- 数学视角：DR 目标函数（MDP 分布）、贝叶斯平均 vs 鲁棒优化、ADR 熵增 curriculum、LSTM 隐式辨识、hedging 代价方程、三路线决策表 ✅

## 连接
- 上一篇: day-20-2023-rlpd — RLPD：交互贵时的 off-policy 解法（用数据覆盖不确定性）；Day21 是其对偶——用世界分布覆盖不确定性，零真机数据
- 下一篇预告: day-22-2021-rma — RMA：rapid motor adaptation——把 ADR 里 LSTM 隐式做的"在线辨识"显式化为 adaptation module，毫秒级适应地形与载荷

## 问答补充
（本日暂无用户追问。）
