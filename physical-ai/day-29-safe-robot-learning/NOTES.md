# Day 29 — Safe Robot Learning：约束 MDP / CBF / Shield / Runtime Monitor（安全四层栈）

## 元信息
- Title: Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning（主综述，安全分类学与统一比较框架）+ Safe Reinforcement Learning via Shielding（对照：反应式安全修正）+ Control Barrier Function Based Quadratic Programs for Safety Critical Systems（对照：逐时刻安全证书）
- Authors / Org: Brunke* / Greeff / Hall / Yuan / Zhou / Panerati / Schoellig（U Toronto UTIAS；Annual Review of Control, Robotics, and Autonomous Systems，arXiv:2108.06266）；Alshiekh / Bloem / Ehlers / Könighofer / Niekum / Topcu（UT Austin / TU Graz / Bremen；AAAI 2018）；Ames / Xu / Grizzle / Tabuada（IEEE TAC 2017）
- Link / arXiv: 综述 — https://arxiv.org/abs/2108.06266；Shielding — https://arxiv.org/abs/1708.08611；CBF-QP — https://arxiv.org/abs/1609.06408
- 官方代码：safe-control-gym（UTIAS，PyBullet cart-pole/quadrotor + CasADi 符号动力学 + 可注入扰动的安全基准）— https://github.com/utiasDSL/safe-control-gym；Safety-Gymnasium（CMDP 约束 API 的大规模 safe RL 基准）— https://github.com/PKU-Alignment/safety-gymnasium
- Date read: 2026-09-20
- Tags: [physical-ai, safe-robot-learning, cmdp, constrained-rl, cbf, shielding, runtime-monitor, rta, simplex, safety-filter]
- Thread: physical-ai
- Folder: day-29-safe-robot-learning
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-29-safe-robot-learning

## 一句话总结
Day29 是 Physical AGI 块里的**安全日**：机器人从仿真走向真机、从实验室走向开放家庭（Day14 π₀.₅ 的方向），"成功"必须升级为"安全地成功"——这一天把安全拆成四层栈：**CMDP**（训练时期望约束， $J_c\le d$ ）、**CBF**（部署时逐时刻证书，前向不变集）、**Shield**（动作级反应式修正，最小干预）、**Runtime Monitor / RTA**（系统级仲裁，Simplex 切换备份控制器）；四层对应"安全"这个词的四种数学强度，而 Brunke 综述把控制理论与 RL 两大社区的方法第一次放进同一个可比较的框架（safe-control-gym 开源基准）。

## 和之前工作的关系

- **接了哪条线**：Day19（PPO）→ Day20（RLPD 真机在线 RL）→ Day21–24（sim2real）这一整条"真机部署线"的**缺失拼图**。PPO 的 clipped surrogate 只保证"更新别走太远"，RLPD 让真机在线 RL 变得样本高效，但两者都不回答：探索的第一步就撞坏硬件怎么办？Day29 是真机 RL 能落地的**前置条件**。
- **补了哪个短板**：Day28 的数学视角已经点名——benchmark 的 $S$ 只判"成功"不判"安全"，二项估计 $\hat{J}$ 完全不统计擦碰、超限、跌倒。CBF/shield/RTA 正是补"**怎么成功**"的那块：安全是一个独立的、可证书化的维度，不能被成功率平均掉。
- **替代 / 分叉 / 改进**：
  - vs Day19（PPO）：PPO 是 reward 最大化器；CPO（Achiam et al. 2017，CMDP 的信任域版本）是它的约束对偶——同一个 actor-critic 骨架，目标函数从 $\max J_r$ 变成 $\max J_r\ \text{s.t.}\ J_c\le d$ 。Day19 建立的"BC-MLE vs on-policy-RL"两轴，Day29 加上第三轴：**无约束优化 vs 约束优化**。
  - vs Day20（RLPD）：RLPD 的 50/50 对称采样 + 高 UTD 让真机在线 RL 可行；但真机 exploration 的每一步都是硬件风险。安全栈的工程含义：**RLPD 是油门，shield/CBF 是刹车**——没有刹车的在线 RL 不敢上真机。
  - vs Day06（DreamerV3）：Dreamer 在 latent 想象空间里 rollout，可以"免费"犯错；但想象里犯的错不带安全约束，解码到真机仍可能灾难。安全世界模型（safe MBRL）把 CBF 约束写进想象 rollout——"想"的时候就要守规矩。
  - vs Day21/24（DR / SimOpt）：DR 让 $p_\phi$ 覆盖真机，SimOpt 校准 $p_\phi$ ；但**覆盖 ≠ 安全**——即使测度对齐，单条轨迹仍可能出界。CMDP 的期望约束 vs CBF 的逐轨迹证书，正好对应"分布级安全 vs 轨迹级安全"。
  - vs Day25–27（Gato / GR00T / Cosmos）：模型越大、部署越开放（Day14 π₀.₅ 的未见家庭），运行时监护越从"可选"变"必备"——大模型的 $u_{\text{nom}}$ 可以直接套 CBF 滤波器做事后加装，这是 CBF 最大的工程红利。
- **对之前 Day X 的直接对比**：Day21 的 $J_{\text{DR}}=\mathbb{E}_{\xi\sim p_\phi}[J(\theta;\xi)]$ 是"期望性能"；Day29 的安全四层栈回答"期望之外的东西"——一次灾难的代价不是可平均的，安全需要比期望更强的数学（逐时刻不变集）或更冗余的架构（运行时仲裁）。

## 为什么今天读它

Roadmap Day29 的官方主题就是"约束 MDP、control barrier function、shield 和 runtime monitor 共同覆盖训练与部署安全"。它是 Day20（真机 RL）的安全前提、Day28（评测）的"安全维度"补完、Day30（数据飞轮）的 release gate 安全版。读完它，"安全"从一句口号变成一个**四层可计算栈**（见数学视角）。

## 今天的 3 问
1. **期望约束 vs 逐时刻证书，数学强度差几层？** CMDP 的 $J_c(\pi)\le d$ 只管期望——一条灾难轨迹可以被 1000 条安全轨迹平均掉；CBF 的 $h(x_t)\ge 0,\ \forall t$ 管每条轨迹的每个时刻。为什么前者"可平均"、后者"不可平均"？部署安全到底需要哪一层？
2. **CBF-QP 里 class- $\mathcal{K}$ 函数 $\alpha$ 的物理含义是什么？** 约束 $L_f h+L_g h\,u\ge -\alpha(h)$ 中取 $\alpha(h)=\gamma h$ （线性）时， $\gamma$ 大/小分别对应什么驾驶风格？保守和灵活的 trade-off 在公式里藏在哪？
3. **Post-shield 为什么不破坏 learner 的收敛保证？** Shield 在 learner 之后改写动作，agent 实际交互的已经是"被修正过的 MDP"了——Alshiekh 凭什么还能谈收敛？Minimal intervention 在这里起了什么关键作用？

## 核心

### 1. Motivation
- Baseline 为什么不行：Day19 的 PPO / Day20 的 RLPD / Day11–14 的 VLA 全是 **reward 最大化器**——它们的数学里没有"不许"这个词。真机上，一次关节超限可能烧电机、一次跌倒可能砸坏人形机器人（Day07/08 的 H1/XBot）、一次误抓可能伤人。仿真里 cost 只是数字，真机上 cost 是硬件和人身。
- 更深一层：安全和性能是**耦合但冲突**的——最快的抓取轨迹往往贴着奇异点和限位走。Ames 的原话逻辑：需要一个形式框架，把"安全条件"和"性能目标"统一进同一个实时优化问题，而不是事后打补丁。
- 和 Physical AGI 的关系：AGI 级别的具身智能必然部署在开放环境（Day14 π₀.₅ 的未见家庭、Day10 的 Habitat 协作）。开放环境 = 不可穷举的风险 = 必须有**运行时**的安全层。没有 Day29，Day25–27 的"通用"越大，部署风险越大。

### 2. System / Method（安全四层栈）

**第 0 层 — 问题定义：从 MDP 到 CMDP（训练时，期望级）**

$$\max_{\pi}\ J_r(\pi)\quad \text{s.t.}\quad J_{c_i}(\pi)\le d_i,\quad i=1,\dots,m$$

$J_r(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t r_t]$ 是 reward， $J_{c_i}$ 是第 $i$ 个代价（碰撞次数、关节超限、功耗）， $d_i$ 是预算。Lagrangian 对偶把它变成可训练的无约束问题：

$$\mathcal{L}(\pi,\lambda)=J_r(\pi)-\sum_{i=1}^m\lambda_i\big(J_{c_i}(\pi)-d_i\big),\qquad \lambda_i\ge 0$$

$\lambda_i$ 是"安全价格"：违反越多，价格越高，策略被推回安全区。CPO（Achiam et al. 2017）是 Day19 PPO 的约束版——在信任域内解带约束的 QP，保证每次更新**单调**地不增加约束违反。这是"把安全写进训练目标"的标准做法，Safety-Gymnasium 就是为这类算法建的基准。

**第 1 层 — CBF 安全滤波器（部署时，轨迹级证书）**

连续系统 $\dot x=f(x)+g(x)u$ ， $x\in\mathbb{R}^n$ ， $u\in\mathbb{R}^m$ 。定义安全集 $\mathcal{C}=\{x:h(x)\ge 0\}$ （如"躯干与人距离 ≥ $d_{\min}$ "）。 $h$ 是 control barrier function，若存在 extended class- $\mathcal{K}$ 函数 $\alpha$ 使

$$\sup_{u\in U}\big[L_f h(x)+L_g h(x)\,u+\alpha(h(x))\big]\ge 0,\quad \forall x$$

其中 $L_f h=\nabla h\cdot f$ 、 $L_g h=\nabla h\cdot g$ 是 Lie 导数（ $h$ 沿动力学的变化率拆成"漂移项 + 控制项"）。安全滤波器是一个 QP——**离 $u_{\text{nom}}$ 最近的安全动作**：

$$u^*=\arg\min_u\|u-u_{\text{nom}}\|^2\quad \text{s.t.}\quad L_f h(x)+L_g h(x)\,u\ge -\alpha(h(x))$$

定理：任何满足约束的 Lipschitz 控制器都让 $\mathcal{C}$ **前向不变**（forward invariant）——从安全集内出发，永远不出去。这是安全版的 Lyapunov：Lyapunov $V$ 证"收敛到目标"（ $\dot V\le -\gamma(V)$ ，往下走），CBF $h$ 证"不离开安全集"（ $\dot h\ge -\alpha(h)$ ，不许掉太快）——控制理论里最漂亮的一对对偶。

**第 2 层 — Shield（动作级，反应式修正）**

Alshiekh et al. 2018：给定 LTL 时序逻辑规范 $\varphi$ （如 $G(\neg(\text{gripper\_closed}\land \text{hand\_in\_grasp}))$ ——"永远不要在有人手时闭爪"），**合成**一个反应式系统 shield。两种部署位置：
- **pre-shield**：learner 做决策前，shield 先给出"安全动作集合"，learner 只能在其中选；
- **post-shield**：learner 输出动作后，shield 监控 \$(s,a)\$ ——安全就放行，不安全才改写为 $a'\in\text{Safe}(s)$ 。

核心性质 **minimal intervention**：只在违反时介入，平时零干扰。收敛保证的关键正在这里——post-shield 下 agent 看到的仍是 MDP（shield 只是环境动力学的一部分），且 shield 只在"必然违反"的动作上改写，learner 的探索空间几乎不受损。这是对 CMDP 的降维打击：不需要重训策略，**事后加装**。

**第 3 层 — Runtime Monitor / RTA（系统级，架构冗余）**

Simplex 架构（Sha 2001）+ ASTM F3269-17 run-time assurance：**复杂控制器**（ML 策略，性能强、不可验证）+ **monitor**（简单、可验证的谓词）+ **备份控制器**（简单、可验证的安全策略）。Monitor 检测到风险谓词为真 → 切换权交给备份控制器。Recovery RL（Thananjeyan et al. 2021）是"学出来"的版本：训一个安全 critic $Q_{\text{risk}}(s,a)$ ，超过阈值就切到恢复策略 $\pi_{\text{rec}}$ ——monitor 本身也是学的，但切换逻辑是硬的。

**四层的数学强度 ladder**（从弱到强）：

$$\mathbb{E}_\pi[\text{cost}]\le d \;<\; P_\pi(\text{violation})\le\delta \;<\; \forall t:\ x_t\in\mathcal{C} \;<\; \text{certificate}+\text{runtime arbitration}$$

（从左到右：期望级 → 概率级 → 逐时刻证书 → 证书+运行时仲裁。）CMDP 管期望，chance constraint 管概率，CBF 管逐时刻不变，RTA 再加一层"证书也可能错"的冗余。部署安全不是选一层，而是**按风险等级叠层**。

### 3. Training / Data Details
- **安全基准（verifiable signal 的来源）**：safe-control-gym——PyBullet 的 cart-pole / 1D / 2D quadrotor，稳定 + 轨迹跟踪两类任务；关键设计是**符号先验**（CasADi 符号动力学/约束/代价，控制理论方法可直接用）+ **可重复注入扰动**（输入/状态/惯性参数三处扰动，测鲁棒性）。Safety-Gymnasium——把 OpenAI Safety Gym 升级为约束信息 API 标准化的大规模 safe RL 基准（导航/操作多智能体任务族）。
- **Verifiable signal**：三维——任务性能 $J_r$ 、约束违反率/期望代价 $J_c$ 、干预率（shield/CBF 触发频率：触发太频说明 $u_{\text{nom}}$ 本身不安全）。论文比较的是**帕累托前沿**，不是单点 SOTA。
- **Sim 数据怎么来**：同 Day02/03 的仿真栈（PyBullet/MuJoCo）；安全任务的"难"不在物理精度，在**约束的符号化表达**（ $h$ 写不写得出来）。
- **Reward 设计**：CMDP 把"别撞"从 reward shaping 里拿出来，变成硬约束 $c_i$ ——这是方法论要点：shaping 是"劝"，约束是"罚"，证书是"锁"。

### 4. Key Tricks（最值得抄的 3 个）
1. **证书与策略解耦，事后加装**：CBF 滤波器 / post-shield 不碰 $u_{\text{nom}}$ 的训练——VLA、diffusion policy、PPO 策略训完直接套。这意味着 Day11–14 的任何策略都能"一夜变安全"，代价只是在线解一个小 QP。这是整个 Day29 工程价值最高的一句话。
2. **Minimal intervention 保留探索**：shield 只在"这个动作必然导致违反"时改写，平时完全透明——安全层不吃掉 learner 的样本效率。对比 CMDP-Lagrangian（训练全程被 $\lambda$ 拽着），shield 是"平时不管、犯规才吹哨"的裁判。
3. **把"别做"从 reward 里搬出来**：安全需求写成约束/证书/时序逻辑，而不是 reward shaping 的负权重。Shaping 的权重是玄学，约束的 $d_i$ 是合同——可审计、可验收，这正是 Day28"评测合同"思想在安全维度的复用。

### 5. Results
- **Brunke 综述**（safe-control-gym，cart-pole / quadrotor 约束稳定任务）：三类方法——① 学习不确定动力学（learning uncertain dynamics，模型越学越准）、② RL 中鼓励安全（CMDP/Lagrangian 系）、③ 证书化 learning-based control（CBF/Reachability 系）——都能在满足约束下完成任务；综述的结论不是"谁赢"，而是**三类方法的假设-代价对照表**：① 要模型结构先验、② 要大量安全违反样本（期望约束允许训练时犯错）、③ 要手写 $h$ /可达集计算（维度灾难）。这是选型手册，不是排行榜。
- **Shielding**（AAAI 2018）：在多个 RL 场景（含连续控制）上零违反地学到最优策略——post-shield 的干预只发生在早期探索，后期策略自己学会绕开 shield（"裁判下场次数递减"是健康信号）。
- **CBF-QP**（TAC 2017）：自适应巡航 + 车道保持，安全约束与执行器饱和同时满足——QP  mediation 的范式此后成为安全关键控制的标准件。
- **诚实注脚**：所有"零违反"都是**相对于建模了的** $h$ / $\varphi$ 而言的——没写进规范的风险，shield/CBF 一律看不见。这是形式方法的原罪，也是 RTA 要加架构冗余的原因。

### 数学视角（统一框架：安全 = 期望 → 证书 → 仲裁的三级跳）

**框架**：从 Day19 的 MDP 出发，安全是在优化问题上**逐步加码**：

$$\underbrace{\max_\pi J_r}_{\text{Day19 PPO}}\;\to\; \underbrace{\max_\pi J_r\ \text{s.t.}\ J_c\le d}_{\text{CMDP: expectation}}\;\to\; \underbrace{u^*\ \text{s.t.}\ \dot h\ge -\alpha(h)}_{\text{CBF: trajectory cert.}}\;\to\; \underbrace{\text{monitor}\triangleright(u_c,u_s)}_{\text{RTA: architecture}}$$

**符号全解**： $x\in\mathbb{R}^n$ 状态（关节角/位姿/速度）， $u\in\mathbb{R}^m$ 控制（力矩/末端速度）， $f(x)$ 漂移动力学（重力/科氏力）， $g(x)$ 控制矩阵（ $u$ 怎么影响 $\ddot x$ ）， $h:\mathbb{R}^n\to\mathbb{R}$ 安全函数（ $h\ge 0$ 安全）， $L_f h=\nabla h\cdot f$ 、 $L_g h=\nabla h\cdot g$ 为 $h$ 沿 $f$ / $g$ 的 Lie 导数（变化率）， $\alpha$ 为 extended class- $\mathcal{K}$ 函数（严格递增、 $\alpha(0)=0$ ，控制"允许 $h$ 下降多快"）， $u_{\text{nom}}$ 为性能策略（VLA/PPO/diffusion 输出的"想做的动作"）， $\lambda_i\ge 0$ 为第 $i$ 个约束的对偶价格， $Q_{\text{risk}}$ 为学到的风险 critic。

**为什么期望约束"可平均"而证书"不可平均"**： $J_c(\pi)=\mathbb{E}_{\tau\sim\pi}[\sum_t c_t]$ 是对轨迹分布取期望——单条灾难轨迹的 $c=10^6$ 可以被 1000 条 $c=0$ 的轨迹稀释；而 $\forall t: h(x_t)\ge 0$ 是对**每条轨迹每个时刻**的全称量词，没有求和就没有平均。这是测度论的基本事实：期望是积分，全称是逐点——安全要的是逐点。

** $\alpha$ 的物理直觉**（Q2 的答案）：取 $\alpha(h)=\gamma h$ ，约束变成 $\dot h\ge -\gamma h$ ，即 $h$ 至多指数衰减、衰减率不超过 $\gamma$ 。 $\gamma$ 大 = "允许快速逼近边界"（激进驾驶，贴着限位走，性能高但扰动一来就越界）； $\gamma$ 小 = "提前减速"（保守驾驶，离边界还远就开始刹车）。**保守-灵活 trade-off 在公式里就是 $\gamma$ 这一个旋钮**——CBF 把"多保守"从玄学变成参数。

**Shield 不破坏收敛**（Q3 的答案）：post-shield 下，learner 面对的仍是一个 MDP——只是转移函数被 shield"修正"了（ $P'(s'|s,a)=P(s'|s,S(s,a))$ ）。Alshiekh 的收敛论证依赖两点：① shield 是**确定性反应式系统**（给定 \$(s,a)\$ 输出唯一），修正后的环境仍满足 Markov 性；② minimal intervention 保证"安全动作原样通过"，最优安全策略在修正 MDP 中仍可达。换句话说，shield 没有缩小**安全策略类**，只剪掉了注定违反的枝——Q-learning 的收敛定理照用。

**贯穿例子：人形双臂端热汤（厨房帮手）**
- CMDP 层： $\mathbb{E}[\text{汤洒出量}]\le d$ ——"平均别洒"，训练目标。
- CBF 层： $h(x)=\|p_{\text{torso}}-p_{\text{human}}\|_2-d_{\min}$ ——"任何时刻躯干离人不小于 $d_{\min}$ "；VLA 输出 $u_{\text{nom}}$ 想伸手够远处的碗，QP 把它拉回——**性能策略不变，动作被投影到安全集**。
- Shield 层：LTL 规范 $G(\neg(\text{gripper\_closed}\land \text{hand\_in\_grasp}))$ —— diffusion policy  sampled 到一个"闭爪"动作而视觉显示人手在抓取区，shield 当场改写为"保持张开"。
- RTA 层：monitor 检测 ZMP 跌出支撑多边形（跌倒前兆）→ 切换备份控制器（保护性蹲姿 + 汤碗扶稳），ML 策略被隔离——**连 CBF 的模型都信不过时的最后一道门**。

**统一到之前 Days 的数学线**：Day06 Dreamer 的 RSSM 学的是信念 $b(s)$ （"世界长什么样"），Day29 的 $h$ / $Q_{\text{risk}}$ 学的是**危险度量**（"离灾难多远"）——一个管认知，一个管生存。Day21 的 $p_\phi$ 管"世界分布多宽"，Day29 的 $\mathcal{C}$ 管"状态空间哪里不许去"——分布覆盖 vs 集合禁区。Day28 的二项估计 $\hat{J}$ 回答"多成功"，Day29 的四层栈回答"多安全"——**成功率和安全是两个正交的测度**，Day30 的数据飞轮 gate 必须同时看两者。

**数学模型没覆盖的**：① CBF 要**手写** $h$ ——"离人不小于 $d_{\min}$ "好写，"别把热汤洒到小孩身上"难写成 $h(x)\ge 0$ ；学 CBF（learning CBF）正在补，但证书的"证"字就打了折；② 高相对度系统（relative degree > 1，如位置约束靠力矩控制，中间隔了两阶导）CBF 条件退化，需要 high-order CBF，公式变复杂、保守性上升；③ Shield 要**离散抽象**——连续机器人状态得先抽象成有限自动机，抽象本身可能漏掉真风险；④ RTA 的 monitor 谓词若太保守，备份控制器频繁接管，ML 策略等于没部署——"安全但无用"是 RTA 最常见的死法。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？安全方法的 transfer 形态特殊：**证书 transfer，策略不 transfer**——CBF 的 $h$ （如"关节限位""最小人机距离"）是物理常识，换机器人、换任务照用；而 CMDP 的 $\lambda$ / $Q_{\text{risk}}$ 跟任务绑定。Shield 的 LTL 规范是任务相关的，但"合成 shield"的**工具链**可复用。模型 vs 框架：这里**框架（四层栈的划分方式）贡献 >> 单个算法**。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. **安全是 infra 层，不是算法层**：以后设计任何真机实验，先画四层栈——训练用什么约束（CMDP）、部署套什么滤波器（CBF/shield）、monitor 谓词是什么、备份策略是什么。再强的策略，没有第 1/3 层就不上真机。
  2. **把 $\gamma$ （CBF 激进度）当超参调**：像调 PPO 的 clip 系数一样调 $\alpha$ ——先保守（小 $\gamma$ ）跑通，再逐步放开。这是"安全"从哲学变成工程的抓手。
- Infra 视角：可扩展性（CBF-QP 是每步一个小 QP，kHz 可解；shield 是查表/自动机，O(1)）/ 成本（safe-control-gym 的符号动力学让约束可审计，省掉真机试错成本）/ 评测自动化（Safety-Gymnasium 把 $J_c$ 变成标准 API 字段）/ 可复现性（约束写成 $h$ / $\varphi$ 就是可版本化的合同）。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：High-order CBF（相对度 > 1）在实际机械臂力矩控制里到底多保守？论文多是仿真和小车——人形机器人（Day07/08 的 H1/XBot）上有没有 CBF 落地的实证？这是 Q2 的实证缺口。
- 如果要复现 / 小规模试，第一个实验做什么：pip 装 safe-control-gym，跑通 cart-pole 的约束稳定任务；先训一个无约束 PPO 记录约束违反率，再套 CBF-QP 滤波器（手写 $h(x)=x_{\text{lim}}-|x|$ ）看违反率 → 0 且任务性能掉多少——亲手摸一下"安全税"（safety tax）。

## 原文金句 (1-2句)
> "Reinforcement learning algorithms discover policies that maximize reward, but do not necessarily guarantee safety during learning or execution phases."（Alshiekh et al., Shielding 摘要首句——整个 Day29 的合法性来源：reward 最大化器天生没有"不许"。）
> "Safety conditions are specified in terms of forward invariance of a set, and are verified via two novel generalizations of barrier functions."（Ames et al., CBF-QP 摘要——安全的数学定义：一个集合的前向不变性。）

## 今晚产出
- 按模板补齐 System（四层栈）/ Training（安全基准）/ Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节
- 数学视角：安全三级跳 $\max J_r \to J_c\le d \to \dot h\ge -\alpha(h) \to \text{monitor}\triangleright(u_c,u_s)$ ；期望可平均、逐点不可平均； $\alpha$ / $\gamma$ 是保守-灵活旋钮；shield 最小干预保收敛；贯穿"端热汤"四层实例

## 连接
- 上一篇: day-28-maniskill-robosuite-eval（ManiSkill3 / robosuite 可复现评测基准；Day28 的 $S$ 只判成功不判安全 → Day29 补安全维度）
- 下一篇预告: day-30-physical-ai-eval-data-flywheel（Physical AI Eval + Data Flywheel — 成功率 $\hat{J}$ 与安全四层栈合流，组成 release gate 的双测度）

## 问答补充
（本篇暂无 side chat 问答；跨篇通识问答见 README 问答记录。）
