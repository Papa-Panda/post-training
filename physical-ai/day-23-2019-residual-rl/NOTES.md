# Day 23 — Residual RL：Residual Reinforcement Learning for Robot Control

## 元信息
- Title: Residual Reinforcement Learning for Robot Control
- Authors / Org: Tobias Johannink*, Shikhar Bahl*, Ashvin Nair*（*共同一作）, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, Sergey Levine（Siemens Corporation / UC Berkeley / Hamburg University of Technology；ICRA 2019）
- Link / arXiv: https://arxiv.org/abs/1812.03201（2018-12-07 v1，v2 2018-12-18；ICRA 2019 接收）/ 项目页（含视频）residualrl.github.io（论文内提及）
- Official code: 论文未提供官方实现仓库（诚实标注）；底层 TD3 用的是 rlkit 的公开实现，residual 框架本身与 RL 算法解耦
- Date read: 2026-09-14
- Tags: [physical-ai, residual-rl, robot-control, classical-control, feedback-control, actor-critic, td3, sim2real, contact-rich-manipulation]
- Thread: physical-ai
- Folder: day-23-2019-residual-rl
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-23-2019-residual-rl

## 一句话总结
真机学接触任务太贵、纯 RL 从零学太慢太危险——Residual RL 把控制问题拆成两半：**经典反馈控制器** $\pi_H$ 管"几何部分"（走到哪、跟轨迹，刚体动力学先验足够），**TD3 学的残差策略** $\pi_\theta$ 只管"接触部分"（推 block 不推倒、补偿摩擦），最终执行 $u=\pi_H+\pi_\theta$ 叠加；Sawyer 7 自由度臂在 **真机上只用约 8000 步（3 小时）** 学会把 block 插进两个会倒的立块之间——同样条件下手写控制器在 block 被转 ±20° 时成功率 2/20，residual RL 15/20。

## 和之前工作的关系

- **接了哪条线**：Day19–24 "RL for Robotics / Sim2Real" 块的第四天，roadmap 标题 "Residual RL — combine classical control priors with learned correction"。Day19（PPO/TD3 这类 actor-critic 是残差头的算法本体——本篇底层用的就是 TD3）；Day21（DR：把世界做宽来 robust）、Day22（RMA：privileged sim + 零真机交互的显式辨识）之后，Day23 给出第三条 sim2real/真机学习路线：**先验来自控制理论，真机只做小量在线学习**。
- **补了哪个短板**：Day19–22 的路线里，"先验"要么是数据（Day20 RLPD 的 offline data）、要么是随机化（Day21 DR）、要么是特权仿真（Day22 RMA）。但工厂里最现成的先验其实是**随机器人一起发货的经典控制器**（PID、阻抗控制、笛卡尔位置控制器）——论文原话："our method only requires a conventional controller for motion, which ships with most robots"。Residual RL 是第一套把这个"免费先验"形式化进 RL 目标函数的 canonical 方法。
- **替代 / 分叉 / 改进**：不是替代 DR/RMA/RLPD，而是**正交的先验来源轴**。三分法更新：① 先验=无 → Day21 ADR（世界做宽）；② 先验=privileged sim → Day22 RMA（零真机交互，只做 inference-time 辨识）；③ 先验=经典控制器 → Day23 residual（真机小量交互，权重继续学）；④ 先验=离线数据 → Day20 RLPD（真机在线更新权重）。四条路线的选择标准：你手里有什么免费的东西——可参数化仿真器？现成控制器？人类演示数据？什么都没有？
- **对之前 Day X 的直接对比**：
  - **vs Day22（RMA）**：RMA 真机上权重 frozen、"适应"=条件变量 $\hat z_t$ 更新；residual RL 真机上 TD3 权重**继续梯度更新**——"学"还在，只是被先验框在了一个很小的残差流形里。RMA 是"辨而不学"，residual 是"带着镣铐学"。
  - **vs Day20（RLPD）**：RLPD 的先验是**数据**（replay 里常驻 offline transitions），residual 的先验是**模型/控制器**（加法结构里常驻 $\pi_H$）。关键差异：RLPD 的 BC 初始化策略会**继承并可能遗忘**先验偏置（catastrophic forgetting）；residual 的加法结构让 $\pi_\theta$ 可以**主动抵消** $\pi_H$ 的偏置——"undo" 能力是结构自带的（见数学视角 §5）。
  - **vs Day12（Diffusion Policy）**：DP 是纯 behavior cloning（学 $p(A\mid O)$），residual 是 RL（优化回报）。DP 需要高质量演示、换任务重训；residual 需要可写的经典控制器 + 可定义的回报——**先验的形态决定了路线的形态**。

## 为什么今天读它

Roadmap Day23 的标题就是它。这是"控制理论 × 深度 RL"结合最干净的一篇：没有花哨架构，就一个加法 $u=\pi_H+\pi_\theta$，却同时回答了真机 RL 的三个硬问题——**样本效率**（先验吃掉几何部分）、**安全性**（先验稳定 $s_m$，探索被限制在小残差内）、**可解释接口**（$\pi_H$ 是工程师能看懂、能单独 debug 的模块）。对 Jun 的路线：① 这是"先验注入"最便宜的形态——不需要 teleop 数据（RLPD 的痛）、不需要可参数化仿真器（RMA 的痛）；② "把 MDP re-center 到先验周围再学"是一个通用设计模式，LLM 训练里 SFT-then-RL、RLHF 里 reference-policy KL 约束都是它的表亲；③ Day24（系统辨识）的前置：residual 学出来的 $\pi_\theta$ 本身就是"先验模型误差"的测量。

## 今天的 3 问
1. Algorithm 1 里 critic 学的 $Q(s,u)$ 中的 $u$ 到底指什么？把 $\pi_H$ 看成环境的一部分后，critic 实际在解哪个 MDP？（提示：wrapped MDP——$p'(s'\mid s,u_{\text{res}})=p(s'\mid s,u_{\text{res}}+\pi_H(s))$；Q 定义在残差动作上，TD3 本体一行不用改。）
2. 为什么 residual 结构能"undo"一个有偏的 $\pi_H$、理论上仍收敛到最优策略，而 BC 预训练再 finetune 的路线做不到？（提示：加法永远在线 vs 初始化继承偏置；finetune 还会 catastrophic forgetting 掉演示先验。）
3. $r_t=f(s_m)+g(s_o)$ 的"几何/接触"分解什么时候失效？（提示：当修正 $s_o$ 必须违背 $f$ 时——比如为了扶正 block 必须先偏离几何目标——$\pi_\theta$ 被迫和 $\pi_H$ 打架，"小残差"的高效论证就塌了。Block insertion 恰好可分：$\pi_H$ 管"走到缝隙"，$\pi_\theta$ 只管"轻推不倒"。）

## 核心
1. **Motivation**: 工厂里的机器人做重复任务，缺的是对不确定性的适应力。经典反馈控制（PID、computed torque）跟轨迹很高效，但一碰到**接触和摩擦**就脆——接触参数辨识极难，调参成本"可能和机器人硬件本身相当"（论文原话）。纯 RL 能处理接触，但从零学：① 真机交互贵且初期不安全；② block insertion 这种任务随机探索几乎撞不到成功。Residual RL 的赌注：**任务里"能被经典控制高效解决的部分"和"必须靠交互学的部分"是可分的**——前者用先验免费吃掉，RL 只学后者。
2. **System / Method**: 核心方程（论文式 5）：
   $$u=\pi_H(s_m)+\pi_\theta(s_m,s_o),$$
   $\pi_H$ 为手写控制器（只看机器人本体状态 $s_m$），$\pi_\theta$ 为神经网络残差策略（看 $(s_m,s_o)$，含物体状态）。执行流程（Algorithm 1）：$u_t=\pi_\theta(s_t)+\mathcal{N}_t$（探索噪声只加在残差上）→ 实际执行 $u'_t=u_t+\pi_H(s_t)$ → 存 $(s_t,u_t,s_{t+1})$ 进 replay → TD3 更新 $\theta$。注意：**replay 里存的是残差动作**，$\pi_H$ 被吸收进了环境转移核——RL 算法本体完全不用改。
   - **系统论分解**（论文式 1）：$[s_{m,t+1};s_{o,t+1}]=\big[[A(s_{m,t}),0];[B(s_{m,t},s_{o,t}),C(s_{o,t})]\big][s_{m,t};s_{o,t}]+D[u_t;0]$，$s_m$ 为全驱动机器人状态，$s_o$ 为欠驱动物体状态。$A,C$ 是已知的刚体动力学，**耦合矩阵 $B(s_m,s_o)$ 未知**——接触和摩擦就藏在 $B$ 里。经典控制能镇定 $s_m$（$f$ 部分），但对 $B$ 无能为力（$g$ 部分）。
   - **奖励分解**（论文式 4）：$r_t=f(s_m)+g(s_o)$，$f$ 为几何项（到目标距离/轨迹误差，先验可解），$g$ 为物体项（block 不倒、不移位，必须靠交互学）。
3. **Training / Data Details**: **底层 RL = TD3**（twin delayed DDPG，rlkit 公开实现；作者强调 residual 框架与 RL 算法解耦，可换任意算法）。Off-policy + replay 是真机可行的关键（sample efficient）。
   - **Sim**：MuJoCo，Sawyer 7 自由度臂 + 平行夹爪，笛卡尔空间位置控制器；两个立块各 3 自由度（顶部带斜角），可在 $x$/$y$ 滑动、绕 $y$ 轴倾倒；观测 = 末端位置 + 末端六维力/力矩 + block 位置 + 目标位置；初始状态夹爪已抓起待插入 block 悬在上方。Sim reward（式 6）：$r_t=-\lVert x_g-x_t\rVert_2-\lambda(\lVert\theta_l\rVert_1+\lVert\theta_r\rVert_1)$，$\theta_{l/r}$ 为左右立块相对桌面的倾角。
   - **Real**：compliant joint-space 阻抗控制器（自研，顺滑且容忍接触）；block 位姿来自**相机跟踪系统**（非真值）；因 block 轻且会滑动、Sawyer 测不到 $x$/$y$ 向接触力，观测里只保留 $z$ 向末端力。Real reward（式 7）在 sim 版基础上加立块位置误差 $-\mu\lVert X_g-X_t\rVert_2$ 和绕 $z$ 轴偏角项 $-\beta(\lVert\phi_l\rVert_1+\lVert\phi_r\rVert_1)$。**Verifiable signal**：block 是否保持直立且在原位——相机可测、人工可判定（成功率人工计数）。
   - **Real 数据量级**：从零真机训练约 **8000 步 ≈ 3 小时**收敛；sim 初始化后再做 residual，**不到 1000 步**真机交互即解。
4. **Key Tricks**: ① **残差动作进 replay**：$(s_t,u_t,s_{t+1})$ 存的是 $\pi_\theta$ 的输出，$\pi_H(s_t)$ 不进 buffer——数学上等价于把先验"编译"进环境，TD3 一行不用改，这是框架能即插即用的原因。② **信息分工是故意的**：$\pi_H$ 只看 $s_m$（几何），$\pi_\theta$ 看 $(s_m,s_o)$（接触）——先验的信息瓶颈不是缺陷，是设计：所有物体反馈只走学习通道，工程师 debug $\pi_H$ 时不用管接触。③ **探索噪声只加在残差上**：$\mathcal{N}_t$ 的幅度天然小（残差本身小），探索半径被先验框住——纯 RL 的"broad spatial variance"（论文原话，真机危险）在 residual 里被结构性消除。④ **有偏控制器实验**：$\sigma=0.05,\mu\in[0,0.2]$ 的偏置下 residual 性能恒定、手写控制器崩——残差学的是**积分动作/扰动抑制**，等于把 sensor drift 在线吃掉（论文明确点出 automation 里 sensor drift 的应用）。
5. **Results**: **Block insertion 任务**（把 block 插进两个立块之间，不碰倒、不移位）：
   - **样本效率**（sim + 真机）：residual RL 收敛更快、最终性能更高；纯 RL 要从零学位置控制结构，探索状态空间更广（真机危险）。
   - **环境变化**：立块初始朝向随机倾斜（sim：$U[-r,r],r$ 到 0.3 rad；真机：直立/顺时针/逆时针 ±20° 三档）——手写控制器 **2/20**，residual RL **15/20**；学到的是"consistent small corrective feedback behaviors…slightly nudge the blocks in the right direction without tipping them over"（论文原话）——**小修正反馈**，正是 residual 名字的由来。
   - **控制噪声/偏置**：输出加 $\mathcal{N}(\mu,\sigma^2)$ 噪声，$\mu=0,\sigma\in[0.01,0.1]$ 下性能几乎不受影响；偏置 $\mu\in[0,0.2]$ 下 residual 保持、手写控制器性能随偏置单调崩塌；真机偏置实验同样通过。
   - **Sim2Real**：$\pi_H$ 取"sim 里用 deep RL 训出的参数化策略"（block 固定场景），真机 residual **<1000 步**解决——"highly sample efficient, practical way to solve robotics problems with difficult contact dynamics"。
   - 失败/局限（论文 Discussion 诚实标注）：依赖精心搭建的视觉系统估计位姿，限制新场景应用；若手写视觉丢掉边缘等关键信息，学出的反馈控制器质量受限；未来方向是端到端视觉输入。

## 数学视角

> 符号统一：$s_t=(s_{m,t},s_{o,t})$，$s_m$ 全驱动机器人状态，$s_o$ 欠驱动物体状态；$u_t$ 控制输入；$\pi_H$ 手写控制器（$s_m\mapsto u$）；$\pi_\theta$ 残差策略（$(s_m,s_o)\mapsto u_{\text{res}}$）；$u'_t=u_{\text{res},t}+\pi_H(s_t)$ 实际执行；$r_t=f(s_m)+g(s_o)$；$B(s_m,s_o)$ 未知接触耦合矩阵；$\gamma$ 折扣因子。

### 1. 先验已知的世界：$A,C$ vs 未知的 $B$

论文式 (1) 把机器人+物体写成块矩阵系统。控制论视角：**$A(s_m)$ 和 $C(s_o)$ 是刚体动力学，先验已知**——经典控制（PID、computed torque、阻抗控制）镇定 $s_m$ 误差动态是指数稳定的（论文明确给出这个结论：忽略 $\pi_\theta$ 时，$\pi_H$ 给出 $s_m$ 子空间的指数稳定误差动态，前提是子空间可镇定）。**$B(s_m,s_o)$ 是接触/摩擦耦合，先验未知**——这就是为什么"调参成本堪比硬件"：工程师在用试错法拟合 $B$。Residual RL 的数学分工：$\pi_H$ 吃掉 $A$ 已知的部分（最大化 $f$），$\pi_\theta$ 只负责学 $B$ 未知的部分（最大化 $g$）。**先验的形状决定了学习的形状**——这是整篇论文的数学母题。

### 2. 核心方程：加法即接口

$$u=\pi_H(s_m)+\pi_\theta(s_m,s_o)\tag{5}$$

三个性质一次读懂：
- **叠加是线性接口**：$\pi_H$ 和 $\pi_\theta$ 之间唯一的契约是"加法"——工程师可以独立设计、测试、替换 $\pi_H$（比如从 PID 换成 MPC），$\pi_\theta$ 的训练代码不用动。这是 Day13 Octo"接口即泛化"在控制层的回声：**窄接口 + 强先验 = 可维护性**。
- **信息不对称是故意的**：$\pi_H$ 只看 $s_m$，$\pi_\theta$ 看 $(s_m,s_o)$。$s_o$ 的反馈只走学习通道——先验被故意蒙在鼓里，这样它的失败模式是可预测的（几何部分永远对），残差只需要"修正"，不需要"重做"。
- **$f/g$ 分解是设计假设，不是定理**：$r_t=f(s_m)+g(s_o)$ 要求任务可分——"走到缝隙"（$f$）和"轻推不倒"（$g$）在 block insertion 里恰好正交。如果任务不可分（见 §7），加法结构还在，但"小残差"的高效论证失效。

### 3. Wrapped MDP：先验被编译进环境（回答 Q1）

Algorithm 1 的关键细节：replay 存的是 $(s_t,u_t,s_{t+1})$，其中 $u_t=\pi_\theta(s_t)+\mathcal{N}_t$ 是**残差动作**，而环境转移发生在 $u'_t=u_t+\pi_H(s_t)$ 下。定义 wrapped 转移核：

$$p'(s_{t+1}\mid s_t,u_{\text{res},t})\;\triangleq\;p(s_{t+1}\mid s_t,\,u_{\text{res},t}+\pi_H(s_t)).$$

于是 critic 学的 $Q_\phi(s,u_{\text{res}})$ 是**以 $\pi_H$ 为新原点的 MDP** 的值函数：

$$Q^{\pi_\theta}(s,u_{\text{res}})=\mathbb{E}\left[\,r_t+\gamma\,Q^{\pi_\theta}(s_{t+1},\pi_\theta(s_{t+1}))\;\middle|\;s_t=s,u_{\text{res},t}=u_{\text{res}}\,\right],\quad s_{t+1}\sim p'(\cdot\mid s,u_{\text{res}}).$$

直觉：**residual RL = 标准 actor-critic，但坐标系被平移到了先验上**。TD3 的 twin critics、delayed update、target smoothing 全都不用改——因为"先验"在数学上只是一个环境包装器。这就是为什么论文说"Our method is independent of the choice of RL algorithm"：任何 RL 算法都可以套这个包装器。这也是统一框架：**凡是"已知结构 + 未知残差"的问题，都可以写成 wrapped MDP，把已知结构编译进 $p'$**。

### 4. 为什么样本高效：三重红利

1. **探索流形被先验锚定**：$\pi_H$ 把状态分布 $d^{\pi}(s)$ 钉在任务相关流形上（"走到缝隙"附近）。纯 RL 的初始 $d^{\pi_0}$ 是均匀随机游走——block insertion 的成功集在状态空间中测度极小，随机探索几乎撞不到。数学上：先验把**有效探索半径**从整个动作空间压缩到残差球 $\lVert u_{\text{res}}\rVert\le\delta$ 内。
2. **信用分配变短**：$f$ 部分的回报被 $\pi_H$ 提前饱和，critic 只需要拟合 $g$ 相关的残差回报——TD 误差的方差主要来自"推 block 那一下"，而不是"走过去那 50 步"。这就是 sim+真机学习曲线里 residual 前期陡峭的原因。
3. **安全性是结构性的**：$\pi_H$ 镇定 $s_m$ ⇒ 即使 $\pi_\theta$ 输出垃圾，总动作 $u'_t$ 仍被 $\pi_H$ 主导在稳定域附近。论文原话：纯 RL "needs to explore a wider set of states…potentially dangerous in hardware deployments"——residual 用**加法结构**把危险探索在数学上排除，而不是靠 reward shaping 祈祷。

### 5. Undo 偏置：残差是可学习的积分动作（回答 Q2）

设手写控制器带偏置 $\tilde\pi_H(s)=\pi_H(s)+b$（$b$ 常数或慢变，对应 sensor drift / 参数失配）。加法结构下残差可以学到：

$$\pi_\theta(s)\;\approx\;-b\;+\;\text{feedback correction}(s_m,s_o).$$

第一项是**常值抵消**——经典控制里这正是积分器的活（消除稳态误差）；第二项是真正的接触反馈。实验（式 8，$\mu\in[0,0.2]$ 偏置）证实：residual 性能恒定，手写控制器单调崩塌。

对比 BC-pretrain-then-finetune（Day20 RLPD 路线）：finetune 是**初始化**在先验上——梯度路径被先验偏置污染，且可能 catastrophic forgetting 掉演示知识（RRLfD, arXiv:2106.08050 明确指出 residual 形式能缓解这一点）。Residual 是**加法永远在线**：$\pi_H$ 永不消失，$\pi_\theta$ 永远有机会纠正它。**初始化是"起点"，加法是"结构"**——起点会被优化带偏，结构一直在场。这是回答 Q2 的核心：residual 理论上能收敛到最优策略（把 $\pi_H$ 完全 undo 掉：$\pi_\theta^*=\pi^*-\pi_H$），而 finetune 的收敛性依赖初始化盆地。

### 6. 四条路线的统一坐标系（Day19–23 串联）

| 路线 | 先验来源 | 真机上更新什么 | 真机交互量 |
|---|---|---|---|
| Day21 ADR | 无（世界做宽） | 什么都不更新（LSTM 隐式辨识是 inference） | 0 |
| Day22 RMA | privileged sim（$e_t$） | 条件变量 $\hat z_t$（inference），权重 frozen | 0 |
| Day23 Residual | 经典控制器 $\pi_H$ | 残差权重 $\theta$（gradient） | ~8000 步 / 3h |
| Day20 RLPD | 离线数据（replay 常驻） | $Q,\pi$ 权重（gradient） | 在线持续 |

横轴是"先验从哪来"（模型/控制器 vs 数据 vs 特权仿真 vs 无），纵轴是"真机上动什么"（权重 vs 条件变量 vs 不动）。**选路线 = 看你手里有什么免费的东西**：有可参数化仿真器 → RMA；有现成控制器 → residual；有人类演示 → RLPD；什么都没有 → ADR 把世界做宽。Day24（系统辨识）的伏笔：residual 学出的 $\pi_\theta$ 本身就是"$B$ 矩阵有多未知"的测量——辨识和残差是同一枚硬币的两面。

### 7. 数学没有覆盖的东西（诚实标注）

- **残差幅度无约束**："小残差"高效、安全，论文里是**涌现**的（学出来的是 small corrective feedback），不是**强制**的——目标函数里没有 $\lVert u_{\text{res}}\rVert$ 正则。如果 $\pi_H$ 很烂，$\pi_\theta$ 会被迫长大，"带着镣铐学"退化成"纯 RL + 一个常数偏置"。
- **$f/g$ 可分性假设**：若修正 $s_o$ 必须违背 $f$（比如为了扶正 block 必须先大幅偏离几何目标），$\pi_\theta$ 被迫和 $\pi_H$ 打架——此时最优残差不再是"小修正"，样本效率论证塌掉。Block insertion 恰好可分，这是任务选择的眼光，不是方法的普适性。
- **$\pi_H$ 的信息瓶颈**：$\pi_H$ 看不到 $s_o$ 是设计，但也意味着所有"几何目标本身该随物体变化"的场景（如目标缝隙在动）先验天然盲区——残差要同时干"重规划几何"和"接触修正"两份活。
- **视觉瓶颈**：论文 Discussion 明确承认——位姿来自精心搭建的相机跟踪系统，非端到端；视觉丢掉边缘信息会直接限制反馈控制器质量。这是 Day09–14 VLA 路线（端到端视觉）相对 residual 的优势区。
- **Wrapped MDP 的代价**：critic 仍要在 $(s,u_{\text{res}})$ 全空间学 $Q$——先验只帮了**探索**，没帮**函数逼近**。$Q$ 的拟合难度（接触动力学的非光滑性）一点没少，这也是为什么底层必须用 TD3 这种 sample-efficient 的 off-policy 方法。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？** 贡献在**框架**：$u=\pi_H+\pi_\theta$ + wrapped MDP + 奖励/信息分工。证据：① 同一框架被并发独立提出（Silver et al. "Residual Policy Learning", arXiv:1812.06298，仿真长程稀疏奖励任务）——框架收敛，任务各异；② 后续 RRLfD（arXiv:2106.08050）把 residual 从"控制器先验"推广到"演示先验"，框架复用；③ sim2real 实验里 $\pi_H$ 本身就是"sim 训出的 RL 策略"——先验可以是任何东西（手写、MPC、sim policy），接口不变。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发**：
  1. **"已知结构编译进环境"是通用模式**：LLM 的 SFT-then-RL（SFT 是 $\pi_H$，RL 是残差）、RLHF 的 reference-policy KL 约束（$\pi_{\text{ref}}$ 是先验锚点），数学上都是 wrapped MDP——先把已知的钉死，再学未知的。以后设计 post-training 流程，先问"我的 $\pi_H$ 是什么"。
  2. **加法 vs 初始化**：凡是"已有 baseline + 想改进"的场景（线上模型迭代、控制器升级），优先考虑**加法结构**（residual head / adapter）而非 finetune——前者保留 undo 能力、后者继承偏置。这和 LLM adapter/LoRA 的哲学同构。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性**：residual 把"工程师可维护的 $\pi_H$"和"数据驱动的 $\pi_\theta$"解耦——$\pi_H$ 可单元测试、可形式化验证（指数稳定是定理），$\pi_\theta$ 走数据飞轮；这是 Day29（安全）的前置思想：**安全关键的部分用可验证的先验 covering，学习只发生在安全包络内**。评测上"2/20 vs 15/20"的人工计数 + 相机可测位姿是接触任务评测的诚实写法（Day28 评测主题会 formalize）。

## 疑问 / 下一步

- **没看懂的 / 想深挖的 1 个问题**：$\pi_\theta$ 学出的残差场 $u_{\text{res}}(s)$ 的几何——在"block 已对准" vs "block 倾斜 20°"两种状态下，残差是"常值偏置抵消"主导还是"状态相关反馈"主导？把残差场画出来，能直接验证 §5 的"积分动作"解释 vs 纯反馈解释。
- **如果要复现 / 小规模试，第一个实验做什么？** MuJoCo 里最小闭环：① 写一个笛卡尔 PD 控制器当 $\pi_H$（setpoint = 目标缝隙上方，5 行代码）；② 用 stable-baselines3 的 TD3 学残差，任务 = 论文的简化版（固定立块，只学插入）；③ 先复现"无扰动下手写控制器 20/20、加 ±20° 扰动掉到 ~2/20、residual 拉回 15/20"的三段对比——先验证框架，再谈真机。

## 原文金句 (1-2句)
> "The final control policy is a superposition of both control signals."（摘要）
> "We believe this approach can accelerate learning of many tasks, especially those where the control problem can be solved in large part by prior knowledge but requires some model-free reasoning to solve perfectly."（Discussion）

## 今晚产出
- [x] 按模板补齐 System / Training / Key Tricks / Results / 可迁移（含数学视角 §1–7）
- [x] 保留并完善「和之前工作的关系」小节（Day19–23 四路线统一坐标系）
- [x] 论文逐节核验（arXiv HTML 全文 761 行已读，公式/数据/引用均来自原文非记忆）

## 连接
- 上一篇: day-22-2021-rma（RMA：privileged 两阶段 + 显式在线辨识，零真机交互）
- 下一篇预告: day-24-sim2real-system-identification（System Identification + Sim2Real Evaluation——把"先验有多准"变成 release gate；residual 学出的 $\pi_\theta$ 本身就是模型误差的测量）

## 问答补充
（本日 side chat 若有用户提问，在此归档；推送卡片类消息跳过。）
