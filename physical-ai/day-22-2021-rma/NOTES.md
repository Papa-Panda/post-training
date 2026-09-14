# Day 22 — RMA：Rapid Motor Adaptation for Legged Robots

## 元信息
- Title: RMA: Rapid Motor Adaptation for Legged Robots
- Authors / Org: Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik（UC Berkeley；RSS 2021）
- Link / arXiv: https://arxiv.org/abs/2107.04034（2021-07 preprint；RSS 2021 报告）/ 项目页（含视频）https://ashish-kmr.github.io/rma-legged-robots/
- Official code: 本次运行未检索到可直接验证的官方仓库 URL，不臆测拼接；实现细节与视频以作者项目页为准（诚实标注）
- Date read: 2026-09-13
- Tags: [physical-ai, rma, legged-robots, quadruped, motor-adaptation, privileged-learning, sim2real, unitree-a1]
- Thread: physical-ai
- Folder: day-22-2021-rma
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-22-2021-rma

## 一句话总结
腿式机器人换地形就摔、真机试错不起——RMA 把"适应"拆成两阶段 privileged learning：**Phase 1** 用 PPO 在仿真里"开天眼"训 base policy $\pi(a_t\mid x_t,a_{t-1},z_t)$，其中 $z_t=\mu(e_t)$ 是 17 维环境参数（质量/质心/电机强度/摩擦/地形高度）经 encoder 压成的低维"环境指纹"；**Phase 2** 用本体感知历史 $(x,a)$ 监督训练 adaptation module $\varphi$，"闭眼"从身体感受反推 $\hat z_t=\varphi(\text{history})$；部署时 $\varphi$ 以 10Hz 估计环境、$\pi$ 以 100Hz 生成期望关节位置、经 A1 的 PD 控制器转力矩。**同一个策略、零仿真标定、零真机微调**，Unitree A1 走过沙地/泥地/徒步小径/高草/土堆全部 trial 零失败，下楼梯 70%——训练时从未见过松软下陷地面、植被和楼梯。

## 和之前工作的关系

- **接了哪条线**：Day21 明确预告的"**显式自适应路线**"（vs ADR 的隐式路线、RLPD 的真机在线路线）；Day19（PPO：Phase 1 的算法本体就是 Day19 那套 clipped surrogate + GAE）；Day08（Humanoid-Gym：足式 RL locomotion 的训练配方，速度跟踪 reward + 并行仿真）；Day02/03（仿真器是 privileged $e_t$ 的来源——没有可参数化的仿真器就没有 RMA）。
- **补了哪个短板**：Day21 把 ADR+LSTM 的"隐式在线辨识"讲透了——但它是**涌现的、不可检查的**（hidden state 里到底辨识了什么，只能靠行为反推）。RMA 把"辨识"**显式化**为一个可监督训练、可可视化、可单独 debug 的模块 $\varphi$：你可以直接画 $\hat z_t$ 随地形变化的轨迹，验证它是不是真的在"辨识"。
- **替代 / 分叉 / 改进**：不是替代 DR，而是 **DR 之上的第二层**：Phase 1 训练本身就在随机化的 $e_t$ 分布上做（DR 把世界做宽），RMA 再加一层"部署时把分布坍缩成点估计 $\hat z_t$"（在线辨识把 hedging 赎回来一部分）。三条路线对照：DR = 世界做宽（Day21）；RLPD = 真机在线学（Day20）；RMA = 仿真预学辨识、真机零交互。
- **对之前 Day X 的直接对比**：
  - **vs Day21（ADR）**：数学上的"隐式 vs 显式"对照，见数学视角 §5——这是昨天留下的第一问，今天正面回答。
  - **vs Day20（RLPD）**：RMA 的 $\varphi,\pi$ 权重在真机上 **frozen**——"适应"（inference-time 条件变量 $\hat z_t$ 更新）vs "学习"（权重更新）的分野。RLPD 真机在线更新权重；RMA 真机只做"辨"，不做"学"。
  - **vs Day08（Humanoid-Gym）**：同一类任务（足式 locomotion）、同一类算法（PPO），Day08 的答案是"DR 做宽 + zero-shot"，RMA 的答案是"DR 做宽 + 在线辨识"——后者在未见过的极端地形上多一层保险。

## 为什么今天读它

Roadmap Day19–24 "RL for Robotics / Sim2Real" 块的第三天，标题就是"RMA — rapid motor adaptation under latent dynamics"。它是足式机器人 sim2real 从"**鲁棒**"（hedge 一切）走向"**适应**"（在线辨识再行动）的分水岭，也是 privileged learning（"Learning by Cheating" 家族）在 locomotion 上的 canonical 实现。对 Jun 的路线：① 理解"适应 vs 学习"的数学分野（这个区分在 LLM agent 的 in-context adaptation 里同样成立）；② "双时间尺度"（慢变量低频估计、快变量高频生成）是控制与生成里反复出现的设计模式，值得收进工具箱。

## 今天的 3 问
1. Privileged learning 为什么不直接训 $\pi(a\mid x, e)$（把 17 维环境参数直接喂给 policy），而要经 encoder $\mu$ 压成低维 $z$？（提示：$e_t$ 里大量维度冗余/相关；bottleneck 逼出"环境指纹"流形，也让 Phase 2 的回归目标更光滑——直接回归 17 维物理量是病态的。）
2. Phase 2 为什么必须用"当前 $\varphi$ 的预测去 unroll"迭代收集数据，而不是直接用真 $z$ unroll 收集 $(h,z)$ 对做监督学习？（提示：部署时 $\varphi$ 的输入分布是它自己预测失误后产生的轨迹——DAgger 式分布偏移；随机初始化 $\varphi$ 的 rollout 自带 exploration，专治"好轨迹上训出的辨识器一摔就懵"。）
3. RMA 的"适应"和 RL 的"学习"在数学上差在哪？为什么真机上 $\varphi,\pi$ 权重 frozen 也算 adaptation？（提示：adaptation = inference 时条件变量 $\hat z_t$ 更新；learning = 权重更新。RMA 把"学"全部留在仿真，真机只做"辨"——辨识是前向推理，不是梯度更新。）

## 核心
1. **Motivation**: 腿式机器人换地形就摔：沙地一脚踩空、重心一偏就是一次硬件损伤。真机试错成本让"多摔几次再优化"不可行——论文开宗明义：**收集 3–5 分钟真机行走数据做 adaptation 在实践中都不可行**，"适应"必须在仿真里预先学会、真机上零样本执行。Baseline 为什么不行：① 纯 DR 策略对所有环境用同一套保守步态（Day21 的 hedging 代价）；② system identification 需要停下来做激励实验、拟合参数——慢，还要真机交互；③ 手调 controller 遇到分布外地形直接崩。RMA 的赌注：**人换鞋走路不需要知道摩擦系数，但小脑一直在"辨"**——把这个"辨"显式学出来。
2. **System / Method**: 双系统 $\pi$（base policy）+ $\varphi$（adaptation module）。
   - **Privileged 环境向量** $e_t\in\mathbb{R}^{17}$：质量与质心位置（3 维）、12 个电机强度、摩擦系数（1）、局部地形高度（1）——全是仿真器里免费、真机上不可测的量。
   - **Encoder** $\mu: e_t\mapsto z_t$（MLP），$z_t$ 为低维连续"环境指纹"；**base policy** $\pi_\theta(a_t\mid x_t,a_{t-1},z_t,\text{cmd})$（MLP），$x_t$ 为本体感知（12 关节位置/速度、IMU 的 roll/pitch、二值足端接触、上一动作），输出 12 维**期望关节位置**，经 A1 自带 PD 控制器转力矩。$\mu$ 与 $\pi$ 在 Phase 1 联合端到端训练。
   - **Adaptation module** $\varphi$（CNN）：输入约 50 步本体感知+动作历史 $h_t=(x_{t-50:t},a_{t-50:t-1})$，输出 $\hat z_t$。**部署双频异步**：$\varphi$ 跑 10Hz，$\pi$ 跑 100Hz，$\pi$ 永远消费最新的 $\hat z_t$——A1 机载算力有限，CNN 不必每步跑。
3. **Training / Data Details**: **Sim 数据 = 全部**（RaiSim 并行仿真；论文未公开精确 GPU 小时账，不做数字断言）。
   - **Phase 1（RL）**：每 episode 随机采样 $e_t$（这本身就是 dynamics DR），$\mu,\pi$ 联合 PPO 最大化速度指令跟踪回报。Reward = 指令速度跟踪 + 能耗/关节平滑/足端惩罚——**verifiable signal** 是指令速度跟踪误差，仿真里直接可测。
   - **Phase 2（监督）**：先用真 $z$ unroll $\pi$ 收集 $(h_t,z_t)$；再用**随机初始化的 $\varphi$ 的预测** $\hat z$ unroll $\pi$ 收集第二批数据（带预测误差的探索轨迹），MSE 回归 $\min_\varphi\mathbb{E}\|\varphi(h_t)-z_t\|^2$，迭代至收敛——这是 DAgger 式的数据聚合，专治部署分布偏移。
   - **Real 数据 = 零**（训练期不用任何真机数据；部署期零微调——这是 RMA 路线的定义性特征，也是它和 Day20 最硬的分界）。
4. **Key Tricks**: ① **双频异步 = 时间尺度分离**：环境参数 $e_t$ 在 episode 内准静态（慢变量），步态控制 100Hz（快变量）——$\varphi$ 没必要每步重估；慢变量低频估计、快变量高频生成，是 edge 部署"按时间尺度分配算力"的通用原则。② **用"烂 $\varphi$"收集训练数据**：反直觉但关键——只在好轨迹上训出的辨识器，部署时一次预测失误就进入没见过的 $(h,z)$ 区域直接崩；用当前 $\varphi$（含误差）的预测去 unroll，训练分布自动覆盖"犯错后的恢复轨迹"。③ **纯本体感知、无视觉**：辨识不需要眼睛——"脚的感觉"就是地形的观测算子 $o=h(s;\xi)$，50 步历史里藏着摩擦/软硬/坡度的全部证据；输入做减法也是 sim2real（视觉的 reality gap 在这里被绕开，不是被解决）。④ **Encoder bottleneck**：17 维物理量压成低维 $z$，"环境指纹"流形让 Phase 2 回归良态——直接回归 17 维物理量，不同量纲、强相关，监督学习会病态。
5. **Results**: **同一个策略、无仿真标定、无真机微调**，Unitree A1 真机：沙地/泥地/徒步小径/高草/土堆**全部 trial 零失败**；沿徒步小径下楼梯 70%；水泥堆/石子堆 80%——而训练从未见过松软下陷地面、植被和楼梯。Berkeley/CMU 官方报道口径：**超越无 adaptation、纯 DR、系统辨识等对照系统**（"outperformed competing systems"），是"第一个完全基于学习、不依赖任何手写动作、从零开始在世界中探索交互来适应环境的腿式系统"。消融定性：去掉 $\varphi$（只用 DR）→ 未见地形性能塌；去掉 Phase 2 的迭代数据收集 → 部署时预测误差累积崩。

## 数学视角

> 符号统一：$s_t$ 真实状态；$x_t$ 本体感知状态；$a_t\in\mathbb{R}^{12}$ 期望关节位置；$e_t\in\mathbb{R}^{17}$ privileged 环境参数；$z_t=\mu(e_t)$ latent extrinsics（环境指纹）；$\hat z_t=\varphi(h_t)$ 在线估计；$h_t=(x_{t-50:t},a_{t-50:t-1})$ 历史窗口；$\xi$ 沿用 Day21 的"环境参数"记号，$e_t$ 是它的 privileged 可测版本。

### 1. Phase 1：开天眼的 RL（privileged teacher）

$$\max_{\theta,\psi}\; J(\theta,\psi)=\mathbb{E}_{e\sim p(e)}\,\mathbb{E}_{\tau\sim p_e(\cdot\mid\pi_{\theta,\psi})}\left[\sum_t\gamma^t r_t\right],\qquad \pi_{\theta,\psi}(a_t\mid x_t,a_{t-1},z_t),\; z_t=\mu_\psi(e_t).$$

直觉：先给 policy **开天眼**——"如果知道摩擦/载荷/地形，**应该**怎么走"。这是 "Learning by Cheating"（Chen et al.）在 locomotion 的实例：teacher 享受真机上不存在的信息，把"最优适应行为"先学出来。注意外层期望 $\mathbb{E}_{e\sim p(e)}$——Phase 1 本身就站在 Day21 的 DR 地基上：$e$ 每 episode 重采样，$\pi$ 被迫对"环境指纹流形"上的每一点都给出好动作。

### 2. 为什么经 encoder 压成低维 $z$（回答 Q1）

$e_t$ 的 17 维里：12 维电机强度高度相关、质量×质心×摩擦存在强耦合——直接喂给 $\pi$ 是冗余输入，直接做回归目标是病态的（量纲不一、共线性）。$\mu$ 的 bottleneck 做两件事：① **信息论**：逼出 $e$ 的低维充分统计量——"环境指纹" $z$ 是"给定 $z$，最优动作与 $e$ 条件独立"的压缩表示；② **优化**：Phase 2 的回归目标 $\|\varphi(h)-z\|^2$ 定义在光滑低维流形上，比回归 17 维物理量稳定一个数量级。代价：bottleneck 宽度是超参——太窄丢信息（重载荷 vs 大摩擦不可分），太宽 Phase 2 难学。

### 3. Phase 2：闭眼辨识（supervised system identification）

$$\min_\varphi\; \mathcal{L}(\varphi)=\mathbb{E}_{(h_t,z_t)\sim\mathcal{D}(\varphi)}\left[\,\|\varphi(h_t)-z_t\|_2^2\,\right],$$

其中数据分布 $\mathcal{D}(\varphi)$ **依赖于 $\varphi$ 自身**：用当前 $\varphi$ 的预测 $\hat z$ unroll $\pi$ 收集 $(h_t, z_t)$（$z_t$ 真值仿真器免费给），迭代至收敛。

直觉：$\varphi$ 学的是**贝叶斯反演的回归近似**——$\hat z_t\approx\mathbb{E}[z\mid h_t]$，即"给定身体 0.5 秒的感觉，环境指纹的后验均值"。脚陷进沙子里→关节位置偏离期望→历史 $h_t$ 里藏着"软"的证据→$\hat z_t$ 滑向"软地面"区域→$\pi$ 切换步态。这正是人小脑干的事：**你不需要知道摩擦系数是 0.4 还是 0.6，你的身体知道"该换步态了"**。

### 4. 为什么必须用"烂 $\varphi$"收集数据（回答 Q2）

若只用真 $z$ unroll 收集数据，$\mathcal{D}$ 全是"完美辨识下的好轨迹"——部署时 $\varphi$ 第一次预测失误，$(h_t,\hat z_t)$ 进入训练从未见过的区域，误差累积→摔。这是**行为克隆的分布偏移**穿了个马甲（Day12 的 DP 用 closed-loop 重规划治，Day17 的 DAgger 思想治）。RMA 的解法是 DAgger 式的：**用当前 $\varphi$（含误差）的预测去 unroll**，让训练分布主动包含"犯错后的恢复轨迹"；随机初始化的 $\varphi$ 第一轮提供最大熵 exploration。数学上这是在解一个不动点：$\varphi^*=\arg\min_\varphi \mathcal{L}(\varphi;\mathcal{D}(\varphi^*))$。

### 5. 隐式 vs 显式：正面回答 Day21 的第一问

| | Day21 ADR + LSTM（隐式） | Day22 RMA（显式） |
|---|---|---|
| 辨识载体 | $m_t=f_\theta(m_{t-1},o_t,a_{t-1})$，belief 藏在 hidden state | $\hat z_t=\varphi(h_t)$，belief 是显式向量 |
| 训练信号 | 只有 RL 回报——"辨识"是为回报服务的涌现行为 | RL 回报 + 监督回归 $\|\\varphi(h)-z\\|^2$——"辨识"有独立目标 |
| 可检查性 | 不可：$m_t$ 含义只能靠行为反推 | 可：直接画 $\hat z_t$ 随地形变化的轨迹验证 |
| 代价 | 不需要 privileged $e_t$ 的设计 | 需要仿真器能参数化 $e_t$ + encoder 设计 |
| 失效模式 | 分布外 $o_t$ → hidden state 漂移，无从 debug | $\hat z$ 估计误差传给 $\pi$ 未建模（A-RMA 2022 补的就是这块） |

一句话：**两者都是"在线辨识"，差别在辨识目标是涌现的还是被监督的**。RMA 用"多一个监督 loss"换来了可解释性和训练稳定性——这正是 privileged learning 的通用红利：仿真器里免费的真值，不用白不用。

### 6. 部署双频：时间尺度分离（回答"为什么 10Hz/100Hz"）

$$a_t = \pi_\theta(x_t, a_{t-1}, \hat z_{\lfloor t/10\rfloor}), \qquad \hat z \text{ 每 0.1s 更新一次}.$$

数学上这是**慢-快变量分离**：$e_t$ 在 episode 内准静态（$\dot e_t\approx 0$，地形不会每 10ms 突变），而步态动力学需要 100Hz 闭环。$\varphi$ 是 CNN、算力贵，10Hz 足以跟踪 $\dot e_t$ 的带宽——**采样定理视角**：辨识环路的 Nyquist 频率只需覆盖环境变化的带宽，不是步态的带宽。工程上这让 A1 的小机载电脑跑得动；理论上这是"按时间尺度分配算力"的通用原则（和 Day14 π₀.₅ 的"高层慢想、低层快做"分层 POMDP 同构）。

### 7. 适应 ≠ 学习：RMA 的权重在真机上 frozen

真机部署时 $\theta,\psi,\varphi$ **全部 frozen**，唯一的在线更新是 $\hat z_t=\varphi(h_t)$——这是**前向推理**，不是梯度更新。对比：
- **RMA（适应）**：$\hat z_t$ 是 inference-time 条件变量；"学"（权重）全部留在仿真。
- **Day20 RLPD（学习）**：真机在线梯度更新 $Q,\pi$ 权重。
- **Day21 ADR+LSTM（隐式适应）**：$m_t$ 在线更新，也是 inference-time，但载体不可解释。

这个三分法是 roadmap 后半程（Day23–30）反复出现的透镜：**凡是真机交互贵的地方，能用"辨"（inference）解决就别用"学"（gradient）**。

### 8. 数学没有覆盖的东西（诚实标注）

- **$e_t$ 必须被仿真器参数化**：RMA 的 privileged 建立在"仿真器结构对、参数可随机化"上——Day21 决策表里"仿真器结构都不对"（软体接触、复杂摩擦）的情形，$e_t$ 本身就是错的，$\varphi$ 辨识的是一个错误参数化的世界。
- **辨识误差传导未建模**：Phase 1 的 $\pi$ 吃的是**真** $z$，部署吃的是**估计** $\hat z$——$\|\hat z-z\|$ 的误差如何影响回报，RMA 没有显式处理。这是 A-RMA（2022，Cassie 双足）加的一步：**用 imperfect $\hat z$ 再 finetune $\pi$**，弥合"训练-部署辨识质量 gap"。
- **准静态假设**：10Hz 假设环境慢变；踩空、被绊这种**毫秒级突变**，$\hat z_t$ 跟不上——那 0.1s 里靠的是 $\pi$ 自身的鲁棒性（DR 地基），不是辨识。
- **$z$ bottleneck 的信息损失**：低维指纹可能把"重载荷+高摩擦"和"轻载荷+低摩擦"压到同一点——encoder 宽度是拍脑袋的超参，论文未给选择理论。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？** 贡献在**框架**：privileged 两阶段 + 显式辨识 + 双频部署。证据：同一框架被原样搬到灵巧手（HORA: Humanoid Object Rotation via RMA）、双足 Cassie（A-RMA），base policy 本体只是标准 PPO——换身体、换任务，框架复用，policy 重训。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发**：
  1. **"适应 ≠ 学习"的三分法是通用透镜**：LLM agent 的 in-context adaptation（prompt 里给 few-shot，权重不动）= RMA 的 $\hat z_t$ 更新；finetune = RLPD 的权重更新。以后看到任何"在线改进"系统，先问：它更新的是条件变量还是权重？
  2. **Privileged learning 是仿真/合成数据的通用范式**：凡是训练时有免费真值（仿真器参数、合成数据生成器的隐变量）的地方，都可以"先开天眼训 teacher，再闭眼训 student"——这和 Day04–06 世界模型的"用仿真真值监督表征"同构。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性**：Phase 2 的"用当前模型的错误分布收集下一轮数据"是**数据飞轮**的雏形——和 RLHF/DPO 的 on-policy 数据迭代同构，Day30 的 flywheel 主题在这里预演；双频部署是"按时间尺度分配算力"的 edge 范式；评测上 RMA 的"同一策略、零标定、多地形 trial 成功率"是 sim2real 评测的 gold 标准写法（Day24 系统辨识/评测主题会 formalize）。

## 疑问 / 下一步

- **没看懂的 / 想深挖的 1 个问题**：$z$ 的"环境指纹"流形几何——不同地形在 $z$ 空间里是否聚类？$\hat z_t$ 随地形变化的轨迹是检验"真辨识 vs 死记历史模式"的关键实验，论文的可视化值得单独看一眼。
- **如果要复现 / 小规模试，第一个实验做什么？** Isaac Gym（或 MuJoCo）里复刻两阶段最小闭环：① 训带 privileged $e_t$（先只做摩擦+载荷 2 维）的速度跟踪 base policy，$e_t$ 每 episode 随机化；② 训 $\varphi$ 从 50 步本体感知历史回归 $z$；③ 先在"高/低摩擦二分类"上验证 $\hat z$ 可分，再谈泛化。先别碰 17 维。
- **A-RMA（arXiv:2205.15299）**：2022 把 RMA 搬到 Cassie 双足，核心增量是"用 imperfect $\hat z$ finetune $\pi$"——直接回答本节 §8 的第二个诚实标注，值得在 Day23/24 对照读。

## 原文金句 (1-2句)
> "It's not an afterthought, but a forethought. That's our secret sauce."
> —— Ashish Kumar（Berkeley Engineering 官方报道原话；论文正文金句本次未逐字核对，不臆引）
>
> "Our RMA-enabled robot shows strong adaptation performance to previously unseen environments and learns this adaptation entirely by interacting with its surroundings and learning from experience. That is new."
> —— 同上出处。呼应本篇核心：适应发生在 inference 时与世界交互的 0.5 秒历史里，不发生在权重里。

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节（含 Day21 预告的第一问的正面回答）
- 数学视角 8 节：Phase 1 开天眼 RL、encoder bottleneck、Phase 2 贝叶斯反演、DAgger 式数据收集、隐式 vs 显式对照表、双频时间尺度分离、适应≠学习三分法、诚实标注 4 条

## 连接
- 上一篇: Day21 Domain Randomization / ADR（隐式在线辨识的涌现版本）
- 下一篇预告: Day23 Residual RL——"在模型控制器动作上学习 residual"：另一条"先验 + 学习"的 sim2real 路线，先验从"privileged 仿真"换成"经典控制器"，对照着看哪种先验更便宜。

## 问答补充（2026-09-13）

> 以下 3 组问答来自 2026-09-13 当晚用户对 Day22 RMA 卡片的追问（三问 quiz：Q1 用户先抛砖、助手拆三层；Q2/Q3 助手出题、用户作答、助手判卷），原样归档。

### Q1：为什么 $\varphi$ 预测 8 维 $z$ ，而不是 17 维 $e$ ？

**问题原文**："为了正则化？意思是形状更好预测一点 要不然真实的区域可能是一个形状奇怪的"

**字母表**：
- $e \in \mathbb{R}^{17}$ ：完整物理参数向量（摩擦、质量、电机增益等）——"物理学家眼中的世界"；
- $z \in \mathbb{R}^8$ ：encoder $\mu$ 从 $e$ 学出的低维环境指纹；
- $\varphi$ ：adaptation module，从本体感知历史 $h_t$ 预测 $\hat z_t = \varphi(h_t)$ ；
- $\pi$ ：base policy， $\pi(a \mid x, a_{t-1}, \hat z_t)$ ， $x$ 为本体感知观测。

**核心答案**：用户"形状奇怪"的直觉全踩中，三层意思拧紧：

1. **正则化（信息瓶颈）**：17 维的 $e$ 里很多组合对行为根本没区别——瓶颈逼着 policy 只依赖"真正影响动力学"的因子。
2. **"形状奇怪"（最深的一点）**：从本体感知历史出发，很多物理参数是**不可辨识**的。具体例子：躯干重了 0.5kg，和电机弱了 10%，从关节编码器 + IMU 的感受来看几乎一模一样——都是"腿变迟钝了"。 $\varphi$ 既分不清、也没必要分清。 $e$ 空间里"产生相同动力学的参数集合"是一个很扭曲的流形（多对一），而 $\mu$ 学出来的 8 维 $z$ 把它熨平了：只保留动力学上可区分的"环境指纹"，扔掉冗余自由度。所以不是 $e$ 不好，是 $e$ 的"真实有效区域"形状太怪，直接预测它等于让 $\varphi$ 去解一个病态问题。
3. **工程**： $e$ 的 17 个分量量纲不同（kg、摩擦系数、无量纲增益……），Phase 2 的 MSE loss 会被大尺度分量绑架；压到归一化的 8 维 $z$ ，loss 才 well-behaved。

**结论**： $e$ 是"物理学家眼中的世界"， $z$ 是"机器人身体感受到的世界"—— $\varphi$ 只能预测后者。

### Q2：Phase 2 为什么要用当前 $\varphi$ 的预测去 rollout，而不是只用真 $z$ 跑完美轨迹？

**问题原文**："Q2 不太懂"

**核心答案**：一句话——**因为 $\varphi$ 部署时一定会犯错，而"只在完美轨迹上训练"等于没收了它"犯错后如何恢复"的训练数据。**

naive 做法的崩溃链条：
1. Phase 1 的 $\pi$ 配上**真** $z$ 走得极其漂亮 → 在这些完美轨迹上收集 $(h_t, z_t)$ → 训 $\varphi$ ；
2. 部署时 $\varphi$ 第一步就可能把 $\hat z$ 估偏一点（比如把沙地误判成硬地）；
3. $\pi$ 拿到错误的 $\hat z$ ，用了硬地步态 → 脚陷进沙子 → 机身前倾；
4. 此时历史 $h_t$ = "下陷 + 前倾 + 挣扎"，**这种历史在完美轨迹数据里从没出现过**；
5. $\varphi$ 被问了一个训练分布外的问题 → 输出更离谱 → 下一步更错 → 雪崩。

这就是行为克隆的 compounding error / covariate shift：**训练分布（专家的状态分布）≠ 部署分布（学生自己诱导出的状态分布）**。Day12 的 Diffusion Policy 用 closed-loop 重规划治的是同一个病，RMA 用的是 DAgger 的思路。

DAgger 式解法：
1. 用**当前** $\varphi_k$ 的预测 $\hat z$ 去 rollout（注意：仿真器**免费知道真 $z_t$ **，这是 privileged 信息）；
2. $\varphi_k$ 会犯错 → 机器人踉跄、下陷、挣扎 → 这些"犯错 + 恢复"的历史 $h_t$ 被收集下来，**标注上真 $z_t$ **（如"沙地"）；
3. 在这批数据上训出 $\varphi_{k+1}$ ，重复。附赠彩蛋：第一轮 $\varphi_0$ 是随机初始化的，随机预测 $\hat z$ ≈ 最大熵 exploration，白送探索。

公式点睛就在这里：

$$\mathcal{L}(\varphi) = \mathbb{E}_{(h_t, z_t) \sim \mathcal{D}(\varphi)}[\|\varphi(h_t) - z_t\|_2^2]$$

注意期望的下标 $\mathcal{D}(\varphi)$ ——**数据分布本身依赖于正在优化的 $\varphi$ **，这不是普通监督学习，是一个不动点问题 $\varphi^* = \arg\min_\varphi \mathcal{L}(\varphi; \mathcal{D}(\varphi^*))$ 。DAgger 就是解这类问题的标准姿势。

**用户自己答对的题眼**："真实世界中完美的 $z$ 不存在——训练时假装它存在，部署时就要还债。"

**判卷校准**：用户说" $\pi$ 没有在不准确的 $z$ 上训练过"——对了一半。 $\pi$ 在 Phase 1 确实只见过真 $z$ ，但崩溃的**直接导火索**是 $\varphi$ 那一环： $\varphi$ 在没见过的 $h$ （下陷+挣扎）上输出垃圾 $\hat z$ ， $\pi$ 拿到垃圾输入才跟着错。整个 $\varphi \to \pi$ 链条在完美数据下训练，对"辨识误差"零容忍、零恢复余量。DAgger 式迭代就是给系统**买容错**：提前在训练时摔够，部署时才摔不倒。

### Q3：部署时 $\varphi$ 和 $\pi$ 的权重完全 frozen，为什么这仍然叫 adaptation？

**问题原文**："这个简单 因为z事变的 RL在学习真实的物理参数"

**核心答案**：

**对的部分**：变的是条件变量 $\hat z_t$ ，不是权重——部署时 $\varphi, \pi$ 的参数 frozen，但 $\hat z_t = \varphi(h_t)$ 每 0.1 秒更新一次， $\pi(a \mid x, a_{t-1}, \hat z_t)$ 的行为跟着变。

**校准一句**："RL 在学习真实的物理参数"——RL 学的不是物理参数本身，而是**权重** $\theta$ （策略/价值函数的参数）： $\theta \leftarrow \theta + \eta \nabla J$ ，函数本身变了。RMA 的对比是：

- **Learning**：函数变（权重更新），同一个输入 → 不同输出；
- **Adaptation**：函数不变（权重 frozen），输入的条件变（ $\hat z_t$ 更新）→ 不同输出。

类比： $\pi$ 是司机， $\varphi$ 是副驾驶。副驾驶不断报路况"现在是沙地"（ $\hat z_t$ ），司机切换驾驶模式——司机本人没重新学开车，但行为适应了路况。**适应的定义看的是"行为是否随环境变"，不是"权重动没动"。**

**与 Day21 的关系**：Day21 ADR 里 LSTM 的 hidden state $m_t$ 干的正是这件事——inference 时更新的条件变量。RMA 只是把它从黑盒 hidden state 里拎出来，变成可监督、可可视化的显式向量 $\hat z_t$ 。隐式 → 显式，这就是 Day21 到 Day22 的那条线。

**三条 sim2real 路线齐了**：Day20 RLPD（真机学，权重更新）、Day21 ADR（仿真做宽，隐式适应）、Day22 RMA（仿真做宽 + 显式辨识，条件变量更新）。
