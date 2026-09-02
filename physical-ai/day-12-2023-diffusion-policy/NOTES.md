# Day 12 — Diffusion Policy：条件动作扩散与 receding-horizon 视觉控制

## 元信息
- Title: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- Authors / Org: Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song / Columbia University, Toyota Research Institute, MIT（RSS 2023）
- Link / arXiv / Blog: https://arxiv.org/abs/2303.04137v5 / http://diffusion-policy.cs.columbia.edu
- Official code: https://github.com/real-stanford/diffusion_policy
- Date read: 2026-09-02
- Tags: [physical-ai, visuomotor-policy, diffusion, behavior-cloning, action-chunking, receding-horizon-control, robot-manipulation]
- Thread: physical-ai
- Folder: day-12-2023-diffusion-policy
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-12-2023-diffusion-policy

## 一句话总结
Diffusion Policy 把视觉条件机器人策略写成动作序列上的 DDPM：从高斯噪声迭代去噪出一段连续控制命令，只执行前 $T_a$ 步再观察并重规划；这同时保留示范中的多峰行为、整段动作的时间一致性和闭环纠错能力，在 IJRR 扩展版覆盖的 15 个仿真与真机任务上相对既有方法平均提升 46.9%。

## 和之前工作的关系

- **接了哪条线：** Day09 建立 VLA 总览，Day11 的 π₀ 再把 VLM 与连续 action chunk 接起来；Day12 回到更纯粹的 visuomotor behavior cloning，隔离出“条件生成式 action head + receding horizon”本身为什么有效。
- **补了哪个短板：** π₀ 的 flow matching 看起来像一个大模型部件，但 Diffusion Policy 把关键控制接口说得更清楚：observation horizon $T_o$、prediction horizon $T_p$、execution horizon $T_a$ 是三个不同时间尺度，chunk 不能等同于 open-loop 执行完整段。
- **替代 / 分叉 / 改进：** 它不学习 Day05/06 那样的环境 transition/world model，也没有语言语义或跨 embodiment backbone；它直接学习 $p(A_t\mid O_t)$。因此训练更像监督式行为克隆，代价是继承 demonstration coverage 与 covariate shift 的上限。
- **对之前 Day X 的直接对比：** Day11 的 π₀ 用连续时间 flow ODE 从噪声积分到动作，Diffusion Policy 用离散噪声日程和 DDPM/DDIM 反向去噪。二者都联合生成 action chunk，但 Diffusion Policy 每次只执行较短前缀并滚动重规划，闭环接口更显式；π₀ 则用 VLM、大规模跨机器人数据和 action expert 扩展语义与规模。

## 为什么今天读它

路线从 Day11–14 专题展开 VLA / 生成式 policy。Diffusion Policy 是连续动作生成的关键基线：它把“多峰示范分布”从一个统计问题，变成可部署的控制系统设计——动作序列表示、视觉条件缓存、去噪延迟、控制频率和重规划 horizon 必须联合选择。它也提供了一个很好的桥，把扩散/score matching 与 state-space control、MPC 式 receding horizon 放在同一张图里。

## 今天的 3 问
1. 为什么对多峰示范直接做单步 MSE 会输出“平均但不可执行”的动作，而对整段 $A_t$ 做 conditional diffusion 能在每次 rollout 中选定并坚持一个模式？
2. $T_p$ 预测得长有利于时间一致性，$T_a$ 执行得短有利于闭环响应；怎样把模型推理延迟、传感器频率和扰动时间尺度共同写成 horizon 选择问题？
3. 去噪 loss 只保证拟合专家动作分布，不保证动力学可行、安全或稳定；真实部署中应在哪一层加入碰撞约束、延迟补偿、uncertainty gate 与底层反馈控制？

## 数学视角：POMDP 上的条件轨迹生成 + receding-horizon control

### 1) State / observation / action / objective

把真实机器人写成部分可观测状态空间模型：

$$x_{t+1}=f(x_t,a_t,w_t),\qquad o_t=h(x_t,v_t).$$

- $x_t\in\mathbb R^{d_x}$：隐藏物理状态，包括机器人位姿/速度、物体位姿、接触模式、摩擦、液体或柔性物体状态。
- $a_t\in\mathbb R^{d_a}$：连续控制命令；论文任务从 2-DoF 平面动作到 6/7-DoF 末端位姿或关节/夹爪命令，双臂时维度继续增加。
- $o_t$：可观测量，通常是多视角 RGB 加 proprioception；图像可写成 $I_t\in\mathbb R^{H\times W\times3}$。
- $w_t,v_t$：未建模动力学扰动与传感噪声；控制时间 $t$ 是真实环境步，不是 diffusion step。

在时刻 $t$，策略输入最近 $T_o$ 步观测

$$O_t=[o_{t-T_o+1},\ldots,o_t],$$

联合预测长度 $T_p$ 的动作块

$$A_t^0=[a_t,\ldots,a_{t+T_p-1}]\in\mathbb R^{T_p\times d_a},\qquad \pi_\theta(A_t^0\mid O_t)\approx p_{\mathcal D}(A_t^0\mid O_t).$$

训练目标不是最大化环境 reward，而是用示范数据集 $\mathcal D$ 逼近专家条件动作分布。对 Push-T 而言，同一观测下可从 T 块左侧或右侧绕行；整段分布建模允许不同轨迹模式存在，而不是把两条轨迹平均到障碍物上。

### 2) Forward diffusion 与 noise-prediction objective

令 $k\in\{1,\ldots,K\}$ 是 diffusion/noise level，注意它与真实控制时间 $t$ 不同。把干净示范动作块 $A_t^0$ 加噪：

$$A_t^k=\sqrt{\bar\alpha_k}\,A_t^0+\sqrt{1-\bar\alpha_k}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I_{T_p d_a}).$$

其中 $\alpha_k=1-\beta_k$、$\bar\alpha_k=\prod_{i=1}^{k}\alpha_i$，$\beta_k$ 是 noise schedule。模型接收 $O_t,A_t^k,k$，预测噪声：

$$\mathcal L_{\mathrm{DP}}(\theta)=\mathbb E_{(O_t,A_t^0)\sim\mathcal D,\,k,\,\epsilon}\left[\left\|\epsilon-\epsilon_\theta(O_t,A_t^k,k)\right\|_2^2\right].$$

- $A_t^k,\epsilon,\epsilon_\theta\in\mathbb R^{T_p\times d_a}$；loss 同时覆盖全部未来步和动作维度。
- $K$ 是训练时的去噪层数。论文真机设置用 $K=100$ 训练、DDIM 10 步推理；这是生成时间尺度，不对应 100 个机器人动作。
- 论文实测 square cosine noise schedule 最好；它控制模型在不同频率/噪声尺度上学习动作信号的权重。

这也可视为条件 score matching：$-\epsilon_\theta$ 在尺度因子下近似 $\nabla_A\log p(A\mid O)$。推理不是一次回归均值，而是从 $A_t^K\sim\mathcal N(0,I)$ 出发，沿高概率动作流形迭代移动。

### 3) Reverse transition 与系统实现

论文把每次去噪写成 noisy gradient step：

$$A_t^{k-1}=\alpha_k\left(A_t^k-\gamma_k\epsilon_\theta(O_t,A_t^k,k)+\eta_k\right),\qquad \eta_k\sim\mathcal N(0,\sigma_k^2I).$$

- $\alpha_k,\gamma_k,\sigma_k$ 由 noise schedule 决定；直觉上 $\gamma_k$ 是沿 learned action-score field 的步长，$\eta_k$ 保留采样多样性。
- CNN 版本用 1D temporal U-Net/ConvNet 建模动作时间轴，并用 FiLM 注入 observation feature；Transformer 版本让动作 token 因果 self-attend，并对 observation embedding 做 cross-attention。
- 视觉编码器只对 $O_t$ 算一次，之后 $K$ 次去噪复用条件特征；论文的 end-to-end 版本使用修改的 ResNet-18（Spatial Softmax 保留空间位置、GroupNorm 避免 EMA 与 BatchNorm 冲突）。
- DDIM 将训练 100 步压到推理 10 步；论文报告 NVIDIA 3080 上约 0.1 s inference latency。真机 Push-T 以 10 Hz 产生 command，再线性插值到 125 Hz 执行。

### 4) Receding horizon：预测、执行、反馈三个时间尺度

生成 $A_t^0$ 后，只执行前 $T_a$ 步：

$$a_{t:t+T_a-1}\leftarrow A_t^0[0:T_a],\qquad t\leftarrow t+T_a,\qquad O_t\leftarrow\text{new observations},$$

然后再次采样。需要区分：

- $T_o$：看多少历史，决定部分可观测状态估计能力；
- $T_p$：预测多长，决定轨迹级时间一致性与隐式计划长度；
- $T_a$：一次真正执行多少步，决定 feedback bandwidth 与 inference amortization。

$T_a=1$ 最灵敏但每步都要完成去噪，延迟成本高；$T_a$ 太大则接近 open loop。论文消融中多数任务 $T_a=8$ 最优，并在仿真中维持到 4-step latency 的峰值表现。这与 MPC 相似：每轮优化一段未来控制，但只落地前缀；不同之处是 Diffusion Policy 的目标来自示范分布，而不是显式动力学模型和 cost function。

### 5) 假设与数学没有覆盖的真实误差

这个框架假设示范中的 action timestamp、相机帧和 proprioception 已正确同步，动作归一化和控制接口一致，且观测足以辨别任务阶段。它没有显式建模 actuator saturation、backlash、contact impulse、friction drift、camera exposure/occlusion、网络 jitter、碰撞和硬件安全约束；也不保证从训练分布外状态恢复。DDPM loss 能让动作“像数据”，却不等价于满足 $x_{t+1}=f(x_t,a_t,w_t)$ 下的可达性、稳定性或 constraint feasibility。因此真机还需要低层 position/impedance loop、限速/限力、collision checker、watchdog 与 OOD/uncertainty gate。

## 核心

1. **Motivation：显式回归和普通 implicit BC 各有硬伤**
   - 单一 Gaussian/MSE 容易在多峰动作上做均值；mixture/categorical action 随维度增加而难扩展。
   - IBC 的 energy-based policy 可表达多峰，但负样本近似 normalization constant 会带来训练不稳定。
   - Diffusion Policy 学 action distribution 的 score/noise，既不显式求 normalization constant，又能把输出扩展到高维动作序列。

2. **System / Method：conditioned action diffusion + closed-loop chunk execution**
   - 输入最近 $T_o$ 步视觉/状态，输出 $T_p$ 步连续动作；只执行 $T_a$ 步后重规划。
   - CNN 是默认稳健起点；高频变化或 velocity control 场景可用 time-series diffusion Transformer，代价是更敏感的调参。
   - position-control action 在论文中通常优于 velocity control：序列预测中的误差不会像速度积分那样持续累积，且 diffusion 更能容纳 position action 的多峰性。

3. **Training / Data Details：完全是离线示范学习，但数据与控制接口决定上限**
   - 仿真覆盖 robomimic 的 Lift/Can/Square/Transport/ToolHang、Push-T、Block Push 和 Franka Kitchen；既含 state policy 也含 visual policy、单臂/双臂、刚体/流体和短/长时程任务。
   - Robomimic 每个 proficient-human task 约 200 demonstrations，部分 mixed-human variant 约 300；Push-T 为 200，Block Push 为 1,000 scripted episodes，Kitchen 为 566 demonstrations。
   - 真机单臂包括 Push-T（136 demos）、mug flipping（250）、sauce pouring/spreading（各 90）；IJRR 扩展版还加入 egg beater、mat unrolling 和 shirt folding 双臂任务。
   - verifiable signal 仍是 task success、Push-T target area IoU、sauce coverage 等外部评测；noise-prediction validation loss 不能替代闭环 rollout。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — 三个 horizon 分离：** $T_p$ 负责计划/平滑，$T_a$ 负责闭环带宽，$T_o$ 负责状态估计；论文多数任务的 sweet spot 是执行 8 步再重规划。
   - **Trick 2 — 条件只编码一次：** 不对 observation-action 联合轨迹做 diffusion，而只扩散 action；视觉特征可跨 10 次去噪复用，使实时控制可行。
   - **Trick 3 — position action + receding horizon：** 位置命令降低误差积分，对感知/网络延迟也更鲁棒；命令在 10 Hz 生成、底层插值到 125 Hz。
   - **Trick 4 — Spatial Softmax + GroupNorm：** 从零端到端学 visuomotor feature 时保留图像空间位置，并避开 BatchNorm 与 EMA 的不稳定组合。
   - **Trick 5 — 整段采样后“mode commitment”：** 一个 rollout 的动作序列共同落进同一轨迹 basin，避免逐步从不同模式抽样造成左右摇摆。

5. **Results：强基线来自生成表示与控制系统共同设计**
   - 当前 arXiv v5 / IJRR 扩展版覆盖 15 个仿真与真机任务，报告相对既有 SOTA 平均 success-rate 提升 46.9%；原始项目页/RSS 版本表述为 12 tasks。
   - 仿真 visual policy 在复杂的 Transport / ToolHang 上提升明显；长期多峰指标上，Block Push `p2` 提升 32%，Kitchen `p4` 提升 213%。
   - 真机 Push-T 的 end-to-end Diffusion Policy 达 95% success、0.80 IoU，接近 human 的 1.00/0.84；相同表中 best LSTM-GMM 为 20% success，IBC 为 0%。
   - Mug flipping 在 20 trials 中成功率 90%；sauce pouring/spreading 的 success 为 79%/100%，并在 coverage 上接近 human。
   - 局限也很具体：训练仍受示范覆盖限制，去噪比 LSTM-GMM 等单步 policy 计算更贵；action chunk 只能部分掩盖延迟，对真正高频 torque control 仍不够。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？** Push-T 的遮挡和物理扰动实验显示短期恢复：被挡 3 秒仍继续完成、T 块被移动后会重新规划；但这不是跨任务 foundation-model transfer。结果主要证明 policy representation、action space、视觉训练和 control loop 的系统组合，而不是互联网语义或跨 embodiment 泛化。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：**
  1. policy serving 的正确单位不是单次 forward latency，而是 `observe → encode → K-step sample → queue Ta commands → execute → replan` 的 deadline graph；需要同时记录 p50/p95 sampling latency、observation age、action queue depth 和 interruption rate。
  2. post-training/data flywheel 应按 failure state 采集纠错示范，而不只增加成功轨迹；多峰 coverage、idle actions、阶段切换和 OOD recovery 都要成为数据标签与分桶评测。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：** 官方代码把 observation/action tensor 明确成 `(B,To,Do)` 与 `(B,Ta,Da)`，适合做 shape/schema contracts。可扩展训练还应版本化 normalizer、noise schedule、camera latency、horizon 配置和 controller mode；评测必须在固定 initial-condition buckets 上跑闭环成功率，不能只看 MSE。

## 疑问 / 下一步

- **没看懂 / 想深挖：** 怎样把固定 $T_a$ 改为 event-triggered replanning？例如接触突变、视觉残差、去噪样本方差或 safety monitor 触发时提前中断 chunk，同时避免频繁重采样造成动作不连续。
- **如果要复现 / 小规模试，第一个实验做什么？** 跑官方 Push-T low-dim Colab/仓库，以相同 demonstrations 比较 `single-step MSE BC` 与 `Diffusion Policy`；做 $T_a\in\{1,4,8,16\}$、DDIM steps $\in\{2,5,10\}$ 的矩阵，记录 success、轨迹 jerk、p95 inference latency 与受扰后恢复时间。
- **下一步：** Day13 读 Octo，观察 diffusion action head 放入 Open X-Embodiment 数据和通用 Transformer policy 后，真正增加的是跨任务/跨机器人 transfer，还是只是模型与数据规模。

## 原文金句 (1-2句)
> “Diffusion policy refines noise into actions via a learned gradient field. This formulation provides stable training, allows the learned policy to accurately model multimodal action distributions, and accommodates high-dimensional action sequences.”

> “This design allows the policy to continuously replan its action in a closed-loop manner while maintaining temporal action consistency — achieving a balance between long-horizon planning and responsiveness.”

## 今晚产出
- [ ] 画 `To observations → visual encoder once → K denoising steps → Tp actions → execute Ta → re-observe` 时序图
- [ ] 跑官方 Push-T low-dim smoke test，记录输入 `(B,To,Do)`、输出 `(B,Ta,Da)` 与 checkpoint config
- [ ] 比较 `MSE BC / Diffusion Policy` 在双峰二维 action toy 上的 mode coverage 与均值动作失败
- [ ] 扫描 `Ta={1,4,8,16} × DDIM steps={2,5,10}`，记录 success / jerk / p95 latency
- [ ] 注入 1–4 step observation delay 与一次 block perturbation，测恢复时间并定义 deployment gate

## 连接
- 上一篇: Day11 — π₀（VLM + flow matching + high-frequency action chunks）
- 下一篇预告: Day13 — Octo（Open X-Embodiment 上的开放通用机器人策略 + diffusion readout）
- 相关: Day05 UniSim；Day06 DreamerV3；Day09 RT-2 / OpenVLA

## 参考链接
- Paper: https://arxiv.org/abs/2303.04137v5
- Project: http://diffusion-policy.cs.columbia.edu
- Official code: https://github.com/real-stanford/diffusion_policy
