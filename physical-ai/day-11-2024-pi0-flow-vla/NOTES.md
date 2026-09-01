# Day 11 — π₀：用 flow matching 生成高频连续动作块的 VLA

## 元信息
- Title: π₀: A Vision-Language-Action Flow Model for General Robot Control
- Authors / Org: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky / Physical Intelligence
- Link / arXiv / Blog: https://arxiv.org/abs/2410.24164 / https://www.physicalintelligence.company/blog/pi0
- Official code: https://github.com/Physical-Intelligence/openpi
- Date read: 2026-09-01
- Tags: [physical-ai, vla, pi0, flow-matching, action-chunking, cross-embodiment, dexterous-manipulation, robot-foundation-model]
- Thread: physical-ai
- Folder: day-11-2024-pi0-flow-vla
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-11-2024-pi0-flow-vla

## 一句话总结
π₀ 把 PaliGemma 的视觉—语言语义与一个 300M action expert 接在同一 Transformer 中，用 conditional flow matching 一次生成 50-step 连续动作块；再以 10,000+ 小时、7 类机器人配置、68 个任务的跨 embodiment 数据预训练，并用少量高质量数据 post-train，使单一 3.3B VLA 能在最高 50 Hz 的双臂精细操作中兼顾语义、动作流畅度和错误恢复。

## 和之前工作的关系

- **接了哪条线：** Day09 用 RT-2 / OpenVLA 建立“视觉 + 语言 → 动作”的 VLA 总览；Day11 专门深挖连续动作生成这一分支，回答 action token 在高频、双臂、接触丰富任务上的限制。
- **补了哪个短板：** Day09 的 OpenVLA 把每个动作维度离散成 token，单帧自回归输出 7D action；π₀ 直接建模条件分布 $p(A_t\mid o_t)$，一次联合生成整段 $H=50$ 的连续动作，因此能表达跨时间、跨关节耦合与多峰策略。
- **替代 / 分叉 / 改进：** π₀ 不替代 Day07/08 的 torque / locomotion 稳定环，也不显式学习 Day04–06 那种环境 transition model；它是低频语义条件与中频 manipulation command 之间的生成式 policy。真实电机安全、力控和碰撞约束仍由底层 controller 兜底。
- **对之前 Day X 的直接对比：** RT-2 / OpenVLA 的主轴是“动作即 token”，π₀ 的主轴是“动作块即条件生成轨迹”。前者复用 next-token CE infra；后者用 10 次 flow integration 换取连续精度和 action-chunk 联合建模。

## 为什么今天读它

路线保留 Day09/10 的总览后再进入 Day11–18 分专题扩展。π₀ 是 VLA 从离散动作 token 走向连续生成式控制的关键节点，也把 LLM 式 pre-training / post-training 配方直接迁移到机器人：广而杂的预训练数据提供覆盖和恢复行为，高质量 post-training 数据提供流畅、稳定、任务特定的执行风格。这和 Jun 当前关注的 data flywheel、post-training、rollout infra 有直接可迁移性。

## 今天的 3 问
1. 为什么对高频双臂 manipulation，联合生成 $H=50$ 的连续 action chunk 会比逐动作、逐维 token 解码更合适；收益来自连续精度、时间耦合，还是推理摊销？
2. 预训练混合中“低质量但多样”的纠错轨迹和 post-training 中“高质量且一致”的示范如何配比，才能同时避免笨拙与脆弱？
3. action chunk 有 0.5–0.8 秒 open-loop 执行窗口；接触、遮挡或人类干预使真实状态偏离后，何时应提前重规划，怎样把 uncertainty 接到底层 safety controller？

## 数学视角：统一成一个部分可观测的条件生成控制问题

### 1) State / observation / action / objective

真实系统可视为 POMDP：隐藏物理状态

$$x_t=(q_t,\dot q_t,\text{object poses},\text{contact modes},\text{friction},\text{deformation},\ldots),$$

但 policy 只能看到

$$o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t].$$

- $I_t^i\in\mathbb{R}^{h\times w\times 3}$：第 $i$ 个 RGB 相机，论文每个平台用 $n=2$ 或 $3$ 个视角。
- $\ell_t=(w_1,\ldots,w_L)$：长度为 $L$ 的语言 token 序列，既可以是总任务，也可以是约 2 秒粒度的 segment annotation / 高层子指令。
- $q_t\in\mathbb{R}^{d_q}$：本体状态，主要是关节角；跨机器人训练时 padding 到最大维度 18，并 mask 不存在的 image slot。
- 单步动作 $a_t\in\mathbb{R}^{d_a}$，整段动作为 $A_t=[a_t,\ldots,a_{t+H-1}]\in\mathbb{R}^{H\times d_a}$；$H=50$，$d_a$ 随 embodiment 变化并 padding 到统一维度。

行为克隆目标不是预测一个均值动作，而是学习完整条件分布：

$$\pi_\theta(A_t\mid o_t)\approx p_{\mathcal D}(A_t\mid o_t).$$

这很重要：同一件衣服可能向左或向右展平，双臂绕过障碍也可能有多条可行轨迹；普通 MSE 容易平均出不可执行动作，flow matching 则允许多峰连续分布。

### 2) Conditional flow matching

从噪声 $\epsilon\sim\mathcal N(0,I_{H d_a})$ 到示范动作块 $A_t$ 建立线性概率路径：

$$A_t^{\tau}=(1-\tau)\epsilon+\tau A_t,\qquad \tau\in[0,1].$$

对应的目标速度场为 $u=A_t-\epsilon$（若采用反向时间参数化，符号相反），训练 action expert $v_\theta$：

$$\mathcal L_{\mathrm{FM}}(\theta)=\mathbb E_{(o_t,A_t)\sim\mathcal D,\epsilon,\tau}\left[\left\|v_\theta(A_t^\tau,o_t,\tau)-(A_t-\epsilon)\right\|_2^2\right].$$

- $A_t^\tau\in\mathbb R^{H\times d_a}$：第 $\tau$ 个 flow time 的 noisy action chunk；这里的 $\tau$ 是生成积分时间，不是机器人控制时间 $t$。
- $v_\theta\in\mathbb R^{H\times d_a}$：模型预测的整段动作速度场；所有 $H$ 个 action slots 彼此双向 attention，因此关节与时间步联合建模。
- 论文对 $\tau$ 使用偏向低 $\tau$ / 高噪声区的 shifted Beta 分布，而不是均匀采样；直觉是给定机器人观测后，“从纯噪声找出合理平均动作”并不容易，应增加这一区域的训练权重。

推理从 $A_t^0\sim\mathcal N(0,I)$ 出发，用 10 步 Euler 积分（$\delta=0.1$）：

$$A_t^{\tau+\delta}=A_t^\tau+\delta\,v_\theta(A_t^\tau,o_t,\tau).$$

最后得到 $A_t^1$ 并执行其中一部分动作。20 Hz 的 UR5e / Franka 每 0.8 秒重推理一次、执行 16 步；其他 50 Hz 平台每 0.5 秒重推理一次、执行 25 步。因此 $H=50$ 是模型预测 horizon，不等于每次都 open-loop 执行 50 步。

### 3) 和系统实现的对应

- **VLM prefix：** $[I_t^1,\ldots,I_t^n,\ell_t]$ 进入 PaliGemma 3B backbone，承接 Internet-scale 视觉语义。
- **Robot suffix：** $[q_t,A_t^\tau]$ 进入约 300M 的 action expert；两组参数只通过 self-attention 交互，避免让机器人连续量直接挤进语言词表。
- **三段 blockwise mask：** image/language、state、noisy actions 各自块内双向 attention，前块不看未来块；于是 image/language/state 的 KV cache 可在 10 个 flow steps 间复用。
- **延迟预算：** 3 个相机、RTX 4090 上 image encoder 14 ms、observation pass 32 ms、10 次 action pass 合计 27 ms，总 on-board 73 ms；off-board 加网络后 86 ms。action chunk 将高频控制和较慢模型推理解耦。

### 4) 假设与数学没有覆盖的真实误差

这个 loss 默认示范动作在时间上对齐、归一化和 embodiment mask 正确，也默认观测足以消除关键歧义；它没有显式建模相机延迟、控制通信抖动、joint backlash、力 / 触觉、摩擦变化、柔性物体内部状态、碰撞约束和执行器饱和。更关键的是，0.5–0.8 秒 open-loop 窗口内真实 transition $x_{t+1}=f(x_t,a_t,w_t)$ 仍会受未观测扰动 $w_t$ 影响；flow loss 只拟合示范分布，不保证闭环稳定性、安全性或约束可行性。

## 核心

1. **Motivation：离散 action token 不够高频、也不够连续**
   - 机器人 foundation model 既要继承 VLM 的语义泛化，又要执行双臂、柔性物体和接触丰富的连续动作；逐 token 自回归动作在控制频率、量化误差和长 action sequence 延迟上受限。
   - π₀ 用 flow matching 输出连续 action chunk，使整个时间窗内的动作相关性一次建模，也把多次控制命令摊到一次视觉编码和一次条件生成。
   - 目标不是只做一个漂亮模型，而是形成“广覆盖 pre-training → 高质量 post-training → 复杂任务部署”的机器人训练 recipe。

2. **System / Method：3B VLM backbone + 300M action expert**
   - PaliGemma 负责 image / language tokens，action expert 负责 proprioception / noisy actions；总参数约 3.3B，action expert 从零初始化。
   - 模型采用 3-block attention：视觉语言、机器人状态、动作块。动作 token 之间全双向 attention，适合联合生成 $H=50$ 的轨迹。
   - 每次采样做 10 次 flow integration；条件 prefix 的 KV cache 复用，使多步生成不必重复算视觉与语言。
   - 控制动作并非 torque-level safety policy；它输出 robot-specific joint / gripper / base command，底层执行器仍需速率、位置、碰撞和急停约束。

3. **Training / Data Details：覆盖与质量分阶段优化**
   - 自有数据超过 10,000 小时，涵盖 7 种机器人配置和 68 个宽定义任务；另混合 OXE、Bridge V2、DROID。自有数据约 903M timesteps，其中 106M 单臂、797M 双臂；开源数据占训练 mixture 的 9.1%。
   - 不平衡的 task-robot 组合按 $n^{0.43}$ 加权，而不是按样本数线性采样，从而压低 laundry 等大数据任务的垄断效应。
   - 预训练强调任务、场景、机器人和错误恢复的覆盖；post-training 用 5 小时到 100+ 小时不等的任务特定高质量数据，强化流畅、稳定和一致策略。
   - 语言标签同时使用 coarse task name 与约 2 秒的细粒度 segment annotation；复杂任务还可由高层 VLM 动态输出子指令。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — action expert 解耦参数、共享 attention：** 保留 VLM 语义参数，同时给连续状态 / 动作独立容量；比把动作硬塞进语言 token 更适合高维、连续和高频控制。
   - **Trick 2 — action chunk + prefix KV cache：** 一次生成 50 步、实际执行 16 或 25 步再重规划；10 次 flow pass 只重算 action suffix，达到 73 ms on-board inference。
   - **Trick 3 — pretrain 学恢复，post-train 学风格：** 广而杂的数据包含失误、纠正和长尾状态，高质量小数据教 fluent execution；只用后者会脆，只用前者会笨。
   - **Trick 4 — $n^{0.43}$ mixture weighting：** 以次线性频率保留大数据集信号，又给小任务足够曝光，是 data curation 与 model training 联动的简单可复用接口。
   - **Trick 5 — 低噪声区少采、高噪声区多采：** flow timestep 分布针对 action prediction 调整，不照搬图像生成的 logit-normal recipe。

5. **Results：生成架构、VLM 初始化和预训练 recipe 都有贡献，但评测仍偏小样本**
   - Base model 在 shirt folding、两档 bussing、grocery bagging、toast 等 5 个任务上，以每方法每任务 10 trials 评估；完整 700k-step π₀ 最强，160k-step compute-parity 版本也超过同 mixture 的 OpenVLA / Octo。
   - 新任务 fine-tuning 中，π₀ 通常优于 OpenVLA、Octo、ACT 和 Diffusion Policy；相似于预训练分布的任务收益更大，部分设置中预训练相对 scratch 可到约 2×。
   - 复杂任务包括随机皱褶衣物折叠、移动取衣、桌面清理、纸箱组装、装蛋和外卖盒打包；完整 pretrain + post-train 在所有任务平均得分超过满分的 50%，并普遍优于 zero-shot / scratch ablation。
   - 但大多结果只有每任务 10 次真机 trial，baseline 训练步数并非全部严格匹配；论文也承认尚不清楚不同数据源的边际价值，以及跨得很远的 domain（驾驶、导航、legged locomotion）是否正迁移。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？** 有新物体、新任务、跨机器人和少量数据 fine-tuning 的正迁移证据；但 π₀ 同时改变了模型规模、VLM 初始化、连续 action head、action chunk、数据规模和训练 recipe，单项贡献不能只凭总成绩归因。更严谨的视角是“完整系统栈”的胜利。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：**
  1. 机器人 post-training 与 LLM post-training 的结构高度相似：base model 的 coverage / recovery repertoire 与高质量 task data 的行为塑形必须分开管理；真正的机会在 mixture observability、dataset value、failure replay 和 eval flywheel。
  2. rollout infra 不只要吞吐，还要建模 control deadline：图像编码、prefix pass、10 次 flow pass、网络、机器人执行必须形成 latency budget；p95 jitter 会直接改变闭环 policy，而不是单纯服务质量指标。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：** 按 task × embodiment × camera config 记录数据权重、action norm stats、mask、loss 与 success；serving 同时记录 flow-step latency、achieved action Hz、stale-observation age、chunk interruption、controller saturation、collision / recovery。官方 openpi 已提供 base checkpoints、JAX / PyTorch 实现和 remote policy server，但截至其文档，PyTorch 路线仍缺 FSDP、LoRA、EMA 与 mixed precision 等能力。

## 疑问 / 下一步

- **没看懂 / 想深挖：** 怎样在保持 chunk-level 多峰性的同时显式引入 closed-loop feedback？固定每 0.5–0.8 秒重规划会把 inference schedule 当超参，但真正需要的是根据接触变化、uncertainty 或 safety residual 自适应中断。
- **如果要复现 / 小规模试，第一个实验做什么？** 用 openpi 的无机器人 inference / LIBERO example 做 smoke test：固定同一 observation batch，比 10 / 5 / 2 flow steps 的 p50/p95 latency、action smoothness 与 task success；再人为增加 50–150 ms observation delay，测 chunk length 与重规划频率的交互。
- **下一步：** Day12 深挖 Diffusion Policy，分离“连续生成式 action head”本身与 π₀ 的 VLM / 大规模跨 embodiment 数据贡献。

## 原文金句 (1-2句)
> “Our generalist robot policy uses a pre-trained vision-language model (VLM) backbone, as well as a diverse cross-embodiment dataset with a variety of dexterous manipulation tasks.”

> “Intuitively, the diverse (but lower quality) pre-training data allows the model to recover from mistakes and handle highly varied situations, which might not otherwise occur in the high-quality post-training data, while the post-training data teaches the model to perform the task well.”

## 今晚产出
- [ ] 画 `multi-view RGB + language + proprioception → VLM prefix / action expert → 10-step flow → H=50 action chunk → low-level controller` 数据流图
- [ ] 用 NumPy 写 20 行 conditional flow toy：二维双峰 action distribution，比较 MSE mean 与 flow samples
- [ ] 跑 openpi 无机器人 inference smoke test，记录 GPU 显存、image / prefix / flow latency 和输出 shape
- [ ] 做 `2 / 5 / 10 flow steps × 8 / 16 / 25 executed actions` 小矩阵，明确算力—反馈频率 trade-off
- [ ] 为 robot data mixture 设计日志字段：task、embodiment、source、quality、recovery flag、sampling weight、held-out success

## 连接
- 上一篇: Day10 — Habitat 3.0（动态人类伙伴、协作环境与分层 skill 评测）
- 下一篇预告: Day12 — Diffusion Policy（视觉条件动作扩散与 receding-horizon 控制）
- 相关: Day05 UniSim；Day06 DreamerV3；Day09 RT-2 / OpenVLA

## 参考链接
- Paper: https://arxiv.org/abs/2410.24164
- Project: https://www.physicalintelligence.company/blog/pi0
- Official code: https://github.com/Physical-Intelligence/openpi
