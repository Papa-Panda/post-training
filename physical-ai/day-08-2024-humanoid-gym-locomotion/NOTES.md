# Day 08 — Humanoid-Gym：Humanoid Locomotion 的 Zero-Shot Sim2Real

## 元信息
- Title: Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer
- Authors / Org: Xinyang Gu, Yen-Jen Wang, Jianyu Chen / RobotEra, Shanghai Qi Zhi Institute, Tsinghua University
- Link / arXiv / Blog: https://arxiv.org/abs/2404.05695
- Date read: 2026-08-29
- Tags: [physical-ai, humanoid, locomotion, reinforcement-learning, sim2sim, sim2real, isaac-gym, mujoco, domain-randomization]
- Thread: physical-ai
- Folder: day-08-2024-humanoid-gym-locomotion
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-08-2024-humanoid-gym-locomotion

## 一句话总结
Humanoid-Gym 用 Isaac Gym 中的 8192 个并行环境训练 100 Hz 的 humanoid velocity-command policy，再把同一策略放进经真机轨迹校准的 MuJoCo 做 cross-simulator gate，并通过 asymmetric actor-critic、15 帧历史、周期 gait prior 和覆盖传感器/时延/动力学的 domain randomization，在 1.2 m XBot-S 与 1.65 m XBot-L 上展示 zero-shot sim-to-real 行走。

## 和之前工作的关系

- **接了哪条线：** 接 Day03 Isaac Lab / Isaac Sim 的 GPU 并行仿真与 Day07 H2O 的 deployable PPO，把关注点从“跟踪人类全身动作”收窄为最基础但必须可靠的 command-conditioned locomotion。
- **补了哪个短板：** Day07 主要解决 motion retargeting、动作可行性过滤和 whole-body imitation；本篇补上周期步态、速度命令跟踪、跨引擎验证与两种尺寸 humanoid 的 locomotion 部署链路。
- **替代 / 分叉 / 改进：** 它没有 world model，也不做 H2O 式 reference-motion tracking；策略直接从 proprioception、clock 和速度命令输出 joint-position target。核心不是更复杂模型，而是 reward / observation / randomization / simulator calibration 的系统配方。
- **对之前 Day X 的直接对比：** Day02 MuJoCo 在这里不是训练主引擎，而是 Isaac Gym 与真机之间的独立 sim2sim gate；Day03 的“GPU rollout engine”负责吞吐，Day02 的较慢 CPU simulator负责检查策略是否过拟合单一物理实现。

## 为什么今天读它

路线图 Day08 进入 humanoid locomotion。H2O 已说明“可部署 observation + domain randomization”能让全身动作上真机；Humanoid-Gym进一步给出一个开源、较小而完整的 `Isaac Gym train → MuJoCo validate → robot deploy` 基线，适合拆解 locomotion 的最小闭环，也暴露了该类论文常见的评测短板：真机展示充分，但统一量化指标不足。

## 今天的 3 问
1. 为什么 `Isaac Gym → MuJoCo` 的 sim2sim 迁移可以作为真机前 gate；什么条件下跨引擎一致仍不能预测 sim2real？
2. 周期 clock、stance mask 与 joint-reference reward 给了多少 gait prior；它们是在提高样本效率，还是限制了非周期步态与复杂地形适应？
3. 15 帧 observation history、asymmetric critic 和 domain randomization 各自覆盖 partial observability、训练稳定性与动力学偏差中的哪一部分？

## 核心

1. **Motivation：让 full-size humanoid locomotion 有一个可复现的最小 sim2real 基线**
   - 人形机器人结构更复杂、自由度耦合更强、跌倒代价更高，sim2real gap 通常大于四足机器人；当时开源的 full-size humanoid locomotion 训练与部署资源仍有限。
   - Humanoid-Gym 的主张不是提出新网络，而是公开一套 end-to-end recipe：大规模并行 PPO、humanoid-specific reward、domain randomization、sim2sim 验证和真机部署。

2. **System / Method：周期先验 + asymmetric PPO + 双模拟器 gate**
   - **控制目标**：输入期望平面速度与偏航命令，策略输出 12 维目标关节位置；内部 PD controller 将其转为 torque。
   - **步态先验**：一个 gait cycle 被分为两段 double support 与两段 single support；`[sin(2πt/CT), cos(2πt/CT)]` clock 驱动参考腿部运动，periodic stance mask 指定左右脚预期 swing / stance。
   - **Actor observation**：clock 2 维、command 3 维、joint position 12、joint velocity 12、base angular velocity 3、orientation 3、last action 12，共 47 维；堆叠 15 帧以补偿 POMDP 中看不到的状态。
   - **Asymmetric critic**：critic 额外接收 friction、body mass、base linear velocity、push force/torque、tracking difference、stance mask 和 foot contact 等 privileged state；单帧 privileged observation 为 73 维，堆叠 3 帧。
   - **运行频率与验证**：policy 100 Hz，底层 PD 1000 Hz；在 Isaac Gym 训练，再将同一 policy 放入经轨迹校准的 MuJoCo，在 flat / unseen uneven terrain 上做 sim2sim stress test，最后 zero-shot 部署。

3. **Training / Data Details：Sim 数据、Real 数据与可验证信号**
   - **Sim rollout**：8192 个并行环境；episode 2400 steps；PPO + GAE，discount 0.994、GAE factor 0.95、learning rate `1e-5`。论文表中的“Number Training Epochs = 2”是每次 update 的 epoch，而不是总训练只跑两轮。
   - **Reward**：velocity tracking、orientation / base-height stability、contact-pattern、joint-reference tracking，再加 energy、action second-difference、过大接触力等 regularization；目标 base height 为 0.7 m。
   - **Domain randomization**：关节位置/速度、角速度、姿态观测噪声；0–10 ms system delay；friction 0.1–2.0；motor strength 95%–105%；payload 加性扰动 -5–5 kg，并注入 push force / torque。
   - **Real 数据的角色**：策略训练不使用真机 trajectory 做梯度更新；真机轨迹用于校准/比较 MuJoCo 动力学，尤其检查腿部关节 sine trajectory 与 left-knee / left-ankle phase portrait。
   - **Verifiable signal**：训练中用速度、姿态、base height、接触计划与关节参考误差；部署前用 cross-simulator trajectory agreement 与 flat / unseen uneven-terrain traversal 做 gate。论文没有给出标准化真机 success rate、速度跟踪误差或跌倒率。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — 把第二个 simulator 当独立 evaluator**：高吞吐 Isaac Gym 负责搜索策略，校准后的 MuJoCo 负责暴露对 contact solver / dynamics implementation 的过拟合；它比“同一引擎换 seed”更像真正的 held-out eval。
   - **Trick 2 — Actor / critic 信息边界分离**：actor 只看真机可得观测，critic 使用 friction、mass、push 与 contact 等 privileged state；训练时提高 value estimation，部署时不增加传感器依赖。
   - **Trick 3 — 历史窗口显式补 POMDP**：47 维 actor observation 堆叠 15 帧，让前馈 policy 从时间差分中推断速度、接触与未建模动力学，不必直接依赖复杂 recurrent / transformer 架构。
   - **Trick 4 — Gait prior 进入 observation 和 reward 两侧**：clock 告诉 policy 当前 phase，stance mask 奖励约束预期落脚；这样更容易得到稳定步态，但也可能压制非周期的恢复动作。
   - **Trick 5 — 同时 randomize sensing 与 dynamics**：不仅改 friction / payload / motor strength，也扰动 joint / IMU-like observations 和系统时延；sim2real 不是单一 physics 参数问题。

5. **Results：证据与边界**
   - 框架在 RobotEra 的 **1.2 m XBot-S** 与 **1.65 m XBot-L** 上展示 zero-shot sim-to-real locomotion，覆盖不同尺寸 embodiment。
   - 同一 policy 在 MuJoCo flat terrain 和训练外 uneven terrain 上均成功行走；作者报告经校准后 MuJoCo 的关节轨迹/phase portrait 更接近真机，而 Isaac Gym 与真机差异更大。
   - 论文的主要证据是轨迹图和视频演示，没有报告统一的 success rate、fall rate、command-tracking RMSE、训练时长/GPU budget，也没有 ablation 分离 sim2sim gate、frame stacking、gait prior 和各 randomization 项的贡献。因此应把结论理解为“可工作的开源 recipe”，而非已充分量化的 SOTA 比较。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？** 跨 simulator、未见 uneven terrain、两种身高 humanoid 和真机的结果支持一定 transfer；但缺少量化与消融，不能判断哪一项贡献最大。现有证据更支持 **framework / recipe**，而不是 policy architecture 创新。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：**
  1. `Isaac Gym train → MuJoCo eval → real deploy` 对应 post-training 的 generator / independent verifier / online canary：优化环境和评测环境必须解耦，否则 reward 高可能只是 simulator overfitting。
  2. Asymmetric actor-critic 是“训练时富信息、推理时窄接口”的机器人版本；和 privileged judge / process supervision 类似，关键是严格定义不能泄漏到 deployment 的信号。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：** 把 checkpoint export、跨引擎 replay、trajectory alignment、terrain sweep 与 regression threshold 自动化；记录 simulator version、URDF、PD gains、policy / PD frequency、observation normalization、frame stack、randomization distributions，否则 zero-shot 结果难复现。

## 疑问 / 下一步

- **没看懂 / 想深挖：** MuJoCo 使用少量真机轨迹校准后还能否被称为独立 held-out evaluator？若调参反复看真机轨迹，sim2sim gate 也可能过拟合 hardware calibration set。
- **如果要复现 / 小规模试，第一个实验做什么？** 在 Humanoid-Gym 中训练 XBot baseline，固定 policy 后同时在 Isaac Gym 与 MuJoCo 扫 friction、payload、latency、uneven-terrain level；统计 command-tracking RMSE、fall rate、energy / m 和 cross-simulator rank correlation，比较 `single-frame` vs `15-frame`、`clock on/off` 两个消融。
- **下一步：** Day09 进入 RT-2 / OpenVLA，从低层 proprioceptive locomotion policy 转向视觉—语言—动作模型，明确高层任务语义如何与低层稳定控制对接。

## 原文金句 (1-2句)
> “Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to MuJoCo that allows users to verify the trained policies in different physical simulations to ensure the robustness and generalization of the policies.”

> “Our control policy operates at a high frequency of 100Hz, providing enhanced granularity and precision beyond standard RL locomotion approaches. The internal PD controller runs at an even higher frequency of 1000Hz.”

## 今晚产出
- [ ] 画出 `Isaac Gym 8192 env → PPO → MuJoCo sim2sim gate → XBot-S/L` 四段闭环
- [ ] 把 47 维 actor observation 与 73 维 privileged state 做成 train-only / deploy-time 对照表
- [ ] 跑一个小型 cross-simulator sweep，至少记录 velocity RMSE、fall rate、energy / m
- [ ] 做 `15-frame vs 1-frame` 或 `clock on vs off` 的一个消融，并写清 gait prior 的收益与限制
- [ ] 在笔记中保留证据边界：论文未给统一真机 success rate 与完整训练 compute

## 连接
- 上一篇: Day07 — H2O（sim-to-data + whole-body teleoperation）
- 下一篇预告: Day09 — RT-2 / OpenVLA（Vision-Language-Action）
- 相关: Day02 MuJoCo；Day03 Isaac Lab / Isaac Sim

## 参考链接
- Paper: https://arxiv.org/abs/2404.05695
- Code: https://github.com/ahucc/humanoid-gym
