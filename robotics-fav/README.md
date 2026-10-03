# Robotics 收藏夹纪要

来源：Bilibili 收藏夹「Robotics」（UP：geek_UNCLE，fid=8685233），共 21 个视频。与青稞Talk 同体例：字幕存档（transcripts/）+ 纪要 + HTML 阅读版。

| # | 标题 | 阅读版 | 一句话 |
|---|---|---|---|
| F21 | [Pulkit Agrawal：RL与IL的物理层真相（还原版）](episodes/F21-agrawal-rl-il-physics.md) | [阅读版](episodes/F21-agrawal-rl-il-physics.html) | 语义层归 IL、接触物理层只能靠交互学，先定层再选工具 |
| F06 | [宇树G1关节解剖（还原版）](episodes/F06-unitree-g1-teardown-2.md) | [阅读版](episodes/F06-unitree-g1-teardown-2.html) | 背隙测出来而不是造没有；中空走线把绞线失效置换为接插件失效 |
| F05 | [拆解宇树G1：执行器、热管理与踝关节（还原版）](episodes/F05-unitree-g1-teardown-1.md) | [阅读版](episodes/F05-unitree-g1-teardown-1.html) | 一套执行器架构缩放，钱花在双编码器与电流估力矩 |
| F19 | [Jason Ma：生产级机器人的RL迷思与数据飞轮（还原版）](episodes/F19-jason-ma-production-rl.md) | [阅读版](episodes/F19-jason-ma-production-rl.html) | 演示成功率是集体错觉，生产级看 99.4% 基准与部署数据飞轮 |
| F20 | [Robert Platt：几何等变性与机器人学习的 Scaling Curve（还原版）](episodes/F20-platt-equivariance-scaling.md) | [阅读版](episodes/F20-platt-equivariance-scaling.html) | 等变是数据 scaling 的乘数，乘的只是几何一维 |
| F18 | [David Held：机器人辅助穿衣的 RL 与 VLM 奖励（还原版）](episodes/F18-held-robot-dressing.md) | [阅读版](episodes/F18-held-robot-dressing.html) | 力约束独立成层；VLM 奖励走成对偏好而非直接打分 |
| F16 | [Sergey Levine：机器人基础模型的 RL 后训练范式（还原版）](episodes/F16-levine-rl-post-training.md) | [阅读版](episodes/F16-levine-rl-post-training.html) | VLA 是通用型原型，RECAP 用优势条件化做后训练 |
| F15 | [RL4IL 研讨会：等变策略、Offline RL 与多模态预训练（还原版）](episodes/F15-rl4il-workshop.md) | [阅读版](episodes/F15-rl4il-workshop.html) | 共识已从「RL 还是 IL」换成「RL 以何种形态接在 IL 后面」 |
| F17 | [VLA 后训练：Progress-to-Go 与 ExPO（还原版）](episodes/F17-vla-post-training-expo.md) | [阅读版](episodes/F17-vla-post-training-expo.html) | EXPO 编辑策略绕开 VLA 长链路，19 分钟在线数据 30/30 |
| F14 | [结构化世界模型：可扩展数据引擎（还原版）](episodes/F14-structured-world-model.md) | [阅读版](episodes/F14-structured-world-model.html) | 结构化中间表示才决定数据引擎性质，不是模型规模 |
| F13 | [人形机器人运动控制的「数据飞轮」（还原版）](episodes/F13-humanoid-data-flywheel.md) | [阅读版](episodes/F13-humanoid-data-flywheel.html) | 生成管覆盖、仿真管可行，按失败模式定向回流补数据 |
| F12 | [合成数据在机器人领域的闭环基础设施（还原版）](episodes/F12-synthetic-data-closed-loop.md) | [阅读版](episodes/F12-synthetic-data-closed-loop.html) | 评估是闭环的限速环，仿真评测只当 gate 用 |
| F11 | [具身智能的数据解法（还原版）](episodes/F11-embodied-data-solutions.md) | [阅读版](episodes/F11-embodied-data-solutions.html) | 造什么/造多少/怎么迁移三问各有解，物理对齐优先于渲染逼真 |
| F10 | [TRI 的机器人可扩展学习基础设施（还原版）](episodes/F10-tri-scalable-learning.md) | [阅读版](episodes/F10-tri-scalable-learning.html) | 评估基础设施才是核心资产，连负面结果都保留 |
| F09 | [物理感知视觉：逆向图形学与世界模型（还原版）](episodes/F09-icra2026-physics-vision.md) | [阅读版](episodes/F09-icra2026-physics-vision.html) | 「看对≠算对」开始被量化，逆动力学把生成视频落成动作 |
| F08 | [机器人学习的数据挑战（还原版）](episodes/F08-icra2026-data-challenges.md) | [阅读版](episodes/F08-icra2026-data-challenges.html) | 可微物理/LTL/Sim-to-Real 省的是同一种真机交互资源 |
| F07 | [加速机器人物理仿真（还原版）](episodes/F07-icra2026-physics-sim.md) | [阅读版](episodes/F07-icra2026-physics-sim.html) | 引擎层规格变为「并行世界数 × 梯度质量」 |
| F04 | [ICRA 2026 机器人巡礼（还原版）](episodes/F04-icra2026-robot-tour.md) | [阅读版](episodes/F04-icra2026-robot-tour.html) | 触觉从选配传感器变成具身数据管线的入口 |
| F02 | [All-In 巴黎峰会：四大机器人巨头共话](episodes/F02-all-in-paris-robotics.md) | [阅读版](episodes/F02-all-in-paris-robotics.html) | 形态之争是伪命题，四家卖的都是数据与结果而非机器人本体 |
| F03 | [从VLA模型预训练到博世量产：Humanoid 端到端落地架构](episodes/F03-vla-bosch-humanoid.md) | [阅读版](episodes/F03-vla-bosch-humanoid.html) | Artem Sokolov：1500+ 场景筛后轮式底盘+并联夹爪覆盖 90% 需求 |
| F01 | [具身智能时代：人形机器人如何重塑我们的未来](episodes/F01-humanoid-future.md) | [阅读版](episodes/F01-humanoid-future.html) | Agility：市场按仓库→零售/医院→家庭递进，部署仍依赖物理屏障 |

## 说明

- 字幕实录版：F01–F03（字幕存 transcripts/）。
- 公开资料还原版（F04–F21 中除上之外）：标注「（还原版）」的均已显著声明非字幕实录；其中 ICRA 2026 一批（Levine/Platt/Held/Ma/Agrawal 等）字幕轨存在但接口串台，多次重试未果，先以还原版占位，拿到字幕后升级实录版。
- 收藏夹共 21 个视频已全部覆盖。
