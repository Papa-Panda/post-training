# Day 09 — RT-2 / OpenVLA：把动作变成 token 的 Vision-Language-Action 路线

## 元信息
- Title: RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control / OpenVLA: An Open-Source Vision-Language-Action Model
- Authors / Org: Anthony Brohan et al. / Google DeepMind；Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti et al. / Stanford University, UC Berkeley, Toyota Research Institute, Google DeepMind, Physical Intelligence, MIT
- Link / arXiv / Blog: https://arxiv.org/abs/2307.15818 / https://arxiv.org/abs/2406.09246
- Project: https://robotics-transformer2.github.io/ / https://openvla.github.io/
- Official code: https://github.com/openvla/openvla （RT-2 未公开训练代码与权重）
- Date read: 2026-08-30
- Tags: [physical-ai, vla, robot-manipulation, action-tokenization, co-finetuning, open-x-embodiment, lora, inference]
- Thread: physical-ai
- Folder: day-09-2024-rt2-openvla
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-09-2024-rt2-openvla

## 一句话总结
RT-2 证明了把连续机器人动作离散成语言 token、再把 web-scale vision-language data 与 robot trajectories 联合训练，可以把语义知识迁移到闭环控制；OpenVLA 则把这条路线变成可复现的 7B 开源系统，用 970k Open X-Embodiment demonstrations、DINOv2+SigLIP 双视觉编码器和 Llama 2，在 29 个跨 embodiment 任务上以少 7 倍参数超过 RT-2-X 16.5 个绝对成功率百分点，并支持 LoRA 与量化部署。

## 和之前工作的关系

- **接了哪条线：** Day04 Genie、Day05 UniSim、Day06 DreamerV3 都在回答“如何得到可预测、可交互的环境模型”；Day09 转向另一条 Physical AI 主线：不显式 rollout 世界模型，而是从图像与语言直接 autoregressive 地输出动作 token。
- **补了哪个短板：** Day07 H2O 与 Day08 Humanoid-Gym 的策略主要依赖 proprioception、reference motion 或速度命令，擅长稳定低层控制，但不理解开放世界语义。RT-2 / OpenVLA 补上“看到新物体、理解新指令、选择正确对象”的高层感知—语义—动作接口。
- **替代 / 分叉 / 改进：** RT-2 的关键是保留 web 数据做 co-fine-tuning，换取 semantic generalization；OpenVLA 选择只在 robot action data 上微调开源 VLM，牺牲部分困难的 Internet-concept generalization，换来开放权重、训练代码、跨 embodiment 数据和可适配性。
- **对之前 Day X 的直接对比：** Day08 以 100 Hz policy + 1000 Hz PD 完成 locomotion；RT-2 55B 云端推理只有 1–3 Hz，OpenVLA bf16 在 RTX 4090 约 6 Hz。VLA 适合 semantic manipulation / 高层 action proposal，尚不能直接替代 humanoid 的高频稳定环。

## 为什么今天读它

路线图 Day09 从 locomotion 进入 VLA。RT-2 给出范式：**action is another language**；OpenVLA 给出工程化开源基线：数据混合、模型架构、FSDP 训练、LoRA 适配、量化推理和真机评测。两篇一起读，能把“概念突破”和“可复现系统”分开看，也能更清楚地识别当前 VLA 的能力边界：web knowledge 能重组已有 motion primitives，但不会凭空创造训练数据里没有的新运动技能。

## 今天的 3 问
1. 把每个连续动作维度独立量化成 256 个 token，为什么足以支持闭环 manipulation；它在哪些高频、精细接触或多峰动作任务上会输给 continuous / diffusion action head？
2. RT-2 的 web+robot co-fine-tuning 与 OpenVLA 的 robot-only fine-tuning，分别如何权衡 semantic retention、embodiment coverage、训练成本与 catastrophic forgetting？
3. 对真实机器人而言，VLA 的核心瓶颈究竟是模型规模、robot data mixture、控制频率，还是缺少 proprioception / temporal history；应该怎样做独立消融？

## 核心

1. **Motivation：把 Internet-scale 语义先验直接接到机器人控制**
   - 传统 imitation policy 能在训练分布内学会动作，却容易在新物体、背景、指令与概念上失效；纯 VLM planner 又通常只负责高层分解，低层 controller 并没有共享 web-scale 预训练知识。
   - RT-2 把 VLM 的输出空间直接扩展为动作 token，让同一组参数既处理 vision-language task，也输出 closed-loop robot action。目标不是从 web 学到全新运动，而是用 web 语义重新组合 robot data 中已有的技能。
   - OpenVLA 针对 RT-2 / RT-2-X 闭源、难以复现和难以适配的问题，提供开放模型、权重、PyTorch pipeline 与下游 fine-tuning recipe。

2. **System / Method：视觉 patch + 语言 instruction → 自回归 action tokens**
   - **RT-2 action encoding**：控制量包含末端执行器 6-DoF 位姿增量、gripper extension，以及 episode termination；连续维度均匀离散成 256 bins，再复用已有 tokenizer 的数字 token 或覆盖 256 个低频 token。
   - **RT-2 co-fine-tuning**：robot image + instruction + action-token sequence 与原始 VQA / caption / interleaved image-text 数据共同训练，并提高 robot dataset 的采样权重；机器人请求解码时把 vocabulary mask 到合法 action tokens。
   - **OpenVLA architecture**：Prismatic-7B backbone = DINOv2（空间细节）+ SigLIP（语义）双视觉编码器，视觉特征拼接后经 2-layer MLP projector 进入 Llama 2 7B；给单张图像和语言指令，输出 7D relative action。
   - **OpenVLA action encoding**：每个动作维度按训练数据第 1–99 percentile 区间独立量化为 256 bins，以避免极端 outlier 拉大 bin width；覆盖 Llama tokenizer 最末 256 个低频 token，只在 action tokens 上计算 next-token cross-entropy。
   - **控制栈位置**：两者都以较低频率给出末端动作增量，不负责 torque-level stabilization；实际系统仍需要安全约束、低层 controller、超时与动作合法性检查。

3. **Training / Data Details：RT-2 保知识，OpenVLA 扩 embodiment**
   - **RT-2**：使用 RT-1 机器人数据（13 台机器人、17 个月、office-kitchen manipulation）和原始 web-scale VLM 数据；PaLI-X 版本为 5B / 55B，PaLM-E 版本为 12B。论文在约 6,000 条 evaluation trajectories 上测试 seen / unseen 和 emergent semantics。
   - **OpenVLA**：从 Open X-Embodiment 的 70+ datasets、2M+ raw trajectories 中筛出至少有第三人称相机、single-arm end-effector control 的 manipulation 数据，并按 Octo mixture heuristic 重加权，最终训练 970k demonstrations。
   - **数据质量细节**：OpenVLA 发现 DROID action-token accuracy 长期偏低，因此只给 10% conservative weight，并在最后三分之一训练中移除；这说明“更多异构数据”并不自动等于更好，mixture compatibility 需要在线监控。
   - **OpenVLA full pretraining**：224×224 输入、batch size 2048、固定 learning rate `2e-5`，训练 27 epochs，直到 action-token accuracy 超过 95%；64×A100 训练 14 天，共约 21,500 A100-hours。384×384 未提升真机表现，却令训练慢约 3 倍。
   - **Adaptation / serving**：OpenVLA full fine-tuning 每个任务用 8×A100、5–15 小时；LoRA rank 32 只训练 1.4% 参数，单 A100 约 10–15 小时，性能接近 full FT。bf16 在 RTX 4090 约 6 Hz、约 15 GB；4-bit 在论文实验里以 7.0 GB 显存达到与 bf16 相近成功率。

4. **Key Tricks：最值得抄的细节**
   - **Trick 1 — 动作复用语言模型 vocabulary**：不另加 action-only head，而是让 action 与 text 共享 autoregressive interface；这样可直接复用 VLM 训练和 serving infra，也使语义知识更容易流入动作预测。
   - **Trick 2 — 保留预训练分布 vs 适配机器人分布**：RT-2 在 fine-tuning 时继续混入 web data，明显改善 generalization；OpenVLA 的对照显示，robot-only FT 在困难 Internet concepts 上会落后 RT-2-X。数据 mixing 是能力保留机制，不只是吞吐问题。
   - **Trick 3 — 双视觉特征且 vision encoder 必须适配**：OpenVLA 融合 SigLIP semantic feature 与 DINOv2 spatial feature；冻结 vision encoder 会显著掉点，说明精细控制所需空间特征不能只靠冻结的 web representation。
   - **Trick 4 — percentile action bins**：OpenVLA 用第 1–99 percentile 而不是 min/max 定量化范围，避免极少数异常动作吞掉有效分辨率；这是小改动，但直接改善 action token 的有效容量。
   - **Trick 5 — latency 是 policy quality 的一部分**：RT-2 55B 只有 1–3 Hz；OpenVLA 的 int8 版本因量化算子开销降到 1.2 Hz，真机成功率反而明显下降，而更快的 int4 接近 bf16。不能只看离线 token accuracy，必须把 end-to-end control rate 放进评测。

5. **Results：语义泛化上台阶，但高频与新技能仍未解决**
   - **RT-2**：在 unseen objects / backgrounds / environments 上平均约为 RT-1 与 MOO 的 2 倍；在 symbol understanding、reasoning、human recognition 三类 emergent evaluation 上，最好模型平均成功率超过 RT-1 的 3 倍；Language-Table simulation 为 `90 ± 10`，对比此前 SoTA `77 ± 4`。
   - **RT-2 ablation**：co-fine-tuning 优于 robot-only fine-tuning；55B 优于 5B；但 web pretraining 不会赋予训练 robot data 中不存在的新 motion skill。
   - **OpenVLA out-of-box**：在 WidowX + Google Robot 共 29 个任务上，7B 模型比 55B RT-2-X 高 16.5 个绝对成功率百分点；其优势并非全轴成立——RT-2-X 在困难 semantic generalization 上更好。
   - **OpenVLA adaptation**：在 7 个 Franka task（每项 10–150 demonstrations）上总体最好，是唯一所有任务都至少 50% success 的方法；但 Diffusion Policy 在狭窄、精细单指令任务上动作更平滑、精确。
   - **边界**：OpenVLA 只看单帧图像、无 proprioception / history / action chunking，通常仍低于 90% success；RT-2 与 OpenVLA 的 1–15 Hz 控制频率离 humanoid 低层控制所需频率很远。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？** RT-2 的 unseen semantics 与 OpenVLA 的跨 WidowX / Google Robot 结果支持 visual、semantic、physical 和 motion generalization；但 OpenVLA 自己的分析把收益归因于数据规模/清洗、双视觉 encoder 和架构多项因素，不能把 16.5-point 提升只归给“开源 7B 更强”。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：**
  1. VLA 是最直接的接口复用：robot demonstrations 变成另一种 supervised sequence，FSDP / FlashAttention / LoRA / quantization / data mixture observability 都可从 LLM post-training 迁移。
  2. 机器人评测必须把模型质量与系统时延联合看：离线 action-token accuracy 相同，控制频率不同也会改变 closed-loop dynamics。对应 agentic RL，端到端 tool latency 与 timeout 同样是 policy behavior 的一部分。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：** 每个 checkpoint 同时记录 action-token accuracy、per-dataset loss、mixture sampling weight、机器人 success / recovery / safety violation、p50/p95 inference latency 与 achieved control Hz；按 embodiment / task / semantic novelty 分桶，防止 aggregate success 掩盖 semantic forgetting 或单一机器人过拟合。

## 疑问 / 下一步

- **没看懂 / 想深挖：** 动作逐维量化并用 token-level cross-entropy 优化，默认各维在 decoder 中按固定顺序分解；它对多峰 action distribution、跨维几何耦合和 contact-rich precision 的代价有多大？
- **如果要复现 / 小规模试，第一个实验做什么？** 用 OpenVLA 官方代码在 LIBERO 或 BridgeData V2 子集上做 LoRA：固定同一数据与 seed，对比 `1–99 percentile bins vs min/max bins`、`frozen vs trainable vision encoder`，同时报告 action-token accuracy、task success、p95 latency 和 achieved control Hz。
- **下一步：** Day10 进入 Habitat 3.0 / Habitat Lab，从桌面 manipulation 转向 embodied navigation、human-robot interaction 与 simulator benchmark infrastructure。

## 原文金句 (1-2句)
> “We represent robot actions as another language, which can be cast into text tokens and trained together with Internet-scale vision-language datasets.” — RT-2 project page

> “OpenVLA demonstrates strong results for generalist manipulation, outperforming closed models such as RT-2-X (55B) by 16.5% in absolute task success rate across 29 tasks and multiple robot embodiments, with 7x fewer parameters.” — OpenVLA abstract

## 今晚产出
- [ ] 画一张 `image + instruction → visual tokens → LLM → action tokens → de-tokenize → low-level controller` 数据流图
- [ ] 做 RT-2 vs OpenVLA 对照表：web co-finetuning、robot data、参数量、开放性、控制频率、semantic retention
- [ ] 跑通 OpenVLA 官方 inference 示例，记录显存、p50/p95 latency 与实际 action Hz（无机器人可先用固定图片）
- [ ] 选一个小数据集跑 LoRA smoke test，确认 loss 只计算 action tokens，并记录 per-dataset action accuracy
- [ ] 写清 VLA 与 Day08 locomotion policy 的分层边界：VLA 做低频语义动作建议，安全/稳定由高频 controller 兜底

## 连接
- 上一篇: Day08 — Humanoid-Gym（高频 proprioceptive locomotion + sim2sim gate）
- 下一篇预告: Day10 — Habitat 3.0 / Habitat Lab（embodied navigation 与 human-robot interaction）
- 相关: Day04 Genie；Day05 UniSim；Day06 DreamerV3；Day07 H2O

## 参考链接
- RT-2 paper: https://arxiv.org/abs/2307.15818
- RT-2 project: https://robotics-transformer2.github.io/
- OpenVLA paper: https://arxiv.org/abs/2406.09246
- OpenVLA project: https://openvla.github.io/
- OpenVLA official code: https://github.com/openvla/openvla
