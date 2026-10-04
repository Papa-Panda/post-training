# Day 09 — RT-2 / OpenVLA：把动作变成 token 的 Vision-Language-Action 路线

> 📖 阅读版：https://papa-panda.github.io/post-training/physical-ai/day-09-2024-rt2-openvla/

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

## 大纲
- 问题：纯机器人策略不认识训练外物体与指令，纯 VLM 又不会输出低层动作；缺的是语义到控制的桥
- 表示：7 维末端动作逐维切 256 个 bin，动作即 token，与文字共用词表、自回归与交叉熵
- 训练：RT-2 用 web 与 robot 数据 co-fine-tuning 保语义；OpenVLA 用 970k 条 Open X-Embodiment 演示训 7B 开源模型
- 系统：DINOv2 加 SigLIP 双编码器接 Llama 2，单帧图像加语言进，7 维相对动作出，低层控制器兜底执行
- 证据：未见物体与 emergent 语义上约为基线 2 倍；OpenVLA 在 29 个任务上超 RT-2-X 16.5 个百分点；代价是 1–6 Hz 与量化精度

## 流程图
```mermaid
graph LR
  A[图像与指令输入] --> B[双编码器提特征]
  B --> C[语言模型推理]
  C --> D[动作令牌生成]
  D --> E[解码末端动作]
  E --> F[低层控制器执行]
```

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

## 问答补充（2026-09-07）

> 以下问答归档自 physical AI side chat（2026-09-05）的用户主动提问；Day N 推送卡片与提醒类消息已跳过。每个条目保留问题原文（精简）、核心答案与符号定义。

### VLA 是类别名，不是具体架构

**问**：VLA 不是一种模型而是模型架构？类似 transformer？

**答**：VLA 更像一个**类别名**，定义的是输入输出契约：图像和语言进来，动作出去，即 $\pi(A\mid O,\text{language})$ ，其中 $A$ 为动作（序列/块）， $O$ 为观测。凡是满足这个契约的都算 VLA。Transformer 是实现引擎（积木），VLA 是用积木搭出来的车型：RT-2/OpenVLA 用 Transformer 解码器加离散 action token，π₀ 用 VLM 加 flow matching 动作头，Octo 用 Transformer 加 diffusion head——三个都是 VLA，但内部引擎完全不同。反例：Day12 的 Diffusion Policy 连语言输入都不吃，严格讲不算 VLA，只是 action diffusion。这说明语言那个 L 是 VLA 的身份特征：去掉语言，它就退化成纯动作生成模型。

**关联**：Day09（RT-2/OpenVLA）/ Day11（π₀）/ Day12（Diffusion Policy）/ Day13（Octo）。

### VLA 与 transformer 的关系

**问**：VLA 也是 transformer？

**答**：定义上不是必须，现实中几乎全是。VLA 只要求"图像+语言→动作"这个契约（CNN 编码器加 RNN 理论上也算），但 VLA 的价值在于借用预训练 VLM 的语义（知道"杯子"是什么、"放进水槽"意味着什么），而今天好用的 VLM 都是 transformer；连 π₀ 的 flow matching 动作头，前面半个 PaliGemma 也是 transformer。结论：**VLA 名义上是类别，实际上市面上 99% 都是 transformer 实现**。

### Day09 "以什么为条件"：芒果的例子

**问**：Day09 解决的是"以什么为条件"……没看懂。

**答**：假设机器人训练数据里只有"拿苹果"和"拿香蕉"的演示，现在指令是"拿起那个芒果"——纯机器人策略会懵：它这辈子没见过芒果，图像里那坨黄色的东西跟它学过的"抓取目标"对不上号。RT-2 的做法：拿一个在互联网上看过几百万张图片的 VLM（它在网上见过芒果，知道"芒果"长什么样、跟"黄色水果"是什么关系），把机器人动作也变成 token，跟语言放在同一个词表里一起训练。于是模型用它在网上学到的"芒果长这样"的知识，在当前图像里定位芒果，再输出对应的动作 token。**语义知识是从 web 数据里"借"来的，机器人数据只负责教它"看到东西之后手该怎么动"**。这就是"解决语义问题"：之前机器人是文盲，只能认训练里见过的几样东西；RT-2 让它识字了。代价：动作被量化成 256 个 bin、频率只有几 Hz——认得字，不代表手就稳。

### VLM 是什么

**问**：VLM 是啥？

**答**：VLM = Vision-Language Model，视觉语言模型：图片和文字一起吃进去，文字出来（比如给它一张照片问"这是什么"，它回答"一只芒果"）。它在互联网规模的图文对上训练，认识的东西远超任何机器人数据集——这就是 RT-2 要"借"的语义。一句话关系：**VLA = VLM + 动作输出**。RT-2 是把 VLM 的词表扩充了动作 token；π₀ 是拿 PaliGemma 当 VLM 底座再外挂一个动作头。

### t₁..t₇ 不是文字：Day09 的突破点

**问**： $p(t\mid I,\ell)$ 里 t 是文字？那这不就是一个 VLM 吗？Day09 有啥突破？

**答**：t 不是文字，是**动作 token**：7 维连续动作 $a=(\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,\text{gripper})$ 每维切成 256 个 bin，每个 bin 分配一个 token ID， $t_1$ 到 $t_7$ 就是"这次手该怎么动"的离散编码。公式长得像 VLM 完全是故意的——**突破恰恰是这个表示上的 trick**：把动作塞进 VLM 的词表，让"预测下一个动作 token"和"预测下一个文字 token"变成同一个任务、同一个 transformer、同一套交叉熵 loss。在此之前是两个世界：VLM 认识芒果但手不会动，机器人策略手会动但不认识芒果。RT-2 第一个把它们打通——web 数据教它认识世界，机器人数据教它把"认识"翻译成"动作"，翻译的桥梁就是"动作即 token"。所以 Day09 的突破不是新数学（自回归 CE 都是现成的），而是证明了：**VLM 的语义能力可以通过 token 化迁移到机器人控制上**。

**符号**： $t$ 动作 token 序列； $I$ 输入图像； $\ell$ 语言指令 token； $a$ 连续动作向量，各分量为末端位姿增量与夹爪开合。

### 目标动作 token 来自哪里

**问**：预测下一个动作 token，那这个目标动作来自于哪里？

**答**：来自人类演示，标准的 behavior cloning。训练数据是人遥操作机器人录下来的：操作员用 VR 手柄或 spacemouse 控制机械臂做"拿杯子"，系统同步记录每一时刻的图像、指令和 7 维连续动作；训练时把这些连续动作按 256 bin 离散化成 token，就成了交叉熵 loss 的 target（模型预测 $t_3$ ，target 就是演示里那一刻真实的 $t_3$ ）。RT-2 的 co-finetraining 有意思在 target 是两种混在一起的：web 数据的 target 是文字 token（看图说话、问答），机器人数据的 target 是动作 token。同一个 loss 下，模型自己学会了"看到芒果图片时输出描述文字，看到机器人视角+指令时输出动作"。

**与之前工作的关系**：本节 6 问构成 Day09 的"表示"主线——VLA 的契约定义 → 语义从 web 借 → 动作 token 化 → 演示数据即 target；与 Day11/12 的"生成式连续动作"路线形成对照（对照见 README 问答记录）。

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

<!-- viz:stats: 7B Prismatic backbone | DINOv2+SigLIP 双编码器 | 7D 相对动作输出 -->
<!-- viz:vs: RT-2 | fine-tune 混入 web data; generalization 好 || robot-only FT | 困难概念上落后 -->

## 第二轮复习（2026-10-04）

> 本轮做三处元信息校准（两篇的会议定位、OpenVLA 等贡献标注、RT-2 评测规模措辞）＋补大纲与流程图（初读缺失，本轮补齐）。核心收获：把 Day09 从「动作变成 token 的技巧」重读为「接口复用换表示瓶颈」的交易——复用 VLM 词表与训练栈买到语义重组，代价是逐维量化、自回归顺序与单帧无本体感知；这笔交易的价格后来由 Day11–14 用连续动作头分期偿还，又在 Day34 被组合泛化重新定价。

### 元信息修正

1. **发表定位**：RT-2 为 CoRL 2023（第 7 届 Conference on Robot Learning，Atlanta）；OpenVLA 为 CoRL 2024（第 8 届，Munich；PMLR v270, pp. 2679–2713；arXiv v3 为 2024-09-05）。NOTES 元信息只记 arXiv，本轮补上会议定位。
2. **作者标注**：RT-2 论文作者按字母序排列（贡献见其附录 A），「Brohan et al.」只是首名缩写，不表示贡献排序；OpenVLA 的 Moo Jin Kim、Karl Pertsch、Siddharth Karamcheti 三人为等贡献（论文以星号标注），NOTES 原记未标此项。
3. **评测规模措辞**：RT-2 摘要原文为 6k evaluation trials；NOTES 原写「约 6,000 条 evaluation trajectories」，本轮校准为 trials（试验次数），不与训练轨迹数混同。
4. **定位校准**：初读把 16.5 个百分点记成「7B 开源模型更强」的证据；复习校准为**数据混合、双编码器与架构的合成结果**——论文自身未做单因子归因（见边界第 1 条）。

### 一句话总结

三十多天后回看，Day09 的耐久贡献不是「动作即语言」这句口号，而是证明了一笔可审计的交易：把动作塞进 VLM 词表，就能用同一套自回归训练栈把 web 语义重组到机器人已有的运动技能上；OpenVLA 则证明这笔交易可以开源复现（970k 条演示、29 个任务、超 RT-2-X 16.5 个百分点）。但交易的价格写在表示里：256 bin 逐维量化吃精度、固定顺序自回归吃多峰耦合、1–6 Hz 吃闭环动态——后续 VLA 的演进史，基本就是逐项赎回这三笔抵押。

### 和之前工作的关系

- **vs Day08 Humanoid-Gym（直接对比）**：Day08 是 100 Hz 本体感知步态加 1000 Hz PD 的执行层，语义为零但闭环快；Day09 是 1–6 Hz 的语义层，认得出芒果但手不稳。两者不是替代关系，是同一栈的上下两层：Day09 输出的末端增量必须落到 Day08 一类高频控制器上才算动作。这也解释了为什么本篇 NOTES 一开始就把「低层 controller 兜底」写进系统位置——VLA 从第一天起就是分层架构的上层，不是端到端全身控制。
- **vs Day11 / Day12 / Day13（分专题，同轴分叉）**：四篇学的是同一个条件动作分布 $p_\theta(A\mid O,c)$ ，差别在表示。Day09 用离散 token 加自回归；Day11 π₀ 用 flow matching 一次生成 50 步连续动作块；Day12 Diffusion Policy 用去噪生成连续轨迹但不吃语言；Day13 Octo 保留 diffusion 头，把创新让给跨 embodiment 的模块化数据接口。复习后看得更清：Day09 解决的是「以什么为条件」（语义），Day11/12 解决的是「预测什么」（连续多峰动作），Day13 解决的是「在什么数据上训」（跨身体混合）——三个问题被初读混成了一个，后来在 README 问答里才被拆成 2×2。
- **跨阶段 vs Day15–18（数据专题）**：OpenVLA 的 970k 条演示来自 Day15 Open X-Embodiment 的统一数据，其 DROID 低权重与末段移除（NOTES §3 自记）正是 Day16/17 的数据质量问题在训练侧的显形：异构数据不是越多越好，mixture 兼容性要在线监控。Day09 是数据专题的第一个消费者证据。
- **跨阶段 vs Day25–30 与 Day31–34（评测、安全与最新进展）**：Day09 的评测以 seen / unseen 与 emergent 语义分桶，这是 Day28「固定测度下的成功率估计」与 Day30 分桶门禁的早期形态，但缺 Day29 的安全层（动作合法性、超时与约束全在模型外）。到 Day34 π0.7，组合泛化要求模型做没教过的任务——那正是 Day09「web 语义重组已有技能」命题的严格化版本：重组被单独拎出来当评测轴，而不是混在 emergent 示例里展示。

### 核心

1. **动机重读：赌「接口复用 > 表示保真」**：2023 年机器人缺的不是又一个动作回归头，而是把互联网语义接进控制的桥。RT-2 的赌注是让动作与文字共用词表、共用交叉熵、共用 serving 栈，语义迁移就免费搭车。两年后看赌下对了方向、也标好了价格：被复用的是 VLM 的指代与常识，被抵押的是动作本身的几何——逐维独立量化默认各维可分，接触任务的跨维耦合不在这个表示的语言里。
2. **机制深挖：自回归分解把顺序写进先验**：观测是一帧图像 $I$ 加语言指令 $\ell$ ，真实状态（物体位姿、摩擦、遮挡关系）不可见，是 POMDP 而非 MDP；动作 $a\in\mathbb{R}^{7}$ 为末端位姿增量加夹爪，每维切 256 bin 成 token $t_{1:7}$ 。模型学的是

$$p_\theta(t_{1:7}\mid I,\ell)=\prod_{j=1}^{7}p_\theta(t_j\mid t_{<j},I,\ell)$$

   训练即对此分布做 behavior cloning 的交叉熵，target 来自人类演示（web 数据的 target 是文字 token，同一 loss 里混训）。固定解码顺序意味着平移与旋转的误差会沿 token 链传播，且单值采样难以表达「绕左还是绕右都行」的多峰抓取——这正是 Day12 要用扩散、Day11 要用 flow 赎回的那一笔。OpenVLA 的 1–99 percentile 定 bin 是表示内的小修：不用 min/max 定范围，避免离群动作吞掉有效分辨率。
3. **语义重组，不是技能创造**：RT-2 的消融钉死了边界——co-fine-tuning 优于只训机器人数据、55B 优于 5B，但 web 预训练不会给模型训练数据里不存在的新运动。换句话说， $p_\theta$ 的动作支撑集由机器人演示决定，web 数据只改条件分布的语义索引。Day34 π0.7 的「组合泛化」之所以是进展，正因为它把这个重组能力从定性示例推进到可测的任务组合。
4. **延迟即动态**：RT-2 55B 云端 1–3 Hz、OpenVLA bf16 约 6 Hz，而 int8 因量化算子开销掉到 1.2 Hz 时真机成功率反而下降、更快的 int4 接近 bf16（NOTES §4 自记）。离线 action-token accuracy 超过 95% 不等于闭环好：控制周期进入了被控对象的有效动力学，评测必须报告 achieved control Hz 与 p95 latency，不能只报 token 级指标——这是本篇对 Day28/30 评测线最硬的一条遗产。

### 边界

1. **证据没有证明什么**：OpenVLA 超 RT-2-X 的 16.5 个百分点是数据规模与清洗、双视觉编码器、架构三者的合成，论文未隔离单因子，不能引用为「开源 7B 架构更强」；OpenVLA 在困难语义泛化上仍逊于 RT-2-X，聚合成功率掩盖了这一轴（NOTES §5 自记）。RT-2 的 emergent 能力以分类评测与示例为主，不是可复现的成功率曲线。
2. **何时失效**：接触丰富的精细操作（量化吃精度、多峰被自回归顺序压平）、需要历史与本体的任务（单帧、无 proprioception、无 action chunking，成功率通常仍低于 90%）、高频稳定任务（1–6 Hz 进不了 Day08 的稳定环）、以及演示支撑集之外的全新运动（语义再对也变不出没练过的动作）。
3. **系统边界**：动作合法性、安全约束与超时不在模型内，由外部低层控制器与人工规则兜底；把 Day09 单独部署成端到端控制器，等于把 Day29 的安全层整层拆掉。

### 迁移到 post-training / Agentic RL Infra

可执行的同构：把「co-fine-tuning 保语义」做成 agentic post-training 的防遗忘门禁。每次混入新域 tool-use 数据训练后，不只看新域成功率，必须同时跑三桶回归——旧域任务成功率、语义新颖桶（未见指令表述与对象指代）、以及端到端 p95 latency 与实际 achieved step 频率；任一桶相对上一 checkpoint 出现排序翻转，就回滚该域的 mixture 采样权重并把翻转样本回流做 failure triage。配套记录直接抄本篇 NOTES 的 transfer 清单：per-dataset loss、mixture sampling weight、action-token（tool-call token）accuracy、按任务与语义新颖度分桶的成功率。验收标准写死：离线 token accuracy 上升但闭环成功率或控制频率下降的 checkpoint 不放行——对应本篇 int8 的教训。

### 思考题（综合 Day08 / Day09 / Day11 / Day12 / Day15 / Day29 / Day34）

- **(a) 表示赎回的顺序**：Day09 的三笔抵押——256 bin 量化、固定顺序自回归、单步无 chunking——Day11 用 flow 一次赎回「连续加整块」，Day12 用扩散赎回「多峰」，但两者都放弃了与文字共用词表的接口红利。若只能保留一项 Day09 接口（词表共享 / web co-fine-tuning / 自回归 serving 栈），哪一项对 Day34 式组合泛化的贡献最不可替代？请用「动作支撑集由演示决定、web 只改语义索引」这条边界论证，并设计一个能把三项贡献分开测的消融。
- **(b) 频率分工的合同**：Day09 在 1–6 Hz 出末端增量，Day08 在 100 Hz 站稳。若把 OpenVLA 直接接到一个人形机器人上做家庭整理，上层语义错一次（抓错物体）与下层稳定慢一拍（失衡）分别由谁的指标先报警？请按 Day29 的安全四层栈，给这条 VLA 到 locomotion 的接口写一份合同：哪些检查在 token 解码后、哪些在 PD 之前、超时多久触发降级动作，并说明为什么 token 级 accuracy 不能充当任何一层的放行指标。
- **(c) 16.5 个百分点该怎么拆**：OpenVLA 的优势被归因于数据混合、双编码器与架构三者，Day15 的教训又是「异构数据要在线监控兼容性」。若算力只够跑两个消融，你选哪两个因子先拆？请写清每个消融固定什么、变什么、看哪一桶（29 任务聚合 / 困难语义桶 / Franka 适配桶），并预判哪种结果会推翻「接口复用 > 表示保真」这个 Day09 命题本身。

参考链接（本轮复习）：
- RT-2（arXiv，CoRL 2023）：https://arxiv.org/abs/2307.15818
- OpenVLA（arXiv，CoRL 2024）：https://arxiv.org/abs/2406.09246
- OpenVLA 正式版（PMLR v270）：https://proceedings.mlr.press/v270/kim25c.html
- OpenVLA 项目页：https://openvla.github.io/

GitHub NOTES：https://github.com/Papa-Panda/post-training/blob/master/physical-ai/day-09-2024-rt2-openvla/NOTES.md
