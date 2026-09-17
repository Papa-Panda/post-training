# Day 25 — Gato：一个网络，604 个任务 — generalist agent 的范式宣言

## 元信息
- Title: A Generalist Agent（Gato）
- Authors / Org: Scott Reed, Konrad Zolna, Emilio Parisotto 等（DeepMind；TMLR 2022）
- Link / arXiv: https://arxiv.org/abs/2205.06175?context=cs.LG（v1 2022-05-12；TMLR 11/2022 接收）
- 官方博客（含训练/部署示意图）: https://deepmind.google/blog/a-generalist-agent/
- Official code: DeepMind 未公开官方实现（诚实标注）；核心是标准 decoder-only Transformer + BC，无私有算子，可复写
- Date read: 2026-09-16
- Tags: [physical-ai, gato, generalist-agent, tokenization, behavior-cloning, multi-embodiment, deepmind]
- Thread: physical-ai
- Folder: day-25-2022-gato
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-25-2022-gato

## 一句话总结
Gato 把"一个网络做一切"从口号变成可训练的工程：1.2B decoder-only Transformer，把 604 个任务（Atari、DM Lab、Meta-World、真机 Sawyer 叠方块、看图说话、对话）的观测 / 动作 / 文本全部序列化成**同一种 token 流**，用带 mask 的自回归 next-token 预测一次训完；部署时靠 prompt（一段演示）指定任务，同一套权重既能聊天又能输出关节力矩——代价是离散化精度损失和单任务有效容量被摊薄，论文自己把出路押在 scale 上。

## 和之前工作的关系

- **接了哪条线**：Day25–30 "Physical AGI / Eval / Safety" 块的开篇。Day01–24 搭好了三条腿（仿真 / 世界模型 / 控制、VLA、RL + sim2real），Gato 把"generalist"第一次写成可复现的训练配方：**统一序列建模**。
- **补了哪个短板**：Day09 RT-2 / OpenVLA、Day11 π₀、Day12 Diffusion Policy、Day13 Octo 都是"机器人域内"的 generalist（跨 embodiment，但不出机器人域）。Gato 证明连"机器人 vs 聊天 vs 打游戏"的**域边界**都可以不要——模态和 embodiment 只是 token 流里的不同位置。
- **替代 / 分叉 / 改进**：不是替代 Day09–14 的 VLA 路线，而是它们的**祖先**：RT-1 / RT-2 的 action tokenization 直接继承 Gato 的离散化路线；π₀ 的 flow matching、Diffusion Policy 的 DDPM 则是对 Gato 离散精度损失的**分叉修正**（连续动作头）。
- **对之前 Day X 的直接对比**：
  - **vs Day09（RT-2）**：RT-2 = Gato 的 token 哲学 + web-scale 语义迁移。Gato 靠 604 任务混合学"跨域共享"，RT-2 靠 web 预训练学"语义先验"——两种不同的泛化货币。
  - **vs Day11（π₀）/ Day12（Diffusion Policy）**：Gato 用 1024-bin 离散 BC 做动作，π₀ 用 flow matching、DP 用 DDPM 做**连续**动作生成。离散 token 在精细接触任务里是精度瓶颈——这是 Day11 / 12 存在的核心理由之一。
  - **vs Day13（Octo）**：Octo 也是 generalist，但用"统一 backbone + 可插拔 diffusion readout"——对 Gato "一根 token 流走到底"的**模块化修正**：共享表征，专用解码。
  - **vs Day06（DreamerV3）**：都是序列模型，但 Dreamer 是 latent RSSM + imagined actor-critic（学世界模型 + RL），Gato 是 flat token + 纯 BC。Gato **不预测观测**（loss 只打在动作 / 文本 token 上），所以它不能做 planning——这是 "policy as sequence model" vs "world model as sequence model" 的分野。
  - **vs Day19–24（RL 块）**：Gato 是纯 BC，没有 RL。RL 块回答"如何超越演示者"，Gato 回答"如何一个网络装下所有演示者"——正交的两轴，合起来才是后来的路线（拿 RL 微调 generalist）。
  - **vs Day15（OpenX / RT-X）**：Gato 的 604 任务混合是 OXE 跨 embodiment 数据思想的**概念祖先**；区别是 Gato 连非机器人任务也混进来了。

## 为什么今天读它

Roadmap 把 Day25–30 定为 "Physical AGI / Eval / Safety"。Gato 是这个块的范式宣言：它第一次用可训练的系统回答"Physical AGI 的模型长什么样"——不是分模块的 pipeline，而是一个**统一的序列模型**，任务由 prompt 指定。读完它再看 Day26 GR00T N1（humanoid foundation model）、Day27 Cosmos（世界基础模型造数据）、Day28 评测、Day29 安全，才有坐标系：Gato 定义了 generalist 的**上限想象**，后面几天逐一检视它的**代价**（容量税、精度损失、评测缺失、安全无保障）。

## 今天的 3 问
1. μ-law + 1024-bin 离散化：连续动作经 companding 后均匀量化，近零区分辨率最细、两端最粗。这个量化误差在什么量级上会吃掉接触丰富（contact-rich）任务的控制精度？和 Day12 diffusion 的连续建模相比，离散 BC 的精度损失有没有可计算的下界？
2. Loss 只打在动作 / 文本 token 上，观测 token 被 mask 掉不预测——Gato 因此**没有世界模型**，不能做 planning / imagination（对比 Day06 Dreamer）。如果让它也预测观测 token，会得到什么、失去什么（容量？训练稳定性？）？
3. "Generalist tax"：1.2B 参数、604 任务，单任务有效容量被摊薄——论文自己承认多数任务上不如专用模型，押注 scale。RT-2（55B）和后来的 VLA 把这个赌注兑现了多少？精度和容量的 trade-off 有没有出现拐点？

## 核心
1. **Motivation**: 2022 年之前，机器人、游戏、对话各有各的网络和训练范式；LLM 证明了"一个 Transformer + next-token 预测"可以吞掉整个文本域。Gato 问：这个配方能不能吞掉**动作**？baseline 的问题不是性能，而是**范式碎片化**：每个 embodiment 都要重搭一套感知-决策 pipeline。与 Physical AGI 的关系：Gato 是第一个把 "generalist agent" 从哲学口号变成**可训练、可评测**的工程实体的尝试——哪怕它在很多任务上只是 mediocre（ZDNet 当时的刻薄标题恰恰点中要害：意义不在单点 SOTA，在范式验证）。
2. **System / Method**: 万物皆 token 流（见数学视角 §1–2）：
   - 文本 → SentencePiece（32k 词表）；图像 → 16×16 patch 切块 + 线性投影（ViT 式，连续 embedding，不离散化）；连续观测（proprio）与连续动作 → μ-law companding 后均匀量化为 1024 bins；离散动作（Atari 按键）→ 整数 token。
   - 训练序列形如 `[prompt tokens, o_0, a_0, o_1, a_1, …, o_T, a_T]`，decoder-only Transformer（24 层 / 2048 维 / 16 头 / 上下文 1024）做自回归建模。
   - Loss 只在**动作 token 和文本 token**上计算（观测 token 被 mask）——"只预测要输出的东西"。
   - 部署：prompt（一段成功演示的 token 化）+ 环境观测 → 自回归采样动作 token → 解码为连续动作 → 执行 → 循环。任务切换 = 换 prompt，**权重不动**。
3. **Training / Data Details**: 604 个任务一次混合训练：ALE Atari、DM Lab、DM Control、Meta-World、真机 RGB Stacking（Sawyer 臂叠方块）、图像描述、VQA、对话（MassiveText 子集）。Sim 数据（游戏 / 仿真机器人）+ Real 数据（真机叠方块、人类标注的图文对话）统一进同一个 token 流；采样权重手工平衡各域配比。Reward / verifiable signal：**没有 RL，没有 reward**——纯监督 BC，能验证的只有"token 预测对不对"。Sim2Real 在这里退化成"真机数据也是训练混合的一部分"，没有专门的 transfer 机制（和 Day19–24 的整套方法论形成对照）。
4. **Key Tricks**: ① **μ-law companding 再量化**：小动作（精细修正）分得多 bins、大动作分得少——把"控制精度需求"编码进 token 分配，这是从电话语音编码借来的老智慧；② **loss masking**：不预测观测，只预测动作 / 文本——省掉大半建模负担，让 1.2B 参数装得下 604 任务（代价：失去世界模型，见第 2 问）；③ **prompt 即任务接口**：不靠 task id、不靠 finetune 权重，靠"演示的 token 前缀"指定任务——后来整个 prompt-based robot learning（含 in-context RL）的接口祖先。
5. **Results**: 604 任务中约 450 个达到专家分数 50% 以上（论文主结果）；真机 Sawyer 叠方块达到专用 BC 策略水平；Atari 上超人类但不如专用 SOTA；看图说话 / 对话质量 mediocre。Held-out 任务上 finetune 快于从零训练——"generalist 权是好的初始化"是第二个可验证结论。诚实注：**没有官方开源实现**，以上数字来自论文 / 博客原文。

## 数学视角

### 0. 符号与维度（先摆清楚）
- $\tau\in\{1,\dots,604\}$ ：任务 id；每个任务有自己的观测空间 $\mathcal{O}_\tau$ 与动作空间 $\mathcal{A}_\tau$ （维度、模态都不同）。
- $T_\tau$ ：任务相关的 tokenizer，把 $(o_t,a_t)$ 映成 token 序列。
- $s=(s_1,\dots,s_L)$ ， $L\le 1024$ ：一条训练序列，形如 $[\text{prompt},o_0,a_0,o_1,\dots,o_T,a_T]$ 的 token 化。
- $\theta$ ：同一套 Transformer 权重（1.2B，不随 $\tau$ 变化）。
- $M\subset\{1,\dots,L\}$ ：loss 的 mask 集合——只含动作 token 与文本 token 的位置。

### 1. 万物皆 token：μ-law 离散化

连续标量 $x\in[-1,1]$ （关节角、力矩、proprio 读数）先做 μ-law companding：

$$F(x)=\mathrm{sign}(x)\frac{\ln(1+\mu\lvert x\rvert)}{\ln(1+\mu)}$$

再把 $F(x)\in[-1,1]$ 均匀切成 1024 个 bin。companding 的导数 $F'(0)=\mu/\ln(1+\mu)\gg 1$ 意味着**零附近分辨率最细**——小修正（精细操作恰恰发生在零附近）分得最多 bins，这是把控制先验写进 token 分配。代价：量化是不可逆的信息损失，bin 宽即动作精度的硬下界——Day11 / 12 用连续生成头替代离散 token，正是要拿回这部分精度。

### 2. 训练目标：带 mask 的自回归 BC

$$p_\theta(s)=\prod_{i=1}^{L}p_\theta(s_i\mid s_{<i}),\qquad \mathcal{L}(\theta)=-\mathbb{E}_{\tau\sim p(\tau)}\mathbb{E}_{s\sim\mathcal{D}_\tau}\left[\sum_{i\in M}\log p_\theta(s_i\mid s_{<i})\right]$$

三点解释：① 这就是 LLM 的 next-token 目标，只是"语言"换成了"观测-动作交错流"；② $p(\tau)$ 是任务采样权重——**数据配比是超参**，决定 1.2B 容量在 604 任务间的分配，配比失衡 = 某些任务被饿死；③ mask $M$ 剔掉观测 token：模型只学"给定历史该输出什么"，不学"世界会怎么变"——所以 Gato 是 policy，不是 world model（对比 Day06 Dreamer 的 $\mathcal{L}_{\text{dyn}}$ ）。

### 3. 部署：prompt 条件化的自回归控制

给定 prompt token 前缀 $P$ （一段演示）与历史 $h_t$ ，每步逐 token 自回归采样动作，再经反量化与 μ-law 逆变换解码：

$$a_t\sim p_\theta(\cdot\mid P,h_t),\qquad \hat a_t = F^{-1}(\mathrm{dequantize}(a_t))$$

环境执行 $\hat a_t$ 后返回 $o_{t+1}$ ，token 化并追加进上下文（上限 1024）。时间尺度：每步 token 预算 ≈ 图像 patch 数 + proprio bin 数 + 动作 bin 数（几十量级）→ 上下文只装得下约 10–20 步历史——**Gato 的"记忆"是以 token 步数计的**，长程任务的早期信息会被滑出窗口，这是纯自回归控制的结构性短板（Day11 π₀ 的 50-step action chunk 与历史压缩，可视为对此的回应）。

### 4. Generalist tax：容量与精度的双重税

同一套 $\theta$ 要同时拟合 Atari 按钮时序、Sawyer 力矩映射和图像描述——单任务有效容量被 604 分摊，这是**容量税**；1024-bin 量化是**精度税**。论文的 scaling 赌注是把两项税都看成"规模不够"的暂时现象。但 Day09–14 的后续史给出了更细的答案：RT-2（55B）证明 scale 确实能同时减税；Octo / π₀ 则证明**架构修正**（模块化 readout、连续动作头）是另一条减税路线——"大力出奇迹" vs "结构出效率"，两条路线在 Day26 GR00T N1 还会再碰头。

### 5. 数学没有覆盖的部分
- **安全**：BC 目标里没有约束项，换一个 prompt 演示就能让同一套权重输出危险动作——对齐 / 安全完全缺席，这是 Day29 要补的。
- **评测**："450 / 604 达到专家 50%"是**自选阈值**下的计数，不同任务的"专家分数"不可比——generalist 的评测方法论本身是 open problem，Day28 回来算账。
- **数据配比 $p(\tau)$ 的原理**：论文靠手工调，没有"最优混合比"的理论——Day27 Cosmos（用世界模型造数据）某种意义上是在回答"数据不够时怎么办"的另一半。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？模型 vs 框架哪个贡献更大？Held-out 任务 finetune 快于从零训——**框架贡献**（统一序列接口）大于单点模型贡献：换个新机器人，只要能写出 $T_\tau$ （tokenizer），就能接入同一套训练 / 部署管线。真机 RGB stacking 的成功则说明：在数据覆盖的任务上，generalist 权重不拖后腿。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：① "统一接口吞掉异构"是可迁移的**架构模式**：Gato 用 token 流统一 604 任务，正如 post-training 用统一 rollout 接口吞掉异构环境——设计系统时先定接口（tokenizer / rollout schema），再谈模型；② μ-law 量化是"把领域先验编码进表示"的范例：做 infra 时，观测 / 动作的表示设计（归一化、分桶、压缩）是和模型同等重要的杠杆，别只调模型。
- Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：Gato 把"加一个新 embodiment"的成本从"重搭 pipeline"降到"写一个 tokenizer + 配数据权重"——这是数量级的工程成本下降；但代价是评测复杂度爆炸（604 任务 × 各自的专家基线），"generalist CI"至今没有标准答案——这正是 Day28 评测篇的动机。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：任务采样权重 $p(\tau)$ 在论文里是手工定的——有没有办法从"各任务梯度冲突 / 遗忘曲线"反推最优配比？这和 LLM post-training 的 data mixture 问题是同一个数学问题吗？
- 如果要复现 / 小规模试，第一个实验做什么？拿 3 个差异大的任务（如一个 Atari 游戏、一个 DM Control 任务、一个图像描述），用同一套 GPT-2 级小模型 + μ-law / 整数 token 化跑通"混合训练 → prompt 切换"最小闭环，先验证"接口统一"本身 work，再谈规模。

## 原文金句 (1-2句)
> "The same network with the same weights can play Atari, caption images, chat, stack blocks with a real robot arm and much more, deciding based on its context whether to output text, joint torques, button presses, or other tokens."
> "During the training phase of Gato, data from different tasks and modalities are serialised into a flat sequence of tokens, batched, and processed by a transformer neural network similar to a large language model. The loss is masked so that Gato only predicts action and text targets."（DeepMind 博客）

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节（含 vs Day09 / 11 / 12 / 13 / 06 / 19–24 / 15 七组对比）
- 新增「数学视角」§0–§5：符号维度表 → μ-law 离散化 → mask 自回归 BC → prompt 条件化部署 → generalist tax → 数学未覆盖部分（安全 / 评测 / 数据配比）
- README 两处打 ✅ 2026-09-16；reading-log.csv 新增一行；commit + push

## 连接
- 上一篇: day-24-sim2real-system-identification（SimOpt：分布辨识校准 sim2real）
- 下一篇预告: day-26-2025-groot-n1（GR00T N1：humanoid foundation model，双系统 reasoning + diffusion control——看 generalist 范式在 humanoid 上的下一站）
