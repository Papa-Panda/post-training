# Day17 — BridgeData V2：廉价遥操作的 imitation data 规模化

> Homer Walke、Kevin Black 等 / UC Berkeley、Stanford、Google DeepMind、CMU，arXiv 2308.12952（CoRL 2023）。数据三角的"廉价采集"边：用约 \$4,000 的公开 WidowX 250 廉价臂 + VR 遥操作，在 24 个玩具厨房/桌面环境按"每场景多任务可行"协议采集 60,096 条轨迹（50,365 条人工演示 + 9,731 条脚本化 pick-and-place），全部带事后众包语言标注；同一套数据跑通 6 种 offline 方法，钉死"技能多样性"这条覆盖轴。

## 元信息
- Title: BridgeData V2: A Dataset for Robot Learning at Scale
- Authors / Org: Homer Walke、Kevin Black、Abraham Lee（UC Berkeley）、Moo Jin Kim、Max Du（Stanford）、Chongyi Zheng（CMU）、Tony Zhao（Stanford）、Philippe Hansen-Estruch（UC Berkeley）、Quan Vuong（Google DeepMind）、Andre He、Vivek Myers、Kuan Fang（UC Berkeley）、Chelsea Finn（Stanford）、Sergey Levine（UC Berkeley）/ UC Berkeley、Stanford、Google DeepMind、CMU
- Link / arXiv / Blog: https://arxiv.org/abs/2308.12952 ；项目页 https://rail-berkeley.github.io/bridgedata/ （数据集下载、可视化、评测视频）；官方代码 https://github.com/rail-berkeley/bridge_data_v2 （Jax 实现：GCBC / D-GCBC / GC-IQL / CRL / LCBC + 预训练 checkpoint）
- Date read: 2026-09-08
- Tags: [physical-ai, bridgedata-v2, robot-data, imitation-learning, offline-rl, skill-diversity, low-cost-robot, widowx, language-conditioning, goal-conditioning]
- Thread: physical-ai
- Folder: day-17-2023-bridgedata-v2
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-17-2023-bridgedata-v2

## 一句话总结
BridgeData V2 把"机器人数据规模化"做成一个 \$4,000 买得起的配方：单台 WidowX 250 廉价臂、VR 遥操作，在 24 个玩具厨房/桌面环境里按"每场景多任务可行"的协议采集 60,096 条轨迹（50,365 条人工演示 + 9,731 条脚本化 pick-and-place），全部带事后众包语言标注；然后用同一套数据跑通 6 种 offline 方法（目标条件/语言条件的 IL + 目标条件 RL），证明三件事：模型容量上升带来单调增益、数据量上升带来增益、**技能多样性 13 vs 3 在数据量相当时把未见 pick-and-place 从 0.30 拉到 0.65**——覆盖轴的第三条边（skill 轴）被钉死了。

## 和之前工作的关系

- 接了哪条线：Day15–18 "Robot Data / Benchmark 扩展"专题的第三棒。Day15 OXE 回答"怎么聚合跨身体数据"，Day16 DROID 回答"真实场景覆盖怎么带来增益"，Day17 回答"低成本 + 多技能覆盖怎么做"——三篇正好构成数据三角的三条边：聚合（OXE）、场景覆盖（DROID）、廉价技能覆盖（Bridge）。
- 补了哪个短板：BridgeData V2 是 OXE 60 个数据集中最大的组成之一——读 Bridge 是在读 OXE 地基里最大的一块砖；RT-1（Day09 RT-2 的前身）最早就是在 Bridge 系列数据上验证的，Day09 的"大模型吃机器人数据"故事缺了 Bridge 这一环就不完整。
- 替代 / 分叉 / 改进：
  - 相对 Day16 DROID：**正交取舍**——DROID 固定贵臂（Franka）扩张场景轴（564 真实场景），Bridge 固定便宜臂（WidowX）扩张技能轴（13 技能、24 环境）；两者都证明"支撑扩张 > 单点加密"，只是扩张的轴不同。
  - 相对 Day15 OXE：Bridge 是 OXE 的"上游原料"，OXE 解决跨身体聚合，Bridge 解决单身体内的技能密度——OXE 的混合权重问题（Day15/16 的疑问）在 Bridge 这里有干净得多的对照实验可学。
- 对之前 Day X 的直接对比：
  - Day12 Diffusion Policy：Bridge 评测的 D-GCBC 就是 Day12 那套 DDPM loss 的目标条件版本——Day12 的数学在这里是"被评估的方法"之一；D-GCBC 出现 jerky 行为（扩散 head 无观测历史 → 在 mode 间振荡）正好呼应 Day12 的 $T_o$ / $T_p$ / $T_a$ 三 horizon 设计的必要性。
  - Day11 π₀ / ACT：Bridge 评测的 ACT 也是"一次预测一段动作"（action chunk + CVAE）——和 π₀ 的 50-step action chunk 同一家族，只是参数化不同（CVAE vs flow matching）。
  - Day09 RT-2 / RT-1：RT-1 在 Bridge 上评测（离散 7 维动作、256 bin、观测历史）——Day09 的 tokenization 数学在这里直接复用；RT-1 远超 LCBC（0.49 vs 0.23）说明"大模型 + 离散化 + 历史"三件套在语言条件 IL 里是真增益，不是玄学。
  - 2026-09-07 问答"固定硬件路线"：BridgeData V2 是 repo 里第四个固定硬件代表（Day09 RT-2 / Day12 / Day16 DROID / Day17 Bridge）——"固定硬件 + 便宜 + 玩具厨房"的经典路线，一次只扩张一个多样性轴。

## 为什么今天读它
数据三角的"廉价采集"边必须有一篇代表作；更重要的是它给出了数据论文里少见的**干净消融**：等量数据下 13 skills vs 3 skills（28k vs 27k 轨迹）——把"多样性"和"数据量"两个变量解耦，这是 Day15（OXE 没公开混合权重）和 Day16（DROID 没做等量消融）都没做干净的事。读完这篇，"覆盖三轴"（embodiment × scene × skill）的框架就闭环了。

## 今天的 3 问
1. 采集协议刻意让每个场景"多任务可行"（不 reset、随时换任务），迫使策略必须看任务条件 $c$ 而不是从观测 $o$ 推断任务。用互信息怎么形式化这个设计？旧 Bridge（单场景任务少）的数据分布里 $I(\text{task}; o) \approx H(\text{task})$ 会导致什么捷径？（见数学视角 §2：策略学 $\pi(a \mid o)$ 捷径，条件 $c$ 的梯度约等于 0；Bridge V2 用高 $H(\text{task} \mid o)$ 逼出 $I(c; a \mid o) > 0$ 。）
2. 13 skills vs 3 skills、数据量相当（28k vs 27k），未见 pick-and-place 从 0.30 升到 0.65——sweeping、folding 这些"看起来不相关"的技能为什么能帮到 pick-and-place？是共享表征（视觉编码器、抓取几何）还是状态覆盖？（见数学视角 §3：共享编码器 $\varphi$ 的训练支撑被撑大，未见任务的观测落进已见支撑——"轴扩张 > 点加密"在 skill 轴上的版本。）
3. 16% 的脚本化次优数据对 BC 是噪声、对 CRL（offline RL）是礼物。从"分布覆盖 vs 分布匹配"的角度，次优数据到底补了什么？为什么 offline RL 需要它而 BC 不需要？（见数学视角 §4：BC 要 $p_{\text{expert}}(a \mid o, c)$ 的分布匹配，offline RL 要 $\text{supp}(p_{\text{data}})$ 的分布覆盖以约束 value 外推。）

## 数学视角：六种方法一个目标 × 覆盖三轴的第三条边

Bridge 是数据论文，数学就是 **offline 学习的目标函数 + 覆盖轴的泛化直觉**。统一框架：条件行为克隆 $\pi(a \mid o, c)$ 及其 value 形式。

### 1) State / observation / action / objective

- **Observation** $o$ ：128×128 RGB 过肩主视角图像（训练只用这一路；RT-1 用 320×256 + 观测历史）。原始采集是 640×480，另有两路随机位姿 RGB + 腕部 RGB + RGBD 深度（多数轨迹只有主视角）。
- **Action** $a$ ： $a = (\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g)$ ，前 6 维是末端执行器笛卡尔相对位姿变化（连续）， $g \in \{0, 1\}$ 是夹爪开合（离散）；控制频率 5 Hz，平均轨迹 38 步（约 7.6 秒）。
- **Condition** $c$ ：目标图像 $g_{\text{img}}$ （GCBC / D-GCBC / ACT / CRL）或语言指令 $\ell$ （LCBC / RT-1）。
- **六种方法，一个目标家族**（全部在学 $p(a \mid o, c)$ 或其 value 形式； $\theta$ 是参数， $\mathcal{D}$ 是 BridgeData V2 离线数据集）：
  - GCBC： $\mathcal{L}_{\text{BC}}(\theta) = -\mathbb{E}_{(o,a,c) \sim \mathcal{D}}[\log \pi_\theta(a \mid o, c)]$ ， $o$ 经 ResNet-34 编码的朴素目标条件 BC。
  - D-GCBC： $\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{k,\epsilon}[\|\epsilon - \epsilon_\theta(a^k, o, c, k)\|_2^2]$ ， $a^k$ 是第 $k$ 步加噪后的动作， $k = 1..K$ 是扩散步， $\epsilon \sim \mathcal{N}(0, I)$ ——Day12 的 DDPM 数学，加了条件 $c$ 。
  - ACT：预测动作块 $A = (a_t, \dots, a_{t+H-1})$ ， $H$ 是块长度；CVAE 的 ELBO 为 $\mathbb{E}_{q(z \mid A,o,c)}[\log \pi_\theta(A \mid o, c, z)] - \mathrm{KL}(q(z \mid A,o,c) \| p(z))$ ， $z$ 是捕捉演示多峰性的隐风格变量，transformer 做编码。
  - CRL：目标条件 value 函数 $Q(o, a, g)$ 参数化为对比表示的 log-linear 形式，把 goal-conditioned RL 写成表示学习问题。
  - LCBC： $\ell$ 经 MUSE 句向量编码，ResNet-34 + FiLM 条件注入视觉特征，MLP 策略头输出动作；FiLM 即用语言向量对视觉特征做缩放/平移调制。
  - RT-1：每维动作离散成 256 个 bin， $p_\theta(t_{1:7} \mid o_{\text{hist}}, \ell) = \prod_{j=1}^{7} p_\theta(t_j \mid t_{<j}, o_{\text{hist}}, \ell)$ ，EfficientNet token + decoder-only transformer——Day09 RT-2 数学的直接复用， $o_{\text{hist}}$ 是观测历史。
- 对应到实现：官方 Jax 仓库 `rail-berkeley/bridge_data_v2` 提供 GCBC / D-GCBC / GC-IQL / CRL / LCBC 五套训练代码与预训练 checkpoint，`experiments/train.py` 一键切换 `--config ...:METHOD`。

### 2) 任务注意力协议的互信息形式化：用数据分布设计代替 loss 设计

旧 Bridge：每个场景只有少数预设任务， $I(\text{task}; o) \approx H(\text{task})$ ——从观测几乎能确定任务，策略学到捷径 $\pi(a \mid o)$ ，条件 $c$ 的梯度约等于 0，语言/目标条件形同虚设。

Bridge V2 的采集协议（三件套：每场景多任务可行、不 reset 连续采集、每 50 条随机化相机/物体/工作区）把 $H(\text{task} \mid o)$ 推高：同一张观测可能对应"把勺子放进碗里""擦桌子""开抽屉"中的任意一个。要降低 $\mathcal{L}_{\text{BC}}$ ，策略必须让 $I(c; a \mid o) > 0$ ，即**真正使用条件**。这是"数据集设计 ⇒ 归纳偏置"的干净例子：不是改 loss，而是改数据分布让捷径消失。

工程对应：语言标注是事后众包的（标注员根据轨迹首末帧描述"物体最终位置"，部分由 Microsoft Research 协助标注）——所以 $\ell$ 是对轨迹的**后验描述**而非采集意图。这解释了评测现象：语言方法在未见物体上吃亏（物体名在训练分布里没被 ground 过， $p(\ell_{\text{new}})$ 落在 MUSE 嵌入空间的未见区域）。

### 3) 技能多样性的 scaling：覆盖三轴的第三条边

统一框架（串联 Day15/16/17）：泛化误差 $\lesssim d(P_{\text{test}}, \text{supp}(P_{\text{train}}))$ ， $P_{\text{test}}$ 是部署分布， $\text{supp}(P_{\text{train}})$ 是训练混合的支撑。三篇各扩张一条轴：

- Day15 OXE：embodiment 轴（22 种身体，场景基本固定在实验室）；
- Day16 DROID：scene 轴（564 真实场景，固定 Franka 身体）；
- Day17 Bridge：skill 轴（13 技能、24 环境，固定 WidowX 身体）。

Bridge 的关键消融把"多样性"和"数据量"解耦：28k 轨迹 / 3 skills vs 27k 轨迹 / 13 skills， $N$ 相当，未见 pick-and-place 成功率 0.30 → 0.65。注意这比"加数据"更强：** $N$ 固定时，多样性本身带来超过 2 倍增益**。

机制解释（多任务表示学习）：设共享编码器 $\varphi: o \mapsto z$ ， $z$ 是视觉-运动表征，策略 $\pi(a \mid z, c)$ 。13 个技能共享 $\varphi$ 和底层原语（抓取几何、接触前对齐、手眼协调）；sweeping / folding 的数据虽然任务不同，但它们把 $\varphi$ 的训练支撑撑大，让未见 pick-and-place 的观测 \$o'\$ 落进已见支撑——于是 $d(P_{\text{test}}, \text{supp}(P_{\text{train}}))$ 缩小。这是 Q2 的答案：不是"sweeping 教会了 pick-and-place 扫地"，而是"所有技能合起来把表征的支撑铺得更宽"；也是 Day16"支撑扩张 > 单点加密"在 skill 轴上的版本。

### 4) 次优数据的双面性：BC 要分布匹配，offline RL 要分布覆盖

16% 脚本化 pick-and-place 数据（高随机、常失败）：对 BC 是标签噪声—— $\mathcal{L}_{\text{BC}}$ 会把它当专家拟合，稀释 $p_{\text{expert}}(a \mid o, c)$ ；对 CRL 这类 offline RL 是覆盖礼物—— $Q(o, a, g)$ 的估计需要在 \$(o, a)\$ 空间有支撑，随机策略的宽覆盖缓解了 value 外推高估。形式化：BC 要的是**分布匹配**（mode 越干净越好），offline RL 要的是**分布覆盖**（ $\text{supp}(p_{\text{data}})$ 越宽，value 在越宽的动作上被约束）。论文明说次优数据可被 offline RL 利用——数据论文里少见的诚实：**同一批数据对不同算法家族价值符号相反**，采集时就想好"这批数据是给谁吃的"。

### 5) 和系统实现的对应

- **硬件 infra**：WidowX 250 六自由度臂 + VR 遥操作，整套约 \$4,000、全公开采购、两周到货——"把数据采集变成本科生 lab 也能复现的事"；这是 Bridge 路线相对 DROID（Franka 贵臂 + 52 栋建筑物流）的根本 infra 差异。
- **采集 infra**：不 reset 连续采集（省掉 reset 的人力/时间税）；每 50 条随机化相机位姿 + 物体 + 工作区位置；事后众包语言标注。
- **数据 infra**：JPEG/PNG + pkl 原始包；转换链 raw → numpy → TFRecord（`data_processing/`）；TFDS RLDS 版（256×256 下采样）+ 推荐 Octo 的 data loader——直接喂给 Day13 Octo 那套生态；CC-BY 4.0。
- **评测**：10 trials/任务（scaling 消融 20 trials），seen / unseen / 跨机构三档；RT-1 在跨机构 zero-shot 退化最小（平均 0.47 → 0.40），语言方法退化最大——容量和离散化带来鲁棒性。

### 6) 假设与数学没有覆盖的真实误差

- 任务全是低精度操作（论文 Discussion 自认）：无大力/动态/高精度插入——泛化结论的外推边界是"玩具厨房级"任务； $\mathcal{L}_{\text{BC}}$ 在这套数据上可解，不代表在精细操作上可解。
- 全部数据单机构采集（Berkeley），"跨机构"评测只是 Lab1 → Lab2 的一次验证——环境多样性的真实上限不如 DROID 的 52 栋建筑。
- 语言标注是事后众包：标注员没看采集意图，只能从首末帧推断—— $\ell$ 与真实意图的对齐有噪声；未见物体名直接 OOV，这是语言方法 unseen 任务拉胯的结构性原因，数学上即 $p(\ell)$ 的支撑缺口。
- 单一 embodiment（WidowX）：论文自己说下一步是 multi-robot 数据集——这正是 Day15 OXE 回答的问题，Bridge 是 OXE 的"上游原料"之一。
- 平均轨迹 38 步 @5Hz ≈ 7.6 秒：全是短时程技能，长程任务组合不在验证范围（Day14 π₀.₅ 的子任务分层是另一条线）。

## 核心
1. **Motivation**: CV/NLP 靠"大数据 + 大模型"拿到泛化，机器人学想抄这条路，但已有数据集要么单环境（换个 lab 就用不了）、要么私有机器人（学术界复现不起）、要么单任务（策略从观测猜任务、不看任务条件）。问题：能不能用 \$4,000 的公开硬件，做出一个"别的 lab 拿去就能用"的多任务数据集？和 Physical AGI 的关系：这是数据飞轮里"规模化采集"那一环的平民化方案。
2. **System / Method**: WidowX 250 + VR 遥操作 + 四路相机（RGBD 过肩 + 两路随机位姿 RGB + 腕部 RGB，640×480，5 Hz）；采集协议三件套：每场景多任务可行、不 reset 连续采、每 50 条随机化；50,365 条人工演示 + 9,731 条脚本化 pick-and-place；全部事后众包语言标注；7 维动作（6D 笛卡尔相对 + 夹爪离散）。
3. **Training / Data Details**: 60,096 轨迹 / 24 环境 / 13 技能 / 100+ 物体；CC-BY 4.0；TFRecord + TFDS RLDS 双格式；评测 6 种方法（GCBC、D-GCBC、ACT、CRL、LCBC、RT-1），10 trials/任务；Sim 数据：无；Reward：纯 BC / 对比 RL，无外部 reward。
4. **Key Tricks**（3个最值得抄的）:
   - **"多任务可行"采集协议**：不预设任务清单、不 reset，让采集员在场景里做任何可行的事——用数据分布设计（ $H(\text{task} \mid o)$ 高）代替 loss 设计，逼策略真正使用任务条件。这是数据 infra 里最便宜的归纳偏置。
   - **脚本化次优数据当 RL 的覆盖礼物**：9,731 条高随机 pick-and-place 对 BC 是噪声、对 offline RL 是 \$(o, a)\$ 覆盖——采集时就想好"这批数据是给谁吃的"，一种数据、两种算法价值。
   - **等量消融钉死多样性因果**：28k/3-skills vs 27k/13-skills， $N$ 相当时多样性带来 0.30 → 0.65——数据论文的 claim 就该这么做对照（对比 Day15 OXE 没公开混合权重、Day16 DROID 没做等量消融）。
5. **Results**: seen 任务上目标条件方法相当（平均约 0.41–0.49）、RT-1 碾压 LCBC（0.49 vs 0.23）；未见物体/环境非零成功，语言方法在未见物体名上吃亏；跨机构 zero-shot 全部非零，RT-1 退化最小（0.47 → 0.40）；模型容量上升严格单调增益；数据量上升 seen + unseen 都增益；技能多样性 13 vs 3：未见 pick-and-place 0.30 → 0.65。

## 可迁移 / Transfer

- 方法在 held-out 上是否 transfer？是——unseen 物体/环境/机构三档都是 held-out；技能多样性的 0.30 → 0.65 是跨技能泛化的直接证据。模型 vs 框架贡献：**框架（数据）贡献主导**——6 种方法同一套数据，增益与排序主要由数据多样性解释；RT-1 的架构增益是例外（0.49 vs 0.23 说明容量 + 离散化 + 观测历史也是真东西）。
- 对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发：
  1. Post-training 数据配方的"三轴覆盖"思维：embodiment（OXE）× scene（DROID）× skill（Bridge）——做 agent RL 数据时先问"我的三轴分别覆盖到哪"，一次扩张一轴，别三轴一起铺；评测按"与训练支撑的距离"分桶。
  2. "采集协议即归纳偏置"：Bridge 证明不用改 loss，改数据分布（多任务可行、不 reset）就能逼模型用条件——做 SFT/RL 数据时，prompt 与任务的可辨识性设计和 loss 设计同等重要；先检查 $I(\text{task}; o)$ 是不是已经把任务泄露了。
- Infra 视角：\$4,000 公开硬件 + VR 遥操作 + 不 reset 采集 + 事后众包标注 = 低成本真实数据采集的 infra 四件套；TFRecord + TFDS RLDS 双格式 + 5 个方法的预训练 checkpoint 是数据集发布的黄金标准（对比 OXE 没公开混合权重）；评测成本锚点：seen/unseen/跨机构三档 × 10 trials/任务是"数据论文"eval 预算的参考线。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：CRL（对比 RL）的 log-linear value 参数化在 Bridge 数据上和 GCBC 成功率相当（平均 0.42 vs 0.49）——offline RL 的理论优势（stitching、次优数据利用）在这套评测里没兑现，是任务太简单（短时程、BC 可解）还是 CRL 的实现没吃到脚本数据的红利？值得翻附录 B 的实现细节和 GC-IQL 的 checkpoint 对比。
- 如果要复现 / 小规模试，第一个实验做什么？拿 Bridge 开源 TFDS 子集（只取 3 个技能 vs 13 个技能、等量轨迹），在仿真或真机 WidowX 上训 GCBC，复现 0.30 → 0.65 的技能多样性增益曲线——验证"轴扩张 > 点加密"在 skill 轴上成立，顺手验证 $H(\text{task} \mid o)$ 协议是否真让策略用上了语言条件（mask 掉 $\ell$ 看成功率掉多少）。

## 原文金句 (1-2句)
> "BridgeData V2 contains 60,096 trajectories collected across 24 environments on a publicly available low-cost robot."（abstract 原文）
> "We also demonstrate that the performance of these methods improves with more data and higher capacity models, and that training on a greater variety of skills leads to improved generalization."（abstract 原文）
> "This ensures that a policy must pay attention to the task specification, rather than inferring the task from its observations."（§3.2 采集协议原文）
> "This result indicates that data from other skills can improve the robustness of the pick-and-place skill."（§5.3 技能多样性消融原文）

## 今晚产出
- [x] 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- [x] 保留并完善「和之前工作的关系」小节（含 Day09/11/12/13/15/16 对比：RT-1 tokenization 复用、D-GCBC 即 Day12 数学、ACT 与 π₀ 同家族、OXE 上游原料、DROID 正交取舍）
- [x] 数学视角：六种方法统一目标 $\pi(a \mid o, c)$ + 任务注意力协议的互信息形式化 + 覆盖三轴（embodiment × scene × skill）+ 等量消融 0.30 → 0.65 的表示学习解释 + 次优数据"分布匹配 vs 分布覆盖"双面性
- [ ] 深挖 CRL 在 Bridge 上没兑现 offline RL 理论优势的原因（疑问/下一步：翻附录 B 与 GC-IQL checkpoint 对比）

## 连接
- 上一篇: day-16-2024-droid — DROID 固定贵臂扩张场景轴，Bridge 固定便宜臂扩张技能轴；"支撑扩张 > 单点加密"是同一条数学直觉的两条轴
- 下一篇预告: day-18-2024-robocasa — RoboCasa：程序化家庭场景 + 大规模仿真轨迹，看"仿真合成"这条轴怎么补数据三角的最后一条边
