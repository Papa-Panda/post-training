# EP5 — 3D-VLA：构建生成式三维具身世界模型

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP05-3d-vla.html

> 「当这个模型有了想象最终状态这样的能力的时候，它其实就是一个世界模型。」——讲者对 3D-VLA 的定性判词（字幕 10:12 附近）

## 元信息

- 期号：5
- 标题：3D-VLA：构建生成式三维具身世界模型
- BV：BV1Loe2z8EFW
- 时长：00:59:44（讲授约 38 分钟 + Q&A 约 21 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：甄浩宇（上海交通大学人工智能专业大四；在 MIT-IBM Watson AI Lab 访问半年；字幕自述姓名与经历，正式头衔以论文作者页为准）
- 相关论文：Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, Chuang Gan, *3D-VLA: A 3D Vision-Language-Action Generative World Model*（ICML 2024），https://arxiv.org/abs/2403.09631
- 相关代码：https://github.com/UMass-Embodied-AGI/3D-VLA （讲者 Q&A 称代码将随更大规模重训练分阶段放出：先 embodied diffusion model，再整个模型）
- B站链接：https://www.bilibili.com/video/BV1Loe2z8EFW/
- 官网预告：青稞Talk 官网第 5 期（链接略）
- 字幕原文存档：本地 `transcripts/EP05.txt`（918 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP5，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名有识别误差，如 3D-LLM 在字幕中作「CDLM / three d l m」、3D-VLA 作「CDVA / CDBLA / 顺DBLA」、ZoeDepth 作「ZEDEPTH」、RAFT 作「rap」、InstructPix2Pix 作「inside a pks to pick」、ScanNet 作「SKYNET」、Objaverse 作「object verse」、FLAN-T5 作「布雷弗兰T5」、drawer 作「java / 招儿」、MIT-IBM Watson AI Lab 作「vs AI live」，均以论文与公开资料为准）。论文（arXiv:2403.09631）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

这期把 3D-LLM 从「能看懂三维场景」推进到「能想象动作执行后的三维世界、再据此行动」：讲者用现成模型链（深度估计 + 光流 + 分割 + GPT-4V）把无 3D 信息的 Open X-Embodiment 机器人视频提升为带点云与指令标注的具身指令微调数据，再通过一组 interactive tokens 把 3D 语言模型与 embodied diffusion 模型（图像 / 点云两路）拼成一个生成式世界模型，并用消融证明「先想象最终状态」确实能反哺语言理解与机械臂抓取——同时坦白其控制精度、物理理解与长时记忆都还远未达标。

## 核心

### 引入：为什么需要一个「能想象」的三维基础模型

讲者先给人类与物理世界交互的分解：探索环境 → 构建 3D 表征 → 基于表征推理 → 规划步骤 → 想象执行后的目标状态 → 真正执行 → 执行改变场景、回到重新探索。这条循环里，前作 3D-LLM 只覆盖了「探索、表征、推理、规划」，缺的正是**想象（goal imagination）与动作（action）**两环；补上想象，模型才配称世界模型。讲者给世界模型下了一个三要素定义：对三维世界构建表征的能力、预测未来事件的模拟能力、推理与规划能力——3D-VLA 的全部功夫都压在中间这一项上。

### 前作 3D-LLM 的管线与四条局限

3D-LLM 的做法是：对三维场景渲染多视角图像，用 CLIP 提取特征，再把 CLIP 特征投影回 3D 空间得到 3D feature，与问题一同喂给 LLM，训练其中的 perceiver / conformer 类模块来回答「椅子在哪里」这类问题。讲者自陈其四条局限（字幕口径）：

1. **数据集过拟合与幻觉**：主要在 ScanNet 与 Objaverse 等场景 / 物体数据集上训练，对新场景会产生「桌子是棕色的」式幻觉；
2. **低层任务打不过专用方法**：定位、导航这类任务上传统专用方法效果更好；
3. **黑盒**：给 3D feature 与问题直接出答案，无法解释为什么；
4. **天然缺机器人能力**：训练数据里没有抓取这类任务，无法直接充当机器人的 foundation model。

### 数据：把 2D 机器人视频「提升」成 3D 指令数据

训练世界模型需要带 3D 信息的大数据集，但 2023 年 9 月发布的 Open X-Embodiment（34 个机构共建）只有视频与指令标注、没有 3D。讲者的解法是整条 off-the-shelf 模型链（字幕口径）：

- **ZoeDepth** 逐帧估深度，再投影成点云，得到每帧的 3D 信息；
- **RAFT 光流**区分静止背景与运动部分——动的部分必然是机械臂或被抓取物体，用来 refine 数据；
- **Grounded SAM** 拿目标物体的 mask；
- **GPT-4V** 生成多样化语言数据；再为每个任务定义模板，模板数据构成 Embodied Instruction Tuning Dataset 的主体。

讲者强调两个数据事实：其一，单视角点云的质量已经足够看出场景深度变化与机械臂、物体的相对位置；其二，机器人数据是**长尾分布**——RT-1（Fractal）约 70K 条轨迹量级最大（字幕口径），不少数据集不到 1K 条，这直接构成后文的数据上限问题。

### 架构：interactive tokens 拼起语言模型与扩散模型

3D-VLA 的交互协议由一组新增 token 定义（字幕口径）：

- **scene token**：把 3D feature 夹在两个 scene token 之间注入，处理 interleaved 的 3D-语言数据；
- **object token / location token**：表示物体及其位置（定位靠输出 3 个 location token，即离散化坐标点，映射回真实 3D 空间）；
- **多模态（modality）token**：告诉解码端当前要生成哪个模态的 goal state——图像走 InstructPix2Pix 式 image diffusion，点云走魔改的 Point-E；
- **action token**：沿用 RT-1 / RT-2 的做法，把机械臂 7 自由度动作空间离散化。

推理时的闭环是：用户输入 3D feature → 模型自行决定输出 / 想象最终状态 → 调 diffusion 模型生成 goal state → goal state 再喂回输入 → 模型据此输出 action 执行抓取。

训练分三阶段（字幕口径）：**阶段一**单独训练 embodied diffusion 模型（以初始状态 + 指令为条件生成最终状态，图像与点云各一个）；**阶段二**沿用 3D-LLM 的训练法，但把场景 / 物体数据换成具身指令微调数据，训练 interactive tokens 并学会 robot control 与 manipulation 任务；**阶段三**用一个 projector 把 LLM 输出映射为 diffusion 模型的 condition，并用 LoRA 微调 diffusion 模型完成拼接。

### 实验：想象能力真的有用吗

讲授给出的定量口播不多，结论以定性对比为主（字幕口径）：

- **语言与定位任务**：在自建数据集上与微调的 FLAN-T5 及若干 zero-shot 模型对比，3D-VLA 在 action 序列理解、任务分解（把目标拆成 4 步）、首尾帧问答、bounding box 检测与描述等任务上显著占优；
- **先预测 bbox 再生成有增益**：让模型先预测场景物体的 bounding box、再输出最终状态，goal generation 质量更好；讲者同时提醒 CLIP 分数在 robotics 领域并不实用，如何设计更好的 goal generation 指标是 future work；
- **消融（核心证据）**：在 what-if query 式语言任务与 RLBench manipulation 上，「先用 goal generation 想象最终状态再回答 / 行动」的结果优于不用想象；用 Open X 上预训练的 checkpoint 也优于 from scratch。失败案例也被如实归因：put knife 提升不大是因为会与场景碰撞，属于仿真环境问题而非模型问题；take umbrella 与 pick up cup 则明显受益于「想象后知道目标物在哪、是哪个」。

讲者展示的涌现行为值得一提：对训练中没见过的 drawer，模型能在纹理与物理有扭曲的情况下定位 plate、想象出合理的抽屉结构——讲者的判断是「尽管 texture 或者物理上是错的，这样的图依然可以指导模型去预测 action」。但长时任务暴露记忆短板：把茶包放进抽屉再关上、打开后茶包消失——模型记不住被遮挡物体的存在。

### 讲者自陈的三条局限

1. **控制不精准**：LLM 输出 action token 的形式把空间切成 ， $256^3$ ，个离散点，抓不了很小的木块，也做不了「用杆推动物块到目标位置」的细节任务；
2. **diffusion 模型不懂物理**：会把勺子「移除」而非放到毛巾上，不知铁锅是硬的、毛巾是软的，texture 也无法保持一致——讲者明言连 Sora 也有类似问题；
3. **真实数据太脏**：深度相机点云噪声大，真机部署的功夫不在跑通模型，而在处理 3D 数据（他们最终剔除桌面与背景点云才让模型收敛）；加上机器人数据集长尾、低清（如 BC-Z 人眼都难辨认物体），数据质量是整个社区的长路。

后续计划（字幕口径）：在 DROID 等带多视角深度的数据上重训练 3D-VLA；跟进 Figure 01 式人形 / 移动机器人应用；给模型加 video diffusion（1.5 版）；探索多 agent 交互。

### Q&A 要点

- **刚体 / 柔体的影响**：RLBench 多为刚体任务，软体效果未知；diffusion 模型本身不理解材质（会把锅当软体、毛巾当刚体），是世界模型要补的课。同组 Robot Dreamer（字幕称 ICML 2024）在做视频扩散处理软体 / 平面物体。
- **仿真到真机的鸿沟**：仿真数据干净、容易刷高分；真机难在预测 action 与真实三维空间的对齐、noisy depth 处理与专家数据的手工采集——「在 simulator 里给一个任务瞬间收集 1 万个 sample，真机只能手动收集或写 hard code」（字幕口径）。
- **goal imagination 与思维链的关系**：讲者认同它就是多模态版 chain-of-thought——先输出最终状态的 goal，再基于 goal 执行；同期 Minecraft 方向的 MineDreamer 也是同一思想。
- **打不过专用小模型怎么办**：讲者坦承 grounding / navigation 等低层能力短期内超不过专精模型（专用 transformer 多视角方法简单任务可刷到 90 以上，语言模型大概只有 70–80 的级别，字幕口径）；工作的意义是把这些能力装进一个 general purpose model，而不是单项夺冠。
- **为什么不用 Gaussian Splatting 提特征**：速度账算不过来——单场景特征提取要 20–30 分钟量级，foundation model 需要百万级数据支撑，乘起来不可行；单场景理解 / 抓取场景下它效果很好、值得尝试。
- **2D 到 3D 能否与空间理解解耦**：完全可以，PointLLM 式从点云出发训练点云语言模型就是另一条路；3D-LLM 借 2D 纯粹是因为 VLM 先验强、能显著提分。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| Open X-Embodiment 最大单一数据集（RT-1 / Fractal） | 其他数据集多不足 1K 条 | 约 70K 条轨迹量级 | 字幕（数据部分） |
| 动作空间离散化粒度 | 连续 7 自由度动作 | ， $256^3$ ，个离散点 | 字幕（局限部分） |
| 任务分解示例步数 | 单步目标描述 | 拆成 4 个步骤 | 字幕（实验部分） |
| 低层 grounding / navigation 能力 | 专用小模型简单任务 90 分以上 | 语言模型约 70–80 分量级 | 字幕（Q&A） |
| 仿真单任务专家数据采集 | 真机需手动 / hard code 采集 | 仿真可瞬间收集约 1 万个 sample | 字幕（Q&A） |
| Gaussian Splatting 单场景特征提取耗时 | 3D-VLA 特征管线约 5 秒量级 | 约 20–30 分钟量级 | 字幕（Q&A） |
| 3D-VLA 定量实验表 | 讲授未逐项口播完整数值表 | 仅定性结论：优于微调 FLAN-T5 与 zero-shot 模型 | 字幕（实验部分） |

注：3D-VLA 论文（arXiv:2403.09631）有完整的 RLBench 等定量表格，讲授现场未逐项口播，本表只收字幕明确给出的数字。

## 可迁移

- **「先想象终局再行动」是一条可迁移的推理时脚手架**：在 agent 任务里先让模型显式生成目标状态（goal state / 终局描述）再规划动作，等价于多模态 chain-of-thought；对纯文本 agent 同理——先写「完成态应该长什么样」再执行，能给后续验证留一个可比对的锚点。讲者的消融（想象反哺语言理解与抓取）是这条经验的直接证据。
- **数据管线的组织方式**：当目标模态数据不存在时，用一串 off-the-shelf 模型（深度、光流、分割、VLM 描述）把已有数据「提升」到目标模态，再用任务模板兜住主体分布——这与用强模型给弱模态数据打标、合成训练集的工程套路同构，关键在每一步都留可视化校验（讲者反复展示点云可视化来证明 lift 质量）。
- **评测指标要怀疑现成货**：讲者明言 CLIP 分数在 robotics 里不实用；迁移到任何新领域，先验证指标与真实目标的相关性，再拿它做优化信号，否则会优化出指标好看、任务没做成的模型。

## 疑问 / 下一步

- 讲授未口播 RLBench 等 benchmark 的具体成功率数字，3D-VLA 相对专用 VLA 方法（如 RT-2、OpenVLA 一代）的定量差距需回论文表格核对；讲者在 Q&A 已承认低层能力打不过专精模型，这条边界在新一代 VLA 上是否还成立值得追踪。
- 「被遮挡物体消失」的长时记忆失败，讲者归为 future work；后续的视频扩散（1.5 版）与多 agent 交互能否补上物体恒存性，没有给出机制层面的方案。
- 真机部署靠「剔除桌面与背景点云」才收敛，这说明 3D feature 对噪声的鲁棒性是落地瓶颈；DROID 重训练后这条是否改善，值得等其开源新版验证。

## 原文金句（1-2句）

> 「当这个模型有了想象最终状态这样的能力的时候，它其实就是一个世界模型。」——讲者界定 3D-VLA 与 3D-LLM 分界的判词（字幕 10:12 附近）

> 「其实在 real world 中，功夫并不是把（模型）下载了部署、或者把这个跑通，而是怎么去处理我们的 3D data。」——讲者谈真机落地的真实成本（字幕 34:55 附近，按干净口径转写）
