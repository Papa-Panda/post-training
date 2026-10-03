# EP135 — 如何设计一个好的自回归 VLA：从问题构建到工程落地的探索之旅
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP135-autoregressive-vla.html

> 「适合 LM 学习的一门语言，才是一个好的 tokenization。」——讲者论 action tokenizer 的评判标准（字幕口径）

## 元信息

- 期号：青稞Talk EP135（官网期号；本视频经字幕内容互证为自回归 VLA 一讲。台账 B135 行旧标题 DCR 持续指令微调已过时，实际视频为本期）
- 标题：如何设计一个好的自回归 VLA：从问题构建到工程落地的探索之旅
- BV：BV1aTM86bEND
- 时长：01:07:45（讲授约 51 分钟 + Q&A 约 17 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：张士铎（字幕自述「我是世铎」，主持人结尾称「张士铎」；单位字幕未明示，以论文作者页为准）
- 相关论文：讲授涉及讲者团队多篇工作——VLA 能力定义与仿真基准（字幕作「VLA Bench」）、FASTER action tokenizer、Action Codec、coarse-to-control 规划、ETC 训练 recipe（Q&A 提示检索其 Google Scholar，recipe 论文名近「One Pathway, Two Bridges」，字幕口径）；均未在字幕中给出编号，以论文原文为准
- B站链接：https://www.bilibili.com/video/BV1aTM86bEND/
- 官网预告：https://qingkeai.online/blog/Autoregressive-VLA
- 字幕原文存档：本地 `transcripts/EP135.txt`（1754 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP135，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 VLA 在字幕中多作「标 A/VRA」、LIBERO 作「LIBR」、flow matching 作「flow mansion」，均以论文与公开资料为准）。讲者的方法论自称「3S principles」（字幕作 skating/simple/search，按 Scaling、Simple、Search 理解）。

## 一句话总结

讲者用两年工作论证一条路线：VLA 的问题定义应先于动作拟合（先定义要继承 VLM 的语言、常识与推理能力），建模上坚持自回归——前提是把 action tokenizer 做成「适合 LM 学习的语言」（高保真、高压缩、语义一致、可跨本体），再用 ETC 式分阶段 recipe 桥接 VL 数据与具身数据的分布 gap；结论是 AR 在语言跟随与 zero-shot 上天然占优，短板不在范式而在 tokenizer 与训练桥接。

## 核心

### 先定义问题：VLA 不是更大的 imitation learning

讲者开场先立定义：现有 VLA（3D-VLA、OpenVLA、π 一系）从根上强调动作拟合与 skill acquisition，忽略高层智能。他在 2024 年（OpenVLA 之前）做的第一件事是把期望能力写进基准：丰富纹理/OCR 认知、常识指代（「抓索尔的弟弟」）、空间关系推理、语义鲁棒（揣测「味道有点淡」背后的需求）、物理规律与组合推理。配套的仿真平台可无限扩任务、做数据自动化，定位是站住 position 而非刷榜。

他对仿真的态度明确：仿真长期只是原型与辅助——开发成本可能比扩真实数据还高，真机数据来得比预期快得多。仿真数据的用法是 co-train 的中间态；他在 8 张 H100 上一天可生成 200 小时不重样数据（字幕口径）。他同时点名任务定义翻车案例：LIBERO 把 vision 或 language 去掉成功率仍能到 90（字幕口径），说明基准本身没在测语言——「这就是任务定义和 benchmark 设计的问题」。

### 为什么押注自回归：平台实验给的底气

转折来自他自己平台上的对照：在相同数据与初始化下，π-FAST 一类自回归模型在语言鲁棒与 language following 上明显好于 flow matching。他的解释是结构性的：AR 保持 next-token prediction 目标，对 VLM 语言能力的结构保持更好；而智能目前只在 language 上涌现，加了视觉模态都会打折，所以用 VLM 做 backbone 就必须强化其语言理解，这一点被 robotics 背景的研究者长期 overlook。

AR 的附带红利：zero-shot 更好、收敛更快更稳；且与 LM/VLM 全部基建（in-context learning、CoT 等范式）无缝兼容。他的生成范式类比：视频生成是「头轻脚重」、LM 是头尾均衡、VLA 是「头重脚轻」（一个 V 一个 L 进、一小段 action 出），低维输出空间本就适合 AR，flow matching 对它是「大材小用」——团队另一篇 one-step flow matching 工作侧面验证：一次 denoise step 的精度已足够（字幕口径）。

### AR 的两个真问题，都指向 action tokenizer

讲者不回避 AR 的短板：

1. **离散化重建损失**。tokenizer 重建失真等于给 LM 喂错误的学习目标（「学了个词叫 apple，重建后变成 APLE」）；VQ 类方法重建甚至不如分箱。
2. **推理速度**。以 LIBERO 的 10×7 action chunk 为例（字幕口径）：朴素分箱生成 70 个 token、FAST 压到 17 个、π0 只需 10 个 denoise step；同等单步耗时下 AR 解码慢 7~8 倍，不可接受。

他给 tokenizer 定了四条设计准则：重建保真、压缩率高、利用 action chunk 的 2D 结构（时间×维度，像图像一样做 2D tokenization）、跨本体灵活（单臂/双臂、EEF/joint position/joint velocity 通吃）。

### FASTER：把语音 tokenizer 的经验搬过来

FASTER 本质是把基本盘做对：借鉴语音模型的 RVQ（讲者明言「RVQ 本身没有任何创新，工程鲁棒已经足够」），外加两处改动——频域（DCT）重建 loss，让高频细微动作与低频信号在同一尺度受监督；按行优先（沿时间维逐维度）压缩，10×7 的 chunk 用 RVQ 深度 3 压到约 20 个 token（字幕口径）。配套加速：blockwise decoding——同一维度的 3 个 RVQ token 并无强时序依赖，可整块并行生成（深度 3 时效果最好，再往前就是整块并行解码）；action expert 后被证明没用，在后续版本中舍弃。结果：LIBERO 上刷到 87（字幕口径），与 π 系列大体持平或略优；真实机器人表现才是团队的验收标准。

两个 tokenizer 层面的 insight：

- **数据 scaling 性**：tokenizer 是在「拟合一门语言」，见的数据越杂，词表分布越趋于稳定（大数定律式论证）。
- **跨本体与 zero-shot**：单臂 EEF 数据训的 tokenizer 可直接给双臂 joint space 做重建编码。zero-shot 差距可从词表统计解释：只在单一数据集训的 FAST 词表利用率显著更低、某个 token 出现频率高达 10%（字幕口径），这种长尾失衡的词表分布不利于 LM 学习——「从 tokenizer 本身就开始过拟合了」。

### Action Codec：从重建正确到语义正确，再到原生规划

下一步是语义：相同 token 在不同本体/任务上应表示相似动作（如「下坠」）。Action Codec 把语义部分定义为动作相似度：训练时用 VL feature 对齐，让相近行为落到相近 token；配套发现——词表不是越大越好（高频噪声被学进去）、相邻 chunk 的 token overlap 越大越像一门「流畅的语言」、模型约 5K step 即收敛（字幕口径）。工程细节：不同本体用 soft prompt 区分、Perceiver cross-attention、两阶段（先按语义分箱、再用 RVQ 残差层补重建）。

再进一步是把规划变成动作 tokenization 的原生能力：把 16 秒轨迹联合编码得到的粗粒度 token 就是 planning token；coarse-to-control 工作从生成侧做同样的粗到细——先生成 planning token 再生成 execution token，用规划约束探索空间。讲者强调对照实验的严谨：所有 backbone 对齐、用 π 初始化、复现基线而非引原文数字；planning token 比文本 CoT（256 token、真机太慢）生成的 token 更少、效果更好（字幕口径）。

### ETC recipe：VLA 训练的两个 mismatch 与三阶段桥接

最后一部分是训练 recipe。讲者把 VLM→VLA 的 gap 拆成两个 mismatch：输入侧（普通视觉编码器特征不满足 manipulation 需要，冻结 vision encoder 训练效果大掉）和输出侧（next-token 交叉熵与 flow matching 的 L2 目标不同；即便同为 AR，text token 的理解与 action token 的生成在语义上也错位）。数据分布上，VL 数据与具身数据交集很小，只用 action 数据硬训等于把模型分布强行拽到具身分布、与 VL 分布几乎无重叠，OOD 即失效；普通 co-train 只是把桥搭在原有小交集上。

ETC（Embodied Trajectory Couple，字幕口径）的做法是构造与 mid-training 匹配的具身多模态数据，把桥加宽。三阶段：

1. **预训练**：不用任何本体/action 数据，用具身 VL 数据再增强一次 VLM——讲者称只要起点是增强过的 VLM，后续任何形式效果都更好，在其基准上不用 π/FAST 式预训练也能到约 70 成功率（字幕口径）。
2. **Mid-training**：才引入 action 生成目标，此时视觉编码器与 backbone 已在图文上对齐，只剩生成侧 mismatch 要补；消融显示 2D trajectory planning 数据的桥接作用最好（文本理解任务生成 2.5D 动作），其余 VQA、grounding、caption 数据加上都有益、叠加更好。
3. **Post-training（retention）**：in-distribution 上 co-train 收益不大；关键在 OOD 组合泛化——物体从 ABC 换成 DEF 时成功率直接砍半（如 90 到 40、50 到 25，字幕口径），补救办法是给新物体造只有 VL 理解标签、无 action 标签的数据，靠已搭好的桥把已知 action 映射到新物体的视觉分布上。而且每个环节都要加 co-train，中途停掉效果大掉。

### Q&A 中的明确表态

- 精度质疑：AR 做亚毫米级精细操作没问题（插 5mm 宽的环、错 1mm 就掉一类任务，字幕口径），与 π0.5 的差距在 60 与 70 分量级（字幕口径）。
- On-policy RL：直言「真机不要指望 on-policy」，只能做短程任务；长程问题大概率靠 offline RL；常识类能力用 online RL 几乎不可能。
- 落地阻碍排序：技术路线未收敛 > 场景与需求定义不清 > 算法侧瓶颈（最小）；硬件稳定性被低估。
- 潜力估计：当前 VLA 对 VLM 潜力的挖掘只有 5% 到 10%（字幕口径）——VLM context 是 8K/32K/100K 量级，VLA 还不到 1~2K。

## 关键数字

| 指标 | 数值 | 来源 |
|---|---|---|
| LIBERO 去 vision/language 的成功率 | 仍达 90 | 字幕 |
| 仿真数据生成 | 8 张 H100 一天 200 小时不重样数据 | 字幕 |
| 朴素分箱 / FAST / π0 的解码量（10×7 chunk） | 70 token / 17 token / 10 步 denoise | 字幕 |
| AR 相对解码速度差 | 慢约 7~8 倍（同等单步耗时假设） | 字幕 |
| FASTER 压缩结果 | 10×7 chunk 约 20 token（RVQ 深度 3） | 字幕 |
| FASTER 在 LIBERO | 87（成功率口径） | 字幕 |
| 单数据集 FAST 的最高频 token 占比 | 10% | 字幕 |
| Action Codec 收敛 | 约 5K step | 字幕 |
| ETC 预训练起点效果 | 基准约 70 成功率（无 π/FAST 式预训练） | 字幕 |
| OOD 物体替换的成功率跌落 | 90 到 40、50 到 25 | 字幕 |
| VLA 对 VLM 潜力的挖掘估计 | 5% 到 10% | 字幕 |

## 可迁移

- 「先定义能力、再选范式」对 post-training 同样适用：先把要继承的基座能力（语言、常识、推理）写成可测的基准，再谈 SFT/RL recipe；LIBERO 去语言仍 90 的例子说明，基准测不出能力，训练就不会保留能力。
- Tokenizer 评判标准可直接借用：不只看重建误差，还要看词表分布是否均衡、是否利于 LM 学习——评估任何离散化（动作、工具调用、结构化输出）时都该加上「LM 可学习性」这一维。
- ETC 的桥接思路是通用的 continual-learning 叙事：两个分布交集太小时，与其在小交集上 co-train，不如构造与下游对齐的中间数据把桥加宽；且抗遗忘要全程在场，不能只在最后一个阶段补。

## 疑问 / 下一步

- 讲授涉及的 VLA Bench、FASTER、Action Codec、ETC 四篇工作的正式名称与出处字幕均未给出（Q&A 只提示查 Google Scholar），引用具体数字前需按讲者姓名检索论文核对。
- FASTER 的约 20 token 与 blockwise decoding 的实际闭环频率（20Hz 落地）只在 Q&A 被追问时一句带过，工程细节需查论文。

## 原文金句

> 「你给他一个错误的学习目标，那他基本上就是学了一个错误的知识。」——论离散化重建失真（字幕口径）

> 「我真的认为现在的 VLA 对 VLM 的潜力挖掘只开发了 5% 到 10%。」——Q&A 论建模空间（字幕口径）
