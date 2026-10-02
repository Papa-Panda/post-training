# EP119 — FIPO & Future-KL：突破大语言模型在复杂推理中的性能瓶颈

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP119-fipo-future-kl.html

> 「DAPO 会卡在 4000 左右，我们会涨到 1 万 2。」——讲者对长 CoT 是否被真正激发的判据（字幕口径）

## 元信息

- 期号：119（B站编号 B119，bilibili-only；台账 B119 行登记标题为「Farm：LLM 训练与推理的确定性系统，Unsloth 社区新项目」，与本视频实际内容不符，以实际内容为准；官网无对应期——官网 119 无预告文）
- 标题：FIPO & Future-KL：突破大语言模型在复杂推理中的性能瓶颈
- BV：BV1cXDyBWEwr
- 时长：00:42:45（讲授约 29 分钟 + Q&A 约 13 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：马诗雨（达特茅斯学院博士三年级；本工作为其在千问团队的实习项目，字幕作「千万 PO 组」，机构口径以论文作者页为准）
- 相关论文：FIPO（Future-KL Influenced Policy Optimization）及前序分析工作 Sparse...（字幕作「sparser critical」，ICLR 在投/录用口径待查）、Direction of ... Updates（字幕作「direction of 821 updates」，ACL 口径待查）、Dark Secrets 博客（wasted moments 主题）；具体标题与编号以论文原文为准，本纪要不臆造
- 相关代码：32B 模型已开源（代码在 GitHub、模型在 HuggingFace，讲者口径；repo 地址待从论文页补录）
- B站链接：https://www.bilibili.com/video/BV1cXDyBWEwr/
- 官网预告：无（官网无对应期）
- 字幕原文存档：本地 `transcripts/EP119.txt`（1084 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；别名极多，如 GRPO/DAPO 在字幕中作「G2PO/DAO/DEO」、CoT 作「coo t」、wasted moment 作「乌斯 moment / wood moment」、Qwen2.5-32B base 作「2.53102B 贝斯」、AIME 作「A米」，均以论文与公开资料为准）。台账注记：`episode-index.csv` 中 BV1cXDyBWEwr 原登记为 bilibili-only 的 Farm 一场，实际视频内容为 FIPO 本场，官网亦无对应期；编号沿用 B站台账位置 B119（与 EP106 Intern-S1、EP124 人机信任的编号先例一致），讲授中主持人未报期号，此编号为台账位置而非官方期号。凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

复现 DAPO 时性能能涨但 CoT 长度卡在约 4000 token，讲者追问「这算不算激发了长 CoT」并给出否定答案；FIPO 的解法是不用 critic、不用逐 token 人工信用分配，而是用当前 token 之后整段轨迹的 log-prob 累积（Future-KL）判断这个 token 到底引出了好行为还是坏行为，再据此对 DAPO 的均一 advantage 做逐 token 的重加权，从而在 Qwen2.5-32B base 上把思维链稳定推到 1.1 万–1.2 万 token 并超过基线性能。

## 核心

### 引入：性能涨了，但长 CoT 没被激发

动机来自一个复现落差：按公开 recipe 跑 DAPO，分数能到 47–50（超过 DeepSeek R1-Zero 当时汇报的 47，字幕口径），但平均 CoT 长度始终停在约 4000、偶尔 5000 token。讲者的质疑是：长度不涨的「长 CoT recipe」可能并没有真正激发长推理，只是把短推理做得更对。这与 PPO 路线的工作形成对照——有工作回到纯 PPO 后长度能涨到约 1 万、分数却和 DAPO 差不多（47–50，字幕口径），说明 base 模型的长 CoT 潜力在、只是 GRPO 系的信用分配方式没把它引出来。

### 前置分析一：RL 只改写了极少数关键 token

FIPO 的设计建立在讲者团队三步前序研究上。第一步用 JS 散度逐 token 比较 base 与 RL 后模型的输出分布，发现约 83% 的 token 分布没有区别（一个 simple RL 变体里甚至 98% 没区别，字幕口径）：RL 没有改写整体 policy，只动了一小部分。交叉改写实验（cross sampling）把这件事量化：用 RL 模型替换 base 的仅 1%–4% 的 token 就能恢复 RL 的性能，反过来用 base 替换约 5%–8% 就能把 RL 模型打回 base（均为字幕口径）。这些被改写的 token 稀疏地集中在生成过程的头部与关键节点上——RL 像是在少数分叉点改写、其余位置仍由 base 的轨迹接管。

第二步是找指标：entropy、KL 这类无方向的度量分不出哪些 token 是关键节点，带方向的 log-prob 比值（字幕作 log p）可以——按 log-prob 大小挑 token 改写同样能恢复 RL 性能，说明它能定位关键 token。

第三步是 wasted moment 现象：理想中的 aha moment（停顿、反思、把错的改对）其实是少数；更常见的是模型中途已经得到正确答案、又强行 double-check 把自己改错（讲者举的例子中途已算出答案、最终却错）。在 outcome-only 的均一 advantage 下，这条轨迹前面所有正确 token 会被同等惩罚——这是 FIPO 要修的核心伤害。

### 方法：Future-KL 把「后续轨迹」变成逐 token 信号

基线选 DAPO 而非 GRPO（GRPO 的 sequence-level 均一 advantage、以及对短的正确/长的错误回答的隐含偏置是讲者点名的问题；所有改进都在 DAPO 之上做）。FIPO 的核心想法：一个 token 的好坏，由它引出的后续轨迹判断。具体做法（按字幕口径转述）：

- 把当前 token 之后各 token 的 log-prob（importance ratio 的对数形式）累积成 Future-KL 的和；和大于零说明该 token 引出了训练中更被偏好的后续行为，应予强化，小于零则应压低。
- 稳定化三件套：其一，importance ratio 超过极值的样本直接 mask 掉、不计入 Future-KL（讲者复现时发现个别样本的 ratio 过大会把整个 advantage 信号拉爆、把训练推回 DAPO 水平约 51–52 分，字幕口径）；其二，加 GAE 式的衰减窗（GAE window），让近处 token 权重高、远处递减，既符合因果、又防序列变长后累积和爆炸，再做 clip；其三，先把 Future-KL 和变成指数形式使其在 1 处对称，再配合指数变换把重加权做成「大于 1 放大该 token 的 advantage、小于 1 压缩」的乘性缩放。

讲者给的理论直觉（他自陈先有直觉、后有一位热心读者补了证明）：在「最大化 reward 且限制 policy 变化」的目标下，最优 policy 近似正比于先验 policy 乘上 advantage 的指数项；FIPO 的 Future-KL 恰与 PPO 里由 critic 与 reward model 估出的 advantage 项等价，GAE window 对应 GAE 的形式。区别只在误差来源：PPO 需要一个足够准、且适应长 CoT 的 critic（讲者引 VAPO 的冷启动 critic 与回到纯 PPO 的工作为证），FIPO 则用更新后 policy 本身做隐式 critic 近似，省掉显式 critic 模型。

### 实验：32B base 上的保守数字

设定刻意保持与基线可比：Qwen2.5-32B base（要干净的 base 才能回答「如何从 base 引出长 CoT」）、训练数据与 DAPO 相同的 7K（字幕口径）、128 卡训 32B（字幕后段又作 7B 用 32 卡，卡数口径前后不一，以论文为准）。结果（讲者明说报得保守）：

| 观察项 | 基线/口径 | FIPO（保守汇报 / 讲者称最高） | 来源 |
|---|---|---|---|
| AIME 2024 | DAPO 复现 47–50 | 56 / 最高 58，裸版曾跑到 60（与 VAPO 相当） | 字幕 |
| AIME 2025 | — | 43 / 最高约 45–47 | 字幕 |
| 平均 CoT 长度 | DAPO 卡在约 4000 | 稳定升至约 11000–12000，且单调上升不回摆 | 字幕 |
| 长度与性能关系 | DAPO 后期被长的负样本主导 | 正样本逐渐变长并压过负样本长度，长度–性能正相关 | 字幕（其长度加权 advantage 度量） |
| 训练稳定性 | DAPO 后期 entropy、grad norm、policy KL 波动/爆炸 | 三者均更平滑（同等 mini-batch 下亦然） | 字幕 |

讲者同时给出小模型边界：在 7B 上先升后降的 entropy 轨迹更有效，硬套 32B 的探索型参数 7B 反而涨不上去——小模型可能更适合强蒸馏而非这类激发式 RL，他明确说训小模型与大模型不能用同一套参数。pass@k（如 pass@32）只涨一两个点也被坦白：没有新知识注入时这类算法主要教模型「把题做对」，大幅 pass@k 提升更依赖教师蒸馏。

### Q&A 要点（含讲者最坦白的边界）

- **纯 on-policy 还成立吗**：成立性不依赖 on/off-policy 标签，本质是更新前后 policy 的对比；纯 on-policy 每次 rollout 成本高（他们测过约 90% 时间在 rollout），但把梯度更新前后的对比做成 Future-KL 信号是他们后续在做的版本，预期噪声更低。
- **mini-batch 是关键稳定性旋钮**：一次 rollout 多次更新的意义是效率与总 rollout 次数，但更新多噪声大；FIPO 里把 DAPO 的 mini-batch 从 32 个 prompt 翻倍（字幕口径）后 importance sampling 方差明显下降、稳定性提升。mini-batch 越大 log-prob 精度越高、信号越准，代价是更新次数变少，要权衡。
- **与调高 clip 的区别**：把 clip 上界从 1.28（字幕作 1.8 应为 clip 口径，待查）调到更高（如 1.4）确实涨长度更快（约 100 步到 1.2 万），但涨出来很多是 LaTeX 格式冗余而非有效推理，性能不跟；FIPO 的变长速度与「步长、思维链完整度」相关，不是单纯放开探索。Future-KL 本身也可被强制压小当作稳定性正则，取决于怎么用。
- **为什么不在 Qwen3 等已蒸馏模型上训**：已预置长 CoT 的蒸馏模型与 base 的训练机制不同、RL 很脆容易崩，他们要研究的是从 base 引出而非在蒸馏模型上巩固；稳定性类算法（如 GSPO）更适合那类模型。FIPO 与 GSPO 不冲突，正在做把 Future-KL 嫁接到 GSPO、以及多智能体与数学以外任务（字幕作 ME 方向，待查）的 V2。
- **数据与泛化边界**：只在 DAPO 的 7K 数学数据上验证过（32B 训练两三周一轮、换数据集调参成本太高，字幕口径）；pass@k 提升有限、需要外部 process reward 或蒸馏补知识，是他明确认领的下一步而非已解决项。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| RL 后分布未变的 token 占比 | base vs RL 逐 token JS 散度 | 约 83%（simple RL 变体约 98%） | 字幕（分析部分） |
| 交叉改写恢复 RL 性能所需替换 | 用 RL token 改写 base | 仅 1%–4% | 字幕（分析部分） |
| 交叉改写打回 base 所需替换 | 用 base token 改写 RL | 约 5%–8% | 字幕（分析部分） |
| DAPO 复现性能 | AIME 口径 | 47–50（CoT 卡在约 4000 token） | 字幕（引入部分） |
| FIPO AIME 2024 | 保守汇报 | 56（最高 58，裸版曾到 60） | 字幕（实验部分） |
| FIPO AIME 2025 | 保守汇报 | 43（最高约 45–47） | 字幕（实验部分） |
| FIPO 平均 CoT 长度 | 训练全程 | 约 11000–12000，稳定上升 | 字幕（实验部分） |
| 训练配置 | DAPO 同款 | 7K 数据、512 prompt × 16 重复、32B 用 128 卡 | 字幕（Q&A，卡数口径待查） |
| rollout 时间占比 | 训练总时长 | 约 90% | 字幕（Q&A） |

## 可迁移

- 对 RL post-training 的直接清单：当 outcome-only 均一 advantage 出现「中途做对又改错却被全程惩罚」的 wasted moment 时，可用后续轨迹的 log-prob 累积做逐 token 信用重加权，不引入 critic 模型；但先查 mini-batch 大小——信号噪声对它极其敏感，翻倍 mini-batch 是讲者验证过的最便宜的降噪手段。
- 评判一个 RL recipe 是否「激发长 CoT」，不要只看分数：同时看平均长度是否持续上升、长度与性能是否正相关、正样本是否在变长。只涨分不涨长度的 recipe 可能只是在把短推理做得更对。
- base 与蒸馏模型的 RL 参数不能共用：小模型/蒸馏模型的稳定性要求更高，先分开调参再谈算法优劣，避免把参数 regime 的差异误记为算法差异。

## 疑问 / 下一步

- Future-KL 的确切公式（累积范围、指数变换与 clip 的具体形式、GAE window 的衰减系数）字幕只给了口径，需查 FIPO 论文原文核对后再引用；前序两篇分析工作的准确标题与发表状态同样待查。
- AIME 数字讲者明说报保守值且裸版最高到 60，论文表格口径与讲授口径的对应关系需逐项对齐；32B/7B 的卡数在字幕前后段不一致（128 卡 vs 32 卡），以论文实验设置页为准。
- wasted moment 与 aha moment 的比例是用 R1 系模型做验证得到的定性结论，具体统计口径（哪个 judge、多少样本）字幕未给，需查 Dark Secrets 博客原文。

## 原文金句（1-2句）

> 「DAPO 会卡在 4000 左右，我们会涨到 1 万 2。」——讲者对比基线与 FIPO 的 CoT 长度（字幕口径）

> 「训小模型和训大模型，我们不能用同样的参数。」——讲者关于模型规模与 RL 参数 regime 的判词（字幕口径）

> 「很多 RL 算法都不太能提高 pass@k，没有介入额外的新知识，只是教模型怎么更好地把问题做对。」——Q&A 中对这类方法能力边界的坦白（字幕口径）
