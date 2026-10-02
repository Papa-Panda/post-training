# EP25 — LLMC：大语言模型的量化基准

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP25-llmc-benchmark.html

> 「实际上不能完全随机的生成这个校准数据，还需要满足这个句子间的一个逻辑。」——讲者谈校准数据构造时的一条经验判词

## 元信息

- 期号：25
- 标题：LLMC：大语言模型的量化基准
- BV：BV1SjNzeAEXe
- 时长：01:10:39（讲授约 55 分钟 + Q&A 约 15 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：讲者未在字幕中自述姓名（讲授口径以「我们论文」方式介绍 LLMC 团队工作；嘉宾姓名以论文作者页为准）
- 相关论文：*LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit*（配套工具 LLMC；论文与代码地址以论文页为准）
- 相关代码：讲授开场给出扫码入口；LLMC 工具另有专场介绍（讲者明确说「这个工具的一些具体的实现……由之后我们的同事会来讲」）
- B站链接：https://www.bilibili.com/video/BV1SjNzeAEXe/
- 官网期号：EP25（官网预告链接本文省略）
- 字幕原文存档：本地 `transcripts/EP25.txt`（1447 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP25，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 GPTQ 在字幕中作「GPDQ/GBTQ」、QuaRot 作「QRT/q out」、OmniQuant 作「OMINIQUANT」、Os/+ 作「OS加」，均以论文与公开资料为准）。本期为讲者介绍论文配套工具 LLMC 的学术报告：讲者把已有经典量化算法归结为 transformation / clip / reconstruction 三类技术并逐项 benchmark，本文只写讲授实际讲到的内容（约 44% 处 slide 引用到论文 Table 3，未在字幕中复述的具体数字以论文为准）。

## 一句话总结

LLMC 这篇论文（及配套同名压缩工具）不提新算法，而是把 SmoothQuant、AWQ、OS+、OmniQuant、QuaRot、GPTQ 等经典 PTQ 方法归结为 transformation（等价变换）、clip（权重裁剪）、reconstruction（误差补偿重建）三类可组合技术，逐项拆解并在统一工具内公平地 benchmark，其核心结论可压缩为四条可迁移的工程判据：①校准数据的 token 分布必须与测试分布一致、且句子间逻辑不能打乱；②等价变换的收益以牺牲权重 outlier 为代价，weight 低比特时未必为正；③旋转法压低 outlier 不等于压低输出误差，两者不正相关；④clip 与对称性必须匹配（对称量化配对称 clip），且 clip 在低比特时收益巨大、在高比特时反而可能成为主要误差源。讲者把论文定位为「指导书」——给出层级误差的诊断与启发，而非一键最优配置。

## 核心

### 量化背景：为什么这期只谈均匀量化与 PTQ

讲者定的范围很明确：聚焦**均匀量化**（把浮点数均匀映射到整数格点；省显存、整数乘法加速），聚焦**训练后量化 PTQ**（先收校准数据、求量化参数、得量化模型），因为 QAT 在大模型上开销过大「在大模型上的开销实在太大了」。PTQ 里所有决策（变换参数、量化参数）都由校准数据决定，所以校准数据的分布是论文第一个、最「接地气」的变量。

讲者随后按三类技术铺经典算法的地图（这是整期的骨架）：

- **Transformation（等价变换）**：rule-based 手工定 scale（SmoothQuant）；search-based 搜 scale（AWQ、OS+，其中 AWQ 同时搜 clip 的 ratio；OS+ 是对称量化 + channel-wise shift + scale）；learning-based 学 scale 与 shift（OmniQuant，逐 block 反向传播）；旋转矩阵法（QuaRot：利用 RMSNorm 的等式性质把正交矩阵 $Q$ 折进权重、降低激活 outlier，须在线做哈达玛变换处只有注意力与 MLP 中间激活，down 层处同样可在线；讲者评其「空间会更大」，实测效果优于一维 scale 向量法）。
- **Clip**：把 weight 的最大/最小值按比例截断，间接优化量化 scale（AWQ 用 search 学、OmniQuant 用训练学 $\gamma$ / $\beta$ ）。
- **Reconstruction**：GPTQ 继承 OBS/OBC 逐列量化、用 Hessian 估计误差并回补到未量化列（lazy block 更新 + 矩阵分解省显存、降延迟）；本质是「weight 自己给自己做互动」。

三个细节从方法论上区分了这些工作：只有 GPTQ 一类不做变换、只在 weight 内部回补；其余方法在 weight 与 activation 之间倒换难度。讲者说清了这个区分，是为了后文「为什么 AWQ+GPTQ 复合收益有限」埋伏笔。

### 校准数据：token 分布一致 + 句间逻辑一致，两条都不是免费的

讲者用 WikiText-2 perplexity 做实验组：C4、Pile validation、WikiText 三种校准数据中，WikiText 与测试集的 KL 散度最低、量化后 PPL 最低（6.133 / 6.144 两组相近数值并列出现，语涉具体配置，以论文表为准）；结论一句话：「量化参数对于更加拟合到这个测试数据上面」——真实场景的建议是直接用真实对话数据做校准。

第二条更细：把 WikiText 的句子顺序随机打乱再做校准，PPL 明显变差（字幕称「涨了」）。讲者把这归因于「句间逻辑」：随机生成的 token（哪怕词频对）破坏了真实语料的句法连接关系。「并不能完全随机的生成这个校准数据」是讲者给 synthetic calibration 的直接告诫。

### Transformation 的三个结论：全是「不得 free lunch」的实测

讲者用了一个逐 channel 峰度型度量 $K$ （字幕近似表述为逐 channel 偏移累加）量化 outlier 严重程度，在 attention 与 FFN 各层上对比 FP 模型与变换后的 outlier 与层输出余弦相似度：

1. **W4A8 不如 W6A6**（Table 3 口径）：等价变换「消耗 weight 来弥补激活」——激活 outlier 降下去的代价是 weight outlier 增加；只有当 weight 比特高于 activation 时这个交换才划算，W4A8 的 PPL 反而大于 W6A6。这直接给了低比特 weight 场景下 transformation 的使用边界。
2. **down 层是模型中 outlier 最重的层**，去掉 down 层的变换（早期方法为了避开在线计算常常不做 down）量级上吃亏：「做了 down 层，其实这个（ $K$ 值）减了 1000 多」；加上 down 变换后各方法 PPL「都降低了很多」。讲者在 Q&A 中又把这条接到混合精度上：down 层该给高比特（「这一层可能需要用混合精度，也要用高比特」），是 benchmark 给出的层级敏感度启发。
3. **旋转法打破了「outlier 越小越好」的朴素信仰**：QuaRot 的 $K$ 值最低，却因为「只优化了这个 tensor 的 outlier，并没有去看这一层的输出」，其层输出（量化后/前）余弦相似度在靠后层上不如 AWQ，weight-only 例会上 PPL 最差；讲者特别声明这是「为了举这个例子」——weight+activation 同量化时 QuaRot「还是很不错的」。修复方向是已有后续工作把最终输出纳入旋转学习（字幕称「spring框S」，开销较大）。这一条把 benchmark 从「评算法」推进到「评指标」：outlier 不是输出保真度的代理指标。

### Clip 的两个结论：对称性匹配与比特依赖

- **对称量化配对称 clip、非对称量化配非对称 clip**：AWQ 本来是非对称量化却用了对称 clip，属 suboptimal；W3 weight-only 用非对称 clip 后「都有提升」，W2 极低比特下「如果你只要上了这个非对称 clip，其实这个提升是非常非常明显的……提升到了 13 点几」（字幕口径：修正前误差量级达 $10^5$ 量级表述）。
- **高低比特的误差主因不同**：低比特时 rounding 误差主导，clip 掉极端值反而让 scale 变好、正收益；高比特时 clip 的信息损失本身成为主要误差源，W3A16/W4A16 关掉 clip 分数反而更高。weight-activation 量化（W6A6、W8A8）则「无论怎样，clip 一定都是正向的作用」，讲者的机理解释是激活误差在总误差中占大头，clip 是在防止激活误差被放大——这又和 AWQ「显著权重」动机呼应上了。

### 复合方式与数值格式：复合收益取决于机制互补

- **AWQ + GPTQ 收益有限**：两者都会放大 weight 的 outlier（AWQ 为激活变换放大权重、GPTQ 把误差回补到后面未量化列），靠后列的绝对误差明显更大；
- **QuaRot + GPTQ 明显好于单用 QuaRot**：旋转补的是「降 outlier」、GPTQ 补的是「降层输出误差」，机制互补，输出相似度「回来了」——与第 3 条结论自洽；
- **整数 vs 浮点格式**：同比特下，weight+activation 时浮点格式更好（非均匀格点适配近高斯的张量分布，SmoothQuant 在 W4A4 整数下「完全不行」、换浮点后「还能看」）；weight-only 且分组量化后每组分布更均匀，整数量化反胜（且浮点格式有正零/负零等不可用码点，低比特时信息损失更明显）。讲者半开玩笑地指出一个现实 gap：理想做法是「激活用浮点、weight 用整数」，但「现在没有这种 kernel」。

### Q&A 精选（讲者的坦率与边界）

- **PPL 能说明量化好吗？** 不能单独说明：差 0.5 能定性，差 0.01–0.02「并不一定」；有下游任务就用下游任务评测。
- **AdaRound 为什么大家不用？** CNN 式 block reconstruction 需要约 20K iteration，7B 模型「超过十个（小）种（时）」（字幕口径）；减 iteration 效果变差、但调好参数「还是有收益的」。讲者对 20K 与 7B 的开销比是口头估算，数值以论文/工具实测为准。
- **二比特以下怎么样？** binary 量化「实际上根本就不怎么能用」（移位运算本身并不比乘除快），偏学术；2-bit 本身可用 codebook 路线（讲者举 AQLM），且 HuggingFace 已有支持。
- **LLM Quant 的 QAT 为什么不如 PTQ？** 今天所谓的大模型 QAT（LLM-QAT 等）本质是小数据蒸馏，「他们很多的效果还不如现在一些 PTQ 好」；资源充足可以真做 QAT，「但应该会挺久的」。
- **LLaMA 3 为什么难量化？** 主因是它「这个 outlier 会比 LLaMA 2 更加的严重」；「训练比较好的模型比较好量化」是普遍现象而非定理，70B 尤其如此，405B「其实还是很不错的」。
- **AWQ 是 weight-only 的，为什么 LLMC 里能量化激活？** AWQ 的本质是量化前 transformation，「我做完 transformation 之后，我把激活量化了，这也没有关系」；这正是把算法拆成三类技术再重组的方法论收益。
- **int4 权重读入 SRAM 转 FP16 的加速主因？** weight-only 量化的主要加速来自数据搬运（decode 阶段是 memory-bound），不是计算；prefill 才是计算密集。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 校准=WikiText 时的 WikiText-2 PPL | C4 / Pile validation 作校准 | 6.133 / 6.144（两组相近数值，字幕口径） | 字幕（校准数据部分） |
| 句序打乱的校准数据 | 与测试同源校准 | PPL 上升（变差） | 字幕（校准数据部分） |
| W4A8 vs W6A6 的 PPL 对比 | Table 3 口径 | W4A8 的 PPL 大于 W6A6 | 字幕（transformation 部分） |
| 做 down 层变换对 outlier 度量 $K$ 的影响 | 未做 down 变换 | $K$ 值减少 1000+（字幕口径：「减了1000多」，量级 $10^3$ ） | 字幕（transformation 部分） |
| W2 极低比特关/开非对称 clip 的 PPL | 对称 clip（AWQ 原配） | 非对称 clip 后回落到「13 点几」 | 字幕（clip 部分） |
| W3/W4（A16，weight-only）关掉 clip | 开 clip | 关 clip 后反而提升 | 字幕（clip 部分） |
| AdaRound 的 block reconstruction 迭代量级 | CNN 传统配置 | 约 20K iteration；7B 模型开销大 | 字幕（Q&A） |
| weight-only 量化加速来源 | 计算 vs 数据搬运 | 主要来自数据搬运（decode 为 memory-bound） | 字幕（Q&A） |

## 可迁移

- **评测纪律可直接搬进任何后训练/压缩流水线**：校准集与评测集「分布一致」不是形式合规，而是量化参数拟合质量的直接决定变量；「句序打乱能测出差距」这个对照实验，任何做合成校准数据的团队都应该自检一遍。
- **outlier 不是保真度的代理**：这条与本仓库 eval 方法论完全同构——优化中间指标而不验目标输出相似度，会得到「看起来更干净但更差」的模型。量化、训练监控（loss 平顺 vs 下游分数）都是同一个病。
- **层敏感度是一等公民**：down 层（以及一般意义上的「最难层」）该给高比特/混合精度，这个结论在 Q&A 中被讲者直接推广为层级 mixed-precision 的启发——与 RL infra 里对敏感模块（logit 层、归一化层）给高精度的做法同构。
- **拆解优于整表排名**：把 SmoothQuant/AWQ/GPTQ 拆成 transform/clip/reconstruct 再重组（AWQ 拿去量激活、QuaRot+GPTQ），是 benchmark 类论文最大的复用价值：工具化之后算法是「模块货架」，不是固定菜单。
- **「没有这种 kernel」提醒**：混合数值格式（FP 激活 + INT 权重）在 paper 里再对，落不了 kernel 就落不了地；选量化方案时先确认目标推理引擎的算子支持。

## 疑问 / 下一步

- 讲者引用的 Table 3 具体数字（W4A8 vs W6A6 的 PPL 差值）字幕未逐格复述，建议直接看论文 Table 3 与 LLMC 仓库的复现入口。
- 讲者提到的「以最终输出学习旋转矩阵」的后续工作（字幕音近「spring框S」）未展开；这类「输出感知旋转」与 QuaRot 的差距值得单独核对。
- 讲者对 synthetic calibration「句间逻辑」的解释停留在现象层（句序打乱→PPL 变差）；如果改用 LLM 生成的真实句式合成数据、或只打乱段间而保句内，差距如何变化？讲授未覆盖。
- 「down 层给高比特」与 mix-precision 的一般规律（哪几层最敏感、敏感度随模型规模如何变）讲者在 Q&A 中表示「暂时还不在我们这个 benchmark 的系列里面」——仍是开放经验问题。

## 原文金句

> 「实际上不能完全随机的生成这个校准数据，还需要满足这个句子间的一个逻辑。」

> 「单独的 PPL 确实不能说明量化的很好……但是如果你说 PPL 相差 0.01、0.02，这种其实确实并不一定（能）说你谁好谁坏。」

> 「因为我们这个论文其实提供了一种就是指导书的一个作用。」（讲者谈论文定位，Q&A）
