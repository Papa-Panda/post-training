# EP28 — DuQuant：基于正交变换实现大型语言模型的 SOTA 级 4 bit 量化

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP28-duquant.html

> 「我们会进一步发现 massive outlier 它其实会在这个 FFN module 的第二个层、就是 down projection 的这个输入层……这个是我们一个比较重要的一个发现，也是 motivate 我们去提出两个正交变换来解决这个问题。」——讲者陈述 DuQuant 的立论起点（字幕 20:40–21:15）

## 元信息

- 期号：28
- 标题：DuQuant：基于正交变换实现大型语言模型的 SOTA 级 4 bit 量化
- BV：BV16CaRzWEvx
- 时长：01:01:44（总时长 3704 秒；讲授约 47 分钟 + Q&A 约 12 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：本期讲者（字幕未自报姓名；自述来自香港城市大学，导师孙正南、魏莹；工作为中科院自动化所、香港城市大学、清华大学、浙江大学合作，正式作者信息以论文作者页为准）
- 相关论文：*DuQuant: Distributing Outliers via Dual Transformation for Superior Quantization*（NeurIPS 2024 oral；讲者字幕自述 oral），arXiv:2406.11235
- 相关代码：已开源（讲者欢迎在 GitHub 提 issue；talk 中亦提及团队 KV cache 方向的 AttentionKV 工作）
- B站链接：https://www.bilibili.com/video/BV16CaRzWEvx/
- 字幕原文存档：本地 `transcripts/EP28.txt`（1450 条，带时间戳，02:12–61:38）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP28，已存档）清洗提炼（经人工清洗；个别专名可能有识别误差，如字幕中「地球矿/丢油矿/DIO框」为 DuQuant、「欧米矿/OMEIQU」为 OmniQuant、「q a road/QA6」为 QuaRot、「spring框」为 SpinQuant、「autumn」为 Atom、「VK test two」为 WikiText-2，均以论文与公开资料为准）。讲者未逐项口播的表格细数字不在此臆测；凡引论文口径处会特别标注。

## 一句话总结

DuQuant 先把既有方法在 4 bit 翻车的原因定位到 FFN down-projection 输入层的 massive outlier，再用两级正交变换——基于 outlier 先验构造的 block-wise 旋转，加 zigzag 通道重排——同时平滑 activation 与 weight 两个曲面，于是最朴素的 RTN 就能在 W4A4 上打出 SOTA 精度，且 7B 模型单卡 50 秒量化完毕、不需要 GPTQ 兜底。

## 核心

### 背景与量化设定

讲者先铺了 PTQ（post-training quantization）的完整设定：训练好的全精度模型 + 少量 calibration 数据、无梯度训练（区别于 QAT）；DuQuant 采用均匀量化 + 非对称量化（硬件友好），activation 按 per-token、weight 按 per-channel 分配 scale 与 zero point，且把 KV cache、query、hidden states 等所有 activation 一并量化。基础量化公式是 min-max 形式：

$$Q(x) = \mathrm{clamp}\left( \left\lfloor \frac{x}{\Delta} \right\rceil + z, 0, 2^b - 1 \right), \quad \Delta = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

其中 $\Delta$ 是步长、 $z$ 是 zero point。后面所有问题都出在同一个地方： $x_{\max}$ 被极端值撑爆后， $\Delta$ 变大，有限的整数格点被少数 outlier 独占。

### 两类 outlier，以及 down-projection 的新定位

讲者严格区分了两种 outlier。**Normal outlier**：少数 channel 在所有 token 上都显著大（SmoothQuant 图中那种），per-token 量化时每列的 scale 都被它拉大。**Massive outlier**：数值极端（可超过 1000、甚至 5000），但只集中在极少数 token 上，典型出现在起始的 BOS token 附近；这一现象来自《Massive Activations in Large Language Models》（COLM）与讲者参与的 KV cache 工作。

DuQuant 的关键新观察是把定位粒度推进了一步：massive outlier 明确出现在 **FFN 第二层 down-projection 的输入**，而此前工作只定位到 transformer block 的输出层。讲者称这是整个工作的 motivation 起点——它能解释为什么 SmoothQuant 与 OmniQuant 在这一层、进而在 4 bit 下集体表现不好。

### 为什么旧方法恰好在这一层失效

讲者逐个拆解了基线在 down-projection 输入上的失败模式：

- **SmoothQuant** 把该层的量化难度往 weight 转移时，由于 outlier 值特别大，转移因子 $S$ 也特别大，结果在原本平坦的 weight 上**人为制造出新的 outlier**，反而增加 weight 的量化难度；且 massive 值经转移后仍有约 100（原约 1000），难以消除。
- **OmniQuant / FLAQuant** 这类可学习方法在这一层会遇到梯度更新不稳定，实践中直接丢弃了该层的可学习参数，等于绕开问题。

结论：4 bit 量化精度差不是「比特不够」，而是这一层的数值 landscape 没人处理过。

### 方法一：基于 outlier 先验的 block-wise 旋转

第一级变换是旋转：用正交矩阵把 outlier 能量分摊到邻近 channel。由于理想正交矩阵不可得，DuQuant 用先验知识贪心地构造：先定位最大的 outlier channel（取该列在所有 token 上的最大值、再 argmax 得列号），用列置换把这一列换到第一列，与一个「第一行均匀分布」的初始旋转矩阵相乘，使最大通道被均匀分摊到其他列；再重复旋转多步，取整体 outlier 最小的一步。为控制开销，旋转是 block-wise 的，最终采用 128 个 channel 一个 block。

正交变换的关键性质是输出不变性：线性层 $X \cdot W$ 在激活侧右乘 $R$ 、在权重侧左乘 $R^{\top}$ 之后，计算结果严格等价：

$$X \cdot W = (X \cdot R) \cdot (R^{\top} \cdot W)$$

讲者特别点出：这套变换同时作用在 $X$ 与 $W$ 上，所以它不只平滑 activation，也平滑 weight——这是后面「不需要 GPTQ」的原因。

### 方法二：zigzag 通道重排（permutation）

Block-wise 旋转留下残余问题：被旋转过的 block 整体仍明显大于其他 block，分布跨 block 不均。第二级变换是通道置换：按每列最大值从高到低排队，用 zigzag 顺序（像按身高排队、先递减到最后一个 block、再折返回来）把大 channel 分散到不同 block，使 block 间均值与方差尽量接近，再做第二次旋转即可进一步压低 outlier。消融对比很说明问题：不用置换最差，随机置换已有大幅提升，zigzag 再进一步，而模拟退火学出的置换性能无明显优势、速度却慢得多——最终选 zigzag，只存 channel id，实现开销可忽略。

完整流水线是：先做 SmoothQuant 式对角变换把难度从 activation 转移一部分，再旋转 → 置换 → 旋转（讲者的默认配置是旋转两次、置换一次），并可选用 OmniQuant 的 LWC（learnable weight clipping）在部分情形再提一点。可视化结果：LLaMA-1-65B 的 normal outlier 被压到很平；量级约 800 的 massive outlier 经两级变换后降到易于量化的平面。

### 实验：W4A4 是主战场，且校准数据没想象中重要

实验覆盖 LLaMA 1/2/3、Vicuna、Mistral，任务含 perplexity、commonsense QA、MMLU、MT-Bench 与 LongBench（解码 3500 步测长文本生成）。讲者的主结论是：

- 在重点的 W4A4 设定下 DuQuant（含 LWC 变体）明显优于既有基线；以难量化著称的 LLaMA3（8B 与 70B）上相对 Atom 等方法优势依然显著；Vicuna 在 MT-Bench 上甚至与 FP16 模型用 GPT-4 打分对比时仍保持竞争力。
- LongBench 上量化后的 Vicuna 7B/13B 相比 FP16 只掉约 2–3 个点（字幕口径）。
- **量化耗时极小**：因为变换已把 weight 一并平滑，权重侧只需 RTN、不必跑 GPTQ，单卡 A100 上 50 秒即可完成 7B 模型量化。
- 一个讲者自己觉得有趣的发现：calibration 数据换成「从词表随机生成的 128 条 2048 长度序列」，性能基本不变。他的推论是 outlier 更多是模型自身的固有属性、与校准数据关系不大，这意味着 **data-free 量化**（无需真实数据，利好隐私场景）可能成立。
- 与最接近的 QuaRot 对比：DuQuant 速度略慢、但精度更好且显存占用相当；变换带来的额外推理开销约 10%（Q&A 口径），讲者判断在可接受范围。置换只做一次的理由是：两次置换 perplexity 无明显变化、性能劣化却更多。

### Q&A 与讲者的自我定位

- 与同期工作 SpinQuant 的区别被专门澄清：SpinQuant 是在 QuaRot 的 Hadamard 矩阵上做可学习化且依赖 GPTQ；DuQuant 是用 outlier 先验构造旋转、不依赖 GPTQ，也不认为两者「撞车」（投稿与挂 arXiv 时间相近，本质不同）。讲者还委婉质疑了 SpinQuant 在 MacBook 上测速的可比性。
- LLaMA3 量化下降更明显，讲者归因于词表变大等新特性，提到把旋转矩阵完全可训练的工作在 LLaMA3 上有提升。
- 4 bit 是否到头：讲者认为 W4A4 的 W/A 量化仍会继续做（4 bit 乘法最好加速），weight-only 方向则已走向二值化；与 QAT 结合在资源允许时是值得考虑的进一步提升路径。
- 硬件落地现状坦白：TensorRT 一类推理栈的支持还没有做完，旋转与置换矩阵 fuse 进 kernel 是 decoding 继续加速的方向。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| massive outlier 量级 | 普通激活值 | 超过 1000、甚至 5000，且只集中在少数 token | 字幕 |
| SmoothQuant 转移后的残值 | 原约 1000 | 仍约 100，难以消除 | 字幕 |
| 旋转 block 大小 | — | 128 个 channel 一个 block | 字幕 |
| 默认流水线配置 | — | 旋转 2 次 + 置换 1 次；可选 LWC | 字幕 |
| 7B 模型量化耗时 | 依赖 GPTQ 的方法更慢 | 单卡 A100 约 50 秒（权重侧只需 RTN） | 字幕 |
| LongBench 长文本生成 | FP16 Vicuna | 量化后约低 2–3 个点 | 字幕 |
| calibration 数据 | 128 条 WikiText-2 | 换成随机生成数据，性能基本不变 | 字幕 |
| 变换的推理额外开销 | 无变换基线 | 约 +10%，与 QuaRot 相当 | 字幕（Q&A） |
| 硬件对比（prefill 速度 / decoding 显存） | QuaRot | 速度略慢、精度更好、显存相当 | 字幕 |

## 可迁移

- 「先定位最难的那一层，再设计变换」是可直接抄的排障顺序：做任何 W/A 低比特量化前，先按层扫描 activation 的最大值分布，找出 massive outlier 的具体落点（很可能在 down-projection 输入），再决定变换与 scale 策略，而不是全局调参。
- 正交变换的「输出不变、只改数值 landscape」思路可迁移到训练与推理两侧的数值治理：凡是形如 $X \cdot W$ 的线性层，都可以在不改变函数的前提下重新分配 outlier 能量；做 rollout 量化或 FP8 训练前的 activation 平滑时可先试这一类无训练变换。
- Calibration 数据的结论（outlier 近似是模型固有属性、随机数据可替代）值得在自己的 PTQ 流程里复验：若成立，校准环节可省去真实数据准备，尤其适合拿不到用户数据的部署场景。

## 疑问 / 下一步

- Massive outlier 为何集中在 down-projection 输入，讲者给的是定位而非机制解释：它与 attention sink / BOS 的关系、是否随模型家族（大词表 LLaMA3）系统性增强，值得追问与实测。
- 旋转矩阵若改为全程可训练（讲者提到这类工作在 LLaMA3 上更好），与 DuQuant 的先验构造能否叠加，稳定性与耗时如何，未在 talk 中展开。
- Data-free 量化的边界条件只在一个随机数据实验中验证：对更新的模型（MoE、大词表）是否仍成立，需要在自己的模型上复验后再依赖。

## 原文金句（1-2句）

> 「我们会进一步发现 massive outlier 它其实会在这个 FFN module 的第二个层、就是 down projection 的这个输入层……也是 motivate 我们去提出两个正交变换来解决这个问题。」——讲者陈述立论起点（字幕 20:40–21:15）

> 「欢迎大家使用 DuQuant 作为一个 baseline，在后面的工作中打败它，或者提出更好的解决方案。」——讲者收尾（字幕 61:26–61:33）
