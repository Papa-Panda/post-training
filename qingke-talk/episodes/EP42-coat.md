# EP42 — COAT：显存高效的 FP8 训练，实现高效深度学习

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP42-coat.html

> 「我们去优化的目标，也是希望减少 activation 和 optimizer states 的显存占用——比如说将它们 quant 到 FP8 的格式，这样就能很好地减小训练当中的显存占用。」——讲者提出 COAT 的动机（字幕 08:59）

## 元信息

- 期号：42
- 标题：COAT：显存高效的 FP8 训练，实现高效深度学习
- BV：BV1REpYzZE19
- 时长：00:47:01（讲授约 42 分钟 + Q&A 约 4 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：徐浩诚（加州大学伯克利分校一年级博士生；清华姚班本科；导师朱军、陈建飞；本工作为其在 NVIDIA 实习期间与韩松合作；均为字幕自述 [03:17]）
- 相关论文：Haocheng Xu et al., *COAT: Compressing Optimizer States and Activations for Memory-Efficient FP8 Training*，https://arxiv.org/abs/2410.19313 （COAT 即 Compressing Optimizer states and Activations for FP8 Training 的缩写，字幕作「code / Fps s training」为识别误差）
- 相关代码：COAT（开源实现；讲者 Q&A 提到与 Transformer Engine 均为静态计算图）
- B站链接：https://www.bilibili.com/video/BV1REpYzZE19/
- 字幕原文存档：本地 `transcripts/EP42.txt`（943 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP42，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 COAT 在字幕中作「code/cos」、quantization 作「框太」、E4M3 作「ECM3」，均以论文为准）。误差数值一项字幕作「MIC/MICE」，应为 MSE，已在表中注明。

## 一句话总结

COAT 瞄准的是 FP8 训练里一直没人省的那块显存：不只算子用 FP8 加速，而是把**优化器状态**和**激活**也都压成 FP8 存。关键发现是这两类张量的数值分布与权重完全不同——优化器状态的组内动态范围太小、FP8 的 256 个刻度只用到约 20 个，需要先用扩张函数把动态范围撑开再量化；激活则大头在非线性层，需要混合粒度的量化粒度与一条全 FP8 的精度流。结果是与 BF16 近乎无损、显存约降到 1/1.54、还能跑起 BF16 会 OOM 的配置。

## 核心

### 背景：显存到底被谁吃掉

讲者先算显存账。模型相关张量里，每个参数要存 FP32 master weight（4 字节）、AdamW 的一阶与二阶动量（各 4 字节）、FP32 梯度（4 字节），合计 16 字节/参数——7B 模型就是一百多 GB，单卡根本训不起来（字幕 [05:53]；Q&A 中讲者补了一笔：混合精度下还有一份 BF16 权重副本 2 字节，slide 未计入）。另一部分是 activation，随 batch size 与序列长度涨。FSDP 这类分片能把优化器状态、梯度、权重摊到多卡（16 卡就除以 16），但卡数越少这部分越致命，而且分片之后 activation 的占比反而更高。结论：**activation 与 optimizer states 是显存的两大头**，这正是 COAT 的优化对象。

对照组是 NVIDIA Transformer Engine：它用 FP8 主要是**加速计算**，并不以省显存为目标。COAT 的理想账是：优化器状态从 FP32 压到 FP8 省 75%，activation 从 BF16 压到 FP8 省一半，两者叠加显存大降（字幕口径约 75% 的理想空间，实际端到端约 1.54 倍）。

### 方法一：优化器状态量化——先扩张动态范围，再量化

难点先行：现成的 per-group 量化（每 128 个元素共享一个 scale factor）套到优化器状态上效果很差。讲者的可视化显示，各量化组内最大值/最小值的比值（dynamic range）非常小：二阶动量常小于 10，一阶动量也大不到哪去；而 E4M3 本身的动态范围约 $2 \times 10^{5}$ 。结果是 FP8 的 256 个可表示值只用到约 20 个，可能只用满 8 个 bit 中的 4–6 个，量化误差自然高。

解法是量化前先做**动态范围扩张**：对组内元素施加保号幂函数

$$f(x) = \mathrm{sgn}(x) \cdot |x|^{K}$$

当 $K > 1$ 时，组内最大值与最小值的比值被放大为原来的 $K$ 次幂。 $K$ 不用调参：令扩张后的动态范围恰好等于 E4M3 的动态范围， $K$ 可解析算出，并被 fuse 进量化 kernel、每个 optimizer step 在线计算。由于二阶动量的动态范围比一阶更小，它算出的 $K$ 也更大（字幕口径：二阶约 5–15，一阶约 1–3）。整条 optimizer step（反量化 → FP32 更新动量与权重 → 再量化）融合成单个 kernel，显存中始终只躺 FP8 版本，没有 FP32 峰值。

误差证据：讲者用 AdamW 实际更新项 $m / \sqrt{v}$ 的量化误差来衡量（这是真正进到参数更新里的量），全用 E4M3 无扩张时 MSE 为 20.1，加扩张后降到 12.3；格式选择上，一阶动量用 E4M3 + 扩张最好，二阶动量 E4M3 与 E5M2 都可以、但扩张不可省。（E4M3 最大可表示 448，E5M2 约 5.7 万但精度更粗，字幕 [11:44]。）

### 方法二：激活量化——大头在非线性层，要混合粒度

第二个发现：以往 FP8 工作多只量化 linear 层（因为计算在那里），但 activation 的构成恰恰相反——以 LLaMA 式 transformer 做 breakdown，非线性层占 activation 的 60%–70%，linear 层只有约 30%。只省 linear 层等于只省了小头。

COAT 的设计是两件套：

1. **FP8 precision flow**：每一层的输入与输出都保持 FP8 精度。既然输入本来就是 FP8，backward 需要存的张量直接存 FP8 输入即可，activation 显存先天减半；量化/反量化 kernel 还能与相邻层 fuse 掉，没有额外 overhead。
2. **Mixed granularity（混合粒度）**：linear 层计算密集，用 per-tensor 量化（对硬件友好）；非线性层用更细的 per-group 量化压误差。但实验给出一条硬边界：**LayerNorm 的输入不能用 per-block 量化**——用 4×4 block 量化 LayerNorm 输入时，量化误差随层数累积、显著高于其他方案；其余层用 per-group 还是 per-block 差别不大，最终方案是所有层统一 per-group。

linear 层还有一个加速技巧叫 **group scaling**：per-tensor 量化需要先对整个张量做 max reduction 才能定 scale，单独起一个 kernel 开销大（Transformer Engine 的 delayed scaling 用过去约 100 步的历史估 scale，实现复杂且可能伤精度）。COAT 把 reduction 拆成两步：先在组内（组大小 $G$ ）求最大值（这一步可 fuse 进上一个 kernel），再对缩小后的张量做全局 max。量化环节因此加速 5 倍（小张量）到 14 倍（大张量）， $G = 16$ 时理论上限是 16 倍。

### 实验：无损、省显存、能跑起 OOM 的配置

- **精度**：两个规模的 LM 从零预训练（讲者口径：较大者 250B tokens、每 batch 约 4M tokens，约 6–7 万步；1B 模型 300B tokens），training loss、perplexity、下游任务均与 BF16 非常接近；MATH 微调与 BF16 无 degradation，与 Transformer Engine 相当；VILA-1.5 7B 视觉语言模型 SFT 同样近乎无损。
- **单层效率**：不同序列长度/hidden size/batch size 下，加速 1.4–1.5 倍、显存降 1.65 倍。
- **端到端**：7B/13B/30B 上验证，显存约降 1.54 倍；省出的显存可以开更大的 batch，而 batch 越大 FP8 的加速收益越高，形成正循环。讲者举例：2 张 GPU 训 7B 时 BF16 直接 OOM、COAT 可以；13B 在 2 卡、30B 在单机 8 卡下同样是「BF16 做不了全参数微调、COAT 能做」。

### Q&A 边界（讲者自述）

- FP8 optimizer 与 FP8 activation 是两个独立模块，可单独使用都有收益。
- FP8 optimizer **只省显存、不加速计算**：计算大头在 linear 层，optimizer step 本身很快。
- 与 torch.compile 的关系：讲者称 TE「目前不支持」的原因自己也不确定；COAT 与 TE 的计算流程都是静态的，原理上应与 torch.compile 兼容，但团队没实测过——引用时注意这是判断而非实测。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 模型相关显存 | 每参数 FP32 权重 + 两动量 + FP32 梯度 | 16 字节/参数，7B 约一百多 GB | 字幕（背景部分） |
| 优化器状态量化 | FP32 存储 | FP8 省 75% 显存 | 字幕（背景部分） |
| 激活量化 | BF16 存储 | FP8 省一半显存 | 字幕（背景部分） |
| FP8 刻度利用率 | per-group 直接量化 | 256 个可表示值只用到约 20 个 | 字幕（方法一） |
| 扩张指数 $K$ | 一阶动量 | 约 1–3；二阶动量约 5–15 | 字幕（方法一） |
| 更新项量化误差 | 无扩张 MSE 20.1 | 加扩张后 12.3 | 字幕（方法一） |
| 激活构成 | linear 层约 30% | 非线性层占 60%–70% | 字幕（方法二） |
| group scaling 加速 | 单 kernel 做 per-tensor reduction | 量化环节快 5–14 倍（ $G = 16$ ，上限 16 倍） | 字幕（方法二） |
| 单层效率 | BF16 | 加速 1.4–1.5 倍、显存降 1.65 倍 | 字幕（实验部分） |
| 端到端显存 | BF16 | 约降 1.54 倍（7B/13B/30B） | 字幕（实验部分） |
| 预训练规模 | BF16 对照 | 250B / 300B tokens 两档，均近乎无损 | 字幕（实验部分） |

## 可迁移

- 给自己的训练框架做 FP8 省显存时，先 profiling 再下手：FSDP 分片后优化器状态不再是大头，activation 才是；而 activation 里非线性层占 60%–70%，只量化 matmul 输入输出等于没省到大头。
- 「量化前先适配分布」是可迁移的一般原则：目标张量的组内动态范围远小于 FP8 刻度范围时，先做保号幂扩张这类单调变换把刻度用满，比换格式（E4M3/E5M2）更管用；且扩张指数可以像 COAT 这样按组在线解析求解、不引入调参。
- 省显存的价值不只是省卡：显存换 batch、batch 换低精度加速收益，这条正循环在决定是否上 FP8 训练时应纳入核算。

## 疑问 / 下一步

- 扩张函数对动量做非线性变换后，AdamW 的偏差校正与有效步长是否仍严格等价，讲者只给了 $m / \sqrt{v}$ 的误差证据，没有展开理论——自己在长训练里用之前，值得先小规模复现验证 loss 曲线。
- 实验最大规模只到 30B（字幕口径），且都在 Hopper FP8 语境下；70B+ 与 MoE 架构上 per-group + 扩张的误差累积是否仍可控，talk 未覆盖。

## 原文金句（1-2句）

> 「（之前的很多工作）只关注了 linear layer 的 quantization，因为 linear layer 占据了绝大多数的 compute；但是我们发现 non-linear layer 其实才是占据更多 activation 的这一部分。」——讲者论激活量化的着力点（字幕 24:51）

> 「（FP8）optimizer 的话，它能节省显存，但是不能加速——因为主要的计算都发生在 linear 层上。」——Q&A 中讲者的明确边界（字幕 43:49）
