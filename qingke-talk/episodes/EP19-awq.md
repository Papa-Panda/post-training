# EP19 — AWQ：激活值感知的 LLM 低位权重量化

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP19-awq.html

> 「大概只有 1% 的 weight channels 是最重要的……通过保护这 1% 的 channels，就可以很好的恢复这个模型在量化之后的一个性能。」——讲者在总结部分对 AWQ 核心观察的概括（字幕 33:51–34:12）

## 元信息

- 期号：19
- 标题：AWQ：激活值感知的 LLM 低位权重量化
- BV：BV1mDaYzuE7o
- 时长：00:49:46（总时长 2986 秒；讲授约 32 分钟 + Q&A 约 13 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：唐佳明（AWQ 作者之一；字幕中自述姓名与作者身份，所在机构字幕未提，正式头衔以论文作者页为准）
- 相关论文：Ji Lin, Jiaming Tang, et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*（MLSys 2024，最佳论文奖；讲者字幕自述获奖），arXiv:2306.00978；后续 serving 工作 QServe（W4A8KV4，字幕提及）
- 相关代码：AWQ 与推理引擎 TinyChat 已开源（讲者多处提及社区集成与下载量）
- B站链接：https://www.bilibili.com/video/BV1mDaYzuE7o/
- 字幕原文存档：本地 `transcripts/EP19.txt`（691 条，带时间戳，04:37–49:41）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP19，已存档）清洗提炼（经人工清洗；个别专名可能有识别误差，如字幕中「QUALIZATION」为 quantization、「拉玛」为 LLaMA、「crowd」为 Claude、「 tan chat」为 TinyChat、「A6Q」为 AWQ，均以论文与公开资料为准）。凡讲授口径与论文版数字不同，以下标注（字幕口径）；讲者未逐项口播的表格细数字不在此臆测。

## 一句话总结

AWQ 的关键发现是：权重通道的重要性由**激活**而非权重本身决定——按 activation magnitude 找出约 1% 的 salient weight channels，再用逐通道等价缩放把这批通道放大（scale up），使其量化误差降为原来的 $1/s$ ，全程保持均匀低比特、不引入 mixed precision；配合为边缘设备设计的 TinyChat 推理引擎，在 memory-bound 的 single-batch 生成场景拿到 3–4 倍加速。

## 核心

### 问题：边缘端 single-batch 推理是 memory-bound 的

讲者先用 roofline 图划定战场。以桌面 GPU 4090 为例，横轴是计算强度（近似正比于推理 batch size），分界点约在 batch size 等于 8：batch 更大时计算受限，W8A8 量化可以调用 INT8 tensor core，是更优选择；batch 更小时落入 memory-bound 区域，tensor core 再快也没用，瓶颈是权重搬运。个人 laptop 等边缘设备恰好在后一区域。

把推理拆成 prefill 与 generation 两阶段看更清楚：prefill 一次并行处理整段文本、计算密度高；generation 是逐 token 自回归，等效 batch size 等于 1，速度由显存带宽决定。此时的 memory footprint 又以权重为主导。两个因素叠加，结论是：**在边缘 single-batch 场景，减少 weight footprint 是第一杠杆，weight-only 量化比 W8A8 更契合**。动机数字：70B 的 LLaMA 用 FP16 部署需要 140GB 显存（两张 80G A100），量化到整型后一张 40GB 的 A100 即可。

### 前作 SmoothQuant 与它的适用边界

讲者顺带回顾了同组前作 SmoothQuant：activation 中存在 outlier channels、量化困难，而 weight 相对平坦，于是把 activation 的 outlier 通道除以一个缩放因子、对应 weight 通道乘以逆因子，数学等价地把量化难度从 activation 迁移到 weight，实现 W8A8。讲者明确划界：SmoothQuant 适用于 batch 较大或 prefill 加速；AWQ 针对的是 single-batch generation，两者不是替代关系（Q&A 中再次确认）。

### 关键发现：重要的通道要按 activation 找

起点实验很朴素：用最简单的 RTN（round-to-nearest）把 OPT-6.7B 量化到 3 bit，perplexity 从十几涨到 40 多，基本不可用。但若**保留 1% 的通道不量化**，perplexity 立刻从 40 多降回十几。于是问题变成：这 1% 该怎么选？

实验给了反直觉的答案：按 weight magnitude 选 salient channels，效果与随机选几乎无差别；而按 activation magnitude 选，性能显著提升。讲者给出的解释框架是：一个通道对输出的影响由「激活 × 权重」共同决定，权重数值大不等于它承载的信号大。这一条决定了 AWQ 全部后续设计的方向——以 activation 分布为准绳。

### 方法：把重要通道 scale up，而不是给它们开 FP16 小灶

直白的做法是把 1% 通道以 FP16 保存（mixed precision），但讲者明确拒绝：需要两套 kernel（一套 FP16 运算、一套整型乘法），系统复杂度陡增。替代方案是对 salient 通道做等价缩放：把权重通道乘上 $s$ 、对应激活通道除以 $s$ ，输出严格不变。实验现象是：乘 1.5 倍 perplexity 大降，乘 2 倍继续降，乘 4 倍反而回升——存在最优缩放，这正是下一节误差分析要解释的。

### 误差分析：为什么 scale up 有效、又为什么不能过头

量化即把浮点数按步长 $\Delta$ 映射到整数网格：

$$Q(w) = \Delta \cdot \mathrm{round}(w / \Delta)$$

其中 $\Delta$ 由一个 group（AWQ 取 128 个值一组）内的最大值决定。对第 $i$ 个通道做等价缩放后，该通道参与计算的是 $(x_i / s_i)$ 与量化后的 $s_i w_i$ ：

$$x_i w_i = (x_i / s_i) \cdot (s_i w_i)$$

误差来源只有 round 一项，其期望量级约为 $0.25 \Delta$ ，且在缩放前后不变；但输出端要再乘回 $1/s_i$ ，于是该通道的量化误差被压到约 $0.25 \Delta / s_i$ ——这就是 scale up 有效的原因。过头则失效：若 $s_i$ 大到改变了 group 内的最大值， $\Delta$ 本身变大，所有通道的误差一起放大。两个效应竞争，决定了最优 $s$ 的存在（字幕 19:27–23:16 的完整推导）。

### 搜索：activation-aware 地定每个通道的缩放

最优缩放不能手工指定，AWQ 用 activation 统计量构造搜索空间：以每个通道的平均激活幅值 $\bar{x}$ 的 $\alpha$ 次幂作为缩放候选，

$$s = \bar{x}^{\alpha}$$

再在小规模 calibration 数据上网格搜索 $\alpha$ 。讲者强调这一步是「activation-aware」的落点：重要性先验直接来自激活分布。实际校准开销很低，论文口径是 128 条、长度 512 的 calibration 样本即可收敛（Q&A 确认），且换不同 calibration 数据集结果无明显变化；AWQ 也可直接用在 VLM 上（ViT 先把图像转成 tokens 再拼接），个别 benchmark 上量化后甚至略优于 FP16。

### 系统：TinyChat 把算法收益兑现成速度

讲者反复强调算法必须配系统才成立。TinyChat 的设计目标是「像 HuggingFace Transformers 一样易用、像 TensorRT-LLM 一样快」，关键工作有二：一是按平台定制 weight packing（如为 Intel CPU 设计重排存储，配合 SIMD 指令，解包 32 个值只需 3 条逻辑运算，带来约 1.2–1.3 倍加速；此数字字幕识别略有模糊，按字幕口径记）；二是 kernel fusion，把 attention 与 GEMM 各自融成单个 kernel。结果是相对 HF Transformers 推理 3 倍以上加速、相对 AutoGPTQ 快 2.6 倍（同等灵活性下）；在 A100、4090、Jetson Orin Nano 上约 3 倍。讲者还展示了与哈佛学生合作的 7GB RAM 小盒子运行 7B 模型，点明边缘部署的隐私与离线价值。社区侧：HuggingFace 下载量超过 100 万，IBM、Berkeley 的 VLM、Intel 等均已集成。

后续工作 QServe 则把镜头转回 serving：batch 大于 1 时 W4A16 不再最优，QServe 用 W4A8 加 KV cache 4 bit 量化，在 L40S 上达到 A100 上 TensorRT-LLM 的推理速度。

### Q&A 中讲者给的边界

- AWQ 与 GPTQ 的权重完全可以共享，图中速度差来自 weight packing 与推理框架不同；同框架下速度一样，差别只在精度。
- AWQ 只优化 generation 阶段，prefill 未做专门优化，长 prompt 下首 token 返回慢（观众实测问题，讲者确认原因）。
- NF4 这类按高斯分布设计的数据类型可能在精度上占优，AWQ/GPTQ 的整型路线胜在硬件友好，是精度与性能的取舍。
- AWQ 的缩放依赖相邻层之间传递 scale，CNN 的卷积结构下不能直接迁移；堆叠式 Transformer 架构（含多模态）则普遍适用。
- 把 scale 作用到相邻 projection（如 V projection）不会显著抬高其量化难度，因为 outlier 只存在于 activation 侧、weight 本身平坦。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 70B LLaMA 部署显存 | FP16 需 140GB（2 张 80G A100） | 量化后 1 张 40GB A100 可跑 | 字幕 |
| OPT-6.7B INT3 RTN 的 perplexity | FP16 时十几 | 涨到 40 多，不可用 | 字幕 |
| 保留 1% salient channels 后 | RTN 的 40 多 | 降回十几 | 字幕 |
| salient 通道选择标准 | 按 weight magnitude 选 | ≈ 随机；按 activation magnitude 选显著更好 | 字幕 |
| scale up 实验（1.5 / 2 / 4 倍） | 未缩放 | 1.5、2 倍 perplexity 下降，4 倍回升 | 字幕 |
| group 量化粒度 | — | 每 128 个值共享一个 scaling factor | 字幕（Q&A） |
| calibration 规模 | — | 128 条 × 512 长度即收敛 | 字幕（Q&A） |
| TinyChat 相对 HF Transformers | HF 推理 | 3 倍以上；相对 AutoGPTQ 快 2.6 倍 | 字幕 |
| TinyChat 总体加速 | memory-bound 设备 | 约 3–4 倍 | 字幕 |
| CPU packing 解包 | 常规 packing | 32 个值仅需 3 条逻辑运算，约 1.2–1.3 倍加速 | 字幕 |
| 社区采用 | — | HuggingFace 下载量超 100 万 | 字幕 |

## 可迁移

- 做低比特 rollout / 部署压缩时，「哪些权重重要」的判据应取 activation 统计而非 weight magnitude：用一小批真实 prompt 的激活幅值给通道排序，比按权重大小裁剪/保护可靠得多。
- 先定位自己处在 roofline 的哪一侧再选量化形态：RL rollout 的 decode 阶段同样是 memory-bound，weight-only 低比特对生成吞吐的杠杆大于 W8A8；serving 侧（大 batch）再切换到 QServe 式 W4A8 + KV 量化。
- 算法与 kernel 必须一起验收：同一份量化权重换 packing 与推理框架，速度可以差出 2 倍以上——量化方案的 benchmark 必须带框架口径，否则不可比。

## 疑问 / 下一步

- Prefill 未被 AWQ 优化：长上下文 prompt 下首 token 延迟如何与 prefill 侧的量化/并行方案（如 SmoothQuant、chunked prefill）拼接，talk 未展开。
- Scale 搜索目前是逐层贪心 + 单参数 $\alpha$ 网格：它与 GPTQ 式误差补偿、clip 搜索能否统一进一个目标函数，讲者未给出结论。
- VLM 上 AWQ 的 kernel 支持被观众指出不完善（讲者归因为代码适配问题）：多模态模型上 packing 策略是否需要按 ViT 与 LLM 分别设计，值得实测。

## 原文金句（1-2句）

> 「大概只有 1% 的 weight channels 是最重要的……通过保护这 1% 的 channels，就可以很好的恢复这个模型在量化之后的一个性能。」——讲者总结（字幕 33:51–34:12）

> 「实际上 AWQ 跟 GPTQ 他的这个权重是完全可以共享的，只是因为我们的 weight packing 的方式不同。」——Q&A 回答两者速度差来源（字幕 40:27–40:40）
