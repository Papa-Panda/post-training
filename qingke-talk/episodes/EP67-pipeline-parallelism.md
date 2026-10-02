# EP67 — 大模型训练流水线并行四部曲：吞吐、内存、负载均衡与线性扩展

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP67-pipeline-parallelism.html

> 「在 PP 里面，activation memory 无法通过加设备摊薄——每个设备的持有量由构造单元的生命周期决定；那我们就去缩短生命周期（V 形排布、BW 分离），或者干脆把长寿命的那一份 activation 搬去 CPU（选择性 offload），这两条路走通之后，PP 的内存甚至可以做到比 TP 还低。」（本期核心框架，语义据字幕）

## 元信息

- 期号：67
- 标题：大模型训练流水线并行四部曲：吞吐、内存、负载均衡与线性扩展
- BV：BV1v14QzoEZf
- 时长：01:37:28（主讲与答疑交织进行）
- 提炼日期：2026-10-02
- 分享嘉宾：万欣怡（新加坡国立大学在读博士，CIAI Lab（Sea 旗下独立 AI 研究院）；字幕中主持人介绍与称呼均为「万欣怡」，讲者自述单位为 CIAI Lab 与 NUS）
- 相关论文：（均为 Sea AI Lab / NUS 方向的连续工作，字幕未给 arXiv 号，链接待补）
  1. Zero Bubble Pipeline Parallelism（ZB，下称「ZB 流水线」）
  2. Pipeline Parallelism with Controllable Memory（V 形 schedule 家族：V-min / V-half / V-ZB）
  3. Pipeline Offload（选择性 activation offload，实现线性扩展）
  4. Vocabulary Parallelism（词表并行）
- 相关代码：讲者称四篇工作均已开源；ZB 已被 Megatron-LM、PyTorch、DeepSpeed、PaddlePaddle 等框架集成（字幕口径），DeepSeek V3 的 DualPipe 由 ZB 的 B/W split 与 Chimera schedule 叠加而成，团队后续贡献的 DualPipe V 已被 DeepSeek 接受并挂在其仓库首页
- B站链接：https://www.bilibili.com/video/BV1v14QzoEZf/
- 字幕原文存档：本地 `transcripts/EP67.txt`（1968 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP67，已存档）清洗提炼，来源为 AI 字幕原文（经清洗，专名可能有识别误差，如 Zero Bubble 的两个 schedule 在字幕中作「ZBH1/ZBH2」、Chimera 作「camera」、Megatron 作「MCTRL/麦克创」、Sea Sailor 模型作「C罗二」等，均按公开资料校订）。凡关键数字均出自字幕口径，与论文版可能有出入。

## 一句话总结

讲者把流水线并行（PP）的四大缺陷拆成一条逻辑链：先用 B/W 分离把 backward 拆成「算输入梯度（B）」和「算参数梯度（W）」两段可独立调度的 pass，消除流水线气泡（ZB）；再提出「构造单元生命周期（lifespan）÷ 重复间隔（interval）」的峰值内存公式，用 V 形排布把 lifespan 压到原来的 1/3（Controllable Memory）；再用 CPU offload 选择性搬走长寿命分片的 activation、拿到超线性收益，让每卡 activation 随设备数线性下降（Pipeline Offload）；最后把词表（embedding/LM head）计算均分到所有设备、用 online softmax 把同步移出关键路径，治首尾设备的负载不均（Vocabulary Parallelism）——合起来是一套「用 PP 替代 TP」的完整主张。

## 核心

### 背景：PP 好在哪、坏在哪

讲者先横向对比四种并行：DP 简单但训不了超单卡模型；TP 通信量大、跨不了节点；ZeRO/FSDP 是 DP 的改进，参数通信频繁、节点多时性能受限。PP 的核心优势是通信量最低、天生适合多节点，所以称它是「跨节点并行的首选策略」。但 PP 有四个硬伤，本期四篇工作正好一一对应：

1. **气泡**：流水线首尾填充期 GPU 空转，外加各设备负载不均时的「不均匀气泡」；
2. **激活内存大且不随设备数下降**：见下一节的推导，这是 PP 最反直觉的性质；
3. **负载不均衡**：词表计算压在首尾设备上（见 Vocabulary Parallelism 一节）；
4. **不能线性扩展**：加再多设备，每卡 activation 不变（由第 2 点直接推出）。

一个常被忽略的推导值得单独记下：设设备数为 $D$ ，第一个设备要连续跑 $D$ 次 forward 才等到第一次 backward，每次 forward 只产生全模型 $1/D$ 的 activation，于是它手里攥着 $D \times (1/D) = 1$ 份全量模型 activation——**parameter 随层数切分摊薄了，activation 却没有**。这就是为什么「模型大了之后没办法只用 PP 训练」，也是整场的出发点。

经验配比上讲者也给了口径（问答环节）：先开 DP，只有模型大到单卡放不下才动 TP/PP；TP 限于机内（一般 8 卡），开满还爆内存再上 PP 做跨节点；最后用 DP 横向扩数据。序列长度超过 64K 才需要认真考虑 sequence parallelism。

### 工作一：Zero Bubble——把 backward 拆成 B 和 W

关键观察是 backward 里两个矩阵乘法可以解耦：对一个线性层，forward 是一个 matmul，backward 是两个——算输入梯度（下称 B）和算参数梯度（下称 W）。流水线里上游设备只需要 B 的结果就能继续往前传，W 只被本设备自己的 optimizer 用。框架默认把两者塞在同一个 backward 调用里，ZB 把它们拆开、先算 B 传给上游、W 往后挪一位再算，得到两个 schedule：

- **ZBH1**：气泡降到 GPipe 基线的 1/3（字幕口径），峰值内存与 1F1B 基线相同；
- **ZBH2**：继续挪 W，气泡清零（需要先跑 \$2D - 1\$ 个 pass 把流水线填满），代价是 activation 累积翻倍到 2 倍峰值内存。

大模型训练实测加速约 30%（字幕口径）。讲者特意强调 B/W 分离本身是通用技巧，不必照搬 ZB 的编排——套到任何 schedule 上都有可观收益。Zero Bubble 也因此被公认为社区里第一个把流水线气泡做到零的方案。

> 一个容易被追问的工程细节（讲者在答疑里回答）：ZBH2 的甘特图允许每个设备的 optimizer step 异步，是因为 Adam 类 optimizer 逐参数独立；但 grad clipping 与 NaN/Inf 检查需要全局信息。论文的做法是先按「训练稳定后 clip/NaN 极少触发」的假设推进 optimizer step，在下一个 iteration 做后验检查，若发现越界就 rollback optimizer step 与已跑的 forward 重做。

### 工作二：Controllable Memory——峰值内存由 lifespan 决定

这篇给出了整场最有价值的分析框架。把一个 micro-batch 在所有设备上的全部 forward/backward 看成一个「构造单元」（building block），它以固定间隔（interval）周期性重复。峰值内存可以写成：

$$M_{peak} = \frac{L}{I} \times M_{mb}$$

其中 $L$ 是构造单元的生命周期（lifespan，一个 micro-batch 从 forward 到 backward 释放 activation 的时间跨度）、 $I$ 是重复间隔、 $M_{mb}$ 是单个 micro-batch 的 activation。直觉：forward 与 backward 隔得越久，中间穿插的 micro-batch 越多，峰值越高。

降内存就是缩短 $L$ ，两招叠加：B/W 分离只等 B 不等 W，关键路径变短；V 形排布把层按「第 1 层和第 8 层放同一设备」的对折方式切成 $2D$ 段、每设备放两段，使第一个 backward 再次大幅提前。数值上 lifespan 从 1F1B 基线的 $3D$ 降到 BW 分离后的 $2D$ 、再到 V 形的 $D$ ——**峰值内存只要基线的 1/3**。由此得到一个可调家族（相对 1F1B 基线）：

- **V-min**：内存 1/3（讲者称已证明这是不用 offload/recompute 时纯调度的下界），气泡约为基线的 2/3；
- **V-half**：内存 1/2，气泡 1/2；
- **V-ZB**：内存与基线相同、气泡为零——即同样零气泡，内存只有 ZBH2 的一半。

讲者称这一族推进了吞吐—内存的 Pareto frontier，而且 V 形排布是「free lunch」：不是拿内存换吞吐，而是既省又快。

### 工作三：Pipeline Offload——让 PP 的 activation 也能线性扩展

为什么 V-min 的 1/3 内存还不够？因为 TP 用 $D$ 个设备时 activation 是 $M/D$ ，而 PP 恒为 $M$ （V-min 也还有 $M/3$ ）——要在内存上追平 8 路 TP，PP 得做到 $M/8$ 以下。问题出在峰值公式里的 $L$ 不随 $D$ 变小。解法：既然 lifespan 长既是祸（内存高）也是福（有足够时间把 activation 搬去 CPU 再搬回来），就用 CPU offload 把长寿命分片的 activation 挪走，等效缩短它在 GPU 上的存活。

可行性由一个比值决定， $K = T_{offload} / T_{compute}$ ，而 $K$ 与 hidden size 和序列长度成反比：模型越大、序列越长，offload 越划算。在 A100 上以约 200 TFLOPS 算力、PCIe 约 15 GB/s 估算（字幕口径），hidden 约 8K（对应 30B 量级模型）以上、或序列长 32K 以上时可以全量 offload 而不损性能（实测降速小于 1%）；且所有情形下 $K < 2$ ，即至少能 offload 一半。

选择性 offload 有超线性收益：不同模型分片的 lifespan 不等，先搬最长的那个。实测设置中（每设备 16 个分片）offload 掉一半分片，剩余内存只有约 25%，而非线性预期的 50%。综合下来每卡 activation 大致为 $M/4 \sim M/6$ ，全量 offload 时：

$$M_{per\,GPU} = \frac{M}{V \times D}$$

其中 $V$ 是每设备的分片数、 $D$ 是设备数——此时 PP 的 activation 反而比 TP 更低。讲者自述这个结论「有些颠覆认知」：对比实验里全量 offload 的纯 PP 方案内存比 TP8 更少、吞吐还更高；半量 offload 内存约为 TP 的 2 倍但吞吐也更高。

### 工作四：Vocabulary Parallelism——把词表均分给所有设备

decoder-only 模型有 input embedding 和 output embedding 两个词表，通常被塞在第一个和最后一个设备上。讲者估算社区词表规模多在 1~3 个 transformer layer 的计算量（字幕口径），即尾设备负载是别人的 2~3 倍，流水线被拖出气泡。靠重排 transformer layer 摊平也行不通：层数是整数切不碎，且「计算拍平了内存就不平，内存拍平了计算就不平」。

方案是把词表切成 $D$ 份分到每个设备。但词表计算里横着一个 softmax cross-entropy，它要求对全词表 logits 做同步——这正是朴素切分做不了的原因。解法借 FlashAttention 的 online softmax 思路：先在本地算局部 softmax 并完成 backward 的两个 matmul，最后再做一次轻量同步并修正输入梯度与参数梯度，把同步移出关键路径。词表主计算称 S-pass、同步修正称 T-pass，依赖关系理清（S 依赖最后一层 forward 完成、跑在最后一层 backward 之前）后即可并入流水线调度。

### 结论/观点：社区采纳与未尽问题

讲者把四篇的分工收束为：前两篇把 PP 的吞吐与内存推到纯调度的极限，Vocabulary Parallelism 治负载不均，Pipeline Offload 治不扩展，「我们希望社区里面能够用 PP 替代 TP」。采纳情况（字幕口径）：DeepSeek V3 的 DualPipe = ZB 的 B/W split + Chimera（对折一半即 V-ZB 的特例），后续团队用 V 形 schedule 免掉 Chimera 的一份参数拷贝，成果被 DeepSeek 接受命名为 DualPipe V；PyTorch 集成了 ZBH1 及与 interleaved 结合的 ZBV；Sea 自家东南亚语言模型 Sea Sailor 用 ZBV + Vocabulary Parallelism 训练。

剩下的坑讲者也列了：纯 PP 设备数不能超过层数；ZB 类策略需要 $2D$ 个 micro-batch 喂饱，长序列训练 micro-batch 本来就少，会成为瓶颈；推理侧 PP 延迟高、KV cache 生命周期长，所以推理里 PP 总是最后才用。

## 关键数字

以下均为字幕口径，未与论文版核对。

| 指标 | 基线（1F1B） | 结果 |
|---|---|---|
| ZBH1 气泡 | 1 | 1/3（峰值内存不变） |
| ZBH2 气泡 | 1 | 0（峰值内存 2 倍） |
| ZB 大模型训练加速 | — | 约 30% |
| V-min 峰值内存 | 1 | 1/3（气泡约 2/3） |
| V-half 峰值内存 | 1 | 1/2（气泡 1/2） |
| V-ZB 峰值内存 | 1 | 1（气泡 0，为 ZBH2 的一半） |
| Construct-unit lifespan | $3D$ | V 形排布后 $D$ |
| 选择性 offload（一半分片，16 分片设置） | 剩 50% | 剩约 25% |
| 每卡 activation（offload 实践） | $M$ | 约 $M/4 \sim M/6$ |
| 词表计算量 | 1 个 transformer layer | 约 1~3 个 layer |
| 参数+优化器内存经验系数 | — | 约 12（ZeRO 经验式）~16（Megatron 混合精度） bytes/param |

## 可迁移

- 对 RL infra/训练框架工作的直接可试点：
  - 峰值 activation 的 `lifespan/interval` 公式是一个通用体检工具：你自己的 PP schedule 里哪一份 activation 活得最久、活着期间穿插了多少个 micro-batch，算出来就知道峰值从哪来、该先 offload 哪段；
  - B/W 分离不依赖 ZB 的具体编排，现成框架（Megatron/PyTorch 已集成）里可以直接开；训 RL 的 rollout+train 流水线时，「先传 B、W 延后」同样能提前释放上游等待。
- Infra 视角的启发：
  - 「省内存 = 提升吞吐」是讲者反复强调的一句：省下的内存可以涨 batch size 提计算密度，或者省掉 TP 这类高通信策略本身就是加速；
  - PP 的选型启发式（DP 打底 → 机内 TP → 跨机 PP → DP 扩数据，64K+ 序列才上 SP）可以直接写进并行度调参 checklist；
  - offload 的可行性比值 $K$ 随 hidden/seq 反比下降——长序列 RL 训练（大 seq）恰好落在全量 offload 不亏的区间，这个结论值得在自己的长上下文 RL 场景里实测一遍。

## 疑问 / 下一步

- ZB 家族需要 $2D$ 个 micro-batch 喂饱流水线，而长序列 RL 场景 micro-batch 数量稀缺（讲者自己列的坑）：在 RL 训练（rollout 长度大、micro-batch 少）中，V-min 这类 1/3 内存但带气泡的配置与 ZBH2 这类零气泡但 2 倍内存的配置，哪个实际 MFU 更高？下一步可找一篇在 RL 框架（如 verl/slime）里实测 PP schedule 的工作对照。

## 原文金句（1-2句）

> 「省内存就等于提升吞吐。」
>
> 「对于流水线并行来讲，activation memory 你用更多的设备是不会降低每个设备上的数量的。」
