# EP20 — S-LoRA：实现多 LoRA 大模型的高效并行化推理

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP20-s-lora.html

> 「S-LoRA 主要是为了高效部署上千个 LoRA adapter 所提出来的一个系统……相比同期的别的推理系统，可以达到将近四倍左右吞吐量的提升，并且它能承载 100 倍数量的 LoRA adapters。」——讲者开场定调（字幕口径，专名已校正）

## 元信息

- 期号：20
- 标题：S-LoRA：实现多 LoRA 大模型的高效并行化推理
- BV：BV1mDaYzuEqK
- 时长：01:00:14
- 提炼日期：2026-10-02
- 分享嘉宾：字幕中未自述姓名（自述为 UC Berkeley 在读博士生，研究方向为 LLM 推理加速；论文一作为 Ying Sheng，姓名对应以论文作者页为准）
- 相关论文：Ying Sheng, Shiyi Cao, Dacheng Li, et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*，https://arxiv.org/abs/2311.03285 （MLSys 2024）
- B站链接：https://www.bilibili.com/video/BV1mDaYzuEqK/
- 官网预告：青稞Talk 官网第 20 期预告（链接未确认，略）
- 字幕原文存档：本地 `transcripts/EP20.txt`（1183 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP20，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名有识别误差，如 LoRA 在字幕中作「LAURA」、S-LoRA 作「SLAURA/SLOL」、adapter 作「ADAPTTER/debtor」、vLLM 作「VRM/V2M」、Punica 作「普妮卡」，均已按论文与公开资料校正）。论文（arXiv:2311.03285）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

S-LoRA 把多租户 serving 拆成一个显存问题加一个算子问题：所有 adapter 平时躺在 CPU 内存、只把当前 batch 用到的搬上 GPU，KV cache 与 adapter 权重共用一个以 hidden size 为页大小的 Unified Memory Pool 动态配比，再用异构 batching 的自定义 kernel 对付「同一 batch 里 rank 各不相同、权重还不连续存放」的计算；结果是单卡可挂数千个 adapter 而吞吐几乎不降（字幕口径约 4 倍于对照系统、承载量约 100 倍）。讲授后半还引出了多租户公平调度这一后续问题，并给出 Virtual Token Counter 的 max-min 公平解法（字幕口径）。

## 核心

### 引入：每人一份模型为什么不行

服务商想给每个用户提供定制化服务（个人助手的人格与偏好、企业在自有数据上的微调、把长文档微调成 adapter 当检索的替代品，字幕口径），但给每人部署一份大模型不经济：单个用户的请求是稀疏的，大部分时间在空转。讲者的解法是共享一份基座模型、每个用户只维护一个 LoRA adapter，请求随之变成「基座加 adapter」的二元组，serving 系统要解决的是如何同时、高效地跑成百上千个 adapter。

### 规模感：adapter 小到可以「不心疼」地存

LoRA adapter 只有两个 $H \times r$ 的小矩阵：以 $H = 4096$ 、 $r = 16$ 计，单层 adapter 不过零点几 MB（字幕口径），这决定了它可以全量放在 CPU 内存里、按需上卡。真正的瓶颈随之转移：adapter 数量一大，它们在 GPU 上与 KV cache 抢显存——显存被 adapter 占死，batch 就上不去，吞吐就上不去。讲者把问题归结为两个挑战：大量 adapter 的存储层级安排，以及 rank 异构时批量矩阵乘怎么做。

### 系统设计三件套

1. **存储层级**：全部 adapter 存 CPU 内存，仅当前 batch 需要的搬到 GPU；adapter 尺寸小，搬运通信可以被计算重叠掉，不构成额外关键路径成本（字幕口径）。
2. **Unified Memory Pool**：KV cache 与 adapter 权重放进同一个分页内存池，页大小取两者的公共维度 hidden size $H$ ——一个 token 的 KV 占一页、一个 adapter 占 $r$ 页，两类张量在池内交错、非连续存放，配比随负载动态伸缩（请求集中时多给 KV、请求分散时多给 adapter），避免固定分区两头浪费（字幕口径）。
3. **异构 batching kernel**：基座计算与 adapter 计算分开走；adapter 部分是大量小矩阵向量乘，且每个矩阵 rank 不同、还分散在内存池里，现成的 batched GEMM 用不了。讲者的做法是借助索引表定位每个 adapter 的页，再在一个 thread block 内完成一个 $H$ 维向量乘（decode 阶段口径，prefill 类似）。Q&A 确认该 kernel 是在 Punica 的 BGMV kernel 之上改写、对齐 unified memory pool 的（字幕口径）。

补充：张量并行沿 Megatron 式策略，adapter 部分的通信与基座通信做融合，额外开销很小；实验在 LLaMA-30B / 70B、10-100 个 adapter 上验证（字幕口径，setting S5 / S6）。

### 实验：adapter 从 5 个加到 2000 个，吞吐基本不掉

主实验在单 GPU 上设五个 setting（基座 hidden size 不同、adapter rank 从单一到 8-64 混合不等，字幕口径），adapter 数量从 5 一直扫到 2000：S-LoRA 吞吐相比同期系统约 4 倍提升，且 adapter 数量增长时吞吐下降可忽略；对照方案（字幕提及的 naive 方案等）在 adapter 增多时会直接显存溢出。消融最说明问题的是 unified memory：adapter 很少时开不开差别不大，数量一大，不开的吞吐明显塌下去——因为被浪费的显存本可以给 KV cache 换更大 batch（字幕口径）。讲者也提醒对照口径是论文提交当时的状态，其后 vLLM 等系统已陆续补上多 adapter 支持（字幕口径）。

### 后续问题：多租户公平调度

讲授后半把问题从「跑得动」推进到「分得公道」：FCFS 会让高频用户饿死低频用户，per-user rate limit 能隔离却浪费产能（讲者给的算例：系统容量 20k token/s，四个用户各限 5k，实际只服务掉 14k，字幕口径）。目标是 max-min 公平：先给所有人等额份额，再把富余产能均分给没吃饱的人，如此迭代。落到 LLM serving 有三个传统调度没有的难点：输出长度事先未知、input 与 output token 成本不同且 output 成本随生成长度变化、preemption 在 KV cache 搬运下代价太大不能频繁做（字幕口径）。讲者的解法 Virtual Token Counter 是后续工作：给每个用户（或每个 adapter）维护一个虚拟计数器，decode 每步按一个与输入输出长度相关的 cost function 累加，进 batch 时永远优先 counter 最低的用户；用户离开后回归时，把其 counter 抬升到当前活跃用户的最小值，防止「回归用户」瞬间变成激进客户端。讲者称其有理论分析保证任意时间窗内两用户所得服务之差有常数界（字幕口径）。

### Q&A 要点

- **现在能直接用吗**：讲者坦承 S-LoRA 仓库已不活跃维护，定位是研究原型；相关能力应去看 vLLM 与 SGLang，两者都已（在与其团队协作集成后）支持多 LoRA serving（字幕口径）。
- **与 Punica 的区别**：kernel 部分本就建立在 Punica 的 BGMV 之上，Punica 的贡献在 kernel 优化，S-LoRA 想推的是「内存管理加 batching」的一整套 serving recipe，各组件要合在一起才有最高吞吐（字幕口径）。
- **多 adapter 的延迟代价**：对比 0 个与 200 个 adapter 的延迟曲线，增加很少、近似常数（字幕口径）。
- **adapter 权重从哪来**：直接用 HuggingFace PEFT 就能微调得到，serving 系统不关心训练侧细节（字幕口径）。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 吞吐量 | 同期推理系统 | 约 4 倍提升 | 字幕（讲授与结果部分） |
| 可承载 adapter 数量 | 同期系统 | 约 100 倍，实测扫到 2000 个仍不降吞吐 | 字幕 |
| 单层 adapter 体积 | $H = 4096$ 、 $r = 16$ | 约 零点几 MB | 字幕 |
| 公平调度算例 | 系统容量 20k token/s，4 用户各限速 5k | rate limit 实际仅服务 14k token/s | 字幕 |
| 张量并行验证 | LLaMA-30B / 70B | 10-100 个 adapter | 字幕（setting S5 / S6） |
| 多 adapter 延迟 | 200 个 vs 0 个 adapter | 增加很少，近似常数 | 字幕（Q&A） |

## 可迁移

- 对 serving 架构的直接启发：KV cache 与权重类张量共用一个分页池、按公共维度定页大小，是把 PagedAttention 的思想从单类张量推广到多类张量；凡是「很多小权重加一份大权重」的工作负载（adapter、多租户微调模型、MoE 专家缓存）都可借这套配比方式。
- 对评测自动化的启发：吞吐结论必须连同 adapter 数量与 rank 分布一起报——同一系统在 5 个与 2000 个 adapter 下是两个系统；做 serving 基准时应把「并发 adapter 数」当作与 batch size 同级的扫描维度。
- Infra 视角的组织结论（讲者 Q&A 口径）：研究原型 kernel 最终要活下去得进入 vLLM / SGLang 这类主线引擎；自研系统验证 recipe、主线系统承接维护，是这类系统的现实生命周期。

## 疑问 / 下一步

- Unified Memory Pool 的页取 $H$ 是对当前 LoRA 形状的巧合式对齐；若 adapter 形式变化（如对 FFN 与 attention 分别用不同 rank、或换到非 LoRA 的 PEFT 形式），页大小与索引机制要怎么改，讲授未展开。
- Virtual Token Counter 的 cost function 在讲授中只给了「与输入输出长度相关」的定性描述，具体形式与常数界证明在后续工作论文里，需回原文核对。
- 讲者提到 S-LoRA 已与 vLLM / SGLang 做集成；两引擎当前版本的多 adapter 显存管理是否已完整吸收 unified memory 的设计，值得在选型前实测对齐。

## 原文金句（1-2句）

> 「我们并不是想为每一个人提供一个 dedicated GPU……这样是非常不经济的。」——讲者对多租户 serving 问题的定性（字幕口径）

> 「所有人都没有 get starvation，并且请求多的用户也相应地 receive 到了多一点的 service……系统的 throughput 能达到整个系统 maximum 的 capacity。」——讲者对 max-min 公平调度目标状态的描述（字幕口径）
