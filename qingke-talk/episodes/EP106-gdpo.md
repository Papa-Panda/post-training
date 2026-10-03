# EP106 — GDPO：解决 GRPO 在多奖励 RL 训练中的"优势崩溃"问题

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP106-gdpo.html

> "applying GRPO directly to the summed reward can cause different reward combinations to collapse into the same advantage values." —— GDPO 论文 §6（结论），一句话点出本期的核心诊断：多奖励场景下不稳定的元凶不在奖励设计，而在 advantage 的归一化方式。

## 元信息

- 期号：106（官网期号；本期存在编号冲突，见下文专节）
- 标题：GDPO：解决 GRPO 在多奖励 RL 训练中的"优势崩溃"问题
- 分享嘉宾：刘诗扬（Shih-Yang "Sean" Liu，香港科技大学博士候选人、NVIDIA 研究实习生；官网嘉宾介绍与论文第一作者一致，工作在 NVIDIA 实习期完成）
- 直播时间：2026-01-30 9:00–10:00（官网预告；以 episode-index 台账登记日期为准）
- BV：无（官网第 106 期在合集无对应视频；合集同号期 B站视频实为 Intern-S1，见编号冲突专节）
- 提炼日期：2026-10-02
- 相关论文：Shih-Yang Liu, Xin Dong, Ximing Lu 等，*GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization*，https://arxiv.org/abs/2601.05242（arXiv:2601.05242）
- 相关代码：https://github.com/NVlabs/GDPO （含 verl、TRL、Nemo-RL 三种实现）；项目页 https://nvlabs.github.io/GDPO/
- 官网预告：https://qingkeai.online/blog/GDPO-talk

> ⚠️ 提炼方式说明：本期无视频、无字幕（官网有预告，B站合集无对应视频），本纪要基于对应论文还原，非逐字稿。讲授提纲以官网预告为准（四段：多奖励 RL 训练挑战与传统 GRPO 的局限性、通过逐奖励解耦归一化的 GDPO、GDPO 的性能验证与多目标 RL 讨论、AMA）；方法、公式与实验数字全部以 GDPO 论文（arXiv:2601.05242）为准，未取自讲授口播。AMA 环节未覆盖；若后续取得可验证字幕或回放，应以其为准修订。

## 编号冲突说明（106 期）

官网枚举的第 106 期是本场 GDPO（讲者刘诗扬）；但合集中该期号对应的 B站视频实际内容是 Intern-S1（上海 AI Lab 科学多模态基础模型），其纪要见 EP106-intern-s1.md。按「以视频实际内容为准」的原则，B站实录归 Intern-S1，本篇为官网 GDPO 的论文还原，两篇互指、编号同为 106。

## 一句话总结

当 RL 训练同时优化正确性、格式、长度、bug 率等多个奖励时，社区的默认做法是把奖励求和后再做 GRPO 的组内归一化。GDPO 论文证明这一步会把不同的奖励组合压成相同的 advantage（advantage collapse），训练信号分辨率下降，轻则收敛变差、重则早期崩溃。GDPO 的修法极简：先对每个奖励分别做组内归一化、再聚合，最后加一层 batch 级归一化稳定数值尺度——在工具调用、数学推理、代码推理三类多奖励任务上一致优于 GRPO，且可作为 GRPO 的 drop-in 替代（verl/TRL 改几行代码）。

## 核心

### 背景/问题：不稳定的元凶是归一化，不是奖励冲突（论文 §1、§2）

多奖励 RL 的常见叙事把训练不稳归因于奖励设计或目标冲突。论文提出另一诊断：GRPO 的 advantage 计算本身在多奖励下有结构性缺陷。标准做法先把 $n$ 个奖励求和成 $r_{\mathrm{sum}}^{(i,j)}$ ，再组内标准化：

$$A_{\mathrm{sum}}^{(i,j)}=\frac{r_{\mathrm{sum}}^{(i,j)}-\mathrm{mean}\{r_{\mathrm{sum}}^{(i,1)},\ldots,r_{\mathrm{sum}}^{(i,G)}\}}{\mathrm{std}\{r_{\mathrm{sum}}^{(i,1)},\ldots,r_{\mathrm{sum}}^{(i,G)}\}}$$

论文用一个最小例子把问题钉死（Figure 2）：每题 2 条 rollout、2 个二值奖励，组内总奖励组合（不计顺序）有 6 种，但 GRPO 标准化后只有 2 组不同的 advantage——组合 (0,1) 、(0,2) 、(1,2) 全部得到 $(-0.7071,\,0.7071)$ ，(0,0) 、(1,1) 、(2,2) 全部得到 (0,0) 。直觉上总量 2 的 rollout 同时满足两个奖励，理应比只满足一个的 rollout 拿到更强的相对优势，但 GRPO 给不了这个区分。实际后果不是抽象的：论文 Figure 5 里 GRPO 训练数学任务约 400 步后 correctness 分数开始下滑，呈现部分崩溃。

附带检验了一个「看似能治」的改动：Dr.GRPO 式的去掉 std 归一化。组合层面它确实多分出一点不同 advantage（(0,1) 与 (0,2) 变得可分），但 Figure 3 显示 rollout 数或奖励数一多，增益很有限；实证上更糟——工具调用任务中 GRPO w/o std 的 format 奖励完全学不起来，BFCL-v3 上 Correct Format 为 0%（论文 Table 2）。

### 方法/设计：逐奖励解耦归一化 + batch 归一化（论文 §3）

GDPO 只改 advantage 的构造顺序。对第 $k$ 个奖励先独立做组内标准化：

$$A_{k}^{(i,j)}=\frac{r_{k}^{(i,j)}-\mathrm{mean}\{r_{k}^{(i,1)},\ldots,r_{k}^{(i,G)}\}}{\mathrm{std}\{r_{k}^{(i,1)},\ldots,r_{k}^{(i,G)}\}}$$

然后带权重聚合 $A_{\mathrm{sum}}^{(i,j)}=w_{1}A_{1}^{(i,j)}+\cdots+w_{n}A_{n}^{(i,j)}$ （等权时 $w_{k}=1$ ），最后对全 batch 做一次标准化，使最终 advantage 的数值尺度不随奖励个数增加而膨胀。在同一玩具例子下， `(0,1)` 与 `(0,2)` 分别得到 `(-0.7071, 0.7071)` 与 `(-1.4142, 1.4142)` ——信号分辨率保住了。论文还数了「不同 advantage 组的数量」随 rollout 数与奖励数的增长（Figure 3）：GDPO 始终显著多于 GRPO 与 GRPO w/o std，且差距随规模放大。Appendix A 口径：去掉 batch 归一化偶尔会导致收敛失败，所以这一层不是装饰。

**优先级怎么表达**（论文 §3.2，这是讲授提纲里「多目标 RL 讨论」的实质内容）：当各奖励难度悬殊时，调权重不可靠——模型总会先去抠容易的奖励，权重差要大到足以补偿难度差才有效（§4.2.1 实证）。更可靠的做法是**条件奖励**：把容易奖励的发放条件绑在难奖励达标之上， $r_{k}$ 仅当 $r_{l}\ge t$ 时才发放。论文在数学任务里把 length 奖励改成「答对且不超长才给」（记作 $\tilde{\mathcal{R}}_{\mathrm{length}}$ ），效果见下文。解决完难度支配问题后，权重微调才重新变得可预测（Figure 6）。

### 实验/实战：三类任务一致优于 GRPO（论文 §4）

- **工具调用**（正确性 + 格式两奖励，Qwen2.5-Instruct 1.5B/3B，ToolRL 设置，5 次运行平均，论文 Table 1）：1.5B 上 GDPO 平均准确率 32.81% vs GRPO 30.18% ，Correct Format 80.66% vs 76.33% ；3B 上 40.87% vs 39.20%。训练曲线上 GDPO 两个奖励都收敛到更高点。
- **数学推理**（准确率 + 长度约束两奖励，DeepScaleR 4 万题、500 步，论文 Table 3 与 Figure 5）：DeepSeek-R1-1.5B 上 GDPO 在 MATH/AIME/Olympiad 分别比 GRPO 高 2.6/6.7/2.3 个百分点（86.2/29.4/46.6 vs 83.6/23.1/44.3 ——AIME 提升即官网预告所称的「高达 6.3%」量级），且超长率全面更低；GRPO 在约 400 步后 correctness 下滑、batch 最大长度反弹，GDPO 持续改进。Qwen3-4B-Instruct 上 AIME 高 2.3 个百分点。
- **优先级实验**（论文 §4.2.1、Table 4、Figure 6）：降低 length 权重对超长率几乎没影响（AIME 上 GRPO/GDPO 仅动 0.2–1.3 个百分点量级），印证「权重调不动难度支配」。改用条件奖励后，GDPO 在 AIME 上 57.7% vs 同配置 GRPO 的 53.3%，且超长率增幅小得多（12.3% vs 29.2%）。
- **代码推理**（通过率 + 条件长度 + bug 率三奖励，DeepSeek-R1-7B，Eurus-2-RL 2.4 万题，论文 Table 5）：两奖励配置下 GDPO 三个 benchmark 的通过率全面高于 GRPO-2obj（Codeforces 71.2% vs 68.1%）；三奖励配置下 GDPO 与 GRPO-3obj 通过率相近，但 bug 率与超长率明显更低（Codeforces bug 1.8% vs 2.5%，Apps bug 18.8% vs 20.3%），说明奖励数增加时 GDPO 的优势保持。

### 结论/观点（区分事实与作者主张）

事实层（论文实验口径）：在工具调用、数学、代码三类多奖励设置与 1.5B–7B/4B 多个底座上，GDPO 在最终性能与训练稳定性上一致优于 GRPO，且优势不随奖励个数增加而消失。作者主张（观点层，论文 §6 与官网预告一致）：多目标 RL 的核心不是继续发明新奖励函数，而是先保证训练信号的清晰度——GRPO 不应该是不加检验的默认算法；「sum-then-normalize」这一习惯性步骤本身就是需要被逐项审计的对象。

## 关键数字

数字均为论文（arXiv:2601.05242）口径。

| 指标 | GRPO | GDPO | 来源（arXiv:2601.05242） |
|---|---|---|---|
| 工具调用 Avg Acc / Correct Format（Qwen2.5-1.5B） | 30.18% / 76.33% | 32.81% / 80.66% | Table 1 |
| 工具调用 Avg Acc（Qwen2.5-3B） | 39.20% | 40.87% | Table 1 |
| GRPO w/o std 的 Correct Format（1.5B） | 0%（格式奖励完全没学起来） | 80.66%（GDPO） | Table 2 及 §4.1.1 |
| 数学 AIME / MATH Acc（DeepSeek-R1-1.5B） | 23.1% / 83.6% | 29.4% / 86.2% | Table 3 |
| 数学 AIME Acc（Qwen3-4B-Instruct） | 54.6% | 56.9%（论文称 +2.3%） | Table 3 及 §1 |
| 条件奖励下 AIME Acc（DeepSeek-R1-7B） | 53.3%（超长率 29.2%） | 57.7%（超长率 12.3%） | Table 4 |
| 代码 Codeforces Pass / Bug（三奖励） | 69.5% / 2.5% | 69.4% / 1.8% | Table 5 |
| GRPO 部分崩溃起点 | 数学任务约 400 步后 correctness 下滑 | GDPO 持续改进 | Figure 5 及 §4.2 |

## 可迁移

- 对 coding data / RL infra 的可试图：任何「多奖励 + GRPO」的训练（正确性、格式、长度、风格、安全等组合）都可以先检查 advantage 构造顺序——逐奖励归一化 + batch 归一化是几行代码的改动（官方已有 verl/TRL/Nemo-RL 实现），且不改变策略目标本身，适合在现有 GRPO 管线里做 A/B。
- 奖励优先级的做法可直接借鉴：难度悬殊的目标不要靠权重硬调，先把易奖励的发放条件绑到难奖励达标上（如「答对才计长度分」），再用权重做细粒度调整——这与代码任务里常见的「先过测试再谈简洁」直觉一致，论文给了受控证据。
- Infra/评测视角：多奖励训练的监控不应只看总奖励曲线。论文的做法是分开画每个奖励的曲线与 batch 最大长度这类极端量——GRPO 的崩溃在总曲线上出现前，correctness 子曲线与最大长度已先报警。

## 疑问 / 下一步

- GDPO 保住了 advantage 分辨率，但权重与条件阈值仍需按任务设计：条件奖励的阈值 $t$ 在奖励是连续量（如部分通过率）时怎么取、会不会引入新的不连续性，论文未给系统做法，是把 GDPO 搬到代码 RL（通过率本身是连续奖励）前最想核实的一点。
- 想看 AMA 里讲者对「奖励数继续增加（如十个以上）」时 batch 归一化是否仍够用、以及与 GSPO/DAPO 等其他 GRPO 变体叠加时的相互作用——此为讲授层面信息，本期无字幕未覆盖。

## 原文金句（1-2句）

> "Our analysis shows that applying GRPO directly to the summed reward can cause different reward combinations to collapse into the same advantage values. This collapse eliminates important distinctions across reward dimensions, produces inaccurate policy updates and weaker optimization performance, and can in many cases lead to early training failure." —— GDPO 论文 §6
>
> 「该工作指出，多目标 RL 的核心在于保持训练信号清晰度。」—— 官网预告对本期的定位（https://qingkeai.online/blog/GDPO-talk）

## 未确证项

1. 本期无字幕、无视频可看，本文全部内容基于论文 arXiv:2601.05242 与官网预告；讲授顺序与 AMA 未覆盖，与论文的差异无从比对。
2. 台账 BV 冲突：episode-index 记官网 EP106 为 official-only；合集对应期号的 B站视频（BV1FY6sBuECe）实际内容为 Intern-S1（见 EP106-intern-s1.md），GDPO 一期无 B站视频。
3. 官网预告页内日期表述自相矛盾（正文写「1月30日（周五）上午9点」、直播时间栏写「1月30日(周二)9:00 - 10:00」），本文按 episode-index 台账登记的 2026-01-30 著录，星期与实际开讲细节未经视频核实。
4. 论文代码仓 README 标注 [ICML2026]，会议收录状态以代码仓口径记录，未从会议官网二次核实。
