# EP95 — 从 π_0 到 π_RL：面向流匹配 VLA 的强化学习后训练框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP95-pi-rl.html

> "applying large-scale RL to flow-based VLAs remains challenging due to intractable action log-likelihoods from iterative denoising." —— π_RL 论文 Abstract，本期要解决的核心问题

## 元信息

- 期号：95（官网期，B站合集有对应视频 BV1Tt2sBPEix）
- 标题：从 π_0 到 π_RL：面向流匹配 VLA 的强化学习后训练框架
- BV：BV1Tt2sBPEix（有视频；本期字幕经接口多次尝试均返回串台内容、无法验证，未取得可用字幕，故走论文还原路线）
- 直播时间：2025-12-06 10:00–11:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：陈康（北京大学计算机学院在读博士生；官网预告嘉宾介绍；论文一作 Kang Chen）
- 相关论文：Kang Chen, Zhihao Liu, Tonghe Zhang 等，*π_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models*，https://arxiv.org/abs/2510.25889（v3, 2026-01-29；代码集成于 RLinf 框架）
- 相关代码：https://github.com/RLinf/RLinf（π_RL 的实现与模型 checkpoint 随 RLinf 开源，论文 Abstract 与官网预告）
- 官网预告：https://qingkeai.online/blog/%CF%80_RL

> ⚠️ 提炼方式说明：本期 B站有视频，但字幕接口多次返回其他视频的串台字幕、无法验证，未能取得可用字幕。本纪要根据该期对应的公开材料还原——官网预告（含讲者与提纲）+ π_RL 论文原文（arXiv:2510.25889）。讲授提纲以官网预告为准，方法与实验数字以论文为准。现场演示与 AMA 环节未覆盖；若后续取得可验证字幕，应以字幕为准修订。

## 一句话总结

这期讲的是怎么对流匹配 VLA（ $\pi_0$ 、 $\pi_{0.5}$ ）做在线 RL：这类模型用迭代去噪生成动作，动作的 log-likelihood 不可解，标准策略梯度用不了。π_RL 给出两条路线——Flow-Noise 在去噪过程里加一个可学习噪声网络，把去噪本身建成离散时间 MDP 以精确计算联合似然；Flow-SDE 把采样 ODE 改写为等边际分布的 SDE，建成「去噪 × 环境交互」的双层 MDP。两者在 LIBERO、ManiSkill、MetaWorld 上都把少样本 SFT 模型大幅抬升（LIBERO 上 $\pi_0$ 从 57.6% 到 97.6%）。

另注意与同主题工作的关系：同期被广泛讨论的 SimpleVLA-RL 解决的是自回归型 VLA（OpenVLA-OFT，动作似然可由 softmax/高斯头直接得到）加 GRPO 的问题；π_RL 面对的是流匹配 VLA 的似然不可解问题，两者底座与技术路线不同，不要混为一谈（π_RL 论文 §2.2 亦将 SimpleVLA-RL 列为自回归路线的先行工作）。

## 核心

### 背景/问题：流匹配 VLA 的似然黑洞

按官网预告提纲，讲授分四块：流匹配 VLA 与 RL 难点、π_RL 框架（Flow-Noise / Flow-SDE）、 $\pi_0$ 与 $\pi_{0.5}$ 微调实践、AMA。论文 §1–§2 把难点说清：

- VLA 的主流训练仍是预训练 + SFT，专家轨迹采集昂贵且 SFT 容易过拟合演示；RL 本可让模型通过环境交互超越演示，但已有 VLA-RL 工作几乎都建立在自回归底座（OpenVLA、OpenVLA-OFT）上，动作似然可直接算。
- $\pi_0$ 、 $\pi_{0.5}$ 这类流匹配 VLA 通过迭代去噪生成动作 chunk：条件流匹配损失训练的是一个速度场 $\mathbf{v}_{\theta}$ ，推理时从高斯噪声出发做欧拉积分。速度场形式为：

$$\mathcal{L}_{\text{CFM}}=\mathbb{E}\left[\left\|\mathbf{v}_{\theta}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t})-\mathbf{u}(\mathbf{A}_{t}^{\tau}|\mathbf{A}_{t})\right\|_{2}^{2}\right]$$

这里 $\mathbf{A}_{t}^{\tau}$ 是时刻 $\tau$ 的带噪动作， $\mathbf{u}$ 是目标速度场。这套生成过程是确定性 ODE：既没有可用的动作 log-likelihood（用 Hutchinson 估计在少去噪步数下不准），ODE 本身也不产生探索所需的随机性（论文 §4 开头）。策略梯度公式里的 $\log\pi_{\theta}(a_{t}|s_{t})$ 无从下手。

### 方法/设计：两条把去噪变成 MDP 的路线

**Flow-Noise（受 ReinFlow 启发，论文 §4.1）**：在去噪每一步注入由神经网络参数化的可学习噪声，把一步转移建成各向同性高斯：

$$p(\mathbf{A}^{\tau+\delta}|\mathbf{A}^{\tau})\sim\mathcal{N}(\mu_{\tau},\Sigma_{\tau})$$

其中均值 $\mu_{\tau}$ 由原 ODE 的欧拉更新给出，方差由噪声网络 $\sigma_{\theta^{\prime}}$ 依当前动作与观测预测。这样整个去噪序列的联合 log 概率可以精确连乘得到，直接代入标准单层 MDP 的策略梯度。噪声网络与速度场联合训练，微调结束后丢弃，推理仍是确定性策略。

**Flow-SDE（受 Flow-GRPO 启发，论文 §4.2）**：利用概率流 ODE 与 SDE 的等价关系，把确定性采样改写为保持边际分布不变的 SDE，漂移项用速度场修正、扩散项提供探索噪声；再套 DPPO 式的双层 MDP——内层是去噪步、外层是环境步，奖励只在去噪完成并与环境交互时发放；由于去噪转移本身就是高斯，每一步的 log 概率直接可算。针对双层 MDP 轨迹过长的问题，采用混合 ODE-SDE 采样：每步只随机选一个去噪时刻做随机转移、其余保持确定性 ODE，缩短有效 horizon 加速训练（§4.2.3）。

有了精确似然之后，π_RL 用 PPO 做优化；论文在 LIBERO 上对比过 GRPO，结论是 PPO 在所有任务套件上一致优于 GRPO（§1）。

### 实验/实战（论文 §1、Abstract）

- **LIBERO**：少样本 SFT 的 $\pi_0$ 平均成功率从 57.6% 提升到 97.6%， $\pi_{0.5}$ 从 77.1% 提升到 98.3%。最亮眼的是 LIBERO-Long 上只用一条轨迹 SFT 的 $\pi_{0.5}$ ：从 43.9% 提升到 94.0%，反超用全部轨迹 SFT 的 92.4%——RL 把数据效率问题部分绕过去了。
- **ManiSkill**：在 320 个并行环境中训练，4352 种抓取-放置组合（16 类物体 × 17 种容器 × 16 个场景）上， $\pi_0$ 从 38.4% 提升到 78.8%， $\pi_{0.5}$ 从 40.1% 提升到 90.8%（v3 Abstract 口径）；附带的 SIMPLER 基准上 $\pi_0$ 从 67.2% 提升到 86.7%。
- **MetaWorld（MT50）**：50 个操作任务上 RL 后 $\pi_0$ 与 $\pi_{0.5}$ 成功率分别达 85.8% 与 70.7%，均超过基线 SmolVLA 的 68.2%（§1）。
- 论文还做了 RL 算法、critic 设计、噪声注入策略、MDP 建模与超参的系统消融（§1 贡献列表），定位是给后续流匹配 VLA 的 RL 工作提供经验基线；全部代码与 checkpoint 随 RLinf 开源。

注：ManiSkill 数字在论文不同版本间有修订（v1 口径 $\pi_0$ 为 41.6% → 85.7%），本纪要采用当前 arXiv v3 Abstract 口径并在此标注，引用时注意版本。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| LIBERO 平均成功率（ $\pi_0$ ，少样本 SFT） | 57.6% | 97.6% | 论文 §1、Abstract |
| LIBERO 平均成功率（ $\pi_{0.5}$ ） | 77.1% | 98.3% | 论文 §1、Abstract |
| LIBERO-Long 单轨迹 SFT 的 $\pi_{0.5}$ | 43.9%（全轨迹 SFT 为 92.4%） | 94.0% | 论文 §1 |
| ManiSkill 4352 组合成功率（ $\pi_0$ / $\pi_{0.5}$ ） | 38.4% / 40.1% | 78.8% / 90.8% | 论文 Abstract（v3） |
| MetaWorld MT50 成功率（ $\pi_0$ / $\pi_{0.5}$ ） | SmolVLA 68.2% | 85.8% / 70.7% | 论文 §1 |
| 优化算法对比 | GRPO | PPO 在所有任务套件上一致更优 | 论文 §1 |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **「似然不可解」有两种通用解法**：要么像 Flow-Noise 那样给生成过程加可学习噪声、把中间过程建成 MDP 拿联合似然；要么像 Flow-SDE 那样把确定性采样换成等边际的随机采样。凡是想对扩散/流式生成器（包括代码生成里的迭代精修）做策略梯度，都可以从这两族方法里选。
  2. **混合 ODE-SDE 采样是降成本的通用技巧**：只需在少数随机步上算似然、其余步走确定性路径，能把双层 MDP 的有效 horizon 压下来——对 rollout 昂贵的 RL 训练是直接的成本项。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. 320 个并行仿真环境做多任务 RL（4352 种组合）说明 VLA 的 RL 瓶颈在仿真吞吐而非单步算法；做具身 RL infra 时并行环境调度与 GPU 仿真利用率是一等公民。
  2. RL 能让单轨迹 SFT 模型反超全轨迹 SFT（43.9% → 94.0% vs 92.4%），提示数据管线预算分配可以变：少量高质量演示 + 在线 RL 可能比堆演示更划算，前提是仿真环境足够真实。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：Flow-Noise 的噪声网络在训练后被丢弃、推理回到确定性策略，那么训练时学到的探索分布与部署策略之间的一致性如何保证、丢弃噪声后性能掉多少，论文未单独报告，值得查 RLinf 代码与 checkpoint 实测。
- 现场内容不可还原：官网提纲第三部分「 $\pi_0$ 与 $\pi_{0.5}$ 微调实践」的工程细节（如并行环境配置、训练时长）无法从论文推知。
- 全部实验在仿真（LIBERO/ManiSkill/MetaWorld/SIMPLER）中完成，真机迁移效果论文未报告；与 SimpleVLA-RL 等自回归路线在同一底座上的直接对比也不存在，选型时注意。

## 原文金句（1-2句）

> "Flow-Noise models the denoising process as a discrete-time MDP with a learnable noise network for exact log-likelihood computation. Flow-SDE integrates denoising with agent-environment interaction, formulating a two-layer MDP that employs ODE-to-SDE conversion for efficient RL exploration." —— 论文 Abstract，两条技术路线的官方概括

> "On LIBERO-Long, π_RL boosts the performance of the π_0.5 one-trajectory SFT model from 43.9% to 94.0%, surpassing the 92.4% performance of the all-trajectories SFT model." —— 论文 §1，本期最能说明 RL 数据效率的数字
