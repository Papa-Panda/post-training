# EP149 — LOPD：不再手动设计 OPSD 特权信息

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP149-lopd.html

> "experience representations should be optimized end-to-end for the policies they are meant to improve." —— LOPD 论文 §5（结论），点出本期的核心主张：经验的表示本身应该是可学习优化的对象，而不是由人预先规定格式。

## 元信息

- 期号：149（官网期；B站合集无对应视频）
- 标题：LOPD：不再手动设计 OPSD 特权信息（Latent On-Policy Self-Distillation）
- BV：无（官网有预告，B站合集无对应视频；episode-index 记为 official-only）
- 直播时间：2026-08-29（周六）10:00–11:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：张桂彬（新加坡国立大学计算学院博士研究生，导师颜水成教授；官网预告嘉宾介绍；论文共同第一作者 Guibin Zhang）
- 相关论文：Guibin Zhang, Jiayang Lyu, Ran Sun, Xinlei Yu, Haoyu Zhao, Qibing Ren, Shuicheng Yan，*Latent On-Policy Self-Distillation*，https://arxiv.org/abs/2608.13040（arXiv:2608.13040，2026-08-13 提交）
- 相关代码：https://github.com/bingreeky/lopd（另有 Hugging Face 上的 Qwen3-8B-LOPD 与 Olmo3-7B-LOPD 模型，论文页给出）
- 官网预告：https://qingkeai.online/blog/LOPD

> ⚠️ 提炼方式说明：本期无字幕（官网有预告、B站合集无对应视频），本纪要基于对应论文还原，非逐字稿。讲授提纲以官网预告为准（官网给出的六段提纲：从 OPD 到 OPSD 的「谁来决定什么经验值得被看见」、从设计特权信息到学习经验表示、Composer 压缩经验为 latent context、实验验证、经验即 substrate、AMA）；方法、公式与实验数字全部以 LOPD 论文（arXiv:2608.13040）为准，未取自讲授口播。现场讨论与 AMA 环节未覆盖；若后续取得可验证字幕或回放，应以其为准修订。

> 关联：本期与 EP125《重探 On-Policy Distillation》同属 OPD/OPSD 线——EP125 关注标准 sampled-token OPD 的失败模式与修复（教师 top- $k$ 局部支持匹配），LOPD 引用的正是这一线工作（论文引 Fu et al. 2026 与 Zhao et al. 2026 的 OPSD 设定，并采用 SDPO 式 top- $M$ logits 加尾桶的蒸馏实现）。EP154《Rethinking On-Policy Distillation》则是 OPD 何时成败的机制化分析。三者可对照读：125 修监督的「实施」，154 查监督的「机制」，149 把监督信号的输入表示本身变成可学习对象。

## 一句话总结

本期讲 LOPD（Latent On-Policy Self-Distillation）：它质疑 OPSD 里一个被默认的前提——教师要看的特权信息（答案、反馈、技能、成功轨迹等）为什么要由人手工规定格式。LOPD 改为从历史成功轨迹的经验库里检索相关经验，用一个可学习的 Composer 把经验压成连续 latent tokens 喂给同骨干的冻结教师，再对学生的 on-policy 轨迹做稠密 token 级蒸馏，并加 privileged-margin 约束防止教师塌向学生。论文结果（arXiv:2608.13040）：在 Qwen3-4B/8B 与 Olmo3-7B 三个骨干、工具调用与代码生成七个基准上全面超过 GRPO 与 OPSD/SDPO/Skill-SD 等代表性 OPSD 方法，且以不到 GRPO 与 Skill-SD 三成的 rollout 预算即反超二者。

## 核心

### 背景/问题：特权信息的「格式」成了 OPSD 的隐形瓶颈（论文 §1、§2）

按论文的梳理，OPD 的价值在于把外部经验转成可持久的策略改进：学生先采样自己的轨迹，教师在学生实际访问过的前缀上给 token 级稠密监督。OPSD 进一步去掉外部强教师——教师与学生是同一个模型，只是教师多看一份特权信息（privileged information），如验证过的推理轨迹、最终答案、环境反馈或成功的历史 rollout。于是 OPSD 的核心问题从「哪个教师够强」变成「该给自教师看什么特权上下文」。

论文指出既有做法的共同毛病：特权上下文都是设计者事先规定的离散工件（artifact）——oracle 答案、文本反思、成功轨迹、检索文档——它们先验地决定了教师能看到什么，因而把蒸馏能利用的信息限制在人选定的格式里。论文 §4.2 给了直接证据：同一种特权上下文在不同设置下效果会反转。例如 Qwen3-4B 上， SDPO 在 BFCL-v3 上从 vanilla 的 22.88 掉到 15.75、在 ACEBench 上从 50.6 掉到 38.0 ，而 OPSD 在 LiveCodeBench 上两骨干都低于 vanilla。结论（论文口径）：问题不在于「有没有特权信息」，而在于它的表示是否适配当前任务与学生的学习状态。

官网预告里讲者的问题提法更直白：当 answer、trajectory、reflection、skill 被设计得越来越精巧，为什么仍要由人来规定「什么经验值得被看见」？LOPD 的答案是：不再提新的手工变体，而是让特权上下文自身可端到端学习。

### 方法/设计：Composer 把经验压成 latent 特权上下文（论文 §3）

LOPD 的训练管线与标准 OPSD 一致（学生 rollout 自己的轨迹、教师在同一前缀上给分布、reverse KL 蒸馏），关键差异只在教师上下文的构造。形式上，传统 OPSD 的上下文是固定变换 $\bm{c}_{\mathrm{fix}}=\Phi_{\mathrm{fix}}(x,\mathcal{E})$ ，其中 $\Phi_{\mathrm{fix}}$ 是设计者规则；LOPD 把它换成可学习参数化 $\bm{c}_{\phi}=\Phi_{\phi}(x,\mathcal{E})$ ，输出 $K$ 个连续 latent token 拼接而成的上下文。教师多看 $\bm{c}_{\phi}$ ，学生仍只看状态 $\bm{s}_{t}$ ，部署时只保留学生——推理不需要经验库、检索或 Composer。

具体构造（论文 §3.2）：

- **经验库与检索**：离线只保留成功 rollout（存任务描述与紧凑的 action–result 轨迹，省去冗长 observation）；对当前任务用 dense retriever 按余弦相似度取 top- $J$ 条经验。论文实现刻意从这个「最小经验管线」起步（§3.2 首段），且只用轨迹这一种原料（Figure 1 注）。
- **Composer**：编码器把「任务 + 经验」对映射为隐藏状态，QFormer 式 cross-attention 压缩器（learned queries）把变长状态压成每条经验固定 $K$ 个 latent token；可训练参数只有编码器的 LoRA 与压缩器，编码骨干冻结。Composer 先在成功轨迹上冷启动（骨干冻结），再进入联合优化。
- **教师**：与学生同骨干、同初始化的冻结副本（权重不再更新），但前向激活仍在计算图里——蒸馏梯度穿过冻结网络回传到 latent 输入，于是 Composer 能学到「哪些经验特征能对当前学生轨迹产生有效监督」。

蒸馏目标（论文 §3.3）：对学生自己轨迹的每个被监督 token，用教师 top- $M$ 词表项加一个尾桶（tail bucket）的重整分布做 reverse KL：

$$\mathcal{L}_{\mathrm{distill}}(\theta,\phi)=\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{\tau\sim\pi_{\theta}^{S}(\cdot\mid x)}\left[\frac{\sum_{t=1}^{|\tau|}\sum_{n=1}^{L_{t}}\omega_{t,n}\,D_{\mathrm{KL}}(\tilde{\bm{p}}^{S}_{t,n}\|\tilde{\bm{p}}^{T}_{t,n})}{\sum_{t=1}^{|\tau|}\sum_{n=1}^{L_{t}}\omega_{t,n}}\right]$$

其中 $\tilde{\bm{p}}^{S}_{t,n}$ 与 $\tilde{\bm{p}}^{T}_{t,n}$ 是师生在同一动作前缀处的 top- $M$ 加尾桶分布， $\omega_{t,n}$ 是动作 token 掩码。reverse KL 让学生集中到教师支持的行为上。

**privileged-margin 约束**（论文 §3.3）：只优化蒸馏损失有个退化解——Composer 可以把教师分布往学生分布上靠来降低 KL，产出无信息上下文。LOPD 用轨迹级验证结果把教师优势钉住：定义单 token 特权量 $\delta_{t,n}(\phi)$ 为教师与学生在同一采样 token 上的 log-prob 差（学生侧 stop-gradient）：

$$\delta_{t,n}(\phi)=\log\pi^{T}_{\bar{\theta},\phi}(a_{t,n}\mid\bm{s}_{t},\bm{c}_{\phi},a_{t,<n})-\mathrm{sg}\left[\log\pi^{S}_{\theta}(a_{t,n}\mid\bm{s}_{t},a_{t,<n})\right]$$

再以结果符号 $A(\tau)=2r(\tau)-1$ 加权，要求加权特权量 $\Delta(\phi)$ 不低于阈值 $m>0$ （配对偶变量 $\beta$ 的约束优化，外加一个把 $\bm{c}_{\phi}$ 锚在冷启动 $\bm{c}_{\phi_{0}}$ 附近的正则项）：

$$\Delta(\phi)=\mathbb{E}_{\tau}\left[\frac{\sum_{t,n}\omega_{t,n}\,A(\tau)\,\delta_{t,n}(\phi)}{\sum_{t,n}\omega_{t,n}}\right]\geq m$$

直觉是：若 Composer 退化成无信息上下文，则 $\pi^{T}\to\pi^{S}$ 、 $\delta_{t,n}\to 0$ 、 $\Delta\to 0<m$ ，对偶惩罚自动激活，结构上排除平凡解；结果加权又让 Composer 偏向支持成功行为的证据。论文默认配置（§4.1）：每任务检索 $J=3$ 条经验、每条压 $K=32$ 个 latent token（共 96 个），top- $M$ 取 $M=20$ ，margin 阈值 $m=0.05$ ；工具调用 rollout 上限 30 个环境步，代码蒸馏用 16,384 token 的响应预算。

### 实验/实战：三骨干七基准全面第一，样本效率是最大卖点（论文 §4）

实验设置（论文 §4.1）：工具调用用 EnvScaler 派生的 2,349 个任务训练，代码用 DeepCoder 的 TACO 子集 7K 个可验证 Python 题；骨干为 Qwen3-4B、Qwen3-8B 与 Olmo3-7B；工具调用报 EnvScaler 成功率、BFCL-v3 与 ACEBench，代码报 LiveCodeBench v5/v6 与 EvalPlus（HumanEval+/MBPP+）的 pass@1；基线含 GRPO、SDFT 与 OPSD、SDPO、Skill-SD 三个手设计特权上下文的 OPSD 方法。

关键结果（均为论文表格口径，详见下节数字表）：

- **十个「骨干 × 基准」聚合比较里 LOPD 全部第一**，且在全部十个设置中都高于 vanilla（论文 §4.2）。工具调用上增益随骨干变大更明显：Qwen3-8B 的 EnvScaler 从最强基线 Skill-SD 的 60.2 提到 66.4 ，ACEBench 从 GRPO 的 58.0 提到 62.7 ；Qwen3-4B 的 EnvScaler 为 63.7（GRPO 61.8），BFCL-v3 平均 27.38（GRPO 25.25）。
- **样本效率**：在 EnvScaler 上，LOPD 仅 320 代即超过 0.61 平均奖励、576 代达 0.637 并在 1,600 代内维持在 0.63–0.64；同预算下 GRPO 与 Skill-SD 在 1,600 代末分别只有 0.611 与 0.588（论文 Figure 4）。摘要口径：LOPD 以不到 GRPO 与 Skill-SD 三成的 rollout 预算反超二者。
- **联合优化的必要性消融**（论文 §4.3、Figure 3）：冻结 Composer（只用冷启动）EnvScaler 为 0.573；联合优化但去掉 margin（ $m=0$ ）反而掉到 0.551——无约束蒸馏梯度会把教师拉向学生； $m=0.05$ 时 0.637、 $m=0.10$ 时 0.626 。即「可学习」本身不够，margin 是把上下文学习变成有效监督的必要件。
- **容量敏感性**（论文 §4.3、Figure 5）：每条经验的 latent token 数从 8/16（约 0.56）到 32（0.637）有明显阈值，64/128 无稳定增益；检索数从 1（0.605）到 3（0.637）提升，更多则平台化。故默认 $K=32$ 、 $J=3$ 是「最早达到最强」的设置。
- **行为内化**（论文 §4.3、Table 3）：蒸馏后的学生（推理时无任何特权上下文）继承了 latent 上下文诱导的交互模式——相比 vanilla，每步工具调用从 3.50 降到 1.11、环境步从 11.12 升到 17.04，首步响应短 37.5% ，重复调用从 8.89 降到 5.25 ，单位工具调用奖励从 0.038 升到 0.050。论文解读为从「一次发很多试探性调用」转向「更顺序化的计划执行」。
- **latent token 到底编码了什么**（论文 §4.4）：把 32 个 latent token 过冻结 LM head 投影，得到的是多语言与代码碎片的混合，任务条件改变表面投影但不产生可读的程序、也不复述检索到的解法——与「分布式 latent 表示」一致，但论文也承认可直接解码性并不能证明教师功能上用到的是哪部分信息。

### 结论/观点：把「经验表示」从设计问题改成优化问题（论文 §5；区分事实与作者主张）

论文的事实层结论是：手设计特权上下文的效果随任务与学习状态反转，没有一种固定格式普遍最优；可学习 latent 上下文在全部十个聚合比较中稳定占优。作者主张（观点层，论文 §5 与官网预告一致）是：自进化不应依赖越来越精巧的人工经验格式——原始轨迹就足以作最小底座（substrate），更丰富的经验库与检索器可以扩充可得经验，但「学什么」应由端到端优化决定；LOPD 更像一条设计原则的证据，而非又一个特权上下文配方。

对边界的交代（论文 §4.4、§5 口径）：当前经验来源只有轨迹一种、经验库只存成功 rollout，latent 内容不可读、机制未被直接证明；论文把「更丰富的仓库与检索器」列为后续扩展方向。

## 关键数字

数字均为论文（arXiv:2608.13040）口径；工具调用为 0–100 分制。

| 指标 | 基线 | 结果（LOPD） | 来源（arXiv:2608.13040） |
|---|---|---|---|
| EnvScaler 成功率（Qwen3-8B） | Skill-SD 60.2（最强基线）；GRPO 57.3；vanilla 49.2 | 66.4 | Table 1 |
| BFCL-v3 平均（Qwen3-8B） | GRPO 29.00；vanilla 28.38 | 29.88 | Table 1 |
| ACEBench 平均（Qwen3-8B） | GRPO 58.0 | 62.7 | Table 1 |
| EnvScaler 成功率（Qwen3-4B） | GRPO 61.8；vanilla 48.6 | 63.7 | Table 1 |
| LiveCodeBench 平均（Qwen3-4B） | GRPO 48.29；vanilla 45.61（OPSD 掉到 40.24） | 48.78 | Table 2 |
| EvalPlus 平均（Qwen3-4B） | SDFT 80.07 | 81.36 | Table 2 |
| LiveCodeBench 平均（Olmo3-7B） | GRPO 48.29 | 50.98（+2.69） | Table 2 |
| 样本效率（EnvScaler 平均奖励） | GRPO 1,600 代 0.611；Skill-SD 0.588 | 576 代 0.637；不到基线三成 rollout 预算反超 | Figure 4 及摘要 |
| 联合优化消融（EnvScaler） | 冻结 Composer 0.573； $m=0$ 时 0.551 | $m=0.05$ 时 0.637（ $m=0.10$ 时 0.626） | Figure 3（§4.3） |
| 行为内化 | vanilla 每步工具调用 3.50、环境步 11.12 | 1.11、17.04；重复调用 8.89 → 5.25 | Table 3 |
| 训练规模与默认配置 | — | 工具训练 2,349 任务；代码 7K 题； $J=3$ 、每条 $K=32$ 个 latent token、 $M=20$ 、 $m=0.05$ | §4.1 |

## 可迁移

- 对 coding data / RL infra 的可试图：把「喂给教师/奖励模型的经验格式」本身当成可学习对象。当前 coding-data 流水线里，哪些历史成功轨迹、哪种粒度的反馈进 prompt 多是人工规则；LOPD 给了一个可直接对照的实现——经验库（成功 rollout 存紧凑 action–result 轨迹）+ 检索 + QFormer 式压缩 + 端到端联合优化，且只动训练侧、部署零开销，适合在已有的 OPD 训练管线（如 EP125 的 top- $k$ 匹配实现）上做增量实验。
- privileged-margin 是一个便宜的「教师健康度」指标： $\Delta(\phi)$ 度量教师相对学生的、经结果加权的 log-prob 优势，只需一次 gather、不额外前向。任何自蒸馏/OPSD 训练都可以顺手监控它——它掉到 0 附近就说明教师监督在退化，这比事后看曲线更早暴露问题。
- Infra 视角：教师是冻结的同骨干副本、梯度只回传到 latent 输入，训练侧多一个检索 + 小压缩器模块，推理侧完全不带；当 rollout 预算是瓶颈（agentic 任务采样贵）时，「用更稠密的监督换更少的 rollout」是明确的成本结构——论文的量化口径是不到三成预算反超 GRPO。

## 疑问 / 下一步

- latent 特权上下文不可读（论文 §4.4 自认投影只是碎片），因此无法直接审计「教师到底在教什么」：当学生学坏时，怎么定位是经验库、检索还是压缩的问题？论文没有给可操作的分段诊断方法，这是把 LOPD 搬进生产 post-training 前最想深挖的一点。
- 想看 AMA 与讲授里讲者对「经验库只存成功轨迹」的回应：失败轨迹（尤其近失轨迹）在 margin 的结果加权里只起压制作用，是否值得入库存疑——此为讲授/AMA 层面信息，本期无字幕未覆盖，待有回放或字幕时补录。

## 原文金句（1-2句）

> "The question behind this work is not which new artifact should be appended to an OPSD teacher, but whether the teacher's privileged context can itself be learned from experience." —— LOPD 论文 §5 首句
>
> 「我们为何仍要替模型规定"什么经验值得被看见"？」—— 官网预告对本期的设问（https://qingkeai.online/blog/LOPD）
