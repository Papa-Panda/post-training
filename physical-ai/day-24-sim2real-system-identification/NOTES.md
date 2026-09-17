# Day 24 — 系统辨识 + Sim2Real 评估：SimOpt — Closing the Sim-to-Real Loop

## 元信息
- Title: Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience（SimOpt）
- Authors / Org: Yevgen Chebotar（USC）, Ankur Handa, Viktor Makoviychuk, Miles Macklin, Jan Issac, Nathan Ratliff, Dieter Fox（NVIDIA / Univ. of Copenhagen / Univ. of Washington；ICRA 2019）
- Link / arXiv: https://arxiv.org/abs/1810.05687（v1 2018-10-12，v4 2019-03-05；ICRA 2019 接收）
- 项目页（含实验视频）: https://sites.google.com/view/simopt
- Official code: 论文未附公开实现仓库（诚实标注）；核心算法是标准组件组合（PPO + REPS + 加权 MLE），可复写
- Date read: 2026-09-15
- Tags: [physical-ai, simopt, system-identification, domain-randomization, sim2real, reps, ppo, behavior-matching, real2sim]
- Thread: physical-ai
- Folder: day-24-sim2real-system-identification
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-24-sim2real-system-identification

## 一句话总结
Day21 的 DR 把随机化分布当**手工/启发式超参**调，SimOpt 把它当**可辨识的对象**：用少量真机 rollout 作为锚点，把"仿真参数分布 $p_\phi(\xi)$ "和"策略 $\pi_\theta$ "放进一个**闭环双循环**里交错优化——内层 PPO 在当前分布上训策略，外层 REPS 以"仿真轨迹 vs 真机轨迹的行为差异"为代价更新分布参数；在 swing-peg-in-hole 和开抽屉两个真机任务上，用 a few 真机轨迹就实现了可靠迁移，且迁移到**不同机器人**上仍然成立。

## 和之前工作的关系

- **接了哪条线**：Day19–24 "RL for Robotics / Sim2Real" 块的第五天，roadmap 标题 "System Identification + Sim2Real Evaluation — calibrate and gate transfer"。本篇是"calibrate"那一半的 canonical 答案：参数辨识不再是辨识"物理真值"，而是辨识"让行为对上"的分布。
- **补了哪个短板**：Day21 DR 回答了"世界做多宽"，但没回答"做多宽是对的"——Tobin 是手工范围、ADR 是熵驱动的无锚扩张（只知道越宽越难、成功了就继续扩）。SimOpt 补上**真机锚点**：分布的更新方向由真实 rollout 的行为残差决定，扩张/收缩都有证据支撑。
- **替代 / 分叉 / 改进**：不是替代 DR，而是 DR 的**闭环升级**。三分法更新为：① 无锚 DR（Tobin 手工 / Peng 2017 / ADR 2019-05）→ ② 有锚分布辨识（SimOpt 2019-03，真机行为匹配）→ ③ 零真机交互的部署端辨识（Day22 RMA）。SimOpt 恰好卡在中间：真机交互极少（few-shot），但不为零。
- **对之前 Day X 的直接对比**：
  - **vs Day21（ADR）**：ADR 的分布更新信号是"策略在当前分布下还能不能成功"（成功率驱动的熵单调扩张）；SimOpt 的更新信号是"仿真轨迹和真机轨迹像不像"（行为差异驱动）。ADR 的世界只会变宽，SimOpt 的世界可以向真值**收缩**——这是辨识 vs 课程的本质区别。
  - **vs Day22（RMA）**：RMA 是**部署时**辨识（权重 frozen，辨识 $\hat z_t$ 在线条件变量）；SimOpt 是**训练时**辨识（用真机 rollout 离线更新 $p_\phi$ ，策略权重重训）。两者互补：SimOpt 负责把训练分布校准到真机附近，RMA 负责部署时残差的在线补偿。
  - **vs Day19（PPO）**：SimOpt 的内层策略优化就是 PPO——Day19 建立的 actor-critic 基线在这里是可替换的执行器。SimOpt 的贡献全部在**外层**：把"仿真器参数分布"变成一个 episodic RL 问题来解。
  - **vs Day23（Residual RL）**：residual 的先验是**控制器**（ $u=\pi_H+\pi_\theta$ ），SimOpt 的先验是**校准后的仿真器**。两者可叠加：先用 SimOpt 把仿真分布校准，再用残差头在真机上学最后的接触修正。
  - **vs Day08（Humanoid-Gym）**：Day08 的 MuJoCo sim2sim gate 是 transfer 的**评估门**；SimOpt 是**校准器**。本篇 NOTES 末尾把两者拼成完整的 release gate 流程。

## 为什么今天读它

Roadmap 给 Day24 的任务是"用参数辨识、sim2sim、hardware-in-the-loop 和分桶指标把 sim2real 从口号变成 release gate"。这四个词里，SimOpt 是第一块拼图最干净的 canonical 实现：它把"系统辨识"重新定义为**分布辨识**（identifying a distribution, not a point estimate），并且第一次证明了"真机行为匹配"这个目标函数足以驱动可靠迁移。对 Jun 的路线：① 这是"真机数据极少时怎么办"的标准答案——RLPD（Day20）要真机在线训、RMA（Day22）要零真机、SimOpt 要"几条"真机轨迹，三者正好覆盖数据预算谱；② "把仿真器参数更新写成 RL 问题"是通用设计模式——REPS 外层 + PPO 内层这种双循环结构，在超参搜索、课程学习里反复出现；③ Day25–30 进入 generalist / eval / safety 之前，先把"怎么证明能 transfer"的方法论钉死。

## 今天的 3 问
1. REPS 外层更新里，"采样 $\xi$ "→"算权重 $w_i$ "→"更新 $\phi$ "三步的精确公式是什么？KL 约束 $\mathrm{KL}(p\Vert p_{\text{old}})\le\epsilon$ 为什么必不可少（没有它分布会塌缩到哪里）？
2. 为什么 SimOpt 匹配的是**行为**（轨迹观测差异）而不是直接辨识物理参数（mass / friction 的真值）？接触任务里"多个 $\xi$ 组合产生相同行为"的参数退化（non-identifiability）会带来什么后果？
3. 外层辨识用的轨迹是**当前策略** $\pi$ 跑出来的行为：策略更新后，之前匹配好的分布还成立吗？为什么 SimOpt 必须把辨识和策略训练**交错**进行，而不能"先辨识完仿真器、再一口气训策略"？

相关讨论（Gemini 网页版，2026-09-16）：https://gemini.google.com/app/290dacd4af1b7a57

## 核心
1. **Motivation**: DR 之后社区留下一个尴尬问题：随机化范围全靠手工调——调宽了策略学不动（Day21 ADR 论文里明确记录了过宽分布导致训练崩溃），调窄了真机不在支撑集里。更深层的问题是**物理参数不可辨识**：接触、摩擦、柔顺性这些量，真机上根本测不准，辨识一个"真值"既不现实也没必要。SimOpt 的赌注：**我们不需要仿真器"对"，只需要"行为对得上"**——在分布层面匹配行为，点估计的不可辨识问题就绕过去了。
2. **System / Method**: 双循环闭环（见数学视角 §1–3）：
   - **内层（policy）**：在当前分布 $p_\phi(\xi)$ 下，用 PPO 训练策略 $\pi_\theta$ （ $\xi$ 每 episode 重采样——这就是标准 DR 训练）。
   - **外层（simulator）**：固定当前 $\pi_\theta$ ，在仿真里采样一批参数 $\{\xi_i\}$ 各跑轨迹，与少量真机参考轨迹算行为差异 $D_i$ ；把 $-D_i$ 当回报，用 REPS 更新 $p_\phi$ （KL 约束防止分布跳变）。
   - 交错执行：分布变了 → 策略重训/继续训 → 新策略跑出新行为 → 再更新分布。**辨识和策略是耦合的**，因为"行为像不像"这个判据本身依赖策略。
3. **Training / Data Details**: Sim 数据来自 FleX（NVIDIA GPU 物理引擎，擅长刚体+绳索/布料；swing-peg-in-hole 里的绳子就是用它仿的）；Real 数据是**极少量的真机 rollout**（论文强调 a few real world rollouts，不是 RLPD 式的大量在线交互）；Reward / verifiable signal 有两层——内层是任务回报（PPO 正常优化），外层是**轨迹差异的负值**（行为匹配代价，见数学视角 §4）。Sim2Real 的做法不是"训完直接搬"，而是"搬之前先用真机行为把仿真分布拽向真机"。
4. **Key Tricks**: ① **分布而非点估计**：输出是 $p_\phi$ （论文用高斯族）不是单个 $\xi^*$ ——保留不确定性，策略继续在分布上做 DR 训练，鲁棒性和校准兼得；② **REPS 的 KL 约束即信任域**：外层更新本质是"分布空间的 PPO-clip"（和 Day19 的 ratio-clip 同构），防止一次真机观测把分布拽飞——真机 rollout 极少时这是保命项；③ **差异度量用观测空间加权 L1+L2**：不比较隐藏物理参数，只比较策略能看到/能影响的观测轨迹——"行为匹配"的操作化定义，也是绕开不可辨识的工程关键。
5. **Results**: 两个真机任务（swing-peg-in-hole、开抽屉）在**不同机器人**上可靠迁移（cross-robot transfer——说明辨识到的是任务相关动力学，不是某台机器的过拟合指纹）；对比手工调参的 DR 基线，SimOpt 的迁移成功率显著更高且调参人力归零。论文同时展示了"分布参数随外层迭代向真机行为收敛"的曲线——这是"辨识真的在发生"的直接证据。

## 数学视角

### 0. 符号与维度（先摆清楚）
- $\xi\in\mathbb{R}^d$ ：仿真器物理参数向量（如摩擦系数、质量、阻尼、柔顺性；论文任务里 $d\approx O(10)$ ）。
- $p_\phi(\xi)$ ：参数分布，论文取高斯族 $\phi=(\mu,\Sigma)$ ； $\phi$ 是**外层优化的变量**。
- $\tau=(o_0,a_0,o_1,\dots,o_T)$ ：一条轨迹， $o_t\in\mathbb{R}^{n_o}$ 为观测（如末端位姿、关节角、绳摆角）， $T$ 为 horizon。
- $\pi_\theta(a\mid o)$ ：内层 PPO 策略； $D(\tau^{\text{sim}},\{\tau^{\text{real}}_m\})$ ：行为差异代价， $m=1..M$ 为少量真机参考轨迹。

### 1. 外层目标：分布辨识 = 最小化期望行为差异

$$\min_{\phi}\; J_{\text{ID}}(\phi)=\mathbb{E}_{\xi\sim p_\phi}\Big[\mathbb{E}_{\tau\sim p_{\pi_\theta}(\cdot\mid\xi)}\big[D(\tau;\{\tau^{\text{real}}_m\})\big]\Big].$$

直觉：**找一个仿真参数分布，使得当前策略在它下面跑出的行为，平均而言最像真机**。注意内层期望依赖 $\pi_\theta$ ——策略变了，"像不像"的标准就变了，这就是必须交错优化的数学原因（3 问之 3）。

### 2. REPS 更新：把"调仿真器"写成 episodic RL
把 $\xi$ 看成"动作"， $R(\xi)=-D(\xi)$ 看成"回报"，外层就是一个单步 RL 问题。REPS（Peters et al. 2010）解：

$$\max_{p}\int p(\xi)R(\xi)\,d\xi \quad \text{s.t.}\quad \mathrm{KL}(p\Vert p_{\text{old}})\le\epsilon,\ \int p=1.$$

拉格朗日解得闭式（ $\eta>0$ 为 KL 约束的对偶变量）：

$$p^*(\xi)\propto p_{\text{old}}(\xi)\,\exp\!\big(R(\xi)/\eta\big),\qquad w_i\propto\exp\!\big(R(\xi_i)/\eta\big).$$

然后 $\phi_{\text{new}}=\arg\max_\phi\sum_i w_i\log p_\phi(\xi_i)$ （加权 MLE；高斯族下就是加权均值/协方差）。 $\eta$ 由对偶问题定： $\min_{\eta>0}\,\eta\epsilon+\eta\log\mathbb{E}_{p_{\text{old}}}[\exp(R/\eta)]$ 。
- **直觉**：行为越像真机（ $R$ 越大）的 $\xi_i$ 权重 $w_i$ 越大，分布向它们靠拢；KL 约束 $\epsilon$ 是"分布空间的信任域"——和 Day19 PPO 的 ratio-clip 同构，没有它，一次带噪声的真机观测就能把分布拽到单点塌缩（3 问之 1）。
- **时间尺度**：内层 PPO 是快循环（成千上万仿真 episode），外层 REPS 是慢循环（每次只用几条真机轨迹更新一次分布）。

### 3. 行为差异 $D$ ：观测空间加权 L1+L2

$$D(\xi)=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T}\Big(w_1\big\Vert o^{\text{sim}}_t(\xi)-o^{\text{real}}_{m,t}\big\Vert_1+w_2\big\Vert o^{\text{sim}}_t(\xi)-o^{\text{real}}_{m,t}\big\Vert_2^2\Big).$$

只比较**观测**轨迹，不碰隐藏物理量——这是"行为匹配"绕开参数不可辨识（3 问之 2）的操作化：接触任务里摩擦 $\mu$ 和接触阻尼 $c$ 的多种组合能产生几乎相同的末端轨迹，辨识真值是病态问题；但只要观测行为对得上，策略 transfer 就成立。**代价**：匹配的是"在当前策略下"的行为——换一个探索区域完全不同的策略，之前匹配好的分布可能失效（外层与内层耦合的另一面）。

### 4. 统一框架：Day19–24 的"先验轴"至此闭环
四条路线的外层数学其实是同一个问题——**怎么选训练分布 $p(\xi)$ **：
- Day21 ADR： $p_\phi$ 按"成功率阈值+熵增"自动扩张（无真机锚点， $R$ =成功示性函数）；
- Day24 SimOpt： $p_\phi$ 按"真机行为残差"收缩/移动（有真机锚点， $R=-D$ ）；
- Day22 RMA：干脆不碰 $p(\xi)$ ，把辨识搬到部署时（在线估计 $\hat z_t$ ）；
- Day20 RLPD：连仿真都不要了，直接用真机+离线数据（ $p$ =真实世界本身）。
SimOpt 的位置：**真机预算在"几条轨迹"量级时的最优解**——比 ADR 多用了真机信息，比 RLPD/RMA 少两个数量级。

### 5. 数学没覆盖的部分（诚实标注）
- **观测匹配 ≠ 动力学对齐**： $D$ 只在观测空间定义，仿真器内部状态可以错得离谱——只要策略看不到/用不到，就不影响 transfer；但策略一更新、探索到新区域，错的内部动力学就会暴露（这就是交错优化不能停的原因）。
- **接触不连续**： $R(\xi)$ 对 $\xi$ 不可微（接触事件是离散的），所以外层只能用 REPS 这类**无梯度**方法——采样效率天然低于有梯度辨识，这是 SimOpt 用"分布+加权"而不用"梯度下降找 $\xi^*$ "的深层原因。
- **真机轨迹的噪声**： $M$ 很小（few）， $D$ 的估计方差大；KL 约束 $\epsilon$ 在这里同时扮演了"防过拟合到噪声真机样本"的正则项。

## 可迁移 / Transfer

- **方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？** 论文做了 cross-robot 迁移（在 A 机器人上辨识+训练，部署到 B 机器人）——transfer 成立，说明贡献主要来自**框架**（行为匹配的分布辨识流程），而非某个过拟合到特定机器的策略。这和 Day23 residual 的"先验免费"结论呼应：SimOpt 让"仿真器"这个先验本身变得可校准。
- **对你 Infra → Post-training → Physical AI 迁移的 1-2 个直接启发**：① "真机数据预算"是可以**分档设计**的：0 条（RMA/ADR）→ 几条（SimOpt）→ 大量在线（RLPD）→ 全真机（residual 8000 步）。做任何真机项目先问"我的预算在第几档"，再选路线——这是 Day19–24 整块内容最实用的选型表。② REPS 外层是"用 RL 调仿真器"的通用模式：内层任何 policy optimizer（PPO/SAC）都可替换，外层任何带约束的分布优化器都可替换——模块化程度和 Day13 Octo 的 readout 哲学一致。
- **Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性**：SimOpt 把"调随机化范围"这个最吃人力、最低复现性的环节变成了**自动化闭环**——真机 rollout 是唯一的外部输入，且只需要 few 条。工程上这意味着 sim2real 调参可以进 CI：每次改仿真器/改任务，先跑 SimOpt 外层对齐，再跑策略回归测试。成本模型也清晰：真机成本 = $M \times$ （单条 rollout 时间），和策略训练的仿真 GPU 成本解耦。

## 附：Sim2Real Release Gate（本篇校准器 + Day08 评估门的完整流程）

SimOpt 只解决了"calibrate"；roadmap 要求的 gate 需要四件套拼齐：

1. **Calibrate（本篇）**：SimOpt 式行为匹配，把 $p_\phi(\xi)$ 拽向真机；输出校准后的分布 + 收敛曲线（分布参数是否稳定是第一道健康指标）。
2. **Sim2Sim gate（Day08 做法）**：在**独立的第二个仿真器**（如 Isaac Sim 训 → MuJoCo 测）上跑策略，成功率跌幅超过阈值（如 >10–15%）不准上真机——防止过拟合到某个物理引擎的数值特性。
3. **Hardware-in-the-loop / few-shot 真机探针**：上真机前先跑少量低风险探针轨迹（不做完整任务），只验证 $D$ 指标在容限内；SimOpt 的 $D$ 公式可以直接复用为探针判据。
4. **分桶指标（per-bucket metrics）**：成功率必须按任务阶段（抓取/搬运/放置）、环境条件（光照/桌面材质/初始位姿分区）分桶报告——平均成功率 90% 可能藏着某个桶 40%，release 判据是**最差桶**，不是平均。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：REPS 的 $\epsilon$ （KL 上限）在论文里是手工超参——它和"真机轨迹条数 $M$ "之间有没有自适应的关系？ $M$ 越小是否应该 $\epsilon$ 越小（更保守）？这直接决定 few-shot 极限。
- 如果要复现 / 小规模试，第一个实验做什么？在 Isaac Lab 里搭一个"已知真值"的 toy 环境（自己设定 $\xi_{\text{true}}$ 生成伪真机轨迹），跑通"内层 PPO + 外层 REPS"双循环，验证 $p_\phi$ 的均值是否向 $\xi_{\text{true}}$ 收敛——先验证辨识回路本身，再碰真机。

## 原文金句 (1-2句)
> "Rather than manually tuning the randomization of simulations, we adapt the simulation parameter distribution using a few real world roll-outs interleaved with policy training."
> "We are able to change the distribution of simulations to improve the policy transfer by matching the policy behavior in simulation and the real world."

## 今晚产出
- 按模板补齐 System / Training / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节（含 vs Day19/20/21/22/23/08 六组对比）
- 新增「数学视角」§0–§5：符号维度表 → 外层 REPS 目标与闭式解 → 行为差异 $D$ 公式 → Day19–24 先验轴统一框架 → 数学未覆盖部分
- 新增「附：Sim2Real Release Gate」：calibrate + sim2sim + HIL 探针 + 分桶指标四件套

## 连接
- 上一篇: day-23-2019-residual-rl（先验=经典控制器，真机小量在线学）
- 下一篇预告: day-25-2022-gato（离开 RL for Robotics 块，进入 Day25–30 "Physical AGI / Eval / Safety"：统一 token 序列的 generalist 范式）


## 问答补充
（本日 side chat 若有用户提问，在此归档；推送卡片类消息跳过。）

### 2026-09-15：Day24 NOTES 的 `$$...$$` 块在 GitHub 上仍显示裸 LaTeX

**问题原文**（用户，2026-09-15 16:17 PDT）："[Day24 NOTES 链接] 数学符号又没修理好 1/修一下 2/你的prompt是不是出问题了？修理下 之后 查看下其他类似的scheduledjob也修理"

**核心答案**（side chat 内的诊断 + 修复）：第一轮只修了行内公式 `$ x $` 的内侧空格，没有修到根上。真凶是 Day24 的 `$$...$$` 块与相邻文字挤在同一个段落里：GitHub 的 markdown 层根本不把它识别为公式，块内的 `_` 被解析成 `<em>` 斜体，整块显示成裸 LaTeX。之前猜的"标题后必须空行"也被证伪——对照检查了 Day20 和 ai-data Day13 的单行 `$$...$$` 块渲染正常，说明根因是"块公式必须独占段落"。验证方法：curl GitHub blob 页 HTML，先确认 currentOid 与推送的 commit 一致（排除缓存），再从 GitHub 自己的渲染标记（`<math-renderer class="js-inline-math">`）看它到底认了哪些块——认出的块带 math-renderer 标记，没认出的直接看源码 `<p>`。修复：所有 `$$` 块前后各加空行独占段落；`tools/check_repo.py` gate 强制三条规则（行内公式外侧空格、`$` 内侧紧贴、`$$` 块独占段落）；四个会写公式的 cron prompt 全部补上新规则并要求提交前跑 lint 门。commit `80df815`（另有 `f896397` 消掉 check_repo.py 的 SyntaxWarning），已 push。

**关联**：这是 AGENTS.md 里那条"块公式必须独占段落"规则的实战来源——旧的"标题后空行"理论被自己的验证数据推翻并修正，调试方法论见该条目。与 2026-09-14 的 Day23 行内公式事件（day-23-2019-residual-rl/NOTES.md 问答补充）是同一主题的连续两天抓包；两次合起来形成了仓库现行的三条数学排版规则。
