# Day 30 — Physical AI Eval + Data Flywheel：评测合同 × 安全红线 × 数据飞轮（30 天收官）

## 元信息
- Title: Physical AI Eval + Data Flywheel — end-to-end synthesis（收官日，无单篇论文）
- Authors / Org: —
- Link / arXiv: 本日为 30 天路线收官综述；锚点材料见 Day28（ManiSkill3 / robosuite）与 Day29（Brunke et al. 安全综述）
- 锚点论文：ManiSkill3 — https://arxiv.org/abs/2410.00425；Safe Learning in Robotics 综述 — https://arxiv.org/abs/2108.06266
- 锚点代码：ManiSkill — https://github.com/haosulab/ManiSkill；safe-control-gym — https://github.com/utiasDSL/safe-control-gym
- Date read: 2026-09-21
- Tags: [physical-ai, evaluation, data-flywheel, release-gate, deployment, synthesis]
- Thread: physical-ai
- Folder: day-30-physical-ai-eval-data-flywheel
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-30-physical-ai-eval-data-flywheel

## 一句话总结
Day30 是收官日：把 30 天从"点"连成"线"——Day28 的评测合同给出 release 的**测量尺**，Day29 的安全栈给出 release 的**红线**，Day30 把两者焊进一条生产闭环：failure → triage → recollect / resimulate → retrain → gated deploy；eval 不是终点，是数据飞轮的转速表；飞轮转起来，Day01–29 的每一项技术才有复利。

## 和之前工作的关系

- **接了哪条线**：全 30 天。Day01（Physical AGI 定义）提出目标；Day02/03（MuJoCo / Isaac Lab）给仿真基座；Day04–06（Genie / UniSim / DreamerV3）世界模型；Day07/08（H2O / Humanoid-Gym）全身控制；Day09–14（RT-2 / OpenVLA / Habitat / π₀ / Diffusion Policy / Octo / π₀.₅）VLA 与开放世界；Day15–18（OXE / DROID / BridgeData V2 / RoboCasa）数据规模化；Day19/20（PPO / RLPD）优化与真机 RL；Day21–24（DR / RMA / Residual RL / SimOpt）sim2real；Day25–27（Gato / GR00T N1 / Cosmos）通用智能体与世界基础模型；Day28 评测合同；Day29 安全栈。Day30 把"怎么训出来"变成"**怎么持续变好且不出事**"。
- **补了哪个短板**：前 29 天回答"单点技术怎么做"，缺的是"生产系统怎么运转"——release 谁说了算、失败数据回哪去、新模型凭什么上线。Day30 是 Day24 的 release gate 四件套（sim2sim / HIL 探针 / 分桶指标 / SimOpt 校准）的终极形态。
- **替代 / 分叉 / 改进**：
  - vs Day28（评测基准）：Day28 把评测写成"固定测度 $\mu$ 下的二项估计"，是**静态合同**；Day30 把评测变成飞轮的**转速表**——每一次 gated deploy 都是一次新的测量，而测度本身随部署分布漂移而演进（golden set 只增不减）。
  - vs Day29（安全栈）：Day29 的 CBF / shield / RTA 是飞轮的**刹车**（保证"转起来不出事"）；没有刹车的飞轮，不敢把真机数据回流开到最大。
  - vs Day27（Cosmos）：WFM 是飞轮的**廉价数据引擎**——resimulate 分支把"失败场景"的复现成本从真机小时降到 GPU 小时（Transfer 的可控重绘正好用来复现长尾失败）。
  - vs LLM post-training 飞轮（RLHF：deploy → 偏好标注 → DPO / RL → redeploy）：LLM 飞轮失败成本约等于 0、可大规模并行人类标注；机器人飞轮失败等于硬件损坏、采集贵 2–3 个量级、动作有不可逆的因果后果——所以机器人飞轮必须多两样东西：Day29 的安全刹车和 Day28 的评测合同。

## 为什么今天读它

Roadmap Day30 的官方主题就是"汇总 state / action / latency / safety 指标，设计 failure → triage → recollect / resimulate → retrain → gated deploy 闭环"。它是 30 天的收官：eval（Day28）+ safety（Day29）+ data（Day15–18、27）三条线在此交汇成一条生产闭环。

## 今天的 3 问
1. **Release gate 的判定函数长什么样？** 成功率 $\hat{p}=k/N$ 是二项估计（有 Wilson 置信区间），安全是逐轨迹证书（0/1 不可平均），latency 是系统级硬约束——三类不同数学强度的测度，怎么合成一个"能不能上线"的布尔判定？各自的置信度 / 显著性怎么给？
2. **Failure triage 的 taxonomy**：失败按什么分桶（perception / planning / control / hardware / distribution shift）？为什么每个桶的修复路径不同——recollect（真机重采）、resimulate（仿真/生成重放）、relabel（标注放大）分别对应哪种失败？"哪条数据最值钱"有没有可计算的定义（边际信息增益）？
3. **飞轮的闭环稳定性**：部署 → 数据回流 → 模型变强 → 部署更广 → 遇到更难的分布 → 更多失败数据……这是正反馈。它什么时候收敛、什么时候发散（比如模型变强后接更难的任务，**观测成功率反而下降**——"越强越敢，越敢越难"）？怎么设计让飞轮"向上收敛"的阻尼？

## 核心

### 1. Motivation
- 前 29 天解决"怎么训出一个能用的 policy"，Day30 回答"怎么让它在生产里一直变好、且不出事"。生产系统的三个敌人：
  1. **分布漂移**：部署环境永远在变（光照、物体、用户行为），Day21–24 的 sim2real 只解决"出厂时"的对齐；
  2. **长尾失败**：成功率 99% 的剩下 1% 杀死产品——而这 1% 恰恰是训练分布里没有的（Day16 DROID 的教训：场景覆盖即泛化增益）；
  3. **回归**：新模型在新数据上变好、在老场景上退化——没有 golden set 的"进步"是不可信的。
- 和 Physical AGI 的关系：Day01 定义的 Physical AGI 不是"一个模型"，而是一个**持续进化的系统**。飞轮是 AGI 的工程形态：数据 → 模型 → 部署 → 数据，复利增长。Day25–27 的"通用"（Gato / GR00T / Cosmos）只有在飞轮里才能从 demo 变成产品。

### 2. 指标体系：state / action / latency / safety 四轴

记一次部署 rollout 为轨迹 $\tau=(s_0,o_0,a_0,s_1,\dots)$ ， $s$ 为 state（本体+环境真值，仿真可得、真机部分可得）， $o$ 为 observation（RGB-D / 触觉等）， $a$ 为 action。

- **Task success（任务轴）**：二项估计 $\hat{p}=k/N$ ，配 Wilson 区间 $[\hat{p}_{\mathrm{low}},\hat{p}_{\mathrm{high}}]$ ——release 判据用的是**下界** $\hat{p}_{\mathrm{low}}$ ，不是点估计。分桶 success（Day24 / 28 的思想）：按场景难度、光照、embodiment 分桶，桶级下界全部过线才算过。
- **Safety（安全轴，Day29 的输出）**：CBF 违反计数、shield 干预率、RTA 备份切换次数——**零容忍型**指标： $n_{\mathrm{violation}}=0$ 是硬门槛，不可被成功率平均（Day29 核心结论：期望可平均、逐点不可平均）。
- **Latency（时间轴）**：端到端闭环 $T_{\mathrm{loop}}=T_{\mathrm{perc}}+T_{\mathrm{infer}}+T_{\mathrm{ctrl}}\le T_{\mathrm{budget}}$ 。Day26 的双系统（10Hz VLM 推理 + 120Hz DiT 控制）就是 latency 预算驱动的架构实例； $T_{99}$ （99 分位）比均值更重要——长尾延迟等于失控。
- **Intervention rate（接管率）**：人类接管次数 / 总步数——最诚实的部署指标，直接对应 autonomy 水平。

### 3. Release gate：判定的数学形态

$$G(\pi)=\big[\hat{p}_{\mathrm{low}}(\pi)\ge p_{\min}\big]\;\land\;\big[n_{\mathrm{safety}}(\pi)=0\big]\;\land\;\big[T_{99}(\pi)\le T_{\max}\big]\;\land\;\big[\mathrm{Golden}(\pi)\succeq\mathrm{Golden}(\pi_{\mathrm{old}})\big]$$

- 四项分别是：成功率下界、安全零违反、延迟上界、golden 回归集不退化——**合取**，一票否决。
- **Golden set**：只增不减的回归测试集（历史上所有修过的失败场景的最小复现），是飞轮的"记忆"。
- 统计诚实： $p_{\min}$ 的选择要和 $N$ 联动—— $N$ 太小，Wilson 区间太宽，谁也过不了 gate；Day28 的"吞吐决定 $N$ 与统计精度"在这里变成发布成本。

### 4. Data flywheel：failure → triage → 数据策略 → retrain → gated deploy

闭环五步：

1. **Deploy + log**：记录 $(s_t,o_t,a_t,r_t,c_t,\mathrm{intervention}_t,\mathrm{fail}_t)$ ——Day29 的 monitor 在这里第一次产生**生产标注**（shield 干预 = 隐式负样本）。
2. **Triage（分桶）**：perception（看错）/ planning（想错）/ control（做错）/ hardware（本体坏）/ shift（分布外）。分桶决定修复路径——这是飞轮最关键的一步，分错桶等于修错方向。
3. **数据策略三选一**：
   - **recollect**：真机重采——贵（Day16 DROID 量级成本），用于 sim / 生成覆盖不了的失败（接触丰富、长尾场景）；
   - **resimulate**：仿真 / 生成重放——Day27 Cosmos / Transfer 把失败场景"重绘"成千条变体，GPU 小时替代真机小时；
   - **relabel / 放大**：Day18 MimicGen 式 SE(3) 搬运，把一条真机失败放大成一批。
4. **Retrain**：co-train 混合（Day16 DROID 配方：本域小数据 + 大池数据），golden set 常驻。
5. **Gated deploy**：过 $G(\pi)$ 才上线；上线后回到步骤 1。
- 数据价值的数学：一条候选数据的价值约等于它带来的期望风险下降 —— $\mathrm{Value}(d)=\mathbb{E}[R(\pi)-R(\pi^{+d})]$ ，实践中用"失败桶覆盖度 × 采集成本"近似。飞轮的调度问题本质是**在预算下最大化边际信息增益**——这是 active learning 在策略空间的版本。

### 5. 飞轮稳定性：什么时候收敛

- 正反馈链：部署更广 → 遇到更难分布 → 失败更多 → 数据更多 → 模型更强 → 敢部署更广……
- **"越强越敢，越敢越难"悖论**：模型变强后接更难的任务，**观测成功率可能下降**——不是模型退化，是任务分布变难了。所以飞轮的健康指标不是成功率本身，而是**同分布桶内的成功率**（分桶指标的深层原因）。
- 阻尼设计：gated deploy（不达标不上线）、golden set（只增不减）、安全刹车（Day29 保证探索步伐可控）。没有阻尼的正反馈等于发散。

## 可迁移 / Transfer

- 对 Infra → Post-training → Physical AI 的启发：
  1. **Eval 即 infra**：Day28 的"吞吐决定统计精度"在这里变成发布成本——评测 infra（并行仿真、真机场测管线）是飞轮的转速上限。
  2. 你在 ML-for-infra 里做过的"指标 → 告警 → 回滚"生产闭环，和 $G(\pi)$ 是同一个形状：SLO（latency / 成功率）+ 安全红线 + 金丝雀发布 = gated deploy。
  3. Post-training 的语言（SFT / RL / 评测门禁）平移到 Physical AI：recollect / resimulate = 数据侧的 SFT，gated deploy = 评测门禁，safety stack = 对齐约束。

## 疑问 / 下一步

- 没看懂的 1 个问题：triage 的自动化——失败分桶目前靠人工看 log，能不能从 $(o_t,a_t,\mathrm{intervention}_t)$ 学一个 classifier 自动分桶？它的训练信号从哪来（shield 干预是弱信号）？
- **Goodhart 警告**：指标一旦成为目标就失效——团队开始"刷" $\hat{p}$ （挑简单场景测）怎么办？分桶 + golden 只增不减是已知的对抗手段，但组织层面的对抗永远存在。
- 30 天之后：路线图走完了，但真正的"第 31 天"是把这套东西用在一个具体机器人上——建议选一个（双臂？人形上肢？），把 Day30 的闭环跑一遍最小实例。

## 今晚产出
- [x] 收官综述：评测四轴指标 + release gate 判定函数 $G(\pi)$ + 数据飞轮五步闭环 + 稳定性分析
- [x] 串联 30 天：Day01→30 主线（仿真 → 世界模型 → VLA → 数据 → 优化 → sim2real → 通用 → 评测 → 安全 → 飞轮）
- [x] 更新 reading-log.csv、README.md 进度（30/30），commit + push

## 连接
- 上一篇：Day 29 — Safe Robot Learning（https://github.com/Papa-Panda/post-training/tree/master/physical-ai/day-29-safe-robot-learning）
- 这是 30 天路线的最后一篇：30/30 收官 🎉
