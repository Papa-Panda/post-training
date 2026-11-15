# Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training

> 📖 阅读版：https://papa-panda.github.io/post-training/synthetic-data/day-08-harness-s/

## 元信息

- Title: Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training
- Authors: Zhang 等
- Link / arXiv: https://arxiv.org/abs/2611.04277
- Date of publication: 2026-11-03
- Tags: [grpo, harness, effective-groups]
- 与前篇的关系：Day 07 Spotify 是合成数据当测试集、修规划层而不动模型权重；Day 08 回到 RL 训练本身——GRPO 的学习信号出了问题。Day 02 AED 把真实失败样本当主资产进 SFT，Day 08 也是“失败变资产”的思路，但救的不是数据、是训练信号：原 harness 下全组失败时 GRPO 组内无奖励方差，兄弟 harness 把失败组“救活”成有效组。Day 04 ToolLoop 的闭环是带着错误原因重生成数据，Harness-S 的闭环是带着失败原因换环境重跑——一个改数据分布，一个改环境分布。

## 一句话总结

GRPO 靠组内奖励方差学东西，但 agent 任务里常出现整组在原 harness 下全军覆没——奖励全是 0，没有方差就没有梯度信号，这组样本等于白烧 rollout 算力。Harness-S 不改模型、不改数据：先由 edit agent 按失败原因沿不同修复方向演化出一批兄弟 harness，组成 Sibling Harness Pool（训练期间固定）；只有当初始尝试在原 harness 下全部失败时才调用救援——同一策略在被选中的兄弟 harness 下重新 rollout，救援组替换掉失败组进 RL；Adaptive Harness Router 按行为画像（Harness Behavior Profiles）加多 LLM 投票选 Top-k，画像随策略学习做 policy-aware 重校准。Qwen3.5-4B 在 AppWorld 上 Avg@3 从 48.98% 涨到 56.87%，有效组比例从 64.69% 涨到 79.38%，代价是多花 18.44% 的 rollouts；而且学到的能力能带回原 harness 下用，在 AppWorld、SWE-Gym、DBBench 三个 benchmark 上都显著优于只训策略。

## 3 个问题

1. 救援组的奖励方差是真信号，还是兄弟 harness 人工抬起来的？全败组的奖励全是 0，救援组在兄弟 harness 下跑出了方差、有了梯度——但这个方差可能是“题变简单了”而不是“策略变强了”。设计一个实验：三臂——A 原 harness 全失败组直接丢弃（基线，GRPO 只用有效组）；B Harness-S 救援组替换失败组（论文方法）；C 救援组照常生成、但奖励按“拿回原 harness 重评”打分（救援轨迹在原 harness 下重新验证，不过仍计 0）。指标：AppWorld Avg@3、有效组比、救援轨迹在原 harness 下的真实通过率。判据：若 B ≫ A 但 C ≈ A → 救援组的方差是兄弟 harness 放水抬起来的，策略学到的是“在兄弟 harness 下好过”的行为，不是原任务能力；若 C ≈ B → 救援信号是真的，兄弟 harness 的修复方向保住了任务本质。再构造一个具体反例：兄弟 harness 把“必须调用三个工具的任务”拆成“调一个工具就行”的变体——策略在兄弟 harness 下学会“跳步”，回原 harness 全败；若 B 臂里这类轨迹占多数，Avg@3 的提升就是假的。再预测一个 2×2 格子：{救援轨迹在原 harness 下能过，不能过} × {救援组内奖励方差高，低}，哪个格子最可能是真能力提升并写理由。漏洞：C 的“重评”本身可能不公平——兄弟 harness 的修复可能改了任务语义，原 harness 的 checker 会把“等价但不同路径”的解判错；而且救援组只在全组失败时触发，C 臂的样本天然是难任务，与 A 臂的有效组难度分布不同，直接比 Avg@3 是在比任务难度。
2. 兄弟 harness 的“兄弟”关系怎么定义才不引入捷径？论文的方法是按失败原因定向演化（Directional Harness Evolution），但“按失败原因改”与“把题改简单”之间只有一线之隔——edit agent 完全可以把导致失败的那个环节整个删掉。设计一个实验：三臂生成兄弟 harness——A 按失败原因定向演化（论文方法）；B 随机扰动原 harness（改提示词、工具描述、超时等，不针对失败原因）；C 人工写“最小改动”兄弟 harness（只改导致失败的那一个环节，其余一字不动）。指标：救援成功率、救援轨迹回原 harness 的真实通过率、Avg@3。判据：若 B 的救援成功率也高、但回原 harness 通过率 ≪ A → “兄弟”太松，救援只是在造简单题，定向演化那套失败原因分析没起作用；若 C 的回原通过率 ≫ A → 定向演化还可以更精准，edit agent 的改动幅度过大。再预测一个 2×2 格子：{兄弟与原任务语义等价，不等价} × {救援成功，救援失败}，哪个格子的救援轨迹最值得进 RL 并写理由。漏洞：C 的人工最小改动本身就是金标准泄露——人已经知道正确答案长什么样，写出来的“最小改动”天然偏向可解方向；而且“语义等价”没有可操作的判定，Router 的多 LLM 投票可能与人类判断不一致，投票本身就可能在奖励“改得多的兄弟”。
3. 只救全败组，那“半败组”的信号是不是白白浪费了？——综合 Day 04 ToolLoop。Harness-S 的救援只在全组失败时触发；ToolLoop 证明闭环反馈（带着错误原因重生成）比单纯堆数据重要（无反馈合成数据比不训还差 2.17 个点）。但组内通过率 1/8 的组也有方差、只是方差小，里面的失败轨迹同样带着错误原因，却得不到任何救援或重生成。设计一个实验：三臂——A 只救全败组（论文方法）；B 全败组 + 半败组（组内通过率 < 50% 的组也触发兄弟重跑）；C 半败组不触发救援、但把失败轨迹的错误原因喂给 edit agent，生成针对性兄弟 harness 供下一轮训练用（ToolLoop 式：失败原因进生成，而不是多跑 rollout）。指标：Avg@3、有效组比、rollout 总成本。判据：若 B 的有效组比 ≫ A 但 Avg@3 ≈ A → 半败组救援只是在抬有效组比这个数字，没转化成能力；若 C 的 Avg@3 ≫ A 且成本低于 B → 失败原因进生成比多跑 rollout 更划算，Harness-S 的救援触发条件可以放宽到“原因复用”而不是“全败才救”。再预测一个 2×2 格子：{全败组，半败组} × {兄弟重跑救援，失败原因进生成}，哪个格子单位 rollout 的 Avg@3 提升最高并写理由。漏洞：B 的成本随触发率线性涨，论文的 +18.44% rollouts 在 B 臂可能翻倍，没控成本就比不出机制；而且“半败”的 50% 阈值是拍脑袋定的，阈值一变结论跟着变；C 臂的“下一轮用”还引入了时间滞后——策略可能已经变了，针对旧失败生成的兄弟 harness 可能已经过时。

## 核心机制

1. **Motivation**：GRPO 的学习信号来自组内奖励方差。agent 任务难，整组在原 harness 下全失败是常态——奖励全是 0，组内无方差，这组样本对训练零贡献，还白白烧了 rollout 算力。痛点不是“数据不够”，是“失败的组没有信号”。以前的办法是直接丢掉这些组，等于把最难的任务整个从训练分布里删掉。
2. **方法**：Directional Harness Evolution——edit agent 按失败原因沿不同修复方向演化兄弟 harness，形成可复用的 Sibling Harness Pool，训练期间固定；Complementary Harness Rescue Training——仅当初始尝试在 base harness 下全部失败时调用救援池，同一策略在选中的兄弟 harness 下重新 rollout，救援组替换失败组进 RL；Adaptive Harness Router——按 Harness Behavior Profiles 加多 LLM 投票选 Top-k，policy-aware profile re-anchoring 随策略学习刷新画像。训练完的策略回到原 harness 下评测，仍显著优于只训策略——harness 激发的那些行为真的迁移进了策略本身，而不是只在兄弟 harness 下才管用。
3. **关键数字**（论文自报，出自 papers.md 注释）：Qwen3.5-4B 在 AppWorld 上 Avg@3 从 48.98% 涨到 56.87%（+7.89 个百分点）；有效组比从 64.69% 涨到 79.38%（+14.69 个百分点）；代价是 +18.44% rollouts。

$$ A_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})} \quad\text{；}\quad r_1 = r_2 = \cdots = r_G = 0 \Rightarrow \mathrm{std}(\mathbf{r}) = 0 \Rightarrow A_i = 0 $$

上式是 GRPO 组内相对优势的定义：全败组奖励全是 0，标准差为 0，优势全是 0——这组样本没有梯度信号。Harness-S 的救援组就是给这组重新造出一个标准差不为 0 的奖励分布。

## 边界

- 救援只在全组失败时触发：半组失败的组（奖励方差小但非零）得不到救援，信号浪费；全败阈值本身是二值开关，组内通过率 0/8 和 1/8 的待遇天差地别，阈值附近的组最容易被误判。
- “兄弟”没有等价性保证：兄弟 harness 是 edit agent 按失败原因演化的，改动可能悄悄降低任务难度或改任务语义——救援组的高奖励可能是“题变简单了”，不是“策略变强了”；论文用“回原 harness 评测”来兜底，但训练过程中策略已经在兄弟分布上走了很多步，评测时的迁移不等于训练时的信号干净。
- Router 的画像依赖多 LLM 投票加行为画像：投票的 LLM 与被训策略同源时，画像可能系统性偏向“策略已经会的那类修复方向”；policy-aware 重校准的频率与成本没有公开口径——校准太稀疏，画像滞后于策略；校准太频繁，投票成本又吃掉救援省下的算力。
- 代价是真实的：+18.44% rollouts 换 +7.89 个百分点的 Avg@3——救援的边际收益随任务变简单而递减；如果原 harness 的失败率本来就不高，救援池大部分时间闲置，性价比更低。
- 评测可质疑处：核心数字只报了 AppWorld 上 Qwen3.5-4B 的 Avg@3；SWE-Gym、DBBench 在注释里只说“显著优于”，没有给数字——跨 benchmark 的泛化幅度没法从注释里核实，AppWorld 一家的提升不能直接外推。

## 迁移

- “失败组救援”可搬到任何组内相对奖励的 RL（GRPO、RLOO 等）：只要奖励是组内比较的，全败组就是纯烧算力的死样本；关键是救援轨迹必须能“带回原任务”——Harness-S 的做法是训练后回原 harness 评测，搬过去时先定好“回原任务验证”这一环，否则救援只是在自嗨。
- 与 Day 02 AED 构成对照：AED 是把真实失败样本当主资产、进 SFT；Harness-S 是把失败组当“待救援信号”、进 RL。两者都拒绝丢掉失败，但 AED 改的是数据、Harness-S 改的是训练时采样的环境——失败样本的价值不在样本本身，在“失败有没有变成可学习的信号”。
- 与 Day 04 ToolLoop 构成对照：ToolLoop 是失败原因进数据生成（重生成数据再训），Harness-S 是失败原因进环境生成（演化兄弟 harness 再跑）；一个改数据分布、一个改环境分布。ToolLoop 证明无反馈合成数据比不训还差（−2.17 个点），Harness-S 的救援组如果奖励不回原 harness 校验，同样可能“越救越偏”。
- 如果复现，第一步先在自己的任务上量“全败组占比”——如果全败组不到 10%，救援的收益天花板很低，不值得建兄弟池；再写一个“兄弟 harness 等价性检查”（救援轨迹拿回原 harness 重跑，统计真实通过率），没有这个检查，救援成功率这个数字没有意义。

## 讨论题

- 这篇的结论里，哪一条最可能被后来的工作推翻？
- 数据 pipeline 中哪个环节最容易藏 bug？（edit agent 演化的兄弟 harness 是不是悄悄降低了任务难度；Router 的多 LLM 投票是不是在奖励“策略已经会的方向”；policy-aware 重校准是不是滞后于策略变化）
- 换成今天的模型规模，这个方法还成立吗？
