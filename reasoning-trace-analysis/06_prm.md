# 06 — Process Supervision: 给中间步骤打分（PRM）

> Outcome supervision 只看答案对错：答案对但步骤错的解会被当成好解（false positive）。"Let's Verify Step by Step"（Hunter Lightman、Vineet Kosaraju、Yura Burda、Harri Edwards、Bowen Baker、Teddy Lee、Jan Leike、John Schulman、Ilya Sutskever、Karl Cobbe；arXiv 2023-05-31，ICLR 2024）训练 **Process-supervised Reward Model (PRM)**：对每一步预测正确性，整解得分为各步正确概率之积。
>
> Outcome supervision only sees final-answer correctness: a solution with the right answer via a wrong step is graded as good (false positive). "Let's Verify Step by Step" trains a Process-supervised Reward Model (PRM) that predicts per-step correctness; the solution score is the product of per-step correctness probabilities.

## 1. 问题：ORM 的 credit assignment 缺陷

记一个解为 $K$ 个 step： $s_{1},\ldots,s_{K}$ ，最终答案正确性为 $y\in\{0,1\}$ 。

**ORM**（outcome-supervised reward model）只用 $y$ 做标签。测试时，解的得分是 ORM 在**最后一个 token** 上的预测：

$$\mathrm{score}_{\mathrm{ORM}}=P_{\mathrm{ORM}}(y=1\mid x,s_{1:K}).$$

缺陷： $y=1$ 但某步 $s_{k}$ 错误的解（false positive），ORM 照样给高分。best-of-N 选出来的"最优解"可能包含错误推理——这正是把 trace 当**证据**用时最危险的情况。

## 2. 数学：PRM 的打分规则

PRM 对**每一步的最后一个 token**预测该步正确性。一次 forward pass 打完全部 $K$ 步，解的得分定义为各步正确概率之积：

$$\mathrm{score}_{\mathrm{PRM}}=\prod_{k=1}^{K}P_{\mathrm{PRM}}(r_{k}=+\mid x,s_{1:k}).$$

乘积形式的含义： $P(\text{全对})=\prod_{k}P(\text{第 }k\text{ 步对})$ ，即假设各步正确性条件独立时"整解无错"的概率。注意这是一个**建模选择**，不是定理——步骤之间显然不独立，但乘积是一个简单有效的聚合。

## 3. 数据：PRM800K

过程监督的最大成本是标注。论文发布 **PRM800K**：800K step-level labels，来自 75K solutions、12K problems。每步标签三档：positive / negative / neutral（neutral 给"虽然没错但对解题无用"的步骤）。

数据效率优化：**active learning** 让过程监督的数据效率提升 $2.6\times$ ——优先标注模型最不确定的步骤，而不是均匀采样。

## 4. 实证：PRM > ORM，但有前提

- 大规模结果：在 MATH test set 的 representative subset 上，PRM 达到 $78.2\%$ （摘要写作约 $78\%$ ），显著优于 ORM。
- 背景：Uesato et al. (2022) 在 grade-school math 上发现 ORM 与 PRM 最终表现相近。Lightman 等的反驳是：换更难的 MATH、更强的 base model、更多的 feedback，PRM 的优势才显现。**PRM 不是无条件赢**，它的赢面在"难到步骤质量拉开差距"的问题上。

## 5. 系统：step 切分是数据与系统的 contract

PRM 有一个常被忽略的系统前提：为了方便解析，论文把 generator 训练成输出 **newline-delimited step-by-step 格式**——每行即一步。这回扣到 00/01 章的观点：**PRM 天生不知道什么是 planning、什么是 calculation、什么是 backtracking**。step boundary 不是语义真理，是数据格式和系统之间的约定。换一种切分，PRM 的 $r_{k}$ 语义就变了。

## 6. 从打分到训练信号：credit assignment 的推广

PRM 回答"选哪个解"（best-of-N 的打分器）。更进一步的问题是：**训练时，每个 token 该分到多少 credit？**

- **GRPO 式**：一个标量 advantage 广播到整条 trace 的每个 token。整条 trace 要么全奖要么全罚。
- **Segment 式**（如 "Know When to Stop" 的 DASH）：按 segment 分配。典型场景是 **answer drift**：trace 在中间已经得出正确答案（checkpoint），后面又漂移到错误答案。标量 advantage 会给"找到正确答案的那段"也打负分；segment 式给 checkpoint 段正 credit，对 drift 段施加递增惩罚。

这是 PRM 思想从"打分器"到"训练信号"的自然延伸：过程监督的终极形态不是选解，而是让每个 token 位置都收到与其贡献相称的梯度。

## 7. 回扣线 1：PRM 需要结构解剖

PRM 的 $r_{k}$ 只说"这步对/错"，不说"这步在干什么"。但 04、05 章告诉我们：同样是"错"的一步，hedging 空转和 genuine 的探索失败是两回事；同样是"对"的一步，deep-thinking token 和 filler token 的价值也不同。**过程监督的下一步，是把行为标签（01、02）和步骤打分（本章）结合起来**：先知道每段在干什么，再决定每段值几分。

## 8. 代码对应

`code/trace_lab.py` 的 (d) 部分：

- `make_candidates` ：合成候选解， $20\%$ 为 false positive（答案对、某步错）。
- `orm_score` / `prm_score` ：与 §2 定义逐字对应。
- `prm_vs_orm_experiment` ：best-of-N 下"选出的解全步正确"的比例。断言 PRM 不低于 ORM，且只要池子里有全对解，PRM 必选中（scoring rule 的单调性）。
- `uniform_advantage` / `segment_advantage` ：GRPO 标量广播 vs DASH 式 segment credit 在 drift trace 上的对比。断言：checkpoint 段得正 credit（即使整条 advantage 为负），drift 段惩罚递增。
