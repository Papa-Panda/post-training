# 03 — Entropy and DiffAdapt: 熵是 U 型的，预算就该分三档

> 生成过程的平均 token 熵随题目难度呈 **U 型**：简单的题熵高但做得对，中等难度熵最低，难题熵高且真的不会。DiffAdapt（Xiang Liu、Xuming Hu、Xiaowen Chu、Eunsol Choi）把这个观察做成三档 inference 路由：Easy / Normal / Hard 各配不同的 prompt、temperature 和 token 预算。
>
> Mean generation entropy is U-shaped in problem difficulty: easy questions have high entropy yet high accuracy, medium difficulty has the lowest entropy, hard questions have high entropy reflecting genuine uncertainty. DiffAdapt turns this into a three-tier inference router with per-tier prompt, temperature, and token budget.

## 1. 数学：token 熵与 trace 熵

对 trace 位置 $t$ ，模型输出词表分布 $p_{t}$ 。**token entropy** 是它的 Shannon 熵：

$$H_{t}=-\sum_{j}p_{t,j}\log p_{t,j}.$$

**trace entropy** 是整条 trace 的平均：

$$\bar{H}=\frac{1}{T}\sum_{t=1}^{T}H_{t}.$$

注意 $\bar{H}$ 和最终答案正确性之间**没有单调关系**——这正是 U 型曲线的要点。

## 2. 实证：U 型曲线

把题目按难度 $d$ 排序， $\bar{H}(d)$ 的形状：

- **Easy**：熵高，但准确率高。模型"想得多但做得对"——高熵在这里不是困惑，而是解法空间的自然发散。
- **Normal / medium**：熵最低。模型走最熟悉的路，分布尖锐。
- **Hard**：熵高，准确率低。这里的高熵是**真实不确定性**（genuine uncertainty），不是发散。

论文报告从 easy 到 medium 熵下降 22–25% 。这个数字的工程含义：如果你只看到"熵下降"，不能直接解读为"模型更确定了"——要先看题目落在哪一段。

## 3. 系统：三档路由

DiffAdapt 的核心动作是 inference 时路由。路由依据是一个小 probe：读 final hidden state，预测题目难度。三个档位（论文表中的配置）：

| 档位 | temperature | token 预算 | 直觉 |
|---|---|---|---|
| Easy | 0.5 | $0.4\times\text{Max}$ | 直接解 + 验证，低温小预算 |
| Normal | 0.8 | $1.0\times\text{Max}$ | 逐步推导，给满预算 |
| Hard | 0.4 | $0.5\times\text{Max}$ | **fail fast**：超出能力边界时继续生成常常无效，不如限预算止损 |

Hard 档的逻辑值得单独说：它的 temperature 比 Normal 还低（0.4 vs 0.8），预算只有一半。这不是"难题多给资源"，而是"难题少浪费资源"——论文认为超出模型能力的问题，拉长 trace 主要是空转。

三阶段训练流程：

1. **数据**：用 proxy model 生成 trace，按启发式规则标注难度，得到训练数据。
2. **探针**：在 final hidden state 上训练一个小 probe 做难度分类。
3. **路由**：inference 时 probe 先判难度，再进对应档位。

## 4. 效果：oracle 上限与真实结果

- **Oracle 路由**（假设难度已知）：约 $50\%$ 的 token 节省，准确率提升超过 $10\%$ 。这是该方法的上限演示。
- **真实结果**（论文摘要）：5 个模型、8 个 benchmark，token 最多下降 $22.4\%$ ，准确率持平或更好。

两个数字的差距（ $50\%$ vs $22.4\%$ ）就是 probe 误分类和启发式标注噪声的代价——路由系统的精度天花板，永远是难度估计的精度。

## 5. 与 02 的对照：两种"不确定性"量的是不同东西

| | Token 熵 $H_{t}$ （本章） | JSD 轨迹 $D_{t,l}$ （02） |
|---|---|---|
| 固定什么 | 固定位置 $t$ 、固定层 | 固定位置 $t$ ，扫层 $l$ |
| 问的问题 | "这一刻模型有多犹豫？" | "这个决定被改了多少次？" |
| U 型 | 对难度 $d$ 呈 U 型 | 未在论文中报告对难度的形状 |
| 系统用途 | 路由前先估计难度 | 在线选高质量样本（Think@n） |

工程上两者可以串联：先用熵/probe 做**难度路由**（花多少钱），再用 DTR 做**样本选择**（选哪条 trace）。预算分配和质量选择是正交的。

## 6. 代码对应

`code/trace_lab.py` 的 (b) 部分：

- `u_entropy(d)` ：U 型熵的解析玩具， $d^{\star}=5.5$ 处取最小。
- `route(d)` ：三档路由，返回 `(策略, token 预算)` ，Easy $0.4\times$ 、Normal $1.0\times$ 、Hard $0.5\times$ ，与论文表一致。
- `routing_experiment` ：uniform 全预算基线 vs 路由预算。玩具设定：easy 题小预算可解、normal 题需近满预算、hard 题任何预算都解不出（fail fast）。断言路由省 token 且准确率不降。
