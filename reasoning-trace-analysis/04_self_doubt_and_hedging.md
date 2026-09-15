# 04 — Self-Doubt and Hedging: 已经做对了，为什么还在检查

> Self-doubt 指模型**已经得到正确信息/答案后**，仍把大量 token 花在重复验证上。三个工作从三个角度处理它：Peng 等给出定义和干预方法，SelfDoubt 给出可计算的 HVR 指标，"Know When to Stop" 给出六个轻量语言信号。
>
> Self-doubt is when a model keeps spending tokens re-verifying an already-correct answer. Three papers attack it from three angles: Peng et al. define it and propose an intervention, SelfDoubt gives the computable HVR metric, and "Know When to Stop" gives six lightweight linguistic signals.

## 1. 定义（Peng et al., 2505.23480）

Self-doubt 的操作化定义：**excessive token usage devoted to re-verifying an already-correct answer**——答案已经对了，模型还在"确认一遍"。论文把 trace 分成三类：

| 类别 | 含义 |
|---|---|
| Overthinking with Self-Doubt | 过度思考，且由自我怀疑驱动（反复验证已得答案） |
| Overthinking without Self-Doubt | 过度思考，但不是自我怀疑（比如真的在探索） |
| Non-Overthinking | 不过度思考 |

这个三分法的价值在于把"长"拆成两种："想太多"和"怀疑太多"。只有后者是纯浪费。

## 2. 干预：先检查输入，再决定想多深

Peng 等的方法分两步：

1. **先检查**：输入问题是否有效、信息是否齐全。
2. **再回答**：信息足够时，用最少的 token 作答。

Prompt 原文要求模型 "Before reasoning deeply, check whether all necessary information is available…"——信息齐全就别展开长推理。论文在其实验表上报告平均 reasoning length 降 $37.1\%$ 、accuracy 增 $3.6\%$ 。注意：这是**该论文实验表的平均**，不是普适结论；它的含义是"砍掉 self-doubt 式空转不伤准确率"，而不是"所有任务都能又短又准"。

## 3. 度量：HVR（SelfDoubt, 2604.06389）

Satwik Pandey、Suresh Raghu、Shashwat Pandey 提出 **Hedge-to-Verify Ratio**：

$$\mathrm{HVR}(T)=\frac{h(T)}{v(T)+1}.$$

其中 $h(T)$ 是 hedge marker 出现次数（表达 doubt 的措辞，如 "wait"、"hold on"、"but maybe"）， $v(T)$ 是 verify marker 出现次数（对 doubt 的**实际检查**，如 "verify"、"substitute back"）。分母 $+1$ 是平滑项。

关键区分：hedge ≠ verify。说"wait"只是表达怀疑，真的代入验算才是 verify。HVR 高 = 光怀疑不验证 = 空转。

论文最硬的数字是 $\mathrm{HVR}=0$ gate：在 7 个模型、3 个数据集上，HVR 为 0 的 trace 正确率 $96.1\%$ ，coverage $25.4\%$ 。解读：约四分之一的 trace 完全没有 hedging 措辞，而这部分几乎全对。这是一个**筛选器**，不是安全保证——不要把它读成"无 hedging 就一定对"。

## 4. 六个轻量信号（Know When to Stop, 2607.00482）

Chia-Hsuan Lee 等提出六个无需白盒访问的语言信号：

| 信号 | 含义 |
|---|---|
| S1 Repetition | 重复已说过的内容 |
| S2 Hedging | hedging 措辞 |
| S3 Abandonment | 放弃当前思路（另起炉灶） |
| S4 Contradiction | 自相矛盾 |
| S5 Recomputation | 重算已算过的东西 |
| S6 Length outlier | 长度异常 |

最强的是 **S3 abandonment**：在所测模型的 incorrect traces 中，abandonment 出现频率是 correct traces 的 4.1–4.3 倍。论文还强调：控制 response length 后，incorrect traces 依然有更多**无效** self-reflection——"长"本身不是病因，空转才是。

该论文还提出了 DASH（segment-level credit assignment），但那是线 3（过程监督）的次级指针，这里不展开。

## 5. 三者的关系：检测 vs 干预

```
检测（事后/在线判断 trace 有没有空转）
├── HVR：一个标量，白盒不需要，阈值可当 gate 用
└── 六信号：更细的分类，S3 abandonment 是最强单项

干预（事前减少空转）
└── Peng 的两步法：先验输入完整性，够信息就短答
```

HVR 和六信号回答"怎么发现"，Peng 的方法回答"怎么减少"。下一章（05）的 batch prompting 是另一种干预——不靠 prompt 指令，而靠任务结构本身压缩空转。

## 6. 代码对应

`code/trace_lab.py` 的 (c) 部分：

- `scan_trace` ：正则扫描 hedge / verify / recheck / re-derivation / tangent 五类 marker，输出 `TraceScan` 。
- `TraceScan.hvr` ：与论文公式逐字对应， $h/(v+1)$ 。
- `pattern_frequencies` ：按每 100 token 归一化——与 batch prompting 论文的归一化方式一致。
- `hedging_experiment` ：长 trace vs batched 短 trace 的四模式频率对比，演示"结构压缩空转"。
