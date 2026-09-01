# 10 — RICo：用 ICL 干预近似训练数据价值

## 元信息
- Paper: [RICo: Refined In-Context Contribution for Automatic Instruction-Tuning Data Selection](https://arxiv.org/abs/2505.05327)
- Official code: [annayang2020/rico_data_selection_for_instruction-tuning](https://github.com/annayang2020/rico_data_selection_for_instruction-tuning)
- 内容类型：跨方法坐标中的新增锚点（gradient-free、assessment-set-conditioned valuation）
- 本章定位：补全 LESS / DataInf 之外的 **ICL-based model-aware valuation**，不在 `ai-data/` 新增 Day。

## 1. 它补的是哪一个空位

LESS / DataInf 用参数空间信号回答：候选样本的训练梯度是否能推动目标能力？RICo 改用函数空间干预：把候选样本放入 context 后，模型在 assessment set 上是否立刻变好？

```text
candidate T_j
   ├─ gradient branch:  g(T_j) vs. g(V)          -> LESS / DataInf
   └─ ICL branch:       ΔPPL(V | demonstration)  -> RICo
```

两条线解决的是同一个 data 问题：**一条候选 SFT 数据对目标能力有没有边际价值？** 区别只在代理信号。

## 2. RICo score

候选 instruction-response pair 为

$$T_j=(x_j^t,y_j^t),$$

assessment sample 为

$$S_i=(x_i^a,y_i^a).$$

直接看 $\mathrm{PPL}(S_i\mid T_j)$ 会受到 context 长度影响。RICo 用与 $T_j$ 等长、但无语义的随机序列 $T_j^{\mathrm{rand}}$ 做控制：

$$\mathrm{task\mbox{-}RICo}(T_j\to S_i)=\frac{\mathrm{PPL}(S_i\mid T_j^{\mathrm{rand}})-\mathrm{PPL}(S_i\mid T_j)}{\mathrm{PPL}(S_i)+\epsilon}.$$

再对 assessment set 平均：

$$\mathrm{global\mbox{-}RICo}(T_j)=\frac1{|D_a|}\sum_{S_i\in D_a}\mathrm{task\mbox{-}RICo}(T_j\to S_i).$$

这个分数可理解为一次受控的功能性干预：若真实 demonstration 比等长随机 context 更能降低目标样本 perplexity，则候选样本被认为具有正贡献。

论文的 assessment set 混合 OpenOrca-GPT3.5、OpenOrca-GPT4 和 Dolly-15K，意图覆盖不同生成来源与任务类型。这里的关键不是三个数据源本身，而是：**assessment set 实际定义了 selector 眼中的“价值”。**

## 3. 从 $O(nm)$ 到 $O(m)$

若有 $m$ 条候选和 $n$ 条 assessment samples，直接打分需要 $O(nm)$ 次推理。RICo 的扩展方式是：

1. 只对候选子集计算真实 global-RICo；
2. 将 top-$K\%$ 标为高贡献数据；
3. 用 LoRA 训练轻量 selection classifier；
4. 对完整候选池做 $O(m)$ 线性扫描。

因此第二阶段不再直接计算 contribution，而是在学习“高 RICo 数据长什么样”。这带来可扩展性，也引入第二层代理误差：classifier 可能学到长度、格式、语言风格或数据来源，而不是真实贡献。

## 4. 和现有方法的关系

| 方法 | 主要信号 | 回答的问题 | RICo 的关系 |
|---|---|---|---|
| Influence / TracIn | Hessian / checkpoint gradient | 谁影响了某个行为 | attribution 前驱 |
| LESS / DataInf | candidate–target gradient geometry | 谁最能推动目标能力 | 同问题的 parameter-space 分支 |
| Nuggets / ICP | ICL 中的粗粒度比较 | demonstration 是否有帮助 | RICo 最直接的方法前驱 |
| SuperFiltering / IFD | 样本自身条件难度 | 哪些数据更难、可能更可学 | intrinsic difficulty，不等于 contribution |
| DEITA | quality × complexity × diversity | 什么样的数据看起来更好 | heuristic 属性，与行为干预互补 |
| Vendi / G-Vendi / SPICE | 集合覆盖与协调 | top-$k$ 是否重复或互相抵消 | RICo 之后的集合级选择 |

一句话：**LESS 更接近真实训练机制但需要梯度；RICo 更轻、更易迁移到黑盒/弱白盒设置，但 ICL→SGD 的代理错配更大。**

## 5. 不能把它当成完整 selector

### 5.1 ICL 帮助不等于训练帮助

一个 demonstration 可能因为语义相似、模板匹配、答案风格或局部复制降低当前 PPL，却不保证参数更新后仍产生同方向收益。RICo 论文也明确把 ICL 视为 full gradient training 的近似，而不是等价替代。

### 5.2 Assessment set 决定价值函数

若 assessment set 以通用 QA 为主，coding、tool use、长上下文或 repo navigation 数据可能被系统性低估。等权平均还可能让稀有但关键的能力被多数任务淹没。生产系统应按能力桶报告 task-RICo，并显式设置权重或最低覆盖，而不是只保留一个 global average。

### 5.3 长度控制不等于完全去偏

等长随机 context 主要修正长度效应，不能消除模板、语言、格式、主题和答案风格相似性造成的 PPL 改善。

### 5.4 单样本 top-$k$ 会重复

RICo 给的是 pointwise value。若高分样本都来自同一种 Python 算法题，top-$k$ 仍可能高度冗余。应在 RICo shortlist 后增加 Vendi / G-Vendi coverage，必要时再加 protected-set conflict gate。

## 6. Coding data 的可证伪实验

不要先假设 RICo 能迁移到 coding。最小实验应直接比较三种分数对真实训练收益的预测能力：

1. 构造分层 assessment set：compile、unit test、debugging、repo navigation、multi-file edit、timeout/efficiency；
2. 对同一候选池计算 RICo、LESS gradient alignment 和 IFD；
3. 从各方法的高/中/低分区间抽样，做多个固定 token-budget 的小规模 SFT；
4. 在严格去污染的 held-out tasks 上记录 pass-rate delta；
5. 比较三种 score 与真实 $\Delta\text{pass-rate}$ 的 rank correlation，并按任务桶检查偏差；
6. 对 RICo top-$k$ 再加入 coverage constraint，测冗余减少是否带来额外收益。

```text
quality / provenance / decontamination gate
    -> RICo or LESS value shortlist
    -> G-Vendi / SPICE set-level selection
    -> execution verification
    -> train + held-out pass-rate audit
```

真正要回答的不是“RICo 分数是否漂亮”，而是：**ICL contribution 是否能稳定预测 coding SFT 后的能力增量；若不能，错在哪些任务桶、哪些模型和哪些数据格式。**

## 7. 结论

RICo 值得进入本专题，因为它把 model-aware valuation 从“必须读取梯度”扩展到“可以观察受控的模型行为变化”。但它不是新的主干范式，也不替代 coverage、correctness、execution verification 或 protected-set safety。

最合适的位置是：

$$\text{quality gate}\rightarrow\{\text{LESS/DataInf},\ \text{RICo}\}\rightarrow\text{coverage/conflict}\rightarrow\text{train/eval}.$$

<!-- NAVIGATION -->
## 导航

- 上一篇：[09 SPICE 协调性](09_spice_information_conflict.md)
- 下一篇：[论文证据](papers.md)
- 回到：[目录 README](README.md) | [ai-data 边界](08_ai_data_boundary.md)

> 串联：归因基础 → 梯度/ICL 两类目标化估值 → 集合覆盖与冲突 → 生成与持续学习闭环
