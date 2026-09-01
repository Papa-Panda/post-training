# 08 ICL 的系统成本与评测协议

[← 07 Coding data](07_coding_data.md) · [回到 README](README.md) · [证据账本](references.md)

ICL 把训练成本换成推理期上下文成本。shot 数 $k$ 不是系统变量；真正决定成本的是 token 数 $n$、模型结构、batching、cache 策略和输出长度。

## 1. Prefill 与 decode 分开记账

令层数为 $L$，query heads 为 $H_q$，KV heads 为 $H_{kv}$，head dimension 为 $d_h$，每元素字节数为 $b$，prompt 长度为 $n$。以下均按**单条未 padding 序列**计；batch size 为 $B$ 时再乘 $B$，但实际服务还受 padding、paged allocation 与并行切分影响。

密集 attention 的 prefill 算术量随 $n^2$ 增长：

$$C_{\mathrm{prefill,attn}}\propto L H_q n^2 d_h$$

朴素实现若显式保存每层每头的 attention score，存储上界为：

$$M_{\mathrm{scores}}=L H_q n^2 b$$

memory-efficient attention kernels 可以避免完整 $n\times n$ score materialization，但不自动消除 dense attention 的二次算术量。

KV cache 存 keys 与 values，因此：

$$M_{\mathrm{KV}}=2L n H_{kv}d_hb$$

它对 $n$ 线性增长。生成一个新 token 的 attention 需要读取历史 cache，成本也随当前上下文长度增长；因此只报“每秒输出 token”会掩盖 many-shot 的 prefill 成本。

[`kv_cache_bytes`](icl_mechanisms.py) 与 [`attention_score_bytes`](icl_mechanisms.py) 可做维度 sanity check。它们不包含 allocator、padding、quantization metadata、MLP activations、logits 或 kernel workspace。

## 2. Prefix caching 的边界

当多个 query 共享完全相同的 demo prefix 时，可复用 prefix KV cache，摊薄 prefill。以下变化通常会破坏或缩小复用：

- demo 顺序变化；
- 动态 system/guideline 拼接；
- tokenizer 或 chat template 变化；
- 每个请求插入不同 repository context；
- cache key 未包含模型版本或位置编码相关配置。

评测时要区分 cold-prefix 与 warm-prefix，否则线上收益无法复现。

## 3. 质量实验：最小因子设计

固定模型、解码参数、数据 split 与 verifier，交叉以下变量：

- shots：$k\in\{0,1,2,4,8,\ldots\}$，同时记录真实 $n$；
- selector：random、similarity、coverage、oracle；
- order：至少多个随机 permutation，加 best/worst order 诊断；
- labels：correct、shuffled、format-only；
- rules：none、retrieved、matched-token irrelevant、oracle；
- budget：固定 shot 与固定 token 两种实验都做。

对每个 cell 报：

```text
pass@1 / task score
calibration or confidence
prompt tokens / generated tokens
prefill latency / time-to-first-token
decode latency or tokens-per-second
peak device memory / KV-cache estimate
retry count / tool calls / verifier time
```

## 4. 统计与污染控制

- 以 task 为配对单位，保留每个 task 的 baseline 与 treatment 差值。
- order、sampling seed 与 executor nondeterminism 分开记录。
- repository family、near-duplicate patch、issue 模板必须 group split。
- 先冻结 task set 与 metric，再查看 treatment 结果。
- 同时报平均数、分位数和失败类型；只报总体平均会隐藏长尾 latency 与 negative transfer。

## 5. 何时停止增加示例

不要假设性能随 $k$ 单调。可用受预算约束的选择目标：

$$D^*=\arg\max_{D\subseteq\mathcal P}\left[\widehat U(D)-\lambda_n n(D)-\lambda_t T(D)-\lambda_m M(D)\right]$$

其中 $\widehat U$ 是 held-out utility estimate，$n,T,M$ 分别为 token、latency、memory。工程上的停止条件可以是：新增 demo 的置信区间内收益不再覆盖其成本，或 negative-transfer rate 超过预设阈值。

## 6. 与三条机制的接口

- [Bayesian](02_line_I_bayesian.md)：看 coverage、冲突证据与 calibration 是否符合 posterior 直觉。
- [GD](03_line_II_gd.md)：看复制、置换、符号翻转与 query-demo similarity 响应。
- [Circuit](04_line_III_circuit.md)：看 token-level probe 与 causal intervention，而不是只看 attention 图。
- [Trajectory rules](06_trajectory_error_prompt.md)：把检索成本、规则误触发与 staleness 算入总账。

系统数据本身也能反驳机制故事：若更长 prompt 的收益完全来自更多 compute，而 matched-token control 同样提升，就不能把提升直接归功于示例语义。
