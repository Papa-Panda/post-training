# 03 Hermes Harness — ahuachen/hermes-compression-eval

> Repo: https://github.com/ahuachen/hermes-compression-eval — offline probe-based harness for `hermes-agent`'s `ContextCompressor`

## 架构

```
real conversation fixture (jsonl, ~80K)
  ├─ compress(): ContextCompressor (from hermes-agent, python module)
  ├─ probes.yaml (30 Q per fixture, with gold + dimension tag)
  ├─ judge_prompt (6 dims)
  └─ evaluator.py → score.csv + markdown report
```

图示 (照搬 repo README):

- Fixture: SWE-like long trajectory, multi-turn tool_use
- Compress: calls `compressor.compress(messages)` — no LLM in loop optional, can be extractive
- Probe: from compressed memory, ask LLM 30 Qs
- Judge: GPT-4 / Claude as judge, 0-5 per dim

## 安装 & Env

```bash
git clone https://github.com/ahuachen/hermes-compression-eval
cd hermes-compression-eval
pip install -r requirements.txt

# Point to your hermes-agent checkout
export HERMES_AGENT_ROOT=/path/to/hermes-agent
# OPENAI key for probe answering + judge
export OPENAI_API_KEY=sk-...
python evaluator.py --fixture fixtures/swe_long_01.jsonl --probes probes/swe_long_01.yaml
```

`HERMES_AGENT_ROOT` env 是关键 — harness 不自带 compressor，而是 `import` 你的实现，这样同一 harness 可测不同 compressor。

## 成本 & 可复现

- 单 fixture: 1 compress + 30 probes answering + 30 judge = ~61 LLM calls
- 如果用 gpt-4o-mini / claude-haiku，~$0.30 / fixture
- Repo 自带 3 fixtures，跑全 $1 以内；scale 到 50 fixtures → $15 左右，CI 友好

## Signal 是什么？

Output `report.md` 示例：

```
| probe_id | dimension | question | gold | answer | score | reason |
|----------|-----------|----------|------|--------|-------|--------|
| 7 | artifact_trail | What files were edited in round 2? | [app.py, util.py] | app.py | 3/5 | missed util.py |
```

Average 6 dims:

```
accuracy: 4.2/5
context_awareness: 3.8
artifact_trail: 2.9  <- 弱点
completeness: 3.5
continuity: 4.0
instruction_following: 4.4
```

From this you know: current compressor is extractive and drops diffs → artifact weak → need to keep tool output verbatim or add file-change log side channel.

Factory's key point: **Failure mode localization** > single number.

## 我们可抄的点

- Toy 版我们做 `toy_compressor_eval.py` (see code/): 不依赖 hermes-agent，用 naive summarizer 模拟
- 真实版：把 vLLM rollout 里的 long context log 转成 fixture，直接塞 harness，30 probes 手写 5 个先跑通
- 与 GLM-5.2 token-level PPO 对比：Hermes 的 token-level advantage 需要好压缩，压缩不好 advantage 噪声大

## Limitations

- Judge LLM bias: 用 GPT-4 judge 自己的压缩可能 inflate
- Probe coverage: 30 Q 不一定覆盖长尾 (e.g. security constraint)
- No online eval: offline probe ≠ online rollout 断点续跑能力，但 Factory 认为离线先行已能过滤 80% bad compressor
