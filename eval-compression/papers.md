# Papers — Eval Compression / Context Management

## Core track

- **Factory AI "Evaluating Compression" — Dec 2025**  
  `https://factory.ai/news/evaluating-compression`  
  Defines agent context compression eval vs prompt vs KV-cache. Introduces 6-dim probe method (accuracy, context_awareness, artifact_trail, completeness, continuity, instruction_following). Scoreboard framing explicitly rejected. Cost: ~60 LLM calls / fixture. Foundation for this folder.

- **Hermes Compression Eval Harness**  
  `https://github.com/ahuachen/hermes-compression-eval` —  MIT license, Python, offline probe harness. Requires `HERMES_AGENT_ROOT`. 30 probes / fixture, 0-5 judge. Adopted in `03_hermes_harness.md`. Reference impl for our toy version.

## Prompt / Context Compression

- **Prompt Compression in the Wild — 2025.04** [arxiv:2604.02985]  
  `https://arxiv.org/abs/2604.02985`  
  LLMLingua-style entropy filtering in production RAG. When operating window matches prefill window, end-to-end speedup up to 18%, memory offload enables larger batch. Shows latency vs quality tradeoff curve for prompt comp. Used in `04`.

- **LLM Eval Compression Tradeoffs — 2024.09** [arxiv:2409.11233]  
  `https://arxiv.org/abs/2409.11233`  
  Study: magnitude pruning, SparseGPT, Wanda preserve perplexity at 50% sparsity but degrade downstream 8-15%. Proposes JS divergence, downstream preservation as better metrics than ppl. Core argument for `05_metrics.md`.

- **Language Modeling is Compression (Chinchilla claim) + Compression as Evaluation — 2024.02** [arxiv:2402.00861]  
  `https://arxiv.org/abs/2402.00861v1`  
  Formalizes LM eval via compression (MDL). $L = -\log P(data)$ ↔ codelength. Inspiration but for LM, not agent memory. Factory explicitly departs from this.

## Long-horizon RL connection (why eval matters)

- **GLM-5.2 Technical Report — 2026.06** Z.ai / Zhipu  
  `https://z.ai/blog/glm-5` (blog), `https://arxiv.org/abs/2602.15763` (report)  
  744B MoE 40B active, 1M context Index Sharing, **abandons GRPO for PPO in long-horizon RL** due to sub-trajectory length variance making group comparison impossible. Uses SLIME async RL infra. Parallel to our context compression variance problem. Discovered in `grpo-vs-ppo/`.

  Key quote: *"After compression, the number and length of sub-trajectories become highly uneven. This strikes directly at GRPO's weak point: it requires grouping outputs from same prompt."* 【BigGo Finance】

- **Learning Without Critics? Revisiting GRPO — 2025 late**  
  Controlled study: GRPO (no critic) underperforms PPO with value in long-horizon no-early-termination tasks, only parity in CartPole short tasks. Academic support for GLM-5.2 switch.

## Related infra

- **TurboQuant / LeanKV / H2O** — KV-cache compression papers (for `04` comparison)
- **DeepSeekMath GRPO — 2024.02** [arxiv:2402.03300] origin of GRPO — group mean/std baseline no critic, saves 30-50% mem.
- **DAPO, Dr.GRPO, GSPO, GMPO** — GRPO variants trying to fix stability, still group-based, academia continues while industry (Z.ai) turns back.

---

## Reading order suggested

1. Factory article → understand *what* to eval
2. Hermes harness README → how to run it
3. 2604.02985 → prompt comp latency story
4. 2409.11233 → why ppl misleading
5. GLM-5.2 report sec 3 RL → why compression eval affects RL algorithm choice
