# Evidence ledger

[← README](README.md)

Checked against the linked primary arXiv records on 2026-09-01. “Supports” means only the scoped statement below; it is not a license to transfer a result to arbitrary models, tasks, or scales.

| Work | Evidence | Supports here | Does **not** establish |
|---|---|---|---|
| [Brown et al., *Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) | `[E-model]` | GPT-3 was evaluated in zero-, one-, and few-shot text settings without task-specific gradient updates or fine-tuning. | A mechanism for ICL, or that few-shot always improves over zero-shot. |
| [Xie et al., *An Explanation of In-context Learning as Implicit Bayesian Inference*](https://arxiv.org/abs/2111.02080) | `[T-bound] [E-synthetic]` | Mixtures of HMMs can induce Bayesian-style latent-concept inference; synthetic GINC experiments probe order and zero/few-shot effects. | Every Transformer literally computes a normalized posterior; IID finite-concept equations are the paper's full model. |
| [Min et al., *Rethinking the Role of Demonstrations*](https://arxiv.org/abs/2202.12837) | `[E-model]` | In studied classification/multiple-choice tasks across 12 models, replacing labels with random labels barely hurt; label space, input distribution, and format mattered. | Labels never matter, or the result transfers unchanged to generation/repository tasks. |
| [von Oswald et al., *Transformers Learn In-Context by Gradient Descent*](https://arxiv.org/abs/2212.07677) | `[T-construct] [E-synthetic]` | A linear self-attention construction can implement one linear-regression GD step; trained synthetic models show related behavior. | Every layer is exactly one GD step in pretrained LLMs. |
| [Akyürek et al., *What Learning Algorithm Is In-Context Learning?*](https://arxiv.org/abs/2211.15661) | `[T-construct] [E-synthetic]` | Transformer constructions can implement GD and closed-form ridge; trained synthetic predictors can resemble different estimators depending on regime. | A unique optimizer describes all ICL or natural-language tasks. |
| [Garg et al., *What Can Transformers Learn In-Context?*](https://arxiv.org/abs/2208.01066) | `[E-synthetic]` | Transformers trained on synthetic function classes learn in-context predictors that match or exceed task-specific baselines in studied settings. | A fixed correlation to a GD trajectory, or a mechanism claim for coding LLMs. |
| [Olsson et al., *In-context Learning and Induction Heads*](https://arxiv.org/abs/2209.11895) | `[E-model] [C-causal]` | Previous-token plus induction-head circuits implement `[A][B] ... [A] -> [B]`; small attention-only models receive strong causal evidence, larger MLP models mainly correlational evidence. | Induction heads are the sole cause of all ICL, or chain-of-thought is an induction circuit. |
| [Hendel et al., *In-Context Learning Creates Task Vectors*](https://arxiv.org/abs/2310.15916) | `[E-model] [C-causal]` | In studied settings, demonstration information can often be compressed into a query-agnostic task vector and intervened on. | One universal vector exists for every task/model or is identical to an induction head. |
| [Todd et al., *Function Vectors in Large Language Models*](https://arxiv.org/abs/2310.15213) | `[E-model] [C-causal]` | Causal mediation identifies a small number of heads carrying compact function-vector effects, strongest in middle layers in studied settings. | These heads implement Bayesian normalization or gradient descent. |
| [Shinn et al., *Reflexion*](https://arxiv.org/abs/2303.11366) | `[E-model]` | Verbal feedback/reflection is stored in episodic memory and reused without updating base-model weights. | Reflection text is always correct, or retrieval is free of stale/negative-transfer effects. |
| [Zhang et al., *In-Context Principle Learning from Mistakes (LEAP)*](https://arxiv.org/abs/2402.05403) | `[E-model]` | Paper abstract reports GPT-4 gains of 7.5% on DROP and 3.3% on HotpotQA while using no more input examples than standard few-shot prompting. | General coding-agent gains or a universal percentage improvement. |
| [Zhao et al., *ExpeL*](https://arxiv.org/abs/2308.10144) | `[E-model]` | Collects successful/failed experiences, extracts cross-task insights, and recalls them during later tasks. | Extracted insights are automatically causally valid. |
| [Fu et al., *AutoGuide*](https://arxiv.org/abs/2403.08978) | `[E-model]` | Extracts conditional, context-aware guidelines from offline trajectories and dynamically selects relevant ones. | More guidelines monotonically improve performance. |
| [Ding et al., *CYCLE*](https://arxiv.org/abs/2403.18746) | `[E-model]` | Trains code models to refine faulty generations using feedback such as execution/test results on studied code benchmarks. | Repository-level repair quality or improvement without a reliable verifier. |
| [Chen et al., *Teaching Large Language Models to Self-Debug*](https://arxiv.org/abs/2304.05128) | `[E-model]` | Few-shot prompting can teach code explanation/execution-feedback-based self-debugging in the reported benchmark settings. | Self-generated explanations certify correctness. |
| [Pan et al., *Training Software Engineering Agents and Verifiers with SWE-Gym*](https://arxiv.org/abs/2412.21139) | `[E-model]` | SWE-Gym v2 reports 2,438 executable real-world Python task instances and experiments training agents and trajectory verifiers. | Those counts describe every release/version, or its reported ranking remains current. |
| [Xiao et al., *Socratic-SWE*](https://arxiv.org/abs/2606.07412) | `[E-model]` | Trace-derived skills guide targeted repository repair-task generation; candidates receive execution validation and solver-gradient alignment; abstract reports 50.40% SWE-bench Verified after three iterations. | A 2025 result, a universal SOTA claim, or that the closed loop is equivalent to ordinary SFT. |

## Evidence labels

- `[T-construct]`: an explicit mathematical construction or representational result.
- `[T-bound]`: a theorem or bound under stated assumptions.
- `[E-synthetic]`: empirical evidence on controlled synthetic tasks.
- `[E-model]`: empirical evidence on trained language models or agents.
- `[C-causal]`: ablation, activation patching, or another causal intervention.
- `[H]`: a hypothesis or extrapolation proposed here, not established evidence.

When a chapter makes a claim stronger than this ledger, narrow the chapter rather than broadening the citation.
