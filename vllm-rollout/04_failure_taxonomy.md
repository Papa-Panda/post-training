# 04 — Failure taxonomy and response

A useful taxonomy is mutually exclusive at the **attempt** level, while preserving contributing signals. Do not call every dropped trajectory an engine failure.

| Class | Boundary | Example evidence | Retry guidance |
|---|---|---|---|
| `client_transport` | client, proxy, network | connection reset, client deadline | retry only if request idempotency and budget allow |
| `admission_reject` | validation/admission | context length exceeds limit, malformed request | do not retry unchanged |
| `queue_timeout` | before or during engine service | oldest queue age, server/client deadline | retry with backoff only if load has changed |
| `kv_preemption` | scheduler/cache pressure | preemption counter delta, recompute-token delta, high cache usage | not automatically fatal; tune capacity/concurrency |
| `worker_failure` | engine process or device rank | worker exit, health check failure, collective error | restart/fail over; retry with attempt lineage |
| `tool_environment` | external action execution | tool timeout, sandbox crash, invalid tool response | policy-dependent; distinguish deterministic from transient |
| `response_invalid` | generated protocol/content | parse error, missing required fields | possibly regenerate; count generated waste |
| `scorer_verifier` | reward/scoring path | scorer timeout, verifier crash, unavailable tests | preserve rollout; rescore when possible |
| `freshness_drop` | learner acceptance gate | policy lag beyond configured maximum | do not hide as model-quality failure |
| `cancelled` | orchestration | run cancelled after target batch completed | normally no retry |
| `unknown` | unclassified | missing evidence | page on rising rate; never redistribute silently |

## Attempt identity and lineage

Each record should include:

```json
{
  "rollout_id": "stable logical id",
  "attempt_id": "unique execution id",
  "parent_attempt_id": null,
  "policy_version": 41,
  "engine_instance": "replica-2",
  "prompt_tokens": 512,
  "generated_tokens": 933,
  "finish_reason": "stop",
  "failure_class": null,
  "accepted": true
}
```

Do not reuse an `attempt_id` after retry. Otherwise token cost, duplicate delivery, and policy age cannot be reconstructed.

## Diagnostic decision tree

1. **Was the request admitted?** If not, classify validation or admission rejection.
2. **Did the client lose the response?** Cross-check server completion by rollout/attempt ID before retrying.
3. **Did engine work begin?** Compare admission/first-scheduled timestamps and queue age.
4. **Did cache pressure occur?** Inspect version-correct KV usage, preemption, and recomputation counters plus logs.
5. **Did generation finish?** Preserve finish reason and token counts.
6. **Did tool or scorer fail after generation?** Attribute failure to that boundary while retaining generated-token cost.
7. **Was the sample dropped for staleness or filtering?** Mark it as completed-but-not-accepted.

## Common mistakes corrected

- Paged KV allocation invalidates a generic “contiguous free span” diagnosis. Do not infer fragmentation from fabricated internals.
- A preemption is a performance event, not necessarily a failed request.
- An empty or truncated output can be a stop configuration, transport truncation, or client parser error—not automatically an out-of-memory event.
- P95 greater than a multiple of P50 is a tail symptom, not a timeout definition.
- Enabling CPU swap or changing eager execution is not a universal remedy; both are version- and workload-dependent experiments.
- A verifier rejection must not be mixed with transport or engine availability when computing infrastructure reliability.

## Retry budget

Bound retries by both attempts and tokens:

$$retry\ amplification = \frac{attempted\ generation\ tokens}{first\ attempt\ generation\ tokens}$$

Use exponential backoff with jitter only for transient classes. Admission errors and deterministic invalid requests should fail fast. Keep a deduplication key at the downstream dataset boundary so a client timeout does not create duplicate training examples.
