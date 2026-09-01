# NOTES — Large-model FSDP Capacity Planning

## Audit correction

The earlier version presented unmeasured throughput, failure-rate, communication-percentage, and fit ranges beside analytical values. It also described a script as “ready for H100,” although selecting `--model 7b/13b/70b` instantiated the nominal full-size proxy and the activation-checkpoint flag was unused.

Those claims were removed rather than relabeled as predictions.

## Retained result

The retained tool is a dependency-free byte-accounting model with explicit state precision:

$$M_{resident}=P(b_p+b_g+b_o+b_m)/G.$$

It adds only the missing parameter shards for the largest materialized wrapped unit. The output calls itself a lower bound and lists omitted memory categories.

Semantic tests verify:

- the default mixed-precision Adam example is 16 bytes/parameter;
- 7B unsharded state is 112 decimal GB under that assumption;
- resident state scales as $1/G$;
- removing a master copy changes the accounting to 12 bytes/parameter;
- largest-unit materialization adds only the non-resident parameter shards;
- invalid parameter/rank/layer inputs fail.

## Removed artifacts

- `README.md.tmp`: tracked placeholder with no technical content;
- local `vllm_rollout_stress_test.py`: byte-identical duplicate of the copy under `day-10-vllm/code/`;
- the synthetic evaluation delta and all unmeasured H100 throughput/failure distributions.

The historical `infra_note_latest.md` now contains a deprecation notice rather than a second source of estimates.
