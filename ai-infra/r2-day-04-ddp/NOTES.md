# NOTES — r2-Day04 DDP

## Audit corrections

- Dataset generation now uses one deterministic seed on every rank before `DistributedSampler`; the previous rank-dependent seed created different datasets.
- Correctness checks compare initial and post-step parameters, not final rank-local minibatch losses.
- The demo uses `torchrun` environment variables rather than positional rank arguments.
- Checkpointing is rank-0 only, followed by a barrier before process-group teardown.
- Gloo is selected on CPU and NCCL when CUDA is available.
- Bucket size is described as a tuning parameter rather than a timeless fixed default.

## Evidence available here

Seven dependency-free tests cover sampler ownership and padding, averaged-gradient semantics, replicated parameter updates, ring traffic, and invalid inputs.

PyTorch is not installed in the audit environment, so the two-process integration path was syntax-checked but not executed. No GPU performance claim follows from the pure-Python tests.
