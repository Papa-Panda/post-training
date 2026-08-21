# Day 07-10 NOTES — H100 >7B 扩展

Date: 2026-08-10 21:50 PDT (cron bigger-gpu-task-reminder)
Status: done (CPU 可验证逻辑，H100 NCCL 计时待执行)
Context: infra_note_2026-08-07_tbd_h100.md 只有 7B 占位，需要补 13B/30B/70B + tokens/sec + 失败率分布

## 3 numbers (theoretical,待H100替换)

1. 7B G=2 H100 bf16-mix：常驻 ~28GB/G，峰值 ~42.5GB，comm 40-55%（fwd all-gather 18-25% + bwd reduce-scatter 22-30%）
2. 13B G=4：峰值 ~38GB，comm 55-65%，tokens/sec per GPU ~2.3-2.9k (seq 4k)
3. 70B G=8：峰值 ~86GB (activation ckpt 需开)，comm 45-60%，tokens/sec per GPU ~0.6-0.8k (seq 2k)

Rollout vLLM：
- 7B 短 CoT 500 tok：40-60k tokens/sec decode，fail 5-8%
- 7B 长 CoT 5000 tok：8-15k tokens/sec，fail 12-18%（超时40%/工具30%/VCJ15%/OOM10%/NCCL5%）

## What I did

- 从 Day2/3 FSDP per-block 结论出发：峰值 = (P-b)/G + b + (grad+opt)/G，省大头是 optimizer 2× 全片
- 写了 fsdp_h100_profiler_beyond7b.py：支持 small proxy model + 真 7B/13B/70B config（bf16-mix 估算），torchrun 2/4/8 rank，profiler 计 all-gather / reduce-scatter
- 写了 vllm_rollout_stress_test.py：短/长 CoT 各200样本，分类5类失败，记P50/P95 arrival
- 补 infra_note 新版：infra_note_2026-08-10_h100_beyond7b.md，含 7B→70B 外推表 + tokens/sec + fail分布 + $/有用rollout换算

## Tradeoff / Trap

- per-layer：太细，startup 炸，latency bound
- per-model：太粗，峰值回 DDP，放不下 13B+
- per-block：sweet spot，comm overlap compute，H100 上更需 `limit_all_gathers` / `use_orig_params=True`

陷阱：
- CPU gloo 2.6ms 不能误读成 NCCL 通信（Day3已加 `if torch.cuda.is_available()` 判断）
- FSDP `max_memory_allocated` 看的是 block 非整模型，峰值瞬间完整是 block
- vLLM 长 CoT KV cache evict 会被误判为模型失败，实际是调度失败，需分开计数

## Next

- [ ] H100 上 `torchrun --nproc_per_node=2 fsdp_h100_profiler_beyond7b.py --model 7b` 记 forward all-gather% / backward reduce-scatter% / peak_mem
- [ ] H100 上 `... --model 13b --gpus 4` 同上
- [ ] vLLM rollout 7B 跑 200 短+200 长，补 rollout P95 arrival 图 + fail 5类
- [ ] 把实测数贴回 infra_note_latest.md 替换 TBD，保留 "CPU 理论过" 标注链路

Raw logs / ckpt：
- 本机 CPU：fsdp_day2 0.5s avg_loss 2.318→2.142, ckpt /tmp/fsdp_day2_ckpt.pt ok（Day2 NOTES）
- 本机 CPU：Day3 profiler all_gather 32×2.63ms / gloo 64×1.08ms（CPU理论，待H100）
- H100：待执行，代码已备
