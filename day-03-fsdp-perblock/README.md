# Day 3 - FSDP per-block 显存 + profiler 对应 ai_daily 2026-08-04：Code data flywheel 是主线，今天用 FSDP per-block 作为量化切入，补 infra 可见性。 ## 内存模型
- DDP 每卡常驻 4P (param + grad + 2×adam)
- FSDP 常驻 4P/G
- 峰值 = (P-b)/G + b + (grad+opt)/G
- b = 一个 block 的参数，越小峰值越低，但启动次数 = 2*num_blocks
- per-transformer-block 是甜点 7B 例子 (fp32 P=28GB, bf16混 14GB): - 2×A100 80GB: 常驻 ~42-56GB, 峰值 ~42.5GB bf16 mix → 可塞下
- 结论写进 infra note: “FSDP per-block可把7B塞进2×A100可跑eval” ## Profiler 3行
```python
prof = torch.profiler.profile(activities=[CPU,CUDA], record_shapes=True)
prof.__enter__
...
prof.__exit__
print(prof.key_averages.table(sort_by="cpu_time_total"))
``` - CPU gloo 环境下 gloo:all_gather 显示为 CPU op，时间不能当 NCCL 通信量 → 标注“待H100验证”
- GPU 上再区分 forward all-gather vs backward reduce-scatter 占比 ## Demo 结果 (CPU gloo 2-rank, 2026-08-04 PDT)
- FSDP per-block ok (b1,b2,root)
- epoch 0 2.318 / epoch 1 2.142
- profilerTop: Optimizer.step 39.7% / FSDP::all_gather (0) 14.5% (32 calls, 2.6ms avg) / gloo:all_gather 69ms 64 calls
- 注: CPU下 all_gather = memcpy，不能当 NCCL 读 ## 5行 infra note (copy到每日问题库)
1. 链路：产 X 类 coding 数据 50k → 训 Y 7B
2. 评测：eval A73%→B78%
3. 成本：tokens/sec 1.2k, GPU-hour $3.2, 失败率12%
4. 瓶颈：rollout 约占80%墙钟
5. 动作：FSDP per-block + vLLM rollout 可把7B塞进2×A100省Z小时bad case / 可跑eval
