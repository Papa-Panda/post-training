# NOTES — r2-Day12 Profiling（Nsight 双剑）

## 准确术语

- **Nsight Systems（`nsys`）**：全系统性能分析。采集时间线：CUDA API 调用、kernel launch 与执行、H2D / D2H memcpy、NCCL collective、OS runtime、NVTX 标记区间。回答"时间去哪了"。tracing 开销相对低，是 profiling 的第一站。
- **Nsight Compute（`ncu`）**：kernel 级 profiler。对**单个** kernel 做 kernel replay（为采 counter 把同一个 kernel 跑很多遍，每遍采一组 counter），输出 HW PM counter 的 section。回答"这个 kernel 为什么慢"。开销大、扰动 timing——它的耗时数字**不能**当 wall-time 用。
- **SpeedOfLight（SOL）**：ncu 的 section，对每个 GPU 单元（Compute/SM、DRAM、L1/TEX、L2…）报告 achieved throughput 占理论峰值的百分比。memory-bound kernel 的**正常形状**是 DRAM SOL% 高、SM SOL% 低——这是 bound 的签名，不是 bug。
- **GPU Speed Of Light Roofline Chart**：ncu section（名 `SpeedOfLight_RooflineChart`，`--set detailed` / `--set full` 自带）：被 profile kernel 的 roofline 图，x 轴 AI、y 轴 achieved FLOP/s。
- **Kernel replay**：ncu 的采集方式——replay 每个被 profile 的 kernel dispatch 很多次。后果：profile 时间远大于运行时间；不能用 ncu 的时间做 benchmark。
- **NVTX**：标注 API（`nvtx.range_push/pop`，或经由 torch profiler 的挂钩），用来在 Systems 时间线上标出区间（比如一个 train iter），否则几千个 kernel 看不出结构。
- **Tracing overhead / perturbation（扰动）**：测量改变被测对象。Systems 加 API 拦截开销，ncu replay 改变缓存状态。所以：先 Systems 后 Compute；正确性没跑通之前不 profile。
- **Arithmetic intensity / ridge point / roofline**：Day06 术语复用——是**模型预测**，不是测量。SOL% 才是检验模型的测量。

## Connection to Prev 的实质

Day06 给了理论 roofline（$I^*=20$ FLOP/byte，vector add 只能摸到带宽屋顶）；Day11 给了 fused 的流量账（$6N\to2N$，theoretical estimate）。Day12 把"理论"和"测量"接起来：

1. **Systems 验证 fuse 真生效**：时间线上 kernel 个数从 4 个变成 1 个；`--trace=cuda,nvtx,osrt` 抓一次 train iter，看 GPU busy% 和 gap 归因。
2. **Compute 验证 bound 判断**：fused kernel 的 SpeedOfLight 应该呈现 memory-bound 形状（DRAM SOL% 高、SM SOL% 低），roofline 点落在带宽屋顶附近——如果 Day06 的模型是对的。
3. 顺序不能反：全系统问题（dataloader、NCCL、launch 开销）先用 Systems 看，ncu 只打 Systems 定位到的热点 kernel。

## 可手算小例子（N=8 FP32 vector add，H100 SXM）

$bytes=96$，$flops=8$，$AI=1/12\approx0.0833$；$I^*=67/3.35=20$ → memory-bound；

$$P = 0.0833\times3.35 = 0.2792\ \text{TFLOP/s},\quad SOL_{flops}=0.2792/67=0.417\%,\quad t_{min}=96/3.35\times10^{12}\approx28.7\ \text{ps}$$

对照组（compute-bound）：$flops=800$、$bytes=8$ → $AI=100$，$SOL_{flops}=100\%$，$SOL_{bw}=20\%$。两组形状都进了测试（`test_sol_report_vecadd_n8`、`test_compute_bound_contrast`）。

FP16 tensor ridge：$989/3.35\approx295.2$ FLOP/byte——Tensor Core 把 ridge 推高约 15 倍，喂饱它需要多得多的片上复用（呼应 Day06 的 Tensor Core 备注）。

## 命令速查（参考，未在本环境执行）

```bash
# Systems：先看时间线（全系统，低开销先行）
nsys profile -o train_iter --trace=cuda,nvtx,osrt --force-overwrite=true ./train.py
# 用 nsys-ui 打开 train_iter.nsys-rep：看 GPU busy%、kernel 个数、gap 归因

# Compute：再看单个 kernel 的 SOL（replay，高开销后行）
ncu --set detailed --section SpeedOfLight --section SpeedOfLight_RooflineChart -o kernel_sol ./app
ncu --set full -o kernel_full ./app   # 全量深挖，只打热点 kernel
```

本课 `nsys_ncu_recipes.py` 把上面三条存成了 argv 构造器 + `tools_present()` 探针；测试断言了 argv 内容与本环境双工具缺失。**命令本身 execution not validated / 待H100验证。**

## 何时不赚（再强调一次）

1. 正确性没跑通就 profile——量的都是错的东西。
2. 只看 SOL% 不看端到端——热点 kernel 的 DRAM 100% 可能只占 step 的 1%。
3. 小 kernel 上 ncu 的 % 外推到大 shape——replay 放大的是固定开销。
4. profile 一次下结论——时钟/缓存/系统噪声都会抖，看分布。
5. 拿 ncu 去查全系统问题——dataloader 慢、NCCL hang 是 Systems 的活。

## 与 Day13 的连接

Day13（Attention 变种 MHA/MQA/GQA/MLA/MoE）："省 KV" 的 claim 最终也要过这一关——Systems 看 decode 阶段的 GPU busy%（memory-bound 的 decode 是否被 KV 搬运拖住），Compute 看 attention kernel 的 DRAM SOL% 是否随 KV 缩小而下降。profiler 是尺子，架构是被量的对象。
