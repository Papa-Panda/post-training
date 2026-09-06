# r2-Day12 — Profiling：Nsight 双剑（Systems 看拖后，Compute 看 SOL）

## Connection to Prev

r2-Day11 把 softmax 从 eager 4 遍 fuse 成 1 个 kernel，流量模型说 HBM 往返 $6N\to2N$ （theoretical estimate）。但"省下的流量变成 wall-time 了吗"——流量模型回答不了。r2-Day12 给出验证的仪器：**Nsight Systems** 看时间线（kernel 个数是不是真从 4 个变成 1 个、host 端有没有 launch / recompile 拖后），**Nsight Compute** 看单个 kernel 的 Speed-of-Light（DRAM / SM throughput 离理论峰值差多远、roofline 落在哪）。**牺牲**：profiling 扰动被测对象——Systems 的 tracing 有开销，Compute 对每个 kernel replay 很多遍采 counter（时间被放大，不能直接当 wall-time）；**换取**：把"我觉得快了"变成可复核的测量（GPU busy%、SOL%、gap 归因）；**何时不赚**：正确性还没跑通就 profile；只盯单个 kernel 的 SOL%、不看端到端；小 kernel 上 ncu replay 几十遍的数字外推到大 shape；profile 一次就下结论（忽略噪声）。

## 1. 双剑分工：一句话

- **Nsight Systems（`nsys`）**：回答"时间去哪了"。全系统时间线：CUDA API 调用、kernel launch 与执行、H2D / D2H memcpy、NCCL、OS runtime、NVTX 标记区间。开销相对低，**先用它**看 GPU 有没有在干活、idle gap 的来源是 host 还是 device。
- **Nsight Compute（`ncu`）**：回答"这个 kernel 为什么慢"。对**单个** kernel 做 kernel replay（为采 counter 把同一个 kernel 跑很多遍，每遍采一组 counter），输出 PM counter section：**SpeedOfLight**（DRAM / L1 / L2 / SM 各单元 throughput 占理论峰值的百分比）、**SpeedOfLight_RooflineChart**（roofline 图）、MemoryWorkloadAnalysis、Occupancy 等。代价高、扰动大，所以流程是：**先 Systems 定位，再 Compute 深挖**。

## 2. 可手算例子 A：SOL（N=8 FP32 vector add）

Day06 的同一算子，N 缩到 8——AI 是 scale-free 的，bound 不变，数字小到可以手算。

- 流量：读 x、读 y、写 z → $bytes = 3\times8\times4 = 96\ \text{B}$ ； $flops = 8$ 。

$$AI = 8/96 = 1/12 \approx 0.0833\ \text{FLOP/byte}$$

- H100 SXM：FP32 峰值 67 TFLOP/s，HBM 3.35 TB/s（Day06 规矩：十进制前缀）→ ridge：

$$I^* = 67/3.35 = 20\ \text{FLOP/byte}$$

- $AI \ll I^*$ → memory-bound；roofline 可达：

$$P = AI \times 3.35 = 0.2792\ \text{TFLOP/s}$$

- SOL（相对 FP32 峰值）：$$0.2792/67 = 0.417\%$$——这就是 ncu SpeedOfLight 里"SM throughput ≈ 0.4%、DRAM ≈ 100%"的含义：**不是 kernel 写坏了，是算子本身只能摸到带宽屋顶**。
- 理论最短时间：$$t_{min} = 96/3.35\times10^{12} \approx 28.7\ \text{ps}$$——ps 级！真实 kernel 光 launch 就是 µs 级（示意量级，非测量）：**小 N 时 Systems 会告诉你 GPU 几乎全程 idle，瓶颈全在 host 端**。这正是 Day11"小 N 不赚"的测量版表述。
- 对比（compute-bound 长什么样）： $flops=800$ 、 $bytes=8$ → $AI=100 > 20$ → compute-bound，SOL_compute = 100%，SOL_bw = 20%（ $=I^*/AI$ ）， $t_{min}=800/67\times10^{12}\approx11.9\ \text{ps}$ 。ncu 里看到 SM 100%、DRAM 20% 就是这个形状。

## 3. 可手算例子 B：Systems 时间线（N=2^20 softmax，eager vs fused）

把 Day11 的 $6N\to2N$ 接上时间。模型参数（**非测量**）：单次 host launch 开销 $L=5\ \mu s$ （示意值）；kernel 时间取 roofline 下限 $bytes/3.35\ \text{TB/s}$ 。

- Eager（4 个 kernel，总流量 $6N$ 元素 $= 25{,}165{,}824\ \text{B}$ ）：kernel 时间 $7.512\ \mu s$ ； $span = 4\times5 + 7.512 = 27.512\ \mu s$ ；GPU busy $= 7.512/27.512 = 27.3\%$ ；4 段 idle gap，每段 $5\ \mu s$ ，归因 host launch。
- Fused（1 个 kernel，总流量 $2N$ 元素 $= 8{,}388{,}608\ \text{B}$ ）：kernel 时间 $2.504\ \mu s$ ； $span = 5 + 2.504 = 7.504\ \mu s$ ；GPU busy $= 33.4\%$ 。
- 模型加速比： $27.512/7.504 \approx 3.67\times$ ——流量省 $3\times$ ，wall-time 省 $3.67\times$ （还省了 3 次 launch）。**但这是模型**：kernel 时间取的是 roofline 下限、launch 取的是示意值，真实比例必须用 nsys 量。
- Systems 视角的读法：先看 GPU busy%（有没有在干活）→ 再看 kernel 个数（4→1，验证 fuse 真生效）→ 再看 gap 归因（host launch / H2D memcpy / NCCL / dataloader）。ncu 只在"某个 kernel 的 SOL% 值得深挖"时出场。

## 4. 可执行代码

- `roofline_sol.py`：`Hardware` / `KernelTraffic` + AI / ridge / bound / SOL% / $t_{min}$ ，全是纯函数；`vecadd_traffic` 把"1 FLOP、3 元素流量"的算子知识编码成代码。
- `timeline_model.py`：`simulate(events)` 真走一遍事件序列累加 span / busy / gap（串行假设已注明）；`softmax_demo` 把 roofline 算出的 kernel 时间喂给时间线——两层模型是**算出来**连起来的，不是手填的。
- `nsys_ncu_recipes.py`：参考命令的 argv 构造器 + `tools_present()` 探针（本环境 nsys / ncu 均不存在），**不执行任何 profiling**。

```bash
python3 ai-infra/r2-day-12-profiling/roofline_sol.py
python3 ai-infra/r2-day-12-profiling/timeline_model.py
python3 ai-infra/r2-day-12-profiling/nsys_ncu_recipes.py
python3 -m unittest discover -s ai-infra/r2-day-12-profiling -p 'test_*.py' -v
```

## 5. 何时不赚（profiling 版）

1. **正确性未跑通时**：profile 一个错的结果毫无意义。
2. **只看 SOL% 不看端到端**：单个 kernel DRAM 100% 也可能只占 step 的 1%。
3. **小 kernel / 小 shape**：ncu replay 放大的是固定开销，% 数字不能外推；先 Systems 看 busy%。
4. **profile 一次就下结论**：GPU 时钟、缓存状态、系统噪声都会抖；同一配置跑多次，看分布。
5. **在错误层级用错工具**：全系统问题（dataloader 慢、NCCL hang）用 ncu 单 kernel 深挖是南辕北辙。

## 状态

- 已验证：Python 语法；20 个 CPU 单元测试（手算 AI=1/12、ridge=20、SOL=0.417%、 $t_{min}=28.7$ ps；时间线 eager $27.512\mu s$ /27.3%、fused $7.504\mu s$ /33.4%、模型加速比 $3.67\times$ ；recipe argv 断言；本环境无 nsys/ncu 探针断言）。
- **execution not validated / 待H100验证**：本环境无 CUDA GPU / nsys / ncu；参考命令未执行；launch 开销 $5\mu s$ 为示意参数非测量；kernel 时间取 roofline 理论下限非测量；任何真实 SOL%、wall-time、加速比均未测量。 $3.67\times$ 是 theoretical model，不是 benchmark。
- 本课状态保持 `blocked`；没有声称 loss、耗时、带宽实测、comm%、MFU、设备拓扑。

## 原始 / 官方来源

- Nsight Compute 文档（SpeedOfLight / Roofline Chart sections）：<https://docs.nvidia.com/nsight-compute/2023.1/NsightCompute/index.html>
- NVIDIA 技术博客（Nsight Compute roofline 方法，`--set detailed/full` 与 `SpeedOfLight_RooflineChart`）：<https://developer.nvidia.com/blog/?p=22103>
- NVIDIA H100 官方规格（FP32 67 TFLOPS、HBM 3.35TB/s；同 r2-day-06 引用）：<https://www.nvidia.com/en-us/data-center/h100/?trk=article-ssr-frontend-pulse_little-text-block>
- Roofline 模型原始论文：Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 2009
