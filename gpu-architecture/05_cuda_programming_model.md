# 05 — CUDA 编程模型：Groups、Streams 与依赖关系

## 1. Kernel launch 是异步工作提交

典型 kernel：

```cuda
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

launch geometry：

$$N_{\mathrm{blocks}}=\left\lceil\frac{N}{T_b}\right\rceil.$$

host launch 通常异步返回；错误可能在后续同步点才暴露。正确 benchmark 需要 CUDA events 或明确 device synchronization，不能用包围异步 launch 的普通 CPU timer 当 kernel latency。

## 2. 同步作用域

| primitive | 作用域 | 典型用途 |
|---|---|---|
| `__syncwarp(mask)` | 指定 warp lanes | warp-local phase/memory ordering |
| `__syncthreads()` | thread block | shared-memory tile 生产者/消费者 |
| Cooperative Groups `group.sync()` | 显式 group | 把参与者作为 API contract |
| event/stream dependency | device runtime | kernel/copy 间有向依赖 |
| cooperative grid sync | 满足 cooperative launch 的 grid | 少数需要单 kernel 全局 phase 的场景 |
| kernel boundary | grid | 最通用的全局阶段分隔 |

barrier 不是越多越安全：它会等待最慢参与者，减少 overlap；但缺 barrier 会产生 race。首先定义 ownership 和 happens-before。

## 3. Cooperative Groups

Cooperative Groups 将“哪些线程共同执行 collective”编码为对象：

```cuda
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
auto tile = cg::tiled_partition<32>(block);
```

相比把 `__syncthreads()` 隐藏在函数里，显式 group 参数能说明调用者必须让整个 group 参与，减少 partial-block deadlock。`coalesced_threads()` 可在分支内部发现当前 active group，但 group 成员是动态的，不应假定固定 lane identity。

## 4. Streams 是有序队列，不是“线程”

同一 stream 中操作按提交顺序建立依赖；不同 streams **可能**并发，但不保证并发。并发需同时满足：

- 硬件资源允许；
- 数据无隐式依赖；
- copy 方向/engine 支持；
- host memory 条件（异步 H2D/D2H 常需 pinned memory）；
- default-stream 语义和 library handle stream 设置正确。

用 event 建图：

```text
stream A: H2D(batch k) -> event ready_k
stream B: wait ready_k -> kernel(k) -> event done_k
stream A: wait done_k -> reuse buffer
```

双缓冲的理想 steady-state：

$$T_{\mathrm{stage}}\approx\max(T_{\mathrm{copy}},T_{\mathrm{compute}}),$$

但首尾 pipeline bubble、copy engine 数、PCIe/NVLink 和资源竞争仍在。

## 5. CUDA Graphs

重复执行稳定 DAG 时，Graph 可摊薄 host launch/dispatch overhead。适合小 kernel 密集、shape/control flow 稳定的迭代。它不减少 kernel 内 FLOPs/bytes，也不能自动解决动态 shape、内存地址生命周期或错误依赖。

## 6. Stream-ordered memory 与 lifetime

异步分配/释放需要把内存 lifetime 纳入 stream dependency。最危险的错误不是 OOM，而是 buffer 在另一个 stream 尚未消费完成时被复用。工程上记录：

- producer stream/event；
- consumer waits；
- allocator pool；
- capture compatibility；
- library handle 所属 stream。

## 7. Warp-level collectives 示例

归约常写成：

```cuda
for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(mask, value, offset);
```

它假设 participating group 与 mask 一致。若只部分 lanes 有效，必须让 mask 反映真实 active set，并处理结果所在 lane。浮点加法不满足结合律，不同 reduction tree 可能产生微小数值差异。

## 8. 常见误区

- “不同 stream 就会 overlap”：资源或依赖可能使它们串行。
- “event 只用于计时”：event 也是设备侧依赖边。
- “warp 内不用同步”：Independent Thread Scheduling 下应使用明确同步 primitive。
- “更多 streams 总更快”：会增加调度、内存占用和 contention。
- “Graph 一定更快”：大而长的 compute-bound kernel 几乎不受 host launch 开销影响。

## 9. 正确验证顺序

1. CPU reference / invariant；
2. CUDA error checking 与 sanitizer；
3. warmup；
4. events 测量 device elapsed time；
5. 多次运行报告分布；
6. profiler 验证 overlap/指令/流量，而非从 API 结构猜执行。

## 导航

- 上一篇：[04 Tensor Core/GEMM](04_tensor_core_gemm_dataflows.md)
- 下一篇：[06 库与 Kernel DSL](06_libraries_and_kernel_dsl.md)
- 相关：[02 SIMT](02_simt_warp_scheduling.md) · [09 Profiling](09_roofline_profiling_tuning.md)
