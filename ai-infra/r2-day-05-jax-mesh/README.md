> Connection to Prev: r2-Day04 DDP → r2-Day05 JAX Mesh/pjit: DDP是PyTorch命令式“你管通信AllReduce”，JAX是声明式“你管切分，编译器管通信”，同样8卡，JAX只写PartitionSpec。

# r2-Day05 - JAX Mesh / pjit 声明式切分

## 昨日复盘
r2-Day04 DDP 每卡全模型，DistributedSampler切数据，AllReduce grad，bucket 25MB，CPU gloo proxy 3.12s。

## 今日主题
**Mesh / jit / pjit 声明切分 vs PyTorch命令式**

- PyTorch DDP: 你显式 init_process_group, all_reduce, DistributedSampler
- JAX pjit: 你声明 Mesh(2,4), PartitionSpec('data','model'), 编译器自动插 AllGather/ReduceScatter
- 10行 matmul 看 device mesh，Sharding从逻辑切分到物理设备

## 最小可跑任务（30-60min）
写 `jax_demo.py`：
- `mesh = Mesh(jax.devices(), ('x','y'))`
- `pjit matmul` 按 P('x',None) 切分
- 对比 PyTorch DDP 需手写 all_reduce，JAX 只改 PartitionSpec

## 检验
- 不查说出 pjit 是声明式，DDP是命令式
- 能写出 8卡 2x4 mesh，batch切x，hidden切y
- 能说清为何 JAX适合TP+DP混合

## 资源
- JAX pjit docs
- https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html

## 待H100
CPU proxy，待H100 8卡真机 `jax.device_count()` + `pjit` Sharding可视化
