# NOTES - r2-Day05 JAX Mesh/pjit

CPU proxy，待H100验证

- Mesh: 逻辑设备网格，如 8卡 2x4，轴 x=data y=model
- PartitionSpec: 声明张量如何切，P('x',None) 表示batch切x，hidden不切
- pjit: 编译时根据Mesh+PSpec自动插通信，AllGather/ReduceScatter对用户透明
- 对比：
  - PyTorch: 你管通信，dist.init, all_reduce, DistributedSampler手写
  - JAX: 你管切分，写PSpec，编译器管通信
- 10行matmul例子：A 1024x1024 切行，B 1024x1024 切列，C = A@B 自动AllReduce

CPU proxy：
- jax未装，结构演示
- PyTorch DDP 2-rank 3.12s vs JAX pjit 声明式 0行通信代码

待H100：
- 8卡 `mesh = Mesh(np.array(jax.devices()).reshape(2,4), ('data','model'))`
- `pjit(lambda x,y: x@y, in_shardings=(P('data',None), P(None,'model')), out_shardings=P('data','model'))`
- 真机 Sharding可视化 `jax.debug.visualize_sharding`
