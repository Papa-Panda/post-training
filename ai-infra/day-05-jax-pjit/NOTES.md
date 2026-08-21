# 2026-08-06 JAX pjit - done

CPU run after JAX 0.11.0 installed:
- devices=[CpuDevice(id=0)] count=1, mesh axis='data' size=1
- pjit A(8,4) @ B(4,2) -> C(8,2), C[0]=[4. 4.] ok
- 声明式 sharding: P('data',None) 行切，编译器管通信

fallback 时模拟：shard 0 (4,4)->(4,2) shard 1 (4,4)->(4,2)

Conclusion: JAX pjit 逻辑通了，真分片待多卡 H100。
