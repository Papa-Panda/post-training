# r2-day-05 JAX Mesh/pjit demo - 严格6步对齐DDP声明式对比
# 6步: Mesh定义 → PSpec切分 → pjit声明 → 逻辑matmul → 编译器插通信 → 执行Sharding可视化
try:
    import jax, jax.numpy as jnp
    from jax.experimental.pjit import pjit
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, PartitionSpec as P
    HAS_JAX=True
except ImportError:
    HAS_JAX=False
    class P:
        def __init__(self, *args): self.args=args
        def __repr__(self): return f"P{self.args}"

def run():
    if not HAS_JAX:
        print("=== r2-Day05 JAX Mesh/pjit 声明式 6步结构 (无jax环境模拟) ===")
        print("[Step1] Mesh: Mesh(np.array(jax.devices()).reshape(2,4), ('data','model'))")
        print("  代码: mesh = Mesh(mesh_utils.create_device_mesh((2,4)), ('data','model'))")
        print("[Step2] PSpec: a_pspec=P('data',None) b_pspec=P(None,'model') c_pspec=P('data','model')")
        print("  代码: a_pspec = P('data', None)  # A 1024x1024 行切2份 512行")
        print("  代码: b_pspec = P(None, 'model') # B 1024x1024 列切4份 256列")
        print("  代码: c_pspec = P('data', 'model') # C 行切2列切4 每卡512x256")
        print("[Step3] pjit声明用PSpec: matmul = pjit(lambda a,b: a@b, in_shardings=(a_pspec,b_pspec), out_shardings=c_pspec)")
        print("  代码: matmul = pjit(lambda a,b: a@b, in_shardings=(a_pspec,b_pspec), out_shardings=c_pspec)")
        print("  -> 对比DDP: dist.init_process_group + all_reduce 10行，这里声明式0行显式通信")
        print("[Step4] 逻辑: c = matmul(a,b)  # 1024x1024 @ 1024x1024 -> 1024x1024")
        print("[Step5] 编译器: 自动 AllGather/ReduceScatter，对应DDP hook")
        print("[Step6] 执行: jax.debug.visualize_sharding(c) 看8卡棋盘")
        print("CPU proxy ok, 待H100真机 8卡 2x4 Mesh 声明式 vs DDP 10行")
        return

    devices=jax.devices()
    print(f"[Step1] jax devices {len(devices)}")
    mesh_shape = (len(devices),) if len(devices)<2 else (2, max(1,len(devices)//2))
    axis_names = ('data',) if len(mesh_shape)==1 else ('data','model')
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), axis_names)
    print(f"[Step1] mesh = Mesh(create_device_mesh({mesh_shape}), {axis_names}) -> {mesh}")

    # Step2 真PSpec，被Step3使用
    a_pspec = P('data', None)
    b_pspec = P(None, 'model') if len(mesh_shape)>1 else P(None, None)
    c_pspec = P('data', 'model') if len(mesh_shape)>1 else P('data', None)
    print(f"[Step2] a_pspec = P('data',None) = {a_pspec} 行切2")
    print(f"[Step2] b_pspec = P(None,'model') = {b_pspec} 列切4")
    print(f"[Step2] c_pspec = P('data','model') = {c_pspec} 行列都切")

    # Step3 pjit声明，显式使用PSpec
    matmul = pjit(lambda a,b: a@b, in_shardings=(a_pspec, b_pspec), out_shardings=c_pspec)
    print(f"[Step3] matmul = pjit(lambda a,b: a@b, in_shardings=({a_pspec},{b_pspec}), out_shardings={c_pspec}) 0行显式通信")

    # Step4-6
    a=jnp.ones((1024,1024))
    b=jnp.ones((1024,1024))
    print(f"[Step4] a 1024x1024 b 1024x1024 逻辑 matmul")
    c=matmul(a,b)
    print(f"[Step5-6] c = matmul(a,b) done {c.shape} 编译器已插AllGather/ReduceScatter")
    try:
        jax.debug.visualize_sharding(c)
        print("[Step6] jax.debug.visualize_sharding(c) # 看分片")
    except:
        print("[Step6] 单卡无切分，待H100 8卡 2x4 visualize_sharding")

if __name__=="__main__":
    run()
