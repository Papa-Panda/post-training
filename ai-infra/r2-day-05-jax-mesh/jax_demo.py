# r2-day-05 JAX Mesh/pjit demo CPU proxy
try:
    import jax, jax.numpy as jnp
    from jax.experimental.pjit import pjit
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, PartitionSpec as P
    HAS_JAX=True
except ImportError:
    HAS_JAX=False

def run():
    if not HAS_JAX:
        print("=== r2-Day05 JAX Mesh/pjit 声明式 6步结构 (无jax环境模拟) ===")
        print("[Step1] Mesh: Mesh(jax.devices() 8卡, ('data','model')) 2x4网格")
        print("  -> 逻辑轴 data=2 model=4，物理8卡")
        print("[Step2] PSpec: P('data',None) 切batch，不切hidden")
        print("  -> A 1024x1024 切行 data，B 1024x1024 不切")
        print("[Step3] pjit: pjit(lambda x,y: x@y, in_shardings=(P('data',None), P(None,'model')), out=P('data','model'))")
        print("  -> 声明式，编译器自动插 AllGather/ReduceScatter")
        print("[Step4] 对比 PyTorch DDP: dist.init_process_group + all_reduce hand-write 10行")
        print("[Step5] JAX: 0行通信，改PSpec即可切 TP+DP混合")
        print("[Step6] 8卡 matmul 1024² CPU proxy ok, 待H100真机 jax.device_count()=8 + visualize_sharding")
        return
    # real JAX path (needs 8 devices, else 1 device demo)
    devices=jax.devices()
    print(f"jax devices {len(devices)}")
    mesh=Mesh(mesh_utils.create_device_mesh((len(devices),)), ('data',))
    print(f"Mesh {mesh}")
    @pjit
    def matmul(a,b):
        return a@b
    a=jnp.ones((1024,1024))
    b=jnp.ones((1024,1024))
    c=matmul(a,b)
    print(f"matmul done {c.shape} 声明式 ok")

if __name__=="__main__":
    run()
