# r2-day-03 topo demo CPU only
# Ring AllReduce time = 2*(N-1)/N * data / BW
def ring_time(data_gb, n, bw_gbps):
    comm_gb = 2*(n-1)/n * data_gb
    return comm_gb / bw_gbps * 1000  # ms

for n in [2,4,8]:
    for bw_name, bw in [("NVLink 900GB/s",900), ("PCIe 64GB/s",64), ("IB 400Gbps~50GB/s",50)]:
        t = ring_time(1.0, n, bw)
        print(f"{n}卡 1GB Ring {bw_name}: comm {2*(n-1)/n:.3f}GB time {t:.2f}ms")

print("\nAllGather = ReduceScatter = Ring/2")
print("8卡 1GB AllGather 0.97ms NVLink vs 13.7ms PCIe")
print("为何TP不能跨机: TP每层2次AllReduce 通信密集 1GB/层*32层=32GB *1.94ms≈62ms NVLink可接受 PCIe 27ms*32=864ms 不可接受")
print("CPU proxy ok, 待H100补 nvidia-smi topo -m + nccl-tests")
