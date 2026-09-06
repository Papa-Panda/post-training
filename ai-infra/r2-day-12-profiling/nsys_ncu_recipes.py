"""Reference nsys / ncu CLI recipes (NOT executed here).

Each recipe is a real argv list you can copy onto a GPU machine.
``tools_present()`` probes PATH; in this environment both tools are
absent, so every recipe below is reference-only:
execution not validated / 待H100验证.
"""

import shutil

RECIPES = {
    "systems_train_iter": {
        "tool": "nsys",
        "argv": ["nsys", "profile",
                 "-o", "train_iter",
                 "--trace=cuda,nvtx,osrt",
                 "--force-overwrite=true",
                 "./train.py"],
        "purpose": ("抓一次 train iter 的全系统时间线：先看 GPU busy%、"
                    "kernel 个数、idle gap 的来源。"),
        "look_for": ["GPU busy%（GPU 有没有在干活）",
                     "kernel 个数 4 vs 1（Day11 的 fuse 是否真生效）",
                     "gap 归因：cudaLaunchKernel / memcpy H2D / NCCL / dataloader"],
    },
    "compute_sol": {
        "tool": "ncu",
        "argv": ["ncu", "--set", "detailed",
                 "--section", "SpeedOfLight",
                 "--section", "SpeedOfLight_RooflineChart",
                 "-o", "kernel_sol",
                 "./app"],
        "purpose": ("对单个 kernel 看 SpeedOfLight：各单元 throughput 占理论峰值的"
                    "百分比，以及它在 roofline 图上的落点。"),
        "look_for": ["SOL%：SM vs DRAM（memory-bound 应见 DRAM 高、SM 低）",
                     "roofline 图：点落在带宽屋顶还是算力屋顶"],
    },
    "compute_full": {
        "tool": "ncu",
        "argv": ["ncu", "--set", "full",
                 "-o", "kernel_full",
                 "./app"],
        "purpose": ("全量 section 深挖（MemoryWorkloadAnalysis / Occupancy / "
                    "SchedulerStats…）。replay 开销最大，只对 Systems 定位到的"
                    "热点 kernel 使用。"),
        "look_for": ["MemoryWorkloadAnalysis 的 DRAM/L2/L1 throughput",
                     "Occupancy 与 launch 配置是否合理"],
    },
}


def get_recipe(name: str) -> dict:
    """Return the recipe dict; raises KeyError for unknown names."""
    return RECIPES[name]


def tools_present() -> dict:
    """Probe PATH for nsys/ncu. Harmless; does not run any profiling."""
    return {"nsys": shutil.which("nsys") is not None,
            "ncu": shutil.which("ncu") is not None}


if __name__ == "__main__":
    print("tool availability (this machine):", tools_present())
    for name, r in RECIPES.items():
        print(f"\n[{name}] {r['purpose']}")
        print("  $" + " ".join(r["argv"]))
        for lf in r["look_for"]:
            print(f"    -> {lf}")
    print("\nAll recipes are reference-only: execution not validated / 待H100验证.")
