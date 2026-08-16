#!/usr/bin/env python3
"""
Day 12 — Reward Model uncertainty + OAS calibration
CPU gloo 2-rank runnable, GPU NCCL待验证

Uncertainty → calibration mapping:
- Reward OAS: observed human preference = true utility + annotation noise + rollout jitter (thermal/eval flaky)
  calibrated reward = a*raw + b, uncertainty σ from ensemble, filtered if σ > threshold

Minimal runnable teaches:
- train simple classification RM on noisy preference pairs
- compute ECE before/after Platt scaling
- ensemble 5 bootstraps → std as uncertainty, OAS-style spread = |calibrated - raw|
- distributed gloo all_reduce metrics (CPU ok, H100 NCCL待补 max_memory_allocated)

Run:
  python reward_oas_calibration.py
  torchrun --nproc_per_node=2 reward_oas_calibration.py
"""
import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.distributed as dist

def is_dist():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_dist():
    if is_dist():
        dist.init_process_group(backend="gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def generate_data(N, dim, w_true, noise_flip=0.15, seed=42):
    g = torch.Generator().manual_seed(seed)
    x_a = torch.randn(N, dim, generator=g)
    x_b = torch.randn(N, dim, generator=g)
    r_a = x_a @ w_true + 0.1*torch.randn(N, generator=g)
    r_b = x_b @ w_true + 0.1*torch.randn(N, generator=g)
    y_true = (r_a > r_b).float()
    # flip labels to simulate annotator disagreement / rollout thermal jitter
    flip_mask = torch.rand(N, generator=g) < noise_flip
    y = torch.where(flip_mask, 1 - y_true, y_true)
    diff = x_a - x_b  # RM input is diff; reward(A) - reward(B)
    return diff, y, (r_a - r_b)

def compute_ece(probs, labels, n_bins=10):
    # probs, labels tensor [N]
    bin_boundaries = torch.linspace(0, 1, n_bins+1)
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i+1]
        mask = (probs > low) & (probs <= high) if i>0 else (probs >= low) & (probs <= high)
        bin_size = mask.sum().float()
        if bin_size == 0:
            continue
        bin_acc = labels[mask].float().mean()
        bin_conf = probs[mask].mean()
        ece += (bin_size/total) * torch.abs(bin_acc - bin_conf)
    return ece.item()

def train_rm(diff_train, y_train, dim, seed, epochs=40, lr=0.05):
    torch.manual_seed(seed)
    model = nn.Linear(dim, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        # full batch for simplicity (CPU fast)
        opt.zero_grad()
        logits = model(diff_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
    return model

def fit_platt_scaling(logits_val, y_val, iters=80, lr=0.1):
    # learn a,b s.t. calibrated_logit = a*logit + b minimizes BCE
    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=iters)
    bce = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        scaled = a * logits_val + b
        loss = bce(scaled, y_val)
        loss.backward()
        return loss
    opt.step(closure)
    return a.detach().item(), b.detach().item()

def ensemble_predict(models, diff):
    # returns mean prob, std prob across ensemble
    probs_list = []
    with torch.no_grad():
        for m in models:
            logits = m(diff).squeeze(-1)
            probs = torch.sigmoid(logits)
            probs_list.append(probs)
    stacked = torch.stack(probs_list)  # [K, N]
    mean = stacked.mean(0)
    std = stacked.std(0)
    return mean, std, stacked

def main():
    rank, world_size = setup_dist()
    is_main = rank == 0
    set_seed(42 + rank*7)

    dim = 16
    N_train = 2000
    N_val = 500
    w_true = torch.randn(dim)
    w_true = w_true / w_true.norm()

    # all ranks generate same data for reproducibility? Use fixed seed 42 for data then shard
    diff_train_full, y_train_full, _ = generate_data(N_train, dim, w_true, noise_flip=0.15, seed=42)
    diff_val_full, y_val_full, delta_true = generate_data(N_val, dim, w_true, noise_flip=0.15, seed=123)

    # shard for distributed metrics demonstration
    def shard_tensor(t, rank, world_size):
        n = len(t)
        per = n // world_size
        start = rank*per
        end = start+per if rank < world_size-1 else n
        return t[start:end]

    diff_val = shard_tensor(diff_val_full, rank, world_size)
    y_val_shard = shard_tensor(y_val_full, rank, world_size)

    # train ensemble on full train (each bootstrap resampled with different seed)
    K = 5
    models = []
    for k in range(K):
        # bootstrap resample indices
        g = torch.Generator().manual_seed(100 + k*13)
        idx = torch.randint(0, N_train, (N_train,), generator=g)
        diff_boot = diff_train_full[idx]
        y_boot = y_train_full[idx]
        model = train_rm(diff_boot, y_boot, dim, seed=200+k*11, epochs=35, lr=0.05)
        models.append(model)

    # pick first model as primary for Platt
    base_model = models[0]
    with torch.no_grad():
        logits_val_full = base_model(diff_val_full).squeeze(-1)
        probs_val_full = torch.sigmoid(logits_val_full)

    # ECE raw (full val)
    ece_raw = compute_ece(probs_val_full, y_val_full)

    # Platt scaling on full val logits
    a, b = fit_platt_scaling(logits_val_full, y_val_full)
    logits_cal_full = a * logits_val_full + b
    probs_cal_full = torch.sigmoid(logits_cal_full)
    ece_cal = compute_ece(probs_cal_full, y_val_full)

    # ensemble uncertainty
    mean_probs, std_probs, stacked = ensemble_predict(models, diff_val_full)
    reward_std_mean = std_probs.mean().item()
    # OAS spread = |calibrated - raw| analogy to option-adjusted spread
    oas_spread_mean = (probs_cal_full - probs_val_full).abs().mean().item()
    # high-uncertainty filter rate: std > 0.15 → would drop rollout (like vLLM failure taxonomy)
    high_uncert_rate = (std_probs > 0.15).float().mean().item()

    # accuracy, brier
    acc_raw = ((probs_val_full > 0.5).float() == y_val_full).float().mean().item()
    acc_cal = ((probs_cal_full > 0.5).float() == y_val_full).float().mean().item()
    brier_raw = ((probs_val_full - y_val_full)**2).mean().item()
    brier_cal = ((probs_cal_full - y_val_full)**2).mean().item()

    # distributed all_reduce for shard metrics (demo gloo sync)
    metrics = torch.tensor([ece_raw, ece_cal, reward_std_mean, oas_spread_mean, high_uncert_rate, acc_raw], dtype=torch.float32)
    if is_dist():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics = metrics / world_size

    result = {
        "rank": rank,
        "world_size": world_size,
        "ece_raw": float(metrics[0]) if is_dist() else ece_raw,
        "ece_cal": float(metrics[1]) if is_dist() else ece_cal,
        "ece_improve": float(ece_raw - ece_cal),
        "reward_std_mean": float(metrics[2]) if is_dist() else reward_std_mean,
        "oas_spread_mean": float(metrics[3]) if is_dist() else oas_spread_mean,
        "high_uncert_rate": float(metrics[4]) if is_dist() else high_uncert_rate,
        "acc_raw": float(metrics[5]) if is_dist() else acc_raw,
        "acc_cal": acc_cal,
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "platt_a": a,
        "platt_b": b,
        "K_ensemble": K,
        "notes": "CPU gloo ok 2-rank, 待H100 NCCL + torch.cuda.max_memory_allocated + 真人偏好数据"
    }

    if is_main:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # write NOTES friendly
        print(f"\n=> 3 CPU真数: ECE_raw {ece_raw:.4f}→ECE_cal {ece_cal:.4f} improve {ece_raw-ece_cal:.4f}, "
              f"reward_std_mean {reward_std_mean:.4f}, oas_spread {oas_spread_mean:.4f}, high_uncert {high_uncert_rate:.2%}")

    if is_dist():
        dist.destroy_process_group()
    return result

if __name__ == "__main__":
    main()
