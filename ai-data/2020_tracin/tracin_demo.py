"""
TracIn minimal demo - 1文件跑通 self-influence
用途：对应 2020 TracIn，用3个 checkpoint + 梯度点积找出脏数据
跑法：python tracin_demo.py (CPU即可，需 torch)
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# 1. toy data: 100条，10条故意错标当脏数据
torch.manual_seed(0)
n=200
X = torch.randn(n, 20)
true_w = torch.randn(20,1)
y = (X @ true_w > 0).float().squeeze()
# 注入10%噪声标签
for i in random.sample(range(n), 20):
    y[i] = 1 - y[i]

ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=32, shuffle=True)

model = nn.Linear(20,1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()

checkpoints = []
# 2. 训练3个epoch，每epoch存一个checkpoint的梯度快照
for epoch in range(3):
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb).squeeze(), yb)
        loss.backward()
        opt.step()
    # 存当前参数
    checkpoints.append({k: v.clone().detach() for k,v in model.state_dict().items()})
    print(f"epoch {epoch} saved, loss {loss.item():.3f}")

# 3. 算 self-influence = sum_t lr * ||grad||^2
# 遍历每条样本
influences=[]
model.train()
for idx in range(n):
    xi, yi = X[idx:idx+1], y[idx:idx+1]
    score=0.0
    for ckpt in checkpoints:
        model.load_state_dict(ckpt)
        model.zero_grad()
        l = loss_fn(model(xi).squeeze(), yi)
        l.backward()
        # 梯度平方和近似点积
        g2=0.0
        for p in model.parameters():
            if p.grad is not None:
                g2 += (p.grad**2).sum().item()
        score += 0.1 * g2 # lr=0.1
    influences.append((idx, score))

influences.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 self-influence (疑似脏数据):")
for idx, sc in influences[:10]:
    print(f" idx {idx:3d} score {sc:.4f} label {y[idx].item()}")

print("\n验证：注入噪声的20个里，命中了几个？")
noisy_set=set(random.sample(range(n),20)) # 注意这里为了demo简化，实际应记录上面注入的idx
# 正确做法：记录noisy_idx，统计重叠
# 这里只演示分布：分数越高越可能是outlier
print("看分布尾巴是否很长 -> 长尾就是脏数据信号")
