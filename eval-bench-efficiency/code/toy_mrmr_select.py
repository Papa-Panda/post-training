"""
toy_mrmr_select.py — minimal mRMR vs random for efficient benchmarking
Demonstrates idea from arXiv:2605.25773 : same tiny code shows stable selection
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n_models, d_items = 80, 500  # 80 LLMs, 500 questions
# simulate item accuracies
X = np.random.binomial(1, 0.6, size=(n_models, d_items)).astype(float)
# total score y = weighted sum + noise
true_weights = np.random.randn(d_items)
y = X @ true_weights + np.random.randn(n_models)*2

def mrmr_select(X, y, k=30):
    selected=[]
    remaining=list(range(X.shape[1]))
    # precompute relevance I(f;y) via mutual info (proxy for I)
    relevance = mutual_info_regression(X, y, random_state=0)
    for _ in range(k):
        best, best_score = None, -1e9
        for f in remaining:
            rel = relevance[f]
            if not selected:
                red = 0
            else:
                # avg mutual info between f and selected (approx redundancy)
                # use correlation as fast proxy
                red = np.mean([abs(np.corrcoef(X[:,f], X[:,s])[0,1]) for s in selected])
                red = 0 if np.isnan(red) else red
            score = rel - red
            if score > best_score:
                best_score, best = score, f
        selected.append(best)
        remaining.remove(best)
    return selected

for k in [10,30,50]:
    sel_mrmr = mrmr_select(X, y, k)
    sel_rand = np.random.choice(d_items, k, replace=False).tolist()
    for name, sel in [("mrmr", sel_mrmr), ("random", sel_rand)]:
        kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
        kr.fit(X[:,sel], y)
        y_pred = kr.predict(X[:,sel])
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"k={k} {name} RMSE={rmse:.3f} stable_set_head={sel[:5]}")

# stub IRT info filtering proxy: high variance + high discrimination (std * corr)
vars = X.var(axis=0)
disc = np.array([abs(np.corrcoef(X[:,j], y)[0,1]) for j in range(d_items)])
disc = np.nan_to_num(disc)
info_proxy = vars * disc
top_irt = np.argsort(-info_proxy)[:30]
print("irt_proxy top 5:", top_irt[:5])
