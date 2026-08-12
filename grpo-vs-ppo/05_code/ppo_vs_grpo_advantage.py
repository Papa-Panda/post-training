"""
Minimal reproduction: PPO GAE vs GRPO group baseline advantage
Bilingual ZH/EN concise, torch only, no external deps
"""
import torch

def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    # rewards: [T], values: [T+1]
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv

def compute_grpo_advantages(rewards_group, eps=1e-8):
    """
    rewards_group: [G] scalar rewards for each output in group
    Returns: [G] normalized advantage, broadcast to tokens later
    DeepSeekMath: A_i = (r_i - mean)/std
    """
    r = torch.tensor(rewards_group, dtype=torch.float32)
    mu = r.mean()
    std = r.std(unbiased=False)
    # skip case: all same -> zero advantage
    if std < eps:
        return torch.zeros_like(r)
    adv = (r - mu) / (std + eps)
    return adv

def demo():
    print("=== PPO GAE demo ===")
    # simulate 5-step trajectory with sparse final reward 1.0
    rewards = torch.tensor([0.,0.,0.,0.,1.0])
    values = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.0])  # V(s_t), last = 0
    adv_ppo = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    print(f"rewards {rewards.tolist()} -> GAE adv {adv_ppo.tolist()}")
    # intuition: early steps get discounted credit via V propagation

    print("\n=== GRPO group demo ===")
    # same query, G=4 samples: 2 correct (reward 1), 2 wrong (0)
    group_rewards = [1.0, 0.0, 1.0, 0.0]
    adv_grpo = compute_grpo_advantages(group_rewards)
    print(f"group rewards {group_rewards} -> mu {torch.tensor(group_rewards).mean():.2f} std {torch.tensor(group_rewards).float().std(unbiased=False):.2f}")
    print(f"GRPO advantages {adv_grpo.tolist()}  # +1 for good, -1 for bad after norm")
    # broadcasting: each output's tokens share same A_i
    token_lens = [10, 10, 12, 11]
    for i, L in enumerate(token_lens):
        print(f"  output {i} len {L} token_adv = {adv_grpo[i].item():.2f} repeated")

    print("\n=== Edge: all correct ===")
    print(compute_grpo_advantages([1,1,1,1]).tolist(), "-> zero, should skip update")

if __name__ == "__main__":
    demo()
