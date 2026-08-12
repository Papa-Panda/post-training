# OpenRLHF PPO vs GRPO switch minimal
"""
OpenRLHF example: 4-model PPO vs 2-model GRPO
"""
ppo_config = dict(
    actors_lr=1e-6,
    critic_lr=2e-6,
    kl_coef=0.02,
    ptx_coef=0.0,
    clip_range=0.2,
    gamma=1.0,
    lam=0.95,
    num_model=4,  # actor, critic, ref, reward
    zero_stage=2,
    vllm_engine="vllm",
)

grpo_config = dict(
    actors_lr=1e-6,
    kl_coef=0.02,
    clip_range=0.2,
    clip_range_high=0.28,  # DAPO
    adv_estimator="grpo",  # no GAE, no critic
    group_size=16,
    num_model=2,  # actor, ref (+ rule reward no model)
    dynamic_filtering=True,  # skip all-correct/all-wrong groups
    overlong_filter=True,
    token_mean_loss=True,
)

print("PPO needs", ppo_config["num_model"], "models; GRPO", grpo_config["num_model"], "-> save ~1x model memory")
