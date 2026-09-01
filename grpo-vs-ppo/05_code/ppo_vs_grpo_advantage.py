"""Runnable PPO-vs-GRPO semantics demo; standard library only."""

from rl_objectives import (
    aggregate_masked,
    clipped_surrogate,
    compute_gae,
    group_advantages,
    leave_one_out_advantages,
    sequence_ratio_from_token_ratios,
)


def demo() -> None:
    rewards = [0.0, 0.0, 1.0]
    values = [0.2, 0.3, 0.4, 0.0]
    gae = compute_gae(rewards, values, terminated=[False, False, True], gamma=1.0, lam=0.95)
    print("PPO GAE:", [round(x, 4) for x in gae])

    group_rewards = [1.0, 0.0, 1.0, 0.0]
    print("GRPO z-score:", group_advantages(group_rewards))
    print("RLOO:", leave_one_out_advantages(group_rewards))
    print("all equal:", group_advantages([1.0, 1.0, 1.0, 1.0]))

    ratios = [1.3, 0.7]
    print("clip, positive A:", clipped_surrogate(ratios, [1.0, 1.0]))
    print("clip, negative A:", clipped_surrogate(ratios, [-1.0, -1.0]))
    print("sequence ratio:", sequence_ratio_from_token_ratios(ratios))

    values_by_response = [[3.0], [0.0, 0.0, 0.0]]
    masks = [[1], [1, 1, 1]]
    print("response mean:", aggregate_masked(values_by_response, masks, "response_mean"))
    print("token mean:", aggregate_masked(values_by_response, masks, "token_mean"))


if __name__ == "__main__":
    demo()
