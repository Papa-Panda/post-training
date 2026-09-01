# 05 — Runnable reference code

这是一组 **standard-library-only** 语义参考，不是某个训练框架的 drop-in config。之前两个伪配置文件把未核验的 key、固定超参数和“模型数 = 显存”混在一起，现已删除。

## Run

```bash
cd grpo-vs-ppo/05_code
python3 ppo_vs_grpo_advantage.py
python3 -m unittest -v test_rl_objectives.py test_docs.py
python3 -m py_compile rl_objectives.py ppo_vs_grpo_advantage.py test_rl_objectives.py test_docs.py
```

## Covered semantics

- token ratio 与 exact sequence ratio；
- PPO sign-dependent clipping；
- terminal vs truncated GAE bootstrap；
- GRPO population-standard-deviation convention；
- equal-reward 与 $G=1$ edge cases；
- group-centered 与 leave-one-out advantage 的 scale 关系；
- $k_1/k_2/k_3$ KL value estimators；
- response mean vs token mean；
- action mask 排除 observation token。

分布式 trainer 需要额外处理：packed sequences、cross-rank groups、precision、sharding、policy versioning、reward failures、weight synchronization 与 metrics reduction。这些不应伪装在一个“看起来像真实框架”的静态 YAML 中。
