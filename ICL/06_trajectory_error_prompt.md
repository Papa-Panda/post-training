# 06 从 Trajectory Error 到可检验规则

[← 05 比较](05_comparison.md) · [下一章：Coding data →](07_coding_data.md)

问题不是“能否把错误总结放进 prompt”——可以；关键是如何防止反思文本成为未经验证的新噪声。

## 1. 四类机制不要混用

| 机制 | 状态放在哪里 | 是否改权重 | 最小验证 |
|---|---|---|---|
| episodic reflection | 当前/后续 trial 的 memory | 否 | 同任务 retry 对照 |
| retrieved guideline | 外部规则库，按 state/query 选择 | 否 | held-out task + retrieval 消融 |
| training example | SFT/preference/RL 数据 | 是 | 数据 provenance + 训练后 eval |
| executable verifier | 测试、静态检查、规则 reward | 不一定 | false-positive/negative 与隔离执行 |

Reflexion 属于第一类：语言反馈写入 episodic memory，不更新基础模型权重。ExpeL 收集成功/失败经验、抽象跨任务 insights，再在评测时应用。LEAP 从 few-shot 示例上诱发错误并抽取 task principles。AutoGuide 从 offline contrastive trajectories 抽取条件式 guidelines，再按当前 context 选择。

## 2. 推荐的数据结构

每条规则都必须绑定证据与适用域，而不是只存一句自然语言：

```json
{
  "rule_id": "python-import-shadowing-v1",
  "condition": "a local module has the same name as an imported package",
  "action": "inspect module resolution before changing dependency versions",
  "evidence": ["trace-0142", "trace-0198"],
  "counterexamples": ["trace-0211"],
  "verifier": "tests/test_import_resolution.py",
  "scope": {"language": "python", "failure": "import"},
  "status": "candidate"
}
```

状态建议：`candidate -> validated -> promoted -> retired`。只有在 held-out tasks 上相对无规则 baseline 有净收益，才进入 `validated`。

## 3. 从轨迹到规则的闭环

1. **记录**：任务、环境版本、动作、观察、退出码、测试结果、token/cost。
2. **定位首个可归因分叉**：不要把最终报错当根因。
3. **聚类**：按 failure mechanism，而非错误字符串表面聚类。
4. **对比**：至少一条成功轨迹与一条失败轨迹，找最小动作差。
5. **抽取条件式规则**：`WHEN condition, DO action, VERIFY check`。
6. **反例搜索**：主动找规则会误导的场景。
7. **held-out 评测**：冻结规则与任务集后比较。
8. **部署选择**：prompt、retrieval、训练样本或 verifier，四者分别记账。

## 4. 评测与停止条件

对规则集合 $G$，至少报告：

$$\Delta\mathrm{Pass}=\mathrm{Pass}(G)-\mathrm{Pass}(\varnothing)$$

$$\Delta C=C(G)-C(\varnothing),\qquad \Delta T=T(G)-T(\varnothing)$$

并做：

- **matched-token control**：用等 token 数的无关规则排除“多思考”效应；
- **oracle retrieval upper bound**：判断瓶颈在规则质量还是检索；
- **leave-one-cluster-out**：测试规则是否跨错误簇泛化；
- **negative transfer**：记录原本成功、加规则后失败的案例；
- **staleness**：依赖/API 更新后自动降级或退休规则。

没有稳定提升时，优先缩窄 condition 或转成 verifier，而不是继续堆 prompt。

## 5. 已核验的代表工作

- [Reflexion](https://arxiv.org/abs/2303.11366)：反馈 → verbal reflection → episodic memory → 后续 trial。
- [ExpeL](https://arxiv.org/abs/2308.10144)：collect → extract/abstract → apply/recall。
- [LEAP](https://arxiv.org/abs/2402.05403)：从 few-shot mistakes 抽取 principles；摘要报告 GPT-4 上 DROP +7.5%、HotpotQA +3.3%，且不增加输入示例数。数字只适用于论文设置。
- [AutoGuide](https://arxiv.org/abs/2403.08978)：offline experiences → context-aware guidelines → test-time selection。

这些工作支持“可行”，不支持“对任意 2026 模型仍单调有效”或“几百条规则可零成本常驻”。系统成本见 [08](08_systems_and_evaluation.md)，代码数据闭环见 [07](07_coding_data.md)。
