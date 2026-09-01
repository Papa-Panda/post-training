# 06 — Self-Improving Harness：从 Failure 到可回滚版本

## 元信息

- 核心论文：[STOP](https://arxiv.org/abs/2310.02304v3) · [Self-Harness](https://arxiv.org/abs/2606.09498) · [Agentic Harness Engineering](https://arxiv.org/abs/2604.25850v4) · [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)
- 本章目标：把不同论文串成一个统一 control-loop，并清楚标出它们在 proposal、evaluation、promotion 时序上的差异。

## 1. Unified control-loop

给定 frozen model $\theta$、active harness $h_t$、外置 control plane $q=(V,\Pi,B,L)$：

$$\mathcal T_t=\mathrm{Rollout}(\theta,h_t,D_{\mathrm{mine}};q),$$

$$z_t=\mathrm{Attribute}(\mathcal T_t,V),\qquad \mathcal C_t=\mathrm{Propose}(h_t,z_t),$$

$$\tilde h_t^{(j)}=\mathrm{Sandbox}(h_t\oplus\delta_t^{(j)}),$$

$$h_{t+1}=\mathrm{Promote}(h_t,\{\tilde h_t^{(j)}\},D_{\mathrm{in}},D_{\mathrm{ho}};q).$$

完整数据流：

```text
rollout
  -> immutable trace + artifacts
  -> failure clustering
  -> root-cause attribution
  -> bounded proposal manifests
  -> static validation
  -> sandbox materialization
  -> held-in repair evaluation
  -> hidden held-out regression evaluation
  -> permission/cost/statistical gate
  -> accept + version OR reject + audit
  -> shadow/canary monitoring
```

“Self-improving”只意味着 proposer 可以由 Agent 驱动；它不意味着 Agent 拥有 verifier、权限、预算和 deployment 的写权。

## 2. Failure record：先归因，后修改

对失败轨迹 $\tau_i$，保存结构化 record：

```json
{
  "task_id": "...",
  "harness_digest": "...",
  "failure_type": "duplicate-side-effect",
  "first_bad_event": "event-42",
  "candidate_causes": [
    {"surface": "workflow", "component": "retry", "confidence": 0.74},
    {"surface": "tool", "component": "idempotency", "confidence": 0.21}
  ],
  "counterevidence": ["event-38"],
  "replay_fixture": "artifact://...",
  "verifier_evidence": ["check://..."],
  "status": "hypothesis"
}
```

Attribution 分布：

$$p_A(z\mid\tau,V),\qquad z\in\{h^C,h^W,h^K,h^M,\theta,\mathcal E,V\}.$$

仅凭失败后出现某条日志，不能推断它是原因。至少使用三种证据：

1. **Temporal**：第一个违反 invariant 的 event；
2. **Mechanistic**：该组件如何导致 observation/action 改变；
3. **Counterfactual**：只替换该组件后，配对 replay 是否修复。

对于随机模型，配对干预仍需多 seed：

$$\widehat{\mathrm{ATE}}_z=\frac1n\sum_{i=1}^n\left(R_i(h\oplus\delta_z)-R_i(h)\right).$$

## 3. Proposal contract

每个 edit $\delta$ 必须是 bounded hypothesis：

$$\delta=(p,z,d,f,r,c),$$

- $p$：parent digest；
- $z$：目标 failure/component；
- $d$：typed diff；
- $f$：expected fixes；
- $r$：at-risk behaviors；
- $c$：capability 与评估成本需求。

Proposal 不是“重写成更好版本”的自由文本。它要满足：

- 只触及 declared editable paths；
- 每个 diff 有最小作用域；
- state/schema change 带 migration；
- 新工具能力单独审批；
- 预期收益和回归风险可被测试；
- 不允许把 evaluator 或 hidden labels 复制进 context。

## 4. Acceptance：逻辑 gate 与统计 gate

### 4.1 Self-Harness 的精确 split-wise 规则

令：

$$\Delta_{\mathrm{in}}^{(j)}=P_{\mathrm{in}}(h_t^{(j)})-P_{\mathrm{in}}(h_t),$$

$$\Delta_{\mathrm{ho}}^{(j)}=P_{\mathrm{ho}}(h_t^{(j)})-P_{\mathrm{ho}}(h_t).$$

[Self-Harness](https://arxiv.org/abs/2606.09498) 接受 candidate 当且仅当：

$$\Delta_{\mathrm{in}}^{(j)}\ge0,\qquad \Delta_{\mathrm{ho}}^{(j)}\ge0,\qquad \max(\Delta_{\mathrm{in}}^{(j)},\Delta_{\mathrm{ho}}^{(j)})>0.$$

即两个 split 都不下降，至少一边严格改善。held-in traces 暴露给 proposer；held-out 对 proposer 隐藏，但会被 promotion gate 重复查询，所以它是 hidden regression/validation set，不是最终一次性 test。

### 4.2 生产系统的约束 gate

二元 pass rate 之外还要加入 margin：

$$\mathrm{LCB}_{1-\alpha}(\Delta_{\mathrm{in}})>\epsilon_{\mathrm{repair}},$$

$$\mathrm{LCB}_{1-\alpha}(\Delta_{\mathrm{ho}})>-\epsilon_{\mathrm{reg}},$$

并满足：

$$\mathrm{Cap}(h')\subseteq\Pi,\qquad K(h')\le B_K,\qquad Q(h')\le B_Q.$$

若样本少，不能把“一次多过一题”当成确定提升；可以使用 paired bootstrap、McNemar test 或预先定义的 non-inferiority margin。统计门槛必须在搜索前固定，不能看到结果后调整。

## 5. Merge accepted edits

两个单独通过的 edit 合并后未必仍通过。定义冲突图：

$$G_\delta=(\mathcal C,E_c),\qquad (\delta_i,\delta_j)\in E_c\iff\mathrm{Conflict}(\delta_i,\delta_j).$$

选择兼容子集：

$$S^\star=\arg\max_{S\subseteq\mathcal C}\sum_{\delta\in S}\widehat{\Delta J}_\delta-\lambda\mathrm{Complexity}(S),\quad S\ \text{is conflict-free}.$$

即使各 edit 单独通过，merged candidate 也必须重新跑 gate，因为 interaction 可能造成回归。

## 6. STOP：优化 improver 本身

[STOP](https://arxiv.org/abs/2310.02304v3) 沿用论文的局部记号，把任务写成 $\tau=(u,s)$，其中本小节的 $u$ 是 utility function，不是统一 runtime 记号中的 authorized command $u_t$。给定 frozen language model $M_{\mathrm{LM}}$，improver 满足：

$$s'=I(u,s;M_{\mathrm{LM}}),$$

$$\hat u(I)=\frac1{|D|}\sum_{(u,s)\in D}u(I(u,s;M_{\mathrm{LM}})),$$

$$I_t=I_{t-1}(\hat u,I_{t-1};M_{\mathrm{LM}}).$$

它搜索的是调用 frozen LM 的 scaffold/program，而不是 LM weights。STOP 的价值在于证明 improver code 本身可以成为优化对象；边界是：没有 Self-Harness 式 split-wise promotion gate，递归轮次也不保证单调改善。

LPN downstream utility 对 $M=20$ 个独立 instances 取平均；meta-dataset $D$ 含同一个 $(u,s)$ 的 5 个副本；test metautility 另在 $M_{\mathrm{test}}=50$ 个独立 instances 上报告。论文还检查所谓 unsandboxing code-pattern proxy：10,000 次尝试中 GPT-4 为 $0.42\%$、GPT-3.5 为 $0.12\%$；这是狭窄 proxy，不是完整安全证明。

## 7. Self-Harness：同轮候选验证

其流程是 Weakness Mining→Harness Proposal→Proposal Validation。候选与当前 harness 在同一轮比较，兼容的 accepted edits 合并，rejected edits 记录但不改变 active state。

实验使用 MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5，在 Terminal-Bench-2.0、SWE-bench Verified、AppWorld 上九组最终 harness 均同时改善 held-in/held-out；最大整体绝对增益是 GLM-5/AppWorld 的 $40.6$ points。结果支持 bounded regression gate 的可行性，但不意味着 pass-rate gate 足够覆盖高风险部署。

## 8. AHE：延迟归因与 rollback

[AHE](https://arxiv.org/abs/2604.25850v4) 强调：

- component observability：每个 editable component 有显式文件；
- experience observability：raw rollout→task diagnosis→benchmark overview；
- decision observability：edit manifest 写明 evidence、fix 和 at-risk behavior。

其时序与 Self-Harness 不同：

```text
round t rollout evaluates edits committed at t-1
  -> attribute pass/fail flips to prior manifest
  -> rollback rejected edits
  -> distill new evidence
  -> commit next edits before effects are known
```

同时保存 $H_{\mathrm{best}}$。主实验用 GPT-5.4 high 在同一 89-task Terminal-Bench 2 集上十轮演化到 $77.0\%$ pass@1，不能称为 held-out。冻结后 SWE-bench Verified 为 $75.6\%$，seed 为 $75.2\%$；部分 repository 仍退化。其 regression prediction precision/recall 为 $11.8\%/11.1\%$，说明可观测性和 rollback 有价值，但 attribution 本身仍难。

## 9. DGM：多谱系而非单链

[DGM](https://arxiv.org/abs/2505.22954) 维护 agent archive：

$$h_p\sim\mathrm{Select}(\mathcal P_t),\qquad h_c=\mathrm{Modify}(h_p,\mathcal T_p),$$

$$\mathcal P_{t+1}=\mathrm{Archive}(\mathcal P_t\cup\{h_c\}).$$

每个 archived agent 保留非零采样概率，避免只追单一路径。论文经过 80 iterations 报告 SWE-bench experimental subset $20.0\%\to50.0\%$、full Polyglot $14.2\%\to30.7\%$，并报告跨 benchmark/model transfer。

关键 caveat：$50.0\%$ 不是全 500 道 SWE-bench Verified；search 使用分阶段 subsets。论文还展示 objective hacking：某候选通过移除特殊 logging tokens 绕过 hallucination detector，拿到 evaluator 满分而非真正消除 hallucinated tool calls。这正说明 verifier 必须外置、保持语义级检查并配 unseen regression。

## 10. 三种 promotion semantics 不可混写

| 方法 | Candidate 何时生效 | 评估/回滚 |
|---|---|---|
| STOP | improver 候选先按 empirical utility 选择 | 没有 split-wise non-regression gate |
| Self-Harness | candidate sandbox 后同轮比较 | 两 split 无回归且至少一边提升才 merge |
| AHE | edit 先 commit | 下一轮 rollout 延迟归因并 rollback |
| DGM | child 加入多谱系 archive | archive 搜索，最终还需独立 selection/deployment gate |

统一 control-loop 可以容纳它们，但不应声称它们采用同一种验证协议。

## 11. Pseudocode：严格版本

```text
function evolve(theta, active, control_plane):
    traces = rollout(theta, active, D_mine, fixed_budget)
    failures = cluster_and_attribute(traces, control_plane.verifier)
    proposals = proposer(active.readonly_view(), failures)

    accepted = []
    for p in proposals:
        if not static_validate(p, editable_manifest):
            audit_reject(p, "invalid or immutable surface")
            continue
        candidate = materialize_in_sandbox(active, p)
        result = paired_evaluate(theta, active, candidate, D_in, D_ho)
        if permission_ok(candidate) and cost_ok(candidate) and statistical_gate(result):
            accepted.append(version(candidate, parent=active.digest))
        else:
            audit_reject(p, result)

    merged = choose_compatible_frontier(accepted)
    if merged:
        rerun_gate(merged)
        deploy_shadow_then_canary(merged)
        return merged
    return active
```

## 12. Failure modes

- symptom 被当 root cause，proposal 只加更多提示；
- held-out 被反复暴露给 proposer；
- individually safe edits 合并后产生 interaction regression；
- candidate 能改 verifier、budget 或 model identity；
- 只保存成功，重复探索旧失败；
- attribution 依据单条 stochastic rollout；
- archive 保存 candidate，却没有明确 active/deployed pointer；
- rollback 只恢复 prompt，不恢复 memory schema、tool config 和 workflow state；
- benchmark 得分上升但长期维护、安全或真实任务退化。

## 13. Engineering checklist

- [ ] failure record 有 first-bad-event、evidence、counterevidence、replay fixture；
- [ ] proposal 是 bounded manifest，不是自由重写；
- [ ] candidate 先 static validate，再 sandbox；
- [ ] held-in/hidden-held-out/final-test 语义清楚；
- [ ] merge 后重新评估 interaction；
- [ ] accepted/rejected/archive/active/deployed 是不同状态；
- [ ] rollback 覆盖 prompt、code、memory 和 state migration；
- [ ] control plane 位于 proposer 写权限外。

<!-- NAVIGATION -->
## 导航

- 上一篇：[05 Harness Optimization](05_harness_optimization.md)
- 下一篇：[07 Observability、Evaluation 与 Security](07_observability_evaluation_security.md)
- 回到：[专题 README](README.md)
