# 07 — Observability、Evaluation 与 Security：外置 Control Plane

## 元信息

- 论文接口：[Self-Harness](https://arxiv.org/abs/2606.09498) · [AHE](https://arxiv.org/abs/2604.25850v4) · [Meta-Harness](https://arxiv.org/abs/2603.28052) · [DGM](https://arxiv.org/abs/2505.22954)
- 本章目标：让“自改进”成为可审计的软件发布流程，而不是允许 proposer 自己改题、改分、改权限后宣布成功。

## 1. Control plane 必须外置

可编辑 harness：

$$h=(h^C,h^W,h^K,h^M).$$

外置控制面：

$$q=(V,\Pi,B,L).$$

- $V$：verifier 和评分逻辑；
- $\Pi$：permission/capability policy；
- $B$：token、工具、时间、金钱和风险预算；
- $L$：append-only trace/audit log。

Model、tool、environment 与 evaluator identity 另存于只读 version manifest，并由 $L$ 中的每条事件引用；它不新增可编辑 harness 维度。

```text
editable workspace                  external control plane
-------------------------------     --------------------------------
context templates                   held-out tasks and labels
retrieval/compression policy        verifier implementation
workflow / retry / routing          permission policy and secrets
skills and subagent config          model identity / decoding
allowed tool/middleware code        budget accounting
memory read/write policy            tracer / deployment gate
```

“不在 prompt 中暴露”不等于隔离。真正边界包括：独立进程身份、最小文件 ACL、read-only mount、网络 policy、签名 artifact、separate CI/deployer 和不可由 proposer 修改的 budget meter。

## 2. Observability 的三层结构

AHE 的三层可以统一到 trace schema：

### 2.1 Component observability

每个组件有：路径、owner、schema、version、依赖、能力和 tests。否则 failure 只能归到“整个 Agent”。

### 2.2 Experience observability

```text
benchmark overview
  -> failure cluster
    -> per-task diagnosis
      -> event timeline
        -> raw tool/model artifact
```

每个摘要可以下钻到 evidence pointer；没有 pointer 的结论不能驱动自动 edit。

### 2.3 Decision observability

每个 proposal 是可证伪假设：

$$H_\delta:\quad \Delta J_{\mathrm{target}}>0,\quad \Delta J_{\mathrm{retain}}\ge0,\quad \Delta K\le B_K,\quad \Delta Q\le B_Q.$$

Manifest 记录 predicted fixes 和 at-risk behaviors；下一轮直接检验。AHE 的 regression prediction precision/recall 只有 $11.8\%/11.1\%$，说明结构化预测仍很难，但至少让失败可归因和 rollback。

## 3. 四类数据 split

| Split | Proposer 可见？ | 查询频率 | 作用 |
|---|---:|---:|---|
| $D_{\mathrm{mine}}$ | 是 | 多次 | 收集 trace、聚类失败、生成 proposal |
| $D_{\mathrm{in}}$ | 结果可见 | 多次 | 检查目标 failure 是否修复 |
| $D_{\mathrm{ho}}$ | labels/traces 隐藏 | promotion 时多次 | 回归门控；实质是 hidden validation |
| $D_{\mathrm{test}}$ | 否 | 冻结后一次 | 最终泛化报告 |

如果 proposer 根据“candidate 是否通过 $D_{\mathrm{ho}}$”反复调整，哪怕没看到样本内容，也能从一比特反馈逐渐过拟合。因此要记录 query budget，并保留真正未使用的 $D_{\mathrm{test}}$。

## 4. Deterministic gate 与 stochastic gate

### 4.1 Deterministic hard constraints

任何一项失败都拒绝：

$$A_{\mathrm{hard}}=\mathbf1[\mathrm{SchemaOK}]\mathbf1[\mathrm{Capabilities}\subseteq\Pi]\mathbf1[\mathrm{NoVerifierEdit}]\mathbf1[\mathrm{Budget}\le B].$$

### 4.2 Performance constraints

目标修复与回归：

$$\Delta_{\mathrm{in}}=J_{\mathrm{in}}(h')-J_{\mathrm{in}}(h),\qquad \Delta_{\mathrm{ho}}=J_{\mathrm{ho}}(h')-J_{\mathrm{ho}}(h).$$

Self-Harness 的论文规则：

$$\Delta_{\mathrm{in}}\ge0,\qquad \Delta_{\mathrm{ho}}\ge0,\qquad \max(\Delta_{\mathrm{in}},\Delta_{\mathrm{ho}})>0.$$

生产环境可更严格：target split 要超过最小修复量，regression split 要满足 non-inferiority：

$$\mathrm{LCB}_{1-\alpha}(\Delta_{\mathrm{in}})>\epsilon_{\mathrm{repair}},$$

$$\mathrm{LCB}_{1-\alpha}(\Delta_{\mathrm{ho}})>-\epsilon_{\mathrm{reg}}.$$

对同一任务 paired pass/fail，可用 paired bootstrap 或 McNemar-style comparison；连续 reward 可对 per-task difference 做 bootstrap。不要把每次随机 rollout 当独立任务样本。

### 4.3 Sequential testing

搜索会不断看结果并决定是否继续，普通固定样本 $p$ 值会失真。简单工程策略：

- 预注册最大候选数和每 candidate attempts；
- 将 search 与 promotion evaluator 分开；
- 用 alpha spending 或置信序列；
- 先便宜 checks，再昂贵 hidden regression；
- 对最终候选在 untouched $D_{\mathrm{test}}$ 一次评估。

## 5. Metrics：不要只看 pass rate

每个 candidate 输出向量：

$$\mathbf y(h)=(J_{\mathrm{task}},K_{\mathrm{token}},K_{\mathrm{tool}},L_{p50},L_{p95},Q_{\mathrm{policy}},C_{\mathrm{maint}}).$$

还要报告：

- run-to-run variance 和 failure distribution；
- timeout/unknown-outcome 比率；
- retries 与 duplicate side effects；
- per-category/per-repository regressions；
- model/harness/tool/evaluator versions；
- search 中查询 evaluator 的次数。

平均分可掩盖长尾。AHE 的 frozen SWE-bench 总体略升，但部分 repository slice 出现退化，这就是为什么必须同时报告分组结果。

## 6. Failure attribution protocol

```text
1. detect: verifier marks first violated contract
2. localize: map event to component/version
3. hypothesize: list possible causes and counterevidence
4. replay: reproduce with same artifact/environment
5. intervene: change one surface or node
6. compare: paired outcomes under fixed budget
7. decide: propose edit, route to another owner, or mark unknown
```

归因标签集合：

$$z\in\{\mathrm{context},\mathrm{workflow},\mathrm{tool},\mathrm{memory},\mathrm{model},\mathrm{environment},\mathrm{verifier}\}.$$

如果根因是 model capability、环境不稳定或 verifier bug，不应为了让 pipeline 闭环而强行生成 harness edit。

## 7. Threat model

### 7.1 Verifier tampering

候选删除 tests、放宽阈值、替换 judge 或吞掉失败。防线：只读 mount、独立 identity、signed evaluator digest、CI 在候选容器外运行。

### 7.2 Test leakage

将 hidden labels、expected output、judge reasoning 写入 prompt/memory。防线：split ACL、只返回最小 aggregate verdict、query budget、final untouched test。

### 7.3 Permission escalation

候选增加 shell/network/send/delete 能力。防线：capability allowlist 由外部 broker 强制，proposal 只能请求、不能自授。

### 7.4 Budget hacking

换更强模型、增加 calls/context/retries。防线：budget meter 在 runtime 外，所有资源统一计费，candidate 不可改 model identity。

### 7.5 Trace suppression

删除失败日志或只报告成功。防线：append-only external log、sequence number、missing-event alarm。

### 7.6 Objective hacking

DGM 论文中的具体例子：候选移除特殊 logging tokens 绕过 hallucination detector，获得 evaluator 满分，却没有真正消除 hallucinated tool calls。防线不是再加一句 prompt，而是语义级 verifier、diverse checks、hidden adversarial cases 与人工审计。

### 7.7 Memory poisoning

错误/恶意 tool output 被升级为长期规则。防线：data/instruction separation、provenance、candidate state、独立验证和 ACL。

## 8. Permission lattice

能力不是布尔 `safe/unsafe`，可以形成偏序：

```text
read_public < read_private
write_sandbox < write_workspace < external_send
reversible_delete < irreversible_delete
```

若 $c_1\preceq c_2$ 表示 $c_2$ 权限更强，则候选需要能力集合 $C(h')$。自动 promotion 的必要条件：

$$C(h')\subseteq C_{\mathrm{auto}},$$

超出部分进入人工审批。审批绑定 candidate digest、目的、目标资源和 expiry，不能变成永久 wildcard。

## 9. Version、lineage 与 rollback

Harness version manifest：

```json
{
  "digest": "...",
  "parent": "...",
  "model_id": "...",
  "editable_schema": "v3",
  "context": "...",
  "workflow": "...",
  "tools": "...",
  "memory_snapshot": "...",
  "permission_policy": "external-ref",
  "evaluator_digest": "external-ref",
  "metrics": {},
  "migration": "..."
}
```

Rollback 不是切回旧 prompt。必须恢复兼容的 workflow、tool versions、memory schema、state migration 和 permissions。若新版本已产生不可逆外部副作用，只能 compensation/reconciliation，不能声称状态被完全回滚。

## 10. Deployment state machine

```text
DRAFT -> STATIC_VALID -> SANDBOXED -> VALIDATED -> SHADOW -> CANARY -> ACTIVE
   |          |              |           |          |          |
   +----------+--------------+-----------+----------+-> REJECTED
                                                  ACTIVE -> ROLLED_BACK
```

- `SHADOW`：读取真实输入但不产生真实副作用；
- `CANARY`：小流量、严格权限与自动 rollback；
- `ACTIVE`：仍持续监控 slice metrics 和 drift；
- `ROLLED_BACK`：保留失败证据和已发生副作用清单。

Promotion 和 rollback 都由外部 deployer 执行；candidate process 只能产出 signed proposal artifact。

## 11. Long-term utility

短期任务分不包含维护与迁移：

$$J_{\mathrm{long}}=J_{\mathrm{task}}-\lambda_1C_{\mathrm{maintenance}}-\lambda_2C_{\mathrm{migration}}-\lambda_3C_{\mathrm{debug}}-\lambda_4Q_{\mathrm{security}}.$$

这些成本常在 rollout 后才出现。补充信号：静态分析、依赖扫描、兼容测试、代码复杂度、incident rate、恢复时间和人工 review。自动 verifier 无法完整衡量 scientific taste、用户信任和长期架构质量。

## 12. Failure modes

- proposer、runner、verifier 共用同一写身份；
- hidden regression 反馈过细，形成标签侧信道；
- 看 overall 不看 slices；
- 重复 rollout 当独立样本，置信区间过窄；
- accepted/rejected/archive/deployed 状态混在一起；
- rollback 只恢复文件，不处理外部副作用；
- verifier 自身有 bug，却被当作真值；
- 权限申请没有 scope/expiry；
- shadow 环境与真实依赖不同，canary 才暴露问题。

## 13. Engineering checklist

- [ ] control plane 使用独立权限、存储和 deployment identity；
- [ ] 四类 data split 和 query budget 明确；
- [ ] hard gate 与 stochastic performance gate 分开；
- [ ] paired evaluation、置信区间和 slice metrics 完整；
- [ ] trace append-only，所有摘要可回到原始 evidence；
- [ ] capability approval 绑定 digest、scope 和 expiry；
- [ ] rollback 包含 state/memory/tool migration 与副作用 reconciliation；
- [ ] final test 在 harness 冻结后只运行一次。

<!-- NAVIGATION -->
## 导航

- 上一篇：[06 Self-Improving Harness](06_self_improving_harness.md)
- 下一篇：[08 Harness、RL 与权重更新](08_harness_rl_and_weight_updates.md)
- 回到：[专题 README](README.md)
