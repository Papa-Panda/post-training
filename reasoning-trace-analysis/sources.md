# Sources: 一手来源清单

> 本专题每个数字、年份、作者、结论都来自下面实际打开过的页面。条目按章节引用顺序排列；"核验内容"只写打开页面时亲眼看到的事实。
>
> Every number, year, author, and claim in this topic comes from a page actually opened below. Entries follow chapter citation order; "verified" notes record only what was seen on the page.

## 线 1：trace 结构解剖

- **DeepSeek-R1**（01）
  https://arxiv.org/html/2501.12948
  核验：摘要 "emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation"；§2 "incorporating verification, reflection, and the exploration of alternative approaches"；正文 "backtrack and explore alternative approaches"；Table 2 "aha moment" 标题与 "rethink using an anthropomorphic tone"；R1-Zero reward "solely based on the correctness of final predictions…without imposing constraints on the reasoning process itself"；"poor readability and language mixing…combining English and Chinese within a single chain-of-thought response"。

- **Think Deep, Not Just Long**（02）
  https://arxiv.org/html/2602.13517
  核验：首位作者 Wei-Lin Chen，2026；settling depth 定义（JSD、阈值 $g$ 、min-envelope）；deep-thinking token 判定 $c_{t}\ge\lceil\rho L\rceil$ ；DTR 定义；benchmark 为 AIME 2024、AIME 2025、HMMT 2025、GPQA-Diamond；模型族 GPT-OSS、DeepSeek-R1、Qwen3；Figure 1 长度相关 $r=-0.544$ 、DTR 相关 $r=0.828$ （GPT-OSS-120B-medium 图示，非普适常数）；Think@n 高 DTR 选择、短 prefix 估计与早停、约一半 inference cost 达到或超过 self-consistency。

## 线 2：长度、效率与 overthinking

- **DiffAdapt**（03）
  https://arxiv.org/html/2510.19669
  核验：作者 Xiang Liu、Xuming Hu、Xiaowen Chu、Eunsol Choi；U 型熵（easy 高熵高准确、medium 低熵、hard 高熵真不确定）；easy→medium 熵降 22–25%；三档配置 Easy（temperature 0.5、 $0.4\times$ Max）、Normal（0.8、 $1.0\times$ Max）、Hard（0.4、 $0.5\times$ Max）；Hard 为 fail-fast；oracle 约 $50\%$ token 节省、准确率提升超 $10\%$ ；三阶段（proxy 标注 → final-hidden-state probe → 推理路由）；摘要：5 模型、8 benchmark、token 最多降 $22.4\%$ 、准确率持平或更好。

- **Self-doubt**（Peng et al.，04）
  https://arxiv.org/html/2505.23480
  核验：作者 Keqin Peng、Liang Ding、Yuanxin Ouyang、Meng Fang、Dacheng Tao；self-doubt 定义为已得正确答案后仍重复验证（excessive token usage devoted to re-verifying an already-correct answer）；三分类 Overthinking with Self-Doubt / without Self-Doubt / Non-Overthinking；两步法（先查输入有效完整、再简洁回答）；prompt 原文 "Before reasoning deeply, check whether all necessary information is available…"；实验表平均 reasoning length 降 $37.1\%$ 、accuracy 增 $3.6\%$ （该表平均，非普适）。

- **SelfDoubt / HVR**（04）
  https://arxiv.org/html/2604.06389
  核验：作者 Satwik Pandey、Suresh Raghu、Shashwat Pandey； $\mathrm{HVR}(T)=h(T)/(v(T)+1)$ ； $\mathrm{HVR}=0$ gate 在 7 模型 3 数据集上正确率 $96.1\%$ 、coverage $25.4\%$ 。

- **Know When to Stop**（04、06 指针）
  https://arxiv.org/html/2607.00482
  核验：首位作者 Chia-Hsuan Lee；六信号 S1 Repetition、S2 Hedging、S3 Abandonment、S4 Contradiction、S5 Recomputation、S6 Length outlier；S3 abandonment 在 incorrect traces 中多 4.1–4.3 倍；控制长度后 incorrect traces 仍有更多无效 self-reflection；DASH 为 segment-level credit assignment（本专题仅作指针）。

- **Reasoning Under Constraint / batch prompting**（05）
  https://arxiv.org/html/2511.04108
  核验：作者含 Saurabh Srivastava、Janit Bidhan；DeepSeek-R1 与 OpenAI-o1、13 benchmark；batch size 增大推理 token 2,950 → 710（降 $76\%$ ）、准确率持平或升；Figure 1 主对比 BS=1→15；Figure 4 四模式 BS=1→5：hedging $0.42\to0.18$ 、rechecking $0.55\to0.22$ 、re-derivation $0.38\to0.16$ 、tangents $0.31\to0.12$ ，平均降 $59\%$ ，频率按 trace length 归一化；输出 token 降 83–88%；显式约束（"Use no more than 100 tokens in thinking"）被忽略或牺牲准确率；候选解释 shared-context pressure、sequential anchoring / in-context pattern induction、implicit difficulty calibration 明确标注为 hypotheses。

## 线 3：过程监督 PRM

- **Let's Verify Step by Step**
  https://arxiv.org/abs/2305.20050
  会议版 PDF：https://proceedings.iclr.cc/paper_files/paper/2024/file/aca97732e30bcf1303bc22ac3924fd16-Paper-Conference.pdf
  核验：作者 Hunter Lightman、Vineet Kosaraju、Yura Burda、Harri Edwards、Bowen Baker、Teddy Lee、Jan Leike、John Schulman、Ilya Sutskever、Karl Cobbe；arXiv 初次提交 2023-05-31；ORM 只用整解最终正确性标签、测试时用 final token 预测作解得分；PRM 每步末 token 预测该步正确性、一次 forward pass、解得分=各步正确概率之积；标签 positive / negative / neutral；MATH 子集上 $78.2\%$ （摘要约 $78\%$ ）；active learning 数据效率 $2.6\times$ ；PRM800K：800K step 标签、75K solutions、12K problems；Uesato et al. 2022 在 grade-school math 上 ORM≈PRM；generator 用 newline-delimited step-by-step 格式方便解析。

## Deferred：faithfulness 后续入口（本专题不展开）

- https://arxiv.org/abs/2505.05410
- https://arxiv.org/abs/2510.04040v1
- https://arxiv.org/abs/2509.13334v1

以上三条仅在 arXiv abs 页确认存在，未核验内容，不引用其结论。
