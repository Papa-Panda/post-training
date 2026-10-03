# EP131 — UniRL：面向统一多模态模型的分布式 RL 后训练框架
> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP131-unirl.html

> 「我们首先拿到一个前提：我们的 action 是否一定是 token？」——讲者全场论证的出发点

## 元信息

- 期号：青稞Talk EP131（B站期号 B131）
- 标题：UniRL：面向统一多模态模型的分布式 RL 后训练框架
- BV：BV1DuJp6PEP6
- 时长：01:02:25（讲授约 41 分钟 + Q&A 约 21 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：吴林玉（字幕自述：NUS 博士生，参与腾讯 UniRL 项目；正式头衔以项目主页为准）
- 相关论文：讲授中未给出论文编号（以 UniRL 项目主页与论文原文为准）
- 相关代码：讲授末尾给出 UniRL 的 GitHub 仓库链接、文档与开发者讨论群（字幕未报具体地址）
- B站链接：https://www.bilibili.com/video/BV1DuJp6PEP6/
- 字幕原文存档：本地 `transcripts/EP131.txt`（1579 条，末条 3736.48 秒，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP131，已存档）清洗提炼，信息来源为 AI 字幕原文（经清洗，专名可能有识别误差）：字幕中框架名作「U02L / unit2 / uni2」、扩散作「DEBUTION / DEPTION」、SGLang 作「as long」、BAGEL 作「背狗 / BGO」、混元作「混元1米3 / 混圆」，均按上下文还原为 UniRL、diffusion、SGLang、BAGEL、混元，正式写法以项目主页与论文原文为准。讲者姓名「吴林玉」为字幕自述口径。凡讲授口径数字均标注「字幕」。

## 一句话总结

UniRL 质疑传统 LLM RL 框架的隐含前提「action 一定是 token」：它把 trajectory 重新组织成由 text segment 与 latent segment 组成的树，用同一条 GRPO 循环（generate → reward → advantage → update → synchronize）同时训练文本推理与扩散生成，并通过「driver 只调度、worker 间点对点传 handle」的数据面与「模型 / 算法两个维度解耦」的可插拔设计，让统一模型（unified model）与拼装管线（unified pipeline）都能在同一个框架里做联合 RL。

## 核心

### 出发点：传统 RL 框架的四个隐含假设在多模态下逐个失效

讲者先把标准 GRPO 循环的既定假设摆到台面上：trajectory 是一串 token；reward 是对整条回答打一个分；group 是同一 prompt 采样出的 N 条回答；ratio 是在每个 token 上把新旧 policy 的概率再 forward 一遍查出来。这四个假设在纯 LLM 场景里非常自然，所以被现有框架当作前提写死。

但当 RL 的对象变成同时具备理解与生成能力的统一多模态模型时，前提逐个失效：扩散模型的 action 是一步去噪得到的下一个 latent，policy 是 latent 空间上的高斯分布，log probability 是该 latent 在高斯下的密度而不是查表；统一模型的一条轨迹会先生成一段文本 reasoning、再生成 image latent，轨迹不再是单一模态的序列；如果框架内部仍默认 action 是 token，生成出来的 latent 没有合理表示、reward 也不知道该对什么打分。讲者明确说，他们不是要再堆一个框架，而是回到同一个 while loop，逐项检查「去掉 token 假设之后什么还成立、什么不成立」（字幕）。

### 扩散 GRPO 与 LLM GRPO：循环不变，四个部件全变

讲者用 diffusion GRPO 做过渡：扩散过程可以理解成逐步去噪，每一步的 action 必须是随机的——用 SDE 形式的采样时每一步都有明确概率，因此也能写出 action 的 log probability。对比表（字幕口径整理）：

| 部件 | LLM GRPO | Diffusion GRPO |
|---|---|---|
| action | 一个 token | 下一个 latent（一步去噪转移结果）|
| policy | 词表上的 softmax 分布 | latent 空间上的高斯分布 |
| log probability | 查 token probability | latent 在高斯分布下的密度 |
| reward | 对最终回答打分 | 对最终生成样本（图像）打分，流程不变 |
| group / advantage | 同 prompt 多样本作 baseline | 同样成立，计算流程不变 |

结论：GRPO 的外层逻辑没有变，变的是 action 分布、policy、log probability 与 reward model 本身。这正是抽象的切入点——框架应该围绕不变的循环组织，把可变部件做成可替换的 segment 级接口。

### 核心抽象：trajectory 是 segment 组成的树

UniRL 最大的设计改动是 trajectory 的组织方式：一条轨迹由若干 segment 组成，text segment 是一段 token 序列，latent segment 是一段扩散 latent；轨迹整体是树形结构——下层 latent segment 的生成依赖上层 text segment（plan）作为条件，而下层 segment 的 reward 不应回传到上层 text segment，除非设计上明确要共享。生成轨迹的 engine 可以替换：可以用训练引擎直接采样，也可以用推理引擎采样（字幕）。

讲者用四类轨迹说明这套抽象的覆盖范围：

1. **纯文本轨迹**：树退化成一条序列，就是传统 LLM GRPO；
2. **纯 latent 轨迹**：对应 diffusion GRPO（如 FlowGRPO 一类，字幕作「举漏 GRPO」），action 从 token 换成 latent，结构仍是序列；
3. **先想后画（think-then-generate）**：统一模型先生成文本 plan、再生成图像，需要同时对文本能力与图像生成能力做 RL；
4. **交错生成**：文本、图像交替出现，是框架在抽象上预留的更复杂情形（暂未实际处理，字幕）。

### 数据面：driver 只调度，数据在 worker 间点对点流动

多模态轨迹打破的第二个假设是「轨迹很小、可以过中心 driver」。LLM 的 token id 只有几 KB 级，走网络协议回传毫无压力；但 diffusion 的 latent 本质上可以 decode 成图片或视频，单个样本数据量可能是 MB 级甚至 GB 级（字幕），还要完整保留中间 latent 以便 replay。此时单 controller 中心化搬运会成为瓶颈。

UniRL 的做法是：仍保留 single controller，但它只负责调度。rollout worker 只把很小的 handle / metadata（字幕作「BSE / biss level」，应为 bytes 级）传给 driver；driver 用 handle 决定哪个 train worker 该去哪个 rollout worker 取数据；真正消费时 train worker 直接从 rollout worker 点对点拉取，数据流不经过 driver。多机场景同样走这条路：通过 NCCL / RDMA 等底层通信做 GPU 到 GPU 的点对点传输，handle 以 tensor 粒度记录它位于哪个 node、哪张 GPU（字幕）。

### 执行面：separate 与 collocated 两种 rollout 布局都支持

框架支持两种布局（字幕作「separate road / collocated robo」，按上下文还原为 separate / collocated rollout）：

- **Separate**：rollout 与 training 不在同一批 GPU 上，天然支持异步训练。讲者指出 LLM 更适合这种布局，因为每条轨迹长短不一、完成时间不一致，异步能消掉长尾；
- **Collocated**：同一批 GPU 上交替做 rollout 与 training。diffusion 每步去噪的完成时间基本一致、没有长尾问题，且其训练侧与推理侧的重放逻辑本质等价，可以直接用 FSDP 这类训练引擎采样，反而更简单、效率也好。

rollout engine 的统一方式是加一层中间层：让 engine 生成 segment 而非原始轨迹输出，再由不同 segment 对应不同 algorithm 去训练。字幕提到的 engine 包括：纯 diffusion 可直接拿 FSDP engine 采样，LLM / VLM 用 SGLang 做推理，扩散侧还有 HF diffusers 一类支持（字幕识别含糊，具体以项目文档为准）。

### reward、group、ratio：三个部件如何随 segment 重定义

- **Reward 可插拔、按 segment 打分**：reward model 必须能接受各种多模态输入，每种模态有各自对应的 reward model，打完分标注回对应 segment。对统一模型，可以对最终图像按审美等指标打分，也可以对中间 plan 本身打分；图像的 reward 还可以向上回传，作为 plan 的 reward（字幕举例：三个 image reward 取平均作为上层 plan 的 reward）。reward 可以共享也可以不共享，取决于具体设计；stepwise reward 在设计上可处理，但他们暂时还没用到（字幕，Q&A）。
- **Group 按树的每一层定义**：同一 plan 生成的多个 image 是一个 group；同一 prompt 生成的多个 plan 也是一个 group。树的每一层都形成自己的 group，advantage 在各自 group 内计算。讲者在 Q&A 中给了具体规模例子：64 个 prompt、每个 prompt 生成 8 个 plan、每个 plan 再生成 8 个 image（字幕）——同一次 rollout 产出的轨迹树上，不同层由各自的算法（LLM 侧 GRPO 与 diffusion 侧 FlowGRPO 类算法）分别处理，因为 latent 的 clip 等细节与 AR 侧并不完全兼容，不能强行共用一个算法。更新目前按 group level 进行；设计上可以联合更新，但他们暂时没有往这个方向研究（字幕，Q&A）。
- **Ratio 靠 segment 级重放**：更新需要新旧 policy 下同一 action 的概率之比，其定义为：

$$r = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)}$$

对 LLM 这只是查 token probability；对 diffusion 则要把这一步去噪转移（从 $x_t$ 到 $x_{t-1}$ ，字幕作「XT 减一到 XT」）完整重放一遍，才能得到当前 latent 在新 policy 分布下的密度，会比查表更重（字幕作「更加 happy」，按上下文应为 heavy）。但抽象是统一的：每个 segment 都能在当前 policy 下被逐步重算出 log probability，这是 UniRL 非常重要的一个接口约定（字幕）。

### 联合训练与模型/算法解耦

因为 text segment 与 latent segment 共用同一个 backbone 与同一条循环，AR loss 与 diffusion loss 可以在同一个 optimizer step 里一起更新，而不是先训完 AR 再训 diffusion（字幕）。讲者强调这不是把两个框架塞进一个仓库：分开训练容易顾此失彼（加强语言能力伤害图像生成，反之亦然），联合训练让两部分互相影响、共同进步，也更不容易出现单侧 reward hacking。主流现状仍是分开训练，讲者判断部分原因是市面上缺少支持联合训练的框架（字幕，Q&A）。

解耦方式：model 侧只负责提供参数与 log probability 重放接口，algorithm 侧只关心 ratio 怎么算、clip 怎么做、regularization 怎么做，不需要知道 segment 是 Stable Diffusion 生成的、还是某个视频模型生成的（字幕作「亲吻英语」，识别不明，按上下文指某种生成模型）。于是模型与算法是笛卡尔积式组合：每个算法大概写 100 行左右（字幕），不需要为每种模型 × 每种算法各写一份适配。

同一套抽象还支持**拼装管线**：用 Qwen3（字幕作「切问三 / 天问三 / 千问三」）做文本 plan、接 Stable Diffusion 3 做图像生成（字幕作「新闻三」，按上下文还原），两个独立模型被组合成一个新的 policy 做联合 RL——前者学会生成能导出更好图片的 plan，后者学会基于 plan 生成更好的图片（字幕）。

### 现状与路线图

当前支持重点在 diffusion 模型与统一模型：讲者称主流扩散模型大多已支持，统一模型方面支持混元与 BAGEL（其 LLM 能力部分），LLM / VLM 方面支持 Qwen3 与 Qwen-VL（字幕作「千问3000问VO」，按上下文还原），在这些模型上训练都能稳定工作（字幕）。路线图四项：补齐更多模型与算法支持（庞大的 LLM / VLM 模型家族仍需逐个适配）；把 AR 侧的效率优化迁移过来（如 load balancing、micro-batching，字幕作「deep 的 balancing、MICHAELBCHING」）；支持更多训练后端（当前是 FSDP，未来引入 vLLM、Megatron 一类，字幕作「VOMEI、metro」）；支持异步训练（先可以从把 reward 打分与 rollout 异步化这种简单形态开始，字幕）。

Q&A 其他要点：AR 训练时需要把 latent mask 掉（这也是 BAGEL、混元这类原生统一模型自身的设计），且两个模态的序列分开存储、按树的层级分别存（字幕）；Omni 类带语音的模型（如 thinker/talker 两阶段）暂不支持、在路线图上，但抽象设计时考虑过（字幕作「OMI / OMEI 模型、千万像」，按上下文指 Qwen-Omni 一类）；与只支持纯 diffusion GRPO 的框架（如 FlowGRPO、DanceGRPO，字幕作「BO 欧 only」）相比，UniRL 的出发点是统一模型本身，包括把两个独立模型拼成一个统一 policy（字幕）。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| 单样本轨迹数据量（diffusion latent） | MB 级或 GB 级（LLM token id 仅几 KB 级） | 字幕（数据面部分） |
| 树形 group 的规模例子 | 64 个 prompt × 8 个 plan × 8 个 image | 字幕（Q&A） |
| 每个 algorithm 的代码量 | 约 100 行 | 字幕（总结部分） |
| 多机验证规模 | 至少在 8 个 node 上验证过，点对点传输、效率高（讲者原话对上限 node 数表述含糊） | 字幕（Q&A） |
| 资源起步规模 | 小实验可从 1 个 node（8 张卡）开始 | 字幕（Q&A） |
| Stable Diffusion 验证实验的 step 时长 | 约一分钟量级（配置表述字幕识别含糊，如「48×16」「1×8 18」无法确读） | 字幕（Q&A） |

## 可迁移

- 做多模态或异构 policy 的 RL infra 时，先把「action / policy / log-prob / group」四个接口按 segment 抽象出来，再让每种模态各自实现重放与打分；不要把 token 序列的假设写死在循环里——这是本期对 RL infra 最直接的一条。
- 数据面纪律：当轨迹载荷从 KB 级涨到 MB/GB 级，中心化 driver 搬运必然成为瓶颈；「调度与数据分离、handle 寻址、worker 间点对点拉取」是可直接借用的系统结构。
- 布局选择经验：轨迹长度方差大的（AR）优先 separate + 异步消长尾；每步时长齐整的（diffusion）collocated + 训练引擎直接采样反而更简单——按 workload 的时间分布选布局，而不是按习惯。

## 疑问 / 下一步

- 联合训练的实际收益缺少量化：讲者称联合训练「天然效果很好」、能减少单侧 hacking，但全场没有给出联合 vs 分开训练的对照实验数字；协同更新时 diffusion 训练影响 AR、AR 影响 diffusion 的算法设计，讲者自己也说「需要比较复杂的算法设计」（字幕，Q&A）。
- 更新粒度停在 group level：不同层的 group 目前分别更新，跨层联合更新（让 image reward 更系统地塑造 plan）在设计上可行但未研究（字幕，Q&A）。
- 多机规模与吞吐数字口径含糊：Q&A 中 node 上限与实验配置的数字字幕识别质量差（见数字总表注），需要看项目文档或论文核对后再引用。
- 交错生成（文本-图像交替）与 Omni 语音模型都还停在抽象预留与路线图阶段，实际落地形态待观察。

## 原文金句（1-2句）

> 「我们首先拿到一个前提……我们的 action 是否一定是 token？在传统 LM 里面，我们的 action 一定是个 token……但在一个多模态的角度下，我们的 action 还是一个 token 吗？」（字幕 00:44–01:05，按干净口径转写）

> 「我们不再关心的事情是某种模型长成什么样，而关心的事情是我们是否能找到一套统一的抽象，来把各种模态的 RL 都统一到一个框架里。」（字幕 16:37–16:53，按干净口径转写）

> 「UniRL 并不是把 AR 的训练和 diffusion 的训练放到同一个仓库里，而是我们重新设计 RL loop 里面的一些抽象，来达到这种效果。」（字幕 36:29–36:42，按干净口径转写；字幕原作「air china / DEBUTION 的 china」，按上下文还原）
