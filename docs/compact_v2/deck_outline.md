# COMPACT Slides 总控大纲（10 页）

> 用途：制作 slides 的 master document。每页包含：标题、页面文案、转场句、图来源。
> 叙事主线：**四性质评分表（页2）→ 审判现有方法（页3-4）→ 我们的哲学（页5）→ 哲学落地为架构（页6）→ 三模块逐一兑现（页7-9）→ 实验全绿呼应（页10）**。
> 两个榫头：页 2 的"理想记忆四性质"是全场评分表；页 5 的"好记忆三判据"解释四性质从何而来。

---

## 章节划分与 agenda 页

10 页切成 **4 个章节**，章节名统一用问句，层层递进：

| 章节 | 页码 | 副标题 |
|---|---|---|
| **Part 1｜为什么需要记忆？（Why）** | 页 2 | 流式交互的上下文困境，与一张"理想记忆"的评分表 |
| **Part 2｜现有方法差在哪？（What's missing）** | 页 3-4 | 用评分表审判三大记忆形态与两条 latent 路线 |
| **Part 3｜我们的答案：COMPACT（How）** | 页 5-9 | 一条设计哲学，三个兑现它的模块（哲学 页5 → 总览 页6 → 模块①②③ 页7-9） |
| **Part 4｜效果与验证（Does it work）** | 页 10 | 评分表全绿，数字说话 |

**Agenda 页（放在页 1 标题页之后）文案**：
> - **01 为什么需要记忆** —— 流式 GUI 交互的困境
> - **02 现有方法差在哪** —— 三大形态，无一及格
> - **03 我们的答案 COMPACT** —— 一个模型，压缩记忆、预期未来、做出决策
> - **04 效果与验证** —— 评分表全绿

**使用建议**：
- Part 3 是主体（5 页，约占一半时长），agenda 上 03 可展开两行小字"设计哲学 + 三个模块"，避免听众在模块页迷路；
- 每个章节隔页重放 agenda 并高亮当前章节（15-20 分钟分享很管用）；
- 页 5（哲学）放在 Part 3 开头而非独立成章：先给信念，再给实现。

---

## 页 1：标题页

**标题（中文 PPT 定稿）**
COMPACT：一个模型，压缩记忆、预期未来、做出决策
—— 流式场景下的自压缩行动者

**英文标题（投会/论文备用）**
COMPACT: Streaming Memory Compression with a Self-Compressing Actor for Long-Horizon GUI Agents

**副标题（backronym）**
**C**ompressing **O**nline **M**emory via **P**redict-**A**nd-**C**orrect **T**ransitions

**文案**
- 一句话：一个模型，同时压缩历史、维护记忆、做出决策 —— 无需外置压缩器，上下文恒定 $O(1)$。
- 标题三个排比动词对应叙事主线：**压缩记忆**（模块①②，Faithful）→ **预期未来**（模块②预测步 + $\mathcal{L}_{obs}$，Predictive）→ **做出决策**（模块③，Useful）。开场念标题时即可预告全场结构。

**转场句**
> "要讲清楚为什么需要这样一个模型，先看 GUI Agent 面临的根本困境。"

**图来源**
- AI prompt（标题页概念图：机器人侧脸，截图胶片螺旋压缩成记忆芯片，一条箭头指向动作图标）：
  > A minimal, elegant title-slide illustration: a single robot head silhouette in profile; inside its head, a long film strip of GUI screenshots spirals inward and compresses into a small glowing memory chip; from the chip, a single clean arrow points to a mouse-cursor action icon. Convey "one brain that compresses, remembers, and acts". Flat vector style, white background, deep blue and orange palette, lots of negative space, top-conference keynote aesthetic, no text.

---

## 页 2：背景 — 为什么需要记忆？（立评分表）

**标题**
Why Memory? —— 没有记忆，每一步都是"初见"

**文案**
- **先直接回答 Why**：没有记忆，Agent 每一步都是第一次见到这个世界 —— 不知道来过这里、不知道刚做过什么、不知道哪条路是死路。长程任务无从谈起
- **但最朴素的记忆 = 全量历史塞上下文**：GUI Agent 是流式决策者，历史线性增长 → 上下文爆炸、注意力稀释、成本 $O(t)$
- 所以需要**外挂的压缩记忆**，且它必须满足四条性质（**全场评分表**）：
  1. **固定容量**：memory bank 大小不随步数增长
  2. **$O(1)$ 更新**：每步写入成本恒定
  3. **按需查询**：只读取与当前决策相关的信息
  4. **按需遗忘**：选择性保留/擦除，而非无脑累积
- 现状：没有任何一类方法同时满足四条

**转场句**
> "这四条性质就是我们的评分表。用它去审判现有的三大记忆形态——没有一家能及格。"

**图来源**
- AI prompt（上下文增长 vs 固定记忆对比图）：
  > A clean two-part infographic about context growth in GUI agents. Left: a chat-style context window filling up with stacked screenshot thumbnails and action text, overflowing with a red warning icon, labeled "Full history: O(t) growth". Right: the same agent with a small fixed-size memory chip icon, context stays tiny, labeled "Ideal memory: fixed size, O(1) update". Flat vector style, white background, blue/orange palette, minimal text, academic slide illustration.

---

## 页 3：相关工作 I — Agent 记忆的三大形态

**标题**
Forms of Agent Memory（分类依据：*Memory in the Age of AI Agents: A Survey*, 2025）

**文案**（三栏，短板全部对照页 2 评分表的四个性质【容量】【更新】【查询】【遗忘】标注）

| 形态 | 代表工作 | 评分表上的失分 |
|---|---|---|
| **文本记忆** (Token-level) | MemGPT、Reflexion、MemoryBank、Generative Agents、A-MEM | ✗【更新】写入 = LLM 摘要调用，贵且有损；✗【查询】文本与模型内部表征不对齐 |
| **参数化记忆** (Parametric) | 长上下文 SFT、StreamingLLM、MemLoRA、Memory Decoder | ✗【更新】更新 = 训练，无法在线写入；✗【遗忘】无法选择性擦除（灾难性遗忘） |
| **隐状态记忆** (Latent) | Gist、AutoCompressor、MemoryLLM、SnapKV、JAMEL | ✓【容量】【更新】有潜力：机器原生、token 高效；✗【查询】【遗忘】仍无好解 |

- takeaway：隐状态路线最有希望拿下【容量】【更新】；【查询】【遗忘】的成败取决于怎么压缩、怎么更新 —— 这正是页 4 的细分。

**转场句**
> "隐状态路线有望拿下容量和更新两条。但查询和遗忘做得如何，要看它的两条细分技术路线。"

**图来源**
- draw.io：`docs/compact_v2/slide2_memory_taxonomy.drawio`（三分支分类图）
- AI prompt ×3（形态图标，备选）：
  > 文本记忆：Flat vector icon-style illustration for "textual memory": an AI robot head writing natural-language notes into an open notebook, with a magnifying glass searching the notes later. White background, blue palette, minimal, academic slide icon.
  > 参数化记忆：Flat vector icon-style illustration for "parametric memory": a neural network brain with memory engraved into its weight connections, glowing synapses, a small gear indicating training. White background, purple palette, minimal, academic slide icon.
  > 隐状态记忆：Flat vector icon-style illustration for "latent memory": a stream of long documents entering a funnel, emerging as a short row of glowing compact tokens/vectors. White background, green palette, minimal, academic slide icon.

---

## 页 4：相关工作 II — 隐状态记忆的两条传统路线（缺口 = 设计目标）

**标题**
Latent Memory — 两条传统路线的未竟之业

**文案**（两栏 + 底部五维对比表）

- **(a) 压缩器式**（Gist / AutoCompressor / JAMEL）：独立 compressor 把历史压成 latent tokens，append-only
  → ✗【容量】bank 线性增长 $O(t)$；✗【遗忘】无遗忘机制；✗【查询】全量 attend 随 $t$ 膨胀；外加 compressor 与 actor 分离、目标不一致
- **(b) 后压缩式**（SnapKV / PyramidKV / H2O）：事后对 KV cache 剪枝/聚合
  → ✗【遗忘】剪枝 = 暴力删除，信息**永久丢失**；✗【查询】被动截断而非按需读取；为省显存而压，不为决策而学

| 性质（评分表） | 压缩器式 | 后压缩式 | **COMPACT（页 7-9 兑现）** |
|---|---|---|---|
| Bank 大小 | 线性增长 | 有界但被动 | **固定** |
| 更新成本 | 追加 $O(1)$，读取 $O(t)$ | 启发式剪枝 | **$O(1)$ 残差更新** |
| 按需查询 | 全量 attend | 无 | **Cross-Attn + 门控** |
| 按需遗忘 | 无 | 暴力删除 | **可学习 Kalman 增益** |
| 训练-推理一致 | 分离模型 | — | **统一 + 零初始化** |

- 脚注（口播）：最接近的 δ-mem 同样是固定矩阵 + 残差更新，但容量极小（8×8）、无不确定性建模、未在真实 GUI 任务验证。

**转场句**
> "这些失败不是工程问题，而是对'什么是好记忆'的回答就不完整。我们的回答有三条——这就是 COMPACT 的设计哲学。"

**图来源**
- draw.io：`docs/compact_v2/slide3_latent_memory_compare.drawio`（三 pipeline 对比图；本页只取 (a)(b) 两行，(c) 行留给页 6）
- AI prompt ×2（传统路线，备选）：
  > 压缩器式：Flat vector pipeline diagram: a frozen neural network "compressor" block squeezing each interaction step (screenshot + click icons) into one small token, tokens appending into an ever-growing horizontal tape that extends off the page edge, labeled "append-only, grows O(t)". A separate "actor" block reads the whole tape. White background, blue/orange, minimal, academic slide illustration.
  > 后压缩式：Flat vector pipeline diagram: a tall stack of key-value cache tokens with scissors cutting away most of them, leaving a small surviving subset; crossed-out tokens fade to gray and are labeled "permanently lost". White background, blue/red accent, minimal, academic slide illustration.

---

## 页 5：设计哲学 — 什么是"好"的记忆？（转折页）

**标题**
Design Philosophy — What Makes a Memory "Good"?

**文案**（三段递进）

- 常见答案：**对决策有用** —— 优化动作 ≈ 优化记忆（$\mathcal{L}_{act}$）
  → 必要但不充分：只服务下一步动作的"投机特征"，无法跨步积累
- 我们的三个判据：
  1. **Faithful**：固定容量下忠实压缩历史
  2. **Predictive**：给定动作 $a_t$，能**预期**下一帧观测 $O_{t+1}$
  3. **Useful**：被决策真正消费
- 桥接方法（判据如何推出性质）：对每个判据问一句 **"什么情况算违反它？"** —— 违反的方式，就是它强加的工程约束：

| 判据 | "违反它长什么样" ⟹ 推导 | 兑现的性质（页 2 评分表） | 落成的模块（按流水线 ①→②→③） |
|---|---|---|---|
| **Predictive** | 预期必然有时出错；**错了却不知道**比没有预期更危险 ⟹ 需要显式预期 + "信谁"可计算 ⟹ $K=\hat P/(\hat P+R)$ | 【按需查询】【按需遗忘】 | ① Predict（预期产生）+ ② Correct（预期校验） |
| **Faithful** | bank 涨爆后截断（JAMEL）、或被剪枝删除（SnapKV）= 不忠实 ⟹ 形态有界（前提）+ 写入最小代价（只写意外） | 【固定容量】【$O(1)$ 更新】 | 有界形态（页 6 交代）+ ② Correct 残差写入 |
| **Useful** | 记忆不被读（v1 推理旁路）或读了毁掉基座 = 没用 ⟹ 注入必须可门控、零初始化 | 训练-推理一致 | ③ Zero-Born Steering |

- 模块①②是 Predictive 的一体两面：先显式预期（动作作控制量），再**只按"意外"校正** —— 解耦让校正只响应预测之外的新息 $\Delta M$，这正是 $O(1)$ 残差更新的来源；② 的残差写入同时兑现 Faithful 的写入要求。
- 损失各有归属（页 7-9 页顶标注）：$\mathcal{L}_{obs}$ 监督①的预测质量；$\mathcal{L}_{nll}$ 校准②的 $R$；$\mathcal{L}_{mem}$ 约束②产出的 $M'$；$\mathcal{L}_{act}$ 是③注入通路的唯一主流监督。
- 点睛句：**先预期、再对账、后导引——Predictive 驱动①②，Faithful 沉淀在②的残差里，Useful 落地在③。**

**金句（页底居中）**
> 我们不为"记住过去"而设计记忆，而为"预期未来"而设计记忆 —— **记忆即世界模型**。

**转场句**
> "这套哲学落到架构上，就是层内三步流水线：Predict → Observe+Correct → Inject。"

**图来源**
- AI prompt（三柱图）：
  > A conceptual three-pillar illustration for "what makes a good memory". Three stone pillars standing on a base shaped like a brain: pillar 1 topped with a mirror reflecting a film strip labeled "Faithful (compresses history)", pillar 2 topped with a crystal ball showing a future screen labeled "Predictive (anticipates the next observation)", pillar 3 topped with a dartboard icon labeled "Useful (serves decisions)". The middle pillar glows brightest. Flat vector, white background, blue/orange palette, minimal, academic keynote style.

---

## 页 6：整体结构 — 数据流与三个模块的位置

**标题**
Overview — One Pass Through COMPACT

**文案**
- 每步输入：指令 + 当前观测（+上一步动作 $a_{t-1}$，独立 side 输入）→ 上下文恒定 $O(1)$
- 每层解码器外挂侧边记忆（固定 16 槽，**有界形态是 Faithful 的前提**），层内三步流水线：
  - **① Anticipative Transition（预期式状态转移）** — 状态怎么转移：动作作控制输入，先"预期"世界（Predictive 前半，引入 $P$ 轨，$\mathcal{L}_{obs}$）
  - **② Calibrated Memory Update（校准式记忆更新）** — 观测怎么入账：与预期对账、按置信度残差写入（Predictive 后半 + Faithful，$\mathcal{L}_{nll}$、$\mathcal{L}_{mem}$）
  - **③ Zero-Born Steering（零起点记忆导引）** — 记忆怎么用：导引向量注入，零初始化（Useful，$\mathcal{L}_{act}$）
- $(M_t, P_t, e_t)$ 跨步携带；动作由记忆条件化生成（KV-cache 预填充）

**转场句**
> "接下来三页按层内计算顺序（Predict → Observe+Correct → Inject）各放大一步，页顶灰字标明兑现的判据与相关损失。"

**图来源**
- draw.io：`docs/compact_v2/method_overall_dataflow.drawio`（整体数据流，①②③ zoom 徽章已标注；含跨步携带回路、动作反馈回路、总损失条）
- token 级细节图：`docs/compact_v2/layer_l_token_flow.drawio`（序列分段 + 掩码 + 层内五步，见附录页 11）
- 可参考：`docs/compact_v2/fig3_session_loop_v2.png`（session loop 现成图）

---

## 页 7：模块 ① Anticipative Transition（预期式状态转移，4a Predict）

**页顶回扣行（灰字）**：兑现判据 **Predictive（前半：预期的产生）**；引入不确定度轨 $P$；相关损失 $\mathcal{L}_{obs}$

**挑战**（引用页 4 的具体缺陷）
append 式记忆（JAMEL）只是"过去的堆积"：不知道"我做了什么、世界因此变成什么样" —— 无因果、无预期；无条件的 RNN 转移则让动作与状态演化脱钩

**我们的方法**：动作条件下的先验状态推演（action-conditioned prior transition）

1. **动作条件化的状态转移**：上一步动作经独立嵌入通路（side input，不进入主序列）生成控制向量，以 FiLM 调制 GRU 完成先验推演 $\hat M = \mathrm{GRU}(W_a a_\downarrow,\ \gamma \odot M + \beta)$，$\hat M\ [B,16,512]$ —— 显式建立"动作 → 状态演化"的因果联系
   ↩ 回应"动作与状态演化脱钩"
2. **不确定度的同步推演**：逐槽方差轨 $P$ 与内容轨联合更新，$\hat P = \sigma(\lambda_l)\odot P + \mathrm{softplus}(Q_\theta(a_\downarrow)) + \gamma_e \bar e$；含层级化时间常数（浅 0.70 / 中 0.85 / 深 0.95）与 surprise 反馈项，使预测置信度成为可计算量
   ↩ 回应"无因果、无预期"
3. **预测能力的显式监督**：观测预测头 $\hat Z = g_\phi(\bar{\hat M})$ 以 $\mathcal{L}_{obs}$ 直接优化；预测残差定义为 surprise $e = \|\hat Z - \bar Z_t\|^2$，并反馈驱动下一步的不确定度膨胀
   ↩ 预测能力不依赖 $\mathcal{L}_{act}$ 间接获得

**一句话**
状态转移由动作显式驱动，且其预测质量受到直接监督 —— 记忆由此具备"预期"的语义。

**转场句**
> "预期有了，接下来让真实观测进场——和预期对账。"

**图来源**
- draw.io：`docs/compact_v2/method_modules_zoom.drawio` 第 1 页（locator 红圈标 Predict + FiLM-GRU / $P$ 轨 / $\mathcal{L}_{obs}$ 细节）
- AI prompt（split-view zoom 图，备选）：
  > Split-view academic diagram. Left: small overview of a transformer layer with the state-transition step highlighted by a red zoom circle. Right: magnified view of an "anticipation" module: a mouse-click action icon enters as a control knob (FiLM) on a GRU gear, rolling a memory rack (16 slots) forward into a dashed "predicted" state; beside it a row of small gauge icons labeled "uncertainty track P" inflating after a spark labeled "surprise e". Flat vector, white background, blue/orange palette, minimal text.

---

## 页 8：模块 ② Calibrated Memory Update（校准式记忆更新，4c Observe + 4d Correct）

**页顶回扣行（灰字）**：兑现 **Predictive（后半：校正）+ Faithful（残差写入）** → 性质【$O(1)$ 更新】【按需查询】【按需遗忘】；相关损失 $\mathcal{L}_{nll}$、$\mathcal{L}_{mem}$

**挑战**
观测如何进记忆？v1 退化成单向量 mean（16 槽收到同一更新，秩-1）；JAMEL 全量 attend（成本随 $t$ 膨胀）—— 共同病根：写入无选择、无量化依据

**我们的方法** —— *"Write only the surprise：对账式写入"*

1. **只看该看的**：$k{=}4$ 个可学习 query 对 prompt 位做掩码 AttnPool，把当前观测萃取为 $Z_t\ [B,4,512]$ —— 排除 response：记忆不许偷看答案
   ↩ 回应"秩-1 退化"：$k>1$，16 槽各自收到不同的修正
2. **只记意外的**：新息 $\Delta M = \mathrm{CrossAttn}(\hat M, Z_t)$ 经增益 $K = \hat P/(\hat P{+}R)$ 按比例入账 —— $M' = \hat M + K\odot\Delta M$，$P' = (1-K)\hat P$
   ↩ 回应"写入无选择、无量化依据"：改不改、改多少，由 $\hat P$ 与 $R$ 的博弈显式决定
3. **让置信度兑现**：$\mathcal{L}_{nll}$ 强迫 $R$ 对齐真实误差，$\mathcal{L}_{mem}$ 保持 $M'$ 有界
   ↩ 信任机制是训练出来的，不是手调规则
4. **该稳则稳，该变则变**：平稳期 $K$ 小、观测只润色；界面突变时 surprise 顶高 $\hat P$、$K \to 0.93$ 瞬间重写
   ↩ 评分表【按需查询】【按需遗忘】的落地形态

**一句话**
差额入账，比例由置信度说了算。

**备询 Q&A**（大概率被问，可放附录）
- **Q：$P$ 大 → $K$ 大？** A：对，$K=\hat P/(\hat P+R)$ 随 $\hat P$ 单调增 —— $\hat P$ 大 = 不信预测 → 多信观测。但 $K$ 是 $\hat P$ 与 $R$ 的**相对博弈**：观测噪声大时 $K$ 同样被压低。更新后 $P'=(1-K)\hat P$ 收缩（"刚校对过，比较有把握"），下一步 Predict 再经 $Q(a)$ 与 surprise 膨胀 —— 收缩-膨胀呼吸循环，见 `fig2_kf_dynamics_v2.png`。
- **Q：层级化 $\lambda<1$，$P$ 每步衰减，与"久不观测则越不确定"矛盾吗？** A：不矛盾。$\hat P=\sigma(\lambda)P + \mathrm{softplus}(Q_\theta(a)) + \gamma_e\bar e$ 是 stable-KF 形式（收缩型转移 $FPF^\top{+}Q$，$\|F\|{<}1$）：GRU 有界激活使转移本身收缩，$\lambda$ 管"旧方差多快失效"（时间常数：浅层快 / 深层慢），"增长"由显式的 $Q(a)$ 与 surprise 反馈承担。若想要"久不观测单调增"，应改 $Q$ 项而非把 $\lambda$ 放到 1 以上（破坏稳定性）。$\lambda$ 可学习，层级值只是初始化。

**转场句**
> "记忆更新完毕，最后一步是让它回到主干参与决策——而且不能碰坏预训练模型。"

**图来源**
- draw.io：`docs/compact_v2/method_modules_zoom.drawio` 第 2 页（locator 红圈标 Observe+Correct + 掩码 / Kalman / 损失细节）
- 现成仿真图：`docs/compact_v2/fig2_kf_dynamics_v2.png`（界面突变时 $K$ 跳至 0.93 再回落 —— 优先用真图）
- AI prompt（split-view zoom 图，备选）：
  > Split-view academic diagram. Left: small overview with the observe-and-correct step highlighted by a red zoom circle. Right: magnified view of an accounting scene: an eye icon (4 query tokens) reads the current screen into "observation Z", then a balance scale compares "prediction" vs "observation", the difference (delta) flows through a valve labeled "gain K = P/(P+R)" into the memory rack; a small gauge labeled "R, calibrated by NLL". Flat vector, white background, blue/green palette, minimal text.

---

## 页 9：模块 ③ Zero-Born Steering（零起点记忆导引，4e Inject）

**页顶回扣行（灰字）**：兑现判据 **Useful** → 训练-推理一致；相关损失 $\mathcal{L}_{act}$

**挑战**
外挂记忆通路若随机初始化，第 0 步就扰动预训练模型；v1 甚至在推理时把记忆整体丢弃

**我们的方法** —— *"Born zero, steer on demand：从零出生，按需导引"*

1. **从零出生**：$W_\uparrow{=}0$ 零初始化 + 小门控 $g_l{=}0.1$ —— 第 0 步 ≡ 预训练 LLM，梯度却立即可训（$\tanh(g_l)\neq 0$）
   ↩ 回应"扰动预训练模型"
2. **按需导引**：记忆经 Cross-Attn 作为 steering vector 注入主干 —— $H = h + w_l \tanh(g_l)\, W_\uparrow\, \mathrm{CrossAttn}(Q{=}h_\downarrow,\, K,V{=}M)$；浅层强（0.8）深层弱（0.3）
   ↩ 与页 7 的层级 $\lambda_l$ 首尾呼应：浅层管"看得清"，深层管"记得住"
3. **全程在线**：推理时记忆增强前向 → KV-cache 预填充 → 增量解码，动作真正以记忆为条件
   ↩ 回应"v1 推理丢弃记忆"：训练-推理一致，通路端到端闭环
4. **一个目标收尾**：response 段 $\mathcal{L}_{act}$ 是注入通路的唯一主流监督；零起步保证它从"无害"开始学
   ↩ Useful 判据的最终兑现

**一句话**
记忆不是硬拼接的上下文，而是可查询、可门控、从零长起的导引信号。

**转场句**
> "三步合起来：先预期、再对账、后导引——Faithful、Predictive、Useful 各就各位。但全绿不是嘴上说的——看实验。"

**图来源**
- draw.io：`docs/compact_v2/method_modules_zoom.drawio` 第 3 页（locator 红箭头标注入通路 + 公式 / 推理闭环 / $\mathcal{L}_{act}$）
- AI prompt（split-view zoom 图，备选）：
  > Split-view academic diagram. Left: small overview of the architecture with the injection arrow from memory into the transformer highlighted by a red zoom circle. Right: magnified view of the injection gate — a memory bank feeding through a faucet/valve icon labeled "learnable gate, zero at birth" into the main hidden-state stream as a steering current bending the flow; a small badge reads "step 0 = pretrained model". Flat vector, white background, purple/orange palette, minimal text.

---

## 页 10：实验（待制作）

**开头固定动作**
重放页 4 的五维对比表 —— 此时 COMPACT 一列**全部打绿**，与页 2 的评分表首尾呼应。

**候选内容**（按实验进度选用）
- 主结果：vs. 全量历史 ReAct / 滑动窗口 / 文本摘要基线的成功率与 token 成本
- 效率曲线：上下文长度恒定 vs. 基线线性增长（呼应页 2 的困境图）
- 消融：去 $\mathcal{L}_{obs}$ / 去 $\mathcal{L}_{nll}$ / 去 surprise 反馈 / 去层级先验
- 记忆动力学可视化：长会话中 $K$、$P$、$e$ 的演化（复用 `fig2_kf_dynamics_v2.png` 风格）
- 训练-推理一致性验证：记忆条件化生成 vs. v1 旁路

**转场句**（收尾）
> "COMPACT 的答案：好记忆不是记住更多，而是预期得更准。"

---

## 页 11（附录）：Token 级流程图（被追问实现细节时用）

**标题**
Appendix — Layer $l$ Token-level I/O Flow

**内容要点**
- 主序列 7 段解剖（指令+axtree / 图像 / 格式尾 / response）+ 两条掩码行（labels / obs_mask）
- 层内五步 4a→4e 与张量形状（$H[B,N,2048]$、$M[B,16,512]$、$Z_t[B,4,512]$）
- 损失作用位置（$\mathcal{L}_{act}$ 只算 response 段；$\mathcal{L}_{obs}$/$\mathcal{L}_{nll}$ 每层均值）

**三个常见理解纠偏**（观众常问）：
1. $a_{t-1}$ **不在序列里** —— 独立 side input（tokenize ≤32 → mean-pool + Linear → a_emb），只喂 Predict
2. 记忆 token **从不拼进主序列** —— $M, P$ 是 per-layer side 张量，经 Cross-Attn 交互
3. obs 池化覆盖**整个 prompt 段**（指令 + axtree + 图像），不只图像；response 段被排除（记忆不许偷看答案）

**图来源**
- draw.io：`docs/compact_v2/layer_l_token_flow.drawio`

---

## 附：绘图资源索引

| 资源 | 路径 |
|---|---|
| 整体数据流（页 6） | `docs/compact_v2/method_overall_dataflow.drawio` |
| 三页 zoom 详图（页 7-9：Predict / Observe+Correct / Inject） | `docs/compact_v2/method_modules_zoom.drawio`（一个文件三页，底部切页签） |
| token 级流程图（页 11 附录） | `docs/compact_v2/layer_l_token_flow.drawio` |
| 三形态分类图（页 3） | `docs/compact_v2/slide2_memory_taxonomy.drawio` |
| latent 路线对比图（页 4） | `docs/compact_v2/slide3_latent_memory_compare.drawio` |
| KF 动力学仿真（页 8/10） | `docs/compact_v2/fig2_kf_dynamics_v2.png` |
| 单层 pipeline 现成图 | `docs/compact_v2/fig1_layer_v2.png` |
| Session loop 现成图（页 6） | `docs/compact_v2/fig3_session_loop_v2.png` |
| 方法细节 reference | `docs/COMPACT_V2_METHOD.md` |

注：draw.io 中公式用 LaTeX 书写，导入后开启 Extras → Mathematical Typesetting 渲染。旧文件 `slide1_overall_structure.drawio` 已被 `method_overall_dataflow.drawio` 取代，可删。
