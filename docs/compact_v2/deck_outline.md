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
- 我们的三个判据（**按记忆生命周期排序：形态 → 更新 → 使用**，与模块①②③一一对应）：
  1. **Faithful**：固定容量下忠实压缩历史（→ 形态）
  2. **Predictive**：给定动作 $a_t$，能**预期**下一帧观测 $O_{t+1}$（→ 更新）
  3. **Useful**：被决策真正消费（→ 使用）
- 桥接方法（判据如何推出性质）：对每个判据问一句 **"什么情况算违反它？"** —— 违反的方式，就是它强加的工程约束：

| 判据 | "违反它长什么样" ⟹ 推导 | 兑现的性质（页 2 评分表） | 落成的模块 |
|---|---|---|---|
| **Faithful** | bank 涨爆后截断（JAMEL）、或被剪枝删除（SnapKV）= 不忠实 ⟹ 形态本身必须有界 | 【固定容量】 | ① Compact State Formation |
| **Predictive** | 预期必然有时出错；**错了却不知道**比没有预期更危险 ⟹ "信记忆还是信观测"必须可计算 ⟹ $K=\hat P/(\hat P+R)$，且只写预期之差 | 【$O(1)$ 更新】【按需查询】【按需遗忘】 | ② Anticipate–Correct Update |
| **Useful** | 记忆不被读（v1 推理旁路）或读了毁掉基座 = 没用 ⟹ 注入必须可门控、零初始化 | 训练-推理一致 | ③ Zero-Born Steering |

- Predictive 内部为何再分 Predict / Correct 两步：先显式预期（动作作控制量），再**只按"意外"校正** —— 解耦是为了让校正只响应预测之外的新息 $\Delta M$，这正是 $O(1)$ 残差更新的来源。
- 为什么监督落在预测步：$\mathcal{L}_{act}$ 只担保 Useful（还会学成投机特征）；**Predictive 判据它管不到** ⟹ $\mathcal{L}_{obs}$ 强迫记忆学会预期；置信度的"声称"要兑现 ⟹ $\mathcal{L}_{nll}$ 校准。
- 点睛句：**Faithful 决定记忆长什么样，Predictive 决定记忆怎么动，Useful 决定记忆怎么用——三个判据各管一段生命周期，三个模块各兑现一个判据。**

**金句（页底居中）**
> 我们不为"记住过去"而设计记忆，而为"预期未来"而设计记忆 —— **记忆即世界模型**。

**转场句**
> "这套哲学落到架构上，就是一张数据流图——三个模块已经按'形态、更新、使用'在图上各就各位。"

**图来源**
- AI prompt（三柱图）：
  > A conceptual three-pillar illustration for "what makes a good memory". Three stone pillars standing on a base shaped like a brain: pillar 1 topped with a mirror reflecting a film strip labeled "Faithful (compresses history)", pillar 2 topped with a crystal ball showing a future screen labeled "Predictive (anticipates the next observation)", pillar 3 topped with a dartboard icon labeled "Useful (serves decisions)". The middle pillar glows brightest. Flat vector, white background, blue/orange palette, minimal, academic keynote style.

---

## 页 6：整体结构 — 数据流与三个模块的位置

**标题**
Overview — One Pass Through COMPACT

**文案**
- 每步输入：指令 + 当前观测（+上一步动作 $a_{t-1}$）→ 上下文恒定 $O(1)$
- 每层解码器外挂侧边记忆，三模块按判据各就各位：
  - **① Compact State Formation** — 记忆长什么样（兑现 **Faithful** →【固定容量】）
  - **② Anticipate–Correct Update** — 记忆怎么动（兑现 **Predictive** →【$O(1)$ 更新】【按需查询】【按需遗忘】）
  - **③ Zero-Born Steering** — 记忆怎么用（兑现 **Useful** → 训练-推理一致）
- $(M_t, P_t, e_t)$ 跨步携带；动作由记忆条件化生成（KV-cache 预填充）

**转场句**
> "接下来三页按判据顺序各放大一个模块——页顶灰字标明它兑现哪个判据、哪几条性质。"

**图来源**
- draw.io：`docs/compact_v2/method_overall_dataflow.drawio`（整体数据流，①②③ zoom 徽章已标注；含跨步携带回路、动作反馈回路、总损失条）
- 可参考：`docs/compact_v2/fig3_session_loop_v2.png`（session loop 现成图）

---

## 页 7：模块 ① Compact State Formation（紧致记忆形态）

**页顶回扣行（灰字）**：兑现判据 **Faithful** → 性质【固定容量】（记忆"长什么样"）

**挑战**（引用页 4 的具体缺陷）
append 式（JAMEL）bank 线性增长、溢出即截断；剪枝式（SnapKV）直接删除历史 —— 形态本身无界、或可被外力破坏，"忠实"无从谈起

**我们的方法**
把记忆做成一个**天生有界、不受外力裁剪**的压缩状态：
- 每层外挂固定 **16 槽 × 512 维**内容轨 $M_l$，bank 大小与步数无关
- 可学习初始化 $\sim\mathcal{N}(0, 0.02^2)$：从零学起但不随机扰动主干
- 逐层挂载 + 层级化分工：浅层记 UI 细节，深层记任务逻辑
- 参数开销 ≈12%（2B）/ ≈6%（8B）

**一句话**
形态本身有界，忠实才有可能。

**转场句**
> "形态只回答了'装得下'。静止的记忆还不是世界模型——它必须每步向前'动'，而且要会预期。"

**图来源**
- draw.io：`docs/compact_v2/method_modules_zoom.drawio` 第 1 页（locator 红框 + 内容轨与层级化分工；$P$ 轨不在此页，留给模块②按需引入）
- AI prompt（split-view zoom 图，备选）：
  > Split-view academic diagram. Left: a small simplified overview of a transformer stack with a side memory box highlighted by a red zoom circle. Right: magnified detail of the side memory: a fixed rack with exactly 16 glowing slots labeled "fixed 16-slot content track M", with a "bounded by design" badge; shallow-layer slots hold small UI-element icons, deep-layer slots hold goal/chess icons. Flat vector, white background, blue/green palette, minimal text.

---

## 页 8：模块 ② Anticipate–Correct Update（预期-校正更新）

**页顶回扣行（灰字）**：兑现判据 **Predictive** → 性质【$O(1)$ 更新】【按需查询】【按需遗忘】（记忆"怎么动"）

**挑战**
好记忆必须能预期未来；但预期**必然有时出错** —— 一个会预期却没有置信度的系统，比没有预期更危险（错误记忆污染决策）。"这一步该不该改记忆、改多少"必须可计算

**我们的方法**
先预期，再按置信度校正；只对"意外"更新：
- **引入不确定度轨 $P$**（按需出场）：逐槽方差，$P_0 = 0.5$，全程可学习 —— 预期会错，所以需要一杆秤
- **Predict**（动作作控制量）：$\hat M = \mathrm{GRU}(W_a a_\downarrow,\; \gamma \odot M + \beta)$，$\hat P = \lambda \odot P + \mathrm{softplus}(W_q a_\downarrow) + \gamma_e \bar e$
- **Observe**：掩码 AttnPool（$k{=}4$ queries）$\to Z_t$
- **Correct**：$\Delta M = \mathrm{CrossAttn}(\hat M, Z_t)$，$K = \dfrac{\hat P}{\hat P + R}$，$M \leftarrow \hat M + K \odot \Delta M$，$P \leftarrow (1-K)\hat P$
- **监督落在预测步**（兑现哲学页的承诺）：$\mathcal{L}_{obs}$ 让记忆学会预期；$\mathcal{L}_{nll}$ 校准 $R$；surprise $e_t$ 闭环反馈下一步 $\hat P$

**一句话**
界面突变 → $e$ 飙升 → $\hat P$ 膨胀 → $K$ 跳到 0.93 → 新观测瞬间重写记忆。

**转场句**
> "记忆会动了，最后一步是让它回到主干、参与决策——而且不能碰坏预训练模型。"

**图来源**
- draw.io：`docs/compact_v2/method_modules_zoom.drawio` 第 2 页（locator 红圈标预测-校正循环；含 $P$ 轨引入 + 四步公式流）
- 现成仿真图：`docs/compact_v2/fig2_kf_dynamics_v2.png`（界面突变时 $K$ 跳至 0.93 再回落 —— 优先用真图）
- AI prompt（split-view zoom 图，备选）：
  > Split-view academic diagram. Left: small overview of the architecture with the memory-update cycle highlighted by a red zoom circle. Right: magnified circular flow with four stations — "Predict" (clock icon rolling a memory ball forward with a mouse-click arrow), "Observe" (eye icon reading the current screenshot), "Correct" (two arrows merging: prediction vs observation, a delta symbol between them, a dial labeled "gain K" controlled by a small scale icon labeled "uncertainty P"), and a "surprise e" spark icon feeding back to Predict. Flat vector, white background, blue/orange palette, minimal text.

---

## 页 9：模块 ③ Zero-Born Steering（零起点记忆导引）

**页顶回扣行（灰字）**：兑现判据 **Useful** → 训练-推理一致（记忆"怎么用"；统一模型，呼应标题页）

**挑战**
外挂记忆通路若随机初始化，第 0 步就扰动预训练模型；v1 甚至在推理时把记忆整体丢弃

**我们的方法**
记忆以 steering vector 方式被"按需查询"，从零出生、无损起步：
- 注入即查询：$H = h + w_l \tanh(g_l)\, W_\uparrow\, \mathrm{CrossAttn}(Q{=}h_\downarrow,\, K,V{=}M)$
- **零初始化** $W_\uparrow{=}0$ + 小门控 $g_l{=}0.1$ ⟹ 第 0 步 ≡ 预训练 LLM，梯度立即可训
- 层级权重 $w_l$：浅层 0.8（细节）→ 深层 0.3（保留推理）
- 推理闭环：记忆增强前向 → KV-cache 预填充 → 增量解码，动作**真正以记忆为条件**

**一句话**
记忆不是硬拼接的上下文，而是可查询、可门控、从零长起的导引信号。

**转场句**
> "三模块合起来，Faithful、Predictive、Useful 各就各位，评分表全绿。但全绿不是嘴上说的——看实验。"

**图来源**
- draw.io：`docs/compact_v2/method_modules_zoom.drawio` 第 3 页（locator 红箭头标注入通路 + 公式与推理闭环）
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

## 附：绘图资源索引

| 资源 | 路径 |
|---|---|
| 整体数据流（页 6） | `docs/compact_v2/method_overall_dataflow.drawio` |
| 三页 zoom 详图（页 7-9） | `docs/compact_v2/method_modules_zoom.drawio`（一个文件三页，底部切页签） |
| 三形态分类图（页 3） | `docs/compact_v2/slide2_memory_taxonomy.drawio` |
| latent 路线对比图（页 4） | `docs/compact_v2/slide3_latent_memory_compare.drawio` |
| KF 动力学仿真（页 8/10） | `docs/compact_v2/fig2_kf_dynamics_v2.png` |
| 单层 pipeline 现成图 | `docs/compact_v2/fig1_layer_v2.png` |
| Session loop 现成图（页 6） | `docs/compact_v2/fig3_session_loop_v2.png` |
| 方法细节 reference | `docs/COMPACT_V2_METHOD.md` |

注：draw.io 中公式用 LaTeX 书写，导入后开启 Extras → Mathematical Typesetting 渲染。旧文件 `slide1_overall_structure.drawio` 已被 `method_overall_dataflow.drawio` 取代，可删。
