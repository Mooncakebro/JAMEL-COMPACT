# BrowserGym 版 ScaleWoB demo

这个版本不是直接连 CDP，而是按 BrowserGym 的官方扩展方式写：

- 自定义一个 `AbstractBrowserTask`
- 用 `BrowserEnv(task_entrypoint=...)` 启动
- 在 `setup()` 里打开本地网页
- 在 `validate()` 里直接调用页面的奖励函数

主脚本在：

- `jamel/core/env/web/scalewob/browsergym_scalewob_demo.py`

## 依赖

按 BrowserGym 官方 README，至少需要：

```bash
uv sync --locked --extra dev --extra train
uv run playwright install chromium
```

BrowserGym README 里给的最小入口是 `gym.make("browsergym/openended", ...)`；WorkArena 官方示例则直接使用：

```python
from browsergym.core.env import BrowserEnv
```

并通过：

```python
reward, stop, message, info = env.task.validate(env.page, cheat_messages)
```

来做校验。这个 demo 采用的就是后一种形式，因为你这里的重点不是 agent loop，而是“打开页面后调用任务与奖励函数”。

## 核心思路

`ScaleWoBRewardTask` 做了几件事：

1. 起一个本地静态服务，把当前仓库当成站点根目录
2. 打开 `http://127.0.0.1:<port>/<env>/index.html`
3. 等待页面暴露：
   - `window.AppStore`
   - `window.getTasks`
   - `window.evaluateTask` 或 `window.evaluateTasks`
4. 可选执行一段预处理 JS，修改页面状态
5. 在 `validate()` 里调用奖励函数
6. 把 ScaleWoB 的 `score` 从 `[0, 100]` 映射成 BrowserGym 的 `reward` `[0.0, 1.0]`

## 最小示例

只看任务列表，不传评估参数：

```bash
uv run python jamel/core/env/web/scalewob/browsergym_scalewob_demo.py --env weibo
```

这会打开 `weibo` 页面，抓到 `window.getTasks()`，但不会实际算分。

如果你的 `scalewob-env` 不在默认位置，也可以显式传：

```bash
uv run python jamel/core/env/web/scalewob/browsergym_scalewob_demo.py \
  --repo-root /path/to/scalewob-env \
  --env weibo
```

## 调奖励函数的示例

还是用前面那个 `xiaoheihe` task 5：

```bash
uv run python jamel/core/env/web/scalewob/browsergym_scalewob_demo.py \
  --env xiaoheihe \
  --task-id 5 \
  --params '{"bio":"Hardcore Gamer"}' \
  --pre-eval-file jamel/core/env/web/scalewob/xiaoheihe-task5-before.js
```

这里会：

- 先执行 `jamel/core/env/web/scalewob/xiaoheihe-task5-before.js`
- 再调用页面内的：

```js
window.evaluateTasks?.(params) ?? window.evaluateTask(params)
```

最后输出：

- `reward`
  BrowserGym 风格分数，范围 `[0.0, 1.0]`
- `validation.snapshot.evaluationRaw`
  页面原始返回值
- `validation.snapshot.evaluationNormalized`
  归一化后的 `{ success, score, message }`

## 奖励函数怎么映射

这个仓库里的页面通常返回：

```js
{ success: true, score: 100 }
{ success: false, score: 0, message: "..." }
{ success: false, msg: "..." }
```

本 demo 的映射规则是：

```python
reward = normalized_score / 100.0
```

所以：

- `100 -> 1.0`
- `50 -> 0.5`
- `0 -> 0.0`

这和 BrowserGym 的 `validate()` 里返回浮点 reward 的风格是一致的。
