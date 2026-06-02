# ScaleWoB reward function demo

这个仓库里的站点环境，常见约定是：

- 任务列表接口：`window.getTasks()`
- 奖励/评估接口：`window.evaluateTask(params)`
- 上层 bridge 的 `evaluate` 命令会继续转调 `window.evaluateTask(...)`

你提到的 `window.evaluateTasks` 在这个仓库里没有直接定义；如果你外层运行器额外包了一层复数接口，它本质上也应该是在转调单数版。`scalewob-eval-demo.js` 已经同时兼容两种名字：

- 优先调用 `window.evaluateTasks`
- 若不存在，则回退到 `window.evaluateTask`

当前文件已经移动到 `JAMEL` 的环境脚本目录下：

- `jamel/core/env/web/scalewob/scalewob-eval-demo.js`
- `jamel/core/env/web/scalewob/xiaoheihe-task5-before.js`

## 返回值约定

返回值没有完全统一，但大体长这样：

```js
{ success: true, score: 100 }
{ success: false, score: 0, message: "..." }
{ success: false, msg: "..." }
```

规律基本是：

- `score` 最大值通常是 `100`
- 多数任务只有二值分数，也就是 `0` 或 `100`
- 少数任务支持更细粒度的部分分，例如 `50`

仓库里能直接看到两类例子：

- `weibo/js/evaluation.js` 几乎都是最终结果式的 `0/100`
- `slack/js/task_engine.js` 有明显的部分分，例如资料改对一半时给 `50`

## 脚本做了什么

`scalewob-eval-demo.js` 会：

1. 在仓库根目录启动一个本地静态服务
2. 连接一个开启了 CDP 远程调试端口的浏览器
3. 打开目标环境页面
4. 在页面里执行 JS，读取 `window.getTasks()`
5. 可选地先执行一段 `before` JS 改状态
6. 再调用奖励函数并打印标准化结果

输出 JSON 里有两层结果：

- `snapshot.evaluationRaw`
  页面真实返回值，保留原样
- `snapshot.evaluation`
  归一化后的通用结构，方便外部脚本消费

## 最简单的用法

先自己起一个 Chrome 或 Chromium，并打开远程调试端口：

```bash
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/scalewob-cdp \
  about:blank
```

然后跑：

```bash
node jamel/core/env/web/scalewob/scalewob-eval-demo.js --env weibo
```

脚本会自己起本地静态服务，并通过 CDP 新开 `weibo/index.html`。这时不会执行评估，只会把页面任务列表和接口暴露情况打印出来。

如果你已经知道本机浏览器路径，也可以让脚本直接启动浏览器：

```bash
node jamel/core/env/web/scalewob/scalewob-eval-demo.js \
  --env weibo \
  --browser /path/to/google-chrome
```

## 直接调用奖励函数

下面这个例子用的是 `xiaoheihe` 的 task 5。这个任务要求：

- 开启深色模式
- 把签名改成 `Hardcore Gamer`

先用 `before` 脚本直接改页面状态，再调用 evaluator：

```bash
node jamel/core/env/web/scalewob/scalewob-eval-demo.js \
  --env xiaoheihe \
  --task-id 5 \
  --params '{"bio":"Hardcore Gamer"}' \
  --before-file jamel/core/env/web/scalewob/xiaoheihe-task5-before.js
```

如果状态写入成功，返回通常会是：

```js
{
  success: true,
  score: 100
}
```

## 看部分分的例子

`slack/js/task_engine.js` 的 task 1 会检查：

- Job Title 是否是 `Senior Architect`
- Status Text 是否是 `Focusing`

如果只改对一半，它会返回 `50`。例如只改 title：

```bash
node jamel/core/env/web/scalewob/scalewob-eval-demo.js \
  --env slack \
  --task-id 1 \
  --before '(() => { window.AppStore.state.currentUser.title = "Senior Architect"; return window.AppStore.state.currentUser; })()'
```

这个环境里，评估逻辑是：

```js
score = bothCorrect ? 100 : oneCorrect ? 50 : 0
```

## 如果你要接你自己的 `window.evaluateTasks`

浏览器内的最小调用形式就是：

```js
const fn = window.evaluateTasks || window.evaluateTask;
const result = fn({ taskId: 1 });
console.log(result);
```

如果你只是想在 DevTools Console 里验证，这一段就够了。
