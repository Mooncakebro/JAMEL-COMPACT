# Browser Environment

The ScaleWoB browser environment is not bundled in this release. It is
downloaded into:

```text
env/browser_env/scalewob-env/
```

Each app is a static web app served locally and opened through BrowserGym. The
JAMEL reward uses V8 precise JavaScript coverage and converts coverage growth
into a binary novelty reward.

## Download Static Apps

Mirror all 96 apps:

```bash
bash shell/download_scalewob_env.sh --mode all
```

Mirror only the paper test split:

```bash
bash shell/download_scalewob_env.sh --mode test10
```

## Serve Static Apps

From the repository root:

```bash
bash shell/serve_scalewob.sh
```

Default URL:

```text
http://127.0.0.1:8000/weibo/index.html
```

Override host, port, or app root:

```bash
HOST=0.0.0.0 PORT=8010 \
SCALEWOB_ROOT=/path/to/scalewob-env \
bash shell/serve_scalewob.sh
```

## BrowserGym Integration

Main environment/eval code:

```text
jamel/utils/eval/eval_memory_aug_episode.py
jamel/cli/weak_model_sft_label.py
jamel/cli/baseline_eval.py
jamel/core/env/web/
```

The default ScaleWoB path is:

```text
JAMEL/env/browser_env/scalewob-env
```

You can override it with `SCALEWOB_ROOT` or `--scalewob-root`.

## Coverage Reward

Coverage collection and reward calculation live in:

```text
jamel/core/env/web/coverage.py
jamel/core/reward/web/utils.py
jamel/coverage_artifact.py
```

Per-step outputs contain trajectory rows, screenshots, coverage artifacts, and
summary JSON files. A step receives reward `1` when cumulative JavaScript
coverage increases relative to the session baseline; otherwise reward is `0`.
