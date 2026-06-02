#!/usr/bin/env python3
"""
BrowserGym demo for local ScaleWoB environments.

This script defines a custom BrowserGym task that:
1. serves this repository over a local HTTP server
2. opens one of the local env pages
3. optionally runs some JS in the page to modify app state
4. calls window.getTasks() and window.evaluateTask()/window.evaluateTasks()
5. maps the ScaleWoB score [0, 100] to BrowserGym reward [0.0, 1.0]

The code is meant as a reference implementation. It does not need to be run
in this environment.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask


DEFAULT_SCALEWOB_ENV_ROOT = Path(
    os.environ.get(
        "SCALEWOB_ROOT",
        str(
            Path(__file__).resolve().parents[5]
            / "env"
            / "browser_env"
            / "scalewob-env"
        ),
    )
)


class LocalStaticServer:
    def __init__(self, root_dir: Path, host: str = "127.0.0.1", port: int = 8787):
        self.root_dir = Path(root_dir)
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        handler = partial(SimpleHTTPRequestHandler, directory=str(self.root_dir))
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="scalewob-static-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


class ScaleWoBRewardTask(AbstractBrowserTask):
    @classmethod
    def get_task_id(cls):
        return "scalewob.local-reward-demo"

    def __init__(
        self,
        seed: int,
        repo_root: str,
        env_id: str,
        port: int = 8787,
        task_id: int | None = None,
        eval_params: dict[str, Any] | None = None,
        pre_eval_script: str | None = None,
    ) -> None:
        super().__init__(seed)
        self.repo_root = Path(repo_root)
        self.env_id = env_id
        self.port = port
        self.task_id = task_id
        self.eval_params = dict(eval_params or {})
        self.pre_eval_script = pre_eval_script
        self.server = LocalStaticServer(self.repo_root, port=self.port)
        self.viewport = {"width": 430, "height": 932}
        self.slow_mo = 0
        self.timeout = 10_000

    @property
    def start_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self.env_id}/index.html"

    def setup(self, page) -> tuple[str, dict]:
        env_index = self.repo_root / self.env_id / "index.html"
        if not env_index.exists():
            raise FileNotFoundError(f"Environment not found: {env_index}")

        self.server.start()
        page.goto(self.start_url, wait_until="domcontentloaded")
        page.wait_for_function(
            """
            () => {
              const hasStore = typeof window.AppStore !== "undefined";
              const hasTaskApi = typeof window.getTasks === "function";
              const hasEvaluator =
                typeof window.evaluateTask === "function" ||
                typeof window.evaluateTasks === "function";
              return hasStore && hasTaskApi && hasEvaluator;
            }
            """
        )

        goal = (
            "Inspect a local ScaleWoB environment and call its reward function. "
            "Read window.getTasks(), then call window.evaluateTasks(params) if it "
            "exists, otherwise fall back to window.evaluateTask(params)."
        )

        return goal, {
            "start_url": self.start_url,
            "env_id": self.env_id,
            "task_id": self.task_id,
            "eval_params": self.eval_params,
        }

    def validate(self, page, chat_messages) -> tuple[float, bool, str, dict]:
        params = dict(self.eval_params)
        if self.task_id is not None:
            params["taskId"] = self.task_id

        if self.pre_eval_script:
            before_result = page.evaluate(self.pre_eval_script)
        else:
            before_result = None

        snapshot = page.evaluate(
            """
            ({ params }) => {
              const evaluator =
                typeof window.evaluateTasks === "function"
                  ? window.evaluateTasks
                  : typeof window.evaluateTask === "function"
                    ? window.evaluateTask
                    : null;

              const tasks =
                typeof window.getTasks === "function" ? window.getTasks() : null;

              let raw = null;
              if (evaluator && params && Object.keys(params).length > 0) {
                raw = evaluator(params);
              }

              const normalized = raw
                ? {
                    success: Boolean(raw.success),
                    score:
                      typeof raw.score === "number"
                        ? raw.score
                        : raw.success
                          ? 100
                          : 0,
                    message:
                      typeof raw.message === "string"
                        ? raw.message
                        : typeof raw.msg === "string"
                          ? raw.msg
                          : null,
                  }
                : null;

              return {
                url: window.location.href,
                title: document.title,
                hasEvaluateTask: typeof window.evaluateTask === "function",
                hasEvaluateTasks: typeof window.evaluateTasks === "function",
                tasks,
                evaluationInput: params,
                evaluationRaw: raw,
                evaluationNormalized: normalized,
              };
            }
            """,
            {"params": params},
        )

        score = 0
        if snapshot["evaluationNormalized"] is not None:
            score = int(snapshot["evaluationNormalized"]["score"])

        reward = max(0.0, min(1.0, score / 100.0))
        done = True
        user_message = ""
        info = {
            "before_result": before_result,
            "snapshot": snapshot,
        }
        return reward, done, user_message, info

    def teardown(self) -> None:
        self.server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BrowserGym demo for calling ScaleWoB reward functions."
    )
    parser.add_argument("--env", required=True, help="Environment id, e.g. weibo")
    parser.add_argument("--task-id", type=int, default=None, help="Task id to evaluate")
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON object passed to the evaluator, excluding taskId unless you want to set it yourself",
    )
    parser.add_argument(
        "--pre-eval-file",
        default=None,
        help="Path to a JS file that will be executed in the page before validation",
    )
    parser.add_argument("--port", type=int, default=8787, help="Local HTTP port")
    parser.add_argument("--seed", type=int, default=0, help="Task seed")
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_SCALEWOB_ENV_ROOT),
        help="Path to the local scalewob-env repository root",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run BrowserGym in headless mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_params = json.loads(args.params)
    if not isinstance(eval_params, dict):
      raise ValueError("--params must decode to a JSON object")

    pre_eval_script = None
    if args.pre_eval_file:
        pre_eval_script = Path(args.pre_eval_file).read_text(encoding="utf-8")

    env = BrowserEnv(
        task_entrypoint=ScaleWoBRewardTask,
        task_kwargs={
            "repo_root": str(Path(args.repo_root).resolve()),
            "env_id": args.env,
            "port": args.port,
            "task_id": args.task_id,
            "eval_params": eval_params,
            "pre_eval_script": pre_eval_script,
        },
        headless=args.headless,
    )

    try:
        obs, info = env.reset(seed=args.seed)
        reward, done, message, task_info = env.task.validate(env.page, env.chat.messages)
        result = {
            "goal": obs["goal"],
            "task_info": info.get("task_info", {}),
            "reward": reward,
            "done": done,
            "message": message,
            "validation": task_info,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
