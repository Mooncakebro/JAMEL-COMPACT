"""Resilient BrowserGym lifecycle and shared evaluation loop."""
from __future__ import annotations

import gc
import io
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from jamel.core.env.web.coverage import save_coverage, start_coverage
from jamel.core.reward.web.utils import compute_monocart_coverage_reward_details


class RecoveringBrowserSession:
    """Create, reset, and replace a BrowserGym environment after failures."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        reset_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._env_factory = env_factory
        self._reset_retries = max(1, int(reset_retries))
        self._retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._env = None

    @property
    def env(self):
        if self._env is None:
            self._env = self._env_factory()
        return self._env

    @property
    def page(self):
        return self.env.unwrapped.page

    def reset(self, seed: int | None = None):
        last_error: Exception | None = None
        for attempt in range(1, self._reset_retries + 1):
            try:
                return self.env.reset(seed=seed)
            except Exception as error:
                last_error = error
                print(
                    f"  [env] Reset attempt {attempt}/{self._reset_retries} failed: "
                    f"{type(error).__name__}: {error}"
                )
                self.close()
                if attempt < self._reset_retries and self._retry_delay_seconds:
                    time.sleep(self._retry_delay_seconds * attempt)
        raise RuntimeError(
            f"Browser reset failed after {self._reset_retries} attempts"
        ) from last_error

    def step(self, action: str):
        return self.env.step(action)

    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            try:
                env.close()
            except Exception as error:
                print(f"  [env] Close failed: {type(error).__name__}: {error}")
        gc.collect()


def _obs_to_bytes(obs: dict | None, key: str) -> bytes | None:
    if obs is None:
        return None
    image_array = obs.get(key)
    if image_array is None:
        return None
    buffer = io.BytesIO()
    Image.fromarray(image_array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _save_trajectory(trajectory: list[dict], path: Path) -> Path:
    fallback_path = path.with_suffix(".jsonl")
    with fallback_path.open("w", encoding="utf-8") as output_file:
        for row in trajectory:
            output_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    converter = """
import json
import sys
import pandas as pd

rows = []
with open(sys.argv[1], "r", encoding="utf-8") as input_file:
    for line in input_file:
        if line.strip():
            rows.append(json.loads(line))
pd.DataFrame(rows).to_parquet(sys.argv[2])
"""
    conversion = subprocess.run(
        [sys.executable, "-c", converter, str(fallback_path), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if conversion.returncode == 0 and path.is_file():
        fallback_path.unlink(missing_ok=True)
        return path

    error = conversion.stderr.strip().splitlines()
    detail = error[-1] if error else f"exit code {conversion.returncode}"
    print(f"  [save] Parquet conversion failed ({detail}); saved JSONL instead.")
    return fallback_path


def _write_summary(
    output_path: Path,
    checkpoint: str,
    model_type: str,
    apps: list[str],
    num_sessions: int,
    max_steps: int,
    results: list[dict],
) -> Path:
    summary_path = output_path / "eval_summary.json"
    summary = {
        "checkpoint": checkpoint,
        "model_type": model_type,
        "apps": apps,
        "num_sessions": num_sessions,
        "max_steps": max_steps,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    temporary_path.replace(summary_path)
    return summary_path


def run_browser_evaluation(
    *,
    agent: Any,
    checkpoint: str,
    apps: list[str],
    scalewob_root: str,
    max_steps: int,
    num_sessions: int,
    output_dir: str,
    headless: bool,
    port: int,
    seed: int,
    model_type: str,
    summary_title: str,
    browser_timeout_ms: int = 30_000,
    reset_retries: int = 3,
    save_screenshots: bool = False,
    on_step_decision: Callable[[str, int], None] | None = None,
) -> list[dict]:
    """Evaluate an agent while isolating failures to one session."""
    from browsergym.core.env import BrowserEnv
    from jamel.utils.eval.eval_memory_aug_episode import ScaleWoBTask

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for app in apps:
        print(f"\n{'=' * 60}")
        print(f"App: {app}")
        print(f"{'=' * 60}")
        target_url = f"http://127.0.0.1:{port}/{app}/index.html"
        start_url = target_url

        for session_idx in range(num_sessions):
            print(f"\n  Session {session_idx + 1}/{num_sessions}")
            agent.reset_session()

            session_dir = output_path / app / f"session{session_idx}"
            coverage_dir = session_dir / "coverage"
            session_dir.mkdir(parents=True, exist_ok=True)
            coverage_dir.mkdir(parents=True, exist_ok=True)

            def _make_env():
                return BrowserEnv(
                    task_entrypoint=ScaleWoBTask,
                    task_kwargs={
                        "env_id": app,
                        "port": port,
                        "viewport_width": 1280,
                        "viewport_height": 720,
                        "scalewob_root": scalewob_root,
                        "timeout": browser_timeout_ms,
                        "navigation_retries": 2,
                    },
                    headless=headless,
                    viewport={"width": 1280, "height": 720},
                )

            browser = RecoveringBrowserSession(
                _make_env,
                reset_retries=reset_retries,
            )
            cdp_session = None

            def _start_coverage_on_page() -> None:
                nonlocal cdp_session
                cdp_session = None
                try:
                    cdp_session = start_coverage(browser.page)
                except Exception as error:
                    print(f"  [coverage] Failed to start coverage: {error}")

            def _save_step_coverage(global_step: int) -> Path | None:
                if cdp_session is None:
                    return None
                coverage_path = coverage_dir / f"step_{global_step:04d}.json"
                try:
                    save_coverage(cdp_session, str(coverage_path))
                    return coverage_path if coverage_path.is_file() else None
                except Exception as error:
                    print(f"  [coverage] Save failed at step {global_step}: {error}")
                    return None

            trajectory: list[dict] = []
            cumulative_reward = 0.0
            last_coverage_score = 0
            history_coverage_paths: list[Path] = []
            episode_idx = 0
            global_step = 0
            session_started = False
            session_error: str | None = None

            try:
                obs, _ = browser.reset(seed=seed)
                session_started = True
                _start_coverage_on_page()
                print(f"  [env] Reset done. Start URL: {target_url}\n")

                for step_idx_in_session in range(max_steps):
                    print(
                        f"  ── Step {step_idx_in_session + 1}/{max_steps} "
                        f"(ep {episode_idx}) " + "─" * 30
                    )
                    timestamp = datetime.now().isoformat()
                    result = agent.decide_action(obs, target_url, start_url, max_steps)
                    action = result["action"] or "noop()"
                    think = result["think"]
                    raw_response = result["raw_response"]
                    prompt = result["prompt"]

                    if on_step_decision is not None:
                        on_step_decision(app, step_idx_in_session)

                    print(f"    think:  {think[:120]}{'...' if len(think) > 120 else ''}")
                    print(f"    action: {action}")
                    if not result["action"]:
                        print("    [WARN] Empty action, inserting noop()")

                    if save_screenshots:
                        before_bytes = _obs_to_bytes(obs, "screenshot")
                        if before_bytes:
                            screenshot_path = session_dir / f"step_{step_idx_in_session + 1:03d}_before.png"
                            screenshot_path.write_bytes(before_bytes)

                    if action.strip() == "reset()":
                        print("    [reset] Browser reset; agent session state is retained.")
                        coverage_path = _save_step_coverage(global_step)
                        trajectory.append({
                            "step": global_step,
                            "session_idx": session_idx,
                            "episode_idx": episode_idx,
                            "action": action,
                            "think": think,
                            "raw_response": raw_response,
                            "prompt": prompt,
                            "reward": 0.0,
                            "delta_score": 0,
                            "previous_score": last_coverage_score,
                            "current_score": last_coverage_score,
                            "cumulative_reward": cumulative_reward,
                            "coverage_path": str(coverage_path) if coverage_path else None,
                            "target_url": target_url,
                            "start_url": start_url,
                            "timestamp": timestamp,
                            "episode_boundary": True,
                        })
                        if coverage_path is not None:
                            history_coverage_paths.append(coverage_path)
                        global_step += 1
                        episode_idx += 1
                        obs, _ = browser.reset(seed=seed)
                        _start_coverage_on_page()
                        print(f"    [reset] Done. Back at: {target_url}")
                        continue

                    try:
                        next_obs, _raw_reward, terminated, truncated, _ = browser.step(action)
                    except Exception as error:
                        print(f"    [ERROR] env.step failed: {type(error).__name__}: {error}")
                        coverage_path = _save_step_coverage(global_step)
                        trajectory.append({
                            "step": global_step,
                            "session_idx": session_idx,
                            "episode_idx": episode_idx,
                            "action": action,
                            "think": think,
                            "raw_response": raw_response,
                            "prompt": prompt,
                            "reward": 0.0,
                            "delta_score": 0,
                            "previous_score": last_coverage_score,
                            "current_score": last_coverage_score,
                            "cumulative_reward": cumulative_reward,
                            "coverage_path": str(coverage_path) if coverage_path else None,
                            "target_url": target_url,
                            "start_url": start_url,
                            "timestamp": timestamp,
                            "error": f"{type(error).__name__}: {error}",
                        })
                        if coverage_path is not None:
                            history_coverage_paths.append(coverage_path)
                        global_step += 1
                        episode_idx += 1
                        browser.close()
                        obs, _ = browser.reset(seed=seed)
                        _start_coverage_on_page()
                        print("    [env] Recovered with a fresh browser.")
                        continue

                    coverage_path = _save_step_coverage(global_step)
                    reward_details = compute_monocart_coverage_reward_details(
                        current_path=coverage_path,
                        baseline_paths=history_coverage_paths,
                        previous_score=last_coverage_score,
                    )
                    if coverage_path is not None:
                        history_coverage_paths.append(coverage_path)
                    reward = float(reward_details["reward"])
                    cumulative_reward += reward
                    last_coverage_score = int(
                        reward_details.get("current_score", last_coverage_score) or 0
                    )

                    if save_screenshots:
                        after_bytes = _obs_to_bytes(next_obs, "screenshot")
                        if after_bytes:
                            screenshot_path = session_dir / f"step_{step_idx_in_session + 1:03d}_after.png"
                            screenshot_path.write_bytes(after_bytes)

                    print(
                        f"    reward: {reward:+.4f}  "
                        f"Δcov={reward_details.get('delta_score', 0)}  "
                        f"(cumulative: {cumulative_reward:.4f})"
                    )
                    trajectory.append({
                        "step": global_step,
                        "session_idx": session_idx,
                        "episode_idx": episode_idx,
                        "action": action,
                        "think": think,
                        "raw_response": raw_response,
                        "prompt": prompt,
                        "reward": reward,
                        "delta_score": reward_details.get("delta_score", 0),
                        "previous_score": reward_details.get("previous_score", 0),
                        "current_score": reward_details.get("current_score", 0),
                        "cumulative_reward": cumulative_reward,
                        "coverage_path": str(coverage_path) if coverage_path else None,
                        "terminated": terminated,
                        "truncated": truncated,
                        "target_url": target_url,
                        "start_url": start_url,
                        "timestamp": timestamp,
                    })
                    obs = next_obs
                    global_step += 1
                    if terminated or truncated:
                        print("\n    [env] Episode finished early (terminated/truncated).")
                        break
            except Exception as error:
                session_error = f"{type(error).__name__}: {error}"
                print(f"  [session ERROR] {session_error}")
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                browser.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_path = _save_trajectory(
                trajectory,
                session_dir / f"trajectory_{app}_{timestamp}.parquet",
            )
            print(f"\n  [save] Trajectory: {trajectory_path}  ({len(trajectory)} rows)")

            session_result = {
                "app": app,
                "session_idx": session_idx,
                "status": "failed" if session_error else "completed",
                "error": session_error,
                "total_reward": cumulative_reward,
                "num_steps": len(trajectory),
                "num_episodes": episode_idx + (1 if session_started else 0),
                "actions": [row["action"] for row in trajectory],
                "rewards": [row["reward"] for row in trajectory],
                "coverage_delta_scores": [row.get("delta_score", 0) for row in trajectory],
                "trajectory_path": str(trajectory_path),
            }
            all_results.append(session_result)
            _write_summary(
                output_path,
                checkpoint,
                model_type,
                apps,
                num_sessions,
                max_steps,
                all_results,
            )
            print(
                f"  Session status: {session_result['status']}  "
                f"reward: {cumulative_reward:.4f}"
            )

    summary_path = _write_summary(
        output_path,
        checkpoint,
        model_type,
        apps,
        num_sessions,
        max_steps,
        all_results,
    )
    print(f"\n[eval] Summary saved to {summary_path}")
    print(f"\n{'=' * 60}")
    print(summary_title)
    print(f"{'=' * 60}")
    for result in all_results:
        print(
            f"  {result['app']:<20s} session {result['session_idx']}: "
            f"status={result['status']}  reward={result['total_reward']:.4f}  "
            f"steps={result['num_steps']}"
        )
    return all_results
