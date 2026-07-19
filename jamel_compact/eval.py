"""
Evaluation script for JAMEL-COMPACT.

Runs the trained model on ScaleWoB benchmark apps, collecting JS coverage
as the exploration reward.  Memory is maintained across episodes within a
session (not reset on reset()).

Usage:
    python -m jamel_compact.eval \
        --checkpoint outputs/compact_ckpt/final \
        --apps-mode test10 \
        --max-steps 50 \
        --num-sessions 3 \
        --eval-output outputs/compact_eval
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CompactConfig
from .model import JAMELCompactWrapper

# ── Coverage imports (same modules as original JAMEL eval) ──
from jamel.core.env.web.coverage import start_coverage, save_coverage
from jamel.core.reward.web.utils import compute_monocart_coverage_reward_details
from jamel.coverage_artifact import build_coverage_artifact_fields


# ── Action / think parsing ──

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
THINK_RE  = re.compile(r"<think>(.*?)</think>",  re.DOTALL)


def parse_action(response: str) -> tuple[str, str]:
    """Extract action and think from model response.

    Returns (action, think).
    """
    action_m = ACTION_RE.search(response)
    think_m = THINK_RE.search(response)
    action = action_m.group(1).strip().split("\n")[0] if action_m else ""
    think = think_m.group(1).strip() if think_m else ""
    return action, think


# ── Helper functions (same pattern as original JAMEL eval) ──

def _obs_to_bytes(obs: dict | None, key: str) -> bytes | None:
    """Encode screenshot numpy array → PNG bytes."""
    if obs is None:
        return None
    arr = obs.get(key)
    if arr is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _compute_coverage_details(
    current_path: Path | None,
    history_paths: list[Path],
    previous_score: int = 0,
) -> dict:
    """Compute reward through the shared Monocart cumulative coverage interface."""
    return compute_monocart_coverage_reward_details(
        current_path=current_path,
        baseline_paths=history_paths,
        previous_score=previous_score,
    )


# ── JAMEL-COMPACT Agent ──

class CompactAgent:
    """
    Wraps JAMELCompactWrapper for step-by-step session inference.
    Maintains per-layer memory states across episodes (not reset on reset()).
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        max_input_tokens: int = 8192,
        image_resize: tuple = (640, 360),
    ):
        print(f"[agent] Loading JAMEL-COMPACT from {checkpoint} ...")
        self.model = JAMELCompactWrapper.from_pretrained(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.tokenizer = self.model.tokenizer
        self.processor = self.model.processor
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.image_resize = image_resize

        # Session state
        self._memory_states = None
        self._confidence_states = None
        self._session_step_idx = 0
        self._last_action = "noop()"

    def reset_session(self):
        """Full reset — start of a new session."""
        self._memory_states, self._confidence_states = self.model.init_memory(
            batch_size=1, device=self.device,
        )
        self._session_step_idx = 0
        self._last_action = "noop()"

    def _build_prompt(self, obs_dict: dict, target_url: str, start_url: str,
                      max_steps: int) -> str:
        """Build the canonical web prompt (same as original JAMEL eval)."""
        from jamel.train.memory.web_prompt import build_web_prompt, extract_axtree_from_observation_str
        from jamel.core.env.web.axtree_utils import prune_axtree
        from jamel.core.env.web.observer import Observer
        import urllib.parse as _urlparse

        obs_text = Observer.get_observation(obs_dict)
        axtree_raw = extract_axtree_from_observation_str(obs_text)
        pruned_axtree = prune_axtree(axtree_raw, max_chars=8000)

        # Extract target_app from start_url path (same as original JAMEL eval)
        path_parts = _urlparse.urlparse(start_url).path.strip("/").split("/")
        target_app = path_parts[0] if path_parts else "app"
        open_urls = obs_dict.get("open_pages_urls", (start_url,))

        return build_web_prompt(
            step_idx=int(self._session_step_idx),
            target_app=target_app,
            start_url=start_url,
            open_urls=open_urls,
            pruned_axtree=pruned_axtree,
        )

    def _get_action_embedding(self) -> torch.Tensor:
        """Get action embedding from the last action string."""
        tokens = self.tokenizer.encode(
            self._last_action, add_special_tokens=False, max_length=32, truncation=True,
        )
        if not tokens:
            tokens = [self.tokenizer.pad_token_id or 0]
        token_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        embed_layer = self.model._get_input_embeddings()
        return embed_layer(token_ids).mean(dim=1)  # [1, d]

    @torch.inference_mode()
    def decide_action(self, obs_dict: dict, target_url: str, start_url: str,
                      max_steps: int) -> dict:
        """Decide the next action given the current observation.

        Returns dict with keys: action, think, raw_response, prompt.
        """
        prompt = self._build_prompt(obs_dict, target_url, start_url, max_steps)

        # Process screenshot (required — same as original JAMEL eval)
        screenshot_arr = obs_dict.get("screenshot")
        if screenshot_arr is None:
            raise RuntimeError("Web prompt requires a screenshot in obs_dict; got None.")
        image = Image.fromarray(screenshot_arr.astype(np.uint8))
        if image.size != self.image_resize:
            image = image.resize(self.image_resize, Image.BILINEAR)

        # Build model inputs — always use processor (same as original JAMEL eval)
        segments = prompt.split("<image>")
        content = []
        for idx, seg in enumerate(segments):
            if seg:
                content.append({"type": "text", "text": seg})
            if idx < len(segments) - 1:
                content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        inputs.pop("second_per_grid_ts", None)

        # Truncate to max_input_tokens
        orig_len = inputs["input_ids"].shape[1]
        if orig_len > self.max_input_tokens:
            for k in list(inputs.keys()):
                if hasattr(inputs[k], "shape") and inputs[k].shape[-1] == orig_len:
                    inputs[k] = inputs[k][..., -self.max_input_tokens:]

        # Get action embedding
        action_embed_input = self._get_action_embedding()

        # Generate with 180s kill timer (same as original JAMEL eval)
        _kill_timer = threading.Timer(180, lambda: os.kill(os.getpid(), signal.SIGKILL))
        _kill_timer.daemon = True
        _kill_timer.start()
        try:
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                action_embed_input=action_embed_input,
                memory_states=self._memory_states,
                confidence_states=self._confidence_states,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
        finally:
            _kill_timer.cancel()

        # Update memory states
        self._memory_states = outputs["new_memory"]
        self._confidence_states = outputs["new_confidence"]

        # Decode response — batch_decode matches original JAMEL eval exactly
        generated_ids = outputs["generated_ids"]
        raw_response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        action, think = parse_action(raw_response)

        # Update state
        self._last_action = action
        self._session_step_idx += 1

        return {
            "action": action,
            "think": think,
            "raw_response": raw_response,
            "prompt": prompt,
        }


# ── Evaluation loop ──

def run_eval(
    checkpoint: str,
    apps: list[str],
    scalewob_root: str,
    max_steps: int = 50,
    num_sessions: int = 3,
    output_dir: str = "outputs/compact_eval",
    device: str = "cuda",
    temperature: float = 0.8,
    top_p: float = 0.9,
    headless: bool = True,
    port: int = 8790,
    seed: int = 42,
):
    """
    Run JAMEL-COMPACT evaluation on ScaleWoB apps.

    For each app, runs `num_sessions` sessions of `max_steps` steps each.
    Memory is maintained across episodes within a session (not reset on reset()).

    Reward is computed from JS coverage (same as original JAMEL eval), NOT
    from env.step() reward (which is always 0 for ScaleWoBTask).
    """
    from browsergym.core.env import BrowserEnv
    from jamel.utils.eval.eval_memory_aug_episode import ScaleWoBTask

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize agent once (model stays loaded across all sessions)
    agent = CompactAgent(
        checkpoint=checkpoint,
        device=device,
        temperature=temperature,
        top_p=top_p,
    )

    all_results = []

    for app in apps:
        print(f"\n{'='*60}")
        print(f"App: {app}")
        print(f"{'='*60}")

        target_url = f"http://127.0.0.1:{port}/{app}/index.html"
        start_url = target_url

        for session_idx in range(num_sessions):
            print(f"\n  Session {session_idx + 1}/{num_sessions}")

            # Reset session memory
            agent.reset_session()

            # Create browser env with ScaleWoB task
            # ScaleWoBTask manages its own static server internally
            env = BrowserEnv(
                task_entrypoint=ScaleWoBTask,
                task_kwargs={
                    "env_id": app,
                    "port": port,
                    "viewport_width": 1280,
                    "viewport_height": 720,
                    "scalewob_root": scalewob_root,
                },
                headless=headless,
                viewport={"width": 1280, "height": 720},
            )

            # Per-session output directory and coverage directory
            session_dir = output_path / app / f"session{session_idx}"
            coverage_dir = session_dir / "coverage"
            session_dir.mkdir(parents=True, exist_ok=True)
            coverage_dir.mkdir(parents=True, exist_ok=True)

            # Coverage tracking state
            cdp_session = None

            def _start_coverage_on_page():
                nonlocal cdp_session
                try:
                    page = env.unwrapped.page
                    cdp_session = start_coverage(page)
                except Exception as e:
                    print(f"  [coverage] Failed to start coverage: {e}")
                    cdp_session = None

            def _save_step_coverage(global_step: int) -> Path | None:
                if cdp_session is None:
                    return None
                cov_path = coverage_dir / f"step_{global_step:04d}.json"
                try:
                    save_coverage(cdp_session, str(cov_path))
                    return cov_path
                except Exception as e:
                    print(f"  [coverage] Save failed at step {global_step}: {e}")
                    return None

            def _reset_with_timeout(seed=None):
                _kt = threading.Timer(600, lambda: os.kill(os.getpid(), signal.SIGKILL))
                _kt.daemon = True
                _kt.start()
                try:
                    result = env.reset(seed=seed)
                finally:
                    _kt.cancel()
                return result

            trajectory: list[dict] = []
            cumulative_reward = 0.0
            last_coverage_score = 0
            history_cov_paths: list[Path] = []
            episode_idx = 0
            global_step = 0

            try:
                obs, info = _reset_with_timeout(seed=seed)
                _start_coverage_on_page()
                print(f"  [env] Reset done. Start URL: {target_url}\n")

                for step_idx_in_session in range(max_steps):
                    print(f"  ── Step {step_idx_in_session + 1}/{max_steps} (ep {episode_idx}) " + "─" * 30)
                    ts = datetime.now().isoformat()

                    # ── inference ──
                    result = agent.decide_action(obs, target_url, start_url, max_steps)
                    action_str = result["action"]
                    think_str = result["think"]
                    raw_resp = result["raw_response"]
                    prompt_str = result["prompt"]

                    print(f"    think:  {think_str[:120]}{'...' if len(think_str) > 120 else ''}")
                    print(f"    action: {action_str}")

                    if not action_str:
                        print("    [WARN] Empty action, inserting noop()")
                        action_str = "noop()"

                    # ── reset() action: browser reset, memory retained ──
                    if action_str.strip() == "reset()":
                        print("    [reset] Browser reset — memory retained across episode boundary.")

                        cov_path = _save_step_coverage(global_step)
                        cov_artifact = build_coverage_artifact_fields(cov_path)

                        row = {
                            "step": global_step,
                            "session_idx": session_idx,
                            "episode_idx": episode_idx,
                            "action": action_str,
                            "think": think_str,
                            "raw_response": raw_resp,
                            "prompt": prompt_str,
                            "reward": 0.0,
                            "delta_score": 0,
                            "previous_score": last_coverage_score,
                            "current_score": last_coverage_score,
                            "cumulative_reward": cumulative_reward,
                            "coverage_path": str(cov_path) if cov_path else None,
                            "target_url": target_url,
                            "start_url": start_url,
                            "timestamp": ts,
                            "episode_boundary": True,
                        }
                        trajectory.append(row)

                        before_bytes = _obs_to_bytes(obs, "screenshot")
                        if before_bytes:
                            (session_dir / f"step_{step_idx_in_session+1:03d}_before.png").write_bytes(before_bytes)

                        global_step += 1
                        episode_idx += 1
                        obs, info = _reset_with_timeout(seed=seed)
                        _start_coverage_on_page()
                        print(f"    [reset] Done. Back at: {target_url}")
                        continue

                    # ── normal step ──
                    before_bytes = _obs_to_bytes(obs, "screenshot")
                    if before_bytes:
                        (session_dir / f"step_{step_idx_in_session+1:03d}_before.png").write_bytes(before_bytes)

                    try:
                        _kt = threading.Timer(600, lambda: os.kill(os.getpid(), signal.SIGKILL))
                        _kt.daemon = True
                        _kt.start()
                        try:
                            next_obs, _raw_reward, terminated, truncated, next_info = env.step(action_str)
                        finally:
                            _kt.cancel()
                    except Exception as e:
                        print(f"    [ERROR] env.step failed: {e}")
                        cov_path = _save_step_coverage(global_step)
                        cov_artifact = build_coverage_artifact_fields(cov_path)
                        row = {
                            "step": global_step,
                            "session_idx": session_idx,
                            "episode_idx": episode_idx,
                            "action": action_str,
                            "think": think_str,
                            "raw_response": raw_resp,
                            "prompt": prompt_str,
                            "reward": 0.0,
                            "delta_score": 0,
                            "previous_score": last_coverage_score,
                            "current_score": last_coverage_score,
                            "cumulative_reward": cumulative_reward,
                            "coverage_path": str(cov_path) if cov_path else None,
                            "target_url": target_url,
                            "start_url": start_url,
                            "timestamp": ts,
                            "error": str(e),
                        }
                        trajectory.append(row)
                        if cov_path and cov_path.exists():
                            history_cov_paths.append(cov_path)
                        global_step += 1
                        continue

                    # ── save coverage and compute reward ──
                    cov_path = _save_step_coverage(global_step)
                    cov_artifact = build_coverage_artifact_fields(cov_path)
                    reward_details = _compute_coverage_details(
                        cov_path,
                        history_cov_paths,
                        previous_score=last_coverage_score,
                    )
                    if cov_path and cov_path.exists():
                        history_cov_paths.append(cov_path)
                    reward = reward_details["reward"]
                    cumulative_reward += reward
                    last_coverage_score = int(reward_details.get("current_score", last_coverage_score) or 0)

                    # after-screenshot
                    after_bytes = _obs_to_bytes(next_obs, "screenshot")
                    if after_bytes:
                        (session_dir / f"step_{step_idx_in_session+1:03d}_after.png").write_bytes(after_bytes)

                    print(f"    reward: {reward:+.4f}  Δcov={reward_details.get('delta_score', 0)}"
                          f"  (cumulative: {cumulative_reward:.4f})")

                    row = {
                        "step": global_step,
                        "session_idx": session_idx,
                        "episode_idx": episode_idx,
                        "action": action_str,
                        "think": think_str,
                        "raw_response": raw_resp,
                        "prompt": prompt_str,
                        "reward": reward,
                        "delta_score": reward_details.get("delta_score", 0),
                        "previous_score": reward_details.get("previous_score", 0),
                        "current_score": reward_details.get("current_score", 0),
                        "cumulative_reward": cumulative_reward,
                        "coverage_path": str(cov_path) if cov_path else None,
                        "terminated": terminated,
                        "truncated": truncated,
                        "target_url": target_url,
                        "start_url": start_url,
                        "timestamp": ts,
                    }
                    trajectory.append(row)

                    obs, info = next_obs, next_info
                    global_step += 1

                    if terminated or truncated:
                        print("\n    [env] Episode finished early (terminated/truncated).")
                        break

            finally:
                env.close()

            # ── save trajectory ──
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            traj_df = pd.DataFrame(trajectory)
            traj_path = session_dir / f"trajectory_{app}_{ts_str}.parquet"
            traj_df.to_parquet(traj_path)
            print(f"\n  [save] Trajectory: {traj_path}  ({len(traj_df)} rows)")

            session_result = {
                "app": app,
                "session_idx": session_idx,
                "total_reward": cumulative_reward,
                "num_steps": len(trajectory),
                "num_episodes": episode_idx + 1,
                "actions": [t["action"] for t in trajectory],
                "rewards": [t["reward"] for t in trajectory],
                "coverage_delta_scores": [t.get("delta_score", 0) for t in trajectory],
                "trajectory_path": str(traj_path),
            }
            all_results.append(session_result)
            print(f"  Session reward: {cumulative_reward:.4f}")

    # ── save aggregate summary ──
    summary_path = output_path / "eval_summary.json"
    summary = {
        "checkpoint": checkpoint,
        "model_type": "jamel_compact",
        "apps": apps,
        "num_sessions": num_sessions,
        "max_steps": max_steps,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[eval] Summary saved to {summary_path}")

    # Print aggregate results
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY (JAMEL-COMPACT)")
    print(f"{'='*60}")
    for result in all_results:
        print(f"  {result['app']:<20s} session {result['session_idx']}: "
              f"reward={result['total_reward']:.4f}  steps={result['num_steps']}")


def main():
    # ── set CUDA_VISIBLE_DEVICES before importing torch ──
    _gpu_ids = os.environ.get("GPU_IDS", "")
    if _gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids

    parser = argparse.ArgumentParser(description="JAMEL-COMPACT Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    parser.add_argument("--apps", default="", help="Comma-separated app list (overrides --apps-mode)")
    parser.add_argument("--apps-mode", default="test10", choices=["test10", "train86", "all"])
    parser.add_argument("--scalewob-root", default="env/browser_env/scalewob-env")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-sessions", type=int, default=3)
    parser.add_argument("--eval-output", default="outputs/compact_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--port", type=int, default=8790, help="Port for ScaleWoB static server")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env.reset()")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU IDs (e.g. '0' or '0,1')")
    args = parser.parse_args()

    # Set GPU visibility
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Resolve apps
    if args.apps:
        apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        app_config = repo_root / "configs" / "benchmark_apps.json"
        if app_config.exists():
            import subprocess
            apps_str = subprocess.check_output([
                "python", str(repo_root / "scripts" / "print_app_split.py"),
                args.apps_mode, "--config", str(app_config),
            ]).decode().strip()
            apps = apps_str.split()
        else:
            apps = ["vipshop", "alibaba", "expedia", "taobao", "pinduoduo",
                    "dongchedi", "youku", "keep", "meituan", "temu"]

    print(f"[eval] Apps: {apps}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] Max steps: {args.max_steps}")
    print(f"[eval] Sessions: {args.num_sessions}")
    print(f"[eval] GPU IDs: {args.gpu_ids or os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    run_eval(
        checkpoint=args.checkpoint,
        apps=apps,
        scalewob_root=args.scalewob_root,
        max_steps=args.max_steps,
        num_sessions=args.num_sessions,
        output_dir=args.eval_output,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        headless=not args.no_headless,
        port=args.port,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()