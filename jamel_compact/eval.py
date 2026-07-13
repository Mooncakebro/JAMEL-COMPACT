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
import json
import os
import re
import signal
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CompactConfig
from .model import JAMELCompactWrapper


# ── Action parsing ──

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def parse_action(response: str) -> str:
    """Extract action from model response."""
    match = ACTION_RE.search(response)
    if match:
        return match.group(1).strip()
    # Fallback: take first line
    return response.strip().split("\n")[0].strip()


# ── Static server for ScaleWoB ──

class LocalStaticServer:
    def __init__(self, root_dir: Path, host: str = "127.0.0.1", port: int = 8790):
        self.root_dir = Path(root_dir)
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    def start(self):
        from functools import partial
        if self._server is not None:
            return
        handler = partial(SimpleHTTPRequestHandler, directory=str(self.root_dir))
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="scalewob-eval", daemon=True,
        )
        self._thread.start()
        print(f"[server] Serving {self.root_dir} at http://{self.host}:{self.port}/")

    def stop(self):
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None


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

    def _build_prompt(self, obs_dict: dict, target_app: str, start_url: str,
                      max_steps: int) -> str:
        """Build the canonical web prompt."""
        from jamel.train.memory.web_prompt import build_web_prompt, extract_axtree_from_observation_str
        from jamel.core.env.web.axtree_utils import prune_axtree

        obs_text = obs_dict.get("axtree_object", "") or obs_dict.get("observation", "")
        if not obs_text and "dom_object" in obs_dict:
            # Fallback: use Observer if available
            try:
                from jamel.core.env.web import Observer
                obs_text = Observer.get_observation(obs_dict)
            except Exception:
                obs_text = str(obs_dict)

        axtree_raw = extract_axtree_from_observation_str(obs_text)
        pruned_axtree = prune_axtree(axtree_raw, max_chars=8000)

        return build_web_prompt(
            step_idx=self._session_step_idx,
            target_app=target_app,
            start_url=start_url,
            open_urls=obs_dict.get("open_pages_urls", (start_url,)),
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
        embed_layer = self.model.llm.get_input_embeddings()
        return embed_layer(token_ids).mean(dim=1)  # [1, d]

    @torch.inference_mode()
    def decide_action(self, obs_dict: dict, target_app: str, start_url: str,
                      max_steps: int) -> dict:
        """Decide the next action given the current observation."""
        prompt = self._build_prompt(obs_dict, target_app, start_url, max_steps)

        # Process screenshot
        screenshot_arr = obs_dict.get("screenshot")
        image = None
        if screenshot_arr is not None:
            image = Image.fromarray(screenshot_arr.astype(np.uint8))
            if image.size != self.image_resize:
                image = image.resize(self.image_resize, Image.BILINEAR)

        # Build model inputs
        has_image = "<image>" in prompt
        segments = prompt.split("<image>")
        content = []
        for idx, seg in enumerate(segments):
            if seg:
                content.append({"type": "text", "text": seg})
            if idx < len(segments) - 1:
                content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]

        if self.processor is not None and has_image and image is not None:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt_text], images=[image], return_tensors="pt",
            )
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = {"input_ids": self.tokenizer.encode(prompt_text, return_tensors="pt")}
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

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

        # Generate
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            action_embed_input=action_embed_input,
            memory_states=self._memory_states,
            confidence_states=self._confidence_states,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Update memory states
        self._memory_states = outputs["new_memory"]
        self._confidence_states = outputs["new_confidence"]

        # Decode response
        generated_ids = outputs["generated_ids"]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        action = parse_action(response)

        # Update state
        self._last_action = action
        self._session_step_idx += 1

        return {
            "action": action,
            "response": response,
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
):
    """
    Run JAMEL-COMPACT evaluation on ScaleWoB apps.

    For each app, runs `num_sessions` sessions of `max_steps` steps each.
    Memory is maintained across episodes within a session.
    """
    from browsergym.core.env import BrowserEnv
    from browsergym.utils.obs import flatten_axtree_to_str

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start static server
    server = LocalStaticServer(Path(scalewob_root), port=8790)
    server.start()

    # Initialize agent
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

        start_url = f"http://127.0.0.1:{server.port}/{app}/index.html"

        for session_idx in range(num_sessions):
            print(f"\n  Session {session_idx + 1}/{num_sessions}")

            # Reset session memory
            agent.reset_session()

            # Create browser env
            env = BrowserEnv(
                headless=headless,
                viewport={"width": 1280, "height": 720},
            )

            try:
                obs, info = env.reset(start_url=start_url)

                session_reward = 0.0
                step_records = []

                for step in range(max_steps):
                    # Decide action
                    result = agent.decide_action(obs, app, start_url, max_steps)
                    action = result["action"]

                    # Execute action
                    try:
                        obs, reward, terminated, truncated, info = env.step(action)
                    except Exception as e:
                        print(f"    [step {step}] env.step error: {e}")
                        break

                    session_reward += reward
                    step_records.append({
                        "session_idx": session_idx,
                        "step": step,
                        "action": action,
                        "reward": reward,
                        "cumulative_reward": session_reward,
                    })

                    print(f"    [step {step}] action={action[:50]}  reward={reward:.4f}  cumul={session_reward:.4f}")

                    if terminated or truncated:
                        print(f"    [step {step}] episode ended (terminated={terminated}, truncated={truncated})")
                        # Reset env but keep memory
                        obs, info = env.reset(start_url=start_url)

                # Save session results
                session_result = {
                    "app": app,
                    "session_idx": session_idx,
                    "total_reward": session_reward,
                    "num_steps": len(step_records),
                    "steps": step_records,
                }
                all_results.append(session_result)

                # Save trajectory parquet
                traj_df = pd.DataFrame(step_records)
                traj_path = output_path / f"{app}_session{session_idx}.parquet"
                traj_df.to_parquet(traj_path)
                print(f"  Session reward: {session_reward:.4f}  → {traj_path}")

            finally:
                env.close()

    # Save summary
    summary_path = output_path / "eval_summary.json"
    summary = {
        "checkpoint": checkpoint,
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
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        print(f"  {result['app']:<20s} session {result['session_idx']}: "
              f"reward={result['total_reward']:.4f}  steps={result['num_steps']}")

    server.stop()


def main():
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
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    # Resolve apps
    if args.apps:
        apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    else:
        # Use benchmark_apps.json split
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
            # Default test10
            apps = ["vipshop", "alibaba", "expedia", "taobao", "pinduoduo",
                    "dongchedi", "youku", "keep", "meituan", "temu"]

    print(f"[eval] Apps: {apps}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] Max steps: {args.max_steps}")
    print(f"[eval] Sessions: {args.num_sessions}")

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
    )


if __name__ == "__main__":
    main()