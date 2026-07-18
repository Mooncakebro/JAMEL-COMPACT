"""
Baseline evaluation: pure Qwen3-VL (no side memory).

Evaluates the SFT'd Qwen3-VL baseline on the same ScaleWoB benchmark apps
as JAMEL-COMPACT, using the same prompt format and action parsing.  The only
difference is that there is no memory module — each step is independent.

This lets us directly compare:
  - JAMEL-COMPACT (with side memory)  vs.
  - Pure Qwen3-VL SFT (no memory)

on the same data, same eval harness, same metrics.

Usage:
    python -m jamel_compact.baseline_eval \
        --checkpoint outputs/baseline_ckpt/final \
        --apps-mode test10 \
        --max-steps 50 \
        --num-sessions 3 \
        --eval-output outputs/baseline_eval
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


# ── Action parsing (same as compact eval) ──

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def parse_action(response: str) -> str:
    """Extract action from model response."""
    match = ACTION_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip().split("\n")[0].strip()


# ── Static server for ScaleWoB (same as compact eval) ──

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


# ── Baseline Agent (pure Qwen3-VL, no memory) ──

class BaselineAgent:
    """
    Wraps a pure Qwen3-VL model for step-by-step session inference.

    Unlike CompactAgent, there is no memory state — each step is independent.
    The model sees only the current screenshot and prompt.
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
        print(f"[agent] Loading baseline Qwen3-VL from {checkpoint} ...")

        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[agent] AutoModelForImageTextToText failed ({e}), "
                  f"trying AutoModelForCausalLM...")
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.model = self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        try:
            self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception:
            self.processor = None

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.image_resize = image_resize

        # Step counter (for prompt building, same format as compact eval)
        self._session_step_idx = 0

    def reset_session(self):
        """Start of a new session — no memory to reset, just the step counter."""
        self._session_step_idx = 0

    def _build_prompt(self, obs_dict: dict, target_app: str, start_url: str,
                      max_steps: int) -> str:
        """Build the canonical web prompt (same as compact eval)."""
        from jamel.train.memory.web_prompt import build_web_prompt, extract_axtree_from_observation_str
        from jamel.core.env.web.axtree_utils import prune_axtree

        obs_text = obs_dict.get("axtree_object", "") or obs_dict.get("observation", "")
        if not obs_text and "dom_object" in obs_dict:
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

    @torch.inference_mode()
    def decide_action(self, obs_dict: dict, target_app: str, start_url: str,
                      max_steps: int) -> dict:
        """Decide the next action given the current observation.

        Uses the standard HuggingFace generate() — no memory, no side modules.
        """
        prompt = self._build_prompt(obs_dict, target_app, start_url, max_steps)

        # Process screenshot
        screenshot_arr = obs_dict.get("screenshot")
        image = None
        if screenshot_arr is not None:
            image = Image.fromarray(screenshot_arr.astype(np.uint8))
            if image.size != self.image_resize:
                image = image.resize(self.image_resize, Image.BILINEAR)

        # Build messages
        has_image = "<image>" in prompt
        segments = prompt.split("<image>")
        content = []
        for idx, seg in enumerate(segments):
            if seg:
                content.append({"type": "text", "text": seg})
            if idx < len(segments) - 1:
                content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]

        # Tokenize
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

        # Truncate to max_input_tokens (keep the most recent tokens)
        orig_len = inputs["input_ids"].shape[1]
        if orig_len > self.max_input_tokens:
            for k in list(inputs.keys()):
                if hasattr(inputs[k], "shape") and inputs[k].shape[-1] == orig_len:
                    inputs[k] = inputs[k][..., -self.max_input_tokens:]

        # ── Generate (standard HF generate, no memory) ──
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0, input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        action = parse_action(response)

        self._session_step_idx += 1

        return {
            "action": action,
            "response": response,
            "prompt": prompt,
        }


# ── Evaluation loop (same structure as compact eval) ──

def run_eval(
    checkpoint: str,
    apps: list[str],
    scalewob_root: str,
    max_steps: int = 50,
    num_sessions: int = 3,
    output_dir: str = "outputs/baseline_eval",
    device: str = "cuda",
    temperature: float = 0.8,
    top_p: float = 0.9,
    headless: bool = True,
):
    """
    Run baseline Qwen3-VL evaluation on ScaleWoB apps.

    For each app, runs `num_sessions` sessions of `max_steps` steps each.
    No memory is maintained — each step is independent.
    """
    from browsergym.core.env import BrowserEnv
    from browsergym.utils.obs import flatten_axtree_to_str

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Start static server
    server = LocalStaticServer(Path(scalewob_root), port=8790)
    server.start()

    # Initialize agent
    agent = BaselineAgent(
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

            agent.reset_session()

            env = BrowserEnv(
                headless=headless,
                viewport={"width": 1280, "height": 720},
            )

            try:
                obs, info = env.reset(start_url=start_url)

                session_reward = 0.0
                step_records = []

                for step in range(max_steps):
                    result = agent.decide_action(obs, app, start_url, max_steps)
                    action = result["action"]

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

                    print(f"    [step {step}] action={action[:50]}  "
                          f"reward={reward:.4f}  cumul={session_reward:.4f}")

                    if terminated or truncated:
                        print(f"    [step {step}] episode ended "
                              f"(terminated={terminated}, truncated={truncated})")
                        obs, info = env.reset(start_url=start_url)

                session_result = {
                    "app": app,
                    "session_idx": session_idx,
                    "total_reward": session_reward,
                    "num_steps": len(step_records),
                    "steps": step_records,
                }
                all_results.append(session_result)

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
        "model_type": "baseline_qwen3vl_sft",
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
    print("BASELINE EVALUATION SUMMARY (Pure Qwen3-VL SFT)")
    print(f"{'='*60}")
    for result in all_results:
        print(f"  {result['app']:<20s} session {result['session_idx']}: "
              f"reward={result['total_reward']:.4f}  steps={result['num_steps']}")

    server.stop()


def main():
    parser = argparse.ArgumentParser(description="Baseline Qwen3-VL SFT Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    parser.add_argument("--apps", default="",
                        help="Comma-separated app list (overrides --apps-mode)")
    parser.add_argument("--apps-mode", default="test10", choices=["test10", "train86", "all"])
    parser.add_argument("--scalewob-root", default="env/browser_env/scalewob-env")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-sessions", type=int, default=3)
    parser.add_argument("--eval-output", default="outputs/baseline_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

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
    print(f"[eval] Model type: baseline (pure Qwen3-VL SFT, no memory)")
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
