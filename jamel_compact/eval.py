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
import gc
import json
import os
import re
import sys
from pathlib import Path

from .gpu import configure_cuda_visibility

configure_cuda_visibility()

import numpy as np
import torch
from PIL import Image

from .model import JAMELCompactWrapper
from .eval_reservation import reserve_eval_run

# F8: Linear probe for memory diagnostics
import sys as _sys
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
try:
    from scripts.probe_memory import LinearProbeMemory
except ImportError:
    LinearProbeMemory = None

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
        use_model_parallel = device == "cuda" and torch.cuda.device_count() > 1
        self.model = JAMELCompactWrapper.from_pretrained(
            checkpoint,
            model_parallel_override=use_model_parallel,
        )
        if self.model.model_parallel:
            self.device = self.model.input_device
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
        self.model.eval()

        # ── Override training-only settings that waste memory at inference ──
        # The checkpoint config may have gradient_checkpointing=True / use_cache=False
        # (saved during training).  These are harmful during eval:
        #   - gradient_checkpointing recomputes activations, wasting GPU memory
        #   - use_cache=False prevents KV-cache reuse, slowing generation
        try:
            self.model.llm.gradient_checkpointing_disable()
        except Exception:
            pass
        if hasattr(self.model.llm, 'config'):
            self.model.llm.config.use_cache = True

        # Free CPU-side weight copies before the browser env starts (avoids OOM kill)
        gc.collect()
        torch.cuda.empty_cache()

        self.tokenizer = self.model.tokenizer
        self.processor = self.model.processor

        if self.processor is None:
            raise RuntimeError(
                "Processor failed to load from checkpoint. The processor files "
                "(preprocessor_config.json etc.) are missing. Ensure the checkpoint "
                f"was saved with processor files: {checkpoint}"
            )
        if self.tokenizer is None:
            raise RuntimeError(
                f"Tokenizer failed to load from checkpoint: {checkpoint}"
            )
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.image_resize = image_resize

        # Session state
        self._memory_states = None
        self._variance_states = None
        self._e_prev_list = None  # v2: surprise from previous step
        self._session_step_idx = 0
        self._last_action = "noop()"
        self._freeze_memory = False  # set by --freeze-memory-init flag

    def reset_session(self):
        """Full reset — start of a new session."""
        self._memory_states, self._variance_states = self.model.init_memory(
            batch_size=1, device=self.device,
        )
        self._e_prev_list = None
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

        # Do not kill the whole evaluator from a background timer.  A slow
        # generation should be handled by the per-session error boundary.
        outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                action_embed_input=action_embed_input,
                memory_states=self._memory_states,
                variance_states=self._variance_states,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                e_prev_list=self._e_prev_list,
                freeze_memory=self._freeze_memory,
            )

        # Update memory states
        self._memory_states = outputs["new_memory"]
        self._variance_states = outputs["new_variance"]
        self._e_prev_list = outputs.get("e_list")

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

    def get_memory_snapshot(self) -> list:
        """Return current memory states for F8 linear probe."""
        return self._memory_states


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
    freeze_memory_init: bool = False,
    browser_timeout_ms: int = 30_000,
    reset_retries: int = 3,
    max_input_tokens: int = 8192,
    max_new_tokens: int = 256,
    save_screenshots: bool = False,
    enable_linear_probe: bool = False,
):
    """Run JAMEL-COMPACT evaluation with per-session crash recovery."""
    from .eval_browser import run_browser_evaluation

    agent = CompactAgent(
        checkpoint=checkpoint,
        device=device,
        temperature=temperature,
        top_p=top_p,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
    agent._freeze_memory = freeze_memory_init
    if freeze_memory_init:
        print("[eval] WARNING: --freeze-memory-init is active. Memory will NOT update during eval.")

    probe = None
    if enable_linear_probe and LinearProbeMemory is not None:
        probe = LinearProbeMemory(
            num_layers=agent.model.num_layers,
            mem_dim=agent.model.mem_dim,
            num_mem=agent.model.num_mem,
            num_apps=len(apps),
            max_steps=max_steps,
        )
        print(
            f"[eval] F8 linear probe initialized: {agent.model.num_layers} layers, "
            f"d_mem={agent.model.mem_dim}, N_m={agent.model.num_mem}"
        )
    elif enable_linear_probe:
        print("[eval] F8 linear probe unavailable (scripts.probe_memory import failed)")

    app_id_map = {app: index for index, app in enumerate(apps)}

    def _collect_probe_snapshot(app: str, step_idx: int) -> None:
        if probe is not None:
            probe.add_snapshot(
                agent.get_memory_snapshot(),
                app_id=app_id_map[app],
                step_idx=step_idx,
            )

    results = run_browser_evaluation(
        agent=agent,
        checkpoint=checkpoint,
        apps=apps,
        scalewob_root=scalewob_root,
        max_steps=max_steps,
        num_sessions=num_sessions,
        output_dir=output_dir,
        headless=headless,
        port=port,
        seed=seed,
        model_type="jamel_compact",
        summary_title="EVALUATION SUMMARY (JAMEL-COMPACT)",
        browser_timeout_ms=browser_timeout_ms,
        reset_retries=reset_retries,
        save_screenshots=save_screenshots,
        on_step_decision=_collect_probe_snapshot if probe is not None else None,
    )

    if probe is not None:
        probe_results = probe.train_and_eval()
        probe.print_results(probe_results)
        probe_path = Path(output_dir) / "linear_probe_results.json"
        probe_path.write_text(json.dumps(
            {str(key): value for key, value in probe_results.items()}, indent=2,
        ))
        print(f"\n[eval] Linear probe results saved to {probe_path}")

    return results


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
    parser.add_argument("--port", type=int, default=0,
                        help="ScaleWoB server port; 0 automatically reserves a free port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env.reset()")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU IDs (e.g. '0' or '0,1')")
    parser.add_argument("--freeze-memory-init", action="store_true",
                        help="Freeze memory at initial state (ablation: tests "
                             "whether memory writes improve generation)")
    parser.add_argument("--browser-timeout-ms", type=int, default=30000,
                        help="Per-page navigation timeout; failed sessions are retried")
    parser.add_argument("--reset-retries", type=int, default=3,
                        help="Fresh-browser attempts for each reset")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-screenshots", action="store_true",
                        help="Save before/after screenshots (uses extra disk and CPU)")
    parser.add_argument("--enable-linear-probe", action="store_true",
                        help="Enable the optional memory probe; disabled by default")
    args = parser.parse_args()

    # Resolve apps
    if args.apps:
        apps = [a.strip() for a in args.apps.split(",") if a.strip()]
        # Guard against common mistake: using an apps-mode name as a literal app
        _mode_names = {"test10", "train86", "all"}
        if len(apps) == 1 and apps[0] in _mode_names:
            print(f"[eval] ERROR: '{apps[0]}' is an apps-mode, not an app name.")
            print(f"  Use: APPS_MODE={apps[0]}  (not APPS={apps[0]})")
            sys.exit(2)
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

    try:
        run_reservation = reserve_eval_run(args.port, args.eval_output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[eval] ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    args.port = run_reservation.port

    print(f"[eval] Apps: {apps}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] Max steps: {args.max_steps}")
    print(f"[eval] Sessions: {args.num_sessions}")
    print(f"[eval] ScaleWoB port: {args.port}")
    visible_gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[eval] Physical GPU IDs visible: {visible_gpu_ids or 'all'}")
    if torch.cuda.is_available():
        print(f"[eval] Logical CUDA devices: {torch.cuda.device_count()} "
              f"(cuda:0 maps to physical GPU {(visible_gpu_ids.split(',')[0] if visible_gpu_ids else '0')})")

    try:
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
            freeze_memory_init=args.freeze_memory_init,
            browser_timeout_ms=args.browser_timeout_ms,
            reset_retries=args.reset_retries,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            save_screenshots=args.save_screenshots,
            enable_linear_probe=args.enable_linear_probe,
        )
    finally:
        run_reservation.close()


if __name__ == "__main__":
    main()
