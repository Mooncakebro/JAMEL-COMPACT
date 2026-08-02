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
import gc
import os
import re
import sys
from pathlib import Path

from .gpu import configure_cuda_visibility

configure_cuda_visibility()

import numpy as np
import torch
from PIL import Image

from .lora import load_lora_adapter, read_lora_adapter_config
from .eval_reservation import reserve_eval_run


def _load_baseline_checkpoint(model_class, checkpoint: str, load_kwargs: dict):
    adapter_config_path = Path(checkpoint) / "adapter_config.json"
    if not adapter_config_path.is_file():
        return model_class.from_pretrained(checkpoint, **load_kwargs), None

    adapter_config = read_lora_adapter_config(checkpoint)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"LoRA checkpoint is missing base_model_name_or_path: {adapter_config_path}"
        )

    model = model_class.from_pretrained(base_model_name, **load_kwargs)
    try:
        model = load_lora_adapter(model, checkpoint, is_trainable=False)
    except Exception:
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    print(f"[agent] Loaded LoRA adapter over base model {base_model_name}")
    return model, base_model_name


# ── Action / think parsing (same as compact eval) ──

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

        # ── Load model directly onto GPU to avoid CPU RAM doubling ──
        # Loading to CPU then .to(device) temporarily holds 2× the weights
        # in system RAM. On memory-constrained servers this causes OOM-kill
        # when the browser environment (Chromium) starts afterwards.
        # Keep weights off CPU after loading. With multiple visible GPUs,
        # distribute the model instead of forcing every layer onto cuda:0.
        _load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                _load_kwargs["device_map"] = "balanced"
            else:
                _load_kwargs["device_map"] = {"": device}

        adapter_base_model = None
        try:
            self.model, adapter_base_model = _load_baseline_checkpoint(
                AutoModelForImageTextToText, checkpoint, _load_kwargs,
            )
        except Exception as e:
            print(f"[agent] AutoModelForImageTextToText failed ({e}), "
                  f"trying AutoModelForCausalLM...")
            gc.collect()
            torch.cuda.empty_cache()
            from transformers import AutoModelForCausalLM
            self.model, adapter_base_model = _load_baseline_checkpoint(
                AutoModelForCausalLM, checkpoint, _load_kwargs,
            )

        # Only .to(device) if device_map wasn't used (CPU fallback path)
        if "device_map" not in _load_kwargs:
            self.model = self.model.to(device)
        self.model.eval()

        # ── Override training-only settings for inference ──
        # Checkpoints saved during gradient-checkpointing training may have
        # use_cache=False, which breaks generation and wastes memory.
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:
            pass
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True

        # Free CPU-side weight copies before the browser env starts (avoids OOM kill)
        gc.collect()
        torch.cuda.empty_cache()

        input_embedding = self.model.get_input_embeddings()
        self.device = next(input_embedding.parameters()).device
        asset_source = checkpoint
        try:
            self.processor = AutoProcessor.from_pretrained(
                asset_source, trust_remote_code=True,
            )
        except Exception:
            if adapter_base_model is not None:
                self.processor = AutoProcessor.from_pretrained(
                    adapter_base_model, trust_remote_code=True,
                )
            else:
                raise RuntimeError(
                    f"AutoProcessor failed to load from checkpoint: {checkpoint}"
                )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            tokenizer_source = adapter_base_model or asset_source
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, trust_remote_code=True,
            )
        gc.collect()

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

    @torch.inference_mode()
    def decide_action(self, obs_dict: dict, target_url: str, start_url: str,
                      max_steps: int) -> dict:
        """Decide the next action given the current observation.

        Uses the standard HuggingFace generate() — no memory, no side modules.
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

        # Build messages — always use processor (same as original JAMEL eval)
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

        # Truncate to max_input_tokens (keep the most recent tokens)
        orig_len = inputs["input_ids"].shape[1]
        if orig_len > self.max_input_tokens:
            for k in list(inputs.keys()):
                if hasattr(inputs[k], "shape") and inputs[k].shape[-1] == orig_len:
                    inputs[k] = inputs[k][..., -self.max_input_tokens:]

        # ── Generate (standard HF generate, no memory) ──
        # Do not kill the whole evaluator from a background timer.  A slow
        # generation should be handled by the per-session error boundary.
        generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens — batch_decode matches original JAMEL eval
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        raw_response = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        action, think = parse_action(raw_response)

        self._session_step_idx += 1

        return {
            "action": action,
            "think": think,
            "raw_response": raw_response,
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
    port: int = 8790,
    seed: int = 42,
    browser_timeout_ms: int = 30_000,
    reset_retries: int = 3,
    max_input_tokens: int = 8192,
    max_new_tokens: int = 256,
    save_screenshots: bool = False,
    resume: bool = True,
):
    """Run baseline Qwen3-VL evaluation with per-session crash recovery."""
    from .eval_browser import run_browser_evaluation

    agent = BaselineAgent(
        checkpoint=checkpoint,
        device=device,
        temperature=temperature,
        top_p=top_p,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
    return run_browser_evaluation(
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
        model_type="baseline_qwen3vl_sft",
        summary_title="BASELINE EVALUATION SUMMARY (Pure Qwen3-VL SFT)",
        browser_timeout_ms=browser_timeout_ms,
        reset_retries=reset_retries,
        save_screenshots=save_screenshots,
        resume=resume,
    )


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
    parser.add_argument("--port", type=int, default=0,
                        help="ScaleWoB server port; 0 automatically reserves a free port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env.reset()")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU IDs (e.g. '0' or '0,1')")
    parser.add_argument("--browser-timeout-ms", type=int, default=30000,
                        help="Per-page navigation timeout; failed sessions are retried")
    parser.add_argument("--reset-retries", type=int, default=3,
                        help="Fresh-browser attempts for each reset")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-screenshots", action="store_true",
                        help="Save before/after screenshots (uses extra disk and CPU)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore prior eval_summary.json results and rerun all sessions")
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
    print("[eval] Model type: baseline (pure Qwen3-VL SFT, no memory)")
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
            browser_timeout_ms=args.browser_timeout_ms,
            reset_retries=args.reset_retries,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            save_screenshots=args.save_screenshots,
            resume=not args.no_resume,
        )
    finally:
        run_reservation.close()


if __name__ == "__main__":
    main()
