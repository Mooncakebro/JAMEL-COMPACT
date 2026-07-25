#!/usr/bin/env python
"""
F4 Acceptance Test: Verify zero-init injection at model initialization.

At init, the model should behave exactly like the base LLM because:
  1. delta_up.weight and delta_up.bias are zeros → delta_up(x) = 0
  2. inject_gate = 0 → tanh(0) = 0 → w = 0

This script checks these conditions and confirms the injection is a no-op.

Usage:
    python scripts/check_zero_init.py --model-path outputs/compact_ckpt/final
    python scripts/check_zero_init.py --base-model Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def check_zero_init(model_path: str = None, base_model: str = None):
    """Check that all side memory injection paths are zero-initialized."""
    # Load model
    if model_path:
        from jamel_compact.model import JAMELCompactWrapper
        print(f"[check] Loading from checkpoint: {model_path}")
        model = JAMELCompactWrapper.from_pretrained(model_path)
    else:
        from jamel_compact.config import CompactConfig
        from jamel_compact.model import JAMELCompactWrapper
        bm = base_model or "Qwen/Qwen3-VL-2B-Instruct"
        print(f"[check] Building fresh model from: {bm}")
        config = CompactConfig(base_model_name=bm)
        model = JAMELCompactWrapper(config)

    model.eval()
    all_pass = True

    print(f"\n[check] Checking {len(model.side_memories)} side memory modules...\n")

    for i, sm in enumerate(model.side_memories):
        layer_pass = True

        # Check delta_up weights and bias are zero
        delta_w = sm.delta_up.weight
        delta_b = sm.delta_up.bias
        delta_w_max = delta_w.abs().max().item()
        delta_b_max = delta_b.abs().max().item() if delta_b is not None else 0.0

        if delta_w_max > 1e-7:
            print(f"  Layer {i}: FAIL — delta_up.weight max abs = {delta_w_max:.2e} (expected 0)")
            layer_pass = False
        else:
            print(f"  Layer {i}: OK   — delta_up.weight max abs = {delta_w_max:.2e}")

        if delta_b_max > 1e-7:
            print(f"  Layer {i}: FAIL — delta_up.bias max abs = {delta_b_max:.2e} (expected 0)")
            layer_pass = False
        else:
            print(f"  Layer {i}: OK   — delta_up.bias max abs = {delta_b_max:.2e}")

        # Check inject_gate is zero
        gate_val = sm.inject_gate.item()
        if abs(gate_val) > 1e-7:
            print(f"  Layer {i}: FAIL — inject_gate = {gate_val:.6f} (expected 0)")
            layer_pass = False
        else:
            print(f"  Layer {i}: OK   — inject_gate = {gate_val:.6f}")

        # Compute effective injection weight: w_max * tanh(gate)
        w_max = sm.inject_w_max
        w_effective = w_max * torch.tanh(sm.inject_gate).item()
        print(f"  Layer {i}: OK   — effective injection weight = {w_effective:.6f} "
              f"(w_max={w_max}, tanh(gate)={torch.tanh(sm.inject_gate).item():.6f})")

        if not layer_pass:
            all_pass = False
        print()

    # ── Functional test: verify inject(h, m) == h at init ──
    print("[check] Functional test: inject(h, m_new) should equal h at init...\n")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    B, N, d = 2, 16, model.side_memories[0].hidden_dim

    for i, sm in enumerate(model.side_memories):
        h = torch.randn(B, N, d, device=device, dtype=dtype)
        m = sm.init_memory.unsqueeze(0).expand(B, -1, -1).clone().to(device=device, dtype=dtype)

        h_injected = sm.inject(h, m)
        diff = (h_injected - h).abs().max().item()

        if diff > 1e-5:
            print(f"  Layer {i}: FAIL — max |inject(h,m) - h| = {diff:.2e} (expected ~0)")
            all_pass = False
        else:
            print(f"  Layer {i}: OK   — max |inject(h,m) - h| = {diff:.2e}")

    print()
    if all_pass:
        print("=" * 50)
        print("✓ ALL CHECKS PASSED — Zero-init injection is correct.")
        print("  At init, the model behaves identically to the base LLM.")
        print("=" * 50)
    else:
        print("=" * 50)
        print("✗ SOME CHECKS FAILED — See above for details.")
        print("=" * 50)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="F4: Verify zero-init injection")
    parser.add_argument("--model-path", default=None,
                        help="Path to saved checkpoint (uses from_pretrained)")
    parser.add_argument("--base-model", default=None,
                        help="Base model name (only if --model-path not given)")
    args = parser.parse_args()

    if args.model_path is None and args.base_model is None:
        # Default: try to find a checkpoint
        for candidate in [
            "outputs/compact_ckpt/final",
            "outputs/compact_ckpt/best",
        ]:
            if Path(candidate).exists():
                args.model_path = candidate
                break

    check_zero_init(model_path=args.model_path, base_model=args.base_model)


if __name__ == "__main__":
    main()
