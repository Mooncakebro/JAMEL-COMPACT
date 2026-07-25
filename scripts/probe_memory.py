#!/usr/bin/env python
"""
F8 Diagnostic: Probe memory module behavior during training.

Logs per-layer statistics:
  - Memory state norm (is memory being used?)
  - Variance P values (is uncertainty evolving?)
  - Surprise e (is the observation model detecting novelty?)
  - Kalman gain K (is the filter updating?)
  - Injection weight w (is injection growing from zero-init?)
  - Observation loss and NLL (are the auxiliary losses training?)

Usage:
    # Standalone probe (creates a fresh model)
    python scripts/probe_memory.py --base-model Qwen/Qwen3-VL-2B-Instruct

    # Probe a trained checkpoint
    python scripts/probe_memory.py --model-path outputs/compact_ckpt/final

    # As a training callback (import and call)
    # from scripts.probe_memory import log_memory_stats
    # log_memory_stats(model, writer, global_step)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_memory_stats(model) -> dict:
    """Collect per-layer memory statistics from a JAMELCompactWrapper.

    Returns a dict with keys:
        layer_{i}_mem_norm:    mean L2 norm of memory state
        layer_{i}_var_mean:    mean variance P
        layer {i}_var_std:     std of variance P
        layer_{i}_gate:        effective injection weight (w_max * tanh(gate))
        layer_{i}_lambda:      mean sigmoid(lambda_l_raw) — variance decay
        layer_{i}_Q_mean:      mean process noise Q (softplus(Q_theta))
        layer_{i}_R_mean:      mean observation noise R (softplus(R_psi))
    """
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    stats = {}

    for i, sm in enumerate(raw.side_memories):
        prefix = f"layer_{i}"

        # Injection gate effective weight
        gate_eff = sm.inject_w_max * torch.tanh(sm.inject_gate).item()
        stats[f"{prefix}_gate"] = gate_eff

        # Lambda decay
        lam = torch.sigmoid(sm.lambda_l_raw)
        stats[f"{prefix}_lambda_mean"] = lam.mean().item()
        stats[f"{prefix}_lambda_std"] = lam.std().item()

        # Process noise Q (using a zero input as proxy)
        zero_input = torch.zeros(1, sm.mem_dim, device=lam.device, dtype=lam.dtype)
        Q = F.softplus(sm.Q_theta(zero_input))
        stats[f"{prefix}_Q_mean"] = Q.mean().item()

        # Observation noise R
        R = F.softplus(sm.R_psi(zero_input))
        stats[f"{prefix}_R_mean"] = R.mean().item()

        # Init memory norm
        init_m_norm = sm.init_memory.norm(dim=-1).mean().item()
        stats[f"{prefix}_init_mem_norm"] = init_m_norm

    return stats


@torch.no_grad()
def collect_runtime_stats(
    model,
    memory_states: list,
    variance_states: list,
    e_list: Optional[list] = None,
) -> dict:
    """Collect statistics from actual runtime memory states.

    Call this after a forward pass with the actual memory_states and
    variance_states from the model output.

    Args:
        model:           JAMELCompactWrapper (or DataParallel)
        memory_states:   list of [B, N_m, d_mem] — from outputs["new_memory"]
        variance_states: list of [B, N_m] — from outputs["new_confidence"]
        e_list:          list of [B] — from outputs["e_list"] (surprise)

    Returns:
        dict of statistics
    """
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    stats = {}

    for i, (m, p) in enumerate(zip(memory_states, variance_states)):
        prefix = f"layer_{i}"

        # Memory state norm
        m_norm = m.norm(dim=-1).mean().item()  # mean over B and N_m
        stats[f"{prefix}_mem_norm"] = m_norm

        # Variance stats
        stats[f"{prefix}_var_mean"] = p.mean().item()
        stats[f"{prefix}_var_std"] = p.std().item()
        stats[f"{prefix}_var_min"] = p.min().item()
        stats[f"{prefix}_var_max"] = p.max().item()

        # Kalman gain proxy: K = P / (P + R)
        # Using mean R from the layer's R_psi with zero input
        zero_input = torch.zeros(1, raw.side_memories[i].mem_dim,
                                 device=p.device, dtype=p.dtype)
        R = F.softplus(raw.side_memories[i].R_psi(zero_input))  # [1, N_m]
        K = p.mean(dim=0) / (p.mean(dim=0) + R.squeeze(0) + 1e-8)  # [N_m]
        stats[f"{prefix}_kalman_gain_mean"] = K.mean().item()

        # Surprise
        if e_list is not None and i < len(e_list) and e_list[i] is not None:
            stats[f"{prefix}_surprise_mean"] = e_list[i].mean().item()
            stats[f"{prefix}_surprise_max"] = e_list[i].max().item()

    return stats


def log_memory_stats(model, writer, global_step: int):
    """Log memory module statistics to TensorBoard.

    Usage during training:
        from scripts.probe_memory import log_memory_stats
        log_memory_stats(model, writer, global_step)
    """
    stats = collect_memory_stats(model)

    for key, val in stats.items():
        # Split layer index and metric name
        parts = key.split("_", 2)
        if len(parts) == 3:
            layer_tag = parts[1]
            metric = parts[2]
            writer.add_scalar(f"memory/L{layer_tag}_{metric}", val, global_step)
        else:
            writer.add_scalar(f"memory/{key}", val, global_step)


def print_stats(stats: dict, title: str = "Memory Stats"):
    """Pretty-print memory statistics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    # Group by layer
    layers = {}
    for key, val in stats.items():
        parts = key.split("_", 2)
        if len(parts) == 3:
            layer = parts[1]
            metric = parts[2]
            if layer not in layers:
                layers[layer] = {}
            layers[layer][metric] = val

    for layer in sorted(layers.keys(), key=int):
        s = layers[layer]
        print(f"\n  Layer {layer}:")
        for metric in sorted(s.keys()):
            val = s[metric]
            if isinstance(val, float):
                print(f"    {metric:30s} = {val:.6f}")
            else:
                print(f"    {metric:30s} = {val}")


def main():
    parser = argparse.ArgumentParser(description="F8: Probe memory module statistics")
    parser.add_argument("--model-path", default=None,
                        help="Path to saved checkpoint")
    parser.add_argument("--base-model", default=None,
                        help="Base model name (if no checkpoint)")
    parser.add_argument("--device", default="cpu",
                        help="Device to load model on (cpu or cuda)")
    args = parser.parse_args()

    # Load model
    if args.model_path:
        from jamel_compact.model import JAMELCompactWrapper
        print(f"[probe] Loading from checkpoint: {args.model_path}")
        model = JAMELCompactWrapper.from_pretrained(args.model_path)
    else:
        from jamel_compact.config import CompactConfig
        from jamel_compact.model import JAMELCompactWrapper
        bm = args.base_model or "Qwen/Qwen3-VL-2B-Instruct"
        print(f"[probe] Building fresh model from: {bm}")
        config = CompactConfig(base_model_name=bm)
        model = JAMELCompactWrapper(config)

    model = model.to(args.device)
    model.eval()

    # ── 1. Static parameter stats (from init) ──
    stats = collect_memory_stats(model)
    print_stats(stats, "Static Parameter Stats (from init)")

    # ── 2. Simulated forward pass stats ──
    print("\n[probe] Running simulated forward pass...")
    B = 2
    device = torch.device(args.device)
    memory_states, variance_states = model.init_memory(B, device)

    # Create dummy input
    d = model.side_memories[0].hidden_dim
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device).expand(B, -1)
    attention_mask = torch.ones_like(input_ids)
    action_embed = torch.randn(B, d, device=device)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                action_embed_input=action_embed,
                memory_states=memory_states,
                confidence_states=variance_states,
            )
        runtime_stats = collect_runtime_stats(
            model,
            outputs["new_memory"],
            outputs["new_confidence"],
            outputs.get("e_list"),
        )
        print_stats(runtime_stats, "Runtime Stats (after 1 forward step)")

        # Also show loss components if available
        if "loss_dict" in outputs:
            print(f"\n{'='*60}")
            print(f"  Loss Components")
            print(f"{'='*60}")
            for k, v in outputs["loss_dict"].items():
                val = v.mean().item() if isinstance(v, torch.Tensor) else float(v)
                print(f"    {k:20s} = {val:.6f}")

    except Exception as e:
        print(f"\n[probe] Could not run forward pass (expected if no torch/CUDA): {e}")
        print("[probe] Showing static stats only.")

    print(f"\n{'='*60}")
    print("Done.")


class LinearProbeMemory:
    """F8: Linear probe on memory snapshots.

    Trains a simple linear classifier on flattened memory states M_t to
    predict metadata (app_id, step_idx). If the probe achieves accuracy
    significantly above chance, the memory is encoding meaningful information.

    Usage during eval:
        probe = LinearProbeMemory(num_layers, mem_dim, num_mem,
                                  num_apps=10, max_steps=50)
        # Collect snapshots during eval sessions
        probe.add_snapshot(memory_states, app_id=3, step_idx=5)
        # After collecting, train and evaluate
        results = probe.train_and_eval()
    """

    def __init__(self, num_layers: int, mem_dim: int, num_mem: int,
                 num_apps: int = 10, max_steps: int = 50):
        self.num_layers = num_layers
        self.mem_dim = mem_dim
        self.num_mem = num_mem
        self.num_apps = num_apps
        self.max_steps = max_steps

        # Storage: list per layer of (features, app_label, step_label)
        self._snapshots = [[] for _ in range(num_layers)]

    def add_snapshot(self, memory_states: list, app_id: int, step_idx: int):
        """Record one memory snapshot from an eval step.

        Args:
            memory_states: list of [B, N_m, d_mem] — one tensor per layer
            app_id:        int — which app this session is running
            step_idx:      int — step number within the session
        """
        for l, m in enumerate(memory_states):
            if l >= self.num_layers:
                break
            # Flatten [B, N_m, d_mem] → [B, N_m * d_mem] and take batch 0
            feat = m[0].reshape(-1).detach().cpu().float()
            self._snapshots[l].append((feat, app_id, step_idx))

    def train_and_eval(self, test_ratio: float = 0.3, seed: int = 42) -> dict:
        """Train linear probes for app_id and step_idx prediction.

        Returns dict: {
            layer_i: {
                "app_accuracy": float,
                "app_chance": float,
                "step_mae": float,
                "step_chance_mae": float,
                "n_samples": int,
            }
        }
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression, Ridge

        rng = np.random.RandomState(seed)
        results = {}

        for l in range(self.num_layers):
            snaps = self._snapshots[l]
            if len(snaps) < 4:
                results[l] = {"app_accuracy": 0.0, "app_chance": 1.0 / max(self.num_apps, 1),
                              "step_mae": float(self.max_steps), "step_chance_mae": float(self.max_steps / 2),
                              "n_samples": len(snaps)}
                continue

            X = torch.stack([s[0] for s in snaps]).numpy()
            y_app = np.array([s[1] for s in snaps])
            y_step = np.array([s[2] for s in snaps], dtype=float)

            n = len(snaps)
            perm = rng.permutation(n)
            n_test = max(1, int(n * test_ratio))
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

            X_train, X_test = X[train_idx], X[test_idx]
            y_app_train, y_app_test = y_app[train_idx], y_app[test_idx]
            y_step_train, y_step_test = y_step[train_idx], y_step[test_idx]

            # App ID classification
            unique_apps = np.unique(y_app_train)
            if len(unique_apps) > 1:
                clf = LogisticRegression(max_iter=500, multi_class='multinomial')
                clf.fit(X_train, y_app_train)
                app_acc = float((clf.predict(X_test) == y_app_test).mean())
            else:
                app_acc = 0.0
            app_chance = 1.0 / max(len(np.unique(y_app)), 1)

            # Step index regression (MAE)
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_step_train)
            step_pred = reg.predict(X_test)
            step_mae = float(np.abs(step_pred - y_step_test).mean())
            step_chance_mae = float(self.max_steps / 2)  # random guess average

            results[l] = {
                "app_accuracy": app_acc,
                "app_chance": app_chance,
                "step_mae": step_mae,
                "step_chance_mae": step_chance_mae,
                "n_samples": n,
            }

        return results

    def print_results(self, results: dict):
        """Pretty-print linear probe results."""
        print(f"\n{'='*60}")
        print(f"  F8: Linear Probe Results (memory → metadata)")
        print(f"{'='*60}")
        for l in sorted(results.keys()):
            r = results[l]
            app_ratio = r["app_accuracy"] / max(r["app_chance"], 1e-8)
            step_ratio = r["step_chance_mae"] / max(r["step_mae"], 1e-8)
            print(f"\n  Layer {l} ({r['n_samples']} samples):")
            print(f"    App ID accuracy:    {r['app_accuracy']:.3f}  "
                  f"(chance={r['app_chance']:.3f}, ratio={app_ratio:.1f}x)")
            print(f"    Step MAE:           {r['step_mae']:.2f}  "
                  f"(chance={r['step_chance_mae']:.2f}, ratio={step_ratio:.1f}x)")
        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
