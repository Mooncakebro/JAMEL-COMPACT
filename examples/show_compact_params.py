"""Print detailed parameter breakdown for JAMEL-COMPACT."""
import sys
sys.path.insert(0, '.')

from jamel_compact.model import SideMemoryModule
from jamel_compact.config import CompactConfig

print("=" * 80)
print("JAMEL-COMPACT: Detailed Parameter Breakdown")
print("=" * 80)

for model_name, d, L, base_b in [
    ("Qwen3-VL-2B", 1536, 28, 2.0),
    ("Qwen3-VL-8B", 4096, 36, 8.0),
]:
    d_mem = 512
    N_m = 16
    n_heads = 8

    print("\n" + "=" * 80)
    print("Base model: {}  (d={}, L={} layers)".format(model_name, d, L))
    print("Memory dim: d_mem={}, N_m={} tokens, heads={}".format(d_mem, N_m, n_heads))
    print("=" * 80)

    cfg = CompactConfig(base_model_name=model_name, mem_dim=d_mem,
                        num_mem_tokens=N_m, num_heads=n_heads)
    sm = SideMemoryModule(0, L, hidden_dim=d, mem_dim=d_mem,
                          num_mem=N_m, num_heads=n_heads, config=cfg)

    print("\n--- Per-layer SideMemoryModule breakdown ---")
    total = 0
    for name, param in sm.named_parameters():
        n = param.numel()
        total += n
        shape_str = str(tuple(param.shape))
        print("  {:<40s} {:<20s} {:>8.1f}K".format(name, shape_str, n / 1e3))
    print("  " + "-" * 70)
    print("  {:<40s} {:<20s} {:>8.2f}M".format("TOTAL per layer", "", total / 1e6))

    action_embed_params = d * d + d
    total_new = total * L + action_embed_params
    base_params = base_b * 1e9

    print("\n--- Summary ---")
    print("  Per layer:    {:.2f}M".format(total / 1e6))
    print("  All layers:   {:.1f}M  ({} layers)".format(total * L / 1e6, L))
    print("  Action embed: {:.2f}M".format(action_embed_params / 1e6))
    print("  Total new:    {:.1f}M".format(total_new / 1e6))
    print("  Base model:   ~{:.1f}B".format(base_params / 1e9))
    print("  Overhead:     {:.1f}%".format(total_new / base_params * 100))

    print("\n--- Module architecture details ---")
    print("  FiLMGRUCell (d_mem={}):".format(d_mem))
    print("    gru:         GRUCell({}, {})".format(d_mem, d_mem))
    print("      weight_ih: [{}, {}]  = {:.0f}K".format(3*d_mem, d_mem, 3*d_mem*d_mem/1e3))
    print("      weight_hh: [{}, {}]  = {:.0f}K".format(3*d_mem, d_mem, 3*d_mem*d_mem/1e3))
    print("      bias_ih:   [{}]           = {:.0f}K".format(3*d_mem, 3*d_mem/1e3))
    print("      bias_hh:   [{}]           = {:.0f}K".format(3*d_mem, 3*d_mem/1e3))
    print("    film_mlp:    Linear({}, {}) + Tanh".format(d_mem, 2*d_mem))
    print("      weight:    [{}, {}]  = {:.0f}K".format(2*d_mem, d_mem, 2*d_mem*d_mem/1e3))
    print("      bias:      [{}]           = {:.0f}K".format(2*d_mem, 2*d_mem/1e3))
    print("    action_proj: Linear({}, {})".format(d_mem, d_mem))
    print("      weight:    [{}, {}]  = {:.0f}K".format(d_mem, d_mem, d_mem*d_mem/1e3))
    print("      bias:      [{}]           = {:.0f}K".format(d_mem, d_mem/1e3))

    print("\n  Projections (d={} <-> d_mem={}):".format(d, d_mem))
    for pname, pshape in [
        ("obs_down",    (d_mem, d)),
        ("action_down", (d_mem, d)),
        ("h_down",      (d_mem, d)),
        ("delta_up",    (d, d_mem)),
    ]:
        w = pshape[0] * pshape[1]
        b = pshape[0]
        print("    {:<14s} Linear({} -> {})  weight={}K  bias={}K".format(
            pname, d, d_mem, w/1e3, b/1e3))

    print("\n  Cross-Attention (nn.MultiheadAttention, dim={}, heads={}):".format(d_mem, n_heads))
    print("    mem_cross_attn (innovation):")
    print("      in_proj_weight:  [{}, {}] = {:.0f}K".format(3*d_mem, d_mem, 3*d_mem*d_mem/1e3))
    print("      in_proj_bias:    [{}]        = {:.0f}K".format(3*d_mem, 3*d_mem/1e3))
    print("      out_proj.weight: [{}, {}] = {:.0f}K".format(d_mem, d_mem, d_mem*d_mem/1e3))
    print("      out_proj.bias:   [{}]        = {:.0f}K".format(d_mem, d_mem/1e3))
    print("    inject_cross_attn (injection):")
    print("      (same shape as mem_cross_attn)")

    print("\n  Kalman Gate:")
    print("    k_gate: Linear({}, {}) + Sigmoid".format(2*d_mem, d_mem))
    print("      weight: [{}, {}] = {:.0f}K".format(d_mem, 2*d_mem, 2*d_mem*d_mem/1e3))
    print("      bias:   [{}]        = {:.0f}K".format(d_mem, d_mem/1e3))

    print("\n  Innovation Proj:")
    print("    innovation_proj: Linear({}, {})".format(d_mem, d_mem))
    print("      weight: [{}, {}] = {:.0f}K".format(d_mem, d_mem, d_mem*d_mem/1e3))
    print("      bias:   [{}]        = {:.0f}K".format(d_mem, d_mem/1e3))

    print("\n  LayerNorm:")
    print("    inject_norm: LayerNorm({}) = {:.0f}K".format(d, 2*d/1e3))

    print("\n  Learnable init:")
    print("    init_memory: [{}, {}] = {:.0f}K".format(N_m, d_mem, N_m*d_mem/1e3))

    print("\n  Global (not per-layer):")
    print("    action_embed: Linear({}, {}) = {:.0f}K".format(d, d, (d*d+d)/1e3))

print("\n" + "=" * 80)
print("DONE")