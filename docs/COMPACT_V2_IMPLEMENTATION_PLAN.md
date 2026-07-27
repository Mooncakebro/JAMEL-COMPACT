# JAMEL-COMPACT v2 — Implementation Plan

This document is the work order for the coding agent. It lists **Phase 1 (correctness
fixes, method unchanged)** and **Phase 2 (method upgrades)** as independently testable
tasks with exact file references and acceptance criteria.

The companion method description (math + figures) is in
[COMPACT_V2_METHOD.md](COMPACT_V2_METHOD.md).

## Background: symptoms to fix

1. `train/loss_uncert` never decreases.
2. `train/loss_action` stays high; eval coverage does not improve over baseline.

Root causes (verified in code — do not skip the diagnosis):

| # | Root cause | Location |
|---|---|---|
| R1 | Uncertainty target computed in mismatched space: raw hidden state `Z[..., :512]` vs memory space; forward uses learned `obs_down` space | `jamel_compact/loss.py:84-98` vs `jamel_compact/model.py:184-186` |
| R2 | Confidence `C` is a pinned recursion (`C0=0.5`, `α=0.1`); with `chunk_size=1` it lives in `[0.35, 0.53]` and carries almost no gradient | `model.py:161,187`, `config.py:59` |
| R3 | Eval generation bypasses memory: injected hidden states are discarded, tokens come from base `llm.generate()` on the raw prompt | `model.py:762-836` |
| R4 | Train truncates from the right (can cut the *response*); eval truncates from the left | `data.py:225-227` vs `eval.py:239-242` |
| R5 | Injection branch not zero-initialized; random memory perturbs pretrained hidden states from step 0 (`w_inj` up to 0.8) | `model.py:108,143-148,196` |
| R6 | Observation = unmasked mean over **all** positions (padding included) | `model.py:198-205` |
| R7 | Innovation cross-attention has a single KV token ⇒ `ΔM` identical across all 16 slots (rank-1 write) | `model.py:169-173` |
| R8 | Default `chunk_size=1`: memory never evolves during training; eval uses 50-step evolved memory (distribution shift) | `config.py:59`, `train.py:289-341` |
| R9 | Wrapper bypasses the pretrained final norm and adds an extra per-layer injection norm, so zero injection is not base-model-equivalent | `model.py` manual decoder/LM-head path |
| R10 | Training feeds the current target action instead of $a_{t-1}$ and orders reset-local episodes by `step_idx` | `data.py`, `SessionChunkDataset` |
| R11 | `e` is already an MSE, but NLL uses `e**2 / R`, making the residual term quartic | `model.py` `SideMemoryModule.correct()` |
| R12 | Observation pooling sees teacher-forced response tokens during training, which are absent at inference | `data.py`, `model.py` pooling mask |

## Ground rules

- Make minimal, reviewable diffs. One task per commit.
- After each task, run its acceptance check. Do not proceed to Phase 2 until all
  Phase 1 checks pass.
- Do not change the baseline (`baseline_train.py`, `baseline_eval.py`) — it is the
  reference. Compare against it.
- Keep `CompactConfig` backward compatible where possible; new fields get defaults
  that reproduce v1 behavior unless the task says otherwise.
- Checkpoint compatibility: tasks U1/U2 change `side_memories.pt` keys. Old
  checkpoints are not loadable after U1/U2 — retrain from base. Note this in the
  checkpoint's `compact_config.json` by bumping a new `model_version: 2` field.

---

## Phase 1 — Correctness fixes (method unchanged)

### F1 — Well-posed uncertainty target (fixes R1, partial R2)

**Change**:
- `model.py` `SideMemoryModule.correct()`: also return the per-token `match`
  (already computed at `model.py:184-186`).
- `model.py` `JAMELCompactWrapper.forward()`: collect `matches` (detached) per layer
  into the result dict.
- `loss.py` `compute_compact_loss()`: replace the mean-pool + `min_dim` truncation
  block with per-token `F.mse_loss(c_new, match.detach())`, averaged over layers.
  Remove `predicted_memory`/`observation_feat` plumbing.

**Acceptance**: on a 200-step overfit run (small subset), `train/loss_uncert`
decreases measurably; TensorBoard shows per-token match values spread in `[0,1]`,
not a single constant.

### F2 — Train/eval truncation consistency (fixes R4)

**Change** (`data.py`, `__getitem__`, currently lines 224-235):
- Tokenize prompt and full text separately (already done). When
  `len(full_ids) > max_length`, left-truncate the **prompt** to
  `max_length - response_len` and keep the full response (same side as
  `eval.py:239-242`).
- If the response alone does not fit, or the surviving prompt would be shorter than
  64 tokens, drop the sample (return `None` and filter in `collate_fn`; count drops
  and log the count).
- Build labels **after** truncation; assert at least one non-`-100` label per sample.

**Acceptance**: unit test with a synthetic overlong row: supervised tokens are
response tokens, never prompt tokens; zero all-`-100` samples in a full pass over
`data/compact_sft_data_all/compact_train.parquet`.

### F3 — Masked observation pooling (fixes R6)

**Change**:
- Thread a dedicated observation mask through training and generation. It masks
  padding and, during teacher forcing, masks assistant response tokens.
- Phase-1 mean pooling uses this mask; U1 replaces the mean with latent-query
  attention while preserving identical mask semantics.

**Acceptance**: for the same unpadded sample, `z` computed alone vs inside a padded
batch differs by < 1e-5; changing response tokens does not change pooled $Z_t$.

### F4 — Zero-initialized injection (fixes R5)

**Change** (`model.py` `SideMemoryModule.__init__` / `inject`):
- `nn.init.zeros_(self.delta_up.weight); nn.init.zeros_(self.delta_up.bias)`.
- Replace the fixed `inject_weight` float with a small nonzero learnable gate
  (default `0.1`); inject uses
  `h + w_max * tanh(self.inject_gate) * delta_up` where `w_max` keeps the current
  hierarchical values (0.8/0.5/0.3).
- Remove/bypass the extra `inject_norm`; zero injection must return `h` exactly.
- Apply the pretrained decoder's final norm before the LM head in both forward
  and generation paths.

Zeroing both `delta_up` and the gate is forbidden: it gives both factors zero
gradient and permanently disconnects memory from action CE.

**Acceptance**: script `scripts/check_zero_init.py` (new): load base model and
wrapper (fresh side memory), assert wrapper logits == base logits (`allclose`,
rtol 1e-4, atol 1e-4) on a real batch. After this fix, compact action CE should be
≤ baseline CE at equal steps.

### F5 — Memory-conditioned generation (fixes R3)

**Change** (`model.py` `generate()`):
- During the memory-update prompt pass, call each decoder layer with
  `use_cache=True` and populate a `transformers.cache_utils.DynamicCache`.
- Preserve processor-provided `mm_token_type_ids` and use the base Qwen3-VL
  `compute_3d_position_ids`/`get_rope_index` path for prompt and incremental
  generation M-RoPE; do not substitute plain 1D positions for image prompts.
- Use the memory-injected prompt hidden state for the first-token logits, then
  run the pretrained decoder one token at a time against the populated cache.
  This is the current cache-backed implementation of memory-conditioned
  continuation; verify position IDs / RoPE handling for Qwen3-VL (M-RoPE) matches
  the base model's own prefill.
- Gate the cache-backed path behind `config.memory_conditioned_generate`
  (default True). Setting it to `False` remains an explicit raw-base-generation
  ablation; a wrapper-forward fallback for incompatible Transformers versions is
  not silently substituted because that would change the evaluation contract.
- Add eval flag `--freeze-memory-init` (eval.py): never write `new_memory` back
  into the session state. Used for ablation.

**Acceptance**: first-position logits from the cache-prefill path match the
wrapper's forward logits (rtol 1e-3). Ablation: eval with `--freeze-memory-init`
produces materially different action sequences than normal eval (if not, memory is
still not influencing outputs — investigate).

### F6 — Session-chunked training by default (fixes R8, completes R2)

**Change**:
- `config.py`: `chunk_size: int = 8`.
- `shell/run_compact_train.sh`: pass through `CHUNK_SIZE=${CHUNK_SIZE:-8}`.
- Verify `SessionChunkDataset` groups by `session_id` and orders by
  continuous `session_step_idx` (not reset-local `step_idx`)
  (fix if not).
- Build action conditioning from the preceding session action $a_{t-1}$, with
  `noop()` at session start; mask action-token padding during pooling.
- Split train/validation by complete `session_id`, never by individual rows.
- Keep TBPTT-1 detach (`train.py:161-176`) for now; add `config.tbptt_detach`
  (default True) so it can be ablated.

**Acceptance**: TensorBoard `memory/*` stats differ between chunk step 0 and step 7
(memory actually evolves); every chunk has consecutive `session_step_idx`; train
and validation session sets are disjoint; loss curves stable.

### F7 — Coverage-weighted SFT (uses the novelty signal already in the data)

**Change**:
- `config.py`: `coverage_weight_eta: float = 0.0` (0 = off).
- `data.py`: return `sample_weight = 1 + eta * max(coverage_delta_score, 0)`
  (column exists in the parquet).
- `loss.py`: `F.cross_entropy(..., reduction='none')`, per-sample mean, then
  weighted mean over batch. Log weighted and unweighted CE separately.

**Acceptance**: with `eta=1.0`, high-novelty samples dominate the logged weighted
loss; training runs stable.

### F8 — Diagnostics

**Change**:
- Throttled TensorBoard histograms (every `log_steps` optimizer steps, cheap):
  per-layer `K`, `P`, `R`, surprise `e`, and `||M||`.
- New `scripts/probe_memory.py`: linear probe on eval-time memory snapshots —
  predict `app id`, `step_idx` from `M_t` per layer; report accuracy per layer.
- This task has no model changes.

**Acceptance**: probe accuracy significantly above chance at mid/deep layers after
a short training run (otherwise memory encodes nothing — flag before Phase 2).

---

## Phase 2 — Method upgrades (v2)

See [COMPACT_V2_METHOD.md](COMPACT_V2_METHOD.md) for the full math. Order matters:
U1 → U2 → U3. U4 items are independent experiments, one at a time.

### U1 — Multi-token observation (fixes R7)

**Change** (`SideMemoryModule`):
- Add `self.obs_queries = nn.Parameter(torch.randn(k, d_mem) * 0.02)` with `k = 4`
  (config `num_obs_tokens`).
- Project first: `h_down = obs_down(h)` → `[B, N, d_mem]`; then
  `Z = MultiheadAttention(Q=obs_queries.expand(B,-1,-1), KV=h_down,
  key_padding_mask=~observation_mask)` → `[B, k, d_mem]`.
- This pre-projection is the required low-overhead implementation; attention must
  not operate in full hidden width `d`.
- `mem_cross_attn` now gets `k` KV tokens (was 1) — innovation is no longer
  rank-1.
- U2's scalar Kalman gain consumes the resulting per-slot innovation; the old
  learned `k_gate` path is removed with the confidence recursion.

**Acceptance**: `ΔM` variance across slots > 0; F3's mask test still passes;
checkpoint `model_version: 2`.

### U2 — Learned Kalman track (replaces confidence; fixes R2 fully)

**Change** (`SideMemoryModule.predict/correct`, `loss.py`, `config.py`):
- Replace `C` with per-slot variance `P ∈ R+^{B×N_m}` (init 0.5). Rename
  `confidence_states` → `variance_states` through wrapper/train/eval.
- Predict: `P_hat = lambda_l * P + Q_theta(a_down) + gamma_e * e_prev.detach()`
  - `lambda_l`: learnable per-slot vector, `sigmoid`-constrained, initialized to
    the hierarchical values (0.70/0.85/0.95).
  - `Q_theta`: `Linear(d_mem → N_m)` + `softplus`, input the layer's action
    embedding.
  - `e_prev`: previous step's observation-prediction error (per sample, detached);
    0 at chunk start. `gamma_e = 1.0` (config). Clamp only this recurrent
    inflation input to `surprise_clip` for stability.
- Correct: `R = softplus(MLP(mean-pooled z_down → N_m)) + r_min`
  (config-hidden 128; `r_min=0.01` prevents variance collapse);
  `K = P_hat / (P_hat + R)` → `[B, N_m, 1]`; `M = M_hat + K * ΔM`;
  `P = (1 - K) * P_hat`.
- Loss: replace `loss_uncert` with
  `L_nll = 0.5 * (log R + e / R).mean()` in float32 (e from U3 is already a
  squared residual and is not squared or clipped again; until U3 lands use
  placeholder `e = ||mean-pool(m_hat) − pooled z_down||²`, detached).
  Remove the Bernoulli-entropy term; keep `loss_mem_l2`. Keep `lambda_uncert`
  as the weight (rename to `lambda_nll`, default 0.01).

**Acceptance**: `train/loss_nll` decreases; logged `K` distribution spreads over
`[0,1]` (not pinned in `[0.475, 0.65]`); synthetic unit test: feeding a large
`e_prev` raises `K` on the next step.

### U3 — Observation model + surprise (feeds U2)

**Change** (`SideMemoryModule`):
- Add `self.obs_model = MLP(d_mem → d_mem → d_mem)` (GELU).
- `z_pred = obs_model(mean-pool(m_hat))`; target `pooled z_down.detach()`.
- `L_obs = MSE(z_pred, target)` per layer, weight `lambda_obs = 0.01` (config).
- `e = per-sample MSE.detach()` → carried to the next step for U2's
  `gamma_e * e_prev` term (plumb through wrapper result dict, train loop,
  `CompactAgent` session state).
- Later variant (flag `obs_loss: mse|infonce`, default `mse`): InfoNCE against
  other samples' `z` in the batch.

**Acceptance**: `train/loss_obs` decreases; probe (F8) accuracy improves over
Phase-1 model at equal steps.

### U4 — Independent later experiments (one at a time, behind config flags)

- **U4a — Shared memory across layers** (`shared_memory: bool = False`): one
  memory state; each layer keeps its own projections/gates but reads/writes the
  shared state. Expect fewer params, better cross-layer consistency.
- **U4b — Two-timescale memory** (`slow_memory: bool = False`): add slow track
  `M̄_t = (1−η)M̄_{t−1} + η M_t` (η=0.05), injected alongside `M_t` in deep layers.
- **U4c — Sparse top-k writes** (`sparse_write_k: int = 0`): per step update only
  the `k` slots with highest `K` (straight-through estimator).

**Acceptance (each)**: coverage on `test10` ≥ the U3 model at equal eval budget;
otherwise revert the flag.

---

## Phase 3 — Novelty-driven RL (after U3 stabilizes)

1. **RFT** (cheapest first): run eval sessions with the U3 model, keep episodes
   with coverage gain above a threshold, fine-tune on them (echoes original
   JAMEL's rejection fine-tuning).
2. **GRPO** via `third_party/verl-agent` with coverage-delta as reward. Memory
   state must be part of the rollout state (init per session, carried per step).

---

## Global regression criteria

Run after each phase:

1. Action CE: compact ≤ `baseline_train.py` CE at equal optimizer steps (after F4
   this should already hold).
2. Coverage on `test10` (3 sessions × 50 steps): Phase-1 model ≥ baseline;
   U3 model > Phase-1 model.
3. Ablation table (report all): full model / `--freeze-memory-init` /
   `tbptt_detach=False` / `coverage_weight_eta=0`.

## Suggested task order

```
F2, F3, F4 (independent, small) → F1 → F8 → F5 → F6 → F7
→ U1 → U2 → U3 → regression + ablations → U4a/b/c → Phase 3
```
