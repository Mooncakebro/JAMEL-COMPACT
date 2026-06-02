from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

from jamel.config.settings import get_settings
from jamel.log import log_utils
from .model import FSDPTextPolicyModel
from .rollout import collect_trajectory, save_rollout_records
from .trainer import aggregate_pg_stats, run_policy_gradient_update, save_fsdp_checkpoint

logger = log_utils.get_logger(__name__)


class FSDPOnlinePolicyGradientPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        self.device = self._init_distributed()
        self.tokenizer = None
        self.model = None
        self.rollout_model = None
        self.optimizer = None

        logger.info(
            "fsdp_online_pipeline_initialized",
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            model=self.settings.model,
            start_url=self.settings.start_url,
        )

    def _init_distributed(self) -> torch.device:
        if self.world_size > 1 and not torch.cuda.is_available():
            raise RuntimeError("Single-node multi-GPU FSDP requires CUDA devices.")

        if dist.is_available() and not dist.is_initialized() and self.world_size > 1:
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def _resolve_torch_dtype(self):
        mode = (self.settings.fsdp_mixed_precision or "").lower()
        if mode == "bf16" and torch.cuda.is_available():
            return torch.bfloat16
        if mode == "fp16" and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _resolve_mixed_precision(self):
        dtype = self._resolve_torch_dtype()
        if dtype == torch.float32:
            return None
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def _infer_transformer_layer_cls(self, model):
        candidates = [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(getattr(model, "transformer", None), "h", None),
            getattr(model, "layers", None),
        ]
        for layers in candidates:
            if layers is not None and len(layers) > 0:
                return {layers[0].__class__}
        return None

    def _build_model(self):
        if not self.settings.model:
            raise ValueError("`model` must be configured for the fsdp_online pipeline.")
        if not self.settings.start_url:
            raise ValueError("`start_url` must be configured for the fsdp_online pipeline.")

        dtype = self._resolve_torch_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model,
            trust_remote_code=True,
        )
        if self.settings.jinja_template_path:
            self.tokenizer.chat_template = Path(self.settings.jinja_template_path).read_text()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if self.settings.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False

        if self.world_size > 1:
            transformer_layer_cls = self._infer_transformer_layer_cls(base_model)
            auto_wrap_policy = None
            if transformer_layer_cls is not None:
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_layer_cls,
                )
            self.model = FSDP(
                base_model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=self._resolve_mixed_precision(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=self.device,
                use_orig_params=True,
            )
        else:
            self.model = base_model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.settings.pg_learning_rate,
            weight_decay=self.settings.pg_weight_decay,
        )
        self.rollout_model = FSDPTextPolicyModel(
            model_path=str(self.settings.model),
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def _rollout_data_dir(self) -> Path:
        return Path(self.settings.online_pg_data_dir)

    def _checkpoint_root_dir(self) -> Path:
        if self.settings.online_pg_output_dir:
            return Path(self.settings.online_pg_output_dir)
        if self.settings.output_base_dir:
            return Path(self.settings.output_base_dir) / "fsdp_online"
        return Path("models/fsdp_online")

    def _all_ranks_have_chunks(self, local_records) -> bool:
        local_chunk_count = sum(
            len((record.rollout_data or {}).get("chunks", [])) for record in local_records
        )
        if not dist.is_available() or not dist.is_initialized():
            return local_chunk_count > 0
        chunk_tensor = torch.tensor(local_chunk_count, dtype=torch.long, device=self.device)
        dist.all_reduce(chunk_tensor, op=dist.ReduceOp.MIN)
        return int(chunk_tensor.item()) > 0

    def _log_iteration_stats(self, iteration: int, stats) -> None:
        logger.info(
            "fsdp_online_pg_iteration",
            iteration=iteration,
            rank=self.rank,
            world_size=self.world_size,
            loss=stats.loss,
            mean_step_reward=stats.mean_step_reward,
            total_step_reward=stats.total_step_reward,
            num_steps=stats.num_steps,
            num_chunks=stats.num_chunks,
            num_completion_tokens=stats.num_completion_tokens,
        )

    def run(self):
        self._build_model()
        checkpoint_root_dir = self._checkpoint_root_dir()
        rollout_root_dir = self._rollout_data_dir()

        for iteration in range(
            self.settings.start_iteration_step,
            self.settings.online_pg_iterations,
        ):
            local_records = []
            for trajectory_index in range(self.settings.trajectories_per_rank_per_iteration):
                trajectory_result = collect_trajectory(
                    self.rollout_model,
                    iteration=iteration,
                    rank=self.rank,
                    trajectory_index=trajectory_index,
                )
                local_records.extend(trajectory_result.step_records)

            if self.settings.save_rollout_data:
                save_rollout_records(
                    local_records,
                    iteration=iteration,
                    rank=self.rank,
                    output_dir=rollout_root_dir,
                )

            if not self._all_ranks_have_chunks(local_records):
                logger.warning(
                    "skip_pg_update_due_to_empty_chunks",
                    iteration=iteration,
                    rank=self.rank,
                    local_step_count=len(local_records),
                )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                continue

            local_stats = run_policy_gradient_update(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                rollout_records=local_records,
                device=self.device,
                micro_batch_size=self.settings.pg_micro_batch_size,
                max_grad_norm=self.settings.pg_max_grad_norm,
                pg_epochs=self.settings.pg_epochs,
                rollout_temperature=self.settings.rollout_temperature,
                rollout_top_p=self.settings.rollout_top_p,
            )
            stats = aggregate_pg_stats(local_stats, device=self.device)
            if self.rank == 0:
                self._log_iteration_stats(iteration, stats)
            save_fsdp_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                output_dir=checkpoint_root_dir / f"iteration_{iteration:04d}",
                rank=self.rank,
                extra_state={
                    "iteration": iteration,
                    "stats": stats.__dict__,
                    "settings": self.settings.model_dump(),
                }
                if self.rank == 0
                else None,
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def close(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    pipeline = FSDPOnlinePolicyGradientPipeline()
    try:
        pipeline.run()
    finally:
        pipeline.close()
