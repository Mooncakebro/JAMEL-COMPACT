from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List
from uuid import uuid4

from jamel.config.settings import get_settings
from jamel.core.env.web import get_environment, stop_envrionment
from jamel.core.policy.agent import PolicyAgent
from jamel.core.policy.hooks import HookContext, HookEvent
from jamel.log import log_utils

logger = log_utils.get_logger(__name__)


@dataclass
class RolloutStepRecord:
    iteration: int
    rank: int
    trajectory_id: str
    step: int
    observation_str: str
    next_observation_str: str
    reward: float
    terminated: bool
    truncated: bool
    memory_content: Any
    raw_content: str
    parsed_content: dict
    rollout_data: dict


class OnlineRolloutCollector:
    def __init__(self, iteration: int, rank: int, trajectory_id: str):
        self.iteration = iteration
        self.rank = rank
        self.trajectory_id = trajectory_id
        self.records: List[RolloutStepRecord] = []

    def _after_reward(self, context: HookContext):
        decision = context.data.decision
        rollout_data = getattr(decision, "rollout_data", {"chunks": []}) or {"chunks": []}
        self.records.append(
            RolloutStepRecord(
                iteration=self.iteration,
                rank=self.rank,
                trajectory_id=self.trajectory_id,
                step=context.agent.current_step,
                observation_str=context.data.observation,
                next_observation_str=context.data.new_observation,
                reward=float(context.data.reward or 0.0),
                terminated=bool(context.data.terminated),
                truncated=bool(context.data.truncated),
                memory_content=decision.memory_content,
                raw_content=decision.raw_content,
                parsed_content=decision.parsed_content,
                rollout_data=rollout_data,
            )
        )
        return context

    def attach(self, agent: PolicyAgent) -> None:
        agent.register_hook(
            HookEvent.AFTER_REWARD,
            self._after_reward,
            name=f"online_rollout_collector_{self.trajectory_id}",
        )


@dataclass
class TrajectoryResult:
    trajectory_id: str
    history_path: str | None
    step_records: List[RolloutStepRecord]
    success: bool
    steps_completed: int
    error: str | None


def collect_trajectory(
    model,
    iteration: int,
    rank: int,
    trajectory_index: int,
) -> TrajectoryResult:
    settings = get_settings()
    if not settings.start_url:
        raise ValueError("`start_url` must be configured for the fsdp_online pipeline.")

    trajectory_id = (
        f"iter_{iteration:04d}_rank_{rank:02d}_traj_{trajectory_index:03d}_{uuid4().hex[:8]}"
    )
    env_context = get_environment(
        start_url=settings.start_url,
        headless=settings.headless_mode,
        record_coverage=settings.record_coverage,
        timeout=settings.browser_timeout,
    )
    agent = PolicyAgent(
        model,
        env=env_context.env,
        max_steps=settings.max_steps_per_trajectory,
    )
    agent.start_url = settings.start_url

    collector = OnlineRolloutCollector(
        iteration=iteration,
        rank=rank,
        trajectory_id=trajectory_id,
    )
    collector.attach(agent)

    try:
        result = agent.run(env_context.obs, env_context.info, settings.goal)
        return TrajectoryResult(
            trajectory_id=trajectory_id,
            history_path=agent.saved_file_path,
            step_records=collector.records,
            success=bool(result.success),
            steps_completed=int(result.steps_completed),
            error=result.error,
        )
    finally:
        coverage_path = None
        if settings.record_coverage:
            coverage_path = (
                Path(settings.history_data_dir)
                / ".."
                / agent.extra_info_dirname
                / "coverage.json"
            )
        stop_envrionment(
            env_context,
            record_coverage=settings.record_coverage,
            record_coverage_path=coverage_path,
        )


def save_rollout_records(
    records: List[RolloutStepRecord],
    iteration: int,
    rank: int,
    output_dir: str | Path,
) -> Path:
    base_dir = Path(output_dir) / f"iteration_{iteration:04d}" / f"rank_{rank:02d}"
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / "rollout.jsonl"
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.info(
        "saved_online_rollout_records",
        output_path=str(output_path),
        record_count=len(records),
    )
    return output_path
