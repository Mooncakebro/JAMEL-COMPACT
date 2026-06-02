"""
Policy Agent - 基于 Gym 环境的智能体
和环境解耦，不管是什么环境，都用同一个 PolicyAgent 框架代码。
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import traceback
from typing import Dict, Any, List, Optional, Callable
import uuid
import numpy as np
import gymnasium as gym
import os
from PIL import Image
from jamel.core.env.web import Observer # 之后挪到一个 get_env 的部分，这里只保留一个通用框架。
from jamel.core.env.web.action_space import get_action_space
from jamel.core.env.web.utils import StepHistory
from jamel.core.memory.base import MemoryBase
from jamel.core.memory.types import get_memory_cls
from jamel.core.policy.brains.base import BrainBase
from jamel.core.policy.brains.types import get_brain_cls
from jamel.core.reward.web.reward_funcs import jamel_reward_fn_web_coverage as reward_fn

from jamel.models.service.base import ModelInterface
from jamel.config import get_settings
from jamel.utils.general_utils import summarize_str

from jamel.log import log_utils
from jamel.core.policy.hooks import HookData, HookManager, HookEvent, HookContext, Hook
logger = log_utils.get_logger(__name__)

@dataclass
class AgentResult:
    success: bool
    trace_path: str
    steps_completed: int
    data: HookData
    error: str = None

    def to_json(self):
        return {
            "success": self.success,
            "trace_path": self.trace_path,
            "steps_completed": self.steps_completed,
            "error ": self.error,
        }

class PolicyAgent:

    def __init__(
        self,
        model: ModelInterface,
        env: gym.Env,
        max_steps: int = 50,
        save_step_coverage_fn: Optional[Callable[[Path, int], None]] = None,
        frozen_global_coverage_paths: Optional[List[str]] = None,
        run_output_dir: Optional[os.PathLike | str] = None,
        start_url: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.settings = get_settings()
        self.max_steps = max_steps
        self.env = env
        self.save_step_coverage_fn = save_step_coverage_fn
        self.frozen_global_coverage_paths = list(frozen_global_coverage_paths or [])
        self.run_output_dir = Path(run_output_dir) if run_output_dir is not None else None
        self.default_start_url = start_url
        self.run_metadata = dict(run_metadata or {})
        # 组件初始化
        self.brain: BrainBase = get_brain_cls(brain_type=self.settings.brain_type)(model, get_action_space=get_action_space)
        self.memory: MemoryBase = get_memory_cls(memory_type=self.settings.memory_type)(self)

        # Hook 系统
        self.hooks = HookManager()

        # 状态管理
        self.current_step = 0
        self.history: List[StepHistory] = []
        self.save_filename = None
        self.saved_file_path = None
        self.goal = None
        self.start_url = None
        self.coverage_dir = None

    def register_hook(
        self,
        event: HookEvent,
        callback,
        priority: int = 0,
        name: Optional[str] = None
    ) -> Hook:
        """
        注册 hook

        Args:
            event: 事件类型
            callback: 回调函数
            priority: 优先级
            name: Hook 名称

        Returns:
            Hook 实例
        """
        return self.hooks.register(event, callback, priority, name)

    def unregister_hook(self, hook: Hook) -> bool:
        """注销 hook"""
        return self.hooks.unregister(hook)

    def reset(self, goal: str):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id = uuid.uuid4().hex[:8]
        self.current_step = 0
        self.history = []

        self.info_dir = self.run_output_dir or (Path(self.settings.history_data_base_dir) / self.settings.exp_name)
        self.extra_info_dir = self.info_dir / f"extra/{self.timestamp}_{self.run_id}"
        self.history_data_dir = self.info_dir / "histories"
        
        self.save_filename = f"agent_history_{self.timestamp}_{self.run_id}.parquet"
        self.saved_file_path = None
        self.goal = goal
        self.start_url = self.default_start_url
        self.coverage_dir = self.extra_info_dir / "coverage"
        self.memory.reset()
        logger.info("Agent state reset", frozen_global_coverage_count=len(self.frozen_global_coverage_paths))

    def run(
        self,
        obs: dict, 
        info: dict, 
        goal: str
    ) -> AgentResult:
        result = None
        try:
            self.reset(goal)
            result = self._main_loop(obs, info, goal)

            self.saved_file_path = self._save(result)

            return result

        except Exception as e:
            traceback.print_exc()
            logger.error("Agent 运行出错", error=str(e), exc_info=True)

            # 即使出错也保存历史记录
            if result is None:
                result = AgentResult(
                    success=False,
                    error=str(e),
                    trace_path=self.saved_file_path,
                    data=None,
                    steps_completed=self.current_step
                )
            self.saved_file_path = self._save(result)
            return result
        finally:
            # 清理资源
            self._cleanup()


    def _main_loop(self, obs: dict, info: dict, goal: str) -> AgentResult:
        """主执行循环"""
        policy_agent_data = HookData(
            before_obs=None,
            before_info=None,
            before_observation=None,
            after_obs=obs,
            after_info=info,
            after_observation=Observer.get_observation(obs),
            cummulated_reward=0,
        )
        while self.current_step < self.max_steps:
            policy_agent_data.before_obs = policy_agent_data.after_obs
            policy_agent_data.before_info = policy_agent_data.after_info
            policy_agent_data.before_observation = Observer.get_observation(policy_agent_data.before_obs)
            policy_agent_data.after_obs = None
            policy_agent_data.after_info = None
            policy_agent_data.after_observation = None
            policy_agent_data.raw_reward = None
            policy_agent_data.reward = None
            policy_agent_data.terminated = None
            policy_agent_data.truncated = None
            logger.info(f"执行步骤 {self.current_step + 1}/{self.max_steps}")

            # 触发 BEFORE_STEP hook
            context = HookContext(
                event=HookEvent.BEFORE_STEP,
                agent=self,
                data=policy_agent_data
            )
            context = self.hooks.trigger(context)

            # 1. 预处理观察
            logger.info(f"Observation: {summarize_str(500)(policy_agent_data.before_observation)}")
            self._save_step_screenshot(
                obs=policy_agent_data.before_obs,
                step_index=self.current_step + 1,
            )

            # 2. 思考决策
            context = HookContext(
                event=HookEvent.BEFORE_DECIDE_ACTION,
                agent=self,
                data=policy_agent_data
            )
            context = self.hooks.trigger(context)

            policy_agent_data.decision = self.brain.decide_action(
                observation=policy_agent_data.before_observation,
                user_goal=goal,
                history=self.history,
                memory=self.memory
            )

            # 触发 AFTER_DECIDE_ACTION hook
            context = HookContext(event=HookEvent.AFTER_DECIDE_ACTION, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            # 3. 执行行动 - 使用 gym.step()

            # 触发 BEFORE_EXECUTE_ACTION hook
            context = HookContext(event=HookEvent.BEFORE_EXECUTE_ACTION, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            (
                policy_agent_data.after_obs,
                policy_agent_data.raw_reward,
                policy_agent_data.terminated,
                policy_agent_data.truncated,
                policy_agent_data.after_info,
            ) = self.env.step(policy_agent_data.decision.action)
            policy_agent_data.after_observation = Observer.get_observation(policy_agent_data.after_obs)
            self._save_step_screenshot(
                obs=policy_agent_data.after_obs,
                step_index=self.current_step + 1,
                suffix="after",
            )
            current_step_history = self._get_current_step_history(
                policy_agent_data
            ) # 注意，这个 policy_agent_data 在后续还会被修改。
            self._save_step_coverage(step_index=self.current_step + 1)

            # 触发 AFTER_EXECUTE_ACTION hook
            context = HookContext(event=HookEvent.AFTER_EXECUTE_ACTION, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            # 触发 BEFORE_REWARD hook
            context = HookContext(event=HookEvent.BEFORE_REWARD, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            # 触发 AFTER_REWARD hook
            policy_agent_data.reward = reward_fn(
                current_step_history,
                frozen_global_coverage_paths=self.frozen_global_coverage_paths,
                trajectory_history=self.history,
            )
            current_step_history.reward = policy_agent_data.reward

            context = HookContext(event=HookEvent.AFTER_REWARD, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            policy_agent_data.cummulated_reward += policy_agent_data.reward
            logger.info(f"got reward: {policy_agent_data.reward}")

            self._add_to_history(current_step_history=current_step_history)
            logger.info(
                "step finished",
                trajectory_history_len=len(self.history),
                frozen_global_coverage_count=len(self.frozen_global_coverage_paths),
                history_reward=self.history[-1].reward,
            )

            # 5. 检查是否完成
            if policy_agent_data.terminated or policy_agent_data.truncated:
                logger.info("任务完成")
                return AgentResult(
                    success=True,
                    steps_completed=self.current_step + 1,
                    trace_path=self.saved_file_path,
                    data=policy_agent_data
                )

            self.saved_file_path = self._save(AgentResult(
                success=True,
                steps_completed=self.current_step,
                trace_path=self.saved_file_path,
                data=policy_agent_data
            ))

            # 触发 BEFORE_MEMORY_UPDATE hook
            context = HookContext(event=HookEvent.BEFORE_MEMORY_UPDATE, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            self.memory.update_memory()

            # 触发 AFTER_MEMORY_UPDATE hook
            context = HookContext(event=HookEvent.AFTER_MEMORY_UPDATE, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            # 触发 AFTER_STEP hook
            context = HookContext(event=HookEvent.AFTER_STEP, agent=self, data=policy_agent_data)
            context = self.hooks.trigger(context)

            self.current_step += 1

        # 达到最大步数
        logger.warning(f"达到最大步数限制 {self.max_steps}")

        return AgentResult(
            success=False,
            error=f"达到最大步数限制 {self.max_steps}",
            steps_completed=self.current_step + 1,
            trace_path=self.saved_file_path,
            data=policy_agent_data,
        )

    def _get_current_step_history(
        self,
        policy_agent_data: HookData
    ):
        coverage_path = self._get_step_coverage_path(step_index=self.current_step + 1)
        return StepHistory(
            before_obs=policy_agent_data.before_obs,
            after_obs=policy_agent_data.after_obs,
            before_info=policy_agent_data.before_info,
            after_info=policy_agent_data.after_info,
            before_observation=policy_agent_data.before_observation,
            after_observation=policy_agent_data.after_observation,
            step=self.current_step,
            reward=policy_agent_data.reward,
            raw_content=policy_agent_data.decision.raw_content,
            memory_content=policy_agent_data.decision.memory_content,
            parsed_content=policy_agent_data.decision.parsed_content,
            result=None,
            timestamp=self.timestamp,
            extra_fields={
                "coverage_path": str(coverage_path) if coverage_path else None,
                **self.run_metadata,
            },
        )

    def _add_to_history(
        self,
        current_step_history: StepHistory
    ):
        """添加到历史记录"""
        self.history.append(current_step_history)

    def _save(self, result: AgentResult):
        metadata = {
            "start_url": self.start_url,
            "user_goal": self.goal,
            "total_steps": self.current_step + 1,
            "max_steps": self.max_steps,
            "final_result": result.to_json(),
            "model_path": self.model.model_path
        }
        saved_file_path = Observer.save_trajectory(history=self.history, history_dir=os.path.join(os.getcwd(), self.history_data_dir), metadata=metadata, filename=self.save_filename)
        return saved_file_path

    def _save_step_coverage(self, step_index: int):
        if not self.settings.record_coverage:
            logger.info("跳过步骤 coverage 保存：record_coverage 未开启", step_index=step_index)
            return

        if not self.save_step_coverage_fn:
            logger.info("跳过步骤 coverage 保存：环境未提供保存实现", step_index=step_index)
            return

        coverage_path = self._get_step_coverage_path(step_index=step_index)
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("保存步骤 coverage 快照", step_index=step_index, coverage_path=coverage_path)
        self.save_step_coverage_fn(coverage_path, step_index)

    def _get_step_coverage_path(self, step_index: int) -> Optional[Path]:
        if self.coverage_dir is None:
            logger.info("当前运行未初始化 coverage 目录", step_index=step_index)
            return None
        return self.coverage_dir / f"coverage_{step_index}.json"

    def _save_step_screenshot(
        self,
        obs: Optional[dict],
        step_index: int,
        suffix: Optional[str] = None,
    ) -> Optional[Path]:
        if not isinstance(obs, dict):
            logger.warning("跳过截图保存：obs 不是 dict", step_index=step_index, suffix=suffix)
            return None

        screenshot_data = obs.get("screenshot")
        if screenshot_data is None:
            logger.warning("跳过截图保存：obs 中没有 screenshot", step_index=step_index, suffix=suffix)
            return None

        img = Image.fromarray(screenshot_data.astype(np.uint8))
        output_dir = self.extra_info_dir / "images"
        output_dir.mkdir(exist_ok=True, parents=True)

        filename = f"step_{step_index}.png" if not suffix else f"step_{step_index}_{suffix}.png"
        screenshot_path = output_dir / filename
        img.save(screenshot_path, format='PNG')
        return screenshot_path

    def _cleanup(self):
        """清理资源"""
        pass
