from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
import multiprocessing
from pathlib import Path
import queue
import shutil
import time

from jamel.config.settings import get_settings
from jamel.core.env.web import get_environment, stop_envrionment
from jamel.core.policy.agent import PolicyAgent
from jamel.core.policy.brains.types import get_brain_prompt_func
from jamel.log import log_utils
from jamel.models.openai_model import OpenAIModel
from jamel.models.service.client import InferenceClient
from jamel.models.service.server import run_model_server
from jamel.pipeline.local.curriculum import CurriculumState
from jamel.train.agent.format_funcs import get_memory_format_func, get_policy_format_func
from jamel.train.agent.main import prepare_explorer_data, run_training
from jamel.train.agent.processor import ProcessorStats, get_processor_func

logger = log_utils.get_logger(__name__)


@dataclass
class TrajectoryTask:
    base_url: str
    api_key: str
    model_checkpoint: str
    target_url: str
    stage_index: int
    iteration_step: int
    iteration_output_dir: str
    frozen_global_coverage_paths: list[str]
    trajectory_index: int
    max_steps_per_trajectory: int
    headless_mode: bool
    record_coverage: bool
    browser_timeout: int


@dataclass
class TrajectoryRunResult:
    target_url: str
    trajectory_index: int
    iteration_step: int
    stage_index: int
    saved_file_path: str | None
    positive_coverage_paths: list[str]
    success: bool
    error: str | None = None


def _build_model_client(base_url: str, api_key: str, model_checkpoint: str) -> OpenAIModel:
    return OpenAIModel(model_name=str(model_checkpoint), base_url=base_url, api_key=api_key)


def _collect_positive_coverage_paths(agent: PolicyAgent | None) -> list[str]:
    if agent is None:
        return []

    positive_paths: list[str] = []
    for step in agent.history:
        if step.reward <= 0:
            continue
        extra_fields = step.extra_fields or {}
        coverage_path = extra_fields.get("coverage_path")
        if not coverage_path:
            continue
        coverage_path = str(Path(coverage_path).resolve())
        if not Path(coverage_path).exists():
            continue
        positive_paths.append(coverage_path)
    return positive_paths


def _run_single_trajectory_task(task: TrajectoryTask) -> TrajectoryRunResult:
    settings = get_settings()
    env_context = None
    agent = None
    goal = ""
    try:
        env_context = get_environment(
            task.target_url,
            headless=task.headless_mode,
            record_coverage=task.record_coverage,
            timeout=task.browser_timeout,
        )
        logger.info(
            "Initializing Agent",
            goal=goal,
            target_url=task.target_url,
            stage_index=task.stage_index,
            iteration_step=task.iteration_step,
            trajectory_index=task.trajectory_index,
        )
        agent = PolicyAgent(
            _build_model_client(task.base_url, task.api_key, task.model_checkpoint),
            env=env_context.env,
            max_steps=task.max_steps_per_trajectory,
            save_step_coverage_fn=env_context.save_step_coverage,
            frozen_global_coverage_paths=task.frozen_global_coverage_paths,
            run_output_dir=task.iteration_output_dir,
            start_url=task.target_url,
            run_metadata={
                "target_url": task.target_url,
                "stage_index": task.stage_index,
                "iteration_step": task.iteration_step,
                "trajectory_index": task.trajectory_index,
                "brain_type": settings.brain_type,
                "memory_type": settings.memory_type,
            },
        )
        logger.info(
            "Agent starting execution",
            target_url=task.target_url,
            stage_index=task.stage_index,
            iteration_step=task.iteration_step,
            trajectory_index=task.trajectory_index,
        )
        result = agent.run(env_context.obs, env_context.info, goal)
        positive_coverage_paths = _collect_positive_coverage_paths(agent)
        logger.info(
            "Agent finished execution",
            target_url=task.target_url,
            stage_index=task.stage_index,
            iteration_step=task.iteration_step,
            trajectory_index=task.trajectory_index,
            success=result.success,
            saved_file_path=agent.saved_file_path,
            positive_coverage_count=len(positive_coverage_paths),
        )
        return TrajectoryRunResult(
            target_url=task.target_url,
            trajectory_index=task.trajectory_index,
            iteration_step=task.iteration_step,
            stage_index=task.stage_index,
            saved_file_path=agent.saved_file_path,
            positive_coverage_paths=positive_coverage_paths,
            success=result.success,
            error=result.error,
        )
    except Exception as e:
        logger.error(
            "Trajectory execution failed",
            target_url=task.target_url,
            stage_index=task.stage_index,
            iteration_step=task.iteration_step,
            trajectory_index=task.trajectory_index,
            error=str(e),
            exc_info=True,
        )
        return TrajectoryRunResult(
            target_url=task.target_url,
            trajectory_index=task.trajectory_index,
            iteration_step=task.iteration_step,
            stage_index=task.stage_index,
            saved_file_path=agent.saved_file_path if agent is not None else None,
            positive_coverage_paths=_collect_positive_coverage_paths(agent),
            success=False,
            error=str(e),
        )
    finally:
        if env_context is not None:
            final_coverage_path = None
            if agent is not None and getattr(agent, "extra_info_dir", None) is not None:
                final_coverage_path = agent.extra_info_dir / "coverage.json"
            stop_envrionment(
                env_context,
                record_coverage=task.record_coverage,
                record_coverage_path=final_coverage_path,
            )


class LocalExplorePipeline:
    def __init__(self):
        self.model_server_process = None
        self.settings = get_settings()
        logger.info(
            f"Pipeline initialized with settings: port={self.settings.model_api_port}, model={self.settings.model}"
        )
        self.base_url = f"http://{self.settings.model_api_host}:{self.settings.model_api_port}"
        self.api_key = self.settings.model_api_key
        self.inference_client = InferenceClient(base_url=self.base_url, api_key=self.settings.model_api_key)

    def open(self):
        """启动模型服务子进程"""
        if self.model_server_process and self.model_server_process.is_alive():
            logger.warning("Model server process is already running.")
            return

        logger.info("Starting model server process...")
        try:
            ctx = multiprocessing.get_context("spawn")
            self.model_server_process = ctx.Process(
                target=run_model_server,
                kwargs={"port": self.settings.model_api_port, "host": self.settings.model_api_host},
            )
            self.model_server_process.start()
            logger.info(
                f"Model server process started with PID: {self.model_server_process.pid}. Waiting for readiness..."
            )
            self.inference_client.wait_for_server_ready()
            logger.info("Model server should be ready now.")
        except Exception as e:
            logger.error(f"Failed to start model server: {e}", exc_info=True)
            raise e

    def close(self):
        """关闭模型服务子进程"""
        if self.model_server_process:
            if self.model_server_process.is_alive():
                logger.info(f"Terminating model server process (PID: {self.model_server_process.pid})...")
                self.model_server_process.terminate()
                self.model_server_process.join(timeout=5)

                if self.model_server_process.is_alive():
                    logger.warning("Model server did not terminate gracefully, killing it.")
                    self.model_server_process.kill()

                logger.info("Model server process terminated.")
            else:
                logger.info("Model server process was already stopped.")

            self.model_server_process = None
        else:
            logger.debug("No model server process to close.")

    def _get_target_urls(self) -> list[str]:
        if self.settings.target_urls:
            return self.settings.target_urls
        return ["http://localhost:8000/weibo/"]

    def _get_experiment_dir(self) -> Path:
        return (Path(self.settings.history_data_base_dir) / self.settings.exp_name).resolve()

    def _get_curriculum_state_path(self) -> Path:
        return self._get_experiment_dir() / "curriculum_state.json"

    def _get_iteration_output_dir(self, stage_index: int, iteration_step: int) -> Path:
        return (self._get_experiment_dir() / f"stage_{stage_index}" / f"iteration_{iteration_step}").resolve()

    def _reset_iteration_output_dir(self, stage_index: int, iteration_step: int) -> Path:
        iteration_output_dir = self._get_iteration_output_dir(stage_index=stage_index, iteration_step=iteration_step)
        if iteration_output_dir.exists():
            shutil.rmtree(iteration_output_dir)
        iteration_output_dir.mkdir(parents=True, exist_ok=True)
        return iteration_output_dir

    def _build_iteration_tasks(
        self,
        curriculum_state: CurriculumState,
        iteration_step: int,
        model_checkpoint: str,
        iteration_output_dir: Path,
    ) -> list[list[TrajectoryTask]]:
        target_urls = self._get_target_urls()
        url_parallelism = max(1, min(self.settings.url_parallelism, len(target_urls)))
        trajectory_parallelism = max(
            1,
            min(self.settings.trajectory_parallelism_per_url, self.settings.explore_num_per_iteration),
        )

        task_batches: list[list[TrajectoryTask]] = []
        for url_start in range(0, len(target_urls), url_parallelism):
            current_urls = target_urls[url_start:url_start + url_parallelism]
            for trajectory_start in range(0, self.settings.explore_num_per_iteration, trajectory_parallelism):
                batch: list[TrajectoryTask] = []
                for target_url in current_urls:
                    frozen_paths = curriculum_state.get_frozen_coverage_paths(target_url)
                    for trajectory_index in range(
                        trajectory_start,
                        min(trajectory_start + trajectory_parallelism, self.settings.explore_num_per_iteration),
                    ):
                        batch.append(
                            TrajectoryTask(
                                base_url=self.base_url,
                                api_key=self.api_key,
                                model_checkpoint=model_checkpoint,
                                target_url=target_url,
                                stage_index=curriculum_state.stage_index,
                                iteration_step=iteration_step,
                                iteration_output_dir=str(iteration_output_dir),
                                frozen_global_coverage_paths=frozen_paths,
                                trajectory_index=trajectory_index,
                                max_steps_per_trajectory=self.settings.max_steps_per_trajectory,
                                headless_mode=self.settings.headless_mode,
                                record_coverage=self.settings.record_coverage,
                                browser_timeout=self.settings.browser_timeout,
                            )
                        )
                if batch:
                    task_batches.append(batch)
        return task_batches

    def _collect_iteration_data(
        self,
        curriculum_state: CurriculumState,
        model_checkpoint: str,
        iteration_step: int,
    ) -> tuple[Path, list[TrajectoryRunResult]]:
        stage_index = curriculum_state.stage_index
        iteration_output_dir = self._reset_iteration_output_dir(stage_index=stage_index, iteration_step=iteration_step)
        task_batches = self._build_iteration_tasks(
            curriculum_state=curriculum_state,
            iteration_step=iteration_step,
            model_checkpoint=model_checkpoint,
            iteration_output_dir=iteration_output_dir,
        )
        worker_count = max(1, self.settings.url_parallelism * self.settings.trajectory_parallelism_per_url)
        ctx = multiprocessing.get_context("spawn")

        all_results: list[TrajectoryRunResult] = []
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            for task_batch in task_batches:
                futures = [executor.submit(_run_single_trajectory_task, task) for task in task_batch]
                for future in as_completed(futures):
                    all_results.append(future.result())

        return iteration_output_dir / "histories", all_results

    def _record_stage_candidates(
        self,
        curriculum_state: CurriculumState,
        run_results: list[TrajectoryRunResult],
    ) -> None:
        ordered_results = sorted(
            run_results,
            key=lambda item: (item.target_url, item.trajectory_index, item.saved_file_path or ""),
        )
        for result in ordered_results:
            if not result.positive_coverage_paths:
                continue
            curriculum_state.add_stage_candidate_coverage_paths(
                target_url=result.target_url,
                coverage_paths=result.positive_coverage_paths,
            )

    def _prune_old_checkpoints(self):
        if not self.checkpoint_models.full():
            return
        model_to_delete = self.checkpoint_models.get()
        logger.info(f"reached max size, try to delete oldest model: {model_to_delete}")
        if Path(model_to_delete).exists():
            shutil.rmtree(model_to_delete)
        else:
            logger.warning(f"failed to delete oldest model because this path is missing: {model_to_delete}")

    def _run_training_stage(
        self,
        stage_name: str,
        stage_index: int,
        iteration_step: int,
        processed_data_path: Path,
        base_model: str | None = None,
    ) -> str:
        output_base_dir = Path(self.settings.output_base_dir)
        output_stage_dir = (
            output_base_dir
            / self.settings.exp_name
            / stage_name
            / f"stage_{stage_index}"
            / f"iteration_{iteration_step}"
        )
        stage_training_args = dict(self.settings.stage_training_args[stage_name])
        if base_model is not None:
            stage_training_args["model"] = base_model
        output_model_dir = run_training(
            self.settings.training_api_type,
            data_path=str(processed_data_path),
            output_dir=output_stage_dir,
            **stage_training_args,
        )
        return str(output_model_dir)

    def _prepare_stage_data(
        self,
        stage_name: str,
        stage_index: int,
        data_path: Path,
        format_func,
        processor_config: dict,
        iteration_step: int,
    ) -> tuple[Path, ProcessorStats | None]:
        processed_data_path = (
            data_path / ".." / f"processed_{stage_name}_data" / f"stage_{stage_index}_iteration_{iteration_step}.parquet"
        ).resolve()
        processor = get_processor_func(
            processor_config["name"],
            format_func=format_func,
            least_sample_num=self.settings.least_sample_num_per_iteration,
            **processor_config.get("kwargs", {}),
        )
        stats = prepare_explorer_data(
            data_path=str(data_path),
            processed_data_path=str(processed_data_path),
            processor=processor,
            dataset_num_proc=self.settings.dataset_num_proc,
            return_stats=True,
        )
        return processed_data_path, stats

    def _prepare_memory_data(
        self,
        stage_index: int,
        data_path: Path,
        iteration_step: int,
    ) -> tuple[Path, ProcessorStats | None]:
        format_func = partial(
            get_memory_format_func(self.settings.memory_type),
            get_user_prompt=get_brain_prompt_func(self.settings.brain_type),
        )
        memory_processor_config = self.settings.data_processors["memory"]
        return self._prepare_stage_data(
            stage_name="memory",
            stage_index=stage_index,
            data_path=data_path,
            format_func=format_func,
            processor_config=memory_processor_config,
            iteration_step=iteration_step,
        )

    def _prepare_policy_data(
        self,
        stage_index: int,
        data_path: Path,
        iteration_step: int,
    ) -> tuple[Path, ProcessorStats | None]:
        format_func = partial(
            get_policy_format_func(self.settings.memory_type),
            get_user_prompt=get_brain_prompt_func(self.settings.brain_type),
        )
        policy_processor_config = self.settings.data_processors["policy"]
        return self._prepare_stage_data(
            stage_name="policy",
            stage_index=stage_index,
            data_path=data_path,
            format_func=format_func,
            processor_config=policy_processor_config,
            iteration_step=iteration_step,
        )

    def _has_min_samples(self, stage_name: str, stats: ProcessorStats | None, min_required: int) -> bool:
        filtered_count = stats.filtered_count if stats is not None else 0
        if filtered_count < min_required:
            logger.warning(
                f"Skip training: insufficient {stage_name} data",
                min_required=min_required,
                filtered_count=filtered_count,
            )
            return False
        return True

    def _run_memory_training(self, stage_index: int, iteration_step: int, processed_data_path: Path) -> str:
        logger.info("[Memory] Collecting data and training...")
        output_model_dir = self._run_training_stage(
            stage_name="memory",
            stage_index=stage_index,
            iteration_step=iteration_step,
            processed_data_path=processed_data_path,
        )
        logger.info("[Memory] training completed!")
        return output_model_dir

    def _run_policy_training(
        self,
        stage_index: int,
        iteration_step: int,
        base_model: str,
        processed_data_path: Path,
    ) -> str:
        logger.info("[Policy] Collecting data and training...")
        output_model_dir = self._run_training_stage(
            stage_name="policy",
            stage_index=stage_index,
            iteration_step=iteration_step,
            processed_data_path=processed_data_path,
            base_model=base_model,
        )
        logger.info("[Policy] training completed!")
        return output_model_dir

    def _run_explore_pipeline_core(self):
        self.open()

        current_model_checkpoint = self.settings.model
        logger.info(f"Using initial checkpoint: {current_model_checkpoint}")
        self.checkpoint_models = queue.Queue(maxsize=self.settings.max_version_limit)

        target_urls = self._get_target_urls()
        curriculum_state = CurriculumState.load(
            path=self._get_curriculum_state_path(),
            target_urls=target_urls,
            curriculum_stage_iterations=self.settings.curriculum_stage_iterations,
            start_iteration_step=self.settings.start_iteration_step,
        )
        curriculum_state.save(self._get_curriculum_state_path())

        for iteration_step in range(self.settings.start_iteration_step, self.settings.update_iterations):
            logger.info(
                f"--- Iteration {iteration_step + 1}/{self.settings.update_iterations} ---",
                stage_index=curriculum_state.stage_index,
                completed_iterations_in_stage=curriculum_state.completed_iterations_in_stage,
            )

            current_online_models = [model_info.id for model_info in self.inference_client.list_models()]
            logger.info(f"current online models: {current_online_models}")
            if current_model_checkpoint not in current_online_models:
                logger.info("Launching model...", model_path=current_model_checkpoint, model_type=self.settings.model_type)
                self.inference_client.launch_model(
                    model_path=str(current_model_checkpoint),
                    model_type=self.settings.model_type,
                    jinja_template_path=self.settings.jinja_template_path,
                )

            stage_index = curriculum_state.stage_index
            data_path, run_results = self._collect_iteration_data(
                curriculum_state=curriculum_state,
                model_checkpoint=str(current_model_checkpoint),
                iteration_step=iteration_step,
            )

            logger.info("Trying to unload current online model", model=current_model_checkpoint)
            self.inference_client.terminate_model(current_model_checkpoint)

            if not data_path.exists():
                logger.warning("Skip training: no saved trajectory found for this iteration.")
                continue

            self._record_stage_candidates(curriculum_state=curriculum_state, run_results=run_results)
            curriculum_state.save(self._get_curriculum_state_path())

            min_required = max(1, self.settings.least_sample_num_per_iteration)

            memory_processed_path, memory_stats = self._prepare_memory_data(
                stage_index=stage_index,
                data_path=data_path,
                iteration_step=iteration_step,
            )
            if not self._has_min_samples("memory", memory_stats, min_required):
                continue

            policy_processed_path, policy_stats = self._prepare_policy_data(
                stage_index=stage_index,
                data_path=data_path,
                iteration_step=iteration_step,
            )
            if not self._has_min_samples("policy", policy_stats, min_required):
                continue

            self._prune_old_checkpoints()
            output_model_dir = self._run_memory_training(
                stage_index=stage_index,
                iteration_step=iteration_step,
                processed_data_path=memory_processed_path,
            )
            current_model_checkpoint = str(output_model_dir)
            self.checkpoint_models.put(str(current_model_checkpoint))

            self._prune_old_checkpoints()
            output_model_dir = self._run_policy_training(
                stage_index=stage_index,
                iteration_step=iteration_step,
                base_model=current_model_checkpoint,
                processed_data_path=policy_processed_path,
            )
            current_model_checkpoint = str(output_model_dir)
            self.checkpoint_models.put(str(current_model_checkpoint))

            promoted_by_url = curriculum_state.complete_iteration()
            curriculum_state.save(self._get_curriculum_state_path())

            if promoted_by_url:
                logger.info(
                    "Curriculum stage promoted",
                    completed_stage_index=stage_index,
                    next_stage_index=curriculum_state.stage_index,
                    promoted_counts={key: len(value) for key, value in promoted_by_url.items()},
                )

        time.sleep(2)

    def run_explore_pipeline(self):
        """自动探索，自动收集数据，自动训练的闭环。"""
        logger.info("=== Starting Exploration Pipeline ===")

        try:
            self._run_explore_pipeline_core()
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user.")
        except Exception as e:
            logger.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
        finally:
            logger.info("Cleaning up resources...")
            self.close()
            logger.info("=== Exploration Pipeline Finished ===")


if __name__ == "__main__":
    pipeline = LocalExplorePipeline()
    pipeline.run_explore_pipeline()
