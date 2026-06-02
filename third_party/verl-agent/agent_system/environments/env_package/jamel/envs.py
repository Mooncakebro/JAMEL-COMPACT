import inspect
import json
import logging
import os
import re
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import browsergym.core
import gymnasium as gym
import numpy as np
import playwright.sync_api
import ray
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .coverage import start_coverage, take_coverage_snapshot
from .observer import get_observation, get_screenshot

logger = logging.getLogger(__name__)
_SYS_PATH_LOCK = threading.Lock()

browser_timeout = 5000
OBS_REFRESH_WAIT_MS = 400
OBS_REFRESH_MAX_TRIES = 6

TRACE_COLUMNS = [
    "type",
    "step",
    "prompt_text",
    "raw_response",
    "projected_action",
    "action",
    "reward",
    "done",
    "terminated",
    "truncated",
    "target_url",
    "phase",
    "round_id",
    "goal",
    "episode_dir",
    "coverage_path",
    "coverage_reward_source",
    "coverage_previous_score",
    "coverage_current_score",
    "coverage_delta_score",
    "coverage_skip_reason",
    "positive_reward_steps",
    "last_action",
    "last_action_error",
    "observation_text",
    "image_png_bytes",
    "image_height",
    "image_width",
    "image_channels",
]


def _normalize_jamel_root(root: str | os.PathLike | None) -> Path | None:
    if not root:
        return None

    root_path = Path(root).expanduser().resolve()
    if (root_path / "jamel").is_dir():
        return root_path
    if root_path.name == "jamel" and (root_path / "__init__.py").is_file():
        return root_path.parent
    return None


def _resolve_jamel_root(configured_root: str | os.PathLike | None = None) -> Path | None:
    if configured_root:
        normalized_configured = _normalize_jamel_root(configured_root)
        if normalized_configured is None:
            raise RuntimeError(
                f"Invalid env.jamel.jamel_root: {configured_root!s}. "
                "Expected the JAMEL repo root or its jamel package directory."
        )
        return normalized_configured

    repo_root = Path(__file__).resolve().parents[4]
    env_root = os.environ.get("JAMEL_ROOT")
    if env_root:
        normalized_env_root = _normalize_jamel_root(env_root)
        if normalized_env_root is not None:
            return normalized_env_root
        logger.warning("JAMEL_ROOT is set but invalid, ignoring: %s", env_root)

    candidates = [
        repo_root.parent / "JAMEL",
        Path("/workspace/JAMEL"),
    ]

    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_jamel_root(candidate)
        if normalized is None:
            continue
        normalized_str = str(normalized)
        if normalized_str in seen:
            continue
        seen.add(normalized_str)
        return normalized

    return None


def _load_jamel_reward_components(configured_root: str | os.PathLike | None = None):
    jamel_root = _resolve_jamel_root(configured_root)
    if jamel_root is not None:
        with _SYS_PATH_LOCK:
            if str(jamel_root) not in sys.path:
                sys.path.insert(0, str(jamel_root))

    try:
        from jamel.core.env.web.utils import StepHistory
        from jamel.core.reward.web.reward_funcs import jamel_reward_fn_web_coverage_details
    except Exception as exc:
        raise RuntimeError(
            "Failed to import JAMEL Monocart coverage reward. "
            "Set JAMEL_ROOT or env.jamel.jamel_root "
            "to the repository containing the jamel package."
        ) from exc

    reward_params = inspect.signature(jamel_reward_fn_web_coverage_details).parameters
    required_params = {"frozen_global_coverage_paths", "trajectory_history"}
    missing_params = required_params - set(reward_params)
    if missing_params:
        raise RuntimeError(
            "JAMEL reward function has an incompatible signature. "
            f"Missing parameters: {sorted(missing_params)}"
        )

    return StepHistory, jamel_reward_fn_web_coverage_details


def _patch_setup_for_open_ended_task(self: browsergym.core.OpenEndedTask, page: playwright.sync_api.Page) -> tuple[str, dict]:
    page.goto(self.start_url, timeout=browser_timeout, wait_until="commit")
    return self.goal, {}


browsergym.core.OpenEndedTask.setup = _patch_setup_for_open_ended_task


def _sanitize_url_for_path(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_") or "target"


def _coverage_detail_value(reward_details: dict, key: str, default=0):
    value = reward_details.get(key, default)
    return default if value is None else value


class JAMELWorker:
    def __init__(self, seed: int, env_kwargs: dict | None = None):
        self._rng = np.random.RandomState(seed)
        self._env_kwargs = dict(env_kwargs or {})
        self._env_id = self._env_kwargs.get("env_id", "browsergym/openended")
        self._headless = bool(self._env_kwargs.get("headless", True))
        self._timeout = int(self._env_kwargs.get("timeout", 60000))
        self._record_coverage = bool(self._env_kwargs.get("record_coverage", True))
        self._trace_image = bool(self._env_kwargs.get("trace_image", True))
        self._coverage_root = Path(
            self._env_kwargs.get("coverage_dir")
            or Path(tempfile.gettempdir()) / "verl-agent-jamel"
        )
        self._goal_template = self._env_kwargs.get(
            "goal",
            "Explore the website to maximize novel JavaScript execution coverage.",
        )
        self._jamel_root = self._env_kwargs.get("jamel_root")
        self._step_history_cls = None
        self._coverage_reward_details_fn = None
        if self._record_coverage:
            self._step_history_cls, self._coverage_reward_details_fn = _load_jamel_reward_components(
                self._jamel_root
            )
        self._global_frozen_paths = [
            str(Path(path).expanduser().resolve())
            for path in self._env_kwargs.get("frozen_coverage_paths", [])
        ]
        self._frozen_manifest = self._load_frozen_manifest(self._env_kwargs.get("frozen_coverage_manifest"))

        self.env = None
        self.cdp_session = None
        self.target_url = None
        self.goal = None
        self.episode_step = 0
        self.positive_reward_steps = 0
        self.current_frozen_coverage_paths: list[str] = []
        self.trajectory_history: list = []
        self.episode_dir = None
        self.trace_path = None
        self.summary_path = None
        self.trace_records: list[dict] = []
        self.phase = "train"
        self.round_id = 0
        self._current_obs = None
        self._current_info = None
        self._current_text_observation = ""

    def _encode_image_payload(self, image) -> dict:
        if image is None:
            return {
                "image_png_bytes": None,
                "image_height": None,
                "image_width": None,
                "image_channels": None,
            }

        np_image = np.asarray(image)
        if np_image.ndim != 3:
            raise ValueError(f"Expected HWC image for trace logging, got shape={np_image.shape!r}")
        if np_image.shape[2] not in (3, 4):
            raise ValueError(f"Expected 3/4-channel image for trace logging, got shape={np_image.shape!r}")

        pil_image = Image.fromarray(np_image.astype(np.uint8))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return {
            "image_png_bytes": buffer.getvalue(),
            "image_height": int(np_image.shape[0]),
            "image_width": int(np_image.shape[1]),
            "image_channels": int(np_image.shape[2]),
        }

    def _flush_trace(self) -> None:
        if self.trace_path is None:
            return
        existing_rows = {}
        if self.trace_path.is_file():
            try:
                for row in pq.read_table(self.trace_path).to_pylist():
                    existing_rows[(row.get("type"), row.get("step"))] = row
            except Exception:
                logger.exception("Failed to read existing JAMEL trace parquet")

        rows = []
        for record in self.trace_records:
            key = (record.get("type"), record.get("step"))
            merged = dict(existing_rows.get(key, {}))
            merged.update(record)
            rows.append({column: merged.get(column) for column in TRACE_COLUMNS})
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, self.trace_path)

    def _append_trace(self, payload: dict, *, image=None) -> None:
        if self.trace_path is None:
            return
        trace_record = dict(payload)
        trace_record.update(self._encode_image_payload(image if self._trace_image else None))
        self.trace_records.append(trace_record)
        self._flush_trace()

    def _write_summary(self) -> None:
        if self.summary_path is None:
            return
        payload = {
            "target_url": self.target_url,
            "phase": self.phase,
            "round_id": self.round_id,
            "goal": self.goal,
            "episode_step": self.episode_step,
            "positive_reward_steps": self.positive_reward_steps,
            "trajectory_history_count": len(self.trajectory_history),
            "frozen_coverage_path_count": len(self.current_frozen_coverage_paths),
            "episode_dir": str(self.episode_dir) if self.episode_dir is not None else None,
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_frozen_manifest(self, manifest_path: str | None) -> dict[str, list[str]]:
        if not manifest_path:
            return {}

        manifest = Path(manifest_path).expanduser().resolve()
        if not manifest.exists():
            logger.warning("Frozen coverage manifest does not exist: %s", manifest)
            return {}

        try:
            with manifest.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Failed to load frozen coverage manifest", extra={"manifest_path": str(manifest)})
            return {}

        normalized_payload: dict[str, list[str]] = {}
        for target_url, coverage_paths in payload.items():
            normalized_payload[target_url] = [
                str(Path(path).expanduser().resolve()) for path in (coverage_paths or [])
            ]
        return normalized_payload

    def _get_frozen_paths_for_url(self, target_url: str) -> list[str]:
        return [*self._global_frozen_paths, *self._frozen_manifest.get(target_url, [])]

    def _get_step_coverage_path(self) -> Path | None:
        if not self._record_coverage or self.episode_dir is None:
            return None
        return self.episode_dir / f"coverage_{self.episode_step}.json"

    def _build_step_history(
        self,
        *,
        action: str,
        before_obs: dict | None,
        after_obs: dict | None,
        before_info: dict | None,
        after_info: dict | None,
        before_observation: str,
        after_observation: str,
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        coverage_path: Path | None,
    ):
        if self._step_history_cls is None:
            raise RuntimeError("StepHistory is unavailable because record_coverage is disabled")

        return self._step_history_cls(
            before_obs=before_obs or {},
            after_obs=after_obs or {},
            before_info=before_info or {},
            after_info=after_info or {},
            before_observation=before_observation,
            after_observation=after_observation,
            step=self.episode_step,
            reward=float(reward),
            raw_content=action,
            memory_content=None,
            parsed_content={"action": action},
            result={
                "reward": float(reward),
                "done": bool(done),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            extra_fields={
                "coverage_path": str(coverage_path) if coverage_path is not None else None,
                "target_url": self.target_url,
                "phase": self.phase,
                "round_id": self.round_id,
                "episode_dir": str(self.episode_dir) if self.episode_dir is not None else None,
            },
        )

    def _compute_coverage_reward_details(self, current_step_history) -> dict:
        if not self._record_coverage:
            return {
                "reward": 0.0,
                "previous_score": 0,
                "current_score": 0,
                "delta_score": 0,
                "skip_reason": "coverage_disabled",
            }

        details = self._coverage_reward_details_fn(
            current_step_history,
            frozen_global_coverage_paths=self.current_frozen_coverage_paths,
            trajectory_history=self.trajectory_history,
        )
        return dict(details)

    def _close_env(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                logger.exception("Failed to close JAMEL worker environment")
        self.env = None
        self.cdp_session = None

    def _refresh_obs(self, obs: dict | None) -> dict | None:
        if self.env is None or not isinstance(obs, dict):
            return obs
        refreshed_obs = obs
        for _ in range(OBS_REFRESH_MAX_TRIES):
            busy_flag = str(refreshed_obs.get("axtree_object", "")).find("busy=1") >= 0
            if not busy_flag:
                return refreshed_obs
            try:
                self.env.unwrapped.page.wait_for_timeout(OBS_REFRESH_WAIT_MS)
                refreshed_obs = self.env.unwrapped._get_obs()
            except Exception:
                logger.exception("Failed to refresh busy JAMEL observation")
                return obs
        return refreshed_obs

    def reset(self, target_url: str, trace_enabled: bool = False, phase: str = "train", round_id: int = 0):
        global browser_timeout
        browser_timeout = self._timeout

        self._close_env()
        self.target_url = target_url
        self.phase = phase
        self.round_id = round_id
        self.goal = f"{self._goal_template}\nTarget URL: {target_url}"
        self.episode_step = 0
        self.positive_reward_steps = 0
        self.current_frozen_coverage_paths = self._get_frozen_paths_for_url(target_url)
        self.trajectory_history = []
        self.episode_dir = self._coverage_root / phase / f"round_{round_id:06d}" / _sanitize_url_for_path(target_url) / uuid.uuid4().hex
        if trace_enabled or self._record_coverage:
            self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.episode_dir / "trace.parquet" if trace_enabled else None
        self.summary_path = self.episode_dir / "summary.json" if trace_enabled else None
        self.trace_records = []

        self.env = gym.make(
            self._env_id,
            task_kwargs={"start_url": target_url},
            wait_for_user_message=False,
            headless=self._headless,
            timeout=self._timeout,
        )

        obs, info = self.env.reset()
        obs = self._refresh_obs(obs)
        if self._record_coverage:
            page = self.env.unwrapped.page
            self.cdp_session = start_coverage(page)

        text_observation = get_observation(obs)
        screenshot = get_screenshot(obs)
        info = dict(info or {})
        info.update(
            {
                "goal": self.goal,
                "target_url": target_url,
                "won": False,
                "episode_dir": str(self.episode_dir),
                "trace_path": str(self.trace_path) if self.trace_path is not None else None,
                "trace_enabled": trace_enabled,
            }
        )
        self._current_obs = obs
        self._current_info = dict(info)
        self._current_text_observation = text_observation
        self._append_trace(
            {
                "type": "reset",
                "step": 0,
                "target_url": target_url,
                "goal": self.goal,
                "episode_dir": str(self.episode_dir),
                "phase": phase,
                "round_id": round_id,
                "observation_text": text_observation,
            },
            image=screenshot,
        )
        self._write_summary()

        return {"text": text_observation, "image": screenshot}, info

    def step(self, action: str):
        if self.env is None:
            raise RuntimeError("JAMEL worker step called before reset")

        before_obs = self._current_obs if isinstance(self._current_obs, dict) else {}
        before_info = self._current_info if isinstance(self._current_info, dict) else {}
        before_observation = self._current_text_observation or ""

        obs, _raw_reward, terminated, truncated, info = self.env.step(action)
        obs = self._refresh_obs(obs)
        self.episode_step += 1

        coverage_path = None
        if self._record_coverage and self.cdp_session is not None:
            coverage_path = self._get_step_coverage_path()
            take_coverage_snapshot(self.cdp_session, coverage_path)

        done = bool(terminated or truncated)

        text_observation = get_observation(obs)
        screenshot = get_screenshot(obs)
        info = dict(info or {})

        reward = 0.0
        reward_details = {
            "reward": 0.0,
            "previous_score": 0,
            "current_score": 0,
            "delta_score": 0,
            "skip_reason": "coverage_disabled",
        }
        if self._record_coverage:
            step_history_kwargs = {
                "action": action,
                "before_obs": before_obs,
                "after_obs": obs if isinstance(obs, dict) else {},
                "before_info": before_info,
                "after_info": info,
                "before_observation": before_observation,
                "after_observation": text_observation,
                "done": done,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "coverage_path": coverage_path,
            }
            reward_input_history = self._build_step_history(reward=0.0, **step_history_kwargs)
            reward_details = self._compute_coverage_reward_details(reward_input_history)
            reward = float(reward_details.get("reward", 0.0) or 0.0)
            current_step_history = self._build_step_history(reward=reward, **step_history_kwargs)
            self.trajectory_history.append(current_step_history)
        self.positive_reward_steps += int(reward > 0)
        coverage_reward_source = "jamel_monocart" if self._record_coverage else "disabled"

        info.update(
            {
                "goal": self.goal,
                "target_url": self.target_url,
                "coverage_path": str(coverage_path) if coverage_path else None,
                "coverage_reward_source": coverage_reward_source,
                "coverage_previous_score": _coverage_detail_value(reward_details, "previous_score"),
                "coverage_current_score": _coverage_detail_value(reward_details, "current_score"),
                "coverage_delta_score": _coverage_detail_value(reward_details, "delta_score"),
                "coverage_skip_reason": reward_details.get("skip_reason"),
                "won": self.positive_reward_steps > 0,
                "step_reward": float(reward),
                "episode_step": self.episode_step,
                "episode_dir": str(self.episode_dir),
                "trace_path": str(self.trace_path) if self.trace_path is not None else None,
                "trace_enabled": self.trace_path is not None,
            }
        )
        self._append_trace(
            {
                "type": "step",
                "step": self.episode_step,
                "action": action,
                "reward": float(reward),
                "done": done,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "target_url": self.target_url,
                "phase": self.phase,
                "round_id": self.round_id,
                "coverage_path": str(coverage_path) if coverage_path else None,
                "coverage_reward_source": coverage_reward_source,
                "coverage_previous_score": _coverage_detail_value(reward_details, "previous_score"),
                "coverage_current_score": _coverage_detail_value(reward_details, "current_score"),
                "coverage_delta_score": _coverage_detail_value(reward_details, "delta_score"),
                "coverage_skip_reason": reward_details.get("skip_reason"),
                "positive_reward_steps": self.positive_reward_steps,
                "last_action": obs.get("last_action") if isinstance(obs, dict) else None,
                "last_action_error": obs.get("last_action_error") if isinstance(obs, dict) else None,
                "observation_text": text_observation,
            },
            image=screenshot,
        )
        self._write_summary()
        self._current_obs = obs
        self._current_info = dict(info)
        self._current_text_observation = text_observation

        return {"text": text_observation, "image": screenshot}, reward, done, info

    def close(self):
        self._close_env()


class JAMELMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self._rng = np.random.RandomState(seed)
        self._env_kwargs = dict(env_kwargs or {})
        self._target_urls = list(self._env_kwargs.get("target_urls", []))
        if not self._target_urls:
            raise ValueError("env.jamel.target_urls must not be empty")
        self._trace_mode = str(self._env_kwargs.get("trace_mode", "both")).lower()
        self._trace_freq = max(int(self._env_kwargs.get("trace_freq", 1)), 1)
        self._max_traces_per_dump = max(int(self._env_kwargs.get("max_traces_per_dump", 2)), 0)
        self._worker_timeout_ms = int(self._env_kwargs.get("worker_timeout", self._env_kwargs.get("timeout", 60000)))
        self._worker_timeout_s = max(self._worker_timeout_ms / 1000.0, 1.0)
        self._reset_round = 0

        self._env_worker_cls = ray.remote(**resources_per_worker)(JAMELWorker)
        self._worker_seeds = [
            seed + (worker_idx // max(group_n, 1))
            for worker_idx in range(self.num_processes)
        ]
        self._workers = [
            self._make_worker(worker_idx)
            for worker_idx in range(self.num_processes)
        ]
        self._last_obs = [
            {"text": "JAMEL worker is initializing.", "image": None}
            for _ in range(self.num_processes)
        ]
        self._last_infos = [
            self._build_recovery_info(worker_idx, reason="worker_initializing")
            for worker_idx in range(self.num_processes)
        ]
        self._episode_specs = [None for _ in range(self.num_processes)]
        self._episode_actions = [[] for _ in range(self.num_processes)]
        self._closed = False

    def _make_worker(self, worker_idx: int):
        return self._env_worker_cls.remote(self._worker_seeds[worker_idx], self._env_kwargs)

    def _restart_worker(self, worker_idx: int, *, reason: str) -> None:
        worker = self._workers[worker_idx]
        try:
            ray.kill(worker, no_restart=True)
        except TypeError:
            ray.kill(worker)
        except Exception:
            logger.exception(
                "Failed to kill timed-out JAMEL worker",
                extra={"worker_idx": worker_idx, "reason": reason},
            )
        self._workers[worker_idx] = self._make_worker(worker_idx)
        logger.warning(
            "Restarted JAMEL worker",
            extra={
                "worker_idx": worker_idx,
                "reason": reason,
                "worker_timeout_ms": self._worker_timeout_ms,
            },
        )

    def _build_recovery_info(self, worker_idx: int, *, reason: str) -> dict:
        return {
            "goal": self._env_kwargs.get("goal", "Explore the website to maximize novel coverage."),
            "target_url": "",
            "won": False,
            "episode_step": 0,
            "worker_idx": worker_idx,
            "worker_restarted": True,
            "worker_restart_reason": reason,
            "coverage_reward_source": "worker_recovery",
            "coverage_previous_score": 0,
            "coverage_current_score": 0,
            "coverage_delta_score": 0,
            "coverage_skip_reason": reason,
            "step_reward": 0.0,
            "trace_enabled": False,
            "trace_path": None,
        }

    def _reset_worker_episode(self, worker_idx: int):
        spec = self._episode_specs[worker_idx]
        if spec is None:
            return None
        return self._workers[worker_idx].reset.remote(
            spec["target_url"],
            trace_enabled=spec["trace_enabled"],
            phase=spec["phase"],
            round_id=spec["round_id"],
        )

    def _replay_worker_episode(self, worker_idx: int, actions: list[str]):
        reset_future = self._reset_worker_episode(worker_idx)
        if reset_future is None:
            raise RuntimeError(f"Cannot replay worker {worker_idx}: no episode spec")
        obs, info = ray.get(reset_future, timeout=self._worker_timeout_s)
        for action in actions:
            obs, _reward, done, info = ray.get(
                self._workers[worker_idx].step.remote(action),
                timeout=self._worker_timeout_s,
            )
            if done:
                break
        self._last_obs[worker_idx] = obs
        self._last_infos[worker_idx] = info
        return obs, info

    def _retry_step_after_restart(self, worker_idx: int, action: str):
        replay_actions = [*self._episode_actions[worker_idx], action]
        logger.warning(
            "Retrying JAMEL worker step after restart",
            extra={
                "worker_idx": worker_idx,
                "replay_action_count": len(replay_actions),
                "worker_timeout_ms": self._worker_timeout_ms,
            },
        )
        obs, info = self._replay_worker_episode(worker_idx, replay_actions[:-1])
        obs, reward, done, info = ray.get(
            self._workers[worker_idx].step.remote(action),
            timeout=self._worker_timeout_s,
        )
        self._last_obs[worker_idx] = obs
        self._last_infos[worker_idx] = info
        self._episode_actions[worker_idx].append(action)
        return obs, reward, done, info

    def _wait_for_indexed_results(self, indexed_futures: list[tuple[int, ray.ObjectRef]], *, operation: str):
        pending_refs = [future for _, future in indexed_futures]
        ref_to_idx = {future: idx for idx, future in indexed_futures}
        ready_refs, pending_refs = ray.wait(
            pending_refs,
            num_returns=len(pending_refs),
            timeout=self._worker_timeout_s,
        )

        results = {}
        failed_indices = []
        for future in ready_refs:
            worker_idx = ref_to_idx[future]
            try:
                results[worker_idx] = ray.get(future)
            except Exception as exc:
                logger.exception(
                    "JAMEL worker failed during %s; restarting worker %s",
                    operation,
                    worker_idx,
                )
                self._restart_worker(worker_idx, reason=f"{operation}_exception:{type(exc).__name__}")
                failed_indices.append(worker_idx)

        for future in pending_refs:
            worker_idx = ref_to_idx[future]
            logger.warning(
                "JAMEL worker timed out during %s; restarting worker %s after %.1fs",
                operation,
                worker_idx,
                self._worker_timeout_s,
            )
            try:
                ray.cancel(future, force=False)
            except Exception:
                logger.debug(
                    "Ignoring Ray cancel failure for timed-out JAMEL actor task",
                    exc_info=True,
                )
            self._restart_worker(worker_idx, reason=f"{operation}_timeout")
            failed_indices.append(worker_idx)

        return results, failed_indices

    def _sample_target_urls(self, round_id: int) -> list[str]:
        num_urls = len(self._target_urls)
        if num_urls == 0:
            raise ValueError("env.jamel.target_urls must not be empty")

        chosen_urls = [
            self._target_urls[(round_id * self.env_num + env_idx) % num_urls]
            for env_idx in range(self.env_num)
        ]
        return np.repeat(chosen_urls, self.group_n).tolist()

    def _sample_trace_flags(self, round_id: int) -> list[bool]:
        phase = "train" if self.is_train else "val"
        mode_enabled = self._trace_mode in ("both", phase)
        round_enabled = round_id % self._trace_freq == 0
        trace_count = min(self._max_traces_per_dump, self.num_processes) if mode_enabled and round_enabled else 0
        return [worker_idx < trace_count for worker_idx in range(self.num_processes)]

    def reset(self):
        round_id = self._reset_round
        self._reset_round += 1
        phase = "train" if self.is_train else "val"
        target_urls = self._sample_target_urls(round_id)
        trace_flags = self._sample_trace_flags(round_id)
        indexed_futures = [
            (
                worker_idx,
                worker.reset.remote(target_url, trace_enabled=trace_enabled, phase=phase, round_id=round_id),
            )
            for worker_idx, (worker, target_url, trace_enabled) in enumerate(zip(self._workers, target_urls, trace_flags))
        ]
        result_map, failed_indices = self._wait_for_indexed_results(indexed_futures, operation="reset")

        # Reset is the start of a rollout, so retry once on fresh workers instead of
        # returning a synthetic initial observation.
        if failed_indices:
            retry_futures = [
                (
                    worker_idx,
                    self._workers[worker_idx].reset.remote(
                        target_urls[worker_idx],
                        trace_enabled=trace_flags[worker_idx],
                        phase=phase,
                        round_id=round_id,
                    ),
                )
                for worker_idx in failed_indices
            ]
            retry_results, retry_failed_indices = self._wait_for_indexed_results(retry_futures, operation="reset_retry")
            result_map.update(retry_results)
            failed_indices = retry_failed_indices

        results = []
        for worker_idx in range(self.num_processes):
            if worker_idx in result_map:
                obs, info = result_map[worker_idx]
                self._episode_specs[worker_idx] = {
                    "target_url": target_urls[worker_idx],
                    "trace_enabled": trace_flags[worker_idx],
                    "phase": phase,
                    "round_id": round_id,
                }
                self._episode_actions[worker_idx] = []
                self._last_obs[worker_idx] = obs
                self._last_infos[worker_idx] = info
                results.append((obs, info))
            else:
                info = self._build_recovery_info(worker_idx, reason="reset_timeout")
                info.update({"target_url": target_urls[worker_idx], "phase": phase, "round_id": round_id})
                obs = {"text": "JAMEL worker reset timed out; this rollout item was recovered.", "image": None}
                self._episode_specs[worker_idx] = {
                    "target_url": target_urls[worker_idx],
                    "trace_enabled": trace_flags[worker_idx],
                    "phase": phase,
                    "round_id": round_id,
                }
                self._episode_actions[worker_idx] = []
                self._last_obs[worker_idx] = obs
                self._last_infos[worker_idx] = info
                results.append((obs, info))
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")

        indexed_futures = [
            (worker_idx, worker.step.remote(action))
            for worker_idx, (worker, action) in enumerate(zip(self._workers, actions))
        ]
        result_map, failed_indices = self._wait_for_indexed_results(indexed_futures, operation="step")

        results = []
        for worker_idx in range(self.num_processes):
            if worker_idx in result_map:
                obs, reward, done, info = result_map[worker_idx]
                self._last_obs[worker_idx] = obs
                self._last_infos[worker_idx] = info
                self._episode_actions[worker_idx].append(actions[worker_idx])
                results.append((obs, reward, done, info))
            else:
                try:
                    results.append(self._retry_step_after_restart(worker_idx, actions[worker_idx]))
                except Exception as exc:
                    logger.exception(
                        "JAMEL worker retry failed; ending trajectory for worker %s",
                        worker_idx,
                    )
                    self._restart_worker(worker_idx, reason=f"step_retry_failed:{type(exc).__name__}")
                    obs = self._last_obs[worker_idx]
                    info = dict(self._last_infos[worker_idx])
                    info.update(
                        {
                            "won": False,
                            "worker_idx": worker_idx,
                            "worker_restarted": True,
                            "worker_restart_reason": "step_retry_failed",
                            "coverage_reward_source": "worker_recovery",
                            "coverage_skip_reason": "step_retry_failed",
                            "step_reward": 0.0,
                        }
                    )
                    # Last-resort fallback: keep rollout alive if retry also fails.
                    results.append((obs, 0.0, True, info))
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def close(self):
        if self._closed:
            return

        close_futures = [worker.close.remote() for worker in self._workers]
        ready_refs, pending_refs = ray.wait(
            close_futures,
            num_returns=len(close_futures),
            timeout=min(self._worker_timeout_s, 30.0),
        )
        if ready_refs:
            ray.get(ready_refs)
        for future in pending_refs:
            try:
                ray.cancel(future, force=False)
            except Exception:
                logger.debug("Ignoring Ray cancel failure while closing JAMEL worker", exc_info=True)
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True

    def __del__(self):
        self.close()


def build_jamel_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict | None = None,
):
    return JAMELMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
