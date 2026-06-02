import json
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
JAMEL_ROOT = REPO_ROOT.parent.parent


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


coverage_module = _load_module(
    "jamel_coverage_test_module",
    "agent_system/environments/env_package/jamel/coverage.py",
)
projection_module = _load_module(
    "jamel_projection_test_module",
    "agent_system/environments/env_package/jamel/projection.py",
)

extract_covered_units = coverage_module.extract_covered_units
merge_coverage_units_from_paths = coverage_module.merge_coverage_units_from_paths
DEFAULT_ACTION = projection_module.DEFAULT_ACTION
jamel_projection = projection_module.jamel_projection


def _load_envs_module():
    pytest.importorskip("browsergym.core")
    pytest.importorskip("ray")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("agent_system.environments.env_package.jamel.envs")


def _require_jamel_root() -> Path:
    if not (JAMEL_ROOT / "jamel").is_dir():
        pytest.skip("JAMEL repository root is not available above third_party/verl-agent")
    return JAMEL_ROOT


def test_jamel_projection_extracts_action():
    actions, valids = jamel_projection(
        ["<think>inspect the CTA</think><action>click('12')</action>"]
    )
    assert actions == ["click('12')"]
    assert valids == [1]


def test_jamel_projection_falls_back_to_noop():
    actions, valids = jamel_projection(["click('12')"])
    assert actions == [DEFAULT_ACTION]
    assert valids == [0]


def test_extract_covered_units_only_keeps_positive_ranges():
    coverage_records = [
        {
            "url": "http://localhost:8000/app.js",
            "functions": [
                {
                    "ranges": [
                        {"startOffset": 0, "endOffset": 10, "count": 1},
                        {"startOffset": 10, "endOffset": 20, "count": 0},
                    ]
                }
            ],
        }
    ]

    assert extract_covered_units(coverage_records) == {
        ("http://localhost:8000/app.js", 0, 10)
    }


def test_merge_coverage_units_from_paths_dedupes_ranges(tmp_path):
    first_path = tmp_path / "coverage_1.json"
    second_path = tmp_path / "coverage_2.json"

    first_payload = [
        {
            "url": "http://localhost:8000/app.js",
            "functions": [{"ranges": [{"startOffset": 0, "endOffset": 10, "count": 1}]}],
        }
    ]
    second_payload = [
        {
            "url": "http://localhost:8000/app.js",
            "functions": [
                {
                    "ranges": [
                        {"startOffset": 0, "endOffset": 10, "count": 1},
                        {"startOffset": 10, "endOffset": 20, "count": 1},
                    ]
                }
            ],
        }
    ]

    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    second_path.write_text(json.dumps(second_payload), encoding="utf-8")

    assert merge_coverage_units_from_paths([str(first_path), str(second_path)]) == {
        ("http://localhost:8000/app.js", 0, 10),
        ("http://localhost:8000/app.js", 10, 20),
    }


def test_jamel_root_resolves_repo_and_package_dir():
    envs_module = _load_envs_module()
    root = _require_jamel_root().resolve()

    assert envs_module._resolve_jamel_root(str(root)) == root
    assert envs_module._resolve_jamel_root(str(root / "jamel")) == root


def test_invalid_configured_jamel_root_fails(tmp_path):
    envs_module = _load_envs_module()

    with pytest.raises(RuntimeError, match="Invalid env.jamel.jamel_root"):
        envs_module._resolve_jamel_root(str(tmp_path / "missing"))


def test_worker_record_coverage_false_skips_jamel_import(tmp_path):
    envs_module = _load_envs_module()
    worker = envs_module.JAMELWorker(
        seed=0,
        env_kwargs={
            "record_coverage": False,
            "jamel_root": str(tmp_path / "missing"),
            "coverage_dir": str(tmp_path),
        },
    )
    worker.episode_dir = tmp_path
    worker.episode_step = 1

    assert worker._step_history_cls is None
    assert worker._coverage_reward_details_fn is None
    assert worker._get_step_coverage_path() is None
    assert worker._compute_coverage_reward_details(object()) == {
        "reward": 0.0,
        "previous_score": 0,
        "current_score": 0,
        "delta_score": 0,
        "skip_reason": "coverage_disabled",
    }


def test_imported_jamel_reward_signature_is_supported():
    envs_module = _load_envs_module()
    root = _require_jamel_root()

    _, reward_fn = envs_module._load_jamel_reward_components(str(root))
    params = inspect.signature(reward_fn).parameters

    assert "frozen_global_coverage_paths" in params
    assert "trajectory_history" in params


def test_worker_builds_step_history_for_imported_reward(tmp_path):
    envs_module = _load_envs_module()
    root = _require_jamel_root()
    worker = envs_module.JAMELWorker(
        seed=0,
        env_kwargs={
            "jamel_root": str(root),
            "record_coverage": True,
            "coverage_dir": str(tmp_path),
        },
    )
    worker.target_url = "http://localhost:8000/weibo/"
    worker.phase = "train"
    worker.round_id = 7
    worker.episode_dir = tmp_path
    worker.episode_step = 1
    coverage_path = tmp_path / "coverage_1.json"
    coverage_path.write_text("[]", encoding="utf-8")

    step_history = worker._build_step_history(
        action="click('12')",
        before_obs={"last_action": "noop()"},
        after_obs={"last_action": "click('12')", "last_action_error": "Success."},
        before_info={"before": True},
        after_info={"after": True},
        before_observation="before",
        after_observation="after",
        reward=0.0,
        done=False,
        terminated=False,
        truncated=False,
        coverage_path=coverage_path,
    )

    assert isinstance(step_history, worker._step_history_cls)
    assert step_history.extra_fields["coverage_path"] == str(coverage_path)
    assert step_history.after_obs["last_action"] == "click('12')"
    assert step_history.after_obs["last_action_error"] == "Success."
    assert step_history.parsed_content == {"action": "click('12')"}


def test_worker_coverage_path_does_not_require_trace(tmp_path):
    envs_module = _load_envs_module()
    root = _require_jamel_root()
    worker = envs_module.JAMELWorker(
        seed=0,
        env_kwargs={
            "jamel_root": str(root),
            "record_coverage": True,
            "coverage_dir": str(tmp_path),
        },
    )
    worker.episode_dir = tmp_path
    worker.trace_path = None
    worker.episode_step = 3

    assert worker._get_step_coverage_path() == tmp_path / "coverage_3.json"


def test_trace_columns_expose_monocart_scores_not_raw_range_counts():
    envs_module = _load_envs_module()

    assert "coverage_unit_count" not in envs_module.TRACE_COLUMNS
    assert "coverage_novel_unit_count" not in envs_module.TRACE_COLUMNS
    assert "trajectory_unit_count" not in envs_module.TRACE_COLUMNS
    assert "coverage_previous_score" in envs_module.TRACE_COLUMNS
    assert "coverage_current_score" in envs_module.TRACE_COLUMNS
    assert "coverage_delta_score" in envs_module.TRACE_COLUMNS
    assert "coverage_skip_reason" in envs_module.TRACE_COLUMNS
    assert envs_module._coverage_detail_value({"delta_score": 0.75}, "delta_score") == 0.75
    assert envs_module._coverage_detail_value({"delta_score": None}, "delta_score") == 0


def test_worker_compute_reward_details_calls_imported_reward(tmp_path):
    envs_module = _load_envs_module()
    root = _require_jamel_root()
    worker = envs_module.JAMELWorker(
        seed=0,
        env_kwargs={
            "jamel_root": str(root),
            "record_coverage": True,
            "coverage_dir": str(tmp_path),
        },
    )
    worker.current_frozen_coverage_paths = ["frozen.json"]
    worker.trajectory_history = ["previous-step"]
    calls = []
    expected_details = {
        "reward": 1.0,
        "previous_score": 2,
        "current_score": 5,
        "delta_score": 3,
        "skip_reason": None,
    }

    def fake_reward_details(current_step, frozen_global_coverage_paths=None, trajectory_history=None):
        calls.append((current_step, frozen_global_coverage_paths, trajectory_history))
        return expected_details

    worker._coverage_reward_details_fn = fake_reward_details
    current_step = object()

    assert worker._compute_coverage_reward_details(current_step) == expected_details
    assert calls == [(current_step, ["frozen.json"], ["previous-step"])]
