from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import random
import re
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image

from jamel.core.env.web import Observer, get_environment, stop_envrionment
from jamel.coverage_artifact import (
    build_coverage_artifact_fields,
    coverage_artifact_extra_fields,
)
from jamel.core.env.web.utils import StepHistory
from jamel.core.reward.web.reward_funcs import (
    jamel_reward_fn_web_coverage_details,
)
from jamel.core.reward.web.utils import dedupe_coverage_paths
from jamel.log import log_utils
from jamel.weak_model_labeling_utils import (
    extract_action_response,
    is_action_execution_valid,
    select_successful_prefix,
)

logger = log_utils.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RELEASE_SCALEWOB_ROOT = REPO_ROOT / "env" / "browser_env" / "scalewob-env"
DEFAULT_SCALEWOB_ROOT = Path(
    os.environ.get(
        "SCALEWOB_ROOT",
        str(_DEFAULT_RELEASE_SCALEWOB_ROOT),
    )
)

ACTION_SPACE = """
noop(wait_ms: float = 1000)
send_msg_to_user(text: str)
report_infeasible(reason: str)
scroll(delta_x: float, delta_y: float)
fill(bid: str, value: str)
select_option(bid: str, options: str | list[str])
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
hover(bid: str)
press(bid: str, key_comb: str)
focus(bid: str)
clear(bid: str)
drag_and_drop(from_bid: str, to_bid: str)
upload_file(bid: str, file: str | list[str])
tab_close()
tab_focus(index: int)
new_tab()
go_back()
go_forward()
goto(url: str)
""".strip()

ACTION_NAMES = tuple(
    line.split("(", 1)[0].strip()
    for line in ACTION_SPACE.splitlines()
    if line.strip() and "(" in line
)

PROMPT_TEMPLATE = """
You are a weak autonomous browser exploration model.
Your goal is to explore the target app and maximize novel JavaScript execution coverage.

Target app: {target_app}
Start URL: {start_url}

Action space:
{action_space}

Current observation:
{observation}

Current valid interactive element ids:
{interactive_elements}

Respond with exactly:
<think>short reason</think><action>one action</action>

The <action> content must be one single BrowserGym action call. If the action
uses a bid, use one exact id from the valid interactive element list. Never
invent bids and never combine two actions in one response.
""".strip()


class LocalStaticServer:
    def __init__(
        self,
        root_dir: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        port_search: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.host = host
        self.port = port
        self.port_search = port_search
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        if self._server is not None:
            return self.port

        last_error: OSError | None = None
        ports = [self.port]
        if self.port_search:
            ports.extend(range(self.port + 1, self.port + 100))

        handler = partial(SimpleHTTPRequestHandler, directory=str(self.root_dir))
        for candidate_port in ports:
            try:
                self._server = ThreadingHTTPServer((self.host, candidate_port), handler)
                self.port = candidate_port
                break
            except OSError as exc:
                last_error = exc
                continue

        if self._server is None:
            raise RuntimeError(f"Failed to start static server: {last_error}")

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="weak-label-scalewob-static-server",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _parse_apps(raw_apps: str) -> list[str]:
    stripped = raw_apps.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in stripped.split(",") if item.strip()]


def _start_url(host: str, port: int, app: str) -> str:
    return f"http://{host}:{port}/{app}/index.html"


def _normalize_label(label: str) -> str:
    compact = " ".join(label.lower().split())
    compact = re.sub(r"\d+", "#", compact)
    return compact[:80]


def _extract_axtree_text(observation: str) -> str:
    marker = "Current Observation:"
    if marker not in observation:
        return observation.strip()
    return observation.split(marker, 1)[-1].strip()


def _state_signature(observation: str) -> str:
    lines: list[str] = []
    for raw_line in _extract_axtree_text(observation).splitlines():
        line = raw_line.strip()
        if any(keyword in line.lower() for keyword in ("button", "link", "tab", "search", "textbox", "menu")):
            lines.append(re.sub(r"\[\d+\]", "[]", line))
    return "\n".join(lines[:80])


def _extract_interactive_elements(observation: str) -> list[dict[str, str]]:
    elements: list[dict[str, str]] = []
    pattern = re.compile(r"\[(\d+)\]\s+([A-Za-z]+)\s+'([^']*)'")
    for raw_line in observation.splitlines():
        line = raw_line.strip()
        match = pattern.search(line)
        if not match:
            continue
        bid, role, label = match.groups()
        elements.append(
            {
                "bid": bid,
                "role": role.lower(),
                "label": label.strip(),
                "line": line,
            }
        )
    return elements


def _format_interactive_elements(observation: str, limit: int = 120) -> str:
    elements = _extract_interactive_elements(observation)
    if not elements:
        return "(none)"
    lines = [f"- {item['bid']}: {item['role']} {item['label']!r}" for item in elements[:limit]]
    if len(elements) > limit:
        lines.append(f"- ... {len(elements) - limit} more omitted")
    return "\n".join(lines)


def _parse_action_call(action: str) -> ast.Call | None:
    try:
        parsed = ast.parse(action.strip(), mode="eval")
    except SyntaxError:
        return None
    if not isinstance(parsed.body, ast.Call):
        return None
    if not isinstance(parsed.body.func, ast.Name):
        return None
    if parsed.body.func.id not in ACTION_NAMES:
        return None
    return parsed.body


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword_str(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _literal_str(keyword.value)
    return None


def _action_referenced_bids(action: str) -> tuple[list[str], str | None]:
    call = _parse_action_call(action)
    if call is None:
        return [], "action is not a single supported BrowserGym call"
    if not isinstance(call.func, ast.Name):
        return [], "action function is invalid"
    name = call.func.id

    def positional(index: int) -> str | None:
        if len(call.args) <= index:
            return None
        return _literal_str(call.args[index])

    if name in {"click", "dblclick", "hover", "focus", "clear", "fill", "press", "select_option", "upload_file"}:
        bid = positional(0) or _keyword_str(call, "bid")
        if not bid:
            return [], f"{name} action is missing a string bid"
        return [bid], None
    if name == "drag_and_drop":
        from_bid = positional(0) or _keyword_str(call, "from_bid")
        to_bid = positional(1) or _keyword_str(call, "to_bid")
        missing = [label for label, bid in (("from_bid", from_bid), ("to_bid", to_bid)) if not bid]
        if missing:
            return [], f"drag_and_drop action is missing {', '.join(missing)}"
        return [str(from_bid), str(to_bid)], None
    return [], None


def _action_validation_error(action: str, observation: str) -> str | None:
    referenced_bids, parse_error = _action_referenced_bids(action)
    if parse_error:
        return parse_error
    valid_bids = {item["bid"] for item in _extract_interactive_elements(observation)}
    invalid_bids = [bid for bid in referenced_bids if bid not in valid_bids]
    if invalid_bids:
        return f"action references missing bid(s): {', '.join(invalid_bids)}"
    return None


def _site_fill_values(target_app: str) -> list[str]:
    target = target_app.lower()
    if target in {"agoda", "airbnb", "trip", "huazhu"}:
        return ["Tokyo", "Paris", "2 guests", "2026-05-01"]
    if target in {"douban", "bilibili", "weibo", "ximalaya", "qqmusic"}:
        return ["movie", "travel", "music", "OpenAI"]
    if target == "wikipedia":
        return ["Artificial intelligence", "Travel", "China", "Machine learning"]
    return ["OpenAI", "Travel", "News", "Music"]


@dataclass
class WeakHeuristicModel:
    target_app: str
    start_url: str
    rng: random.Random
    visited_bids: set[str] = field(default_factory=set)
    clicked_bids: set[str] = field(default_factory=set)
    tried_state_actions: set[tuple[str, str]] = field(default_factory=set)
    tried_action_signatures: set[str] = field(default_factory=set)
    rewarded_action_signatures: set[str] = field(default_factory=set)
    failed_action_signatures: set[str] = field(default_factory=set)
    no_reward_streak: int = 0
    cursor: int = 0
    fill_cursor: int = 0
    pending_text_bid: str | None = None
    submitted_text_bids: set[str] = field(default_factory=set)

    def _action_signature(self, action: str, observation: str) -> str:
        elements = {item["bid"]: item for item in _extract_interactive_elements(observation)}
        bid_match = re.search(r"['\"]?(\d+)['\"]?", action)
        bid = bid_match.group(1) if bid_match else ""
        element = elements.get(bid, {})
        action_type = action.split("(", 1)[0]
        role = element.get("role", "")
        label = _normalize_label(element.get("label", ""))
        if action_type in {"click", "hover", "fill", "press", "dblclick"}:
            return f"{action_type}:{role}:{label}"
        return action_type

    def _element_score(self, element: dict[str, str]) -> float:
        role = element["role"]
        label = element["label"].lower()
        bid = element["bid"]
        semantic_click = f"click:{role}:{_normalize_label(element['label'])}"
        score = 0.0
        if bid not in self.visited_bids:
            score += 5.0
        if semantic_click not in self.tried_action_signatures:
            score += 10.0
        if semantic_click in self.rewarded_action_signatures:
            score -= 20.0
        if semantic_click in self.failed_action_signatures:
            score -= 8.0
        if bid not in self.clicked_bids:
            score += 2.0

        score += {
            "button": 10.0,
            "link": 9.0,
            "textbox": 8.0,
            "searchbox": 8.0,
            "combobox": 7.0,
            "tab": 7.0,
            "menuitem": 6.0,
            "heading": 2.0,
            "generic": -4.0,
            "image": -4.0,
        }.get(role, 1.0)

        for keyword, bonus in {
            "search": 6.0,
            "next": 5.0,
            "discover": 4.0,
            "profile": 4.0,
            "setting": 4.0,
            "message": 3.0,
            "video": 3.0,
            "travel": 3.0,
            "home": -18.0,
            "logo": -18.0,
        }.items():
            if keyword in label:
                score += bonus
        return score

    def choose_action(self, observation: str, history: list[StepHistory]) -> str:
        if self.pending_text_bid:
            action = f"press('{self.pending_text_bid}', 'Enter')"
            self.pending_text_bid = None
            return action

        elements = _extract_interactive_elements(observation)
        state_signature = _state_signature(observation)
        candidates: list[tuple[float, str]] = []
        fill_values = _site_fill_values(self.target_app)

        for element in elements:
            bid = element["bid"]
            role = element["role"]
            base = self._element_score(element)
            if role in {"button", "link", "tab", "menuitem", "heading", "generic", "image"}:
                candidates.append((base + 1.0, f"click('{bid}')"))
                candidates.append((base - 2.0, f"hover('{bid}')"))
            if role in {"textbox", "searchbox", "combobox"}:
                value = fill_values[self.fill_cursor % len(fill_values)]
                if bid not in self.submitted_text_bids:
                    candidates.append((base + 4.0, f"click('{bid}')"))
                    candidates.append((base + 9.0, f"fill('{bid}', '{value}')"))

        if self.no_reward_streak >= 4:
            candidates.extend(
                [
                    (11.0, "scroll(0, 900)"),
                    (10.0, "scroll(0, -900)"),
                    (9.0, "go_back()"),
                    (8.0, "go_forward()"),
                    (8.0, f"goto('{self.start_url}')"),
                ]
            )
        else:
            candidates.extend([(4.0, "scroll(0, 700)"), (2.0, "scroll(0, -500)")])

        filtered: list[tuple[float, str]] = []
        last_action = ""
        if history:
            last_action = str(history[-1].parsed_content.get("action") or "")
        for score, action in candidates:
            signature = self._action_signature(action, observation)
            if (state_signature, signature) in self.tried_state_actions:
                continue
            if action == last_action:
                continue
            filtered.append((score, action))

        if not filtered:
            fallback = [
                "scroll(0, 800)",
                "scroll(0, -600)",
                "go_back()",
                f"goto('{self.start_url}')",
                "noop(300)",
            ]
            action = fallback[self.cursor % len(fallback)]
        else:
            max_score = max(score for score, _ in filtered)
            top = [action for score, action in filtered if score >= max_score - 2.0]
            action = top[self.cursor % len(top)]

        self.cursor += 1
        if action.startswith("fill("):
            self.fill_cursor += 1
        return action

    def respond(
        self,
        prompt: str,
        observation: str,
        history: list[StepHistory],
        obs: dict[str, Any] | None = None,
    ) -> str:
        action = self.choose_action(observation, history)
        return (
            "<think>I will try a simple valid browser action that reaches a less explored UI state.</think>"
            f"<action>{action}</action>"
        )

    def update(self, action: str, reward: float, before_observation: str) -> None:
        signature = self._action_signature(action, before_observation)
        self.tried_action_signatures.add(signature)
        self.tried_state_actions.add((_state_signature(before_observation), signature))
        bid_match = re.search(r"\(\"?([0-9]+)\"?\)", action)
        if bid_match:
            bid = bid_match.group(1)
            self.visited_bids.add(bid)
            if action.startswith(("click(", "dblclick(")):
                self.clicked_bids.add(bid)
            if action.startswith("fill("):
                self.pending_text_bid = bid
            if action.startswith("press("):
                self.submitted_text_bids.add(bid)
        if reward > 0:
            self.rewarded_action_signatures.add(signature)
            self.no_reward_streak = 0
        else:
            self.failed_action_signatures.add(signature)
            self.no_reward_streak += 1


@dataclass
class OpenAICompatibleWeakModel:
    model_name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: int
    use_screenshot: bool = False

    def _screenshot_data_url(self, obs: dict[str, Any] | None) -> str | None:
        if not self.use_screenshot or not isinstance(obs, dict):
            return None
        screenshot = obs.get("screenshot")
        if screenshot is None:
            return None
        image = Image.fromarray(screenshot.astype("uint8"))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _repair_response(self, content: str) -> str:
        think, action, valid = extract_action_response(content)
        if valid:
            return content

        action_pattern = "|".join(re.escape(name) for name in ACTION_NAMES)
        match = re.search(rf"\b({action_pattern})\s*\([^\n<>`]*\)", content, flags=re.DOTALL)
        repaired_action = match.group(0).strip() if match else content.strip().splitlines()[-1].strip()
        repaired_action = repaired_action.strip("` ")
        return (
            f"<think>{think or 'Choose a valid exploratory browser action.'}</think>"
            f"<action>{repaired_action}</action>"
        )

    def respond(
        self,
        prompt: str,
        observation: str,
        history: list[StepHistory],
        obs: dict[str, Any] | None = None,
    ) -> str:
        recent_history = []
        for step in history[-8:]:
            extra = step.extra_fields or {}
            recent_history.append(
                {
                    "step": int(step.step),
                    "action": extra.get("action") or step.parsed_content.get("action"),
                    "reward": float(step.reward or 0.0),
                    "coverage_delta_score": int(extra.get("coverage_delta_score", 0) or 0),
                    "last_action_error": str((step.after_obs or {}).get("last_action_error") or ""),
                }
            )
        system_prompt = (
            "You are a weak but real browser exploration model. Output exactly "
            "<think>...</think><action>...</action>. The action must be one BrowserGym "
            "action call. Do not output JSON or markdown. Use only exact bid ids shown "
            "in the current valid interactive element list. Never invent ids and never "
            "put two actions in one <action>. Explore app functionality, not just text "
            "boxes. If you fill a search or comment input, submit it on the next "
            "action with press('<bid>', 'Enter') or click the visible search/send button. "
            "Do not repeatedly fill the same textbox with unrelated words. Prefer actions "
            "that open new panels, search results, comments, detail pages, tabs, settings, "
            "or submit forms."
        )
        user_prompt = (
            f"{prompt}\n\nRecent action history:\n"
            f"{json.dumps(recent_history, ensure_ascii=False, indent=2)}\n\n"
            "Choose the next single action that is most likely to reveal new UI state or "
            "new JavaScript coverage."
        )
        user_content: str | list[dict[str, Any]] = user_prompt
        screenshot_url = self._screenshot_data_url(obs)
        if screenshot_url:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": screenshot_url}},
            ]
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            if response.is_error:
                raise RuntimeError(
                    f"Weak model request failed: status={response.status_code}, "
                    f"url={url}, body={response.text[:1000]}"
                )
            data = response.json()
        content = str(data["choices"][0]["message"]["content"])
        return self._repair_response(content)

    def update(self, action: str, reward: float, before_observation: str) -> None:
        return None


def _arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _resolve_model_config(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    explicit_model = _arg(args, "model_name")
    model_name = explicit_model or _env_first("WEAK_MODEL_NAME", "MODEL_NAME", "UI_TARS_MODEL_NAME")
    ui_tars_model = os.getenv("UI_TARS_MODEL_NAME") or ""
    prefer_doubao = bool(ui_tars_model and model_name == ui_tars_model)

    base_url = _arg(args, "model_base_url")
    api_key = _arg(args, "model_api_key")
    if not base_url:
        if prefer_doubao:
            base_url = _env_first("WEAK_MODEL_BASE_URL", "DOUBAO_BASE_URL", "OPENAI_BASE_URL")
        else:
            base_url = _env_first("WEAK_MODEL_BASE_URL", "MODEL_BASE_URL", "OPENAI_BASE_URL", "DOUBAO_BASE_URL")
    if not api_key:
        if prefer_doubao:
            api_key = _env_first("WEAK_MODEL_API_KEY", "ARK_API_KEY", "OPENAI_API_KEY")
        else:
            api_key = _env_first("WEAK_MODEL_API_KEY", "MODEL_API_KEY", "OPENAI_API_KEY", "ARK_API_KEY")

    return {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": _arg(args, "model_temperature", 0.7),
        "max_tokens": _arg(args, "model_max_tokens", 512),
        "timeout": _arg(args, "model_timeout", 120),
        "use_screenshot": bool(_arg(args, "model_use_screenshot", False)),
    }


def _requested_policy(args: argparse.Namespace) -> str:
    return str(_arg(args, "policy", "heuristic"))


def _effective_policy(args: argparse.Namespace, model_config: dict[str, Any], *, strict: bool) -> str:
    requested = _requested_policy(args)
    if requested not in {"auto", "model", "heuristic"}:
        raise ValueError(f"Unsupported policy: {requested}")
    has_model_config = bool(model_config["model"] and model_config["base_url"] and model_config["api_key"])
    if requested == "heuristic":
        return "heuristic"
    if requested == "model":
        if not has_model_config and strict:
            missing = [key for key in ("model", "base_url", "api_key") if not model_config[key]]
            raise ValueError(f"Model policy requested but missing config fields: {missing}")
        return "model"
    return "model" if has_model_config else "heuristic"


def _public_weak_model_config(args: argparse.Namespace, model_config: dict[str, Any]) -> dict[str, Any]:
    effective_policy = _effective_policy(args, model_config, strict=True)
    return {
        "requested_policy": _requested_policy(args),
        "effective_policy": effective_policy,
        "model": model_config["model"] if effective_policy == "model" else "heuristic_weak_model_v0",
        "base_url": model_config["base_url"] if effective_policy == "model" else None,
        "api_key": "set" if effective_policy == "model" and model_config["api_key"] else None,
        "temperature": model_config["temperature"] if effective_policy == "model" else None,
        "max_tokens": model_config["max_tokens"] if effective_policy == "model" else None,
        "timeout": model_config["timeout"] if effective_policy == "model" else None,
        "use_screenshot": model_config["use_screenshot"] if effective_policy == "model" else False,
        "invalid_action_retries": int(_arg(args, "model_invalid_action_retries", 0) or 0),
        "seed": _arg(args, "seed", None),
    }


def _build_weak_model(
    *,
    args: argparse.Namespace,
    target_app: str,
    start_url: str,
    episode_id: int,
):
    model_config = _resolve_model_config(args)
    if _effective_policy(args, model_config, strict=True) == "model":
        return OpenAICompatibleWeakModel(
            model_name=model_config["model"],
            base_url=model_config["base_url"],
            api_key=model_config["api_key"],
            temperature=float(model_config["temperature"]),
            max_tokens=int(model_config["max_tokens"]),
            timeout=int(model_config["timeout"]),
            use_screenshot=bool(model_config["use_screenshot"]),
        )
    return WeakHeuristicModel(
        target_app=target_app,
        start_url=start_url,
        rng=random.Random(args.seed + episode_id),
    )


def _extract_validated_action(response: str, observation: str) -> tuple[str, str, bool, str | None]:
    think, action, tag_valid = extract_action_response(response)
    if not tag_valid:
        return think, action, False, "response is missing non-empty <think>...</think><action>...</action>"
    validation_error = _action_validation_error(action, observation)
    if validation_error:
        return think, action, False, validation_error
    return think, action, True, None


def _respond_with_validated_action(
    *,
    weak_model: Any,
    prompt: str,
    observation: str,
    history: list[StepHistory],
    obs: dict[str, Any],
    args: argparse.Namespace,
    effective_policy: str,
) -> tuple[str, str, str, bool, str | None, int]:
    max_retries = int(_arg(args, "model_invalid_action_retries", 0) or 0)
    retry_count = 0
    current_prompt = prompt
    last_response = ""
    last_think = ""
    last_action = ""
    last_valid = False
    last_error: str | None = None

    while True:
        last_response = weak_model.respond(current_prompt, observation, history, obs)
        last_think, last_action, last_valid, last_error = _extract_validated_action(last_response, observation)
        if last_valid or effective_policy != "model" or retry_count >= max_retries:
            return last_response, last_think, last_action, last_valid, last_error, retry_count

        retry_count += 1
        current_prompt = (
            f"{prompt}\n\n"
            "Your previous response was invalid and was not executed.\n"
            f"Invalid response:\n{last_response}\n\n"
            f"Reason: {last_error}\n\n"
            "Return exactly one valid BrowserGym action call in <action>. "
            "Use one of these current bids if the action needs a bid:\n"
            f"{_format_interactive_elements(observation)}"
        )


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _default_state(target_app: str, start_url: str, weak_model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target_app": target_app,
        "start_url": start_url,
        "weak_model_config": weak_model_config,
        "round_id": 0,
        "next_episode_id": 0,
        "global_step": 0,
        "exhausted": False,
        "exhausted_reason": None,
        "cumulative_coverage_manifest": [],
        "accepted_prefix_records": [],
        "exhaustion": {
            "no_positive_episode_count": 0,
            "recent_coverage_growth_window": [],
            "valid_action_rate_window": [],
            "repeated_action_window": [],
        },
    }


def _coverage_baseline_paths(state: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in state.get("cumulative_coverage_manifest", []):
        path = item.get("coverage_path")
        if path and Path(path).exists():
            paths.append(str(Path(path).resolve()))
    return dedupe_coverage_paths(paths)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _update_exhaustion(
    state: dict[str, Any],
    *,
    positive_count: int,
    coverage_growth: int,
    valid_action_rate: float,
    actions: list[str],
    no_positive_limit: int,
    growth_window_size: int,
    growth_threshold: int,
    valid_action_rate_threshold: float,
) -> None:
    exhaustion = state.setdefault("exhaustion", {})
    if positive_count > 0:
        exhaustion["no_positive_episode_count"] = 0
    else:
        exhaustion["no_positive_episode_count"] = int(exhaustion.get("no_positive_episode_count", 0)) + 1

    growth_window = list(exhaustion.get("recent_coverage_growth_window", []))
    growth_window.append(int(coverage_growth))
    exhaustion["recent_coverage_growth_window"] = growth_window[-growth_window_size:]

    valid_window = list(exhaustion.get("valid_action_rate_window", []))
    valid_window.append(float(valid_action_rate))
    exhaustion["valid_action_rate_window"] = valid_window[-growth_window_size:]

    action_window = list(exhaustion.get("repeated_action_window", []))
    action_window.extend(actions)
    exhaustion["repeated_action_window"] = action_window[-growth_window_size:]

    if exhaustion["no_positive_episode_count"] >= no_positive_limit:
        state["exhausted"] = True
        state["exhausted_reason"] = f"no_positive_episode_count>={no_positive_limit}"
        return

    if len(growth_window) >= growth_window_size and sum(growth_window[-growth_window_size:]) < growth_threshold:
        state["exhausted"] = True
        state["exhausted_reason"] = f"coverage_growth_last_{growth_window_size}<{growth_threshold}"
        return

    if len(valid_window) >= growth_window_size:
        recent_valid_rate = sum(valid_window[-growth_window_size:]) / growth_window_size
        recent_actions = action_window[-growth_window_size:]
        repeated_ratio = 0.0
        if recent_actions:
            repeated_ratio = 1.0 - (len(set(recent_actions)) / len(recent_actions))
        if recent_valid_rate < valid_action_rate_threshold and repeated_ratio > 0.8:
            state["exhausted"] = True
            state["exhausted_reason"] = "low_valid_action_rate_and_repeated_actions"


def export_accepted_sft_samples(output_root: Path) -> dict[str, Any]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for accepted_path in sorted(output_root.glob("apps/*/accepted/accepted_prefixes.jsonl")):
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            trace_path = Path(record["trace_path"])
            last_step = int(record["last_step"])
            if not trace_path.exists():
                logger.warning("Skip accepted prefix with missing trace", trace_path=str(trace_path))
                continue
            df = pd.read_parquet(trace_path)
            for _, row in df.iterrows():
                step = int(row.get("step", 0))
                if step > last_step:
                    continue
                rows.append(
                    {
                        "prompt": row.get("prompt"),
                        "response": row.get("response"),
                        "target_app": row.get("target_app") or record.get("target_app"),
                        "episode_id": int(row.get("episode_id", record.get("episode_id", 0))),
                        "step": step,
                        "reward": float(row.get("reward", 0.0) or 0.0),
                        "coverage_delta_score": int(row.get("coverage_delta_score", 0) or 0),
                        "coverage_sha256": row.get("coverage_sha256"),
                        "coverage_path": row.get("coverage_path"),
                        "checkpoint_id": row.get("checkpoint_id") or record.get("checkpoint_id"),
                        "trace_path": str(trace_path),
                    }
                )

    sft_dir = output_root / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = sft_dir / "accepted_samples.parquet"
    jsonl_path = sft_dir / "accepted_samples.jsonl"
    pd.DataFrame(rows).to_parquet(parquet_path)
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "samples": len(rows),
        "parquet_path": str(parquet_path.resolve()),
        "jsonl_path": str(jsonl_path.resolve()),
    }
    (sft_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _run_episode(
    *,
    target_app: str,
    start_url: str,
    app_dir: Path,
    output_root: Path,
    episode_id: int,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode_dir = app_dir / "raw" / f"episode_{episode_id:06d}"
    coverage_dir = episode_dir / "coverage"
    checkpoint_id = f"{target_app}-{episode_id:06d}"
    baseline_paths = _coverage_baseline_paths(state)

    env_context = get_environment(
        start_url,
        headless=not args.show_browser,
        record_coverage=True,
        timeout=args.timeout,
    )
    history: list[StepHistory] = []
    actions: list[str] = []
    weak_model = _build_weak_model(
        args=args,
        target_app=target_app,
        start_url=start_url,
        episode_id=episode_id,
    )
    model_config = _resolve_model_config(args)
    effective_policy = _effective_policy(args, model_config, strict=True)
    try:
        obs, info = env_context.obs, env_context.info
        for step in range(1, args.steps_per_episode + 1):
            before_obs = obs
            before_info = info
            before_observation = Observer.get_observation(before_obs)
            prompt = PROMPT_TEMPLATE.format(
                target_app=target_app,
                start_url=start_url,
                action_space=ACTION_SPACE,
                observation=before_observation,
                interactive_elements=_format_interactive_elements(before_observation),
            )
            (
                response,
                think,
                action,
                action_format_valid,
                action_validation_error,
                model_retry_attempts,
            ) = _respond_with_validated_action(
                weak_model=weak_model,
                prompt=prompt,
                observation=before_observation,
                history=history,
                obs=before_obs,
                args=args,
                effective_policy=effective_policy,
            )
            actions.append(action)

            if action_format_valid:
                after_obs, raw_reward, terminated, truncated, after_info = env_context.env.step(action)
            else:
                after_obs, raw_reward, terminated, truncated, after_info = before_obs, 0.0, False, False, before_info
            after_observation = Observer.get_observation(after_obs)

            coverage_path = coverage_dir / f"coverage_{step}.json"
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            if env_context.save_step_coverage is not None:
                env_context.save_step_coverage(coverage_path, step)

            artifact_fields = build_coverage_artifact_fields(coverage_path)
            action_execution_valid = bool(action_format_valid and is_action_execution_valid(after_obs))
            extra_fields = {
                "prompt": prompt,
                "response": response,
                "think": think,
                "action": action,
                "action_format_valid": action_format_valid,
                "action_validation_error": action_validation_error,
                "action_execution_valid": action_execution_valid,
                "model_retry_attempts": model_retry_attempts,
                "weak_policy": effective_policy,
                "weak_model_name": model_config["model"] if effective_policy == "model" else "heuristic_weak_model_v0",
                "target_app": target_app,
                "target_url": start_url,
                "start_url": start_url,
                "episode_id": episode_id,
                "global_step": int(state.get("global_step", 0)) + 1,
                "checkpoint_id": checkpoint_id,
                "coverage_path": str(coverage_path),
                **coverage_artifact_extra_fields(artifact_fields),
            }

            step_history = StepHistory(
                before_obs=before_obs,
                after_obs=after_obs,
                before_info=before_info,
                after_info=after_info,
                before_observation=before_observation,
                after_observation=after_observation,
                step=step,
                reward=0.0,
                raw_content=response,
                memory_content=None,
                parsed_content={"think": think, "action": action},
                result={
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "raw_reward": float(raw_reward or 0.0),
                },
                timestamp=datetime.now().isoformat(timespec="seconds"),
                extra_fields=extra_fields,
            )

            reward_details = jamel_reward_fn_web_coverage_details(
                step_history,
                frozen_global_coverage_paths=baseline_paths,
                trajectory_history=history,
            )
            reward = float(reward_details["reward"])
            if not artifact_fields["coverage_exists_at_write"]:
                reward = 0.0
                reward_details["reward"] = 0.0
                reward_details["skip_reason"] = reward_details.get("skip_reason") or "missing_coverage_at_write"

            step_history.reward = reward
            step_history.result["reward_source"] = "coverage" if reward > 0 else "none"
            step_history.extra_fields.update(
                {
                    "reward_source": step_history.result["reward_source"],
                    "coverage_previous_score": int(reward_details.get("previous_score", 0) or 0),
                    "coverage_current_score": int(reward_details.get("current_score", 0) or 0),
                    "coverage_delta_score": int(reward_details.get("delta_score", 0) or 0),
                    "coverage_skip_reason": reward_details.get("skip_reason"),
                }
            )
            weak_model.update(action, reward, before_observation)
            history.append(step_history)
            state["global_step"] = int(state.get("global_step", 0)) + 1

            obs = after_obs
            info = after_info
            if terminated or truncated:
                break

        trace_path = Observer.save_trajectory(
            history=history,
            history_dir=str(episode_dir),
            filename="trace.parquet",
            metadata={
                "target_app": target_app,
                "start_url": start_url,
                "episode_id": episode_id,
                "checkpoint_id": checkpoint_id,
                "steps": len(history),
            },
        )
        trace_path_obj = Path(trace_path) if trace_path else episode_dir / "trace.parquet"

        positive_manifest_entries: list[dict[str, Any]] = []
        for step_history in history:
            extra = step_history.extra_fields or {}
            if float(step_history.reward or 0.0) <= 0:
                continue
            if not extra.get("coverage_exists_at_write"):
                continue
            entry = {
                "episode_id": episode_id,
                "step": int(step_history.step),
                "coverage_path": str(Path(str(extra["coverage_path"])).resolve()),
                "coverage_sha256": extra.get("coverage_sha256"),
                "coverage_size_bytes": int(extra.get("coverage_size_bytes", 0) or 0),
                "reward": float(step_history.reward),
                "coverage_delta_score": int(extra.get("coverage_delta_score", 0) or 0),
            }
            positive_manifest_entries.append(entry)

        state.setdefault("cumulative_coverage_manifest", []).extend(positive_manifest_entries)
        last_prefix_step, prefix_reason = select_successful_prefix(history)
        accepted_record = None
        if last_prefix_step is not None:
            accepted_record = {
                "episode_id": episode_id,
                "last_step": last_prefix_step,
                "trace_path": str(trace_path_obj.resolve()),
                "checkpoint_id": checkpoint_id,
                "target_app": target_app,
            }
            state.setdefault("accepted_prefix_records", []).append(accepted_record)
            _append_jsonl(app_dir / "accepted" / "accepted_prefixes.jsonl", accepted_record)

        positive_steps = [int(step.step) for step in history if float(step.reward or 0.0) > 0]
        coverage_growth = sum(item["coverage_delta_score"] for item in positive_manifest_entries)
        valid_action_count = sum(1 for step in history if (step.extra_fields or {}).get("action_execution_valid"))
        valid_action_rate = valid_action_count / max(1, len(history))
        _update_exhaustion(
            state,
            positive_count=len(positive_manifest_entries),
            coverage_growth=coverage_growth,
            valid_action_rate=valid_action_rate,
            actions=actions,
            no_positive_limit=args.no_positive_exhaustion,
            growth_window_size=args.coverage_growth_window,
            growth_threshold=args.coverage_growth_threshold,
            valid_action_rate_threshold=args.valid_action_rate_threshold,
        )
        state["next_episode_id"] = episode_id + 1

        return {
            "target_app": target_app,
            "episode_id": episode_id,
            "trace_path": str(trace_path_obj.resolve()),
            "checkpoint_id": checkpoint_id,
            "steps": len(history),
            "positive_steps": positive_steps,
            "accepted_prefix_last_step": last_prefix_step,
            "accepted_prefix_reason": prefix_reason,
            "coverage_files": sum(1 for step in history if Path(str((step.extra_fields or {}).get("coverage_path"))).exists()),
            "coverage_embedded_steps": sum(1 for step in history if (step.extra_fields or {}).get("coverage_exists_at_write")),
            "valid_action_rate": valid_action_rate,
            "coverage_growth": coverage_growth,
            "exhausted": bool(state.get("exhausted")),
            "accepted_record": accepted_record,
        }
    finally:
        stop_envrionment(
            env_context,
            record_coverage=True,
            record_coverage_path=coverage_dir / "coverage_final.json",
        )


def run_labeling(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scalewob_root = Path(args.scalewob_root).resolve()
    if not scalewob_root.exists():
        raise FileNotFoundError(f"ScaleWoB root not found: {scalewob_root}")

    server = None
    port = args.port
    if args.use_existing_server:
        if not _port_is_open(args.host, args.port):
            raise RuntimeError(f"--use-existing-server was set, but {args.host}:{args.port} is not reachable")
    else:
        server = LocalStaticServer(
            scalewob_root,
            host=args.host,
            port=args.port,
            port_search=not args.no_port_search,
        )
        port = server.start()

    model_config = _resolve_model_config(args)
    weak_model_config = _public_weak_model_config(args, model_config)
    logger.info(
        "Weak labeling policy resolved",
        requested_policy=weak_model_config["requested_policy"],
        effective_policy=weak_model_config["effective_policy"],
        model=weak_model_config["model"],
        use_screenshot=weak_model_config["use_screenshot"],
    )

    apps = _parse_apps(args.apps)
    results: list[dict[str, Any]] = []
    try:
        for target_app in apps:
            start_url = _start_url(args.host, port, target_app)
            app_dir = output_root / "apps" / target_app
            checkpoint_path = app_dir / "checkpoints" / "exploration_state.json"
            if args.fresh and checkpoint_path.exists():
                checkpoint_path.unlink()
            state = _load_json(checkpoint_path, _default_state(target_app, start_url, weak_model_config))
            state["start_url"] = start_url
            state["weak_model_config"] = weak_model_config

            if state.get("exhausted") and not args.ignore_exhausted:
                logger.info("Skip exhausted app", target_app=target_app, checkpoint=str(checkpoint_path))
                continue

            if getattr(args, "target_episodes_per_app", None) is not None:
                current_episode_id = int(state.get("next_episode_id", 0))
                episodes_to_run = max(0, int(args.target_episodes_per_app) - current_episode_id)
                logger.info(
                    "Resolved target episode budget",
                    target_app=target_app,
                    current_episode_id=current_episode_id,
                    target_episodes_per_app=args.target_episodes_per_app,
                    episodes_to_run=episodes_to_run,
                )
            else:
                episodes_to_run = int(args.episodes_per_app)

            for _ in range(episodes_to_run):
                if state.get("exhausted") and not args.ignore_exhausted:
                    break
                episode_id = int(state.get("next_episode_id", 0))
                episode_summary = _run_episode(
                    target_app=target_app,
                    start_url=start_url,
                    app_dir=app_dir,
                    output_root=output_root,
                    episode_id=episode_id,
                    state=state,
                    args=args,
                )
                results.append(episode_summary)
                checkpoint_payload = dict(state)
                _write_json_atomic(checkpoint_path, checkpoint_payload)
                manifest_entry = {
                    **{key: value for key, value in episode_summary.items() if key != "accepted_record"},
                    "checkpoint_path": str(checkpoint_path.resolve()),
                }
                _append_jsonl(output_root / "manifest.jsonl", manifest_entry)

        sft_export = export_accepted_sft_samples(output_root)
        root_state = {
            "schema_version": 1,
            "output_root": str(output_root),
            "apps": apps,
            "episodes": results,
            "sft_export": sft_export,
        }
        _write_json_atomic(output_root / "exploration_state.json", root_state)
        return root_state
    finally:
        if server is not None:
            server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect weak-model ScaleWoB SFT labeling trajectories.")
    parser.add_argument("--apps", default="weibo", help="Comma-separated app ids, e.g. weibo,agoda")
    parser.add_argument("--output-dir", default=None, help="Output root")
    parser.add_argument("--scalewob-root", default=str(DEFAULT_SCALEWOB_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-port-search", action="store_true")
    parser.add_argument("--use-existing-server", action="store_true")
    parser.add_argument("--episodes-per-app", type=int, default=30)
    parser.add_argument(
        "--target-episodes-per-app",
        type=int,
        default=None,
        help="Collect until each app has this total episode count; overrides --episodes-per-app append count.",
    )
    parser.add_argument("--steps-per-episode", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=["auto", "model", "heuristic"], default="auto")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument("--model-api-key", default=None)
    parser.add_argument("--model-temperature", type=float, default=0.7)
    parser.add_argument("--model-max-tokens", type=int, default=512)
    parser.add_argument("--model-timeout", type=int, default=120)
    parser.add_argument("--model-use-screenshot", action="store_true")
    parser.add_argument("--model-invalid-action-retries", type=int, default=2)
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--ignore-exhausted", action="store_true")
    parser.add_argument("--no-positive-exhaustion", type=int, default=30)
    parser.add_argument("--coverage-growth-window", type=int, default=100)
    parser.add_argument("--coverage-growth-threshold", type=int, default=1)
    parser.add_argument("--valid-action-rate-threshold", type=float, default=0.2)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    if not args.scalewob_root:
        args.scalewob_root = str(DEFAULT_SCALEWOB_ROOT)
    if not args.output_dir:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "outputs" / "weak_model_sft_labeling" / run_id)
    result = run_labeling(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
