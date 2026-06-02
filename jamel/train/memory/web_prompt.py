"""Canonical web-agent prompt builder.

Single source of truth for the prompt format used by the JAMEL web
agent. Both training (`prepare_sft_dataset.py`) and inference
(`jamel/utils/eval/eval_memory_aug_episode.py`) MUST import from this
module and use `build_web_prompt()` to construct prompts. Any divergence is a
bug.

Design contract:
- Single-step observation only. No multi-step concatenation, no JSON memory
  blocks, no in-prompt action history. Long-term context flows through
  ``memory_tokens`` only.
- AXTree must be pruned via the shared ``prune_axtree(max_chars=8000)`` from
  ``jamel.core.env.web.axtree_utils`` BEFORE calling the builder.
- ``Current valid interactive element ids:`` list is derived from the pruned
  AXTree's bid set so that ids and tree are strictly aligned.
- Response template is ``<action>...</action>`` (no ``<think>``).
- The screenshot fed to the VLM is fixed to ``WEB_MODEL_IMAGE_SIZE``;
  the browser viewport stays 1280x720 to preserve responsive layout.
"""
from __future__ import annotations

import re

# ── Constants ────────────────────────────────────────────────────────────────

# Resize target applied to the screenshot just before the VLM processor (both
# training and inference). Viewport (BrowserGym) stays 1280x720 to preserve
# responsive DOM layout; we only shrink on the model side.
WEB_MODEL_IMAGE_SIZE: tuple[int, int] = (640, 360)

# Canonical BrowserGym action space. Must remain byte-identical between training
# and inference. Includes reset() so the agent can return to initial app state.
WEB_ACTION_SPACE = """\
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
reset()"""

# Canonical prompt template. All `\n` are real newlines. Curly braces are
# str.format placeholders; nothing else uses braces.
WEB_PROMPT_TEMPLATE = """\
You are an autonomous browser exploration agent.

This is session step {step_idx}.
Your goal is to explore the target app and maximize novel JavaScript execution coverage.

Target app: {target_app}
Start URL: {start_url}

Browser action space:
{action_space}

Current open pages URLs:
{open_urls}

Current Observation:
{pruned_axtree}

Current valid interactive element ids:
{element_ids}

The current webpage screenshot is:
<image>

Respond with exactly:
<action>one action</action>

The <action> content must be one single BrowserGym action call. If the action
uses a bid, use one exact id from the valid interactive element list. Never
invent bids and never combine two actions in one response."""


# ── Helpers ─────────────────────────────────────────────────────────────────-

# Matches `[bid] role 'label'` lines produced by browsergym's flatten_axtree_to_str.
_BID_LINE_RE = re.compile(r"\[(\d+)\]\s+([A-Za-z]+)\s+'([^']*)'")


def _format_open_urls(urls: object) -> str:
    """Format open_pages_urls as a Python tuple repr, matching Observer output.

    Accepts tuple/list/np.ndarray/str. If `urls` is already a tuple-repr string,
    return it as-is. Falls back to single-element tuple containing str(urls).
    """
    if isinstance(urls, str):
        s = urls.strip()
        if s.startswith("(") and s.endswith(")"):
            return s
        return repr((s,))
    try:
        items = tuple(str(u) for u in urls)  # type: ignore[arg-type]
        return repr(items)
    except TypeError:
        return repr((str(urls),))


def extract_element_ids(pruned_axtree: str, limit: int = 200) -> str:
    """Build the `Current valid interactive element ids:` block from a pruned AXTree.

    The bid set must be EXACTLY the bids that appear in `pruned_axtree`, in
    document order, so the model can ground its action onto a matching tree.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for raw in pruned_axtree.splitlines():
        m = _BID_LINE_RE.search(raw)
        if not m:
            continue
        bid, role, label = m.groups()
        if bid in seen:
            continue
        seen.add(bid)
        lines.append(f"- {bid}: {role.lower()} {label.strip()!r}")
        if len(lines) >= limit:
            lines.append(f"- ... {limit}+ ids omitted")
            break
    if not lines:
        return "(none)"
    return "\n".join(lines)


def build_web_prompt(
    *,
    step_idx: int,
    target_app: str,
    start_url: str,
    open_urls: object,
    pruned_axtree: str,
    element_ids: str | None = None,
) -> str:
    """Assemble the canonical web-agent prompt.

    Caller is responsible for calling `prune_axtree(max_chars=8000)` on the
    raw AXTree text BEFORE passing it here. If `element_ids` is None, it is
    derived from `pruned_axtree`.
    """
    if element_ids is None:
        element_ids = extract_element_ids(pruned_axtree)
    return WEB_PROMPT_TEMPLATE.format(
        step_idx=int(step_idx),
        target_app=str(target_app),
        start_url=str(start_url),
        action_space=WEB_ACTION_SPACE,
        open_urls=_format_open_urls(open_urls),
        pruned_axtree=pruned_axtree,
        element_ids=element_ids,
    )


def strip_think(response: str) -> str:
    """Remove `<think>...</think>` blocks from a response."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def extract_axtree_from_observation_str(obs_str: str) -> str:
    """Extract the AXTree segment from `before_observation_str` / Observer output.

    Observer.get_observation() emits a block ending with::

        Current Observation: \n{flatten_axtree_to_str(...)}

    We return everything after `Current Observation:` (trimmed). If the marker
    is missing, return the whole string (caller will prune anyway).
    """
    marker = "Current Observation:"
    idx = obs_str.rfind(marker)
    if idx == -1:
        return obs_str.strip()
    return obs_str[idx + len(marker):].strip()


__all__ = [
    "WEB_MODEL_IMAGE_SIZE",
    "WEB_ACTION_SPACE",
    "WEB_PROMPT_TEMPLATE",
    "build_web_prompt",
    "extract_element_ids",
    "extract_axtree_from_observation_str",
    "strip_think",
]
