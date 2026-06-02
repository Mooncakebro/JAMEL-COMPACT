"""
Shared prune_observation module.

Applies semantic pruning to AXTree / observation text before tokenization,
preserving interactive elements and semantic content while dropping
redundant structural lines.

Used by both:
  - prepare_sft_dataset.py  (compressor: prune → if overflow → truncate)
  - jamel_sft_dataset.py  (main model: prune → if overflow → discard)

Design principle (documented in TRAINING.md §4.4):
  - Compressor: prune first, keep as much as possible within context length;
    if still overflow, truncate chars further.
  - Main model: prune first; if pruned prompt still exceeds max_length,
    **discard the entire sample** — never train on truncated garbage.
"""

from __future__ import annotations

import re
from typing import List, Tuple


# ── line classifiers ──────────────────────────────────────────────────────

_INTERACTIVE_ROLES = {
    "button", "link", "textbox", "combobox", "checkbox", "radio",
    "slider", "option", "menuitem", "switch", "tab", "spinbutton",
    "listbox", "treeitem", "menuitemcheckbox", "menuitemradio",
}

_SEMANTIC_ROLES = {
    "heading", "paragraph", "StaticText", "label", "caption", "legend",
    "alert", "status", "log", "note",
}


def _has_bid(line: str) -> bool:
    """Check if a line contains a [bid] pattern at the start."""
    return bool(re.match(r'^\s*\[\d+\]', line))


def _classify_line(line: str) -> str:
    """Classify a line as 'interactive', 'semantic', or 'structural'."""
    stripped = line.strip()
    # Interactive: has bid AND interactive role
    if _has_bid(line):
        for role in _INTERACTIVE_ROLES:
            if role in stripped:
                return "interactive"
    # Semantic: heading, text content
    for role in _SEMANTIC_ROLES:
        if role in stripped:
            return "semantic"
    # Bidded but not interactive (images, containers) → keep as context
    if _has_bid(line):
        return "context"
    return "structural"


# ── core prune function ───────────────────────────────────────────────────

def prune_observation(text: str, max_chars: int = 4096) -> str:
    """
    Prune observation/AXTree text to fit within max_chars while preserving
    interactive elements and semantic content.

    Priority (highest → lowest):
      1. Interactive elements with bids (must keep)
      2. Semantic content (headings, paragraphs, StaticText)
      3. Context elements (bidded containers, images)
      4. Structural lines (dropped first when budget exceeded)

    Args:
        text: Raw observation or AXTree text.
        max_chars: Target max character count.

    Returns:
        Pruned text guaranteed ≤ max_chars (approximately).
    """
    if len(text) <= max_chars:
        return text

    lines = text.split("\n")
    classified: List[Tuple[int, str, str]] = []  # (orig_idx, line, class)

    for i, line in enumerate(lines):
        cls = _classify_line(line)
        classified.append((i, line, cls))

    # Gather lines by priority
    interactive = [(i, l) for i, l, c in classified if c == "interactive"]
    semantic = [(i, l) for i, l, c in classified if c == "semantic"]
    context = [(i, l) for i, l, c in classified if c == "context"]
    structural = [(i, l) for i, l, c in classified if c == "structural"]

    # Reconstruct within budget
    kept: List[Tuple[int, str]] = []
    budget = max_chars

    for group in (interactive, semantic, context, structural):
        for idx, line in group:
            cost = len(line) + 1  # +1 for newline
            if budget >= cost:
                kept.append((idx, line))
                budget -= cost
            else:
                break
        if budget <= 0:
            break

    # Sort back to original order
    kept.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in kept)


# ── utility for higher-level prompt pruning ───────────────────────────────
# Called by the dataset to prune the entire prompt (not just observation).

# Pattern: everything between the observation start marker and the end marker
# is the prune-able observation block.

_OBS_START_MARKERS = [
    "Current open pages",
    "Current observation:\n",
    "Current Observation:\n",
    "Your current observation is:\n",
]

_OBS_END_MARKERS = [
    "\nCurrent valid interactive element ids:",
    "\nRespond with exactly:",
    "\nNow take exactly",
    "\n\nThe current webpage screenshot is:\n<image>",
]


def prune_prompt(prompt: str, max_chars: int = 4096) -> str:
    """
    Prune a full prompt by compressing only the observation/AXTree block,
    preserving system prompt, valid ids, and response format instructions.

    This replaces the old _truncate_prompt_observation() with semantic pruning.
    """
    for start_marker in _OBS_START_MARKERS:
        if start_marker not in prompt:
            continue

        prefix, rest = prompt.split(start_marker, 1)

        for end_marker in _OBS_END_MARKERS:
            if end_marker not in rest:
                continue

            observation, suffix = rest.split(end_marker, 1)

            # Calculate budget for observation
            overhead = len(prefix) + len(start_marker) + len(end_marker) + len(suffix)
            obs_budget = max(0, max_chars - overhead)

            if obs_budget <= 0:
                # No budget left for observation — just keep start marker
                return f"{prefix}{start_marker}{end_marker}{suffix}"

            # Prune the observation block
            pruned_obs = prune_observation(observation, max_chars=obs_budget)
            return f"{prefix}{start_marker}{pruned_obs}{end_marker}{suffix}"

        # End marker not found — prune rest as observation
        obs_budget = max(0, max_chars - len(prefix) - len(start_marker))
        pruned_rest = prune_observation(rest, max_chars=obs_budget)
        return f"{prefix}{start_marker}{pruned_rest}"

    # No known markers — prune entire prompt
    return prune_observation(prompt, max_chars=max_chars)