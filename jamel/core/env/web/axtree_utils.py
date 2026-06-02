"""
AXTree pruning utilities for reducing prompt context window.

Shared by eval (eval_memory_aug_episode.py) and training data prep
(prepare_sft_dataset.py) to keep AXTree size under control.

Strategy:
  1. Keep every line with an interactive bid ([N] role 'label').
  2. For each interactive line, trace its ancestor chain (indentation-based)
     and keep structural ancestors (headings, landmarks, etc.).
  3. Also keep heading/labeltext lines even without bids (page structure).
  4. Merge consecutive StaticText lines into one to save tokens.
  5. Reduce indent from tabs (4-space equivalent) to 1 space per level.
  6. If the result still exceeds max_chars, fall back to minimal:
     RootWebArea header + interactive-element lines only.

Target: ~8000 chars (~2000-4000 tokens), max 16000.
"""
from __future__ import annotations

import re

# ── patterns ────────────────────────────────────────────────────────────────
_IE_PATTERN  = re.compile(r"\[(\d+)\]\s+([A-Za-z]+)\s+'([^']*)'")
_ST_CONTENT  = re.compile(r"^StaticText\s+(.+)$")

# ── structural roles that provide useful page context ───────────────────────
_STRUCTURAL: set[str] = {
    "heading", "labeltext", "banner", "navigation", "main", "region",
    "complementary", "contentinfo", "search", "article", "section",
    "list", "listitem", "table", "rowgroup", "row", "gridcell",
    "menu", "menubar", "menuitem", "toolbar", "tablist", "tab",
    "tabpanel", "dialog", "alert", "form", "group",
}


def prune_axtree(axtree_text: str, max_chars: int = 8000) -> str:
    """
    Prune AXTree text to reduce prompt length while keeping all interactive elements.

    Improvements over naive pruning:
      - Consecutive StaticText lines at the same indent are merged (| separated).
      - Indentation reduced from tabs to 1 space per level.
    """
    if not axtree_text or len(axtree_text) <= max_chars:
        return axtree_text

    lines = axtree_text.split("\n")
    n = len(lines)
    if n <= 1:
        return axtree_text

    # ── parse lines ─────────────────────────────────────────────────────────
    parsed: list[dict] = []
    for line in lines:
        stripped = line.lstrip("\t")
        indent = len(line) - len(stripped)
        m = _IE_PATTERN.search(stripped)
        parsed.append({
            "indent": indent,
            "text": stripped,
            "has_bid": m is not None,
            "role": m.group(2).lower() if m else None,
        })

    keep = [False] * n
    if n > 0:
        keep[0] = True  # RootWebArea header

    # ── mark interactive lines and their structural ancestors ───────────────
    for i, p in enumerate(parsed):
        if not p["has_bid"]:
            continue
        keep[i] = True
        target_indent = p["indent"] - 1
        for j in range(i - 1, -1, -1):
            if parsed[j]["indent"] == target_indent:
                role = parsed[j]["role"]
                if role and role in _STRUCTURAL:
                    keep[j] = True
                elif parsed[j]["indent"] <= 1:
                    keep[j] = True
                elif role is None:
                    txt_lower = parsed[j]["text"].lower()
                    if any(kw in txt_lower for kw in _STRUCTURAL):
                        keep[j] = True
                target_indent -= 1
                if target_indent < 0:
                    break

    # ── also keep heading / labeltext lines ─────────────────────────────────
    for i, p in enumerate(parsed):
        if not keep[i] and p["role"] in ("heading", "labeltext"):
            keep[i] = True

    # ── keep StaticText children of kept parents (will be merged) ─────────
    for i, p in enumerate(parsed):
        if keep[i]:
            continue
        if p["role"] is not None or p["has_bid"]:
            continue
        if not p["text"].startswith("StaticText"):
            continue
        # Find parent: nearest preceding line with strictly lower indent
        for j in range(i - 1, -1, -1):
            if parsed[j]["indent"] < p["indent"]:
                if keep[j]:
                    keep[i] = True
                break

    # ── build output with StaticText merging + indent reduction ─────────────
    output_lines: list[str] = []
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue

        p = parsed[i]

        # Detect StaticText (no bid, no known role, starts with "StaticText")
        is_st = (
            p["role"] is None
            and not p["has_bid"]
            and p["text"].startswith("StaticText")
        )

        if is_st:
            # Collect consecutive kept StaticText at the *same indent* level
            st_contents: list[str] = []
            st_indent = p["indent"]
            j = i
            while j < n and keep[j]:
                pj = parsed[j]
                pj_is_st = (
                    pj["role"] is None
                    and not pj["has_bid"]
                    and pj["text"].startswith("StaticText")
                )
                if not pj_is_st or pj["indent"] != st_indent:
                    break
                m = _ST_CONTENT.match(pj["text"])
                if m:
                    raw = m.group(1).strip()
                    # Strip surrounding quotes: 'text' or "text"
                    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
                        raw = raw[1:-1]
                    st_contents.append(raw)
                j += 1

            indent_str = " " * st_indent
            if len(st_contents) == 1:
                output_lines.append(f"{indent_str}StaticText {st_contents[0]}")
            else:
                merged = " ".join(st_contents)
                output_lines.append(f"{indent_str}StaticText {merged}")
            i = j
        else:
            indent_str = " " * p["indent"]
            output_lines.append(f"{indent_str}{p['text']}")
            i += 1

    pruned = "\n".join(output_lines)

    # ── fallback: minimal interactive-only if still too long ────────────────
    # Only trigger when even merged StaticText result exceeds 2x the budget.
    if len(pruned) > max_chars * 2:
        minimal_lines: list[str] = []
        for i, p in enumerate(parsed):
            if i == 0:
                minimal_lines.append(p["text"])
            elif p["has_bid"]:
                minimal_lines.append(f"{' ' * p['indent']}{p['text']}")
        pruned = "\n".join(minimal_lines)

    return pruned


def prune_observation_text(obs_text: str, max_chars: int = 8000) -> str:
    """
    Prune the 'Current Observation:' section inside an Observer.get_observation() string.
    Only the AXTree block (after 'Current Observation:') is pruned; the metadata
    headers (Last Action, pages, etc.) are kept as-is.
    """
    marker = "Current Observation:"
    idx = obs_text.find(marker)
    if idx < 0:
        return prune_axtree(obs_text, max_chars)

    prefix = obs_text[:idx + len(marker)]
    axtree_block = obs_text[idx + len(marker):]
    pruned_axtree = prune_axtree(axtree_block.strip(), max_chars)
    return prefix + "\n" + pruned_axtree