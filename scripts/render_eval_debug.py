#!/usr/bin/env python3
"""Render complete per-step VLM debug frames and an optional scrolling video.

The renderer is backward compatible with existing JAMEL-COMPACT trajectories:
old rows use ``prompt`` as the canonical user prompt; newer rows additionally
show the exact processor-rendered chat template, token counts, previous action,
and compact recurrent-state summaries.

Examples:
    python scripts/render_eval_debug.py outputs/compact_eval/weibo/session0

    python scripts/render_eval_debug.py outputs/compact_eval/weibo/session0 \
        --video outputs/compact_eval/weibo/session0/vlm_debug.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#F7F7F5"
PANEL = "#FFFFFF"
BORDER = "#C8C8C8"
TEXT = "#202020"
MUTED = "#666666"
INPUT_COLOR = "#3568A8"
OUTPUT_COLOR = "#C65F32"
MEMORY_COLOR = "#6B4FA1"


def _font(size: int, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(34, bold=True)
SECTION_FONT = _font(24, bold=True)
MONO_FONT = _font(18, mono=True)
MONO_BOLD_FONT = _font(18, mono=True, bold=True)
SMALL_FONT = _font(16)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Reading parquet trajectories requires pandas and pyarrow.") from error
    dataframe = pd.read_parquet(path)
    return [
        {key: _clean_value(value) for key, value in row.items()}
        for row in dataframe.to_dict(orient="records")
    ]


def _discover_trajectory(session_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"Trajectory does not exist: {requested}")
        return requested
    candidates = list(session_dir.glob("trajectory_*.parquet"))
    candidates.extend(session_dir.glob("trajectory_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory_*.parquet or trajectory_*.jsonl found in {session_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _wrap_text(text: Any, width: int) -> list[str]:
    value = "" if text is None else str(text)
    lines = []
    for raw_line in value.expandtabs(4).splitlines() or [""]:
        wrapped = textwrap.wrap(
            raw_line,
            width=max(width, 1),
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    font: ImageFont.ImageFont,
    fill: str,
    chars_per_line: int,
    line_spacing: int = 5,
) -> int:
    x, y = xy
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + line_spacing
    for line in _wrap_text(text, chars_per_line):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _panel_height(text: Any, chars_per_line: int, font: ImageFont.ImageFont, padding: int = 28) -> int:
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 5
    return len(_wrap_text(text, chars_per_line)) * line_height + padding * 2 + 45


def _format_json(value: Any) -> str:
    if value in (None, "", []):
        return "(not recorded)"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _step_number(row: dict, row_index: int) -> int:
    try:
        return int(row.get("step", row_index)) + 1
    except (TypeError, ValueError):
        return row_index + 1


def _screenshot_path(session_dir: Path, step_number: int, kind: str) -> Path | None:
    path = session_dir / f"step_{step_number:03d}_{kind}.png"
    return path if path.is_file() else None


def _load_screenshot(path: Path | None, target_width: int) -> Image.Image:
    if path is None:
        image = Image.new("RGB", (target_width, int(target_width * 9 / 16)), "#262626")
        draw = ImageDraw.Draw(image)
        message = "Screenshot not saved\nRun eval with SAVE_SCREENSHOTS=1"
        bbox = draw.multiline_textbbox((0, 0), message, font=SECTION_FONT, align="center")
        draw.multiline_text(
            ((image.width - (bbox[2] - bbox[0])) / 2, (image.height - (bbox[3] - bbox[1])) / 2),
            message,
            font=SECTION_FONT,
            fill="#FFFFFF",
            align="center",
            spacing=10,
        )
        return image
    with Image.open(path) as source:
        screenshot = source.convert("RGB")
    target_height = max(1, round(screenshot.height * target_width / screenshot.width))
    return screenshot.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _draw_panel(
    canvas: Image.Image,
    x: int,
    y: int,
    width: int,
    title: str,
    text: Any,
    accent: str,
    chars_per_line: int,
    font: ImageFont.ImageFont = MONO_FONT,
) -> int:
    height = _panel_height(text, chars_per_line, font)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=14,
        fill=PANEL,
        outline=BORDER,
        width=2,
    )
    draw.rectangle((x, y, x + 9, y + height), fill=accent)
    draw.text((x + 28, y + 18), title, font=SECTION_FONT, fill=accent)
    _draw_wrapped(
        draw,
        (x + 28, y + 62),
        text,
        font=font,
        fill=TEXT,
        chars_per_line=chars_per_line,
    )
    return y + height


def _render_step(
    row: dict,
    row_index: int,
    session_dir: Path,
    output_path: Path,
    width: int,
) -> dict:
    margin = 36
    gap = 28
    left_width = int(width * 0.39)
    right_width = width - margin * 2 - gap - left_width
    right_chars = max(55, int(right_width / 10.5))
    left_chars = max(42, int(left_width / 10.5))
    step_number = _step_number(row, row_index)

    before_screenshot = _screenshot_path(session_dir, step_number, "before")
    after_screenshot = _screenshot_path(session_dir, step_number, "after")
    screenshot = _load_screenshot(before_screenshot, left_width)
    after_image = (
        _load_screenshot(after_screenshot, left_width)
        if after_screenshot is not None
        else None
    )

    exact_input = row.get("model_input_text")
    if exact_input:
        input_title = "Exact Processor-Rendered VLM Text Input"
        input_note = "Recorded after apply_chat_template()."
    else:
        exact_input = row.get("prompt", "")
        input_title = "Canonical User Prompt (Old Eval)"
        input_note = "Exact chat-template text was not recorded in this older trajectory."

    role_summary = (
        "System role message: none\n"
        "User role: exploration instructions + action space + pruned AXTree + <image>\n"
        "Assistant role: generation begins after the processor chat template"
    )
    if row.get("chat_messages_json"):
        role_summary += "\n\nRecorded messages:\n" + _format_json(row["chat_messages_json"])

    metadata = {
        "step": step_number,
        "episode_idx": row.get("episode_idx"),
        "target_url": row.get("target_url"),
        "start_url": row.get("start_url"),
        "previous_action": row.get("previous_action", "not recorded"),
        "input_image_size": row.get("input_image_size", "640x360 by eval configuration"),
        "input_tokens_before_truncation": row.get(
            "input_token_count_before_truncation", "not recorded"
        ),
        "model_input_tokens": row.get("model_input_token_count", "not recorded"),
        "input_truncated": row.get("input_truncated", "not recorded"),
    }
    output_text = (
        f"RAW RESPONSE\n{row.get('raw_response', '')}\n\n"
        f"PARSED THINK\n{row.get('think', '') or '(empty)'}\n\n"
        f"PARSED ACTION\n{row.get('action', '') or '(empty)'}\n\n"
        f"REWARD\nstep={row.get('reward', 0)}  delta_coverage={row.get('delta_score', 0)}  "
        f"cumulative={row.get('cumulative_reward', 0)}"
    )
    memory_text = (
        "BEFORE DECISION\n"
        + _format_json(row.get("memory_before_summary"))
        + "\n\nAFTER DECISION\n"
        + _format_json(row.get("memory_after_summary"))
    )

    right_panels = [
        (input_title, f"{input_note}\n\n{exact_input}", INPUT_COLOR, MONO_FONT),
        ("VLM Output and Environment Result", output_text, OUTPUT_COLOR, MONO_FONT),
    ]
    right_height = margin + 70
    for _, panel_text, _, font in right_panels:
        right_height += _panel_height(panel_text, right_chars, font) + gap

    left_height = margin + 70 + screenshot.height + gap
    if after_image is not None:
        left_height += after_image.height + gap
    left_height += _panel_height(_format_json(metadata), left_chars, MONO_FONT) + gap
    left_height += _panel_height(role_summary, left_chars, MONO_FONT) + gap
    if row.get("memory_before_summary") or row.get("memory_after_summary"):
        left_height += _panel_height(memory_text, left_chars, MONO_FONT) + gap
    canvas_height = max(right_height, left_height) + margin
    canvas = Image.new("RGB", (width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title = f"VLM Evaluation Debug — Step {step_number:03d}"
    draw.text((margin, margin), title, font=TITLE_FONT, fill=TEXT)
    draw.text(
        (width - margin, margin + 8),
        str(row.get("timestamp", "")),
        font=SMALL_FONT,
        fill=MUTED,
        anchor="ra",
    )

    left_x = margin
    left_y = margin + 70
    draw.rounded_rectangle(
        (left_x, left_y, left_x + left_width, left_y + screenshot.height),
        radius=14,
        fill="#222222",
        outline=BORDER,
        width=2,
    )
    canvas.paste(screenshot, (left_x, left_y))
    draw.text(
        (left_x + 12, left_y + 10),
        "IMAGE INPUT (browser screenshot; resized to 640×360 for VLM)",
        font=MONO_BOLD_FONT,
        fill="#FFFFFF",
        stroke_width=2,
        stroke_fill="#000000",
    )
    left_y += screenshot.height + gap
    if after_image is not None:
        draw.rounded_rectangle(
            (left_x, left_y, left_x + left_width, left_y + after_image.height),
            radius=14,
            fill="#222222",
            outline=BORDER,
            width=2,
        )
        canvas.paste(after_image, (left_x, left_y))
        draw.text(
            (left_x + 12, left_y + 10),
            "ENVIRONMENT OUTPUT AFTER ACTION",
            font=MONO_BOLD_FONT,
            fill="#FFFFFF",
            stroke_width=2,
            stroke_fill="#000000",
        )
        left_y += after_image.height + gap
    left_y = _draw_panel(
        canvas, left_x, left_y, left_width,
        "Input Metadata", _format_json(metadata), INPUT_COLOR, left_chars,
    ) + gap
    left_y = _draw_panel(
        canvas, left_x, left_y, left_width,
        "Chat Roles", role_summary, INPUT_COLOR, left_chars,
    ) + gap
    if row.get("memory_before_summary") or row.get("memory_after_summary"):
        _draw_panel(
            canvas, left_x, left_y, left_width,
            "COMPACT Recurrent State Summary", memory_text,
            MEMORY_COLOR, left_chars,
        )

    right_x = margin + left_width + gap
    right_y = margin + 70
    for panel_title, panel_text, accent, font in right_panels:
        right_y = _draw_panel(
            canvas, right_x, right_y, right_width,
            panel_title, panel_text, accent, right_chars, font,
        ) + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return {
        "step": step_number,
        "frame": str(output_path),
        "height": canvas.height,
        "has_before_screenshot": before_screenshot is not None,
        "has_after_screenshot": after_screenshot is not None,
        "has_exact_model_input": bool(row.get("model_input_text")),
    }


def _write_scrolling_video(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    seconds_per_step: float,
    video_width: int,
    video_height: int,
) -> None:
    if video_width % 2 or video_height % 2:
        raise RuntimeError("Video width and height must be even for yuv420p output.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{video_width}x{video_height}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Video output requires the ffmpeg executable.") from error

    frames_per_step = max(1, round(fps * seconds_per_step))
    hold_frames = max(1, round(fps * min(1.0, seconds_per_step / 4)))
    try:
        assert process.stdin is not None
        for frame_path in frame_paths:
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
            resized_height = max(1, round(image.height * video_width / image.width))
            image = image.resize(
                (video_width, resized_height), Image.Resampling.LANCZOS,
            )
            if image.height <= video_height:
                page = Image.new("RGB", (video_width, video_height), BACKGROUND)
                page.paste(image, (0, 0))
                frame_bytes = page.tobytes()
                for _ in range(frames_per_step):
                    process.stdin.write(frame_bytes)
                continue

            max_offset = image.height - video_height
            scroll_frames = max(1, frames_per_step - 2 * hold_frames)
            offsets = [0] * hold_frames
            offsets.extend(
                round(max_offset * index / max(scroll_frames - 1, 1))
                for index in range(scroll_frames)
            )
            offsets.extend([max_offset] * hold_frames)
            for offset in offsets:
                page = image.crop((0, offset, video_width, offset + video_height))
                process.stdin.write(page.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BrokenPipeError as error:
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.wait()
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg stopped while writing video: {detail}") from error
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

    if return_code != 0:
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {return_code}"
        raise RuntimeError(f"ffmpeg failed to create video: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render complete per-step VLM inputs/outputs from an eval session."
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional scrolling MP4 path.",
    )
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--seconds-per-step", type=float, default=8.0)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        parser.error(f"Session directory does not exist: {session_dir}")
    try:
        trajectory_path = _discover_trajectory(session_dir, args.trajectory)
        rows = _load_rows(trajectory_path)
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as error:
        parser.error(str(error))
    if args.max_steps is not None:
        rows = rows[: max(args.max_steps, 0)]
    if not rows:
        parser.error(f"Trajectory contains no rows: {trajectory_path}")

    output_dir = args.output_dir or (session_dir / "vlm_debug_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    frame_paths = []
    for row_index, row in enumerate(rows):
        step_number = _step_number(row, row_index)
        frame_path = output_dir / f"step_{step_number:03d}_vlm_debug.png"
        manifest.append(
            _render_step(row, row_index, session_dir, frame_path, args.width)
        )
        frame_paths.append(frame_path)
        print(f"[debug] Rendered {frame_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_dir": str(session_dir),
                "trajectory": str(trajectory_path),
                "frames": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[debug] Manifest: {manifest_path}")

    if args.video is not None:
        try:
            _write_scrolling_video(
                frame_paths,
                args.video,
                args.fps,
                args.seconds_per_step,
                args.video_width,
                args.video_height,
            )
        except RuntimeError as error:
            parser.error(str(error))
        print(f"[debug] Video: {args.video}")


if __name__ == "__main__":
    main()
