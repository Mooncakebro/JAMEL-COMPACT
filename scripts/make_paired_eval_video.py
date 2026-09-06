#!/usr/bin/env python3
"""Create a vertically stacked COMPACT/Baseline evaluation comparison video.

Each video frame contains the COMPACT screenshot on top and the Baseline
screenshot below. A colored header on each panel shows the method, evaluation
step, and accumulated reward.

Example:
    python scripts/make_paired_eval_video.py \
        --compact-trajectory outputs/compact_eval/app/session0/trajectory_*.parquet \
        --baseline-trajectory outputs/baseline_eval/app/session0/trajectory_*.parquet \
        --output outputs/analysis/app_pair/app_comparison.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COMPACT_COLOR = "#1769AA"
BASELINE_COLOR = "#C65F32"
BACKGROUND = "#161B24"
TEXT_COLOR = "#FFFFFF"
REWARD_COLOR = "#C9F2D0"
HEADER_HEIGHT = 76
GAP_HEIGHT = 8


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


LABEL_FONT = _font(28, bold=True)
REWARD_FONT = _font(24, bold=True)


def _trajectory_rows(path: Path) -> dict[int, dict[str, Any]]:
    frame = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Trajectory is empty: {path}")
    if "step" not in frame or "reward" not in frame:
        raise ValueError(f"Trajectory must contain step and reward columns: {path}")

    rows: dict[int, dict[str, Any]] = {}
    running_reward = 0.0
    for row_index, row in frame.iterrows():
        try:
            step = int(row["step"])
        except (TypeError, ValueError):
            step = row_index
        reward = float(row.get("reward", 0.0) or 0.0)
        cumulative = row.get("cumulative_reward")
        try:
            if cumulative is None or pd.isna(cumulative):
                running_reward += reward
            else:
                running_reward = float(cumulative)
        except (TypeError, ValueError):
            running_reward += reward
        rows[step] = {
            "step": step,
            "cumulative_reward": running_reward,
        }
    return rows


def _screenshot_path(session_dir: Path, step: int, kind: str) -> Path:
    # Trajectory steps are zero-based; saved screenshots are one-based.
    return session_dir / f"step_{step + 1:03d}_{kind}.png"


def _load_screenshot(path: Path, width: int, height: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing screenshot: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _draw_panel(
    screenshot: Image.Image,
    method: str,
    color: str,
    step: int,
    cumulative_reward: float,
) -> Image.Image:
    width, height = screenshot.size
    panel = Image.new("RGB", (width, HEADER_HEIGHT + height), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width, HEADER_HEIGHT), fill=color)
    draw.text(
        (18, 11),
        f"{method}    Evaluation step: {step + 1}/50",
        font=LABEL_FONT,
        fill=TEXT_COLOR,
    )
    draw.text(
        (width - 18, 15),
        f"Accumulated reward: {cumulative_reward:.0f}",
        font=REWARD_FONT,
        fill=REWARD_COLOR,
        anchor="ra",
    )
    panel.paste(screenshot, (0, HEADER_HEIGHT))
    return panel


def _write_video(
    frames: list[Image.Image],
    output: Path,
    fps: float,
) -> None:
    if not frames:
        raise ValueError("No video frames were generated")
    width, height = frames[0].size
    if width % 2 or height % 2:
        raise ValueError("Video dimensions must be even")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Creating the video requires ffmpeg.") from error

    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BrokenPipeError as error:
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.wait()
        raise RuntimeError(stderr.strip() or "ffmpeg stopped while writing") from error
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

    if return_code != 0:
        raise RuntimeError(stderr.strip() or f"ffmpeg exited with code {return_code}")


def make_video(
    compact_trajectory: Path,
    baseline_trajectory: Path,
    output: Path,
    snapshot_kind: str,
    fps: float,
    width: int,
    height: int,
    summary_output: Path | None,
) -> None:
    compact_rows = _trajectory_rows(compact_trajectory)
    baseline_rows = _trajectory_rows(baseline_trajectory)
    steps = sorted(set(compact_rows) & set(baseline_rows))
    if len(steps) != 50 or steps != list(range(50)):
        raise ValueError(
            "Expected both trajectories to contain exactly zero-based steps 0..49; "
            f"shared steps were {steps}"
        )

    compact_session = compact_trajectory.parent
    baseline_session = baseline_trajectory.parent
    frames: list[Image.Image] = []
    manifest: list[dict[str, Any]] = []
    for step in steps:
        compact_image = _load_screenshot(
            _screenshot_path(compact_session, step, snapshot_kind), width, height
        )
        baseline_image = _load_screenshot(
            _screenshot_path(baseline_session, step, snapshot_kind), width, height
        )
        compact_panel = _draw_panel(
            compact_image,
            "COMPACT",
            COMPACT_COLOR,
            step,
            float(compact_rows[step]["cumulative_reward"]),
        )
        baseline_panel = _draw_panel(
            baseline_image,
            "Baseline",
            BASELINE_COLOR,
            step,
            float(baseline_rows[step]["cumulative_reward"]),
        )
        frame = Image.new(
            "RGB",
            (width, compact_panel.height + GAP_HEIGHT + baseline_panel.height),
            BACKGROUND,
        )
        frame.paste(compact_panel, (0, 0))
        frame.paste(baseline_panel, (0, compact_panel.height + GAP_HEIGHT))
        frames.append(frame)
        manifest.append(
            {
                "step": step,
                "compact_reward": compact_rows[step]["cumulative_reward"],
                "baseline_reward": baseline_rows[step]["cumulative_reward"],
            }
        )

    _write_video(frames, output, fps)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(
                {
                    "compact_trajectory": str(compact_trajectory),
                    "baseline_trajectory": str(baseline_trajectory),
                    "snapshot_kind": snapshot_kind,
                    "fps": fps,
                    "frame_size": list(frames[0].size),
                    "frames": manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-trajectory", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-kind", choices=["before", "after"], default="before")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    make_video(
        args.compact_trajectory,
        args.baseline_trajectory,
        args.output,
        args.snapshot_kind,
        args.fps,
        args.width,
        args.height,
        args.summary_output,
    )
    print(f"Saved paired video to {args.output}")


if __name__ == "__main__":
    main()
