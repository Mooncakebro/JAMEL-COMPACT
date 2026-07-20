#!/usr/bin/env python3
"""
Convert step_*_before.png / step_*_after.png screenshots into an MP4 video.

Images are ordered by step number, with each step showing "before" then
"after" side by side (or sequentially). Skips non-image files (parquet,
coverage dir, etc.) automatically.

Usage:
    # Basic — sequential before/after, 2 fps:
    python scripts/snapshots_to_mp4.py outputs/compact_10epo_eval_debug/weibo/session0/

    # Side-by-side before|after, specify fps and output:
    python scripts/snapshots_to_mp4.py session0/ --mode side_by_side --fps 4 -o eval.mp4

    # Only "after" screenshots:
    python scripts/snapshots_to_mp4.py session0/ --mode after_only --fps 2

Requirements:
    pip install opencv-python numpy
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: This script requires opencv-python and numpy.")
    print("  Install:  pip install opencv-python numpy")
    sys.exit(1)


_STEP_RE = re.compile(r"step_(\d+)_(before|after)\.png$", re.IGNORECASE)


def collect_steps(session_dir: Path) -> list[dict]:
    """Return sorted list of {step, before_path, after_path}."""
    steps: dict[int, dict] = {}

    for f in session_dir.iterdir():
        if not f.is_file():
            continue
        m = _STEP_RE.match(f.name)
        if not m:
            continue
        step_num = int(m.group(1))
        kind = m.group(2).lower()
        if step_num not in steps:
            steps[step_num] = {"step": step_num, "before": None, "after": None}
        steps[step_num][kind] = f

    return [steps[k] for k in sorted(steps)]


def load_image(path: Path | None) -> np.ndarray | None:
    """Load an image, returning a BGR array for OpenCV. None if path is None."""
    if path is None:
        return None
    img = cv2.imread(str(path))
    return img


def build_frames(
    step_entries: list[dict],
    mode: str,
    gap_color=(0, 0, 0),
    gap_width=4,
    label_height=36,
    label_bg=(40, 40, 40),
    label_fg=(255, 255, 255),
) -> list[np.ndarray]:
    """Build video frames from step entries.

    Modes:
      - sequential:   before then after, each as a separate frame
      - side_by_side: before | after concatenated horizontally, one frame per step
      - before_only:  only the before screenshot
      - after_only:   only the after screenshot
    """
    frames: list[np.ndarray] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        """Add a label bar at the top of the image."""
        if img is None:
            return None
        h, w = img.shape[:2]
        bar = np.full((label_height, w, 3), label_bg, dtype=np.uint8)
        cv2.putText(
            bar, text, (10, label_height - 10),
            font, font_scale, label_fg, font_thickness, cv2.LINE_AA,
        )
        return np.vstack([bar, img])

    for entry in step_entries:
        step = entry["step"]
        before_img = load_image(entry["before"])
        after_img = load_image(entry["after"])

        if mode == "sequential":
            if before_img is not None:
                before_labeled = add_label(before_img, f"Step {step} — Before")
                frames.append(before_labeled)
            if after_img is not None:
                after_labeled = add_label(after_img, f"Step {step} — After")
                frames.append(after_labeled)

        elif mode == "side_by_side":
            before_labeled = add_label(before_img, f"Step {step} — Before")
            after_labeled = add_label(after_img, f"Step {step} — After")

            # If one side is missing, use a black placeholder
            if before_labeled is None and after_labeled is None:
                continue
            if before_labeled is None:
                h, w = after_labeled.shape[:2]
                before_labeled = np.zeros((h, w, 3), dtype=np.uint8)
            if after_labeled is None:
                h, w = before_labeled.shape[:2]
                after_labeled = np.zeros((h, w, 3), dtype=np.uint8)

            # Match heights
            h = max(before_labeled.shape[0], after_labeled.shape[0])
            if before_labeled.shape[0] < h:
                pad = np.zeros(
                    (h - before_labeled.shape[0], before_labeled.shape[1], 3),
                    dtype=np.uint8,
                )
                before_labeled = np.vstack([before_labeled, pad])
            if after_labeled.shape[0] < h:
                pad = np.zeros(
                    (h - after_labeled.shape[0], after_labeled.shape[1], 3),
                    dtype=np.uint8,
                )
                after_labeled = np.vstack([after_labeled, pad])

            gap = np.full((h, gap_width, 3), gap_color, dtype=np.uint8)
            frame = np.hstack([before_labeled, gap, after_labeled])
            frames.append(frame)

        elif mode == "before_only":
            if before_img is not None:
                frames.append(add_label(before_img, f"Step {step} — Before"))

        elif mode == "after_only":
            if after_img is not None:
                frames.append(add_label(after_img, f"Step {step} — After"))

    return frames


def write_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float = 2.0,
    codec: str = "mp4v",
) -> None:
    """Write frames to an MP4 file."""
    if not frames:
        print("No frames to write!")
        return

    # Use the max dimensions across all frames to avoid truncation
    max_w = max(f.shape[1] for f in frames)
    max_h = max(f.shape[0] for f in frames)

    # Pad all frames to the same size
    uniform_frames = []
    for f in frames:
        h, w = f.shape[:2]
        if w < max_w or h < max_h:
            padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            padded[:h, :w] = f
            uniform_frames.append(padded)
        else:
            uniform_frames.append(f)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (max_w, max_h))

    if not writer.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}")
        print("  Try installing ffmpeg:  sudo apt install ffmpeg")
        return

    for frame in uniform_frames:
        writer.write(frame)
    writer.release()
    print(f"  Wrote {len(uniform_frames)} frames → {output_path}")
    print(f"  Resolution: {max_w}×{max_h}, FPS: {fps}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert step screenshots into an MP4 video.",
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Directory containing step_XXX_before.png / step_XXX_after.png files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: <session_dir>/eval_video.mp4)",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "side_by_side", "before_only", "after_only"],
        default="sequential",
        help="Frame arrangement (default: sequential — before then after per step)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec (default: mp4v)",
    )
    args = parser.parse_args()

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"Error: {session_dir} is not a directory")
        sys.exit(1)

    print(f"Scanning: {session_dir}")
    step_entries = collect_steps(session_dir)
    if not step_entries:
        print("Error: No step_XXX_before.png or step_XXX_after.png files found!")
        sys.exit(1)

    print(f"  Found {len(step_entries)} steps "
          f"(step {step_entries[0]['step']} – {step_entries[-1]['step']})")

    output_path = args.output or (session_dir / "eval_video.mp4")
    print(f"  Mode: {args.mode}, FPS: {args.fps}")
    print(f"  Output: {output_path}")

    frames = build_frames(step_entries, mode=args.mode)
    print(f"  Built {len(frames)} frames")

    write_video(frames, output_path, fps=args.fps, codec=args.codec)

    print(f"\nDone! Video: {output_path}")


if __name__ == "__main__":
    main()
