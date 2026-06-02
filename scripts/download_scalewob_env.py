#!/usr/bin/env python3
"""Download and extract the released ScaleWoB static browser apps."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ARCHIVE_URL = "https://cloud.tsinghua.edu.cn/f/2d5201e728bf4c58af3e/?dl=1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_scalewob_root(extract_root: Path) -> Path:
    candidates = [extract_root / "scalewob-env", extract_root]
    candidates.extend(path for path in extract_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "environments.json").is_file():
            return candidate
    raise RuntimeError("Archive does not contain a scalewob-env root with environments.json.")


def _download(url: str, timeout_s: float) -> Path:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"ScaleWoB archive download failed with HTTP {status}: {url}")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            shutil.copyfileobj(response, tmp)
            return Path(tmp.name)


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Download and extract scalewob-env.zip.")
    parser.add_argument("--url", default=DEFAULT_ARCHIVE_URL, help="Direct scalewob-env.zip URL.")
    parser.add_argument(
        "--output",
        default=str(repo_root / "env" / "browser_env" / "scalewob-env"),
        help="Output directory for the extracted scalewob-env root.",
    )
    parser.add_argument("--timeout-ms", type=int, default=600000)
    parser.add_argument(
        "--mode",
        choices=("test10", "train86", "all"),
        default="all",
        help="Accepted for compatibility; the archive contains the full environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    timeout_s = max(args.timeout_ms / 1000, 1)

    print(f"Downloading ScaleWoB archive: {args.url}")
    archive_path = _download(args.url, timeout_s=timeout_s)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            extract_root = Path(tmp_dir)
            shutil.unpack_archive(str(archive_path), str(extract_root))
            source_dir = _find_scalewob_root(extract_root)

            output_dir.parent.mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.move(str(source_dir), str(output_dir))

        manifest = {
            "archive_url": args.url,
            "output_dir": str(output_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (output_dir / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        archive_path.unlink(missing_ok=True)

    print(f"ScaleWoB environment ready: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
