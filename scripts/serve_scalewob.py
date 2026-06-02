#!/usr/bin/env python3
"""Serve the downloaded ScaleWoB static browser environments."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "env" / "browser_env" / "scalewob-env"

    parser = argparse.ArgumentParser(description="Serve ScaleWoB static apps.")
    parser.add_argument("--root", default=str(default_root), help="Path to scalewob-env.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"ScaleWoB root not found: {root}. "
            "Run `bash shell/download_scalewob_env.sh --mode all` first."
        )

    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/weibo/index.html"
    print(f"Serving {root}")
    print(f"Example app: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
