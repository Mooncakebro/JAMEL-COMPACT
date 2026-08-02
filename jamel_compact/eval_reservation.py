"""Cross-process reservations for concurrent evaluation runs."""
from __future__ import annotations

import fcntl
import os
import socket
from pathlib import Path
from typing import TextIO


class EvalRunReservation:
    def __init__(
        self,
        port: int,
        port_lock: TextIO,
        output_lock: TextIO,
    ) -> None:
        self.port = port
        self._port_lock = port_lock
        self._output_lock = output_lock

    def close(self) -> None:
        for lock_file_name in ("_output_lock", "_port_lock"):
            lock_file = getattr(self, lock_file_name, None)
            if lock_file is None:
                continue
            setattr(self, lock_file_name, None)
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _try_lock(path: Path) -> TextIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _reserve_port(requested_port: int) -> tuple[int, TextIO]:
    if requested_port < 0 or requested_port > 65535:
        raise ValueError(f"Invalid eval port: {requested_port}")

    lock_root = Path(
        os.environ.get("JAMEL_EVAL_LOCK_DIR", "/tmp/jamel_compact_eval_locks")
    )
    candidates = [requested_port] if requested_port else range(8790, 8991)
    for port in candidates:
        lock_file = _try_lock(lock_root / f"port_{port}.lock")
        if lock_file is None:
            continue
        if _port_is_available(port):
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"pid={os.getpid()} port={port}\n")
            lock_file.flush()
            return port, lock_file
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    if requested_port:
        raise RuntimeError(
            f"ScaleWoB port {requested_port} is already in use. "
            "Use --port 0 for automatic allocation or choose another port."
        )
    raise RuntimeError("No free ScaleWoB eval port found in range 8790-8990")


def reserve_eval_run(requested_port: int, output_dir: str | Path) -> EvalRunReservation:
    """Reserve one HTTP port and one output directory for this process."""
    port, port_lock = _reserve_port(int(requested_port))
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_lock = _try_lock(output_path / ".eval.lock")
    if output_lock is None:
        fcntl.flock(port_lock.fileno(), fcntl.LOCK_UN)
        port_lock.close()
        raise RuntimeError(
            f"Eval output directory is already used by another process: {output_path}. "
            "Set EVAL_OUTPUT to a different directory for each concurrent run."
        )

    output_lock.seek(0)
    output_lock.truncate()
    output_lock.write(f"pid={os.getpid()} port={port}\n")
    output_lock.flush()
    return EvalRunReservation(port, port_lock, output_lock)
