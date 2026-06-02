from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Any


COVERAGE_ARTIFACT_COLUMNS = (
    "coverage_path",
    "coverage_json_gzip_bytes",
    "coverage_sha256",
    "coverage_size_bytes",
    "coverage_exists_at_write",
)


def build_coverage_artifact_fields(coverage_path: str | Path | None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "coverage_path": str(coverage_path) if coverage_path else None,
        "coverage_json_gzip_bytes": None,
        "coverage_sha256": None,
        "coverage_size_bytes": 0,
        "coverage_exists_at_write": False,
    }
    if not coverage_path:
        return fields

    path = Path(coverage_path)
    fields["coverage_path"] = str(path)
    if not path.exists() or not path.is_file():
        return fields

    raw_bytes = path.read_bytes()
    fields.update(
        {
            "coverage_json_gzip_bytes": gzip.compress(raw_bytes, mtime=0),
            "coverage_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "coverage_size_bytes": len(raw_bytes),
            "coverage_exists_at_write": True,
        }
    )
    return fields


def coverage_artifact_extra_fields(artifact_fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact_fields.items()
        if key != "coverage_json_gzip_bytes"
    }
