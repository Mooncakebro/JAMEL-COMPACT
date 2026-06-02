from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path


_MODEL_FORMAT = "jamel_model"
_MODEL_FORMAT_VERSION = 1
_MEMORY_CONFIG_NAME = "memory_augment_config.json"
_HIDDEN_MODEL_FILES = (".msc", ".mdl", ".mv")


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint is not a directory: {checkpoint}")

    candidates: list[tuple[int, Path]] = []
    for child in checkpoint.iterdir():
        if not child.is_dir() or not child.name.startswith("global_step_"):
            continue
        try:
            step = int(child.name.removeprefix("global_step_"))
        except ValueError:
            continue
        candidates.append((step, child))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1].resolve()
    return checkpoint


def _ensure_safe_output_path(output_model_path: str | Path) -> Path:
    output = Path(output_model_path).expanduser().resolve()
    if output == Path("/") or output == Path.cwd().resolve():
        raise ValueError(f"refusing to write model to unsafe path: {output}")
    return output


def _assert_not_nested(output: Path, source: Path, *, source_name: str) -> None:
    if output == source:
        raise ValueError(f"output path must differ from {source_name}: {source}")
    if output.is_relative_to(source):
        raise ValueError(f"output path cannot be inside {source_name}: {source}")
    if source.is_relative_to(output):
        raise ValueError(f"output path cannot contain {source_name}: {source}")


def _copy_directory(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _clean_model_metadata(path: Path, *, remove_readme: bool) -> None:
    for filename in _HIDDEN_MODEL_FILES:
        (path / filename).unlink(missing_ok=True)
    if remove_readme:
        (path / "README.md").unlink(missing_ok=True)


def _normalize_actor_memory_config(actor_path: Path) -> None:
    memory_config_path = actor_path / _MEMORY_CONFIG_NAME
    if not memory_config_path.is_file():
        return

    memory_config = json.loads(memory_config_path.read_text(encoding="utf-8"))
    memory_config["format"] = "jamel_memory_augmented_lm"
    memory_config["format_version"] = int(memory_config.get("format_version", 1))
    memory_config["self_contained"] = True
    memory_config["base_model_name_or_path"] = "."
    memory_config_path.write_text(
        json.dumps(memory_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def package_jamel_model(
    *,
    checkpoint: str | Path,
    compressor_model: str | Path,
    output_model_path: str | Path,
) -> Path:
    """Create a released JAMEL model directory from an actor checkpoint and compressor.

    `checkpoint` may be either a concrete actor checkpoint or a training output
    directory containing `global_step_*` subdirectories. The latest global step is
    used when such subdirectories exist.
    """
    actor_checkpoint = _resolve_checkpoint(checkpoint)
    compressor = Path(compressor_model).expanduser().resolve()
    if not compressor.is_dir():
        raise FileNotFoundError(f"compressor model is not a directory: {compressor}")

    output = _ensure_safe_output_path(output_model_path)
    _assert_not_nested(output, actor_checkpoint, source_name="checkpoint")
    _assert_not_nested(output, compressor, source_name="compressor model")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    actor_output = output / "actor"
    compressor_output = output / "compressor"
    _copy_directory(actor_checkpoint, actor_output)
    _copy_directory(compressor, compressor_output)

    _clean_model_metadata(actor_output, remove_readme=True)
    _clean_model_metadata(compressor_output, remove_readme=False)
    _normalize_actor_memory_config(actor_output)

    manifest = {
        "format": _MODEL_FORMAT,
        "format_version": _MODEL_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "components": {
            "actor": "actor",
            "compressor": "compressor",
        },
    }
    (output / "model.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a JAMEL actor checkpoint with its compressor model.")
    parser.add_argument("--checkpoint", required=True, help="Actor checkpoint or output dir containing global_step_*.")
    parser.add_argument("--compressor-model", required=True, help="Local Qwen3-VL compressor model directory.")
    parser.add_argument("--output-model-path", required=True, help="Output JAMEL model directory.")
    args = parser.parse_args()

    output = package_jamel_model(
        checkpoint=args.checkpoint,
        compressor_model=args.compressor_model,
        output_model_path=args.output_model_path,
    )
    print(f"Packaged JAMEL model at: {output}")
    print(f"Evaluate with: MODEL_PATH={output} bash shell/run_eval.sh")


if __name__ == "__main__":
    main()
