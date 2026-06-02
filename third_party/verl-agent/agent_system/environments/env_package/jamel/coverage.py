import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import playwright.sync_api

logger = logging.getLogger(__name__)

CoverageUnit = tuple[str, int, int]


def start_coverage(page: "playwright.sync_api.Page"):
    """Enable Chrome precise coverage collection on the current page."""
    try:
        cdp = page.context.new_cdp_session(page)
        cdp.send("Profiler.enable")
        cdp.send("Debugger.enable")
        cdp.send(
            "Profiler.startPreciseCoverage",
            {
                "callCount": False,
                "detailed": True,
            },
        )
        return cdp
    except Exception:
        logger.exception("Failed to start browser coverage collection")
        return None


def extract_covered_units(coverage_records: list[dict]) -> set[CoverageUnit]:
    covered_units: set[CoverageUnit] = set()
    for record in coverage_records:
        url = record.get("url")
        if not url or not url.startswith("http"):
            continue

        for function_info in record.get("functions", []):
            for range_info in function_info.get("ranges", []):
                count = int(range_info.get("count", 0))
                if count <= 0:
                    continue

                start = int(range_info.get("startOffset", 0))
                end = int(range_info.get("endOffset", 0))
                if end <= start:
                    continue

                covered_units.add((url, start, end))

    return covered_units


def take_coverage_snapshot(cdp: "playwright.sync_api.CDPSession", output_path: str | Path | None = None) -> set[CoverageUnit]:
    if not cdp:
        return set()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        coverage_data = cdp.send("Profiler.takePreciseCoverage")
        script_coverages = coverage_data.get("result", [])
        final_data: list[dict] = []

        for script in script_coverages:
            url = script.get("url")
            if not url or not url.startswith("http"):
                continue

            try:
                source_obj = cdp.send("Debugger.getScriptSource", {"scriptId": script["scriptId"]})
            except Exception:
                continue

            final_data.append(
                {
                    "url": url,
                    "scriptId": script["scriptId"],
                    "source": source_obj.get("scriptSource", ""),
                    "functions": script.get("functions", []),
                }
            )

        if output_path is not None:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False)

        return extract_covered_units(final_data)
    except Exception:
        logger.exception("Failed to persist browser coverage snapshot", extra={"output_path": str(output_path)})
        return set()


@lru_cache(maxsize=2048)
def load_coverage_units_from_file(coverage_path: str) -> frozenset[CoverageUnit]:
    path = Path(coverage_path).expanduser().resolve()
    if not path.exists():
        logger.warning("Coverage file does not exist: %s", path)
        return frozenset()

    try:
        with path.open("r", encoding="utf-8") as f:
            coverage_records = json.load(f)
    except Exception:
        logger.exception("Failed to load coverage file", extra={"coverage_path": str(path)})
        return frozenset()

    return frozenset(extract_covered_units(coverage_records))


def merge_coverage_units_from_paths(coverage_paths: Iterable[str]) -> set[CoverageUnit]:
    merged_units: set[CoverageUnit] = set()
    for coverage_path in coverage_paths:
        if not coverage_path:
            continue
        merged_units.update(load_coverage_units_from_file(str(coverage_path)))
    return merged_units
