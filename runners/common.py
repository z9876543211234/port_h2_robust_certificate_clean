"""Small CLI parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def case_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    names = {f"C{index}": PROJECT_ROOT / "data" / "cases" / f"C{index}.yaml" for index in range(1, 5)}
    try:
        return names[value.upper()]
    except KeyError as error:
        raise ValueError("case must be C1, C2, C3, C4, or a case file path") from error


def parse_resolutions(items: list[str]) -> dict[str, object]:
    result = {}
    for item in items:
        path, separator, raw_value = item.partition("=")
        if not separator or not path:
            raise ValueError(f"invalid --set value: {item}")
        result[path] = json.loads(raw_value)
    return result


def current_git_commit() -> str:
    head = PROJECT_ROOT / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        reference = PROJECT_ROOT / ".git" / value[5:]
        return reference.read_text(encoding="utf-8").strip() if reference.is_file() else "uncommitted"
    return value
