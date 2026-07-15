"""Controlled vector/raster export and source-manifest writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def export_figure(
    figure,
    output_directory: str | Path,
    stem: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Path]:
    output_root = Path(output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor("white")
    for axis in figure.axes:
        axis.set_facecolor("white")
    figure.canvas.draw()

    paths = {
        "pdf": output_root / f"{stem}.pdf",
        "svg": output_root / f"{stem}.svg",
        "png": output_root / f"{stem}.png",
        "manifest": output_root / f"{stem}_source_manifest.json",
    }
    common = {"facecolor": "white", "transparent": False}
    figure.savefig(paths["pdf"], format="pdf", **common)
    figure.savefig(paths["svg"], format="svg", **common)
    figure.savefig(paths["png"], format="png", dpi=600, **common)

    payload = dict(source_manifest)
    payload["exports"] = {
        "pdf": paths["pdf"].name,
        "svg": paths["svg"].name,
        "png": paths["png"].name,
        "png_dpi": 600,
        "background": "white",
    }
    paths["manifest"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
