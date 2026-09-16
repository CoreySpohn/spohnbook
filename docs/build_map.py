"""Render the library map table from libraries.yaml at documentation build time."""

from pathlib import Path

import yaml

LAYER_NAMES = {
    "foundation": "Foundation",
    "geometry": "Geometry",
    "scene": "Scene",
    "data": "Data",
    "simulation": "Simulation",
    "wavefront": "Wavefront",
    "exposure-time": "Exposure time",
    "analysis": "Analysis",
    "inference": "Inference",
    "planning": "Planning",
    "orchestration": "Orchestration",
    "visualization": "Visualization",
}


def _link(label, url):
    return f"[{label}]({url})" if url else ""


def write_map_table(docs_dir: Path) -> Path:
    """Write docs/_generated/map-table.md from the repository's libraries.yaml."""
    data = yaml.safe_load((docs_dir.parent / "libraries.yaml").read_text())
    libraries = data["libraries"]
    out_dir = docs_dir / "_generated"
    out_dir.mkdir(exist_ok=True)
    lines = []
    for layer, title in LAYER_NAMES.items():
        rows = [lib for lib in libraries if lib["layer"] == layer]
        if not rows:
            continue
        lines.append(f"## {title}\n")
        lines.append("| Library | Stage | What it owns | Depends on | Links |")
        lines.append("|---|---|---|---|---|")
        for lib in rows:
            links = ", ".join(
                s
                for s in (
                    _link("docs", lib.get("docs")),
                    _link("PyPI", lib.get("pypi")),
                )
                if s
            )
            deps = ", ".join(lib.get("depends_on") or []) or "none"
            lines.append(
                f"| {_link(lib['name'], lib['repo'])} | {lib['stage']} | "
                f"{lib['summary']} | {deps} | {links} |"
            )
        lines.append("")
    path = out_dir / "map-table.md"
    path.write_text("\n".join(lines))
    return path
