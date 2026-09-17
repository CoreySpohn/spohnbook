"""Render the library dependency graph and the static D2 diagrams for the site.

Run from the repository root: python tools/build_map_figures.py
(needs hwostyle, d2 with the ELK layout, and rsvg-convert).

The dependency graph is generated from libraries.yaml, so it stays in step with
the map table; the other diagrams are the D2 sources under docs/figures/source/.
"""

import json
import subprocess
from pathlib import Path

import hwostyle
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
SOURCE = OUT / "source"

LAYER_ORDER = [
    "orchestration",
    "planning",
    "inference",
    "analysis",
    "exposure-time",
    "wavefront",
    "simulation",
    "data",
    "scene",
    "geometry",
    "foundation",
    "visualization",
]
LAYER_LABELS = {
    "orchestration": "Orchestration",
    "planning": "Planning",
    "inference": "Inference",
    "analysis": "Analysis",
    "exposure-time": "Exposure time",
    "wavefront": "Wavefront",
    "simulation": "Simulation",
    "data": "Data",
    "scene": "Scene",
    "geometry": "Geometry",
    "foundation": "Foundation",
    "visualization": "Visualization",
}


def d2_prefix(mode):
    colors = hwostyle.palette
    bg, fg, fill, edge = (
        ("#FFFFFF", "#20252B", "#F2F6F8", "#74818C")
        if mode == "light"
        else ("#101419", "#F5F7F8", "#1D2731", "#8A99A7")
    )
    return f"""vars: {{
  bg: "{bg}"
  fg: "{fg}"
  fill: "{fill}"
  edge: "{edge}"
  cyan: "{colors.cyan}"
  pink: "{colors.pink}"
  yellow: "{colors.yellow}"
}}
style.fill: ${{bg}}
*.style.font-color: ${{fg}}
*.style.font-size: 18
classes: {{
  layer: {{style: {{fill: ${{bg}}; stroke: ${{edge}}; stroke-width: 1; stroke-dash: 4; border-radius: 12; font-size: 16}}}}
  lib: {{style: {{fill: ${{fill}}; stroke: ${{cyan}}; stroke-width: 2; border-radius: 10}}}}
  viz: {{style: {{fill: ${{fill}}; stroke: ${{pink}}; stroke-width: 2; border-radius: 10}}}}
  step: {{style: {{fill: ${{fill}}; stroke: ${{cyan}}; stroke-width: 2; border-radius: 10}}}}
  external: {{style: {{fill: ${{fill}}; stroke: ${{yellow}}; stroke-width: 2; border-radius: 10}}}}
  dep: {{style: {{stroke: ${{edge}}; stroke-width: 2}}}}
}}
"""


def quote(name):
    return f'"{name}"' if "-" in name else name


UTILITY = ("hwoutils", "hwostyle")


def dependency_graph_source():
    """Flat layered graph: color encodes the layer, edges point at what a library imports.

    Edges into the two utility libraries are left out because nearly every
    library carries them; they and the libraries with no edges sit in a
    container pinned to the bottom so the layered part stays readable.
    """
    data = yaml.safe_load((ROOT / "libraries.yaml").read_text())
    libs = data["libraries"]
    layer_of = {lib["name"]: lib["layer"] for lib in libs}
    edges = [
        (lib["name"], dep)
        for lib in libs
        for dep in (lib.get("depends_on") or [])
        if dep in layer_of and dep not in UTILITY
    ]
    connected = {a for a, _ in edges} | {b for _, b in edges}

    def node(lib):
        layer = lib["layer"]
        cls = "viz" if layer == "visualization" else ("external" if lib["name"] in UTILITY else "lib")
        return f'{quote(lib["name"])}: {{label: "{lib["name"]}\\n{LAYER_LABELS[layer].lower()}"; class: {cls}}}'

    lines = ["direction: down", ""]
    for layer in LAYER_ORDER:
        for lib in libs:
            if lib["layer"] == layer and lib["name"] in connected:
                lines.append(node(lib))
    lines += ["", "base: {", "  near: bottom-center", '  label: ""', "  class: layer", "  grid-columns: 1"]
    lines += ["  utility: {", '    label: "imported by every library above"', "    class: layer", "    grid-rows: 1"]
    lines += ["    " + node(lib) for lib in libs if lib["name"] in UTILITY]
    lines += ["  }", "  standalone: {", '    label: "no dependencies within the suite"', "    class: layer", "    grid-rows: 1"]
    lines += ["    " + node(lib) for lib in libs if lib["name"] not in connected and lib["name"] not in UTILITY]
    lines += ["  }", "}", ""]
    for a, b in edges:
        lines.append(f"{quote(a)} -> {quote(b)}: {{class: dep}}")
    return "\n".join(lines) + "\n"


def render(name, body, mode):
    svg = OUT / f"{name}-{mode}.svg"
    subprocess.run(
        ["d2", "--layout", "elk", "--theme", "0" if mode == "light" else "200", "--pad", "30", "-", str(svg)],
        input=d2_prefix(mode) + body,
        text=True,
        check=True,
        capture_output=True,
    )
    subprocess.run(["rsvg-convert", "-z", "2", str(svg), "-o", str(svg.with_suffix(".png"))], check=True)
    return svg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for mode in ("light", "dark"):
        hwostyle.use(mode)
        outputs.append(render("library-graph", dependency_graph_source(), mode))
        for path in sorted(SOURCE.glob("*.d2")):
            outputs.append(render(path.stem, path.read_text(), mode))
    manifest = {
        "build_command": "python tools/build_map_figures.py",
        "generated_from": "libraries.yaml and docs/figures/source/*.d2",
        "outputs": sorted(p.name for p in OUT.iterdir() if p.suffix in {".svg", ".png"}),
    }
    (OUT / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(outputs)} diagrams into {OUT}")


if __name__ == "__main__":
    main()
