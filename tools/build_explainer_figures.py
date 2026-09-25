"""Build the handbook's physical explainer diagrams and their talk assets.

Run from anywhere:

    python tools/build_explainer_figures.py                 # every module
    python tools/build_explainer_figures.py --only d02 d05  # some modules
    python tools/build_explainer_figures.py --no-anim       # stills only
    python tools/build_explainer_figures.py --preview       # review PNGs
    python tools/build_explainer_figures.py --cast-sheet    # the vocabulary

Each diagram is a module tools/explainers/dNN_<topic>.py declaring FIGURES
and ANIMATIONS (see tools/explainers/__init__.py). Outputs go to
docs/conventions/figures/ (documentation stills and players), talks/stills/
and talks/animations/ (talk assets), with one manifest per module in
docs/conventions/figures/explainer-manifests/. A module writes only files
named by its own dNN prefix. --preview writes to the ignored
.explainer-preview/ directory and changes nothing else.

Needs NumPy, Matplotlib, hwostyle, hwoutils and eyepiece; animations also
need ffmpeg.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explainers import _common as ex
from explainers import _export as exporter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="dNN",
        help="build only the modules with these prefixes, for example d02",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="skip animations (keeps their manifest entries)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="write review PNGs to .explainer-preview/ instead of publishing",
    )
    parser.add_argument(
        "--cast-sheet",
        action="store_true",
        help="preview the shared entity vocabulary and exit",
    )
    args = parser.parse_args(argv)

    if args.cast_sheet:
        for rel in exporter.preview_builder(ex.cast_sheet, "cast-sheet"):
            print(rel)
        return 0

    names = exporter.discover()
    if args.only:
        wanted = {w.lower()[:3] for w in args.only}
        names = [n for n in names if n[:3] in wanted]
        missing = wanted - {n[:3] for n in names}
        if missing:
            parser.error(f"no module for {sorted(missing)}; have {exporter.discover()}")
    if not names:
        print("no diagram modules found in tools/explainers/")
        return 0
    for name in names:
        written = exporter.build_module(
            exporter.load(name), animations=not args.no_anim, preview=args.preview
        )
        for rel in written:
            print(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
