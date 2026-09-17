"""Build the handbook's independent teaching figures and editable D2 diagrams.

Run from the repository root: python tools/build_conventions_figures.py
(needs NumPy, Matplotlib, hwostyle, d2 and rsvg-convert).
The examples derive from stated definitions; they do not execute repaired models.
"""

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import hwostyle
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "conventions" / "figures"
SOURCE = OUT / "source"


def save(fig, name, mode):
    """Export the same layout to screen and vector publication formats."""
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUT / f"hwo-conventions-{name}-{mode}.{suffix}",
            dpi=180 if suffix == "png" else 300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def quiet_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def arrow(ax, start, end, color, text=None, offset=(0, 0)):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
    )
    if text:
        ax.text(end[0] + offset[0], end[1] + offset[1], text, color=color)


def geometry(mode):
    colors = hwostyle.palette
    neutral = "#777777" if mode == "light" else "#8A8A8A"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    ax, phase, rv = axes
    ax.set(xlim=(-1.5, 1.65), ylim=(-0.6, 1.75), aspect="equal")
    ax.axis("off")
    ax.set_title("(a) Illumination is a planet-centered angle", loc="left")
    arrow(ax, (-1.4, 0), (1.5, 0), neutral)
    arrow(ax, (0, -0.35), (0, 1.45), neutral)
    ax.text(0.8, -0.44, "+z toward observer", ha="center")
    ax.text(0, 1.6, "projected coordinate", ha="center", color=neutral)
    ax.scatter([0], [0], marker="*", s=240, color=colors.yellow, zorder=5)
    ax.text(0, -0.25, "star", ha="center", color=neutral)
    ax.add_patch(Circle((-1, 0), 0.12, color=colors.cyan, zorder=6))
    ax.add_patch(
        Circle((1, 0), 0.12, fc=fig.get_facecolor(), ec=colors.cyan, lw=2, zorder=6)
    )
    ax.scatter([0], [1], s=130, color=colors.cyan, zorder=6)
    ax.text(-1, 0.28, "$\\alpha=0$\nfull", ha="center")
    ax.text(1, 0.28, "$\\alpha=\\pi$\ndark", ha="center")
    ax.text(-0.08, 1.15, "$\\alpha=\\pi/2$", ha="right")
    arrow(ax, (0.04, 1), (0.75, 1), neutral)
    arrow(ax, (0, 0.91), (0, 0.33), neutral)
    ax.add_patch(Arc((0, 1), 0.5, 0.5, theta1=-90, theta2=0, color=colors.pink, lw=2))
    ax.text(0.27, 0.76, "$\\alpha$", color=colors.pink)

    beta = np.linspace(0, np.pi, 301)
    alpha = np.pi - beta
    physical = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
    legacy = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    phase.plot(
        np.rad2deg(beta),
        physical,
        color=colors.cyan,
        label=r"$\Phi(\pi-\beta)$: illumination",
    )
    phase.plot(
        np.rad2deg(beta),
        legacy,
        color=colors.pink,
        ls="--",
        label=r"$\Phi(\beta)$: angle confused",
    )
    phase.set(
        xlim=(0, 180),
        ylim=(-0.025, 1.12),
        xlabel=r"Observer-axis angle $\beta$ (deg)",
        ylabel=r"Lambert phase function $\Phi$",
    )
    phase.set_xticks([0, 45, 90, 135, 180])
    phase.set_title("(b) Quadrature hides the error", loc="left")
    phase.scatter(
        [90], [1 / np.pi], s=55, marker="s", color=plt.rcParams["text.color"], zorder=5
    )
    phase.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    quiet_axes(phase)

    rv.axis("off")
    rv.set(xlim=(-1.55, 1.7), ylim=(-0.9, 1.6))
    rv.set_title("(c) Recession fixes the stellar RV sign", loc="left")
    arrow(rv, (-1.4, 1.24), (1.45, 1.24), neutral)
    rv.text(0, 1.4, "+z toward observer", ha="center")
    rv.scatter([0], [0.75], s=120, color=colors.cyan)
    arrow(rv, (0.15, 0.75), (1, 0.75), colors.cyan)
    rv.text(-1.4, 0.75, "planet", va="center", color=colors.cyan)
    rv.text(0.5, 0.91, r"$v_{z,\mathrm{rel}}>0$", ha="center")
    rv.scatter([0], [0.12], s=180, marker="*", color=colors.yellow)
    arrow(rv, (-0.13, 0.12), (-0.9, 0.12), colors.pink)
    rv.text(0.4, 0.12, "star recedes", va="center", color=colors.pink)
    rv.text(0, -0.4, r"$v_{r,\star}=-\dot z_\star>0$", ha="center")
    rv.text(
        0,
        -0.75,
        "Velocity arrows are schematic, not to scale.",
        ha="center",
        color=neutral,
    )
    save(fig, "geometry", mode)


def time_coordinates(mode):
    colors = hwostyle.palette
    neutral = "#777777" if mode == "light" else "#8A8A8A"
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set(xlim=(-8, 29), ylim=(-0.7, 3))
    ax.set_title(
        "(a) Different numeric origins can describe the same instants", loc="left"
    )
    for row, label, numbers in (
        (2, "JD (TDB)", ["2451545.0", "2451555.0", "2451565.0"]),
        (1, "MJD (TDB)", ["51544.5", "51554.5", "51564.5"]),
        (0, "Elapsed days", ["0", "10", "20"]),
    ):
        ax.text(-3, row, label, ha="right", va="center")
        arrow(ax, (-0.5, row), (24, row), neutral)
        for x, number in zip([0, 10, 20], numbers, strict=True):
            ax.scatter([x], [row], color=colors.cyan, s=40, zorder=3)
            ax.text(x, row + 0.23, number, ha="center")
    ax.text(
        10,
        -0.6,
        "Same scale and reference position. UTC/TAI/TDB conversion is a separate operation.",
        ha="center",
    )

    drift = fig.add_subplot(gs[1, 0])
    years = np.linspace(0, 3, 601)
    base = date(2000, 1, 1)
    calendar_days = []
    for elapsed in years:
        whole = int(np.floor(elapsed))
        start, end = date(2000 + whole, 1, 1), date(2001 + whole, 1, 1)
        calendar_days.append(
            (start - base).days + (elapsed - whole) * (end - start).days
        )
    error = np.asarray(calendar_days) - 365.25 * years
    drift.plot(years, error, color=colors.pink)
    drift.scatter([1], [0.75], color=plt.rcParams["text.color"], s=45, zorder=4)
    drift.annotate(
        "+0.75 day at one elapsed year",
        xy=(1, 0.75),
        xytext=(1.1, 0.91),
        arrowprops={"arrowstyle": "->", "color": neutral},
    )
    drift.set(
        xlim=(0, 3),
        ylim=(-0.05, 1.1),
        xlabel="Elapsed Julian years from start of 2000",
        ylabel="Imported duration error (days)",
    )
    drift.set_title("(b) Calendar years change duration", loc="left")
    quiet_axes(drift)

    precision = fig.add_subplot(gs[1, 1])
    values = np.array([2451545.0, 51544.5, 10.0], dtype=np.float32)
    seconds = np.spacing(values).astype(float) * 86400
    precision.bar(
        [0, 1, 2], seconds, color=[colors.cyan, colors.pink, colors.yellow], width=0.5
    )
    precision.set_yscale("log")
    precision.set(ylim=(0.01, 1e6), ylabel="float32 spacing (seconds)")
    precision.set_xticks([0, 1, 2], ["JD\n2451545", "MJD\n51544.5", "Elapsed day\n10"])
    for x, value in enumerate(seconds):
        precision.text(x, value * 1.7, f"{value:g} s", ha="center")
    precision.set_title("(c) Precision depends on representation", loc="left")
    quiet_axes(precision)
    save(fig, "time", mode)
    return {
        "float32_spacing_s": seconds.tolist(),
        "calendar_duration_error_at_one_year_d": float(error[200]),
    }


def pixels(mode):
    colors = hwostyle.palette
    neutral = "#777777" if mode == "light" else "#8A8A8A"
    furniture = "#D6D6D6" if mode == "light" else "#353535"
    fig, (grid, roll, flux) = plt.subplots(
        1, 3, figsize=(15, 5.5), layout="constrained"
    )
    grid.set_title("(a) Pixel centers and the optical origin", loc="left")
    for x in np.arange(-0.5, 4, 1):
        grid.axvline(x, color=furniture, lw=1)
        grid.axhline(x, color=furniture, lw=1)
    xx, yy = np.meshgrid(np.arange(4), np.arange(4))
    grid.scatter(xx, yy, color=colors.cyan, s=30)
    grid.scatter(
        [1.5], [1.5], color=colors.pink, marker="+", s=200, linewidth=2.5, zorder=4
    )
    grid.annotate(
        "origin (1.5, 1.5)",
        (1.5, 1.5),
        (0.1, 2.7),
        arrowprops={"arrowstyle": "->", "color": neutral},
    )
    grid.set(
        xlim=(-0.5, 3.5),
        ylim=(-0.5, 3.5),
        aspect="equal",
        xlabel="column index j",
        ylabel="row index i (displayed upward)",
    )
    grid.set_xticks(range(4))
    grid.set_yticks(range(4))
    grid.text(1.5, -1.2, r"$x=(j-c_x)s,\quad y=(i-c_y)s$", ha="center")

    roll.set_title("(b) Telescope roll is a frame change", loc="left")
    roll.set(xlim=(-1.4, 1.5), ylim=(-1.4, 1.5), aspect="equal", xlabel="x", ylabel="y")
    roll.axhline(0, color=furniture)
    roll.axvline(0, color=furniture)
    arrow(roll, (0, 0), (1, 0), colors.cyan)
    arrow(roll, (0, 0), (0, -1), colors.pink)
    roll.scatter([1, 0], [0, -1], color=[colors.cyan, colors.pink], s=65)
    roll.plot([0, 0], [0, 1], color=neutral, ls="--")
    roll.scatter([0], [1], marker="x", color=neutral, s=65)
    roll.text(0.4, 0.15, "sky (+1, 0)", color=colors.cyan)
    roll.text(0.12, -1.12, "detector (0, -1)", color=colors.pink)
    roll.text(0.12, 1.03, "active +90 deg", color=neutral)
    roll.set_xticks([-1, 0, 1])
    roll.set_yticks([-1, 0, 1])
    roll.text(
        0,
        -1.9,
        r"$\mathbf{x}_{det}=R(-\theta)\mathbf{x}_{sky},\quad\theta=90^\circ$",
        ha="center",
    )

    n = np.array([2, 4, 8])
    flux.plot(
        n,
        np.full(3, 4),
        color=colors.cyan,
        marker="o",
        label="Integrate density over each pixel",
    )
    flux.plot(
        n,
        n**2,
        color=colors.pink,
        marker="s",
        ls="--",
        label="Treat density samples as pixel rates",
    )
    flux.set(
        xlabel="Pixels along each side",
        ylabel="Sum of pixel rates (photons / s)",
        xlim=(1.5, 8.5),
        ylim=(0, 70),
    )
    flux.set_xticks(n)
    flux.set_title("(c) Refinement must preserve total flux", loc="left")
    flux.text(
        2,
        60,
        "Uniform brightness: 1 photon / s / arcsec$^2$\nFixed field: 2 arcsec by 2 arcsec",
    )
    flux.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    quiet_axes(flux)
    save(fig, "pixels", mode)


def diagrams(mode):
    colors = hwostyle.palette
    bg, fg, fill, edge = (
        ("#FFFFFF", "#20252B", "#F2F6F8", "#74818C")
        if mode == "light"
        else ("#101419", "#F5F7F8", "#1D2731", "#8A99A7")
    )
    prefix = f'''vars: {{
  bg: "{bg}"
  fg: "{fg}"
  fill: "{fill}"
  edge: "{edge}"
  cyan: "{colors.cyan}"
  pink: "{colors.pink}"
}}
style.fill: ${{bg}}
*.style.font-color: ${{fg}}
*.style.font-size: 18
classes: {{
  stage: {{style: {{fill: ${{fill}}; stroke: ${{cyan}}; stroke-width: 2; border-radius: 10}}}}
  model: {{style: {{fill: ${{fill}}; stroke: ${{pink}}; stroke-width: 2; border-radius: 10}}}}
}}
'''
    for name in ("pipeline", "roadmap", "measurement"):
        source = prefix + (SOURCE / f"{name}.d2").read_text()
        svg = OUT / f"hwo-conventions-{name}-{mode}.svg"
        subprocess.run(
            [
                "d2",
                "--layout",
                "elk",
                "--theme",
                "0" if mode == "light" else "200",
                "--pad",
                "30",
                "-",
                str(svg),
            ],
            input=source,
            text=True,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["rsvg-convert", "-z", "2", str(svg), "-o", str(svg.with_suffix(".png"))],
            check=True,
        )
        subprocess.run(
            ["rsvg-convert", "-f", "pdf", str(svg), "-o", str(svg.with_suffix(".pdf"))],
            check=True,
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for mode in ("light", "dark"):
        hwostyle.use(mode)
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "svg.fonttype": "none",
            }
        )
        geometry(mode)
        anchors = time_coordinates(mode)
        pixels(mode)
        diagrams(mode)
    manifest = {
        "purpose": "Independent explanatory schematics; not evidence of repaired library behavior",
        "build_command": "python tools/build_conventions_figures.py",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(SOURCE.glob("*.d2"))]
        },
        "versions": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "d2": subprocess.run(
                ["d2", "--version"], check=True, text=True, capture_output=True
            ).stdout.strip(),
            "rsvg": subprocess.run(
                ["rsvg-convert", "--version"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
        },
        "anchors": anchors,
        "outputs": [
            p.name
            for p in sorted(OUT.iterdir())
            if p.suffix in {".png", ".svg", ".pdf"}
        ],
    }
    (OUT / "conventions-figure-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Built {len(manifest['outputs'])} figure files in {OUT}")


if __name__ == "__main__":
    main()
