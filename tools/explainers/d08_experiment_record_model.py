"""The physical experiment, the acquired record and the assumed model.

Two stills. The overview separates what exists in the simulated world (the
scene and its truth record), what is actually measured (the commanded visit,
the detector pixels and the retained records), what is hypothesized (the
inference model) and what decides (the observing policy), with evaluation
in its own simulation-only band. Truth reaches the pixels only through the
simulated light and the noise draw, and reaches evaluation directly; neither
truth nor scores feed inference or observing policy. Two visits of one
target give a detection record and a nondetection record, and the second
visit's pixels also show the alternative forced-photometry experiment,
whose reported value lies below the threshold.

The specialization repeats the fit-and-forecast tutorial on a time axis:
five acquired astrometric epochs, the posterior of the position from the
tutorial's own fit, and three forecast epochs that carry no data, with the
simulated truth drawn only for comparison.

The numbers printed on the record cards are computed from the drawn pixels;
tests/test_explainer_d08.py recomputes them independently.
"""

import dataclasses
import functools
import math

import hwostyle
import matplotlib
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Rectangle,
)

from explainers import _common as ex

# Scalar reporting experiment on a small detector cutout. Every pixel holds a
# known sky background plus the planet's pixel-integrated Gaussian image plus
# independent Gaussian noise; the reduction subtracts the known background
# and sums the pixels whose centers lie within APERTURE_RADIUS_PX of the fixed
# target position (the 3 by 3 block). The reported flux F is that sum, its
# standard deviation before any selection is PIXEL_NOISE_E * sqrt(9), and
# D = 1 precisely when F > THRESHOLD_E.
N_PIX = 9
PSF_SIGMA_PX = 0.9
BACKGROUND_E = 20.0
PIXEL_NOISE_E = 2.0
APERTURE_RADIUS_PX = 1.5
THRESHOLD_E = 30.0
VISIT_FLUX_E = (100.0, 20.0)
NOISE_SEED = 20260925

# The fit-and-forecast tutorial's setup (docs/examples/fit-and-forecast.md).
ORBIT = {
    "period_d": 600.0,
    "eccentricity": 0.3,
    "inclination_deg": 60.0,
    "node_deg": 30.0,
    "arg_periastron_deg": 70.0,
    "periastron_jd": 2461000.0,
    "distance_pc": 10.0,
}
T_REF_JD = 2461160.0
T_OBS_D = (-140.0, -70.0, 0.0, 70.0, 140.0)
T_FUTURE_D = (240.0, 320.0, 400.0)
SIGMA_MAS = 5.0
TUTORIAL_SEED = 7


# The image experiment


def _pixel_integrated_gaussian(n_pix, sigma_px):
    """Unit-flux Gaussian image integrated over each pixel, centered."""
    edges = np.arange(n_pix + 1) - 0.5 * n_pix
    cdf = np.array(
        [0.5 * (1.0 + math.erf(x / (sigma_px * math.sqrt(2.0)))) for x in edges]
    )
    one_d = np.diff(cdf)
    return np.outer(one_d, one_d)


def aperture_mask():
    """Pixels whose centers lie within the aperture radius of the center."""
    centers = np.arange(N_PIX) - 0.5 * (N_PIX - 1)
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    return np.hypot(xx, yy) <= APERTURE_RADIUS_PX


def visit_images():
    """The two simulated raw cutouts, in electrons per pixel."""
    rng = np.random.default_rng(NOISE_SEED)
    psf = _pixel_integrated_gaussian(N_PIX, PSF_SIGMA_PX)
    return [
        BACKGROUND_E + flux * psf + PIXEL_NOISE_E * rng.standard_normal((N_PIX, N_PIX))
        for flux in VISIT_FLUX_E
    ]


def reduce(image):
    """Background-subtracted aperture sum and its standard deviation."""
    mask = aperture_mask()
    flux = float(np.sum(image[mask] - BACKGROUND_E))
    sigma = PIXEL_NOISE_E * math.sqrt(int(mask.sum()))
    return flux, sigma


def record_cards():
    """The three record cards: title, optional subtitle, and rows.

    A row is ``(field, name, value, gloss, kind)`` with ``kind`` either
    ``"measured"`` (a random outcome of the visit) or ``"setting"`` (a fixed,
    known property of the experiment); ``None`` marks the divider before the
    settings.
    """
    (f1, s1), (f2, s2) = (reduce(img) for img in visit_images())
    if not (f1 > THRESHOLD_E >= f2):
        msg = "the example needs a detection on visit 1 and none on visit 2"
        raise ValueError(msg)
    unit = r"e$^-$"

    def settings(sigma):
        return [
            None,
            ("L", "L", f"{THRESHOLD_E:.0f} {unit}", "threshold", "setting"),
            ("sigma", r"$\sigma$", f"{sigma:.1f} {unit}", "noise of F", "setting"),
        ]

    return {
        "visit1": (
            "record R, visit 1",
            None,
            [
                ("D", "D", "1 (yes)", "detected?", "measured"),
                ("F", "F", f"{f1:.1f} {unit}", "flux", "measured"),
                *settings(s1),
            ],
        ),
        "visit2": (
            "record R, visit 2",
            None,
            [
                ("D", "D", "0 (no)", "detected?", "measured"),
                ("F", "F", "not reported", "flux", "measured"),
                *settings(s1),
            ],
        ),
        "forced": (
            "forced photometry",
            "alternative experiment:\nreplaces visit 2 record",
            [
                ("D", "D", "0 (no)", "detected?", "measured"),
                ("F", "F", f"{f2:.1f} {unit}", "flux, below L", "measured"),
                *settings(s2),
            ],
        ),
    }


# Hand-rolled boxes, cards and arrows


def _lighter(*colors):
    """The lightest of the given colors, for marks over the dark end of magma."""
    return max(colors, key=lambda c: sum(to_rgb(c)))


def _box(ax, center, size, cast, *, style, title, lines=()):
    """A labeled box: truth (dashed gray), command (gray) or model (round pink)."""
    (cx, cy), (w, h) = center, size
    if style == "truth":
        edge, ls, shape = (
            cast["scenery"].color,
            "--",
            "round,pad=0.02,rounding_size=0.12",
        )
    elif style == "model":
        edge, ls, shape = hwostyle.roles.model, "-", "round,pad=0.02,rounding_size=0.25"
    else:
        edge, ls, shape = cast.neutral(0.6), "-", "square,pad=0.02"
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.5 * w, cy - 0.5 * h),
            w,
            h,
            boxstyle=shape,
            facecolor=cast.background,
            edgecolor=edge,
            lw=cast.layout.lw,
            ls=ls,
            zorder=3,
        )
    )
    n = 1 + len(lines)
    step = h / (n + 0.4)
    top = cy + 0.5 * h - 0.7 * step
    ax.text(
        cx,
        top,
        title,
        ha="center",
        va="center",
        color=cast.text,
        fontweight="bold",
        zorder=4,
    )
    for k, line in enumerate(lines, start=1):
        ax.text(
            cx,
            top - k * step,
            line,
            ha="center",
            va="center",
            color=cast.neutral(0.8),
            fontsize=cast.layout.small_pt,
            zorder=4,
        )


def _card_height(rows, subtitle, row_h):
    """Height of a record card with these rows and subtitle."""
    n_sub = 0 if subtitle is None else subtitle.count("\n") + 1
    return row_h * (len(rows) + n_sub + 1.7)


def _card(ax, key, origin, width, row_h, cast, *, title, subtitle, rows, dashed=False):
    """A table-like record card: measured fields above, fixed settings below."""
    x0, y_top = origin
    n_sub = 0 if subtitle is None else subtitle.count("\n") + 1
    height = _card_height(rows, subtitle, row_h)
    edge = hwostyle.roles.measured
    small = cast.layout.small_pt
    ax.add_patch(
        Rectangle(
            (x0, y_top - height),
            width,
            height,
            facecolor=cast.background,
            edgecolor=edge,
            lw=cast.layout.lw,
            ls="--" if dashed else "-",
            zorder=3,
        )
    )
    ax.text(
        x0 + 0.5 * width,
        y_top - 0.55 * row_h,
        title,
        ha="center",
        va="center",
        color=cast.text,
        fontweight="bold",
        fontsize=small,
        zorder=4,
    )
    y = y_top - 1.1 * row_h
    if subtitle is not None:
        ax.text(
            x0 + 0.5 * width,
            y - 0.05 * row_h,
            subtitle,
            ha="center",
            va="top",
            color=cast.neutral(0.8),
            fontstyle="italic",
            fontsize=small,
            linespacing=1.0,
            zorder=4,
        )
        y -= n_sub * row_h
    ax.plot([x0, x0 + width], [y, y], color=edge, lw=0.6 * cast.layout.lw, zorder=4)
    col = x0 + 0.3
    for row in rows:
        y -= row_h
        if row is None:
            ax.plot(
                [x0 + 0.08, x0 + width - 0.08],
                [y + 0.5 * row_h] * 2,
                color=cast.neutral(0.35),
                lw=0.5 * cast.layout.lw,
                zorder=4,
            )
            ax.text(
                x0 + 0.1,
                y - 0.05 * row_h,
                "fixed, known settings",
                ha="left",
                va="center",
                color=cast.neutral(0.7),
                fontstyle="italic",
                fontsize=small,
                zorder=4,
            )
            continue
        field_id, name, value, gloss, kind = row
        value_color = (
            hwostyle.roles.measured if kind == "measured" else cast.neutral(0.8)
        )
        ax.text(
            col,
            y,
            name,
            ha="right",
            va="center",
            color=cast.neutral(0.8),
            fontsize=small,
            zorder=4,
        )
        ax.text(
            col + 0.1,
            y,
            value,
            ha="left",
            va="center",
            color=value_color,
            fontsize=small,
            zorder=4,
            gid=f"record-{key}-{field_id}",
        )
        ax.text(
            x0 + width - 0.1,
            y,
            gloss,
            ha="right",
            va="center",
            color=cast.neutral(0.6),
            fontstyle="italic",
            fontsize=small,
            zorder=4,
        )
    return height


def _flow(ax, start, end, cast, *, truth=False, gid=None):
    """A thin line arrow with an open head.

    Solid and neutral for a command (control flow, not data); dashed and in
    the truth gray for simulation-truth information.
    """
    color = cast["scenery"].color if truth else cast.neutral(0.65)
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="->,head_length=0.5,head_width=0.3",
        mutation_scale=1.6 * cast.layout.marker_pt,
        color=color,
        lw=cast.layout.lw,
        ls="--" if truth else "-",
        shrinkA=0,
        shrinkB=0,
        zorder=4,
        gid=gid,
    )
    ax.add_patch(patch)
    return patch


def _blocked(ax, start, end, cast, *, at=0.5):
    """A data arrow struck through with a cross: a path that must not exist."""
    ex.arrow(ax, start, end, "data", cast, color=hwostyle.roles.alert)
    mx = start[0] + at * (end[0] - start[0])
    my = start[1] + at * (end[1] - start[1])
    ax.plot(
        [mx],
        [my],
        marker="X",
        ms=1.9 * cast.layout.marker_pt,
        color=hwostyle.roles.alert,
        mec=cast.background,
        mew=0.5,
        ls="none",
        zorder=6,
    )
    return mx, my


def _no_path_label(ax, xy, cast, *, side="right"):
    """Label a crossed arrow, leaving about 3 points between cross and text."""
    offset = 0.95 * cast.layout.marker_pt + 3.0
    return ex.halo(
        ax.annotate(
            "no path",
            xy,
            xytext=(offset if side == "right" else -offset, 0),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="center",
            color=hwostyle.roles.alert,
            fontstyle="italic",
            fontsize=cast.layout.small_pt,
        ),
        cast,
    )


# Overview figure


def build_overview(layout, cast):
    """Simulated world, acquisition and record, model and policy, and scoring.

    The slide keeps the documentation geometry, which is height-limited on a
    16:9 frame, so its type is set 1.4 times the documentation sizes rather
    than the talk defaults; that is as large as the cards allow.
    """
    if not layout.is_slide:
        return _draw_overview(layout, cast)
    talk = dataclasses.replace(
        layout, font_pt=14.0, small_pt=12.0, title_pt=20.0, marker_pt=10.0, lw=1.8
    )
    rc = {"font.size": talk.font_pt, "axes.titlesize": talk.title_pt}
    with matplotlib.rc_context(rc):
        return _draw_overview(talk, ex.Cast(dict(cast), mode=cast.mode, layout=talk))


def _draw_overview(layout, cast):
    """Draw the overview in one layout."""
    fig, ax = ex.figure(layout, doc_height_in=6.1)
    ax.set(xlim=(0.0, 14.4), ylim=(-0.1, 11.2), aspect="equal")
    ax.axis("off")
    small = layout.small_pt
    lift = 9 if layout.is_slide else 4  # arrow label clearance, points
    if layout.is_slide:
        fig.suptitle(
            "What exists, what is measured, what is inferred", fontweight="bold"
        )

    # Zone headings and separators.
    heads = [
        (1.8, "Simulated world", "exists; hidden from inference"),
        (7.1, "Acquisition and record", "what is actually measured"),
        (12.55, "Model and policy", "use only the record"),
    ]
    for x, head, sub in heads:
        ax.text(
            x, 10.85, head, ha="center", va="center", color=cast.text, fontweight="bold"
        )
        ex.note(ax, (x, 10.42), sub, cast, ha="center", va="center")
    for x in (3.6, 10.65):
        ax.plot(
            [x, x], [0.2, 10.15], color=cast["furniture"].color, lw=layout.lw, zorder=1
        )

    # Simulated world: the truth region holds the scene and its truth record.
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.3),
            3.2,
            9.75,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor="none",
            edgecolor=cast["scenery"].color,
            lw=layout.lw,
            ls="--",
            zorder=1,
        )
    )
    ex.note(ax, (0.35, 9.8), "simulation truth", cast, ha="left", va="center")
    # A schematic orbit with the star at one focus of the ellipse.
    aperture_y = 7.3
    star, semi_major, semi_minor, tilt = (1.0, 7.15), 1.05, 0.78, math.radians(10.0)
    focal = math.sqrt(semi_major**2 - semi_minor**2)
    axis = (math.cos(tilt), math.sin(tilt))
    center = (star[0] + focal * axis[0], star[1] + focal * axis[1])
    phase = math.radians(75.0)
    local = (semi_major * math.cos(phase), semi_minor * math.sin(phase))
    planet = (
        center[0] + local[0] * axis[0] - local[1] * axis[1],
        center[1] + local[0] * axis[1] + local[1] * axis[0],
    )
    ax.add_patch(
        Ellipse(
            center,
            2 * semi_major,
            2 * semi_minor,
            angle=math.degrees(tilt),
            fill=False,
            **cast["scenery"].line_kw(),
        )
    )
    ex.mark(ax, "star", star, cast, label="star")
    ex.mark(ax, "planet", planet, cast, label="planet", label_offset_pt=(0, 7))
    truth_box = (1.8, 1.6)
    _box(
        ax,
        truth_box,
        (2.95, 1.9),
        cast,
        style="truth",
        title="Truth record",
        lines=[r"true orbit and flux, $\theta_{\rm true}$", "planet ID, noise seed"],
    )

    # Physical light from the scene to the telescope, not to scale.
    planet_ray = (planet[0] + 0.2, planet[1] - 0.03)
    star_ray = (star[0] + 0.3, star[1] - 0.01)
    rays = (
        (planet_ray, (4.05, aperture_y + 0.25)),
        (star_ray, (4.05, aperture_y - 0.25)),
    )
    for (start, end), who in zip(rays, ("planet", "star"), strict=True):
        ex.arrow(ax, start, end, "ray", cast, source=who)
        frac = (3.2 - start[0]) / (end[0] - start[0])
        ex.scale_break(
            ax,
            (3.2, start[1] + frac * (end[1] - start[1])),
            cast,
            along_deg=math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])),
        )
    ex.note(ax, (2.4, 6.15), "simulated light", cast, ha="center", va="center")

    # The truth record reaches the detector only through the noise draw.
    detector_foot = (5.15, aperture_y - 0.58)
    _flow(ax, (3.2, 2.58), detector_foot, cast, truth=True, gid="noise-draw")
    ex.note(ax, (4.42, 4.6), "noise draw", cast, ha="left", va="center")

    # A crossed stub from the truth record toward the model and policy
    # column, cut where it would leave the simulated world.
    _blocked(ax, (3.32, 1.85), (5.3, 1.85), cast, at=0.14)
    ex.note(
        ax,
        (0.35, 2.75),
        "no path: truth or scores\nnever reach inference\nor policy",
        cast,
        ha="left",
        va="bottom",
        color=hwostyle.roles.alert,
    )

    # Commanded visit and the instrument.
    ex.aperture(ax, (4.1, aperture_y), 1.2, cast)
    ex.lens(ax, (4.55, aperture_y), 1.0, cast)
    ex.region(ax, "detector", Rectangle((5.02, aperture_y - 0.55), 0.26, 1.1), cast)
    ex.note(ax, (4.45, 8.35), "telescope,\ndetector", cast, ha="right", va="center")
    _box(
        ax,
        (5.15, 9.45),
        (2.95, 1.2),
        cast,
        style="command",
        title="Configured visit",
        lines=["a command, not data:", "target, epoch, exposure"],
    )
    _flow(ax, (4.6, 8.83), (4.6, aperture_y + 0.68), cast)

    # Raw pixels of the two visits, on one pinned linear readout scale, with
    # the summed 3 by 3 block outlined.
    images = visit_images()
    lo = BACKGROUND_E - 3.0 * PIXEL_NOISE_E
    hi = float(max(img.max() for img in images))
    pitch = 0.14
    side = N_PIX * pitch
    x_thumb = 5.72
    thumbs = [(x_thumb, 6.95), (x_thumb, 4.69)]
    names = ("visit 1", "visit 2 (fainter)")
    edge = _lighter(cast.background, cast.text)
    for k, ((x0, y0), img, name) in enumerate(zip(thumbs, images, names, strict=True)):
        ex.pixel_grid(
            ax, (x0, y0), (N_PIX, N_PIX), pitch, cast, values=img, vmin=lo, vmax=hi
        )
        cx, cy = x0 + 0.5 * side, y0 + 0.5 * side
        half = (int(APERTURE_RADIUS_PX) + 0.5) * pitch
        ax.add_patch(
            Rectangle(
                (cx - half, cy - half),
                2 * half,
                2 * half,
                fill=False,
                ec=edge,
                lw=0.9 * layout.lw,
                ls=":",
                zorder=4,
                gid=f"aperture-outline-visit{k + 1}",
            )
        )
        ex.halo(
            ax.text(
                cx,
                y0 + side + 0.06,
                name,
                ha="center",
                va="bottom",
                fontsize=small,
                color=cast.text,
            ),
            cast,
        )
    ex.arrow(ax, (5.33, 7.45), (x_thumb - 0.05, 7.45), "data", cast)
    ex.arrow(ax, (5.2, aperture_y - 0.62), (x_thumb - 0.05, 5.3), "data", cast)
    ex.note(
        ax,
        (4.12, 4.3),
        "raw pixels [e$^-$]\n\nreduction:\nF = sum of the\ndotted pixels at\nthe fixed target\nposition; report\nD = 1 if F > L",
        cast,
        ha="left",
        va="top",
    )

    # The retained records, stacked from the bottom lane upward.
    cards = record_cards()
    x_card, width, row_h, gap_y = 7.45, 2.95, 0.3, 0.25
    mids = {}
    y_bottom = 1.45
    for key in ("forced", "visit2", "visit1"):
        title, subtitle, rows = cards[key]
        h = _card_height(rows, subtitle, row_h)
        top = y_bottom + h
        _card(
            ax,
            key,
            (x_card, top),
            width,
            row_h,
            cast,
            title=title,
            subtitle=subtitle,
            rows=rows,
            dashed=key == "forced",
        )
        mids[key] = top - 0.5 * h
        y_bottom = top + gap_y
    right = x_thumb + side + 0.07
    ex.arrow(ax, (right, 7.45), (x_card - 0.05, mids["visit1"]), "data", cast)
    ex.arrow(ax, (right, 5.45), (x_card - 0.05, mids["visit2"]), "data", cast)
    ex.arrow(ax, (right, 4.95), (x_card - 0.05, mids["forced"] + 0.3), "data", cast)

    # Model and policy, which see only the record; scoring, which sees truth.
    box_x, box_w = 12.55, 3.5
    box_left = box_x - 0.5 * box_w
    _box(
        ax,
        (box_x, 6.9),
        (box_w, 1.9),
        cast,
        style="model",
        title="Inference model",
        lines=[
            r"hypotheses $\theta$: orbit, flux",
            r"$p(\theta \mid R) \propto p(R \mid \theta)\,p(\theta)$",
        ],
    )
    _box(
        ax,
        (box_x, 9.5),
        (box_w, 1.25),
        cast,
        style="model",
        title="Observing policy",
        lines=["a decision rule", "acts on the posterior only"],
    )
    ax.plot(
        [10.65, 14.4],
        [3.4, 3.4],
        color=cast["scenery"].color,
        lw=0.8 * layout.lw,
        ls="--",
        zorder=1,
    )
    _box(
        ax,
        (box_x, 2.0),
        (box_w, 1.9),
        cast,
        style="truth",
        title="Evaluation",
        lines=["simulation-only scoring:", "posterior against truth"],
    )
    stop = box_left - 0.1
    ex.arrow(ax, (x_card + width + 0.05, mids["visit1"]), (stop, 7.3), "data", cast)
    ex.arrow(ax, (x_card + width + 0.05, mids["visit2"]), (stop, 6.35), "data", cast)
    ex.arrow(ax, (box_x, 7.9), (box_x, 8.82), "data", cast)
    _flow(ax, (box_left - 0.05, 9.6), (6.68, 9.6), cast)
    ex.halo(
        ax.annotate(
            "next visit",
            (8.8, 9.6),
            xytext=(0, lift),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=cast.neutral(0.65),
        ),
        cast,
    )
    ex.arrow(ax, (11.7, 5.9), (11.7, 3.0), "data", cast)
    ex.note(ax, (11.85, 3.7), "posterior", cast, ha="left", va="center")
    _no_path_label(ax, _blocked(ax, (13.5, 3.0), (13.5, 5.9), cast), cast, side="left")

    # Truth reaches evaluation directly, along the bottom lane.
    _flow(
        ax,
        (truth_box[0] + 1.5, 1.1),
        (box_left - 0.05, 1.1),
        cast,
        truth=True,
        gid="truth-lane",
    )
    ex.halo(
        ax.annotate(
            "truth, for scoring only",
            (6.3, 1.1),
            xytext=(0, -lift),
            textcoords="offset points",
            ha="center",
            va="top",
            color=cast["scenery"].color,
        ),
        cast,
    )
    return fig


# Fit-and-forecast specialization


@functools.lru_cache(maxsize=1)
def forecast_data():
    """Rerun the tutorial's simulation and fit; return plain arrays in mas.

    This follows docs/examples/fit-and-forecast.md step by step (same orbit,
    epochs, noise level, random key and samplers), in 64-bit precision
    scoped to this function so no other diagram inherits the setting.
    """
    import jax
    import jax.numpy as jnp
    import photomancy as pm
    from hwoutils import constants
    from orbix.orbit import KeplerianOrbit
    from photomancy.orbit import (
        RelativeAstromData,
        build_orbit_logdensity,
        find_init,
        orbits_from_samples,
    )

    with jax.enable_x64(True):
        ms_kg = constants.Msun2kg
        dist = ORBIT["distance_pc"]
        tp_d = ORBIT["periastron_jd"] - T_REF_JD
        cos_i = jnp.cos(jnp.deg2rad(ORBIT["inclination_deg"]))
        w_node = jnp.deg2rad(ORBIT["node_deg"])
        w_peri = jnp.deg2rad(ORBIT["arg_periastron_deg"])
        truth = KeplerianOrbit.from_period(
            ORBIT["period_d"],
            ORBIT["eccentricity"],
            cos_i,
            w_node,
            jnp.cos(w_peri),
            jnp.sin(w_peri),
            tp_d,
            Ms_kg=ms_kg,
        )
        t_obs = jnp.array(T_OBS_D)
        sigma_as = SIGMA_MAS * constants.mas2arcsec
        ra_true, dec_true = truth.position_arcsec(t_jd=t_obs, Ms_kg=ms_kg, dist_pc=dist)
        k_ra, k_dec, k_nuts, _ = jax.random.split(jax.random.PRNGKey(TUTORIAL_SEED), 4)
        ra_obs = ra_true[0] + sigma_as * jax.random.normal(k_ra, t_obs.shape)
        dec_obs = dec_true[0] + sigma_as * jax.random.normal(k_dec, t_obs.shape)
        data = RelativeAstromData.pad(
            times=t_obs,
            ra=ra_obs,
            dec=dec_obs,
            ra_err=jnp.full(t_obs.shape, sigma_as),
            dec_err=jnp.full(t_obs.shape, sigma_as),
            corr=jnp.zeros(t_obs.shape),
            planet_id=jnp.zeros(t_obs.shape, dtype=int),
        )
        problem = build_orbit_logdensity(ms_kg, dist, relative_astrom_data=data)
        z_init = problem.init_to_z(find_init(data, ms_kg, dist))
        laplace = pm.LaplaceBackend(n_steps=100, min_eigenvalue=1.0).run(
            problem.logdensity, z_init
        )
        nuts = pm.NUTSBackend(n_warmup=500, n_samples=2000).run(
            problem.logdensity, laplace.mean, k_nuts
        )
        phys = jax.vmap(problem.to_physical)(nuts.samples)

        t_grid = jnp.linspace(-170.0, 430.0, 301)
        ra_d, dec_d = orbits_from_samples(phys, ms_kg).position_arcsec(
            t_jd=t_grid, Ms_kg=ms_kg, dist_pc=dist
        )
        t_future = jnp.array(T_FUTURE_D)
        ra_f, dec_f = orbits_from_samples(phys, ms_kg).position_arcsec(
            t_jd=t_future, Ms_kg=ms_kg, dist_pc=dist
        )
        ra_t, dec_t = truth.position_arcsec(t_jd=t_grid, Ms_kg=ms_kg, dist_pc=dist)
        ra_tf, dec_tf = truth.position_arcsec(t_jd=t_future, Ms_kg=ms_kg, dist_pc=dist)
        mas = constants.arcsec2mas
        return {
            "t_grid": np.asarray(t_grid),
            "obs": mas * np.stack([np.asarray(ra_obs), np.asarray(dec_obs)]),
            "draws": mas * np.stack([np.asarray(ra_d), np.asarray(dec_d)]),
            "future": mas * np.stack([np.asarray(ra_f), np.asarray(dec_f)]),
            "truth": mas * np.stack([np.asarray(ra_t[0]), np.asarray(dec_t[0])]),
            "truth_future": mas
            * np.stack([np.asarray(ra_tf[0]), np.asarray(dec_tf[0])]),
        }


DRAW_CURVES = (0, 400, 800, 1200, 1600)


def build_forecast(layout, cast):
    """Acquired epochs, posterior of the position and forecast epochs in time."""
    d = forecast_data()
    fig, axes = ex.figure(layout, doc_height_in=4.6, nrows=2, sharex=True)
    if layout.is_slide:
        fig.suptitle("Acquired epochs versus forecasts", fontweight="bold")
    model = hwostyle.roles.model
    measured = hwostyle.roles.measured
    truth_color = cast["scenery"].color
    truth_kw = {"color": truth_color, "lw": layout.lw, "ls": "-", "zorder": 3}
    alpha_band = 0.22 if cast.mode != "dark" else 0.3
    t_last = T_OBS_D[-1]
    t = d["t_grid"]
    names = ("East offset [mas]", "North offset [mas]")
    for k, (ax, name) in enumerate(zip(axes, names, strict=True)):
        ax.axvspan(
            t_last,
            t[-1],
            color=cast.neutral(0.08),
            lw=0,
            zorder=0,
            gid=f"forecast-shade-{k}",
        )
        ax.axvline(
            t_last,
            color=truth_color,
            lw=0.8 * layout.lw,
            ls=":",
            zorder=1,
            gid=f"last-acquired-{k}",
        )
        lo, hi = np.percentile(d["draws"][k], [5.0, 95.0], axis=0)
        ax.fill_between(
            t, lo, hi, color=model, alpha=alpha_band, lw=0, zorder=1, gid=f"band-{k}"
        )
        for j in DRAW_CURVES:
            ax.plot(
                t,
                d["draws"][k][j],
                color=model,
                lw=0.6 * layout.lw,
                alpha=0.8,
                zorder=2,
            )
        ax.plot(t, d["truth"][k], **truth_kw)

        # Each forecast is the whole distribution of the draws at that epoch:
        # a violin, with the 90 percent interval as a capped line.
        fut = d["future"][k]
        parts = ax.violinplot(
            [fut[:, j] for j in range(len(T_FUTURE_D))],
            positions=T_FUTURE_D,
            widths=44.0,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(model)
            body.set_edgecolor(model)
            body.set_alpha(0.32 if cast.mode != "dark" else 0.45)
            body.set_zorder(4)
        f_lo, f_hi = np.percentile(fut, [5.0, 95.0], axis=0)
        for j, epoch in enumerate(T_FUTURE_D):
            ax.plot(
                [epoch, epoch],
                [f_lo[j], f_hi[j]],
                color=model,
                lw=0.8 * layout.lw,
                marker="_",
                ms=0.9 * layout.marker_pt,
                mew=layout.lw,
                zorder=5,
                gid=f"forecast-interval-{k}-{j}",
            )
        ax.plot(
            T_FUTURE_D,
            d["truth_future"][k],
            ls="none",
            marker="o",
            ms=0.6 * layout.marker_pt,
            color=truth_color,
            mec=cast.background,
            mew=0.6,
            zorder=6,
            gid=f"truth-future-{k}",
        )
        acquired = ax.errorbar(
            T_OBS_D,
            d["obs"][k],
            yerr=SIGMA_MAS,
            fmt="o",
            ms=0.8 * layout.marker_pt,
            color=measured,
            ecolor=measured,
            elinewidth=layout.lw,
            capsize=0.4 * layout.marker_pt,
            zorder=7,
        )
        acquired[0].set_gid(f"acquired-{k}")
        ax.set_ylabel(name)
        ax.spines[["top", "right"]].set_visible(False)
    top, bottom = axes
    bottom.set_xlim(t[0], t[-1])
    bottom.set_xticks([*T_OBS_D, *T_FUTURE_D])
    bottom.set_xticklabels([f"{v:+.0f}" if v else "0" for v in (*T_OBS_D, *T_FUTURE_D)])
    bottom.set_xlabel(f"Days from reference epoch (JD {T_REF_JD:.0f})")
    band = top.get_xaxis_transform()
    top.text(
        0.5 * (t[0] + t_last),
        1.03,
        f"acquired: {len(T_OBS_D)} epochs, {SIGMA_MAS:.0f} mas per axis",
        transform=band,
        ha="center",
        va="bottom",
        color=measured,
        gid="header-acquired",
    )
    top.text(
        0.5 * (t_last + t[-1]),
        1.03,
        "forecast: no data yet",
        transform=band,
        ha="center",
        va="bottom",
        color=model,
        gid="header-forecast",
    )

    handles = [
        bottom.errorbar(
            [], [], yerr=[], fmt="o", color=measured, ms=0.8 * layout.marker_pt
        )[0],
        (
            Patch(facecolor=model, alpha=alpha_band, lw=0),
            bottom.plot([], [], color=model, lw=0.6 * layout.lw)[0],
        ),
        (
            Patch(facecolor=model, alpha=0.45, edgecolor=model),
            bottom.plot([], [], color=model, lw=0.8 * layout.lw)[0],
        ),
        bottom.plot([], [], marker="o", ms=0.5 * layout.marker_pt, **truth_kw)[0],
    ]
    labels = [
        "measurement (acquired)",
        "posterior of position: 90% band, 5 draws",
        "forecast position: all draws, 90% interval",
        "simulated truth (comparison only)",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        frameon=False,
        ncols=2,
        borderpad=1.0 if not layout.is_slide else 0.4,
    )
    ex.badge(top, cast, "simulated fit", loc="upper left")
    return fig


def _numbers():
    """The printed record values, as the captions quote them."""
    (f1, s1), (f2, _) = (reduce(img) for img in visit_images())
    return {"f1": f1, "f2": f2, "sigma": s1, "n_ap": int(aperture_mask().sum())}


_N = _numbers()

OVERVIEW_CAPTION = (
    "The proposed information boundary of a simulated imaging experiment "
    "({ref}`records-reporting-law`, {ref}`records-identities`, "
    "{ref}`records-campaign-causality`). The simulated world holds the scene and "
    "a truth record. Truth reaches the detector only through the simulated light "
    "and the noise draw, and reaches evaluation directly, for simulation-only "
    "scoring; neither truth nor scores reach the inference model or the observing "
    "policy, which is a decision rule rather than a hypothesis. Two visits of one "
    "target give the two records of the report-only-on-detection experiment, "
    "R = (D = 1, F) and R = (D = 0), where D records whether F exceeded the "
    "threshold L. The threshold L and the standard deviation sigma of F before "
    "any selection are fixed, known settings of the experiment, not measured "
    "values. The dashed card is a different experiment, forced photometry, which "
    f"replaces the visit 2 record and reports F = {_N['f2']:.1f} e$^-$ from the "
    "same pixels although F is below L ({ref}`Efron and Hastie 2016, Sec. 9.6 "
    "<source-efron2016>`). The 9 by 9 pixel cutouts are simulated raw readouts: a "
    f"known background of {BACKGROUND_E:.0f} e$^-$ per pixel, a pixel-integrated "
    "Gaussian planet image and independent Gaussian noise of "
    f"{PIXEL_NOISE_E:.0f} e$^-$ per pixel. The reduction subtracts the background "
    f"and sums the {_N['n_ap']} dotted pixels, those whose centers lie within "
    f"{APERTURE_RADIUS_PX} pixels of the fixed target position, so sigma = "
    f"{_N['sigma']:.1f} e$^-$. The next figure details the reporting law. The "
    "figure is a schematic of a proposed convention, not an example "
    "implementation, and does not assert that the current tutorial or campaign "
    "software enforces every separation it shows."
)

OVERVIEW_ALT = (
    "Schematic in three labeled zones. Left, a dashed region named simulated "
    "world holds a star at the focus of an orbit, a planet on it, and a truth "
    "record box listing the true orbit and flux, the planet ID and the noise "
    "seed. Rays of starlight and planet light cross a scale break to a telescope "
    "aperture and a detector in the middle zone; a dashed gray arrow labeled "
    "noise draw runs from the truth record to the detector, and a red crossed "
    "arrow from the truth record carries the note: no path, truth or scores "
    "never reach inference or policy. A configured visit box, a command, "
    "receives a next visit arrow from the observing policy. The detector feeds "
    "two 9 by 9 pixel images of raw electrons, visit 1 with a bright planet and "
    "visit 2 fainter, each with the summed 3 by 3 pixels outlined by a dotted "
    "square. Each record card lists measured fields above fixed experiment "
    f"settings. Visit 1 reads D = 1, detected, and F = {_N['f1']:.1f} electrons; "
    "visit 2 reads D = 0 and F not reported; both list the threshold L = "
    f"{THRESHOLD_E:.0f} electrons and sigma = {_N['sigma']:.1f} electrons. A dashed "
    "forced photometry card, labeled as an alternative experiment that replaces "
    f"the visit 2 record, reads D = 0 and F = {_N['f2']:.1f} electrons, below L. "
    "The two records feed an inference model box on the right, which feeds the "
    "observing policy. Below a dashed divider, an evaluation box for "
    "simulation-only scoring receives the posterior and, along the bottom, the "
    "truth record; a red crossed arrow from evaluation back to the inference "
    "model is labeled no path."
)

FORECAST_CAPTION = (
    "The fit-and-forecast tutorial on a time axis, a specialization of the "
    "experiment diagram in {ref}`records-reporting-law` (see also "
    "{ref}`records-campaign-causality`). Filled markers are the five acquired "
    f"relative-astrometry epochs, simulated with {SIGMA_MAS:.0f} mas of Gaussian "
    "noise per axis; the error bars are smaller than the markers. The pink band "
    "and curves are the posterior of the planet's position from the tutorial's "
    "own fit, rerun with the same orbit, epochs, random key and samplers: the 90 "
    "percent band of all 2000 draws and five of those draws. At days "
    f"{T_FUTURE_D[0]:.0f}, {T_FUTURE_D[1]:.0f} and {T_FUTURE_D[2]:.0f} each violin "
    "is the distribution of the forecast position over the 2000 draws, with its "
    "90 percent interval as a capped line; these are forecasts of the position, "
    "not measurements, and a future 5 mas measurement would add that noise in "
    "quadrature. The shaded interval holds no acquired data. The period ridge of "
    "the posterior, about 430 to 720 days, makes the day 400 east forecast "
    "bimodal, so no single central value summarizes it. The gray curve and dots "
    "are the simulated truth, drawn for comparison only and never an input to "
    "the fit; one of the six forecast truths lies outside its 90 percent "
    "interval, which is expected for a few intervals of this kind. The fit uses "
    "photomancy's current conventions and the reporting law is pending "
    "({ref}`decision-reporting-law`); the figure illustrates the proposed "
    "separation and does not assert that the tutorial software enforces it."
)

FORECAST_ALT = (
    "Two stacked panels sharing a time axis in days from the reference epoch, "
    f"JD {T_REF_JD:.0f}, east offset above and north offset below, in "
    "milliarcseconds. Five filled cyan markers at days -140, -70, 0, 70 and 140 "
    "lie on a narrow pink posterior band with five thin pink draws and a gray "
    "truth curve. A dotted vertical line at day 140 separates the acquired "
    "interval, labeled acquired: 5 epochs, 5 mas per axis, from a shaded "
    "interval labeled forecast: no data yet. In the shaded interval the pink "
    "band widens and the draws fan apart. Pink violins with capped 90 percent "
    "interval lines show the forecast positions at days 240, 320 and 400, and "
    "small gray dots mark the truth there. The day 400 east violin has two "
    "separated lobes. The north truth at day 400 lies just above its 90 percent "
    "interval; the other five truths lie inside theirs."
)


FIGURES = [
    ex.FigureSpec(
        slug="d08-experiment-record-model",
        build=build_overview,
        caption=OVERVIEW_CAPTION,
        alt=OVERVIEW_ALT,
        status="schematic, proposed information boundary; simulated pixels",
        params={
            "n_pix": N_PIX,
            "psf_sigma_px": PSF_SIGMA_PX,
            "background_e": BACKGROUND_E,
            "pixel_noise_e": PIXEL_NOISE_E,
            "aperture_radius_px": APERTURE_RADIUS_PX,
            "threshold_e": THRESHOLD_E,
            "visit_flux_e": list(VISIT_FLUX_E),
            "noise_seed": NOISE_SEED,
        },
    ),
    ex.FigureSpec(
        slug="d08-fit-forecast-timeline",
        build=build_forecast,
        caption=FORECAST_CAPTION,
        alt=FORECAST_ALT,
        status="simulated; tutorial setup",
        params={
            **ORBIT,
            "t_ref_jd": T_REF_JD,
            "t_obs_d": list(T_OBS_D),
            "t_future_d": list(T_FUTURE_D),
            "sigma_mas": SIGMA_MAS,
            "seed": TUTORIAL_SEED,
        },
    ),
]
ANIMATIONS = []
