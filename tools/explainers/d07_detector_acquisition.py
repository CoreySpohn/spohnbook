"""A pixel's acquisition and readout schedule.

Teaches which detector terms accumulate with live time (photoelectrons and
dark charge), which occur once per frame or read (clock-induced charge and
read noise), and why accumulated live time differs from the elapsed time a
frame schedule occupies. Every number comes from the radiometry chapter's
worked count example (a four-pixel aperture observed in ten 10 s frames) or
from the photon-to-electron reference case, and the tests recompute each one
by hand.

Motion ground for the animation: narration. The clock advances through
integrations and reads while the aperture's charge and its recorded sum
change; the still carries the complete timeline and budget without it.
"""

import math

import hwostyle
import numpy as np
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from matplotlib.patches import Polygon, Rectangle

from explainers import _common as ex

# The worked count example of the radiometry chapter, section
# "Worked count example". Rates of source photons are aperture totals; the
# detector terms are per pixel.
COUNT = {
    "n_pixels": 4,
    "planet_photon_rate_per_s": 20.0,
    "background_photon_rate_per_s": 80.0,
    "qe": 0.5,
    "n_frames": 10,
    "frame_time_s": 10.0,
    "read_time_s": 0.05,
    "dark_e_per_pixel_s": 0.01,
    "cic_e_per_pixel_frame": 0.02,
    "read_noise_e_rms_per_pixel_read": 2.0,
}

# The photon-to-electron reference case: one 1 s exposure of a 1 nm
# interval at the photon density of 1 Jy at 700 nm, on 1 m^2, spread over
# four pixels, with no dark current, CIC or read noise.
REFERENCE = {
    "flux_jy": 1.0,
    "wavelength_nm": 700.0,
    "area_m2": 1.0,
    "bandwidth_nm": 1.0,
    "exposure_s": 1.0,
    "n_pixels": 4,
    "qe": 0.5,
}

# Seed of the one simulated realization in the animation. Seeds 0 to 399
# were scanned; this one ends 0.5 standard deviations below the expected
# total with every frame value within 1.1 standard deviations of its mean,
# so the realization is neither a lucky match nor an outlier.
SEED = 16


def count_budget(p=COUNT):
    """Aperture-summed mean and variance of the worked count example.

    Returns a dict with the per-term rows ``(name, grows_with, calc, mean,
    variance)``, the totals, the SNR after subtracting the known background,
    dark and CIC means, the live time and the occupied elapsed time.
    """
    live = p["n_frames"] * p["frame_time_s"]
    n_pix, n_f, qe = p["n_pixels"], p["n_frames"], p["qe"]
    planet = p["planet_photon_rate_per_s"] * qe * live
    background = p["background_photon_rate_per_s"] * qe * live
    dark = n_pix * p["dark_e_per_pixel_s"] * live
    cic = n_pix * p["cic_e_per_pixel_frame"] * n_f
    read = n_pix * p["read_noise_e_rms_per_pixel_read"] ** 2 * n_f
    rows = [
        (
            "planet photoelectrons",
            "live time",
            f"{_fmt(p['planet_photon_rate_per_s'])}/s (aperture) x {_fmt(qe)} x {_fmt(live)} s",
            planet,
            planet,
        ),
        (
            "background photoelectrons",
            "live time",
            f"{_fmt(p['background_photon_rate_per_s'])}/s (aperture) x {_fmt(qe)} x {_fmt(live)} s",
            background,
            background,
        ),
        (
            "dark charge",
            "live time",
            f"{n_pix} px x {_fmt(p['dark_e_per_pixel_s'])}/s x {_fmt(live)} s",
            dark,
            dark,
        ),
        (
            "clock-induced charge (CIC)",
            "frames",
            f"{n_pix} px x {_fmt(p['cic_e_per_pixel_frame'])} x {n_f} frames",
            cic,
            cic,
        ),
        (
            "read noise",
            "reads",
            f"{n_pix} px x {_fmt(p['read_noise_e_rms_per_pixel_read'])}$^2$ x {n_f} reads",
            0.0,
            read,
        ),
    ]
    mean = sum(r[3] for r in rows)
    variance = sum(r[4] for r in rows)
    return {
        "rows": rows,
        "mean": mean,
        "variance": variance,
        "snr": planet / math.sqrt(variance),
        "live_s": live,
        "elapsed_s": n_f * (p["frame_time_s"] + p["read_time_s"]),
    }


def reference_budget(p=REFERENCE):
    """Summed expected variance of the noiseless reference case."""
    density = float(jy_to_photons_per_nm_per_m2(p["flux_jy"], p["wavelength_nm"]))
    photons = density * p["area_m2"] * p["bandwidth_nm"] * p["exposure_s"]
    return {"density": density, "photons": photons, "variance": p["qe"] * photons}


def simulate(p=COUNT, seed=SEED):
    """One realization of the aperture sum, frame by frame.

    Photoelectrons and dark charge arrive as one Poisson process at the
    aperture's summed rate during each live interval; each read adds one
    Poisson CIC draw and one Gaussian read-noise draw for the four pixels
    together (sums of independent Poisson and Gaussian terms keep their
    form). Draw order: per frame, the arrival count, the arrival times, the
    CIC count, then the read noise.

    Returns a dict of arrays: ``starts`` (elapsed start of each live
    interval, s), ``arrivals`` (list of arrival times within each frame, s),
    ``values`` (the frame values, electrons), ``rate`` (expected e/s in the
    wells) and ``frame_mean`` (expected frame value, electrons).
    """
    rng = np.random.default_rng(seed)
    n_pix, t_f, t_r = p["n_pixels"], p["frame_time_s"], p["read_time_s"]
    rate = (p["planet_photon_rate_per_s"] + p["background_photon_rate_per_s"]) * p[
        "qe"
    ] + n_pix * p["dark_e_per_pixel_s"]
    cic_mean = n_pix * p["cic_e_per_pixel_frame"]
    read_sd = math.sqrt(n_pix) * p["read_noise_e_rms_per_pixel_read"]
    arrivals, values = [], []
    for _ in range(p["n_frames"]):
        n_k = int(rng.poisson(rate * t_f))
        arrivals.append(np.sort(rng.uniform(0.0, t_f, n_k)))
        cic_k = int(rng.poisson(cic_mean))
        eps_k = float(rng.normal(0.0, read_sd))
        values.append(n_k + cic_k + eps_k)
    starts = np.arange(p["n_frames"]) * (t_f + t_r)
    return {
        "starts": starts,
        "arrivals": arrivals,
        "values": np.asarray(values),
        "rate": rate,
        "frame_mean": rate * t_f + cic_mean,
    }


def _fmt(value):
    """Plain number: integers without a decimal point, else trimmed."""
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):d}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


# Small hand-rolled glyphs


def _gauss_icon(ax, center, width, height, color, lw):
    """A small Gaussian curve: the read-noise glyph."""
    x = np.linspace(-1.0, 1.0, 41)
    y = np.exp(-0.5 * (x / 0.33) ** 2)
    return ax.plot(
        center[0] + 0.5 * width * x,
        center[1] - 0.5 * height + height * y,
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=5,
    )[0]


def _cic_glyph(ax, xy, cast, size):
    """An open diamond: the clock-induced-charge glyph."""
    return ax.plot(
        [xy[0]],
        [xy[1]],
        marker="D",
        ms=size,
        mfc=cast.background,
        mec=cast.neutral(0.85),
        mew=cast.layout.lw * 0.8,
        ls="none",
        zorder=6,
    )[0]


def _live_bar(ax, x, y, w, h, cast, *, zorder=2):
    """A live interval: the hatched bar of the timeline."""
    return ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=cast.neutral(0.15),
            edgecolor=cast.neutral(0.75),
            lw=0.8 * cast.layout.lw,
            hatch="////",
            zorder=zorder,
        )
    )


def _read_bar(ax, x, y, w, h, cast, *, zorder=2):
    """A read: the solid bar of the timeline."""
    return ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=cast.neutral(0.7),
            edgecolor=cast.neutral(0.75),
            lw=0.8 * cast.layout.lw,
            zorder=zorder,
        )
    )


def _bracket(ax, x0, x1, y, text, cast, *, fill=None, height=2.6):
    """A bracket drawn as a thin copy of a timeline bar, labeled beneath.

    ``fill`` is ``"live"`` (hatched, like the live intervals), ``"read"``
    (solid, like the reads) or None (an outline only), so the bracket over a
    part of the pixel diagram carries the same fill as the part of the
    schedule it corresponds to.
    """
    if fill == "live":
        _live_bar(ax, x0, y, x1 - x0, height, cast, zorder=4)
    elif fill == "read":
        _read_bar(ax, x0, y, x1 - x0, height, cast, zorder=4)
    else:
        ax.plot(
            [x0, x0, x1, x1],
            [y + height, y, y, y + height],
            color=cast.text,
            lw=0.8 * cast.layout.lw,
            zorder=4,
        )
    return ex.halo(
        ax.text(
            0.5 * (x0 + x1),
            y - 1.0,
            text,
            ha="center",
            va="top",
            color=cast.text,
        ),
        cast,
    )


def _amplifier(ax, x, y, w, h, cast):
    """A read amplifier as the usual triangle symbol."""
    tri = Polygon(
        [(x, y - 0.5 * h), (x, y + 0.5 * h), (x + w, y)],
        closed=True,
        facecolor=cast.background,
        edgecolor=cast["optics"].color,
        lw=cast.layout.lw,
        zorder=4,
    )
    ax.add_patch(tri)
    return tri


def _well(ax, x0, x1, y0, y1, cast, *, faint=False):
    """A pixel's charge well in side view: two walls and a floor."""
    color = cast.neutral(0.45) if faint else cast["detector"].color
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=cast.neutral(0.08),
            edgecolor="none",
            zorder=1,
        )
    )
    ax.plot(
        [x0, x0, x1, x1],
        [y1, y0, y0, y1],
        color=color,
        lw=2.0 * cast.layout.lw,
        solid_joinstyle="miter",
        zorder=3,
    )


def _plain_arrow(ax, start, end, cast, *, color=None):
    """A thin arrow for charge moving inside the detector (not light, not data)."""
    color = cast.neutral(0.7) if color is None else color
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": cast.layout.lw,
            "mutation_scale": 1.6 * cast.layout.marker_pt,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def _label(ax, xy, text, cast, **kw):
    """Plain label in the text color with a background halo."""
    kw.setdefault("color", cast.text)
    return ex.halo(ax.text(*xy, text, **kw), cast)


# The acquisition diagram: one frame in detail above the frame schedule


def _electrons(ax, cast, box, *, n_photo, n_dark, ms):
    """Photoelectrons (filled dots) and dark charge (open dots) in a well."""
    x0, x1, y0, y1 = box
    rng = np.random.default_rng(3)
    xs = rng.uniform(x0 + 1.5, x1 - 1.5, n_photo + n_dark)
    ys = rng.uniform(y0 + 2.0, y0 + 0.5 * (y1 - y0), n_photo + n_dark)
    ax.plot(
        xs[:n_photo],
        ys[:n_photo],
        ls="none",
        marker="o",
        ms=ms,
        color=cast.neutral(0.85),
        zorder=5,
    )
    if n_dark:
        ax.plot(
            xs[n_photo:],
            ys[n_photo:],
            ls="none",
            marker="o",
            ms=ms,
            mfc=cast.background,
            mec=cast.neutral(0.85),
            mew=0.8 * cast.layout.lw,
            zorder=5,
        )


def _frame_in_detail(ax, layout, cast, case):
    """Upper zone (y 48 to 100): light into one pixel, then its readout.

    ``case`` is ``"count"`` (the worked count example) or ``"reference"``
    (the noiseless restricted reference case), which switches off the
    detector-noise terms and the random draw in the annotations.
    """
    slide = layout.is_slide
    reference = case == "reference"
    ms = layout.marker_pt * 0.55
    small = layout.small_pt
    box = (30.0, 44.0, 62.0, 83.0)
    wx0, wx1, wy0, wy1 = box
    cy = 71.0

    # Light sources above the well; rays run down into its opening.
    if reference:
        src = cast.neutral(0.8)
        ax.plot(
            [37.0],
            [96.0],
            marker="o",
            ms=0.8 * layout.marker_pt,
            mfc=cast.background,
            mec=src,
            mew=layout.lw,
            ls="none",
            zorder=5,
        )
        _label(
            ax,
            (39.5, 96.5),
            "synthetic source: photon spectral flux density\n"
            "of 1 Jy at 700 nm, constant over 1 nm",
            cast,
            ha="left",
            va="center",
            fontsize=small,
            linespacing=1.1,
        )
        ex.arrow(ax, (37.0, 93.5), (37.0, 71.0), "ray", cast, color=src)
    else:
        ex.mark(ax, "planet", (33.0, 96.0), cast, scale=0.8)
        _label(
            ax,
            (31.0, 96.0),
            "planet light",
            cast,
            ha="right",
            va="center",
            color=cast["planet"].color,
        )
        ax.add_patch(
            Rectangle((39.4, 94.2), 3.2, 3.6, zorder=4, **cast["local_zodi"].patch_kw())
        )
        _label(
            ax,
            (44.5, 96.0),
            "background light",
            cast,
            ha="left",
            va="center",
            color=cast["local_zodi"].color,
        )
        ex.arrow(ax, (33.0, 93.5), (35.0, 72.0), "ray", cast, source="planet")
        ex.arrow(
            ax, (41.0, 93.5), (39.5, 70.5), "ray", cast, color=cast["local_zodi"].color
        )

    # The well, its photoelectrons and its dark charge.
    _well(ax, wx0, wx1, wy0, wy1, cast)
    _electrons(ax, cast, box, n_photo=11, n_dark=0 if reference else 2, ms=ms)
    gx = wx0 - 2.5
    q_y, d_y = 74.5, 65.5
    ax.plot([gx], [q_y], marker="o", ms=ms, color=cast.neutral(0.85), ls="none")
    ax.plot(
        [gx],
        [d_y],
        marker="o",
        ms=ms,
        mfc=cast.background,
        mec=cast.neutral(0.85),
        mew=0.8 * layout.lw,
        ls="none",
    )
    if reference or slide:
        q_text = "photoelectrons, quantum\n" + r"efficiency (QE) $q$"
    else:
        q_text = "photoelectrons, quantum\n" + (
            rf"efficiency (QE) $q$ = {_fmt(COUNT['qe'])}"
        )
    if reference:
        d_text = "dark charge: 0 here"
    elif slide:
        d_text = r"dark charge, rate $d$"
    else:
        d_text = rf"dark charge, $d$ = {_fmt(COUNT['dark_e_per_pixel_s'])} e/s"
    _label(
        ax,
        (gx - 2.0, q_y),
        q_text,
        cast,
        ha="right",
        va="center",
        fontsize=small,
        linespacing=1.1,
    )
    _label(ax, (gx - 2.0, d_y), d_text, cast, ha="right", va="center", fontsize=small)
    _label(
        ax,
        (wx1 + 1.2, 80.5),
        "one of the 4\naperture pixels",
        cast,
        ha="left",
        va="center",
        fontsize=small,
        linespacing=1.1,
        color=cast["detector"].color,
    )

    # Readout chain: clock the charge out (CIC), amplify and read (read
    # noise), giving one frame value.
    chain = cast.neutral(0.4) if reference else None
    cic_x, amp_x = 53.5, 63.0
    _plain_arrow(ax, (wx1 + 0.8, cy), (amp_x, cy), cast, color=chain)
    if not reference:
        _cic_glyph(ax, (cic_x, cy), cast, layout.marker_pt * 0.9)
    if reference:
        cic_text = "clock out:\nno CIC here"
        read_text = "read: no read noise here"
    elif slide:
        cic_text = "clock out: + clock-\ninduced charge (CIC)"
        read_text = "read: + read noise"
    else:
        cic_text = (
            "clock out: + clock-\ninduced charge (CIC),\n"
            + rf"$c$ = {_fmt(COUNT['cic_e_per_pixel_frame'])} e/frame"
        )
        read_text = (
            rf"read: + read noise, $\sigma_r$ = "
            f"{_fmt(COUNT['read_noise_e_rms_per_pixel_read'])} e RMS"
        )
    _label(
        ax,
        (cic_x, cy - 2.0),
        cic_text,
        cast,
        ha="center",
        va="top",
        fontsize=small,
        linespacing=1.0,
    )
    _amplifier(ax, amp_x, cy, 6.5, 11.0, cast)
    if not reference:
        _gauss_icon(
            ax, (amp_x + 3.25, cy + 10.5), 6.0, 4.0, cast.neutral(0.85), layout.lw
        )
    _label(
        ax,
        (amp_x + 3.25, cy + 14.0),
        read_text,
        cast,
        ha="center",
        va="bottom",
        fontsize=small,
    )
    ex.arrow(ax, (amp_x + 7.0, cy), (77.0, cy), "data", cast)
    ax.add_patch(
        Rectangle(
            (77.5, cy - 5.5),
            15.5,
            11.0,
            facecolor=cast.neutral(0.1),
            edgecolor=cast["data"].color,
            lw=layout.lw,
            zorder=4,
        )
    )
    value_text = "expected\nvariance only" if reference else "one frame\nvalue"
    ax.text(
        85.25,
        cy,
        value_text,
        ha="center",
        va="center",
        color=cast.text,
        fontsize=small,
        zorder=6,
    )

    # The answer to "which terms scale with what". Each bracket carries the
    # fill of the timeline bar it corresponds to.
    if reference:
        _bracket(ax, wx0 - 1.0, wx1 + 1.0, 54.0, "one 1 s exposure", cast, fill="live")
        _bracket(ax, 46.5, 76.0, 54.0, "computed, not sampled", cast)
    else:
        _bracket(
            ax, wx0 - 1.0, wx1 + 1.0, 53.5, "grows with live time", cast, fill="live"
        )
        _bracket(ax, 46.5, 76.0, 53.5, "once per frame, at its read", cast, fill="read")


def _schedule(ax, layout, cast):
    """Lower zone (y 0 to 48): the frame schedule, frames 3 to 9 abbreviated."""
    p = COUNT
    budget = count_budget()
    slide = layout.is_slide
    small = layout.small_pt
    y0, y1 = 30.0, 38.0
    live_w, read_w = 15.0, 3.0
    starts = {1: 16.0, 2: 16.0 + live_w + read_w, p["n_frames"]: 62.0}
    ink = cast.neutral(0.75)
    reads = {}
    for k, x in starts.items():
        _live_bar(ax, x, y0, live_w, y1 - y0, cast)
        _read_bar(ax, x + live_w, y0, read_w, y1 - y0, cast)
        _label(
            ax,
            (x + 0.5 * live_w, y1 + 1.0),
            f"frame {k}",
            cast,
            ha="center",
            va="bottom",
        )
        reads[k] = x + live_w + 0.5 * read_w
    # The first read is named on its own bar.
    _label(
        ax,
        (reads[1], y1 + 1.0),
        "read" if slide else f"read\n{_fmt(p['read_time_s'])} s",
        cast,
        ha="center",
        va="bottom",
        fontsize=small,
        linespacing=1.0,
    )
    # Frame 2 is the frame drawn in detail above.
    x2 = starts[2]
    ax.add_patch(
        Rectangle(
            (x2 - 0.6, y0 - 1.2),
            live_w + read_w + 1.2,
            y1 - y0 + 2.4,
            facecolor="none",
            edgecolor=cast.text,
            lw=1.3 * layout.lw,
            zorder=3,
        )
    )
    _label(
        ax,
        (1.0, 95.0),
        "frame 2 in detail",
        cast,
        ha="left",
        va="center",
        fontweight="bold",
    )
    # The live-interval label sits on an opaque box over the hatching.
    ax.text(
        starts[1] + 0.5 * live_w,
        0.5 * (y0 + y1),
        rf"live $t_f$ = {_fmt(p['frame_time_s'])} s",
        ha="center",
        va="center",
        color=cast.text,
        fontsize=small,
        zorder=5,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": cast.background,
            "edgecolor": "none",
        },
    )
    # Frames 3 to 9 are omitted: a break on the time axis.
    gap0, gap1 = x2 + live_w + read_w, starts[p["n_frames"]]
    ex.note(
        ax,
        (0.5 * (gap0 + gap1), 0.5 * (y0 + y1)),
        "frames\n3 to 9",
        cast,
        ha="center",
        va="center",
    )
    axis_y = y0 - 3.0
    ax.plot([starts[1], gap0 + 3.0], [axis_y, axis_y], color=ink, lw=0.8 * layout.lw)
    ax.plot([gap1 - 3.0, 81.0], [axis_y, axis_y], color=ink, lw=0.8 * layout.lw)
    ex.scale_break(ax, (gap0 + 3.0, axis_y), cast)
    ex.scale_break(ax, (gap1 - 3.0, axis_y), cast)
    _plain_arrow(ax, (80.0, axis_y), (83.5, axis_y), cast, color=ink)
    _label(
        ax, (84.5, axis_y), "elapsed time", cast, ha="left", va="center", fontsize=small
    )

    # Each read: one CIC draw and one read-noise draw, giving one frame
    # value that flows into the recorded sum.
    glyph_y, rail_y = axis_y - 4.0, 11.0
    for xr in reads.values():
        _cic_glyph(ax, (xr - 1.3, glyph_y), cast, layout.marker_pt * 0.7)
        _gauss_icon(
            ax, (xr + 1.6, glyph_y), 2.8, 3.0, cast.neutral(0.85), 0.8 * layout.lw
        )
        ex.arrow(ax, (xr, glyph_y - 3.5), (xr, rail_y + 1.0), "data", cast)
    _label(
        ax,
        (reads[1] - 3.2, glyph_y + 1.5),
        "at each read: one CIC draw\nand one read-noise draw",
        cast,
        ha="right",
        va="top",
        fontsize=small,
        linespacing=1.1,
    )
    ax.plot(
        [reads[1], 83.0], [rail_y, rail_y], color=cast["data"].color, lw=1.2 * layout.lw
    )
    ex.arrow(ax, (82.5, rail_y), (85.5, rail_y), "data", cast)
    ax.add_patch(
        Rectangle(
            (86.0, rail_y - 6.0),
            13.5,
            12.0,
            facecolor=cast.neutral(0.1),
            edgecolor=cast["data"].color,
            lw=layout.lw,
            zorder=4,
        )
    )
    ax.text(
        92.75,
        rail_y,
        r"$E_p$ = sum of" + f"\n{p['n_frames']} frame values",
        ha="center",
        va="center",
        color=cast.text,
        fontsize=small,
        zorder=6,
    )

    # The answer: live time and elapsed time differ by the reads.
    answer = (
        rf"live time $t_{{\rm live}}$ = {p['n_frames']} x {_fmt(p['frame_time_s'])} s"
        f" = {_fmt(budget['live_s'])} s,   elapsed time"
        rf" {p['n_frames']} x ({_fmt(p['frame_time_s'])} s + {_fmt(p['read_time_s'])} s)"
        f" = {_fmt(budget['elapsed_s'])} s"
    )
    ax.text(
        0.5, 0.0, answer, ha="left", va="bottom", color=cast.text, fontweight="bold"
    )


# How the table shows each scaling category.
GROWS_DISPLAY = {
    "live time": "live time",
    "frames": "frames",
    "reads": "reads (variance only)",
}


def _budget_table(ax, layout, cast, rows, total, notes):
    """A mean and variance table drawn as text.

    ``rows`` are ``(term, grows_with, calculation, mean, variance)`` tuples
    of strings, ``total`` is ``(label, mean, variance)`` or None, and
    ``notes`` are the lines written under the table.
    """
    slide = layout.is_slide
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    cols = [0.0, 0.275, 0.5, 0.85, 0.99] if not slide else [0.02, 0.4, None, 0.74, 0.96]
    n = len(rows) + (2 if total else 1) + len(notes)
    dy = 1.0 / n
    y = 1.0 - 0.5 * dy
    headers = [
        "sum over the 4 aperture pixels",
        "grows with",
        "calculation",
        "mean (e)",
        r"variance (e$^2$)",
    ]
    for x, h, ha in zip(
        cols, headers, ["left", "left", "left", "right", "right"], strict=True
    ):
        if x is not None:
            ax.text(
                x,
                y,
                h,
                ha=ha,
                va="center",
                color=cast["annotation"].color,
                fontstyle="italic",
                fontsize=layout.small_pt,
            )
    for term, grows, calc, mean, var in rows:
        y -= dy
        ax.text(cols[0], y, term, ha="left", va="center", color=cast.text)
        ax.text(cols[1], y, grows, ha="left", va="center", color=cast.text)
        if cols[2] is not None:
            ax.text(
                cols[2],
                y,
                calc,
                ha="left",
                va="center",
                color=cast.neutral(0.7),
                fontsize=layout.small_pt,
            )
        ax.text(cols[3], y, mean, ha="right", va="center", color=cast.text)
        ax.text(cols[4], y, var, ha="right", va="center", color=cast.text)
    if total:
        y -= dy
        ax.plot(
            [0.0, 1.0],
            [y + 0.5 * dy, y + 0.5 * dy],
            color=cast.neutral(0.45),
            lw=0.6 * layout.lw,
        )
        label, mean, var = total
        for x, text, ha in (
            (cols[0], label, "left"),
            (cols[3], mean, "right"),
            (cols[4], var, "right"),
        ):
            ax.text(x, y, text, ha=ha, va="center", color=cast.text, fontweight="bold")
    for line in notes:
        y -= dy
        ex.note(ax, (cols[0], y), line, cast, ha="left", va="center")


def _count_table(ax, layout, cast):
    budget = count_budget()
    rows = [
        (n, GROWS_DISPLAY[g], c, _fmt(m), _fmt(v)) for n, g, c, m, v in budget["rows"]
    ]
    total = (
        r"total, $\Sigma_p E_p$ over 4 pixels",
        _fmt(budget["mean"]),
        _fmt(budget["variance"]),
    )
    planet = budget["rows"][0][3]
    notes = []
    if not layout.is_slide:
        notes = [
            "expected values and variances of the worked count example; no random draw",
            "SNR after subtracting the known background, dark and CIC means = "
            rf"{_fmt(planet)} / $\sqrt{{{_fmt(budget['variance'])}}}$ = {budget['snr']:.3f}",
        ]
    _budget_table(ax, layout, cast, rows, total, notes)


def _pin_layout(*axes):
    """Keep drawn labels out of constrained layout, so the axes keep their size."""
    for ax in axes:
        for artist in ax.get_children():
            artist.set_in_layout(False)


HEADLINE = (
    "Charge grows with live time; clock-induced charge and read noise "
    "arrive once per read"
)


def build_acquisition(layout, cast):
    """Still: frame 2 in detail over the frame schedule, then the budget."""
    slide = layout.is_slide
    fig, (ax, bx) = ex.figure(
        layout,
        doc_height_in=5.7,
        nrows=2,
        height_ratios=[3.7, 1.9] if not slide else [5.6, 2.4],
    )
    if slide:
        fig.suptitle(HEADLINE, fontsize=layout.font_pt + 2, fontweight="bold")
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    _frame_in_detail(ax, layout, cast, "count")
    _schedule(ax, layout, cast)
    ex.badge(ax, cast, "schematic, not to scale; reads drawn wider", loc="upper right")
    ex.note(
        ax,
        (1.0, 90.5),
        "independent Poisson charge,\nindependent Gaussian reads",
        cast,
        ha="left",
        va="top",
    )
    _count_table(bx, layout, cast)
    _pin_layout(ax, bx)
    return fig


def _reference_table(ax, layout, cast):
    """The two-column arithmetic of the noiseless reference case."""
    ref = reference_budget()
    p = REFERENCE

    def spaced(v):
        return f"{v:,.2f}".replace(",", " ")

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    rows = [
        (
            "photon spectral flux density of 1 Jy at 700 nm",
            rf"{spaced(ref['density'])} photon s$^{{-1}}$ m$^{{-2}}$ nm$^{{-1}}$",
        ),
        (
            rf"x {_fmt(p['area_m2'])} m$^2$ x {_fmt(p['bandwidth_nm'])} nm x "
            rf"{_fmt(p['exposure_s'])} s: photons on the 4 pixels",
            f"{spaced(ref['photons'])} photons",
        ),
        (
            rf"x QE $q$ = {_fmt(p['qe'])}: summed expected photoelectrons",
            f"{spaced(ref['variance'])} e",
        ),
        (
            "summed Poisson variance, equal to that mean",
            rf"{spaced(ref['variance'])} e$^2$",
        ),
        ("dark charge, CIC, read noise", "0"),
    ]
    n = len(rows) + 1
    dy = 1.0 / n
    y = 1.0 - 0.5 * dy
    for term, value in rows:
        ax.text(0.0, y, term, ha="left", va="center", color=cast.text)
        ax.text(0.99, y, value, ha="right", va="center", color=cast.text)
        y -= dy
    ex.note(
        ax,
        (0.0, y),
        "the variance is computed from the model, not sampled: no stochastic readout",
        cast,
        ha="left",
        va="center",
    )


def build_reference(layout, cast):
    """Still: the same pixel, annotated for the noiseless reference case."""
    fig, (ax, bx) = ex.figure(
        layout,
        doc_height_in=4.0,
        nrows=2,
        height_ratios=[2.1, 1.6] if not layout.is_slide else [5.0, 3.3],
    )
    ax.set(xlim=(0, 100), ylim=(47, 100))
    ax.axis("off")
    _frame_in_detail(ax, layout, cast, "reference")
    ex.badge(ax, cast, "schematic, not to scale", loc="upper right")
    ex.note(
        ax,
        (99.5, 62.0),
        "restricted case:\nno detector noise terms",
        cast,
        ha="right",
        va="top",
    )
    _reference_table(bx, layout, cast)
    _pin_layout(ax, bx)
    return fig


# The animation: the clock runs through integrations and reads


def _states(layout):
    """Elapsed times shown, as self-contained frame states."""
    p = COUNT
    t_f, t_r = p["frame_time_s"], p["read_time_s"]
    taus = (5.0, 10.0) if not layout.is_slide else (2.0, 4.0, 6.0, 8.0, 10.0)
    frames = []
    for k in range(p["n_frames"]):
        s = k * (t_f + t_r)
        frames += [{"t": s + tau} for tau in taus]
        frames.append({"t": s + t_f + t_r})
    return frames[: layout.n_frames(len(frames))]


def _clock(t, p=COUNT):
    """Live time and completed reads at elapsed time ``t``."""
    t_f, t_r = p["frame_time_s"], p["read_time_s"]
    starts = np.arange(p["n_frames"]) * (t_f + t_r)
    live = float(np.clip(t - starts, 0.0, t_f).sum())
    reads = int(np.sum(starts + t_f + t_r <= t + 1e-9))
    return live, reads


def _charge(sim, t, p=COUNT):
    """Realized and expected charge in the wells at elapsed time ``t``."""
    t_f = p["frame_time_s"]
    _, reads = _clock(t, p)
    if reads >= p["n_frames"]:
        return 0.0, 0.0
    tau = t - sim["starts"][reads]
    if tau > t_f:
        return 0.0, 0.0
    return float(np.sum(sim["arrivals"][reads] <= tau)), sim["rate"] * tau


def frame_variance(p=COUNT):
    """Variance of one frame value summed over the aperture, electrons^2."""
    live = p["frame_time_s"]
    n_pix = p["n_pixels"]
    return (
        (p["planet_photon_rate_per_s"] + p["background_photon_rate_per_s"])
        * p["qe"]
        * live
        + n_pix * p["dark_e_per_pixel_s"] * live
        + n_pix * p["cic_e_per_pixel_frame"]
        + n_pix * p["read_noise_e_rms_per_pixel_read"] ** 2
    )


def build_count_animation(layout, cast):
    """Animation: the aperture's charge and its recorded sum as time runs."""
    p = COUNT
    sim = simulate()
    t_f, t_r = p["frame_time_s"], p["read_time_s"]
    ends = sim["starts"] + t_f + t_r
    cum = np.concatenate([[0.0], np.cumsum(sim["values"])])
    k_reads = np.arange(p["n_frames"] + 1)
    expected_cum = k_reads * sim["frame_mean"]
    expected_sd = np.sqrt(k_reads * frame_variance())
    fig, axes = ex.figure(
        layout,
        doc_height_in=5.6,
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 6.0],
        height_ratios=[1.0, 1.0, 0.62],
    )
    (ag, ac), (asum, ar), (blank, ad) = axes
    blank.axis("off")
    t_max = float(ends[-1])
    # Pinned scales: the charge panel spans the largest realized or
    # expected per-frame charge with room for its label; the record panel
    # spans the larger final sum. Both are fixed for every frame.
    q_top = 50.0 * math.ceil(
        1.3 * max(max(len(a) for a in sim["arrivals"]), sim["rate"] * t_f) / 50.0
    )
    r_top = 500.0 * math.ceil(1.12 * max(cum[-1], expected_cum[-1]) / 500.0)
    # The residual strip spans three standard deviations of the final sum.
    d_top = 50.0 * math.ceil(3.0 * expected_sd[-1] / 50.0)
    ad.set_ylim(-d_top, d_top)
    for a in (ag, ac):
        a.set_ylim(0.0, q_top)
    for a in (asum, ar):
        a.set_ylim(0.0, r_top)
    for a in (ac, ar, ad):
        a.set_xlim(-1.0, t_max + 1.0)
        for e in ends:
            a.axvline(e, color=cast.neutral(0.3), lw=0.6 * layout.lw, zorder=0)
    for a in (ac, ar):
        a.tick_params(labelleft=False, labelbottom=False)
    ad.set_xlabel("elapsed time (s)")
    # The strip's scale sits on its right, so all three time panels keep
    # one left edge; its name sits in the empty cell to its left.
    ad.yaxis.tick_right()
    blank.text(
        1.0,
        0.5,
        "recorded\nminus\nexpected (e)",
        transform=blank.transAxes,
        ha="right",
        va="center",
        color=cast.text,
        fontsize=layout.font_pt,
    )
    ag.set_ylabel("charge in the\n4 pixels (e)")
    asum.set_ylabel("recorded sum (e)")
    model = hwostyle.roles.model
    sim_ink = cast.neutral(0.5)
    lab = layout.font_pt

    # The realization's charge history, as a step function.
    hist_t, hist_q = [], []
    for s, arr in zip(sim["starts"], sim["arrivals"], strict=True):
        hist_t += [s, *(s + arr), s + t_f, s + t_f + t_r]
        hist_q += [0.0, *np.arange(1, len(arr) + 1), float(len(arr)), 0.0]
    hist_t, hist_q = np.asarray(hist_t), np.asarray(hist_q, dtype=float)
    (real_q,) = ac.plot(
        [], [], color=sim_ink, lw=1.6 * layout.lw, drawstyle="steps-post", zorder=3
    )
    (real_r,) = ar.plot(
        [], [], color=sim_ink, lw=1.6 * layout.lw, drawstyle="steps-post", zorder=3
    )
    (real_m,) = ar.plot(
        [],
        [],
        ls="none",
        marker="s",
        ms=0.8 * layout.marker_pt,
        color=sim_ink,
        zorder=3,
    )

    # Expected curves are drawn in full and on top: the prediction is known
    # before observing. The record's band is one standard deviation of the
    # recorded sum after each read.
    saw_t, saw_q = [], []
    for s in sim["starts"]:
        saw_t += [s, s + t_f, s + t_f + t_r]
        saw_q += [0.0, sim["rate"] * t_f, 0.0]
    ac.plot(saw_t, saw_q, color=model, lw=layout.lw, zorder=5)
    step_t = np.concatenate([[0.0], ends])
    ad.fill_between(
        step_t,
        -expected_sd,
        expected_sd,
        step="post",
        color=model,
        alpha=0.22,
        lw=0,
        zorder=2,
    )
    ad.axhline(0.0, color=model, lw=layout.lw, zorder=5)
    (real_d,) = ad.plot(
        [], [], color=sim_ink, lw=1.6 * layout.lw, drawstyle="steps-post", zorder=3
    )
    (real_dm,) = ad.plot(
        [],
        [],
        ls="none",
        marker="s",
        ms=0.8 * layout.marker_pt,
        color=sim_ink,
        zorder=4,
    )
    ex.halo(
        ad.annotate(
            r"expected $\pm 1\sigma$, growing as $\sqrt{\rm reads}$",
            (t_max + 1.0, d_top),
            xytext=(-3, -3),
            ha="right",
            va="top",
            color=model,
            **{"fontsize": lab, "textcoords": "offset points", "zorder": 6},
        ),
        cast,
    )
    ar.plot(
        step_t,
        expected_cum,
        color=model,
        lw=layout.lw,
        drawstyle="steps-post",
        zorder=5,
    )

    lab_kw = {"fontsize": lab, "textcoords": "offset points", "zorder": 6}
    ex.halo(
        ac.annotate(
            f"expected, known before observing: {_fmt(sim['rate'])} e/s during live time",
            (0.5 * t_max, q_top),
            xytext=(0, -3),
            ha="center",
            va="top",
            color=model,
            **lab_kw,
        ),
        cast,
    )
    ex.halo(
        ar.annotate(
            f"expected, known before\nobserving: {_fmt(sim['frame_mean'])} e per read",
            (ends[5], expected_cum[5]),
            xytext=(8, -12),
            ha="left",
            va="top",
            color=model,
            **lab_kw,
        ),
        cast,
    )
    sim_q_lab = ex.halo(
        ac.annotate("simulated", (0, 0), xytext=(0, 0), color=sim_ink, **lab_kw),
        cast,
    )
    sim_r_lab = ex.halo(
        ar.annotate("simulated", (0, 0), xytext=(0, 0), color=sim_ink, **lab_kw),
        cast,
    )
    ex.halo(
        ar.annotate(
            "reads",
            (ends[0], 0.55 * r_top),
            xytext=(3, 0),
            ha="left",
            va="center",
            color=cast.neutral(0.6),
            fontstyle="italic",
            **lab_kw,
        ),
        cast,
    )

    # Physical anchors: the wells filling, and the record growing by one
    # frame value per read, on the same vertical scales as the curves.
    for a in (ag, asum):
        a.set_xlim(0.0, 1.0)
        a.set_xticks([])
    ag.add_patch(
        Rectangle(
            (0.2, 0.0),
            0.6,
            q_top * 0.98,
            facecolor=cast.neutral(0.06),
            edgecolor="none",
            zorder=1,
        )
    )
    ag.plot(
        [0.2, 0.2, 0.8, 0.8],
        [q_top * 0.98, 0.0, 0.0, q_top * 0.98],
        color=cast["detector"].color,
        lw=1.6 * layout.lw,
        zorder=3,
    )
    fill = ag.add_patch(
        Rectangle(
            (0.2, 0.0),
            0.6,
            0.0,
            facecolor=cast.neutral(0.55),
            edgecolor="none",
            zorder=2,
        )
    )
    (exp_fill,) = ag.plot([0.1, 0.9], [0.0, 0.0], color=model, lw=layout.lw, zorder=4)
    ag.set_xlabel("4 pixels", fontsize=layout.small_pt)
    blocks = []
    for k, v in enumerate(sim["values"]):
        blocks.append(
            asum.add_patch(
                Rectangle(
                    (0.2, cum[k]),
                    0.6,
                    v,
                    facecolor=cast.neutral(0.55 if k % 2 == 0 else 0.75),
                    edgecolor=cast.background,
                    lw=0.5 * layout.lw,
                    zorder=2,
                    visible=False,
                )
            )
        )
    (exp_sum,) = asum.plot([0.1, 0.9], [0.0, 0.0], color=model, lw=layout.lw, zorder=4)
    asum.set_xlabel("record", fontsize=layout.small_pt)

    top_title = ac.set_title("", loc="left", fontsize=layout.font_pt)
    bottom_title = ar.set_title("", loc="left", fontsize=layout.font_pt)
    ex.badge(
        ad,
        cast,
        "one simulated realization, selected as representative",
        loc="lower left",
    )

    def draw(fig, frame):
        t = frame["t"]
        live, reads = _clock(t)
        q_now, q_exp = _charge(sim, t)
        keep = hist_t <= t
        real_q.set_data(np.append(hist_t[keep], t), np.append(hist_q[keep], q_now))
        rec_t = np.concatenate([[0.0], ends[:reads]])
        real_r.set_data(np.append(rec_t, t), np.append(cum[: reads + 1], cum[reads]))
        real_m.set_data(ends[:reads], cum[1 : reads + 1])
        resid = cum[: reads + 1] - expected_cum[: reads + 1]
        real_d.set_data(np.append(rec_t, t), np.append(resid, resid[-1]))
        real_dm.set_data(ends[:reads], resid[1:])
        # The realization's labels sit up and to the left of its current
        # end, where its own past trace leaves room, except in the first
        # frame, where that side is the axis edge.
        early = t < 1.5 * t_f
        for label, xy, below in (
            (sim_q_lab, (t, q_now), True),
            (sim_r_lab, (t, cum[reads]), False),
        ):
            label.xy = xy
            if early:
                label.set_position((6, -6) if below else (6, 30))
                label.set_ha("left")
                label.set_va("top" if below else "bottom")
            else:
                label.set_position((-6, 4 if below else 8))
                label.set_ha("right")
                label.set_va("bottom")
        fill.set_height(q_now)
        exp_fill.set_ydata([q_exp, q_exp])
        for k, block in enumerate(blocks):
            block.set_visible(k < reads)
        exp_sum.set_ydata([expected_cum[reads]] * 2)
        top_title.set_text(
            f"elapsed {t:6.2f} s = live {live:6.2f} s + {reads:2d} {'read' if reads == 1 else 'reads'}"
            f" x {_fmt(t_r)} s"
        )
        bottom_title.set_text(
            f"recorded sum {cum[reads]:7.1f} e    expected {expected_cum[reads]:7.1f}"
            rf" $\pm$ {expected_sd[reads]:4.1f} e"
        )

    return ex.AnimationScene(fig=fig, draw=draw, frames=_states(layout))


CAPTION_SCHEDULE = (
    "One pixel's acquisition under the simple detector model of "
    "{ref}`radiometry-detector-counts`: independent Poisson photoelectrons, dark "
    "charge and clock-induced charge (CIC), and independent Gaussian read noise, "
    "with one read per frame, unit gain, and no saturation, gain fluctuation, "
    "correlated reads or charge-transfer effects. Photoelectrons and dark charge "
    "accumulate during the live intervals (hatched), so their means and "
    "variances scale with the live time $t_{\\rm live}$; CIC and read noise enter "
    "once per frame at its read (solid), so theirs scale with the frame count "
    "$n_f$, and read noise adds variance only. The schedule and the budget are "
    "the {ref}`worked count example <radiometry-count-example>`, a four-pixel "
    "aperture observed in ten 10 s frames, each followed by a 0.05 s "
    "nonoverlapping read, which occupy 100.5 s of elapsed time for 100 s of "
    "live time. The drawn pixel is one of the four; $E_p$ is one pixel's "
    "accumulated output, and the table sums it over the four pixels, with the "
    "source rates stated for the whole aperture and the detector terms per "
    "pixel. Frames 3 to 9 are omitted and the reads are drawn wider than to "
    "scale. The table lists expected values and numerical variances, not a "
    "random draw; the SNR is rounded to three decimals (13.9147 in the "
    "chapter). The terms match the analog-mode noise model of "
    "{ref}`Nemati et al. (2023, Sec. 4.1, eq. 32; Sec. 4.5) <source-nemati2023>` "
    "with unit gain and no excess-noise factor. The expectation and variance are "
    "identities under these assumptions; the read cadence is the example's "
    "stated schedule, and the {ref}`acquisition experiment decision "
    "<decision-acquisition-experiment>` that will fix read cadence is pending. "
    "The drawing does not describe nondestructive sampling or a particular "
    "detector architecture."
)
ALT_SCHEDULE = (
    "Schematic in three bands. Top: planet light and background light enter one "
    "of the four aperture pixels, drawn as a charge well holding filled dots for "
    "photoelectrons and open dots for dark charge. An arrow clocks the charge out "
    "past a diamond marking clock-induced charge to a read amplifier marked with "
    "a small Gaussian curve for read noise, giving one frame value. Under the "
    "well a hatched bar reads grows with live time; under the clocking and read "
    "a solid bar reads once per frame, at its read. Middle: a time axis with "
    "hatched 10 s live intervals and solid read bars for frames 1, 2 and 10, "
    "frames 3 to 9 omitted behind a break; the first read bar is labeled read, "
    "0.05 s, and frame 2 is outlined as the frame shown in detail. Under each "
    "read a diamond and a Gaussian curve mark one CIC draw and one read-noise "
    "draw, and a data arrow carries each frame value to a box reading E_p equals "
    "the sum of 10 frame values. A bold line states live time 10 x 10 s = 100 s "
    "and elapsed time 10 x (10 s + 0.05 s) = 100.5 s. Bottom: a table of means "
    "and variances summed over the four aperture pixels: planet photoelectrons "
    "1000 and 1000, background photoelectrons 4000 and 4000, dark charge 4 and "
    "4, all growing with live time; clock-induced charge 0.8 and 0.8, growing "
    "with frames; read noise 0 and 160, growing with reads as variance only; "
    "total 5004.8 electrons and 5164.8 electrons squared, and an SNR of 13.915 "
    "after subtracting the known background, dark and CIC means."
)
CAPTION_REFERENCE = (
    "The same pixel annotated for the restricted "
    "{ref}`photon-to-electron reference case <photon-electron-reference-experiment>`: "
    "a synthetic source with the photon spectral flux density of 1 Jy at 700 nm, "
    "held constant over 1 nm, collected on 1 m$^2$ for one 1 s exposure and "
    "spread over four pixels with constant quantum efficiency. Dark current, "
    "clock-induced charge and read noise are zero, and the case compares a "
    "computed expected variance with the library's, not a sampled readout, so "
    "the clocking and read operations add nothing. For independent Poisson "
    "photoelectrons the summed variance equals the summed mean "
    "({ref}`radiometry-detector-counts`); the photon density follows from the "
    "exact SI Planck constant ({ref}`BIPM 2019, Sec. 2.2 and Table 1 "
    "<source-bipm2019>`). The picture is a schematic of that one restricted "
    "experiment; it is not evidence of a stochastic readout, nonzero detector "
    "noise or the band integration of a real spectrum."
)
ALT_REFERENCE = (
    "Schematic. A synthetic source labeled with the photon spectral flux density "
    "of 1 Jy at 700 nm, constant over 1 nm, sends a ray into one of four pixels, "
    "drawn as a charge well with filled dots for photoelectrons; labels say the "
    "dark charge is 0 here, the clocking adds no clock-induced charge and the "
    "read adds no read noise, and the chain ends in a box reading expected "
    "variance only. Brackets read one 1 s exposure and computed, not sampled. A "
    "table below gives the photon spectral flux density 21 559.86 photon per "
    "second per square meter per nanometer, 21 559.86 photons on the four "
    "pixels, 10 779.93 expected photoelectrons for a quantum efficiency of 0.5, "
    "a summed Poisson variance of 10 779.93 electrons squared, and zero for dark "
    "charge, clock-induced charge and read noise."
)
CAPTION_ANIMATION = (
    "One simulated realization of the "
    "{ref}`worked count example <radiometry-count-example>`, summed over its "
    "four-pixel aperture, as elapsed time runs through ten 10 s live intervals "
    "and their 0.05 s reads. The realization was selected as representative: of "
    "random seeds 0 to 399, seed 16 was chosen because its total ends 0.5 "
    "standard deviations below the expected total and no frame value lies more "
    "than 1.1 standard deviations from its mean. Top: the charge held in the "
    "four pixels rises during each live interval at the expected 50.04 e/s "
    "(photoelectrons plus dark charge) and empties at each read. Bottom: the "
    "recorded sum grows by one frame value per read; a frame value is the "
    "frame's charge plus one Poisson draw of clock-induced charge and one "
    "Gaussian read-noise draw. Pink curves are expectations under the model of "
    "{ref}`radiometry-detector-counts`, known before observing, and the pink "
    "band is one standard deviation of the recorded sum, which grows as the "
    "square root of the number of reads; the gray trace and squares are the "
    "realization. The top title keeps the identity elapsed time = live time + "
    "reads x 0.05 s, so ten frames occupy 100.5 s of elapsed time for 100 s of "
    "live time. Photoelectrons and dark charge are simulated as one Poisson "
    "arrival process, and every draw is independent; the animation illustrates "
    "the stated model and is not a detector simulation."
)
ALT_ANIMATION = (
    "Animation on fixed axes with four panels. Upper left, a gauge for the four "
    "pixels fills during each live interval and empties at each read. Upper "
    "right, charge in the four pixels against elapsed time from 0 to 100.5 s: a "
    "pink sawtooth of expected charge rising to about 500 electrons in every 10 s "
    "live interval and dropping to zero at each read, over a gray simulated "
    "trace drawn up to the current time. Lower left, a stack of frame-value "
    "blocks grows by one block per read. Lower right, the recorded sum against "
    "elapsed time: a pink expected staircase rising 500.48 electrons per read "
    "inside a pink one-standard-deviation band, and a gray simulated staircase "
    "with square markers at the reads, labeled as one simulated realization "
    "selected as representative. The top title states elapsed time as live time "
    "plus the number of reads times 0.05 s, ending at 100.5 s = 100 s + 10 reads "
    "x 0.05 s; the bottom title gives the recorded sum, 4969.0 electrons at the "
    "end, and the expected sum, 5004.8 plus or minus 71.9 electrons."
)


FIGURES = [
    ex.FigureSpec(
        slug="d07-acquisition-schedule",
        build=build_acquisition,
        caption=CAPTION_SCHEDULE,
        alt=ALT_SCHEDULE,
        status="schematic, not to scale; worked example values",
        params={"count": COUNT},
    ),
    ex.FigureSpec(
        slug="d07-reference-restriction",
        build=build_reference,
        caption=CAPTION_REFERENCE,
        alt=ALT_REFERENCE,
        status="schematic, not to scale; restricted reference case",
        params={"reference": REFERENCE},
    ),
]
ANIMATIONS = [
    ex.AnimationSpec(
        slug="d07-acquisition-count",
        build=build_count_animation,
        ground="narration",
        caption=CAPTION_ANIMATION,
        alt=ALT_ANIMATION,
        status=f"simulated; representative realization (seed {SEED}, selected)",
        params={"count": COUNT, "seed": SEED, "seed_scan": [0, 399]},
        fps=5,
        preview_frames=(0, 4, 5, 29),
    ),
]
