"""Build the dust-model chapter's teaching figures with eyepiece and hwostyle.

Run from anywhere: python tools/build_dust_figures.py
(needs NumPy, Matplotlib, hwostyle, hwoutils and eyepiece >= 0.4.0).

Every panel is computed from the analytic fixtures in
tools/dust_reference_cases.py. No scene, zodi or instrument model is
executed, nothing is downloaded, and no figure is evidence about a library's
current behavior.
"""

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from hwostyle.colors import adjust_lightness
from hwoutils.constants import arcsec2rad, c, h, nm2m
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "conventions" / "figures"
sys.path.insert(0, str(ROOT / "tools"))
import dust_reference_cases as drc  # noqa: E402

SCRIPT = "tools/build_dust_figures.py"
FORMATS = ("png", "svg", "pdf")
RADIANCE_UNIT = r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$"

# Fixture parameters, recorded in the manifest.
SPHERE = {"radius_au": 2.0, "emissivity": 3.0}
SPHERE_RAYS = [
    ("inside, +x", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), "x"),
    ("inside, -x", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "x"),
    ("outside, -x", (5.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "x"),
    ("outside, +x", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0), "x"),
    ("grazing, +y", (2.0, -3.0, 0.0), (0.0, 1.0, 0.0), "y"),
]
CLUMP = {"peak_sr": 7.0, "sigma_arcsec": 0.25, "field_arcsec": 2.0}
GRIDS = (24, 48, 96)
BAND = {"b0": 1.0e-6, "lam0_nm": 550.0, "lo_nm": 500.0, "hi_nm": 600.0}
SLOPES = (0.0, -4.0, 2.0)
RING = {"radius_au": 3.0, "width_au": 0.6, "field_au": 10.0, "n_pix": 101}
AMPLITUDES = [(2.0, 0.15), (1.0, 0.30)]
SLAB = {
    "radius_au": 4.0,
    "thickness_au": 0.1,
    "emissivity": 5.0,
    "inclination_deg": 120.0,
    "field_au": 10.0,
    "n_pix": 121,
}


def neutral(level):
    """Blend from the background (0) toward the text color (1)."""
    bg = np.array(to_rgb(plt.rcParams["axes.facecolor"]))
    fg = np.array(to_rgb(plt.rcParams["text.color"]))
    return tuple(bg + level * (fg - bg))


def cast():
    """The one entity-to-style cast every dust figure uses."""
    roles = hwostyle.roles
    light = hwostyle.current_mode() == "light"
    return {
        "dust": {"color": roles.disk, "ls": "-", "marker": "o"},
        "dust_second": {
            "color": adjust_lightness(roles.disk, 1.45 if light else 0.7),
            "ls": "--",
            "marker": "s",
        },
        "dust_third": {
            "color": adjust_lightness(roles.disk, 0.65 if light else 1.35),
            "ls": "-.",
            "marker": "^",
        },
        "star": {"color": roles.star, "marker": "*"},
        "wrong": {"color": roles.model, "ls": ":", "marker": "x"},
        "scenery": {"color": neutral(0.45)},
        "furniture": {"color": neutral(0.2)},
    }


def require_physical(image, name):
    """Refuse to log-display radiance that is negative or not finite.

    eyepiece.imshow_log floors negative pixels, so a failing model would
    otherwise look like faint valid dust.
    """
    image = np.asarray(image)
    if not np.all(np.isfinite(image)):
        msg = f"{name}: non-finite radiance"
        raise ValueError(msg)
    if np.any(image < 0.0):
        msg = f"{name}: negative radiance; use a signed display"
        raise ValueError(msg)
    return image


def quiet(ax):
    ax.spines[["top", "right"]].set_visible(False)


def panel_label(ax, text):
    ax.set_title(text, loc="left")


# Figure 1: rays, scattering angle and the cumulative integral


def geometry(fig_cast):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.8), layout="constrained", width_ratios=[1.25, 1, 1]
    )
    tracks, angle, cumul = axes
    radius, emis = SPHERE["radius_au"], SPHERE["emissivity"]
    dust = fig_cast["dust"]
    scenery = fig_cast["scenery"]["color"]

    # (a) each fixture ray on its own track
    panel_label(tracks, "(a) Only the half-ray $s \\geq 0$ is integrated")
    for row, (label, obs, direc, axis) in enumerate(SPHERE_RAYS):
        y = -row
        k = 0 if axis == "x" else 1
        # The sphere's footprint along this ray's line.
        other = obs[1 - k] if axis == "x" else obs[0]
        half = math.sqrt(max(radius**2 - other**2, 0.0))
        start = obs[k]
        if half > 0:
            tracks.fill_between(
                [-half, half], y - 0.28, y + 0.28, color=dust["color"], alpha=0.15, lw=0
            )
        else:
            tracks.plot([0.0], [y], "|", color=dust["color"], ms=14)
        tracks.annotate(
            "",
            xy=(start + 2.0 * direc[k], y + 0.18),
            xytext=(start, y + 0.18),
            arrowprops={"arrowstyle": "->", "color": scenery, "lw": 1.2},
        )
        length = drc.sphere_path_length(obs, direc, radius)
        if length > 0:
            # Line coordinate = start + s * d; clip the footprint to s >= 0.
            d = direc[k]
            s_a, s_b = sorted(((-half - start) / d, (half - start) / d))
            s0, s1 = max(s_a, 0.0), max(s_b, 0.0)
            tracks.plot(
                [start + s0 * d, start + s1 * d],
                [y, y],
                color=dust["color"],
                lw=4,
                solid_capstyle="butt",
            )
        tracks.plot(start, y, marker="D", color=scenery, ms=6, ls="none")
        value = drc.sphere_ray_radiance(obs, direc, radius, emis)
        tracks.text(7.8, y, f"{label}:  L = {length:g} AU, I = {value:g}", va="center")
    tracks.set(xlim=(-3.2, 14.5), ylim=(-4.6, 0.6), yticks=[])
    tracks.spines["bottom"].set_bounds(-3, 7)
    tracks.set_xticks([-2, 0, 2, 4, 6])
    tracks.set_xlabel("position along the ray's line [AU]", x=0.35)
    tracks.spines[["top", "right", "left"]].set_visible(False)
    tracks.text(
        -3.1,
        0.45,
        "shaded: sphere, R = 2 AU; diamond: observer",
        color=scenery,
        fontsize="small",
    )

    # (b) scattering angle at one grain
    panel_label(angle, "(b) Scattering angle $\\Theta$ and $\\alpha = \\pi - \\Theta$")
    theta_deg = 60.0
    grain = 1.5 * np.array(
        [math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))]
    )
    direction = (-1.0, 0.0, 0.0)  # observer far to the right, looking -x
    theta = drc.scattering_angle((*grain, 0.0), direction)
    alpha = drc.illumination_angle((*grain, 0.0), direction)
    angle.plot(0, 0, ls="none", ms=18, **fig_cast["star"])
    angle.text(0.0, -0.35, "star", ha="center", color=scenery)
    angle.annotate(
        "",
        xy=grain,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": dust["color"], "lw": 2},
    )
    fwd = grain + 1.1 * grain / np.linalg.norm(grain)
    angle.plot(*zip(grain, fwd, strict=True), color=scenery, ls="--", lw=1.2)
    angle.annotate(
        "",
        xy=grain + np.array([1.9, 0.0]),
        xytext=grain,
        arrowprops={"arrowstyle": "->", "color": dust["color"], "lw": 2},
    )
    angle.text(
        *(grain + np.array([1.95, 0.0])), "  toward observer, $-\\hat n$", va="center"
    )
    angle.plot(*grain, marker="o", ms=7, color=dust["color"], ls="none")
    angle.text(*(grain + np.array([-0.2, 0.1])), "grain", ha="right", color=scenery)
    angle.add_patch(
        Arc(
            grain,
            1.3,
            1.3,
            theta1=0,
            theta2=theta_deg,
            color=fig_cast["wrong"]["color"],
        )
    )
    angle.text(
        *(grain + np.array([0.75, 0.3])),
        f"$\\Theta$ = {math.degrees(theta):.0f} deg",
        ha="left",
    )
    angle.add_patch(
        Arc(grain, 0.7, 0.7, theta1=180 + theta_deg, theta2=360, color=scenery)
    )
    angle.text(
        *(grain + np.array([0.5, -0.55])),
        f"$\\alpha$ = {math.degrees(alpha):.0f} deg",
        ha="left",
    )
    angle.text(
        0.9,
        -0.55,
        "incident light propagates star to grain",
        color=scenery,
        fontsize="small",
    )
    angle.set(xlim=(-0.6, 4.6), ylim=(-0.9, 2.9), aspect="equal")
    angle.axis("off")

    # (c) cumulative integral
    panel_label(cumul, "(c) Radiance accumulates only inside the cloud")
    s = np.linspace(0.0, 8.0, 401)
    for key, obs, text in (
        ("dust", (1.0, 0.0, 0.0), "inside observer"),
        ("dust_second", (5.0, 0.0, 0.0), "outside observer"),
    ):
        style = fig_cast[key]
        curve = drc.cumulative_sphere_radiance(obs, (-1.0, 0.0, 0.0), radius, emis, s)
        cumul.plot(s, curve, color=style["color"], ls=style["ls"])
        total = curve[-1]
        cumul.plot(s[-1], total, marker=style["marker"], color=style["color"])
        cumul.text(s[-1] - 0.15, total + 0.4, f"{text}: {total:g}", ha="right")
    cumul.set(xlabel="distance along the ray, s [AU]", ylim=(0, 13.5))
    cumul.set_ylabel("cumulative radiance\n[" + RADIANCE_UNIT + "]")
    quiet(cumul)
    return fig


# Figure 2: radiance versus pixel flux


def sampling(fig_cast):
    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4.2), layout="constrained", width_ratios=[1, 1, 1, 1.25]
    )
    field = CLUMP["field_arcsec"]
    images = [
        require_physical(
            drc.gaussian_patch_pixel_flux(
                CLUMP["peak_sr"], CLUMP["sigma_arcsec"], field, n
            ),
            f"clump {n}",
        )
        for n in GRIDS
    ]
    half = 0.5 * field
    res = ep.compare_row(
        images,
        titles=[f"({t}) {n} x {n} pixels" for t, n in zip("abc", GRIDS, strict=True)],
        axes=axes[:3],
        norm="log",
        extent=(-half, half, -half, half),
        cbar_label="pixel-integrated flux\n[photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ per pixel]",
    )
    for ax, img in zip(res.axes, images, strict=True):
        ep.label_arcsec(ax)
        ax.text(
            0.03,
            0.04,
            f"peak pixel {img.max():.2e}\nsum {img.sum():.4e}",
            transform=ax.transAxes,
            fontsize="small",
            bbox={"fc": plt.rcParams["axes.facecolor"], "ec": "none", "alpha": 0.8},
        )
    for ax in res.axes[1:]:
        ax.set_ylabel("")

    ladder = axes[3]
    panel_label(ladder, "(d) The integral holds; unweighted sums do not")
    grids = np.array(GRIDS)
    totals = np.array([img.sum() for img in images])
    d_omega = (field / grids * arcsec2rad) ** 2
    centers = [np.linspace(-half, half, n + 1) for n in GRIDS]
    sample_sums = []
    for edges in centers:
        mid = 0.5 * (edges[1:] + edges[:-1])
        xx, yy = np.meshgrid(mid, mid)
        sampled = CLUMP["peak_sr"] * np.exp(
            -(xx**2 + yy**2) / (2 * CLUMP["sigma_arcsec"] ** 2)
        )
        sample_sums.append(sampled.sum())
    sample_sums = np.array(sample_sums)
    ref = totals[0]
    ok, bad = fig_cast["dust"], fig_cast["wrong"]
    ladder.plot(
        grids, totals / ref, color=ok["color"], ls=ok["ls"], marker=ok["marker"]
    )
    ladder.plot(
        grids,
        sample_sums * d_omega[0] / ref,
        color=bad["color"],
        ls=bad["ls"],
        marker=bad["marker"],
    )
    ladder.text(96, 1.25, "sum of pixel fluxes", ha="right", color=ok["color"])
    ladder.text(
        90,
        14.0,
        "sum of radiance samples,\nno $d\\Omega$ weight",
        ha="right",
        va="top",
        color=bad["color"],
    )
    ladder.set(xscale="log", yscale="log", xticks=list(GRIDS), xticklabels=GRIDS)
    ladder.minorticks_off()
    ladder.set_xlabel("pixels per side (fixed 2 arcsec field)")
    ladder.set_ylabel("relative to the 24-pixel sum")
    quiet(ladder)
    return fig, {"totals": totals.tolist(), "sample_sums": sample_sums.tolist()}


# Figure 3: where wavelength, band and solid angle enter


def radiometry(fig_cast):
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.6), layout="constrained", width_ratios=[1, 1, 1.15]
    )
    spec, err, chain = axes
    b0, lam0 = BAND["b0"], BAND["lam0_nm"]
    lo, hi = BAND["lo_nm"], BAND["hi_nm"]
    lam = np.linspace(400.0, 800.0, 401)
    styles = [fig_cast["dust"], fig_cast["dust_second"], fig_cast["dust_third"]]

    panel_label(spec, "(a) Convert to photons at each wavelength")
    for k, style in zip(SLOPES, styles, strict=True):
        photon = b0 * (lam / lam0) ** k * lam * nm2m / (h * c) / 1e12
        spec.plot(lam, photon, color=style["color"], ls=style["ls"])
        spec.text(lam[-1] + 5, photon[-1], f"k = {k:g}", va="center")
    spec.axvspan(lo, hi, color=fig_cast["furniture"]["color"], alpha=0.5, lw=0)
    spec.text(
        0.5 * (lo + hi),
        spec.get_ylim()[1] * 0.97,
        "band R = 1",
        ha="center",
        va="top",
        fontsize="small",
    )
    spec.set(xlim=(400, 850), xlabel="wavelength [nm]")
    spec.set_ylabel("photon spectral radiance\n[$10^{12}$ " + RADIANCE_UNIT + "]")
    quiet(spec)

    panel_label(err, "(b) Converting once at band center")
    frac = np.linspace(0.05, 0.5, 46)
    for k, style in zip(SLOPES, styles, strict=True):
        rel = []
        for f in frac:
            a, b = lam0 * (1 - f / 2), lam0 * (1 + f / 2)
            exact = drc.power_law_photon_band_integral(b0, lam0, k, a, b)
            approx = drc.center_conversion_band_integral(b0, lam0, k, a, b)
            rel.append(approx / exact - 1.0)
        err.plot(
            frac * 100,
            np.array(rel) * 100,
            color=style["color"],
            ls=style["ls"],
        )
        err.text(51, rel[-1] * 100, f"k = {k:g}", va="center")
    err.axhline(0.0, color=fig_cast["furniture"]["color"], lw=1)
    err.set(xlabel="fractional bandwidth [%]", xlim=(0, 58))
    err.set_ylabel("relative error of the band integral [%]")
    quiet(err)

    panel_label(chain, "(c) Each factor enters once")
    steps = [
        ("$I_\\lambda$ photon spectral radiance", RADIANCE_UNIT),
        (
            "$\\int R(\\lambda)\\,I_\\lambda\\,d\\lambda$  (band, once)",
            "photon s$^{-1}$ m$^{-2}$ sr$^{-1}$",
        ),
        (
            "$\\times\\,\\Delta\\Omega_{\\rm pix}$  (solid angle)",
            "photon s$^{-1}$ m$^{-2}$ per pixel",
        ),
        ("$\\times\\,A\\,T_{\\rm opt}$  (area, optics)", "photon s$^{-1}$ per pixel"),
        ("$\\times\\,q$  (QE, once)", "electron s$^{-1}$ per pixel"),
    ]
    for i, (op, unit) in enumerate(steps):
        y = 1.0 - i * 0.2
        chain.text(
            0.02,
            y,
            op,
            va="center",
            bbox={"boxstyle": "round", "fc": "none", "ec": fig_cast["dust"]["color"]},
        )
        chain.text(0.62, y, unit, va="center", color=fig_cast["scenery"]["color"])
        if i:
            chain.annotate(
                "",
                xy=(0.12, y + 0.05),
                xytext=(0.12, y + 0.15),
                arrowprops={"arrowstyle": "->", "color": fig_cast["scenery"]["color"]},
            )
    chain.text(
        0.02,
        -0.08,
        "A legacy ratio divides by a declared zero point $F_0$ in the\nreceiving adapter; it is not a physical exchange quantity.",
        va="top",
        fontsize="small",
        color=fig_cast["scenery"]["color"],
    )
    chain.set(xlim=(0, 1.3), ylim=(-0.3, 1.1))
    chain.axis("off")
    return fig


# Figure 4: amplitude identifiability


def ring_morphology():
    n, half = RING["n_pix"], 0.5 * RING["field_au"]
    x = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(x, x)
    r = np.hypot(xx, yy)
    return np.exp(-((r - RING["radius_au"]) ** 2) / (2 * RING["width_au"] ** 2))


def identifiability(fig_cast):
    fig, axes = plt.subplots(
        1, 4, figsize=(17, 4.2), layout="constrained", width_ratios=[1, 1, 1, 1.15]
    )
    morph = ring_morphology()
    images = [
        require_physical(drc.product_amplitude_image(nz, alb, morph), "ring")
        for nz, alb in AMPLITUDES
    ]
    half = 0.5 * RING["field_au"]
    extent = (-half, half, -half, half)
    res = ep.compare_row(
        images,
        titles=[
            f"({t}) nzodis = {nz:g}, albedo = {alb:g}"
            for t, (nz, alb) in zip("ab", AMPLITUDES, strict=True)
        ],
        axes=axes[:2],
        norm="linear",
        extent=extent,
        cbar_label="brightness (normalized morphology\ntimes amplitude, arbitrary units)",
    )
    for ax in res.axes:
        ep.label_au(ax)
    res.axes[1].set_ylabel("")
    residual = images[0] - images[1]
    peak = float(images[0].max())
    diff = ep.imshow_diverging(
        residual,
        ax=axes[2],
        extent=extent,
        vlim=1e-3 * peak,
        cbar_label="(a) minus (b), same units",
    )
    ep.label_au(diff.ax)
    diff.ax.set_ylabel("")
    diff.ax.set_title(f"(c) (a) minus (b): max |diff| = {np.abs(residual).max():g}")

    like = axes[3]
    panel_label(like, "(d) Data constrain only the product")
    nz = np.linspace(0.4, 4.0, 300)
    alb = np.linspace(0.05, 0.6, 300)
    nn, aa = np.meshgrid(nz, alb)
    # Per-pixel noise chosen so the product is measured to +-0.02.
    sigma = 0.02 * math.sqrt(np.sum(morph**2))
    target = AMPLITUDES[0][0] * AMPLITUDES[0][1]
    chi2 = (nn * aa - target) ** 2 * np.sum(morph**2) / sigma**2
    delta = chi2 - chi2.min()
    levels = [1.0, 4.0, 9.0]
    contours = like.contour(
        nn,
        aa,
        delta,
        levels=levels,
        colors=[fig_cast["dust"]["color"]],
        linestyles=["-", "--", ":"],
        linewidths=1.2,
    )
    like.clabel(
        contours, fmt={1.0: "1", 4.0: "4", 9.0: "9"}, fontsize="small", inline=True
    )
    like.text(
        0.97,
        0.97,
        "contours: $\\Delta\\chi^2$ = 1, 4, 9",
        transform=like.transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )
    curve = fig_cast["wrong"]
    like.plot(nz, target / nz, color=curve["color"], ls=curve["ls"], lw=1.5)
    like.text(3.9, target / 3.9 + 0.035, "nzodis x albedo = 0.3", ha="right")
    for (n0, a0), key in zip(AMPLITUDES, ("dust", "dust_second"), strict=True):
        style = fig_cast[key]
        like.plot(
            n0,
            a0,
            marker=style["marker"],
            color=style["color"],
            ms=9,
            mec=plt.rcParams["text.color"],
        )
    like.set(xlabel="nzodis", ylabel="albedo (free spectral amplitude)")
    like.set_ylim(alb[0], alb[-1])
    return fig


# Figure 5: sign preflight and the negative control


def slab_images():
    n, half = SLAB["n_pix"], 0.5 * SLAB["field_au"]
    x = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(x, x)
    inc = math.radians(SLAB["inclination_deg"])
    args = (SLAB["radius_au"], SLAB["thickness_au"], SLAB["emissivity"], inc)
    good = drc.thin_slab_radiance(xx, yy, *args)
    bad = drc.thin_slab_radiance(xx, yy, *args, signed=True)
    return good, bad, (-half, half, -half, half)


def sign_control(fig_cast):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    good, bad, extent = slab_images()
    peak = float(good.max())
    vmin = peak / 100.0
    label = "radiance [" + RADIANCE_UNIT + "]"
    a = ep.imshow_log(
        require_physical(good, "slab"),
        ax=axes[0],
        extent=extent,
        vmin=vmin,
        vmax=peak,
        cbar_label=label,
    )
    a.ax.set_title("(a) path $h/|\\cos i|$, log display", loc="left")
    # Deliberate bypass of require_physical: this panel shows what the
    # floored log display does to a negative map.
    b = ep.imshow_log(
        bad, ax=axes[1], extent=extent, vmin=vmin, vmax=peak, cbar_label=label
    )
    b.ax.set_title("(b) path $h/\\cos i$, same log display", loc="left")
    b.ax.text(
        0.5,
        0.5,
        "negative pixels floored:\nindistinguishable from empty sky",
        transform=b.ax.transAxes,
        ha="center",
        va="center",
        color=fig_cast["wrong"]["color"],
    )
    c = ep.imshow_diverging(bad, ax=axes[2], extent=extent, vlim=peak, cbar_label=label)
    c.ax.set_title("(c) the same map, signed display", loc="left")
    for res in (a, b, c):
        ep.label_au(res.ax)
    for res in (b, c):
        res.ax.set_ylabel("")
    return fig


FIGURES = {
    "geometry": {
        "question": "How can the same cloud be viewed from inside and outside?",
        "status": "analytic fixture and schematic",
        "parameters": {"sphere": SPHERE, "rays": SPHERE_RAYS},
        "quantity": "photon radiance along positive half-rays; scattering angle",
        "normalization": "absolute fixture units (emissivity per AU)",
    },
    "sampling": {
        "question": "Why does resolution change a pixel value but not the integral?",
        "status": "analytic fixture",
        "parameters": {"clump": CLUMP, "grids": GRIDS},
        "quantity": "pixel-integrated photon flux density per pixel",
        "normalization": "absolute; shared log norm across the three grids",
    },
    "radiometry": {
        "question": "Where do wavelength, solid angle, zero point and band enter?",
        "status": "synthetic spectrum and schematic",
        "parameters": {"band": BAND, "slopes": SLOPES},
        "quantity": "photon spectral radiance; relative band-integral error",
        "normalization": "synthetic energy radiance b0 at lam0",
    },
    "identifiability": {
        "question": "Why can two parameter vectors make the same image?",
        "status": "analytic fixture",
        "parameters": {"ring": RING, "amplitudes": AMPLITUDES, "sigma": "0.02 peak"},
        "quantity": "normalized morphology times amplitude; Gaussian likelihood",
        "normalization": "arbitrary units; shared linear norm for the two images",
    },
    "sign-control": {
        "question": "What does a negative line-of-sight weight look like?",
        "status": "negative control (deliberately wrong kernel)",
        "parameters": {"slab": SLAB},
        "quantity": "photon radiance of a thin inclined slab",
        "normalization": "absolute fixture units; log vmin = peak / 100",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(path):
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(
        (ROOT / SCRIPT).read_bytes()
        + (ROOT / "tools" / "dust_reference_cases.py").read_bytes()
    ).hexdigest()[:12]
    outputs = []
    extras = {}
    builders = {
        "geometry": geometry,
        "sampling": sampling,
        "radiometry": radiometry,
        "identifiability": identifiability,
        "sign-control": sign_control,
    }
    for mode in ("light", "dark"):
        hwostyle.use(mode)
        fig_cast = cast()
        for name, builder in builders.items():
            made = builder(fig_cast)
            fig, extra = made if isinstance(made, tuple) else (made, None)
            if extra is not None:
                extras[name] = extra
            fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
            ep.stamp(
                fig,
                script=SCRIPT,
                sha=source_hash,
                note=f"spohnbook dust fixture: {FIGURES[name]['status']}",
            )
            for suffix in FORMATS:
                dpi = {"dpi": 150} if suffix == "png" else {}
                path = ep.save_fig(fig, f"dust-{name}-{mode}.{suffix}", dir=OUT, **dpi)
                outputs.append({"file": path.name, "sha256": sha256(path)})
            plt.close(fig)

    versions = {
        pkg: importlib.metadata.version(pkg)
        for pkg in ("eyepiece", "hwostyle", "hwoutils", "matplotlib", "numpy")
    }
    manifest = {
        "script": SCRIPT,
        "fixtures": "tools/dust_reference_cases.py",
        "source_hash": source_hash,
        "source_sha256": {
            SCRIPT: sha256(ROOT / SCRIPT),
            "tools/dust_reference_cases.py": sha256(
                ROOT / "tools" / "dust_reference_cases.py"
            ),
        },
        "spohnbook_revision": git_revision(ROOT),
        "eyepiece_revision": git_revision(Path(ep.__file__).resolve().parents[2]),
        "hwostyle_revision": git_revision(Path(hwostyle.__file__).resolve().parents[2]),
        "eyepiece_api_floor": "0.4.0: only names exported at tag v0.4.0 are used",
        "package_versions": versions,
        "figures": FIGURES,
        "derived_values": extras,
        "outputs": outputs,
    }
    (OUT / "dust-figure-manifest.json").write_text(
        json.dumps(manifest, indent=2, default=list) + "\n"
    )
    return outputs


if __name__ == "__main__":
    for item in build():
        print(item["file"])
