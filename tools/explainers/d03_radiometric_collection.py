"""What an aperture and a pixel collect: densities and the integrals that count them.

The overview is a side view of one collection chain (a point source and an
extended source, the receiving aperture, the optics, one detector pixel) with
each symbol of the expected-count integral written where it is defined, and
three adjacent panels for the three integrals that turn densities into one
pixel's count: solid angle (with the spatial response), wavelength (with the
optical response and QE inside the integral) and live time.

The same panel constructors make a relay for the book and for talks. Piece 1,
"where each factor lives", splits the side view into a point-source row and an
extended-source row and ends on the face-on detector; piece 2 carries that
detector beside the spectral integral; piece 3 carries the spectral integral
beside the live interval. Each piece carries the previous piece's new panel as
its left panel, drawn by the same constructor.

The specialization draws the photon-to-electron reference experiment (1 Jy
at 700 nm held constant over 1 nm, 1 m2, 1 s, four equally illuminated
pixels, constant QE, no detector noise). Its printed numbers come from
hwoutils here and are checked against an independent decimal calculation in
tests/test_explainer_d03.py.

Hand-rolled shapes (no eyepiece or domain primitive exists): the side-view
collection chain (sky patch, parallel beam, footprint cone, converging beam,
side-on pixel column with an image profile), the chief-ray mapping of a sky
cell onto a pixel, the rectangular reference spectrum, and the 2 by 2 pixel
count grid with per-pixel text. Piece 1 replaces the stroked label halos of the
shared helpers with a plain backing box. The face-on point-spread image uses
eyepiece ``imshow_log``.
"""

import eyepiece as ep
import numpy as np
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text
from scipy.special import j1

from explainers import _common as ex

# Reference experiment of the photon-to-electron page.
REFERENCE = {
    "flux_jy": 1.0,
    "wavelength_nm": 700.0,
    "area_m2": 1.0,
    "bandwidth_nm": 1.0,
    "exposure_s": 1.0,
    "n_pixels": 4,
    "qe_values": (0.0, 0.5, 1.0),
}

# Illustrative point-spread image: an unobscured circular aperture, sampled
# at one lambda/D per pixel on a 7 by 7 pixel window.
PSF_PIXELS = 7
PSF_PITCH_LOD = 1.0
PSF_OVERSAMPLE = 15

# Scene geometry in data units (side view, light travels toward +x).
AXIS_Y = 5.5
APERTURE_X = 8.0
APERTURE_D = 3.6
LENS_X = 8.8
PLATE_X = 12.5
DETECTOR_X = 17.3
PIXEL_H = 0.8
N_SIDE_PIXELS = 7
BEAM_HALF = 1.5

STATUS = "schematic, not to scale"
STATUS_REFERENCE = "reference experiment, idealized"


def reference_numbers():
    """The printed numbers of the reference experiment, from hwoutils.

    Returns:
        A dict with the photon density (photon s-1 m-2 nm-1), the photons
        collected in the exposure, the expected photons per pixel, and the
        summed deterministic variance (electron2) for each QE value.
    """
    r = REFERENCE
    density = float(jy_to_photons_per_nm_per_m2(r["flux_jy"], r["wavelength_nm"]))
    photons = density * r["area_m2"] * r["bandwidth_nm"] * r["exposure_s"]
    return {
        "density": density,
        "rate": density * r["area_m2"] * r["bandwidth_nm"],
        "photons": photons,
        "per_pixel": photons / r["n_pixels"],
        "variance": {q: q * photons for q in r["qe_values"]},
    }


def fmt(value):
    """Format a printed count with thousands separators and two decimals."""
    return f"{value:,.2f}"


def airy(r_lod):
    """Airy intensity for an unobscured circular aperture, peak 1.

    Args:
        r_lod: Radius in lambda/D.

    Returns:
        ``[2 J1(pi r) / (pi r)]^2``, equal to 1 at ``r = 0``.
    """
    x = np.pi * np.asarray(r_lod, dtype=float)
    safe = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 1.0, (2.0 * j1(safe) / safe) ** 2)


def psf_pixel_fractions(n=PSF_PIXELS, pitch=PSF_PITCH_LOD, oversample=PSF_OVERSAMPLE):
    """Fraction of a centered point source's photons in each pixel.

    The Airy pattern integrates to ``4 / pi`` in (lambda/D)^2 for unit peak,
    so dividing the pixel integrals by that total gives photon fractions. A
    finite window captures less than all the light.

    Args:
        n: Pixels on a side (odd, so one pixel is centered).
        pitch: Pixel side in lambda/D.
        oversample: Sub-samples per pixel side.

    Returns:
        An ``(n, n)`` array of fractions.
    """
    step = pitch / oversample
    edge = 0.5 * n * pitch
    centers = -edge + step * (np.arange(n * oversample) + 0.5)
    xx, yy = np.meshgrid(centers, centers)
    sub = airy(np.hypot(xx, yy)) * step**2 / (4.0 / np.pi)
    return sub.reshape(n, oversample, n, oversample).sum(axis=(1, 3))


# Panel constructors


def _text(ax, xy, text, cast, *, color=None, **kw):
    """Plain label in the given color (text color by default), with a halo."""
    return ex.halo(
        ax.text(*xy, text, color=cast.text if color is None else color, **kw), cast
    )


def _carried(ax, cast, text="(from the previous slide)"):
    ex.note(ax, (0.0, 1.1), text, cast, transform=ax.transAxes, ha="left", va="bottom")


def _extended_region(ax, patch, cast):
    """The extended source: a quiet fill and hatch with a bright outline."""
    ex.region(ax, "local_zodi", patch, cast)
    green = np.array(to_rgb(cast["local_zodi"].color))
    bg = np.array(to_rgb(cast.background))
    dark = cast.mode == "dark"
    hatch = bg + (0.45 if dark else 0.8) * (green - bg)
    patch.set_facecolor(to_rgba(cast["local_zodi"].color, 0.06 if dark else 0.10))
    patch.set_hatchcolor(tuple(hatch))
    return patch


def draw_scene(ax, cast, *, detail=True):
    """Side view of the collection chain with each symbol at its location.

    Args:
        ax: Axes to draw into; set to equal aspect here.
        cast: The active ``Cast``.
        detail: Whether to add the ownership tags under each element.
    """
    ax.set(xlim=(-0.2, 30.4), ylim=(-1.1 if detail else 0.6, 12.5), aspect="equal")
    ax.axis("off")
    small = cast.layout.small_pt
    ink = cast.text
    guide = cast.neutral(0.55)
    lw = cast.layout.lw
    y0 = AXIS_Y

    # Sky: an extended source with a point source in front of it, and the
    # sky cell that an idealized mapping assigns to pixel p.
    patch_top, patch_bot = 8.5, 2.5
    _extended_region(ax, Rectangle((0.3, patch_bot), 1.8, patch_top - patch_bot), cast)
    foot_lo, foot_hi = y0 - 1.0, y0 + 1.0
    ax.add_patch(
        Rectangle(
            (0.3, foot_lo),
            1.8,
            foot_hi - foot_lo,
            facecolor="none",
            edgecolor=guide,
            ls="--",
            lw=lw,
            zorder=4,
        )
    )
    star = (1.2, y0)
    ex.mark(ax, "star", star, cast, scale=0.9)
    _text(
        ax,
        (0.3, patch_top + 0.25),
        r"extended source $I_\lambda$",
        cast,
        color=cast["local_zodi"].color,
        ha="left",
        va="bottom",
    )
    _text(
        ax,
        (0.3, patch_bot - 0.25),
        r"point source $\Phi_\lambda$",
        cast,
        color=cast["star"].color,
        ha="left",
        va="top",
    )
    ex.note(
        ax,
        (0.3, patch_top + 1.05),
        r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$",
        cast,
        ha="left",
        va="bottom",
    )
    ex.note(
        ax,
        (0.3, patch_bot - 1.05),
        r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$",
        cast,
        ha="left",
        va="top",
    )
    ex.note(
        ax,
        (2.5, y0 - BEAM_HALF - 0.25),
        "sky cell of $p$\n(idealized)",
        cast,
        ha="left",
        va="top",
        linespacing=1.1,
    )

    # Starlight leaves the star, the distance is broken, and the beam from
    # the distant source arrives parallel.
    for y in (y0 - BEAM_HALF, y0 + BEAM_HALF):
        ex.arrow(ax, (2.5, y), (APERTURE_X - 0.05, y), "ray", cast, source="star")
        ex.scale_break(ax, (4.4, y), cast)
    ex.note(
        ax,
        (2.5, y0 + BEAM_HALF + 0.25),
        "parallel starlight",
        cast,
        ha="left",
        va="bottom",
        linespacing=1.1,
    )

    # The solid angle of the sky cell, seen from the aperture.
    for y in (foot_lo, foot_hi):
        ax.plot([2.1, APERTURE_X], [y, y0], color=guide, lw=0.7 * lw, ls=":", zorder=3)
    half_deg = np.degrees(np.arctan2(foot_hi - y0, APERTURE_X - 2.1))
    ex.angle_arc(
        ax,
        (APERTURE_X, y0),
        180.0 - half_deg,
        180.0 + half_deg,
        cast,
        radius=3.0,
        sense=False,
        color=guide,
    )
    _text(
        ax,
        (4.8, y0),
        r"$\Omega_p$ [sr]",
        cast,
        color=cast.neutral(0.75),
        ha="right",
        va="center",
    )

    # Aperture, focusing optics and a transmitting element.
    ex.aperture(ax, (APERTURE_X, y0), APERTURE_D, cast)
    _text(
        ax,
        (APERTURE_X, y0 + 0.5 * APERTURE_D + 0.8),
        "aperture\n" + r"area $A$ [m$^2$]",
        cast,
        ha="center",
        va="bottom",
        linespacing=1.1,
    )
    ex.lens(ax, (LENS_X, y0), APERTURE_D, cast)
    for sign in (-1.0, 1.0):
        ex.arrow(
            ax,
            (LENS_X, y0 + sign * BEAM_HALF),
            (DETECTOR_X, y0),
            "ray",
            cast,
            source="star",
        )
    plate_half = BEAM_HALF * (DETECTOR_X - PLATE_X) / (DETECTOR_X - LENS_X) + 0.5
    ex.region(
        ax,
        "optics",
        Rectangle((PLATE_X - 0.18, y0 - plate_half), 0.36, 2 * plate_half),
        cast,
    )
    _text(
        ax,
        (PLATE_X, y0 + plate_half + 0.35),
        "optics\n" + r"$T_{\rm opt}(\lambda)$",
        cast,
        ha="center",
        va="bottom",
        linespacing=1.1,
    )

    # Side-on pixel column with the line-spread profile of the image at one
    # wavelength: the light is not one ray per pixel.
    col_lo = y0 - 0.5 * N_SIDE_PIXELS * PIXEL_H
    col_hi = col_lo + N_SIDE_PIXELS * PIXEL_H
    for k in range(N_SIDE_PIXELS):
        rect = Rectangle((DETECTOR_X, col_lo + k * PIXEL_H), 0.6, PIXEL_H)
        ex.region(ax, "detector", rect, cast)
        if k == N_SIDE_PIXELS // 2:
            rect.set_edgecolor(ink)
            rect.set_linewidth(1.8 * lw)
            rect.set_zorder(5)
    ys = np.linspace(col_lo, col_hi, 400)
    x_base = DETECTOR_X + 0.8
    xs = x_base + 2.0 * airy((ys - y0) / PIXEL_H * PSF_PITCH_LOD)
    in_p = np.abs(ys - y0) <= 0.5 * PIXEL_H
    ax.plot(xs, ys, color=guide, lw=0.8 * lw, zorder=4)
    ax.fill_betweenx(ys, x_base, xs, where=in_p, color=guide, alpha=0.6, lw=0, zorder=3)
    ax.fill_betweenx(
        ys, x_base, xs, where=~in_p, color=guide, alpha=0.2, lw=0, zorder=3
    )
    ax.plot([x_base, x_base], [col_lo, col_hi], color=cast.neutral(0.4), lw=0.5 * lw)
    _text(
        ax,
        (DETECTOR_X + 0.3, col_hi + 0.35),
        "detector QE\n" + r"$q_p(\lambda)$",
        cast,
        ha="center",
        va="bottom",
        linespacing=1.1,
    )
    _text(ax, (x_base + 2.2, y0), r"pixel $p$", cast, ha="left", va="center")
    _text(
        ax,
        (x_base + 0.3, col_lo - 0.3),
        r"image spread: $P_p(\lambda)$",
        cast,
        ha="center",
        va="top",
    )

    # The photon rate arriving in the pixel, and the electron count kept there.
    rx = 23.3
    _text(
        ax,
        (rx, 12.3),
        "arriving in $p$\n(point source):\n"
        + r"$A\,\Phi_\lambda T_{\rm opt}P_p$"
        + "\n"
        + r"photon s$^{-1}$ nm$^{-1}$",
        cast,
        ha="left",
        va="top",
        fontsize=small,
        linespacing=1.2,
    )
    ex.note(
        ax,
        (rx, 9.2),
        "extended source:\n"
        + r"$\int I_\lambda P_p\,d\Omega$ replaces"
        + "\n"
        + r"$\Phi_\lambda P_p$",
        cast,
        ha="left",
        va="top",
        linespacing=1.2,
    )
    ex.arrow(ax, (rx + 0.6, 6.5), (rx + 0.6, 4.9), "data", cast)
    _text(
        ax,
        (rx + 1.4, 5.7),
        r"$\times\,q_p$, $\int d\lambda$" + "\n" + r"$\int_{\rm live}dt$",
        cast,
        ha="left",
        va="center",
        linespacing=1.3,
    )
    _text(
        ax,
        (rx, 4.4),
        "accumulated in $p$:\n" + r"$\mu_{e,p}$ [electron]",
        cast,
        ha="left",
        va="top",
        fontsize=small,
        linespacing=1.2,
        fontweight="bold",
    )

    if detail:
        # Where each loss is assigned, so that no loss enters two factors.
        ex.note(
            ax,
            (APERTURE_X + 0.3, 0.95),
            "obscuration: in $A$, or\ndeclared by the primary;\nnever also in $P_p$",
            cast,
            ha="center",
            va="top",
            linespacing=1.1,
        )
        ex.note(
            ax,
            (PLATE_X, y0 - plate_half - 0.3),
            "optical losses:\nin $T_{\\rm opt}$, QE excluded",
            cast,
            ha="center",
            va="top",
            linespacing=1.1,
        )
        ex.note(
            ax,
            (x_base + 0.3, col_lo - 1.05),
            "coronagraph attenuation and\nlight lost outside $p$: in $P_p$;\n"
            "QE once, in $q_p$",
            cast,
            ha="center",
            va="top",
            linespacing=1.1,
        )


def draw_detector(ax, cast, *, carried=False):
    """Face-on detector window with the point source's spread image.

    Args:
        ax: Axes to draw into.
        cast: The active ``Cast``.
        carried: Tag the panel as carried over from the previous slide.
    """
    frac = psf_pixel_fractions()
    n = PSF_PIXELS
    half = 0.5 * n
    ep.imshow_log(
        frac,
        ax=ax,
        extent=(-half, half, -half, half),
        vmin=float(frac.max()) * 1e-3,
        vmax=float(frac.max()),
        cmap=ex.image_cmap("intensity"),
        colorbar=False,
    )
    ex.pixel_grid(ax, (-half, -half), (n, n), 1.0, cast)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            1.0,
            1.0,
            facecolor="none",
            edgecolor=cast.text,
            lw=2.2 * cast.layout.lw,
            zorder=6,
        )
    )
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(r"angle: $P_p$, point source", loc="left")
    _text(ax, (0.0, 0.62), r"$p$", cast, ha="center", va="bottom")
    ex.note(
        ax,
        (0.5, -0.03),
        "face-on, one wavelength, log scale:\nthe image spreads into neighbors",
        cast,
        transform=ax.transAxes,
        ha="center",
        va="top",
        linespacing=1.1,
    )
    if carried:
        _carried(ax, cast, "(pixel $p$ from the previous slide)")


def _spectral_curves():
    """Shapes on a normalized wavelength axis from 0 to 1.

    ``P_p`` is the centered pixel's share of the Airy image, with the pixel
    pitch in lambda/D shrinking as wavelength grows (0.85 to 1.2 times the
    reference wavelength), so the share falls toward longer wavelengths.
    """
    lam = np.linspace(0.0, 1.0, 400)
    phi = 1.0 - 0.45 * lam + 0.08 * np.sin(5.0 * lam)
    t_opt = 0.82 - 0.25 * (lam - 0.55) ** 2
    qe = 0.92 - 0.6 * lam**2
    coarse = np.linspace(0.0, 1.0, 15)
    ratio = 0.85 + 0.35 * coarse
    c = PSF_PIXELS // 2
    share = [psf_pixel_fractions(pitch=PSF_PITCH_LOD / r)[c, c] for r in ratio]
    p_p = np.interp(lam, coarse, share)
    return lam, phi / phi.max(), t_opt, p_p / p_p.max(), qe


BAND = (0.33, 0.67)


def draw_spectral(ax, cast, *, carried=False):
    """Wavelength integral: density, responses and QE inside it.

    Args:
        ax: Axes to draw into.
        cast: The active ``Cast``.
        carried: Tag the panel as carried over from the previous slide.
    """
    lam, phi, t_opt, p_p, qe = _spectral_curves()
    integrand = phi * t_opt * p_p * qe
    lo, hi = BAND
    band = (lam >= lo) & (lam <= hi)
    lw = cast.layout.lw
    hardware = cast.neutral(0.7)
    ax.axvspan(lo, hi, color=cast.neutral(0.1), lw=0, zorder=0)
    ax.fill_between(
        lam[band], 0.0, integrand[band], color=cast.text, alpha=0.25, lw=0, zorder=1
    )
    ax.plot(lam, phi, color=cast["star"].color, lw=lw, zorder=3)
    ax.plot(lam, t_opt, color=hardware, lw=lw, zorder=3)
    ax.plot(lam, p_p, color=hardware, lw=lw, ls=":", zorder=3)
    ax.plot(lam, qe, color=hardware, lw=lw, ls="--", zorder=3)
    ax.plot(lam, integrand, color=cast.text, lw=0.9 * lw, zorder=3)
    lab = {"ha": "left", "va": "center"}
    _text(ax, (1.01, t_opt[-1] + 0.05), r"$T_{\rm opt}$", cast, color=hardware, **lab)
    _text(ax, (1.01, p_p[-1] - 0.03), r"$P_p$", cast, color=hardware, **lab)
    _text(
        ax,
        (1.01, phi[-1] - 0.05),
        r"$\Phi_\lambda$",
        cast,
        color=cast["star"].color,
        **lab,
    )
    _text(ax, (1.01, qe[-1] - 0.06), r"$q_p$", cast, color=hardware, **lab)
    _text(
        ax,
        (1.01, integrand[-1] + 0.02),
        r"$\Phi_\lambda T_{\rm opt}P_p q_p$",
        cast,
        fontsize=cast.layout.small_pt,
        ha="left",
        va="bottom",
    )
    _text(ax, (0.5, 0.1), r"$\int d\lambda$", cast, ha="center", va="center")
    ax.set(xlim=(0.0, 1.5), ylim=(0.0, 1.08), yticks=[])
    ax.set_xticks([lo, hi], [r"$\lambda_b^-$", r"$\lambda_b^+$"])
    ax.set_xlabel(r"wavelength $\lambda$")
    ax.set_ylabel("normalized (shape only)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(r"wavelength: $\int d\lambda$ after $q_p$", loc="left")
    if carried:
        _carried(ax, cast)


LIVE = (0.15, 0.8)


def draw_time(ax, cast, *, link=False):
    """Live-time integral: the expected electron count accumulating in p.

    Args:
        ax: Axes to draw into.
        cast: The active ``Cast``.
        link: Name the slope as the band integral of the panel beside it.
    """
    t0, t1 = LIVE
    t = np.linspace(0.0, 1.0, 300)
    mu = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    ax.axvspan(t0, t1, color=cast.neutral(0.1), lw=0, zorder=0)
    ax.plot(t, mu, color=cast.text, lw=cast.layout.lw, zorder=3)
    ax.plot(
        [t1],
        [1.0],
        marker="o",
        color=cast.text,
        ms=0.8 * cast.layout.marker_pt,
        zorder=4,
    )
    _text(ax, (t1, 1.06), r"$\mu_{e,p}$", cast, ha="center", va="bottom")
    slope = (
        "slope = band integral\n" + r"at left, electron s$^{-1}$"
        if link
        else r"slope: electron s$^{-1}$"
    )
    _text(
        ax,
        (t0 + 0.02, 0.62 if link else 0.7),
        slope,
        cast,
        ha="left",
        va="bottom",
        fontsize=cast.layout.small_pt,
        linespacing=1.15,
    )
    ex.note(
        ax,
        (0.5 * (t0 + t1), 1.4),
        "live: exposing,\nnot reading out",
        cast,
        ha="center",
        va="top",
        linespacing=1.1,
    )
    ex.arrow(ax, (t0, 0.1), (t1, 0.1), "integration", cast)
    _text(
        ax,
        (t1 - 0.12, 0.15),
        r"$t_{\rm live}$",
        cast,
        color=cast["integration"].color,
        ha="center",
        va="bottom",
    )
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.42), xticks=[], yticks=[])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("expected electrons in $p$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(r"time: $\int_{\rm live}dt$", loc="left")


# Figures


def build_overview(layout, cast):
    """The overview: the side-view chain, with the three integrals below it
    on the documentation page."""
    if layout.is_slide:
        fig, ax = ex.figure(layout)
        draw_scene(ax, cast, detail=True)
        fig.suptitle(
            "A pixel's count integrates densities over angle, wavelength and time"
        )
        ex.badge(ax, cast, STATUS, loc="lower right")
        return fig
    fig, _ = ex.figure(layout, doc_height_in=6.0)
    fig.clear()
    grid = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.0])
    ax_s = fig.add_subplot(grid[0, :])
    draw_scene(ax_s, cast, detail=True)
    ex.badge(ax_s, cast, STATUS, loc="lower right")
    draw_detector(fig.add_subplot(grid[1, 0]), cast)
    draw_spectral(fig.add_subplot(grid[1, 1]), cast)
    draw_time(fig.add_subplot(grid[1, 2]), cast)
    return fig


def build_wavelength(layout, cast):
    """Relay step 2: the face-on detector, then the spectral integral."""
    fig, (ax_d, ax_w) = ex.figure(
        layout, doc_height_in=2.9, ncols=2, width_ratios=[1.0, 1.5]
    )
    draw_detector(ax_d, cast, carried=layout.is_slide)
    draw_spectral(ax_w, cast)
    if layout.is_slide:
        fig.suptitle("Response and QE act at each wavelength, before the band integral")
    ex.badge(ax_w, cast, STATUS, loc="upper right")
    return fig


def build_electrons(layout, cast):
    """Relay step 3: the spectral integral, then the live-time integral."""
    fig, (ax_w, ax_t) = ex.figure(layout, doc_height_in=2.9, ncols=2)
    draw_spectral(ax_w, cast, carried=layout.is_slide)
    draw_time(ax_t, cast, link=True)
    if layout.is_slide:
        fig.suptitle("Electrons accumulate only while the pixel is live")
    ex.badge(ax_w, cast, STATUS, loc="upper right")
    return fig


DENSITY_UNIT = r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$"


def _reference_spectrum(ax, cast, nums):
    r = REFERENCE
    lam0, dlam = r["wavelength_nm"], r["bandwidth_nm"]
    x = [lam0 - 1.5, lam0 - 0.5 * dlam, lam0 - 0.5 * dlam]
    x += [lam0 + 0.5 * dlam, lam0 + 0.5 * dlam, lam0 + 1.5]
    y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    small = cast.layout.small_pt
    ax.fill_between(x, 0.0, y, color=cast["star"].color, alpha=0.25, lw=0)
    ax.plot(x, y, color=cast["star"].color, lw=cast.layout.lw)
    ax.set(xlim=(lam0 - 1.5, lam0 + 1.5), ylim=(0.0, 1.75), yticks=[])
    ax.set_xticks([lam0], [f"{lam0:.0f} nm"])
    ax.set_xlabel(r"wavelength $\lambda$")
    ax.spines[["top", "right", "left"]].set_visible(False)
    _text(
        ax,
        (lam0, 1.07),
        r"$\Phi_{\lambda,\mathrm{nm}}$ = "
        + fmt(nums["density"])
        + "\n"
        + DENSITY_UNIT
        + "\n(1 Jy at 700 nm)",
        cast,
        ha="center",
        va="bottom",
        fontsize=small,
        linespacing=1.15,
    )
    ax.annotate(
        "",
        (lam0 - 0.5 * dlam, 0.45),
        (lam0 + 0.5 * dlam, 0.45),
        arrowprops={
            "arrowstyle": "<->",
            "color": cast.text,
            "lw": 0.8 * cast.layout.lw,
        },
    )
    _text(
        ax,
        (lam0, 0.5),
        r"$\Delta\lambda$ = 1 nm",
        cast,
        ha="center",
        va="bottom",
        fontsize=small,
    )
    ax.set_title("synthetic spectrum", loc="left")


def build_reference(layout, cast):
    """The photon-to-electron reference experiment, with its numbers."""
    nums = reference_numbers()
    fig, (ax_l, ax) = ex.figure(
        layout, doc_height_in=3.2, ncols=2, width_ratios=[1.0, 2.5]
    )
    _reference_spectrum(ax_l, cast, nums)
    ax_l.set_anchor("N")

    ax.set(xlim=(0.0, 25.0), ylim=(-2.3, 10.4), aspect="equal")
    ax.set_anchor("N")
    ax.axis("off")
    small = cast.layout.small_pt
    lw = cast.layout.lw
    ax.set_title("collection and counting", loc="left")

    # Aperture, exposure, and the collected photon rate.
    ex.aperture(ax, (1.2, 5.0), 4.0, cast)
    _text(
        ax,
        (1.2, 2.5),
        r"$A$ = 1 m$^2$" + "\n" + r"$t$ = 1 s",
        cast,
        ha="center",
        va="top",
        linespacing=1.3,
    )
    for y in (3.5, 6.5):
        ex.arrow(
            ax, (1.3, y), (7.3, 5.0 + 0.55 * (y - 5.0)), "ray", cast, source="star"
        )
    _text(
        ax,
        (4.2, 7.3),
        r"$\times A \times \Delta\lambda$ ="
        + "\n"
        + fmt(nums["rate"])
        + "\n"
        + r"photon s$^{-1}$",
        cast,
        ha="center",
        va="bottom",
        fontsize=small,
        linespacing=1.15,
    )

    # Four pixels, equal share by construction.
    x0, y0, pitch = 7.6, 2.4, 2.7
    for row in range(2):
        for col in range(2):
            ax.add_patch(
                Rectangle(
                    (x0 + col * pitch, y0 + row * pitch),
                    pitch,
                    pitch,
                    facecolor=cast.neutral(0.08),
                    edgecolor=cast["detector"].color,
                    lw=lw,
                )
            )
            _text(
                ax,
                (x0 + (col + 0.5) * pitch, y0 + (row + 0.5) * pitch),
                fmt(nums["per_pixel"]) + "\nphoton",
                cast,
                ha="center",
                va="center",
                fontsize=0.82 * small,
                linespacing=1.1,
            )
    _text(
        ax,
        (x0 + pitch, y0 + 2 * pitch + 0.25),
        r"$\times\,t\,/\,4$: photons per pixel" + "\n" + r"in $t$ = 1 s",
        cast,
        ha="center",
        va="bottom",
        fontsize=small,
        linespacing=1.15,
    )
    ex.note(
        ax,
        (x0 + pitch, y0 - 0.25),
        "4 pixels, equal split by\nconstruction, not a\npoint-spread function",
        cast,
        ha="center",
        va="top",
        linespacing=1.1,
    )

    # Photons to electrons: summed expectation equals summed variance.
    ex.arrow(ax, (x0 + 2 * pitch + 0.3, 5.1), (x0 + 2 * pitch + 1.7, 5.1), "data", cast)
    tx = x0 + 2 * pitch + 2.0
    _text(
        ax,
        (tx, 10.3),
        "expected electrons",
        cast,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ex.note(
        ax,
        (tx, 9.2),
        "constant QE $q$; summed electrons [e]\n= summed Poisson variance [e$^2$],\n"
        "numerically",
        cast,
        ha="left",
        va="top",
        linespacing=1.15,
    )
    for k, (q, v) in enumerate(nums["variance"].items()):
        _text(
            ax,
            (tx, 6.3 - 0.95 * k),
            f"q = {q:.1f}:  {fmt(v)}",
            cast,
            ha="left",
            va="top",
            fontsize=small,
        )
    ex.note(
        ax,
        (tx, 3.2),
        "no dark current, clock-induced\ncharge or read noise; expected\n"
        "values, not a readout",
        cast,
        ha="left",
        va="top",
        linespacing=1.1,
    )
    ex.badge(ax, cast, STATUS_REFERENCE, loc="lower left")
    if layout.is_slide:
        fig.suptitle("The reference experiment: one conversion chain, exact arithmetic")
    return fig


# Relay piece 1, "where each factor lives": the side view split into a point
# source row and an extended source row, ending on the face-on detector that
# the wavelength figure carries. It has its own geometry so that the chief
# ray through the lens center maps the sky cell exactly onto pixel p (the
# overview's side view is not drawn to that constraint).
WHERE = {
    "axis_y": 4.2,
    "cell_x": 2.4,  # plane of the sky patch (its right edge)
    "aperture_x": 9.4,
    "aperture_d": 3.4,
    "lens_x": 10.0,  # the chief-ray pivot
    "outside_x": 1.35,  # a sky direction outside the cell
    "plate_x": 12.4,
    "detector_x": 15.0,
    "pixel_h": 1.0,
    "n_pixels": 7,
    "beam_half": 1.25,
    "xlim": (-0.2, 25.6),
    "ylim": (-0.1, 9.5),
}


def where_cell_half():
    """Half-height of the sky cell that the chief ray maps onto pixel p.

    Lines through the lens center keep their angle, so a cell edge at
    ``cell_x`` lands on a pixel edge at ``detector_x`` when the heights
    scale with the distances from the pivot.
    """
    w = WHERE
    return (
        0.5
        * w["pixel_h"]
        * (w["lens_x"] - w["cell_x"])
        / (w["detector_x"] - w["lens_x"])
    )


def where_outside_y():
    """Height of the drawn direction one pixel pitch outside the sky cell.

    Its chief ray through the lens center lands on the center of the pixel
    next to p, on the other side of the axis (the image is inverted).
    """
    w = WHERE
    return w["axis_y"] + w["pixel_h"] * (w["lens_x"] - w["outside_x"]) / (
        w["detector_x"] - w["lens_x"]
    )


def _backing(cast):
    """A plain backing box in the background color (no stroked halo)."""
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.25",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _plain_text(fig, cast):
    """Replace the stroked halos that shared helpers add with a backing box.

    Labels on this figure carry no stroke, which at documentation size turns
    glyphs into blobs; a label that had one sits on a plain box instead.
    """
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                text.set_bbox(_backing(cast))
    return fig


def _label(ax, xy, text, cast, *, color=None, boxed=False, **kw):
    """A label in data coordinates, with no halo, optionally on a box."""
    kw.setdefault("fontsize", cast.layout.small_pt)
    kw.setdefault("linespacing", 1.1)
    t = ax.text(*xy, text, color=cast.text if color is None else color, **kw)
    if boxed:
        t.set_bbox(_backing(cast))
    return t


def _where_hardware(ax, cast):
    """Aperture, lens, optics plate and the side-on pixel column.

    Returns:
        The y limits of pixel p.
    """
    w = WHERE
    y0, ph, lw = w["axis_y"], w["pixel_h"], cast.layout.lw
    ex.aperture(ax, (w["aperture_x"], y0), w["aperture_d"], cast)
    ex.lens(ax, (w["lens_x"], y0), w["aperture_d"], cast)
    plate_half = 0.5 * w["aperture_d"] - 0.2
    ex.region(
        ax,
        "optics",
        Rectangle((w["plate_x"] - 0.16, y0 - plate_half), 0.32, 2 * plate_half),
        cast,
    )
    col_lo = y0 - 0.5 * w["n_pixels"] * ph
    for k in range(w["n_pixels"]):
        rect = Rectangle((w["detector_x"], col_lo + k * ph), 0.55, ph)
        ex.region(ax, "detector", rect, cast)
        if k == w["n_pixels"] // 2:
            rect.set_edgecolor(cast.text)
            rect.set_linewidth(1.8 * lw)
            rect.set_zorder(5)
            rect.set_gid("pixel-p")
    return (y0 - 0.5 * ph, y0 + 0.5 * ph)


def _where_profile(ax, cast, center_y, p_lims, *, gid):
    """Image profile beside the column, centered on ``center_y``.

    A cut through the illustrative Airy image at one wavelength, one
    lambda/D per pixel; the part that falls inside pixel p is shaded dark.
    """
    w = WHERE
    ph = w["pixel_h"]
    col_lo = w["axis_y"] - 0.5 * w["n_pixels"] * ph
    col_hi = col_lo + w["n_pixels"] * ph
    ys = np.linspace(col_lo, col_hi, 561)
    x_base = w["detector_x"] + 0.75
    xs = x_base + 2.6 * airy((ys - center_y) / ph * PSF_PITCH_LOD)
    in_p = (ys >= p_lims[0]) & (ys <= p_lims[1])
    guide = cast.neutral(0.55)
    lw = cast.layout.lw
    ax.plot(xs, ys, color=guide, lw=0.8 * lw, zorder=4, gid=gid)
    ax.fill_betweenx(
        ys, x_base, xs, where=in_p, color=cast.neutral(0.7), alpha=0.9, lw=0, zorder=3
    )
    ax.fill_betweenx(
        ys, x_base, xs, where=~in_p, color=guide, alpha=0.18, lw=0, zorder=3
    )
    ax.plot([x_base, x_base], [col_lo, col_hi], color=cast.neutral(0.4), lw=0.5 * lw)
    return x_base


def _where_axes(ax):
    w = WHERE
    ax.set(xlim=w["xlim"], ylim=w["ylim"], aspect="equal")
    ax.axis("off")


def draw_where_point(ax, cast):
    """Row 1: the point source, parallel starlight, and its spread image."""
    w = WHERE
    _where_axes(ax)
    y0, bh = w["axis_y"], w["beam_half"]
    slide = cast.layout.is_slide
    star = cast["star"].color

    ex.mark(ax, "star", (1.3, y0), cast, scale=0.9)
    _label(
        ax,
        (0.2, y0 + bh + 0.25),
        r"point source $\Phi_\lambda$ at $\boldsymbol{\theta}$",
        cast,
        color=star,
        ha="left",
        va="bottom",
        fontsize=cast.layout.font_pt,
    )
    if not slide:
        ex.note(
            ax,
            (0.2, y0 - bh - 0.5),
            r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$",
            cast,
            ha="left",
            va="top",
        )
    for y in (y0 - bh, y0 + bh):
        ex.arrow(ax, (2.4, y), (w["aperture_x"] - 0.05, y), "ray", cast, source="star")
        ex.scale_break(ax, (3.6, y), cast)
    _label(
        ax,
        (6.4, y0),
        "parallel\nstarlight",
        cast,
        color=cast["annotation"].color,
        fontstyle="italic",
        ha="center",
        va="center",
    )

    p_lims = _where_hardware(ax, cast)
    for sign in (-1.0, 1.0):
        ex.arrow(
            ax,
            (w["lens_x"], y0 + sign * bh),
            (w["detector_x"], y0),
            "ray",
            cast,
            source="star",
        )
    x_base = _where_profile(ax, cast, y0, p_lims, gid="profile-point")

    top = y0 + 0.5 * w["aperture_d"] + 0.25
    _label(
        ax,
        (w["aperture_x"] - 0.2, top),
        "aperture\n$A$",
        cast,
        ha="center",
        va="bottom",
    )
    _label(
        ax,
        (w["plate_x"], top - 0.2),
        "optics\n" + r"$T_{\rm opt}(\lambda)$",
        cast,
        ha="center",
        va="bottom",
    )
    col_hi = y0 + 0.5 * w["n_pixels"] * w["pixel_h"]
    _label(
        ax,
        (w["detector_x"] + 0.3, col_hi + 0.15),
        "detector QE\n" + r"$q_p(\lambda)$",
        cast,
        ha="center",
        va="bottom",
    )
    _label(
        ax,
        (x_base + 3.1, y0),
        r"pixel $p$",
        cast,
        ha="left",
        va="center",
        fontsize=cast.layout.font_pt,
    )
    _label(
        ax,
        (x_base + 3.1, y0 - 1.4),
        "image spread: share in $p$\n"
        + r"of a source at $\boldsymbol{\theta}$: $P_p(\lambda,\boldsymbol{\theta})$",
        cast,
        ha="left",
        va="top",
    )


def draw_where_extended(ax, cast):
    """Row 2: the extended source, its sky cell, and light from outside it."""
    w = WHERE
    _where_axes(ax)
    y0 = w["axis_y"]
    slide = cast.layout.is_slide
    green = cast["local_zodi"].color
    guide = cast.neutral(0.55)
    lw = cast.layout.lw
    half = where_cell_half()
    cx = w["cell_x"]

    patch_lo, patch_hi = y0 - 3.0, y0 + 3.0
    _extended_region(
        ax, Rectangle((0.3, patch_lo), cx - 0.3, patch_hi - patch_lo), cast
    )
    ax.add_patch(
        Rectangle(
            (0.3, y0 - half),
            cx - 0.3,
            2 * half,
            facecolor="none",
            edgecolor=cast.text,
            ls="--",
            lw=lw,
            zorder=4,
            gid="sky-cell",
        )
    )
    _label(
        ax,
        (0.2, patch_hi + 0.15),
        r"extended source $I_\lambda$",
        cast,
        color=green,
        ha="left",
        va="bottom",
        fontsize=cast.layout.font_pt,
    )
    if not slide:
        ex.note(
            ax,
            (0.2, patch_lo - 0.15),
            r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$ sr$^{-1}$",
            cast,
            ha="left",
            va="top",
        )

    # The idealized mapping: straight lines through the lens center carry
    # the cell's edges onto pixel p's edges (inverted).
    pivot = (w["lens_x"], y0)
    for sign in (-1.0, 1.0):
        edge_y = y0 + sign * half
        land_y = y0 - sign * 0.5 * w["pixel_h"]
        ax.plot(
            [cx, pivot[0], w["detector_x"]],
            [edge_y, y0, land_y],
            color=guide,
            lw=0.8 * lw,
            ls=":",
            zorder=3,
            gid="cell-edge",
        )
    # The solid angle is the cone the cell subtends, with its vertex at the
    # lens center: a shaded wedge and an arc at the vertex.
    ax.add_patch(
        Polygon(
            [(cx, y0 - half), (cx, y0 + half), pivot],
            closed=True,
            facecolor=cast.neutral(0.5),
            alpha=0.22,
            edgecolor="none",
            zorder=2,
            gid="omega-cone",
        )
    )
    half_deg = np.degrees(np.arctan2(half, pivot[0] - cx))
    ex.angle_arc(
        ax,
        pivot,
        180.0 - half_deg,
        180.0 + half_deg,
        cast,
        radius=2.6,
        sense=False,
        color=cast.text,
    )
    _label(
        ax,
        (pivot[0] - 3.05, y0),
        r"$\Omega_p$",
        cast,
        ha="right",
        va="center",
        fontsize=cast.layout.font_pt,
    )
    _label(
        ax,
        (cx + 0.2, y0 - half - 0.15),
        "sky cell of $p$\n(idealized)",
        cast,
        color=cast["annotation"].color,
        fontstyle="italic",
        ha="left",
        va="top",
    )

    # A direction just outside the cell: its chief ray lands one pixel
    # below p, and its spread image reaches into p.
    ox, out_y = w["outside_x"], where_outside_y()
    ax.plot(
        [ox],
        [out_y],
        marker="o",
        ms=0.75 * cast.layout.marker_pt,
        mfc=green,
        mec=cast.text,
        mew=0.6 * lw,
        zorder=6,
        gid="outside-point",
    )
    _label(
        ax,
        (cx + 0.2, out_y + 0.35),
        r"outside $\Omega_p$",
        cast,
        color=green,
        ha="left",
        va="bottom",
    )
    land = w["detector_x"], y0 - w["pixel_h"]
    # The chief ray through the lens center is undeviated: one straight ray.
    ray = ex.arrow(ax, (ox, out_y), land, "ray", cast, source="local_zodi")
    ray[0].set_gid("outside-ray")
    # A smaller head, so it plainly ends on the neighbor and not on p.
    ray[0].set_mutation_scale(0.5 * ray[0].get_mutation_scale())
    along = np.degrees(np.arctan2(land[1] - out_y, land[0] - ox))
    brk_x = 3.6
    ex.scale_break(
        ax,
        (brk_x, out_y + (brk_x - ox) * np.tan(np.radians(along))),
        cast,
        along_deg=along,
    )

    p_lims = _where_hardware(ax, cast)
    x_base = _where_profile(ax, cast, land[1], p_lims, gid="profile-outside")
    _label(
        ax,
        (x_base + 3.1, y0),
        r"pixel $p$",
        cast,
        ha="left",
        va="center",
        fontsize=cast.layout.font_pt,
    )
    _label(
        ax,
        (x_base + 3.1, y0 - 1.1),
        "its image\nreaches into $p$",
        cast,
        color=cast["annotation"].color,
        fontstyle="italic",
        ha="left",
        va="top",
    )


def build_where(layout, cast):
    """Relay piece 1: where each factor lives, ending on the face-on detector."""
    if layout.is_slide:
        fig = ex.figure(layout)[0]
        fig.clear()
        grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0])
    else:
        fig = ex.figure(layout, doc_height_in=4.4)[0]
        fig.clear()
        grid = fig.add_gridspec(2, 2, width_ratios=[2.3, 1.0])
    ax_p = fig.add_subplot(grid[0, 0])
    ax_e = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[:, 1])
    draw_where_point(ax_p, cast)
    draw_where_extended(ax_e, cast)
    draw_detector(ax_d, cast)
    # This figure opens the relay, so the face-on title names the pixel and
    # the direction instead of the integral it stands for in the overview.
    ax_d.set_title(r"face-on: $P_p(\lambda,\boldsymbol{\theta})$ in $p$", loc="left")
    ax_p.set_title(
        r"point source: $A\,\Phi_\lambda\,T_{\rm opt}\,"
        r"P_p(\lambda,\boldsymbol{\theta})\,q_p$ in $p$",
        loc="left",
    )
    ax_e.set_title(
        r"extended source: $\int I_\lambda P_p(\lambda,\boldsymbol{\theta})\,"
        r"d\Omega$ replaces $\Phi_\lambda P_p$",
        loc="left",
    )
    ex.badge(ax_e, cast, STATUS, loc="lower right")
    if layout.is_slide:
        fig.suptitle("Where each factor of a pixel's count lives")
    return _plain_text(fig, cast)


CAPTION_OVERVIEW = (
    "What an aperture and a pixel collect ({ref}`radiometry-response-ownership`, "
    "{ref}`radiometry-pixel-brightness`). Top, side view with light traveling to the "
    "right: a point source with photon flux density $\\Phi_\\lambda$ at the aperture and "
    "an extended source with brightness $I_\\lambda$ feed an aperture of area $A$; the "
    "optics transmit $T_{\\rm opt}(\\lambda)$; the image of the point source spreads over "
    "several pixels, and $P_p(\\lambda)$ is the share that lands in pixel $p$; the "
    "detector converts photons to electrons with QE $q_p(\\lambda)$. For the extended "
    "source, $\\int I_\\lambda P_p\\,d\\Omega$ replaces $\\Phi_\\lambda P_p$; the dashed "
    "sky cell $\\Omega_p$ is the idealized footprint of pixel $p$, and the spread image "
    "also brings light from outside it. The tags under each element state where a loss "
    "is assigned, so that no loss enters two factors. Bottom, the three integrals that "
    "turn densities into one pixel's expected count $\\mu_{e,p}$: over solid angle, where "
    "the face-on detector shows an illustrative Airy image at one wavelength on a "
    "logarithmic scale; over wavelength, where $T_{\\rm opt}$, $P_p$ and $q_p$ multiply "
    "$\\Phi_\\lambda$ at each wavelength before the band from $\\lambda_b^-$ to "
    "$\\lambda_b^+$ is integrated; and over the live interval $t_{\\rm live}$, while the "
    "pixel is exposing rather than being read out, with constant rates assumed. Curves "
    "are normalized and show shapes only. The picture illustrates the chapter's "
    "expected-count integral; it is an original schematic, not an instrument "
    "prescription, and the ownership assignments are the chapter's proposed ledger."
)
ALT_OVERVIEW = (
    "Top: a side-view diagram. At left, a hatched green extended source with a yellow star "
    "in front of it and a dashed gray square marking a pixel's sky cell. Two yellow rays "
    "leave the star, pass scale breaks, and continue as parallel starlight to a vertical "
    "aperture bar labeled area A, where dotted gray lines from the sky cell meet at an "
    "angle labeled Omega p. A lens and an optics plate labeled T opt of lambda focus the "
    "rays onto a column of detector pixels labeled QE q p of lambda; a gray spread profile "
    "beside the column shades the part falling in the outlined pixel p, labeled P p of "
    "lambda. At right, the photon rate arriving in p from the point source, a note that "
    "the integral of I lambda P p over solid angle replaces Phi lambda P p for the "
    "extended source, and a data arrow labeled times q p, integral over wavelength and "
    "integral over live time leading to the electron count mu e p. Italic tags under the "
    "aperture, optics and detector say which factor holds each loss. Bottom: a face-on 7 "
    "by 7 pixel window with a spread point-source image at one wavelength and pixel p "
    "outlined; normalized curves of Phi lambda, T opt, P p and q p against wavelength "
    "with their product shaded between the band edges; and expected electrons in p "
    "rising linearly during a shaded live interval labeled t live."
)

CAPTION_WAVELENGTH = (
    "The spatial response and the spectral integral for pixel $p$ "
    "({ref}`radiometry-response-ownership`, {ref}`radiometry-spectral-covariance`). Left, "
    "the face-on detector at one wavelength: an illustrative Airy image of a point source "
    "spreads over neighboring pixels, and $P_p$ is the share in the outlined pixel. "
    "Right, the photon density $\\Phi_\\lambda$, optical transmission "
    "$T_{\\rm opt}(\\lambda)$, spatial response $P_p(\\lambda)$ (the share falls as the "
    "image grows with wavelength) and QE $q_p(\\lambda)$ multiply at each wavelength, and "
    "only then is the band from $\\lambda_b^-$ to $\\lambda_b^+$ integrated (shaded). "
    "Curves are normalized and show shapes only. Schematic of the chapter's "
    "expected-count integral, not an instrument prescription."
)
ALT_WAVELENGTH = (
    "Left: a face-on 7 by 7 pixel window with a spread point-source image at one "
    "wavelength on a log color scale and the center pixel p outlined. Right: normalized "
    "curves of photon density, optical transmission, dotted P p and dashed QE against "
    "wavelength, with their product, labeled Phi lambda T opt P p q p, shaded between "
    "the band edges lambda b minus and lambda b plus and labeled integral d lambda."
)

CAPTION_ELECTRONS = (
    "From the spectral integral to electrons in pixel $p$ "
    "({ref}`radiometry-response-ownership`, {ref}`radiometry-detector-counts`). Left, the "
    "integrand with $T_{\\rm opt}$, $P_p$ and $q_p$ applied at each wavelength, "
    "integrated over the band, gives an electron rate. Right, that band integral is the "
    "slope of the expected count, which accumulates only during the live interval "
    "$t_{\\rm live}$, while the pixel is exposing rather than being read out, reaching "
    "$\\mu_{e,p}$. Constant rates are assumed; this is an expected count, not a noisy "
    "readout, and dark current, clock-induced charge and read noise are not drawn. "
    "Schematic, not to scale."
)
ALT_ELECTRONS = (
    "Left: normalized curves of photon density, optical transmission, dotted P p and "
    "dashed QE against wavelength with their product shaded between the band edges. "
    "Right: expected electrons in pixel p against time, flat at zero, rising linearly "
    "across a shaded live interval labeled t live and marked by a dashed integration "
    "arrow, then flat at the end value mu e p; the slope is labeled as the band "
    "integral at left, in electrons per second."
)

CAPTION_REFERENCE = (
    "The reference experiment drawn with its actual parameters "
    "({ref}`photon-electron-reference-experiment`). A synthetic source whose photon "
    "spectral density $\\Phi_{\\lambda,\\mathrm{nm}}$ equals that of 1 Jy at 700 nm is "
    "held constant over a 1 nm rectangular interval; 1 m$^2$ collects it for 1 s, and the "
    "photons are split equally over four pixels by construction, not by a point-spread "
    "function. With constant QE $q$ and no dark current, clock-induced charge or read "
    "noise, the summed expected electron count, and numerically the summed Poisson "
    "variance in electron$^2$, is $q$ times the photon count, shown for the three tested "
    "values of $q$. The density comes from the exact SI value of the Planck constant "
    "({ref}`BIPM 2019, Sec. 2.2 and Table 1 <source-bipm2019>`) and the definition of the "
    "jansky; numbers are rounded to two decimals. This is a code-verification fixture "
    "against an analytic anchor, not validation, and it covers no real spectrum, optical "
    "loss, coronagraph or stochastic readout."
)
ALT_REFERENCE = (
    "Left: a rectangular spectrum 1 nm wide centered at 700 nm, labeled with photon "
    "density Phi lambda nm equal to 21,559.86 photon per second per square meter per "
    "nanometer, 1 Jy at 700 nm. Right: an aperture labeled A equals 1 square meter and t "
    "equals 1 second; times A times delta lambda gives 21,559.86 photon per second, sent "
    "to a 2 by 2 pixel grid; times t over 4 gives each pixel 5,389.96 photon in 1 second, "
    "an equal split by construction, not a point-spread function. A data arrow leads to "
    "a column titled expected electrons: summed electrons equal the summed Poisson "
    "variance numerically, 0.00 for q 0.0, 10,779.93 for q 0.5 and 21,559.86 for q 1.0, "
    "with a note that there is no dark current, clock-induced charge or read noise."
)

CAPTION_WHERE = (
    "Where each factor of one pixel's count lives "
    "({ref}`radiometry-pixel-brightness`, {ref}`radiometry-response-ownership`). Side "
    "views with light traveling to the right, one row per kind of source. Top, a point "
    "source at direction $\\boldsymbol{\\theta}$ with photon flux density "
    "$\\Phi_\\lambda$ sends parallel starlight into an aperture of area $A$; the optics "
    "transmit $T_{\\rm opt}(\\lambda)$; the image spreads over several pixels, and "
    "$P_p(\\lambda,\\boldsymbol{\\theta})$ is the share in pixel $p$ of a point source "
    "at direction $\\boldsymbol{\\theta}$; the detector converts photons to electrons "
    "with QE $q_p(\\lambda)$, so the electron rate per unit wavelength in $p$ is "
    "$A\\Phi_\\lambda T_{\\rm opt}P_p(\\lambda,\\boldsymbol{\\theta})q_p$. Bottom, an "
    "extended source with brightness $I_\\lambda(\\boldsymbol{\\theta})$. Straight lines "
    "through the lens center carry the dashed sky cell onto pixel $p$; the shaded cone "
    "with its vertex at the lens center is the solid angle $\\Omega_p$ of that cell. "
    "This idealized footprint is the cell of "
    "$\\Phi_{\\lambda,p}=\\int_{\\Omega_p}I_\\lambda\\,d\\Omega$. A direction one "
    "pixel pitch from the cell center, half a pitch beyond its edge, has its image "
    "centered on the neighboring pixel, on the opposite side of the axis because the "
    "image is inverted, and the spread of that image still reaches into $p$; light from "
    "inside the cell likewise spreads out of $p$. The pixel therefore receives "
    "$\\int I_\\lambda P_p(\\lambda,\\boldsymbol{\\theta})\\,d\\Omega$ over all "
    "directions, which replaces $\\Phi_\\lambda P_p$ in the expected-count integral. The "
    "part of each profile beside the pixel column that falls inside $p$ is shaded more "
    "heavily. Right, pixel $p$ face-on at one wavelength: an illustrative Airy image of "
    "an unobscured circular aperture, integrated over pixels one $\\lambda/D$ wide and "
    "shown on a logarithmic scale; the profiles at left are cuts through the same Airy "
    "pattern before pixel integration. The next figure, "
    "{ref}`the spectral integral for pixel p <fig-explainer-d03-collection-wavelength>`, "
    "opens with this face-on panel as its left panel. An original schematic of the "
    "chapter's expected-count integral, not to scale and not an instrument prescription."
)
ALT_WHERE = (
    "Two side-view rows and a face-on panel. Top row, titled point source, A Phi "
    "lambda T opt P p of lambda and theta q p in p: a yellow star labeled point source Phi lambda at theta, in photon per "
    "second per square meter per nanometer, sends two parallel yellow rays, cut by scale "
    "breaks and labeled parallel starlight, to a vertical aperture bar labeled aperture "
    "A. A lens focuses them through a plate labeled optics T opt of lambda onto a column "
    "of seven detector pixels labeled detector QE q p of lambda, whose center pixel p is "
    "outlined. A gray profile beside the column peaks at pixel p, and its part inside p "
    "is shaded more heavily; the label reads image spread, share in p of a source at theta, P p of lambda and theta. "
    "Bottom row, titled extended source, integral of I lambda P p of lambda and theta d Omega replaces Phi "
    "lambda P p: a dotted green patch labeled extended source I lambda, in photon per "
    "second per square meter per nanometer per steradian, with a dashed sky cell of p, "
    "marked idealized. A lightly shaded cone labeled Omega p narrows from the cell to "
    "its vertex at the lens center, where an arc spans it; dotted gray lines from the "
    "cell edges cross at that vertex and end on the edges of pixel p. A green dot "
    "labeled outside Omega p sits above the cell; its green ray, with a small head, passes "
    "through the lens center and ends on the pixel just below p. The gray profile beside the column now "
    "peaks on that neighboring pixel, and its tail inside p is shaded more heavily, labeled "
    "its image reaches into p. A badge reads schematic, not to scale. Right: a face-on 7 "
    "by 7 pixel window titled face-on, P p of lambda and theta in p, with a spread point-source image "
    "at one wavelength on a log color scale and the center pixel p outlined."
)

FIGURES = [
    ex.FigureSpec(
        slug="d03-radiometric-collection",
        build=build_overview,
        caption=CAPTION_OVERVIEW,
        alt=ALT_OVERVIEW,
        status=STATUS,
        params={
            "psf": "unobscured Airy, 1 lambda/D per pixel, 7 by 7 window",
            "curves": "illustrative normalized shapes",
        },
    ),
    ex.FigureSpec(
        slug="d03-collection-where",
        build=build_where,
        caption=CAPTION_WHERE,
        alt=ALT_WHERE,
        status=STATUS,
        params={
            "psf": "unobscured Airy, 1 lambda/D per pixel, 7 by 7 window",
            "profile": "cut through the same Airy image, one wavelength",
            "cell_mapping": "chief ray through the lens center, inverted",
        },
    ),
    ex.FigureSpec(
        slug="d03-collection-wavelength",
        build=build_wavelength,
        caption=CAPTION_WAVELENGTH,
        alt=ALT_WAVELENGTH,
        status=STATUS,
    ),
    ex.FigureSpec(
        slug="d03-collection-electrons",
        build=build_electrons,
        caption=CAPTION_ELECTRONS,
        alt=ALT_ELECTRONS,
        status=STATUS,
    ),
    ex.FigureSpec(
        slug="d03-four-pixel-reference",
        build=build_reference,
        caption=CAPTION_REFERENCE,
        alt=ALT_REFERENCE,
        status=STATUS_REFERENCE,
        params=dict(REFERENCE),
    ),
]
ANIMATIONS = []
