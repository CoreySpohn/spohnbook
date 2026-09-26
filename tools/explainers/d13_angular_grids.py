"""Fixed versus native angular grids.

A small-multiple strip, two rows by three wavelengths. One point source at a
fixed sky angle is propagated through the disk aperture of the optical-planes
explainer with physicaloptix ``Fraunhofer`` at each wavelength and sampled on
two kinds of focal grid:

- the fixed angular grid, a reference-wavelength grid that stores
  ``u_ref = alpha D / lambda_ref`` (physicaloptix ``reference_wavelength_nm``
  path, one chromatic field), so its angular spacing is the same at every
  wavelength;
- the native grid, with fixed native spacing ``du`` in ``u_lambda = alpha D /
  lambda``, so its angular spacing ``du lambda / D`` grows with wavelength.

On the fixed grid the source stays put and its first dark ring grows; on the
native grid the ring keeps its size and the source moves toward the axis.
The comparison is across non-adjacent wavelengths, so it is a strip, not an
animation.
"""

import functools
import math

import eyepiece as ep
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_lambda_d, lambda_d_to_arcsec
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
from matplotlib.text import Text
from scipy.special import jn_zeros

from explainers import _common as ex
from explainers import d04_optical_planes as d04

PARAMS = {
    # Evaluated wavelengths and the reference wavelength of the fixed grid.
    "wavelengths_nm": (500.0, 750.0, 1000.0),
    "reference_wavelength_nm": 500.0,
    # Both focal grids: the same stored spacing, du = du_ref, and size.
    "du": 0.5,
    "nfoc": 32,
    # Aperture diameter D of the disk pupil (outer and inscribed coincide).
    "diameter_m": 6.0,
    # The sky source, on the +x axis at u_ref = alpha D / lambda_ref.
    "source_u_ref": 4.0,
    # Display: fraction of the source power per pixel, one shared log norm.
    "power_floor": 1.0e-3,
    "power_ceiling": 0.2,
    "view_x": (-3.0, 7.0),
    "view_y": (-5.0, 5.0),
}

ROW_KEYS = ("fixed", "native")
# Image overlays sit on the viridis intensity map in both modes, so they use
# one light color rather than the theme text color.
OVERLAY = "#f2f2f2"


# The propagated example


def first_zero_lod():
    """Radius of the first dark ring of a clear disk, in lambda/D: j_1,1 / pi."""
    return float(jn_zeros(1, 1)[0] / math.pi)


def source_alpha_arcsec():
    """The source angle from the axis, in arcsec."""
    p = PARAMS
    return float(
        lambda_d_to_arcsec(
            p["source_u_ref"], p["reference_wavelength_nm"], p["diameter_m"]
        )
    )


def source_u(row, wavelength_nm):
    """Stored source coordinate on a row's grid at one wavelength."""
    p = PARAMS
    if row == "fixed":
        return p["source_u_ref"]
    return float(
        arcsec_to_lambda_d(source_alpha_arcsec(), wavelength_nm, p["diameter_m"])
    )


def ring_radius(row, wavelength_nm):
    """First dark ring radius in a row's stored coordinate."""
    if row == "native":
        return first_zero_lod()
    return first_zero_lod() * wavelength_nm / PARAMS["reference_wavelength_nm"]


def pixel_mas(row, wavelength_nm):
    """Angular spacing of one stored pixel, in milliarcseconds."""
    p = PARAMS
    lam = p["reference_wavelength_nm"] if row == "fixed" else wavelength_nm
    return 1.0e3 * float(lambda_d_to_arcsec(p["du"], lam, p["diameter_m"]))


@functools.cache
def compute():
    """Propagate the tilted source at every wavelength onto both grids.

    Returns:
        A dict with ``focal_coords`` (stored coordinates of both grids) and
        ``power`` mapping each row key to an array ``(nlam, y, x)`` of the
        fraction of the source power in each pixel (density times the stored
        cell area; the pupil field carries unit power).
    """
    import physicaloptix as po

    p = PARAMS
    q = d04.PARAMS
    lams = np.asarray(p["wavelengths_nm"], dtype=float)
    with jax.enable_x64(True):
        pupil_grid = po.Grid.pupil(q["npup"])
        focal_grid = po.Grid.focal(p["nfoc"], p["du"])
        x = np.asarray(pupil_grid.coords)
        dx = float(pupil_grid.dx)
        aperture = d04.gray_disk(x, dx, q["aperture_radius_d"], q["gray_subsamples"])
        aperture = aperture / math.sqrt(float((aperture**2).sum()) * dx * dx)
        xx, _ = np.meshgrid(x, x)
        # A source at angle alpha tilts the pupil field by the phase
        # 2 pi u_lambda x (x in units of D), which the proposed transform
        # exp(-i 2 pi u x) places at +u_lambda.
        tilts = [source_u("native", lam) for lam in lams]
        pupil = np.stack([aperture * np.exp(2j * np.pi * u * xx) for u in tilts])
        spectrum = po.Spectrum(
            wavelengths_nm=jnp.asarray(lams), weights=jnp.full(lams.size, 1.0)
        )
        chromatic = po.Field(
            data=jnp.asarray(pupil),
            grid=pupil_grid,
            plane=po.PlaneKind.PUPIL,
            spectrum=spectrum,
        )
        fixed = po.Fraunhofer(
            pupil_grid,
            focal_grid,
            reference_wavelength_nm=p["reference_wavelength_nm"],
            min_wavelength_nm=float(lams.min()),
        )(chromatic)
        native_camera = po.Fraunhofer(pupil_grid, focal_grid)
        native = [
            native_camera(
                po.Field(data=jnp.asarray(e), grid=pupil_grid, plane=po.PlaneKind.PUPIL)
            ).data
            for e in pupil
        ]
    area = focal_grid.weights
    return {
        "focal_coords": np.asarray(focal_grid.coords),
        "power": {
            "fixed": np.abs(np.asarray(fixed.data)) ** 2 * area,
            "native": np.abs(np.stack([np.asarray(d) for d in native])) ** 2 * area,
        },
    }


# Text helpers: labels over the image sit on a plain backing box, never on a
# stroked halo.


def _backing(cast):
    return {
        "boxstyle": "round,pad=0.2,rounding_size=0.2",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _plain_text(fig, cast):
    """Replace any stroked halo a shared helper added with the backing box."""
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                text.set_bbox(_backing(cast))
    return fig


# Layout

ROW_HEADERS = {
    "fixed": (
        "Fixed angular grid: the source stays put, the dark ring grows",
        r"stores $u_{\rm ref}=\alpha D/\lambda_{\rm ref}$ with "
        r"$\lambda_{\rm ref}=500$ nm; angular spacing "
        r"$du_{\rm ref}\,\lambda_{\rm ref}/D$ at every $\lambda$",
    ),
    "native": (
        "Native grid: the dark ring keeps its size, the source moves",
        r"stores $u_\lambda=\alpha D/\lambda$ with fixed native spacing $du$; "
        r"angular spacing $du\,\lambda/D$ grows with $\lambda$",
    ),
}
SLIDE_HEADERS = {
    "fixed": (
        "Fixed angular grid",
        "the source stays put;\nthe dark ring grows",
        r"$u_{\rm ref}=\alpha D/\lambda_{\rm ref}$",
    ),
    "native": (
        "Native grid",
        "the dark ring keeps its size;\nthe source moves",
        r"$u_\lambda=\alpha D/\lambda$",
    ),
}
ROW_XLABELS = {
    "fixed": r"$u_{\rm ref}$ along $x$  [$\lambda_{\rm ref}/D$]",
    "native": r"$u_\lambda$ along $x$  [$\lambda/D$]",
}
ROW_YLABELS = {
    "fixed": r"$y$  [$\lambda_{\rm ref}/D$]",
    "native": r"$y$  [$\lambda/D$]",
}
ROW_SYMBOL = {"fixed": r"u_{\rm ref}", "native": r"u_\lambda"}
ROW_UNIT = {"fixed": r"\lambda_{\rm ref}/D", "native": r"\lambda/D"}


def panel_label(row, wavelength_nm, *, short=False):
    """The numbers printed in a panel's corner."""
    pixel = f"pixel {pixel_mas(row, wavelength_nm):.1f} mas"
    if short:
        return pixel
    u = source_u(row, wavelength_nm)
    return (
        f"{pixel}\n"
        rf"source ${ROW_SYMBOL[row]}={u:.2f}$" + "\n"
        f"dark ring radius ${ring_radius(row, wavelength_nm):.2f}\\,{ROW_UNIT[row]}$"
    )


def _geometry(layout):
    """Panel, header and colorbar placement, in inches from the bottom left."""
    font, small = layout.font_pt / 72.0, layout.small_pt / 72.0
    if layout.is_slide:
        size, gap = 2.5, 0.2
        header_w = 3.7
        x0 = 0.35 + header_w + 0.9
        top = layout.height_in - 0.95
        title = 1.6 * font
        xlab = 1.45 * small + 1.45 * font + 0.06
        row_gap = 0.2
        key, stamp = 0.5, 0.25
        panel_top = top - title
        rows = {"fixed": panel_top - size}
        rows["native"] = rows["fixed"] - xlab - row_gap - size
        height = layout.height_in
        key_y = rows["native"] - xlab - 0.5 * key
    else:
        gap = 0.1
        x0 = 0.55
        cbar_room = 0.1 + 0.12 + 0.62
        size = (layout.width_in - x0 - 2 * gap - cbar_room - 0.05) / 3.0
        header = 1.35 * font + 1.35 * small + 1.5 * font + 0.04
        xlab = 1.4 * small + 1.4 * font + 0.04
        key, stamp = 0.3, 0.2
        height = 0.05 + 2 * (header + size + xlab) + key + stamp
        rows = {}
        y = height - 0.05
        for row in ROW_KEYS:
            y -= header + size
            rows[row] = y
            y -= xlab
        key_y = y - 0.5 * key
    return {
        "size": size,
        "gap": gap,
        "x0": x0,
        "rows": rows,
        "key_y": key_y,
        "height": height,
    }


def _panel_axes(fig, layout, g):
    """One square axes per row and wavelength."""
    width, height = layout.width_in, g["height"]
    size, gap, x0 = g["size"], g["gap"], g["x0"]
    return {
        row: [
            fig.add_axes(
                [
                    (x0 + k * (size + gap)) / width,
                    g["rows"][row] / height,
                    size / width,
                    size / height,
                ]
            )
            for k in range(3)
        ]
        for row in ROW_KEYS
    }


def _row_headers(overlay, layout, cast, g):
    """Name each grid: above its row on a page, beside it on a slide."""
    size = g["size"]
    for row in ROW_KEYS:
        panel_y = g["rows"][row]
        if layout.is_slide:
            name, lesson, definition = SLIDE_HEADERS[row]
            x = 0.35
            mid = panel_y + 0.5 * size
            overlay.text(
                x,
                mid + 0.75,
                name,
                ha="left",
                va="bottom",
                fontsize=layout.title_pt,
                fontweight="bold",
                color=cast.text,
            )
            overlay.text(
                x,
                mid + 0.6,
                lesson,
                ha="left",
                va="top",
                fontsize=layout.font_pt,
                color=cast.text,
                linespacing=1.2,
            )
            overlay.text(
                x,
                mid - 0.75,
                definition,
                ha="left",
                va="top",
                fontsize=layout.font_pt,
                color=cast.text,
            )
            continue
        head, detail = ROW_HEADERS[row]
        font, small = layout.font_pt / 72.0, layout.small_pt / 72.0
        top = panel_y + size + 1.5 * font + 1.35 * small + 1.35 * font
        overlay.text(
            0.08,
            top,
            head,
            ha="left",
            va="top",
            fontsize=layout.font_pt,
            fontweight="bold",
            color=cast.text,
        )
        overlay.text(
            0.08,
            top - 1.35 * font,
            detail,
            ha="left",
            va="top",
            fontsize=layout.small_pt,
            color=cast.text,
        )


def _key(overlay, layout, cast, g, lw):
    """What each overlay mark is, in one line under the strip."""
    ky = g["key_y"]
    x = 0.08 if not layout.is_slide else g["x0"] - 1.5
    small = layout.small_pt
    em = small / 72.0
    items = (
        ("source", "sky source at one fixed angle"),
        ("ring", "first dark ring (analytic zero)"),
        ("axis", "optical axis"),
    )
    for kind, words in items:
        if kind == "ring":
            overlay.plot(
                [x, x + 2.0 * em], [ky, ky], color=cast.text, lw=lw, ls=(0, (3, 2))
            )
        elif kind == "source":
            overlay.plot(
                [x + em],
                [ky],
                marker="o",
                ls="none",
                ms=1.1 * layout.marker_pt,
                mfc="none",
                mec=cast["planet"].color,
                mew=lw,
            )
        else:
            overlay.plot(
                [x + em],
                [ky],
                marker="+",
                ls="none",
                ms=layout.marker_pt,
                color=cast.text,
                mew=lw,
            )
        label = overlay.text(
            x + 2.6 * em,
            ky,
            words,
            va="center",
            fontsize=small,
            color=cast.text,
        )
        x += 2.6 * em + 0.55 * em * len(words) + 2.5 * em
    return label


def build_strip(layout, cast):
    """Two rows (fixed, native grid) by three wavelengths."""
    p = PARAMS
    data = compute()
    lams = p["wavelengths_nm"]
    extent = ep.extent_lod(data["focal_coords"])
    g = _geometry(layout)
    width, height = layout.width_in, g["height"]
    fig, ax = ex.figure(layout, doc_height_in=height)
    ax.remove()
    fig.set_layout_engine("none")
    size, gap, x0 = g["size"], g["gap"], g["x0"]
    small = layout.small_pt

    overlay = fig.add_axes([0, 0, 1, 1], zorder=-1)
    overlay.set(xlim=(0, width), ylim=(0, height))
    overlay.axis("off")

    axes = _panel_axes(fig, layout, g)
    # All six panels go to one compare_row call, so they share one norm.
    flat = [ax for row in ROW_KEYS for ax in axes[row]]
    images = [data["power"][row][k] for row in ROW_KEYS for k in range(3)]
    result = ep.compare_row(
        images,
        axes=flat,
        norm="log",
        floor=p["power_floor"],
        extent=extent,
        cmap=ex.image_cmap("intensity"),
        vmin=p["power_floor"],
        vmax=p["power_ceiling"],
    )
    # One colorbar for both rows, beside the right column. The unreleased
    # eyepiece compare_grid with a cax argument would replace this re-hang.
    result.artists["cbar"].remove()
    top_y = g["rows"]["fixed"] + size
    bottom_y = g["rows"]["native"]
    cax = fig.add_axes(
        [
            (x0 + 3 * size + 2 * gap + 0.1) / width,
            bottom_y / height,
            (0.2 if layout.is_slide else 0.12) / width,
            (top_y - bottom_y) / height,
        ]
    )
    cbar = fig.colorbar(result.artists["image"][0], cax=cax)
    cbar.set_label("fraction of the source power per pixel")

    lw = 0.9 * cast.layout.lw
    for row in ROW_KEYS:
        for k, lam in enumerate(lams):
            ax = axes[row][k]
            ax.set_xlim(*p["view_x"])
            ax.set_ylim(*p["view_y"])
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks([-4, -2, 0, 2, 4])
            if row == "fixed" or not layout.is_slide:
                ax.set_title(
                    rf"$\lambda$ = {lam:.0f} nm", fontsize=layout.font_pt, pad=3
                )
            u = source_u(row, lam)
            ax.add_patch(
                Circle(
                    (u, 0.0),
                    ring_radius(row, lam),
                    fill=False,
                    ec=OVERLAY,
                    lw=lw,
                    ls=(0, (3, 2)),
                    zorder=4,
                    gid=f"ring-{row}-{k}",
                )
            )
            ax.plot(
                [u],
                [0.0],
                marker="o",
                ls="none",
                ms=1.3 * layout.marker_pt,
                mfc="none",
                mec=cast["planet"].color,
                mew=1.1 * lw,
                zorder=5,
                gid=f"source-{row}-{k}",
            )
            ax.plot(
                [0.0],
                [0.0],
                marker="+",
                ls="none",
                ms=layout.marker_pt,
                color=OVERLAY,
                mew=lw,
                zorder=5,
            )
            label = ax.text(
                0.04,
                0.96,
                panel_label(row, lam, short=layout.is_slide),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=small,
                color=cast.text,
                linespacing=1.15,
                zorder=6,
                gid=f"label-{row}-{k}",
            )
            label.set_bbox(_backing(cast))
            if k == 0:
                ax.set_ylabel(ROW_YLABELS[row], labelpad=1)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(ROW_XLABELS[row] if k == 1 else "", labelpad=1)
    _row_headers(overlay, layout, cast, g)
    _key(overlay, layout, cast, g, lw)
    ex.badge(overlay, cast, "simulated, noiseless", loc="upper right")
    if layout.is_slide:
        fig.suptitle(
            "One sky source, three wavelengths, two kinds of focal grid",
            x=0.02,
            ha="left",
            y=0.965,
            fontsize=layout.title_pt,
        )
    return _plain_text(fig, cast)


# Captions and alternative text

STRIP_CAPTION = (
    "One point source at a fixed sky angle, 68.8 mas from the optical axis "
    "along +x, imaged through a clear circular aperture of diameter D = 6 m at "
    "500, 750 and 1000 nm ({ref}`optics-angular-grids`). Top row, the fixed "
    "angular grid: a reference-wavelength grid that stores "
    "u_ref = alpha D/lambda_ref with lambda_ref = 500 nm and du_ref = 0.5, so "
    "every pixel spans 8.6 mas at every wavelength. The source stays at "
    "u_ref = 4 while the diffraction pattern grows: the first dark ring has "
    "radius 1.22 lambda/D ({ref}`Hecht 2017, Sec. 10.2.5, eq. 10.58 "
    "<source-hecht2017>`), which is 1.22, 1.83 and 2.44 in u_ref. The dashed "
    "circle is that analytic first zero, drawn whether or not the sampling "
    "resolves a dark pixel ring under it, as at 500 nm. Bottom row, "
    "the native grid: fixed native spacing du = 0.5 in u_lambda = alpha D/lambda, "
    "so a pixel spans 8.6, 12.9 and 17.2 mas. The ring keeps its radius of 1.22 "
    "while the same source moves toward the axis, to u_lambda = 4, 2.67 and 2. "
    "Converting the stored top-row scale again with each evaluated wavelength "
    "would magnify the pattern a second time. The aperture is a clear disk, so "
    "its outer and inscribed diameters coincide; for any other aperture the "
    "grid record must name which diameter D is. Each panel shows the fraction "
    "of the source power in each pixel, center-sampled (density times the "
    "stored cell area; {ref}`optics-pixel-measure`), as raw pixels on one "
    "shared log scale from 1e-3 to 0.2: the top-row source dims with wavelength because the same "
    "power spreads over more fixed pixels. Simulated and noiseless with the "
    "physicaloptix Fraunhofer transform, its reference-wavelength mode for the "
    "top row and its native mode for the bottom row, on a 64 by 64 gray-pixel "
    "disk pupil and 32 by 32 focal grids. The arrays are drawn with x right and "
    "y up; how they map to the sky and to the detector belongs to the pending "
    "{ref}`image coordinates and PSFlet origin decision "
    "<decision-image-coordinates-and-psflet-origin>`. The grid definitions are "
    "identities; the numbers are one example."
)
STRIP_ALT = (
    "A two-row by three-column grid of simulated log-scaled images of one point "
    "source, with columns at 500, 750 and 1000 nanometers. Each panel marks the "
    "optical axis with a plus at the origin, the source with an open cyan "
    "circle on the positive x axis, and the first dark ring of its diffraction "
    "pattern with a dashed circle, and a corner box gives the pixel size in "
    "milliarcseconds, the source coordinate and the dark ring radius. Top row, fixed "
    "angular grid: the source stays at u_ref = 4 in every column while the "
    "ring radius grows from 1.22 to 1.83 to 2.44 and the peak dims; every pixel "
    "is 8.6 mas. Bottom row, native grid: the ring radius stays 1.22 and the "
    "pattern looks the same in every column, while the source moves toward the "
    "axis from 4 to 2.67 to 2, and the pixel size grows from 8.6 to 12.9 to "
    "17.2 mas. One colorbar, the fraction of the source power per pixel, is "
    "shared by all six panels."
)


FIGURES = [
    ex.FigureSpec(
        slug="d13-angular-grids",
        build=build_strip,
        caption=STRIP_CAPTION,
        alt=STRIP_ALT,
        status="simulated, noiseless",
        params=PARAMS,
    ),
]
ANIMATIONS = []
