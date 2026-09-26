"""Which denominator? Four ratios for one planet.

One off-axis planet PSF beside the stellar leakage map of the same
coronagraph, both in the package's image unit (fraction of the star's
incident photons per pixel), and a table of four denominators with the ratio
each gives for that one planet:

- host-star flux, the textbook host-relative ratio, defined before the
  coronagraph;
- the unocculted PSF peak, a peak-referenced pixel ratio;
- the total incident stellar flux, the package's own pixel unit;
- an aperture sum, the package's raw-contrast normalization: a sum inside
  the photometric aperture divided by the same sum for a source of unit
  host-relative ratio at that position (the core throughput).

Every number comes from the yippy ``EqxCoronagraph`` built on the
``eac1_optimal_order_6_1d`` yield input package: ``throughput``,
``core_area`` and ``raw_contrast`` for the tables, ``create_psf`` for the
planet image and its peak, and ``stellar_intens`` for the leakage map. The
planet sits on a tabulated offset, so its image is the package's own PSF with
no interpolation.
"""

import functools
import math

import eyepiece as ep
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import lambda_d_to_arcsec
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
from matplotlib.text import Text

from explainers import _common as ex

PARAMS = {
    "package": "eac1_optimal_order_6_1d",
    # The design the package header names: circumscribed diameter and
    # central wavelength, used only to print the planet's angle in mas.
    "diameter_m": 7.2,
    "wavelength_nm": 1000.0,
    # Example planet: host-relative ratio at one reference plane.
    "planet_ratio": 1.0e-10,
    # Planet separation, snapped to the nearest tabulated PSF offset.
    "planet_sep_lod": 3.22,
    # A tabulated offset where the core throughput has leveled off; its PSF
    # stands in for the unocculted PSF, which the package does not carry.
    "reference_sep_lod": 20.5,
    # Star diameter, snapped to the nearest tabulated leakage map (the Sun
    # at 10 pc is 0.032 lambda/D at 1000 nm on 7.2 m).
    "star_diam_lod": 0.032,
    # Photometric aperture radius of the package's performance tables.
    "aperture_radius_lod": 0.7,
    # Display window, half width in pixels about the star, and log norm.
    "half_window_px": 24,
    "norm_vmin": 1.0e-19,
    "norm_vmax": 1.0e-11,
}

# Overlays sit on the viridis intensity map in both modes, so they use one
# light color rather than the theme text color.
OVERLAY = "#f2f2f2"


# The tables


@functools.cache
def compute():
    """Load the package and evaluate every table the figure prints.

    The package is loaded as the exposure-time tutorial loads it: a
    ``yippy.Coronagraph`` whose performance curves are recomputed for the
    tabulated star diameter nearest ``star_diam_lod`` (the curves at load
    time are for a point-source star, which this design nulls to yippy's
    1e-20 floor), wrapped in an ``EqxCoronagraph``.

    Returns:
        A dict of the tabulated inputs (separations, star diameter, pixel
        scale, obscured fraction), the images (planet PSF and leakage map,
        each the fraction of that source's incident photons per pixel) and
        the scalar table values.
    """
    import yippy

    p = PARAMS
    yippy.logger.setLevel("ERROR")
    with jax.enable_x64(True):
        yip = yippy.Coronagraph(yippy.fetch_yip(p["package"]))
        diams = yip.stellar_intens.diams
        k = int(np.argmin(np.abs(diams.value - p["star_diam_lod"])))
        yip.compute_all_performance_curves(
            stellar_diam=diams[k],
            aperture_radius_lod=p["aperture_radius_lod"],
            save_to_fits=False,
            plot=False,
        )
        coro = yippy.EqxCoronagraph(yippy_coro=yip)
        offsets = np.asarray(yip.offax.x_offsets, dtype=float)
        sep = float(offsets[np.argmin(np.abs(offsets - p["planet_sep_lod"]))])
        ref = float(offsets[np.argmin(np.abs(offsets - p["reference_sep_lod"]))])
        planet = np.asarray(coro.create_psf(sep, 0.0), dtype=float)
        reference = np.asarray(coro.create_psf(ref, 0.0), dtype=float)
        leakage = np.asarray(coro.stellar_intens(float(diams[k].value)), dtype=float)
        tables = {
            "throughput": float(coro.throughput(jnp.asarray(sep))),
            "core_area": float(coro.core_area(jnp.asarray(sep))),
            "raw_contrast": float(coro.raw_contrast(jnp.asarray(sep))),
        }
    return {
        "sep_lod": sep,
        "ref_sep_lod": ref,
        "star_diam_lod": float(diams[k].value),
        "pixel_scale_lod": float(coro.pixel_scale_lod),
        "frac_obscured": float(coro.frac_obscured),
        "planet_psf": planet,
        "leakage": leakage,
        "planet_peak": float(planet.max()),
        "unocculted_peak": float(reference.max()),
        **tables,
    }


def ratios():
    """The four ratios for the example planet, and their ingredients.

    Returns:
        A dict with ``host``, ``peak``, ``incident`` and ``aperture`` (the
        ratio each denominator gives), plus ``aperture_numerator`` (the
        planet's sum inside the aperture, as a fraction of the star's
        incident photons) and ``aperture_pixels`` (the aperture area in
        pixels).
    """
    d = compute()
    c = PARAMS["planet_ratio"]
    planet_peak = c * d["planet_peak"]
    in_aperture = aperture_sum(c * d["planet_psf"])
    return {
        "host": c,
        "peak": planet_peak / d["unocculted_peak"],
        "peak_fraction": d["planet_peak"] / d["unocculted_peak"],
        "incident": planet_peak,
        "incident_per_lod2": planet_peak / d["pixel_scale_lod"] ** 2,
        "aperture": in_aperture / d["throughput"],
        "aperture_numerator": in_aperture,
        "aperture_pixels": d["core_area"] / d["pixel_scale_lod"] ** 2,
    }


def aperture_sum(image, sub=8):
    """Sum of an image inside the planet's aperture, by pixel area fraction.

    The aperture is the circle of area ``core_area`` centered on the planet's
    nominal position; the star sits on the corner between the four central
    pixels. Each pixel is weighted by the fraction of its area inside the
    circle, estimated on a ``sub`` by ``sub`` grid of sub-pixel centers.
    """
    d = compute()
    pix = d["pixel_scale_lod"]
    radius_px = math.sqrt(d["core_area"] / math.pi) / pix
    n = image.shape[0]
    cx, cy = (n - 1) / 2.0 + d["sep_lod"] / pix, (n - 1) / 2.0
    yy, xx = np.mgrid[:n, : image.shape[1]]
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    weight = np.zeros(image.shape)
    for a in offsets:
        for b in offsets:
            weight += np.hypot(xx + a - cx, yy + b - cy) <= radius_px
    return float((image * weight).sum() / sub**2)


# The host-relative ratio. The negative spaces close the side bearings of the
# small star glyph, which otherwise leave a gap before the next character.
PHI_RATIO = r"\Phi_p/\Phi_{\!\star\!}"


def sci(value, digits=2):
    """Mathtext scientific notation, for example ``9.4 x 10^-11``."""
    mant, expo = f"{value:.{digits - 1}e}".split("e")
    return rf"${mant}\times10^{{{int(expo)}}}$"


def planet_mas():
    """The planet's angular separation in milliarcseconds."""
    p = PARAMS
    return 1.0e3 * float(
        lambda_d_to_arcsec(compute()["sep_lod"], p["wavelength_nm"], p["diameter_m"])
    )


# Text helpers: labels over the image sit on a plain backing box, never on a
# stroked halo.


def _backing(cast, alpha=0.85):
    return {
        "boxstyle": "round,pad=0.25,rounding_size=0.2",
        "facecolor": to_rgba(cast.background, alpha),
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


def _key(ax, xy, number, cast, *, size, transform=None):
    """A numbered key that ties a table row to its place on the images."""
    kw = {"transform": transform} if transform is not None else {}
    return ax.text(
        *xy,
        str(number),
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
        color=cast.text,
        bbox={
            "boxstyle": "circle,pad=0.18",
            "facecolor": to_rgba(cast.background, 0.9),
            "edgecolor": cast.text,
            "lw": 0.6 * cast.layout.lw,
        },
        zorder=7,
        gid=f"key-{number}",
        **kw,
    )


def _keyed_note(ax, xy, number, text, cast, *, size, transform):
    """A numbered key followed by a note on a plain backing box."""
    _key(ax, xy, number, cast, size=size, transform=transform)
    note = ax.annotate(
        text,
        xy,
        xycoords=transform,
        xytext=(1.3 * size, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=size,
        color=cast.text,
        linespacing=1.15,
        zorder=6,
    )
    note.set_bbox(_backing(cast))
    return note


# The table


def table_rows(layout):
    """Rows of the denominator table: (key, denominator, ratio, value).

    The documentation rows show the arithmetic behind each value; the slide
    rows keep only the definition.
    """
    d = compute()
    r = ratios()
    c = PARAMS["planet_ratio"]
    if layout.is_slide:
        return [
            (
                1,
                "Host-star flux",
                "planet flux / host flux, before\nthe coronagraph (textbook)",
                sci(r["host"]),
            ),
            (
                2,
                "Unocculted PSF peak",
                "planet peak pixel / the star's\nunmasked peak pixel ({:.0f}%)".format(
                    100 * r["peak_fraction"]
                ),
                sci(r["peak"]),
            ),
            (
                3,
                "Total incident stellar flux",
                "planet peak pixel (0.25 "
                + r"$\lambda/D$"
                + ") /\nall the star's incident photons",
                sci(r["incident"]) + " in the peak pixel",
            ),
            (
                4,
                "An aperture sum",
                "planet aperture sum / same sum\nfor a source with planet flux\nequal to the star's (yield package)",
                sci(r["aperture"]),
            ),
        ]
    return [
        (
            1,
            "Host-star flux\n(textbook)",
            rf"${PHI_RATIO}$: planet flux over host flux at one reference"
            "\nplane before the coronagraph; not in these images",
            sci(r["host"]),
        ),
        (
            2,
            "Unocculted\nPSF peak",
            "planet peak pixel over the star's unmasked peak pixel at the"
            f"\nplanet's pixel phase: {sci(r['incident'])} / {d['unocculted_peak']:.4f};"
            f"\nthe off-axis peak is {100 * r['peak_fraction']:.0f}% of the unocculted peak",
            sci(r["peak"]),
        ),
        (
            3,
            "Total incident\nstellar flux",
            "planet peak pixel as a fraction of all the star's incident"
            f"\nphotons (the colorbar unit): {sci(c, 1)}"
            + r" $\times$ "
            + f"{d['planet_peak']:.4f}",
            sci(r["incident"]) + "\n" + r"in the peak 0.25 $\lambda/D$ pixel",
        ),
        (
            4,
            "An aperture sum\n(yield package)",
            "planet sum in the aperture over the same sum for a source with"
            "\nplanet flux equal to the star's (the core throughput):"
            f"\n{sci(r['aperture_numerator'])} / {d['throughput']:.2f}",
            sci(r["aperture"]),
        ),
    ]


def _draw_table(overlay, g, layout, cast):
    """Documentation: a four-row table under the images, in inches."""
    small, font = layout.small_pt, layout.font_pt
    x0, x1 = g["table_x0"], g["table_x1"]
    cols = g["table_cols"]
    y = g["table_top"]
    header = ("Denominator", "Numerator / denominator", "Value")
    for x, text in zip(cols[1:], header, strict=True):
        overlay.text(
            x,
            y,
            text,
            ha="left",
            va="top",
            fontsize=small,
            fontweight="bold",
            color=cast.text,
        )
    y -= g["table_header"]
    overlay.plot([x0, x1], [y, y], color=cast["scenery"].color, lw=0.6)
    for key, name, ratio, value in table_rows(layout):
        yc = y - 0.5 * g["table_row"]
        _key(overlay, (cols[0], yc), key, cast, size=small)
        overlay.text(
            cols[1],
            yc,
            name,
            ha="left",
            va="center",
            fontsize=small,
            color=cast.text,
            linespacing=1.15,
        )
        overlay.text(
            cols[2],
            yc,
            ratio,
            ha="left",
            va="center",
            fontsize=small,
            color=cast.text,
            linespacing=1.3,
        )
        overlay.text(
            cols[3],
            yc,
            value,
            ha="left",
            va="center",
            fontsize=font,
            color=cast.text,
            linespacing=1.15,
            gid=f"value-{key}",
        )
        y -= g["table_row"]
        overlay.plot([x0, x1], [y, y], color=cast.neutral(0.25), lw=0.5)
    overlay.text(
        cols[1],
        y - 0.06,
        TAKEAWAY,
        ha="left",
        va="top",
        fontsize=small,
        fontweight="bold",
        color=cast.text,
    )


def _draw_list(overlay, g, layout, cast):
    """Slide: the four denominators as a column beside the images."""
    small, font = layout.small_pt, layout.font_pt
    x = g["list_x"]
    y = g["list_top"]
    overlay.text(
        x,
        y,
        "Four denominators, one planet",
        ha="left",
        va="top",
        fontsize=font,
        fontweight="bold",
        color=cast.text,
    )
    y -= 0.75
    for key, name, ratio, value in table_rows(layout):
        _key(overlay, (x + 0.17, y - 0.17), key, cast, size=small)
        overlay.text(
            x + 0.5,
            y,
            name,
            ha="left",
            va="top",
            fontsize=font,
            color=cast.text,
        )
        overlay.text(
            x + 0.5,
            y - 0.42,
            value,
            ha="left",
            va="top",
            fontsize=font,
            fontweight="bold",
            color=cast.text,
            gid=f"value-{key}",
        )
        overlay.text(
            x + 0.5,
            y - 0.9,
            ratio,
            ha="left",
            va="top",
            fontsize=small,
            color=cast.neutral(0.75),
            linespacing=1.2,
        )
        y -= g["list_row"]


# Layout


def _geometry(layout):
    """Panel, colorbar and table placement in inches."""
    if layout.is_slide:
        size = 4.35
        height = layout.height_in
        panel_y = height - 1.35 - 0.6 - size
        return {
            "size": size,
            "gap": 0.25,
            "x0": 0.95,
            "panel_y": panel_y,
            "height": height,
            "cbar_w": 0.2,
            "list_x": 11.45,
            "list_top": height - 1.15,
            "list_row": 1.6,
        }
    f = layout.font_pt / 10.0
    gap, x0 = 0.12, 0.52
    cbar_w = 0.12
    size = (layout.width_in - x0 - gap - (0.12 + cbar_w + 0.62) - 0.05) / 2.0
    head, xlab = 0.4 * f, 0.44 * f
    table_header, table_row = 0.26 * f, 0.56 * f
    stamp = 0.26
    takeaway = 0.24 * f
    height = (
        0.05 + head + size + xlab + 0.1 + table_header + 4 * table_row + takeaway
    ) + stamp
    panel_y = height - 0.05 - head - size
    table_x0 = 0.1
    return {
        "size": size,
        "gap": gap,
        "x0": x0,
        "panel_y": panel_y,
        "height": height,
        "cbar_w": cbar_w,
        "table_x0": table_x0,
        "table_x1": layout.width_in - 0.1,
        "table_cols": (table_x0 + 0.1, table_x0 + 0.34, 1.62, 5.3),
        "table_top": panel_y - xlab - 0.1,
        "table_header": table_header,
        "table_row": table_row,
    }


def _crop(image, half):
    """Symmetric window about the star, which sits on a pixel corner."""
    n = image.shape[0] // 2
    return image[n - half : n + half, n - half : n + half]


def build_denominators(layout, cast):
    """The planet PSF and leakage map, and the four denominators."""
    p = PARAMS
    d = compute()
    r = ratios()
    g = _geometry(layout)
    slide = layout.is_slide
    width, height = layout.width_in, g["height"]
    fig, ax = ex.figure(layout, doc_height_in=height)
    ax.remove()
    fig.set_layout_engine("none")
    size, gap, x0, py = g["size"], g["gap"], g["x0"], g["panel_y"]
    small, font = layout.small_pt, layout.font_pt
    lw = 0.9 * layout.lw
    pix = d["pixel_scale_lod"]
    half = p["half_window_px"]
    c = p["planet_ratio"]

    overlay = fig.add_axes([0, 0, 1, 1], zorder=-1)
    overlay.set(xlim=(0, width), ylim=(0, height))
    overlay.axis("off")

    axes = [
        fig.add_axes(
            [(x0 + k * (size + gap)) / width, py / height, size / width, size / height]
        )
        for k in range(2)
    ]
    images = [c * _crop(d["planet_psf"], half), _crop(d["leakage"], half)]
    result = ep.compare_row(
        images,
        axes=axes,
        norm="log",
        floor=p["norm_vmin"],
        vmin=p["norm_vmin"],
        vmax=p["norm_vmax"],
        extent=ep.extent_lod_from_pixels(2 * half, pix),
        cmap=ex.image_cmap("intensity"),
    )
    result.artists["cbar"].remove()
    cbar_x = x0 + 2 * size + gap + 0.12
    cax = fig.add_axes(
        [cbar_x / width, py / height, g["cbar_w"] / width, size / height]
    )
    cbar = fig.colorbar(result.artists["image"][0], cax=cax)
    cbar.set_label("fraction of the star's incident photons per pixel", fontsize=small)
    cbar.ax.tick_params(labelsize=small)

    sep = d["sep_lod"]
    radius = math.sqrt(d["core_area"] / math.pi)
    titles = (
        rf"Planet, ${PHI_RATIO}=1\times10^{{{int(math.log10(c))}}}$, "
        rf"at {sep:.2f} $\lambda/D$",
        rf"Stellar leakage, {d['star_diam_lod']:.4f} $\lambda/D$ star",
    )
    if slide:
        titles = (
            rf"Planet at {sep:.2f} $\lambda/D$",
            rf"Leakage of a {d['star_diam_lod']:.4f} $\lambda/D$ star",
        )
    lim = half * pix
    for k, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_title(titles[k], fontsize=font, pad=3)
        ax.set_xlabel(r"$x$ from the star [$\lambda/D$]", labelpad=1)
        if k == 0:
            ax.set_ylabel(r"$y$ [$\lambda/D$]", labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.add_patch(
            Circle(
                (sep, 0.0),
                radius,
                fill=False,
                ec=OVERLAY,
                lw=lw,
                ls=(0, (3, 2)),
                zorder=4,
                gid=f"aperture-{k}",
            )
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
        _key(ax, (sep, -(radius + (1.0 if slide else 0.85))), 4, cast, size=small)

    # Corner notes on plain backing boxes.
    if slide:
        psf_note = (
            rf"aperture $r$ = {p['aperture_radius_lod']:.1f} $\lambda/D$ holds"
            + f"\n{d['throughput']:.2f} of the planet light"
        )
        leak_note = f"same aperture: raw\ncontrast {sci(d['raw_contrast'])}"
    else:
        psf_note = (
            rf"aperture $r$ = {p['aperture_radius_lod']:.1f} $\lambda/D$:"
            + "\n"
            + rf"{d['core_area']:.2f} $(\lambda/D)^2$ = "
            + f"{r['aperture_pixels']:.1f} pixels,\n"
            + f"holds {d['throughput']:.2f} of the planet light"
        )
        leak_note = (
            "same aperture, yield package:\n"
            f"raw contrast {sci(d['raw_contrast'])}\n"
            "(star sum / sum for a source with\nplanet flux equal to the star's)"
        )
    axes[0].text(
        0.03,
        0.97,
        psf_note,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=small,
        color=cast.text,
        linespacing=1.2,
        zorder=6,
    ).set_bbox(_backing(cast))
    note = _keyed_note(
        axes[1],
        (0.06, 0.9 if slide else 0.86),
        4,
        leak_note,
        cast,
        size=small,
        transform=axes[1].transAxes,
    )
    note.set_linespacing(1.2)
    axes[0].text(
        0.03,
        0.03,
        f"+ star, {planet_mas():.0f} mas from the planet"
        + ("" if slide else f" at {p['wavelength_nm']:.0f} nm"),
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=small,
        color=cast.text,
        zorder=6,
    ).set_bbox(_backing(cast))
    _keyed_note(
        axes[1],
        (0.06, 0.09 if slide else 0.08),
        2,
        f"without the mask, the star's\npeak pixel is {d['unocculted_peak']:.4f}",
        cast,
        size=small,
        transform=axes[1].transAxes,
    )
    # Key 3 names the colorbar unit, the star's total incident flux: it sits
    # on the colorbar label.
    if slide:
        key3 = (cbar_x + 0.5 * g["cbar_w"], py + size + 0.32)
    else:
        key3 = (cbar_x + g["cbar_w"] + 0.52, py + size + 0.14)
    _key(
        overlay,
        key3,
        3,
        cast,
        size=small,
    )

    if slide:
        _draw_list(overlay, g, layout, cast)
        fig.suptitle(
            "Which denominator? One planet, four ratios",
            x=0.02,
            ha="left",
            y=0.975,
            fontsize=layout.title_pt,
        )
        ex.badge(overlay, cast, STATUS, loc="upper right")
        overlay.text(
            x0,
            py - 1.3,
            TAKEAWAY.replace("; ", ";\n"),
            ha="left",
            va="top",
            fontsize=font,
            fontweight="bold",
            color=cast.text,
        )
    else:
        _draw_table(overlay, g, layout, cast)
        ex.badge(overlay, cast, STATUS, loc="lower right")
    return _plain_text(fig, cast)


# Captions and alternative text

STATUS = "tabulated design, example planet"
TAKEAWAY = (
    "Rows 1 and 4 agree by construction; compare ratios only when their "
    "denominators match."
)

CAPTION = (
    "One planet, four denominators ({ref}`radiometry-contrast-zodi`). Left, the "
    "off-axis PSF of an example planet with band contrast "
    r"$c_b=\Phi_p/\Phi_\star=10^{-10}$ at 3.22 $\lambda/D$ (92 mas at 1000 nm on "
    "the 7.2 m circumscribed aperture), a tabulated offset of the "
    "eac1_optimal_order_6_1d yield input package, so the image is the package's "
    "own PSF with no interpolation. The package's PSFs are averages over its 0.9 "
    "to 1.1 $\\mu$m band, evaluated at five wavelengths, so every ratio here is "
    "a band ratio, not a monochromatic $c_\\lambda$. Right, the stellar leakage of "
    "the same coronagraph for a star 0.0316 $\\lambda/D$ across, the tabulated "
    "diameter nearest the Sun at 10 pc; this design nulls a point-source star. "
    "Both images are in the package's unit, the fraction of the star's photons "
    "incident on the collecting area that lands in each 0.25 $\\lambda/D$ pixel "
    "(the planet image is its PSF times $10^{-10}$), drawn as raw pixels on one "
    "log scale with x right and y up from the star; how these axes map to the sky "
    "belongs to the pending {ref}`image coordinates and PSFlet origin decision "
    "<decision-image-coordinates-and-psflet-origin>`. The dashed circle is the "
    "0.7 $\\lambda/D$ photometric aperture, 1.54 $(\\lambda/D)^2$ or 24.6 pixels, "
    "at the planet position. The table gives, for the same planet, the ratio each "
    "denominator returns. (1) Host-star flux: the host-relative ratio this "
    "section defines at one reference plane before the coronagraph, the flux "
    "ratio of {ref}`Nemati et al. (2023, Sec. 1, text at eq. 1) "
    "<source-nemati2023>`; it appears in neither image. (2) The unocculted PSF "
    "peak: the planet's peak pixel over the star's peak pixel without the "
    "focal-plane mask, a one-pixel form of the normalized intensity of "
    "{ref}`Nemati et al. (2023, Sec. 1.1, eq. 4) <source-nemati2023>` applied to "
    "the planet. The package carries no unocculted PSF, so its off-axis PSF at "
    "20.5 $\\lambda/D$, where the core throughput has leveled off, stands in; it "
    "is sampled at the planet's pixel phase, not at an on-axis star's. The ratio "
    "falls below $10^{-10}$ because the planet's peak is 94 percent of that "
    "unocculted peak. (3) The total incident stellar flux: the planet's peak "
    "pixel read directly in the image unit. That value scales with the pixel "
    "area: it is $3.7\\times10^{-12}$ in one 0.25 $\\lambda/D$ pixel, or "
    "$5.9\\times10^{-11}$ per $(\\lambda/D)^2$, which is the per-pixel versus "
    "per-$(\\lambda/D)^2$ choice of {ref}`radiometry-pixel-brightness`. (4) An "
    "aperture sum, the yield-package normalization: the planet's sum inside the "
    "aperture divided by the same sum for a source with planet flux equal to the "
    "star's, which is the core throughput, 0.58. This is yippy's raw contrast "
    "and the contrast of {ref}`Nemati et al. (2023, Sec. 1.1, eq. 3) "
    "<source-nemati2023>`; for the leakage it gives $2.4\\times10^{-14}$, so "
    "inside this aperture the planet-to-leakage ratio is the band contrast "
    "divided by the raw contrast. Row 4 returns $c_b$ exactly only for a planet "
    "whose spectrum has the star's shape, so that $c_\\lambda$ is constant across "
    "the band; otherwise it weights $c_\\lambda$ by the throughput-weighted "
    "stellar spectrum. Rows 1 and 4 agree by construction of that normalization, "
    "not because they are the same quantity: the first is a flux ratio at a "
    "reference plane, the second a ratio of image sums over one named aperture. "
    "This section lists four denominators, host-star flux, a zero-magnitude "
    "reference, an unocculted PSF peak and total incident stellar flux; the "
    "figure draws three of them, omits the zero-magnitude reference, and adds "
    "the aperture sum, whose aperture {ref}`optics-pixel-measure` requires to be "
    "named. The values come from yippy's throughput, core_area and raw_contrast "
    "tables and its off-axis PSFs; the definitions are identities, the planet is "
    "an example, and the scalar leakage measure every backend exports is the "
    "pending {ref}`stellar leakage measure decision "
    "<decision-stellar-leakage-measure>`."
)
ALT = (
    "Two log-scaled images side by side above a four-row table, sharing one "
    "colorbar, the fraction of the star's incident photons per pixel, from "
    "1e-19 to 1e-11. Left: the off-axis PSF of a planet at 3.22 lambda/D to "
    "the right of a plus sign that marks the star, a bright core inside a "
    "dashed circle keyed 4 with diffraction rings around it; a note gives the "
    "aperture radius 0.7 lambda/D, its area 1.54 square lambda/D or 24.6 "
    "pixels, and that it holds 0.58 of the planet light. Right: the stellar "
    "leakage of a 0.0316 lambda/D star, dark at the star with faint rings, and "
    "the same dashed circle at the planet position; notes give the raw "
    "contrast in that aperture, 2.4e-14, keyed 4, and, keyed 2, that without "
    "the mask the star's peak pixel is 0.0390. Key 3 sits at the top "
    "end of the colorbar. "
    "The table lists four denominators with the numerator and denominator "
    "each uses and the value it gives for the planet: 1, host-star flux, "
    "1.0e-10; 2, unocculted PSF peak, 9.4e-11, because the off-axis peak is 94 "
    "percent of the unocculted peak; 3, total incident stellar flux, 3.7e-12 in "
    "the peak 0.25 lambda/D pixel; 4, an aperture sum, 1.0e-10. A line under "
    "the table reads: rows 1 and 4 agree by construction; compare ratios only "
    "when their denominators match."
)

FIGURES = [
    ex.FigureSpec(
        slug="d12-contrast-denominators",
        build=build_denominators,
        caption=CAPTION,
        alt=ALT,
        status=STATUS,
        params=PARAMS,
    ),
]
ANIMATIONS = []
