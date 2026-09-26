"""How a lenslet IFS turns a sky cube into overlapping detector traces.

Home: the "Lenslet collection, PSFlet placement, and extracted spectra"
section of the optical-fields chapter (clause ``optics-ifs-products``), with
a cross-link from ``radiometry-spectral-covariance``.

Every array panel is drawn by ``coronachrome.viz`` from one synthetic
reference instrument (a 7 by 7 square lenslet grid clocked by arctan(1/2),
174 um pitch over 13 um pixels, linear dispersion in log wavelength, Moffat
PSFlets, ten Nyquist bins from 600 to 720 nm), so the positions a panel
shows are the positions the forward model uses. The side view of the optical
train is a hand-drawn schematic after the layout in Rizzo et al. (2017).

JAX runs in 64-bit inside ``_x64`` only: the template and covariance paths
need it, and scoping it keeps other diagram modules built in the same
process on their own precision.
"""

import functools
import math
from types import SimpleNamespace

import eyepiece as ep
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from coronachrome import (
    IFSRenderer,
    analytic_psflet_pack,
    build_ir,
    spatial_sample,
    spectral_grid,
    spectrum_covariance,
)
from coronachrome import viz as cviz
from coronachrome.build import (
    detector_centroids,
    detector_trace_origin,
    lenslet_cell_centers,
)
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch, Ellipse, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from optixstuff.disperser import LensletDisperser

from explainers import _common as ex

# Scientific inputs (the reference instrument of coronachrome's lenslet
# geometry page). Lengths in meters, wavelengths in nanometers.
PITCH_M = 174e-6
PIXSIZE_M = 13e-6
ANGLE_RAD = math.atan(0.5)
LAM_REF_NM = 660.0
DISPERSION_COEFFS = (140.0, 0.0)
MOFFAT = (1.3, 2.5)
N_LENSLETS = 7
DETECTOR_SHAPE = (120, 120)
RESOLVING_POWER = 50.0
BAND_NM = (600.0, 720.0)
FP_SHAPE = (64, 64)
FP_PX_PER_LENSLET = 6.0
HALF = 4
# The cube producer's optical center: the geometric center (n - 1) / 2.
OPTICAL_CENTER = (31.5, 31.5)
# Lenslet-index (0, 0) and its grid neighbor (1, 0); a corner lenslet whose
# trace runs off the detector's long-wavelength edge.
LENSLET_A = 24
LENSLET_B = 31
LENSLET_EDGE = 43
SCAN_BIN = 4
# Side-view wavelengths and their drawn detector heights (page +y is
# detector +x): the prism bends every wavelength down, the longer one least,
# so it lands higher.
SIDE_SPOTS = ((605.0, -0.95), (660.0, -0.55), (713.0, -0.15))
# An illustrative calibrated centroid correction (dx, dy) in detector pixels,
# the same for every lenslet, varying linearly from the short band edge to
# the long one.
CORRECTION_PX = ((0.3, -0.7), (0.9, -0.9))
# Extra detector columns used only to draw the light that falls past the edge.
EDGE_PAD = 12
NOISE_SIGMA = 0.4
# Seed 5 gives a chi-squared of 14 for the 20 plotted points, within one
# standard deviation of its expectation (20 +/- 6.3).
NOISE_SEED = 5

PARAMS = {
    "pitch_m": PITCH_M,
    "pixsize_m": PIXSIZE_M,
    "angle_rad": ANGLE_RAD,
    "lam_ref_nm": LAM_REF_NM,
    "dispersion_coeffs": list(DISPERSION_COEFFS),
    "psflet": {"kind": "moffat", "params": list(MOFFAT)},
    "n_lenslets": N_LENSLETS,
    "detector_shape": list(DETECTOR_SHAPE),
    "resolving_power": RESOLVING_POWER,
    "band_nm": list(BAND_NM),
    "fp_shape": list(FP_SHAPE),
    "fp_px_per_lenslet": FP_PX_PER_LENSLET,
    "footprint_half_px": HALF,
    "optical_center_px": list(OPTICAL_CENTER),
    "lenslets": [LENSLET_A, LENSLET_B, LENSLET_EDGE],
}


def correction_px(wavelengths_nm):
    """The illustrative centroid correction ``(n_wav, 2)`` at each wavelength."""
    (c0x, c0y), (c1x, c1y) = CORRECTION_PX
    frac = (np.asarray(wavelengths_nm, dtype=float) - BAND_NM[0]) / (
        BAND_NM[1] - BAND_NM[0]
    )
    return np.stack([c0x + frac * (c1x - c0x), c0y + frac * (c1y - c0y)], axis=-1)


def _x64():
    """Scope 64-bit JAX to one build."""
    return jax.enable_x64(True)


def _disperser(**changes):
    kw = {
        "pitch_m": PITCH_M,
        "pixsize_m": PIXSIZE_M,
        "angle_rad": ANGLE_RAD,
        "lam_ref_nm": LAM_REF_NM,
        "pix_per_reselt": 2.0,
        "dispersion_coeffs": jnp.array(DISPERSION_COEFFS),
        "psflet_params": jnp.array(MOFFAT),
        "psflet_ref_nm": LAM_REF_NM,
        "grid_kind": "square",
        "n_lenslets": N_LENSLETS,
        "psflet_kind": "moffat",
        "detector_shape": DETECTOR_SHAPE,
    }
    kw.update(changes)
    return LensletDisperser(**kw)


def _scene():
    """Band-integrated entrance image: a bright core and a faint source."""
    yy, xx = np.mgrid[: FP_SHAPE[0], : FP_SHAPE[1]]
    cx, cy = OPTICAL_CENTER
    image = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.5**2))
    image += 0.05 * np.exp(-((xx - 44.0) ** 2 + (yy - 38.0) ** 2) / (2 * 1.5**2))
    return image


@functools.cache
def _model():
    """Build every array the panels draw, once per process."""
    import warnings

    with _x64(), warnings.catch_warnings():
        # The corner lenslets' long-wavelength footprints leave the detector;
        # the clipping panel shows exactly that.
        warnings.filterwarnings("ignore", message=".*fell off the detector")
        d = _disperser()
        lam, edges = spectral_grid(RESOLVING_POWER, *BAND_NM)
        kw = {
            "fp_px_per_lenslet": FP_PX_PER_LENSLET,
            "wavelength_edges": edges,
            "half": HALF,
        }
        ir = build_ir(d, lam, FP_SHAPE, **kw)
        scene = _scene()

        # Placement with a calibrated centroid correction: a template pack
        # whose planes are stored about their own centroid, plus the
        # correction as separate metadata.
        corr = jnp.asarray(correction_px(np.asarray(lam)))[None]
        pack = analytic_psflet_pack(
            "moffat",
            jnp.array(MOFFAT),
            lam,
            psflet_ref_nm=LAM_REF_NM,
            centroids=corr,
        )
        d_t = _disperser(psflet_kind="template")
        ir_t = build_ir(d_t, lam, FP_SHAPE, psflet_pack=pack, **kw)
        xg, yg = detector_centroids(d, lam)
        xt, yt = detector_centroids(d_t, lam, psflet_pack=pack)

        # The same traces on a detector EDGE_PAD columns wider on the right,
        # built only to show where light past the real edge would land.
        d_w = _disperser(
            detector_shape=(DETECTOR_SHAPE[0], DETECTOR_SHAPE[1] + 2 * EDGE_PAD)
        )
        ir_w = build_ir(d_w, lam, FP_SHAPE, **kw)

        # Extraction: a cube with a gently sloped spectrum, rendered, noised
        # and extracted by unweighted least squares over the columns that
        # reach the detector.
        slope = 1.0 + 0.6 * (np.asarray(lam) - 660.0) / 120.0
        cube = jnp.asarray(scene[None] * slope[:, None, None])
        renderer = IFSRenderer(ir)
        truth = np.asarray(spatial_sample(cube, ir))
        block = np.asarray(
            spectrum_covariance(renderer, channels=jnp.array([LENSLET_A]))
        )
        h = np.asarray(renderer.H_mono.todense())
        live = np.flatnonzero(np.abs(h).sum(axis=0) > 0)
        normal_inv = np.linalg.inv(h[:, live].T @ h[:, live])
        n_wav = len(lam)
        idx = np.r_[
            LENSLET_A * n_wav : (LENSLET_A + 1) * n_wav,
            LENSLET_B * n_wav : (LENSLET_B + 1) * n_wav,
        ]
        where = np.searchsorted(live, idx)
        pair = normal_inv[np.ix_(where, where)]
        rng = np.random.default_rng(NOISE_SEED)
        y = h @ truth.reshape(-1) + NOISE_SIGMA * rng.standard_normal(h.shape[0])
        z_live = normal_inv @ (h[:, live].T @ y)
        z = np.zeros(h.shape[1])
        z[live] = z_live
        extracted = z.reshape(truth.shape)
        return SimpleNamespace(
            d=d,
            lam=np.asarray(lam),
            edges=np.asarray(edges),
            ir=ir,
            scene=scene,
            d_t=d_t,
            ir_t=ir_t,
            pack=pack,
            xg=np.asarray(xg),
            yg=np.asarray(yg),
            xt=np.asarray(xt),
            yt=np.asarray(yt),
            d_w=d_w,
            ir_w=ir_w,
            truth=truth,
            extracted=extracted,
            sigma2=NOISE_SIGMA**2,
            block=block,
            pair=pair,
            trace_origin=detector_trace_origin(d),
            grid_origin=tuple(
                float(np.asarray(v)[0])
                for v in lenslet_cell_centers(
                    d, FP_SHAPE, FP_PX_PER_LENSLET, positions=jnp.zeros((1, 2))
                )
            ),
            cells=tuple(
                np.asarray(v)
                for v in lenslet_cell_centers(d, FP_SHAPE, FP_PX_PER_LENSLET)
            ),
        )


def _styles():
    """Lenslet identities shared by every panel: one color per lenslet."""
    return ep.SourceStyles([f"lenslet {LENSLET_A}", f"lenslet {LENSLET_B}"])


def _lenslet_color(channel):
    return _styles()[f"lenslet {channel}"]["color"]


def _grid_index(channel):
    """Lenslet-grid index ``(i, j)`` of a channel, as drawn in the labels."""
    half = N_LENSLETS // 2
    return (channel // N_LENSLETS - half, channel % N_LENSLETS - half)


def bin_widths_nm(edges):
    """Bin widths from the bin edges."""
    return np.diff(np.asarray(edges, dtype=float))


def bin_smear_px(edges):
    """Detector extent of each bin along the dispersion, in pixels."""
    e = np.asarray(edges, dtype=float)
    return DISPERSION_COEFFS[0] * np.log(e[1:] / e[:-1])


def _correlation(cov):
    sd = np.sqrt(np.diagonal(cov))
    return cov / np.outer(sd, sd)


def _integer_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _tag(ax, xy, text, cast, *, ha="left", va="bottom", gid=None):
    """A short label on a plain background box, for text over an image."""
    return ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=cast.layout.small_pt,
        color=cast.text,
        linespacing=1.0,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": to_rgba(cast.background, 0.85),
            "edgecolor": "none",
        },
        zorder=8,
        gid=gid,
    )


def _backing(cast):
    """A plain backing box in the background color, for a label over marks.

    Labels here carry no stroked halo (at documentation size a halo turns
    every glyph into a blob); a label over an image or a line sits on this
    box instead.
    """
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.25",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _boxed(text, cast):
    """Put ``text`` on the plain backing box, with no stroke."""
    text.set_path_effects([])
    text.set_bbox(_backing(cast))
    return text


def _plain_text(fig, cast):
    """Replace the stroked halos the shared and library helpers add.

    Every label that carried a stroke gets the plain backing box instead,
    unless it already sits on a box of its own.
    """
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                text.set_bbox(_backing(cast))
    return fig


def _label(ax, xy, text, cast, *, color=None, gid=None, **kw):
    """A label in data coordinates on the plain backing box."""
    kw.setdefault("fontsize", cast.layout.small_pt)
    kw.setdefault("linespacing", 1.0)
    return _boxed(
        ax.text(*xy, text, color=cast.text if color is None else color, gid=gid, **kw),
        cast,
    )


def _headline(fig, layout, text):
    """A takeaway headline for the talk slide only."""
    if layout.is_slide:
        fig.suptitle(text, fontsize=layout.title_pt)


# Side view of the optical train


def _ray(ax, start, end, cast, *, color=None):
    """A ray arrow; the talk layout gets a smaller head than the shared default."""
    arts = ex.arrow(ax, start, end, "ray", cast, color=color)
    if cast.layout.is_slide:
        arts[0].set_mutation_scale(0.55 * arts[0].get_mutation_scale())
    return arts


def _side_view(ax, cast, *, slide=False, detector_letter="d"):
    """Schematic side view: entrance focal plane to detector, one lenslet lit.

    After the lenslet-IFS layout of Rizzo et al. (2017), Sec. 2.1, Fig. 1:
    a square lenslet array at the entrance focal plane focuses each cell onto
    a pinhole; the pinhole plane is collimated, dispersed by a prism and
    re-imaged by a camera onto the detector. The dispersion is drawn in the
    page plane. The prism's apex points up, so it deviates every wavelength
    down, toward its base: the longest wavelength least, so it lands highest.
    The camera and detector sit on the deviated axis.
    """
    ax.set(xlim=(-0.45, 10.35), ylim=(-1.8, 2.05), aspect="equal")
    ax.axis("off")
    optics = cast["optics"]
    lenslet = cast["lenslet"]
    lw = cast.layout.lw
    lit_color = _lenslet_color(LENSLET_A)
    # Broadband light before the prism: neutral, not any one wavelength.
    broad = cast.neutral(0.55)
    pitch = 0.34
    x_len, x_pin, x_col, x_pri, x_cam, x_det = 1.0, 2.5, 4.0, 5.55, 7.1, 8.7
    # Entrance focal plane, edge-on, with the lenslet array sitting in it.
    ax.plot([x_len, x_len], [-1.35, 1.35], **cast["reference_plane"].line_kw())
    for yy in (-1.0, 1.0):
        _ray(ax, (-0.4, yy), (x_len - 0.12, 0.3 * yy), cast, color=broad)
    for k in range(-3, 4):
        lit = k == 0
        ax.add_patch(
            Ellipse(
                (x_len, k * pitch),
                0.17,
                0.92 * pitch,
                facecolor=lit_color if lit else ex.neutral(0.12),
                edgecolor=lit_color if lit else lenslet.color,
                lw=lenslet.lw,
                zorder=4,
            )
        )
    # Pinhole mask: a bar with one opening on each lenslet axis.
    for k in range(-4, 4):
        ax.plot(
            [x_pin, x_pin],
            [k * pitch + 0.05, (k + 1) * pitch - 0.05],
            color=optics.color,
            lw=2.2 * lw,
            solid_capstyle="butt",
            zorder=4,
        )
    # The lit lenslet focuses its cell onto its pinhole, which the collimator
    # then sees as a point source.
    ax.fill(
        [x_len, x_pin, x_len],
        [-0.42 * pitch, 0.0, 0.42 * pitch],
        color=broad,
        alpha=0.45,
        lw=0,
        zorder=3,
    )
    ax.fill([x_pin, x_col, x_col], [0.0, -0.8, 0.8], color=broad, alpha=0.22, lw=0)
    ex.lens(ax, (x_col, 0.0), 2.3, cast, thickness=0.12)
    apex_y, base_y = 1.1, -1.0
    prism = ex.region(
        ax,
        "optics",
        Polygon(
            [(x_pri - 0.5, base_y), (x_pri + 0.5, base_y), (x_pri, apex_y)],
            closed=True,
        ),
        cast,
    )
    prism.set_gid("d05-side-prism")

    def face_x(yy, sign):
        """x of the prism's left (sign -1) or right (+1) face at height yy."""
        return x_pri + sign * 0.5 * (apex_y - yy) / (apex_y - base_y)

    for yy in (0.55, -0.35):
        _ray(ax, (x_col + 0.15, yy), (face_x(yy, -1) - 0.05, yy), cast, color=broad)
    # The deviated axis: the camera and detector sit below the entrance axis.
    y_axis = -0.55
    ex.lens(ax, (x_cam, y_axis), 2.0, cast, thickness=0.12)
    # Three wavelengths leave the prism's exit face, every one bent down
    # toward the base, the longest least; the camera focuses each onto its
    # own detector position.
    y_exit = 0.1
    x_exit = face_x(y_exit, +1)
    for lam_nm, y_spot in SIDE_SPOTS:
        color = ex.wavelength_color(lam_nm)
        y_cam = y_spot - 0.05
        ax.plot(
            [x_exit, x_cam],
            [y_exit, y_cam],
            color=color,
            lw=lw,
            zorder=3,
            gid=f"d05-side-ray-{lam_nm:.0f}",
        )
        _ray(ax, (x_cam, y_cam), (x_det, y_spot), cast, color=color)
    ex.region(ax, "detector", Rectangle((x_det, y_axis - 0.85), 0.28, 1.7), cast)
    for lam_nm, y_spot in (SIDE_SPOTS[0], SIDE_SPOTS[-1]):
        _label(
            ax,
            (x_det + 0.4, y_spot),
            f"{lam_nm:.0f} nm",
            cast,
            color=ex.wavelength_color(lam_nm),
            ha="left",
            va="center",
            gid=f"d05-side-{lam_nm:.0f}",
        )
    # Detector axis: a plain coordinate arrow, not a ray or a data flow.
    ax.annotate(
        "",
        (x_det + 0.55, y_axis + 0.25),
        (x_det + 0.55, y_axis - 0.25),
        arrowprops={"arrowstyle": "->", "color": cast.text, "lw": lw},
    )
    _label(ax, (x_det + 0.72, y_axis), "detector +x", cast, ha="left", va="center")
    names = (
        (
            x_len - 0.3,
            "lenslets (b)" if slide else "lenslets in the\nentrance plane (b)",
        ),
        (x_pin, "pinholes"),
        (x_col, "collimator"),
        (x_pri, "prism"),
        (x_cam, "camera"),
        (x_det + 0.14, f"detector ({detector_letter})"),
    )
    for xx, text in names:
        _label(ax, (xx, 1.45), text, cast, ha="center", va="bottom")
    badge = ex.badge(ax, cast, "schematic, not to scale", loc="lower left")
    # Centered under the collimator and prism, clear of the lenslets and the
    # detector.
    badge.set_position((0.44, 0.01))
    badge.set_horizontalalignment("center")


# Entrance plane and its origins

ENTRANCE_WINDOW = (5.0, 59.0, 5.0, 59.0)
ZOOM_WINDOW = (29.6, 34.2, 29.6, 34.2)


def _entrance_panel(
    ax,
    m,
    cast,
    *,
    channels,
    window=ENTRANCE_WINDOW,
    zoom_box=True,
    spaxel_tag=True,
    grid_labels=True,
    zoom_letter="b",
):
    """Entrance image with every lenslet cell; with the zoom box, the origins."""
    res = cviz.plot_lenslet_cells(
        m.d,
        FP_SHAPE,
        fp_px_per_lenslet=FP_PX_PER_LENSLET,
        image=m.scene,
        channels=channels,
        optical_center_px=OPTICAL_CENTER,
        styles=_styles(),
        window=window,
        colorbar=False,
        ax=ax,
    )
    # The chapter's map for a noiseless focal-plane image in either mode.
    res.artists["image"].set_cmap(ex.image_cmap("intensity"))
    for text in res.artists["text"]:
        name = text.get_text()
        if name in ("optical center", "lenslet-grid origin"):
            # The zoom names them; at this scale the two marks overlap.
            text.set_visible(False)
        for ch in channels if grid_labels else ():
            if name == f"lenslet {ch}":
                i, j = _grid_index(ch)
                text.set_text(f"lenslet {ch} = grid ({i}, {j})")
    if not grid_labels:
        # The short names would collide above the two cells; the first
        # lenslet's name moves to the left of its own cell.
        first = res.artists["ellipse"][0]
        xy = np.asarray(first.get_xy())
        left = xy[np.argmin(xy[:, 0])]
        for text in res.artists["text"]:
            if text.get_text() == f"lenslet {channels[0]}":
                text.set_visible(False)
        _label(
            ax,
            (float(left[0]) - 0.8, float(left[1])),
            f"lenslet {channels[0]}",
            cast,
            color=_lenslet_color(channels[0]),
            ha="right",
            va="center",
        )
    if not zoom_box:
        # Without the zoom panel that names them, the two origin marks would
        # be unlabeled glyphs; the origins figure's zoom is their home.
        for line in res.artists.get("lines", []):
            line.set_visible(False)
    ax.set_xlabel("entrance $x$ [cube px]")
    ax.set_ylabel("entrance $y$ [cube px]")
    if spaxel_tag:
        _tag(
            ax,
            (0.98, 0.97),
            "one lenslet cell = one spaxel",
            cast,
            ha="right",
            va="top",
        )
    if zoom_box:
        x0, x1, y0, y1 = ZOOM_WINDOW
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=cast.text,
                lw=0.8 * cast.layout.lw,
                zorder=7,
            )
        )
        _label(
            ax,
            (x0 - 0.5, y0 - 0.5),
            f"zoom ({zoom_letter})",
            cast,
            ha="right",
            va="top",
            fontstyle="italic",
        )
    return res


def _zoom_panel(ax, m, cast):
    """The optical center and the lenslet-grid origin on the cube pixel grid."""
    res = cviz.plot_lenslet_cells(
        m.d,
        FP_SHAPE,
        fp_px_per_lenslet=FP_PX_PER_LENSLET,
        optical_center_px=OPTICAL_CENTER,
        window=ZOOM_WINDOW,
        ax=ax,
    )
    for text in res.artists.get("text", []):
        text.set_visible(False)
    # Only the two origins and the pixel edges belong in this zoom; the
    # corners of the surrounding lenslet cells would be unnamed fragments.
    res.artists["collection"].set_visible(False)
    for line in res.artists.get("lines", []):
        line.set_gid(f"d05-zoom-{line.get_label().replace(' ', '-')}")
    for v in np.arange(29.5, 35.0, 1.0):
        ax.axvline(v, color=cast.neutral(0.35), lw=0.6 * cast.layout.lw, zorder=1)
        ax.axhline(v, color=cast.neutral(0.35), lw=0.6 * cast.layout.lw, zorder=1)
    ax.set(xticks=[30, 31, 32, 33, 34], yticks=[30, 31, 32, 33, 34])
    ax.set_xlabel("entrance $x$ [cube px]")
    ax.set_ylabel("entrance $y$ [cube px]")
    ox, oy = OPTICAL_CENTER
    gx, gy = m.grid_origin
    _label(
        ax,
        (ox - 0.15, oy - 0.3),
        f"optical center ({ox:g}, {oy:g})\n" + r"= $(n_\mathrm{cube} - 1)/2$",
        cast,
        ha="center",
        va="top",
        gid="d05-zoom-optical-center-label",
    )
    _label(
        ax,
        (gx, gy + 0.3),
        f"lenslet-grid origin ({gx:g}, {gy:g})\n"
        + r"= $n_\mathrm{cube}/2$ in coronachrome"
        + "\n(implementation)",
        cast,
        ha="center",
        va="bottom",
        gid="d05-zoom-grid-origin-label",
    )
    _offset_arrow(ax, m, cast)
    _tag(ax, (0.03, 0.03), "lines: cube pixel edges", cast)
    return res


def origin_offset_px(m):
    """Lenslet-grid origin minus optical center, in cube pixels ``(dx, dy)``."""
    return (
        m.grid_origin[0] - OPTICAL_CENTER[0],
        m.grid_origin[1] - OPTICAL_CENTER[1],
    )


def _offset_arrow(ax, m, cast):
    """A one-headed arrow from the optical center to the grid origin."""
    ox, oy = OPTICAL_CENTER
    gx, gy = m.grid_origin
    dx, _ = origin_offset_px(m)
    ax.annotate(
        "",
        (gx, gy),
        (ox, oy),
        arrowprops={
            "arrowstyle": "-|>",
            "color": cast.text,
            "lw": 0.8 * cast.layout.lw,
            "shrinkA": 6,
            "shrinkB": 6,
            "mutation_scale": 8 if not cast.layout.is_slide else 16,
        },
        zorder=7,
        gid="d05-zoom-offset-arrow",
    )
    _label(
        ax,
        (gx + 0.2, 0.5 * (oy + gy) - 0.05),
        f"{dx:+g} px on each\naxis (grid - optical)",
        cast,
        ha="left",
        va="center",
        gid="d05-zoom-offset-label",
    )


# Detector traces


DETECTOR_WINDOW = (32.0, 97.0, 48.5, 74.0)


def _psflet_box(ir, channel, index):
    """Pixel-edge box ``(x0, y0, w, h)`` of one bin footprint in the IR.

    Only pixels that carry weight count: off-detector footprint pixels are
    stored with zero weight at a clipped index.
    """
    nx = ir.det_shape[1]
    rows = np.asarray(ir.det_rows[channel, index])
    rows = rows[np.asarray(ir.det_vals[channel, index]) > 0]
    ys, xs = np.divmod(rows, nx)
    return (
        float(xs.min()) - 0.5,
        float(ys.min()) - 0.5,
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    )


def _hide_library_labels(res):
    """Hide plot_traces' labels and trace-origin mark; we place our own.

    The scan readout is hidden too: the bin-footprint label carries it.
    """
    for text in res.artists["text"]:
        text.set_visible(False)
    for line in res.artists["lines"]:
        if line.get_label() == "detector trace origin":
            line.set_visible(False)


def _detector_panel(ax, m, cast, *, channels, scan_index=SCAN_BIN, window=None):
    """plot_traces with the chapter's colormap for a noiseless model image."""
    res = cviz.plot_traces(
        m.ir,
        m.d,
        m.lam,
        channels=channels,
        scan_index=scan_index,
        styles=_styles(),
        window=window,
        colorbar=False,
        ax=ax,
    )
    # A noiseless model of detector pixels takes the intensity role; the
    # library hard-codes the readouts map, meant for measured frames.
    res.artists["image"].set_cmap(ex.image_cmap("intensity"))
    _integer_ticks(ax)
    _outline_marks(res, cast)
    return res


def _outline_marks(res, cast):
    """Outline lenslet-colored marks so they read over the image colors."""
    stroke = [patheffects.withStroke(linewidth=3.2, foreground=cast.background)]
    marks = [*res.artists["lines"], res.artists["scatter"]]
    marks += [
        p for p in res.artists.get("ellipse", []) if p.get_label() != "detector edge"
    ]
    for mark in marks:
        mark.set_path_effects(stroke)


def _trace_origin_mark(ax, m, cast, text_xy):
    """Detector trace origin: a tick pair that leaves the centroid circles clear."""
    x0, y0 = m.trace_origin
    ticks = []
    for lo, hi in ((y0 + 1.1, y0 + 2.6), (y0 - 2.6, y0 - 1.1)):
        (tick,) = ax.plot(
            [x0, x0],
            [lo, hi],
            color=cast.text,
            lw=1.6 * cast.layout.lw,
            solid_capstyle="butt",
            zorder=7,
            gid="d05-trace-origin",
        )
        ticks.append(tick)
    text = _boxed(
        ax.annotate(
            f"detector trace origin ({x0:g}, {y0:g})\n"
            + r"= $n_\mathrm{det}/2$ in coronachrome (implementation):"
            + f"\nlenslet (0, 0) at reference wavelength {LAM_REF_NM:.0f} nm",
            (x0, y0 - 2.6),
            xytext=text_xy,
            textcoords="data",
            ha="center",
            va="top",
            fontsize=cast.layout.small_pt,
            color=cast.text,
            linespacing=1.0,
            arrowprops={"arrowstyle": "-", "color": cast.text, "lw": 0.6},
            gid="d05-trace-origin-label",
            zorder=7,
        ),
        cast,
    )
    return ticks, text


def _detector_labels(ax, m, cast, res, *, channels, scan_index=SCAN_BIN, detail=True):
    """Name every mark on the trace panel; return the scan-following labels.

    Returns:
        ``(artists, update)``: a dict of the label artists, and
        ``update(k)`` that moves the centroid and footprint labels with the
        scanned bin.
    """
    _hide_library_labels(res)
    a = channels[0]
    ca = _lenslet_color(a)
    ya = float(m.yg[a, 0])
    art = {}
    art["605"] = _label(
        ax,
        (float(m.xg[a, 0]) - 6.5, ya),
        f"{m.lam[0]:.0f} nm",
        cast,
        color=ca,
        ha="right",
        va="center",
        gid="d05-label-605",
    )
    art["713"] = _label(
        ax,
        (float(m.xg[a, -1]) + 5.0, ya),
        f"{m.lam[-1]:.0f} nm",
        cast,
        color=ca,
        ha="left",
        va="center",
        gid="d05-label-713",
    )
    art[a] = _label(
        ax,
        (float(m.xg[a, 0]) - 6.5, ya - 1.6),
        f"lenslet {a}",
        cast,
        color=ca,
        ha="right",
        va="top",
    )
    for ch in channels[1:]:
        art[ch] = _label(
            ax,
            (float(m.xg[ch, -1]) + 5.0, float(m.yg[ch, 0])),
            f"lenslet {ch}",
            cast,
            color=_lenslet_color(ch),
            ha="left",
            va="center",
        )
    art["bin"] = _label(
        ax, (0, 0), "", cast, color=ca, ha="left", va="top", gid="d05-bin-label"
    )
    art["centroid"] = _boxed(
        ax.annotate(
            "centroid",
            (0, 0),
            xytext=(0, 0),
            textcoords="data",
            ha="right",
            va="top",
            fontsize=cast.layout.small_pt,
            color=cast.text,
            arrowprops={"arrowstyle": "-", "color": cast.text, "lw": 0.6},
            zorder=7,
            gid="d05-centroid-label",
        ),
        cast,
    )
    if detail:
        art["circle"] = _tag(
            ax,
            (0.985, 0.97),
            "each circle: center of one wavelength\n"
            f"bin, about {np.mean(bin_widths_nm(m.edges)):.0f} nm wide; "
            "1 square = 1 pixel",
            cast,
            ha="right",
            va="top",
        )

    def update(k):
        x0, y0, _, _ = _psflet_box(m.ir, a, k)
        art["bin"].set_position((x0, y0 - 0.15))
        art["bin"].set_text(
            f"bin footprint:\nlenslet {a}, " + rf"$\lambda$ = {m.lam[k]:.0f} nm"
        )
        art["centroid"].xy = (float(m.xg[a, k]), ya)
        # Below the trace, left of "bin footprint": clear of both traces.
        art["centroid"].set_position((x0 - 1.2, y0 - 0.15))

    update(scan_index)
    return art, update


def build_instrument(layout, cast):
    """Figure: the optical train, the entrance cells, the detector traces.

    The architecture only: the three reference origins have their own figure.
    """
    m = _model()
    with _x64():
        fig = plt.figure(figsize=layout.size(5.6), layout="constrained")
        if layout.is_slide:
            gs = fig.add_gridspec(
                2, 2, height_ratios=(1.0, 1.25), width_ratios=(1.0, 2.0)
            )
        else:
            gs = fig.add_gridspec(
                2, 2, height_ratios=(1.0, 1.0), width_ratios=(1.0, 1.85)
            )
        ax_side = fig.add_subplot(gs[0, :])
        ax_ent = fig.add_subplot(gs[1, 0])
        ax_det = fig.add_subplot(gs[1, 1])
        _side_view(ax_side, cast, slide=layout.is_slide, detector_letter="c")
        ax_side.set_title("(a) side view: the light of one lenslet", loc="left")
        _entrance_panel(
            ax_ent,
            m,
            cast,
            channels=(LENSLET_A, LENSLET_B),
            zoom_box=False,
            grid_labels=False,
        )
        ax_ent.set_title(
            "(b) entrance: lenslet cells"
            if layout.is_slide
            else "(b) entrance plane: lenslet cells",
            loc="left",
        )
        res = _detector_panel(
            ax_det, m, cast, channels=(LENSLET_A, LENSLET_B), window=DETECTOR_WINDOW
        )
        _detector_labels(
            ax_det,
            m,
            cast,
            res,
            channels=(LENSLET_A, LENSLET_B),
            detail=not layout.is_slide,
        )
        ax_det.set_title(
            "(c) detector traces"
            if layout.is_slide
            else "(c) detector: one trace per lenslet",
            loc="left",
        )
        ex.badge(ax_det, cast, "simulated", loc="lower left")
        if not layout.is_slide:
            ex.badge(ax_ent, cast, "simulated", loc="lower left")
        for ax in (ax_ent, ax_det):
            ax.set_anchor("N")
        _headline(
            fig, layout, "Each lenslet becomes one short spectrum on the detector"
        )
    return _plain_text(fig, cast)


# Around the detector trace origin: the trace of lenslet (0, 0) near 660 nm.
ORIGIN_DET_WINDOW = (44.5, 75.5, 50.5, 68.5)
SLIDE_ORIGIN_DET_WINDOW = (49.1, 69.9, 49.5, 70.5)


def _origin_detector_panel(ax, m, cast, *, window):
    """The detector trace origin among the centroids of lenslet (0, 0)."""
    res = _detector_panel(
        ax, m, cast, channels=(LENSLET_A,), scan_index=None, window=window
    )
    _hide_library_labels(res)
    x0 = m.trace_origin[0]
    text_x = 0.5 * (window[0] + window[1]) if cast.layout.is_slide else x0 + 3.0
    _trace_origin_mark(ax, m, cast, (text_x, 55.0))
    ca = _lenslet_color(LENSLET_A)
    k = SCAN_BIN
    ya = float(m.yg[LENSLET_A, k])
    # The two bin centers that bracket the reference wavelength, labeled
    # above the footprints on short leaders.
    for kk, ha, dx in ((k, "right", -1.0), (k + 1, "left", 1.0)):
        xk = float(m.xg[LENSLET_A, kk])
        _boxed(
            ax.annotate(
                f"{m.lam[kk]:.0f} nm",
                (xk, ya + 0.6),
                xytext=(xk + dx, ya + 5.2),
                textcoords="data",
                ha=ha,
                va="bottom",
                color=ca,
                fontsize=cast.layout.small_pt,
                arrowprops={"arrowstyle": "-", "color": ca, "lw": 0.8},
                zorder=7,
                gid=f"d05-origin-bin-{m.lam[kk]:.0f}",
            ),
            cast,
        )
    i, j = _grid_index(LENSLET_A)
    _label(
        ax,
        (window[0] + 0.5, window[3] - 0.5),
        f"lenslet {LENSLET_A} = grid ({i}, {j})",
        cast,
        color=ca,
        ha="left",
        va="top",
        gid="d05-origin-lenslet-label",
    )
    return res


def build_origins(layout, cast):
    """Figure: the three reference origins, each in its own frame.

    The entrance panel is carried over from the instrument figure (same
    constructor) with its zoom box; the zoom names the optical center and
    the lenslet-grid origin on the cube pixel grid, and the detector panel
    names the detector trace origin.
    """
    m = _model()
    with _x64():
        if layout.is_slide:
            # The carried panel is a small anchor; the two origin panels
            # take the frame.
            fig, (ax_ent, ax_zoom, ax_det) = ex.figure(
                layout, ncols=3, width_ratios=(0.65, 1.0, 1.0)
            )
            # Panels hang from the headline: no dead band under it.
            for ax in (ax_ent, ax_zoom, ax_det):
                ax.set_anchor("N")
            window = SLIDE_ORIGIN_DET_WINDOW
        else:
            fig = plt.figure(figsize=layout.size(6.4), layout="constrained")
            gs = fig.add_gridspec(
                2, 2, height_ratios=(1.3, 1.0), width_ratios=(0.6, 1.0)
            )
            # A small anchor: the carried panel takes the upper part of its
            # cell only.
            ax_ent = fig.add_subplot(
                gs[0, 0].subgridspec(2, 1, height_ratios=(0.62, 0.38))[0]
            )
            ax_zoom = fig.add_subplot(gs[0, 1])
            ax_det = fig.add_subplot(gs[1, :])
            window = ORIGIN_DET_WINDOW
        carried = _entrance_panel(
            ax_ent,
            m,
            cast,
            channels=(LENSLET_A, LENSLET_B),
            zoom_box=True,
            spaxel_tag=False,
            grid_labels=False,
            zoom_letter="b",
        )
        # The zoom outline only: at this scale the two origin marks shrink
        # into one unreadable glyph, and (b) is their home.
        for line in carried.artists.get("lines", []):
            line.set_visible(False)
        # The relay tag rides in the title, clear of the small panel's marks.
        title = ax_ent.set_title(
            "(a) entrance\n(previous slide)"
            if layout.is_slide
            else "(a) entrance plane\n(from the previous figure)",
            loc="left",
        )
        title.set_gid("d05-carried-tag")
        _zoom_panel(ax_zoom, m, cast)
        ax_zoom.set_title(
            "(b) cube: two origins"
            if layout.is_slide
            else "(b) cube: two entrance origins",
            loc="left",
        )
        _origin_detector_panel(ax_det, m, cast, window=window)
        ax_det.set_title(
            "(c) detector" if layout.is_slide else "(c) detector: the trace origin",
            loc="left",
        )
        for ax in (ax_ent, ax_det):
            ex.badge(ax, cast, "simulated", loc="lower left")
        _headline(
            fig,
            layout,
            "Three origins in three frames: never exchange one for another",
        )
    return _plain_text(fig, cast)


# PSFlet placement and detector-edge capture


def _footprint_image(ir, pairs):
    """Detector image of unit flux in the listed ``(channel, bin)`` footprints."""
    ny, nx = ir.det_shape
    flat = np.zeros(ny * nx)
    for ch, k in pairs:
        np.add.at(flat, np.asarray(ir.det_rows[ch, k]), np.asarray(ir.det_vals[ch, k]))
    return flat.reshape(ny, nx)


def _lenslet_image(ir, channels):
    return _footprint_image(ir, [(c, k) for c in channels for k in range(ir.n_wav)])


def _set_image(res, image):
    """Show ``image`` in a plot_traces panel under its pinned norm."""
    im = res.artists["image"]
    im.set_data(np.clip(image, im.norm.vmin, None))


def _template_panel(ax, m, cast):
    """The stored template: pixel-integrated PSFlet against centroid offset."""
    k = SCAN_BIN
    plane = np.asarray(m.pack.templates[0, k])
    off = np.asarray(m.pack.offsets)
    step = float(off[1] - off[0])
    extent = (off[0] - 0.5 * step, off[-1] + 0.5 * step) * 2
    peak = float(plane.max())
    res = ep.imshow_log(
        plane,
        ax=ax,
        extent=extent,
        floor=1e-3 * peak,
        cmap=ex.image_cmap("intensity"),
        colorbar=False,
    )
    lim = HALF + 0.5
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.set_xlabel(r"offset from centroid $\Delta x$ [px]")
    ax.set_ylabel(r"offset $\Delta y$ [px]")
    _integer_ticks(ax)
    scale = cast.layout.marker_pt / 7.0
    ax.plot(
        [0],
        [0],
        marker="+",
        ms=14 * scale,
        mew=2.0 * scale,
        color=cast.text,
        ls="none",
        zorder=5,
    )
    _boxed(
        ax.annotate(
            "centroid at (0, 0)",
            (0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color=cast.text,
            fontsize=cast.layout.small_pt,
        ),
        cast,
    )
    _tag(
        ax,
        (0.97, 0.03),
        f"one wavelength: PSFlet at\nthe {m.lam[k]:.0f} nm bin center",
        cast,
        ha="right",
    )
    return res


PLACE_WINDOW = (54.5, 62.5, 55.5, 63.5)


def _placement_panel(ax, m, cast):
    """Geometric trace point, calibrated correction, placed bin footprint."""
    k = SCAN_BIN
    res = cviz.plot_traces(
        m.ir_t,
        m.d_t,
        m.lam,
        channels=(LENSLET_A,),
        psflet_pack=m.pack,
        marked=[k],
        scan_index=k,
        show_trace_origin=False,
        styles=_styles(),
        window=PLACE_WINDOW,
        colorbar=False,
        ax=ax,
    )
    res.artists["image"].set_cmap(ex.image_cmap("intensity"))
    _integer_ticks(ax)
    _set_image(res, _footprint_image(m.ir_t, [(LENSLET_A, k)]))
    # The footprint box is larger than this zoom; the instrument figure shows it.
    for patch in res.artists["ellipse"]:
        if patch.get_label() == "scan footprint":
            patch.set_visible(False)
    for text in res.artists["text"]:
        if text.get_label() == "scan readout":
            text.set_text(f"lenslet {LENSLET_A}, {m.lam[k]:.0f} nm bin")
    small = cast.layout.small_pt
    gx, gy = float(m.xg[LENSLET_A, k]), float(m.yg[LENSLET_A, k])
    tx, ty = float(m.xt[LENSLET_A, k]), float(m.yt[LENSLET_A, k])
    wx, wy = gx + 2 * (tx - gx), gy + 2 * (ty - gy)
    scale = cast.layout.marker_pt / 7.0
    ax.plot([gx], [gy], marker="o", ms=5 * scale, color=cast.text, ls="none", zorder=7)
    arrow_kw = {
        "arrowstyle": "-|>",
        "color": cast.text,
        "lw": cast.layout.lw,
        "shrinkA": 2,
        "shrinkB": 6,
    }
    ax.annotate("", (tx, ty), (gx, gy), arrowprops=arrow_kw, zorder=7)
    # The same correction applied a second time, from the placed centroid.
    ax.annotate("", (wx, wy), (tx, ty), arrowprops={**arrow_kw, "ls": "--"}, zorder=7)
    _label(
        ax,
        (gx - 0.15, gy + 0.2),
        "geometric trace point",
        cast,
        ha="right",
        va="bottom",
    )
    # Below the trace line and left of the arrow, on a short leader.
    _boxed(
        ax.annotate(
            "calibrated\ncorrection",
            (0.5 * (gx + tx), 0.5 * (gy + ty)),
            xytext=(-14, -14),
            textcoords="offset points",
            ha="right",
            va="top",
            color=cast.text,
            fontsize=small,
            linespacing=1.0,
            arrowprops={"arrowstyle": "-", "color": cast.text, "lw": 0.6},
            gid="d05-correction-label",
        ),
        cast,
    )
    _boxed(
        ax.annotate(
            "placed centroid",
            (tx, ty),
            xytext=(12, -1),
            textcoords="offset points",
            ha="left",
            va="center",
            color=cast.text,
            fontsize=small,
        ),
        cast,
    )
    ax.plot(
        [wx],
        [wy],
        marker="o",
        ms=11 * scale,
        mfc="none",
        mec=cast.text,
        mew=1.0 * scale,
        ls="none",
        zorder=7,
    )
    ax.plot([wx], [wy], marker="x", ms=6 * scale, color=cast.text, ls="none", zorder=7)
    _boxed(
        ax.annotate(
            "correction applied again:\ntemplate not recentered,\nshifted twice",
            (wx, wy),
            xytext=(12, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            color=cast.text,
            fontsize=small,
            fontstyle="italic",
            linespacing=1.0,
        ),
        cast,
    )
    _label(
        ax,
        (PLACE_WINDOW[1] - 0.2, gy + 0.12),
        "geometric trace",
        cast,
        color=_lenslet_color(LENSLET_A),
        ha="right",
        va="bottom",
    )
    smear = float(bin_smear_px(m.edges)[k])
    _tag(
        ax,
        (0.03, 0.97),
        f"bin footprint: the PSFlet smeared\nover the bin's {smear:.2f} px along x",
        cast,
        va="top",
        gid="d05-smear-tag",
    )
    for text in res.artists["text"]:
        if text.get_label() == "scan readout":
            text.set_visible(False)
    return res


EDGE_WINDOW = (91.5, 129.5, 45.0, 60.5)


def _edge_panel(ax, m, cast):
    """A trace running off the detector: light past the edge is lost."""
    pad = EDGE_PAD
    nx = DETECTOR_SHAPE[1]
    # Wide-detector image, cropped so its columns share the real detector's
    # x coordinates: column c here is detector x = c.
    wide = _lenslet_image(m.ir_w, [LENSLET_EDGE])[:, pad : pad + nx + pad]
    color = cast.neutral(0.85)
    styles = {f"lenslet {LENSLET_EDGE}": {"color": color, "marker": "o"}}
    res = cviz.plot_traces(
        wide,
        (m.xg, m.yg),
        m.lam,
        channels=(LENSLET_EDGE,),
        show_trace_origin=False,
        show_detector_edge=False,
        styles=styles,
        window=EDGE_WINDOW,
        colorbar=False,
        ax=ax,
    )
    res.artists["image"].set_cmap(ex.image_cmap("intensity"))
    _integer_ticks(ax)
    # A background-colored outline keeps the gray centroid circles visible
    # on the bright trace in both modes.
    _outline_marks(res, cast)
    edge_x = nx - 0.5
    y0, y1 = EDGE_WINDOW[2], EDGE_WINDOW[3]
    # The same light wash in both modes, so the lost light stays visible.
    ax.add_patch(
        Rectangle(
            (edge_x, y0),
            EDGE_WINDOW[1] - edge_x,
            y1 - y0,
            facecolor=to_rgba("white", 0.4),
            edgecolor=cast.neutral(0.5),
            lw=0,
            hatch="//",
            zorder=4,
        )
    )
    ax.plot(
        [edge_x, edge_x],
        [y0, y1],
        color=cast.text,
        ls="--",
        lw=cast.layout.lw,
        zorder=5,
    )
    yc = float(m.yg[LENSLET_EDGE, 0])
    _label(
        ax,
        (edge_x - 0.4, y0 + 0.4),
        "detector edge",
        cast,
        ha="right",
        va="bottom",
        zorder=6,
    )
    _label(
        ax,
        (edge_x + 0.6, y0 + 0.3),
        "lost off the detector:\nnot added back to the\npixels that remain\n"
        "(handbook convention;\ncoronachrome currently\nrenormalizes instead)",
        cast,
        ha="left",
        va="bottom",
        fontstyle="italic",
        zorder=6,
    )
    _label(
        ax,
        (float(m.xg[LENSLET_EDGE, 3]), yc - HALF - 0.7),
        f"lenslet {LENSLET_EDGE}",
        cast,
        color=color,
        ha="left",
        va="top",
    )
    _label(
        ax,
        (EDGE_WINDOW[0] + 0.5, yc + HALF + 0.6),
        f"footprint support +/-{HALF} px: a numerical cutoff, not optics",
        cast,
        ha="left",
        va="bottom",
        fontstyle="italic",
        zorder=6,
    )
    for text in res.artists["text"]:
        text.set_visible(False)
    _label(
        ax,
        (float(m.xg[LENSLET_EDGE, 0]) - 1.0, yc + 1.0),
        f"{m.lam[0]:.0f} nm",
        cast,
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        (float(m.xg[LENSLET_EDGE, -1]) + 0.2, yc + 1.0),
        f"{m.lam[-1]:.0f} nm",
        cast,
        ha="right",
        va="bottom",
    )
    return res


def build_placement(layout, cast):
    """Figure: a recentered template, its calibrated placement, edge capture."""
    m = _model()
    with _x64():
        if layout.is_slide:
            # A talk slide carries the template convention only.
            fig, (ax_t, ax_p) = ex.figure(layout, ncols=2)
            ax_e = None
        else:
            fig = plt.figure(figsize=layout.size(6.8), layout="constrained")
            gs = fig.add_gridspec(
                2, 2, height_ratios=(1.3, 1.0), width_ratios=(1.0, 1.0)
            )
            ax_t = fig.add_subplot(gs[0, 0])
            ax_p = fig.add_subplot(gs[0, 1])
            ax_e = fig.add_subplot(gs[1, :])
        _template_panel(ax_t, m, cast)
        ax_t.set_title("(a) stored template", loc="left")
        _placement_panel(ax_p, m, cast)
        ax_p.set_title("(b) placed on the detector", loc="left")
        axes = [ax_t, ax_p]
        if ax_e is not None:
            _edge_panel(ax_e, m, cast)
            ax_e.set_title("(c) detector edge", loc="left")
            axes.append(ax_e)
        for ax in axes:
            ex.badge(ax, cast, "simulated", loc="lower left")
        _headline(
            fig,
            layout,
            "Store the template about its centroid; apply the correction once",
        )
    return _plain_text(fig, cast)


# Extraction and covariance

OVERLAP_WINDOW = (46.5, 77.5, 51.0, 73.5)
# The talk slide's panel is taller, to match the correlation matrix beside it.
SLIDE_OVERLAP_WINDOW = (46.5, 77.5, 47.5, 76.0)


def _box(ax, ir, channel, index, color, *, ls="-", gid=None):
    x0, y0, w, h = _psflet_box(ir, channel, index)
    return ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor=color,
            lw=1.6,
            ls=ls,
            zorder=6,
            gid=gid,
        )
    )


def _overlap_panel(ax, m, cast, *, window=OVERLAP_WINDOW):
    """Footprints that share detector pixels: along a trace, and across."""
    k = SCAN_BIN
    res = _detector_panel(ax, m, cast, channels=(LENSLET_A, LENSLET_B), window=window)
    _hide_library_labels(res)
    res.artists["line"].set_visible(False)
    for text in res.artists["text"]:
        text.set_visible(False)
    ca, cb = _lenslet_color(LENSLET_A), _lenslet_color(LENSLET_B)
    _box(ax, m.ir, LENSLET_A, k + 1, ca, ls="--")
    _box(ax, m.ir, LENSLET_B, 0, cb)
    xa0, ya0, wa, ha = _psflet_box(m.ir, LENSLET_A, k)
    xn0, _, wn, _ = _psflet_box(m.ir, LENSLET_A, k + 1)
    # The pixels the two neighboring bins of lenslet 24 both claim.
    sx0, sx1 = max(xa0, xn0), min(xa0 + wa, xn0 + wn)
    ax.add_patch(
        Rectangle(
            (sx0, ya0),
            sx1 - sx0,
            ha,
            facecolor="none",
            edgecolor=cast.text,
            hatch="xx",
            lw=0,
            zorder=5,
            gid="d05-shared-neighbor-bins",
        )
    )
    _label(
        ax,
        (xa0 + 0.2, ya0 - 0.3),
        f"{m.lam[k]:.0f} nm bin",
        cast,
        color=ca,
        ha="left",
        va="top",
    )
    _label(
        ax,
        (xn0 + wn + 0.3, ya0 + 0.3),
        f"{m.lam[k + 1]:.0f} nm\nbin",
        cast,
        color=ca,
        ha="left",
        va="bottom",
        fontstyle="italic",
    )
    # Lower right, clear of the bin names, on a leader to the hatched pixels.
    _boxed(
        ax.annotate(
            "shared by neighboring\nbins: anticorrelated",
            (sx1 - 1.0, ya0 + 1.0),
            xytext=(window[1] - 0.4, window[2] + 0.4),
            textcoords="data",
            ha="right",
            va="bottom",
            color=cast.text,
            fontsize=cast.layout.small_pt,
            linespacing=1.0,
            arrowprops={"arrowstyle": "-", "color": cast.text, "lw": 0.6},
            zorder=7,
            gid="d05-shared-label",
        ),
        cast,
    )
    _label(
        ax,
        (window[1] - 0.4, window[3] - 0.4),
        f"lenslet {LENSLET_B},\n{m.lam[0]:.0f} nm: 3\nshared rows,\nwings only",
        cast,
        color=cb,
        ha="right",
        va="top",
        gid="d05-neighbor-rows-label",
    )
    _label(
        ax,
        (float(m.xg[LENSLET_A, 0]) - 1.0, float(m.yg[LENSLET_A, 0]) + 0.8),
        f"lenslet {LENSLET_A}",
        cast,
        color=ca,
        ha="left",
        va="bottom",
    )
    return res


def _flux_panel(ax, m, cast):
    """Extracted bin fluxes with 1-sigma bars, over the input bin fluxes."""
    lam, edges = m.lam, m.edges
    n = len(lam)
    truth_color = cast["scenery"].color
    for j, ch in enumerate((LENSLET_A, LENSLET_B)):
        color = _lenslet_color(ch)
        sd = np.sqrt(np.diagonal(m.pair)[j * n : (j + 1) * n] * m.sigma2)
        for i in range(n):
            ax.plot(
                [edges[i], edges[i + 1]],
                [m.truth[ch, i]] * 2,
                color=truth_color,
                lw=cast.layout.lw,
                solid_capstyle="butt",
                zorder=2,
            )
        ax.errorbar(
            lam,
            m.extracted[ch],
            yerr=sd,
            fmt="o" if j == 0 else "s",
            ms=0.55 * cast.layout.marker_pt,
            color=color,
            mfc=color if j == 0 else "none",
            lw=0.8 * cast.layout.lw,
            capsize=0,
            zorder=3,
        )
        _label(
            ax,
            (edges[-1] + 1.5, m.truth[ch, -1]),
            f"lenslet {ch}",
            cast,
            color=color,
            ha="left",
            va="center",
        )
    ax.set_xlim(edges[0] - 2.0, edges[-1] + 22.0)
    ax.set_ylim(0.0, 1.25 * float(m.extracted[LENSLET_A].max()))
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("bin-integrated flux [arb.]")
    for e in edges:
        ax.axvline(e, color=cast.neutral(0.15), lw=0.5 * cast.layout.lw, zorder=0)
    _boxed(
        ax.annotate(
            "input flux over one bin",
            (0.5 * (edges[0] + edges[1]), m.truth[LENSLET_A, 0]),
            xytext=(0, 16),
            textcoords="offset points",
            color=truth_color,
            ha="left",
            va="bottom",
            fontsize=cast.layout.small_pt,
            arrowprops={"arrowstyle": "-", "color": truth_color, "lw": 0.6},
        ),
        cast,
    )
    k = 6
    _boxed(
        ax.annotate(
            "extracted, 1-sigma bar",
            (lam[k], m.extracted[LENSLET_A, k]),
            xytext=(-10, 18),
            textcoords="offset points",
            color=cast.text,
            ha="right",
            va="bottom",
            fontsize=cast.layout.small_pt,
            arrowprops={"arrowstyle": "-", "color": cast.text, "lw": 0.6},
        ),
        cast,
    )


def second_neighbor_range(pair, n):
    """Range of the correlation between bins two apart, over both lenslets."""
    r = _correlation(pair)
    two = np.concatenate([np.diag(r[:n, :n], 2), np.diag(r[n:, n:], 2)])
    return float(two.min()), float(two.max())


# Label ink on the correlation matrix, whose near-zero cells are near white
# in both modes.
MATRIX_INK = "#1a1a1a"


def _on_matrix(text):
    """Dark text on a light backing: legible on the pale matrix in both modes."""
    text.set_color(MATRIX_INK)
    text.set_path_effects([])
    text.set_bbox(
        {
            "boxstyle": "round,pad=0.2",
            "facecolor": to_rgba("white", 0.85),
            "edgecolor": "none",
        }
    )
    return text


def correlation_summary(pair, n):
    """Neighboring-bin range and largest cross-lenslet |r| of a 2-lenslet pair."""
    r = _correlation(pair)
    adjacent = np.concatenate([np.diag(r[:n, :n], 1), np.diag(r[n:, n:], 1)])
    return float(adjacent.max()), float(adjacent.min()), float(np.abs(r[:n, n:]).max())


def _correlation_panel(ax, m, cast):
    """Correlation of the two lenslets' extracted bins (flattened order)."""
    res = cviz.plot_channel_covariance(
        m.pair,
        wavelengths_nm=m.lam,
        channel_labels=[f"lenslet {LENSLET_A}", f"lenslet {LENSLET_B}"],
        max_ticks=2,
        ax=ax,
    )
    n = len(m.lam)
    hi, lo, cross = correlation_summary(m.pair, n)
    for text in res.artists.get("text", []):
        _on_matrix(text)
    # The label sits in the empty upper-left (cross-lenslet) block, off the
    # diagonal cells, with a leader to one neighboring-bin cell of lenslet 24.
    i = n - 3
    _on_matrix(
        ax.annotate(
            f"neighboring bins:\nr = {hi:.2f} to {lo:.2f}",
            (i + 1, i),
            xytext=(0.5 * n - 0.5, 1.5 * n - 0.5),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=cast.layout.small_pt,
            linespacing=1.0,
            arrowprops={"arrowstyle": "-", "color": MATRIX_INK, "lw": 0.8},
            zorder=7,
            gid="d05-adjacent-r",
        )
    )
    _on_matrix(
        ax.text(
            1.5 * n - 0.5,
            0.5 * n - 0.5,
            f"between\nlenslets:\nlargest |r|\n= {cross:.4f}",
            ha="center",
            va="center",
            fontsize=cast.layout.small_pt,
            linespacing=1.0,
            zorder=7,
            gid="d05-cross-r",
        )
    )
    return res


def build_extraction(layout, cast):
    """Figure: overlapping footprints, correlated bins, extracted fluxes."""
    m = _model()
    with _x64():
        if layout.is_slide:
            # A talk slide carries the mechanism and its consequence only.
            fig, (ax_o, ax_c) = ex.figure(layout, ncols=2, width_ratios=(1.2, 1.0))
            ax_f = None
            # Top-align the two panels so no band opens above the shorter one.
            for ax in (ax_o, ax_c):
                ax.set_anchor("N")
        else:
            fig = plt.figure(figsize=layout.size(5.8), layout="constrained")
            gs = fig.add_gridspec(
                2, 2, height_ratios=(1.35, 1.0), width_ratios=(1.15, 1.0)
            )
            ax_o = fig.add_subplot(gs[0, 0])
            ax_c = fig.add_subplot(gs[0, 1])
            ax_f = fig.add_subplot(gs[1, :])
        _overlap_panel(
            ax_o,
            m,
            cast,
            window=SLIDE_OVERLAP_WINDOW if layout.is_slide else OVERLAP_WINDOW,
        )
        ax_o.set_title("(a) bin footprints share pixels", loc="left")
        _correlation_panel(ax_c, m, cast)
        ax_c.set_title("(b) extracted-bin correlation", loc="left")
        if ax_f is not None:
            _flux_panel(ax_f, m, cast)
            ax_f.set_title("(c) extracted lenslet-bin fluxes", loc="left")
            ex.badge(ax_f, cast, "simulated", loc="upper left")
        ex.badge(ax_o, cast, "simulated", loc="lower left")
        _headline(fig, layout, "Shared pixels make neighboring bins anticorrelated")
    return _plain_text(fig, cast)


# Animation: follow one lenslet across its wavelength bins, then its neighbor

SCAN_ENTRANCE_WINDOW = (17.0, 51.0, 17.0, 47.0)
# The talk frame is taller than the documentation column, so its windows are
# taller too and the panels fill the frame; the traces keep their pixel scale.
SLIDE_SCAN_ENTRANCE_WINDOW = (13.0, 55.0, 11.0, 53.0)
SLIDE_SCAN_WINDOW = (32.0, 97.0, 35.0, 85.0)
NEIGHBOR_FRAMES = 4
DIM = 0.3


def neighbor_offset_px(m):
    """Detector offset of lenslet B's trace from lenslet A's: (along, across)."""
    return (
        float(m.xg[LENSLET_B, 0] - m.xg[LENSLET_A, 0]),
        float(m.yg[LENSLET_B, 0] - m.yg[LENSLET_A, 0]),
    )


def _cell_center(cell):
    """Center of a lenslet-cell polygon (mean of its four corners)."""
    xy = np.asarray(cell.get_xy())
    corners = xy[:-1] if np.allclose(xy[0], xy[-1]) else xy
    return tuple(corners.mean(axis=0))


def _link(fig, ax_from, xy_from, ax_to, cast, channel):
    """A thin connector from an entrance cell to a detector point (no head)."""
    con = ConnectionPatch(
        xyA=xy_from,
        coordsA=ax_from.transData,
        xyB=xy_from,
        coordsB=ax_to.transData,
        color=_lenslet_color(channel),
        lw=0.8 * cast.layout.lw,
        alpha=0.9,
        zorder=9,
    )
    con.set_in_layout(False)
    fig.add_artist(con)
    return con


def build_scan(layout, cast):
    """Animation: a wavelength scan along one trace, then the neighbor's overlap.

    Both lenslets' light is on the detector in every frame; only the
    overlays (highlight, boxes, emphasis) change.
    """
    m = _model()
    n = len(m.lam)
    frames = [("scan", k) for k in range(n)]
    frames += [("neighbor", SCAN_BIN)] * NEIGHBOR_FRAMES
    frames = frames[: layout.n_frames(len(frames))]
    with _x64():
        if layout.is_slide:
            fig, (ax_e, ax_d) = ex.figure(layout, ncols=2, width_ratios=(1.0, 1.5))
        else:
            fig, (ax_e, ax_d) = ex.figure(
                layout, doc_height_in=3.3, ncols=2, width_ratios=(1.0, 1.9)
            )
        ent = _entrance_panel(
            ax_e,
            m,
            cast,
            channels=(LENSLET_A, LENSLET_B),
            window=SLIDE_SCAN_ENTRANCE_WINDOW
            if layout.is_slide
            else SCAN_ENTRANCE_WINDOW,
            zoom_box=False,
            spaxel_tag=False,
            grid_labels=False,
        )
        ax_e.set_title("entrance plane", loc="left")
        window = SLIDE_SCAN_WINDOW if layout.is_slide else DETECTOR_WINDOW
        det = _detector_panel(
            ax_d, m, cast, channels=(LENSLET_A, LENSLET_B), window=window
        )
        _, move = _detector_labels(
            ax_d, m, cast, det, channels=(LENSLET_A, LENSLET_B), detail=False
        )
        ax_d.set_title("detector", loc="left")
        ex.badge(ax_d, cast, "simulated", loc="lower left")
        _tag(
            ax_d,
            (0.985, 0.97),
            "a scan for explanation:\nall bins land at once",
            cast,
            ha="right",
            va="top",
        )
        for ax in (ax_e, ax_d):
            ax.set_anchor("N")
        headline = fig.suptitle("", fontsize=layout.title_pt)
        box_b = _box(ax_d, m.ir, LENSLET_B, 0, _lenslet_color(LENSLET_B))
        box_b_label = _label(
            ax_d,
            (float(m.xg[LENSLET_B, 0]) - 4.5, float(m.yg[LENSLET_B, 0]) + HALF + 0.6),
            f"lenslet {LENSLET_B}, {m.lam[0]:.0f} nm",
            cast,
            color=_lenslet_color(LENSLET_B),
            ha="left",
            va="bottom",
        )
        scatter = det.artists["scatter"]
        full = np.array(scatter.get_edgecolors())
        dim = full.copy()
        dim[n:, 3] = DIM
        trace_b = det.artists["lines"][1]
        cell_a, cell_b = ent.artists["ellipse"][:2]
        # A thin line from each lenslet's entrance cell to the bin footprint
        # of its light that the frame points at.
        # Each link ends at the top-left corner of the box, above the trace
        # and its labels.
        link_a = _link(fig, ax_e, _cell_center(cell_a), ax_d, cast, LENSLET_A)
        link_b = _link(fig, ax_e, _cell_center(cell_b), ax_d, cast, LENSLET_B)
        x0b, y0b, _, hb = _psflet_box(m.ir, LENSLET_B, 0)
        link_b.xy2 = (x0b, y0b + hb)
        along, across = neighbor_offset_px(m)

        def draw(fig, frame):
            beat, k = frame
            both = beat == "neighbor"
            scatter.set_edgecolors(full if both else dim)
            for artist in (trace_b, cell_b):
                artist.set_alpha(1.0 if both else DIM)
            for artist in (box_b, box_b_label, link_b):
                artist.set_visible(both)
            x0, y0, _, h = _psflet_box(m.ir, LENSLET_A, k)
            link_a.xy2 = (x0, y0 + h)
            det.update(k)
            move(k)
            if both:
                headline.set_text(
                    f"Its grid neighbor, lenslet {LENSLET_B}, lands {along:.0f} px "
                    f"along the trace, {across:.0f} px across it: footprints overlap"
                )
            else:
                headline.set_text(
                    f"Follow lenslet {LENSLET_A} across its wavelength bins: "
                    rf"$\lambda$ = {m.lam[k]:.0f} nm"
                )

    return ex.AnimationScene(fig=_plain_text(fig, cast), draw=draw, frames=frames)


INSTRUMENT_CAPTION = (
    "A lenslet integral field spectrograph, from entrance image to detector "
    "traces (clause `optics-ifs-products`). (a) Schematic side view of the "
    "lenslet architecture, after {ref}`Rizzo et al. 2017, Sec. 2.1, Fig. 1 "
    "<source-rizzo2017>`: a square lenslet array in the entrance focal plane "
    "focuses each cell onto a pinhole, and a collimator, prism and camera "
    "re-image every pinhole onto the detector, dispersed along detector +x. "
    "The prism deviates every wavelength toward its base, the shorter ones "
    "more, so 713 nm lands highest; broadband light before the prism is "
    "drawn gray. (b) "
    "Simulated entrance image, log stretch, with every square collection cell. "
    "Each cell integrates the entrance flux of one lenslet, which becomes one "
    "spaxel of the extracted cube ({ref}`Rizzo et al. 2017, Sec. 2.2 "
    "<source-rizzo2017>`); the two outlined cells, lenslets 24 and 31, are "
    "grid neighbors. (c) Simulated detector "
    "response to unit flux in every wavelength bin of the two lenslets, log "
    "stretch. Each open circle is the centroid of one bin footprint, one per "
    "wavelength bin from 605 to 713 nm, and the line through them is the "
    "geometric trace. The box outlines the detector pixels of the 651 nm bin "
    "of lenslet 24. The footprints of the two neighbors share detector rows. "
    "The reference origins of the entrance plane and the detector are drawn "
    "in {ref}`the next figure <fig-explainer-d05-lenslet-ifs-origins>`. "
    "The instrument is synthetic, not an "
    "instrument prescription: a 7-by-7 grid clocked by arctan(1/2), 13.4 "
    "detector pixels per lenslet pitch, Moffat PSFlets, and ten bins at "
    "resolving power 50."
)
INSTRUMENT_ALT = (
    "Three panels. Top: a side-view schematic of a lenslet integral field "
    "spectrograph, with light arriving from the left onto a column of small "
    "lenses in the entrance focal plane, one highlighted lens focusing onto a "
    "pinhole, then a collimator, a prism with its apex up that bends the gray "
    "broadband beam down toward its base into rays from 605 nanometers (bent "
    "most, lowest) to 713 nanometers (bent least, highest), a camera, and a "
    "detector, with an arrow marking detector +x upward. Bottom left: a "
    "simulated entrance image with a rotated grid of square lenslet cells; "
    "the neighboring cells of lenslets 24 and 31 are outlined. Bottom right: "
    "a simulated detector image with two parallel horizontal traces, one per "
    "lenslet, each marked by ten centroid circles from 605 to 713 "
    "nanometers, with a box around the 651 nanometer bin footprint of "
    "lenslet 24 and its centroid."
)
ORIGINS_CAPTION = (
    "Three reference origins of a lenslet integral field spectrograph, each "
    "in its own frame (clause `optics-ifs-products`; pixel origins as in "
    "clause `optics-pixel-directions`). (a) The simulated entrance image of "
    "{ref}`the previous figure <fig-explainer-d05-lenslet-ifs-instrument>`, "
    "with the region drawn in (b) boxed. (b) The "
    "boxed region of (a) on its cube pixel grid, for a cube of "
    "$n_\\mathrm{cube}$ = 64 pixels on a side. The optical center belongs to "
    "the cube and sits at its geometric center (31.5, 31.5). The example "
    "implementation drawn here, coronachrome, places lenslet (0, 0) at "
    "(32, 32), half a pixel away on each axis; the {ref}`limitations page "
    "<limitations-optics>` records that difference. The arrow runs from the "
    "optical center to the lenslet-grid origin, +0.5 pixel on each axis. "
    "(c) Simulated detector "
    "response to unit flux in every wavelength bin of lenslet 24, grid "
    "(0, 0), log stretch. The detector trace origin is the point coronachrome "
    "anchors every trace to, the detector center $n_\\mathrm{det}/2$ = "
    "(60, 60) of the 120-pixel detector; the chapter sets no convention for "
    "this point. With no constant dispersion term, lenslet (0, 0) lands there "
    "at the 660 nm reference wavelength, which is not a bin center, so the "
    "origin lies between the centroids of the 651 nm and 663 nm bins. The "
    "three origins are separate calibrated quantities: taking one for another "
    "shifts every spaxel, and so every planet position, by half a cube pixel "
    "on each axis (1/12 of a lenslet pitch here), or misplaces every bin "
    "footprint on the detector. Simulated, with the synthetic instrument of "
    "the previous figure."
)
ORIGINS_ALT = (
    "Three panels. Top left, small: the simulated entrance image with its "
    "rotated grid of square lenslet cells, the cells of lenslets 24 and 31 "
    "outlined, and a small box near the image center marking the zoomed "
    "region, tagged as carried from the previous figure. Top right: a zoom "
    "on the cube pixel grid, with a cross at the optical center (31.5, "
    "31.5), labeled as the cube size minus one over two, a square at the "
    "lenslet-grid origin (32, 32), labeled as half the cube size in "
    "coronachrome, and an arrow from the cross to the square labeled +0.5 "
    "pixel on each axis. Bottom: a simulated detector image of one "
    "horizontal trace, lenslet 24, grid (0, 0), with ten centroid circles; "
    "a pair of tick marks at (60, 60), between the circles of the 651 and "
    "663 nanometer bins, marks the detector trace origin, labeled as half "
    "the detector size in coronachrome, where lenslet (0, 0) lands at the "
    "660 nanometer reference wavelength."
)
PLACEMENT_CAPTION = (
    "PSFlet templates, calibrated placement and detector-edge capture under "
    "the proposed template profile (clause `optics-ifs-products`). (a) A stored "
    "template: the pixel-integrated PSFlet at the 651 nm bin center, a single "
    "wavelength, tabulated against offset from its own centroid, so its "
    "centroid is at (0, 0). (b) Placement on the detector: the bin footprint "
    "is centered on the geometric trace point plus a separate calibrated "
    "centroid correction, applied once. It is wider along x than the template "
    "because the PSFlet is smeared across the 2.55 pixels the bin spans along "
    "the dispersion; this bin footprint is the narrowband spot of "
    "{ref}`Brandt et al. 2017, Sec. 5.2 <source-brandt2017>`. The correction "
    "drawn, (0.56, -0.79) pixels, is illustrative. A template that still "
    "carried that displacement would receive the correction again (dashed "
    "arrow) and land at the circled cross, shifted twice; the "
    "{ref}`limitations page <limitations-optics>` records this case for "
    "physical template packs. (c) A trace that runs off the detector. The "
    "image comes from a simulation on a wider detector, so the light past the "
    "edge (hatched) is drawn where it would have landed. That light is lost, "
    "and renormalizing the surviving pixels must not restore it; this is the "
    "handbook's convention, and the current implementation does renormalize "
    "them, as the {ref}`limitations page <limitations-optics>` records. The "
    "hard cutoff 4 pixels above and below the trace is the finite numerical "
    "support of the footprint, not an optical edge. Simulated, with the "
    "synthetic instrument of the first figure."
)
PLACEMENT_ALT = (
    "Three panels. Top left: a stored PSFlet template on a fine grid of "
    "offsets from its centroid, with a cross at offset (0, 0). Top right: a "
    "zoom of one simulated bin footprint on detector pixels, elongated along "
    "x, with a horizontal geometric trace line, a dot at the geometric trace "
    "point, a solid arrow labeled calibrated correction to a cross marking the "
    "placed centroid, and a dashed arrow repeating the same step to a circled "
    "cross labeled as the position if the correction were applied again. "
    "Bottom: a simulated trace of lenslet 43 running to the right edge of the "
    "detector, with a dashed detector edge at x = 119.5, a hatched region "
    "beyond it where the lost light would have landed, and a note that the "
    "footprint's 4-pixel vertical cutoff is numerical."
)
EXTRACTION_CAPTION = (
    "Why extracted lenslet-bin fluxes share uncertainty (clauses "
    "`optics-ifs-products` and `radiometry-spectral-covariance`). (a) Simulated "
    "detector response: the 651 nm and 663 nm bin footprints of lenslet 24 "
    "share two thirds of their pixels (hatched). A count in a shared pixel can "
    "be assigned to either bin, so raising one estimate lowers the other: "
    "neighboring bins are anticorrelated, the negative covariance a "
    "chi-squared extraction produces ({ref}`Brandt et al. 2017, Sec. 5.2 "
    "<source-brandt2017>`). The 605 nm footprint of the grid neighbor, lenslet "
    "31, shares only three detector rows, where the PSFlet wings are faint. "
    "(b) Correlation of the unregularized least-squares estimate with uniform "
    "pixel weights, the covariance $[H^\\mathsf{T}H]^{-1}$ scaled to unit "
    "diagonal ({ref}`Riley et al. 2006, Sec. 27.6, eqs. 27.98-27.99 "
    "<source-riley2006>`), in the flattened (channel, wavelength) order; entry "
    "[i, j] is drawn at x = j, y = i with the origin at the lower left. The "
    "matrix borrows the diverging residual map for a signed correlation, since "
    "no correlation role exists yet. Neighboring bins have r = -0.40 to -0.47. "
    "Bins two apart have r = +0.16 to +0.19: each is anticorrelated with the "
    "bin between them, so the sign alternates with separation. The largest "
    "correlation between the two lenslets is 0.0007 in this "
    "layout, and it grows as traces crowd. A selected within-lenslet block, "
    "the form a chi-squared extraction returns, omits it. (c) Extracted fluxes "
    "(points with 1-sigma bars from the same covariance) against the input "
    "bin-integrated flux of each lenslet (bars spanning each bin), for one "
    "simulated readout with uniform Gaussian pixel noise, in arbitrary units. "
    "Simulated, with the synthetic instrument of the first figure."
)
EXTRACTION_ALT = (
    "Three panels. Top left: a simulated detector zoom with two horizontal "
    "traces; boxes outline the 651 and 663 nanometer bin footprints of "
    "lenslet 24, whose shared pixels are hatched and labeled anticorrelated, "
    "and the 605 nanometer footprint of lenslet 31, labeled as sharing three "
    "rows in the wings. Top right: a 20-by-20 correlation matrix in two "
    "lenslet blocks; within each block the diagonal is +1 and the first "
    "off-diagonals are about -0.4, the second off-diagonals about +0.2, and "
    "the blocks between the lenslets are "
    "near zero, with largest magnitude 0.0007. Bottom: extracted fluxes with "
    "error bars for lenslets 24 and 31 over ten wavelength bins from 605 to "
    "713 nanometers, scattered about horizontal bars showing the input flux "
    "of each bin."
)
SCAN_CAPTION = (
    "Following one lenslet across its wavelength bins (clause "
    "`optics-ifs-products`). Left: the entrance cells of lenslet 24 and its "
    "grid neighbor, lenslet 31. Right: the simulated detector response of "
    "both lenslets, log stretch, the same in every frame. The highlighted "
    "centroid, bin footprint box and wavelength readout step through the ten "
    "bin centers from 605 to 713 nm, one per second. This is a scan of the "
    "wavelength being pointed at, not a time sequence: every bin reaches the "
    "detector at once, and each microspectrum is the sum of its monochromatic "
    "PSFlets ({ref}`Brandt et al. 2017, Sec. 4, opening paragraph "
    "<source-brandt2017>`). A thin line in the lenslet's color joins its "
    "entrance cell to the bin footprint in view. The second part brings "
    "forward the neighbor's trace, which lands 12 pixels along the trace and "
    "6 pixels across it, and outlines its 605 nm footprint, which overlaps "
    "the highlighted one. The image, color scale and axes are fixed across "
    "frames; only the overlays change."
)
SCAN_ALT = (
    "Animation in two parts. Left, a simulated entrance image with the "
    "outlined cells of lenslets 24 and 31; right, a simulated detector image "
    "with both lenslets' horizontal traces. In the first part lenslet 31 is "
    "dimmed while a cross, a footprint box and a readout step along the trace "
    "of lenslet 24 from 605 to 713 nanometers, and a thin line joins the "
    "entrance cell of lenslet 24 to the box. In the second part lenslet 31 "
    "is drawn at full strength and a second box, joined to its entrance cell, "
    "marks its 605 nanometer footprint, 12 pixels along the trace and 6 "
    "pixels across it, overlapping the first."
)


FIGURES = [
    ex.FigureSpec(
        slug="d05-lenslet-ifs-instrument",
        build=build_instrument,
        caption=INSTRUMENT_CAPTION,
        alt=INSTRUMENT_ALT,
        status="schematic side view; simulated array panels",
        params=PARAMS,
    ),
    ex.FigureSpec(
        slug="d05-lenslet-ifs-origins",
        build=build_origins,
        caption=ORIGINS_CAPTION,
        alt=ORIGINS_ALT,
        status="simulated; grid and trace origins as coronachrome implements them",
        params=PARAMS,
    ),
    ex.FigureSpec(
        slug="d05-lenslet-ifs-placement",
        build=build_placement,
        caption=PLACEMENT_CAPTION,
        alt=PLACEMENT_ALT,
        status="simulated; illustrative centroid correction",
        params={**PARAMS, "centroid_correction_px": [list(c) for c in CORRECTION_PX]},
    ),
    ex.FigureSpec(
        slug="d05-lenslet-ifs-extraction",
        build=build_extraction,
        caption=EXTRACTION_CAPTION,
        alt=EXTRACTION_ALT,
        status="simulated",
        params={**PARAMS, "noise_sigma": NOISE_SIGMA, "noise_seed": NOISE_SEED},
    ),
]
ANIMATIONS = [
    ex.AnimationSpec(
        slug="d05-lenslet-ifs-scan",
        build=build_scan,
        ground="narration",
        caption=SCAN_CAPTION,
        alt=SCAN_ALT,
        status="simulated; wavelength scan for explanation",
        params=PARAMS,
        fps=1,
        hold_s=(1.0, 2.0),
        preview_frames=(0, 4, 9, 11),
    ),
]
