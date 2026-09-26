"""Optical planes and the quantities that live on them.

One illustrative vortex coronagraph train, drawn unfolded in side view with a
face-on thumbnail of the quantity on each plane. Every thumbnail is an array
propagated through a small physicaloptix path: entrance pupil, a pupil-plane
OPD stage standing for the deformable mirror (it acts on the entrance array,
because conjugate pupils share one array in the model), a charge-6
multi-scale vortex, a Lyot stop and a Fraunhofer transform to the image
plane. A separate detector model (physicaloptix ``read_detector``) turns the
final intensity into simulated counts. A third figure shows the complex
field itself, amplitude and phase, before and after each element.

The side view is drawn here rather than with ``eyepiece.rail``: the rail
places a lens right after every plane and interpolates the beam linearly
between planes, so it cannot draw a pupil relay, a collimated beam after a
collimator, or a stop inside a collimated beam.

Motion ground of the animation: narration. Its frames first step the
highlight through the planes (a narrated tour), then step a deformable
mirror command through a model-parameter sequence. Neither beat is a time
series, and every scale is pinned across all frames.
"""

import functools
import math

import eyepiece as ep
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.patches import Circle, Rectangle
from matplotlib.text import Text
from physicaloptix import viz as po_viz

from explainers import _common as ex

# Scientific inputs. The architecture follows the PSF-and-speckles tutorial
# (gray-pixel disk, charge-6 vortex, 80 percent Lyot stop) on a smaller grid.
PARAMS = {
    "npup": 64,
    "nfoc": 48,
    "pixscale_lod": 0.5,
    "wavelength_nm": 550.0,
    "aperture_radius_d": 0.5,
    "lyot_radius_d": 0.4,
    "vortex_charge": 6,
    "gray_subsamples": 8,
    # Width of the pupil-conjugate thumbnails, in units of D. The path keeps
    # a D-wide array; the thumbnails evaluate the same operators on a wider
    # grid so the light the vortex moves outside the pupil is visible.
    "thumbnail_width_d": 2.0,
    # Static wavefront error at the DM plane: W0 = a0 cos(2 pi k x / D).
    "static_ripple_cycles_per_d": 6.0,
    "static_ripple_nm": 1.0,
    # Extra DM ripple along y used in the perturbation still.
    "probe_ripple_cycles_per_d": 6.0,
    "probe_ripple_nm": 1.0,
    # Detector model: photons per second in a pixel at the unocculted
    # on-axis peak (intensity 1), quantum efficiency 1, detector pixels
    # equal to the focal samples.
    "detector_peak_rate_per_s": 1.0e7,
    "detector_exposure_s": 1.0,
    "detector_read_noise_e": 1.0,
    "detector_seed": 4,
    # Display norms (fixed, shared by every figure and frame).
    "intensity_floor": 1.0e-10,
    "intensity_ceiling": 1.0e-4,
    "delta_unit": 1.0e-5,
    # Phase is drawn only where the intensity exceeds this fraction of the
    # unocculted peak; elsewhere there is too little light to define it.
    "phase_min_intensity": 3.0e-7,
}

# Animation schedule: tour stops, then the DM command sweep c in units of a0.
TOUR_STOPS = ("pupil", "dm", "fpm", "lyot", "image", "detector")
SWEEP_END = -2.0
OPTICAL_KEYS = ("pupil", "dm", "fpm", "lyot", "image")

PLANE_LABELS = {
    "pupil": "entrance\npupil",
    "dm": "DM\nplane",
    "fpm": "focal\nplane",
    "lyot": "Lyot\nplane",
    "image": "image\nplane",
}
ELEMENTS = {
    "pupil": "aperture",
    "dm": "deformable\nmirror",
    "fpm": "vortex\nphase mask",
    "lyot": "Lyot stop\n(pupil stop)",
    "image": "detector",
}
TOUR_TITLES = {
    "pupil": r"Entrance pupil: the aperture multiplies the field by its transmission $A$",
    "dm": r"DM plane: an OPD $W$ multiplies the field by $\exp(+i2\pi W/\lambda)$",
    "fpm": r"Focal plane: a Fourier transform of the pupil field; the vortex adds the phase $6\theta$",
    "lyot": r"Lyot plane: the vortex moved starlight outside the pupil edge; the stop blocks it",
    "image": r"Image plane: a last Fourier transform forms $E$; the detector senses $I=|E|^2$",
    "detector": r"Detector: a separate model turns $I$ into counts, under its own contract",
}
SWEEP_TITLE = (
    r"DM command $c$ changes the field; the intensity change $\Delta I$ is signed"
)


# The propagated example


def gray_disk(coords, dx, radius, subsamples):
    """Fraction of each square cell inside a circle, by subsampling."""
    offsets = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * dx
    inside = np.zeros((coords.size, coords.size))
    for oy in offsets:
        for ox in offsets:
            x, y = np.meshgrid(coords + ox, coords + oy)
            inside += np.hypot(x, y) <= radius
    return inside / subsamples**2


def pad_to_thumbnail(array, fill=0.0):
    """Pad a D-wide pupil array onto the wider thumbnail grid."""
    n = PARAMS["npup"]
    extra = round(0.5 * (PARAMS["thumbnail_width_d"] - 1.0) * n)
    return np.pad(array, extra, constant_values=fill)


@functools.cache
def optics():
    """Build the physicaloptix grids, elements and normalization once."""
    import physicaloptix as po

    p = PARAMS
    n = p["npup"]
    with jax.enable_x64(True):
        pupil_grid = po.Grid.pupil(n)
        focal_grid = po.Grid.focal(p["nfoc"], p["pixscale_lod"])
        coords = np.asarray(pupil_grid.coords)
        dx = float(pupil_grid.dx)
        aperture = gray_disk(coords, dx, p["aperture_radius_d"], p["gray_subsamples"])
        stop = gray_disk(coords, dx, p["lyot_radius_d"], p["gray_subsamples"])
        x, y = np.meshgrid(coords, coords)
        # OPD modes per nanometer of coefficient: a cosine ripple along x
        # and a sine ripple along y.
        modes = np.stack(
            [
                np.cos(2.0 * np.pi * p["static_ripple_cycles_per_d"] * x),
                np.sin(2.0 * np.pi * p["probe_ripple_cycles_per_d"] * y),
            ]
        )
        entrance = po.Field(
            data=jnp.asarray(aperture, dtype=complex),
            grid=pupil_grid,
            plane=po.PlaneKind.PUPIL,
        )
        camera = po.Fraunhofer(pupil_grid, focal_grid)
        vortex = po.MultiScaleVortex.build(charge=p["vortex_charge"], npup=n)
        lyot = po.SampledOptic(
            transmission=jnp.asarray(stop), grid=pupil_grid, plane=po.PlaneKind.PUPIL
        )
    # The unocculted on-axis peak |F[A](0)|^2: the continuous transform at
    # u = 0 is the aperture integral. The focal grid is half-pixel offset,
    # so no sample sits there; the brightest sample is about 0.73 of it.
    peak = float((aperture.sum() * dx * dx) ** 2)
    m = round(p["thumbnail_width_d"] * n)
    wide_coords = (np.arange(m) - 0.5 * m + 0.5) / n
    u = np.asarray(focal_grid.coords)
    uu, vv = np.meshgrid(u, u)
    mask_phase = np.angle(np.exp(1j * p["vortex_charge"] * np.arctan2(vv, uu)))
    return {
        "po": po,
        "pupil_grid": pupil_grid,
        "focal_grid": focal_grid,
        "pupil_coords": coords,
        "wide_coords": wide_coords,
        "focal_coords": u,
        "aperture": aperture,
        "stop": stop,
        "modes": modes,
        "entrance": entrance,
        "camera": camera,
        "vortex": vortex,
        "lyot": lyot,
        "peak": peak,
        "mask_phase": mask_phase,
    }


def run(opd_nm):
    """Propagate one DM-plane OPD map (nm, on the pupil grid) through the path.

    Returns:
        A dict of numpy arrays: ``opd`` (nm), ``entrance_field`` (the
        field leaving the DM stage, on the path's D-wide array),
        ``lyot_field`` (the field
        arriving at the Lyot stop, on the path's D-wide array),
        ``lyot_wide`` (the same field on the wider thumbnail grid),
        ``image_field`` (in units of the square root of the unocculted
        peak), ``intensity`` (the unocculted on-axis peak is one) and
        ``counts`` (simulated detector electrons).
    """
    o = optics()
    po = o["po"]
    p = PARAMS
    with jax.enable_x64(True):
        basis = po.ModeBasis(
            B=jnp.asarray(np.asarray(opd_nm, dtype=float)[None]),
            coeffs=jnp.asarray([1.0]),
        )
        screen = po.PhaseScreen(
            basis, o["pupil_grid"], wavelength_nm=p["wavelength_nm"]
        )
        path = po.OpticalPath(
            stages=(
                po.Stage(name="dm", op=screen),
                po.Stage(name="vortex", op=o["vortex"]),
                po.Stage(name="lyot", op=o["lyot"]),
                po.Stage(name="image", op=o["camera"]),
            )
        )
        out, taps = path.propagate(o["entrance"], taps=("dm", "vortex"))
        # The vortex stage's own operator, evaluated on the wider grid: the
        # same ladder of focal masks, transformed back onto more pupil
        # samples at the same spacing.
        after_dm = taps["dm"].data
        x = o["vortex"].pupil_coords
        wide = jnp.asarray(o["wide_coords"])
        lyot_wide = 0.0
        for u, mask in o["vortex"].levels:
            lyot_wide = lyot_wide + po.cmft_bwd(
                po.cmft_fwd(after_dm, x, u) * mask, wide, u
            )
        intensity = out.intensity() / o["peak"]
        counts = po.read_detector(
            intensity,
            jax.random.PRNGKey(p["detector_seed"]),
            flux=p["detector_peak_rate_per_s"],
            exposure_time=p["detector_exposure_s"],
            read_noise_e=p["detector_read_noise_e"],
            method="gaussian",
        )
    return {
        "opd": np.asarray(opd_nm, dtype=float),
        "entrance_field": np.asarray(after_dm),
        "lyot_field": np.asarray(taps["vortex"].data),
        "lyot_wide": np.asarray(lyot_wide),
        "image_field": np.asarray(out.data) / math.sqrt(o["peak"]),
        "intensity": np.asarray(intensity),
        "counts": np.asarray(counts),
    }


@functools.cache
def propagate(coeffs_nm):
    """Propagate the OPD ``coeffs . modes`` (cached by the coefficients)."""
    modes = optics()["modes"]
    return run(np.tensordot(np.asarray(coeffs_nm, dtype=float), modes, axes=1))


def static_coeffs(scale=0.0):
    """Total OPD coefficients with a DM command of ``scale`` times a0 along x."""
    a0 = PARAMS["static_ripple_nm"]
    return (a0 * (1.0 + scale), 0.0)


def probe_coeffs():
    """Total OPD after the perturbation command: x ripple cancelled, y added."""
    return (0.0, PARAMS["probe_ripple_nm"])


# Display helpers


def pupil_mask(o):
    """NaN outside the aperture, for pupil-plane maps."""
    return np.where(o["aperture"] > 0.0, 1.0, np.nan)


def wide_opd(state):
    """The OPD on the thumbnail grid, blank outside the aperture."""
    o = optics()
    return pad_to_thumbnail(state["opd"] * pupil_mask(o), np.nan)


def phase_field(field):
    """The image field with dark pixels zeroed, so its phase is masked there."""
    bright = np.abs(field) ** 2 >= PARAMS["phase_min_intensity"]
    return np.where(bright, field, 0.0)


def floored(intensity):
    """Intensity clipped to the display floor."""
    return np.clip(intensity, PARAMS["intensity_floor"], None)


def pupil_extent(o):
    """Pixel-edge extent of the pupil grid, in units of D."""
    return ep.extent_lod(o["pupil_coords"])


def focal_extent(o):
    """Pixel-edge extent of the focal grid, in lambda/D."""
    return ep.extent_lod(o["focal_coords"])


def text_color(cast):
    """Body text color, for the labels a reader must read first."""
    return cast.text


# The side view


def draw_rail(
    ax, cast, layout, xp, y0, half, *, x_start, font, details=True, small=None
):
    """Draw the unfolded train in side view, in inch coordinates.

    Args:
        ax: Axes whose data units are inches (equal aspect).
        cast: The active cast.
        layout: ``ex.DOC`` or ``ex.SLIDE``.
        xp: Plane positions by key, inches.
        y0: Optical axis height, inches.
        half: Half-height of the collimated beam, inches.
        x_start: Where the incoming beam starts, inches.
        font: Font size for plane labels, points.
        details: Also draw the element labels and the gap labels.

    Returns:
        A dict with ``lines`` and ``text`` (plane lines and labels, in
        ``OPTICAL_KEYS`` order) and ``elements`` (element labels by key).
    """
    p = PARAMS
    small = layout.small_pt if small is None else small
    pupil, dm, fpm, lyot, image = (xp[k] for k in OPTICAL_KEYS)
    shrink = p["lyot_radius_d"] / p["aperture_radius_d"]
    waist = 0.035 * half
    lens1 = pupil + 0.25 * (dm - pupil)
    relay_focus = 0.5 * (pupil + dm)
    lens2 = dm - 0.25 * (dm - pupil)
    lens3 = 0.5 * (dm + fpm)
    lens4 = 0.5 * (fpm + lyot)
    lens5 = 0.5 * (lyot + image)
    # Collimated where a star at infinity gives a parallel beam, focused at
    # the relay focus, the mask and the image; the stop trims the beam.
    points = [
        (x_start, half),
        (lens1, half),
        (relay_focus, waist),
        (lens2, half),
        (lens3, half),
        (fpm, waist),
        (lens4, half),
        (lyot, half),
        (lyot, shrink * half),
        (lens5, shrink * half),
        (image, waist),
    ]
    bx = np.array([q[0] for q in points])
    bh = np.array([q[1] for q in points])
    beam = cast["scenery"].color
    ax.fill_between(bx, y0 - bh, y0 + bh, color=beam, alpha=0.18, lw=0, zorder=1)
    for sign in (1.0, -1.0):
        ax.plot(bx, y0 + sign * bh, color=beam, lw=0.6 * cast.layout.lw, zorder=1)
    ax.plot(
        [x_start, image],
        [y0, y0],
        color=cast.neutral(0.3),
        lw=0.5 * cast.layout.lw,
        ls=":",
        zorder=1,
    )
    for lx, h in (
        (lens1, half),
        (lens2, half),
        (lens3, half),
        (lens4, half),
        (lens5, shrink * half),
    ):
        ex.lens(ax, (lx, y0), 2.3 * h, cast, thickness=0.13)

    hardware = cast["aperture"]
    stop_lw = 0.55 * hardware.lw
    for sign in (1.0, -1.0):
        ax.plot(
            [pupil, pupil],
            [y0 + sign * half, y0 + sign * 1.45 * half],
            color=hardware.color,
            lw=stop_lw,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            [lyot, lyot],
            [y0 + sign * shrink * half, y0 + sign * 1.45 * half],
            color=hardware.color,
            lw=stop_lw,
            solid_capstyle="butt",
            zorder=3,
        )
    optics_color = cast["optics"].color
    # Deformable mirror: a plate across the beam with a rippled face.
    ax.add_patch(
        Rectangle(
            (dm - 0.05 * half, y0 - 1.2 * half),
            0.1 * half,
            2.4 * half,
            facecolor=optics_color,
            edgecolor="none",
            zorder=3,
        )
    )
    ys = np.linspace(y0 - 1.2 * half, y0 + 1.2 * half, 60)
    ax.plot(
        dm - 0.09 * half + 0.04 * half * np.sin(ys * 9.0 * math.pi / half),
        ys,
        color=optics_color,
        lw=0.6 * cast.layout.lw,
        zorder=3,
    )
    # Vortex mask: a transparent phase plate spanning the focal plane.
    ax.add_patch(
        Rectangle(
            (fpm - 0.04 * half, y0 - 0.75 * half),
            0.08 * half,
            1.5 * half,
            facecolor=optics_color,
            edgecolor="none",
            alpha=0.7,
            zorder=3,
        )
    )
    # Detector: its sensitive surface lies in the image plane.
    ex.region(
        ax, "detector", Rectangle((image, y0 - 0.5 * half), 0.28 * half, half), cast
    )

    plain = cast.neutral(0.55)
    lines, texts = [], []
    for key in OPTICAL_KEYS:
        x = xp[key]
        (line,) = ax.plot(
            [x, x],
            [y0 - 1.55 * half, y0 + 1.55 * half],
            color=plain,
            lw=1.0,
            ls="--",
            zorder=2,
        )
        text = ex.halo(
            ax.text(
                x,
                y0 + 1.62 * half,
                PLANE_LABELS[key] if details else PLANE_LABELS[key].replace("\n", " "),
                ha="center",
                va="bottom",
                color=plain,
                fontsize=font,
                linespacing=1.0,
            ),
            cast,
        )
        lines.append(line)
        texts.append(text)

    ex.arrow(
        ax,
        (x_start, y0),
        (x_start + 0.55 * (pupil - x_start), y0),
        "ray",
        cast,
        source="star",
    )
    elements = {}
    if details:
        ex.halo(
            ax.text(
                x_start - 0.04,
                y0,
                "starlight",
                ha="right",
                va="center",
                color=cast["star"].color,
                fontsize=small,
            ),
            cast,
        )
        for key in OPTICAL_KEYS:
            x = xp[key] + (0.14 * half if key == "image" else 0.0)
            elements[key] = ex.halo(
                ax.text(
                    x,
                    y0 - 1.62 * half,
                    ELEMENTS[key],
                    ha="center",
                    va="top",
                    color=optics_color,
                    fontsize=small,
                    linespacing=1.0,
                ),
                cast,
            )
        # What the model does across each gap between planes.
        for x, word in (
            (relay_focus, "relay"),
            (lens3, "FT"),
            (lens4, "FT"),
            (lens5, "FT"),
        ):
            ex.note(
                ax,
                (x, y0 + 1.2 * half),
                word,
                cast,
                ha="center",
                va="bottom",
                fontsize=small,
            )
    return {"lines": lines, "text": texts, "elements": elements}


def highlight_rail(rail, keys, cast):
    """Pick out the rail planes in ``keys`` (text color, bold, solid line)."""
    for key, line, text in zip(OPTICAL_KEYS, rail["lines"], rail["text"], strict=True):
        on = key in keys
        color = cast.text if on else cast.neutral(0.55)
        line.set_color(color)
        line.set_linewidth(2.0 if on else 1.0)
        line.set_linestyle("-" if on else "--")
        text.set_color(color)
        text.set_fontweight("bold" if on else "normal")


# The train


def _geometry(layout, width_in, height_in, scale):
    """Column centers, card size, beam size and row heights, in inches."""
    sp = scale * layout.small_pt / 72.0
    fp = scale * layout.font_pt / 72.0
    header = 0.95 if not layout.is_slide else 2.0
    extra = 0.30 if not layout.is_slide else 0.6
    colw = (width_in - header - extra - 0.08 * width_in / 7.2) / 6.0
    xs = [header + (i + 0.5) * colw for i in range(5)]
    xs.append(header + 5.5 * colw + extra)
    bracket = 4.9 * sp
    card_label = 3.4 * sp
    tags = 1.7 * sp
    element = 2.6 * sp
    plane_label = 2.5 * (fp if layout.is_slide else sp)
    fixed = bracket + card_label + tags + element + plane_label + 0.1
    room = height_in - fixed
    card = min(0.8 * colw, 0.72 * room)
    half = min(0.3 * card, (room - card) / 3.2)
    spare = max(0.0, room - card - 3.2 * half)
    card_y = bracket + card_label + 0.5 * card + 0.3 * spare
    axis_y = card_y + 0.5 * card + tags + element + 1.6 * half + 0.6 * spare
    return {
        "header": header,
        "colw": colw,
        "xs": xs,
        "card": card,
        "half": half,
        "bracket_y": bracket * 0.8,
        "card_y": card_y,
        "tag_y": card_y + 0.5 * card + 0.35 * sp,
        "axis_y": axis_y,
        "sp": sp,
    }


def _label_card(card, text, cast, font):
    card.label.set_text(text)
    card.label.set_color(text_color(cast))
    card.label.set_fontsize(font)
    card.label.set_linespacing(1.0)


def draw_train(fig, layout, cast, state, *, bottom_in, height_in, scale=1.0):
    """Draw the train with its plane thumbnails in a band of ``fig``.

    Args:
        fig: The figure, with no layout engine.
        layout: ``ex.DOC`` or ``ex.SLIDE``.
        cast: The active cast.
        state: A ``propagate`` result for the thumbnails.
        bottom_in: Bottom of the band, inches from the figure bottom.
        height_in: Height of the band, inches.
        scale: Text scale for this band.

    Returns:
        A dict of handles an animation mutates: ``rail``, ``cards``,
        ``elements`` and ``overlay``.
    """
    o = optics()
    p = PARAMS
    fw, fh = fig.get_size_inches()
    g = _geometry(layout, fw, height_in, scale)
    sp = g["sp"]
    small = scale * layout.small_pt
    overlay = fig.add_axes([0, bottom_in / fh, 1, height_in / fh], zorder=1)
    overlay.set(xlim=(0, fw), ylim=(0, height_in))
    overlay.axis("off")

    xs = g["xs"]
    xp = dict(zip(OPTICAL_KEYS, xs[:5], strict=True))
    rail = draw_rail(
        overlay,
        cast,
        layout,
        xp,
        g["axis_y"],
        g["half"],
        x_start=g["header"] + 0.05,
        font=small if not layout.is_slide else scale * layout.font_pt,
        small=small,
    )

    size = g["card"]
    cy = g["card_y"]
    thumb = o["wide_coords"].size
    cards = {
        "pupil": ex.plane_card(
            overlay,
            (xs[0], cy),
            size,
            pad_to_thumbnail(o["aperture"]),
            cast,
            role="pupil",
            vmin=0.0,
            vmax=1.0,
            label=" ",
        ),
        "dm": ex.plane_card(
            overlay,
            (xs[1], cy),
            size,
            wide_opd(state),
            cast,
            role="opd",
            norm="symmetric",
            vmax=p["static_ripple_nm"],
            label=" ",
        ),
        "fpm": ex.plane_card(
            overlay,
            (xs[2], cy),
            size,
            o["mask_phase"],
            cast,
            role="phase",
            vmin=-np.pi,
            vmax=np.pi,
            label=" ",
        ),
        "lyot": ex.plane_card(
            overlay,
            (xs[3], cy),
            size,
            np.abs(state["lyot_wide"]),
            cast,
            role="pupil",
            vmin=0.0,
            vmax=1.0,
            label=" ",
        ),
        "image": ex.plane_card(
            overlay,
            (xs[4], cy),
            size,
            floored(state["intensity"]),
            cast,
            role="intensity",
            norm="log",
            vmin=p["intensity_floor"],
            vmax=p["intensity_ceiling"],
            label=" ",
        ),
        "detector": ex.plane_card(
            overlay,
            (xs[5], cy),
            size,
            state["counts"],
            cast,
            role="readouts",
            vmin=0.0,
            vmax=float(state["counts"].max()),
            label=" ",
        ),
    }
    labels = {
        "pupil": "transmission $A$\n(2$D$ wide)",
        "dm": "OPD $W$\n(scale $\\pm$1 nm)",
        "fpm": "\nvortex phase,\n6 turns of $2\\pi$",
        "lyot": "field $|E|$;\nedge solid,\nstop dashed",
        "image": "intensity $I$\n(log scale)",
        "detector": "counts, simulated\n(linear scale)",
    }
    for key, card in cards.items():
        _label_card(card, labels[key], cast, small)
    # Pupil edge and Lyot stop on the Lyot-plane thumbnail, in its pixels.
    n = p["npup"]
    center = 0.5 * (thumb - 1)
    for radius, ls in ((p["aperture_radius_d"], "-"), (p["lyot_radius_d"], "--")):
        cards["lyot"].ax.add_patch(
            Circle(
                (center, center),
                radius * n,
                fill=False,
                ec=cast["optics"].color,
                ls=ls,
                lw=cast.layout.lw,
            )
        )
    # Cyclic key for the mask phase, just under its card.
    key_h = 0.45 * sp
    key_w = 0.42 * size
    key_ax = overlay.inset_axes(
        [xs[2] - 0.5 * key_w, cy - 0.5 * size - 0.35 * sp - key_h, key_w, key_h],
        transform=overlay.transData,
    )
    key_ax.imshow(
        np.linspace(-np.pi, np.pi, 64)[None, :],
        cmap=ex.image_cmap("phase"),
        aspect="auto",
        interpolation="nearest",
    )
    key_ax.set_xticks([])
    key_ax.set_yticks([])
    for x, word, ha in (
        (-0.5 * key_w - 0.03 * size, r"$-\pi$", "right"),
        (0.5 * key_w + 0.03 * size, r"$\pi$", "left"),
    ):
        overlay.text(
            xs[2] + x,
            cy - 0.5 * size - 0.35 * sp - 0.5 * key_h,
            word,
            ha=ha,
            va="center",
            fontsize=0.9 * small,
            color=text_color(cast),
        )

    # A detector is not an optical plane: its counts leave the beam.
    ex.arrow(
        overlay, (xs[4] + 0.52 * size, cy), (xs[5] - 0.52 * size, cy), "data", cast
    )

    # What each card holds: applied by an element, or carried by the light.
    tag_y = g["tag_y"]
    for (xa, xb), word, ha in (
        ((xs[0], xs[2]), "what the element applies", "center"),
        ((xs[3], xs[4]), "what the light carries", "center"),
        ((xs[5], xs[5]), "what the detector records", "right"),
    ):
        left, right = xa - 0.45 * size, xb + 0.45 * size
        overlay.plot(
            [left, left, right, right],
            [tag_y - 0.2 * sp, tag_y, tag_y, tag_y - 0.2 * sp],
            color=cast["scenery"].color,
            lw=0.6 * cast.layout.lw,
        )
        ex.note(
            overlay,
            (0.5 * (left + right) if ha == "center" else fw - 0.04, tag_y + 0.1 * sp),
            word,
            cast,
            ha=ha,
            va="bottom",
            fontsize=small,
        )

    # Row headers name the three kinds of object in each column.
    header_kw = {"ha": "left", "va": "center", "linespacing": 1.0, "fontsize": small}
    ex.note(overlay, (0.04, g["axis_y"] + 2.0 * g["half"]), "plane", cast, **header_kw)
    ex.note(
        overlay,
        (0.04, g["axis_y"] - 1.62 * g["half"] - 0.9 * sp),
        "element",
        cast,
        **header_kw,
    )
    ex.note(overlay, (0.04, cy + 0.2 * size), "sampled\narray", cast, **header_kw)
    ex.note(
        overlay, (0.04, cy - 0.3 * size), "face-on,\nx right,\ny up", cast, **header_kw
    )

    # Brackets: where the model carries a field, and where an intensity.
    by = g["bracket_y"]
    tick = 0.35 * sp
    coherent = (xs[0] - 0.5 * size, xs[3] + 0.5 * size)
    incoherent = (xs[4] - 0.5 * size, xs[5] + 0.5 * size)
    for (xa, xb), words in (
        (
            coherent,
            "complex field $E$: each element multiplies it;\n"
            "FT: a Fourier transform between planes;\n"
            "the relayed pupil shares one array in the model",
        ),
        (incoherent, "intensity $I=|E|^2$,\nthen detector counts"),
    ):
        overlay.plot(
            [xa, xa, xb, xb],
            [by + tick, by, by, by + tick],
            color=cast["scenery"].color,
            lw=cast["scenery"].lw,
        )
        ex.halo(
            overlay.text(
                0.5 * (xa + xb),
                by - 0.25 * sp,
                words,
                ha="center",
                va="top",
                color=text_color(cast),
                fontsize=small,
                linespacing=1.05,
            ),
            cast,
        )
    boundary = 0.5 * (coherent[1] + incoherent[0])
    overlay.plot(
        [boundary, boundary],
        [by - 0.2 * sp, by + 1.4 * tick],
        color=cast.text,
        lw=cast.layout.lw,
    )
    ex.halo(
        overlay.text(
            boundary,
            by - 0.35 * sp,
            "$|E|^2$\ntaken\nhere",
            ha="center",
            va="top",
            color=cast.text,
            fontsize=small,
            linespacing=1.0,
        ),
        cast,
    )
    return {
        "rail": rail,
        "cards": cards,
        "elements": rail["elements"],
        "overlay": overlay,
    }


BADGE = "illustrative, unfolded;\nmirrors drawn as\ntransmissive; not\nto scale; thumbnails\nsimulated"


def build_train(layout, cast):
    """The still: the train with a thumbnail of each plane's quantity."""
    state = propagate(static_coeffs(0.0))
    if layout.is_slide:
        fig, ax = ex.figure(layout)
        ax.remove()
        fig.set_layout_engine("none")
        fig.suptitle(
            "Optical planes and the quantities that live on them",
            x=0.02,
            ha="left",
            y=0.975,
            fontsize=layout.title_pt,
        )
        handles = draw_train(fig, layout, cast, state, bottom_in=0.3, height_in=7.8)
    else:
        fig, ax = ex.figure(layout, doc_height_in=3.7)
        ax.remove()
        fig.set_layout_engine("none")
        handles = draw_train(fig, layout, cast, state, bottom_in=0.14, height_in=3.54)
    ex.badge(handles["overlay"], cast, BADGE, loc="upper right")
    return fig


# The perturbation still


def sign_label(ax, xy, text, value, cmap, cast):
    """Label a signed region in the map's own extreme color, on its center."""
    color = cmap(0.0 if value < 0 else 1.0)
    label = ax.text(
        *xy,
        text,
        ha="left",
        va="center",
        color=color,
        fontsize=cast.layout.small_pt,
        fontweight="bold",
    )
    label.set_path_effects(
        [
            patheffects.withStroke(
                linewidth=0.3 * cast.layout.font_pt, foreground=cmap(0.5)
            )
        ]
    )
    return label


def _edge_kw(cast):
    return {"ec": cast["aperture"].color, "lw": cast.layout.lw, "ls": "-"}


def _perturbation_axes(layout):
    """Place the rail, the four panels and the command key by hand, in inches."""
    f = layout.font_pt / 10.0
    width = layout.width_in
    ylab, cbar, shared_gap, gap = 0.5 * f, 0.72 * f, 0.08 * f, 0.4 * f
    size = (width - 0.1 - (2 * ylab + 3 * cbar + shared_gap + gap)) / 4.0
    title, xlab, rail_h = 0.5 * f, 0.5 * f, 1.0 * f
    key_h = 0.36 * f
    key_zone = key_h + 1.4 * layout.small_pt / 72.0 + 0.04
    used = 0.14 + key_zone + xlab + size + title + rail_h
    height = layout.height_in if layout.is_slide else used + 0.04
    fig, ax = ex.figure(layout, doc_height_in=height)
    ax.remove()
    fig.set_layout_engine("none")
    top_pad = 0.8 if layout.is_slide else 0.0
    lift = max(0.0, 0.5 * (height - top_pad - used))
    y_panel = 0.14 + key_zone + xlab + lift
    x0 = 0.05 + ylab
    x1 = x0 + size + cbar + ylab
    x2 = x1 + size + shared_gap
    x3 = x2 + size + cbar + gap
    axes = [
        fig.add_axes([x / width, y_panel / height, size / width, size / height])
        for x in (x0, x1, x2, x3)
    ]
    overlay = fig.add_axes([0, 0, 1, 1], zorder=-1)
    overlay.set(xlim=(0, width), ylim=(0, height))
    overlay.axis("off")
    geometry = {
        "size": size,
        "xs": (x0, x1, x2, x3),
        "y_panel": y_panel,
        "title": title,
        "rail_h": rail_h,
        "key_y": 0.14 + lift + key_zone - 0.5 * key_h,
        "key_h": key_h,
    }
    return fig, axes, overlay, geometry


def build_perturbation(layout, cast):
    """Nominal, perturbed and difference images for one DM command."""
    o = optics()
    p = PARAMS
    nominal = propagate(static_coeffs(0.0))
    perturbed = propagate(probe_coeffs())
    command = (perturbed["opd"] - nominal["opd"]) * pupil_mask(o)
    delta = perturbed["intensity"] - nominal["intensity"]
    fext = focal_extent(o)
    pext = pupil_extent(o)
    unit = p["delta_unit"]
    exponent = round(math.log10(unit))
    small = layout.small_pt

    fig, axes, overlay, g = _perturbation_axes(layout)
    size = g["size"]
    x0, x1, _, x3 = g["xs"]
    top = g["y_panel"] + size + g["title"]
    half = 0.13 * g["rail_h"]
    axis_y = top + 0.4 * g["rail_h"]
    # Planes sit over the panels that show their quantities: the DM plane
    # over the command, the image plane over the three image panels.
    image_x = 0.5 * (x1 + x3 + size)
    dm_x = x0 + 0.5 * size
    xp = {
        "pupil": dm_x - 0.55 * (image_x - dm_x) / 3.0,
        "dm": dm_x,
        "fpm": dm_x + (image_x - dm_x) / 3.0,
        "lyot": dm_x + 2.0 * (image_x - dm_x) / 3.0,
        "image": image_x,
    }
    rail = draw_rail(
        overlay,
        cast,
        layout,
        xp,
        axis_y,
        half,
        x_start=0.05,
        font=small,
        details=False,
    )
    highlight_rail(rail, ("dm", "image"), cast)
    for text in rail["text"]:
        text.set_text(text.get_text().replace(" ", "\n", 1))
    # Leaders from the two planes down to their panels.
    lead_kw = {"color": cast.text, "lw": 0.8 * cast.layout.lw}
    low = axis_y - 1.55 * half
    overlay.plot([dm_x, dm_x], [low, top + 0.05], **lead_kw)
    overlay.plot([image_x, image_x], [low, top + 0.12], **lead_kw)
    overlay.plot(
        [x1, x1, x3 + size, x3 + size],
        [top + 0.05, top + 0.12, top + 0.12, top + 0.05],
        **lead_kw,
    )
    ex.badge(overlay, cast, "simulated, noiseless", loc="upper right")
    if layout.is_slide:
        fig.suptitle(
            "A DM command adds a field, so the intensity change is signed",
            x=0.02,
            ha="left",
            y=0.97,
            fontsize=layout.title_pt,
        )

    ep.imshow_diverging(
        command,
        ax=axes[0],
        extent=pext,
        vlim=float(np.nanmax(np.abs(command))),
        cmap=ex.image_cmap("opd"),
        cbar_label="OPD [nm]",
    )
    axes[0].add_patch(
        Circle((0, 0), p["aperture_radius_d"], fill=False, **_edge_kw(cast))
    )
    axes[0].set_title("DM command $\\delta W$\n(as OPD)")
    axes[0].set_xlabel(r"$x$ [$D$]")
    axes[0].set_ylabel(r"$y$ [$D$]")
    axes[0].set_xticks([-0.5, 0.0, 0.5])
    axes[0].set_yticks([-0.5, 0.0, 0.5])

    # The command spelled out: minus the x cosine plus the y sine.
    modes = o["modes"] * pupil_mask(o)
    key = g["key_h"]
    ky = g["key_y"]
    f = layout.font_pt / 10.0
    first = 0.08 + 0.62 * f + 0.5 * key
    second = first + key + 0.3 * f
    word_kw = {"va": "center", "fontsize": small, "color": text_color(cast)}
    overlay.text(0.08, ky, r"$\delta W = -$", ha="left", **word_kw)
    overlay.text(0.5 * (first + second), ky, "+", ha="center", **word_kw)
    for x, mode, name in ((first, modes[0], "x cosine"), (second, modes[1], "y sine")):
        inset = overlay.inset_axes(
            [x - 0.5 * key, ky - 0.5 * key, key, key], transform=overlay.transData
        )
        inset.imshow(
            mode,
            cmap=ex.image_cmap("opd"),
            vmin=-1,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )
        inset.set_xticks([])
        inset.set_yticks([])
        overlay.text(
            x,
            ky - 0.5 * key - 0.03,
            name,
            ha="center",
            va="top",
            fontsize=small,
            color=text_color(cast),
        )

    row = ep.compare_row(
        [floored(nominal["intensity"]), floored(perturbed["intensity"])],
        titles=[r"nominal $I$", r"perturbed $I$"],
        axes=axes[1:3],
        norm="log",
        extent=fext,
        cmap=ex.image_cmap("intensity"),
        vmin=p["intensity_floor"],
        vmax=p["intensity_ceiling"],
    )
    row.artists["cbar"].set_label(r"$I/I_{\rm peak}$")
    residual = ex.image_cmap("residual")
    ep.imshow_diverging(
        delta / unit,
        ax=axes[3],
        extent=fext,
        vlim=math.ceil(10.0 * float(np.max(np.abs(delta))) / unit) / 10.0,
        cmap=residual,
        cbar_label=rf"$\Delta I/I_{{\rm peak}}$ [$10^{{{exponent}}}$]",
    )
    axes[3].set_title(r"change $\Delta I$")
    for ax in axes[1:]:
        ax.set_xlabel(r"$x$ [$\lambda/D$]")
        ex.mark(ax, "star", (0.0, 0.0), cast, scale=0.55)
    axes[1].set_ylabel(r"$y$ [$\lambda/D$]")
    for ax in (axes[2], axes[3]):
        ax.set_yticklabels([])
    k = p["static_ripple_cycles_per_d"]
    k2 = p["probe_ripple_cycles_per_d"]
    sign_label(axes[3], (k - 1.0, -2.8), r"$<0$", -1.0, residual, cast)
    sign_label(axes[3], (2.2, k2), r"$>0$", 1.0, residual, cast)
    return fig


# The light at every plane

# Columns of the light figure, left to right, and the plane each one samples.
LIGHT_COLUMNS = (
    "entrance",
    "focal_before",
    "focal_after",
    "lyot_before",
    "lyot_after",
    "image",
)
PUPIL_COLUMNS = ("entrance", "lyot_before", "lyot_after")
# What the model does between neighboring columns: a Fourier transform
# between planes, a multiplication by an element within a plane.
# The transform into the Lyot plane is the inverse of the clause's F.
LIGHT_GAPS = (
    "$\\mathcal{F}$",
    "$\\times$",
    "$\\mathcal{F}^{-1}$",
    "$\\times$\nstop",
    "$\\mathcal{F}$",
)
VORTEX_OPERAND = "$e^{i6\\theta}$"
# What each column group's amplitude is relative to.
LIGHT_NORMS = {
    "entrance": "$|E|$ rel. incident",
    "focal_before": "$|E|$ rel. $\\sqrt{I_{\\rm peak}}$",
    "lyot_before": "$|E|$ rel. incident",
}
# Element names under the carried train, as in the train figure.
LIGHT_ELEMENTS = {
    "pupil": "aperture",
    "dm": "deformable\nmirror",
    "fpm": "vortex\nphase mask",
    "lyot": "Lyot stop",
    "image": "detector",
}
# Footnotes scale with the small type so they read at column width.
LIGHT_NOTE_SCALE = 1.08
# The amplitude map extends the pupil ramp past its cyan end toward the
# text color, so the brightest decade reads in light mode too.
LIGHT_AMPLITUDE_TOP_MIX = 0.45
LIGHT_HEADS = {
    "entrance": "$E=A\\,e^{i2\\pi W/\\lambda}$",
    "focal_before": "before",
    "focal_after": "after",
    "lyot_before": "before",
    "lyot_after": "after",
    "image": "$I=|E|^2$",
}
# Plane of the rail that each group of columns hangs from.
LIGHT_GROUPS = (
    (("pupil", "dm"), ("entrance",)),
    (("fpm",), ("focal_before", "focal_after")),
    (("lyot",), ("lyot_before", "lyot_after")),
    (("image",), ("image",)),
)
LIGHT_TICKS = {"pupil": (-1.0, 0.0, 1.0), "focal": (-10.0, 0.0, 10.0)}
# Display sampling of the vortex (focal) plane in the light figure: finer
# than the image grid, over the same extent, so the Airy rings and the
# vortex phase are resolved. The model's own vortex uses its multi-scale
# ladder, not this grid.
LIGHT_PARAMS = {"fpm_display_pixscale_lod": 0.25}
# Blend level of the neutral gray that marks pixels with no defined phase.
LIGHT_BLANK_LEVEL = 0.3
LIGHT_TITLE = (
    "Each element multiplies the complex field; each gap is a Fourier transform"
)


def amplitude_floor():
    """Lowest amplitude drawn: the square root of the intensity floor."""
    return math.sqrt(PARAMS["intensity_floor"])


def entrance_phase_peak():
    """Largest phase the static ripple puts on the field, 2 pi a0/lambda (rad)."""
    return 2.0 * math.pi * PARAMS["static_ripple_nm"] / PARAMS["wavelength_nm"]


def amplitude_cmap(cast):
    """The amplitude map: background to the pupil-role cyan, then deeper.

    The last stop blends the cyan toward the text color, so in light mode
    the top of the ramp darkens and in dark mode it brightens.
    """
    base = ex.image_cmap("pupil")
    cyan = np.array(to_rgb(base(1.0)))
    ink = np.array(to_rgb(cast.text))
    top = cyan + LIGHT_AMPLITUDE_TOP_MIX * (ink - cyan)
    return LinearSegmentedColormap.from_list(
        "explainer_amplitude", [base(0.0), tuple(cyan), tuple(top)], N=256
    )


def fpm_coords():
    """Half-pixel-offset display grid of the vortex plane, in lambda/D.

    It spans the same extent as the image grid at a finer step.
    """
    step = LIGHT_PARAMS["fpm_display_pixscale_lod"]
    half_width = 0.5 * PARAMS["nfoc"] * PARAMS["pixscale_lod"]
    n = round(2.0 * half_width / step)
    return (np.arange(n) - 0.5 * n + 0.5) * step


def plane_fields(state):
    """The complex field on every column of the light figure.

    Pupil-plane fields are relative to the incident amplitude and sit on the
    2 D wide thumbnail grid; focal-plane fields are relative to the square
    root of the unocculted on-axis peak; the vortex-plane pair sits on the
    ``fpm_coords`` display grid and the image on the image grid.

    Args:
        state: A ``propagate`` result.

    Returns:
        A dict of complex numpy arrays keyed by ``LIGHT_COLUMNS``.
    """
    o = optics()
    po = o["po"]
    u = fpm_coords()
    with jax.enable_x64(True):
        focal = po.cmft_fwd(
            jnp.asarray(state["entrance_field"]),
            jnp.asarray(o["pupil_coords"]),
            jnp.asarray(u),
        )
    focal = np.asarray(focal) / math.sqrt(o["peak"])
    uu, vv = np.meshgrid(u, u)
    # The vortex's closed-form phase exp(i 6 theta), theta the focal azimuth.
    vortex = np.exp(1j * PARAMS["vortex_charge"] * np.arctan2(vv, uu))
    return {
        "entrance": pad_to_thumbnail(state["entrance_field"]),
        "focal_before": focal,
        "focal_after": focal * vortex,
        "lyot_before": state["lyot_wide"],
        "lyot_after": state["lyot_wide"] * pad_to_thumbnail(o["stop"]),
        "image": state["image_field"],
    }


def _light_axes(layout):
    """Place the rail band and the two rows of six panels, in inches."""
    f = layout.font_pt / 10.0
    width = layout.width_in
    left, right = (0.84 if layout.is_slide else 0.96) * f, 0.76 * f
    pair_gap, group_gap, vortex_gap = 0.3 * f, 0.3 * f, 0.42 * f
    gaps = (group_gap, vortex_gap, group_gap, pair_gap, group_gap)
    size = (width - left - right - sum(gaps)) / 6.0
    line = 1.15 * layout.small_pt / 72.0
    note_line = LIGHT_NOTE_SCALE * line
    xlab, footer, row_gap = 0.42 * f, 3.0 * note_line + 0.06, 0.07 * f
    # Room under the footer for the provenance stamp the exporter adds.
    stamp = 1.8 * layout.stamp_pt / 72.0 + 0.02
    head, bracket = 2.0 * line + 0.08, 0.1 * f
    # The train: plane labels above and element names below, two lines each.
    half = 0.15 * f
    side = 1.62 * half + 2.0 * line + 0.04
    rail_h = 2.0 * side + 0.04
    # The title band holds the carried-train tag and the status badge (and
    # the headline on a slide).
    title = 1.05 if layout.is_slide else 0.36
    if layout.is_slide:
        size = min(
            size,
            (
                layout.height_in
                - title
                - rail_h
                - head
                - bracket
                - xlab
                - footer
                - row_gap
                - stamp
                - 0.3
            )
            / 2.0,
        )
    used = stamp + footer + xlab + 2 * size + row_gap + head + bracket + rail_h
    height = layout.height_in if layout.is_slide else used + title
    fig, ax = ex.figure(layout, doc_height_in=height)
    ax.remove()
    fig.set_layout_engine("none")
    lift = max(0.0, 0.5 * (height - title - used - 0.04))
    total = 6 * size + sum(gaps)
    x = left + 0.5 * (width - left - right - total)
    xs = []
    for i in range(6):
        xs.append(x)
        x += size + (gaps[i] if i < 5 else 0.0)
    y_phase = stamp + footer + xlab + lift
    y_amp = y_phase + size + row_gap
    rows = {}
    for name, y in (("amplitude", y_amp), ("phase", y_phase)):
        rows[name] = {
            key: fig.add_axes([x0 / width, y / height, size / width, size / height])
            for key, x0 in zip(LIGHT_COLUMNS, xs, strict=True)
        }
    overlay = fig.add_axes([0, 0, 1, 1], zorder=-1)
    overlay.set(xlim=(0, width), ylim=(0, height))
    overlay.axis("off")
    geometry = {
        "size": size,
        "xs": dict(zip(LIGHT_COLUMNS, xs, strict=True)),
        "width": width,
        "y_amp": y_amp,
        "y_phase": y_phase,
        "top": y_amp + size,
        "head": head,
        "bracket": bracket,
        "rail_h": rail_h,
        "rail_half": half,
        "rail_side": side,
        "line": line,
        "note_line": note_line,
        "vortex_gap": vortex_gap,
        "footer_top": stamp + lift + footer,
        "left": left,
        "height": height,
        "title": title,
    }
    return fig, rows, overlay, geometry


def _backing(cast):
    """A plain box in the background color, for a label over an image."""
    return {
        "boxstyle": "round,pad=0.15,rounding_size=0.2",
        "facecolor": to_rgba(cast.background, 0.85),
        "edgecolor": "none",
    }


def _unstroke(fig, cast):
    """Replace the stroked halos the shared helpers add with a backing box."""
    for text in fig.findobj(Text):
        if text.get_path_effects():
            text.set_path_effects([])
            if text.get_bbox_patch() is None:
                text.set_bbox(_backing(cast))
    return fig


def _light_rail(overlay, layout, cast, g):
    """Draw the carried train, with a leader from each plane to its columns.

    The train is the train figure's own drawing (``draw_rail``): the same
    glyphs, element names and starlight label, with the planes placed over
    the columns that show their fields.
    """
    size = g["size"]
    xs = g["xs"]
    centers = {key: xs[key] + 0.5 * size for key in LIGHT_COLUMNS}
    pitch = centers["focal_before"] - centers["entrance"]
    xp = {
        "pupil": centers["entrance"] - 0.36 * pitch,
        "dm": centers["entrance"] + 0.36 * pitch,
        "fpm": 0.5 * (centers["focal_before"] + centers["focal_after"]),
        "lyot": 0.5 * (centers["lyot_before"] + centers["lyot_after"]),
        "image": centers["image"],
    }
    top = g["top"] + g["head"] + g["bracket"]
    half = g["rail_half"]
    axis_y = top + g["rail_side"]
    small = layout.small_pt
    rail = draw_rail(
        overlay,
        cast,
        layout,
        xp,
        axis_y,
        half,
        x_start=0.05,
        font=small,
        details=True,
    )
    highlight_rail(rail, OPTICAL_KEYS, cast)
    for text in rail["text"]:
        text.set_text(text.get_text().replace(" ", "\n", 1))
    gray = cast.neutral(0.55)
    for key, text in rail["elements"].items():
        text.set_text(LIGHT_ELEMENTS[key])
        text.set_color(gray)
        text.set_fontsize(0.9 * small)
    # The transforms are labeled between the panels, where the inverse
    # into the Lyot plane is marked; the train keeps only its relay word.
    for text in list(overlay.texts):
        if text.get_text() == "FT":
            text.remove()
        elif text.get_text() == "starlight":
            text.set_position((0.05, axis_y + 1.1 * half))
            text.set_ha("left")
            text.set_va("bottom")
    lead_kw = {"color": cast.text, "lw": 0.8 * cast.layout.lw}
    # Leaders start under the element names, so no line crosses a word.
    low = axis_y - 1.62 * half - 2.0 * g["line"] - 0.02
    tick = 0.5 * g["bracket"]
    for planes, columns in LIGHT_GROUPS:
        a = min(xs[columns[0]], *(xp[k] for k in planes))
        b = max(xs[columns[-1]] + size, *(xp[k] for k in planes))
        bar = top - 0.5 * g["bracket"]
        overlay.plot([a, a, b, b], [bar - tick, bar, bar, bar - tick], **lead_kw)
        for plane in planes:
            overlay.plot([xp[plane], xp[plane]], [low, bar], **lead_kw)


def _footnote(overlay, xy, text, cast, layout, **kw):
    """An italic footnote at the enlarged note size."""
    kw.setdefault("ha", "center")
    return overlay.text(
        *xy,
        text,
        va="top",
        fontsize=LIGHT_NOTE_SCALE * layout.small_pt,
        fontstyle="italic",
        color=cast["annotation"].color,
        linespacing=1.05,
        **kw,
    )


def build_light(layout, cast):
    """The complex field before and after each element, and the image."""
    o = optics()
    p = PARAMS
    state = propagate(static_coeffs(0.0))
    fields = plane_fields(state)
    small = layout.small_pt
    floor = amplitude_floor()
    fig, rows, overlay, g = _light_axes(layout)
    size = g["size"]
    xs = g["xs"]
    extents = {
        "pupil": ep.extent_lod(o["wide_coords"]),
        "fpm": ep.extent_lod(fpm_coords()),
        "focal": focal_extent(o),
    }
    amp_cmap = amplitude_cmap(cast)
    phase_cmap = ex.image_cmap("phase")
    images = {}
    for key in LIGHT_COLUMNS:
        kind = "pupil" if key in PUPIL_COLUMNS else "focal"
        extent = extents["fpm" if key.startswith("focal") else kind]
        amp_ax = rows["amplitude"][key]
        phase_ax = rows["phase"][key]
        amp_ax.set_gid(f"amplitude-{key}")
        phase_ax.set_gid(f"phase-{key}")
        if key == "image":
            top = ep.imshow_log(
                floored(state["intensity"]),
                ax=amp_ax,
                extent=extent,
                floor=p["intensity_floor"],
                vmin=p["intensity_floor"],
                vmax=p["intensity_ceiling"],
                cmap=ex.image_cmap("intensity"),
                colorbar=False,
            )
            ex.mark(amp_ax, "star", (0.0, 0.0), cast, scale=0.55)
        else:
            top = ep.imshow_log(
                np.abs(fields[key]),
                ax=amp_ax,
                extent=extent,
                floor=floor,
                vmin=floor,
                vmax=1.0,
                cmap=amp_cmap,
                colorbar=False,
            )
        # Where there is too little light for a phase, the panel shows a
        # neutral gray that the cyclic map never takes.
        phase_ax.set_facecolor(cast.neutral(LIGHT_BLANK_LEVEL))
        phase = po_viz.plot_field(
            phase_field(fields[key]),
            kind="phase",
            ax=phase_ax,
            extent=extent,
            cmap=phase_cmap,
            colorbar=False,
        )
        images[key] = (top.artists["image"], phase.artists["image"])
        for ax in (amp_ax, phase_ax):
            ax.set_xticks(LIGHT_TICKS[kind])
            ax.set_yticks([])
            ax.tick_params(labelbottom=False, length=2.0)
        phase_ax.tick_params(labelbottom=True)
        phase_ax.set_xticklabels([f"{t:g}" for t in LIGHT_TICKS[kind]])
        phase_ax.set_xlabel(
            r"$x$ [$D$]" if kind == "pupil" else r"$x$ [$\lambda/D$]", labelpad=1.0
        )
        if kind == "pupil":
            rings = [(p["aperture_radius_d"], "-")]
            if key != "entrance":
                rings.append((p["lyot_radius_d"], "--"))
            for ax in (amp_ax, phase_ax):
                for radius, ls in rings:
                    ax.add_patch(
                        Circle(
                            (0.0, 0.0),
                            radius,
                            fill=False,
                            ec=cast["optics"].color,
                            ls=ls,
                            lw=0.8 * cast.layout.lw,
                        )
                    )
        overlay.text(
            xs[key] + 0.5 * size,
            g["top"] + 0.04,
            LIGHT_HEADS[key],
            ha="center",
            va="bottom",
            fontsize=small,
            color=cast.text,
        )

    # What each group's amplitude is relative to, over its columns.
    for first, last in (
        ("entrance", "entrance"),
        ("focal_before", "focal_after"),
        ("lyot_before", "lyot_after"),
    ):
        overlay.text(
            0.5 * (xs[first] + xs[last] + size),
            g["top"] + 0.06 + g["line"],
            LIGHT_NORMS[first],
            ha="center",
            va="bottom",
            fontsize=0.9 * small,
            color=cast.neutral(0.6),
        )

    # What happens between neighboring columns.
    mid = g["y_phase"] + size + 0.5 * (g["y_amp"] - g["y_phase"] - size)
    for left, right, word in zip(
        LIGHT_COLUMNS[:-1], LIGHT_COLUMNS[1:], LIGHT_GAPS, strict=True
    ):
        x = 0.5 * (xs[left] + size + xs[right])
        if left == "focal_before":
            continue
        overlay.text(
            x,
            mid,
            word,
            ha="center",
            va="center",
            fontsize=small,
            color=cast.text,
            linespacing=1.1,
        )
    # The vortex operand: the train figure's mask-phase thumbnail, between
    # "x" above and its formula below.
    x = 0.5 * (xs["focal_before"] + size + xs["focal_after"])
    thumb = 0.84 * g["vortex_gap"]
    operand = overlay.inset_axes(
        [x - 0.5 * thumb, mid - 0.5 * thumb, thumb, thumb],
        transform=overlay.transData,
    )
    operand.imshow(
        o["mask_phase"],
        cmap=phase_cmap,
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        interpolation="nearest",
    )
    operand.set_xticks([])
    operand.set_yticks([])
    operand.set_gid("operand-vortex")
    overlay.text(
        x,
        mid + 0.5 * thumb + 0.03,
        LIGHT_GAPS[1],
        ha="center",
        va="bottom",
        fontsize=small,
        color=cast.text,
    )
    overlay.text(
        x,
        mid - 0.5 * thumb - 0.03,
        VORTEX_OPERAND,
        ha="center",
        va="top",
        fontsize=small,
        color=cast.text,
    )

    # Assertion: the ripple's phase is too small to see on a full cycle.
    # Enlarged like the footnotes, but never wider than its panel; the unit
    # is the row label's radian.
    bound = rows["phase"]["entrance"].text(
        0.0,
        -0.97,
        f"$\\leq${entrance_phase_peak():.3f}",
        bbox=_backing(cast),
        fontsize=LIGHT_NOTE_SCALE * small,
        color=cast.text,
        ha="center",
        va="bottom",
    )
    renderer = fig.canvas.get_renderer()
    room = (size - 0.04) * fig.dpi
    width = bound.get_window_extent(renderer).width
    if width > room:
        bound.set_fontsize(bound.get_fontsize() * room / width)

    # Keys: amplitude and phase on the left, the image intensity on the right.
    bar_w = 0.07 * layout.font_pt / 10.0
    x_key = xs["entrance"] - 0.1 * layout.font_pt / 10.0 - bar_w
    fw, fh = fig.get_size_inches()
    keys = {}
    for name, y, image, label in (
        ("amplitude", g["y_amp"], images["entrance"][0], "amplitude\n$|E|$"),
        ("phase", g["y_phase"], images["entrance"][1], "phase\n[rad]"),
    ):
        cax = fig.add_axes([x_key / fw, y / fh, bar_w / fw, size / fh])
        cbar = fig.colorbar(image, cax=cax)
        cax.set_gid(f"key-{name}")
        cax.yaxis.set_ticks_position("left")
        keys[name] = cbar
        overlay.text(
            0.04,
            y + 0.5 * size,
            label,
            rotation=90,
            ha="left",
            va="center",
            fontsize=layout.font_pt if layout.is_slide else small,
            color=cast.text,
            linespacing=1.0,
            multialignment="center",
        )
    keys["amplitude"].set_ticks([1e-4, 1e-2, 1.0])
    keys["amplitude"].set_ticklabels(["$10^{-4}$", "$10^{-2}$", "1"])
    keys["phase"].set_ticks([-np.pi, 0.0, np.pi])
    keys["phase"].set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    cax = fig.add_axes(
        [
            (xs["image"] + size + 0.06 * layout.font_pt / 10.0) / fw,
            g["y_amp"] / fh,
            bar_w / fw,
            size / fh,
        ]
    )
    cbar = fig.colorbar(images["image"][0], cax=cax)
    cax.set_gid("key-intensity")
    cbar.set_ticks([1e-10, 1e-7, 1e-4])
    cbar.set_ticklabels(["$10^{-10}$", "$10^{-7}$", "$10^{-4}$"])
    keys["intensity"] = cbar
    overlay.text(
        xs["image"] + size + (0.13 + 0.42) * layout.font_pt / 10.0,
        g["y_amp"] + 0.5 * size,
        r"$I/I_{\rm peak}$",
        rotation=90,
        ha="left",
        va="center",
        fontsize=layout.font_pt,
        color=cast.text,
    )

    _light_rail(overlay, layout, cast, g)
    foot = g["footer_top"]
    _footnote(
        overlay, (0.05, foot), "gray: no light,\nno phase", cast, layout, ha="left"
    )
    _footnote(
        overlay,
        (0.5 * (xs["focal_before"] + xs["focal_after"] + size), foot),
        "the vortex changes\nonly the phase",
        cast,
        layout,
    )
    _footnote(
        overlay,
        (0.5 * (xs["lyot_before"] + xs["lyot_after"] + size), foot),
        "solid: pupil edge; dashed: stop\n"
        "faint light inside (log scale):\n"
        "the ripple light, the speckles",
        cast,
        layout,
    )
    overlay.text(
        0.05,
        g["height"] - (0.72 if layout.is_slide else 0.06),
        "(the train from the optical-planes figure)",
        ha="left",
        va="top",
        fontsize=LIGHT_NOTE_SCALE * small,
        fontstyle="italic",
        color=cast["annotation"].color,
    )
    ex.badge(overlay, cast, "simulated, noiseless;\ntrain schematic", loc="upper right")
    if layout.is_slide:
        fig.suptitle(LIGHT_TITLE, x=0.02, ha="left", y=0.975, fontsize=layout.title_pt)
    return _unstroke(fig, cast)


# The animation


def frame_schedule(layout):
    """Self-contained frame states: the narrated tour, then the DM sweep."""
    if layout.is_slide:
        per_stop, n_sweep, holds, end_holds = 14, 33, 4, 8
    else:
        per_stop, n_sweep, holds, end_holds = 3, 11, 1, 0
    frames = [{"stop": key, "c": 0.0} for key in TOUR_STOPS for _ in range(per_stop)]
    values = [float(v) for v in np.linspace(0.0, SWEEP_END, n_sweep)]
    middle = values.index(-1.0)
    values = values[: middle + 1] + [-1.0] * holds + values[middle + 1 :]
    values += [values[-1]] * end_holds
    frames += [{"stop": "sweep", "c": c} for c in values]
    return frames


def _card_style(card, on, cast):
    plane = cast["reference_plane"]
    for spine in card.ax.spines.values():
        spine.set_color(cast.text if on else plane.color)
        spine.set_linewidth(2.4 * plane.lw if on else plane.lw)
        spine.set_linestyle("-" if on else plane.ls)
    card.label.set_fontweight("bold" if on else "normal")


def _panel_axes(fig, layout, bottom_in, height_in):
    """Three square panels with room for labels and inset colorbars."""
    fw, fh = fig.get_size_inches()
    f = layout.font_pt / 10.0
    if layout.is_slide:
        title, xlab = 0.72, 0.68
    else:
        title, xlab = 0.5, 0.48
    ylab, cbar, margin = 0.52 * f, 0.66 * f, 0.08 * f
    size = min(height_in - title - xlab, (fw - 3 * (ylab + cbar) - 4 * margin) / 3.0)
    block = ylab + size + cbar
    gap = (fw - 3 * block) / 4.0
    bottom = bottom_in + xlab + 0.5 * (height_in - title - xlab - size)
    axes = []
    for i in range(3):
        x = gap + i * (block + gap) + ylab
        axes.append(fig.add_axes([x / fw, bottom / fh, size / fw, size / fh]))
    return axes


def build_sweep(layout, cast):
    """The narrated tour of the planes, then the DM command sequence."""
    o = optics()
    p = PARAMS
    frames = frame_schedule(layout)
    states = {c: propagate(static_coeffs(c)) for c in sorted({f["c"] for f in frames})}
    nominal = states[0.0]
    # Pinned scales, from the union over every frame.
    deltas = {c: s["intensity"] - nominal["intensity"] for c, s in states.items()}
    unit = p["delta_unit"]
    exponent = round(math.log10(unit))
    delta_max = max(float(np.max(np.abs(d))) for d in deltas.values())
    counts_max = max(float(s["counts"].max()) for s in states.values())

    if layout.is_slide:
        fig, ax = ex.figure(layout)
        title_h, train_h, panel_h, scale = 0.6, 4.75, 3.4, 0.8
    else:
        fig, ax = ex.figure(layout, doc_height_in=6.2)
        title_h, train_h, panel_h, scale = 0.44, 3.38, 2.2, 1.0
    ax.remove()
    fig.set_layout_engine("none")
    fh = fig.get_size_inches()[1]
    stamp = 0.16 if not layout.is_slide else 0.28
    handles = draw_train(
        fig,
        layout,
        cast,
        nominal,
        bottom_in=fh - title_h - train_h,
        height_in=train_h,
        scale=scale,
    )
    handles["cards"]["detector"].image.set_clim(0.0, counts_max)
    title = fig.text(
        0.012,
        1.0 - 0.08 / fh,
        "",
        ha="left",
        va="top",
        fontsize=layout.font_pt,
        color=cast.text,
    )
    readout = fig.text(
        0.988,
        1.0 - 0.1 / fh - 1.35 * layout.font_pt / 72.0 / fh,
        "",
        ha="right",
        va="top",
        fontsize=layout.small_pt,
        fontstyle="italic",
        color=cast["annotation"].color,
    )

    panels = _panel_axes(fig, layout, stamp, panel_h)
    fext = focal_extent(o)
    pext = pupil_extent(o)
    mask = pupil_mask(o)
    opd = ep.imshow_diverging(
        nominal["opd"] * mask,
        ax=panels[0],
        extent=pext,
        vlim=p["static_ripple_nm"],
        cmap=ex.image_cmap("opd"),
        cbar_label="OPD [nm]",
    )
    panels[0].add_patch(
        Circle((0, 0), p["aperture_radius_d"], fill=False, **_edge_kw(cast))
    )
    title_kw = {"fontsize": layout.font_pt}
    panels[0].set_title(
        "DM plane: total OPD $W$ =\nstatic ripple + $c$ $\\times$ ripple", **title_kw
    )
    panels[0].set_xlabel(r"$x$ [$D$]")
    panels[0].set_ylabel(r"$y$ [$D$]")
    panels[0].set_xticks([-0.5, 0.0, 0.5])
    panels[0].set_yticks([-0.5, 0.0, 0.5])
    # A band above the aperture holds the value of c.
    panels[0].set_xlim(-0.62, 0.62)
    panels[0].set_ylim(-0.56, 0.8)
    c_text = panels[0].text(
        0.0,
        0.66,
        "",
        ha="center",
        va="center",
        fontsize=layout.small_pt,
        color=cast.text,
    )

    phase = po_viz.plot_field(
        phase_field(nominal["image_field"]),
        kind="phase",
        ax=panels[1],
        extent=fext,
        cmap=ex.image_cmap("phase"),
        cbar_label="rad",
    )
    cbar = phase.artists["cbar"]
    cbar.set_ticks([-np.pi, 0.0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    panels[1].set_title("image plane:\nphase of $E$", **title_kw)
    ex.note(
        panels[1],
        (0.0, 10.8),
        "blank: no light",
        cast,
        ha="center",
        va="top",
        fontsize=0.85 * layout.small_pt,
    )

    residual = ex.image_cmap("residual")
    change = ep.imshow_diverging(
        deltas[0.0] / unit,
        ax=panels[2],
        extent=fext,
        vlim=math.ceil(10.0 * delta_max / unit) / 10.0,
        cmap=residual,
        cbar_label=rf"$\Delta I/I_{{\rm peak}}$ [$10^{{{exponent}}}$]",
    )
    panels[2].set_title("image plane:\nchange $\\Delta I$", **title_kw)
    for ax in panels[1:]:
        ax.set_xlabel(r"$x$ [$\lambda/D$]")
        ex.mark(ax, "star", (0.0, 0.0), cast, scale=0.55)
    panels[1].set_ylabel(r"$y$ [$\lambda/D$]")
    panels[2].set_ylabel(r"$y$ [$\lambda/D$]")

    cards = handles["cards"]
    elements = handles["elements"]
    rail = handles["rail"]
    a0 = p["static_ripple_nm"]

    def draw(fig, frame):
        stop = frame["stop"]
        c = frame["c"]
        state = states[c]
        lit = ("dm", "image") if stop == "sweep" else (stop,)
        highlight_rail(rail, lit, cast)
        for key, card in cards.items():
            _card_style(card, key in lit, cast)
        for key, text in elements.items():
            on = key in lit or (stop == "detector" and key == "image")
            text.set_fontweight("bold" if on else "normal")
            text.set_color(cast.text if on else cast["optics"].color)
        cards["dm"].update(wide_opd(state))
        cards["lyot"].update(np.abs(state["lyot_wide"]))
        cards["image"].update(floored(state["intensity"]))
        cards["detector"].update(state["counts"])
        opd.update(state["opd"] * mask)
        phase.update(phase_field(state["image_field"]))
        change.update(deltas[c] / unit)
        c_text.set_text(rf"$c={c * a0:+.2f}$ nm")
        if stop == "sweep":
            title.set_text(SWEEP_TITLE)
            readout.set_text(
                f"model-parameter sequence, not a time series; static ripple {a0:g} nm"
            )
        else:
            title.set_text(TOUR_TITLES[stop])
            readout.set_text("narrated tour of the planes; DM command c = 0")

    return ex.AnimationScene(fig=fig, draw=draw, frames=frames)


# Captions and alternative text

TRAIN_CAPTION = (
    "An illustrative vortex coronagraph train, unfolded in side view with its "
    "mirrors drawn as transmissive elements; it is not a prescription for any "
    "mission. Each dashed line is a plane, a location along the beam; the glyph "
    "on it is the hardware element there; the framed thumbnail below is a "
    "sampled array of the quantity on that plane. The first three thumbnails "
    "show what an element applies, the next two what the light carries, and the "
    "last what the detector records. The beam envelope is schematic geometry, "
    "not to scale, and is not evidence of coronagraph performance. The "
    "thumbnails come from one propagated physicaloptix example: a 64 by 64 pupil "
    "grid, a 48 by 48 focal grid at 0.5 lambda/D, 550 nm, a charge-6 vortex (its "
    "phase winds six times through 2 pi around the axis; Mawet et al. 2005, ApJ "
    "633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308) and a Lyot stop, the pupil "
    "stop in the relayed pupil, of 0.8 D. The DM plane carries a 1 nm cosine "
    "ripple of 6 cycles per D standing in for a static wavefront error. The "
    "vortex phase is evaluated from its closed form, because the model does not "
    "output its internal focal-plane field. The pupil-plane thumbnails span 2 D: "
    "the path keeps only a D-wide array, which holds about 4 percent of the "
    "entrance energy at the Lyot plane, so the Lyot thumbnail evaluates the same "
    "vortex operator on a wider grid to show the light moved outside the pupil "
    "edge. Image intensities are relative to the unocculted on-axis peak "
    r"$|\mathcal{F}[A](0)|^2$, not to the brightest sample of the half-pixel-offset grid. "
    "Between the entrance pupil and the image plane the model carries a complex "
    "field: each element multiplies it and each gap marked FT is a Fourier "
    "transform. The relay from the entrance pupil to the DM plane is an identity "
    "in the model, which applies the DM to the entrance array, and the model "
    "reaches the Lyot plane with an inverse transform, so it does not invert the "
    "pupil as a physical relay would. An OPD W enters as exp(+i 2 pi W/lambda) "
    "under the proposed coherent profile ({ref}`optics-coherent-phase`). The OPD "
    "is a path difference, not a mirror surface height: at normal reflection a "
    "surface displacement h gives an OPD of 2h ({ref}`Hecht 2017, Sec. 9.4.2, "
    "text before eq. 9.44 <source-hecht2017>`). The intensity forms only at the image plane, "
    "and the detector applies its own boundary contract "
    "({ref}`optics-signed-intensity`): here 1e7 photons per second in a pixel at "
    "the unocculted peak, a 1 s exposure, quantum efficiency 1, a Gaussian "
    "approximation to shot noise, 1 electron of read noise, and detector pixels "
    "equal to the focal samples. The thumbnails show each array with x right "
    "and y up; whether that is the view looking downstream or upstream, and how "
    "a relay's pupil inversion is recorded, belong to the pending "
    "{ref}`image coordinates and PSFlet origin decision "
    "<decision-image-coordinates-and-psflet-origin>`. The picture is an example "
    "implementation of the proposed profile. A backend built on response tables "
    "stores only a final-plane response and does not model the intermediate "
    "planes."
)
TRAIN_ALT = (
    "Side-view diagram of an unfolded optical train with five planes from left "
    "to right: entrance pupil, DM plane, focal plane, Lyot plane and image "
    "plane, each a dashed line across a gray beam. Starlight enters from the "
    "left; a relay lens pair between the entrance pupil and the DM plane and a "
    "single lens in each later gap are marked relay and FT. The beam is "
    "parallel at the pupils and focused at the focal and image planes, and it "
    "narrows at the Lyot stop. Below the beam the hardware is named: aperture, "
    "deformable mirror, vortex phase mask, Lyot stop and detector. Below that a "
    "row of square thumbnails, grouped as what the element applies, what the "
    "light carries and what the detector records, shows a filled disk of pupil "
    "transmission, vertical OPD fringes, a six-fold phase pattern with a cyclic "
    "key, a bright ring of field amplitude just outside a solid pupil-edge "
    "circle and a dashed stop circle, an image with two speckles either side of "
    "a dark center, and, after an arrow, simulated detector counts. A bracket "
    "under the first four marks the complex field, a mark between the fourth "
    "and fifth reads |E| squared taken here, and a second bracket marks the "
    "intensity and counts."
)
PERTURBATION_CAPTION = (
    "A deformable-mirror command changes the image through the field, so the "
    "intensity change is signed ({ref}`optics-signed-intensity`). The nominal "
    "state carries the 1 nm cosine ripple of 6 cycles per D along x from the "
    "train figure, which puts a speckle pair at plus and minus 6 lambda/D on the "
    "x axis. The command (left, as OPD in nm over the aperture, spelled out "
    "below it as minus the x cosine plus the y sine) subtracts that ripple and "
    "adds a 1 nm sine ripple of 6 cycles per D along y. The perturbed image "
    "loses the x pair and gains a y pair. The difference (right) is negative "
    "where the command cancels the existing field and positive where it adds "
    "light; clipping it at zero before adding the nominal image would erase the "
    "cancellation. Both images share one log norm relative to the unocculted "
    "on-axis peak, the difference uses a symmetric norm about zero, and the star "
    "glyph marks the on-axis star behind the vortex. The rail above marks the "
    "DM plane over the command and the image plane over the three image panels. "
    "Same propagated physicaloptix path and parameters as the train figure, "
    "noiseless; an illustration of the stated model, not a performance claim."
)
PERTURBATION_ALT = (
    "A thin optical-train rail with the DM plane and the image plane "
    "highlighted and joined by lines to the panels below. Left: a pattern of "
    "alternating brown and teal spots over a circular aperture, the DM command "
    "as OPD in nanometers, with a small equation under it showing it as minus "
    "a vertical-fringe cosine map plus a horizontal-fringe sine map. Middle: "
    "two log-scaled images relative to the unocculted peak; the nominal image "
    "has two speckles on the horizontal axis and the perturbed image has two "
    "speckles on the vertical axis instead, with a star glyph at the center of "
    "each. Right: the difference on a blue-white-red scale, with blue spots "
    "labeled less than zero at the old speckle positions and red spots labeled "
    "greater than zero at the new ones."
)
TOUR_CAPTION = (
    "The train figure, highlighted one plane at a time as the title names the "
    "quantity each plane carries, followed by a model-parameter sequence that "
    "is not a time series. In the second part a deformable-mirror command c "
    "scales a cosine ripple opposing the static 1 nm ripple, from c = 0 to "
    "c = -2 nm; the left panel shows the total OPD and the value of c. At "
    "c = -1 nm the total OPD vanishes, the speckle pair disappears and the "
    "intensity change there is negative. At c = -2 nm the OPD has reversed "
    "sign: the intensity is back to its nominal value within 4 percent of the "
    "speckle peak (the remainder is interference with the small leakage of "
    "this sampled vortex model) while the field phase at the speckles has "
    "changed by pi, from +pi/2 to -pi/2, so intensity alone barely tells the "
    "two fields apart ({ref}`optics-coherent-phase`, "
    "{ref}`optics-signed-intensity`). Phase is drawn only where the intensity "
    "exceeds 3e-7 of the unocculted on-axis peak. Every color scale is fixed "
    "across the frames: OPD within plus and minus 1 nm, phase over one cycle, "
    "intensity from 1e-10 to 1e-4 of the unocculted peak, and the change within "
    "plus and minus its largest value over the sequence. Same propagated "
    "physicaloptix path as the train figure; the train itself is schematic and "
    "not to scale."
)
TOUR_ALT = (
    "Animation of the optical-train diagram above three panels: the total OPD "
    "at the DM plane with the command value c, the phase of the image-plane "
    "field, and the intensity change at the image plane. First the highlight "
    "moves from the entrance pupil through the DM plane, focal plane, Lyot "
    "plane and image plane to the detector, with a title stating what each "
    "plane carries. Then c steps from 0 to -2 nm: the OPD fringes fade to zero "
    "and return with reversed colors, the two speckles vanish and return, the "
    "intensity change turns blue at the speckles and then back to white, and "
    "the speckle phase changes from red to blue."
)
LIGHT_CAPTION = (
    "The light of the optical-planes train at every plane, before and after "
    "each element ({ref}`optics-coherent-phase`). The top row is the field "
    "amplitude |E| and the bottom row its phase. Amplitude uses the amplitude "
    "colormap on a log norm in every plane, extended past its cyan end toward "
    "the text color so that the brightest decade reads; phase uses the cyclic "
    "phase map in every plane. Pupil-plane amplitudes are relative to the "
    "incident amplitude and span 2 D; focal-plane amplitudes are relative to "
    "the square root of the unocculted on-axis peak and span 24 lambda/D, so "
    "the amplitude floor of 1e-5 matches the intensity floor of 1e-10. Each "
    "element multiplies the field. The entrance array carries the transmission "
    "A of a clear circular aperture times exp(+i 2 pi W/lambda) for the 1 nm "
    "ripple W of 6 cycles per D, a phase of at most 0.011 rad that a "
    "full-cycle phase scale cannot show. The vortex multiplies by "
    "exp(i 6 theta), with theta the focal-plane azimuth, drawn as the mask "
    "thumbnail of the train figure between the focal pair: it leaves |E| "
    "unchanged and winds the phase six times around the axis (Mawet et al. "
    "2005, ApJ 633, 1191; Foo et al. 2005, Opt. Lett. 30, 3308). The Lyot "
    "stop of 0.8 D multiplies by its transmission. The gaps into the focal "
    "and image planes are the transform F of that clause, and the gap into "
    "the Lyot plane is its inverse. Before the vortex the focal field is, "
    "apart from the speckles of the ripple, the Airy pattern of the clear "
    "circular aperture, a real field whose rings alternate in sign (phase 0 "
    "and pi). For that clear circular aperture the vortex moves the starlight "
    "outside the pupil edge (solid circle) in the Lyot plane, and the stop "
    "(dashed circle) blocks it. The faint light inside the pupil, visible "
    "only on the log scale, is the ripple light that forms the speckles of "
    "the perturbation figure: about 4e-5 of the entrance energy passes the "
    "stop, close to the first-order share of the ripple, "
    "(2 pi a0/lambda)^2/2 = 6.5e-5, times the 0.64 of the aperture area "
    "that the stop keeps; the leakage of the sampled vortex without the "
    "ripple is under 5 percent of it. Only at the image plane is the "
    "intensity I = |E|^2 taken, on the log scale of the train and "
    "perturbation figures. Phase is drawn only where the intensity exceeds "
    "3e-7 of the reference; gray marks pixels with too little light for a "
    "phase. The image comes from the propagated physicaloptix path of the "
    "train figure. The path keeps only a D-wide array, which holds about 4 "
    "percent of the entrance energy at the Lyot plane, so the Lyot pair "
    "evaluates the same vortex operator on a 2 D wide grid; its central "
    "D-wide block is the field the path carries. The focal pair is a matrix "
    "Fourier transform of the sampled entrance field onto a display grid of "
    "0.25 lambda/D over the image extent, times the closed-form vortex "
    "phase, because the model's own vortex uses a multi-scale ladder and "
    "does not output its focal-plane field. The model reaches the Lyot plane "
    "with an inverse transform, so it does not invert the pupil as a "
    "physical relay would; the panels show x right and y up, and the viewing "
    "direction belongs to the pending {ref}`image coordinates and PSFlet "
    "origin decision <decision-image-coordinates-and-psflet-origin>`. The "
    "train above is schematic and not to scale; its planes sit over the "
    "columns that show their fields. The picture is a noiseless example "
    "implementation of the proposed coherent profile, not a performance "
    "claim."
)
LIGHT_ALT = (
    "The optical train from the optical-planes figure drawn in side view "
    "across the top: starlight enters from the left through the aperture, a "
    "relay to the deformable mirror, the vortex phase mask at the focal "
    "plane, the Lyot stop and the detector at the image plane, each element "
    "named in small gray text under the beam. Brackets join the planes to "
    "six columns of square panels below. The top row shows field amplitude "
    "on a log scale, labeled relative to the incident amplitude for the pupil "
    "planes and relative to the square root of the peak for the focal plane; "
    "the bottom row shows phase on a cyclic scale, with gray where there is "
    "no light. Left to right: a uniform disk of amplitude with a flat phase "
    "marked at most 0.011 (radians, from the row label); the focal field before the vortex, a bright "
    "core with rings whose phase alternates between dark and light; the same "
    "amplitude after the vortex, with the rings broken into colored spokes "
    "of phase; the Lyot-plane field before the stop, bright outside a solid "
    "pupil-edge circle and faint inside a dashed stop circle; the same field "
    "after the stop, faint light only inside the stop; and the image-plane "
    "intensity with two speckles either side of a star glyph, with phase "
    "shown only at the speckles. Between the columns are the labels F, times "
    "a small thumbnail of the six-fold vortex phase labeled e to the i 6 "
    "theta, F inverse, times stop, and F. Footnotes say that gray means no "
    "light, that the vortex changes only the phase, and that the faint light "
    "inside the stop is the ripple light that forms the speckles."
)


FIGURES = [
    ex.FigureSpec(
        slug="d04-optical-planes",
        build=build_train,
        caption=TRAIN_CAPTION,
        alt=TRAIN_ALT,
        status="schematic, not to scale; thumbnails simulated",
        params=PARAMS,
    ),
    ex.FigureSpec(
        slug="d04-opd-perturbation",
        build=build_perturbation,
        caption=PERTURBATION_CAPTION,
        alt=PERTURBATION_ALT,
        status="simulated, noiseless optical model",
        params=PARAMS,
    ),
    ex.FigureSpec(
        slug="d04-light-every-plane",
        build=build_light,
        caption=LIGHT_CAPTION,
        alt=LIGHT_ALT,
        status="simulated, noiseless optical model; train schematic",
        params={**PARAMS, **LIGHT_PARAMS},
    ),
]
ANIMATIONS = [
    ex.AnimationSpec(
        slug="d04-plane-tour",
        build=build_sweep,
        ground="narration",
        caption=TOUR_CAPTION,
        alt=TOUR_ALT,
        status="schematic train; simulated thumbnails; model-parameter sequence",
        params=PARAMS,
        fps=4,
        preview_frames=(3, 17, 18, 23, 29),
    ),
]
