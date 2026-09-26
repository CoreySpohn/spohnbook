"""Contracts of the contrast-denominator explainer figure.

Every number the figure prints is recomputed here from the yield input
package's FITS files with astropy and numpy: the tabulated off-axis PSFs,
the stellar intensity maps and the header. yippy is used only to locate the
cached package; no yippy table, interpolator or aperture routine enters the
independent values. The ratios are then formed by hand arithmetic and
compared with the text the builder actually drew.
"""

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from matplotlib.image import AxesImage
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from explainers import _common as ex  # noqa: E402
from explainers import _export as exporter  # noqa: E402
from explainers import d12_contrast_denominators as d12  # noqa: E402

PLANET_RATIO = 1.0e-10
APERTURE_LOD = 0.7
ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


@pytest.fixture(scope="module")
def package():
    """The package's own FITS tables, read without yippy's loaders."""
    import yippy

    yippy.logger.setLevel("ERROR")
    path = Path(yippy.fetch_yip(d12.PARAMS["package"]))
    offax = fits.getdata(path / "offax_psf.fits").astype(float)
    header = fits.getheader(path / "offax_psf.fits")
    return {
        "offax": offax,
        "offsets": fits.getdata(path / "offax_psf_offset_list.fits").astype(float),
        "stellar": fits.getdata(path / "stellar_intens.fits").astype(float),
        "diams": fits.getdata(path / "stellar_intens_diam_list.fits").astype(float),
        "pixscale": float(header["PIXSCALE"]),
        "obscured": float(header["OBSCURED"]),
        "diameter_m": float(header["D"]),
        "lambda_um": float(header["LAMBDA"]),
        "band_um": (float(header["MINLAM"]), float(header["MAXLAM"])),
        "n_lam": int(header["N_LAM"]),
    }


def _nearest(values, target):
    return int(np.argmin(np.abs(values - target)))


def _aperture_weights(shape, cx, cy, radius_px, sub=8):
    """Fraction of each pixel inside a circle, by sub-pixel center sampling."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    w = np.zeros(shape)
    for a in offsets:
        for b in offsets:
            w += np.hypot(xx + a - cx, yy + b - cy) <= radius_px
    return w / sub**2


@pytest.fixture(scope="module")
def independent(package):
    p = d12.PARAMS
    i_s = _nearest(package["offsets"], p["planet_sep_lod"])
    i_r = _nearest(package["offsets"], p["reference_sep_lod"])
    k = _nearest(package["diams"], p["star_diam_lod"])
    sep = package["offsets"][i_s]
    pix = package["pixscale"]
    planet = package["offax"][i_s]
    star = package["stellar"][k]
    n = planet.shape[0]
    # The star sits on the corner between the four central pixels.
    center = (n - 1) / 2.0
    w = _aperture_weights(planet.shape, center + sep / pix, center, APERTURE_LOD / pix)
    tau = float((planet * w).sum())
    peak = float(planet.max())
    peak0 = float(package["offax"][i_r].max())
    return {
        "sep": float(sep),
        "star_diam": float(package["diams"][k]),
        "peak": peak,
        "peak0": peak0,
        "reference_sum": float(package["offax"][i_r].sum()),
        "tau": tau,
        "raw_contrast": float((star * w).sum()) / tau,
        "core_area": math.pi * APERTURE_LOD**2,
        "aperture_pixels": math.pi * APERTURE_LOD**2 / pix**2,
        "ratios": {
            "host": PLANET_RATIO,
            "peak": PLANET_RATIO * peak / peak0,
            "incident": PLANET_RATIO * peak,
        },
        "mas": 1.0e3
        * sep
        * package["lambda_um"]
        * 1.0e-6
        / package["diameter_m"]
        * ARCSEC_PER_RAD,
    }


def _sci(value, digits=2):
    """Independent formatting of the printed mantissa and exponent."""
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    if round(mantissa, digits - 1) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    return f"{mantissa:.{digits - 1}f}", exponent


def _drawn(layout, mode="light"):
    spec = d12.FIGURES[0]
    with exporter.venue(mode, layout) as cast:
        fig = spec.build(layout, cast)
        try:
            texts = [t for t in fig.findobj(Text) if t.get_visible()]
            return {
                "strings": [t.get_text() for t in texts if t.get_text().strip()],
                "stroked": [t.get_text() for t in texts if t.get_path_effects()],
                "planet_image": (
                    np.asarray(fig.findobj(AxesImage)[0].get_array(), dtype=float),
                    fig.findobj(AxesImage)[0].get_extent(),
                ),
                "images": [
                    (
                        im.get_interpolation(),
                        type(im.norm).__name__,
                        im.norm.vmin,
                        im.norm.vmax,
                        id(im.norm),
                    )
                    for im in fig.findobj(AxesImage)
                ],
            }
        finally:
            plt.close(fig)


def _contains_sci(strings, value, digits=2):
    mant, expo = _sci(value, digits)
    pattern = re.escape(mant) + r"\\times10\^\{" + str(expo) + r"\}"
    return any(re.search(pattern, s) for s in strings)


# The package tables the figure reads


def test_planet_and_reference_are_tabulated_offsets(package, independent):
    d = d12.compute()
    assert d["sep_lod"] == pytest.approx(independent["sep"], abs=1e-12)
    assert d["ref_sep_lod"] == pytest.approx(
        package["offsets"][_nearest(package["offsets"], 20.5)], abs=1e-12
    )
    assert d["star_diam_lod"] == pytest.approx(independent["star_diam"], rel=1e-9)
    # The planet image is the package's own PSF, not an interpolation.
    i_s = _nearest(package["offsets"], d["sep_lod"])
    np.testing.assert_allclose(d["planet_psf"], package["offax"][i_s], atol=1e-15)


def test_peaks_are_the_tabulated_pixels(independent):
    d = d12.compute()
    assert d["planet_peak"] == pytest.approx(independent["peak"], rel=1e-12)
    assert d["unocculted_peak"] == pytest.approx(independent["peak0"], rel=1e-12)


def test_image_unit_is_the_fraction_of_incident_photons(independent):
    # A PSF the mask no longer touches holds nearly all of its source's
    # photons (the rest falls outside the tabulated field or the Lyot stop).
    assert 0.95 < independent["reference_sum"] <= 1.0


def test_unocculted_peak_matches_the_diffraction_limit(package, independent):
    # A uniformly illuminated pupil of area A puts P A / lambda^2 at the
    # center of its image; per pixel of side s lambda/D, as a fraction of
    # the collected power, that is (1 - f_obscured) (pi / 4) s^2 with D the
    # circumscribed diameter. The package PSF includes the Lyot stop and
    # pixel sampling, so agreement is to a few percent.
    closed_form = (1.0 - package["obscured"]) * math.pi / 4.0 * package["pixscale"] ** 2
    assert independent["peak0"] == pytest.approx(closed_form, rel=0.05)
    assert independent["peak0"] < closed_form


def test_throughput_and_core_area_tables(independent):
    d = d12.compute()
    assert d["core_area"] == pytest.approx(independent["core_area"], rel=1e-9)
    # yippy oversamples and centers differently; an independent area-weighted
    # sum agrees to better than one percent.
    assert d["throughput"] == pytest.approx(independent["tau"], rel=0.01)
    assert d12.ratios()["aperture_pixels"] == pytest.approx(
        independent["aperture_pixels"], rel=1e-9
    )


def test_raw_contrast_table_is_the_aperture_sum_ratio(independent):
    # Star sum over planet sum in the same aperture. yippy truncates the
    # aperture center to a pixel and oversamples by two, so the independent
    # area-weighted sum agrees to ten percent.
    d = d12.compute()
    assert d["raw_contrast"] == pytest.approx(independent["raw_contrast"], rel=0.1)
    assert 1e-15 < d["raw_contrast"] < 1e-13


def test_ratios_by_hand_arithmetic(independent):
    r = d12.ratios()
    ind = independent["ratios"]
    assert r["host"] == pytest.approx(ind["host"], rel=1e-12)
    assert r["peak"] == pytest.approx(ind["peak"], rel=1e-12)
    assert r["incident"] == pytest.approx(ind["incident"], rel=1e-12)
    # The aperture-sum normalization returns the host-relative ratio for a
    # point source, and the pixel ratios differ from it.
    assert r["aperture"] == pytest.approx(PLANET_RATIO, rel=0.01)
    assert r["incident"] < 0.05 * PLANET_RATIO
    assert 0.8 * PLANET_RATIO < r["peak"] < PLANET_RATIO


# What the figure prints


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_figure_prints_the_four_ratios(layout, independent):
    strings = _drawn(layout)["strings"]
    for value in independent["ratios"].values():
        assert _contains_sci(strings, value), value
    joined = "\n".join(strings)
    for word in ("Host-star flux", "Unocculted", "Total incident", "An aperture sum"):
        assert word in joined


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_figure_prints_the_checked_table_values(layout, independent):
    strings = _drawn(layout)["strings"]
    joined = "\n".join(strings)
    assert f"{independent['sep']:.2f}" in joined
    assert f"{independent['star_diam']:.4f}" in joined
    assert f"{independent['tau']:.2f}" in joined
    assert f"{independent['peak0']:.4f}" in joined
    assert f"{independent['mas']:.0f} mas" in joined
    assert _contains_sci(strings, 2.4e-14)
    mant, expo = _sci(independent["raw_contrast"], 2)
    # The printed raw contrast is the table's value to two figures, and the
    # independent sum rounds to within one unit of the last printed digit.
    assert abs(float(mant) * 10.0**expo - 2.4e-14) <= 0.15e-14
    if not layout.is_slide:
        assert f"{independent['core_area']:.2f}" in joined
        assert f"{independent['aperture_pixels']:.1f} pixels" in joined
        assert f"{independent['peak']:.4f}" in joined
        assert _contains_sci(strings, PLANET_RATIO * independent["tau"])


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_images_are_raw_pixels_on_one_log_scale(layout, mode):
    images = _drawn(layout, mode)["images"]
    assert len(images) == 2
    assert {im[0] for im in images} <= {"nearest", "none"}
    assert {im[1] for im in images} == {"LogNorm"}
    assert len({im[4] for im in images}) == 1  # one shared norm object
    assert images[0][2] == pytest.approx(d12.PARAMS["norm_vmin"])
    assert images[0][3] == pytest.approx(d12.PARAMS["norm_vmax"])


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_no_label_carries_a_stroked_halo(layout, mode):
    assert _drawn(layout, mode)["stroked"] == []


def test_caption_names_clauses_sources_and_numbers(independent):
    spec = d12.FIGURES[0]
    cap = spec.caption
    for ref in (
        "radiometry-contrast-zodi",
        "optics-pixel-measure",
        "source-nemati2023",
        "decision-stellar-leakage-measure",
        "decision-image-coordinates-and-psflet-origin",
    ):
        assert ref in cap
    assert "Sec. 1.1, eq. 3" in cap and "Sec. 1.1, eq. 4" in cap
    assert "Sec. 1, text at eq. 1" in cap
    for token in (
        f"{independent['sep']:.2f}",
        f"{independent['mas']:.0f} mas",
        f"{independent['star_diam']:.4f}",
        f"{independent['core_area']:.2f}",
        f"{independent['aperture_pixels']:.1f} pixels",
        f"{independent['tau']:.2f}",
    ):
        assert token in cap, token
    for text in (cap, spec.alt):
        assert "\u2014" not in text and "\u2013" not in text
        assert "n't" not in text
    assert spec.slug.startswith("d12-")
    assert d12.ANIMATIONS == []


def _drawn_aperture_sum(image, extent, sep, sub=8):
    """Sum a drawn image inside the aperture, from its extent in lambda/D."""
    left, right, bottom, top = extent
    ny, nx = image.shape
    dx, dy = (right - left) / nx, (top - bottom) / ny
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    xc = left + (np.arange(nx) + 0.5) * dx
    yc = bottom + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(xc, yc)
    w = np.zeros(image.shape)
    for a in offsets:
        for b in offsets:
            w += np.hypot(xx + a * dx - sep, yy + b * dy) <= APERTURE_LOD
    return float((image * w).sum() / sub**2)


@pytest.fixture(scope="module")
def drawn_aperture(independent):
    image, extent = _drawn(ex.DOC)["planet_image"]
    numerator = _drawn_aperture_sum(image, extent, independent["sep"])
    return {"numerator": numerator, "ratio": numerator / independent["tau"]}


def test_row_four_sums_the_drawn_planet_image(independent, drawn_aperture):
    # The printed aperture-sum ratio: the drawn planet image summed inside the
    # aperture, over the independent core throughput. It returns the band
    # contrast for this point source, to the precision of the two sums.
    r = d12.ratios()
    assert r["aperture_numerator"] == pytest.approx(
        drawn_aperture["numerator"], rel=1e-3
    )
    assert r["aperture"] == pytest.approx(drawn_aperture["ratio"], rel=0.01)
    assert drawn_aperture["ratio"] == pytest.approx(PLANET_RATIO, rel=0.01)
    strings = _drawn(ex.DOC)["strings"]
    assert _contains_sci(strings, drawn_aperture["ratio"])
    assert _contains_sci(strings, drawn_aperture["numerator"])


def test_row_two_states_why_it_is_below_the_band_contrast(independent):
    percent = f"{100 * independent['peak'] / independent['peak0']:.0f}%"
    assert percent == "94%"
    for layout in (ex.DOC, ex.SLIDE):
        assert percent in "\n".join(_drawn(layout)["strings"])
    assert "94 percent" in d12.FIGURES[0].caption


def test_row_three_names_its_pixel_and_scales_with_area(package, independent):
    per_lod2 = PLANET_RATIO * independent["peak"] / package["pixscale"] ** 2
    mant, expo = _sci(per_lod2)
    assert f"${mant}\\times10^{{{expo}}}$ per $(\\lambda/D)^2$" in (
        d12.FIGURES[0].caption
    )
    assert "radiometry-pixel-brightness" in d12.FIGURES[0].caption
    joined = "\n".join(_drawn(ex.DOC)["strings"])
    assert r"in the peak 0.25 $\lambda/D$ pixel" in joined


def test_caption_states_the_band_and_the_chapter_denominators(package):
    cap = d12.FIGURES[0].caption
    lo, hi = package["band_um"]
    assert f"{lo:.1f} to {hi:.1f}" in cap
    assert package["n_lam"] == 5 and "five wavelengths" in cap
    assert "$c_b" in cap and "c_\\lambda" in cap
    assert "Sec. 1, text at eq. 1" in cap
    assert "pixel phase" in cap
    assert "zero-magnitude reference" in cap and "fifth" not in cap


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_wording_fixes(layout):
    joined = "\n".join(_drawn(layout)["strings"])
    assert "unit-ratio" not in joined and "unit-ratio" not in d12.FIGURES[0].caption
    assert "would be" not in joined
    assert "without the mask, the star's" in joined
    assert "agree by construction" in joined
    if not layout.is_slide:
        assert "Numerator / denominator" in joined
        assert "Ratio for this planet" not in joined
