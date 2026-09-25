"""Contracts of the radiometric-collection explainer figures.

The reference-experiment figure prints numbers. Each one is recomputed here
from the exact SI value of the Planck constant and the definition of the
jansky with 40-digit decimal arithmetic, never from hwoutils, and the test
reads the text the builder actually drew. The figure's parameters must be
those of the reference case code. The schematic figures print no numbers.
"""

import re
import sys
from decimal import ROUND_HALF_EVEN, Decimal, getcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text
from scipy.special import j0, j1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from explainers import _common as ex  # noqa: E402
from explainers import _export as exporter  # noqa: E402
from explainers import d03_radiometric_collection as d03  # noqa: E402

H_SI = Decimal("6.62607015e-34")  # J s, exact (SI 2019)
JY_SI = Decimal("1e-26")  # W m-2 Hz-1 per Jy, by definition


def _independent():
    """Printed values of the reference experiment, by hand arithmetic."""
    getcontext().prec = 40
    density = JY_SI / (H_SI * Decimal(700))  # photon s-1 m-2 nm-1
    photons = density * 1 * 1 * 1  # 1 m2, 1 nm, 1 s
    return {
        "density": density,
        "per_pixel": photons / 4,
        "variance": {q: Decimal(q) * photons for q in ("0.0", "0.5", "1.0")},
    }


def _two_dp(value):
    q = Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    return f"{q:,.2f}"


def _texts(slug, layout, mode="light"):
    spec = next(s for s in d03.FIGURES if s.slug == slug)
    with exporter.venue(mode, layout) as cast:
        fig = spec.build(layout, cast)
        try:
            return [
                t.get_text()
                for t in fig.findobj(Text)
                if t.get_text().strip() and t.get_visible()
            ]
        finally:
            plt.close(fig)


def _numbers(texts):
    """Numeric tokens in the drawn text, ignoring exponents and subscripts."""
    tokens = []
    for text in texts:
        stripped = re.sub(r"[\^_]\{?-?\d+\}?", "", text)
        tokens += re.findall(r"\d[\d,]*(?:\.\d+)?", stripped)
    return tokens


def test_anchor_matches_published_value():
    # The chapter prints 21,559.8597091736 photon s-1 m-2 nm-1.
    assert f"{_independent()['density']:.10f}" == "21559.8597091736"


def test_module_numbers_match_independent_arithmetic():
    nums = d03.reference_numbers()
    ind = _independent()
    np.testing.assert_allclose(nums["density"], float(ind["density"]), rtol=1e-12)
    np.testing.assert_allclose(nums["per_pixel"], float(ind["per_pixel"]), rtol=1e-12)
    for q, value in nums["variance"].items():
        np.testing.assert_allclose(
            value, float(ind["variance"][f"{q:.1f}"]), rtol=1e-12
        )


def test_parameters_are_those_of_the_reference_case():
    source = (ROOT / "tests" / "test_radiometry_boundary.py").read_text()
    assert "AREA_M2, BANDWIDTH_NM, EXPOSURE_S, N_PIXELS = 1.0, 1.0, 1.0, 4" in source
    assert '@pytest.mark.parametrize("qe", [0.0, 0.5, 1.0])' in source
    assert "jy_to_photons_per_nm_per_m2(1.0, 700.0)" in source
    r = d03.REFERENCE
    assert (r["area_m2"], r["bandwidth_nm"], r["exposure_s"], r["n_pixels"]) == (
        1.0,
        1.0,
        1.0,
        4,
    )
    assert (r["flux_jy"], r["wavelength_nm"]) == (1.0, 700.0)
    assert r["qe_values"] == (0.0, 0.5, 1.0)


@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_reference_figure_prints_exactly_the_checked_numbers(layout):
    ind = _independent()
    expected = {
        _two_dp(ind["density"]),  # density, and the photon rate over 1 m2 and 1 nm
        _two_dp(ind["per_pixel"]),
        *(_two_dp(v) for v in ind["variance"].values()),
    }
    assert expected == {"21,559.86", "5,389.96", "0.00", "10,779.93"}
    texts = _texts("d03-four-pixel-reference", layout)
    tokens = _numbers(texts)
    for value in expected:
        assert value in tokens, value
    assert tokens.count("5,389.96") == 4  # one label per pixel
    # Parameters printed as labels: 700 nm, 1 Jy, 1 nm, 1 m2, 1 s, 4 pixels,
    # and the three QE values.
    parameters = {"700", "1", "4", "0.0", "0.5", "1.0"}
    stray = set(tokens) - expected - parameters
    assert not stray, stray
    joined = "\n".join(texts)
    for label in ("1 Jy at 700 nm", "= 1 nm", "= 1 m", "= 1 s", "4 pixels"):
        assert label in joined, label
    # Each QE row pairs its q with its own value, in one drawn string.
    for q, value in ind["variance"].items():
        row = f"q = {q}:  {_two_dp(value)}"
        assert row in texts, row
    # The density carries its unit and measure on the next line.
    density = (
        r"$\Phi_{\lambda,\mathrm{nm}}$ = "
        + _two_dp(ind["density"])
        + "\n"
        + r"photon s$^{-1}$ m$^{-2}$ nm$^{-1}$"
    )
    assert any(t.startswith(density) for t in texts), density
    # The collected rate and the per-pixel count carry their units.
    assert any(_two_dp(ind["density"]) + "\n" + r"photon s$^{-1}$" in t for t in texts)
    assert texts.count(_two_dp(ind["per_pixel"]) + "\nphoton") == 4


@pytest.mark.parametrize(
    "slug",
    [
        "d03-radiometric-collection",
        "d03-collection-wavelength",
        "d03-collection-electrons",
    ],
)
@pytest.mark.parametrize("layout", [ex.DOC, ex.SLIDE], ids=["doc", "slide"])
def test_schematics_print_no_numbers(slug, layout):
    assert _numbers(_texts(slug, layout)) == []


def test_overview_names_every_factor_of_the_count_integral():
    joined = "\n".join(_texts("d03-radiometric-collection", ex.DOC))
    for symbol in (
        r"$A$",
        r"\Phi_\lambda",
        r"I_\lambda",
        r"\Omega_p",
        r"T_{\rm opt}",
        r"P_p",
        r"q_p",
        r"\mu_{e,p}",
        r"\int_{\rm live}",
        r"\lambda_b^-",
        r"\lambda_b^+",
    ):
        assert symbol in joined, symbol
    # Rate arriving and count accumulated are separate labels with their units.
    assert "arriving in $p$" in joined
    assert "accumulated in $p$" in joined
    assert r"photon s$^{-1}$ nm$^{-1}$" in joined
    assert "[electron]" in joined
    # The spatial response is chromatic, and the extended source has its own
    # replacement term.
    assert r"$P_p(\lambda)$" in joined
    assert r"$\int I_\lambda P_p\,d\Omega$ replaces" in joined
    assert r"$\Phi_\lambda T_{\rm opt}P_p q_p$" in joined
    assert "one wavelength" in joined


def test_psf_window_holds_less_than_all_light():
    frac = d03.psf_pixel_fractions()
    total = frac.sum()

    # Closed-form Airy encircled energy, 1 - J0^2 - J1^2 at pi r (r in
    # lambda/D): the 7 by 7 window contains the r = 3.5 disk and lies inside
    # the r = 3.5 sqrt(2) disk.
    def encircled(r):
        x = np.pi * r
        return 1.0 - j0(x) ** 2 - j1(x) ** 2

    assert encircled(3.5) < total < encircled(3.5 * np.sqrt(2.0)) < 1.0
    center = frac[3, 3]
    assert center == frac.max()
    assert total - center > 0.4  # much of the point source lands outside p


def test_captions_name_their_clauses():
    labels = {
        "d03-radiometric-collection": "radiometry-response-ownership",
        "d03-collection-wavelength": "radiometry-response-ownership",
        "d03-collection-electrons": "radiometry-detector-counts",
        "d03-four-pixel-reference": "photon-electron-reference-experiment",
    }
    specs = {s.slug: s for s in d03.FIGURES}
    assert set(specs) == set(labels)
    for slug, label in labels.items():
        assert f"`{label}`" in specs[slug].caption
        assert chr(0x2014) not in specs[slug].caption + specs[slug].alt
