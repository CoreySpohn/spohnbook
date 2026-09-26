"""Numbers the lenslet IFS explainer prints or draws, checked independently.

Each expected value is computed here from the instrument parameters by hand
arithmetic or by a small NumPy model of the same physics (square lenslet
cells, a clocked grid, dispersion linear in log wavelength, pixel-integrated
Moffat PSFlets smeared across each bin), never by calling the production
geometry. The figure module supplies the values it actually drew.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coronachrome.viz")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from explainers import d05_lenslet_ifs as d05

N_WAV_EXPECTED = 10


@pytest.fixture(scope="module")
def model():
    return d05._model()


def _grid_positions(n):
    """Lenslet-index coordinates in channel order (row-major over i, then j)."""
    half = n // 2
    idx = np.arange(n) - half
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)


def _edges():
    n = math.ceil(d05.RESOLVING_POWER * math.log(d05.BAND_NM[1] / d05.BAND_NM[0]))
    ratio = (d05.BAND_NM[1] / d05.BAND_NM[0]) ** (1.0 / n)
    return d05.BAND_NM[0] * ratio ** np.arange(n + 1)


def _centers():
    e = _edges()
    return np.sqrt(e[:-1] * e[1:])


def _centroids():
    """Detector centroids from the geometry: offset + clocked grid + dispersion."""
    scale = d05.PITCH_M / d05.PIXSIZE_M
    a = d05.ANGLE_RAD
    pos = _grid_positions(d05.N_LENSLETS)
    ny, nx = d05.DETECTOR_SHAPE
    disp = d05.DISPERSION_COEFFS[0] * np.log(_centers() / d05.LAM_REF_NM)
    gx = scale * (math.cos(a) * pos[:, 0] - math.sin(a) * pos[:, 1])
    gy = scale * (math.sin(a) * pos[:, 0] + math.cos(a) * pos[:, 1])
    xc = nx / 2 + gx[:, None] + disp[None, :]
    yc = (ny / 2 + gy)[:, None] * np.ones_like(disp)[None, :]
    return xc, yc


def test_wavelength_bins_and_printed_labels(model):
    centers = _centers()
    assert centers.size == N_WAV_EXPECTED
    np.testing.assert_allclose(model.lam, centers, rtol=1e-12)
    np.testing.assert_allclose(model.edges, _edges(), rtol=1e-12)
    # Labels printed on the figures and in the captions.
    assert [f"{v:.0f}" for v in centers[[0, 4, 5, 9]]] == ["605", "651", "663", "713"]


def test_origins_are_distinct_and_where_labeled(model):
    ny, nx = d05.FP_SHAPE
    # Optical center: the chapter's geometric center (n - 1) / 2.
    assert d05.OPTICAL_CENTER == ((nx - 1) / 2, (ny - 1) / 2) == (31.5, 31.5)
    # Lenslet-grid origin as the implementation places it: n / 2.
    assert model.grid_origin == (nx / 2, ny / 2) == (32.0, 32.0)
    # Detector trace origin: half the detector shape.
    dy, dx = d05.DETECTOR_SHAPE
    assert model.trace_origin == (dx / 2, dy / 2) == (60.0, 60.0)


def test_detector_centroids_follow_the_geometry(model):
    xc, yc = _centroids()
    np.testing.assert_allclose(model.xg, xc, atol=1e-9)
    np.testing.assert_allclose(model.yg, yc, atol=1e-9)
    # 174 / 13 detector pixels per lenslet pitch, printed as 13.4.
    assert f"{d05.PITCH_M / d05.PIXSIZE_M:.1f}" == "13.4"
    # The grid neighbor lands 6 px higher (printed in the animation).
    dy = yc[d05.LENSLET_B, 0] - yc[d05.LENSLET_A, 0]
    assert dy == pytest.approx(174 / 13 * math.sin(math.atan(0.5)))
    assert f"{dy:.0f}" == "6"
    # Longer wavelengths at larger x for a positive dispersion coefficient.
    assert np.all(np.diff(model.xg[d05.LENSLET_A]) > 0)


def test_shared_pixels_between_neighbors(model):
    xc, yc = _centroids()
    h = d05.HALF
    k = d05.SCAN_BIN

    def box(ch, w):
        x0, y0 = round(xc[ch, w]) - h, round(yc[ch, w]) - h
        return set(
            (y, x) for y in range(y0, y0 + 2 * h + 1) for x in range(x0, x0 + 2 * h + 1)
        )

    a, b = box(d05.LENSLET_A, k), box(d05.LENSLET_B, 0)
    rows = {y for y, _ in a & b}
    assert len(rows) == 3  # "shares three detector rows"
    assert len(a & b) == 21
    # Neighboring bins of one lenslet share two thirds of their pixels.
    assert len(a & box(d05.LENSLET_A, k + 1)) == 6 * (2 * h + 1)
    # The drawn box is the one the implementation assigns.
    x0, y0, w, hh = d05._psflet_box(model.ir, d05.LENSLET_A, k)
    xs = sorted({x for _, x in a})
    ys = sorted({y for y, _ in a})
    assert (x0, y0, w, hh) == (xs[0] - 0.5, ys[0] - 0.5, len(xs), len(ys))


def test_centroid_correction_applied_once(model):
    k = d05.SCAN_BIN
    frac = (_centers()[k] - 600.0) / 120.0
    (c0x, c0y), (c1x, c1y) = d05.CORRECTION_PX
    expected = (c0x + frac * (c1x - c0x), c0y + frac * (c1y - c0y))
    ch = d05.LENSLET_A
    shift = (model.xt[ch, k] - model.xg[ch, k], model.yt[ch, k] - model.yg[ch, k])
    np.testing.assert_allclose(shift, expected, atol=1e-9)
    assert (f"{expected[0]:.2f}", f"{expected[1]:.2f}") == ("0.56", "-0.79")
    # The stored template is recentered: its first moments vanish.
    plane = np.asarray(model.pack.templates[0, k])
    off = np.asarray(model.pack.offsets)
    total = plane.sum()
    assert abs((plane.sum(axis=0) * off).sum() / total) < 1e-9
    assert abs((plane.sum(axis=1) * off).sum() / total) < 1e-9


def test_edge_trace_loses_light_past_the_edge(model):
    xc, _ = _centroids()
    ch = d05.LENSLET_EDGE
    edge = d05.DETECTOR_SHAPE[1] - 0.5
    # Its long-wavelength footprint straddles the edge.
    assert round(xc[ch, -1]) + d05.HALF > edge > round(xc[ch, -1]) - d05.HALF
    # The drawn wide-detector image carries flux beyond the edge.
    pad = d05.EDGE_PAD
    nx = d05.DETECTOR_SHAPE[1]
    wide = d05._lenslet_image(model.ir_w, [ch])[:, pad : pad + nx + pad]
    lost = wide[:, nx:].sum()
    assert lost > 0.05
    # Independent: a Moffat PSFlet centered 0.7 px inside the edge puts a
    # sizable fraction of its light beyond it.
    x = np.linspace(-30, 30, 1201)
    alpha = d05.MOFFAT[0] * _centers()[-1] / d05.LAM_REF_NM
    prof = (1 + (x[:, None] ** 2 + x[None, :] ** 2) / alpha**2) ** (-d05.MOFFAT[1])
    frac = prof[:, x > edge - xc[ch, -1]].sum() / prof.sum()
    assert frac > 0.2


def _moffat_pixel(dx, dy, alpha, beta, n_quad=5):
    sub = np.linspace(-0.5 + 0.5 / n_quad, 0.5 - 0.5 / n_quad, n_quad)
    uu, vv = np.meshgrid(sub, sub, indexing="ij")
    r2 = (dx[..., None] + uu.ravel()) ** 2 + (dy[..., None] + vv.ravel()) ** 2
    return ((1.0 + r2 / alpha**2) ** (-beta)).mean(axis=-1)


def _independent_h():
    """Dense detector response of every (lenslet, bin), from first principles."""
    xc, yc = _centroids()
    edges = _edges()
    lam = _centers()
    ny, nx = d05.DETECTOR_SHAPE
    h = d05.HALF
    off = np.arange(-h, h + 1)
    ody, odx = np.meshgrid(off, off, indexing="ij")
    smear = d05.DISPERSION_COEFFS[0] * np.log(edges[1:] / edges[:-1])
    n_ch = xc.shape[0]
    mat = np.zeros((ny * nx, n_ch * lam.size))
    for ch in range(n_ch):
        for w in range(lam.size):
            px = np.round(xc[ch, w]) + odx
            py = np.round(yc[ch, w]) + ody
            alpha = d05.MOFFAT[0] * lam[w] / d05.LAM_REF_NM
            shifts = np.linspace(-0.5 * smear[w], 0.5 * smear[w], 5)
            g = np.mean(
                [
                    _moffat_pixel(
                        px - xc[ch, w] - s, py - yc[ch, w], alpha, d05.MOFFAT[1]
                    )
                    for s in shifts
                ],
                axis=0,
            )
            g = g / g.sum()
            ok = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
            rows = (py[ok] * nx + px[ok]).astype(int)
            mat[rows, ch * lam.size + w] = g[ok]
    return mat


def test_correlation_pattern_matches_independent_model(model):
    mat = _independent_h()
    live = np.flatnonzero(mat.sum(axis=0) > 0)
    inv = np.linalg.inv(mat[:, live].T @ mat[:, live])
    n = N_WAV_EXPECTED
    idx = np.r_[
        d05.LENSLET_A * n : (d05.LENSLET_A + 1) * n,
        d05.LENSLET_B * n : (d05.LENSLET_B + 1) * n,
    ]
    pair = inv[np.ix_(np.searchsorted(live, idx), np.searchsorted(live, idx))]

    def corr(c):
        sd = np.sqrt(np.diag(c))
        return c / np.outer(sd, sd)

    mine, drawn = corr(pair), corr(model.pair)
    # "neighboring bins: anticorrelated"
    assert np.all(np.diag(mine[:n, :n], 1) < -0.3)
    # "between lenslets: near zero here"
    assert np.abs(mine[:n, n:]).max() < 1e-2
    # The drawn matrix agrees with the independent one.
    np.testing.assert_allclose(drawn, mine, atol=2e-2)


def test_truth_fluxes_are_cell_integrals(model):
    """Input bin fluxes: the pixelized scene integrated over each square cell."""
    scene = d05._scene()
    lam = _centers()
    slope = 1.0 + 0.6 * (lam - 660.0) / 120.0
    p = d05.FP_PX_PER_LENSLET
    a = d05.ANGLE_RAD
    pos = _grid_positions(d05.N_LENSLETS)
    ny, nx = d05.FP_SHAPE
    n_sub = 120
    u = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    uu, vv = np.meshgrid(u * p, u * p, indexing="ij")
    for ch in (d05.LENSLET_A, d05.LENSLET_B):
        cx = nx / 2 + p * (math.cos(a) * pos[ch, 0] - math.sin(a) * pos[ch, 1])
        cy = ny / 2 + p * (math.sin(a) * pos[ch, 0] + math.cos(a) * pos[ch, 1])
        x = cx + math.cos(a) * uu - math.sin(a) * vv
        y = cy + math.sin(a) * uu + math.cos(a) * vv
        values = scene[np.round(y).astype(int), np.round(x).astype(int)]
        flux = values.mean() * p * p
        # The implementation integrates each cell with a coarser quadrature
        # (4 samples per cube pixel); it differs from this 120-by-120
        # reference by about 1 percent on the steep side of the core.
        np.testing.assert_allclose(model.truth[ch], flux * slope, rtol=2e-2)


def test_simulated_extraction_is_representative(model):
    """The pinned noise seed gives a chi-squared near its expectation."""
    n = N_WAV_EXPECTED
    resid, var = [], []
    for j, ch in enumerate((d05.LENSLET_A, d05.LENSLET_B)):
        resid.append(model.extracted[ch] - model.truth[ch])
        var.append(np.diagonal(model.pair)[j * n : (j + 1) * n] * model.sigma2)
    chi2 = float((np.concatenate(resid) ** 2 / np.concatenate(var)).sum())
    assert 2 * n / 3 < chi2 < 2 * n * 1.8


def test_specs_are_consistent():
    slugs = [s.slug for s in d05.FIGURES] + [s.slug for s in d05.ANIMATIONS]
    assert all(slug.startswith("d05-") for slug in slugs)
    assert len(set(slugs)) == len(slugs)
    for spec in (*d05.FIGURES, *d05.ANIMATIONS):
        assert "optics-ifs-products" in spec.caption
        assert chr(0x2014) not in spec.caption + spec.alt


def test_printed_correlation_numbers_match_independent_model(model):
    mat = _independent_h()
    live = np.flatnonzero(mat.sum(axis=0) > 0)
    inv = np.linalg.inv(mat[:, live].T @ mat[:, live])
    n = N_WAV_EXPECTED
    idx = np.r_[
        d05.LENSLET_A * n : (d05.LENSLET_A + 1) * n,
        d05.LENSLET_B * n : (d05.LENSLET_B + 1) * n,
    ]
    w = np.searchsorted(live, idx)
    sd = np.sqrt(np.diag(inv[np.ix_(w, w)]))
    r = inv[np.ix_(w, w)] / np.outer(sd, sd)
    adj = np.concatenate([np.diag(r[:n, :n], 1), np.diag(r[n:, n:], 1)])
    mine = (f"{adj.max():.2f}", f"{adj.min():.2f}", f"{np.abs(r[:n, n:]).max():.4f}")
    hi, lo, cross = d05.correlation_summary(model.pair, n)
    assert (
        (f"{hi:.2f}", f"{lo:.2f}", f"{cross:.4f}")
        == mine
        == ("-0.40", "-0.47", "0.0007")
    )


def test_second_neighbor_correlation_matches_independent_model(model):
    """Bins two apart: positive, the value the caption prints."""
    mat = _independent_h()
    live = np.flatnonzero(mat.sum(axis=0) > 0)
    inv = np.linalg.inv(mat[:, live].T @ mat[:, live])
    n = N_WAV_EXPECTED
    idx = np.r_[
        d05.LENSLET_A * n : (d05.LENSLET_A + 1) * n,
        d05.LENSLET_B * n : (d05.LENSLET_B + 1) * n,
    ]
    w = np.searchsorted(live, idx)
    sd = np.sqrt(np.diag(inv[np.ix_(w, w)]))
    r = inv[np.ix_(w, w)] / np.outer(sd, sd)
    two = np.concatenate([np.diag(r[:n, :n], 2), np.diag(r[n:, n:], 2)])
    assert np.all(two > 0)
    lo, hi = d05.second_neighbor_range(model.pair, n)
    assert (f"{lo:.2f}", f"{hi:.2f}") == (f"{two.min():.2f}", f"{two.max():.2f}")
    assert (f"{lo:.2f}", f"{hi:.2f}") == ("0.16", "0.19")
    (spec,) = [s for s in d05.FIGURES if s.slug == "d05-lenslet-ifs-extraction"]
    assert "+0.16 to +0.19" in spec.caption


def test_neighbor_offset_along_and_across_the_trace(model):
    """Grid neighbor (1, 0): pitch/pixel times (cos, sin) of the clocking."""
    scale = d05.PITCH_M / d05.PIXSIZE_M
    along = scale * math.cos(math.atan(0.5))
    across = scale * math.sin(math.atan(0.5))
    got = d05.neighbor_offset_px(model)
    np.testing.assert_allclose(got, (along, across), atol=1e-9)
    assert (f"{got[0]:.0f}", f"{got[1]:.0f}") == ("12", "6")
    (spec,) = d05.ANIMATIONS
    assert "12 pixels along the trace and 6 pixels across it" in spec.caption


def test_bin_width_and_smear_labels(model):
    e = _edges()
    widths = np.diff(e)
    assert 11.0 < widths.min() and widths.max() < 13.1
    assert f"{widths.mean():.0f}" == "12"  # "about 12 nm wide"
    smear = d05.DISPERSION_COEFFS[0] * math.log(1.2) / N_WAV_EXPECTED
    np.testing.assert_allclose(d05.bin_smear_px(model.edges), smear, rtol=1e-12)
    assert f"{smear:.2f}" == "2.55"


@pytest.fixture(scope="module")
def instrument_fig():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from explainers import _common as ex
    from explainers import _export as exporter

    with exporter.venue("light", ex.DOC) as cast:
        fig = d05.build_instrument(ex.DOC, cast)
        fig.canvas.draw()
        yield fig
    plt.close(fig)


def _by_gid(fig, gid):
    found = [a for a in fig.findobj() if a.get_gid() == gid]
    assert found, gid
    return found


@pytest.fixture(scope="module")
def origins_fig():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from explainers import _common as ex
    from explainers import _export as exporter

    with exporter.venue("light", ex.DOC) as cast:
        fig = d05.build_origins(ex.DOC, cast)
        fig.canvas.draw()
        yield fig
    plt.close(fig)


def _visible_text(fig):
    from matplotlib.text import Text

    return [t.get_text() for t in fig.findobj(Text) if t.get_visible()]


def test_artists_mark_the_three_origins(origins_fig, model):
    fig = origins_fig
    ny, nx = d05.FP_SHAPE
    # Hand arithmetic: (n - 1) / 2 and n / 2 of a 64-pixel cube.
    center = ((nx - 1) / 2, (ny - 1) / 2)
    grid = (nx / 2, ny / 2)
    (oc,) = _by_gid(fig, "d05-zoom-optical-center")
    (go,) = _by_gid(fig, "d05-zoom-lenslet-grid-origin")
    assert (oc.get_xdata()[0], oc.get_ydata()[0]) == center == (31.5, 31.5)
    assert (go.get_xdata()[0], go.get_ydata()[0]) == grid == (32.0, 32.0)
    (oc_text,) = _by_gid(fig, "d05-zoom-optical-center-label")
    (go_text,) = _by_gid(fig, "d05-zoom-grid-origin-label")
    # The wording of the handbook-versus-implementation statement is pinned.
    assert oc_text.get_text() == (
        "optical center (31.5, 31.5)\n= $(n_\\mathrm{cube} - 1)/2$"
    )
    assert go_text.get_text() == (
        "lenslet-grid origin (32, 32)\n= $n_\\mathrm{cube}/2$ in coronachrome"
        "\n(implementation)"
    )
    # The marked offset: n / 2 - (n - 1) / 2 = 1/2 on each axis.
    (arrow,) = _by_gid(fig, "d05-zoom-offset-arrow")
    # One head, from the optical center (tail) to the grid origin (head).
    assert arrow.xy == grid and arrow.xyann == center
    assert arrow.arrow_patch.get_arrowstyle().__class__.__name__ == "CurveFilledB"
    (offset,) = _by_gid(fig, "d05-zoom-offset-label")
    half = (grid[0] - center[0], grid[1] - center[1])
    assert half == (0.5, 0.5)
    assert offset.get_text() == "+0.5 px on each\naxis (grid - optical)"
    # Detector trace origin: half of the 120-pixel detector on each axis.
    dy, dx = d05.DETECTOR_SHAPE
    assert (dx / 2, dy / 2) == (60.0, 60.0)
    ticks = _by_gid(fig, "d05-trace-origin")
    assert all(t.get_xdata()[0] == dx / 2 for t in ticks)
    lo = min(min(t.get_ydata()) for t in ticks)
    hi = max(max(t.get_ydata()) for t in ticks)
    assert lo < dy / 2 < hi
    # The ticks leave the 663 nm centroid circle (0.7 px away) uncovered.
    gap = min(abs(y - dy / 2) for t in ticks for y in t.get_ydata())
    assert gap > 1.0
    (label,) = _by_gid(fig, "d05-trace-origin-label")
    assert label.get_text() == (
        "detector trace origin (60, 60)\n"
        "= $n_\\mathrm{det}/2$ in coronachrome (implementation):\n"
        "lenslet (0, 0) at reference wavelength 660 nm"
    )
    # Lenslet (0, 0) sits on the anchor at the reference wavelength only
    # because the dispersion polynomial has no constant term.
    assert d05.DISPERSION_COEFFS[-1] == 0.0
    assert d05.DISPERSION_COEFFS[0] * math.log(660.0 / d05.LAM_REF_NM) == 0.0


def test_reference_wavelength_falls_between_labeled_bins(origins_fig):
    """660 nm is not a bin center: the origin sits between 651 and 663 nm."""
    xc, _ = _centroids()
    lam = _centers()
    below = np.flatnonzero(lam < d05.LAM_REF_NM).max()
    assert [f"{lam[below]:.0f}", f"{lam[below + 1]:.0f}"] == ["651", "663"]
    ch = d05.LENSLET_A
    x0 = d05.DETECTOR_SHAPE[1] / 2
    assert xc[ch, below] < x0 < xc[ch, below + 1]
    # 140 ln(663.3 / 660) = 0.7 px: the origin is left of the 663 nm circle.
    assert f"{xc[ch, below + 1] - x0:.1f}" == "0.7"
    for k in (below, below + 1):
        (text,) = _by_gid(origins_fig, f"d05-origin-bin-{lam[k]:.0f}")
        assert text.get_text() == f"{lam[k]:.0f} nm"
        assert text.xy[0] == pytest.approx(xc[ch, k])
    (name,) = _by_gid(origins_fig, "d05-origin-lenslet-label")
    assert name.get_text() == "lenslet 24 = grid (0, 0)"
    # Channel 24 is grid index (0, 0) of the 7-by-7 grid, row-major.
    assert tuple(_grid_positions(d05.N_LENSLETS)[ch]) == (0.0, 0.0)


def test_origins_figure_carries_the_entrance_panel(origins_fig):
    texts = _visible_text(origins_fig)
    assert "zoom (b)" in texts
    # The carried panel shows the zoom outline only: its origin marks are
    # hidden, so only the zoom panel's two marks are visible.
    marks = [
        a
        for a in origins_fig.findobj(lambda a: hasattr(a, "get_label"))
        if a.get_label() in ("optical center", "lenslet-grid origin")
        and a.get_visible()
    ]
    assert sorted(a.get_gid() for a in marks) == [
        "d05-zoom-lenslet-grid-origin",
        "d05-zoom-optical-center",
    ]
    # A small anchor: narrower than the zoom panel.
    ent, zoom = origins_fig.axes[0], origins_fig.axes[1]
    assert ent.get_position().width < 0.8 * zoom.get_position().width
    (tag,) = _by_gid(origins_fig, "d05-carried-tag")
    assert tag.get_text() == "(a) entrance plane\n(from the previous figure)"


def test_instrument_figure_shows_no_origins(instrument_fig):
    texts = " ".join(_visible_text(instrument_fig))
    assert "origin" not in texts and "zoom" not in texts
    gids = {a.get_gid() for a in instrument_fig.findobj() if a.get_gid()}
    assert not any("origin" in g or "zoom" in g for g in gids)
    # The two origin marks the library draws are hidden.
    for line in instrument_fig.findobj(lambda a: hasattr(a, "get_label")):
        if line.get_label() in (
            "optical center",
            "lenslet-grid origin",
            "detector trace origin",
        ):
            assert not line.get_visible()


def test_origins_caption_keeps_the_convention_statement():
    specs = {s.slug: s for s in d05.FIGURES}
    origins = specs["d05-lenslet-ifs-origins"].caption
    instrument = specs["d05-lenslet-ifs-instrument"].caption
    statement = (
        "The optical center belongs to the cube and sits at its geometric "
        "center (31.5, 31.5). The example implementation drawn here, "
        "coronachrome, places lenslet (0, 0) at (32, 32), half a pixel away on "
        "each axis; the {ref}`limitations page <limitations-optics>` records "
        "that difference."
    )
    assert statement in origins
    assert "(31.5, 31.5)" not in instrument and "(32, 32)" not in instrument
    assert "fig-explainer-d05-lenslet-ifs-origins" in instrument
    assert "(60, 60)" in origins and "120-pixel" in origins
    assert "651 nm and 663 nm" in origins
    assert "the chapter sets no convention for this point" in origins
    # Half a cube pixel over a lenslet pitch of 6 cube pixels is 1/12.
    assert 0.5 / d05.FP_PX_PER_LENSLET == pytest.approx(1 / 12)
    assert "half a cube pixel on each axis (1/12 of a lenslet pitch here)" in origins
    assert "$n_\\mathrm{cube}$ = 64" in origins and d05.FP_SHAPE == (64, 64)


def test_wavelength_sense_on_detector_and_side_view(instrument_fig):
    fig = instrument_fig
    (short,) = _by_gid(fig, "d05-label-605")
    (long,) = _by_gid(fig, "d05-label-713")
    assert short.get_text() == "605 nm" and long.get_text() == "713 nm"
    # Detector: dispersion along +x, longer wavelengths at larger x.
    assert short.get_position()[0] < long.get_position()[0]
    # Side view: page +y is detector +x, so 713 nm is drawn above 605 nm, and
    # the apex-up prism deviates the shorter wavelength more, toward its base.
    (s605,) = _by_gid(fig, "d05-side-605")
    (s713,) = _by_gid(fig, "d05-side-713")
    assert s713.get_position()[1] > s605.get_position()[1]
    (prism,) = _by_gid(fig, "d05-side-prism")
    xy = prism.get_xy()
    apex_y = xy[:, 1].max()
    base_y = xy[:, 1].min()
    assert apex_y > 0 > base_y
    spots = dict(d05.SIDE_SPOTS)
    assert spots[605.0] < spots[660.0] < spots[713.0]
    # The collimated beam meets the prism horizontally; every wavelength
    # leaves bent toward the base (downward), the longest the least.
    slopes = {}
    for lam in (605, 660, 713):
        (ray,) = _by_gid(fig, f"d05-side-ray-{lam}")
        x, y = ray.get_xdata(), ray.get_ydata()
        slopes[lam] = (y[-1] - y[0]) / (x[-1] - x[0])
    assert slopes[605] < slopes[660] < slopes[713] < 0


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("slide", [False, True])
def test_no_label_carries_a_stroked_halo(mode, slide):
    """Labels sit on plain backing boxes; a stroked halo blobs the glyphs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from explainers import _common as ex
    from explainers import _export as exporter
    from matplotlib.text import Text

    layout = ex.SLIDE if slide else ex.DOC
    with exporter.venue(mode, layout) as cast:
        figs = [spec.build(layout, cast) for spec in d05.FIGURES]
        figs += [spec.build(layout, cast).fig for spec in d05.ANIMATIONS]
        for fig in figs:
            stroked = [t.get_text() for t in fig.findobj(Text) if t.get_path_effects()]
            assert stroked == []
            plt.close(fig)


def test_bin_label_carries_the_scanned_wavelength(instrument_fig, model):
    (label,) = _by_gid(instrument_fig, "d05-bin-label")
    assert label.get_text() == "bin footprint:\nlenslet 24, $\\lambda$ = 651 nm"
    x0, y0, _, _ = d05._psflet_box(model.ir, d05.LENSLET_A, d05.SCAN_BIN)
    assert label.get_position() == (x0, y0 - 0.15)
    # No floating corner readout.
    texts = [t for t in instrument_fig.findobj(lambda a: hasattr(a, "get_label"))]
    for t in texts:
        if getattr(t, "get_label", lambda: "")() == "scan readout":
            assert not t.get_visible()
