# Figures and color: one encoding per quantity

*Draft contract: the choices marked pending or proposed are open.*

The other chapters fix what a number means as it crosses a library boundary.
This chapter fixes what a color means. A reader who has seen one focal-plane
image from the suite should recognize the next one, in another library and
another document, by its colormap alone, and a reader who has followed a planet
through one figure should find it in the same hue in the next. That recognition
is a contract like the others, and it fails the same way: two libraries encode
one quantity two ways, or one document encodes two quantities one way. The rules
below apply to every figure the suite's libraries ship in their documentation,
tutorials and papers.

## The principle

A colormap or a palette color is chosen by the quantity it encodes, never per
figure. The author of a figure decides what is shown; the role decides how it
looks. A pupil transmission map, a focal-plane contrast map and a detector
readout are three quantities, so they take three maps, and a second figure of
the same quantity takes the same map without a new decision.

Dynamic range is the norm's job, not the colormap's. Whether an image is shown
on a log or a linear scale, and where a diverging map places its midpoint, is a
property of the quantity's normalization. The colormap stays fixed while the
norm absorbs the range, so a separate map for wide-range images has no place in
the contract.

The same quantity looks the same on every page and in every library. This is
what makes an image in one library's tutorial comparable with the same image in
another's, and it is what a caption relies on when it omits the colormap name.

Light and dark modes are the same roles retuned by hwostyle. A figure never
hand-picks a hex value for a mode: it names the role, and the active mode
resolves the role to the muted variant for a page or the saturated variant for a
slide. A light-mode hex value in a dark figure, or the reverse, is a defect.

Every pair of series that must survive grayscale printing or color-vision
deficiency carries a second channel: a marker, a linestyle or a direct label.
The light-mode palette is desaturated so that its hues sit in a narrow band of
luminance; two hues alone do not separate two series in print, and the remedy
is a second channel, not a different palette.

```{figure} figures/hwo-conventions-color-light.svg
:class: only-light
:name: fig-color-roles

Color roles: the proposed colormap per image quantity, and the palette role per plotted entity with its second channel.
```

```{figure} figures/hwo-conventions-color-dark.svg
:class: only-dark

Color roles: the proposed colormap per image quantity, and the palette role per plotted entity with its second channel.
```

## Images: colormap by quantity

Every image quantity the suite renders has one role. The role name is the API;
the map names in the table describe what the role resolves to in each mode.

| Quantity | Role name | Light | Dark | Norm |
|---|---|---|---|---|
| Pupil-plane transmission or amplitude (0 to 1) | pupil | white to teal | black to cyan | linear |
| Pupil-plane optical path difference, signed, nm | opd | brown-white-teal (BrBG) | same | symmetric about 0 |
| Pupil-plane phase, cyclic | phase | twilight | twilight | periodic |
| Focal-plane intensity, PSF, contrast map (model, noiseless) | intensity | viridis | viridis | log |
| Detector readout, electrons or electrons per second (measured, noisy) | readouts | magma | magma | log or linear |
| Difference or ratio against a reference image | residual | blue-white-red (RdBu_r) | same | symmetric about 0 (difference) or 1 (ratio) |
| Detection statistic, SNR map | statistic | purple-white-orange (PuOr) | same | symmetric about 0 |
| Probability density, posterior, likelihood | probability | white to teal to black | black to cyan to white | linear |
| Boolean mask, validity | mask | two neutral grays | two neutral grays | none |

`high_dynamic_range` is retired as a role, because dynamic range is the norm's
job. A map name in a call, such as `cmap="magma"`, is a defect; a role name,
such as `cmap=hwostyle.cmaps.readouts` for an image or
`color=hwostyle.roles.planet` for a curve, is the contract. hwostyle ships ten
roles today: `intensity`, `readouts`, `high_dynamic_range`, `residual`, `opd`,
`phase`, `probability`, `mask`, `brand_intensity` and `brand_diverging`. In that
registry `opd` and `residual` resolve to the same diverging map, `intensity`
resolves to magma in dark mode and so shares a map with `readouts`, and
`probability` resolves to YlOrRd in light mode and plasma in dark mode. This
chapter proposes `pupil` and `statistic` as new roles, moves `opd` to BrBG so
that a wavefront map is never mistaken for a residual, moves `probability` to
the brand ramp, keeps `intensity` on viridis in both modes, and sets `mask` to
two neutral grays so that a validity mask never competes with data for a hue.

## Curves and markers: color by role

A plotted entity has one palette role, and the role carries the second channel
that separates it from the roles it most often shares a panel with.

| Role | Color | Second channel |
|---|---|---|
| planet, and measurements of it | cyan | measurements as markers with error bars |
| star, and starlight leakage | yellow | star glyph marker |
| disk, and exozodiacal light | purple | none required |
| local zodiacal light and other sky backgrounds | green | none required |
| model, prediction, posterior draws | pink | solid line; ensemble alpha from the draw count |
| truth, reference geometry (inner and outer working angle rings, lines of nodes, injected values) | neutral gray, scenery rank | solid |
| instrument terms (detector noise, noise floors) | darker neutral gray | dashed |
| uncertainty envelope | the parent series' hue at low alpha | outline when alpha encodes density |
| alert, status | red, reserved | icon or label, never color alone |
| the answer, at most one mark per panel | the text color | none |

Every mark occupies exactly one of four ranks, and its rank is set by its value
(its distance from the background), not by its hue.

**Furniture** is the lowest rank: coordinate cages, gridlines, panes, spines and
shaded exclusion regions. It sits just above the background and measurably below
everything else; if a gridline and a data curve sample to the same value in the
rendered image, the figure has no layers. Grids are off by default and earn
their ink only on a panel a reader will measure values from.

**Scenery** is reference geometry: inner and outer working angle rings, lines of
nodes, star markers, leader lines and their labels. It is a neutral gray one
step above furniture, present so that the data has a frame, and never a palette
hue.

**Data** is the full palette color. It is the thing the figure is about, and it
is the only rank that uses hue.

**Answer** is the text color, and at most one mark per panel may use it: the
decision, the threshold crossing, the measurement everything else rests on. A
leader line, a coordinate cage or a plain marker in the text color spends the
loudest value available on the quietest content; on a black background this is
the commonest failure, because white is the default.

The palette order is fixed: cyan, pink, yellow, green, red, purple. A series
with no role takes the next unused slot in that order. A hue is never cycled to
a second meaning in one figure or in one document: if cyan is the planet in
panel (a), cyan is not a model in panel (b), and the same holds across every
figure of one paper or one documentation page. When a document runs out of
distinct meanings before it runs out of hues, split the figure rather than reuse
a color.

## Time, wavelength and ensembles

Time along a track is an alpha ramp in one hue, faint at the earliest epoch and
full at the latest, never a rainbow. A rainbow spends every hue on one axis,
leaves nothing for the other entities in the panel, and is not ordered under
color-vision deficiency.

Wavelength is encoded by the wavelength-to-color mapping
(`hwostyle.colors.wavelength_to_hex`), so that one channel has the same color in
every figure, or by the spectral palette when the channels are named lines
rather than a continuous band. Neither is a categorical palette, and a
wavelength series never takes a slot from the role palette.

An ensemble (posterior draws, seeds, Monte Carlo trials) is drawn so that its
darkness encodes its density. The per-curve alpha is set from the count, with
`alpha = min(0.6, 3 / n)` as the starting rule and roughly half of that in dark
mode, where a bright stroke on black blooms. If the bundle is at full saturation
in every frame, the ink has stopped reporting the spread. Above about fifty
curves, draw a percentile band plus five individual draws. Opacity that a
reader will interpret as probability must be the probability; driving it from a
rank or a convenience weight is a false claim the panel makes silently.

## Named acceptance fixtures

These fixtures run against hwostyle's registry and against each library's
documentation build. Passing them verifies that the encoding contract holds; it
does not verify the science of any figure.

| Fixture | Required discriminator |
|---|---|
| **C-ROLE-SWATCH** | Every image role and every curve role renders in both modes; within one mode no two image roles resolve to the same map and no two curve roles resolve to the same color. The swatch figure in this chapter is the reference rendering. |
| **C-GRAY-PAIR** | Every pair of curve roles that can share a panel is distinguishable after conversion to grayscale by its second channel alone: marker, linestyle or direct label. A pair separated only by hue fails. |
| **C-DEFAULT-FREE** | No figure in the suite's documentation calls a matplotlib map or a default color by name: no map name in a `cmap` argument, no cycle color such as `C0`, no bare hex value. Every color in figure code is a role. |

## Coverage

Gates are **S0** convention decisions and **S1** boundary repairs, as the
handbook index defines them. Listed owners and gates are proposed, not completed
work.

| Finding | Proposed owner | Gate |
|---|---|---|
| COL-01: hwostyle `intensity` and `readouts` share magma in dark mode | hwostyle | S0 |
| COL-02: hwostyle `opd` and `residual` share one map | hwostyle | S0 |
| COL-03: hwostyle planet and data, disk and envelope, model and highlight share colors | hwostyle | S0 |
| COL-04: hwostyle has no `pupil` or `statistic` role | hwostyle | S0 |
| COL-05: eyepiece corner plots hardcode the truth color and the density map | eyepiece | S1 |
| COL-06: eyepiece image functions take map names, not roles, as their override | eyepiece | S1 |
| COL-07: documentation pages call maps by name where a role should exist | this handbook | S1 |
