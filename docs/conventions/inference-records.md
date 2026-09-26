# Measurements, probability, and records

*Draft contract: the choices marked pending or proposed are open.*

A correct orbit and a correct detector model do not automatically define a correct
likelihood. The likelihood describes the data that actually reached the analyst.
It therefore depends on the reporting rule, the calibration and reduction, and
which other records share the same underlying noise or evidence.

```{figure} figures/explainer-d01-overview-inference-light.png
:class: only-light
:name: fig-explainer-d01-overview-inference
:width: 80%
:alt: The observation map with the raw frame, the records and the inferred orbit at full strength, its Records tag drawn bold and keyed as this chapter, and every other part faded toward the background. A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

Where this chapter sits in the physical observation: the raw frame, the records and the inferred orbit, at full strength, with the rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

```{figure} figures/explainer-d01-overview-inference-dark.png
:class: only-dark
:width: 80%
:alt: The observation map with the raw frame, the records and the inferred orbit at full strength, its Records tag drawn bold and keyed as this chapter, and every other part faded toward the background. A two-row schematic. Top row, left to right: a side view titled target system, side view, observer to the right, of a star with a planet inside a tilted, hatched dust disk, labeled star, planet and exozodiacal dust, with a dotted sky plane; yellow, cyan and purple rays labeled starlight, reflected and scattered leave the system toward the right, cross paired slash marks labeled interstellar distance, not to scale, and converge on an edge-on telescope aperture inside a green dotted cloud labeled local zodiacal dust, which sends two green rays nearly parallel to the others into the aperture; the beam continues through a rail labeled pupil, focal mask, Lyot stop and detector, captioned generic coronagraph, not a flight design. A hollow arrow labeled read out leads down to a small pixelated raw frame, titled raw frame, synthetic, with a saturated central stellar leakage, a planet blob, a faint dust arc and a noisy floor labeled local zodi, uniform floor. Hollow arrows labeled reduce and infer with an assumed model lead left to a card of records, one per visit, listing epoch t sub k, event D sub k, offsets xi sub k and eta sub k, covariance C sub k and calibration revision, and then to a sky panel with an east and north compass, the star at the center, four cyan measured positions, one ringed and labeled current visit, and a bundle of thin pink posterior orbit tracks, tight along the measured arc and fanning out on the opposite side. Small outlined tags name the chapter for each part: Geometry at the system, Dust at both dust clouds, Radiometry at the read out, Optics at the coronagraph and Records above the record card. A key at the top separates light arrows from data arrows and explains the tags, and a badge reads schematic of the simulation scope, not to scale.

Where this chapter sits in the physical observation: the raw frame, the records and the inferred orbit, at full strength, with the rest of the map dimmed ({ref}`full map <fig-explainer-d01-overview>`). An original schematic of the physical system this book's libraries simulate, not an identity, a convention or an instrument design.
```

This chapter defines the intended statistical boundaries. The proposed profiles
remain pending the reporting law decision and the evidence and parameter
chart decision in [the handbook](index.md). Reports that current
implementations disagree with these profiles are recorded under {ref}`limitations-records`. Nothing here asserts that a generic joint imaging likelihood or
portable posterior already exists.

(records-reporting-law)=
## Define the experiment before the uncertainty

Write an experiment as a resolved response, exposure interval, noise process and
reporting rule. Let $\theta$ denote model parameters, $Y$ the complete potential
measurement, and $D$ a reporting event. The actually acquired record $R$ can be
much smaller than $Y$.

```{figure} figures/explainer-d08-experiment-record-model-light.png
:class: only-light
:name: fig-explainer-d08-experiment-record-model
:alt: Schematic in three labeled zones. Left, a dashed region named simulated world holds a star at the focus of an orbit, a planet on it, and a truth record box listing the true orbit and flux, the planet ID and the noise seed. Rays of starlight and planet light cross a scale break to a telescope aperture and a detector in the middle zone; a dashed gray arrow labeled noise draw runs from the truth record to the detector, and a red crossed arrow from the truth record carries the note: no path, truth or scores never reach inference or policy. A configured visit box, a command, receives a next visit arrow from the observing policy. The detector feeds two 9 by 9 pixel images of raw electrons, visit 1 with a bright planet and visit 2 fainter, each with the summed 3 by 3 pixels outlined by a dotted square. Each record card lists measured fields above fixed experiment settings. Visit 1 reads D = 1, detected, and F = 84.8 electrons; visit 2 reads D = 0 and F not reported; both list the threshold L = 30 electrons and sigma = 6.0 electrons. A dashed forced photometry card, labeled as an alternative experiment that replaces the visit 2 record, reads D = 0 and F = 17.7 electrons, below L. The two records feed an inference model box on the right, which feeds the observing policy. Below a dashed divider, an evaluation box for simulation-only scoring receives the posterior and, along the bottom, the truth record; a red crossed arrow from evaluation back to the inference model is labeled no path.

The proposed information boundary of a simulated imaging experiment ({ref}`records-reporting-law`, {ref}`records-identities`, {ref}`records-campaign-causality`). The simulated world holds the scene and a truth record. Truth reaches the detector only through the simulated light and the noise draw, and reaches evaluation directly, for simulation-only scoring; neither truth nor scores reach the inference model or the observing policy, which is a decision rule rather than a hypothesis. Two visits of one target give the two records of the report-only-on-detection experiment, R = (D = 1, F) and R = (D = 0), where D records whether F exceeded the threshold L. The threshold L and the standard deviation sigma of F before any selection are fixed, known settings of the experiment, not measured values. The dashed card is a different experiment, forced photometry, which replaces the visit 2 record and reports F = 17.7 e$^-$ from the same pixels although F is below L ({ref}`Efron and Hastie 2016, Sec. 9.6 <source-efron2016>`). The 9 by 9 pixel cutouts are simulated raw readouts: a known background of 20 e$^-$ per pixel, a pixel-integrated Gaussian planet image and independent Gaussian noise of 2 e$^-$ per pixel. The reduction subtracts the background and sums the 9 dotted pixels, those whose centers lie within 1.5 pixels of the fixed target position, so sigma = 6.0 e$^-$. The next figure details the reporting law. The figure is a schematic of a proposed convention, not an example implementation, and does not assert that the current tutorial or campaign software enforces every separation it shows.
```

```{figure} figures/explainer-d08-experiment-record-model-dark.png
:class: only-dark
:alt: Schematic in three labeled zones. Left, a dashed region named simulated world holds a star at the focus of an orbit, a planet on it, and a truth record box listing the true orbit and flux, the planet ID and the noise seed. Rays of starlight and planet light cross a scale break to a telescope aperture and a detector in the middle zone; a dashed gray arrow labeled noise draw runs from the truth record to the detector, and a red crossed arrow from the truth record carries the note: no path, truth or scores never reach inference or policy. A configured visit box, a command, receives a next visit arrow from the observing policy. The detector feeds two 9 by 9 pixel images of raw electrons, visit 1 with a bright planet and visit 2 fainter, each with the summed 3 by 3 pixels outlined by a dotted square. Each record card lists measured fields above fixed experiment settings. Visit 1 reads D = 1, detected, and F = 84.8 electrons; visit 2 reads D = 0 and F not reported; both list the threshold L = 30 electrons and sigma = 6.0 electrons. A dashed forced photometry card, labeled as an alternative experiment that replaces the visit 2 record, reads D = 0 and F = 17.7 electrons, below L. The two records feed an inference model box on the right, which feeds the observing policy. Below a dashed divider, an evaluation box for simulation-only scoring receives the posterior and, along the bottom, the truth record; a red crossed arrow from evaluation back to the inference model is labeled no path.

The proposed information boundary of a simulated imaging experiment ({ref}`records-reporting-law`, {ref}`records-identities`, {ref}`records-campaign-causality`). The simulated world holds the scene and a truth record. Truth reaches the detector only through the simulated light and the noise draw, and reaches evaluation directly, for simulation-only scoring; neither truth nor scores reach the inference model or the observing policy, which is a decision rule rather than a hypothesis. Two visits of one target give the two records of the report-only-on-detection experiment, R = (D = 1, F) and R = (D = 0), where D records whether F exceeded the threshold L. The threshold L and the standard deviation sigma of F before any selection are fixed, known settings of the experiment, not measured values. The dashed card is a different experiment, forced photometry, which replaces the visit 2 record and reports F = 17.7 e$^-$ from the same pixels although F is below L ({ref}`Efron and Hastie 2016, Sec. 9.6 <source-efron2016>`). The 9 by 9 pixel cutouts are simulated raw readouts: a known background of 20 e$^-$ per pixel, a pixel-integrated Gaussian planet image and independent Gaussian noise of 2 e$^-$ per pixel. The reduction subtracts the background and sums the 9 dotted pixels, those whose centers lie within 1.5 pixels of the fixed target position, so sigma = 6.0 e$^-$. The next figure details the reporting law. The figure is a schematic of a proposed convention, not an example implementation, and does not assert that the current tutorial or campaign software enforces every separation it shows.
```

```{figure} figures/hwo-conventions-measurement-light.svg
:class: only-light
:name: fig-measurement

A report-only-on-detection experiment. A successful nondetection is an
observed event, not a missing datum. Retained measurements and the selection event
belong to one probability model. The bottom condition is an exact information
anchor, independent of the orbit or noise implementation.
```

```{figure} figures/hwo-conventions-measurement-dark.svg
:class: only-dark

A report-only-on-detection experiment. A successful nondetection is an
observed event, not a missing datum. Retained measurements and the selection event
belong to one probability model. The bottom condition is an exact information
anchor, independent of the orbit or noise implementation.
```

Consider a hypothetical flux measurement $F\sim\mathcal N(f(\theta),\sigma^2)$,
with a fixed, known threshold $L$ and $D=1$ precisely when $F>L$. If every attempted
visit is recorded and nondetections report only the event, the contribution is

$$
p(R\mid\theta)=
\begin{cases}
\Phi_{\rm N}((L-f(\theta))/\sigma), & R=(D=0),\\
\mathcal N(F;f(\theta),\sigma^2)\,\mathbf 1_{F>L}, & R=(D=1,F).
\end{cases}
$$

$\Phi_{\rm N}$ is the standard-normal CDF, not the Lambert phase function. This
hybrid record has probability mass for the null event and density for a reported
flux. The integrated contributions sum to one. Do not multiply the reported-flux
density by $P(D=1\mid\theta)$ again: the joint event is already represented
({ref}`Loredo 2004, the survey-likelihood footnote in its trans-Neptunian section <source-loredo2004>`).

A sample consisting only of selected detections under a conditioned sampling
design uses $p(F\mid D=1,\theta)$, including its selection normalization
({ref}`Casella and Berger 2024, Exercise 1.52 <source-casella2024>`). Forced
photometry instead reports $F$ even when $D=0$ and can carry information below the
detection threshold. These are different experiments, the distinction between
censoring and truncation ({ref}`Efron and Hastie 2016, Sec. 9.6 <source-efron2016>`). Choose one explicitly;
neither a detection flag nor a field named `sigma` chooses it for you.

For joint astrometry and flux, replace the scalar $F$ by the actual joint random
vector and apply the real selection rule. Conditioning on detection can change
its mean, covariance and shape. An unconditional inverse Fisher matrix is not by
itself the distribution of selected measurements. The first campaign needs a
calibrated reporting model, not merely an available flux-error calculation.

(records-null-information)=
### The simplest information test

Expected information is about $R$, the record returned by this experiment
({ref}`Lindley 1956, Definition 2 and eq. 9 <source-lindley1956>`):

$$
I(\theta;R)=\mathbb E_{\theta,R}
\left[\log\frac{p(R\mid\theta)}{p(R)}\right].
$$

Use natural logarithms and report nats; the base is this book's choice. For two equally weighted hypotheses, if
both always produce the same null record, posterior weights remain $(1/2,1/2)$
and information is zero ({ref}`Lindley 1956, Theorem 1 <source-lindley1956>`). If the record identifies the hypothesis perfectly,
information is $\ln 2$ nats. The limitations recorded under {ref}`limitations-records` include a report of a
scheduler that fails the first limit.

A response also has a domain. Below an inner working angle, beyond a tabulated
outer boundary, or outside a calibrated wavelength range, record an explicit
unsupported or physically unobservable result according to the response's
contract. These states are not an infinite sensitivity. Prediction, sampling,
likelihood and candidate scoring must use the same boundary policy. When an
exposure or calibration changes the response, its cache identity changes too,
or the changed arrays must remain dynamic function inputs (see {ref}`limitations-records`).

(records-covariance)=
## Covariance describes an ordered, dimensional vector

For a vector such as $(\xi,\eta,f)$, retain axis names, order, units, covariance,
and whether nuisance variables were held fixed or marginalized. Its mixed entries
have mixed units: $C_{\xi f}$ has units of angle times flux. A bare square array is
not a portable uncertainty model.

Under a linear transformation $y=Ax$, covariance transforms exactly as
$C_y=AC_xA^T$ ({ref}`JCGM 102:2011, 6.2.1.3 <source-jcgm2011>`). For a nonlinear
transformation, $JC_xJ^T$ is a local approximation ({ref}`JCGM 100:2008, 5.1.2 <source-jcgm2008>`);
transform posterior samples or use the full distribution where the approximation
is inadequate. Changing units rescales cross terms as well as diagonal entries.

Shared calibration is a concrete source of dependence. Suppose two observations
obey $y_i=f_i(\theta)+b+\epsilon_i$, with independent noise variances
$\sigma_i^2$ and a common offset $b\sim\mathcal N(0,\tau^2)$. Marginalizing $b$
gives

$$
C=\begin{bmatrix}\sigma_1^2&0\\0&\sigma_2^2\end{bmatrix}
+\tau^2\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

An explicit-$b$ model and a correctly marginalized model should agree. Adding a
free $b$ after using that marginalized covariance counts the same uncertainty
twice. Deleting its off-diagonal terms falsely treats the calibration as two
independent offsets ({ref}`JCGM 100:2008, 5.2.2 note 1 <source-jcgm2008>`). Record the nuisance identity and treatment, not just the
resulting error bars.

The same principle applies to IFS extraction: within-spaxel spectral blocks do
not imply zero covariance between overlapping spaxels. A full extracted
covariance is useful only if its consumer preserves the axes, calibration and
estimator meaning. Interfaces that accept only diagonal uncertainties are explicit capability
limitations ({ref}`limitations-records`; {ref}`limitations-optics`).

(records-posterior-chart)=
## A posterior needs coordinates and a normalization history

A posterior bundle must carry the parameter chart used by every array:

| Required information | Why it matters |
|---|---|
| Parameter names, order, shapes and units | A flattened array has no intrinsic scientific coordinate system |
| Physical versus unconstrained coordinates; transform and inverse | A stored log radius or bounded latent variable is not a radius |
| Logarithm base and reference quantity | Logarithms act on dimensionless ratios, such as $\log(R/R_0)$ |
| Angular topology and wrapping | A cloud around 0/2pi is one mode, not two distant clusters |
| Frame, epoch and element/body convention | The same orbital angles at different reference epochs predict different states |
| Model, prior, response and conditioning record revisions | An array alone does not identify which inference problem it represents |
| Evidence kind, normalization validity and uncertainty | A posterior approximation need not supply an absolute marginal likelihood |

The evidence is $Z=\int p(y\mid\theta)p(\theta)\,d\theta$ for a normalized
likelihood and proper prior ({ref}`Kass and Raftery 1995, eq. 2 <source-kass1995>`). A normalized mixture weight is not $Z$. An ELBO is a
lower bound on log evidence under its assumptions, not an interchangeable
estimate ({ref}`Blei et al. 2017, Sec. 2.2, eq. 14 <source-blei2017>`). A log posterior potential that omits constants can be sufficient for a
fixed-noise parameter fit while being insufficient for model evidence or fitting
the noise scale, because every likelihood constant must be retained when evidence
is compared ({ref}`Kass and Raftery 1995, p. 776 and Sec. 5.3 <source-kass1995>`;
see also {ref}`limitations-records`).

**Proposed contract for the evidence and parameter chart decision:** posterior
component weights, evidence value, estimator kind and error are separate
fields. An unavailable quantity is typed unavailable; the claim layer (the
record type that carries a scientific claim and its evidence) rejects an
evidence-based comparison requiring it. Compression preserves evidence with its
provenance or declares it unavailable. It never relabels component mass as
evidence.

Clustering and covariance floors operate in a declared dimensionless chart.
Adding $10^{-6}$ to a variance in meters squared and to the same variance in
kilometers squared implements different approximations. Test unit changes and
angle-wrap boundaries; label any display-only covariance cap as a visualization
product rather than posterior samples for scientific analysis (see {ref}`limitations-records`).

(records-identities)=
## Records distinguish acquisition, interpretation and use

Use a small versioned envelope around domain-owned payloads initially. It needs
to distinguish at least the following identities:

| Identity / metadata | Meaning |
|---|---|
| Subject and observed association | Which sky source or hypothesized companion the analysis attributes a measurement to |
| Acquisition and exposure / visit | The physical data collection; one acquisition is charged once |
| Product and immutable revision | A particular reduction/calibration of that acquisition |
| Response and calibration revision | The experiment that interprets the values, including its supported domain |
| Parent and conditioning records | Which measurements produced a reduction or a posterior |
| Analysis identity | The chosen model/prior/likelihood route through available evidence |
| Acquisition interval and availability time | When the data were collected and when they could first inform a decision |

Truth-only planet IDs belong to simulated worlds and scoring. A renderer's array
index cannot become a known observed association. Reordering planets must not
change record meaning. Reingesting one immutable product is idempotent; a revised
product is an explicit revision, with affected analyses invalidated or rebuilt.

Raw images, extracted spectra and derived orbit summaries can all be stored.
They are not automatically independent evidence. The analysis either selects a
single route or supplies a justified joint model. An imported posterior cannot be
multiplied by all its conditioning data again. The baseline update is a refit
from the original prior over the accepted evidence once.

Missing data, an unsuccessful acquisition, a masked value, an upper limit, and a
successful nondetection are distinct states. Unsupported modalities must raise a
named capability error. They cannot be silently omitted because a particular
multi-planet model lacks a branch. Capacity overflow must name the limit before
array padding fails (see {ref}`limitations-records`).

(records-campaign-causality)=
## A reproducible campaign preserves causality

Science exposure time, occupied facility time and charged budget are separate
quantities. Readout, slew and settling each have one owner. Admission reserves a
feasible interval and resources; completion settles the actual cost; release
makes data available; inference consumes released evidence; policy then acts.
No future reduction or calibration may influence an earlier decision.

Key stochastic streams by stable world, facility, target, exposure and purpose
identity under a stated coupling scheme. Inserting an unrelated visit into an
otherwise identical physical history should not change a target's noise solely
because a global loop counter changed. Different adaptive histories can change
the physical instrument/source state and need not produce identical data.

Set numerical precision at process startup. Store source/configuration/data
versions and calibration identity; reject unintended overwrite of an immutable
run. Checkpoint both acquired-but-unassimilated records and completed charges so
resume neither resimulates the observation nor double counts it. Keep truth out
of every policy/update/claim input; independent structural checks complement
behavioral tests. These requirements answer seven of the findings recorded
under {ref}`limitations-records`.

An SNR, a detection statistic, a false-alarm probability, a posterior probability,
and a sequential evidence process are different quantities. Name the statistic
and its calibration. In particular, repeatedly crossing a fixed-time threshold
does not inherit the error guarantee of one prespecified exposure
({ref}`Howard et al. 2021, Sec. 1 <source-howard2021>`).

(records-fixtures)=
## Named fixtures and coverage

**Reported-measurement distribution** normalizes the scalar threshold example
and checks detection frequency plus the selected distribution. **Null-outcome
information** checks zero and $\ln2$ information limits. **Response support and
cache identity** checks response support and that a warm cache and a fresh cache
give the same answer for the same facility switched A to B to A.
**Shared-calibration covariance** compares the explicit-offset and marginalized
correlated Gaussian. **Evidence normalization** uses a proper conjugate model
with analytic evidence. **Parameter chart and units** checks unit/column/epoch
changes and serialization. **Replay and exactly-once charge** inserts an
unrelated visit and resumes after acquisition, requiring exactly-once
charge/assimilation with matched physical history.

These are fixture definitions, not new passing tests. The conventions stage
selects their contracts; later stages implement them with independently derived
expectations.

Every finding about the current implementations that bears on this chapter, with
its proposed owner, fixture, first gate, status and evidence, is recorded under
{ref}`limitations-records` on the limitations page.
