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

```{figure} figures/explainer-d08-information-boundary-light.png
:class: only-light
:name: fig-explainer-d08-information-boundary
:alt: Schematic in three labeled zones. Left, a dashed region named simulated world holds a star at the focus of an orbit, a planet on it, and a truth record box listing the true orbit and flux, the planet ID and the noise seed. Rays of starlight and planet light cross a scale break to a telescope aperture and a detector in the middle zone; a configured visit box, labeled a command, not data, points a thin arrow at the telescope. A dashed gray elbow arrow labeled noise draw runs from the truth record to the detector. A hollow arrow leads from the detector to one record card, record R sub k, listing t sub k and D sub k and, if reported, the offsets xi sub k and eta sub k and C sub k, and a second hollow arrow leads from the card across the zone line to the inference model box on the right, which feeds the observing policy box, a decision rule, whose next visit arrow returns to the configured visit. A red arrow from the truth record curves toward the inference model and is crossed out just after it leaves the simulated world, labeled no path: truth never reaches inference or policy. Below a dashed divider labeled below: simulation-only scoring, an evaluation box receives the posterior and, along the bottom, the truth record; a red crossed arrow from evaluation back to the inference model is labeled no path: scores.

The proposed information boundary of a simulated imaging experiment ({ref}`records-identities`, {ref}`records-campaign-causality`). The simulated world holds the scene and a truth record. Truth reaches the detector only through the simulated light and the noise draw, and reaches evaluation directly, for simulation-only scoring. The only arrow into the inference model is the acquired record $R_k$ of each visit $k$, which carries its epoch $t_k$ and reporting event $D_k$ and, when reported, the east and north offsets $(\xi_k,\eta_k)$ and their covariance $C_k$; what a record keeps under a reporting law is the subject of the next figure. The crossed arrow from the truth record aims at the inference model. Neither truth nor scores reach the inference model or the observing policy, which is a decision rule that chooses the next visit from the posterior, not a hypothesis. The figure is a schematic of a proposed convention, not an example implementation, and does not assert that the current tutorial or campaign software enforces every separation it shows.
```

```{figure} figures/explainer-d08-information-boundary-dark.png
:class: only-dark
:alt: Schematic in three labeled zones. Left, a dashed region named simulated world holds a star at the focus of an orbit, a planet on it, and a truth record box listing the true orbit and flux, the planet ID and the noise seed. Rays of starlight and planet light cross a scale break to a telescope aperture and a detector in the middle zone; a configured visit box, labeled a command, not data, points a thin arrow at the telescope. A dashed gray elbow arrow labeled noise draw runs from the truth record to the detector. A hollow arrow leads from the detector to one record card, record R sub k, listing t sub k and D sub k and, if reported, the offsets xi sub k and eta sub k and C sub k, and a second hollow arrow leads from the card across the zone line to the inference model box on the right, which feeds the observing policy box, a decision rule, whose next visit arrow returns to the configured visit. A red arrow from the truth record curves toward the inference model and is crossed out just after it leaves the simulated world, labeled no path: truth never reaches inference or policy. Below a dashed divider labeled below: simulation-only scoring, an evaluation box receives the posterior and, along the bottom, the truth record; a red crossed arrow from evaluation back to the inference model is labeled no path: scores.

The proposed information boundary of a simulated imaging experiment ({ref}`records-identities`, {ref}`records-campaign-causality`). The simulated world holds the scene and a truth record. Truth reaches the detector only through the simulated light and the noise draw, and reaches evaluation directly, for simulation-only scoring. The only arrow into the inference model is the acquired record $R_k$ of each visit $k$, which carries its epoch $t_k$ and reporting event $D_k$ and, when reported, the east and north offsets $(\xi_k,\eta_k)$ and their covariance $C_k$; what a record keeps under a reporting law is the subject of the next figure. The crossed arrow from the truth record aims at the inference model. Neither truth nor scores reach the inference model or the observing policy, which is a decision rule that chooses the next visit from the posterior, not a hypothesis. The figure is a schematic of a proposed convention, not an example implementation, and does not assert that the current tutorial or campaign software enforces every separation it shows.
```

(records-reporting-law)=
## Define the experiment before the uncertainty

Write an experiment as a resolved response, exposure interval, noise process and
reporting rule. Let $\theta$ denote model parameters, $Y$ the complete potential
measurement, and $D$ a reporting event. The actually acquired record $R$ can be
much smaller than $Y$.

```{figure} figures/explainer-d08-record-reporting-law-light.png
:class: only-light
:name: fig-explainer-d08-record-reporting-law
:alt: Three columns. Left, two 9 by 9 pixel images of raw electrons, visit 1 with a bright planet and visit 2 fainter, each with the summed 3 by 3 pixels outlined by a dotted square. Hollow arrows lead from each image to its own horizontal F axis; both axes share one scale from 0 to 100 electrons, labeled F, the sum of the 9 dotted pixels minus the known background. A dashed vertical line crosses both axes at L = 30 electrons, fixed, with D = 0 written to its left and D = 1: F > L to its right. A filled cyan marker with a plus or minus sigma bar sits at F sub 1 = 84.8 electrons on the visit 1 axis, right of the line, with the note sigma = 6.0 electrons, fixed. A hollow cyan marker with a dashed bar sits at F sub 2 = 17.7 electrons on the visit 2 axis, left of the line, noted computed; released only by forced photometry. Right, two column headings, report only on detection and forced photometry, a different experiment, sit above a wide card, visit 1, either law, reading D sub 1 = 1, yes, and F sub 1 = 84.8 electrons. Below it the record card R sub k from the information-boundary figure lists t sub k, D sub k and, if reported, F sub k and sigma squared, noted flux F in place of the offsets and sigma squared in place of C sub k; thin lines link it to the visit 1 card and to both visit 2 cards below. Under report only on detection the visit 2 card reads D sub 2 = 0, no, and F sub 2 not reported; beside it, after the word or, a dashed forced photometry card reads D sub 2 = 0, no, and F sub 2 = 17.7 electrons.

The record of a visit under a reporting law, in the chapter's scalar flux example ({ref}`records-reporting-law`). The record card $R_k$ is carried from the information-boundary figure and linked to each visit's record; here the flux $F_k$ stands in place of the offsets and $\sigma^2$ in place of $C_k$. Each 9 by 9 cutout is a simulated raw readout: a known background of 20 e$^-$ per pixel, a pixel-integrated Gaussian planet image and independent Gaussian noise of 2 e$^-$ per pixel. The reduction subtracts the background and sums the 9 dotted pixels, those whose centers lie within 1.5 pixels of the fixed target position, so $F$ has the standard deviation $\sigma$ = 6.0 e$^-$ before any selection; the error bars are $\pm\sigma$ on the axis scale. Both visits share one $F$ axis with the fixed, known threshold $L$ = 30 e$^-$, and $D = 1$ precisely when $F > L$. Visit 1 gives $F$ = 84.8 e$^-$ and the record $R = (D = 1, F)$ under either law. For visit 2 the reduction computes $F$ = 17.7 e$^-$, below $L$; only forced photometry releases it, so its marker is hollow and its bar dashed. Reporting only on detection keeps $R = (D = 0)$, a successful nondetection rather than a missing datum; forced photometry, a different experiment, reports the flux below the threshold ({ref}`Efron and Hastie 2016, Sec. 9.6 <source-efron2016>`). The diagram that follows states the reporting law in general. The figure is a schematic with simulated pixels that illustrates the chapter's example, not an example implementation.
```

```{figure} figures/explainer-d08-record-reporting-law-dark.png
:class: only-dark
:alt: Three columns. Left, two 9 by 9 pixel images of raw electrons, visit 1 with a bright planet and visit 2 fainter, each with the summed 3 by 3 pixels outlined by a dotted square. Hollow arrows lead from each image to its own horizontal F axis; both axes share one scale from 0 to 100 electrons, labeled F, the sum of the 9 dotted pixels minus the known background. A dashed vertical line crosses both axes at L = 30 electrons, fixed, with D = 0 written to its left and D = 1: F > L to its right. A filled cyan marker with a plus or minus sigma bar sits at F sub 1 = 84.8 electrons on the visit 1 axis, right of the line, with the note sigma = 6.0 electrons, fixed. A hollow cyan marker with a dashed bar sits at F sub 2 = 17.7 electrons on the visit 2 axis, left of the line, noted computed; released only by forced photometry. Right, two column headings, report only on detection and forced photometry, a different experiment, sit above a wide card, visit 1, either law, reading D sub 1 = 1, yes, and F sub 1 = 84.8 electrons. Below it the record card R sub k from the information-boundary figure lists t sub k, D sub k and, if reported, F sub k and sigma squared, noted flux F in place of the offsets and sigma squared in place of C sub k; thin lines link it to the visit 1 card and to both visit 2 cards below. Under report only on detection the visit 2 card reads D sub 2 = 0, no, and F sub 2 not reported; beside it, after the word or, a dashed forced photometry card reads D sub 2 = 0, no, and F sub 2 = 17.7 electrons.

The record of a visit under a reporting law, in the chapter's scalar flux example ({ref}`records-reporting-law`). The record card $R_k$ is carried from the information-boundary figure and linked to each visit's record; here the flux $F_k$ stands in place of the offsets and $\sigma^2$ in place of $C_k$. Each 9 by 9 cutout is a simulated raw readout: a known background of 20 e$^-$ per pixel, a pixel-integrated Gaussian planet image and independent Gaussian noise of 2 e$^-$ per pixel. The reduction subtracts the background and sums the 9 dotted pixels, those whose centers lie within 1.5 pixels of the fixed target position, so $F$ has the standard deviation $\sigma$ = 6.0 e$^-$ before any selection; the error bars are $\pm\sigma$ on the axis scale. Both visits share one $F$ axis with the fixed, known threshold $L$ = 30 e$^-$, and $D = 1$ precisely when $F > L$. Visit 1 gives $F$ = 84.8 e$^-$ and the record $R = (D = 1, F)$ under either law. For visit 2 the reduction computes $F$ = 17.7 e$^-$, below $L$; only forced photometry releases it, so its marker is hollow and its bar dashed. Reporting only on detection keeps $R = (D = 0)$, a successful nondetection rather than a missing datum; forced photometry, a different experiment, reports the flux below the threshold ({ref}`Efron and Hastie 2016, Sec. 9.6 <source-efron2016>`). The diagram that follows states the reporting law in general. The figure is a schematic with simulated pixels that illustrates the chapter's example, not an example implementation.
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

```{figure} figures/explainer-d08-campaign-timeline-light.png
:class: only-light
:name: fig-explainer-d08-campaign-timeline
:alt: A schematic time axis running left to right, not to scale, with four horizontal lanes labeled, from top to bottom, facility, telescope time; observing policy, a decision rule box noted as from the information-boundary figure; inference model; and records. A pink diamond in the policy lane labeled decides visit k sends a thin command arrow up to an outlined interval in the facility lane labeled visit k, noted admission reserves the interval, which holds a shorter gray bar labeled exposure and a thick tick after the exposure and before the end of the outline, noted completion settles the actual cost. From the tick a hollow arrow runs down to a record card R sub k in the records lane, noted release: R sub k available; another climbs to a pink square in the inference lane, noted inference consumes R sub k, and a third, labeled posterior, climbs to a second diamond, decides visit k plus 1, which sends a command arrow up to the interval visit k plus 1. A faint dotted vertical line marks the time of that decision. Along the records lane a hollow arrow noted recalibrated later leads past that line to a dashed card R sub k prime. From it a hollow arrow climbs forward to a later pink square noted later updates may use it, and a red crossed arrow pointing back to the earlier decision is labeled no path back in time.

The campaign loop on a time axis ({ref}`records-campaign-causality`; the acquisition interval and availability time of {ref}`records-identities`). The observing-policy box is carried from the information-boundary figure. The policy decides visit $k$; admission reserves a feasible facility interval, the outline, inside which the science exposure is the shorter bar; completion settles the actual cost, the tick, which need not fill the reserved interval; release makes the record $R_k$ available; inference consumes the released record; and the next decision, for visit $k+1$, may use only records already released. Every allowed arrow points forward in time. A later recalibration of visit $k$ is a new product revision, the dashed card $R_k'$: later updates may use it, but it never reaches a decision already made, which the crossed arrow marks against the dotted line at the decision for visit $k+1$. Thin arrows with open heads are commands and hollow arrows are data. Times and durations are schematic and not to scale. The figure is a schematic of a proposed convention, not an example implementation, and does not assert that current campaign software enforces this ordering.
```

```{figure} figures/explainer-d08-campaign-timeline-dark.png
:class: only-dark
:alt: A schematic time axis running left to right, not to scale, with four horizontal lanes labeled, from top to bottom, facility, telescope time; observing policy, a decision rule box noted as from the information-boundary figure; inference model; and records. A pink diamond in the policy lane labeled decides visit k sends a thin command arrow up to an outlined interval in the facility lane labeled visit k, noted admission reserves the interval, which holds a shorter gray bar labeled exposure and a thick tick after the exposure and before the end of the outline, noted completion settles the actual cost. From the tick a hollow arrow runs down to a record card R sub k in the records lane, noted release: R sub k available; another climbs to a pink square in the inference lane, noted inference consumes R sub k, and a third, labeled posterior, climbs to a second diamond, decides visit k plus 1, which sends a command arrow up to the interval visit k plus 1. A faint dotted vertical line marks the time of that decision. Along the records lane a hollow arrow noted recalibrated later leads past that line to a dashed card R sub k prime. From it a hollow arrow climbs forward to a later pink square noted later updates may use it, and a red crossed arrow pointing back to the earlier decision is labeled no path back in time.

The campaign loop on a time axis ({ref}`records-campaign-causality`; the acquisition interval and availability time of {ref}`records-identities`). The observing-policy box is carried from the information-boundary figure. The policy decides visit $k$; admission reserves a feasible facility interval, the outline, inside which the science exposure is the shorter bar; completion settles the actual cost, the tick, which need not fill the reserved interval; release makes the record $R_k$ available; inference consumes the released record; and the next decision, for visit $k+1$, may use only records already released. Every allowed arrow points forward in time. A later recalibration of visit $k$ is a new product revision, the dashed card $R_k'$: later updates may use it, but it never reaches a decision already made, which the crossed arrow marks against the dotted line at the decision for visit $k+1$. Thin arrows with open heads are commands and hollow arrows are data. Times and durations are schematic and not to scale. The figure is a schematic of a proposed convention, not an example implementation, and does not assert that current campaign software enforces this ordering.
```

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
