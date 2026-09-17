# Measurements, probability, and records

*Draft contract: the choices marked pending or proposed are open.*

A correct orbit and a correct detector model do not automatically define a correct
likelihood. The likelihood describes the data that actually reached the analyst.
It therefore depends on the reporting rule, the calibration and reduction, and
which other records share the same underlying noise or evidence.

This chapter defines the intended statistical boundaries. The proposed profiles
remain pending decisions D04 and D07 in [the handbook](index.md). The current
implementations disagree in the ways the coverage table at the end of this
chapter lists. Nothing here asserts that a generic joint imaging likelihood or
portable posterior already exists.

## Define the experiment before the uncertainty

Write an experiment as a resolved response, exposure interval, noise process and
reporting rule. Let $\theta$ denote model parameters, $Y$ the complete potential
measurement, and $D$ a reporting event. The actually acquired record $R$ can be
much smaller than $Y$.

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
density by $P(D=1\mid\theta)$ again: the joint event is already represented.

A sample consisting only of selected detections under a conditioned sampling
design uses $p(F\mid D=1,\theta)$, including its selection normalization. Forced
photometry instead reports $F$ even when $D=0$ and can carry information below the
detection threshold. These are different experiments. Choose one explicitly;
neither a detection flag nor a field named `sigma` chooses it for you.

For joint astrometry and flux, replace the scalar $F$ by the actual joint random
vector and apply the real selection rule. Conditioning on detection can change
its mean, covariance and shape. An unconditional inverse Fisher matrix is not by
itself the distribution of selected measurements. The first campaign needs a
calibrated reporting model, not merely an available flux-error calculation.

### The simplest information test

Expected information is about $R$, the record returned by this experiment:

$$
I(\theta;R)=\mathbb E_{\theta,R}
\left[\log\frac{p(R\mid\theta)}{p(R)}\right].
$$

Use natural logarithms and report nats. For two equally weighted hypotheses, if
both always produce the same null record, posterior weights remain $(1/2,1/2)$
and information is zero. If the record identifies the hypothesis perfectly,
information is $\ln 2$ nats. The current scheduler implementation in planit-py
fails the first limit by adding an alias term for measurements that are never
reported (INF-03).

A response also has a domain. Below an inner working angle, beyond a tabulated
outer boundary, or outside a calibrated wavelength range, record an explicit
unsupported or physically unobservable result according to the response's
contract. These states are not an infinite sensitivity. Prediction, sampling,
likelihood and candidate scoring must use the same boundary policy. When an
exposure or calibration changes the response, its cache identity changes too,
or the changed arrays must remain dynamic function inputs (INF-01/02).

## Covariance describes an ordered, dimensional vector

For a vector such as $(\xi,\eta,f)$, retain axis names, order, units, covariance,
and whether nuisance variables were held fixed or marginalized. Its mixed entries
have mixed units: $C_{\xi f}$ has units of angle times flux. A bare square array is
not a portable uncertainty model.

Under a linear transformation $y=Ax$, covariance transforms exactly as
$C_y=AC_xA^T$. For a nonlinear transformation, $JC_xJ^T$ is a local approximation;
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
independent offsets. Record the nuisance identity and treatment, not just the
resulting error bars.

The same principle applies to IFS extraction: within-spaxel spectral blocks do
not imply zero covariance between overlapping spaxels. A full extracted
covariance is useful only if its consumer preserves the axes, calibration and
estimator meaning. Current diagonal interfaces remain explicit capability
limitations (INF-05 and OPT-18).

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
likelihood and proper prior. A normalized mixture weight is not $Z$. An ELBO is a
lower bound on log evidence under its assumptions, not an interchangeable
estimate. A log posterior potential that omits constants can be sufficient for a
fixed-noise parameter fit while being insufficient for model evidence or fitting
the noise scale (INF-06/07).

**Proposed D07 contract:** posterior component weights, evidence value, estimator
kind and error are separate fields. An unavailable quantity is typed unavailable;
the claim layer (the record type that carries a scientific claim and its
evidence) rejects an evidence-based comparison requiring it. Compression
preserves evidence with its provenance or declares it unavailable. It never
relabels component mass as evidence.

Clustering and covariance floors operate in a declared dimensionless chart.
Adding $10^{-6}$ to a variance in meters squared and to the same variance in
kilometers squared implements different approximations. Test unit changes and
angle-wrap boundaries; label any display-only covariance cap as a visualization
product rather than posterior samples for scientific analysis (INF-08/09).

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
array padding fails (INF-15).

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
behavioral tests (INF-10..16).

An SNR, a detection statistic, a false-alarm probability, a posterior probability,
and a sequential evidence process are different quantities. Name the statistic
and its calibration. In particular, repeatedly crossing a fixed-time threshold
does not inherit the error guarantee of one prespecified exposure.

## Named fixtures and coverage

**I-REPORT** normalizes the scalar threshold example and checks detection
frequency plus the selected distribution. **I-NULL** checks zero and $\ln2$
information limits. **I-RESPONSE** checks response support and that a warm cache
and a fresh cache give the same answer for the same facility switched A to B to
A. **I-COV** compares the explicit-offset and marginalized correlated Gaussian.
**I-EVIDENCE** uses a proper conjugate model with analytic evidence. **I-CHART**
checks unit/column/epoch changes and serialization. **I-REPLAY** inserts an
unrelated visit and resumes after acquisition, requiring exactly-once
charge/assimilation with matched physical history.

These are fixture definitions, not new passing tests. S0 selects their
contracts; later stages implement them with independently derived expectations.

| Finding | Proposed owner and disposition | Fixture | First gate |
|---|---|---|---|
| INF-01: Contrast-curve support has opposite meanings in scheduling and inference | photomancy: one response-support policy | I-RESPONSE | S1; required by S3 |
| INF-02: Changing the response curve can reuse the old compiled experiment | photomancy: immutable response/cache identity or dynamic inputs | I-RESPONSE | S3 |
| INF-03: The information score can count measurements that are never reported | photomancy: information about the reported record | I-NULL | S3 |
| INF-04: Hard detection, probit nondetection, and Gaussian magnitude errors are different experiments | measurement adapter + photomancy: D04 joint reporting law | I-REPORT | S0 -> S2 |
| INF-05: Full covariance has producers but several consumers only accept diagonals | product owners + photomancy: covariance axes and shared nuisances | I-COV | S0 -> S2; S4 spectra |
| INF-06: Evidence, ELBO, and cluster mass share one public field | photomancy: evidence distinct from ELBO/component mass | I-EVIDENCE | S0; before model comparison |
| INF-07: Posterior-only Gaussian helpers feed an evidence-capable backend interface | photomancy: normalized likelihood/proper prior for evidence | I-EVIDENCE | S1; before model comparison |
| INF-08: A posterior's coordinate system is mostly implicit and may be process-local | photomancy: persist reconstructible parameter chart | I-CHART | S1 -> S2 |
| INF-09: Clustering and regularization depend on arbitrary parameter units | photomancy: scale-aware regularization and periodic metrics | I-CHART | S1; before new charts |
| INF-10: Acquisition, product, target, and planet identities are not interchangeable | spaceodyssey (campaign library)/import adapters: stable acquisition/product/association IDs | | S0 -> S2 |
| INF-11: Stable random identity is claimed, but the loop keys draws by global step | spaceodyssey: meaning-keyed RNG and explicit coupling | I-REPLAY | S2 |
| INF-12: Science time, occupied time, and budget time are conflated | spaceodyssey + planit-py: interval and resource ownership | I-REPLAY | S0 -> S2 |
| INF-13: SNR, significance, false-alarm probability, and sequential evidence need distinct types | coronalyze/spaceodyssey claim adapters: statistic and calibration semantics | | S2; sequential extension separately gated |
| INF-14: Precision and artifact identity do not yet pin the scientific interpretation | spaceodyssey: fixed precision, complete run identity, collision behavior | | S2 |
| INF-15: Supported channels and capacity are part of the data contract | photomancy/import adapters: explicit supported data and capacity | | S1 -> S2 |
| INF-16: The truth/model firewall is a convention without full structural enforcement | spaceodyssey: private truth and model-side signatures/checks | | S2 |
| INF-17: Point values, intervals, distributions, and nuisance parameters need different semantics | configuration owners: typed point/distribution/interval meaning | | S0; interval execution deferred |
