# NIFTy in a Day — Lecture Notes

Oct 7, 2026 · @Albert einstein

## How to use this course

One intense day takes you from zero to building real NIFTy models: six theory lectures (this doc) alternate with seven runnable Jupyter notebooks (Hands-on 0–6). Every concept — prior, likelihood, posterior, field, FFT, power spectrum, correlated field, optimizer, KL, MGVI/geoVI — is first explained here, then implemented from scratch in NumPy/JAX, then used through NIFTy.

**Rule for the day:** read the lecture section, run the matching notebook top to bottom, then do its ✏️ exercises before peeking at the solutions at the bottom of each notebook.

### Schedule

| Time | Block | Material |
| --- | --- | --- |
| 08:30 – 08:45 | Setup | Install (below), open notebook 00 |
| 08:45 – 09:30 | Lecture 1 + Lecture 2 | Bayes, fields, FFT, power spectra |
| 09:30 – 10:15 | Hands-on 0 | `00_concepts_from_scratch.ipynb` — every concept in plain NumPy |
| 10:15 – 11:30 | Hands-on 1 | `01_foundations_jax_and_models.ipynb` — JAX, standardized models, first `optimize_kl` |
| 11:30 – 11:45 | Lecture 3 | Optimizers and the Wiener filter |
| 11:45 – 12:45 | Hands-on 2 | `02_gaussian_processes_and_wiener_filter.ipynb` |
| 12:45 – 13:30 | Lunch | — |
| 13:30 – 14:30 | Lecture 4 + Lecture 5 | Correlated field model; variational inference |
| 14:30 – 15:45 | Hands-on 3 | `03_correlated_field_model.ipynb` |
| 16:00 – 16:45 | Hands-on 4 | `04_variational_inference_deep_dive.ipynb` — MAP vs MGVI vs geoVI vs NUTS |
| 16:45 – 17:45 | Lecture 6 + Hands-on 5 | `05_real_world_poisson_and_tomography.ipynb` |
| 17:45 – 18:30 | Hands-on 6 | `06_nifty_cl_classic.ipynb` — the classical `nifty.cl` API |
| Evening | Capstone | One project from the self-test section |

### Setup

The notebooks were written and executed against **NIFTy 9.2.0** and **JAX 0.11.2** (Python 3.13, CPU). NIFTy ships two independent flavours in one package: `nifty.re` (JAX-based, fast, recommended for new work) and `nifty.cl` (classical, NumPy-based, object-oriented).

```bash
python -m venv nifty-env && source nifty-env/bin/activate
pip install 'nifty[re]' ducc0 blackjax matplotlib jupyter
jupyter lab
```

Enable double precision at the top of every `nifty.re` script — CG solvers and log-determinants are fragile in float32:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Runtimes on a 2-core cloud CPU: notebooks 00–02 under 1 minute each, 03 about 2.5 minutes, 04 about 3.5 minutes, 05 about 1.5 minutes, 06 about 1.5 minutes. A laptop is usually faster.

If a run freezes at 0% CPU in the middle of `optimize_kl` (seen twice in notebook 05 on the 2-core test machine), restart Jupyter with single-threaded XLA: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" jupyter lab`.

## Lecture 1 — Bayesian inference: prior, likelihood, posterior

NIFTy answers one question: given noisy, incomplete data d, what do we know about an unknown signal s — and how sure are we? Bayes' theorem is the whole engine; everything else is making it computable for millions of unknowns.

### The four objects

```latex
P(s \mid d) \;=\; \frac{P(d \mid s)\,P(s)}{P(d)}
```

| Object | Name | Plain meaning | In NIFTy |
| --- | --- | --- | --- |
| P(s) | **Prior** | What you believe about s before seeing data: smoothness, positivity, typical size | A model s(ξ) with ξ \~ N(0, 1): `jft.NormalPrior`, `jft.LogNormalPrior`, `jft.CorrelatedFieldMaker` |
| P(d \| s) | **Likelihood** | How probable the observed data are if the truth were s; encodes the instrument (response) and the noise | `jft.Gaussian`, `jft.Poissonian`, `jft.VariableCovarianceGaussian`, `jft.StudentT` |
| P(s \| d) | **Posterior** | Updated belief after the data: the answer | Samples returned by `jft.optimize_kl` |
| P(d) | **Evidence** | Normalisation; how well a whole model explains the data | ELBO via `jft.estimate_evidence_lower_bound` |

**Worked example (notebook 00, §1).** A positive brightness s has a log-normal prior centred at 2. Three measurements around 3.1 with noise σ = 0.8 give a likelihood peaked near 3.3. The posterior sits between, closer to the data because three measurements carry more information than the broad prior. Multiply the curves pointwise, normalise — that is Bayes.

### The information Hamiltonian

NIFTy works with negative log-probabilities, borrowed from statistical physics:

```latex
\mathcal{H}(d, s) = -\ln P(d, s) = \underbrace{-\ln P(d \mid s)}_{\text{likelihood energy}} \; \underbrace{- \ln P(s)}_{\text{prior energy}}, \qquad P(s \mid d) \propto e^{-\mathcal{H}(d,s)}
```

Minimising ℋ = maximising the posterior. Products of probabilities become sums of energies, which is what optimizers and autodiff like. For Gaussian noise the likelihood energy is the familiar ½ χ²:

```latex
\mathcal{H}(d \mid s) = \tfrac12 \big(d - R(s)\big)^{\dagger} N^{-1} \big(d - R(s)\big) + \text{const}
```

In code, `jft.Gaussian(d, noise_cov_inv).amend(model)` *is* this function of the latent parameters; NIFTy adds the prior energy ½ ξ†ξ itself.

### Summarising a posterior

- **MAP** (maximum a posteriori): the peak. Cheap (one optimisation) but no error bar, and in high dimensions the peak is often unrepresentative — it over-fits noise.
- **Posterior mean**: the average over the posterior; minimises expected squared error. Needs samples.
- **Posterior standard deviation**: the uncertainty, pixel by pixel.
- **Samples**: the universal currency. With samples you can compute the mean, std, and the uncertainty of *any* derived quantity (a ratio, a peak position, a power spectrum). NIFTy always returns samples.

### Why grids fail and NIFTy exists

On a grid you evaluate the posterior at n points per dimension: n^D evaluations. A 128 × 128 image has D = 16 384 unknowns. Exact computation is impossible; NIFTy instead computes an **approximate posterior** (variational inference, Lecture 5) whose cost grows roughly linearly in D, tested up to billions of parameters.

### Check yourself

1. With 3 versus 30 measurements, which moves the posterior more: the prior or the data? (Run exercise 1 in notebook 00.)
2. Why is the evidence irrelevant for finding the posterior of one model but essential for comparing two models?

## Lecture 2 — Fields, Fourier space, power spectra and standardization

A Gaussian-process prior on a field costs O(N log N) instead of O(N³) because homogeneous covariances are diagonal in Fourier space. That single fact, plus standardization, is the mathematical heart of NIFTy.

### Fields

A **field** is a function over a continuous domain: a sky image, a 3-D dust density, a temperature over time. In **Information Field Theory (IFT)** — the theory NIFTy implements — the signal is a field, so it formally has infinitely many degrees of freedom. On a computer it becomes an array on pixels, and two rules keep results physical:

- **Pixel volume.** Integrals are volume-weighted sums: ∫ s(x) dx ≈ Σᵢ sᵢ ΔV. Forgetting ΔV makes answers depend on the pixel count.
- **Resolution independence.** Refining the grid must not change the answer. NIFTy's correlated-field normalisations are built so that it doesn't. In `nifty.cl` the domain object (`ift.RGSpace(shape, distances)`) carries ΔV; in `nifty.re` you pass `distances=`.

### The Fourier transform and the FFT

Any field on a periodic grid is a sum of waves with wave numbers k:

```latex
\hat s_k = \sum_{x} s_x\, e^{-2\pi i k x / N}, \qquad s_x = \frac{1}{N}\sum_{k} \hat s_k\, e^{2\pi i k x / N}
```

- Small |k| = large-scale structure; large |k| = fine detail. In 2-D and 3-D every Fourier pixel has a **mode length** |k| = √(kx² + ky²).
- The direct sum costs O(N²); the **Fast Fourier Transform** costs O(N log N). In notebook 00, a hand DFT on 4096 points takes about 1 s; `np.fft.fft` takes 0.2 ms.
- **Convolution theorem:** convolution in position space = pointwise multiplication in Fourier space. Blurring, smoothing and stationary covariances all become cheap.
- NIFTy often uses the **Hartley transform** H = Re F + Im F: real in, real out, its own inverse up to 1/N (`nifty.re.correlated_field.hartley`).

### Covariance and the power spectrum

A Gaussian field prior is s \~ N(0, S), with covariance S\_xy = ⟨s\_x s\_y⟩. For a 128² image S has 2.7 × 10⁸ entries — too many to store. Two physical assumptions rescue us:

- **Homogeneity** (statistics independent of position): S\_xy = C(x − y). Then S is a convolution, so it is **diagonal in Fourier space**. Its diagonal is the **power spectrum** P(k), the Fourier transform of the correlation function C (Wiener–Khinchin theorem).
- **Isotropy** (no preferred direction): P depends only on |k|. An entire N × N covariance is now one 1-D curve.

```latex
S = F^{\dagger}\, \mathrm{diag}\!\big(P(|k|)\big)\, F
```

Steep spectra (P ∝ k⁻⁴) give smooth fields; flat spectra give noise-like fields. NIFTy groups Fourier pixels into shells of equal |k|; the **power distributor** maps a 1-D spectrum onto the full Fourier grid (`grid.harmonic_grid.power_distributor` in `nifty.re`, `ift.PowerDistributor` in `nifty.cl`).

### Gaussian random fields and standardization

If ξ \~ N(0, 𝟙) then A ξ \~ N(0, A A†). Choosing A = F⁻¹ diag(√P) gives a correlated field in three steps: draw white noise, multiply by the **amplitude spectrum** A(k) = √P(k), transform back. Cost: one FFT.

This is **standardization**, the single most important idea in NIFTy: every prior is written as a deterministic function of standard-normal latent variables,

```latex
s = s(\xi), \qquad \xi \sim \mathcal{N}(0, \mathbb{1}), \qquad \text{scalar case: } s(\xi) = \mathrm{CDF}^{-1}_{P(s)}\big(\mathrm{CDF}_{\mathcal{N}}(\xi)\big)
```

Why NIFTy insists on it:

1. Inference algorithms only ever see a standard-normal prior, so one algorithm serves every model.
2. In latent space the posterior is much closer to Gaussian, which makes variational inference accurate.
3. Linear systems become well conditioned. In notebook 00 the same Wiener filter needed 550 CG iterations in signal space and 130 in standardized space.

`jft.LogNormalPrior(4, 3, name="a")` is literally this map for a log-normal with mean 4 and std 3; notebook 01 checks it on 100 000 samples. The `name` becomes the key of the latent parameter, so names must be unique.

### Check yourself

1. Why does a stationary covariance become diagonal in Fourier space? Name the theorem.
2. You double the resolution of an image. Which quantities must not change, and which code argument guarantees that?
3. Write a standardized prior for a quantity uniformly distributed between 2 and 5.

## Lecture 3 — Optimizers and the Wiener filter

Every NIFTy algorithm is built from three solvers: conjugate gradient for linear systems, Newton-CG for minimisation, and both driven only by operator applications — never by stored matrices. The Wiener filter is the one case where these give the exact posterior.

### Optimizers you need to know

| Method | Update | Strength | Weakness |
| --- | --- | --- | --- |
| Gradient descent | x ← x − η ∇E | Trivial to implement | Crawls along curved valleys; notebook 00 reaches E = 0.0017 after 300 steps |
| Newton | x ← x − H⁻¹ ∇E | Quadratic convergence; exact minimum in 10 steps in notebook 00 | Needs the Hessian H and its inverse |
| Conjugate gradient (CG) | Solves A x = b using only products A v | Never forms A; converges in ≤ N steps, far fewer when A is well conditioned | Needs A symmetric positive definite |
| Newton-CG (NIFTy default) | Newton step with H⁻¹∇E computed by CG | Scales to 10⁹ unknowns | Inner CG tolerance must be tuned |

NIFTy replaces the Hessian by the **Fisher information metric** of the likelihood plus the identity from the standardized prior:

```latex
M(\xi) = J(\xi)^{\dagger}\, N^{-1}\, J(\xi) + \mathbb{1}, \qquad J = \frac{\partial R(s(\xi))}{\partial \xi}
```

M is always positive definite (CG works) and needs only Jacobian-vector products, which JAX provides (`lh.metric(x, v)` in `nifty.re`). In `optimize_kl` you control these solvers through `cg_kwargs` (`absdelta`, `maxiter`) and `minimize_kwargs` (`xtol`, `maxiter`); in `nifty.cl` through controllers such as `ift.AbsDeltaEnergyController` and `ift.GradientNormController`.

### The measurement equation

```latex
d = R(s) + n
```

- **Response R**: everything the instrument does — masking, blurring by a point-spread function, line-of-sight integration, Fourier sampling in interferometry, exposure.
- **Noise n**: here Gaussian with covariance N; for photon counts the likelihood is Poisson instead.
- A linear R needs its **adjoint** R† (back-projection from data space to signal space). `nifty.re` gets it from JAX automatically; in `nifty.cl` you implement `apply(x, mode)` for both directions and test it with `ift.extra.check_linear_operator`.

### The Wiener filter

For a Gaussian prior s \~ N(0, S), linear response R and Gaussian noise N, the posterior is exactly Gaussian:

```latex
P(s \mid d) = \mathcal{N}(s;\, m, D), \qquad D = \big(S^{-1} + R^{\dagger} N^{-1} R\big)^{-1}, \qquad m = D\, j, \qquad j = R^{\dagger} N^{-1} d
```

- **j**, the *information source*: data back-projected into signal space and weighted by noise.
- **D**, the *information propagator*: the posterior covariance. Its inverse is prior precision plus data precision.
- The posterior mean m is found by solving D⁻¹ m = j with CG — D itself is never built.
- **Posterior samples** without forming D: draw η with covariance D⁻¹ (η = S^(−1/2) ξ₁ + R† N^(−1/2) ξ₂), solve D⁻¹ y = η, then m + y is a sample. MGVI draws its samples the same way.

What to expect (notebook 02): in observed regions the mean follows the data and the std is small; in gaps the mean relaxes to the prior mean and the std grows towards the prior std, largest in the middle of a gap. About 68% of pixels lie within 1σ of the truth — the calibration check (notebook 02 measured 67.4%).

**Its limit.** The Wiener filter needs the power spectrum S. Assume it 10× too small and the reconstruction is shrunk flat; 10× too large and it chases noise (notebook 02, §6). That is the motivation for Lecture 4.

### Check yourself

1. Why does NIFTy use the Fisher metric rather than the true Hessian inside Newton-CG?
2. What happens to m as N → 0? As N → ∞? Answer from the formula, then verify in notebook 00 exercise 4.
3. Why does `jft.wiener_filter_posterior` solve its system in standardized coordinates?

## Lecture 4 — The correlated field model

The correlated field model infers the field **and** its power spectrum jointly, by putting an interpretable prior on the spectrum itself. It is the workhorse prior of almost every NIFTy application.

### Construction

```latex
\phi(x) = \mu + \sigma_0\, \xi_0 + \frac{1}{V}\, \mathcal{H}\big[\, A_\theta(|k|)\, \xi_k \big](x), \qquad \ln A_\theta(k) \propto \alpha \ln k + \eta\; \mathrm{IWP}(\ln k)
```

The amplitude spectrum A is normalised so that the field's standard deviation equals the `fluctuations` parameter. All ingredients — ξ\_k, ξ₀ and the spectral parameters θ — are standard-normal latents, so one inference handles everything.

| Parameter `(mean, std)` | Prior type | Controls | What you see when you raise it |
| --- | --- | --- | --- |
| `offset_mean` | fixed number | Mean value of the field | Whole field shifts |
| `offset_std` | LogNormal | Uncertainty of the zero mode (overall level) | Overall level free to move |
| `fluctuations` | LogNormal | Std of the field around its mean | Larger amplitude, same shapes |
| `loglogavgslope` | Normal | Average slope of log A versus log k | Less negative = rougher; more negative = smoother |
| `flexibility` | LogNormal | How far log A may bend away from a straight line (an integrated Wiener process, IWP) | Bumps and kinks in the spectrum |
| `asperity` | LogNormal | How sharp those bends may be | Spiky spectral features, e.g. periodicities |

The API (`nifty.re`):

```python
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cfm.add_fluctuations((256,), distances=1/256, fluctuations=(1.0, 0.3),
                     loglogavgslope=(-2.0, 0.5), flexibility=(1.0, 0.5),
                     asperity=(0.2, 0.1), prefix="ax1", non_parametric_kind="amplitude")
cf = cfm.finalize()          # a jft.Model: latents -> field
cfm.power_spectrum(xi)       # the spectrum for a given latent position
```

Its latent parameters are `cfxi` (the field excitations), `cfzeromode`, `cfax1fluctuations`, `cfax1loglogavgslope`, `cfax1flexibility`, `cfax1asperity` and `cfax1spectrum` (the IWP driving noise).

### Expert rules

1. **Prior predictive check first.** Draw 10 or more prior samples of field and spectrum; your real data must look like a plausible draw. Notebook 03 §2 sweeps each parameter with fixed random numbers so you can see its effect in isolation.
2. **Zero-pad non-periodic data.** FFT-based priors are periodic. Make the domain about 1.5–2× the observed region, otherwise structure wraps from one edge to the other.
3. **`amplitude` versus `power`.** With `non_parametric_kind="power"` the slope refers to P(k); with the default `"amplitude"` it refers to A(k) = √P, so the same smoothness needs half the slope.
4. **Product domains** (space × time, space × frequency): call `add_fluctuations` once per sub-domain; the spectrum is the product. Gotcha verified in notebook 03: each `fluctuations` is then *relative to* `offset_std`, so with `offset_std=(1e-3, …)` and two axes the field's std explodes to about 1000. Use an O(1) `offset_std` for product fields.
5. **Hyper-parameters are degenerate.** A shallower slope plus a downward bend can mimic a steeper line, and high-k power is noise-dominated. Judge the inferred field and spectrum, not each knob alone.
6. **Non-linear transforms.** Positive quantities: s = exp(φ) (log-normal). Bounded quantities: s = sigmoid(φ). Either makes the problem non-linear, so the Wiener filter no longer applies — next lecture.
7. **Parametric alternative.** `add_fluctuations_matern` gives a Matérn-type spectrum with fewer degrees of freedom (scale, cutoff, slope).

### Check yourself

1. You expect a smooth field with one dominant periodicity. Which two parameters do you raise, and which do you keep small?
2. Why is it legitimate to infer the power spectrum from a single field realisation? (Hint: how many Fourier modes share each |k| shell in 2-D?)

## Lecture 5 — Variational inference: MGVI and geoVI

For non-linear models the posterior is not Gaussian and cannot be computed exactly. NIFTy fits a Gaussian in the standardized latent space by minimising a KL divergence, with covariance given by the Fisher metric — MGVI — and optionally bends the samples to follow curvature — geoVI.

### The KL divergence

Pick a tractable family Q and make it as close as possible to the true posterior P:

```latex
\mathrm{KL}(Q \,\|\, P) = \int Q(\xi) \ln \frac{Q(\xi)}{P(\xi \mid d)}\, d\xi = \big\langle \mathcal{H}(d, \xi) \big\rangle_{Q} - \mathrm{entropy}(Q) + \ln P(d)
```

Minimising KL is equivalent to maximising the **ELBO** = ⟨ln P(d, ξ)⟩\_Q + entropy(Q) ≤ ln P(d). That inequality is why the ELBO doubles as a model-comparison score.

### MGVI — Metric Gaussian Variational Inference

1. Approximate Q = N(ξ̄, M(ξ̄)⁻¹): the covariance is *not* a free parameter; it is the inverse Fisher metric at the mean. This is what makes MGVI scale to 10⁹ parameters.
2. Draw samples δξᵢ \~ N(0, M⁻¹) with one CG solve each (as in the Wiener filter), and use **antithetic pairs** ξ̄ ± δξᵢ: they cancel odd moments and halve the noise of the KL estimate. `n_samples=20` gives 40 samples.
3. Keeping δξᵢ fixed, move ξ̄ to minimise the sample average (1/n) Σᵢ ℋ(d, ξ̄ + δξᵢ) with Newton-CG.
4. Redraw samples at the new ξ̄ and repeat. The KL is estimated stochastically, so the mean jitters between iterations; more samples, less jitter.

### geoVI — Geometric Variational Inference

geoVI starts from the MGVI samples and pushes each through a non-linear coordinate transformation derived from the metric, solved with a few Newton steps per sample. Samples then follow curved posterior shapes. It costs more (one extra non-linear solve per sample) and is more accurate. In notebook 04's banana posterior MGVI draws a straight ellipse while geoVI's samples follow the arms; started on the symmetry axis, geoVI covered both arms while MAP sat on a saddle point.

&#91;embedded content: one optimize\_kl iteration · 3 solver stages\]

Each box is one solver with its own kwargs: drawing uses `draw_linear_kwargs`, bending uses `nonlinearly_update_kwargs`, moving the mean uses `kl_kwargs`.

### `optimize_kl` controls

| Argument | What it does | Typical expert setting |
| --- | --- | --- |
| `n_samples` | Antithetic pairs per iteration; `0` means MAP | A function of the iteration: few early, many late |
| `sample_mode` | `linear_resample` = MGVI · `nonlinear_resample` = geoVI (default) · `nonlinear_update` = refine existing samples | MGVI early, geoVI in the last iterations |
| `draw_linear_kwargs` | CG that draws the linear samples | `cg_kwargs=dict(absdelta=δ·size/10, maxiter=100)` |
| `nonlinearly_update_kwargs` | Newton-CG that bends samples (geoVI only) | `minimize_kwargs=dict(xtol=δ, maxiter=5)` |
| `kl_kwargs` | Newton-CG that moves the mean | `minimize_kwargs=dict(xtol=δ, maxiter=20–35)` |
| `point_estimates` | Parameter names optimised but not sampled | Nuisance parameters that are well constrained |
| `constants` | Parameter names held fixed | Calibration terms you trust |
| `callback` | Called with `(samples, state)` every iteration | Live plots, custom logging |
| `odir`, `resume` | Save state every iteration; continue after a crash | Always on for runs longer than minutes |

### Validation and diagnostics

- **Reduced χ²** (`jft.minisanity`, also printed in every `optimize_kl` log): normalised data residuals should have χ²/n ≈ 1. Much larger means the model cannot fit the data (noise underestimated, wrong response, too stiff a prior); much smaller means over-fitting. Latent parameters should also look standard-normal.
- **Exact samplers for validation**: `jft.blackjax_nuts` (No-U-Turn Hamiltonian Monte Carlo) on a small version of the problem. NUTS saw both modes of the banana; any single-mean VI cannot represent bimodality, so watch for sign or label symmetries in models.
- **Model comparison**: `jft.estimate_evidence_lower_bound(lh, samples, n_eigenvalues)`. In notebook 04 the ELBO prefers a straight line (−17.1) over degree-2 (−21.4) and degree-5 polynomials (−35.0) for data generated by a line — Occam's razor built in. It omits parameter-independent constants such as −½ ln|2πN|, so compare models with the same likelihood and data, or add them back.

### Check yourself

1. Why does MGVI not optimise the covariance, and what does it gain from that?
2. Your geoVI run shows reduced χ² = 4 for the data. List three things you would check.
3. When would you deliberately choose MAP (`n_samples=0`)?

## Lecture 6 — Building real models, nifty.re vs nifty.cl, best practices

Every NIFTy application is the same four-part recipe: latent ξ → prior model s(ξ) → response R(s) → likelihood → `optimize_kl`. Expertise is in choosing components and checking them, not in new algorithms.

### The recipe in `nifty.re`

```python
class Forward(jft.Model):
    def __init__(self):
        self.diffuse = cfm.finalize()                                     # correlated field
        self.points  = jft.InvGammaPrior(1.5, 0.05, name="points", shape=dims)
        super().__init__(init=self.diffuse.init | self.points.init)      # merge latent specs
    def __call__(self, xi):
        sky = jnp.exp(self.diffuse(xi)) + self.points(xi)                # prior model s(xi)
        return exposure * blur(sky)                                      # response R(s)

lh = jft.Poissonian(counts).amend(Forward())                          # likelihood
samples, state = jft.optimize_kl(lh, jft.Vector(lh.init(key)) * 0.1, key=key2,
                                 n_total_iterations=6, n_samples=lambda i: 2 if i < 3 else 4)
mean, std = jft.mean_and_std(tuple(model.diffuse(s) for s in samples))
```

- **`jft.Model`** = a function plus bookkeeping. `__call__` does the physics; `init=` (merged with `|`) declares the latents. Use `domain=` only for models with one unnamed input array.
- **Component separation** works through prior *shapes*: a smooth multi-scale log-normal field versus sparse heavy-tailed inverse-gamma points — the X-ray/γ-ray imaging pattern. Be warned: it is one of the hardest NIFTy problems. The inverse-gamma map has almost no gradient at ξ = 0, so the diffuse field explains bright spots first; a naive 12-iteration MGVI run in this course recovered none of the point sources. Working pipelines use staged fitting, a stiffer diffuse prior early on and more iterations (notebook 05, §A.5 and exercise E5.2).
- **Line-of-sight tomography**: `jft.SamplingCartesianGridLOS(starts, ends, shape=, distances=)`; the same model scales to 3-D dust maps (notebook 05, Part B; `demos/re/1_tomography.py`).
- **Likelihood menu**: `Gaussian` (known noise), `VariableCovarianceGaussian` (learn the noise; the model returns `(prediction, inverse_std)` with the data's shape), `StudentT` (outliers), `Poissonian` (counts), `Categorical`.
- **Posterior quantities**: always average the derived quantity over samples (`tuple(f(s) for s in samples)`), never apply f to the mean latent — for non-linear f these differ.

### `nifty.cl` in one paragraph

The classical flavour makes geometry explicit: **domains** (`ift.RGSpace`, `ift.HPSpace`, `ift.PowerSpace`, `ift.UnstructuredDomain`, `ift.MultiDomain`), **fields** bound to domains (`ift.Field`, `ift.MultiField`), and **operators** with domain, target, adjoint and inverse, composed with `@`. The Wiener filter is written literally as `D_inv = R.adjoint @ N.inverse @ R + S.inverse`, inverted by `ift.InversionEnabler`. Non-linear models compose operators (`ift.SimpleCorrelatedField(...).exp()`), and `ift.optimize_kl` runs MGVI/geoVI with explicit minimizers and iteration controllers. Notebook 06 covers it end to end, including writing and testing your own `LinearOperator`.

| Topic | `nifty.re` | `nifty.cl` |
| --- | --- | --- |
| Backend | JAX (JIT, CPU/GPU) — generally faster | NumPy + ducc0, optional CuPy |
| Model | Subclass `jft.Model` | Compose `ift.Operator`s |
| Derivatives and adjoints | Automatic via JAX | Built-in `Linearization`; linear adjoints written by hand |
| Extra features | HMC/NUTS, multi-grid (ICR) | Rich operator library, MPI over samples |
| License | BSD-2-Clause or GPL-2.0+ | GPL-3.0+ |
| Use it for | New projects | Existing pipelines, its operator library |

### Debugging and best-practice checklist

1. `jax_enable_x64` on. Most "NaN energy" or non-converging CG problems in float32 disappear.
2. Prior predictive check before any inference.
3. Start from small latents (`init * 0.1`): close to the prior mean, away from extreme non-linear regions.
4. Excluded data are *selected out* by the response — never set to zero. A zero exposure gives λ = 0 and 0·log 0 = NaN in the Poisson energy (hit and fixed in notebook 05).
5. Read the `optimize_kl` log every iteration: energy should fall, reduced χ² should approach 1.
6. Schedule: MGVI and few samples early; geoVI and more samples late; `odir` + `resume=True` for long runs.
7. Validate VI against NUTS on a down-scaled problem; watch for symmetries (sign flips, label switching).
8. Custom linear operators in `nifty.cl`: always `ift.extra.check_linear_operator`; non-linear ones: `ift.extra.check_operator`.
9. Use `jax.vmap` over `samples.samples` for fast posterior statistics of derived quantities.

### Check yourself

1. Sketch the generative model for radio interferometry: what is the prior, what is R, and which likelihood?
2. Why does an inverse-gamma prior favour "a few bright pixels" over "many medium pixels"?

## Glossary, capstone projects and further reading

### Glossary

| Term | One-line meaning | Where implemented |
| --- | --- | --- |
| Adjoint R† | Transpose of a linear operator; maps data back to signal space | NB00 §9, NB02 §5, NB06 §2 |
| Amplitude spectrum A(k) | √P(k); multiplies white noise in Fourier space | NB00 §7, NB02 §2 |
| Antithetic samples | Pairs ξ̄ ± δξ that cancel odd moments | NB01 §5 |
| Conjugate gradient | Solves A x = b using only A v products | NB00 §3.2 |
| Correlated field | Field whose power spectrum is inferred jointly | NB00 §8, NB03 |
| ELBO | Lower bound on the log evidence; model comparison | NB04 §7 |
| Evidence P(d) | Probability of the data under a model | NB00 §1 |
| FFT | O(N log N) Fourier transform | NB00 §5 |
| Field | Function over a continuous domain, discretised on pixels | NB00 §4, NB06 §1 |
| Fisher metric M | JᵀN⁻¹J + 𝟙; NIFTy's stand-in for the Hessian and posterior precision | NB00 §11, NB01 §4 |
| geoVI | VI with samples bent along posterior curvature | NB04 |
| Hartley transform | Real-to-real cousin of the FFT | NB00 §5 |
| Information Hamiltonian ℋ | −ln P(d, s); the energy NIFTy minimises | NB01 §4 |
| Information propagator D | Posterior covariance of the Wiener filter | NB00 §10 |
| Information source j | R†N⁻¹d | NB00 §10 |
| KL divergence | Distance from the approximation Q to the posterior P | NB00 §11 |
| Latent space ξ | Standard-normal parameters every model is written in | NB01 §2 |
| Likelihood P(d \| s) | Probability of the data given the signal | NB00 §1, NB01 §4 |
| MAP | Posterior maximum; `n_samples=0` | NB01 §5, NB04 |
| MGVI | Gaussian VI with covariance = inverse Fisher metric | NB00 §11, NB04 |
| Newton-CG | Newton steps solved with CG | NB00 §3 |
| Posterior P(s \| d) | Belief after the data | NB00 §1–2 |
| Power spectrum P(k) | Variance per Fourier mode; diagonal of a homogeneous covariance | NB00 §6 |
| Prior P(s) | Belief before the data | NB00 §1, NB01 §2 |
| Reduced χ² | Residual check; ≈ 1 when the model fits | NB04 §6 |
| Response R | Map from signal to expected data | NB00 §9, NB02 §3 |
| Standardization | Writing s = s(ξ) with ξ \~ N(0, 𝟙) | NB00 §7, NB01 §2 |
| Wiener filter | Exact posterior for linear Gaussian problems | NB00 §10, NB02, NB06 §3 |

### Capstone projects (pick one for the evening)

1. **Deblurring and inpainting** a real photo (convert to grayscale, mask 30% of pixels, add noise) with a log-normal correlated field. Report the reduced χ² and an uncertainty map.
2. **Time series with unknown noise**: daily temperature data of your city, a 1-D correlated field plus `VariableCovarianceGaussian`; infer the power spectrum and look for the yearly periodicity in it.
3. **Count-data imaging** (extend notebook 05): add an unknown background level and compare models with and without point sources using the ELBO.
4. **Port** your favourite notebook from `nifty.re` to `nifty.cl` (or vice versa) and compare runtime.

### Further reading

- [NIFTy repository](https://github.com/NIFTy-PPL/NIFTy) — source, `demos/re` and `demos/cl` folders, installation.
- [NIFTy documentation](https://ift.pages.mpcdf.de/nifty/) — informal introduction, the six official `nifty.re` notebooks (models, inference, Gaussian processes, Wiener filter, correlated field model, log-normal Poisson), API reference.
- [NIFTy.re paper (JOSS 2024)](https://doi.org/10.21105/joss.06593) — design and benchmarks of the JAX flavour.
- [Standardizing models, arXiv:1812.04403](https://arxiv.org/abs/1812.04403) — the reparameterisation behind latent ξ.
- [Geometric Variational Inference (Entropy 2021)](https://doi.org/10.3390/e23070853) — geoVI and its ELBO.

All notebook outputs and numbers quoted in this doc come from running the seven course notebooks against NIFTy 9.2.0 on the date above.
