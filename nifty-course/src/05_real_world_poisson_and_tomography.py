# %% [markdown]
# # Hands-on 5 — Real-world models: photon-count imaging and tomography
#
# **Course:** NIFTy in a day · **Session:** 16:45 – 17:45
#
# Real NIFTy applications (X-ray / γ-ray imaging, radio interferometry, 3-D dust maps, medical imaging) combine the
# pieces you know into **generative models with several components and non-Gaussian likelihoods**.
#
# **You will build**
# 1. A **photon-count imaging** model: diffuse emission (log-normal correlated field) → blurred by a PSF → multiplied by an exposure map with dead pixels → **Poisson** likelihood. Plus an honest look at why adding point sources makes it much harder.
# 2. A **tomography** model: a positive 2-D density observed only through **line-of-sight integrals** (`jft.SamplingCartesianGridLOS`).

# %%
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

import nifty.re as jft

jax.config.update("jax_enable_x64", True)
plt.rcParams["figure.dpi"] = 90
key = random.PRNGKey(2026)

def im(ax, a, t, **kw):
    h = ax.imshow(np.asarray(a).T, origin="lower", cmap=kw.pop("cmap", "inferno"), **kw)
    ax.set_title(t); ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(h, ax=ax, fraction=0.046)

# %% [markdown]
# ## Part A — Photon-count imaging
#
# ### A.1 The generative model
# $$ \lambda = E \cdot \big(\mathrm{PSF} * e^{\phi}\big), \qquad d \sim \mathrm{Poisson}(\lambda) $$
#
# * **diffuse** $e^{\phi}$, $\phi$ a correlated field → smooth, positive, multi-scale.
# * **PSF** blur via FFT; **exposure** $E$: how long each pixel was observed (here: a gradient, plus a dead strip).
#
# **Poisson likelihood:** $-\ln P(d|\lambda) = \sum_x (\lambda_x - d_x\ln\lambda_x) + \text{const}$. Its Fisher metric is $\mathrm{diag}(1/\lambda)$: brighter pixels carry more information.

# %%
dims = (64, 64)
cfm = jft.CorrelatedFieldMaker("diffuse")
cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(0.5, 0.1))
cfm.add_fluctuations(dims, distances=1 / dims[0], fluctuations=(0.8, 0.3), loglogavgslope=(-4.0, 0.5),
                     flexibility=(0.5, 0.3), asperity=None, prefix="", non_parametric_kind="power")
log_diffuse = cfm.finalize()

kx = jnp.fft.fftfreq(dims[0]) * dims[0]
psf_ft = jnp.exp(-0.5 * (jnp.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2) / 12.0) ** 2)   # Gaussian PSF in Fourier space
exposure = jnp.broadcast_to(jnp.linspace(0.5, 2.0, dims[0])[:, None], dims)
exposure = exposure.at[24:36, :].set(0.0)                                                # dead detector strip (12 px)


class SkyModel(jft.Model):
    def __init__(self):
        self.log_diffuse = log_diffuse
        super().__init__(init=self.log_diffuse.init)

    def diffuse(self, xi):
        return jnp.exp(self.log_diffuse(xi))

    def sky(self, xi):
        return self.diffuse(xi)

    def __call__(self, xi):                         # expected counts λ
        blurred = jnp.fft.ifft2(psf_ft * jnp.fft.fft2(self.sky(xi))).real
        return exposure * jnp.clip(blurred, 1e-10)  # λ must stay > 0 for the Poisson likelihood


model = SkyModel()
print("latent parameters:", list(model.domain.keys()))

# %% [markdown]
# ### A.2 Prior predictive check — always look at prior samples first

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for i, ax in enumerate(axs):
    xi = model.init(random.PRNGKey(i))
    im(ax, model.sky(xi), f"prior sky sample {i}", norm=plt.matplotlib.colors.LogNorm())
plt.tight_layout(); plt.show()

# %% [markdown]
# ### A.3 Synthetic data and the Poisson likelihood

# %%
key, k_t, k_d = random.split(key, 3)
xi_true = jft.Vector(model.init(k_t))
lam_true = model(xi_true)
counts = random.poisson(k_d, lam_true * 20.0).astype(jnp.int64)   # ×20: a moderately bright observation


class Scaled(jft.Model):                                           # include the ×20 brightness in the model
    def __init__(self, m):
        self.m = m
        super().__init__(init=m.init)

    def __call__(self, xi):
        return 20.0 * self.m(xi)[observed]   # only pixels with exposure > 0 are data


# Pixels with zero exposure carry no data. Leaving them in would give λ = 0 and 0·log(0) = NaN in the Poisson
# energy — so the response must *select* the exposed pixels (exactly like the mask in Hands-on 2).
observed = exposure > 0
lh = jft.Poissonian(counts[observed]).amend(Scaled(model))

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
im(axs[0], model.sky(xi_true), "true sky")
im(axs[1], exposure, "exposure", cmap="gray")
im(axs[2], counts, "photon counts (data)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### A.4 Inference

# %%
delta = 1e-4
key, k_i, k_o = random.split(key, 3)
samples, state = jft.optimize_kl(
    lh, jft.Vector(lh.init(k_i)) * 0.1, n_total_iterations=6, key=k_o,
    n_samples=lambda i: 2 if i < 3 else 4,
    # MGVI throughout keeps this cell at a few minutes on a laptop. For the final science run you'd switch the
    # last iterations to "nonlinear_resample" (geoVI) — more accurate, ~5x slower here.
    sample_mode="linear_resample",
    draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10, maxiter=100)),
    nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name=None, xtol=delta, cg_kwargs=dict(name=None), maxiter=5)),
    kl_kwargs=dict(minimize_kwargs=dict(name=None, xtol=delta, cg_kwargs=dict(name=None), maxiter=20)),
)

dm, ds = jft.mean_and_std(tuple(model.sky(s) for s in samples))
fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
im(axs[0], model.sky(xi_true), "true sky"); im(axs[1], dm, "posterior mean sky")
im(axs[2], ds / dm, "relative uncertainty")
z = (dm - model.sky(xi_true)) / ds
im(axs[3], z, "(mean − truth) / std", cmap="RdBu_r", vmin=-3, vmax=3)
plt.tight_layout(); plt.show()
print(f"pixels within 1σ: {float((jnp.abs(z) < 1).mean()):.2f} (ideal 0.68)   within 2σ: {float((jnp.abs(z) < 2).mean()):.2f} (ideal 0.95)")

# %% [markdown]
# Things to notice:
# * The **dead strip** is filled in by the correlated-field prior; the relative uncertainty is higher there than in its exposed neighbours.
# * Relative uncertainty grows towards the left, where the exposure is lower: fewer photons, less information.
# * The last panel is the calibration check. With only 8 posterior samples the std itself is a noisy estimate,
#   so expect fewer pixels inside 1σ than the ideal 68%. More samples (and geoVI in the final iterations) improve the
#   calibration — try `n_samples=lambda i: 2 if i < 3 else 16` and compare the printed fractions.
#
# ### A.5 Expert note — why point sources are hard
# Real X-ray/γ-ray skies also contain **point sources**. The standard NIFTy approach adds a second component
# $p_x \sim \text{InvGamma}(\alpha, q)$ per pixel: mostly tiny, occasionally huge, and the different prior *shapes*
# (smooth vs. sparse) let the inference separate them. In practice this is one of the hardest NIFTy problems:
# the inverse-gamma map has a gradient of ~0.003 at $\xi=0$ but needs $\xi\approx 3$–4 to make a bright source,
# so early on the smooth diffuse field "explains" every bright spot first and the point latents barely move.
# When we tried it here with a naive setup (12 MGVI iterations), the point component recovered none of the bright
# sources. Working pipelines use more iterations, a stiffer diffuse prior early on, staged inference
# (diffuse first, then points) and careful initialisation. Exercise E5.2 lets you explore this.

# %% [markdown]
# ## Part B — Tomography with line-of-sight integrals
#
# We observe a positive 2-D density $\rho = e^{\phi}$ only through integrals along straight lines:
# $d_i = \int_{\text{LOS}_i} \rho\, dl + n_i$. `jft.SamplingCartesianGridLOS(start, end, shape=, distances=)`
# builds that integration operator (it samples points along each line and interpolates the grid).

# %%
dims_t = (64, 64)
dist_t = tuple(1.0 / d for d in dims_t)
cfm_t = jft.CorrelatedFieldMaker("rho")
cfm_t.set_amplitude_total_offset(offset_mean=0.0, offset_std=(0.3, 0.1))
cfm_t.add_fluctuations(dims_t, distances=dist_t, fluctuations=(1.0, 0.3), loglogavgslope=(-4.0, 0.5),
                       flexibility=(0.5, 0.2), asperity=None, prefix="", non_parametric_kind="power")
log_rho = cfm_t.finalize()

n_los = 300
key, k_s, k_e = random.split(key, 3)
starts = random.uniform(k_s, (n_los, 2), minval=0.02, maxval=0.98)
ends = random.uniform(k_e, (n_los, 2), minval=0.02, maxval=0.98)
los = jft.SamplingCartesianGridLOS(starts, ends, distances=dist_t, shape=dims_t, n_sampling_points=200)


class Tomo(jft.Model):
    def __init__(self):
        self.log_rho = log_rho
        super().__init__(init=log_rho.init)

    def rho(self, xi):
        return jnp.exp(self.log_rho(xi))

    def __call__(self, xi):
        return los(self.rho(xi))


tomo = Tomo()
key, k_t, k_n = random.split(key, 3)
xi_t = jft.Vector(tomo.init(k_t))
clean = tomo(xi_t)
sig = 0.05 * clean                                    # 5% relative noise
d_t = clean + sig * random.normal(k_n, clean.shape)
lh_t = jft.Gaussian(d_t, noise_cov_inv=lambda r: r / sig**2).amend(tomo)

key, k_i, k_o = random.split(key, 3)
smp_t, _ = jft.optimize_kl(
    lh_t, jft.Vector(lh_t.init(k_i)) * 0.1, n_total_iterations=6, n_samples=lambda i: 2 if i < 2 else 4, key=k_o,
    sample_mode="linear_resample",
    draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(absdelta=delta * jft.size(lh_t.domain) / 10, maxiter=100)),
    nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name=None, xtol=delta, cg_kwargs=dict(name=None), maxiter=5)),
    kl_kwargs=dict(minimize_kwargs=dict(name=None, xtol=delta, cg_kwargs=dict(name=None), maxiter=20)),
)
rm, rs = jft.mean_and_std(tuple(tomo.rho(s) for s in smp_t))

fig, axs = plt.subplots(1, 4, figsize=(17, 3.9))
im(axs[0], tomo.rho(xi_t), "true density")
for s_, e_ in zip(np.asarray(starts[:60]), np.asarray(ends[:60])):
    axs[1].plot([s_[0] * 64, e_[0] * 64], [s_[1] * 64, e_[1] * 64], "w-", lw=0.4, alpha=0.6)
im(axs[1], tomo.rho(xi_t), "60 of the 300 lines of sight")
im(axs[2], rm, "posterior mean")
im(axs[3], rs, "posterior std")
plt.tight_layout(); plt.show()

# %% [markdown]
# Each datum is a *single number* per line, yet the correlated-field prior makes the 4096-pixel image recoverable.
# The uncertainty is lowest where many lines cross. The same model scales to 3-D (`demos/re/1_tomography.py`) —
# that is how the 3-D Galactic dust maps were made with NIFTy.

# %% [markdown]
# ## ✏️ Exercises
# **E5.1** Reduce the brightness factor from 20 to 2 (fewer photons). How do the relative-uncertainty map and the small-scale structure change?
#
# **E5.2 (advanced)** Add point sources to the *truth* (a handful of pixels set to 50). First fit the diffuse-only model and inspect `jft.minisanity(samples, lh.normalized_residual)` — where is the misfit? Then add `jft.InvGammaPrior(a=0.8, scale=1e-3, name="points", shape=dims)` to the model and try to make the separation work (stiffer diffuse prior, more iterations, two-stage fitting).
#
# **E5.3** Tomography: use only 50 lines of sight, all starting from the same point (an "observer", like in dust mapping, `start=(0.5, 0.5)`). How does the uncertainty map look now?
#
# **E5.4** Add an unknown **calibration factor** to the tomography data (`jft.LogNormalPrior(1.0, 0.2, name="calib")` multiplying the LOS integrals). Is it degenerate with anything in the correlated field? (Hint: `offset_mean`.)

# %%
# Your work here

# %% [markdown]
# ---
# ## Solutions

# %%
# E5.3 — single observer
starts_o = jnp.broadcast_to(jnp.array([0.5, 0.5]), (50, 2))
ends_o = random.uniform(random.PRNGKey(99), (50, 2), minval=0.02, maxval=0.98)
los_o = jft.SamplingCartesianGridLOS(starts_o, ends_o, distances=dist_t, shape=dims_t, n_sampling_points=200)


class TomoObs(Tomo):
    def __call__(self, xi):
        return los_o(self.rho(xi))


tomo_o = TomoObs()
clean_o = tomo_o(xi_t)
sig_o = 0.05 * clean_o
d_o = clean_o + sig_o * random.normal(random.PRNGKey(98), clean_o.shape)
lh_o = jft.Gaussian(d_o, noise_cov_inv=lambda r: r / sig_o**2).amend(tomo_o)
smp_o, _ = jft.optimize_kl(
    lh_o, jft.Vector(lh_o.init(random.PRNGKey(97))) * 0.1, n_total_iterations=5, n_samples=4, key=random.PRNGKey(96),
    sample_mode="linear_resample",
    draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(maxiter=100)),
    nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name=None, xtol=delta, cg_kwargs=dict(name=None), maxiter=5)),
    kl_kwargs=dict(minimize_kwargs=dict(name=None, xtol=delta, cg_kwargs=dict(name=None), maxiter=20)),
)
mo, so = jft.mean_and_std(tuple(tomo_o.rho(s) for s in smp_o))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
im(axs[0], mo, "posterior mean (single observer)"); im(axs[1], so, "posterior std: radial structure")
plt.tight_layout(); plt.show()
