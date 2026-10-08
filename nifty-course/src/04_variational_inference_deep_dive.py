# %% [markdown]
# # Hands-on 4 — Inside the inference engine: MAP vs MGVI vs geoVI vs NUTS
#
# **Course:** NIFTy in a day · **Session:** 16:00 – 16:45
#
# `optimize_kl` has many switches. Experts know *what each one does to the posterior*. On a 2-parameter problem we
# can compute the exact posterior on a grid, so we can **see** the approximation each method makes.
#
# **You will learn**
# 1. How MAP, MGVI (`linear_resample`), and geoVI (`nonlinear_resample`) differ — visually.
# 2. How to get (asymptotically) exact samples with NUTS (`jft.blackjax_nuts`) for validation.
# 3. The remaining `optimize_kl` controls: `n_samples` schedules, `point_estimates`, `constants`, `callback`, `odir`/`resume`.
# 4. Diagnostics: `jft.minisanity` (reduced χ²), and model comparison with the **ELBO**.

# %%
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

import nifty.re as jft

jax.config.update("jax_enable_x64", True)
plt.rcParams["figure.dpi"] = 90

# %% [markdown]
# ## 1. A deliberately non-Gaussian toy problem
#
# Two latent parameters, both standard-normal a priori. We measure $\xi_1 + \xi_2^2$ precisely and $\xi_1$ poorly.
# Because only $\xi_2^2$ enters, the sign of $\xi_2$ is unidentifiable → a curved, two-armed ("banana") posterior.

# %%
class Banana(jft.Model):
    def __init__(self):
        self.a = jft.NormalPrior(0.0, 1.0, name="xi1")
        self.b = jft.NormalPrior(0.0, 1.0, name="xi2")
        super().__init__(init=self.a.init | self.b.init)

    def __call__(self, xi):
        x1, x2 = self.a(xi), self.b(xi)
        return jnp.stack([x1 + x2**2, x1])


model = Banana()
data = jnp.array([1.5, 0.5])
noise_std = jnp.array([0.3, 1.0])
lh = jft.Gaussian(data, noise_std_inv=lambda r: r / noise_std).amend(model)

# Exact posterior on a grid: P(xi|d) ∝ exp(-H(d|xi) - |xi|^2/2)
g1, g2 = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-2.5, 2.5, 300))
H = jax.vmap(jax.vmap(lambda a, b: lh({"xi1": a, "xi2": b}) + 0.5 * (a**2 + b**2)))(g1, g2)
post = np.exp(-(H - H.min()))

def show(ax, pts=None, mean=None, title=""):
    ax.contourf(g1, g2, post, 15, cmap="Blues")
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], ".", c="C3", ms=4, alpha=0.7)
    if mean is not None:
        ax.plot(*mean, "X", c="k", ms=11)
    ax.set_title(title); ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$")

def as_pts(samples):
    return np.stack([np.array([s["xi1"], s["xi2"]]) for s in samples])

# %% [markdown]
# ## 2. Run every method
#
# `optimize_kl` logs every iteration to stderr: the energy `E`, the number of Newton steps, and the
# **reduced χ²** of data residuals and latent parameters (more on those in §6). Read those logs — they are your main
# convergence monitor on real problems.

# %%
solver = dict(
    draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(absdelta=1e-8, maxiter=50)),
    nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name=None, xtol=1e-6, cg_kwargs=dict(name=None), maxiter=20)),
    kl_kwargs=dict(minimize_kwargs=dict(name=None, xtol=1e-6, cg_kwargs=dict(name=None), maxiter=30)),
)
init = jft.Vector({"xi1": jnp.array(0.0), "xi2": jnp.array(0.3)})   # start off the symmetry axis

smp_map, _ = jft.optimize_kl(lh, init, n_total_iterations=5, n_samples=0, key=random.PRNGKey(0), **solver)
smp_mgvi, _ = jft.optimize_kl(lh, init, n_total_iterations=6, n_samples=60, key=random.PRNGKey(1),
                              sample_mode="linear_resample", **solver)
smp_geo, _ = jft.optimize_kl(lh, init, n_total_iterations=6, n_samples=60, key=random.PRNGKey(2),
                             sample_mode="nonlinear_resample", **solver)
smp_nuts, _ = jft.blackjax_nuts(lh, {"xi1": jnp.array(0.0), "xi2": jnp.array(0.3)}, random.PRNGKey(3),
                                n_warmup_steps=300, n_samples=1500)

pts_nuts = np.stack([np.asarray(smp_nuts.samples["xi1"]), np.asarray(smp_nuts.samples["xi2"])], 1)
fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
show(axs[0], mean=(float(smp_map.pos["xi1"]), float(smp_map.pos["xi2"])), title="MAP (a point)")
show(axs[1], as_pts(smp_mgvi), (float(smp_mgvi.pos["xi1"]), float(smp_mgvi.pos["xi2"])), "MGVI: Gaussian, cov = M⁻¹")
show(axs[2], as_pts(smp_geo), (float(smp_geo.pos["xi1"]), float(smp_geo.pos["xi2"])), "geoVI: samples follow curvature")
show(axs[3], pts_nuts[::3], None, "NUTS (reference)")
plt.tight_layout(); plt.show()

for name, pts in [("MGVI", as_pts(smp_mgvi)), ("geoVI", as_pts(smp_geo)), ("NUTS", pts_nuts)]:
    print(f"{name:6s} mean = {pts.mean(0).round(3)}   std = {pts.std(0).round(3)}")

# %% [markdown]
# **What you should see**
# * **MAP** sits on one arm and reports no uncertainty.
# * **MGVI** draws an ellipse aligned with the local Fisher metric at the mean — it can't bend.
# * **geoVI** starts from the MGVI samples and pushes each through a non-linear map derived from the metric's
#   geometry, so samples follow the banana's arm. In the latent space of a standardized model this usually helps
#   a lot for large problems, where posteriors are *mildly* non-Gaussian.
# * **NUTS** (Hamiltonian Monte Carlo) sees both arms. VI methods around a single mean cannot represent bimodality —
#   be aware of symmetries (sign flips, label switching) in your models!
#
# **Cost.** MGVI: 1 CG solve per sample. geoVI: plus one Newton-CG per sample. NUTS: many gradient evaluations per
# sample and scales badly to $10^6$ parameters — use it to **validate** VI on small versions of your problem.

# %% [markdown]
# ## 3. The `sample_mode` options
# | value | meaning |
# |---|---|
# | `"linear_sample"` / `"linear_resample"` | MGVI. *resample* draws fresh samples each iteration; *sample* reuses the old random numbers |
# | `"nonlinear_sample"` / `"nonlinear_resample"` | geoVI (default `nonlinear_resample`) |
# | `"nonlinear_update"` | keep the samples and only update them non-linearly — cheap refinement in late iterations |
#
# `sample_mode` and `n_samples` may be **functions of the iteration index** — the standard expert recipe is
# "cheap and few early, accurate and many late":

# %%
smp_sched, state = jft.optimize_kl(
    lh, init, n_total_iterations=8, key=random.PRNGKey(4),
    n_samples=lambda i: 5 if i < 3 else 50,
    sample_mode=lambda i: "linear_resample" if i < 3 else "nonlinear_resample",
    callback=lambda samples, st: print(f"  iter {st.nit}: mean xi = ({float(samples.pos['xi1']):.3f}, {float(samples.pos['xi2']):.3f}), n_samples={len(samples)}"),
    **solver,
)

# %% [markdown]
# ## 4. `point_estimates` and `constants`
# * `point_estimates=("xi1",)` — optimise these parameters but **don't sample** them (MAP for a subset). Handy for nuisance parameters that are well constrained or very expensive to sample.
# * `constants=("xi1",)` — keep these parameters fixed at the initial value (neither optimised nor sampled).

# %%
smp_pe, _ = jft.optimize_kl(lh, init, n_total_iterations=6, n_samples=50, key=random.PRNGKey(5),
                            point_estimates=("xi1",), **solver)
pe = as_pts(smp_pe)
print("xi1 spread with point_estimates=('xi1',):", pe[:, 0].std().round(6), " | xi2 spread:", pe[:, 1].std().round(3))

# %% [markdown]
# ## 5. Saving and resuming long runs: `odir` and `resume`
# Real reconstructions run for hours. Pass `odir="results"` to write the state (and a `minisanity.txt` log) after
# every iteration; pass `resume=True` to continue from the last saved iteration after a crash.
#
# ```python
# samples, state = jft.optimize_kl(lh, init, ..., odir="results_run1", resume=True)
# ```

# %% [markdown]
# ## 6. Diagnostics: `jft.minisanity` — are the residuals the size of the noise?
#
# For each sample, compute **normalised residuals** $(d - R(s))/\sigma$. If the model is right, they're
# ~$\mathcal N(0,1)$, so the **reduced χ²** ≈ 1. Same check for the latents: $\xi$ should look standard-normal.
# * χ²/n ≫ 1 → model can't fit the data (noise underestimated, response wrong, prior too stiff).
# * χ²/n ≪ 1 → over-fitting (noise overestimated).

# %%
stats, text = jft.minisanity(smp_geo, lh.normalized_residual)
print("Data residuals:\n", text)
stats, text = jft.minisanity(smp_geo)
print("Latent parameters:\n", text)

# %% [markdown]
# ## 7. Model comparison with the ELBO
#
# The **evidence lower bound** $\text{ELBO} = \langle \ln P(d,\xi) \rangle_Q + \text{entropy}(Q) \le \ln P(d)$ lets you compare models.
# Example: data generated by a straight line; compare a *line* model with a *parabola* model.

# %%
key = random.PRNGKey(10)
x = jnp.linspace(-3, 3, 30)
y = 2.0 * x + 1.0 + 0.8 * random.normal(key, x.shape)

class Poly(jft.Model):
    def __init__(self, x, degree):
        self.x = x
        self.coeffs = jft.NormalPrior(0.0, 3.0, shape=(degree + 1,), name="c")
        super().__init__(init=self.coeffs.init)

    def __call__(self, xi):
        c = self.coeffs(xi)
        return sum(c[i] * self.x**i for i in range(c.shape[0]))

for deg in (1, 2, 5):
    m = Poly(x, deg)
    lh_p = jft.Gaussian(y, noise_std_inv=lambda r: r / 0.8).amend(m)
    smp, _ = jft.optimize_kl(lh_p, jft.Vector(lh_p.init(random.PRNGKey(deg))), n_total_iterations=4, n_samples=20,
                             key=random.PRNGKey(20 + deg), **solver)
    elbo, st = jft.estimate_evidence_lower_bound(lh_p, smp, n_eigenvalues=deg + 1, verbose=False)
    print(f"degree {deg}: ELBO = {float(st['elbo_mean']):8.3f}   (1σ range {float(st['elbo_lw']):.2f} … {float(st['elbo_up']):.2f})")

# %% [markdown]
# The line wins: higher-degree models fit the data no better but "waste" prior volume (Occam's razor, built in).
# Caveat: the ELBO omits constants that don't depend on the parameters (e.g. $-\frac12\ln|2\pi N|$) — fine for
# comparing models with the **same likelihood and data**; add them back otherwise.

# %% [markdown]
# ## ✏️ Exercises
# **E4.1** Make the banana less curved (`noise_std = [0.3, 0.1]`: $\xi_1$ is now measured precisely). When do MGVI and geoVI agree?
#
# **E4.2** Start `init` *exactly* on the symmetry axis (`xi2 = 0`). What happens to MAP and to geoVI? Why? (Hint: gradient w.r.t. $\xi_2$.)
#
# **E4.3** Run MGVI with `n_samples=2` vs `n_samples=200`. Plot the mean trajectory over iterations using a `callback`. How noisy is the KL optimisation?
#
# **E4.4** Add a degree-1 model with **unknown noise** (`VariableCovarianceGaussian`, see Hands-on 1) and compare its ELBO to the fixed-noise line. *(Careful: different likelihoods → constants matter!)*

# %%
# Your work here

# %% [markdown]
# ---
# ## Solutions

# %%
# E4.2 — on the symmetry axis the gradient wrt xi2 vanishes (d/dxi2 of xi2^2 at 0 is 0): MAP gets stuck on a saddle.
init0 = jft.Vector({"xi1": jnp.array(0.0), "xi2": jnp.array(0.0)})
s_map0, _ = jft.optimize_kl(lh, init0, n_total_iterations=5, n_samples=0, key=random.PRNGKey(0), **solver)
s_geo0, _ = jft.optimize_kl(lh, init0, n_total_iterations=6, n_samples=60, key=random.PRNGKey(2), **solver)
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
show(axs[0], mean=(float(s_map0.pos["xi1"]), float(s_map0.pos["xi2"])), title="MAP from xi2=0: stuck on the saddle")
show(axs[1], as_pts(s_geo0), (float(s_geo0.pos["xi1"]), float(s_geo0.pos["xi2"])), "geoVI from xi2=0: samples span both arms")
plt.tight_layout(); plt.show()
