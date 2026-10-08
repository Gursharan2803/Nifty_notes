# NIFTy in a Day — hands-on notebooks

Seven executed Jupyter notebooks that accompany the "NIFTy in a Day — Lecture Notes" doc.
Each notebook explains a concept, implements it, then uses NIFTy; ✏️ exercises are near the end, solutions at the bottom.

| # | Notebook | Topic | Runtime (2-core CPU) |
|---|---|---|---|
| 0 | `00_concepts_from_scratch.ipynb` | Prior/likelihood/posterior, optimizers, CG, fields, FFT, power spectra, Gaussian random fields, mini correlated field, Wiener filter, MGVI — in plain NumPy/JAX | < 1 min |
| 1 | `01_foundations_jax_and_models.ipynb` | JAX essentials, standardized priors, `jft.Model`, likelihoods, MAP/MGVI/geoVI with `optimize_kl` | < 1 min |
| 2 | `02_gaussian_processes_and_wiener_filter.ipynb` | Grids, fixed-spectrum GP model, masks, `wiener_filter_posterior`, cross-check vs hand CG, deconvolution | < 1 min |
| 3 | `03_correlated_field_model.ipynb` | `CorrelatedFieldMaker`, hyper-parameter sweeps, joint field + spectrum inference, log-normal 2-D, padding, product fields | ~2.5 min |
| 4 | `04_variational_inference_deep_dive.ipynb` | MAP vs MGVI vs geoVI vs NUTS, sample modes, point estimates, minisanity, ELBO model comparison | ~3.5 min |
| 5 | `05_real_world_poisson_and_tomography.ipynb` | Photon-count (Poisson) imaging with PSF, exposure and dead pixels; why point sources are hard; line-of-sight tomography | ~1.5 min |
| 6 | `06_nifty_cl_classic.ipynb` | `nifty.cl`: domains, fields, operators, custom LinearOperator, Wiener filter, SimpleCorrelatedField + optimize_kl | ~1.5 min |

## Setup

```bash
python -m venv nifty-env && source nifty-env/bin/activate
pip install 'nifty[re]' ducc0 blackjax matplotlib jupyter
jupyter lab
```

Tested with nifty 9.2.0, jax 0.11.2, blackjax 1.7.1, Python 3.13.
**Troubleshooting a stalled run.** On the 2-core test machine, notebook 05 twice froze at 0% CPU in the middle of
`optimize_kl` (a hang inside XLA's multithreaded CPU runtime, not a slow computation). Starting Jupyter with
single-threaded XLA fixed it:

```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" jupyter lab
```

Only use this if you see the stall; on most machines the default is faster.

The `src/` folder holds the same notebooks as plain Python (jupytext percent format) for easy diffing/editing.
