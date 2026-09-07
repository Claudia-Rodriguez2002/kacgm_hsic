# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase for KaCGM (Kolmogorov-Arnold Causal Generative Models). It implements causal generative
models using Kolmogorov-Arnold Networks (KAN) as the structural-equation predictor inside DoWhy `gcm`
causal models, and runs benchmark experiments (synthetic graphs, Sachs semi-synthetic, a cardio case
study) comparing this approach against baselines (ANM, DBCM, normalizing flows). Results back a paper;
notebooks turn stored experiment outputs into paper figures.

> The README notes this is a refactor of the original experimental code and "can contain minimal errors."
> Treat surprising behavior in `src/models/kan.py` and the evaluation pipeline with that in mind.

## Setup

```bash
conda create -n kacgm python==3.11
conda activate kacgm
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The editable install is required: `datasets`, `models`, `plotting`, `utils` live under `src/` and are
imported as top-level packages (e.g. `from models.kan import kan_model_mixed`), not `src.models.kan`.
`pyproject.toml` sets `package-dir = {"" = "src"}`. Tests get `src/` on `sys.path` via
`pythonpath = ["src"]` in `[tool.pytest.ini_options]` (also done manually in `tests/conftest.py`).

## Commands

Run tests:
```bash
pytest                       # whole suite (see testpaths/addopts in pyproject.toml)
pytest tests/test_hsic.py    # single file
pytest tests/test_hsic.py::test_name -q   # single test
```

Run an experiment (all runnables live in `runnables/` and are invoked as `-m runnables.<name>`, e.g.):
```bash
python -m runnables.run_continuous_benchmark --n-jobs 8
python -m runnables.run_discrete_benchmark --n-jobs 8
python -m runnables.run_sachs_benchmark --hp-jobs 8 --run-jobs 4
python -m runnables.run_cardio_case_study --n-jobs 8 [--include-interventions]
python -m runnables.run_cardio_bootstrap_evaluation --n-seeds 10 --n-bootstraps 10 --n-jobs 8 [--store-samples]
python -m runnables.run_sensitivity --jobs 8
python -m runnables.run_symbolic_pruning_study
python -m runnables.run_diagnostic_power
python -m runnables.run_graph_misspecification_sensitivity
```
Every runnable exposes `--help`; `tests/test_runnables_smoke.py` asserts the main ones parse args
without importing heavy/optional dependencies (see "Lazy imports" below) — a good template when adding
a new runnable's CLI.

All runnables write to `outputs/<experiment>/{data,samples,images,figures,checkpoints}/` (created via
`utils.paths.get_experiment_paths`). `outputs/` is git-ignored except for a checked-in skeleton
(`.gitkeep` files), so don't expect experiment output to be committed.

## Architecture

### Package layout and import style

- `src/datasets`, `src/models`, `src/utils`, `src/plotting` are the installed packages; `runnables/`
  contains thin CLI scripts that orchestrate them; `notebooks/` consumes experiment outputs for plots.
- The top-level `__init__.py` of `datasets`, `models`, and `utils` do **not** eagerly import their
  submodules. Instead each defines an `_EXPORTS` dict mapping public name → `(submodule, attr)` and a
  module-level `__getattr__` that lazily imports on first access (PEP 562 style). This is why
  `from models import kan_predictor` works without importing `models.kan` at package-import time, and
  why adding a new public symbol means updating both the submodule and the `_EXPORTS` dict in
  `__init__.py` — grep for `_EXPORTS` before assuming a name isn't exported.
- Runnables and some `utils` functions additionally do function-local imports of optional/heavy
  dependencies (e.g. `from models.flow import causalflow_model` inside a function body rather than at
  module scope). This keeps `--help` and unrelated code paths working even when a model's dependency
  chain isn't installed or its module is missing (see next point) — preserve this pattern rather than
  hoisting such imports to the top of the file.

### Known gap: `models.flow` / `models.dbcm` are referenced but not present

`src/models/factory.py`, `src/utils/hyperparams.py`, `src/utils/evaluation.py`, and several runnables
import `models.flow.causalflow_model` (the normalizing-flow baseline) and
`models.dbcm.create_model_from_graph` (the DBCM baseline), but no `src/models/flow.py` or
`src/models/dbcm.py` exists in this checkout. Because these imports are function-local, most code
(including `--help` and the "kan"/"kaam"/"anm" model paths) still runs; only code paths that actually
select `model="flow"` or `model="dbcm"` will fail with `ModuleNotFoundError`. Don't assume these
baselines are runnable without first checking whether the modules exist.

### Model layer (`src/models`)

- `models/kan.py` is the core: `kan_predictor` wraps a `pykan` `KAN` network with a scikit-like
  `fit`/`predict` interface and a custom training loop (`custom_fit`) supporting three losses:
  `mse`, `hsic` (an HSIC-based independence loss between inputs and residuals, `hsic_loss`), and
  `hybrid` (weighted sum of both, via `alpha_weight`/`beta_weight` hyperparameters). `kan_model_mixed`
  composes per-node `kan_predictor`s (or a `gcm.ScipyDistribution` for root nodes) over a `networkx`
  DAG to form a full SCM supporting discrete and continuous nodes, sampling, interventions, and
  counterfactuals. `symbolic_kan_regressor` fits closed-form symbolic expressions (polynomial or named
  nonlinear families) to a trained single-layer KAN's activations, used to extract human-readable causal
  mechanisms ("KAAM" = KAN + symbolic/formula extraction).
- `models/factory.py::create_model_from_graph(g, model=..., params=..., noise=...)` is the dispatcher
  that builds a DoWhy `gcm.InvertibleStructuralCausalModel` and assigns causal mechanisms per model type
  (`"kan"`/`"kaam"` → KAN-based `AdditiveNoiseModel`, `"anm"` → left for DoWhy auto-assignment,
  `"dbcm"` → delegates to `models.dbcm`, currently missing). This is the main entry point runnables use
  to instantiate a model from a graph + hyperparameters.

### Datasets (`src/datasets`)

- `datasets/synthetic.py::graph_data(name=...).generate(...)` is a large if/elif dispatch generating
  synthetic SCM benchmarks (chains, colliders, forks, Simpson's paradox graphs, triangles — linear and
  non-linear variants) with matching ground-truth structural formulas and precomputed counterfactuals
  for evaluation. Dataset names are looked up by string in `configs/*.yaml` and runnable defaults —
  when adding a new synthetic graph, follow the existing pattern (define noise, standardize, build a
  `networkx.DiGraph`, closed-form `formula` dict, and `cf_*` counterfactual generator functions).
- `datasets/sachs.py` and `datasets/cardio.py` load/prepare the semi-synthetic Sachs protein-signaling
  network and the cardio case-study dataset respectively. The cardio dataset is **not** included in the
  repo; it must be placed at `data/raw/cardio.csv` (gitignored) — see README "Notes".

### Evaluation and hyperparameter search (`src/utils`)

- `utils/hyperparams.py::get_best_hyperparams` runs a `joblib`-parallelized grid search over
  hyperparameter candidates, evaluating each combination by MMD/random-forest-discriminator accuracy
  between real and sampled data (`utils/metrics.py`), and can cache/reload best params via pickle files
  in the experiment's `data/` dir to avoid re-searching (`load_existent` flag). KAN/KAAM search picks
  best hyperparameters **per node**, not globally.
- `utils/evaluation.py::evaluate_model` runs a fitted model through observational sampling,
  interventional sampling, and counterfactual estimation, computing MMD/RF-accuracy/MSE/MAE metrics and
  saving diagnostic histograms/plots to the experiment's `images/` dir. `evaluate_kaam`/
  `evaluate_kaam_mixed` additionally fit a symbolic formula per node and compare it against the
  ground-truth formula (when available) both numerically (MAE over uniform/weighted samples) and
  visually (1D/2D fit plots).
- `utils/metrics.py` has the core statistical primitives: `mmd` (kernel two-sample test), `rf`
  (train/test classifier discriminating real vs. generated), and HSIC/dHSIC utilities
  (`gaussian_grammat`, `centering`, `HSIC`, `dHSIC`) used both as evaluation metrics and inside the KAN
  HSIC training loss.
- `utils/stats.py` implements Friedman + Holm post-hoc statistical significance testing across
  datasets/methods, used to compare benchmark results in the paper's tables.
- `utils/paths.py` centralizes all repo path constants (`REPO_ROOT`, `OUTPUTS_ROOT`, etc.) and
  `get_experiment_paths(name, output_dir=None)`, which is the standard way every runnable resolves and
  creates its `outputs/<name>/{data,samples,images,figures,checkpoints}` tree. Prefer this helper over
  hand-rolling paths in new runnables.

### Runnables (`runnables/`)

Each `run_*.py` is a standalone script with a `run_*(...)` function plus an `argparse` CLI in
`if __name__ == "__main__":`. They share a consistent shape: resolve experiment paths, load/generate
data, grid-search or load cached hyperparameters, fit each model in a configured list, call
`evaluate_model`, and persist CSV/pickle results under `outputs/`. When adding a new experiment, follow
an existing runnable (e.g. `run_continuous_benchmark.py`) as the template rather than inventing new
orchestration.

### Notebooks

Notebooks under `notebooks/` (`paper_plots.ipynb`, `sachs_analysis.ipynb`, `cardio_bootstrap_results.ipynb`,
etc.) read stored CSV/pickle results from `outputs/` via `src/plotting/loaders.py` and
`src/plotting/cardio_formula.py` — run the corresponding runnable first, they do not regenerate data
themselves.
