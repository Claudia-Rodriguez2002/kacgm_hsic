![KaCGM](assets/KaCGM.png)

# KaCGM featuring HSIC

This is the codebase for the Master Thesis (TFM) of Claudia Rodríguez Coromina, supervised by Alejandro Almodóvar. Its title is:

> *Implementation and validation of interpretable causal generative models for treatment effect estimation*

The Master Thesis Report contains a detailed explanation of the reasearch, methology and results of the experiments made in this repository. 

> [!IMPORTANT]
> Please, note that this project is based on the repository by Alejandro Almodóvar:
> https://github.com/aalmodovares/kacgm/tree/main
> Also note that this was a refactored version of the first experimental code, so it can contain minimal errors that we haven´t found.
> If you find any issue, please, report it to us by opening an issue in the repository or contact me to:
> <alejandro.almodovar@upm.es>
>  We will be happy to fix it as soon as possible.

## Context

This Master's Thesis investigates whether explicitly incorporating the Hilbert-Schmidt Independence Criterion (HSIC) into the training objective of Kolmogorov-Arnold Networks (KANs) can improve the causal reliability of structured causal generative models — specifically KaCGM — without sacrificing their interpretability. It designs an HSIC-based regularizer to enforce noise-parent independence, evaluates the resulting accuracy–interpretability trade-off against baseline architectures, and tests the regularizer's robustness under controlled spurious dependencies.

## Repository Layout

```text
.
|-- assets/                  # Static figures used in the repository README and paper support material
|-- configs/                 # Default experiment configurations
|-- notebooks/               # Analysis notebooks and report plots
|   |-- Experimento1_aditivo/          # Synthetic additive-noise experiments
|   |-- Experimento1_no_aditivo/       # Synthetic multiplicative-noise experiments
|   |-- Experimento2 datasets/         # Synthetic graph-benchmark datasets
|   |-- Experimento3 sachs/            # Sachs semi-synthetic experiments
|   |-- Experimento4 cardio/           # Cardio case-study notebooks
|   |-- Resumenes/                     # Summary notebooks aggregating results across experiments
|   `-- extras/                        # Miscellaneous supporting notebooks
|-- outputs/                 # Generated samples, metrics, figures, checkpoints (git-ignored)
|-- plots/                   # Plot outputs and visualizations
|-- runnables/               # Executable experiment entry points
|-- src/
|   |-- datasets/            # Dataset generators and case-study loaders
|   |-- models/              # KAN, flow, and diffusion-based causal models
|   |-- plotting/            # Notebook/result loading helpers
|   `-- utils/               # Paths, metrics, evaluation, hyperparameter search, stats
`-- tests/                   # Lightweight pytest coverage for the refactor
```

## Installation

Create a clean environment and install the project in editable mode:

```bash
conda create -n kacgm_hsic python==3.11 -y
conda activate kacgm_hsic
conda install -c conda-forge llvmlite numba -y
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The editable install is required so the `datasets`, `models`, `plotting`, and `utils` packages resolve correctly in both the runnables and the notebooks.

## Running Notebooks

Notebooks are intended for experimenting with the reasearch line of the project. It is divided into four experiments, organized in the `notebooks/` folder:

- Experiment 1: Synthetic simple graph experiment
- Experiment 2: Synthetic multiple graph and noise experiment
- Experiment 3: Semi-synthetic sachs experiment
- Experiment 4: Cardio use case application

All experiment outputs are written under `outputs/<experiment>/` with separate `data/`, `samples/`, `images/`, `figures/`, and `checkpoints/` subfolders where applicable. These generated artifacts are intentionally ignored by git.

## Tests

Run the lightweight structural checks with:

```bash
pytest
```

## Notes

- The cardio dataset is not included in the repository, but you can find it in the original paper by
[Kyriaocu et al.](https://www.sciencedirect.com/science/article/pii/S0196064422005789). The dataset should be stored as
`data/raw/cardio.csv` for the runnables to work.
- `outputs/` is kept as a skeleton in the repository so all runnables share a predictable output layout. Some plots from this master project are kept though.

## Contact

If you find this work useful for your research, please contact me by <claudia.rodriguez.coromina@alumnos.upm.es> or <alejandro.almodovar@upm.es>

