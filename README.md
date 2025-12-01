# Bauer–Rudebusch 2020 (Python Port)

This repository is an in-progress attempt to port the original Bauer and Rudebusch (2020) replication codebase from R into Python. The goal is to reproduce the empirical results and figures from the paper while providing a Python-native workflow that is easier to extend and integrate with modern data science tooling.

## Project Layout
- `R/` retains the upstream R scripts for reference during the porting effort. `master.R` reproduces the original workflow end to end.
- `data/` holds the public datasets distributed with the original materials. No proprietary data are included.
- `src/bauer_rudebusch/` houses the Python port. Key modules include environment setup (`setup_env.py`), shared utilities (`util.py`), data loading helpers (`data.py`), plotting routines (`analysis/static_outputs.py`), cointegration and unit-root diagnostics (`analysis/cointegration.py`), predictive regressions (`analysis/predict.py`), and the high-level workflow orchestrator (`master.py`).
- `figures/`, `tables/`, and `results/` are populated by the Python workflow with the recreated plots (Figures 1–3), LaTeX/CSV tables (Tables 1–2), and serialized DTSM estimates (`results/ose.pkl`), respectively.
- The repository root contains a Python `master.py` wrapper so `python master.py` mirrors the original `master.R` entry point.

## Getting Started
1. Install the R dependencies listed in `setup.R` if you plan to run the original scripts as a baseline.
2. Create a project-local Python environment (3.10+) so the pinned dependency set does not conflict with a system Anaconda install:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
3. Run `master.R` (optional) to verify the baseline results before comparing them with the Python outputs.
4. Execute the Python workflow with `OMP_NUM_THREADS=1 .venv/bin/python master.py`. Capping OpenMP threads avoids shared-memory errors on macOS Sonoma when NumPy/Scipy link against the Accelerate framework.

## Python Scaffolding
- `requirements.txt` pins a NumPy 1.26 / Pandas 2.1 stack together with `statsmodels`, `arch`, `numexpr`, `bottleneck`, `jinja2`, and the plotting/test libraries needed for the tables and figures. The `<2.0` cap on NumPy prevents crashes with binary wheels that have not yet been rebuilt for the array API in NumPy 2.x.
- Add new Python modules under `src/` (recommended) as you translate the R scripts; update `requirements.txt` when additional libraries are needed.
- Place exploratory notebooks or ad‑hoc scripts in a dedicated directory (e.g., `notebooks/`) to keep the main package importable.
- Run the workflow via `OMP_NUM_THREADS=1 .venv/bin/python master.py`. The script reproduces the static figures, Table 1 (cointegration), Table 2 (return predictability), and estimates the OSE DTSM. The Bayesian ESE system and out-of-sample forecasting blocks are still marked as “pending implementation,” matching the current status in the R materials.

## Current Status
- Core data preparation utilities, plotting routines, predictive regressions, and the OSE DTSM estimator now run natively in Python. Each full run regenerates `figures/`, `tables/`, and `results/ose.pkl`.
- Bayesian DTSM estimation (`ese.R`), the bootstrap infrastructure, and the downstream out-of-sample forecasting workflows remain to be ported.
- Validation tests comparing Python outputs with the established R results are planned as the remaining modules come online. See `agent_documentation/porting_status.md` for a living checklist.

## Contributing
Contributions are welcome—especially around translating specific R scripts to Python, setting up data pipelines, or adding automated tests. Please open an issue to discuss your plans before starting major work.

## License
See `LICENSE` (to be confirmed) for reuse terms; in the absence of a dedicated license, assume the original authors retain all rights.
