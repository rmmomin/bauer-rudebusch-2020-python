# Bauer–Rudebusch 2020 (Python Port)

This repository is an in-progress attempt to port the original Bauer and Rudebusch (2020) replication codebase from R into Python. The goal is to reproduce the empirical results and figures from the paper while providing a Python-native workflow that is easier to extend and integrate with modern data science tooling.

## Project Layout
- `R/` retains the upstream R scripts for reference during the porting effort. `master.R` reproduces the original workflow end to end.
- `data/` holds the public datasets distributed with the original materials. No proprietary data are included.
- `src/bauer_rudebusch/` houses the Python port. Key modules include environment setup (`setup_env.py`), shared utilities (`util.py`), data loading helpers (`data.py`), plotting routines (`analysis/static_outputs.py`), cointegration and unit-root diagnostics (`analysis/cointegration.py`), predictive regressions (`analysis/predict.py`), and the high-level workflow orchestrator (`master.py`).
- The repository root contains a Python `master.py` wrapper so `python master.py` mirrors the original `master.R` entry point.

## Getting Started
1. Install the R dependencies listed in `setup.R` if you plan to run the original scripts as a baseline.
2. Create a Python environment (3.10+) for the port:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Run `master.R` to verify the baseline results before comparing them with the Python outputs.

## Python Scaffolding
- `requirements.txt` tracks the initial dependency set expected to support data handling, econometric routines, plotting, and testing (`numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `pytest`).
- Set up new Python modules under `src/` (recommended) as you translate the R scripts; update `requirements.txt` when additional libraries are needed.
- Add notebooks or scripts under a dedicated directory (e.g., `notebooks/`) to keep exploratory work organized and reproducible.
- Run the Python workflow with `python master.py`. The script now reproduces the baseline figures (Figures 1–3), Tables 1–2, and estimates the observed shifting-endpoint (OSE) DTSM, while flagging the Bayesian ESE and out-of-sample forecasting stages as pending.

## Current Status
- Core data preparation utilities, plotting routines, predictive regressions, and the OSE DTSM estimator now run natively in Python.
- Bayesian DTSM estimation (`ese.R`) and the associated out-of-sample forecasting workflows remain to be ported.
- Validation tests comparing Python outputs with the established R results are planned as the remaining modules come online.

## Contributing
Contributions are welcome—especially around translating specific R scripts to Python, setting up data pipelines, or adding automated tests. Please open an issue to discuss your plans before starting major work.

## License
See `LICENSE` (to be confirmed) for reuse terms; in the absence of a dedicated license, assume the original authors retain all rights.
