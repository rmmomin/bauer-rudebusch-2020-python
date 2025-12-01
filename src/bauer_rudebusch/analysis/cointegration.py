"""
Cointegration regressions (Table 1 / OA Table 3) translated from `coint.R`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from arch.unitroot import ADF, PhillipsPerron
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .. import data as data_module


@dataclass
class RegressionResult:
    coefficients: pd.Series
    std_errors: pd.Series
    r_squared: float
    model: OLS
    fitted: pd.Series
    residuals: pd.Series


def _format_with_sig(value: float, critical_values: Sequence[float], higher_better: bool = False) -> str:
    """
    Append significance stars based on critical values.
    """
    stars = ""
    for threshold in critical_values:
        if higher_better and value > threshold:
            stars += "*"
        elif (not higher_better) and value < threshold:
            stars += "*"
    return f"{value:4.2f}{stars}"


def _lfst_stat(series: np.ndarray, q: int = 8, nsim: int = 5000, seed: int | None = 123) -> tuple[float, float]:
    """
    Mueller-Watson low-frequency stationarity test translated from `ur_tests.R`.
    """
    rng = np.random.default_rng(seed)
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    T = series.size

    def compute_psi(t: int, j: int) -> float:
        s = ((t + 0.5) / T) * np.pi
        return np.sqrt(2.0) * np.cos(j * s)

    psi = np.array([[compute_psi(t, j) for j in range(1, q + 1)] for t in range(T)])
    X = psi.T @ series / T
    norm = np.linalg.norm(X)
    if norm == 0:
        return 0.0, 1.0
    X /= norm

    b = 0.1
    factor = np.diag((1 + 1 / (b * np.pi * np.arange(1, q + 1)) ** 2) ** (-0.5))
    Xa = factor @ X
    stat = float(np.sum(X**2) / np.sum(Xa**2))

    sims = rng.normal(size=(q, nsim))
    sim_norms = np.linalg.norm(sims, axis=0, keepdims=True)
    sim_norms[sim_norms == 0] = 1.0
    sims_norm = sims / sim_norms
    sims_alt = factor @ sims_norm
    sim_stats = np.sum(sims_norm**2, axis=0) / np.sum(sims_alt**2, axis=0)
    pvalue = float(np.mean(sim_stats > stat))
    return stat, pvalue


def _persistence_stats(residuals: pd.Series, nvar: int = 1) -> tuple[str, str, str, str, str, str]:
    residuals = residuals.dropna()
    if residuals.empty:
        return tuple(["NA"] * 6)
    sd_val = float(residuals.std(ddof=0))
    rho = float(acf(residuals, nlags=1, fft=False)[1])
    half_life = np.log(0.5) / np.log(abs(rho)) if abs(rho) < 1 else np.inf

    trend = "c" if nvar == 1 else "n"
    adf = ADF(residuals, trend=trend, lags=None)
    adf_stat = float(adf.stat)
    adf_cv = [adf.critical_values[level] for level in ("1%", "5%", "10%")]

    pp = PhillipsPerron(residuals, trend=trend)
    pp_stat = float(pp.stat)
    pp_cv = [pp.critical_values[level] for level in ("1%", "5%", "10%")]

    lfst_stat, lfst_p = _lfst_stat(residuals.to_numpy())

    return (
        f"{sd_val:4.2f}",
        f"{rho:4.2f}",
        f"{half_life:4.1f}",
        _format_with_sig(adf_stat, adf_cv),
        _format_with_sig(pp_stat, pp_cv),
        f"{lfst_p:4.2f}",
    )


def _build_dols_design(
    data: pd.DataFrame,
    y_col: str,
    regressors: Sequence[str],
    diff_cols: Sequence[str],
    p: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = data.copy()
    extra_cols: list[str] = []
    for col in diff_cols:
        diff = frame[col].diff()
        for k in range(-p, p + 1):
            if k == 0:
                continue
            shifted = diff.shift(k)
            suffix = f"{'m' if k < 0 else 'p'}{abs(k)}"
            name = f"d_{col}_{suffix}"
            frame[name] = shifted
            extra_cols.append(name)
    cols = list(regressors) + extra_cols
    design = frame[cols].dropna()
    y = frame.loc[design.index, y_col]
    X = add_constant(design)
    return X, y


def _fit_dols(data: pd.DataFrame, y_col: str, regressors: Sequence[str], diff_cols: Sequence[str], p: int = 4) -> RegressionResult:
    X, y = _build_dols_design(data, y_col, regressors, diff_cols, p=p)
    model = OLS(y, X).fit()
    cov = cov_hac(model, nlags=6)
    se = pd.Series(np.sqrt(np.diag(cov)), index=X.columns)
    fitted = model.fittedvalues
    residuals = y - fitted
    return RegressionResult(
        coefficients=model.params,
        std_errors=se,
        r_squared=float(model.rsquared),
        model=model,
        fitted=fitted,
        residuals=residuals,
    )


def _johansen_tests(matrix: np.ndarray, det_order: int = 0, k_ar_diff: int = 4) -> tuple[str, str]:
    res = coint_johansen(matrix, det_order, k_ar_diff)
    stat_r0 = res.lr1[-1]
    stat_r1 = res.lr1[-2] if res.lr1.size > 1 else np.nan
    cv_r0 = res.cvt[-1]
    cv_r1 = res.cvt[-2] if res.cvt.shape[0] > 1 else np.full(3, np.nan)
    def format_stat(stat: float, crit: Sequence[float]) -> str:
        stars = "".join("*" for c in crit if stat > c)
        return f"{stat:4.2f}{stars}"
    return format_stat(stat_r0, cv_r0), format_stat(stat_r1, cv_r1)


def _error_correction_model(data: pd.DataFrame, y_col: str, residual_col: str, diff_cols: Sequence[str], p: int = 4) -> tuple[str, str]:
    frame = data.copy()
    frame["dy"] = frame[y_col].diff()
    frame["lag_resid"] = frame[residual_col].shift(1)
    regressors = ["lag_resid"]
    for col in diff_cols:
        diff = frame[col].diff()
        for lag in range(1, p + 1):
            lagged = diff.shift(lag)
            name = f"d_{col}_lag{lag}"
            frame[name] = lagged
            regressors.append(name)
    for lag in range(1, p + 1):
        name = f"dy_lag{lag}"
        frame[name] = frame["dy"].shift(lag)
        regressors.append(name)

    subset = frame.dropna(subset=["dy"] + regressors)
    if subset.empty:
        return "NA", "NA"

    X = add_constant(subset[regressors])
    model = OLS(subset["dy"], X).fit(cov_type="HC0")
    alpha = float(model.params.get("lag_resid", np.nan))
    se = float(model.bse.get("lag_resid", np.nan))
    return f"{alpha:4.2f}", f"({se:4.2f})"


def run_cointegration(yield_var: str = "y10", output_dir: Path | str = "tables") -> pd.DataFrame:
    """
    Compute the cointegration table analogous to `coint.R`.
    """
    data = data_module.load_data()
    data["istar"] = data["pistar.ptr"] + data["rstar.realtime"]

    rstar_vars = ["rstar.filt", "rstar.realtime", "rr.ewma"]
    rstar_desc = ["filtered", "real-time", "mov.~avg."]

    rows = [
        ("constant", "constant"),
        ("constant_se", ""),
        ("pi", r"$\\pi_t^\\ast$"),
        ("pi_se", ""),
        ("r", r"$r_t^\\ast$"),
        ("r_se", ""),
        ("i", r"$i_t^\\ast$"),
        ("i_se", ""),
        ("r2", r"$R^2$"),
        ("memo", "Memo: $r^\\ast$"),
        ("sd", "SD"),
        ("rho", r"$\\hat{\\rho}$"),
        ("half_life", "Half-life"),
        ("adf", "ADF"),
        ("pp", "PP"),
        ("lfst", "LFST"),
        ("johansen_r0", "Johansen $r=0$"),
        ("johansen_r1", "Johansen $r=1$"),
        ("ecm_alpha", r"ECM $\\hat{\\alpha}$"),
        ("ecm_alpha_se", ""),
    ]

    columns = ["label", "Benchmark", "pi* only"] + [f"pi* + {desc}" for desc in rstar_desc] + ["i*"]
    results = pd.DataFrame("", index=[key for key, _ in rows], columns=columns)
    for key, label in rows:
        results.at[key, "label"] = label

    # Constant-only regression
    bench_res = _fit_dols(data, yield_var, [], [], p=0)
    results.at["constant", "Benchmark"] = f"{bench_res.coefficients['const']:4.2f}"
    results.at["constant_se", "Benchmark"] = f"({bench_res.std_errors['const']:4.2f})"
    stats = _persistence_stats(data[yield_var], nvar=1)
    for key, value in zip(["sd", "rho", "half_life", "adf", "pp", "lfst"], stats):
        results.at[key, "Benchmark"] = value

    # pi-star only
    pi_res = _fit_dols(data, yield_var, ["pistar.ptr"], ["pistar.ptr"])
    results.at["pi", "pi* only"] = f"{pi_res.coefficients['pistar.ptr']:4.2f}"
    results.at["pi_se", "pi* only"] = f"({pi_res.std_errors['pistar.ptr']:4.2f})"
    results.at["r2", "pi* only"] = f"{pi_res.r_squared:4.2f}"
    data["resid_pi"] = np.nan
    data.loc[pi_res.fitted.index, "resid_pi"] = pi_res.residuals
    stats = _persistence_stats(data["resid_pi"], nvar=2)
    for key, value in zip(["sd", "rho", "half_life", "adf", "pp", "lfst"], stats):
        results.at[key, "pi* only"] = value
    johansen_r0, johansen_r1 = _johansen_tests(data[[yield_var, "pistar.ptr"]].dropna().to_numpy())
    results.at["johansen_r0", "pi* only"] = johansen_r0
    results.at["johansen_r1", "pi* only"] = johansen_r1
    alpha, alpha_se = _error_correction_model(data, yield_var, "resid_pi", ["pistar.ptr"])
    results.at["ecm_alpha", "pi* only"] = alpha
    results.at["ecm_alpha_se", "pi* only"] = alpha_se

    # pi-star + r-star variants
    for var, desc in zip(rstar_vars, rstar_desc):
        col_name = f"pi* + {desc}"
        res = _fit_dols(data, yield_var, ["pistar.ptr", var], ["pistar.ptr", var])
        results.at["pi", col_name] = f"{res.coefficients['pistar.ptr']:4.2f}"
        results.at["pi_se", col_name] = f"({res.std_errors['pistar.ptr']:4.2f})"
        results.at["r", col_name] = f"{res.coefficients[var]:4.2f}"
        results.at["r_se", col_name] = f"({res.std_errors[var]:4.2f})"
        results.at["r2", col_name] = f"{res.r_squared:4.2f}"
        data[f"resid_{var}"] = np.nan
        data.loc[res.fitted.index, f"resid_{var}"] = res.residuals
        stats = _persistence_stats(data[f"resid_{var}"], nvar=3)
        for key, value in zip(["sd", "rho", "half_life", "adf", "pp", "lfst"], stats):
            results.at[key, col_name] = value
        mat = data[[yield_var, "pistar.ptr", var]].dropna().to_numpy()
        johansen_r0, johansen_r1 = _johansen_tests(mat)
        results.at["johansen_r0", col_name] = johansen_r0
        results.at["johansen_r1", col_name] = johansen_r1
        alpha, alpha_se = _error_correction_model(data, yield_var, f"resid_{var}", ["pistar.ptr", var])
        results.at["ecm_alpha", col_name] = alpha
        results.at["ecm_alpha_se", col_name] = alpha_se
        results.at["memo", col_name] = desc

    # i-star regression
    istar_res = _fit_dols(data, yield_var, ["istar"], ["istar"])
    results.at["i", "i*"] = f"{istar_res.coefficients['istar']:4.2f}"
    results.at["i_se", "i*"] = f"({istar_res.std_errors['istar']:4.2f})"
    results.at["r2", "i*"] = f"{istar_res.r_squared:4.2f}"
    data["resid_istar"] = np.nan
    data.loc[istar_res.fitted.index, "resid_istar"] = istar_res.residuals
    stats = _persistence_stats(data["resid_istar"], nvar=2)
    for key, value in zip(["sd", "rho", "half_life", "adf", "pp", "lfst"], stats):
        results.at[key, "i*"] = value
    mat = data[[yield_var, "istar"]].dropna().to_numpy()
    johansen_r0, johansen_r1 = _johansen_tests(mat)
    results.at["johansen_r0", "i*"] = johansen_r0
    results.at["johansen_r1", "i*"] = johansen_r1
    alpha, alpha_se = _error_correction_model(data, yield_var, "resid_istar", ["istar"])
    results.at["ecm_alpha", "i*"] = alpha
    results.at["ecm_alpha_se", "i*"] = alpha_se
    results.at["memo", "i*"] = "real-time"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / ("coint_level.tex" if yield_var == "level" else "coint.tex")
    table = results.reset_index(drop=True)
    table.to_csv(tex_path.with_suffix(".csv"), index=False)
    table.to_latex(tex_path, index=False, escape=False)
    return results
