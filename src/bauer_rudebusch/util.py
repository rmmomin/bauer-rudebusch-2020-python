"""
Utility functions translated from `R/util_fns.R`.

The implementations prefer NumPy/SciPy/Statsmodels APIs while preserving the
semantics of the original R helpers used throughout the replication code.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.linalg import cholesky, inv
from scipy import stats
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import acf as sm_acf

# Small positive constant used when repairing covariance matrices.
_EPS = 1e-6


@dataclass
class HypothesisTestResult:
    """
    Container for hypothesis test metadata to mimic R's `htest` object.
    """

    statistic: float
    pvalue: float
    alternative: str
    df: int | None
    method: str
    parameter: tuple[int, int]
    data_name: tuple[str, str] | None = None


def _acf(d: np.ndarray, lag_max: int) -> np.ndarray:
    """
    Wrapper around statsmodels' `acf` returning auto-covariances.
    """
    # statsmodels' acf yields autocorrelation; multiply by variance to obtain covariance.
    acf_vals = sm_acf(d, nlags=lag_max, fft=False, adjusted=False)
    variance = np.nanvar(d, ddof=0)
    return acf_vals * variance


def my_dm_test(
    e1: Sequence[float],
    e2: Sequence[float],
    *,
    alternative: str = "two.sided",
    h: int = 1,
    power: int = 2,
    lrvar: str = "HH",
) -> HypothesisTestResult:
    """
    Diebold-Mariano test translated from `myDMtest`.
    """
    alternative = alternative.lower()
    if alternative not in {"two.sided", "less", "greater"}:
        raise ValueError("alternative must be 'two.sided', 'less', or 'greater'")
    lrvar = lrvar.upper()
    if lrvar not in {"HH", "NW"}:
        raise ValueError("lrvar must be 'HH' or 'NW'")

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d = d[~np.isnan(d)]

    if lrvar == "HH":
        lag_max = max(h - 1, 0)
        d_cov = _acf(d, lag_max)
        dv = (d_cov[0] + 2 * np.sum(d_cov[1:])) / len(d)
    else:
        # Newey-West with L = floor(1.5*(h-1))
        L = int(math.floor(1.5 * (h - 1)))
        lag_max = max(L, 0)
        d_cov = _acf(d, lag_max)
        if lag_max > 0:
            weights = 1 - (np.arange(1, lag_max + 1) / (lag_max + 1))
            dv = (d_cov[0] + 2 * np.sum(weights * d_cov[1:])) / len(d)
        else:
            dv = d_cov[0] / len(d)

    if dv <= 0:
        if h == 1:
            raise ValueError("Variance of DM statistic is zero")
        warnings.warn(
            f"Variance for h = {h} is non-positive; retrying with horizon {h-1}",
            RuntimeWarning,
        )
        return my_dm_test(e1, e2, alternative=alternative, h=h - 1, power=power)

    statistic = np.nanmean(d) / math.sqrt(dv)
    n = len(d)
    k = math.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    statistic *= k

    df = n - 1
    if alternative == "two.sided":
        pvalue = 2 * stats.t.sf(abs(statistic), df=df)
    elif alternative == "less":
        pvalue = stats.t.cdf(statistic, df=df)
    else:  # greater
        pvalue = stats.t.sf(statistic, df=df)

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=float(pvalue),
        alternative=alternative,
        df=df,
        method="Diebold-Mariano Test",
        parameter=(h, power),
        data_name=None,
    )


def draw_normal(mu: Sequence[float], omega: np.ndarray) -> np.ndarray:
    """
    Draw a sample from a multivariate normal distribution.
    """
    mu = np.asarray(mu, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return mu + cholesky(omega) @ np.random.normal(size=mu.size)


def get_optim(
    theta: Sequence[float],
    obj,
    *,
    args: Sequence | None = None,
    kwargs: dict | None = None,
    trace: int = 0,
) -> np.ndarray:
    """
    Optimization helper emulating the robust procedure in `get_optim`.
    """
    args = tuple(args or ())
    kwargs = dict(kwargs or {})
    theta = np.asarray(theta, dtype=float)

    def wrapped(x):
        return float(obj(x, *args, **kwargs))

    val = wrapped(theta)
    if trace:
        print(f"Value at starting point: {val}")

    improvement = -np.inf
    current_theta = theta
    current_val = val
    iteration = 1
    while improvement < -0.1:
        res = minimize(
            wrapped,
            current_theta,
            method="Nelder-Mead",
            options={"maxiter": 5000, "disp": False},
        )
        improvement = res.fun - current_val
        current_val = res.fun
        current_theta = res.x
        if trace:
            print(f"iteration {iteration}, value = {current_val}")
        iteration += 1

    if trace:
        print("improvement reached threshold; switching to gradient-based step")

    improvement = -np.inf
    iteration = 1
    while improvement < -0.1:
        try:
            res = minimize(
                wrapped,
                current_theta,
                method="L-BFGS-B",
                options={"maxiter": 50000},
            )
            improvement = res.fun - current_val
            current_val = res.fun
            current_theta = res.x
            if trace:
                print(
                    f"Gradient-based iteration {iteration}, value = {current_val}, "
                    f"status = {res.status}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            if trace:
                print(f"Error in get_optim (L-BFGS-B): {exc}")
            break
        iteration += 1

    return current_theta


def logdinvgamma(x: float, alpha: float, beta: float) -> float:
    """
    Log-density of the Inverse-Gamma distribution evaluated at `x`.
    """
    return alpha * math.log(beta) - math.lgamma(alpha) - (alpha + 1) * math.log(x) - (
        beta / x
    )


def matrix_power(x: np.ndarray, exponent: float) -> np.ndarray:
    """
    Raise a matrix to a (possibly fractional) power using eigendecomposition.
    """
    x = np.asarray(x, dtype=float)
    return fractional_matrix_power(x, exponent)


def make_pd(a: np.ndarray) -> np.ndarray:
    """
    Ensure a symmetric matrix is positive definite by adjusting eigenvalues.
    """
    a = np.asarray(a, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    if np.min(eigenvalues) >= 0:
        return a
    adjusted = np.clip(eigenvalues, _EPS, None)
    return eigenvectors @ np.diag(adjusted) @ eigenvectors.T


def plot_recessions(ax, date_range, y_range, *, alpha: float = 0.5, color: str = "gray"):
    """
    Add US recession shading to a Matplotlib axis, matching `plot_recessions`.
    """
    import pandas as pd

    recessions = np.array(
        [
            (196912, 197011),
            (197311, 197503),
            (198001, 198007),
            (198107, 198211),
            (199007, 199103),
            (200103, 200111),
            (200712, 200906),
        ]
    )

    def _to_date(yyyymm: int) -> pd.Timestamp:
        year = yyyymm // 100
        month = yyyymm % 100
        return pd.Timestamp(year=year, month=month, day=1)

    start = pd.to_datetime(date_range[0])
    min_period = int(start.strftime("%Y%m"))

    for start_mm, end_mm in recessions:
        if end_mm <= min_period:
            continue
        start_date = _to_date(start_mm)
        end_date = _to_date(end_mm)
        ax.fill_between(
            [start_date, end_date],
            y_range[0],
            y_range[1],
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def invalid_cov_mat(omega: np.ndarray) -> bool:
    """
    Determine whether a symmetric matrix is not positive definite.
    """
    omega = np.asarray(omega, dtype=float)
    eigenvalues = np.linalg.eigvalsh(omega)
    return np.min(eigenvalues) < np.sqrt(np.finfo(float).eps)


def make_cov_mat(omega: Sequence[float], n: int = 3) -> np.ndarray:
    """
    Construct a symmetric covariance matrix from its packed lower-triangular entries.
    """
    omega = np.asarray(omega, dtype=float)
    expected = n * (n + 1) // 2
    if omega.size != expected:
        raise ValueError(f"Expected {expected} elements for n={n}, received {omega.size}")
    matrix = np.zeros((n, n))
    lower_indices = np.tril_indices(n)
    matrix[lower_indices] = omega
    matrix[(lower_indices[1], lower_indices[0])] = omega  # mirror
    return matrix


def make_stationary(phi: np.ndarray, *, step: float = 0.001, max_eigen: float = 0.99) -> np.ndarray:
    """
    Scale a VAR transition matrix until its eigenvalues lie inside the unit circle.
    """
    phi = np.asarray(phi, dtype=float)
    delta = 1.0
    phi_new = phi.copy()
    while np.max(np.abs(np.linalg.eigvals(phi_new))) > max_eigen:
        delta -= step
        if delta <= 0:
            raise RuntimeError("Failed to enforce stationarity; step size too large.")
        phi_new = delta * phi
    return phi_new


def make_pcs(y: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Compute principal components to match the R helper `makePCs`.
    """
    y = np.asarray(y, dtype=float)
    cov = np.cov(y, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    eigenvectors = eigenvectors[:, :n_components]
    signs = np.sign(eigenvectors[-1, :])
    eigenvectors *= signs
    pcs = y @ eigenvectors
    return pcs


def excess_returns(y: np.ndarray, maturities: Sequence[float], *, h: int = 4) -> np.ndarray:
    """
    Translate `excess_returns` to Python; works on numpy arrays (yields in percent).
    """
    maturities = np.asarray(maturities, dtype=float)
    y = np.asarray(y, dtype=float)
    if h not in (1, 4):
        raise ValueError("h must be 1 or 4")
    if y.shape[1] != maturities.size:
        raise ValueError("Mismatch between yield columns and maturities.")

    nobs = y.shape[0]
    maturities_gt1 = maturities[maturities > 1]
    result = np.full((nobs, maturities_gt1.size), np.nan)

    for idx, n in enumerate(maturities_gt1):
        nmh = n - 1 if h == 4 else n
        nmh_idx = np.where(np.isclose(maturities, nmh))[0]
        n_idx = np.where(np.isclose(maturities, n))[0]
        h_idx = np.where(np.isclose(maturities, h / 4))[0]
        if not (nmh_idx.size and n_idx.size and h_idx.size):
            raise ValueError("Required maturities are missing for excess return computation.")
        nmh_idx = nmh_idx[0]
        n_idx = n_idx[0]
        h_idx = h_idx[0]

        term = (
            -(n - h / 4) * y[(h):, nmh_idx]
            + n * y[: nobs - h, n_idx]
            - (h / 4) * y[: nobs - h, h_idx]
        )
        result[: nobs - h, idx] = term

    return result


def predict_returns(
    y: np.ndarray,
    istar: Sequence[float],
    maturities: Sequence[float],
    *,
    h: int,
) -> tuple[float, float]:
    """
    Compute restricted and unrestricted R² metrics analogous to `predictReturns`.
    """
    xr = excess_returns(y, maturities, h=h)
    xr_mean = np.nanmean(xr, axis=1)
    pcs = make_pcs(y)
    istar = np.asarray(istar, dtype=float)

    if pcs.shape[0] != xr_mean.size or istar.size != xr_mean.size:
        raise ValueError("Inputs must share the same number of observations.")

    mask = (
        ~np.isnan(xr_mean)
        & np.all(np.isfinite(pcs), axis=1)
        & np.isfinite(istar)
    )
    if mask.sum() < pcs.shape[1] + 2:
        raise ValueError("Insufficient observations to estimate return regressions.")

    y_dep = xr_mean[mask]
    pcs_masked = pcs[mask]
    istar_masked = istar[mask]

    x1 = add_constant(pcs_masked)
    x2 = add_constant(np.column_stack([pcs_masked, istar_masked]))

    model1 = OLS(y_dep, x1).fit()
    model2 = OLS(y_dep, x2).fit()
    return float(model1.rsquared), float(model2.rsquared)


def log_ratio_norm(x1: Sequence[float], x2: Sequence[float], mu: Sequence[float], sigma: np.ndarray) -> float:
    """
    Log ratio of multivariate normal densities evaluated at `x1` and `x2`.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_inv = inv(sigma)
    term1 = -0.5 * (x1 - mu) @ sigma_inv @ (x1 - mu)
    term2 = -0.5 * (x2 - mu) @ sigma_inv @ (x2 - mu)
    return float(term1 - term2)
