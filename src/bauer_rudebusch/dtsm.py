"""
Dynamic Term Structure Model utilities translated from `R/dtsm_fns.R`.

The implementation covers the components needed to estimate the observed
shifting-endpoint (OSE) model.  Bayesian estimation routines for the ESE model
remain to be ported.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pickle
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import linalg
from scipy.optimize import minimize
from scipy.linalg import null_space

from .util import make_stationary, invalid_cov_mat, get_optim

logger = logging.getLogger(__name__)

SCALE_OSE = np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 10.0, 100.0, 100.0, 10.0, 100.0, 100.0, 0.1])
C_PENAL = 1e6


def affine_loadings(
    mu: np.ndarray,
    phi: np.ndarray,
    omega: np.ndarray,
    delta0: float,
    delta1: np.ndarray,
    mats: Sequence[int],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    mats = np.asarray(mats, dtype=int)
    phi_t = phi.T
    mu_t = mu.reshape(1, -1)
    delta1 = delta1.reshape(-1, 1)
    n_factors = delta1.shape[0]
    n_mats = mats.size
    a = np.zeros((1, n_mats))
    b = np.zeros((n_factors, n_mats))
    atmp = 0.0
    btmp = np.zeros((n_factors, 1))
    idx = 0

    for n in range(1, int(mats.max()) + 1):
        ip1 = float(mu_t @ btmp)
        ip2 = float(0.5 * btmp.T @ omega @ btmp)
        atmp += ip1 + ip2 - delta0
        btmp = phi_t @ btmp - delta1
        if n == mats[idx]:
            a[0, idx] = -atmp / n
            b[:, idx : idx + 1] = -btmp / n
            idx += 1
            if idx >= n_mats:
                break

    return a / dt, b / dt


def affine_loadings_bonly(
    phi: np.ndarray,
    delta1: np.ndarray,
    mats: Sequence[int],
    dt: float,
) -> np.ndarray:
    mats = np.asarray(mats, dtype=int)
    phi_t = phi.T
    delta1 = delta1.reshape(-1, 1)
    n_factors = delta1.shape[0]
    n_mats = mats.size
    b = np.zeros((n_factors, n_mats))
    btmp = np.zeros((n_factors, 1))
    idx = 0

    for n in range(1, int(mats.max()) + 1):
        btmp = phi_t @ btmp - delta1
        if n == mats[idx]:
            b[:, idx : idx + 1] = -btmp / n
            idx += 1
            if idx >= n_mats:
                break

    return b / dt
 

def gaussian_loadings(
    maturities: Sequence[int],
    k0d: np.ndarray,
    k1d: np.ndarray,
    h0d: np.ndarray,
    rho0d: float,
    rho1d: np.ndarray,
    timestep: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    maturities = np.asarray(maturities, dtype=int)
    k0d = np.asarray(k0d, dtype=float).reshape(-1, 1)
    k1d = np.asarray(k1d, dtype=float)
    h0d = np.asarray(h0d, dtype=float)
    rho1d = np.asarray(rho1d, dtype=float).reshape(-1, 1)
    if maturities.ndim != 1:
        raise ValueError("maturities must be one-dimensional")

    max_mat = int(maturities.max())
    n_states = k0d.shape[0]
    n_mats = maturities.size

    ay = np.zeros(n_mats)
    by = np.zeros((n_states, n_mats))
    atmp = 0.0
    btmp = np.zeros((n_states, 1))
    idx = 0
    k0d_t = k0d.T
    k1d_t = k1d.T

    for t in range(1, max_mat + 1):
        atmp = atmp + float(k0d_t @ btmp) + 0.5 * float(btmp.T @ h0d @ btmp) - rho0d
        btmp = btmp + k1d_t @ btmp - rho1d
        if t == maturities[idx]:
            ay[idx] = -atmp / t
            by[:, idx] = (-btmp / t).ravel()
            idx += 1
            if idx >= n_mats:
                break

    return ay / timestep, by / timestep


def fit_var1(series: np.ndarray, intercept: bool) -> dict[str, np.ndarray]:
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError("Input series must be 2-dimensional")
    Y = series[1:]
    X = series[:-1]
    if intercept:
        design = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        design = X
    beta, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
    residuals = Y - design @ beta
    dof = max(Y.shape[0] - design.shape[1], 1)
    sigma = (residuals.T @ residuals) / dof
    if intercept:
        mu = beta[0]
        phi = beta[1:].T
    else:
        mu = np.zeros(series.shape[1])
        phi = beta.T
    return {"Phi": phi, "mu": mu, "sigma": sigma, "residuals": residuals}


def theta2pars_jsz(theta: Sequence[float], N: int) -> dict[str, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    expected = N + N * (N + 1) // 2
    if theta.size != expected:
        raise ValueError("Unexpected theta length for JSZ parameters")
    dlamQ = theta[:N]
    lamQ = np.cumsum(dlamQ + np.concatenate(([1.0], np.zeros(N - 1))))
    L_vals = theta[N:]
    L = np.zeros((N, N))
    L[np.tril_indices(N)] = L_vals
    Omega_cp = L @ L.T
    return {"dlamQ": dlamQ, "lamQ": lamQ, "L": L, "Omega.cP": Omega_cp}


def pars2theta_jsz(pars: dict) -> np.ndarray:
    lamQ = np.asarray(pars["lamQ"], dtype=float)
    dlamQ = np.concatenate(([lamQ[0] - 1.0], np.diff(lamQ)))
    L = np.asarray(pars["L"], dtype=float)
    return np.concatenate([dlamQ, L[np.tril_indices_from(L)]])


def check_pars_jsz(pars: dict) -> bool:
    L = pars["L"]
    Omega = pars["Omega.cP"]
    lamQ = pars["lamQ"]
    dlamQ = pars["dlamQ"]
    if np.any(np.diag(L) < 1e-7):
        return False
    if np.any(np.diag(Omega) < 1e-10):
        return False
    if np.any(lamQ > 1.0) or np.any(dlamQ > 0):
        return False
    return True


def obj_jsz(theta: Sequence[float], Y: np.ndarray, WN: np.ndarray, mats: Sequence[float], dt: float) -> float:
    N = WN.shape[0]
    try:
        pars = theta2pars_jsz(theta, N)
    except ValueError:
        return C_PENAL
    if not check_pars_jsz(pars):
        return C_PENAL
    k1qx = np.diag(pars["lamQ"] - 1.0)
    res = jsz_llk(Y, WN, k1qx, pars["Omega.cP"], mats, dt)
    return float(np.sum(res["llk"]))


def get_starting_values_for_mle(L: np.ndarray, Y: np.ndarray, WN: np.ndarray, mats: Sequence[float], dt: float) -> dict:
    Omega_cp = L @ L.T
    N = WN.shape[0]

    def objective(dlamQ: np.ndarray) -> float:
        lamQ = np.cumsum(dlamQ + np.concatenate(([1.0], np.zeros(N - 1))))
        if np.any(lamQ > 1.0) or np.any(dlamQ > 0) or np.any(lamQ < 0.5):
            return C_PENAL
        res = jsz_llk(Y, WN, np.diag(lamQ - 1.0), Omega_cp, mats, dt)
        return float(np.sum(res["llk"]))

    dlam0 = np.linspace(-0.001, -0.2, N)
    result = minimize(objective, dlam0, method="L-BFGS-B")
    lamQ = np.cumsum(result.x + np.concatenate(([1.0], np.zeros(N - 1))))
    return {"lamQ": lamQ, "L": L}


def estimate_jsz(Y: np.ndarray, est_sample: np.ndarray, WN: np.ndarray, mats: Sequence[float], dt: float) -> dict:
    Y = np.asarray(Y, dtype=float)
    est_idx = np.asarray(est_sample, dtype=bool)
    Y_est = Y[est_idx]
    cP = Y @ WN.T
    N = WN.shape[0]

    var_fit = fit_var1(cP[est_idx], intercept=True)
    L = np.linalg.cholesky(var_fit["sigma"]).T
    start = get_starting_values_for_mle(L, Y_est, WN, mats, dt)
    theta_start = pars2theta_jsz(start)
    opt_theta = get_optim(theta_start, obj_jsz, Y_est, WN, mats, dt, trace=0)
    pars = theta2pars_jsz(opt_theta, N)

    res_llk = jsz_llk(Y_est, WN, np.diag(pars["lamQ"] - 1.0), pars["Omega.cP"], mats, dt)
    lamQ = pars["lamQ"]
    dlamQ = pars["dlamQ"]
    Omega_cp = pars["Omega.cP"]
    Phi = var_fit["Phi"]
    mu = var_fit["mu"]
    PhiQ = res_llk["K1Q.cP"] + np.eye(N)

    return {
        "lamQ": lamQ,
        "dlamQ": dlamQ,
        "Omega.cP": Omega_cp,
        "kinfQ": res_llk["kinfQ"],
        "sigma.e": res_llk["sigma.e"],
        "rho0.cP": res_llk["rho0.cP"],
        "rho1.cP": res_llk["rho1.cP"],
        "K0Q.cP": res_llk["K0Q.cP"],
        "K1Q.cP": res_llk["K1Q.cP"],
        "AcP": res_llk["AcP"],
        "BcP": res_llk["BcP"],
        "Phi": Phi,
        "PhiQ": PhiQ,
        "mu": mu,
        "llk": res_llk["llk"],
    }


def persistence(mod: dict) -> None:
    Phi = mod.get("Phi")
    PhiQ = mod.get("PhiQ")
    if Phi is not None:
        logger.info("P-eigenvalues: %s", np.abs(np.linalg.eigvals(Phi)))
    if PhiQ is not None:
        logger.info("Q-eigenvalues: %s", np.abs(np.linalg.eigvals(PhiQ)))


def estimate_ose(data: pd.DataFrame, output_path: Path) -> dict:
    yield_cols = data.attrs["yield_cols"]
    mats = np.array([float(col[1:]) for col in yield_cols])
    dt = 0.25
    Y = data[yield_cols].to_numpy(dtype=float) / 100.0
    W = np.linalg.eigh(np.cov(Y, rowvar=False))[1]
    N = 3
    WN = W[:, :N].T
    est_sample = data["yyyymm"].to_numpy() < 200800
    logger.info("Estimating JSZ model for starting values")
    mod_jsz = estimate_jsz(Y, est_sample, WN, mats, dt)

    istar = (data["pistar.ptr"] + data["rstar.realtime"]).to_numpy(dtype=float) / 100.0
    logger.info("Computing OSE starting values")
    pars_start = startval_ose(Y[est_sample], istar[est_sample], mod_jsz, WN, mats, dt)
    theta_start = pars2theta_ose(pars_start)
    logger.info("Optimizing OSE likelihood")
    opt_theta = get_optim(theta_start, obj_ose, istar[est_sample], Y[est_sample], WN, mats, dt, trace=1)
    opt_pars = theta2pars_ose(opt_theta, WN, mats, dt)
    res_est = llk_ose(
        Y[est_sample],
        istar[est_sample],
        WN,
        opt_pars["kinfQ"],
        opt_pars["lamQ"],
        opt_pars["p"],
        opt_pars["a"],
        opt_pars["Sigma"],
        opt_pars["sigma.tau"],
        mats,
        dt,
    )
    res_full = llk_ose(
        Y,
        istar,
        WN,
        opt_pars["kinfQ"],
        opt_pars["lamQ"],
        opt_pars["p"],
        opt_pars["a"],
        opt_pars["Sigma"],
        opt_pars["sigma.tau"],
        mats,
        dt,
    )

    mod = {**opt_pars, **res_full}
    nobs = Y.shape[0]
    cP = res_full["cP"]
    mod.update(
        {
            "muQ": res_full["K0Q.cP"],
            "PhiQ": res_full["K1Q.cP"] + np.eye(N),
            "cPstar": np.ones((nobs, 1)) @ mod["Pbar"].reshape(1, -1) + istar.reshape(-1, 1) @ mod["gamma"].reshape(1, -1),
        }
    )
    mod["Ystar"] = np.ones((nobs, 1)) @ mod["AcP"] + mod["cPstar"] @ mod["BcP"]
    mod["Yhat"] = np.ones((nobs, 1)) @ mod["AcP"] + cP @ mod["BcP"]
    mod["Ytilde"] = Y - mod["Ystar"]
    Z = np.column_stack([istar, cP])
    mod["Z"] = Z
    Phi = res_full["Phi"]
    mu_Z = np.concatenate([[0.0], (np.eye(N) - Phi) @ mod["Pbar"]])
    mod["mu.Z"] = mu_Z.reshape(1, -1)
    first_col = np.concatenate([[1.0], (np.eye(N) - Phi) @ mod["gamma"]])
    rest_block = np.vstack([np.zeros((1, N)), Phi])
    PhiZ = np.column_stack([first_col.reshape(-1, 1), rest_block])
    mod["Phi.Z"] = PhiZ
    SigmaZ = np.zeros((N + 1, N + 1))
    SigmaZ[0, 0] = opt_pars["sigma.tau"] ** 2
    SigmaZ[0, 1:] = mod["gamma"] * opt_pars["sigma.tau"] ** 2
    SigmaZ[1:, 0] = mod["gamma"] * opt_pars["sigma.tau"] ** 2
    SigmaZ[1:, 1:] = opt_pars["Sigma"]
    mod["Sigma.Z"] = SigmaZ

    if not check_restrictions(mod["Pbar"], mod["gamma"], mod["rho0.cP"], mod["rho1.cP"]):
        logger.warning("Restrictions not perfectly satisfied for OSE estimates")
    persistence(mod)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump({
            "data": data,
            "mod": mod,
            "mod_jsz": mod_jsz,
            "WN": WN,
            "mats": mats,
            "theta": opt_theta,
        }, fh)
    return mod


def jsz_adjust_k1qx(k1qx: np.ndarray) -> tuple[np.ndarray, int]:
    k1qx = np.asarray(k1qx, dtype=float)
    n = k1qx.shape[0]
    if not k1qx.shape[0] == k1qx.shape[1]:
        raise ValueError("K1Q.X must be square")

    is_diagonal = np.allclose(k1qx, np.diag(np.diag(k1qx)))
    if is_diagonal:
        diag_vals = np.diag(k1qx)
        super_diag = np.zeros(n - 1)
        eps = 1e-3
        for i in range(n - 1):
            diff = abs(diag_vals[i + 1] - diag_vals[i])
            if diff < eps:
                super_diag[i] = eps
                k1qx[i, i + 1] += eps
        if not np.any(super_diag):
            super_diag = np.diag(k1qx, k=1)
    else:
        super_diag = np.diag(k1qx, k=1)

    cumprod = 1.0
    m1 = 1
    for idx, val in enumerate(super_diag, start=1):
        cumprod *= val
        if cumprod > 0:
            m1 = idx + 1
    return k1qx, m1


def jsz_loadings(
    w: np.ndarray,
    k1qx: np.ndarray,
    kinfq: float,
    sigma_cp: np.ndarray,
    mats: Sequence[float],
    dt: float,
) -> dict[str, np.ndarray]:
    mats = np.asarray(mats, dtype=float)
    periods = np.round(mats / dt).astype(int)
    n_factors = k1qx.shape[0]
    rho0d = 0.0
    rho1d = np.ones(n_factors)

    k1qx_adj, m1 = jsz_adjust_k1qx(k1qx)
    phiq = k1qx_adj + np.eye(n_factors)

    bx = affine_loadings_bonly(phiq, rho1d * dt, periods, dt)
    wbxp = w @ bx.T
    wbxp_inv = np.linalg.inv(wbxp)
    wbxp_inv_t = wbxp_inv.T
    sigma_x = wbxp_inv @ sigma_cp @ wbxp_inv_t

    k0qx = np.zeros((n_factors, 1))
    k0qx[m1 - 1, 0] = kinfq
    ax, bx = affine_loadings(k0qx, phiq, sigma_x, rho0d * dt, rho1d * dt, periods, dt)

    waxp = w @ ax.T
    bcp = wbxp_inv_t @ bx
    acp = ax - ax @ (w.T @ wbxp_inv_t @ bx)
    k1q_cp = wbxp @ k1qx_adj @ wbxp_inv
    k0q_cp = wbxp @ k0qx - k1q_cp @ waxp
    rho1_cp = wbxp_inv_t @ np.ones((n_factors, 1))
    rho0_cp = -waxp.T @ rho1_cp

    return {
        "AX": ax,
        "BX": bx,
        "AcP": acp,
        "BcP": bcp,
        "K0Q.cP": k0q_cp,
        "K1Q.cP": k1q_cp,
        "rho0.cP": rho0_cp,
        "rho1.cP": rho1_cp,
        "K0Q.X": k0qx,
        "K1Q.X": k1qx_adj,
        "Sigma.X": sigma_x,
        "m1": m1,
    }


def jsz_loadings_prelim(
    w: np.ndarray,
    k1qx: np.ndarray,
    mats: Sequence[float],
    dt: float,
) -> dict[str, np.ndarray]:
    mats = np.asarray(mats, dtype=float)
    periods = np.round(mats / dt).astype(int)
    n_factors = k1qx.shape[0]
    rho1d = np.ones(n_factors)

    k1qx_adj, _ = jsz_adjust_k1qx(k1qx)
    phiq = k1qx_adj + np.eye(n_factors)
    bx = affine_loadings_bonly(phiq, rho1d * dt, periods, dt)
    wbxp = w @ bx.T
    wbxp_inv = np.linalg.inv(wbxp)
    bcp = wbxp_inv.T @ bx
    rho1_cp = wbxp_inv.T @ np.ones((n_factors, 1))
    return {"BcP": bcp, "rho1.cP": rho1_cp, "BX": bx}


def jsz_loadings_post(
    w: np.ndarray,
    bx: np.ndarray,
    k1qx: np.ndarray,
    kinfq: float,
    sigma_cp: np.ndarray,
    mats: Sequence[float],
    dt: float,
) -> dict[str, np.ndarray]:
    mats = np.asarray(mats, dtype=float)
    periods = np.round(mats / dt).astype(int)
    n_factors = k1qx.shape[0]
    rho0d = 0.0
    rho1d = np.ones(n_factors)

    k1qx_adj, m1 = jsz_adjust_k1qx(k1qx)
    phiq = k1qx_adj + np.eye(n_factors)
    wbxp = w @ bx.T
    wbxp_inv = np.linalg.inv(wbxp)
    sigma_x = wbxp_inv @ sigma_cp @ wbxp_inv.T

    k0qx = np.zeros((n_factors, 1))
    k0qx[m1 - 1, 0] = kinfq
    ax, _ = affine_loadings(k0qx, phiq, sigma_x, rho0d * dt, rho1d * dt, periods, dt)
    waxp = w @ ax.T
    acp = ax - ax @ (w.T @ wbxp_inv.T @ bx)
    rho1_cp = wbxp_inv.T @ np.ones((n_factors, 1))
    rho0_cp = -waxp.T @ rho1_cp
    return {"AX": ax, "AcP": acp, "rho0.cP": rho0_cp}


def jsz_loadings_rho0cp(
    w: np.ndarray,
    k1qx: np.ndarray,
    rho0_cp: float,
    sigma_cp: np.ndarray,
    mats: Sequence[float],
    dt: float,
) -> dict[str, np.ndarray]:
    mats = np.asarray(mats, dtype=float)
    periods = np.round(mats / dt).astype(int)
    n_factors = k1qx.shape[0]
    rho0d = 0.0
    rho1d = np.ones(n_factors)

    k1qx_adj, m1 = jsz_adjust_k1qx(k1qx)
    phiq = k1qx_adj + np.eye(n_factors)
    k0qx = np.zeros((n_factors, 1))
    k0qx[m1 - 1, 0] = 1.0

    zero_cov = np.zeros((n_factors, n_factors))
    alpha0_x, bx = affine_loadings(k0qx, phiq, zero_cov, rho0d * dt, rho1d * dt, periods, dt)
    wbxp = w @ bx.T
    wbxp_inv = np.linalg.inv(wbxp)
    wbxp_inv_t = wbxp_inv.T
    rho1_cp = wbxp_inv_t @ np.ones((n_factors, 1))
    bcp = wbxp_inv_t @ bx
    sigma_x = wbxp_inv @ sigma_cp @ wbxp_inv_t

    ax1, _ = affine_loadings(k0qx, phiq, sigma_x, rho0d * dt, rho1d * dt, periods, dt)
    alpha1_x = ax1 - alpha0_x

    beta0 = -np.ones((1, n_factors)) @ wbxp_inv @ w @ alpha0_x.T
    beta1 = -np.ones((1, n_factors)) @ wbxp_inv @ w @ alpha1_x.T

    c_matrix = np.eye(mats.size if isinstance(mats, np.ndarray) else len(mats)) - bx.T @ wbxp_inv @ w
    alpha0_cp = (c_matrix @ alpha0_x.T).T
    alpha1_cp = (c_matrix @ alpha1_x.T).T

    return {
        "BX": bx,
        "BcP": bcp,
        "alpha0.X": alpha0_x,
        "alpha1.X": alpha1_x,
        "alpha0.cP": alpha0_cp,
        "alpha1.cP": alpha1_cp,
        "K1Q.X": k1qx_adj,
        "Sigma.X": sigma_x,
        "rho1.cP": rho1_cp,
        "m1": m1,
        "beta0": beta0,
        "beta1": beta1,
    }


def jsz_rotation(
    w: np.ndarray,
    k1qx: np.ndarray,
    k0qx: np.ndarray,
    dt: float,
    bx: np.ndarray,
    ax: np.ndarray,
) -> dict[str, np.ndarray]:
    wbxp = w @ bx.T
    wbxp_inv = np.linalg.inv(wbxp)
    wbxp_inv_t = wbxp_inv.T
    k1q_cp = wbxp @ k1qx @ wbxp_inv
    k0q_cp = wbxp @ k0qx - k1q_cp @ (w @ ax.T)
    rho1_cp = wbxp_inv_t @ np.ones((k1qx.shape[0], 1))
    rho0_cp = -(w @ ax.T).T @ rho1_cp
    return {"K0Q.cP": k0q_cp, "K1Q.cP": k1q_cp, "rho0.cP": rho0_cp, "rho1.cP": rho1_cp}


def jsz_llk(
    yields_obs: np.ndarray,
    w: np.ndarray,
    k1qx: np.ndarray,
    sigma_cp: np.ndarray,
    mats: Sequence[float],
    dt: float,
    kinfq: float | None = None,
    k0p_cp: np.ndarray | None = None,
    k1p_cp: np.ndarray | None = None,
    sigma_e: float | None = None,
) -> dict[str, np.ndarray]:
    T = yields_obs.shape[0] - 1
    J = yields_obs.shape[1]
    n_factors = w.shape[0]
    cP = yields_obs @ w.T

    if kinfq is None:
        rho0_target = 0.0
        loads = jsz_loadings_rho0cp(w, k1qx, rho0_target, sigma_cp, mats, dt)
        bx = loads["BX"]
        bcp = loads["BcP"]
        alpha0_x = loads["alpha0.X"]
        alpha1_x = loads["alpha1.X"]
        alpha0_cp = loads["alpha0.cP"]
        alpha1_cp = loads["alpha1.cP"]
        rho1_cp = loads["rho1.cP"]
        beta0 = float(loads["beta0"]) if np.size(loads["beta0"]) == 1 else loads["beta0"]
        beta1 = float(loads["beta1"]) if np.size(loads["beta1"]) == 1 else loads["beta1"]
        v_basis = null_space(w.T)
        if v_basis.size:
            proj = v_basis @ v_basis.T
        else:
            proj = np.zeros((w.shape[1], w.shape[1]))
        average_yields = yields_obs[1:, :].mean(axis=0, keepdims=True)
        average_cp = cP[1:, :].mean(axis=0, keepdims=True)
        lhs = (average_yields - alpha1_cp - average_cp @ bcp.T) @ (proj @ alpha0_cp.T)
        rhs = alpha0_cp @ proj @ alpha0_cp.T
        if np.allclose(rhs, 0):
            kinfq = float((rho0_target - beta1) / beta0)
        else:
            kinfq = float(lhs / rhs)
        ax = alpha0_x * kinfq + alpha1_x
        acp = alpha0_cp * kinfq + alpha1_cp
        k0qx = np.zeros((n_factors, 1))
        k0qx[loads["m1"] - 1, 0] = kinfq
        params = jsz_rotation(w, loads["K1Q.X"], k0qx, dt, bx, ax)
        k0q_cp = params["K0Q.cP"]
        k1q_cp = params["K1Q.cP"]
        rho0_cp = params["rho0.cP"]
        rho1_cp = params["rho1.cP"]
    else:
        loads = jsz_loadings(w, k1qx, kinfq, sigma_cp, mats, dt)
        ax = loads["AX"]
        bx = loads["BX"]
        acp = loads["AcP"]
        bcp = loads["BcP"]
        k0q_cp = loads["K0Q.cP"]
        k1q_cp = loads["K1Q.cP"]
        rho0_cp = loads["rho0.cP"]
        rho1_cp = loads["rho1.cP"]

    yields_model = np.ones((T + 1, 1)) @ acp + cP @ bcp
    yield_errors = yields_obs[1:, :] - yields_model[1:, :]
    if sigma_e is None:
        sigma_e = float(np.sqrt(np.sum(yield_errors**2) / (T * (J - n_factors))))

    llk_q = (
        0.5 * np.sum(yield_errors**2, axis=1) / (sigma_e**2)
        + 0.5 * (J - n_factors) * math.log(2 * math.pi)
        + 0.5 * (J - n_factors) * math.log(sigma_e**2)
    )

    if k0p_cp is None or k1p_cp is None:
        var_fit = fit_var1(cP, intercept=True)
        phi = var_fit["Phi"]
        mu = var_fit["mu"]
        eigvals = np.linalg.eigvals(phi)
        if np.max(np.abs(eigvals)) > 0.99:
            phi = make_stationary(phi)
            mu = (np.eye(n_factors) - phi) @ cP[:-1].mean(axis=0)
        k1p_cp = (phi - np.eye(n_factors))
        k0p_cp = mu.reshape(-1, 1)
    innovations = (
        cP[1:].T
        - (k0p_cp @ np.ones((1, T)) + (k1p_cp + np.eye(n_factors)) @ cP[:-1].T)
    )
    sigma_cp_inv = np.linalg.inv(sigma_cp)
    llk_p = (
        0.5 * n_factors * math.log(2 * math.pi)
        + 0.5 * math.log(np.linalg.det(sigma_cp))
        + 0.5 * np.sum(innovations * (sigma_cp_inv @ innovations), axis=0)
    )

    return {
        "llk": (llk_q + llk_p)[:, None],
        "llkQ": llk_q,
        "llkP": llk_p,
        "AcP": acp,
        "BcP": bcp,
        "cP": cP,
        "K0P.cP": k0p_cp,
        "K1P.cP": k1p_cp,
        "K0Q.cP": k0q_cp,
        "K1Q.cP": k1q_cp,
        "rho0.cP": rho0_cp,
        "rho1.cP": rho1_cp,
        "kinfQ": float(kinfq),
        "sigma.e": sigma_e,
        "AX": ax,
        "BX": bx,
    }


def make_pbar(p: np.ndarray, rho0: np.ndarray, rho1: np.ndarray) -> np.ndarray:
    rho0 = float(np.squeeze(rho0))
    rho1 = np.asarray(rho1, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    last = (-rho0 - np.dot(p, rho1[:-1])) / rho1[-1]
    return np.concatenate([p, [last]])


def make_gamma(a: np.ndarray, rho1: np.ndarray) -> np.ndarray:
    rho1 = np.asarray(rho1, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    last = (1.0 - np.dot(a, rho1[:-1])) / rho1[-1]
    return np.concatenate([a, [last]])


def check_restrictions(p0: np.ndarray, gamma: np.ndarray, rho0: np.ndarray, rho1: np.ndarray) -> bool:
    rho0 = float(np.squeeze(rho0))
    rho1 = np.asarray(rho1, dtype=float).reshape(-1)
    p0 = np.asarray(p0, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    a_val = rho0 + float(np.dot(rho1, p0))
    b_val = float(np.dot(rho1, gamma))
    if not np.isclose(a_val, 0.0):
        logger.debug("Restriction check: rho'Pbar = %s", a_val)
    if not np.isclose(b_val, 1.0):
        logger.debug("Restriction check: rho'gamma = %s", b_val)
    return np.isclose(a_val, 0.0) and np.isclose(b_val, 1.0)


def theta2pars_ose(theta: Sequence[float], WN: np.ndarray, mats: Sequence[float], dt: float) -> dict[str, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    N = WN.shape[0]
    expected_len = 1 + N + (N - 1) + (N - 1) + N * (N + 1) // 2 + 1
    if theta.size != expected_len:
        raise ValueError(f"theta length {theta.size} does not match expected {expected_len}")
    theta_scaled = theta / SCALE_OSE
    idx = 0
    kinfQ = theta_scaled[idx]
    idx += 1
    dlamQ = theta_scaled[idx : idx + N]
    idx += N
    lamQ = np.cumsum(dlamQ + np.concatenate(([1.0], np.zeros(N - 1))))
    p = theta_scaled[idx : idx + (N - 1)]
    idx += N - 1
    a = theta_scaled[idx : idx + (N - 1)]
    idx += N - 1
    lower_count = N * (N + 1) // 2
    L_vals = theta_scaled[idx : idx + lower_count]
    idx += lower_count
    L = np.zeros((N, N))
    tril_idx = np.tril_indices(N)
    L[tril_idx] = L_vals
    Sigma = L @ L.T
    sigma_tau = math.exp(theta_scaled[idx])
    return {
        "kinfQ": kinfQ,
        "dlamQ": dlamQ,
        "lamQ": lamQ,
        "p": p,
        "a": a,
        "Sigma": Sigma,
        "sigma.tau": sigma_tau,
    }


def pars2theta_ose(pars: dict) -> np.ndarray:
    lamQ = np.asarray(pars["lamQ"], dtype=float)
    dlamQ = np.concatenate(([lamQ[0] - 1.0], np.diff(lamQ)))
    p = np.asarray(pars["p"], dtype=float)
    a = np.asarray(pars["a"], dtype=float)
    Sigma = np.asarray(pars["Sigma"], dtype=float)
    L = np.linalg.cholesky(Sigma).T
    tril_idx = np.tril_indices_from(L)
    sigma_tau = float(pars["sigma.tau"])
    theta = np.concatenate(
        [
            np.array([pars["kinfQ"]]),
            dlamQ,
            p,
            a,
            L[tril_idx],
            np.array([math.log(sigma_tau)]),
        ]
    )
    return theta * SCALE_OSE


def check_pars_ose(pars: dict) -> bool:
    kinfQ = pars["kinfQ"]
    lamQ = np.asarray(pars["lamQ"])
    dlamQ = np.asarray(pars["dlamQ"])
    Sigma = np.asarray(pars["Sigma"])
    sigma_tau = float(pars["sigma.tau"])
    if kinfQ < 0:
        return False
    if np.any(lamQ > 1) or np.any(lamQ < 0):
        return False
    if np.any(dlamQ > 0):
        return False
    if np.any(np.diag(Sigma) < np.finfo(float).eps):
        return False
    if invalid_cov_mat(Sigma):
        return False
    if sigma_tau > 0.1:
        return False
    return True


def llk_ose(
    Y: np.ndarray,
    istar: Sequence[float],
    W: np.ndarray,
    kinfQ: float,
    lamQ: Sequence[float],
    p: Sequence[float],
    a: Sequence[float],
    Sigma: np.ndarray,
    sigma_tau: float,
    mats: Sequence[float],
    dt: float,
    sigma_e: float | None = None,
) -> dict[str, np.ndarray]:
    Y = np.asarray(Y, dtype=float)
    istar = np.asarray(istar, dtype=float).reshape(-1, 1)
    W = np.asarray(W, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    lamQ = np.asarray(lamQ, dtype=float)
    T = Y.shape[0] - 1
    J = Y.shape[1]
    N = W.shape[0]
    cP = Y @ W.T

    loads = jsz_loadings(W, np.diag(lamQ - 1.0), kinfQ, Sigma, mats, dt)
    acp = loads["AcP"]
    bcp = loads["BcP"]
    rho0_cp = loads["rho0.cP"]
    rho1_cp = loads["rho1.cP"]

    yields_model = np.ones((T + 1, 1)) @ acp + cP @ bcp
    yield_errors = Y[1:, :] - yields_model[1:, :]
    if sigma_e is None:
        sigma_e = float(np.sqrt(np.sum(yield_errors**2) / (T * (J - N))))
    llkQ = (
        -0.5 * (J - N) * math.log(2 * math.pi * sigma_e**2)
        - 0.5 * np.sum(yield_errors**2, axis=1) / (sigma_e**2)
    )

    gamma = make_gamma(a, rho1_cp)
    Pbar = make_pbar(p, rho0_cp, rho1_cp)
    Sigma_tilde = Sigma - np.outer(gamma, gamma) * sigma_tau**2
    if invalid_cov_mat(Sigma_tilde):
        return {"llk": -C_PENAL}

    Ptilde = cP - np.ones((T + 1, 1)) @ Pbar.reshape(1, -1) - istar @ gamma.reshape(1, -1)
    var_fit = fit_var1(Ptilde, intercept=False)
    Phi = var_fit["Phi"]
    eigenvalues = np.linalg.eigvals(Phi)
    if np.max(np.abs(eigenvalues)) > 0.99:
        Phi = make_stationary(Phi)
    innovations = (Ptilde[1:].T - Phi @ Ptilde[:-1].T)
    sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
    llkP1 = (
        -0.5 * N * math.log(2 * math.pi)
        - 0.5 * math.log(np.linalg.det(Sigma_tilde))
        - 0.5 * np.sum(innovations * (sigma_tilde_inv @ innovations), axis=0)
    )
    diff_istar = np.diff(istar, axis=0).ravel()
    llkP2 = -0.5 * math.log(2 * math.pi * sigma_tau**2) - 0.5 * (diff_istar**2) / (sigma_tau**2)
    llkP = llkP1 + llkP2

    return {
        "llk": (llkQ + llkP)[:, None],
        "llkQ": llkQ,
        "llkP": llkP,
        "Phi": Phi,
        "sigma.e": sigma_e,
        "cP": cP,
        "Ptilde": Ptilde,
        "AcP": acp,
        "BcP": bcp,
        "Pbar": Pbar,
        "gamma": gamma,
        "istar": istar,
        "Sigma.tilde": Sigma_tilde,
        "rho0.cP": rho0_cp,
        "rho1.cP": rho1_cp,
        "K0Q.cP": loads["K0Q.cP"],
        "K1Q.cP": loads["K1Q.cP"],
    }


def obj_ose(theta: Sequence[float], istar: Sequence[float], Y: np.ndarray, WN: np.ndarray, mats: Sequence[float], dt: float) -> float:
    try:
        pars = theta2pars_ose(theta, WN, mats, dt)
    except ValueError:
        return C_PENAL
    if not check_pars_ose(pars):
        return C_PENAL
    res = llk_ose(
        Y,
        istar,
        WN,
        pars["kinfQ"],
        pars["lamQ"],
        pars["p"],
        pars["a"],
        pars["Sigma"],
        pars["sigma.tau"],
        mats,
        dt,
    )
    llk = res.get("llk")
    if isinstance(llk, (int, float)) and llk == -C_PENAL:
        return C_PENAL
    return -float(np.sum(llk))


def startval_ose(
    Y: np.ndarray,
    istar: Sequence[float],
    mod_jsz: dict,
    WN: np.ndarray,
    mats: Sequence[float],
    dt: float,
) -> dict:
    Y = np.asarray(Y, dtype=float)
    istar = np.asarray(istar, dtype=float)
    cP = Y @ WN.T
    nobs = Y.shape[0]
    N = WN.shape[0]
    sigtau2 = np.var(np.diff(istar), ddof=1)

    def objective(theta_vec: np.ndarray) -> float:
        p_vec = theta_vec[: N - 1]
        a_vec = theta_vec[N - 1 :]
        if np.any(np.abs(p_vec) > 1):
            return C_PENAL
        res = llk_ose(
            Y,
            istar,
            WN,
            mod_jsz["kinfQ"],
            mod_jsz["lamQ"],
            p_vec,
            a_vec,
            mod_jsz["Omega.cP"],
            math.sqrt(sigtau2),
            mats,
            dt,
        )
        llk = res.get("llk")
        if isinstance(llk, (int, float)) and llk == -C_PENAL:
            return C_PENAL
        return -float(np.sum(llk))

    gamma_init = WN.sum(axis=1)
    Pbar_init = np.mean(cP - np.outer(istar, gamma_init), axis=0)
    theta_start = np.concatenate([Pbar_init[:-1], gamma_init[:-1]])
    rng = np.random.default_rng(616)
    attempts = 0
    while objective(theta_start) >= C_PENAL and attempts < 10:
        theta_start = theta_start + rng.normal(scale=0.1, size=theta_start.shape)
        attempts += 1

    res = minimize(objective, theta_start, method="L-BFGS-B")
    p_opt = res.x[: N - 1]
    a_opt = res.x[N - 1 :]
    Pbar = make_pbar(p_opt, mod_jsz["rho0.cP"], mod_jsz["rho1.cP"])
    gamma = make_gamma(a_opt, mod_jsz["rho1.cP"])
    Ptilde = cP - np.ones((nobs, 1)) @ Pbar.reshape(1, -1) - istar.reshape(-1, 1) @ gamma.reshape(1, -1)
    var_fit = fit_var1(Ptilde, intercept=False)
    Phi = var_fit["Phi"]
    Sigma_tilde = mod_jsz["Omega.cP"] - np.outer(gamma, gamma) * sigtau2
    if invalid_cov_mat(Sigma_tilde):
        raise ValueError("Sigma_tilde is not positive definite in startval_ose")
    return {
        "kinfQ": mod_jsz["kinfQ"],
        "lamQ": mod_jsz["lamQ"],
        "dlamQ": mod_jsz["dlamQ"],
        "p": p_opt,
        "a": a_opt,
        "Sigma": mod_jsz["Omega.cP"],
        "sigma.tau": math.sqrt(sigtau2),
        "Phi": Phi,
        "Pbar": Pbar,
        "gamma": gamma,
        "Sigma.tilde": Sigma_tilde,
    }
