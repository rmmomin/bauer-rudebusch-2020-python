"""
Predictive regressions for excess bond returns (Table 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .. import data as data_module
from ..util import excess_returns


def _principal_components(yields: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(yields, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvectors = eigenvectors[:, order]
    signs = np.sign(eigenvectors[-1, :])
    eigenvectors *= signs
    pcs = yields @ eigenvectors
    return pcs, eigenvectors


def _fit_ols(y: pd.Series, X: pd.DataFrame) -> OLS:
    X_const = add_constant(X)
    model = OLS(y, X_const).fit(cov_type="HC0")
    return model


def _init_table() -> pd.DataFrame:
    rows = [
        ("label", [
            "PC1", "", "PC2", "", "PC3", "",
            r"$\\pi_t^\\ast$", "", "",
            r"$r_t^\\ast$", "", "",
            r"$i_t^\\ast$", "", "",
            r"$R^2$",
            "Memo: $r^\\ast$",
        ])
    ]
    index = [
        "pc1", "pc1_se",
        "pc2", "pc2_se",
        "pc3", "pc3_se",
        "pi", "pi_se", "pi_boot",
        "r", "r_se", "r_boot",
        "i", "i_se", "i_boot",
        "r2",
        "memo",
    ]
    table = pd.DataFrame(index=index, dtype=object)
    table.insert(0, "label", rows[0][1])
    return table


def _column_names(rstar_desc: Sequence[str]) -> list[str]:
    cols = ["Term structure", "+ pi*"]
    cols.extend([f"+ pi* & {desc}" for desc in rstar_desc])
    cols.append("+ i*")
    return cols


def run_predictive_regressions(output_dir: Path | str = "tables", bootstrap: bool = False) -> dict[str, pd.DataFrame]:
    if bootstrap:
        raise NotImplementedError("Bootstrap inference not yet ported to Python.")

    data = data_module.load_data()
    data["istar"] = data["pistar.ptr"] + data["rstar.realtime"]

    yield_cols = data.attrs["yield_cols"]
    mats = np.array([float(col[1:]) for col in yield_cols])
    Y = data[yield_cols].to_numpy(dtype=float)

    xrn = excess_returns(Y, mats, h=1)
    valid_counts = np.sum(~np.isnan(xrn), axis=1)
    xr_mean = np.full(xrn.shape[0], np.nan)
    mask = valid_counts > 0
    if mask.any():
        xr_sum = np.nansum(xrn, axis=1)
        xr_mean[mask] = xr_sum[mask] / valid_counts[mask]
    data["xr"] = xr_mean

    pcs, weights = _principal_components(Y)
    data["PC1"] = pcs[:, 0]
    data["PC2"] = pcs[:, 1]
    data["PC3"] = pcs[:, 2]

    two_mask = np.isclose(mats, 2.0)
    if not two_mask.any():
        raise ValueError("Two-year maturity not found; cannot scale PC3 as in the R code.")
    sc = np.array([
        weights[:, 0].sum(),
        weights[-1, 1] - weights[0, 1],
        weights[-1, 2] - 2 * weights[two_mask][0, 2] + weights[0, 2],
    ])
    sc = np.where(np.isclose(sc, 0), np.nan, sc)
    scaled = pcs / sc
    data["PC1sc"], data["PC2sc"], data["PC3sc"] = scaled.T

    rstar_vars = ["rstar.filt", "rstar.realtime", "rr.ewma"]
    rstar_desc = ["filtered", "real-time", "mov. avg"]

    results = {}
    for subsample in (False, True):
        label = "post1985" if subsample else "full"
        table = _init_table()
        column_order = ["label"] + _column_names(rstar_desc)
        table = table.reindex(columns=column_order)
        for col in column_order:
            table[col] = table[col].astype(object)
        if subsample:
            mask = data["yyyymm"] >= 198501
        else:
            mask = np.ones(len(data), dtype=bool)

        subset = data.loc[mask].copy()
        subset = subset.dropna(subset=["xr", "PC1sc", "PC2sc", "PC3sc", "pistar.ptr"] + rstar_vars + ["istar"])
        if subset.empty:
            results[label] = table
            continue

        def record(column: str, model: OLS, coeff_keys: Sequence[str]) -> None:
            params = model.params
            bse = model.bse
            for key in coeff_keys:
                if key.startswith("PC1"):
                    row, row_se = "pc1", "pc1_se"
                elif key.startswith("PC2"):
                    row, row_se = "pc2", "pc2_se"
                elif key.startswith("PC3"):
                    row, row_se = "pc3", "pc3_se"
                elif key.startswith("pistar"):
                    row, row_se = "pi", "pi_se"
                elif key == "istar":
                    row, row_se = "i", "i_se"
                else:
                    row, row_se = "r", "r_se"
                table.at[row, column] = f"{params[key]:4.2f}"
                table.at[row_se, column] = f"({bse[key]:4.2f})"
            table.at["r2", column] = f"{model.rsquared:4.2f}"

        # Term structure only
        model_ts = _fit_ols(subset["xr"], subset[["PC1sc", "PC2sc", "PC3sc"]])
        record("Term structure", model_ts, ["PC1sc", "PC2sc", "PC3sc"])

        # Add pi*
        model_pi = _fit_ols(subset["xr"], subset[["PC1sc", "PC2sc", "PC3sc", "pistar.ptr"]])
        record("+ pi*", model_pi, ["PC1sc", "PC2sc", "PC3sc", "pistar.ptr"])

        # Add r* variants
        for var, desc in zip(rstar_vars, rstar_desc):
            col_name = f"+ pi* & {desc}"
            model_r = _fit_ols(subset["xr"], subset[["PC1sc", "PC2sc", "PC3sc", "pistar.ptr", var]])
            record(col_name, model_r, ["PC1sc", "PC2sc", "PC3sc", "pistar.ptr", var])
            table.at["memo", col_name] = desc

        # i*
        model_i = _fit_ols(subset["xr"], subset[["PC1sc", "PC2sc", "PC3sc", "istar"]])
        record("+ i*", model_i, ["PC1sc", "PC2sc", "PC3sc", "istar"])
        table.at["memo", "+ i*"] = "real-time"

        results[label] = table

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_subsample" if subsample else ""
        csv_path = output_dir / f"returns{suffix}.csv"
        tex_path = output_dir / f"returns{suffix}.tex"
        table.reset_index(drop=True).to_csv(csv_path, index=False)
        table.reset_index(drop=True).to_latex(tex_path, index=False, escape=False)

    return results
