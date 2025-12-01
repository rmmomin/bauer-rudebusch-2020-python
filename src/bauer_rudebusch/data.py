"""
Data loading and transformation helpers translated from `R/data_fns.R`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DATA_ROOT = Path("data")


def make_quarterly(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Subset a DataFrame to end-of-quarter observations.
    """
    if "date" not in frame.columns:
        raise ValueError("DataFrame must include a `date` column.")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    months = result["date"].dt.month
    if months.isna().any():
        raise ValueError("Encountered invalid month while filtering for quarter-end data.")
    return result.loc[months % 3 == 0].reset_index(drop=True)


def matlab_to_yyyymm(date_values: Sequence[float]) -> np.ndarray:
    """
    Convert Matlab serial quarterly dates (e.g., 2012.75) into integer YYYYMM codes.
    """
    date_values = np.asarray(date_values, dtype=float)
    years = np.floor(date_values).astype(int)
    fractions = date_values - years
    quarters = np.rint(fractions * 4).astype(int) + 1
    months = quarters * 3
    return years * 100 + months


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file missing: {path}")
    return pd.read_csv(path, **kwargs)


def load_lw(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    frame = _load_csv(data_dir / "rstar" / "lw.csv", na_values=["#N/A"])
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    frame["yyyymm"] = frame["Date"].dt.strftime("%Y%m").astype(int)
    frame = frame.rename(columns={frame.columns[1]: "rstar.lw", frame.columns[2]: "rstar.lw.sm"})
    return frame[["yyyymm", "rstar.lw", "rstar.lw.sm"]]


def load_hlw(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    frame = _load_csv(data_dir / "rstar" / "hlw.csv")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    frame["yyyymm"] = frame["Date"].dt.strftime("%Y%m").astype(int) + 2
    frame = frame.rename(columns={frame.columns[1]: "rstar.hlw"})
    return frame[["yyyymm", "rstar.hlw"]]


def load_kiley(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    frame = _load_csv(data_dir / "rstar" / "kiley.csv", header=None)
    frame.columns = ["date", "rstar.kiley.sm", "rstar.kiley"]
    frame["yyyymm"] = matlab_to_yyyymm(frame["date"].to_numpy())
    return frame[["yyyymm", "rstar.kiley", "rstar.kiley.sm"]]


def load_jm(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    jm = _load_csv(data_dir / "rstar" / "jm_realtime.csv")
    first = jm.columns[0]
    jm = jm.rename(columns={first: "rstar.jm"})
    jm["yyyymm"] = matlab_to_yyyymm(jm["Date"].to_numpy())
    jm = jm[["yyyymm", "rstar.jm"]]

    jm_sm = _load_csv(data_dir / "rstar" / "jm_smoothed.csv")
    first_sm = jm_sm.columns[0]
    jm_sm = jm_sm.rename(columns={first_sm: "rstar.jm.sm"})
    jm_sm["yyyymm"] = matlab_to_yyyymm(jm_sm["Date"].to_numpy())
    jm = jm.merge(jm_sm[["yyyymm", "rstar.jm.sm"]], on="yyyymm", how="left")
    return jm


def load_ptr(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    frame = _load_csv(data_dir / "pistar_PTR.csv")
    frame["yyyymm"] = frame["Year"] * 100 + frame["Quarter"] * 3
    frame = frame.rename(columns={"pistar_PTR": "pistar.ptr"})
    return frame[["yyyymm", "pistar.ptr"]]


def load_pcepi(
    *,
    core: bool = True,
    yoy: bool = True,
    data_dir: Path = DATA_ROOT,
) -> pd.DataFrame:
    filename = "PCEPILFE.csv" if core else "PCEPI.csv"
    frame = _load_csv(data_dir / filename)
    frame.columns = ["date", "pce"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = make_quarterly(frame)
    frame["yyyymm"] = frame["date"].dt.strftime("%Y%m").astype(int)
    log_pce = np.log(frame["pce"])
    if yoy:
        frame["pi"] = 100 * log_pce.diff(4)
    else:
        frame["pi"] = 400 * log_pce.diff()
    frame = frame.dropna(subset=["pi"])
    return frame[["date", "yyyymm", "pi"]]


def load_tb(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    frame = _load_csv(data_dir / "TB3MS.csv")
    frame.columns = ["date", "tb3m"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = make_quarterly(frame)
    frame["yyyymm"] = frame["date"].dt.strftime("%Y%m").astype(int)
    frame = frame.loc[frame["yyyymm"] >= 195201]
    return frame.reset_index(drop=True)


def ewma(values: Sequence[float], v: float = 0.95, y0: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    tau = np.zeros_like(values)
    tau[0] = values[0] if y0 is None else y0
    for t in range(1, len(values)):
        tau[t] = tau[t - 1] + (1 - v) * (values[t] - tau[t - 1])
    return tau


def load_eprr(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    inflation = load_pcepi(core=True, yoy=True, data_dir=data_dir)
    tb = load_tb(data_dir=data_dir)
    merged = inflation.merge(tb[["yyyymm", "tb3m"]], on="yyyymm", how="left")
    merged["rr"] = merged["tb3m"] - merged["pi"]
    merged = merged.dropna(subset=["rr"])
    return merged.reset_index(drop=True)


def load_data(
    *,
    start: int = 197112,
    end: int = 201803,
    data_dir: Path = DATA_ROOT,
) -> pd.DataFrame:
    yields = _load_csv(data_dir / "yields.csv")
    yield_cols = [col for col in yields.columns if col != "Date"]

    yields["yyyymm"] = (yields["Date"] // 100).astype(int)
    yields["year"] = yields["yyyymm"] // 100
    yields["month"] = yields["yyyymm"] - yields["year"] * 100
    yields["date"] = pd.to_datetime(yields["Date"].astype(str), format="%Y%m%d")
    yields = yields.loc[(yields["yyyymm"] <= end) & (yields["yyyymm"] >= start)].reset_index(drop=True)

    y_numeric = yields[yield_cols].astype(float)
    cov_matrix = y_numeric.cov().to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    w = eigenvectors[:, -1]
    yields["level"] = (y_numeric.to_numpy() @ w) / np.sum(w)

    data = yields.copy()
    data = data.merge(load_ptr(data_dir=data_dir), on="yyyymm", how="left")
    data = data.merge(load_lw(data_dir=data_dir), on="yyyymm", how="left")
    data = data.merge(load_hlw(data_dir=data_dir), on="yyyymm", how="left")
    data = data.merge(load_kiley(data_dir=data_dir), on="yyyymm", how="left")
    data = data.merge(load_jm(data_dir=data_dir), on="yyyymm", how="left")

    rr = load_eprr(data_dir=data_dir)
    rr = rr.loc[rr["yyyymm"] >= 196111].copy()
    rr["rr.ewma"] = ewma(rr["rr"].to_numpy(), v=0.98)
    data = data.merge(rr[["yyyymm", "rr", "rr.ewma"]], on="yyyymm", how="left")

    dn_rt = _load_csv(data_dir / "rstar" / "delnegro_realtime.csv")
    dn_rt.columns = ["yyyymm", "rstar.dn", "X", "Y"]
    data = data.merge(dn_rt[["yyyymm", "rstar.dn"]], on="yyyymm", how="left")

    dn = _load_csv(data_dir / "rstar" / "delnegro.csv")
    dn.columns = ["yyyymm", "rstar.dn.sm", "rstar.dn.lb", "rstar.dn.ub"]
    data = data.merge(dn[["yyyymm", "rstar.dn.sm"]], on="yyyymm", how="left")

    proxies_rt = _load_csv(data_dir / "rstar" / "proxies_realtime.csv")
    proxies_rt["rstar.proxies"] = proxies_rt["rstar"]
    data = data.merge(proxies_rt[["yyyymm", "rstar.proxies"]], on="yyyymm", how="left")

    proxies = _load_csv(data_dir / "rstar" / "proxies.csv")
    proxies["rstar.proxies.sm"] = proxies["rstar"]
    data = data.merge(proxies[["yyyymm", "rstar.proxies.sm"]], on="yyyymm", how="left")

    uc_rt = _load_csv(data_dir / "rstar" / "uc_realtime.csv")
    uc_rt["rstar.uc"] = uc_rt["rstar"]
    data = data.merge(uc_rt[["yyyymm", "rstar.uc"]], on="yyyymm", how="left")

    uc = _load_csv(data_dir / "rstar" / "uc.csv")
    uc["rstar.uc.sm"] = uc["rstar.mcmc"]
    data = data.merge(uc[["yyyymm", "rstar.uc.sm"]], on="yyyymm", how="left")

    ssm = _load_csv(data_dir / "rstar" / "ssm.csv")
    ssm["rstar.ssm.sm"] = ssm["rstar"]
    ssm["pistar.ssm.sm"] = ssm["pistar"]
    data = data.merge(ssm[["yyyymm", "rstar.ssm.sm", "pistar.ssm.sm"]], on="yyyymm", how="left")

    ssm_rt = _load_csv(data_dir / "rstar" / "ssm_realtime.csv")
    ssm_rt["rstar.ssm"] = ssm_rt["rstar"]
    ssm_rt["pistar.ssm"] = ssm_rt["pistar"]
    data = data.merge(ssm_rt[["yyyymm", "rstar.ssm", "pistar.ssm"]], on="yyyymm", how="left")

    nobs = len(data)
    realtime_names = ["rstar.dn", "rstar.uc", "rstar.proxies", "rstar.ssm", "pistar.ssm"]
    for name in realtime_names:
        smoothed = f"{name}.sm"
        if smoothed in data.columns:
            data.loc[nobs - 1, name] = data.loc[nobs - 1, smoothed]

    nms_realtime = ["rstar.jm", "rstar.dn", "rstar.uc", "rstar.proxies", "rstar.ssm", "rr.ewma"]
    data["rstar.realtime"] = data[nms_realtime].mean(axis=1, skipna=True)

    nms_filt = ["rstar.lw", "rstar.kiley", "rstar.hlw"]
    data["rstar.filt"] = data[nms_filt].mean(axis=1, skipna=True)

    nms_all = sorted(set(nms_filt + nms_realtime))
    data["rstar.mean"] = data[nms_all].mean(axis=1, skipna=True)

    nms_sm = [
        "rstar.lw.sm",
        "rstar.jm.sm",
        "rstar.dn.sm",
        "rstar.kiley.sm",
        "rstar.uc.sm",
        "rstar.proxies.sm",
        "rstar.ssm.sm",
    ]
    data["rstar.sm"] = data[nms_sm].mean(axis=1, skipna=True)

    data.attrs.update(
        {
            "yield_cols": yield_cols,
            "nms.realtime": nms_realtime,
            "nms.filt": nms_filt,
            "nms.sm": nms_sm,
            "nms.all": nms_all,
        }
    )

    if not np.all(data["month"] % 3 == 0):
        raise ValueError("Yield data must be quarterly.")

    return data.reset_index(drop=True)


def load_bluechip(data_dir: Path = DATA_ROOT) -> pd.DataFrame:
    frame = _load_csv(data_dir / "bluechip_10y.csv", header=0)
    year_cols = [f"year{i}" for i in range(1, 6)]
    forecast_cols = [f"f{i}" for i in range(1, 6)]
    frame.columns = ["Date", *year_cols, *forecast_cols, "lr"]
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    frame["year"] = frame["Date"].dt.year
    frame["month"] = frame["Date"].dt.month
    frame["yyyymm"] = frame["year"] * 100 + frame["month"]
    return frame
