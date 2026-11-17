"""Data loading, cleaning, features, slippage estimation, splits and regime labels.

Course steps covered:
  * "the more data the better" (>=100k rows)        -> check_data_size()
  * clean raw data, simple missing-value handling    -> clean()
  * model slippage / market impact                   -> estimate_costs()
  * out-of-sample data the agent has never seen      -> chronological_splits()
  * cover volatile / steady / up / down markets      -> label_regimes()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["ret_1", "ret_5", "ret_20", "vol_20", "rsi_14", "z_20", "vol_ratio", "spread_bps"]


# ---------------------------------------------------------------- loading
def load_csv(path: str) -> pd.DataFrame:
    """Expected columns: timestamp, open, high, low, close, volume.
    Optional: bid, ask (best for slippage), sentiment* (any column starting with 'sentiment')."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts])
    df = df.set_index(ts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def check_data_size(df: pd.DataFrame, minimum: int = 100_000) -> dict:
    n = len(df)
    return {"rows": n, "ok": n >= minimum,
            "note": "OK" if n >= minimum else
            f"Only {n:,} rows; course guideline is >= {minimum:,}. Treat backtest results with caution."}


# ---------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame, max_gap: int = 5, max_missing_frac: float = 0.2,
          period: str = "D", fill: str = "ffill") -> tuple[pd.DataFrame, dict]:
    """'Simpler is better': interpolate short gaps, drop periods with too much missing data.

    max_gap          -- longest run of consecutive NaNs that is interpolated
    max_missing_frac -- if a period (e.g. a day) has more missing than this, drop the whole period
    """
    report = {"rows_in": len(df), "nan_cells_in": int(df.isna().sum().sum())}
    num = df.select_dtypes("number").columns

    # 1) drop whole periods whose missing fraction is too high
    miss = df[num].isna().mean(axis=1).groupby(df.index.to_period(period)).mean()
    bad = miss[miss > max_missing_frac].index
    keep = ~df.index.to_period(period).isin(bad)
    df = df.loc[keep].copy()
    report["periods_dropped"] = len(bad)

    # 2) fill short gaps only (limit), then drop anything still missing.
    #    NOTE: linear interpolation uses the *next* observation -> look-ahead bias in a backtest.
    #    Default is forward-fill (only past data); use fill="interpolate" for research plots only.
    if fill == "interpolate":
        df[num] = df[num].interpolate(method="linear", limit=max_gap, limit_area="inside")
    else:
        df[num] = df[num].ffill(limit=max_gap)
    df = df.dropna(subset=["close"])
    df[num] = df[num].ffill()
    df = df.dropna()

    # 3) basic sanity: positive prices, high>=low
    df = df[(df["close"] > 0)]
    if {"high", "low"} <= set(df.columns):
        df = df[df["high"] >= df["low"]]
    report["rows_out"] = len(df)
    return df, report


# ---------------------------------------------------------------- costs
def estimate_costs(df: pd.DataFrame, trade_notional: float, impact_k: float = 0.1,
                   default_spread_bps: float = 1.0) -> pd.Series:
    """Per-trade one-way cost as a fraction of price.

    If bid/ask exist: half-spread (what a market order pays) + square-root impact.
    Otherwise infer from trade size, volume and volatility, as the course suggests:
        impact = k * sigma * sqrt(trade_size / volume)
    """
    if {"bid", "ask"} <= set(df.columns):
        mid = (df["bid"] + df["ask"]) / 2
        half_spread = ((df["ask"] - df["bid"]) / mid / 2).clip(lower=0)
    else:
        half_spread = pd.Series(default_spread_bps / 1e4 / 2, index=df.index)

    sigma = np.log(df["close"]).diff().rolling(20, min_periods=5).std().bfill()
    dollar_vol = (df["close"] * df.get("volume", pd.Series(np.nan, index=df.index))).replace(0, np.nan)
    participation = (trade_notional / dollar_vol).fillna(0.01).clip(upper=1.0)
    impact = impact_k * sigma * np.sqrt(participation)
    return (half_spread + impact).fillna(half_spread.median())


# ---------------------------------------------------------------- features
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(df: pd.DataFrame, cost: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (features, price_cols, sentiment_cols). All features use only past data."""
    c = df["close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = lr
    f["ret_5"] = lr.rolling(5).sum()
    f["ret_20"] = lr.rolling(20).sum()
    f["vol_20"] = lr.rolling(20).std()
    f["rsi_14"] = (_rsi(c) - 50) / 50
    f["z_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    if "volume" in df.columns:
        f["vol_ratio"] = np.log1p(df["volume"]) - np.log1p(df["volume"]).rolling(20).mean()
    else:
        f["vol_ratio"] = 0.0
    f["spread_bps"] = cost * 1e4

    sent_cols = [col for col in df.columns if col.startswith("sentiment")]
    for col in sent_cols:
        f[col] = df[col]

    f["fwd_ret"] = lr.shift(-1)          # target for the env reward only, never an observation
    f["cost"] = cost
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f, PRICE_FEATURES.copy(), sent_cols


def normalize(train: pd.DataFrame, others: list[pd.DataFrame], cols: list[str]):
    """Fit z-score on the training split only (no look-ahead), clip outliers."""
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    out = []
    for d in [train, *others]:
        d = d.copy()
        d[cols] = ((d[cols] - mu) / sd).clip(-5, 5)
        out.append(d)
    return out


# ---------------------------------------------------------------- splits & regimes
def chronological_splits(f: pd.DataFrame, fracs=(0.60, 0.15, 0.15, 0.10)):
    """train / dev / test / holdout. Holdout = most recent data, never touched until the final check."""
    n = len(f)
    cuts = [0, *np.cumsum([int(n * x) for x in fracs[:-1]]), n]
    parts = [f.iloc[cuts[i]:cuts[i + 1]] for i in range(4)]
    return dict(zip(["train", "dev", "test", "holdout"], parts))


def label_regimes(f: pd.DataFrame, trend_win: int = 60, vol_win: int = 60) -> pd.Series:
    """Label each bar: up / down / steady (trend) x volatile flag."""
    trend = f["ret_1"].rolling(trend_win).sum()
    vol = f["ret_1"].rolling(vol_win).std()
    hi_vol = vol > vol.expanding(vol_win).quantile(0.75)
    t_thr = vol * np.sqrt(trend_win) * 0.5
    lab = np.where(trend > t_thr, "up", np.where(trend < -t_thr, "down", "steady"))
    lab = pd.Series(lab, index=f.index)
    lab[hi_vol] = "volatile"
    return lab.where(trend.notna(), "unknown")
