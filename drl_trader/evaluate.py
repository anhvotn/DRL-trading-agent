"""Backtesting, go-live checks and ensemble weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .env import TradingEnv


# ---------------------------------------------------------------- backtest
def backtest(model, f: pd.DataFrame, obs_cols, window: int, cost_mult: float = 1.0,
             allow_short: bool = True) -> pd.DataFrame:
    """Run the policy deterministically over the whole segment, one pass, no random starts.

    Fast path: the policy's only state is the current position (-1/0/+1), so we precompute its
    action for every bar under each possible position in large batches, then walk forward.
    The result is identical to stepping the env bar by bar."""
    import torch
    env = TradingEnv(f, obs_cols, window=window, episode_len=None, random_start=False,
                     cost_mult=cost_mult, allow_short=allow_short)
    env.reset()
    T = np.arange(window - 1, window - 1 + env.episode_len)
    win = np.lib.stride_tricks.sliding_window_view(env.X, window, axis=0)   # (n-w+1, cols, w)
    flat = win.transpose(0, 2, 1).reshape(win.shape[0], -1)[T - window + 1]
    acts = {}
    with torch.no_grad():
        for p in (-1.0, 0.0, 1.0):
            obs = np.hstack([flat, np.full((len(T), 1), p, np.float32)])
            out = []
            for i in range(0, len(obs), 8192):
                a, _ = model.predict(obs[i:i + 8192], deterministic=True)
                out.append(a)
            acts[p] = np.concatenate(out)
    for i in range(len(T)):
        _, _, done, _, _ = env.step(int(acts[env.pos][i]))
        if done:
            break
    return env.history()


def metrics(h: pd.DataFrame, periods_per_year: int = 252) -> dict:
    net = h["net"]
    eq = net.cumsum()
    sd = net.std()
    return {
        "total_return_%": round(100 * (np.exp(eq.iloc[-1]) - 1), 2) if len(eq) else 0.0,
        "sharpe": round(float(net.mean() / sd * np.sqrt(periods_per_year)), 2) if sd > 0 else 0.0,
        "max_drawdown_%": round(100 * float((np.exp(eq) / np.exp(eq.cummax()) - 1).min()), 2),
        "turnover_per_bar": round(float(h["pos"].diff().abs().mean()), 3),
        "cost_drag_%": round(100 * float(h["cost"].sum()), 2),
        "time_in_market_%": round(100 * float((h["pos"] != 0).mean()), 1),
        "bars": len(h),
    }


def buy_and_hold(f: pd.DataFrame, window: int) -> pd.DataFrame:
    seg = f.iloc[window - 1:-1]
    return pd.DataFrame({"pos": 1.0, "ret": seg["fwd_ret"], "cost": 0.0, "net": seg["fwd_ret"]},
                        index=seg.index)


def by_regime(h: pd.DataFrame, regimes: pd.Series, ppy: int) -> pd.DataFrame:
    lab = regimes.reindex(h.index)
    rows = {r: metrics(h[lab == r], ppy) for r in lab.dropna().unique() if (lab == r).sum() > 20}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- go-live checks
def diagnose(train_sr: float, dev_sr: float, test_sr: float, good: float = 0.5) -> str:
    """Course's three cases mapped to remedies."""
    if train_sr < good * 0.5:
        return ("UNDERFIT: poor on training data. Add inputs, deepen the network, tune PPO/network "
                "hyper-parameters, check weight initialisation.")
    if test_sr < 0 or test_sr < 0.4 * train_sr:
        return ("OVERFIT: good in training, weak/negative on test. Bring in more data, use a shallower "
                "network, or raise dropout.")
    return "OK: decent in train, dev and test. Proceed to holdout, then fine-tune with a small LR and paper trade."


def go_live_checklist(res: dict, ppy: int, min_sharpe: float = 0.5) -> pd.DataFrame:
    """res holds metrics dicts and regime tables produced by run.py."""
    m = res["metrics"]
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})

    add("Data size >= 100k rows", res["data_size"]["ok"], res["data_size"]["note"])
    add("Train / dev / test consistent", diagnose(m["train"]["sharpe"], m["dev"]["sharpe"],
                                                   m["test"]["sharpe"]).startswith("OK"),
        diagnose(m["train"]["sharpe"], m["dev"]["sharpe"], m["test"]["sharpe"]))
    add("Holdout (never seen) Sharpe >= threshold", m["holdout"]["sharpe"] >= min_sharpe,
        f"holdout Sharpe {m['holdout']['sharpe']} vs threshold {min_sharpe}")
    add("Beats buy-and-hold on holdout (risk-adjusted)",
        m["holdout"]["sharpe"] > m["holdout_bh"]["sharpe"],
        f"agent {m['holdout']['sharpe']} vs B&H {m['holdout_bh']['sharpe']}")
    reg = res["regimes_test_holdout"]
    covered = set(reg.index) & {"up", "down", "steady", "volatile"}
    add("Out-of-sample data covers up/down/steady/volatile", len(covered) == 4,
        f"regimes present: {sorted(covered)}")
    neg = [r for r in reg.index if reg.loc[r, "sharpe"] < -0.5]
    add("No regime with badly negative Sharpe", not neg, f"weak regimes: {neg or 'none'}")
    add("Survives 2x transaction costs", m["holdout_2x_cost"]["sharpe"] > 0,
        f"holdout Sharpe at 2x costs: {m['holdout_2x_cost']['sharpe']}")
    if "transfer" in m:
        add("Works on an untrained security, same timestep", m["transfer"]["sharpe"] > 0,
            f"transfer Sharpe {m['transfer']['sharpe']}")
    else:
        add("Works on an untrained security, same timestep", False, "not run: pass --transfer-csv")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ensemble
def ensemble_weights(recent_sharpes: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Higher weight to agents that did better recently; agents with Sharpe <= 0 get zero."""
    names = list(recent_sharpes)
    s = np.array([max(recent_sharpes[n], 0.0) for n in names])
    if s.sum() == 0:
        return {n: 0.0 for n in names}            # nobody works recently -> stay flat
    w = np.exp(s / temperature) * (s > 0)
    w /= w.sum()
    return dict(zip(names, np.round(w, 3)))


def ensemble_action(models: dict, weights: dict, obs) -> int:
    """Weighted vote over buy/sell/hold; 'hold' if no agent has weight."""
    votes = np.zeros(3)
    for n, m in models.items():
        if weights.get(n, 0) > 0:
            a, _ = m.predict(obs, deterministic=True)
            votes[int(a)] += weights[n]
    return int(votes.argmax()) if votes.sum() > 0 else 0
