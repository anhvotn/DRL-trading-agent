"""End-to-end pipeline.

    python -m drl_trader.run --demo                       # synthetic data, quick smoke test
    python -m drl_trader.run --csv spy_15m.csv --ppy 6552 # your data (ppy = bars per year)
    python -m drl_trader.run --csv spy_15m.csv --transfer-csv qqq_15m.csv --ppy 6552

Steps: load -> clean -> costs -> features -> splits -> train N agents (different windows/seeds)
-> weight them by dev Sharpe -> test -> holdout -> regimes -> transfer -> go-live checklist.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import data as D
from .agent import build_agent, train_two_stage
from .env import TradingEnv
from .evaluate import (backtest, buy_and_hold, by_regime, ensemble_weights, go_live_checklist,
                       metrics)


def prepare(df, trade_notional, splits=None, norm_ref=None):
    df, clean_rep = D.clean(df)
    cost = D.estimate_costs(df, trade_notional)
    f, price_cols, sent_cols = D.make_features(df, cost)
    obs_cols = price_cols + sent_cols
    return f, obs_cols, price_cols, sent_cols, clean_rep


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv"); p.add_argument("--transfer-csv")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--ppy", type=int, default=252, help="bars per year (daily 252, 15-min US equity 6552)")
    p.add_argument("--notional", type=float, default=100_000)
    p.add_argument("--windows", default="20,40", help="one agent per window size")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--steps-pnl", type=int, default=60_000)
    p.add_argument("--steps-sharpe", type=int, default=30_000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--out", default="outputs")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # ---- 1. data
    if a.demo:
        from .synthetic import make_market
        raw = make_market(n=120_000, seed=0).set_index("timestamp"); raw.index.name = "timestamp"
        raw_t = make_market(n=40_000, seed=7).set_index("timestamp")
        a.ppy = 6552
    else:
        raw = D.load_csv(a.csv)
        raw_t = D.load_csv(a.transfer_csv) if a.transfer_csv else None
    size = D.check_data_size(raw)
    print("data:", size)

    f, obs_cols, price_cols, sent_cols, clean_rep = prepare(raw, a.notional)
    print("cleaning:", clean_rep, "| sentiment cols:", sent_cols)
    sp = D.chronological_splits(f)
    sp["train"], sp["dev"], sp["test"], sp["holdout"] = D.normalize(
        sp["train"], [sp["dev"], sp["test"], sp["holdout"]], obs_cols)
    regimes = D.label_regimes(f)
    groups = [list(range(len(price_cols)))]
    if sent_cols:
        groups.append(list(range(len(price_cols), len(obs_cols))))
    short = not a.no_short

    # ---- 2. train several agents that behave differently
    agents = {}
    for w in [int(x) for x in a.windows.split(",")]:
        for s in range(a.seeds):
            name = f"w{w}_s{s}"
            mk = lambda reward: (lambda seed: TradingEnv(sp["train"], obs_cols, window=w, reward=reward,
                                                          allow_short=short, seed=seed))
            model = build_agent(mk("pnl"), w, groups, dropout=a.dropout, seed=s)
            print(f"training {name} ...", flush=True)
            train_two_stage(model, mk("sharpe"), a.steps_pnl, a.steps_sharpe, seed=s)
            agents[name] = (model, w)

    # ---- 3. evaluate each agent; weight by dev performance (the "recent past" before test)
    per_agent, dev_sr = {}, {}
    for name, (m, w) in agents.items():
        r = {k: metrics(backtest(m, sp[k], obs_cols, w, allow_short=short), a.ppy)
             for k in ["train", "dev", "test"]}
        per_agent[name] = r
        dev_sr[name] = r["dev"]["sharpe"]
    weights = ensemble_weights(dev_sr)
    best = max(dev_sr, key=dev_sr.get)
    print("dev Sharpe:", dev_sr, "-> weights:", weights, "| best:", best)

    # ---- 4. the selected agent on all splits, holdout, cost stress, regimes, transfer
    m, w = agents[best]
    hist = {k: backtest(m, sp[k], obs_cols, w, allow_short=short) for k in sp}
    M = {k: metrics(h, a.ppy) for k, h in hist.items()}
    M["holdout_bh"] = metrics(buy_and_hold(sp["holdout"], w), a.ppy)
    M["holdout_2x_cost"] = metrics(backtest(m, sp["holdout"], obs_cols, w, cost_mult=2.0,
                                            allow_short=short), a.ppy)
    oos = pd.concat([hist["test"], hist["holdout"]])
    reg_tbl = by_regime(oos, regimes, a.ppy)

    if raw_t is not None:
        ft, *_ = prepare(raw_t, a.notional)
        missing = [c for c in obs_cols if c not in ft.columns]
        for c in missing:
            ft[c] = 0.0
        # normalise with the new security's own early history, evaluate on its later 40%
        cut = int(len(ft) * 0.6)
        ft = D.normalize(ft.iloc[:cut], [ft], obs_cols)[1]
        M["transfer"] = metrics(backtest(m, ft.iloc[int(len(ft) * 0.6):], obs_cols, w, allow_short=short), a.ppy)

    res = {"data_size": size, "metrics": M, "regimes_test_holdout": reg_tbl}
    checks = go_live_checklist(res, a.ppy)

    # ---- 5. outputs
    m.save(os.path.join(a.out, f"agent_{best}"))
    pd.DataFrame(M).T.to_csv(os.path.join(a.out, "metrics.csv"))
    reg_tbl.to_csv(os.path.join(a.out, "regimes_oos.csv"))
    checks.to_csv(os.path.join(a.out, "go_live_checklist.csv"), index=False)
    with open(os.path.join(a.out, "agents.json"), "w") as fh:
        json.dump({"per_agent": per_agent, "dev_sharpe": dev_sr, "ensemble_weights": weights,
                   "selected": best, "cleaning": clean_rep}, fh, indent=2, default=float)
    _plot(hist, sp, w, a.out)

    pd.set_option("display.width", 160)
    print("\n== metrics (selected agent) ==\n", pd.DataFrame(M).T)
    print("\n== out-of-sample by regime ==\n", reg_tbl)
    print("\n== go-live checklist ==\n", checks.to_string(index=False))


def _plot(hist, sp, w, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    for ax, k in zip(axes, ["train", "dev", "test", "holdout"]):
        eq = hist[k]["net"].cumsum() * 100
        bh = buy_and_hold(sp[k], w)["net"].cumsum() * 100
        ax.plot(eq.index, eq, color="#4c78a8", lw=1.3, label="agent (net of costs)")
        ax.plot(bh.index, bh, color="#9aa5b1", lw=1.0, ls="--", label="buy & hold")
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.set_title(k + (" (never seen)" if k == "holdout" else ""), fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7); ax.grid(alpha=0.2)
    axes[0].set_ylabel("cumulative log return, %")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "equity.png"), dpi=130)


if __name__ == "__main__":
    main()
