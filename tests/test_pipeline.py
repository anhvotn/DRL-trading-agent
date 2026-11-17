"""Fast checks for the parts of the pipeline where silent bugs would fake a good backtest.

Run:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from drl_trader import data as D
from drl_trader.env import BUY, HOLD, SELL, TradingEnv
from drl_trader.evaluate import backtest, ensemble_weights
from drl_trader.synthetic import make_market


@pytest.fixture(scope="module")
def market():
    raw = make_market(n=4000, seed=1, missing_frac=0.0).set_index("timestamp")
    df, _ = D.clean(raw)
    cost = D.estimate_costs(df, 1e5)
    f, price_cols, sent_cols = D.make_features(df, cost)
    return df, f, price_cols + sent_cols


# ---------------------------------------------------------------- data
def test_features_have_no_look_ahead(market):
    """Features at time t must not change when data after t is removed."""
    df, f_full, cols = market
    cut = 3000
    df_cut = df.iloc[:cut]
    f_cut, *_ = D.make_features(df_cut, D.estimate_costs(df_cut, 1e5))
    common = f_cut.index.intersection(f_full.index)
    pd.testing.assert_frame_equal(f_cut.loc[common, cols], f_full.loc[common, cols])


def test_clean_fills_only_from_the_past():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({"close": [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10.0], "volume": 1.0}, index=idx)
    out, _ = D.clean(df)
    assert out.loc[idx[3], "close"] == 3.0          # previous value, not the interpolated 4.0


def test_clean_drops_periods_with_too_much_missing_data():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    close = np.arange(48, dtype=float) + 1
    close[24:40] = np.nan                            # day 2 is 2/3 missing
    out, rep = D.clean(pd.DataFrame({"close": close, "volume": 1.0}, index=idx))
    assert rep["periods_dropped"] == 1
    assert (out.index.date == idx[0].date()).all()


def test_splits_are_chronological_and_disjoint(market):
    _, f, _ = market
    sp = D.chronological_splits(f)
    parts = [sp[k] for k in ["train", "dev", "test", "holdout"]]
    assert sum(len(p) for p in parts) == len(f)
    for a, b in zip(parts, parts[1:]):
        assert a.index.max() < b.index.min()


def test_normalization_uses_train_statistics_only(market):
    _, f, cols = market
    sp = D.chronological_splits(f)
    tr, te = D.normalize(sp["train"], [sp["test"]], cols)
    assert np.allclose(tr[cols].mean().abs().max(), 0, atol=0.05)
    raw_te = sp["test"][cols]
    expected = ((raw_te - sp["train"][cols].mean()) / sp["train"][cols].std()).clip(-5, 5)
    pd.testing.assert_frame_equal(te[cols], expected)


# ---------------------------------------------------------------- environment
def test_costs_charged_only_when_position_changes(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=10)
    env.reset()
    t0 = env.t
    _, _, _, _, info = env.step(BUY)                 # flat -> long: pays 1x cost
    assert info["cost"] == pytest.approx(f["cost"].iloc[t0])
    _, _, _, _, info = env.step(HOLD)                # no trade: no cost
    assert info["cost"] == 0.0
    _, _, _, _, info = env.step(SELL)                # long -> short: pays 2x cost
    assert info["cost"] == pytest.approx(2 * f["cost"].iloc[t0 + 2])


def test_long_only_mode_never_goes_short(market):
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=50, allow_short=False)
    env.reset()
    for a in [SELL, BUY, SELL, SELL]:
        env.step(a)
    assert env.history()["pos"].min() >= 0


def test_reward_uses_next_bar_return(market):
    """Position chosen at t earns the return from t to t+1, never the return already observed."""
    _, f, cols = market
    env = TradingEnv(f, cols, window=20, random_start=False, episode_len=5, cost_mult=0.0)
    env.reset()
    t0 = env.t
    _, _, _, _, info = env.step(BUY)
    assert info["net"] == pytest.approx(f["fwd_ret"].iloc[t0])
    assert f["fwd_ret"].iloc[t0] == pytest.approx(f["ret_1"].iloc[t0 + 1])


# ---------------------------------------------------------------- evaluation
def test_fast_backtest_matches_step_by_step(market):
    from drl_trader.agent import build_agent
    _, f, cols = market
    f = D.normalize(f, [], cols)[0]
    groups = [list(range(8)), list(range(8, len(cols)))]
    model = build_agent(lambda s: TradingEnv(f, cols, window=20, seed=s), 20, groups, n_envs=1)
    model.learn(512)

    fast = backtest(model, f, cols, window=20)

    env = TradingEnv(f, cols, window=20, episode_len=None, random_start=False)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(a))
    pd.testing.assert_frame_equal(fast, env.history())


def test_ensemble_weights():
    w = ensemble_weights({"a": 2.0, "b": 1.0, "c": -0.5})
    assert w["c"] == 0
    assert w["a"] > w["b"] > 0
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v == 0 for v in ensemble_weights({"a": -1, "b": 0}).values())
