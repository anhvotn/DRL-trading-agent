"""Trading environment: the agent sees a window of data and chooses buy / sell / hold.

Reward stage 1 ("pnl"):    position * next return - transaction costs (slippage + spread + fees)
Reward stage 2 ("sharpe"): differential Sharpe ratio (Moody & Saffell, 1998) of the same net return,
                           i.e. "PnL first, Sharpe ratio in later stages" as in the course.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

HOLD, BUY, SELL = 0, 1, 2


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, f: pd.DataFrame, obs_cols: list[str], window: int = 30,
                 episode_len: int | None = 2048, reward: str = "pnl", allow_short: bool = True,
                 fee: float = 0.0, cost_mult: float = 1.0, dsr_eta: float = 0.01,
                 random_start: bool = True, seed: int | None = None):
        super().__init__()
        self.X = f[obs_cols].to_numpy(np.float32)
        self.r = f["fwd_ret"].to_numpy(np.float64)
        self.c = f["cost"].to_numpy(np.float64) * cost_mult + fee
        self.index = f.index
        self.window, self.reward_mode = window, reward
        self.allow_short, self.dsr_eta = allow_short, dsr_eta
        self.episode_len = episode_len or (len(f) - window - 1)
        self.episode_len = min(self.episode_len, len(f) - window - 1)
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)

        n_obs = window * len(obs_cols) + 1          # flattened window + current position
        self.observation_space = spaces.Box(-10, 10, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    # ------------------------------------------------------------------
    def _obs(self):
        w = self.X[self.t - self.window + 1: self.t + 1].ravel()
        return np.append(w, np.float32(self.pos)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        lo, hi = self.window - 1, len(self.X) - self.episode_len - 1
        self.t0 = int(self.rng.integers(lo, hi + 1)) if (self.random_start and hi > lo) else lo
        self.t, self.pos, self.steps = self.t0, 0.0, 0
        self.A = self.B = 0.0                        # DSR running moments
        self.log = []
        return self._obs(), {}

    def step(self, action: int):
        target = {HOLD: self.pos, BUY: 1.0, SELL: -1.0 if self.allow_short else 0.0}[int(action)]
        trade = abs(target - self.pos)
        cost = trade * self.c[self.t]
        net = target * self.r[self.t] - cost            # simple-ish log-return approximation
        self.pos = target

        if self.reward_mode == "sharpe":
            dA, dB = net - self.A, net ** 2 - self.B
            denom = (self.B - self.A ** 2) ** 1.5
            reward = (self.B * dA - 0.5 * self.A * dB) / denom if denom > 1e-12 else 0.0
            self.A += self.dsr_eta * dA
            self.B += self.dsr_eta * dB
            reward = float(np.clip(reward, -5, 5))
        else:
            reward = float(net * 100)                    # scale to ~unit size for PPO

        self.log.append((self.index[self.t], target, self.r[self.t], cost, net))
        self.t += 1
        self.steps += 1
        terminated = self.steps >= self.episode_len or self.t >= len(self.X) - 1
        return self._obs(), reward, terminated, False, {"net": net, "cost": cost}

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=["time", "pos", "ret", "cost", "net"]).set_index("time")
