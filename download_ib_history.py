"""Download historical bars from Interactive Brokers (TWS or IB Gateway) into a CSV
that drl_trader can read.

    python download_ib_history.py                       # SPY, 15-min, 10 years -> data/spy_15m.csv
    python download_ib_history.py --symbol QQQ --years 5
    python download_ib_history.py --with-spread         # also fetch bid/ask (better slippage model, 2x slower)

Works backwards one week per request and saves after every chunk, so you can stop it with
Ctrl+C and run the same command again later: it continues where it stopped.
TWS must be running, logged in, with the API enabled.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from ib_async import IB, Stock, util

TZ = "US/Eastern"


def fetch_chunk(ib, contract, end, duration, bar, what):
    bars = ib.reqHistoricalData(contract, endDateTime=end, durationStr=duration,
                                barSizeSetting=bar, whatToShow=what, useRTH=True,
                                formatDate=2, timeout=120)
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    return df.set_index("date")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--bar", default="15 mins")
    p.add_argument("--years", type=float, default=10)
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--out", default=None)
    p.add_argument("--with-spread", action="store_true")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between requests")
    a = p.parse_args()

    out = a.out or os.path.join("data", f"{a.symbol.lower()}_{a.bar.replace(' mins', 'm').replace(' ', '')}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ib = IB()
    ib.connect("127.0.0.1", a.port, clientId=7, timeout=15)
    ib.reqMarketDataType(3)                       # delayed data is fine for history
    contract = Stock(a.symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    head = ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=True, formatDate=2)
    if head:
        head = pd.Timestamp(head)
        head = (head.tz_convert(TZ) if head.tzinfo else head.tz_localize("UTC").tz_convert(TZ)).tz_localize(None)
    else:
        head = None
    stop_at = pd.Timestamp.now() - pd.Timedelta(days=365 * a.years)
    if head is not None and head > stop_at:
        stop_at = head
    print(f"{a.symbol}: downloading back to {stop_at:%Y-%m-%d} (earliest available: {head})")

    # resume support
    if os.path.exists(out):
        have = pd.read_csv(out, parse_dates=["timestamp"]).set_index("timestamp")
        end = have.index.min()
        print(f"Resuming: {len(have):,} bars already in {out}, earliest {end}")
    else:
        have, end = pd.DataFrame(), None          # None = "up to now"

    empty_in_a_row = 0
    try:
        while end is None or end > stop_at:
            end_str = "" if end is None else end.tz_localize(TZ).to_pydatetime()
            trades = fetch_chunk(ib, contract, end_str, "1 W", a.bar, "TRADES")
            if trades.empty:
                empty_in_a_row += 1
                print(f"  no data for week ending {end:%Y-%m-%d}")
                if empty_in_a_row >= 3:
                    print("  three empty weeks in a row, stopping")
                    break
                end = (end or pd.Timestamp.now()) - pd.Timedelta(weeks=1)
                time.sleep(a.pause)
                continue
            empty_in_a_row = 0
            chunk = trades[["open", "high", "low", "close", "volume"]]

            if a.with_spread:
                time.sleep(a.pause)
                ba = fetch_chunk(ib, contract, end_str, "1 W", a.bar, "BID_ASK")
                if not ba.empty:
                    # IBKR BID_ASK bars: open = average bid, close = average ask over the bar
                    chunk = chunk.join(ba[["open", "close"]].rename(columns={"open": "bid", "close": "ask"}))

            have = pd.concat([chunk, have]) if not have.empty else chunk
            have = have[~have.index.duplicated(keep="last")].sort_index()
            have.index.name = "timestamp"
            have.to_csv(out)
            new_end = chunk.index.min()
            # guard: if IBKR returned the same week again, step back manually
            end = new_end if end is None or new_end < end else end - pd.Timedelta(weeks=1)
            print(f"  {chunk.index.min():%Y-%m-%d} .. {chunk.index.max():%Y-%m-%d}  "
                  f"(+{len(chunk)} bars, total {len(have):,})")
            time.sleep(a.pause)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
    finally:
        ib.disconnect()

    print(f"Saved {len(have):,} bars to {out}")
    if len(have) < 100_000:
        print("Note: under 100k rows. Fine for a first test, but download more for real training.")


if __name__ == "__main__":
    main()
