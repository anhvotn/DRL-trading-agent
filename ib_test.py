"""Quick check: can Python connect to TWS and get SPY bars?
Run from PowerShell:  python ib_test.py
TWS must be running and logged in, with API access enabled.
"""
from ib_async import IB, Stock

PORT = 7497   # change this if TWS shows a different port under API settings

ib = IB()
try:
    ib.connect("127.0.0.1", PORT, clientId=1, timeout=10)
except Exception as e:
    print("Could not connect to TWS:", repr(e))
    print("Check that TWS is open, API is enabled, and the port is", PORT)
    raise SystemExit(1)

print("Connected. Accounts:", ib.managedAccounts())
ib.reqMarketDataType(3)   # 3 = delayed data (free)
bars = ib.reqHistoricalData(Stock("SPY", "SMART", "USD"), endDateTime="", durationStr="2 D",
                            barSizeSetting="15 mins", whatToShow="TRADES", useRTH=True)
if bars:
    print(f"Got {len(bars)} bars. Last one: {bars[-1]}")
else:
    print("Connected, but no data came back. Check the TWS window for an error message.")
ib.disconnect()
