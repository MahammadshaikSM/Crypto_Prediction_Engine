"""The stock watchlist: US-listed names that are also traded as USDT perpetuals on crypto exchanges.

SK hynix trades on exchanges as both SKHY and SKHYNIX; both follow the same company, tracked once as SKHY.
The weekly-call rule lives here so the backtest (backtest/watchlist_backtest.py) and the live tracker
(watchlist_tracker.py) score names the same way.
"""
import numpy as np
import pandas as pd

TICKERS = [
    "SNDK", "GOOGL", "MU", "INTC", "PLTR", "NVDA", "NFLX", "MRVL", "SKHY", "HOOD", "CBRS", "AMZN",
    "TSM", "QCOM", "IBM", "DELL", "MSTR", "MSFT", "SPCX", "AMD", "TSLA", "META", "DRAM",
]
BENCHMARKS = ["SPY", "QQQ"]
RULE = "combo"        # chosen on 2021-2024 data, before looking at 2024-2026 (see the backtest)
TOP, COST = 5, 0.001  # names per weekly call; round-trip cost per name


def scores(px, t):
    """Scores for every name with 6 months of history at row t of a daily close table (higher = better)."""
    hist = px.iloc[:t + 1]
    ok = hist.iloc[-127:].notna().all() if len(hist) > 127 else pd.Series(False, index=px.columns)
    p = hist.iloc[-1]
    s = {}
    s["mom_12_1"] = hist.iloc[-22] / hist.iloc[-253] - 1 if len(hist) > 253 else pd.Series(np.nan, index=px.columns)
    s["mom_3m"] = p / hist.iloc[-64] - 1
    s["rev_1w"] = -(p / hist.iloc[-6] - 1)
    s["low_vol"] = -np.log(hist.iloc[-64:]).diff().std()
    ma50, hi = hist.iloc[-50:].mean(), hist.iloc[-253:].max()
    s["trend"] = pd.Series(np.where(p > ma50, p / hi, -1 + p / hi), index=px.columns)
    ranks = {k: v[ok].rank(pct=True) for k, v in s.items()}
    s["combo"] = (ranks["mom_12_1"].fillna(.5) + ranks["rev_1w"] + ranks["trend"]) / 3
    return {k: v[ok].dropna() for k, v in s.items()}
