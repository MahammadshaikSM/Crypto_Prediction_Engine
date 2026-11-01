"""Weekly calls on the stock watchlist: does any simple ranking pick the better names for the next week?

Every Friday close, rank the watchlist names that have at least 6 months of history, "buy" the top 5 in
equal weight and hold to the next Friday close, minus 0.1% round-trip costs. Compare with the
equal-weight average of all eligible watchlist names that week (the fair yardstick: these names were
chosen because they are popular today, so the watchlist itself looks great in hindsight).

Rules are fixed in advance (no fitting beyond picking the rule):
  mom_12_1   12-month return skipping the last month (classic momentum)
  mom_3m     3-month return
  rev_1w     last week's losers (short-term reversal)
  low_vol    lowest 3-month volatility
  trend      price above its 50-day average, ranked by distance from the 52-week high (closest first)
  combo      average rank of mom_12_1, rev_1w and trend
Pick the rule on the training years (to Oct 2024), report the held-out years (Nov 2024 - Oct 2026) once.

Usage: python backtest/watchlist_backtest.py   (downloads daily prices with yfinance)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from watchlist import TICKERS, scores  # noqa: E402

COST, TOP, TRAIN_END = 0.001, 5, "2024-10-31"
rng = np.random.default_rng(5)


def prices():
    import yfinance as yf
    df = yf.download(TICKERS + ["SPY", "QQQ"], period="6y", auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all")


def run(px, rule, start, end):
    fridays = [i for i, d in enumerate(px.index) if d.weekday() == 4 and start <= str(d.date()) <= end]
    rows = []
    for a, b in zip(fridays, fridays[1:]):
        sc = scores(px, a)
        s = sc[rule]
        names = [n for n in s.index if n in TICKERS]
        if len(names) < 10:
            continue
        r = px.iloc[b][names] / px.iloc[a][names] - 1
        pick = s[names].sort_values(ascending=False).index[:TOP]
        rows.append(dict(date=str(px.index[a].date()), pick=float(r[pick].mean()) - COST, avg=float(r.mean()) - COST,
                         spy=float(px.iloc[b]["SPY"] / px.iloc[a]["SPY"] - 1), n=len(names)))
    return rows


def summarize(rows):
    e = np.array([r["pick"] - r["avg"] for r in rows])
    boot = rng.choice(e, (4000, len(e))).mean(1)
    eq = np.cumprod([1 + r["pick"] for r in rows]); eqa = np.cumprod([1 + r["avg"] for r in rows])
    return dict(weeks=len(rows), pick=float(np.mean([r["pick"] for r in rows])), avg=float(np.mean([r["avg"] for r in rows])),
                spy=float(np.mean([r["spy"] for r in rows])), edge=float(e.mean()), lo=float(np.percentile(boot, 5)),
                hi=float(np.percentile(boot, 95)), beat=float((e > 0).mean()), equity=float(eq[-1] - 1),
                equity_avg=float(eqa[-1] - 1), drawdown=float((eq / np.maximum.accumulate(eq) - 1).min()))


RULES = ["mom_12_1", "mom_3m", "rev_1w", "low_vol", "trend", "combo"]


def main():
    px = prices()
    print(f"prices {px.index[0].date()} -> {px.index[-1].date()}, {px.shape[1]} tickers")
    tr = {r: summarize(run(px, r, "2021-11-01", TRAIN_END)) for r in RULES}
    print("\nTRAIN (Nov 2021 - Oct 2024): weekly top-5 vs the watchlist average, after 0.1% costs")
    for r in sorted(RULES, key=lambda k: -tr[k]["edge"]):
        x = tr[r]
        print(f"  {r:<9} edge {x['edge']*100:+.2f}%/wk (90% {x['lo']*100:+.2f}..{x['hi']*100:+.2f}) beat {x['beat']:.0%} weeks {x['weeks']}")
    best = max(RULES, key=lambda k: tr[k]["edge"])
    te = {r: summarize(run(px, r, "2024-11-01", "2026-12-31")) for r in RULES}
    print(f"\nHELD-OUT (Nov 2024 - Oct 2026), reported once; chosen on train: {best}")
    for r in RULES:
        x = te[r]; tag = "  <- chosen" if r == best else ""
        print(f"  {r:<9} pick {x['pick']*100:+.2f}%/wk vs watchlist {x['avg']*100:+.2f}% (SPY {x['spy']*100:+.2f}%) | edge {x['edge']*100:+.2f}%"
              f" (90% {x['lo']*100:+.2f}..{x['hi']*100:+.2f}) | beat {x['beat']:.0%} | equity {x['equity']*100:+.0f}% vs {x['equity_avg']*100:+.0f}% | dd {x['drawdown']*100:.0f}%{tag}")
    (HERE / "watchlist_results.json").write_text(json.dumps(dict(train=tr, heldout=te, chosen=best, top=TOP, cost=COST), indent=1))


if __name__ == "__main__":
    main()
