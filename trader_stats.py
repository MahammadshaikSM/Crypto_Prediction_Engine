"""Trader view of the calls: market light, and what following each call would have earned.

Market light (validated on held-out data for the 7-day calls, backtest/trader_backtest.py):
  TRADE    Bitcoin is flat or up over the last 7 days
  CAUTION  Bitcoin is down over the last 7 days
It is read from BTC's daily CoinMarketCap snapshots and stored with each call when it is made;
older calls are filled in from the stored price snapshots.

Call results: buy the 10 picks in equal weight at the call's price, sell at the grading price,
minus COST round trip per coin. Bitcoin over the same window is the yardstick.
"""
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np

COST = 0.003
ROOT = Path(__file__).resolve().parent
BACKTEST = ROOT / "backtest" / "trader_results.json"
LIGHTS = {
    "trade": ("TRADE", "Bitcoin is flat or up over the last 7 days"),
    "caution": ("CAUTION", "Bitcoin is down over the last 7 days"),
}


def light_state(btc7):
    return None if btc7 is None else "trade" if btc7 >= 0 else "caution"


def light_from_history(rows):
    """Market light from BTC daily snapshots [{'date', 'price'}, ...] (oldest -> newest)."""
    if len(rows) < 8:
        return None
    btc7 = rows[-1]["price"] / rows[-8]["price"] - 1
    return {"state": light_state(btc7), "btc7": round(btc7 * 100, 2), "as_of": rows[-1]["date"]}


def _shift(day, days):
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def btc_on(data, day):
    return ((data.get("predictions", {}).get(day) or {}).get("price_snapshot") or {}).get("bitcoin")


def light_for(data, day, pred=None):
    """Stored light for a call, else BTC's 7-day change from the stored daily snapshots."""
    stored = (pred or {}).get("market_light") or (data.get("predictions", {}).get(day) or {}).get("market_light")
    if stored:
        return stored
    now, before = btc_on(data, day), btc_on(data, _shift(day, -7))
    if not now or not before:
        return None
    btc7 = now / before - 1
    return {"state": light_state(btc7), "btc7": round(btc7 * 100, 2), "as_of": day}


def call_results(data, horizon):
    """One row per graded call: basket return, BTC return over the same window, light, coin returns."""
    preds = data.get("predictions" if horizon == 7 else "predictions_3d", {})
    evals = data.get("evaluations" if horizon == 7 else "evaluations_3d", {})
    rows = []
    for day in sorted(evals):
        ev = evals[day]
        coins = [c["actual_return_pct"] / 100 - COST for c in ev.get("coin_results", [])
                 if c.get("actual_return_pct") is not None]
        if not coins:
            continue
        b0, b1 = btc_on(data, day), btc_on(data, ev.get("evaluation_date", _shift(day, horizon)))
        lt = light_for(data, day, preds.get(day))
        rows.append(dict(date=day, end=ev.get("evaluation_date"), basket=float(np.mean(coins)), coins=coins,
                         btc=(b1 / b0 - 1) if b0 and b1 else None, light=lt["state"] if lt else None))
    return rows


def summarize(rows):
    if not rows:
        return None
    b = np.array([r["basket"] for r in rows]); c = np.concatenate([r["coins"] for r in rows])
    vs = [r for r in rows if r["btc"] is not None]
    wins, losses = c[c > 0], c[c <= 0]
    return dict(calls=len(rows), avg=float(b.mean()), median=float(np.median(b)), in_profit=float((b > 0).mean()),
                beat_btc=float(np.mean([r["basket"] > r["btc"] for r in vs])) if vs else None,
                btc_avg=float(np.mean([r["btc"] for r in vs])) if vs else None,
                coin_win=float((c > 0).mean()), avg_win=float(wins.mean()) if len(wins) else 0.0,
                avg_loss=float(losses.mean()) if len(losses) else 0.0, expectancy=float(c.mean()),
                best=max(rows, key=lambda r: r["basket"])["date"], worst=min(rows, key=lambda r: r["basket"])["date"],
                best_ret=float(b.max()), worst_ret=float(b.min()))


def stats(data, horizon):
    """Live track record overall, by market light, and for the last 30 graded calls."""
    rows = call_results(data, horizon)
    return dict(all=summarize(rows), last30=summarize(rows[-30:]),
                trade=summarize([r for r in rows if r["light"] == "trade"]),
                caution=summarize([r for r in rows if r["light"] == "caution"]),
                calls=[dict(date=r["date"], end=r["end"], basket=round(r["basket"], 4),
                            btc=None if r["btc"] is None else round(r["btc"], 4), light=r["light"]) for r in rows])


def backtest(horizon):
    try:
        return json.loads(BACKTEST.read_text())[str(horizon)]
    except (OSError, KeyError, ValueError):
        return None
