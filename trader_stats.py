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
# Review rule for model v2, set before any v2 call was graded (published on proof.html).
# One call a week from the first v2 call; decide once WEEKS of them are graded.
REVIEW = dict(model="v2", since="2026-10-24", weeks=8, expected="2026-12-19",
              keep=0.0, retire=-0.02, max_drawdown=-0.50,
              text="After 8 weekly model-v2 calls are graded: keep v2 if its average return beats the top-200 average; "
                   "keep it but review again at 16 weeks if it trails by less than 2 points per call; retire it if it "
                   "trails by 2 points or more per call, or if following it weekly ever loses more than 50% from a peak.")
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


def market_return(data, start, end):
    """Equal-weight average return of the coins in both daily snapshots (the ~200 tracked), minus COST.

    Stablecoins and gold tokens are left out, as the model never picks them: anything priced near $1
    that moved less than 1%, and gold-backed tokens.
    """
    a = (data.get("predictions", {}).get(start) or {}).get("price_snapshot") or {}
    b = (data.get("predictions", {}).get(end) or {}).get("price_snapshot") or {}
    r = []
    for k in a:
        if k not in b or not a[k] or not b[k] or "gold" in k:
            continue
        x = b[k] / a[k] - 1
        if 0.9 <= a[k] <= 1.1 and abs(x) < 0.01:
            continue
        r.append(x)
    return float(np.mean(r)) - COST if len(r) >= 50 else None


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
        end = ev.get("evaluation_date", _shift(day, horizon))
        b0, b1 = btc_on(data, day), btc_on(data, end)
        lt = light_for(data, day, preds.get(day))
        rows.append(dict(date=day, end=end, basket=float(np.mean(coins)), coins=coins, model=(preds.get(day) or {}).get("model"),
                         btc=(b1 / b0 - 1) if b0 and b1 else None, market=market_return(data, day, end),
                         light=lt["state"] if lt else None))
    return rows


def summarize(rows):
    if not rows:
        return None
    b = np.array([r["basket"] for r in rows]); c = np.concatenate([r["coins"] for r in rows])
    vs = [r for r in rows if r["btc"] is not None]
    vm = [r for r in rows if r.get("market") is not None]
    wins, losses = c[c > 0], c[c <= 0]
    return dict(calls=len(rows), avg=float(b.mean()), median=float(np.median(b)), in_profit=float((b > 0).mean()),
                beat_btc=float(np.mean([r["basket"] > r["btc"] for r in vs])) if vs else None,
                btc_avg=float(np.mean([r["btc"] for r in vs])) if vs else None,
                market_avg=float(np.mean([r["market"] for r in vm])) if vm else None,
                beat_market=float(np.mean([r["basket"] > r["market"] for r in vm])) if vm else None,
                coin_win=float((c > 0).mean()), avg_win=float(wins.mean()) if len(wins) else 0.0,
                avg_loss=float(losses.mean()) if len(losses) else 0.0, expectancy=float(c.mean()),
                best=max(rows, key=lambda r: r["basket"])["date"], worst=min(rows, key=lambda r: r["basket"])["date"],
                best_ret=float(b.max()), worst_ret=float(b.min()))


def stats(data, horizon, model=None):
    """Live track record overall, by market light, and for the last 30 graded calls (optionally one model only)."""
    rows = [r for r in call_results(data, horizon) if model is None or r["model"] == model]
    return dict(all=summarize(rows), last30=summarize(rows[-30:]),
                trade=summarize([r for r in rows if r["light"] == "trade"]),
                caution=summarize([r for r in rows if r["light"] == "caution"]),
                equity=equity(rows, horizon),
                calls=[dict(date=r["date"], end=r["end"], basket=round(r["basket"], 4),
                            btc=None if r["btc"] is None else round(r["btc"], 4),
                            market=None if r["market"] is None else round(r["market"], 4), light=r["light"]) for r in rows])


def equity(rows, horizon):
    """Growth of 1 when trading one call every `horizon` days (non-overlapping, ending with the latest
    graded call), next to Bitcoin and the equal-weight market over the same windows."""
    if not rows:
        return []
    last = date.fromisoformat(rows[-1]["date"])
    pick = [r for r in rows if (last - date.fromisoformat(r["date"])).days % horizon == 0
            and r["btc"] is not None and r["market"] is not None]
    out, e, b, m = [], 1.0, 1.0, 1.0
    for r in pick:
        e, b, m = e * (1 + r["basket"]), b * (1 + r["btc"]), m * (1 + r["market"])
        out.append(dict(date=r["end"], calls=round(e, 4), btc=round(b, 4), market=round(m, 4)))
    return out


def backtest(horizon):
    try:
        return json.loads(BACKTEST.read_text())[str(horizon)]
    except (OSError, KeyError, ValueError):
        return None


def review(data):
    """Progress of the pre-set model-v2 review: weekly v2 calls graded so far and where they stand."""
    since = date.fromisoformat(REVIEW["since"])
    rows = [r for r in call_results(data, 7) if r["model"] == REVIEW["model"] and r["market"] is not None
            and (date.fromisoformat(r["date"]) - since).days % 7 == 0]
    out = dict(REVIEW, graded_weeks=len(rows), calls=[dict(date=r["date"], basket=round(r["basket"], 4),
                                                         market=round(r["market"], 4)) for r in rows])
    if not rows:
        return dict(out, status="collecting", edge=None, drawdown=None)
    edge = float(np.mean([r["basket"] - r["market"] for r in rows]))
    eq = np.cumprod([1 + r["basket"] for r in rows])
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    if dd <= REVIEW["max_drawdown"]:
        status = "retire"
    elif len(rows) < REVIEW["weeks"]:
        status = "collecting"
    else:
        status = "keep" if edge >= REVIEW["keep"] else "retire" if edge <= REVIEW["retire"] else "extend"
    return dict(out, status=status, edge=edge, drawdown=dd)


def market_sentiment(regime):
    """Market-wide sentiment: CoinMarketCap Fear & Greed (0-100) on the day of a call and 7 days earlier."""
    if not regime or regime.get("fng") is None:
        return None
    v = regime["fng"]
    label = ("Extreme fear" if v < 25 else "Fear" if v < 45 else "Neutral" if v <= 55 else "Greed" if v <= 75 else "Extreme greed")
    return dict(fear_greed=v, fear_greed_7d_ago=regime.get("fng_7"), label=label, as_of=regime.get("date"),
                source="CoinMarketCap Fear & Greed index (market-wide, not per coin)")
