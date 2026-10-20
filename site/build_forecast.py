"""Build site/forecast.json: 3-day and 7-day top-10 forecasts for the landing page.

Inputs
  backtest/results.json   tuned weights per horizon + backtest stats (from backtest_weights.py)
  CoinMarketCap           listings/latest (24h & 7d change, volume, market cap) and
                          quotes/historical daily closes (RSI-14, 7-day volatility), via cmc_client.py
  prediction_data.json    only to anchor the date to the latest daily prediction run

The signals match backtest_weights.py (CMC daily data, 7-day volatility, RSI-14 on daily
closes), so the tuned weights are applied to the same inputs they were tuned on.
Runs in the daily GitHub Action after the trackers (needs CMC_API_KEY).
"""
import json, statistics, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cmc_client import fetch_market  # noqa: E402

STABLE_SYMS = {"usdt", "usdc", "dai", "fdusd", "usde", "usds", "tusd", "pyusd", "frax", "usdd", "usdp", "gusd",
               "xaut", "paxg", "usd1", "rlusd", "usdtb", "bfusd", "susds", "susde", "usdg", "usd0"}


def rsi14(closes):
    """RSI-14 on daily closes, or None when the coin has under 15 days of history."""
    closes = [p for p in closes if p]
    if len(closes) < 15:
        return None
    d = [closes[i] - closes[i - 1] for i in range(len(closes) - 14, len(closes))]
    g = sum(max(x, 0) for x in d) / 14; l = sum(max(-x, 0) for x in d) / 14
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def rsi_score(r):
    return 0 if r >= 80 else 20 if r >= 70 else 50 if r >= 60 else 70 if r >= 40 else 85 if r >= 30 else 100


def norm(rows, key):
    vals = [r[key] for r in rows]; mn, mx = min(vals), max(vals)
    for r in rows:
        r[key + "_n"] = (r[key] - mn) / (mx - mn) * 100 if mx > mn else 50.0


def main():
    bt = json.loads((ROOT / "backtest" / "results.json").read_text(encoding="utf-8"))
    snaps = json.loads((ROOT / "prediction_data.json").read_text(encoding="utf-8"))["predictions"]
    coins = [c for c in fetch_market(limit=200, days=16)
             if "stablecoin" not in c["tags"] and c["symbol"] not in STABLE_SYMS
             and c["current_price"] and c["total_volume"] and c["market_cap"]
             and c["price_change_percentage_24h"] is not None
             and c["price_change_percentage_7d_in_currency"] is not None]
    btc7 = next(c for c in coins if c["cmc_id"] == 1)["price_change_percentage_7d_in_currency"]
    rows = []
    for c in coins:
        closes = c["daily_closes"]
        daily = [p for p in closes[-8:] if p]
        rets = [daily[i] / daily[i - 1] - 1 for i in range(1, len(daily))]
        vol = max(statistics.pstdev(rets), 1e-4) if len(rets) > 1 else 0.01
        mom7 = c["price_change_percentage_7d_in_currency"]; mom24 = c["price_change_percentage_24h"]
        r = rsi14(closes)
        rows.append(dict(id=c["id"], symbol=c["symbol"].upper(), name=c["name"], image=c["image"],
                         price=c["current_price"], mom24h=mom24, mom7d=mom7, rsi=None if r is None else round(r, 1),
                         vol_adj_mom=mom7 / (vol * 100 + 1), btc_rel=mom7 - btc7,
                         vol_surge=c["total_volume"] / c["market_cap"], rsi_s=rsi_score(50 if r is None else r)))
    for k in ("vol_adj_mom", "mom24h", "btc_rel", "vol_surge"):
        norm(rows, k)
    # Anchor to the latest daily prediction run, not the wall clock,
    # so the 3-day/7-day targets line up with the dashboard's dates.
    now = datetime.now(timezone.utc)
    as_of = datetime.strptime(max(snaps), "%Y-%m-%d")
    out = dict(generated_at=now.isoformat(timespec="minutes"), as_of=as_of.strftime("%Y-%m-%d"),
               source="CoinMarketCap", coins_scored=len(rows), horizons={})
    for h, info in bt["horizons"].items():
        w = info["weights"]
        for r in rows:
            r["_s"] = (w["vol_adj_mom"] * r["vol_adj_mom_n"] + w["mom24h"] * r["mom24h_n"] + w["btc_rel"] * r["btc_rel_n"]
                       + w["vol_surge"] * r["vol_surge_n"] + w["rsi_s"] * r["rsi_s"]) / 100
        top = sorted(rows, key=lambda r: -r["_s"])[:10]
        out["horizons"][h] = dict(
            target_date=(as_of + timedelta(days=int(h))).strftime("%Y-%m-%d"), weights=w,
            backtest=dict(test=info["tuned"]["test"], train=info["tuned"]["train"], live_test=info["live"]["test"],
                          train_end=bt["train_end"], range=bt["range"]),
            picks=[dict(rank=i + 1, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                      if k in ("id", "symbol", "name", "image", "price", "mom24h", "mom7d", "rsi")},
                        score=round(r["_s"], 1)) for i, r in enumerate(top)])
    (ROOT / "site" / "forecast.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("forecast.json:", {h: v["target_date"] for h, v in out["horizons"].items()}, f"{len(rows)} coins scored")


if __name__ == "__main__":
    main()
