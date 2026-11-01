"""Build site/coins.json for the Coin Checker page (coins.html) and the API.

For every top-200 coin on CoinMarketCap (stablecoins left out):
  price, 24h / 7d / 30d change, market cap, 24h volume, 92 daily closes (for the bracket odds the page
  computes in the browser), the 7-day model's top-10 chance and rank today, whether the coin is in
  today's 7-day or 3-day call, its Rebound Radar verdict, plus the market light.

Credits: listings ~2 + 92 days of history for ~200 coins ~185 per run.
Runs in the daily GitHub Action after the trackers.

--quick (every 4 hours, .github/workflows/coin-prices.yml): refresh only price, 24h / 7d / 30d change, market
cap and volume from one listings call (~2 credits); history, scores and verdicts stay from the daily run.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cmc_client as cmc  # noqa: E402
import trader_stats       # noqa: E402

DAYS = 92


def sig(x, n=6):
    return None if x is None else float(f"{x:.{n}g}")


def load(name, default=None):
    try:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def quick():
    """Every 4 hours: refresh prices and moves in the existing coins.json from one listings call."""
    path = ROOT / "site" / "coins.json"
    doc = load("site/coins.json")
    if not doc:
        print("coins.json not built yet; the daily run builds it first")
        return
    live = {c["cmc_id"]: c for c in cmc.listings(200)}
    n = 0
    for c in doc["coins"]:
        x = live.get(c["cmc_id"])
        if not x or not x.get("current_price"):
            continue
        c.update(price=sig(x["current_price"]), ch24=sig(x.get("price_change_percentage_24h"), 4),
                 ch7=sig(x.get("price_change_percentage_7d_in_currency"), 4), ch30=sig(x.get("price_change_percentage_30d"), 4),
                 mcap=sig(x.get("market_cap"), 4), vol24=sig(x.get("total_volume"), 4))
        n += 1
    doc["prices_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"coins.json: refreshed prices for {n} coins at {doc['prices_at']}")


def main():
    if "--quick" in sys.argv:
        return quick()
    data = load("prediction_data.json", {})
    preds = data.get("predictions", {})
    day = max(preds) if preds else None
    today = preds.get(day, {})
    scores = today.get("scores") or {}
    ranked = sorted(scores, key=lambda k: -scores[k])
    rank = {k: i + 1 for i, k in enumerate(ranked)}
    in7 = {c["id"] for c in today.get("predicted_coins", [])}
    p3 = data.get("predictions_3d", {})
    in3 = {c["id"] for c in p3.get(max(p3), {}).get("predicted_coins", [])} if p3 else set()
    rb = load("rebound_data.json", {})
    verdict = {e["cmc_id"]: dict(verdict=e["verdict"], flagged=e["flagged"]) for e in (rb.get("open") or {}).values()}
    verdict.update({c["cmc_id"]: dict(verdict=c["verdict"], flagged=(verdict.get(c["cmc_id"]) or {}).get("flagged"))
                    for c in (rb.get("today") or {}).get("coins", [])})

    coins = [c for c in cmc.listings(200) if "stablecoin" not in c["tags"]]
    hist = cmc.daily_history([c["cmc_id"] for c in coins], days=DAYS)
    out = []
    for c in coins:
        rows = hist.get(c["cmc_id"], [])
        out.append(dict(
            id=c["id"], cmc_id=c["cmc_id"], symbol=c["symbol"].upper(), name=c["name"], image=c["image"],
            price=sig(c["current_price"]), ch24=sig(c.get("price_change_percentage_24h"), 4),
            ch7=sig(c.get("price_change_percentage_7d_in_currency"), 4), ch30=sig(c.get("price_change_percentage_30d"), 4),
            mcap=sig(c.get("market_cap"), 4), vol24=sig(c.get("total_volume"), 4),
            score7=scores.get(c["id"]), rank7=rank.get(c["id"]), ranked=len(ranked),
            in7=c["id"] in in7, in3=c["id"] in in3, rebound=verdict.get(c["cmc_id"]),
            start=rows[0]["date"] if rows else None, closes=[sig(r["price"]) for r in rows]))
    light = trader_stats.light_for(data, day, today) if day else None
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = dict(generated_at=now, prices_at=now, call_date=day, light=light, sentiment=trader_stats.market_sentiment(today.get("regime")), coins=out,
               note="closes are CoinMarketCap daily snapshots (00:00 UTC); score7 = model v2's chance (%) of finishing in the "
                    "week's top 10 of ~200 coins; 3-day calls are experimental. Not financial advice.")
    (ROOT / "site" / "coins.json").write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"coins.json: {len(out)} coins, {sum(1 for c in out if c['score7'] is not None)} with a 7-day score")


if __name__ == "__main__":
    main()
