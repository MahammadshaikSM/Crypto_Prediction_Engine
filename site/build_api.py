"""Machine-readable Ichnora data for trading bots and AI assistants (static JSON, no key needed).

Writes
  api/v1/today.json         market light + today's 7-day and 3-day calls + Rebound Radar headline
  api/v1/track-record.json  what following every call earned (after costs) vs Bitcoin, by market light
  api/v1/rebound.json       every top-200 coin down 25%+ in 30 days, with its comeback verdict
  api/v1/calls.json         recent calls with entry prices, latest prices and grades
  api/openapi.json          OpenAPI 3.1 description of the endpoints
  llms.txt                  plain-text guide for AI assistants (llmstxt.org convention)

Everything is a plain file served by the website, so any HTTP client can read it.
Runs in the daily GitHub Action after build_posts.py and rebound_tracker.py.
"""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import trader_stats  # noqa: E402

API = ROOT / "api"
V1 = API / "v1"
VERSION = "1.0"
SITE = os.environ.get("PUBLIC_SITE_URL", "").strip().rstrip("/")
DISCLAIMER = ("Research data from a public experiment, not financial advice. Crypto is volatile and can go to zero. "
              "Past results do not guarantee future returns. Use as one input next to your own judgement and risk limits.")


def load(path, default=None):
    try:
        return json.loads((ROOT / path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def url(path):
    return f"{SITE}/{path}" if SITE else path


def r(x, n=4):
    return None if x is None else round(float(x), n)


def summary(s):
    if not s:
        return None
    keys = ("calls", "avg", "median", "in_profit", "beat_btc", "btc_avg", "coin_win", "avg_win", "avg_loss", "expectancy")
    return {k: (r(s[k]) if isinstance(s[k], float) else s[k]) for k in keys}


def picks(entry, h):
    out = []
    for c in entry.get("predicted_coins", []):
        out.append(dict(rank=c["rank"], symbol=c["symbol"].upper(), name=c["name"], cmc_slug=c["id"],
                        price_at_call=c.get("price_at_prediction"), score=c.get("score"),
                        rsi=c.get("rsi"), change_24h_pct=c.get("mom24h"), change_7d_pct=c.get("mom7d")))
    return out


def light_block(data, day, entry):
    lt = trader_stats.light_for(data, day, entry)
    if not lt:
        return None
    bt = (trader_stats.backtest(7) or {}).get("light", {})
    return dict(state=lt["state"], label=trader_stats.LIGHTS[lt["state"]][0], btc_change_7d_pct=lt["btc7"], as_of=lt["as_of"],
                rule="TRADE when Bitcoin's 7-day change is >= 0, CAUTION when it is negative.",
                evidence=dict(
                    applies_to="7-day calls (no measurable effect on 3-day calls in testing)",
                    backtest_period="2025-10-01 to 2026-09-30, held out from training",
                    backtest_avg_return_per_call_trade=r(bt.get("trade", {}).get("basket_mean")),
                    backtest_avg_return_per_call_caution=r(bt.get("caution", {}).get("basket_mean")),
                    note="Backtest edge is likely but not certain (about 88-94% bootstrap probability). "
                         "Live split is in track-record.json and updates daily."))


def call_block(data, h):
    preds = data.get("predictions" if h == 7 else "predictions_3d", {})
    if not preds:
        return None
    day = max(preds); e = preds[day]
    status = ("main signal: beat comparable coins after costs in a held-out year (see proof.html)" if h == 7 else
              "experimental: no reliable edge over comparable coins after realistic (1%) trading costs; use as a watchlist")
    return dict(horizon_days=h, date=day, grades_on=e.get("maturity_date"), status=status,
                model="model v2 (18 signals + market regime, logistic regression)" if e.get("model") == "v2" and h == 7
                else "5-signal formula with weights tuned on 3 years of CMC data",
                how_to_read=("Buy-and-hold idea for the window: the 10 coins the model rates most likely to be among the "
                             "top-10 gainers of ~200 coins by the end date. Hold all 10 in equal size; returns come from a few "
                             "big winners. score = model rating (7-day: chance in % of finishing top-10, 3-day: 0-100 index)."),
                picks=picks(e, h))


def rebound_block(rb, full=False):
    if not rb:
        return None
    t = rb.get("today", {})
    coins = [dict(symbol=c["symbol"], name=c["name"], cmc_slug=c["slug"], verdict=c["verdict"], comeback_score=c["rank"],
                  change_30d_pct=r(c["r30"] * 100, 2), from_90d_high_pct=r(c["hi90"] * 100, 2),
                  rise_needed_to_recover_pct=r((c["ref_price"] / c["price"] - 1) * 100, 2), price=c["price"])
             for c in t.get("coins", [])]
    out = dict(date=t.get("date"), coins_on_radar=len(coins),
               definition="Top-200 coins down 25%+ over 30 days. comeback_score = rank (0-100) among today's declining coins "
                          "of the model's chance to get back to the price of 30 days ago within 60 days.",
               verdicts=dict(comeback="top 30% of comeback scores", watch="middle 40%", unlikely="bottom 30%"))
    if full:
        out.update(coins=coins, tracking=list(rb.get("open", {}).values()), scorecard=rb.get("scorecard"),
                   replay=rb.get("replay"))
    else:
        out["comeback_candidates"] = [c for c in coins if c["verdict"] == "comeback"]
    return out


def openapi():
    def get(summary_, desc):
        return {"get": {"summary": summary_, "description": desc,
                        "responses": {"200": {"description": "JSON document", "content": {"application/json": {}}}}}}
    return {
        "openapi": "3.1.0",
        "info": {"title": "Ichnora public data", "version": VERSION,
                 "description": "Daily crypto calls, market light, track record and Rebound Radar as static JSON files. "
                                "No key, no rate limit beyond normal web hosting. Updated once a day around 11:30 UTC. " + DISCLAIMER},
        "servers": [{"url": SITE or "/"}],
        "paths": {
            "/api/v1/today.json": get("Today's market light and calls",
                                      "Market light, today's 7-day and 3-day calls (10 coins each, entry prices, model scores) and the Rebound Radar headline."),
            "/api/v1/track-record.json": get("What following the calls earned",
                                             "Average return per call after 0.3% costs, win rates, Bitcoin over the same days, split by market light, plus the held-out backtest."),
            "/api/v1/rebound.json": get("Rebound Radar",
                                        "Every top-200 coin down 25%+ in 30 days with its comeback verdict, tracked coins and the graded scorecard."),
            "/api/v1/calls.json": get("Recent calls", "Recent 7-day and 3-day calls with entry prices, latest prices, grades and basket return vs Bitcoin."),
        },
    }


def llms_txt(today):
    lt = today.get("market_light") or {}
    return f"""# Ichnora

> Daily crypto research calls graded in public. Each day a model scans ~200 coins on CoinMarketCap and publishes
> 10 coins for the next 7 days and 10 for the next 3 days, a market light (TRADE / CAUTION), and a Rebound Radar
> for coins down 25%+ in a month. Every call is graded and kept, wins and losses.

{DISCLAIMER}

## How an AI assistant should use this
- Read {url('api/v1/today.json')} first: market_light, calls_7d, calls_3d, rebound.
- Check {url('api/v1/track-record.json')} before relying on a signal: it shows average return per call after costs,
  how often calls beat Bitcoin, and how calls made on TRADE vs CAUTION days did.
- Treat the calls as a basket: hold all 10 in equal size for the stated window. Most of the return comes from a few
  big winners, so single picks are unreliable (picked coins go up no more often than other coins).
- The 7-day calls are the main signal. The 3-day calls are experimental: no reliable edge after realistic costs.
- How it was tested (matched baselines, independent weeks, calibration, liquidity): {url('proof.html')}
- The market light is based on Bitcoin's 7-day trend. In a held-out backtest 7-day calls did better on TRADE days;
  that evidence is likely but not certain. It had no effect on 3-day calls.
- Data refreshes once a day around 11:30 UTC. "as_of" and "generated_at" fields give the exact times.
- Today the market light is {lt.get('label', 'unknown')} (Bitcoin {lt.get('btc_change_7d_pct', 0):+.1f}% over 7 days).

## Endpoints (JSON, no key)
- [today.json]({url('api/v1/today.json')}): market light, today's calls, Rebound Radar headline
- [track-record.json]({url('api/v1/track-record.json')}): returns after costs vs Bitcoin, by market light, plus backtest
- [rebound.json]({url('api/v1/rebound.json')}): coins down 25%+ in 30 days, comeback verdicts, scorecard
- [calls.json]({url('api/v1/calls.json')}): recent calls with grades
- [openapi.json]({url('api/openapi.json')}): OpenAPI 3.1 description

## Pages for people
- [Home]({url('index.html')}), [7-day calls]({url('prediction_dashboard.html')}),
  [3-day calls]({url('prediction_dashboard_3d.html')}), [Rebound Radar]({url('rebound_dashboard.html')}),
  [Proof: how it was tested]({url('proof.html')}), [API docs]({url('api.html')})
"""


def main():
    data = load("prediction_data.json", {})
    rb = load("rebound_data.json")
    calls = load("site/calls.json", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = max(data.get("predictions", {}) or [None]) if data.get("predictions") else None
    meta = dict(api_version=VERSION, generated_at=now, source="CoinMarketCap", disclaimer=DISCLAIMER,
                docs=url("api.html"))
    today = dict(meta, date=day,
                 market_light=light_block(data, day, data["predictions"][day]) if day else None,
                 calls_7d=call_block(data, 7), calls_3d=call_block(data, 3), rebound=rebound_block(rb))
    record = dict(meta, method="Buy the 10 picks in equal amounts at the call price, sell at the grading price, minus "
                               f"{trader_stats.COST:.1%} round-trip costs per coin. Bitcoin over the same days for comparison.")
    for h in (7, 3):
        s = trader_stats.stats(data, h); bt = trader_stats.backtest(h)
        record[f"calls_{h}d"] = dict(
            all=summary(s["all"]), last_30=summary(s["last30"]),
            by_market_light=dict(trade=summary(s["trade"]), caution=summary(s["caution"])),
            backtest_held_out=bt and dict(period=bt["period"], calls=bt["hold"]["days"], avg=r(bt["hold"]["basket_mean"]),
                                          median=r(bt["hold"]["basket_median"]), in_profit=r(bt["hold"]["basket_win"]),
                                          beat_btc=r(bt["hold"]["beat_btc"]), btc_avg=r(bt["hold"]["btc_mean"])),
            per_call=s["calls"])
    V1.mkdir(parents=True, exist_ok=True)
    dump = lambda p, o: p.write_text(json.dumps(o, indent=1, ensure_ascii=False), encoding="utf-8")
    dump(V1 / "today.json", today)
    dump(V1 / "track-record.json", record)
    dump(V1 / "rebound.json", dict(meta, **(rebound_block(rb, full=True) or {})))
    dump(V1 / "calls.json", dict(meta, calls=calls.get("calls", calls if isinstance(calls, list) else [])))
    dump(API / "openapi.json", openapi())
    (ROOT / "llms.txt").write_text(llms_txt(today), encoding="utf-8")
    print(f"api: today.json ({day}), track-record.json, rebound.json, calls.json, openapi.json, llms.txt")


if __name__ == "__main__":
    main()
