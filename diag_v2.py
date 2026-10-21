"""Manual-run diagnostic: run model v2 end to end against live CMC data without saving picks.
Writes posts/diag_v2.json with each stage's result or error. Uses ~66 CMC credits."""
import json, traceback
from datetime import datetime, timezone
from pathlib import Path

out = {"at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "stages": {}}
path = Path(__file__).resolve().parent / "posts" / "diag_v2.json"
try:
    from cmc_client import fetch_market, market_regime
    import prediction_tracker as pt
    coins = fetch_market(limit=200, days=32)
    btc = next(c for c in coins if c["id"] == "bitcoin")
    out["stages"]["fetch"] = {"coins": len(coins), "btc_history_points": len(btc.get("daily_history", [])),
                              "btc_first_last": [btc["daily_history"][0]["date"], btc["daily_history"][-1]["date"]] if btc.get("daily_history") else None}
    try:
        regime = market_regime(); out["stages"]["regime"] = regime
    except Exception as e:
        out["stages"]["regime"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}; raise
    picks = pt.run_prediction_v2(coins, regime)
    out["stages"]["model_v2"] = {"picks": [c["symbol"].upper() for c in picks], "top_prob_pct": picks[0]["score"]}
    out["ok"] = True
except Exception as e:
    out["ok"] = False
    out["error"] = f"{type(e).__name__}: {e}"
    out["trace"] = traceback.format_exc()[-2500:]
path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str)[:3000])
