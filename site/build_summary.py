"""Build the small JSON files the landing page (index.html) loads.

prediction_data.json is several MB, too heavy for a landing page on mobile.
This script distils it into:
  site/summary.json  - today's picks, scoreboard rows for the chart, stock stats
  site/history.json  - per-day graded picks for the "time machine" (lazy-loaded)

Runs in the daily GitHub Action right after the trackers. Standard library only.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site"
sys.path.insert(0, str(ROOT))
import trader_stats  # noqa: E402


def load(name):
    with open(ROOT / name, encoding="utf-8") as fh:
        return json.load(fh)


def rnd(x, n=2):
    return None if x is None else round(float(x), n)


def market_up_share(start, end):
    """Share (%) of tracked assets whose price rose between two snapshots."""
    common = [k for k in start if k in end]
    if not common:
        return None
    return 100 * sum(end[k] > start[k] for k in common) / len(common)


def crypto():
    data = load("prediction_data.json")
    preds, evals = data["predictions"], data["evaluations"]
    latest = preds[max(preds)]
    rows, history, images = [], [], {}
    for day in sorted(evals):
        ev, p = evals[day], preds.get(day)
        if not p:
            continue
        snap = p.get("price_snapshot") or {}
        later = preds.get(ev.get("evaluation_date") or ev.get("maturity_date"))
        mkt = market_up_share(snap, later["price_snapshot"]) if later and later.get("price_snapshot") and snap else None
        rows.append({"d": day, "hit": rnd(ev["hit_rate"], 1), "base": rnd(1000 / (len(snap) or 200), 2),
                     "pos": rnd(ev["positive_rate"], 1), "mkt": rnd(mkt, 1), "hc": ev["hit_count"]})
        picks = []
        for c in sorted(ev.get("coin_results", []), key=lambda c: c.get("rank_predicted", 0)):
            images.setdefault(c["id"], c.get("image"))
            picks.append([c.get("rank_predicted"), c["id"], c.get("symbol", ""), c.get("name", ""),
                          rnd(c.get("actual_return_pct")), 1 if (c.get("hit") or c.get("in_actual_top10")) else 0])
        actual = [[a.get("symbol", ""), rnd(a.get("return_pct")), 1 if a.get("was_predicted") else 0]
                  for a in ev.get("actual_top10", [])]
        history.append({"d": day, "e": ev.get("evaluation_date") or ev.get("maturity_date"),
                        "hc": ev["hit_count"], "pc": ev["positive_count"], "p": picks, "a": actual})
    today = [{k: c.get(k) for k in ("rank", "id", "symbol", "name", "image", "score", "confidence", "mom24h", "rsi")}
             for c in latest["predicted_coins"][:10]]
    tr = trader_stats.stats(data, 7)
    trader = {k: tr[k] and {x: rnd(v, 4) if isinstance(v, float) else v for x, v in tr[k].items()}
              for k in ("all", "last30", "trade", "caution")}
    summary = {"date": latest["date"], "maturity": latest["maturity_date"], "fetched_at": latest.get("fetched_at"),
               "days": len(preds), "today": today, "rows": rows,
               "light": trader_stats.light_for(data, latest["date"], latest), "trader": trader}
    return summary, {"images": images, "days": history}


def stocks():
    try:
        data = load("stock_data.json")
    except FileNotFoundError:
        return None
    preds, evals = data["predictions"], data["evaluations"]
    rows = []
    for day in sorted(evals):
        p = preds.get(day)
        if not p:
            continue
        n = len(p.get("price_snapshot") or {}) or 100
        rows.append((evals[day]["hit_rate"], 1000 / n, evals[day]["positive_rate"], n))
    if not rows:
        return None
    avg = lambda i: sum(r[i] for r in rows) / len(rows)
    return {"days": len(preds), "graded": len(rows), "hit": rnd(avg(0), 1), "base": rnd(avg(1), 1),
            "pos": rnd(avg(2), 1), "universe": round(avg(3))}


def main():
    summary, history = crypto()
    summary["stocks"] = stocks()
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(history, separators=(",", ":")), encoding="utf-8")
    print(f"summary.json: {len(summary['rows'])} graded days, picks for {summary['date']}")


if __name__ == "__main__":
    main()
