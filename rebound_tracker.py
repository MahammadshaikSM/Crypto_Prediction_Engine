#!/usr/bin/env python3
"""Rebound Radar: track every top-200 coin that is down 25%+ over 30 days, and grade what happens.

Daily run:
  1. Find coins down 25%+ over 30 days (CoinMarketCap listings, then 91 days of history for them).
  2. Score each one's chance of getting back to its price from 30 days ago within 60 days
     (rebound_model.py) and rank it against the day's other declining coins:
     Comeback candidate (top 30%) / Watching (middle 40%) / Unlikely to recover (bottom 30%).
  3. Open a tracking entry the first day a coin lands on the radar, with its verdict frozen.
  4. Update every open entry with today's price and grade it:
       Recovered     back to the pre-drop price (price 30 days before it was flagged), any time
       Bounced       +25% above the flag price at some point, but not recovered, after 60 days
       Fell further  20%+ below the flag price after 60 days
       Still down    anything else after 60 days
  5. Write rebound_data.json and rebound_dashboard.html.

Credits: listings ~2, market regime ~2, 92 days of history for the flagged coins ~1 per coin.
"""
import html, json
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np

import cmc_client as cmc
import rebound_model as rm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rebound_data.json"
PAGE = ROOT / "rebound_dashboard.html"
OUTCOMES = {
    "recovered": ("Recovered", "green"),
    "bounced": ("Bounced +25%", "blue"),
    "down": ("Still down", "gold"),
    "fell": ("Fell further", "red"),
}


def scan(model):
    regime = cmc.market_regime()
    coins = cmc.listings(200)
    pre = [c for c in coins if c["cmc_id"] != 1 and "stablecoin" not in c["tags"]
           and (c.get("price_change_percentage_30d") or 0) <= -20]  # loose pre-filter; exact test below
    hist = cmc.daily_history([1] + [c["cmc_id"] for c in pre], days=rm.HISTORY_DAYS + 1)
    btc = [r["price"] for r in hist.get(1, [])]
    radar = []
    for c in pre:
        rows = hist.get(c["cmc_id"], [])[-rm.HISTORY_DAYS:]
        if len(rows) < rm.HISTORY_DAYS or len(btc) < 31 or not all(r["volume"] and r["mcap"] for r in rows[-31:]):
            continue
        p = [r["price"] for r in rows]
        if p[-1] / p[-31] - 1 > rm.DECLINE:
            continue
        x = rm.coin_features(p, [r["volume"] for r in rows[-31:]], rows[-1]["mcap"], btc[-31:], regime)
        radar.append(dict(cmc_id=c["cmc_id"], slug=c["id"], symbol=c["symbol"].upper(), name=c["name"], image=c["image"],
                          price=c["current_price"] or p[-1], flag_price=p[-1], ref_price=p[-31], r30=p[-1] / p[-31] - 1,
                          hi90=p[-1] / max(p) - 1, prob=rm.prob(model["model"], x), snapshot=rows[-1]["date"]))
    for r, rank in zip(radar, rm.day_rank([r["prob"] for r in radar])):
        r["rank"] = round(float(rank), 1)
        r["verdict"] = rm.verdict(rank)
    radar.sort(key=lambda r: -r["prob"])
    return coins, regime, radar


def update(state, coins, radar, today):
    prices = {c["cmc_id"]: c["current_price"] for c in coins if c["current_price"]}
    missing = [int(k) for k in state["open"] if int(k) not in prices]
    if missing:
        try:
            prices.update(cmc.prices_by_id(missing))
        except RuntimeError as e:
            print(f"  could not price {len(missing)} tracked coins outside the top 200: {e}")
    for key, e in list(state["open"].items()):
        price = prices.get(int(key))
        if price:
            e["last"], e["last_date"] = price, today
            e["best"], e["worst"] = max(e["best"], price), min(e["worst"], price)
        e["days"] = (date.fromisoformat(today) - date.fromisoformat(e["flagged"])).days
        outcome = None
        if e["best"] >= e["ref_price"]:
            outcome = "recovered"
        elif e["days"] >= rm.HORIZON:
            outcome = ("fell" if e["last"] <= e["flag_price"] * (1 + rm.FELL)
                       else "bounced" if e["best"] >= e["flag_price"] * (1 + rm.BOUNCE) else "down")
        if outcome:
            e["outcome"], e["closed"] = outcome, today
            state["closed"].append(e)
            del state["open"][key]
    for r in radar:
        key = str(r["cmc_id"])
        if key not in state["open"]:
            state["open"][key] = dict(
                cmc_id=r["cmc_id"], symbol=r["symbol"], name=r["name"], image=r["image"], flagged=today,
                verdict=r["verdict"], rank=r["rank"], prob=r["prob"], flag_price=r["flag_price"], ref_price=r["ref_price"],
                r30=r["r30"], last=r["price"], last_date=today, best=max(r["price"], r["flag_price"]),
                worst=min(r["price"], r["flag_price"]), days=0)


def scorecard(closed):
    out = {}
    for v in rm.VERDICTS:
        g = [e for e in closed if e["verdict"] == v]
        out[v] = dict(n=len(g), **{o: (sum(e["outcome"] == o for e in g) / len(g) if g else None) for o in OUTCOMES})
    return out


# ---------- page ----------
def pct(x, digits=0, sign=True):
    return "&ndash;" if x is None else f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def money(x):
    return "&ndash;" if not x else f"${x:,.2f}" if x >= 1 else f"${x:.4g}"


def esc(s):
    return html.escape(str(s))


def coin_card(r):
    need = r["ref_price"] / r["price"] - 1
    return f"""<div class="card v-{r['verdict']}">
      <div class="card-top"><img src="{esc(r['image'])}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
        <div><div class="c-name">{esc(r['name'])}</div><div class="c-sym">{esc(r['symbol'])}</div></div>
        <div class="c-score" title="Comeback score: rank among today's declining coins">{r['rank']:.0f}</div></div>
      <div class="row"><span>30 days</span><b class="dn">{pct(r['r30'])}</b></div>
      <div class="row"><span>From 90d high</span><b class="dn">{pct(r['hi90'])}</b></div>
      <div class="row"><span>To recover</span><b>{pct(need)}</b></div>
      <div class="bar"><i style="width:{r['rank']:.0f}%"></i></div></div>"""


def build_page(state, model):
    today = state["today"]; radar = today["coins"]
    groups = {v: [r for r in radar if r["verdict"] == v] for v in rm.VERDICTS}
    held = model["heldout"]; bt = held["groups"]
    live = scorecard(state["closed"])
    open_rows = sorted(state["open"].values(), key=lambda e: (e["flagged"], -e["rank"]), reverse=True)

    def section(v, blurb):
        cards = "".join(coin_card(r) for r in groups[v]) or '<p class="empty">None today.</p>'
        return f"""<div class="group"><div class="g-head"><span class="pill p-{v}">{rm.VERDICTS[v]}</span>
          <span class="g-n">{len(groups[v])}</span><span class="g-blurb">{blurb}</span></div><div class="cards">{cards}</div></div>"""

    def track_row(e):
        since = e["last"] / e["flag_price"] - 1; best = e["best"] / e["flag_price"] - 1; need = e["ref_price"] / e["last"] - 1
        return f"""<tr><td><div class="tc"><img src="{esc(e['image'])}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
          <div><b>{esc(e['symbol'])}</b><small>{esc(e['name'])}</small></div></div></td>
          <td>{e['flagged']}</td><td><span class="pill p-{e['verdict']}">{rm.VERDICTS[e['verdict']]}</span></td>
          <td class="{'up' if since >= 0 else 'dn'}">{pct(since)}</td><td class="up">{pct(best)}</td>
          <td>{pct(need)}</td><td>{max(rm.HORIZON - e['days'], 0)}d</td></tr>"""

    def score_row(v):
        b, l = bt[v], live[v]
        cell = lambda x: "&ndash;" if x is None else pct(x, 0, False)
        return f"""<tr><td><span class="pill p-{v}">{rm.VERDICTS[v]}</span></td>
          <td>{pct(b['comeback'], 1, False)}</td><td>{pct(b['bounce'], 0, False)}</td><td>{pct(b['falling'], 0, False)}</td>
          <td>{l['n']}</td><td>{cell(l['recovered'])}</td><td>{cell(l['bounced'])}</td><td>{cell(l['fell'])}</td></tr>"""

    tracks = "".join(track_row(e) for e in open_rows) or '<tr><td colspan="7" class="empty">Nothing tracked yet.</td></tr>'
    lift = bt["comeback"]["comeback"] / max(bt["unlikely"]["comeback"], 1e-9)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rebound Radar · Ichnora</title>
<link rel="icon" href="brand/ichnora/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="brand/ichnora/apple-touch-icon.png">
<style>
:root{{--void:#020510;--glass:rgba(255,255,255,.025);--glass2:rgba(255,255,255,.05);--border:rgba(255,255,255,.06);
  --blue:#4facfe;--green:#00f5a0;--red:#ff4d6d;--gold:#ffd60a;--lime:#D4FF3A;--cyan:#3DE8FF;
  --text:#e8eaf6;--text2:#8b9cc8;--muted:#5a6b92;--font:'Inter',system-ui,sans-serif}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--void);color:var(--text);font-family:var(--font);line-height:1.6;min-height:100vh}}
.topnav{{display:flex;gap:8px;padding:20px 32px 0;border-bottom:1px solid rgba(255,255,255,.07);background:rgba(6,13,31,.95);position:sticky;top:0;z-index:100}}
.topnav a{{padding:10px 24px;border-radius:10px 10px 0 0;font-size:.85rem;font-weight:600;letter-spacing:.04em;color:#8b9cc8;text-decoration:none}}
.topnav a.on{{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-bottom:1px solid #060d1f;color:#e8eaf6}}
.header{{border-bottom:1px solid var(--border);padding:18px 40px;display:flex;align-items:center;gap:16px}}
.logo-mark{{width:36px;height:36px;border-radius:11px;display:block}}
.logo-name{{font-size:1.05rem;font-weight:800;background:linear-gradient(135deg,var(--lime),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.logo-sub{{font-size:.6rem;color:var(--muted);text-transform:uppercase}}
.hdr-right{{margin-left:auto;font-size:.72rem;color:var(--muted)}}
.horizon-tabs{{display:flex;gap:8px;padding:18px 32px 0;max-width:1400px;margin:0 auto;flex-wrap:wrap}}
.htab{{padding:10px 20px;border-radius:999px;font-size:.85rem;font-weight:700;text-decoration:none;color:#8b9cc8;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.02)}}
.htab:hover{{color:#e8eaf6;border-color:rgba(79,172,254,.4)}}
.htab.active{{color:#020510;background:linear-gradient(135deg,var(--lime),var(--cyan));border-color:transparent}}
.wrap{{max-width:1320px;margin:0 auto;padding:36px 32px 80px}}
.label{{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--blue);margin-bottom:10px}}
h1{{font-size:2rem;font-weight:900;letter-spacing:-.04em;line-height:1.15}}
h2{{font-size:1.35rem;font-weight:800;letter-spacing:-.03em;margin-bottom:4px}}
.sub{{font-size:.86rem;color:var(--text2);max-width:780px;margin:8px 0 22px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:26px 0 46px}}
.stat{{background:var(--glass);border:1px solid var(--border);border-radius:18px;padding:20px 22px}}
.stat-l{{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}}
.stat-v{{font-size:1.9rem;font-weight:900;letter-spacing:-.04em;line-height:1.2}}
.stat-s{{font-size:.72rem;color:var(--text2)}}
.group{{margin-bottom:30px}}
.g-head{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}}
.g-n{{font-weight:800;color:var(--text2)}} .g-blurb{{font-size:.78rem;color:var(--muted)}}
.pill{{font-size:.68rem;font-weight:700;padding:3px 10px;border-radius:20px;white-space:nowrap;border:1px solid}}
.p-comeback{{background:rgba(0,245,160,.1);color:var(--green);border-color:rgba(0,245,160,.25)}}
.p-watch{{background:rgba(255,214,10,.08);color:var(--gold);border-color:rgba(255,214,10,.22)}}
.p-unlikely{{background:rgba(255,77,109,.1);color:var(--red);border-color:rgba(255,77,109,.25)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}}
.card{{background:var(--glass);border:1px solid var(--border);border-radius:16px;padding:14px 16px;position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.card.v-comeback::before{{background:linear-gradient(90deg,var(--lime),var(--cyan))}}
.card.v-watch::before{{background:var(--gold)}} .card.v-unlikely::before{{background:var(--red)}}
.card-top{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.card-top img{{width:30px;height:30px;border-radius:50%;background:rgba(255,255,255,.05)}}
.c-name{{font-size:.8rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:105px}}
.c-sym{{font-size:.62rem;font-weight:800;color:var(--blue)}}
.c-score{{margin-left:auto;font-size:1.25rem;font-weight:900;letter-spacing:-.03em}}
.row{{display:flex;justify-content:space-between;font-size:.7rem;color:var(--text2);margin-bottom:2px}}
.bar{{margin-top:8px;height:4px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}}
.bar i{{display:block;height:100%;background:linear-gradient(90deg,var(--lime),var(--cyan))}}
.up{{color:var(--green)}} .dn{{color:var(--red)}}
.panel{{background:var(--glass);border:1px solid var(--border);border-radius:18px;padding:6px 18px;overflow-x:auto;margin-bottom:46px}}
table{{width:100%;border-collapse:collapse;font-size:.78rem}}
th{{color:var(--muted);font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;padding:12px 8px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:9px 8px;border-bottom:1px solid rgba(255,255,255,.03);color:var(--text2);white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
.tc{{display:flex;align-items:center;gap:9px}} .tc img{{width:22px;height:22px;border-radius:50%}}
.tc b{{color:var(--text);display:block;font-size:.78rem}} .tc small{{color:var(--muted);font-size:.64rem}}
.empty{{color:var(--muted);font-size:.8rem;padding:12px 0}}
.note{{font-size:.76rem;color:var(--muted);max-width:860px;margin-top:10px}}
.how{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}}
.how div{{background:var(--glass);border:1px solid var(--border);border-radius:16px;padding:16px 18px;font-size:.8rem;color:var(--text2)}}
.how b{{color:var(--text);display:block;margin-bottom:4px}}
@media(max-width:700px){{.wrap{{padding:28px 16px 60px}}.header{{padding:16px}}.topnav,.horizon-tabs{{padding-left:16px;padding-right:16px}}h1{{font-size:1.6rem}}}}
</style>
</head>
<body>
<div class="topnav"><a class="on" href="prediction_dashboard.html">&#128640; Crypto</a><a href="stock_dashboard.html">&#128200; Stocks</a></div>
<header class="header">
  <img class="logo-mark" src="brand/ichnora/icon.svg" alt="" width="36" height="36">
  <div><div class="logo-name">ichnora</div><div class="logo-sub">Every call leaves a footprint</div></div>
  <div class="hdr-right">Last updated: {esc(state['updated'])}</div>
</header>
<nav class="horizon-tabs" aria-label="Dashboards">
  <a href="prediction_dashboard.html" class="htab">&#128197; 7-Day Prediction</a>
  <a href="prediction_dashboard_3d.html" class="htab">&#9889; 3-Day Prediction</a>
  <a href="rebound_dashboard.html" class="htab active">&#129517; Rebound Radar</a>
</nav>
<div class="wrap">
  <div class="label">&#129517; Rebound Radar</div>
  <h1>Down 25% in a month. Coming back, or not?</h1>
  <p class="sub">Every top-200 coin that has fallen 25% or more in 30 days, ranked by its chance of getting back to
    where it was a month ago within the next 60 days. Each coin is tracked from the day it lands here and graded in public.</p>
  <div class="stats">
    <div class="stat"><div class="stat-l">On the radar today</div><div class="stat-v">{len(radar)}</div><div class="stat-s">of the top 200, down 25%+ in 30 days</div></div>
    <div class="stat"><div class="stat-l">Comeback candidates</div><div class="stat-v up">{len(groups['comeback'])}</div><div class="stat-s">top 30% of today's comeback scores</div></div>
    <div class="stat"><div class="stat-l">Being tracked</div><div class="stat-v">{len(state['open'])}</div><div class="stat-s">graded at recovery or after {rm.HORIZON} days</div></div>
    <div class="stat"><div class="stat-l">Backtest edge</div><div class="stat-v" style="color:var(--lime)">{lift:.1f}&times;</div><div class="stat-s">comeback rate, candidates vs unlikely</div></div>
  </div>

  <div class="label">Today &middot; {esc(today['date'])}</div>
  <h2>Today's radar</h2>
  <p class="sub">Big number = comeback score (0&ndash;100), this coin's rank among today's declining coins.
    "To recover" = how far it must rise to get back to its price 30 days ago.</p>
  {section('comeback', f"Back-tested: {pct(bt['comeback']['comeback'], 0, False)} got back to their pre-drop price within 60 days")}
  {section('watch', f"Back-tested: {pct(bt['watch']['comeback'], 0, False)} recovered")}
  {section('unlikely', f"Back-tested: only {pct(bt['unlikely']['comeback'], 0, False)} recovered")}

  <div class="label" style="margin-top:20px">Tracking</div>
  <h2>Every flagged coin, followed for {rm.HORIZON} days</h2>
  <p class="sub">The verdict is frozen on the day a coin is flagged, so the scorecard can't be rewritten later.</p>
  <div class="panel"><table>
    <tr><th>Coin</th><th>Flagged</th><th>Verdict then</th><th>Since flagged</th><th>Best since</th><th>To recover</th><th>Days left</th></tr>
    {tracks}
  </table></div>

  <div class="label">Scorecard</div>
  <h2>Does the verdict hold up?</h2>
  <p class="sub">Backtest: {held['events']:,} declines from {esc(held['period'])}, data the model never saw while training.
    Live: coins graded on this page.</p>
  <div class="panel"><table>
    <tr><th>Verdict</th><th>Backtest recovered</th><th>Bounced +25%</th><th>Fell 20%+ more</th>
      <th>Live graded</th><th>Recovered</th><th>Bounced</th><th>Fell further</th></tr>
    {"".join(score_row(v) for v in rm.VERDICTS)}
    <tr><td><b style="color:var(--text)">All declines</b></td><td>{pct(held['all']['comeback'], 1, False)}</td>
      <td>{pct(held['all']['bounce'], 0, False)}</td><td>{pct(held['all']['falling'], 0, False)}</td>
      <td>{len(state['closed'])}</td><td colspan="3"></td></tr>
  </table></div>
  <p class="note">Read it honestly: the model is good at telling which coins tend to win back their losses
    (candidates recover about {lift:.1f}&times; as often as "unlikely"), but about a third of every group still drops
    another 20%+. Candidates are often the more volatile coins: bigger swings both ways. The backtest only includes coins
    still in today's top 200, so coins that died are missing and real recovery rates are lower; the live scorecard has no such gap.</p>

  <div class="label" style="margin-top:40px">How it works</div>
  <div class="how">
    <div><b>1 &middot; Flag</b>Any top-200 coin (no stablecoins) down 25%+ over 30 days on CoinMarketCap data.</div>
    <div><b>2 &middot; Score</b>A model trained on 3 years of declines weighs 20 signals: depth of the drop, distance from its
      90-day low and high, size, volatility, volume, RSI, Bitcoin's trend and market mood.</div>
    <div><b>3 &middot; Rank</b>Today's declining coins are ranked against each other: top 30% Comeback candidate,
      middle 40% Watching, bottom 30% Unlikely to recover.</div>
    <div><b>4 &middot; Grade</b>Recovered when it gets back to its pre-drop price; otherwise after {rm.HORIZON} days:
      Bounced (+25% at some point), Fell further (20%+ lower) or Still down.</div>
  </div>
  <p class="note">An experiment, not financial advice. Crypto is volatile and coins can go to zero.</p>
</div>
</body>
</html>
"""


def main():
    model = rm.load()
    state = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {"open": {}, "closed": []}
    today = datetime.now(timezone.utc).date().isoformat()
    print("Rebound Radar: scanning top 200 for 30-day declines ...")
    coins, regime, radar = scan(model)
    print(f"  {len(radar)} coins down {abs(rm.DECLINE):.0%}+ in 30 days")
    update(state, coins, radar, today)
    state["updated"] = datetime.now(timezone.utc).strftime("%b %d, %Y at %H:%M UTC")
    state["today"] = {"date": today, "regime": regime, "coins": radar}
    state["scorecard"] = scorecard(state["closed"])
    DATA.write_text(json.dumps(state, indent=1), encoding="utf-8")
    PAGE.write_text(build_page(state, model), encoding="utf-8")
    print(f"  tracking {len(state['open'])}, graded {len(state['closed'])} -> {PAGE.name}")


if __name__ == "__main__":
    main()
