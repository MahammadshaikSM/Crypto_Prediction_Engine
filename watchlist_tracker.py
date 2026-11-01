#!/usr/bin/env python3
"""Stock watchlist: daily stats for every name, plus one weekly call graded a week later.

Daily run (Yahoo Finance daily closes, same source as stock_tracker.py):
  1. Download ~2 years of daily closes and volumes for the watchlist and SPY / QQQ.
  2. Stats per name: last close, 1-day / 1-week / 1-month / 3-month change, RSI 14, distance from the
     52-week high, 50- and 200-day trend, volume vs its 20-day average.
  3. Weekly call: for the latest Friday close without a call, rank the names with the rule in
     watchlist.py and record the top 5 with their Friday prices.
  4. Grade every call whose next Friday close is in: the 5 picks in equal weight (minus costs) vs the
     watchlist average, SPY and QQQ.
  5. Write watchlist_data.json and watchlist.html.

Calls are experimental: the rule showed no proven edge in testing (backtest/watchlist_results.json).
"""
import html, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import watchlist as wl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "watchlist_data.json"
PAGE = ROOT / "watchlist.html"
NAMES = {
    "SNDK": "Sandisk", "GOOGL": "Alphabet", "MU": "Micron", "INTC": "Intel", "PLTR": "Palantir", "NVDA": "NVIDIA",
    "NFLX": "Netflix", "MRVL": "Marvell", "SKHY": "SK hynix", "HOOD": "Robinhood", "CBRS": "Cerebras",
    "AMZN": "Amazon", "TSM": "TSMC", "QCOM": "Qualcomm", "IBM": "IBM", "DELL": "Dell", "MSTR": "Strategy",
    "MSFT": "Microsoft", "SPCX": "SpaceX", "AMD": "AMD", "TSLA": "Tesla", "META": "Meta",
    "DRAM": "Roundhill Memory ETF", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF",
}


def download():
    import yfinance as yf
    raw = yf.download(wl.TICKERS + wl.BENCHMARKS, period="2y", auto_adjust=True, progress=False)
    return raw["Close"].dropna(how="all"), raw["Volume"].dropna(how="all")


def rsi14(s):
    d = s.diff().iloc[-14:]
    g, l = d.clip(lower=0).mean(), (-d.clip(upper=0)).mean()
    return 100.0 if l == 0 else float(100 - 100 / (1 + g / l))


def stats(px, vol):
    rows = []
    for t in wl.TICKERS + wl.BENCHMARKS:
        s = px[t].dropna()
        if len(s) < 6:
            continue
        chg = lambda n: float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n else None
        v = vol[t].dropna()
        rows.append(dict(
            ticker=t, name=NAMES.get(t, t), benchmark=t in wl.BENCHMARKS, close=float(s.iloc[-1]), as_of=str(s.index[-1].date()),
            d1=chg(1), w1=chg(5), m1=chg(21), m3=chg(63), rsi=rsi14(s) if len(s) > 15 else None,
            from_high=float(s.iloc[-1] / s.iloc[-252:].max() - 1),
            above50=bool(s.iloc[-1] > s.iloc[-50:].mean()) if len(s) >= 50 else None,
            above200=bool(s.iloc[-1] > s.iloc[-200:].mean()) if len(s) >= 200 else None,
            vol_vs_20d=float(v.iloc[-1] / v.iloc[-21:-1].mean()) if len(v) > 21 and v.iloc[-21:-1].mean() > 0 else None,
            history_days=len(s)))
    return rows


def weekly_calls(state, px):
    fridays = [i for i, d in enumerate(px.index) if d.weekday() == 4]
    if fridays:
        f = fridays[-1]; day = str(px.index[f].date())
        if day not in state["calls"]:
            sc = wl.scores(px[wl.TICKERS], f)[wl.RULE].sort_values(ascending=False)
            pick = list(sc.index[:wl.TOP])
            state["calls"][day] = dict(
                date=day, rule=wl.RULE, picks=[dict(ticker=t, name=NAMES.get(t, t), price=float(px[t].iloc[f]),
                                                    score=round(float(sc[t]), 3)) for t in pick],
                ranking=[dict(ticker=t, score=round(float(v), 3)) for t, v in sc.items()],
                universe=list(sc.index))
    for day, c in state["calls"].items():
        if c.get("graded"):
            continue
        i = px.index.get_loc(px.index[[str(d.date()) == day for d in px.index]][0])
        nxt = [j for j in fridays if j > i]
        if not nxt:
            continue
        j = nxt[0]
        r = lambda t: float(px[t].iloc[j] / px[t].iloc[i] - 1)
        picks = [r(p["ticker"]) for p in c["picks"]]
        avg = [r(t) for t in c["universe"] if np.isfinite(px[t].iloc[j])]
        c.update(graded=str(px.index[j].date()), pick_returns=picks, basket=float(np.mean(picks)) - wl.COST,
                 watchlist=float(np.mean(avg)) - wl.COST, spy=r("SPY"), qqq=r("QQQ"))


def record(calls):
    g = [c for c in calls if c.get("graded")]
    if not g:
        return None
    e = [c["basket"] - c["watchlist"] for c in g]
    return dict(weeks=len(g), basket=float(np.mean([c["basket"] for c in g])), watchlist=float(np.mean([c["watchlist"] for c in g])),
                spy=float(np.mean([c["spy"] for c in g])), edge=float(np.mean(e)), beat=float(np.mean([x > 0 for x in e])))


# ---------- page ----------
def pct(x, d=1):
    return "&ndash;" if x is None else f"{x * 100:+.{d}f}%"


def esc(s):
    return html.escape(str(s))


def build_page(state, bt):
    rows = state["stats"]; calls = sorted(state["calls"].values(), key=lambda c: c["date"], reverse=True)
    latest = calls[0] if calls else None
    rec = record(calls)
    cls = lambda x: "" if x is None else ("up" if x > 0 else "dn" if x < 0 else "")
    def stat_row(r):
        trend = ("&#9650;" if r["above50"] else "&#9660;") if r["above50"] is not None else "&ndash;"
        trend_cls = "" if r["above50"] is None else ("up" if r["above50"] else "dn")
        rsi = "&ndash;" if r["rsi"] is None else f"{r['rsi']:.0f}"
        volx = "&ndash;" if r["vol_vs_20d"] is None else f"{r['vol_vs_20d']:.1f}&times;"
        return (f"<tr{' class=bm' if r['benchmark'] else ''}><td><b>{esc(r['ticker'])}</b><small>{esc(r['name'])}</small></td>"
                f"<td>${r['close']:,.2f}</td><td class='{cls(r['d1'])}'>{pct(r['d1'])}</td><td class='{cls(r['w1'])}'>{pct(r['w1'])}</td>"
                f"<td class='{cls(r['m1'])}'>{pct(r['m1'])}</td><td class='{cls(r['m3'])}'>{pct(r['m3'])}</td>"
                f"<td>{rsi}</td><td>{pct(r['from_high'], 0)}</td><td class='{trend_cls}'>{trend}</td><td>{volx}</td></tr>")
    order = sorted([r for r in rows if not r["benchmark"]], key=lambda r: -(r["w1"] or -9)) + [r for r in rows if r["benchmark"]]
    pick_cards = "".join(
        f"<div class='card'><span class='rk'>#{k + 1}</span><b>{esc(p['ticker'])}</b><small>{esc(p['name'])}</small>"
        f"<div class='row'><span>Call price</span><b>${p['price']:,.2f}</b></div>"
        f"<div class='row'><span>Score</span><b>{p['score'] * 100:.0f}</b></div></div>" for k, p in enumerate(latest["picks"])) if latest else ""
    graded = "".join(
        f"<tr><td>{c['date']} &rarr; {c['graded']}</td><td>{', '.join(esc(p['ticker']) for p in c['picks'])}</td>"
        f"<td class='{cls(c['basket'])}'>{pct(c['basket'])}</td><td>{pct(c['watchlist'])}</td><td>{pct(c['spy'])}</td>"
        f"<td class='{cls(c['basket'] - c['watchlist'])}'>{pct(c['basket'] - c['watchlist'])}</td></tr>"
        for c in calls if c.get("graded")) or "<tr><td colspan='6' class='empty'>The first weekly call is graded one week after it is made.</td></tr>"
    h = bt["heldout"][bt["chosen"]] if bt else None
    evidence = (f"In testing, this rule picked names that did {pct(h['edge'], 2)} a week better than the watchlist average "
                f"(Nov 2024 &ndash; Oct 2026), but the 90% range runs from {pct(h['lo'], 2)} to {pct(h['hi'], 2)}, and in 2021&ndash;2024 it was "
                f"{pct(bt['train'][bt['chosen']]['edge'], 2)}. No proven edge yet: treat it as a watchlist ranking, not a trade signal.") if h else ""
    live = (f"Live so far: {rec['weeks']} graded weeks, top 5 {pct(rec['basket'])} per week vs watchlist {pct(rec['watchlist'])} "
            f"(beat it {rec['beat']:.0%} of weeks).") if rec else "Live record starts with the first graded week."
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Watchlist · Ichnora</title>
<link rel="icon" href="brand/ichnora/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="brand/ichnora/apple-touch-icon.png">
<style>
:root{{--void:#020510;--glass:rgba(255,255,255,.025);--border:rgba(255,255,255,.06);--blue:#4facfe;--green:#00f5a0;--red:#ff4d6d;
  --gold:#ffd60a;--orange:#ff9a3d;--lime:#D4FF3A;--cyan:#3DE8FF;--text:#e8eaf6;--text2:#8b9cc8;--muted:#5a6b92;--font:'Inter',system-ui,sans-serif}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--void);color:var(--text);font-family:var(--font);line-height:1.6}}
.topnav{{display:flex;gap:8px;padding:20px 32px 0;border-bottom:1px solid rgba(255,255,255,.07);background:rgba(6,13,31,.95);position:sticky;top:0;z-index:100}}
.topnav a{{padding:10px 24px;border-radius:10px 10px 0 0;font-size:.85rem;font-weight:600;color:#8b9cc8;text-decoration:none}}
.topnav a.on{{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-bottom:1px solid #060d1f;color:#e8eaf6}}
.header{{border-bottom:1px solid var(--border);padding:18px 40px;display:flex;align-items:center;gap:16px}}
.logo-mark{{width:36px;height:36px;border-radius:11px}}
.logo-name{{font-size:1.05rem;font-weight:800;background:linear-gradient(135deg,var(--lime),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.logo-sub{{font-size:.6rem;color:var(--muted);text-transform:uppercase}}
.hdr-right{{margin-left:auto;font-size:.72rem;color:var(--muted)}}
.wrap{{max-width:1320px;margin:0 auto;padding:36px 32px 80px}}
.label{{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--blue);margin-bottom:10px}}
h1{{font-size:2rem;font-weight:900;letter-spacing:-.04em;line-height:1.15}} h2{{font-size:1.35rem;font-weight:800;letter-spacing:-.03em;margin:40px 0 4px}}
.sub{{font-size:.86rem;color:var(--text2);max-width:820px;margin:8px 0 18px}}
.warn{{border:1px solid rgba(255,154,61,.3);background:rgba(255,154,61,.07);border-radius:16px;padding:16px 20px;font-size:.84rem;color:var(--text2);margin:18px 0}}
.warn b{{color:var(--orange)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}}
.card{{background:var(--glass);border:1px solid var(--border);border-radius:16px;padding:14px 16px;position:relative}}
.card .rk{{position:absolute;top:10px;right:12px;font-size:.62rem;color:var(--muted);font-weight:800}}
.card b{{font-size:1.15rem;display:block}} .card small{{color:var(--muted);font-size:.7rem;display:block;margin-bottom:8px}}
.row{{display:flex;justify-content:space-between;font-size:.72rem;color:var(--text2)}} .row b{{font-size:.75rem;display:inline}}
.panel{{background:var(--glass);border:1px solid var(--border);border-radius:18px;padding:6px 16px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.78rem;font-variant-numeric:tabular-nums}}
th{{color:var(--muted);font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;padding:12px 8px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:9px 8px;border-bottom:1px solid rgba(255,255,255,.03);color:var(--text2);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} td b{{color:var(--text)}} td small{{display:block;color:var(--muted);font-size:.64rem}}
tr.bm td{{background:rgba(79,172,254,.04)}} tr:last-child td{{border-bottom:none}}
.up{{color:var(--green)}} .dn{{color:var(--red)}} .empty{{color:var(--muted);text-align:left}}
.note{{font-size:.76rem;color:var(--muted);max-width:900px;margin-top:12px}}
@media(max-width:700px){{.wrap{{padding:28px 16px 60px}}.header{{padding:16px}}.topnav{{padding:16px 16px 0}}h1{{font-size:1.6rem}}}}
</style>
</head>
<body>
<div class="topnav"><a href="prediction_dashboard.html">&#128640; Crypto</a><a href="stock_dashboard.html">&#128200; Stocks</a><a class="on" href="watchlist.html">&#11088; Watchlist</a></div>
<header class="header">
  <img class="logo-mark" src="brand/ichnora/icon.svg" alt="" width="36" height="36">
  <div><div class="logo-name">ichnora</div><div class="logo-sub">Every call leaves a footprint</div></div>
  <div class="hdr-right">Prices as of {esc(rows[0]['as_of']) if rows else '&ndash;'} &middot; updated {esc(state['updated'])}</div>
</header>
<div class="wrap">
  <div class="label">&#11088; Stock watchlist</div>
  <h1>23 stocks traded as crypto perpetuals</h1>
  <p class="sub">Real stock prices for the names you can trade as USDT perpetuals on crypto exchanges, with a weekly top-5 call
    graded one week later against the whole watchlist, the S&amp;P 500 (SPY) and the Nasdaq-100 (QQQ).</p>
  <div class="warn"><b>Experimental weekly call.</b> {evidence} {live}</div>
  <div class="warn"><b>About leverage.</b> Exchanges offer these perpetuals at 20&times;&ndash;100&times;. At 50&times;, a 2% move against you wipes out
    the position; several names here move more than that in a normal day. Perpetual prices can also drift from the real stock,
    and funding fees apply. Nothing here is financial advice.</div>

  <h2>This week's call</h2>
  <p class="sub">{f"Ranked on the {latest['date']} close, graded on the next Friday close. Rule: 12-month momentum, last week's losers and trend, averaged." if latest else "No call yet."}</p>
  <div class="cards">{pick_cards}</div>

  <h2>The whole watchlist</h2>
  <p class="sub">Sorted by 1-week change. Trend &#9650; = above its 50-day average. Volume = today vs its 20-day average.</p>
  <div class="panel"><table>
    <tr><th>Stock</th><th>Close</th><th>1 day</th><th>1 week</th><th>1 month</th><th>3 months</th><th>RSI</th><th>From 52w high</th><th>Trend</th><th>Volume</th></tr>
    {''.join(stat_row(r) for r in order)}
  </table></div>

  <h2>Graded weekly calls</h2>
  <div class="panel"><table>
    <tr><th>Week</th><th>Top 5</th><th>Top 5 return</th><th>Watchlist avg</th><th>SPY</th><th>vs watchlist</th></tr>
    {graded}
  </table></div>
  <p class="note">Returns use Friday closes, equal weight, minus 0.1% costs per name. Names join the ranking once they have six months
    of trading history (CBRS, SPCX, SKHY and DRAM listed in 2026). Data: Yahoo Finance daily closes. Not financial advice.</p>
</div>
</body>
</html>
"""


def main():
    state = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {"calls": {}}
    print("Watchlist: downloading daily prices ...")
    px, vol = download()
    state["stats"] = stats(px, vol)
    weekly_calls(state, px)
    state["updated"] = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    state["record"] = record(list(state["calls"].values()))
    DATA.write_text(json.dumps(state, indent=1), encoding="utf-8")
    try:
        bt = json.loads((ROOT / "backtest" / "watchlist_results.json").read_text())
    except (OSError, ValueError):
        bt = None
    PAGE.write_text(build_page(state, bt), encoding="utf-8")
    latest = max(state["calls"]) if state["calls"] else None
    print(f"  {len(state['stats'])} names, latest call {latest}: "
          + (", ".join(p["ticker"] for p in state["calls"][latest]["picks"]) if latest else "none"))


if __name__ == "__main__":
    main()
