"""Build proof.html: how the calls were tested, with the numbers, the ranges and the limits.

Reads backtest/evidence_results.json (matched baseline, independent windows, calibration, liquidity,
execution delay), backtest/trader_results.json (market light) and the live record from
prediction_data.json. Runs in the daily GitHub Action so the live numbers stay current.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import trader_stats  # noqa: E402


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pct(x, sign=True, d=1):
    return "&ndash;" if x is None else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def rng(z):
    return f"{pct(z['lo'])} to {pct(z['hi'])}"


def calib_svg(bins):
    W, H, P = 640, 260, 40
    top = max(max(b["predicted"], b["actual"]) for b in bins) * 1.15
    bw = (W - 2 * P) / len(bins)
    y = lambda v: H - P - v / top * (H - 2 * P)
    bars = ""
    for i, b in enumerate(bins):
        x0 = P + i * bw
        bars += (f'<rect x="{x0 + bw * .14:.1f}" y="{y(b["predicted"]):.1f}" width="{bw * .32:.1f}" height="{H - P - y(b["predicted"]):.1f}" fill="#8b9cc8" rx="3"/>'
                 f'<rect x="{x0 + bw * .52:.1f}" y="{y(b["actual"]):.1f}" width="{bw * .32:.1f}" height="{H - P - y(b["actual"]):.1f}" fill="#D4FF3A" rx="3"/>'
                 f'<text x="{x0 + bw / 2:.1f}" y="{H - P + 16}" fill="#A1A1AC" font-size="11" text-anchor="middle">{b["lo"] * 100:.0f}&#8211;{b["hi"] * 100:.0f}%</text>'
                 f'<text x="{x0 + bw * .68:.1f}" y="{y(b["actual"]) - 5:.1f}" fill="#D4FF3A" font-size="10.5" text-anchor="middle">{b["actual"] * 100:.0f}%</text>')
    return (f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto" role="img" '
            f'aria-label="Predicted vs actual chance of a top-10 finish">{bars}'
            f'<line x1="{P}" x2="{W - P}" y1="{H - P}" y2="{H - P}" stroke="#ffffff22"/></svg>'
            '<p class="cap"><span style="color:#8b9cc8">&#9632; what the score said</span> &nbsp; '
            '<span style="color:#D4FF3A">&#9632; what happened</span> &middot; grouped by score, all ~200 coins, held-out days</p>')


def main():
    ev, tr = load("backtest/evidence_results.json"), load("backtest/trader_results.json")
    live = trader_stats.stats(load("prediction_data.json"), 7)["all"]
    e7, e3 = ev["7"], ev["3"]
    i7, i3 = e7["independent"], e3["independent"]
    lt = tr["7"]["light"]
    now = datetime.now(timezone.utc).strftime("%b %d, %Y")

    def baseline_rows(e, h):
        return f"""
        <tr><td>{h}-day picks</td><td>{pct(e['avg']['picks'])}</td><td>{pct(e['up']['picks'], False)}</td><td>{pct(e['hit']['picks'], False)}</td></tr>
        <tr><td>Matched random coins</td><td>{pct(e['avg']['matched'])}</td><td>{pct(e['up']['matched'], False)}</td><td>{pct(e['hit']['matched'], False)}</td></tr>
        <tr><td>Top-200 average</td><td>{pct(e['avg']['market'])}</td><td>{pct(e['up']['market'], False)}</td><td>5.7%</td></tr>
        <tr><td>Bitcoin</td><td>{pct(e['avg']['btc'])}</td><td>&ndash;</td><td>&ndash;</td></tr>"""

    def indep_rows(i, h):
        out = ""
        for k, label in (("vs matched", "vs matched random coins"), ("vs top-200 avg", "vs top-200 average"), ("vs BTC", "vs Bitcoin")):
            z = i[k]
            out += (f"<tr><td>{h}-day {label}</td><td><b>{pct(z['mean'])}</b></td><td>{rng(z)}</td>"
                    f"<td>{z['windows']}</td><td>{z['offsets_positive']}/{z['offsets']}</td></tr>")
        return out

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proof · Ichnora</title>
<meta name="description" content="How Ichnora's crypto calls were tested: matched baselines, independent weeks with confidence ranges, calibration, liquidity and costs. Including what does not work.">
<meta name="theme-color" content="#07070A">
<link rel="icon" href="brand/ichnora/favicon.svg" type="image/svg+xml">
<link rel="icon" href="brand/ichnora/favicon-32.png" sizes="32x32">
<link rel="apple-touch-icon" href="brand/ichnora/apple-touch-icon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="site/style.css">
<style>
  .pf-h {{ font: 800 clamp(40px, 6vw, 76px)/.95 var(--display); letter-spacing: -.04em; margin-top: 14px; }}
  .pf-h em {{ font-style: normal; background: var(--brand-grad); -webkit-background-clip: text; background-clip: text; color: transparent; }}
  .verdicts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 30px; }}
  .vd {{ padding: 22px 24px; border-radius: 20px; background: var(--surface); border: 1px solid var(--line); }}
  .vd.good {{ border-color: #D4FF3A55; }} .vd.warn {{ border-color: #ff9a3d66; }}
  .vd .k {{ font: 600 12px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--fg-2); }}
  .vd b.big {{ display: block; font: 800 40px var(--display); letter-spacing: -.03em; margin: 6px 0; }}
  .vd.good b.big {{ color: var(--lime); }} .vd.warn b.big {{ color: #ff9a3d; }}
  .vd p {{ color: var(--fg-2); font-size: 15px; margin: 0; }}
  h2.sec {{ font: 800 30px var(--display); letter-spacing: -.02em; margin-top: 60px; }}
  .sec-sub {{ color: var(--fg-2); max-width: 780px; margin-top: 8px; }}
  .tbl {{ width: 100%; border-collapse: collapse; margin-top: 18px; font-variant-numeric: tabular-nums; }}
  .tblwrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 6px 18px; margin-top: 18px; }}
  .tbl th {{ text-align: left; font: 600 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--fg-2); padding: 12px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  .tbl td {{ padding: 11px 10px; border-bottom: 1px solid #ffffff0a; white-space: nowrap; }}
  .tbl tr:last-child td {{ border-bottom: 0; }}
  .tbl td:first-child {{ color: var(--fg); }} .tbl td {{ color: var(--fg-2); }} .tbl b {{ color: var(--fg); }}
  .plain {{ margin-top: 16px; padding: 16px 20px; border-left: 3px solid var(--lime); background: #D4FF3A0a; border-radius: 0 14px 14px 0; color: var(--fg); max-width: 860px; }}
  .plain.warn {{ border-color: #ff9a3d; background: #ff9a3d0d; }}
  .cap {{ font-size: 13px; color: var(--fg-2); margin-top: 8px; }}
  ul.lim {{ margin-top: 14px; padding-left: 20px; color: var(--fg-2); display: grid; gap: 10px; max-width: 860px; }}
  ul.lim b {{ color: var(--fg); }}
</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="ambient" aria-hidden="true"><i class="a"></i><i class="b"></i></div>
<header class="nav">
  <div class="wrap">
    <a class="brand" href="./"><img class="brand-icon" src="brand/ichnora/icon.svg" alt="" width="34" height="34">ichnora</a>
    <nav class="nav-links" aria-label="Primary"><a href="./#today">Today</a><a href="call.html">Calls</a><a href="proof.html" aria-current="page">Proof</a><a href="rebound_dashboard.html">Rebound</a><a href="api.html">API</a></nav>
    <a class="btn lime sm" href="prediction_dashboard.html">Live dashboard ↗</a>
  </div>
</header>

<main id="main">
  <section class="block" style="padding-top:56px">
    <div class="wrap">
      <span class="chip">Tested on a year the model never saw &middot; updated {now}</span>
      <h1 class="pf-h">Don't trust us. <em>Check the tests.</em></h1>
      <p class="lede" style="margin-top:18px">Every model looks good on the data it was built on. So we froze the models on data up to
        Sep 2025 and tested them on the next twelve months ({e7['period'][0]} to {e7['period'][1]}), a falling market where Bitcoin lost about 30%.
        All returns are after 0.3% trading costs per coin. Here is what held up, and what didn't.</p>

      <div class="verdicts">
        <div class="vd good"><span class="k">7-day calls &middot; main signal</span><b class="big">{pct(i7['vs matched']['mean'])}</b>
          <p>per call above comparable random coins (90% range {rng(i7['vs matched'])}), positive from every start day.
          Passed all four tests.</p></div>
        <div class="vd warn"><span class="k">3-day calls &middot; experimental</span><b class="big">{pct(i3['vs matched']['mean'])}</b>
          <p>per call above comparable coins, but the range includes zero ({rng(i3['vs matched'])}) and at 1% trading costs the average
          shrinks to {pct(e3['cost_sensitivity']['0.01'])}. A watchlist, not a trade.</p></div>
        <div class="vd"><span class="k">Live record &middot; 7-day</span><b class="big" style="color:var(--fg)">{pct(live['avg'])}</b>
          <p>average per call on this site ({live['calls']} graded calls) vs {pct(live['market_avg'])} for the top-200 average and
          {pct(live['btc_avg'])} for Bitcoin. Earlier calls came from older model versions.</p></div>
      </div>

      <h2 class="sec">1 &middot; Is it just picking wild coins?</h2>
      <p class="sec-sub">Momentum models like volatile coins, and volatile coins land in the top-10 gainers more often by chance.
        So each pick is compared with a random coin from the same day with similar volatility and size (200 random draws per day).</p>
      <div class="tblwrap"><table class="tbl">
        <tr><th></th><th>Avg return per call</th><th>Coins that went up</th><th>Top-10 hit rate</th></tr>
        {baseline_rows(e7, 7)}
      </table></div>
      <p class="plain">Matched random coins hit the top 10 more often than pure chance ({pct(e7['hit']['matched'], False)} vs 5.7%), so part of the
        headline hit rate <b>is</b> volatility. But they lost money on average ({pct(e7['avg']['matched'])}) while the picks made
        {pct(e7['avg']['picks'])}. The edge in returns is not explained by volatility or size.</p>
      <p class="plain warn">The picks do <b>not</b> go up more often: {pct(e7['up']['picks'], False)} of picked coins rose vs {pct(e7['up']['market'], False)} of all coins.
        The gain comes from the winners rising much further than the losers fall. Hold all 10 in equal size; single picks are a coin flip.</p>

      <h2 class="sec">2 &middot; Is it luck from a few lucky weeks?</h2>
      <p class="sec-sub">Daily 7-day calls overlap, so 356 calls are really about {i7['vs matched']['windows']} independent weeks. We re-ran the
        test on non-overlapping windows only, for every possible start day, and resampled the weeks 4,000 times to get a 90% range.</p>
      <div class="tblwrap"><table class="tbl">
        <tr><th>Edge per call</th><th>Average</th><th>90% range</th><th>Independent windows</th><th>Start days positive</th></tr>
        {indep_rows(i7, 7)}{indep_rows(i3, 3)}
      </table></div>
      <p class="plain">For the 7-day calls the whole 90% range is above zero against every yardstick, and every one of the 7 possible
        start days is positive. For the 3-day calls the range touches zero against comparable coins and Bitcoin: not proven.</p>

      <h2 class="sec">3 &middot; Does the score mean what it says?</h2>
      <p class="sec-sub">The 7-day score is the model's chance that a coin finishes in the top 10 of ~200. A good score should come true about
        as often as it says.</p>
      <div class="tblwrap" style="padding:18px">{calib_svg(e7['calibration']['bins'])}</div>
      <p class="plain">The 10 picks scored {pct(e7['calibration']['picks_predicted'], False)} on average and actually hit
        {pct(e7['calibration']['picks_actual'], False)} of the time. Lower scores run a little optimistic (they say ~4%, reality ~3%),
        so read small scores as "unlikely", not as exact odds.</p>

      <h2 class="sec">4 &middot; Can you actually trade it?</h2>
      <div class="tblwrap"><table class="tbl">
        <tr><th>7-day calls</th><th>Result</th></tr>
        <tr><td>Median 24h trading volume of picks</td><td><b>${e7['liquidity']['median_volume_picks'] / 1e6:,.0f}M</b> (all coins ${e7['liquidity']['median_volume_all'] / 1e6:,.0f}M)</td></tr>
        <tr><td>Picks under $1M / $5M daily volume</td><td>{pct(e7['liquidity']['under_1m'], False)} / {pct(e7['liquidity']['under_5m'], False)}</td></tr>
        <tr><td>Thin coins (&lt;$1M) swapped for the next pick</td><td>{pct(e7['liquidity']['liquid_only'])} per call</td></tr>
        <tr><td>At 0.3% / 1% / 2% round-trip costs</td><td>{pct(e7['cost_sensitivity']['0.003'])} / {pct(e7['cost_sensitivity']['0.01'])} / {pct(e7['cost_sensitivity']['0.02'])} per call</td></tr>
        <tr><td>Bought a full day after the signal</td><td>{pct(e7['delay_1day']['avg'])} per call (edge vs market {rng(e7['delay_1day']['vs_market'])})</td></tr>
      </table></div>
      <p class="plain">The picks are as liquid as the average coin, survive 2% trading costs, and still work if you buy a day late.</p>

      <h2 class="sec">The market light</h2>
      <p class="sec-sub">TRADE when Bitcoin is flat or up over 7 days, CAUTION when it is down. Held-out year, 7-day calls:
        {pct(lt['trade']['basket_mean'])} per call on TRADE days vs {pct(lt['caution']['basket_mean'])} on CAUTION days, with smaller drops on TRADE
        days. The difference is likely but not proven (roughly 88&ndash;94% in resampling), the site's live record shows the opposite so far,
        and it had no effect on 3-day calls. One input, not a rule.</p>

      <h2 class="sec">What could still be wrong</h2>
      <ul class="lim">
        <li><b>One test year.</b> Twelve months, mostly a falling market. Other conditions may behave differently.</li>
        <li><b>Survivor data.</b> The history holds coins that are in today's top 200, so coins that collapsed and dropped out are missing.
          This flatters every basket, picks and comparisons alike; the comparisons above are fairer than the raw returns.</li>
        <li><b>Big-winner dependence.</b> A few large moves drive the average. A stretch without them will look poor.</li>
        <li><b>Live record mixes model versions.</b> Calls before Oct 20, 2026 used earlier models. The live numbers will settle the question over time.</li>
      </ul>
      <p class="sec-sub" style="margin-top:28px">Everything is graded in public: <a href="call.html">every call</a>,
        <a href="prediction_dashboard.html">7-day dashboard</a>, <a href="api/v1/track-record.json">track record as JSON</a>.
        Not financial advice; crypto can lose all of its value.</p>
    </div>
  </section>
</main>

<footer class="legal">
  <div class="wrap">
    <div class="who"><b>Ichnora</b><span>Data from <a href="https://coinmarketcap.com/" target="_blank" rel="noopener">CoinMarketCap</a></span></div>
    <ol>
      <li>Backtests freeze each model on data up to Sep 30, 2025 and score it only on later days. Returns are equal-weight baskets after 0.3% costs per coin unless stated.</li>
      <li>Personal research project. Not a broker, adviser or exchange. Nothing here is financial advice; crypto can lose all of its value.</li>
    </ol>
  </div>
</footer>
</body>
</html>
"""
    (ROOT / "proof.html").write_text(page, encoding="utf-8")
    print("proof.html written")


if __name__ == "__main__":
    main()
