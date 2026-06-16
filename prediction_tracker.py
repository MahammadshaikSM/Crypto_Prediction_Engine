#!/usr/bin/env python3
"""
Crypto Prediction Tracker
─────────────────────────
Runs daily to:
  1. Fetch top 200 coins from CoinGecko (with sparkline data)
  2. Run 5-factor model:
       * Volatility-Adjusted Momentum  30%  (mom7d / price_volatility)
       * Volume Surge                  20%  (volume / market_cap)
       * Momentum Acceleration         18%  (is today faster than avg daily?)
       * BTC-Relative Strength         17%  (coin vs BTC 7d return)
       * RSI Score                     15%  (penalise overbought >70, boost <30)
  3. Save today's prediction + price snapshot to prediction_data.json
  4. Evaluate any prediction made exactly 7 days ago
  5. Regenerate prediction_dashboard.html with all data embedded

Accuracy rule: a predicted coin is a "hit" if it appears in the
actual top-10 gainers 7 days later (prices measured from snapshot).
Also tracks "positive rate" = % of predicted coins that simply went up.
"""

import json, os, sys, time, math
import requests
from datetime import datetime, timedelta
from pathlib import Path

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
DATA_FILE    = SCRIPT_DIR / "prediction_data.json"
DASHBOARD    = SCRIPT_DIR / "prediction_dashboard.html"
API_KEY_FILE = SCRIPT_DIR / ".coingecko_key"

# -- CoinGecko URL -------------------------------------------------------------
# sparkline=true gives 7-day hourly prices (~168 points) for RSI + volatility
CG_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
    "&sparkline=true&price_change_percentage=7d"
)

# -- Stablecoin exclusion ------------------------------------------------------
STABLECOINS = {
    'usdt','usdc','busd','dai','tusd','usdp','usdd','frax','gusd','lusd',
    'susd','usdn','cusd','nusd','usdx','husd','ousd','musd','usdk','usds',
    'fdusd','pyusd','crvusd','mkusd','eurs','eurt','xaut','paxg','usde',
    'dola','ageur','alusd','usd0','wusd',
}

def is_stable(coin):
    sym = (coin.get('symbol') or '').lower()
    if sym in STABLECOINS:
        return True
    chg7  = abs(coin.get('price_change_percentage_7d_in_currency') or 0)
    chg24 = abs(coin.get('price_change_percentage_24h') or 0)
    return chg7 < 0.6 and chg24 < 0.3

# -- Fetch coins ---------------------------------------------------------------
def fetch_coins():
    api_key = None
    if API_KEY_FILE.exists():
        api_key = API_KEY_FILE.read_text().strip() or None

    url     = CG_URL + (f"&x_cg_demo_api_key={api_key}" if api_key else "")
    headers = {"x-cg-demo-api-key": api_key} if api_key else {}

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  Rate limited - waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return data[:200]
            raise ValueError("Empty or unexpected API response")
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(15)

    raise RuntimeError("Could not fetch coin data after 3 attempts.")

# -- Technical helpers ---------------------------------------------------------
def calc_rsi(prices, period=14):
    """RSI(14) from a list of prices. Returns 0-100, or 50 if insufficient data."""
    if not prices or len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    g = gains[-period:]
    l = losses[-period:]
    avg_g = sum(g) / period
    avg_l = sum(l) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))

def calc_volatility(prices):
    """Daily-return std-dev from hourly sparkline. Returns small positive float."""
    if not prices or len(prices) < 2:
        return 0.01
    daily = [prices[i] for i in range(0, len(prices), 24) if prices[i] > 0]
    if len(daily) < 2:
        return 0.01
    returns = [(daily[i] - daily[i-1]) / daily[i-1] for i in range(1, len(daily))]
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    return max(math.sqrt(variance), 0.0001)

def rsi_score(rsi):
    """Convert RSI to a 0-100 score. Penalise overbought >70, boost oversold <30."""
    if rsi >= 80:  return 0.0
    if rsi >= 70:  return 20.0
    if rsi >= 60:  return 50.0
    if rsi >= 40:  return 70.0
    if rsi >= 30:  return 85.0
    return 100.0

# -- Prediction model ----------------------------------------------------------
def run_prediction(coins):
    eligible = [
        c for c in coins
        if c.get('price_change_percentage_7d_in_currency') is not None
        and c.get('price_change_percentage_24h') is not None
        and (c.get('total_volume') or 0) > 0
        and (c.get('market_cap') or 0) > 0
        and not is_stable(c)
    ]
    if len(eligible) < 10:
        return []

    btc = next((c for c in coins if c['id'] == 'bitcoin'), None)
    btc_mom7d = (btc.get('price_change_percentage_7d_in_currency') or 0) if btc else 0

    scored = []
    for c in eligible:
        mom7d   = c['price_change_percentage_7d_in_currency']
        mom24h  = c['price_change_percentage_24h']
        vol_rat = c['total_volume'] / c['market_cap']

        sparkline_prices = (c.get('sparkline_in_7d') or {}).get('price') or []
        volatility = calc_volatility(sparkline_prices)
        rsi        = calc_rsi(sparkline_prices)

        vol_adj_mom  = mom7d / (volatility * 100 + 1)
        vol_surge    = vol_rat
        avg_daily    = mom7d / 7 if mom7d != 0 else 0
        accel_raw    = max(0.0, min(3.0, (mom24h / avg_daily) if avg_daily != 0 else 0))
        btc_relative = mom7d - btc_mom7d
        rsi_s        = rsi_score(rsi)

        scored.append(dict(
            coin=c, mom7d=mom7d, mom24h=mom24h, vol_rat=vol_rat,
            vol_adj_mom=vol_adj_mom, vol_surge=vol_surge,
            accel_raw=accel_raw, btc_relative=btc_relative,
            rsi=rsi, rsi_s=rsi_s, volatility=volatility,
        ))

    def normalize(items, key):
        vals = [x[key] for x in items]
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        for x in items:
            x[key + '_n'] = ((x[key] - mn) / rng * 100) if rng > 0 else 50.0

    normalize(scored, 'vol_adj_mom')
    normalize(scored, 'vol_surge')
    normalize(scored, 'accel_raw')
    normalize(scored, 'btc_relative')

    for x in scored:
        x['score'] = (
            0.30 * x['vol_adj_mom_n']  +
            0.20 * x['vol_surge_n']    +
            0.18 * x['accel_raw_n']    +
            0.17 * x['btc_relative_n'] +
            0.15 * x['rsi_s']
        )
        x['confidence'] = min(72.0, x['score'] * 0.72)

    top10 = sorted(scored, key=lambda x: x['score'], reverse=True)[:10]

    result = []
    for i, x in enumerate(top10):
        c = x['coin']
        result.append({
            'rank':                i + 1,
            'id':                  c['id'],
            'symbol':              c['symbol'],
            'name':                c['name'],
            'image':               c.get('image', ''),
            'price_at_prediction': c['current_price'],
            'score':               round(x['score'], 2),
            'confidence':          round(x['confidence'], 2),
            'mom7d':               round(x['mom7d'], 2),
            'mom24h':              round(x['mom24h'], 2),
            'vol_ratio_pct':       round(x['vol_rat'] * 100, 4),
            'btc_relative':        round(x['btc_relative'], 2),
            'rsi':                 round(x['rsi'], 1),
            'volatility_pct':      round(x['volatility'] * 100, 2),
        })
    return result

# -- Evaluate a past prediction ------------------------------------------------
def evaluate_prediction(pred_entry, current_coins):
    old_prices = pred_entry.get('price_snapshot', {})
    pred_coins = pred_entry['predicted_coins']
    pred_ids   = {c['id'] for c in pred_coins}

    cur_map    = {c['id']: c['current_price'] for c in current_coins}
    stable_ids = {c['id'] for c in current_coins if is_stable(c)}
    name_map   = {c['id']: {'name': c['name'], 'symbol': c['symbol']} for c in current_coins}

    all_returns = {}
    for cid, old_p in old_prices.items():
        if cid in cur_map and old_p and old_p > 0:
            all_returns[cid] = (cur_map[cid] - old_p) / old_p * 100

    filtered            = {k: v for k, v in all_returns.items() if k not in stable_ids}
    actual_top10_sorted = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:10]
    actual_top10_ids    = {k for k, _ in actual_top10_sorted}

    coin_results = []
    for pc in pred_coins:
        cid           = pc['id']
        actual_return = all_returns.get(cid)
        coin_results.append({
            'id':                  cid,
            'symbol':              pc['symbol'],
            'name':                pc['name'],
            'image':               pc.get('image', ''),
            'rank_predicted':      pc['rank'],
            'score':               pc.get('score'),
            'confidence':          pc.get('confidence'),
            'price_at_prediction': pc['price_at_prediction'],
            'price_at_evaluation': cur_map.get(cid),
            'actual_return_pct':   round(actual_return, 2) if actual_return is not None else None,
            'in_actual_top10':     cid in actual_top10_ids,
            'went_positive':       (actual_return or 0) > 0,
            'hit':                 cid in actual_top10_ids,
        })

    hits     = sum(1 for c in coin_results if c['hit'])
    positive = sum(1 for c in coin_results if c['went_positive'])

    actual_top10_list = [
        {
            'id':            cid,
            'name':          name_map.get(cid, {}).get('name', cid),
            'symbol':        name_map.get(cid, {}).get('symbol', ''),
            'return_pct':    round(ret, 2),
            'was_predicted': cid in pred_ids,
        }
        for cid, ret in actual_top10_sorted
    ]

    return {
        'prediction_date': pred_entry['date'],
        'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
        'hit_count':       hits,
        'hit_rate':        round(hits / 10 * 100, 1),
        'positive_count':  positive,
        'positive_rate':   round(positive / 10 * 100, 1),
        'coin_results':    coin_results,
        'actual_top10':    actual_top10_list,
    }

# -- Data persistence ----------------------------------------------------------
def load_data():
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  Warning: could not read {DATA_FILE}: {e}. Starting fresh.")
    return {'predictions': {}, 'evaluations': {}}

def save_data(data):
    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

# -- Dashboard HTML generator --------------------------------------------------
def generate_dashboard(data):
    predictions = data.get('predictions', {})
    evaluations = data.get('evaluations', {})

    total_preds  = len(predictions)
    total_evals  = len(evaluations)
    avg_hit_rate = (sum(e['hit_rate'] for e in evaluations.values()) / total_evals
                    if total_evals > 0 else 0)
    avg_pos_rate = (sum(e['positive_rate'] for e in evaluations.values()) / total_evals
                    if total_evals > 0 else 0)
    best_eval    = (max(evaluations.values(), key=lambda e: e['hit_rate'])
                    if evaluations else None)

    timeline = []
    for date in sorted(predictions.keys(), reverse=True):
        pred = predictions[date]
        ev   = next((e for e in evaluations.values()
                     if e.get('prediction_date') == date), None)
        timeline.append({
            'date':            date,
            'maturity_date':   pred.get('maturity_date', ''),
            'predicted_coins': pred.get('predicted_coins', []),
            'evaluation':      ev,
        })

    chart_data = {'labels': [], 'hit_rates': [], 'pos_rates': []}
    for ev in sorted(evaluations.values(), key=lambda e: e['prediction_date']):
        chart_data['labels'].append(ev['prediction_date'])
        chart_data['hit_rates'].append(ev['hit_rate'])
        chart_data['pos_rates'].append(ev['positive_rate'])

    tl_json    = json.dumps(timeline, ensure_ascii=False)
    chart_json = json.dumps(chart_data, ensure_ascii=False)
    updated_at = datetime.now().strftime('%b %d, %Y at %H:%M')
    hit_disp   = f"{avg_hit_rate:.1f}%" if total_evals else "-"
    pos_disp   = f"{avg_pos_rate:.1f}%" if total_evals else "-"
    best_disp  = (f"{best_eval['hit_rate']:.0f}% ({best_eval['prediction_date']})"
                  if best_eval else "-")

    DASHBOARD.write_text(
        _build_html(tl_json, chart_json, updated_at, hit_disp, pos_disp,
                    best_disp, total_preds),
        encoding='utf-8'
    )
    print(f"  Dashboard -> {DASHBOARD}")


def _build_html(tl_json, chart_json, updated_at, hit_disp, pos_disp,
                best_disp, total_preds):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prediction Accuracy Dashboard - CryptoTracker Pro</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --void:#020510; --deep:#070d1e; --surface:#0c1428;
  --glass:rgba(255,255,255,0.025); --glass2:rgba(255,255,255,0.05);
  --border:rgba(255,255,255,0.06);
  --blue:#4facfe; --purple2:#a78bfa;
  --teal:#00d4aa; --green:#00f5a0; --red:#ff4d6d; --gold:#ffd60a;
  --text:#e8eaf6; --text2:#8b9cc8; --muted:#3d4e72;
  --font:'Inter',system-ui,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--void);color:var(--text);font-family:var(--font);line-height:1.6;min-height:100vh}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--muted);border-radius:3px}
.header{background:rgba(2,5,16,.85);backdrop-filter:blur(24px);border-bottom:1px solid var(--border);
        padding:18px 40px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}
.logo-mark{width:36px;height:36px;background:linear-gradient(135deg,#4facfe,#00d4ff);
           border-radius:11px;display:flex;align-items:center;justify-content:center;
           font-weight:900;color:var(--void);font-size:1rem}
.logo-name{font-size:.95rem;font-weight:800;background:linear-gradient(135deg,#4facfe,#00d4ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo-sub{font-size:.6rem;color:var(--muted);text-transform:uppercase;-webkit-text-fill-color:var(--muted)}
.hdr-right{margin-left:auto;font-size:.72rem;color:var(--muted)}
.wrap{max-width:1320px;margin:0 auto;padding:40px 32px 80px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:48px}
.stat{background:var(--glass);border:1px solid var(--border);border-radius:20px;padding:24px 26px;position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:20px 20px 0 0}
.stat.c-blue::before{background:linear-gradient(90deg,#4facfe,#00d4ff)}
.stat.c-green::before{background:linear-gradient(90deg,#00f5a0,#00d4aa)}
.stat.c-gold::before{background:linear-gradient(90deg,#ffd60a,#fb923c)}
.stat.c-purple::before{background:linear-gradient(90deg,#7f5af0,#a78bfa)}
.stat-icon{font-size:1.1rem;margin-bottom:10px}
.stat-label{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:6px}
.stat-val{font-size:2rem;font-weight:900;letter-spacing:-.04em;line-height:1}
.stat-val.blue{color:var(--blue)} .stat-val.green{color:var(--green)}
.stat-val.gold{color:var(--gold)} .stat-val.purple{color:var(--purple2)}
.stat-sub{font-size:.72rem;color:var(--text2);margin-top:6px}
.section-label{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--blue);margin-bottom:14px}
.section-h{font-size:1.4rem;font-weight:800;letter-spacing:-.04em;margin-bottom:4px}
.section-sub{font-size:.82rem;color:var(--text2);margin-bottom:22px}
.model-badge{display:inline-flex;gap:8px;flex-wrap:wrap;margin-bottom:28px}
.mbadge{font-size:.65rem;font-weight:700;padding:4px 10px;border-radius:20px;
        background:rgba(79,172,254,.08);color:var(--blue);border:1px solid rgba(79,172,254,.15)}
.chart-card{background:var(--glass);border:1px solid var(--border);border-radius:20px;padding:28px 28px 24px;margin-bottom:48px}
.chart-wrap{position:relative;height:220px}
.timeline{display:flex;flex-direction:column;gap:14px}
.tl-entry{background:var(--glass);border:1px solid var(--border);border-radius:20px;overflow:hidden}
.tl-header{display:flex;align-items:center;gap:14px;padding:16px 22px;cursor:pointer;transition:background .15s}
.tl-header:hover{background:var(--glass2)}
.tl-date-badge{background:rgba(79,172,254,.1);border:1px solid rgba(79,172,254,.2);
               border-radius:12px;padding:4px 12px;font-size:.72rem;font-weight:700;color:var(--blue);white-space:nowrap}
.tl-arrow{margin-left:auto;color:var(--muted);transition:transform .2s}
.tl-header.open .tl-arrow{transform:rotate(180deg)}
.tl-status-badge{font-size:.67rem;font-weight:700;padding:3px 10px;border-radius:20px}
.badge-hit{background:rgba(0,245,160,.1);color:var(--green);border:1px solid rgba(0,245,160,.2)}
.badge-pending{background:rgba(255,214,10,.08);color:var(--gold);border:1px solid rgba(255,214,10,.2)}
.badge-miss{background:rgba(255,77,109,.1);color:var(--red);border:1px solid rgba(255,77,109,.2)}
.tl-body{display:none;border-top:1px solid var(--border)}
.tl-body.open{display:block}
.tl-cols{display:grid;grid-template-columns:1fr 1fr}
@media(max-width:800px){.tl-cols{grid-template-columns:1fr}}
.tl-col{padding:20px 22px}
.tl-col:first-child{border-right:1px solid var(--border)}
.tl-col-title{font-size:.64rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px}
.tl-col-title.pred{color:var(--blue)} .tl-col-title.actual{color:var(--green)}
.coin-row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.03)}
.coin-row:last-child{border-bottom:none}
.coin-img{width:24px;height:24px;border-radius:50%;object-fit:contain;background:rgba(255,255,255,.05);flex-shrink:0}
.coin-name{flex:1;font-size:.8rem;font-weight:600}
.coin-sym{font-size:.64rem;color:var(--muted);font-weight:700;text-transform:uppercase}
.coin-val{font-size:.78rem;font-weight:700;white-space:nowrap}
.coin-val.up{color:var(--green)} .coin-val.dn{color:var(--red)} .coin-val.neutral{color:var(--text2)}
.hit-icon{font-size:.9rem;flex-shrink:0;width:18px;text-align:center}
.rank-badge{font-size:.62rem;font-weight:800;color:var(--muted);width:22px;flex-shrink:0;text-align:center}
.rsi-tag{font-size:.6rem;font-weight:700;padding:2px 6px;border-radius:8px;white-space:nowrap}
.rsi-hot{background:rgba(255,77,109,.15);color:#ff4d6d}
.rsi-ok{background:rgba(0,245,160,.1);color:#00f5a0}
.rsi-neutral{background:rgba(255,255,255,.05);color:var(--muted)}
.eval-summary{background:rgba(0,0,0,.2);border-top:1px solid var(--border);
              padding:14px 22px;display:flex;gap:20px;flex-wrap:wrap;align-items:center}
.eval-bar-wrap{flex:1;min-width:160px}
.eval-bar-label{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
                color:var(--muted);margin-bottom:5px;display:flex;justify-content:space-between}
.eval-bar-track{height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden}
.eval-bar-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#4facfe,#00d4ff)}
.eval-bar-fill.pos{background:linear-gradient(90deg,#00f5a0,#00d4aa)}
.eval-pill{font-size:.72rem;font-weight:700;padding:5px 12px;border-radius:20px;white-space:nowrap}
.pill-green{background:rgba(0,245,160,.1);color:var(--green);border:1px solid rgba(0,245,160,.2)}
.pill-red{background:rgba(255,77,109,.1);color:var(--red);border:1px solid rgba(255,77,109,.2)}
.pill-gold{background:rgba(255,214,10,.08);color:var(--gold);border:1px solid rgba(255,214,10,.2)}
.pending-notice{padding:32px 22px;text-align:center;color:var(--muted)}
.no-data{text-align:center;padding:80px 24px;color:var(--muted)}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.entry-anim{animation:fadeUp .4s ease both}
</style>
</head>
<body>
<header class="header">
  <div class="logo-mark">CT</div>
  <div>
    <div class="logo-name">CryptoTracker Pro</div>
    <div class="logo-sub">Prediction Accuracy Dashboard</div>
  </div>
  <div class="hdr-right">Last updated: __UPDATED_AT__</div>
</header>
<div class="wrap">
  <div style="margin-bottom:32px">
    <div class="section-label">Model Performance</div>
    <h2 class="section-h">Prediction Accuracy</h2>
    <p class="section-sub">
      Each day the model predicts the top&nbsp;10 coins most likely to outperform over the next 7&nbsp;days.
      A "hit" = coin appeared in the real top&nbsp;10 gainers 7&nbsp;days later.
    </p>
    <div class="model-badge">
      <span class="mbadge">&#9889; Vol-Adj Momentum 30%</span>
      <span class="mbadge">&#128202; Volume Surge 20%</span>
      <span class="mbadge">&#128640; Momentum Accel 18%</span>
      <span class="mbadge">&#8383; BTC-Relative 17%</span>
      <span class="mbadge">&#128201; RSI Score 15%</span>
    </div>
  </div>
  <div class="stats">
    <div class="stat c-blue entry-anim" style="animation-delay:.05s">
      <div class="stat-icon">&#128202;</div>
      <div class="stat-label">Total Predictions</div>
      <div class="stat-val blue">__TOTAL_PREDS__</div>
      <div class="stat-sub">Days tracked so far</div>
    </div>
    <div class="stat c-green entry-anim" style="animation-delay:.1s">
      <div class="stat-icon">&#9989;</div>
      <div class="stat-label">Avg Hit Rate</div>
      <div class="stat-val green">__HIT_DISP__</div>
      <div class="stat-sub">Top-10 intersection accuracy</div>
    </div>
    <div class="stat c-gold entry-anim" style="animation-delay:.15s">
      <div class="stat-icon">&#128200;</div>
      <div class="stat-label">Avg Positive Rate</div>
      <div class="stat-val gold">__POS_DISP__</div>
      <div class="stat-sub">% predicted coins that went up</div>
    </div>
    <div class="stat c-purple entry-anim" style="animation-delay:.2s">
      <div class="stat-icon">&#127942;</div>
      <div class="stat-label">Best Day</div>
      <div class="stat-val purple" style="font-size:1.2rem;line-height:1.3">__BEST_DISP__</div>
      <div class="stat-sub">Highest single-day hit rate</div>
    </div>
  </div>
  <div id="chart-section" style="margin-bottom:48px">
    <div class="section-label">Accuracy Over Time</div>
    <h2 class="section-h" style="margin-bottom:18px">Hit Rate &amp; Positive Rate</h2>
    <div class="chart-card"><div class="chart-wrap"><canvas id="accChart"></canvas></div></div>
  </div>
  <div>
    <div class="section-label">Prediction History</div>
    <h2 class="section-h">All Predictions</h2>
    <p class="section-sub">Click any entry to expand. RSI tags: red=overbought (&gt;70), green=oversold (&lt;30).</p>
    <div class="timeline" id="timeline"></div>
    <div id="no-data" class="no-data" style="display:none">
      <div style="font-size:3rem;margin-bottom:16px">&#128302;</div>
      <div style="font-size:1rem;font-weight:700;color:var(--text2);margin-bottom:8px">No predictions yet</div>
      <div style="font-size:.82rem">Run <code>prediction_tracker.py</code> to make the first prediction.</div>
    </div>
  </div>
</div>
<script>
const TIMELINE   = __TL_JSON__;
const CHART_DATA = __CHART_JSON__;

(function() {
  if (!CHART_DATA.labels.length) { document.getElementById('chart-section').style.display='none'; return; }
  const ctx = document.getElementById('accChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: CHART_DATA.labels,
      datasets: [
        { label:'Hit Rate (top-10 %)', data:CHART_DATA.hit_rates,
          borderColor:'#4facfe', backgroundColor:'rgba(79,172,254,.1)',
          tension:0.4, fill:true, pointRadius:0, pointHoverRadius:4,
          pointBackgroundColor:'#4facfe', borderWidth:2 },
        { label:'Positive Rate (% up)', data:CHART_DATA.pos_rates,
          borderColor:'#00f5a0', backgroundColor:'rgba(0,245,160,.07)',
          tension:0.4, fill:true, pointRadius:0, pointHoverRadius:4,
          pointBackgroundColor:'#00f5a0', borderWidth:2 },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      interaction:{ mode:'index', intersect:false },
      plugins: {
        legend:{ labels:{ color:'#8b9cc8', font:{ size:12 }, boxWidth:12 } },
        tooltip:{ backgroundColor:'#0c1428', borderColor:'rgba(255,255,255,.1)', borderWidth:1,
                  titleColor:'#e8eaf6', bodyColor:'#8b9cc8',
                  callbacks:{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' } }
      },
      scales: {
        x:{ ticks:{ color:'#3d4e72', maxTicksLimit:10, maxRotation:0 },
            grid:{ color:'rgba(255,255,255,.04)' } },
        y:{ min:0, max:100, ticks:{ color:'#3d4e72', callback:v=>v+'%' },
            grid:{ color:'rgba(255,255,255,.04)' } }
      }
    }
  });
})();

function fmtDate(d) {
  if (!d) return '';
  const [y,m,day] = d.split('-');
  return new Date(+y,+m-1,+day).toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric',year:'numeric'});
}
function fmtPct(v) {
  if (v==null) return '-';
  return (v>=0?'+':'')+v.toFixed(2)+'%';
}
function coinImg(img,sym) {
  return `<img class="coin-img" src="${img}" alt="${sym}" onerror="this.style.visibility='hidden'" loading="lazy">`;
}
function rsiTag(rsi) {
  if (rsi==null) return '';
  const cls = rsi>70?'rsi-hot':rsi<30?'rsi-ok':'rsi-neutral';
  return `<span class="rsi-tag ${cls}">RSI ${rsi}</span>`;
}

const container = document.getElementById('timeline');
if (!TIMELINE.length) {
  document.getElementById('no-data').style.display='block';
} else {
  TIMELINE.forEach((entry, idx) => {
    const ev     = entry.evaluation;
    const mature = ev != null;

    let statusBadge = '';
    if (!mature) {
      const daysLeft = Math.ceil((new Date(entry.maturity_date)-new Date())/86400000);
      const dl = daysLeft>0?`Matures in ${daysLeft} day${daysLeft!==1?'s':''}`:'Evaluating soon...';
      statusBadge = `<span class="tl-status-badge badge-pending">&#9203; Pending &middot; ${dl}</span>`;
    } else {
      const hr=ev.hit_rate, cls=hr>=50?'badge-hit':'badge-miss', ico=hr>=50?'&#9989;':'&#10060;';
      statusBadge = `<span class="tl-status-badge ${cls}">${ico} ${ev.hit_count}/10 hits &middot; ${hr}%</span>`;
    }

    const predRows = entry.predicted_coins.map(c => {
      const hitIcon  = mature?(ev.coin_results.find(r=>r.id===c.id)?.hit?'&#9989;':'&#10060;'):'';
      const actualRet = mature?ev.coin_results.find(r=>r.id===c.id)?.actual_return_pct:null;
      const retStr = actualRet!=null
        ? `<span class="coin-val ${actualRet>=0?'up':'dn'}">${fmtPct(actualRet)}</span>`
        : `<span class="coin-val neutral">${fmtPct(c.confidence)} conf.</span>`;
      return `<div class="coin-row">
        ${coinImg(c.image,c.symbol)}
        <span class="rank-badge">#${c.rank}</span>
        <div class="coin-name">${c.name}<br><span class="coin-sym">${c.symbol.toUpperCase()}</span></div>
        ${rsiTag(c.rsi)}
        ${retStr}
        ${mature?`<span class="hit-icon">${hitIcon}</span>`:''}
      </div>`;
    }).join('');

    let actualCol = '';
    if (mature && ev.actual_top10) {
      const rows = ev.actual_top10.map(c => {
        const star = c.was_predicted?'&#11088;':'<span style="opacity:.2">&middot;</span>';
        return `<div class="coin-row">
          <div class="coin-name">${c.name}<br><span class="coin-sym">${c.symbol.toUpperCase()}</span></div>
          <span class="coin-val up">${fmtPct(c.return_pct)}</span>
          <span class="hit-icon">${star}</span>
        </div>`;
      }).join('');
      actualCol = `<div class="tl-col"><div class="tl-col-title actual">&#127942; Actual Top 10 (&#11088; = predicted)</div>${rows}</div>`;
    } else if (!mature) {
      actualCol = `<div class="tl-col"><div style="padding:32px;text-align:center;color:var(--muted)">
        <div style="font-size:2rem;margin-bottom:8px">&#9203;</div>
        <div style="font-size:.88rem;font-weight:600;color:var(--text2)">Not yet evaluated</div>
        <div style="font-size:.75rem">Matures ${fmtDate(entry.maturity_date)}</div>
      </div></div>`;
    }

    let evalSummary = '';
    if (mature) {
      const hp=ev.hit_rate, pp=ev.positive_rate;
      evalSummary = `<div class="eval-summary">
        <div class="eval-bar-wrap">
          <div class="eval-bar-label"><span>Hit Rate</span><span>${hp.toFixed(1)}%</span></div>
          <div class="eval-bar-track"><div class="eval-bar-fill" style="width:${hp}%"></div></div>
        </div>
        <div class="eval-bar-wrap">
          <div class="eval-bar-label"><span>Positive Rate</span><span>${pp.toFixed(1)}%</span></div>
          <div class="eval-bar-track"><div class="eval-bar-fill pos" style="width:${pp}%"></div></div>
        </div>
        <span class="eval-pill ${hp>=50?'pill-green':'pill-red'}">${ev.hit_count}/10 Hits</span>
        <span class="eval-pill ${pp>=50?'pill-green':'pill-gold'}">${ev.positive_count}/10 Positive</span>
      </div>`;
    }

    container.insertAdjacentHTML('beforeend', `
    <div class="tl-entry entry-anim" style="animation-delay:${idx*0.04}s">
      <div class="tl-header" onclick="toggle(this)" data-idx="${idx}">
        <span class="tl-date-badge">&#128197; ${fmtDate(entry.date)}</span>
        ${statusBadge}
        <span style="font-size:.7rem;color:var(--muted);white-space:nowrap">-&gt; Matures ${fmtDate(entry.maturity_date)}</span>
        <span class="tl-arrow">&#9660;</span>
      </div>
      <div class="tl-body" data-body="${idx}">
        <div class="tl-cols">
          <div class="tl-col"><div class="tl-col-title pred">&#128302; Predicted Top 10 on ${fmtDate(entry.date)}</div>${predRows}</div>
          ${actualCol}
        </div>
        ${evalSummary}
      </div>
    </div>\`);
  });

  const firstHeader = container.querySelector('.tl-header');
  if (firstHeader) toggle(firstHeader);
}

function toggle(header) {
  const idx  = header.getAttribute('data-idx');
  const body = document.querySelector(`[data-body="${idx}"]`);
  const open = body.classList.contains('open');
  body.classList.toggle('open', !open);
  header.classList.toggle('open', !open);
}
</script>
</body>
</html>"""
    return (html
        .replace('__UPDATED_AT__',  updated_at)
        .replace('__TOTAL_PREDS__', str(total_preds))
        .replace('__HIT_DISP__',    hit_disp)
        .replace('__POS_DISP__',    pos_disp)
        .replace('__BEST_DISP__',   best_disp)
        .replace('__TL_JSON__;',    tl_json + ';')
        .replace('__CHART_JSON__;', chart_json + ';'))


# -- Main execution -----------------------------------------------------------
def main():
    print("CryptoTracker Pro - Daily Prediction Pipeline")
    print("=" * 50)

    # 1. Fetch coins
    print("\n[1/4] Fetching top coins from CoinGecko...")
    coins = fetch_coins()
    if not coins:
        print("  ERROR: Failed to fetch coin data. Aborting.")
        sys.exit(1)
    print(f"  Fetched {len(coins)} coins.")

    # 2. Load existing data
    print("\n[2/4] Loading prediction history...")
    data = load_data()
    today_str = datetime.now().strftime('%Y-%m-%d')
    maturity  = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

    # 3. Evaluate past prediction (7 days ago)
    print("\n[3/4] Checking for prediction to evaluate...")
    eval_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if eval_date in data['predictions'] and eval_date not in data['evaluations']:
        print(f"  Evaluating prediction from {eval_date}...")
        ev = evaluate_prediction(data['predictions'][eval_date], coins)
        data['evaluations'][eval_date] = ev
        print(f"  Hit rate: {ev['hit_rate']}%  |  Positive rate: {ev['positive_rate']}%")
        print(f"  Hits: {ev['hit_count']}/10")
    else:
        print(f"  No prediction found for {eval_date} (or already evaluated).")

    # 4. Run today's prediction
    print("\n[4/4] Running today's prediction model...")
    if today_str in data['predictions']:
        print(f"  Prediction for {today_str} already exists. Updating...")
    top10 = run_prediction(coins)
    if not top10:
        print("  ERROR: Prediction model returned no results.")
        sys.exit(1)

    price_snapshot = {c['id']: c['current_price'] for c in coins}
    data['predictions'][today_str] = {
        'date':             today_str,
        'maturity_date':    maturity,
        'predicted_coins':  top10,
        'price_snapshot':   price_snapshot,
    }
    save_data(data)
    print(f"  Saved prediction for {today_str}.")

    # Print top 10
    print("\nToday's Predicted Top 10 (7-day outlook):")
    print("-" * 45)
    for coin in top10:
        r,s,n,c = coin["rank"],coin["symbol"].upper(),coin["name"],coin["confidence"]
        print(f"  #{r:2d}  {s:<8}  {n:<22}  confidence: {c:.1f}%")

    # 5. Regenerate dashboard
    print("\nRegenerating dashboard...")
    generate_dashboard(data)
    # 5. Regenerate dashboard
    print("\nRegenerating dashboard...")
    generate_dashboard(data)
