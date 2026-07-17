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
CMC_KEY_FILE = SCRIPT_DIR / ".cmc_key"

# -- CoinMarketCap URL (primary source) ─────────────────────────────────────────
CMC_URL = (
    "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    "?start=1&limit=500&convert=USD"
    "&aux=volume_24h,market_cap,percent_change_1h,percent_change_24h,percent_change_7d"
)

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

# -- Fetch coins (CMC primary → CoinGecko fallback) ----------------------------
def _fetch_cmc():
    """Fetch from CoinMarketCap. Returns CG-compatible list or None."""
    if not CMC_KEY_FILE.exists():
        return None
    cmc_key = CMC_KEY_FILE.read_text().strip()
    if not cmc_key:
        return None
    try:
        r = requests.get(
            CMC_URL,
            headers={"X-CMC_PRO_API_KEY": cmc_key, "Accept": "application/json"},
            timeout=20,
        )
        r.raise_for_status()
        body = r.json()
        raw  = body.get("data", [])
        if not raw:
            return None
        coins = []
        for c in raw:
            q = (c.get("quote") or {}).get("USD", {})
            coins.append({
                "id":            c.get("slug") or c.get("symbol", "").lower(),
                "symbol":        (c.get("symbol") or "").lower(),
                "name":          c.get("name", ""),
                "image":         (f"https://s2.coinmarketcap.com/static/img/coins/64x64/{c['id']}.png"
                                  if c.get("id") else ""),
                "current_price": q.get("price"),
                "market_cap":    q.get("market_cap"),
                "total_volume":  q.get("volume_24h"),
                "price_change_percentage_24h":            q.get("percent_change_24h"),
                "price_change_percentage_7d_in_currency": q.get("percent_change_7d"),
                "sparkline_in_7d": None,
            })
        print(f"  CMC: fetched {len(coins)} coins.")
        return coins
    except Exception as e:
        print(f"  CMC fetch failed: {e}")
        return None


def _fetch_coingecko():
    """Fetch from CoinGecko (includes sparkline for RSI/volatility)."""
    api_key = None
    if API_KEY_FILE.exists():
        api_key = API_KEY_FILE.read_text().strip() or None
    url     = CG_URL + (f"&x_cg_demo_api_key={api_key}" if api_key else "")
    headers = {"x-cg-demo-api-key": api_key} if api_key else {}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  CoinGecko rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                print(f"  CoinGecko: fetched {len(data)} coins (with sparkline).")
                return data[:250]
            raise ValueError("Empty response")
        except Exception as e:
            print(f"  CoinGecko attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(15)
    return None


def _enrich_with_sparkline(cmc_coins):
    """Add CoinGecko sparkline data to CMC coins (for RSI/volatility calc)."""
    api_key = None
    if API_KEY_FILE.exists():
        api_key = API_KEY_FILE.read_text().strip() or None
    try:
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
            "&sparkline=true&price_change_percentage=7d"
        )
        if api_key:
            url += f"&x_cg_demo_api_key={api_key}"
        headers = {"x-cg-demo-api-key": api_key} if api_key else {}
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        cg_data = r.json()
        if not isinstance(cg_data, list):
            return cmc_coins
        sparkline_map = {c["id"]: c.get("sparkline_in_7d") for c in cg_data}
        enriched = 0
        for c in cmc_coins:
            if c.get("sparkline_in_7d") is None:
                sp = sparkline_map.get(c["id"])
                if sp:
                    c["sparkline_in_7d"] = sp
                    enriched += 1
        print(f"  Enriched {enriched} CMC coins with CoinGecko sparklines.")
    except Exception as e:
        print(f"  Sparkline enrichment failed (RSI/vol will use defaults): {e}")
    return cmc_coins


def fetch_coins():
    """Primary: CoinMarketCap → fallback: CoinGecko."""
    coins = _fetch_cmc()
    if coins:
        coins = _enrich_with_sparkline(coins)
        return coins
    print("  Falling back to CoinGecko...")
    coins = _fetch_coingecko()
    if coins:
        return coins
    raise RuntimeError("Could not fetch coin data from any source.")

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
    normalize(scored, 'mom24h')
    normalize(scored, 'vol_surge')
    normalize(scored, 'btc_relative')

    for x in scored:
        x['score'] = (
            0.28 * x['vol_adj_mom_n']  +   # Vol-adjusted 7D momentum  (v3: 28%)
            0.04 * x['mom24h_n']       +   # Momentum acceleration     (v3: 4%)
            0.28 * x['btc_relative_n'] +   # BTC-relative strength     (v3: 28%)
            0.12 * x['vol_surge_n']    +   # Volume surge              (v3: 12%)
            0.28 * x['rsi_s']              # RSI safety filter         (v3: 28% +68.8% lift)
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
            'model_version':       'v3-3d',
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

    evaluations_3d  = data.get('evaluations_3d', {})
    eval_by_date    = {e['prediction_date']: e for e in evaluations.values()}
    eval_3d_by_date = {e['prediction_date']: e for e in evaluations_3d.values()}

    timeline = []
    for date in sorted(predictions.keys(), reverse=True):
        pred  = predictions[date]
        ev    = eval_by_date.get(date)
        ev_3d = eval_3d_by_date.get(date)
        timeline.append({
            'date':            date,
            'maturity_date':   pred.get('maturity_date', ''),
            'maturity_3d':     pred.get('maturity_3d', ''),
            'predicted_coins': pred.get('predicted_coins', []),
            'evaluation':      ev,
            'evaluation_3d':   ev_3d,
        })

    # Build chart across ALL prediction dates; None = pending (chart skips them)
    all_dates = sorted(predictions.keys())
    chart_data = {
        'labels': [], 'hit_rates': [], 'pos_rates': [],
        'hit_rates_3d': [], 'pos_rates_3d': [],
    }
    for d in all_dates:
        chart_data['labels'].append(d)
        ev7 = eval_by_date.get(d)
        ev3 = eval_3d_by_date.get(d)
        chart_data['hit_rates'].append(   ev7['hit_rate']      if ev7 else None)
        chart_data['pos_rates'].append(   ev7['positive_rate'] if ev7 else None)
        chart_data['hit_rates_3d'].append(ev3['hit_rate']      if ev3 else None)
        chart_data['pos_rates_3d'].append(ev3['positive_rate'] if ev3 else None)

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
/* --- tabs --- */
.tabs-bar{display:flex;gap:0;margin-bottom:32px;border-bottom:1px solid var(--border)}
.tab-btn{background:none;border:none;border-bottom:2px solid transparent;padding:10px 24px;font-size:.82rem;font-weight:700;color:var(--muted);cursor:pointer;margin-bottom:-1px;transition:color .15s,border-color .15s;font-family:var(--font);letter-spacing:.02em}
.tab-btn:hover{color:var(--text2)}
.tab-btn.active{color:var(--blue);border-bottom-color:var(--blue)}
/* --- insights --- */
.ins-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:24px}
.ins-card{background:var(--glass);border:1px solid var(--border);border-radius:20px;padding:22px 24px}
.ins-card-title{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--blue);margin-bottom:16px}
.ins-table{width:100%;border-collapse:collapse;font-size:.77rem}
.ins-table th{color:var(--muted);font-size:.59rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;padding:5px 4px 5px 0;text-align:left;border-bottom:1px solid var(--border)}
.ins-table td{padding:7px 4px 7px 0;border-bottom:1px solid rgba(255,255,255,.03);color:var(--text2);vertical-align:middle}
.ins-table td:last-child{text-align:right;font-weight:700;padding-right:0}
.ins-table tr:last-child td{border-bottom:none}
.factor-row{margin-bottom:14px}
.factor-label{display:flex;justify-content:space-between;font-size:.74rem;font-weight:600;margin-bottom:5px}
.factor-track{height:8px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
</style>
</head>
<body>
<div style="display:flex;gap:8px;padding:20px 32px 0;border-bottom:1px solid rgba(255,255,255,0.07);background:rgba(6,13,31,0.95);position:sticky;top:0;z-index:100">
  <a href="prediction_dashboard.html" style="padding:10px 24px;border-radius:10px 10px 0 0;font-size:.85rem;font-weight:600;letter-spacing:.04em;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-bottom:1px solid #060d1f;color:#e8eaf6;text-decoration:none">&#128640; Crypto</a>
  <a href="stock_dashboard.html" style="padding:10px 24px;border-radius:10px 10px 0 0;font-size:.85rem;font-weight:600;letter-spacing:.04em;color:#8b9cc8;text-decoration:none">&#128200; Stocks</a>
</div>
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
      <span class="mbadge">&#128640; 24H Momentum 28%</span>
      <span class="mbadge">&#8383; BTC-Relative 25%</span>
      <span class="mbadge">&#9889; Vol-Adj Momentum 22%</span>
      <span class="mbadge">&#128201; RSI Score 15%</span>
      <span class="mbadge">&#128202; Volume Surge 10%</span>
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
  <div class="tabs-bar">
    <button class="tab-btn active" onclick="switchTab('overview',this)">&#128202; Overview</button>
    <button class="tab-btn" onclick="switchTab('insights',this)">&#128300; Insights</button>
  </div>
  <div id="tab-overview">
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
  <div id="tab-insights" style="display:none">
    <div style="margin-bottom:32px">
      <div class="section-label">Data Analysis</div>
      <h2 class="section-h">4-Factor Model Insights</h2>
      <p class="section-sub">Computed from 6 months of backfill data. Shows which factors actually drive hits.</p>
    </div>
    <div class="ins-grid">
      <div class="ins-card">
        <div class="ins-card-title">&#9889; Factor Correlation with Hits</div>
        <div id="ins-factors"></div>
      </div>
      <div class="ins-card">
        <div class="ins-card-title">&#128197; Monthly Accuracy</div>
        <div id="ins-monthly"></div>
      </div>
      <div class="ins-card">
        <div class="ins-card-title">&#128302; Top Recurring Coins</div>
        <div id="ins-coins"></div>
      </div>
      <div class="ins-card">
        <div class="ins-card-title">&#127942; Best &amp; Worst Days</div>
        <div id="ins-bestworst"></div>
      </div>
    </div>
    <div class="ins-card" style="margin-bottom:40px">
      <div class="ins-card-title">&#128200; Data-Driven Optimal Weights</div>
      <div id="ins-weights"></div>
    </div>
  </div>
</div>
<script>
const TIMELINE   = __TL_JSON__;
const CHART_DATA = __CHART_JSON__;

window.addEventListener('load', function() {
  if (!CHART_DATA.labels.length) { document.getElementById('chart-section').style.display='none'; return; }
  const canvas = document.getElementById('accChart');
  const W = canvas.width  = canvas.parentElement.clientWidth || 600;
  const H = canvas.height = 220;
  const ctx = canvas.getContext('2d');
  const PAD = { top:20, right:20, bottom:40, left:44 };
  const cw = W - PAD.left - PAD.right;
  const ch = H - PAD.top  - PAD.bottom;
  const n  = CHART_DATA.labels.length;
  const xStep = n > 1 ? cw / (n - 1) : cw;

  function toY(v) { return PAD.top + ch - (v / 100) * ch; }
  function toX(i) { return PAD.left + (n > 1 ? i * xStep : cw / 2); }

  // grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth   = 1;
  [0,25,50,75,100].forEach(v => {
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cw, y); ctx.stroke();
    ctx.fillStyle = '#8b9cc8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(v + '%', PAD.left - 6, y + 4);
  });

  // x labels
  ctx.fillStyle = '#8b9cc8';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  CHART_DATA.labels.forEach((lbl, i) => {
    ctx.fillText(lbl.slice(5), toX(i), H - 8);
  });

  function drawLine(vals, color, fillColor) {
    const valid = vals.map((v,i)=>[i,v]).filter(([,v])=>v!=null);
    if (!valid.length) return;
    ctx.save();
    // line (gap-aware: moveTo on null, lineTo on value)
    ctx.beginPath();
    let gap = true;
    vals.forEach((v,i) => {
      if (v == null) { gap = true; return; }
      if (gap) { ctx.moveTo(toX(i), toY(v)); gap = false; }
      else ctx.lineTo(toX(i), toY(v));
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();
    // dots on valid points only
    valid.forEach(([i,v]) => {
      ctx.beginPath();
      ctx.arc(toX(i), toY(v), 3, 0, Math.PI*2);
      ctx.fillStyle = color; ctx.fill();
    });
    ctx.restore();
  }

  drawLine(CHART_DATA.pos_rates,    '#00f5a0', 'rgba(0,245,160,0.07)');
  drawLine(CHART_DATA.hit_rates,    '#4facfe', 'rgba(79,172,254,0.12)');
  drawLine(CHART_DATA.pos_rates_3d, '#ffd60a', 'rgba(255,214,10,0.07)');
  drawLine(CHART_DATA.hit_rates_3d, '#ff6b6b', 'rgba(255,107,107,0.10)');

  // legend
  [['#4facfe','7d Hit'],['#00f5a0','7d Positive'],['#ff6b6b','3d Hit'],['#ffd60a','3d Positive']].forEach(([c,l], i) => {
    const row = Math.floor(i/2), col = i%2;
    const lx = PAD.left + col * 130;
    const ly = 4 + row * 14;
    ctx.fillStyle = c;
    ctx.fillRect(lx, ly, 12, 3);
    ctx.fillStyle = '#8b9cc8';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(l, lx + 16, ly + 7);
  });
});

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
    // 3-day badge
    let badge3d = '';
    const ev3 = entry.evaluation_3d;
    if (ev3) {
      const hr3=ev3.hit_rate, cls3=hr3>=50?'badge-hit':'badge-miss', ico3=hr3>=50?'&#9989;':'&#10060;';
      badge3d = `<span class="tl-status-badge ${cls3}" style="font-size:.62rem;opacity:.85">3d: ${ico3} ${hr3}%</span>`;
    } else if (entry.maturity_3d) {
      const dl3 = Math.ceil((new Date(entry.maturity_3d)-new Date())/86400000);
      if (dl3 > 0) badge3d = `<span class="tl-status-badge badge-pending" style="font-size:.62rem;opacity:.75">3d: &#9203; ${dl3}d</span>`;
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
        ${badge3d}
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
    </div>`);
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

function switchTab(id, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  ['overview','insights'].forEach(t => {
    document.getElementById('tab-'+t).style.display = (t===id)?'block':'none';
  });
  if (id==='insights') buildInsights();
}

let insBuilt = false;
function buildInsights() {
  if (insBuilt) return;
  insBuilt = true;

  const evals = TIMELINE.filter(e => e.evaluation);
  if (!evals.length) {
    ['ins-factors','ins-monthly','ins-coins','ins-bestworst','ins-weights'].forEach(id => {
      document.getElementById(id).innerHTML = '<div style="color:var(--muted);font-size:.8rem;padding:12px 0">No evaluated data yet.</div>';
    });
    return;
  }

  const factors = [
    {key:'mom7d',         label:'7D Momentum'},
    {key:'mom24h',        label:'24H Momentum'},
    {key:'btc_relative',  label:'BTC-Relative'},
    {key:'vol_ratio_pct', label:'Volume Surge'},
  ];
  const fdata = {};
  factors.forEach(({key}) => { fdata[key] = {hSum:0,hN:0,mSum:0,mN:0}; });
  evals.forEach(entry => {
    const cr_map = {};
    (entry.evaluation.coin_results||[]).forEach(cr => { cr_map[cr.id]=cr; });
    entry.predicted_coins.forEach(c => {
      const cr = cr_map[c.id];
      if (!cr) return;
      factors.forEach(({key}) => {
        const val = c[key] != null ? c[key] : (key==='btc_relative' ? c['alignment'] : null);
        if (val == null) return;
        if (cr.hit) { fdata[key].hSum += val; fdata[key].hN++; }
        else        { fdata[key].mSum += val; fdata[key].mN++; }
      });
    });
  });
  const fdiffs = factors.map(({key,label}) => {
    const d = fdata[key];
    const hAvg = d.hN ? d.hSum/d.hN : 0;
    const mAvg = d.mN ? d.mSum/d.mN : 0;
    return {key, label, diff: hAvg - mAvg, hAvg, mAvg};
  }).sort((a,b) => b.diff - a.diff);
  const maxD = Math.max(...fdiffs.map(d => Math.abs(d.diff))) || 1;
  document.getElementById('ins-factors').innerHTML = fdiffs.map(d => {
    const pct = (Math.abs(d.diff)/maxD*100).toFixed(1);
    const col = d.diff >= 0 ? '#00f5a0' : '#ff4d6d';
    const sign = d.diff >= 0 ? '+' : '';
    return '<div class="factor-row">'
      + '<div class="factor-label"><span>'+d.label+'</span>'
      + '<span style="color:'+col+'">'+sign+d.diff.toFixed(1)+' avg diff</span></div>'
      + '<div class="factor-track"><div style="height:8px;border-radius:4px;width:'+pct+'%;background:'+col+'"></div></div>'
      + '<div style="font-size:.66rem;color:var(--muted);margin-top:3px">Hits avg: '+d.hAvg.toFixed(1)+' | Misses avg: '+d.mAvg.toFixed(1)+'</div>'
      + '</div>';
  }).join('');

  const monthly = {};
  evals.forEach(e => {
    const m = e.date.slice(0,7);
    if (!monthly[m]) monthly[m] = {hits:0,total:0,pos:0};
    monthly[m].hits  += e.evaluation.hit_count;
    monthly[m].total += 10;
    monthly[m].pos   += e.evaluation.positive_count;
  });
  const mRows = Object.keys(monthly).sort().map(m => {
    const d = monthly[m];
    const hr = (d.hits/d.total*100).toFixed(1);
    const pr = (d.pos/d.total*100).toFixed(1);
    const days = evals.filter(e => e.date.startsWith(m)).length;
    const [y,mo] = m.split('-');
    const lbl = new Date(+y,+mo-1,1).toLocaleDateString('en-US',{month:'short',year:'2-digit'});
    const hCol = +hr>=20?'#00f5a0':'#ff4d6d';
    return '<tr><td>'+lbl+'</td><td>'+days+'d</td>'
      +'<td style="color:'+hCol+'">'+hr+'%</td>'
      +'<td style="color:#ffd60a">'+pr+'%</td></tr>';
  }).join('');
  document.getElementById('ins-monthly').innerHTML =
    '<table class="ins-table"><tr><th>Month</th><th>Days</th><th>Hit%</th><th>Pos%</th></tr>'+mRows+'</table>';

  const coinCount = {}, coinMeta = {};
  evals.forEach(e => {
    e.predicted_coins.forEach(c => {
      coinCount[c.id] = (coinCount[c.id]||0)+1;
      coinMeta[c.id] = {name:c.name, sym:c.symbol};
    });
  });
  const topCoins = Object.keys(coinCount).sort((a,b)=>coinCount[b]-coinCount[a]).slice(0,10);
  const totalDays = evals.length;
  const cRows = topCoins.map(id => {
    const m = coinMeta[id]; const cnt = coinCount[id];
    const pct = (cnt/totalDays*100).toFixed(0);
    return '<tr><td>'+m.name+'<br><span style="font-size:.6rem;color:var(--muted)">'+m.sym.toUpperCase()+'</span></td>'
      +'<td>'+cnt+'/'+totalDays+' <span style="color:var(--blue);font-size:.7rem">('+pct+'%)</span></td></tr>';
  }).join('');
  document.getElementById('ins-coins').innerHTML =
    '<table class="ins-table"><tr><th>Coin</th><th>Appearances</th></tr>'+cRows+'</table>';

  const days = evals.map(e => ({date:e.date,hr:e.evaluation.hit_rate,hc:e.evaluation.hit_count}))
    .sort((a,b)=>b.hr-a.hr);
  const best5  = days.slice(0,5);
  const worst5 = days.slice(-5).reverse();
  function dayRows(arr,col) {
    return arr.map(d => {
      const [y,m,dy] = d.date.split('-');
      const lbl = new Date(+y,+m-1,+dy).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'2-digit'});
      return '<tr><td>'+lbl+'</td><td style="color:'+col+'">'+d.hc+'/10 ('+d.hr+'%)</td></tr>';
    }).join('');
  }
  document.getElementById('ins-bestworst').innerHTML =
    '<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#00f5a0;margin-bottom:6px">Top Days</div>'
    +'<table class="ins-table" style="margin-bottom:14px"><tr><th>Date</th><th>Hit Rate</th></tr>'+dayRows(best5,'#00f5a0')+'</table>'
    +'<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#ff4d6d;margin-bottom:6px">Toughest Days</div>'
    +'<table class="ins-table"><tr><th>Date</th><th>Hit Rate</th></tr>'+dayRows(worst5,'#ff4d6d')+'</table>';

  const posDiffs = fdiffs.map(d => Math.max(d.diff, 0.001));
  const tot = posDiffs.reduce((s,v)=>s+v,0)||1;
  const wBars = fdiffs.map((d,i) => {
    const w = (posDiffs[i]/tot*100).toFixed(1);
    return '<div class="factor-row">'
      +'<div class="factor-label"><span>'+d.label+'</span><span style="color:var(--blue)">'+w+'%</span></div>'
      +'<div class="factor-track"><div style="height:8px;border-radius:4px;width:'+w+'%;background:linear-gradient(90deg,#4facfe,#00d4ff)"></div></div>'
      +'</div>';
  }).join('');
  document.getElementById('ins-weights').innerHTML =
    '<p style="font-size:.79rem;color:var(--text2);margin-bottom:18px">Weights derived from correlation between each factor score and actual hit outcomes across 164 days of data.</p>'
    +wBars
    +'<div style="margin-top:16px;padding:14px;background:rgba(79,172,254,.06);border:1px solid rgba(79,172,254,.12);border-radius:12px;font-size:.75rem;color:var(--text2)">'
    +'&#128161; <strong style="color:var(--blue)">Current live model</strong> uses 24H Momentum + 7D + BTC-Relative + Volume + RSI (5-factor). '
    +'This analysis covers only the 4-factor backfill period (no RSI available).</div>';
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


def main():
    today = datetime.now().strftime('%Y-%m-%d')
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

    print(f"[{today}] Loading data...")
    data = load_data()
    data.setdefault('evaluations_3d', {})

    # Migrate: backfill maturity_3d for old predictions that lack it
    for _d, _pred in data['predictions'].items():
        if 'maturity_3d' not in _pred:
            _dt = datetime.strptime(_d, '%Y-%m-%d')
            _pred['maturity_3d'] = (_dt + timedelta(days=3)).strftime('%Y-%m-%d')

    print("  Fetching coins (CMC primary, CoinGecko fallback)...")
    coins = fetch_coins()
    print(f"  Fetched {len(coins)} coins.")

    # --- Evaluate 7-day-old prediction ---
    if seven_days_ago in data.get('predictions', {}) and seven_days_ago not in data.get('evaluations', {}):
        print(f"  Evaluating 7-day prediction from {seven_days_ago}...")
        ev = evaluate_prediction(data['predictions'][seven_days_ago], coins)
        data.setdefault('evaluations', {})[seven_days_ago] = ev
        print(f"  7d hit rate: {ev['hit_rate']}%  |  Positive rate: {ev['positive_rate']}%")

    # --- Evaluate 3-day-old prediction ---
    if (three_days_ago in data.get('predictions', {})
            and three_days_ago not in data.get('evaluations_3d', {})):
        print(f"  Evaluating 3-day prediction from {three_days_ago}...")
        ev3 = evaluate_prediction(data['predictions'][three_days_ago], coins)
        data.setdefault('evaluations_3d', {})[three_days_ago] = ev3
        print(f"  3d hit rate: {ev3['hit_rate']}%  |  Positive rate: {ev3['positive_rate']}%")

    # --- Run today's prediction (skip if already done today) ---
    if today not in data.get('predictions', {}):
        print("  Running prediction model...")
        top10 = run_prediction(coins)
        if not top10:
            print("  ERROR: prediction model returned no results.")
            sys.exit(1)

        price_snapshot = {c['id']: c['current_price'] for c in coins if c.get('current_price')}
        maturity   = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        maturity3d = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')

        data.setdefault('predictions', {})[today] = {
            'date':            today,
            'maturity_date':   maturity,
            'maturity_3d':     maturity3d,
            'predicted_coins': top10,
            'price_snapshot':  price_snapshot,
            'fetched_at':      datetime.now().isoformat(),
        }

        print(f"  Today's top 10:")
        for c in top10:
            print(f"    {c['rank']:>2}. {c['symbol'].upper():<10} confidence={c['confidence']}%")
    else:
        print(f"  Prediction for {today} already exists — skipping.")
        for c in data['predictions'][today]['predicted_coins']:
            print(f"    {c['rank']:>2}. {c['symbol'].upper():<10} confidence={c['confidence']}%")

    print("  Saving data...")
    save_data(data)

    print("  Regenerating dashboard...")
    generate_dashboard(data)

    print("Done ✓")


if __name__ == '__main__':
    main()
