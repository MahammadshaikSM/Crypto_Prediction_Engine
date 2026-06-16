#!/usr/bin/env python3
"""
backfill_predictions.py
=======================
Fetches 6 months of historical CoinGecko data and backfills prediction_data.json
with past predictions + evaluations using the 4-factor model.

(RSI and Volatility are skipped — they require sparkline which needs a paid key.)

Run once from CMD:
    cd "C:/Users/moham/OneDrive\Documents\Claude\Projects\Crypto App"
    python backfill_predictions.py

Takes ~10 minutes due to API rate limits. Don't close the window while it runs.
"""

import json, time, sys
from datetime import datetime, timedelta
from pathlib import Path
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE  = Path('prediction_data.json')
KEY_FILE   = Path('.coingecko_key')
DASH_FILE  = Path('prediction_dashboard.html')
DAYS_BACK  = 180    # 6 months of history
RATE_SLEEP = 2.5    # seconds between coin fetches (free tier safe)

STABLECOINS = {
    'usdt','usdc','busd','dai','tusd','usdp','usdd','frax','gusd','lusd',
    'susd','usdn','cusd','nusd','usdx','husd','ousd','musd','usdk','usds',
    'fdusd','pyusd','crvusd','mkusd','eurs','eurt','xaut','paxg','usde',
    'dola','ageur','alusd','usd0','wusd',
}

# ── API helper ──────────────────────────────────────────────────────────────────
def get_api_key():
    if KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding='utf-8').strip()
        if k:
            return k
    return None

def cg_get(path, params=None, api_key=None, retries=3):
    base = 'https://api.coingecko.com/api/v3'
    headers = {}
    if params is None:
        params = {}
    if api_key:
        headers['x-cg-demo-api-key'] = api_key
        params['x_cg_demo_api_key'] = api_key
    for attempt in range(retries):
        try:
            r = requests.get(base + path, params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                print('  Rate limited — waiting 70s...')
                time.sleep(70)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f'  API error: {e}')
                return None
            time.sleep(8)
    return None

# ── Helpers ─────────────────────────────────────────────────────────────────────
def is_stable(symbol):
    return symbol.lower() in STABLECOINS

def normalize(values):
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng == 0:
        return [50.0] * len(values)
    return [(v - mn) / rng * 100 for v in values]

def date_str(dt):
    return dt.strftime('%Y-%m-%d')

def parse_chart(raw, key):
    """Convert [[ts_ms, val], ...] to {date_str: val}."""
    out = {}
    for ts_ms, val in raw.get(key, []):
        d = datetime.utcfromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d')
        out[d] = val
    return out

# ── Step 1: Fetch top 200 coin list ────────────────────────────────────────────
def fetch_coin_list(api_key):
    print('[1/4] Fetching top 200 coins...')
    data = cg_get('/coins/markets', {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 200,
        'page': 1,
        'sparkline': 'false',
    }, api_key)
    if not data:
        print('ERROR: Could not fetch coin list.')
        sys.exit(1)
    print(f'  Got {len(data)} coins.')
    return data

# ── Step 2: Fetch historical market chart for each coin ─────────────────────────
def fetch_all_history(coins, api_key):
    print(f'\n[2/4] Fetching {DAYS_BACK}-day history for {len(coins)} coins...')
    print(f'  This takes ~{int(len(coins) * RATE_SLEEP / 60)} minutes. Please wait.')

    # daily[date_str][coin_id] = {price, volume, market_cap}
    daily = {}
    meta  = {}  # coin_id -> {name, symbol, image}

    for i, coin in enumerate(coins):
        cid = coin['id']
        meta[cid] = {
            'name':   coin['name'],
            'symbol': coin['symbol'],
            'image':  coin.get('image', ''),
        }

        raw = cg_get(f'/coins/{cid}/market_chart', {
            'vs_currency': 'usd',
            'days': DAYS_BACK,
        }, api_key)

        if raw:
            prices   = parse_chart(raw, 'prices')
            volumes  = parse_chart(raw, 'total_volumes')
            mcaps    = parse_chart(raw, 'market_caps')
            for d in prices:
                if d not in daily:
                    daily[d] = {}
                daily[d][cid] = {
                    'price':      prices.get(d),
                    'volume':     volumes.get(d),
                    'market_cap': mcaps.get(d),
                }

        done = i + 1
        if done % 10 == 0 or done == len(coins):
            print(f'  {done}/{len(coins)} coins fetched...')
        time.sleep(RATE_SLEEP)

    print(f'  History covers {len(daily)} dates.')
    return daily, meta

# ── Step 3: Score one historical day ───────────────────────────────────────────
def score_day(target_date, daily, meta):
    """
    4-factor model for a past date (no RSI/sparkline available for history):
      24H Momentum  31%  (strongest predictor from correlation analysis)
      BTC-Relative  30%  (near-equal to 24H)
      7D Momentum   29%  (plain, not vol-adjusted — no sparkline)
      Volume Surge  10%  (weak signal, kept as filter)
    Weights derived from 6-month correlation study on actual hit outcomes.
    """
    d_today     = date_str(target_date)
    d_yesterday = date_str(target_date - timedelta(days=1))
    d_7ago      = date_str(target_date - timedelta(days=7))

    today_data     = daily.get(d_today, {})
    yesterday_data = daily.get(d_yesterday, {})
    ago7_data      = daily.get(d_7ago, {})

    if not today_data or not ago7_data:
        return None, None

    # BTC reference
    btc_today = today_data.get('bitcoin', {}).get('price')
    btc_7ago  = ago7_data.get('bitcoin', {}).get('price')
    btc_7d    = ((btc_today - btc_7ago) / btc_7ago * 100) if (btc_today and btc_7ago) else 0

    eligible = []
    for cid, m in meta.items():
        if is_stable(m['symbol']):
            continue
        p_now  = today_data.get(cid, {}).get('price')
        p_7ago = ago7_data.get(cid, {}).get('price')
        p_yday = yesterday_data.get(cid, {}).get('price')
        vol    = today_data.get(cid, {}).get('volume')
        mcap   = today_data.get(cid, {}).get('market_cap')

        if not (p_now and p_7ago and mcap and mcap > 0):
            continue

        mom7d  = (p_now - p_7ago) / p_7ago * 100
        mom24h = ((p_now - p_yday) / p_yday * 100) if p_yday else 0

        if abs(mom7d) < 0.6 and abs(mom24h) < 0.3:
            continue  # unlisted stable

        vol_rat  = (vol / mcap) if vol else 0
        avg_day  = mom7d / 7
        accel    = max(0.0, min(3.0, (mom24h / avg_day) if avg_day != 0 else 0))
        btc_rel  = mom7d - btc_7d

        eligible.append({
            'id': cid, 'symbol': m['symbol'], 'name': m['name'],
            'image': m['image'], 'price': p_now,
            'mom7d': mom7d, 'mom24h': mom24h,
            'vol_rat': vol_rat, 'accel': accel, 'btc_rel': btc_rel,
        })

    if len(eligible) < 10:
        return None, None

    m7   = normalize([e['mom7d']   for e in eligible])
    vr   = normalize([e['vol_rat'] for e in eligible])
    ac   = normalize([e['accel']   for e in eligible])
    br   = normalize([e['btc_rel'] for e in eligible])

    for i, e in enumerate(eligible):
        e['score']      = 0.29*m7[i] + 0.31*ac[i] + 0.30*br[i] + 0.10*vr[i]
        e['confidence'] = min(72.0, e['score'] * 0.72)

    top10 = sorted(eligible, key=lambda x: x['score'], reverse=True)[:10]

    predicted_coins = []
    for rank, e in enumerate(top10, 1):
        predicted_coins.append({
            'rank':                rank,
            'id':                  e['id'],
            'symbol':              e['symbol'],
            'name':                e['name'],
            'image':               e['image'],
            'price_at_prediction': round(e['price'], 8),
            'score':               round(e['score'], 2),
            'confidence':          round(e['confidence'], 2),
            'mom7d':               round(e['mom7d'], 2),
            'mom24h':              round(e['mom24h'], 2),
            'vol_ratio_pct':       round(e['vol_rat'] * 100, 4),
            'btc_relative':        round(e['btc_rel'], 2),
            'rsi':                 50.0,
            'volatility_pct':      0.0,
            'model_version':       '4-factor-backfill',
        })

    # price snapshot of ALL coins for evaluation later
    price_snapshot = {
        cid: today_data[cid]['price']
        for cid in today_data
        if today_data[cid].get('price')
    }

    return predicted_coins, price_snapshot

# ── Step 4: Evaluate a past prediction ─────────────────────────────────────────
def evaluate_past(pred_entry, eval_date, daily, meta):
    d_eval    = date_str(eval_date)
    eval_data = daily.get(d_eval, {})
    if not eval_data:
        return None

    old_prices = pred_entry.get('price_snapshot', {})
    pred_coins = pred_entry['predicted_coins']
    pred_ids   = {c['id'] for c in pred_coins}

    # Returns for all coins at eval date vs prediction date
    all_returns = {}
    for cid, old_p in old_prices.items():
        new_p = eval_data.get(cid, {}).get('price')
        if new_p and old_p and old_p > 0:
            all_returns[cid] = (new_p - old_p) / old_p * 100

    # Filter stables
    all_returns = {
        k: v for k, v in all_returns.items()
        if not is_stable(meta.get(k, {}).get('symbol', ''))
    }

    actual_sorted    = sorted(all_returns.items(), key=lambda kv: kv[1], reverse=True)[:10]
    actual_top10_ids = {k for k, _ in actual_sorted}

    coin_results = []
    hits = positive = 0
    for pc in pred_coins:
        cid = pc['id']
        ret = all_returns.get(cid)
        went_up = (ret or 0) > 0
        hit     = cid in actual_top10_ids
        if hit:      hits     += 1
        if went_up:  positive += 1
        coin_results.append({
            'id':                  cid,
            'symbol':              pc['symbol'],
            'name':                pc['name'],
            'image':               pc.get('image', ''),
            'rank_predicted':      pc['rank'],
            'score':               pc.get('score'),
            'confidence':          pc.get('confidence'),
            'price_at_prediction': pc['price_at_prediction'],
            'price_at_evaluation': eval_data.get(cid, {}).get('price'),
            'actual_return_pct':   round(ret, 2) if ret is not None else None,
            'in_actual_top10':     hit,
            'went_positive':       went_up,
            'hit':                 hit,
        })

    actual_top10_list = [
        {
            'id':            cid,
            'name':          meta.get(cid, {}).get('name', cid),
            'symbol':        meta.get(cid, {}).get('symbol', ''),
            'return_pct':    round(ret, 2),
            'was_predicted': cid in pred_ids,
        }
        for cid, ret in actual_sorted
    ]

    return {
        'prediction_date': pred_entry['date'],
        'evaluation_date': d_eval,
        'hit_count':       hits,
        'hit_rate':        round(hits / 10 * 100, 1),
        'positive_count':  positive,
        'positive_rate':   round(positive / 10 * 100, 1),
        'coin_results':    coin_results,
        'actual_top10':    actual_top10_list,
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 55)
    print('  CryptoTracker Pro — Historical Backfill')
    print('=' * 55)

    api_key = get_api_key()
    print(f'  API key: {"found" if api_key else "not found (public endpoint)"}')

    # Load existing data (don't overwrite live predictions)
    existing = {'predictions': {}, 'evaluations': {}}
    if DATA_FILE.exists():
        try:
            existing = json.loads(DATA_FILE.read_text(encoding='utf-8'))
            print(f'  Existing data: {len(existing["predictions"])} predictions, '
                  f'{len(existing["evaluations"])} evaluations')
        except Exception as e:
            print(f'  Warning: could not read existing data: {e}')

    # Fetch coin list and history
    coins   = fetch_coin_list(api_key)
    daily, meta = fetch_all_history(coins, api_key)

    # Date range: from DAYS_BACK ago to yesterday
    today     = datetime.utcnow().date()
    start_dt  = today - timedelta(days=DAYS_BACK - 10)
    end_dt    = today - timedelta(days=1)  # don't backfill today (live model handles it)

    print(f'\n[3/4] Generating predictions from {start_dt} to {end_dt}...')
    new_preds = 0
    current_dt = start_dt
    while current_dt <= end_dt:
        d = date_str(current_dt)
        if d in existing['predictions']:
            current_dt += timedelta(days=1)
            continue  # skip — live model already has this date

        predicted_coins, price_snapshot = score_day(current_dt, daily, meta)
        if predicted_coins:
            maturity = date_str(current_dt + timedelta(days=7))
            existing['predictions'][d] = {
                'date':            d,
                'maturity_date':   maturity,
                'predicted_coins': predicted_coins,
                'price_snapshot':  price_snapshot,
            }
            new_preds += 1

        current_dt += timedelta(days=1)

    print(f'  Added {new_preds} new predictions.')

    # Evaluate all predictions that have matured (>= 7 days old)
    print(f'\n[4/4] Evaluating matured predictions...')
    new_evals = 0
    for d, pred in sorted(existing['predictions'].items()):
        if d in existing['evaluations']:
            continue  # already evaluated
        pred_dt  = datetime.strptime(d, '%Y-%m-%d').date()
        eval_dt  = pred_dt + timedelta(days=7)
        if eval_dt > today:
            continue  # not matured yet

        ev = evaluate_past(pred, eval_dt, daily, meta)
        if ev:
            existing['evaluations'][d] = ev
            new_evals += 1

    print(f'  Added {new_evals} new evaluations.')

    # Save
    DATA_FILE.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f'\n  Saved {DATA_FILE}')
    print(f'  Total: {len(existing["predictions"])} predictions, '
          f'{len(existing["evaluations"])} evaluations')

    # Regenerate dashboard
    try:
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location('pt', 'prediction_tracker.py')
        pt   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pt)
        pt.generate_dashboard(existing)
        print('  Dashboard regenerated.')
    except Exception as e:
        print(f'  Note: could not regenerate dashboard automatically: {e}')
        print('  Run prediction_tracker.py once to update the dashboard.')

    print('\nDone! Commit and push to GitHub to see the results live.')
    print('  git add prediction_data.json prediction_dashboard.html')
    print('  git commit -m "data: backfill 6 months of predictions"')
    print('  git push origin master')

if __name__ == '__main__':
    main()
