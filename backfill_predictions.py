#!/usr/bin/env python3
"""
backfill_predictions.py  (v2 — full 200-coin dataset)
======================================================
Fetches 6 months of CoinGecko history and produces TWO outputs:

  prediction_data.json   — top-10 predictions per day (for dashboard)
  all_coins_scores.json  — ALL eligible coins' factor scores + evaluation
                           outcomes per day (for weight optimisation)

Run once from CMD:
    cd "C:/Users/moham/OneDrive/Documents/Claude/Projects/Crypto App"
    python backfill_predictions.py

Takes ~30 minutes due to API rate limits. Don't close the window.

Set OVERWRITE_BACKFILL = True to regenerate even if data already exists.
"""

import json, time, sys
from datetime import datetime, timedelta
from pathlib import Path
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE        = Path('prediction_data.json')
ALL_SCORES_FILE  = Path('all_coins_scores.json')
KEY_FILE         = Path('.coingecko_key')
DAYS_BACK        = 180    # 6 months of history
RATE_SLEEP       = 2.5    # seconds between coin fetches (free tier safe)
OVERWRITE_BACKFILL = True # set False to skip dates already in prediction_data.json

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
        if k: return k
    return None

def cg_get(path, params=None, api_key=None, retries=3):
    base = 'https://api.coingecko.com/api/v3'
    headers = {}
    if params is None: params = {}
    if api_key:
        headers['x-cg-demo-api-key'] = api_key
        params['x_cg_demo_api_key']  = api_key
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
        'order':       'market_cap_desc',
        'per_page':    200,
        'page':        1,
        'sparkline':   'false',
    }, api_key)
    if not data:
        print('ERROR: Could not fetch coin list.')
        sys.exit(1)
    print(f'  Got {len(data)} coins.')
    return data

# ── Step 2: Fetch historical market chart for each coin ─────────────────────────
def fetch_all_history(coins, api_key):
    print(f'\n[2/4] Fetching {DAYS_BACK}-day history for {len(coins)} coins...')
    print(f'  Estimated time: ~{int(len(coins) * RATE_SLEEP / 60)} minutes. Please wait.\n')

    daily = {}   # daily[date_str][coin_id] = {price, volume, market_cap}
    meta  = {}   # coin_id -> {name, symbol, image}

    for i, coin in enumerate(coins):
        cid = coin['id']
        meta[cid] = {
            'name':   coin['name'],
            'symbol': coin['symbol'],
            'image':  coin.get('image', ''),
        }

        raw = cg_get(f'/coins/{cid}/market_chart', {
            'vs_currency': 'usd',
            'days':        DAYS_BACK,
        }, api_key)

        if raw:
            prices  = parse_chart(raw, 'prices')
            volumes = parse_chart(raw, 'total_volumes')
            mcaps   = parse_chart(raw, 'market_caps')
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

    print(f'\n  History covers {len(daily)} dates.')
    return daily, meta

# ── Step 3: Score ALL coins for one historical day ──────────────────────────────
def score_day(target_date, daily, meta):
    """
    5-factor model (4 available from history + 3D momentum):
      24H Momentum  — mom24h
      3D Momentum   — mom3d  (NEW: sustained short-term signal)
      7D Momentum   — mom7d
      BTC-Relative  — btc_rel
      Volume Surge  — vol_ratio

    Returns:
      all_coins     — list of ALL eligible coins with raw factor scores
      top10         — top 10 coins (for dashboard predictions)
      price_snapshot — all coin prices (for evaluation)
    """
    d_today = date_str(target_date)
    d_1ago  = date_str(target_date - timedelta(days=1))
    d_3ago  = date_str(target_date - timedelta(days=3))
    d_7ago  = date_str(target_date - timedelta(days=7))

    today_data = daily.get(d_today, {})
    d1_data    = daily.get(d_1ago,  {})
    d3_data    = daily.get(d_3ago,  {})
    d7_data    = daily.get(d_7ago,  {})

    if not today_data or not d7_data:
        return None, None, None

    # BTC reference for btc_relative
    btc_now  = today_data.get('bitcoin', {}).get('price')
    btc_7ago = d7_data.get('bitcoin', {}).get('price')
    btc_7d   = ((btc_now - btc_7ago) / btc_7ago * 100) if (btc_now and btc_7ago) else 0

    eligible = []
    for cid, m in meta.items():
        if is_stable(m['symbol']):
            continue

        p_now  = today_data.get(cid, {}).get('price')
        p_1ago = d1_data.get(cid,  {}).get('price')
        p_3ago = d3_data.get(cid,  {}).get('price')
        p_7ago = d7_data.get(cid,  {}).get('price')
        vol    = today_data.get(cid, {}).get('volume')
        mcap   = today_data.get(cid, {}).get('market_cap')

        if not (p_now and p_7ago and mcap and mcap > 0):
            continue

        mom7d  = (p_now - p_7ago) / p_7ago * 100
        mom3d  = ((p_now - p_3ago) / p_3ago * 100) if p_3ago else mom7d / 2
        mom24h = ((p_now - p_1ago) / p_1ago * 100) if p_1ago else 0

        # Filter near-stablecoins
        if abs(mom7d) < 0.6 and abs(mom24h) < 0.3:
            continue

        vol_ratio = (vol / mcap) if vol else 0
        btc_rel   = mom7d - btc_7d

        eligible.append({
            'id':        cid,
            'symbol':    m['symbol'],
            'name':      m['name'],
            'image':     m['image'],
            'price':     p_now,
            'mom7d':     mom7d,
            'mom3d':     mom3d,
            'mom24h':    mom24h,
            'vol_ratio': vol_ratio,
            'btc_rel':   btc_rel,
        })

    if len(eligible) < 10:
        return None, None, None

    # Rank-normalize all factors (0-100)
    m7  = normalize([e['mom7d']     for e in eligible])
    m3  = normalize([e['mom3d']     for e in eligible])
    m24 = normalize([e['mom24h']    for e in eligible])
    vr  = normalize([e['vol_ratio'] for e in eligible])
    br  = normalize([e['btc_rel']   for e in eligible])

    for i, e in enumerate(eligible):
        e['mom7d_n']     = round(m7[i],  2)
        e['mom3d_n']     = round(m3[i],  2)
        e['mom24h_n']    = round(m24[i], 2)
        e['vol_ratio_n'] = round(vr[i],  2)
        e['btc_rel_n']   = round(br[i],  2)
        # Default score using current best-guess weights
        # (optimizer will find the real best weights later)
        e['score'] = (
            0.28 * m24[i] +
            0.22 * m3[i]  +
            0.22 * m7[i]  +
            0.22 * br[i]  +
            0.06 * vr[i]
        )
        e['confidence'] = min(72.0, e['score'] * 0.72)

    all_sorted = sorted(eligible, key=lambda x: x['score'], reverse=True)
    top10      = all_sorted[:10]
    top10_ids  = {e['id'] for e in top10}

    # Build prediction records for top 10 (for dashboard)
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
            'mom7d':               round(e['mom7d'],     2),
            'mom3d':               round(e['mom3d'],     2),
            'mom24h':              round(e['mom24h'],    2),
            'vol_ratio_pct':       round(e['vol_ratio'] * 100, 4),
            'btc_relative':        round(e['btc_rel'],   2),
            'rsi':                 50.0,
            'volatility_pct':      0.0,
            'model_version':       '5-factor-backfill-v2',
        })

    # Build full-pool records for ALL eligible coins (for optimizer)
    all_coins_day = []
    for e in all_sorted:
        all_coins_day.append({
            'id':            e['id'],
            'symbol':        e['symbol'],
            'name':          e['name'],
            'mom7d':         round(e['mom7d'],     4),
            'mom3d':         round(e['mom3d'],     4),
            'mom24h':        round(e['mom24h'],    4),
            'vol_ratio':     round(e['vol_ratio'], 6),
            'btc_rel':       round(e['btc_rel'],   4),
            'mom7d_n':       e['mom7d_n'],
            'mom3d_n':       e['mom3d_n'],
            'mom24h_n':      e['mom24h_n'],
            'vol_ratio_n':   e['vol_ratio_n'],
            'btc_rel_n':     e['btc_rel_n'],
            'in_top10_pred': e['id'] in top10_ids,
            # evaluation fields filled in later:
            'actual_return_7d': None,
            'in_actual_top10':  None,
        })

    price_snapshot = {
        cid: today_data[cid]['price']
        for cid in today_data
        if today_data[cid].get('price')
    }

    return predicted_coins, all_coins_day, price_snapshot

# ── Step 4: Evaluate a past prediction ─────────────────────────────────────────
def evaluate_past(pred_entry, eval_date, daily, meta):
    d_eval    = date_str(eval_date)
    eval_data = daily.get(d_eval, {})
    if not eval_data:
        return None

    old_prices = pred_entry.get('price_snapshot', {})
    pred_coins = pred_entry['predicted_coins']
    pred_ids   = {c['id'] for c in pred_coins}

    all_returns = {}
    for cid, old_p in old_prices.items():
        new_p = eval_data.get(cid, {}).get('price')
        if new_p and old_p and old_p > 0:
            all_returns[cid] = (new_p - old_p) / old_p * 100

    all_returns = {
        k: v for k, v in all_returns.items()
        if not is_stable(meta.get(k, {}).get('symbol', ''))
    }

    actual_sorted    = sorted(all_returns.items(), key=lambda kv: kv[1], reverse=True)[:10]
    actual_top10_ids = {k for k, _ in actual_sorted}

    coin_results = []
    hits = positive = 0
    for pc in pred_coins:
        cid     = pc['id']
        ret     = all_returns.get(cid)
        went_up = (ret or 0) > 0
        hit     = cid in actual_top10_ids
        if hit:     hits     += 1
        if went_up: positive += 1
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
        'all_returns':     {k: round(v, 2) for k, v in all_returns.items()},
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  CryptoTracker Pro — Full Backfill v2 (5 factors, all coins)')
    print('=' * 60)

    api_key = get_api_key()
    print(f'  API key: {"found" if api_key else "not found (public endpoint)"}')
    print(f'  Output 1: {DATA_FILE}  (top-10 predictions for dashboard)')
    print(f'  Output 2: {ALL_SCORES_FILE}  (all coins for optimizer)\n')

    # Load existing data
    existing = {'predictions': {}, 'evaluations': {}}
    if DATA_FILE.exists():
        try:
            existing = json.loads(DATA_FILE.read_text(encoding='utf-8'))
            print(f'  Existing: {len(existing["predictions"])} predictions, '
                  f'{len(existing["evaluations"])} evaluations')
        except Exception as e:
            print(f'  Warning: could not read existing data: {e}')

    # Load existing all_coins_scores
    all_scores = {}
    if ALL_SCORES_FILE.exists():
        try:
            all_scores = json.loads(ALL_SCORES_FILE.read_text(encoding='utf-8'))
            print(f'  Existing all_coins_scores: {len(all_scores)} days')
        except Exception as e:
            print(f'  Warning: could not read all_coins_scores: {e}')

    # Fetch data
    coins        = fetch_coin_list(api_key)
    daily, meta  = fetch_all_history(coins, api_key)

    today    = datetime.utcnow().date()
    start_dt = today - timedelta(days=DAYS_BACK - 10)
    end_dt   = today - timedelta(days=1)

    # ── Generate predictions ──
    print(f'\n[3/4] Generating predictions {start_dt} to {end_dt}...')
    new_preds = 0
    current_dt = start_dt
    while current_dt <= end_dt:
        d = date_str(current_dt)

        # Skip if exists and we're not overwriting backfill
        existing_pred = existing['predictions'].get(d)
        is_backfill   = (existing_pred or {}).get('predicted_coins', [{}])[0].get(
                            'model_version', '').startswith('4-factor')
        skip = (existing_pred and not OVERWRITE_BACKFILL) or \
               (existing_pred and not is_backfill)  # never overwrite live predictions

        if skip:
            current_dt += timedelta(days=1)
            continue

        predicted_coins, all_coins_day, price_snapshot = score_day(current_dt, daily, meta)
        if predicted_coins:
            maturity = date_str(current_dt + timedelta(days=7))
            existing['predictions'][d] = {
                'date':            d,
                'maturity_date':   maturity,
                'predicted_coins': predicted_coins,
                'price_snapshot':  price_snapshot,
            }
            all_scores[d] = all_coins_day
            new_preds += 1

        current_dt += timedelta(days=1)

    print(f'  Added/updated {new_preds} prediction days.')

    # ── Evaluate ──
    print(f'\n[4/4] Evaluating matured predictions...')
    new_evals = 0
    for d, pred in sorted(existing['predictions'].items()):
        pred_dt = datetime.strptime(d, '%Y-%m-%d').date()
        eval_dt = pred_dt + timedelta(days=7)
        if eval_dt > today:
            continue

        ev = evaluate_past(pred, eval_dt, daily, meta)
        if not ev:
            continue

        existing['evaluations'][d] = ev
        new_evals += 1

        # Back-fill actual returns into all_coins_scores for this day
        if d in all_scores:
            actual_top10_ids = {c['id'] for c in ev['actual_top10']}
            all_ret          = ev.get('all_returns', {})
            for coin_rec in all_scores[d]:
                cid = coin_rec['id']
                coin_rec['actual_return_7d'] = all_ret.get(cid)
                coin_rec['in_actual_top10']  = cid in actual_top10_ids

    print(f'  Evaluated {new_evals} predictions.')

    # ── Save ──
    DATA_FILE.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    ALL_SCORES_FILE.write_text(
        json.dumps(all_scores, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f'\n  Saved {DATA_FILE} ({DATA_FILE.stat().st_size // 1024} KB)')
    print(f'  Saved {ALL_SCORES_FILE} ({ALL_SCORES_FILE.stat().st_size // 1024} KB)')
    print(f'  Total: {len(existing["predictions"])} predictions, '
          f'{len(existing["evaluations"])} evaluations')
    print(f'  all_coins_scores: {len(all_scores)} days '
          f'({sum(len(v) for v in all_scores.values())} coin-day records)')

    # ── Regenerate dashboard ──
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('pt', 'prediction_tracker.py')
        pt   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pt)
        pt.generate_dashboard(existing)
        print('  Dashboard regenerated.')
    except Exception as e:
        print(f'  Warning: dashboard regeneration failed: {e}')

    print('\n' + '=' * 60)
    print('  Backfill complete! Now run:')
    print('    python optimize_weights.py')
    print('  to find the best formula weights.')
    print('=' * 60)

if __name__ == '__main__':
    main()
