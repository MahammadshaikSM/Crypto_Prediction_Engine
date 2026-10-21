#!/usr/bin/env python3
"""
Crypto Prediction Tracker
─────────────────────────
Runs daily to:
  1. Fetch top 200 coins from CoinMarketCap (listings + 16 days of daily closes)
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
from cmc_client import fetch_market, prices_by_symbol, market_regime
import numpy as np
import model_v2
from datetime import datetime, timedelta
from pathlib import Path

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
DATA_FILE    = SCRIPT_DIR / "prediction_data.json"
DASHBOARD    = SCRIPT_DIR / "prediction_dashboard.html"
DASHBOARD_3D = SCRIPT_DIR / "prediction_dashboard_3d.html"

# -- Stablecoin exclusion ------------------------------------------------------
STABLECOINS = {
    'usdt','usdc','busd','dai','tusd','usdp','usdd','frax','gusd','lusd',
    'susd','usdn','cusd','nusd','usdx','husd','ousd','musd','usdk','usds',
    'fdusd','pyusd','crvusd','mkusd','eurs','eurt','xaut','paxg','usde',
    'dola','ageur','alusd','usd0','wusd',
}

def is_stable(coin):
    sym = (coin.get('symbol') or '').lower()
    if sym in STABLECOINS or 'stablecoin' in (coin.get('tags') or []):
        return True
    chg7  = abs(coin.get('price_change_percentage_7d_in_currency') or 0)
    chg24 = abs(coin.get('price_change_percentage_24h') or 0)
    return chg7 < 0.6 and chg24 < 0.3

# -- Fetch coins (CoinMarketCap) -----------------------------------------------
def fetch_coins():
    """Top 200 coins from CoinMarketCap, each with ~16 daily closes for RSI/volatility."""
    for attempt in range(3):
        try:
            coins = fetch_market(limit=200, days=32)   # 31 daily points for model v2
            if coins:
                return coins
            raise ValueError("Empty CoinMarketCap response")
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(30)
    raise RuntimeError("Could not fetch coin data from CoinMarketCap after 3 attempts.")

# -- Technical helpers ---------------------------------------------------------
def calc_rsi(prices, period=14):
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

def calc_daily_volatility(closes):
    """Std-dev of the last 7 daily returns (matches backtest/backtest_weights.py)."""
    c = [p for p in closes[-8:] if p and p > 0]
    if len(c) < 3:
        return 0.01
    rets = [(c[i] - c[i-1]) / c[i-1] for i in range(1, len(c))]
    mean = sum(rets) / len(rets)
    return max(math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets)), 0.0001)

def rsi_score(rsi):
    if rsi >= 80:  return 0.0
    if rsi >= 70:  return 20.0
    if rsi >= 60:  return 50.0
    if rsi >= 40:  return 70.0
    if rsi >= 30:  return 85.0
    return 100.0

# -- Prediction model ----------------------------------------------------------
# Model weights (% per signal). 7-day = current live model; 3-day = tuned by
# backtest/backtest_weights.py on 3 years of CoinMarketCap data (results.json).
WEIGHTS_7D = {'vol_adj_mom': 22, 'mom24h': 28, 'btc_rel': 25, 'vol_surge': 10, 'rsi_s': 15}
WEIGHTS_3D_DEFAULT = {'vol_adj_mom': 0, 'mom24h': 14.8, 'btc_rel': 79.1, 'vol_surge': 6.0, 'rsi_s': 0.1}
BACKTEST_FILE = SCRIPT_DIR / "backtest" / "results.json"

def load_backtest():
    try:
        return json.loads(BACKTEST_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None

def weights_3d():
    bt = load_backtest()
    try:
        return {k: float(v) for k, v in bt['horizons']['3']['weights'].items()}
    except Exception:
        return dict(WEIGHTS_3D_DEFAULT)

def run_prediction(coins, weights=None):
    w = {k: v / 100 for k, v in (weights or WEIGHTS_7D).items()}
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

        closes     = c.get('daily_closes') or []
        volatility = calc_daily_volatility(closes)
        rsi        = calc_rsi(closes)            # RSI-14 on daily closes

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
            w['vol_adj_mom'] * x['vol_adj_mom_n']  +
            w['mom24h']      * x['mom24h_n']       +
            w['btc_rel']     * x['btc_relative_n'] +
            w['vol_surge']   * x['vol_surge_n']    +
            w['rsi_s']       * x['rsi_s']
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

def run_prediction_v2(coins, regime):
    """7-day picks from model v2 (18 signals + market regime). Raises if inputs are incomplete."""
    btc = next(c for c in coins if c['id'] == 'bitcoin')
    ref_dates = [r['date'] for r in btc.get('daily_history', [])][-model_v2.HISTORY_DAYS:]
    if len(ref_dates) < model_v2.HISTORY_DAYS:
        raise ValueError("not enough BTC history for model v2")
    rows = []
    for c in coins:
        if is_stable(c):
            continue
        h = {r['date']: r for r in c.get('daily_history', [])}
        seq = [h.get(d) for d in ref_dates]
        if any(r is None or not r['price'] or r['price'] <= 0 or r.get('volume') is None for r in seq):
            continue
        if not (seq[-1].get('volume') or 0) > 0 or not (seq[-1].get('mcap') or 0) > 0:
            continue
        rows.append((c, seq))
    if len(rows) < 40:
        raise ValueError(f"only {len(rows)} coins with full history")
    closes = np.array([[r['price'] for r in seq] for _, seq in rows]).T
    vols   = np.array([[r['volume'] for r in seq] for _, seq in rows]).T
    mcaps  = np.array([seq[-1]['mcap'] for _, seq in rows])
    b      = next(i for i, (c, _) in enumerate(rows) if c['id'] == 'bitcoin')
    prob, raw = model_v2.score_day(closes, vols, mcaps, b, regime)
    order = np.argsort(-prob)[:10]
    result = []
    for rank, i in enumerate(order, 1):
        c = rows[i][0]
        result.append({
            'rank': rank, 'id': c['id'], 'symbol': c['symbol'], 'name': c['name'], 'image': c.get('image', ''),
            'price_at_prediction': c['current_price'],
            'score':          round(float(prob[i]) * 100, 2),      # model's chance of a top-10 finish
            'confidence':     round(float(prob[i]) * 100, 2),
            'mom7d':          round(c.get('price_change_percentage_7d_in_currency') or 0, 2),
            'mom24h':         round(c.get('price_change_percentage_24h') or 0, 2),
            'vol_ratio_pct':  round(c['total_volume'] / c['market_cap'] * 100, 4) if c.get('market_cap') else 0,
            'btc_relative':   round(float(raw['btcrel7'][i]), 2),
            'rsi':            round(float(raw['rsi'][i]), 1),
            'volatility_pct': round(float(raw['vol7'][i]) * 100, 2),
            'model':          'v2',
        })
    return result

# -- Evaluate a past prediction ------------------------------------------------
# Predictions saved before the CoinMarketCap switch use CoinGecko ids. Most match the
# CMC slug; these are the common top-200 exceptions (CoinGecko id -> CMC slug).
CG_TO_CMC = {
    'binancecoin': 'bnb', 'ripple': 'xrp', 'avalanche-2': 'avalanche', 'the-open-network': 'toncoin',
    'polkadot': 'polkadot-new', 'hedera-hashgraph': 'hedera', 'crypto-com-chain': 'cronos',
    'near': 'near-protocol', 'render-token': 'render', 'matic-network': 'polygon',
    'polygon-ecosystem-token': 'polygon-ecosystem-token', 'internet-computer': 'internet-computer',
    'leo-token': 'unus-sed-leo', 'hyperliquid': 'hyperliquid', 'sui': 'sui', 'wrapped-bitcoin': 'wrapped-bitcoin',
    'elrond-erd-2': 'multiversx-egld', 'fetch-ai': 'artificial-superintelligence-alliance',
    'injective-protocol': 'injective', 'theta-token': 'theta-network', 'kucoin-shares': 'kucoin-token',
    'okb': 'okb', 'blockstack': 'stacks', 'havven': 'synthetix', 'immutable-x': 'immutable-x',
}

def _to_current_id(cid, cur_map):
    if cid in cur_map:
        return cid
    alt = CG_TO_CMC.get(cid)
    return alt if alt in cur_map else None

def unresolved_picks(pred_entry, current_coins):
    """(symbol, name) of picks that are not in today's top-200 listing."""
    ids = {c['id'] for c in current_coins}
    syms = {c['symbol'].lower() for c in current_coins}
    return [(pc['symbol'], pc['name']) for pc in pred_entry['predicted_coins']
            if not _to_current_id(pc['id'], ids) and pc['symbol'].lower() not in syms]

def evaluate_prediction(pred_entry, current_coins, extra_prices=None):
    """extra_prices: {symbol_lower: price} for picks outside today's top 200. They count
    toward each pick's return and positive rate, but not toward the actual top-10 ranking."""
    extra_prices = extra_prices or {}
    current_ids = {c['id'] for c in current_coins}
    by_symbol   = {}
    for c in current_coins:
        by_symbol.setdefault(c['symbol'].lower(), c['id'])
    # translate ids saved by an older data source (CoinGecko) to today's ids
    old_prices = {}
    for cid, p in pred_entry.get('price_snapshot', {}).items():
        new_id = _to_current_id(cid, current_ids)
        if new_id:
            old_prices[new_id] = p
    pred_coins = []
    for pc in pred_entry['predicted_coins']:
        new_id = _to_current_id(pc['id'], current_ids) or by_symbol.get(pc['symbol'].lower(), pc['id'])
        if new_id not in old_prices and pc.get('price_at_prediction'):
            old_prices[new_id] = pc['price_at_prediction']
        pred_coins.append(dict(pc, id=new_id))
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
        eval_price    = cur_map.get(cid)
        if eval_price is None and pc['symbol'].lower() in extra_prices and pc.get('price_at_prediction'):
            eval_price    = extra_prices[pc['symbol'].lower()]
            actual_return = (eval_price - pc['price_at_prediction']) / pc['price_at_prediction'] * 100
        coin_results.append({
            'id':                  cid,
            'symbol':              pc['symbol'],
            'name':                pc['name'],
            'image':               pc.get('image', ''),
            'rank_predicted':      pc['rank'],
            'score':               pc.get('score'),
            'confidence':          pc.get('confidence'),
            'price_at_prediction': pc['price_at_prediction'],
            'price_at_evaluation': eval_price,
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
LABELS = {'mom24h': ('&#128640;', '24H Momentum', '24H-Mom'), 'btc_rel': ('&#8383;', 'BTC-Relative', 'BTC-Rel'),
          'vol_adj_mom': ('&#9889;', 'Vol-Adj Momentum', 'Vol-Adj Momentum'), 'rsi_s': ('&#128201;', 'RSI Score', 'RSI'),
          'vol_surge': ('&#128202;', 'Volume Surge', 'Volume')}

def _fmt_w(v):
    return f"{v:g}%"

def generate_dashboard(data):
    """Render the 7-day page and the 3-day page from the same template."""
    bt = load_backtest() or {}
    def bt_note(h, which):
        try:
            t = bt['horizons'][str(h)][which]['test']
            return (f"Backtest of these exact weights on held-out CoinMarketCap data "
                    f"(Oct 2025 &ndash; Sep 2026, {t['days']} days): <strong>{t['hit']:.1f}%</strong> hit rate vs "
                    f"{t['base']:.1f}% for random picks.")
        except Exception:
            return ""
    preds7 = data.get('predictions', {})
    v2_days = sorted(d for d, p in preds7.items() if p.get('model') == 'v2')
    override7 = None
    if v2_days and preds7[max(preds7)].get('model') == 'v2':
        try:
            ho = model_v2.load_model()['heldout']
            note7 = (f"Model v2 backtest on held-out CoinMarketCap data (Oct 2025 &ndash; Sep 2026, {ho['days']} days): "
                     f"<strong>{ho['hit']:.1f}%</strong> hit rate vs {ho['current_formula_hit']:.1f}% for the previous "
                     f"formula and {ho['base']:.1f}% for random picks.")
        except Exception:
            note7 = ""
        note7 += f" Model v2 live since {v2_days[0]}; earlier picks used the v1 formula."
        override7 = dict(
            badges="\n      ".join(f'<span class="mbadge">{ic} {txt}</span>' for ic, txt in model_v2.DRIVERS),
            model_text="Model v2 (18 signals + market regime)", note=note7)
    pages = [
        (7, preds7, data.get('evaluations', {}), DASHBOARD, WEIGHTS_7D, bt_note(7, 'live'), override7),
        (3, data.get('predictions_3d', {}), data.get('evaluations_3d', {}), DASHBOARD_3D, weights_3d(), bt_note(3, 'tuned'), None),
    ]
    for horizon, predictions, evaluations, out_path, weights, note, override in pages:
        _render_page(horizon, predictions, evaluations, out_path, weights, note, override)


def _render_page(horizon, predictions, evaluations, out_path, weights, note, override=None):

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
    all_evs = sorted(evaluations.values(), key=lambda e: e['prediction_date'])
    recent_evs = all_evs[-7:]
    for ev in recent_evs:
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

    ordered = sorted(weights.items(), key=lambda kv: -kv[1])
    badges = "\n      ".join(f'<span class="mbadge">{LABELS[k][0]} {LABELS[k][1]} {_fmt_w(v)}</span>' for k, v in ordered if v > 0)
    model_text = " + ".join(f"{LABELS[k][2]}({_fmt_w(v)})" for k, v in ordered if v > 0)
    if override:
        badges, model_text, note = override['badges'], override['model_text'], override['note']
    out_path.write_text(
        _build_html(tl_json, chart_json, updated_at, hit_disp, pos_disp, best_disp, total_preds,
                    horizon=horizon, badges=badges, model_text=model_text, note=note),
        encoding='utf-8'
    )
    print(f"  Dashboard ({horizon}-day) -> {out_path}")


def _build_html(tl_json, chart_json, updated_at, hit_disp, pos_disp,
                best_disp, total_preds, horizon=7, badges='', model_text='', note=''):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__H__-Day Prediction Dashboard - CryptoTracker Pro</title>
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
.tabs-bar{display:flex;gap:0;margin-bottom:32px;border-bottom:1px solid var(--border)}
.tab-btn{background:none;border:none;border-bottom:2px solid transparent;padding:10px 24px;font-size:.82rem;font-weight:700;color:var(--muted);cursor:pointer;margin-bottom:-1px;transition:color .15s,border-color .15s;font-family:var(--font);letter-spacing:.02em}
.tab-btn:hover{color:var(--text2)}
.tab-btn.active{color:var(--blue);border-bottom-color:var(--blue)}
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
/* ── Today's 3-Day Picks ─────────────────────────────────────── */
.picks-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:12px;margin-top:4px}
.pick-card{background:var(--glass);border:1px solid var(--border);border-radius:16px;padding:16px;position:relative;overflow:hidden;transition:transform .15s,border-color .15s;cursor:default}
.pick-card:hover{transform:translateY(-2px);border-color:rgba(79,172,254,.3)}
.pick-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#4facfe,#00d4aa)}
.pick-card.hit-card::before{background:linear-gradient(90deg,#00f5a0,#00d4aa)}
.pick-card.miss-card::before{background:linear-gradient(90deg,#ff4d6d,#fb923c)}
.pick-rank{position:absolute;top:10px;right:12px;font-size:.6rem;font-weight:800;color:var(--muted)}
.pick-img{width:34px;height:34px;border-radius:50%;object-fit:contain;background:rgba(255,255,255,.05);margin-bottom:8px}
.pick-name{font-size:.78rem;font-weight:700;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:1px}
.pick-sym{font-size:.6rem;font-weight:800;color:var(--blue);text-transform:uppercase;margin-bottom:9px;letter-spacing:.05em}
.pick-stat{display:flex;justify-content:space-between;font-size:.63rem;color:var(--text2);margin-bottom:3px}
.pick-stat-val{font-weight:700}
.pick-conf-bar{margin-top:8px;height:4px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
.pick-conf-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#4facfe,#00d4aa);transition:width .6s ease}
.horizon-tabs{display:flex;gap:8px;padding:18px 32px 0;max-width:1400px;margin:0 auto;flex-wrap:wrap}
.htab{padding:10px 20px;border-radius:999px;font-size:.85rem;font-weight:700;letter-spacing:.02em;text-decoration:none;color:#8b9cc8;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.02);transition:all .2s}
.htab:hover{color:#e8eaf6;border-color:rgba(79,172,254,.4)}
.htab.active{color:#020510;background:linear-gradient(90deg,#4facfe,#00d4aa);border-color:transparent}
.pick-footer{display:flex;justify-content:space-between;align-items:center;margin-top:5px;font-size:.6rem;color:var(--muted)}
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
<nav class="horizon-tabs" aria-label="Prediction horizon">
  <a href="prediction_dashboard.html" class="htab __ACTIVE7__">&#128197; 7-Day Prediction</a>
  <a href="prediction_dashboard_3d.html" class="htab __ACTIVE3__">&#9889; 3-Day Prediction</a>
</nav>
<div class="wrap">

  <!-- ── Today's 3-Day Picks ───────────────────────────────────── -->
  <div style="margin-bottom:52px">
    <div class="section-label">&#128293; __H__-Day Spotlight</div>
    <h2 class="section-h">Today's Top Picks</h2>
    <p class="section-sub" id="picks-subtitle" style="margin-bottom:18px">Loading...</p>
    <div class="picks-grid" id="picks-grid"></div>
  </div>

  <div style="margin-bottom:32px">
    <div class="section-label">Model Performance</div>
    <h2 class="section-h">Prediction Accuracy</h2>
    <p class="section-sub">
      Each day the model predicts the top&nbsp;10 coins most likely to outperform over the next __H__&nbsp;days.
      A "hit" = coin appeared in the real top&nbsp;10 gainers __H__&nbsp;days later.
    </p>
    <div class="model-badge">
      __BADGES__
    </div>
    <p class="section-sub" style="margin-top:10px;font-size:.78rem">__BT_NOTE__</p>
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

// ── Utility helpers ──────────────────────────────────────────────────────────
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
  return '<img class="coin-img" src="'+img+'" alt="'+sym+'" onerror="this.style.visibility=\\'hidden\\'" loading="lazy">';
}
function rsiTag(rsi) {
  if (rsi==null) return '';
  const cls = rsi>70?'rsi-hot':rsi<30?'rsi-ok':'rsi-neutral';
  return '<span class="rsi-tag '+cls+'">RSI '+rsi+'</span>';
}

// ── Today's 3-Day Picks ──────────────────────────────────────────────────────
(function renderTodayPicks() {
  const subtitle = document.getElementById('picks-subtitle');
  const grid     = document.getElementById('picks-grid');
  if (!TIMELINE.length) {
    subtitle.textContent = 'No predictions yet. New picks are made daily at 11:30 UTC.';
    return;
  }
  const today  = TIMELINE[0];
  const coins  = today.predicted_coins || [];
  const ev     = today.evaluation;
  const mature = ev != null;

  if (mature) {
    const hc = ev.hit_count, hr = ev.hit_rate, pr = ev.positive_rate;
    subtitle.innerHTML =
      'Results in: <strong style="color:var(--green)">' + hc + '/10 hits (' + hr + '%)</strong>' +
      ' &middot; Positive rate: <strong style="color:var(--gold)">' + pr + '%</strong>' +
      ' &middot; Matured ' + fmtDate(today.maturity_date);
  } else {
    const daysLeft = Math.ceil((new Date(today.maturity_date)-new Date())/86400000);
    subtitle.innerHTML =
      'Matures in <strong style="color:var(--blue)">' + daysLeft + ' day' + (daysLeft!==1?'s':'') + '</strong>' +
      ' (' + fmtDate(today.maturity_date) + ')' +
      ' &middot; Model: __MODEL_TEXT__';
  }

  coins.forEach(function(c) {
    const result  = mature ? (ev.coin_results||[]).find(function(r){return r.id===c.id;}) : null;
    const ret     = result ? result.actual_return_pct : null;
    const hit     = result ? result.hit : null;
    const cardCls = mature ? (hit ? 'hit-card' : 'miss-card') : '';

    const retRow = ret != null
      ? '<div class="pick-stat"><span>Return</span><span class="pick-stat-val" style="color:' +
        (ret>=0?'var(--green)':'var(--red)') + '">' + (ret>=0?'+':'') + ret.toFixed(1) + '%</span></div>'
      : '';
    const hitBadge = mature
      ? '<span style="font-size:.65rem;font-weight:700;color:' + (hit?'var(--green)':'var(--red)') + '">' +
        (hit?'&#9989; HIT':'&#10060; MISS') + '</span>'
      : '<span style="font-size:.6rem;color:var(--muted)">&#9203; Pending</span>';

    const rsiVal  = c.rsi != null ? c.rsi : 50;
    const rsiCol  = rsiVal > 70 ? '#ff4d6d' : rsiVal < 30 ? '#00f5a0' : 'var(--muted)';
    const m7col   = c.mom7d  >= 0 ? 'var(--green)' : 'var(--red)';
    const m24col  = c.mom24h >= 0 ? 'var(--green)' : 'var(--red)';

    grid.insertAdjacentHTML('beforeend',
      '<div class="pick-card ' + cardCls + '">' +
        '<span class="pick-rank">#' + c.rank + '</span>' +
        '<img class="pick-img" src="' + (c.image||'') + '" alt="' + c.symbol + '" onerror="this.style.visibility=\\'hidden\\'" loading="lazy">' +
        '<div class="pick-name">' + c.name + '</div>' +
        '<div class="pick-sym">' + c.symbol.toUpperCase() + '</div>' +
        '<div class="pick-stat"><span>7D</span><span class="pick-stat-val" style="color:'+m7col+'">' + (c.mom7d>=0?'+':'') + c.mom7d.toFixed(1) + '%</span></div>' +
        '<div class="pick-stat"><span>24H</span><span class="pick-stat-val" style="color:'+m24col+'">' + (c.mom24h>=0?'+':'') + c.mom24h.toFixed(1) + '%</span></div>' +
        '<div class="pick-stat"><span>Vol Surge</span><span class="pick-stat-val" style="color:var(--blue)">' + c.vol_ratio_pct.toFixed(2) + '%</span></div>' +
        '<div class="pick-stat"><span>RSI</span><span class="pick-stat-val" style="color:'+rsiCol+'">' + rsiVal.toFixed(0) + '</span></div>' +
        retRow +
        '<div class="pick-conf-bar"><div class="pick-conf-fill" style="width:' + Math.min(100,c.confidence) + '%"></div></div>' +
        '<div class="pick-footer"><span>' + c.confidence.toFixed(1) + '% conf</span>' + hitBadge + '</div>' +
      '</div>'
    );
  });
})();

// ── Chart ────────────────────────────────────────────────────────────────────
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

  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth   = 1;
  [0,25,50,75,100].forEach(function(v) {
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cw, y); ctx.stroke();
    ctx.fillStyle = '#8b9cc8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(v + '%', PAD.left - 6, y + 4);
  });

  ctx.fillStyle = '#8b9cc8';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  CHART_DATA.labels.forEach(function(lbl, i) {
    ctx.fillText(lbl.slice(5), toX(i), H - 8);
  });

  function drawLine(vals, color, fillColor) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(vals[0]));
    vals.forEach(function(v,i) { if(i>0) ctx.lineTo(toX(i), toY(v)); });
    ctx.lineTo(toX(vals.length-1), PAD.top + ch);
    ctx.lineTo(toX(0), PAD.top + ch);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(vals[0]));
    vals.forEach(function(v,i) { if(i>0) ctx.lineTo(toX(i), toY(v)); });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();
    vals.forEach(function(v,i) {
      ctx.beginPath();
      ctx.arc(toX(i), toY(v), 4, 0, Math.PI*2);
      ctx.fillStyle = color; ctx.fill();
    });
    ctx.restore();
  }

  drawLine(CHART_DATA.pos_rates, '#00f5a0', 'rgba(0,245,160,0.07)');
  drawLine(CHART_DATA.hit_rates, '#4facfe', 'rgba(79,172,254,0.12)');

  [['#4facfe','Hit Rate'],['#00f5a0','Positive Rate']].forEach(function(item, i) {
    var c = item[0], l = item[1];
    const lx = PAD.left + i * 140;
    ctx.fillStyle = c;
    ctx.fillRect(lx, 4, 14, 3);
    ctx.fillStyle = '#8b9cc8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(l, lx + 18, 11);
  });
});

// ── Timeline ─────────────────────────────────────────────────────────────────
const container = document.getElementById('timeline');
if (!TIMELINE.length) {
  document.getElementById('no-data').style.display='block';
} else {
  TIMELINE.forEach(function(entry, idx) {
    const ev     = entry.evaluation;
    const mature = ev != null;

    var statusBadge = '';
    if (!mature) {
      const daysLeft = Math.ceil((new Date(entry.maturity_date)-new Date())/86400000);
      const dl = daysLeft>0?'Matures in '+daysLeft+' day'+(daysLeft!==1?'s':''):'Evaluating soon...';
      statusBadge = '<span class="tl-status-badge badge-pending">&#9203; Pending &middot; '+dl+'</span>';
    } else {
      const hr=ev.hit_rate, cls=hr>=50?'badge-hit':'badge-miss', ico=hr>=50?'&#9989;':'&#10060;';
      statusBadge = '<span class="tl-status-badge '+cls+'">'+ico+' '+ev.hit_count+'/10 hits &middot; '+hr+'%</span>';
    }

    const predRows = entry.predicted_coins.map(function(c) {
      const hitIcon  = mature?(ev.coin_results.find(function(r){return r.id===c.id;})?.hit?'&#9989;':'&#10060;'):'';
      const actualRet = mature?ev.coin_results.find(function(r){return r.id===c.id;})?.actual_return_pct:null;
      const retStr = actualRet!=null
        ? '<span class="coin-val '+(actualRet>=0?'up':'dn')+'">'+fmtPct(actualRet)+'</span>'
        : '<span class="coin-val neutral">'+fmtPct(c.confidence)+' conf.</span>';
      return '<div class="coin-row">'+
        coinImg(c.image,c.symbol)+
        '<span class="rank-badge">#'+c.rank+'</span>'+
        '<div class="coin-name">'+c.name+'<br><span class="coin-sym">'+c.symbol.toUpperCase()+'</span></div>'+
        rsiTag(c.rsi)+
        retStr+
        (mature?'<span class="hit-icon">'+hitIcon+'</span>':'')+
      '</div>';
    }).join('');

    var actualCol = '';
    if (mature && ev.actual_top10) {
      const rows = ev.actual_top10.map(function(c) {
        const star = c.was_predicted?'&#11088;':'<span style="opacity:.2">&middot;</span>';
        return '<div class="coin-row">'+
          '<div class="coin-name">'+c.name+'<br><span class="coin-sym">'+c.symbol.toUpperCase()+'</span></div>'+
          '<span class="coin-val up">'+fmtPct(c.return_pct)+'</span>'+
          '<span class="hit-icon">'+star+'</span>'+
        '</div>';
      }).join('');
      actualCol = '<div class="tl-col"><div class="tl-col-title actual">&#127942; Actual Top 10 (&#11088; = predicted)</div>'+rows+'</div>';
    } else if (!mature) {
      actualCol = '<div class="tl-col"><div style="padding:32px;text-align:center;color:var(--muted)">'+
        '<div style="font-size:2rem;margin-bottom:8px">&#9203;</div>'+
        '<div style="font-size:.88rem;font-weight:600;color:var(--text2)">Not yet evaluated</div>'+
        '<div style="font-size:.75rem">Matures '+fmtDate(entry.maturity_date)+'</div>'+
      '</div></div>';
    }

    var evalSummary = '';
    if (mature) {
      const hp=ev.hit_rate, pp=ev.positive_rate;
      evalSummary = '<div class="eval-summary">'+
        '<div class="eval-bar-wrap">'+
          '<div class="eval-bar-label"><span>Hit Rate</span><span>'+hp.toFixed(1)+'%</span></div>'+
          '<div class="eval-bar-track"><div class="eval-bar-fill" style="width:'+hp+'%"></div></div>'+
        '</div>'+
        '<div class="eval-bar-wrap">'+
          '<div class="eval-bar-label"><span>Positive Rate</span><span>'+pp.toFixed(1)+'%</span></div>'+
          '<div class="eval-bar-track"><div class="eval-bar-fill pos" style="width:'+pp+'%"></div></div>'+
        '</div>'+
        '<span class="eval-pill '+(hp>=50?'pill-green':'pill-red')+'">'+ev.hit_count+'/10 Hits</span>'+
        '<span class="eval-pill '+(pp>=50?'pill-green':'pill-gold')+'">'+ev.positive_count+'/10 Positive</span>'+
      '</div>';
    }

    container.insertAdjacentHTML('beforeend',
      '<div class="tl-entry entry-anim" style="animation-delay:'+idx*0.04+'s">'+
        '<div class="tl-header" onclick="toggle(this)" data-idx="'+idx+'">'+
          '<span class="tl-date-badge">&#128197; '+fmtDate(entry.date)+'</span>'+
          statusBadge+
          '<span style="font-size:.7rem;color:var(--muted);white-space:nowrap">-&gt; Matures '+fmtDate(entry.maturity_date)+'</span>'+
          '<span class="tl-arrow">&#9660;</span>'+
        '</div>'+
        '<div class="tl-body" data-body="'+idx+'">'+
          '<div class="tl-cols">'+
            '<div class="tl-col"><div class="tl-col-title pred">&#128302; Predicted Top 10 on '+fmtDate(entry.date)+'</div>'+predRows+'</div>'+
            actualCol+
          '</div>'+
          evalSummary+
        '</div>'+
      '</div>'
    );
  });

  const firstHeader = container.querySelector('.tl-header');
  if (firstHeader) toggle(firstHeader);
}

function toggle(header) {
  const idx  = header.getAttribute('data-idx');
  const body = document.querySelector('[data-body="'+idx+'"]');
  const open = body.classList.contains('open');
  body.classList.toggle('open', !open);
  header.classList.toggle('open', !open);
}

function switchTab(id, btn) {
  document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  ['overview','insights'].forEach(function(t) {
    document.getElementById('tab-'+t).style.display = (t===id)?'block':'none';
  });
  if (id==='insights') buildInsights();
}

let insBuilt = false;
function buildInsights() {
  if (insBuilt) return;
  insBuilt = true;

  const evals = TIMELINE.filter(function(e) { return e.evaluation; });
  if (!evals.length) {
    ['ins-factors','ins-monthly','ins-coins','ins-bestworst','ins-weights'].forEach(function(id) {
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
  factors.forEach(function(f) { fdata[f.key] = {hSum:0,hN:0,mSum:0,mN:0}; });
  evals.forEach(function(entry) {
    const cr_map = {};
    (entry.evaluation.coin_results||[]).forEach(function(cr) { cr_map[cr.id]=cr; });
    entry.predicted_coins.forEach(function(c) {
      const cr = cr_map[c.id];
      if (!cr) return;
      factors.forEach(function(f) {
        const val = c[f.key] != null ? c[f.key] : (f.key==='btc_relative' ? c['alignment'] : null);
        if (val == null) return;
        if (cr.hit) { fdata[f.key].hSum += val; fdata[f.key].hN++; }
        else        { fdata[f.key].mSum += val; fdata[f.key].mN++; }
      });
    });
  });
  const fdiffs = factors.map(function(f) {
    const d = fdata[f.key];
    const hAvg = d.hN ? d.hSum/d.hN : 0;
    const mAvg = d.mN ? d.mSum/d.mN : 0;
    return {key:f.key, label:f.label, diff: hAvg - mAvg, hAvg:hAvg, mAvg:mAvg};
  }).sort(function(a,b){return b.diff-a.diff;});
  const maxD = Math.max.apply(null, fdiffs.map(function(d){return Math.abs(d.diff);})) || 1;
  document.getElementById('ins-factors').innerHTML = fdiffs.map(function(d) {
    const pct = (Math.abs(d.diff)/maxD*100).toFixed(1);
    const col = d.diff >= 0 ? '#00f5a0' : '#ff4d6d';
    const sign = d.diff >= 0 ? '+' : '';
    return '<div class="factor-row">'+
      '<div class="factor-label"><span>'+d.label+'</span><span style="color:'+col+'">'+sign+d.diff.toFixed(1)+' avg diff</span></div>'+
      '<div class="factor-track"><div style="height:8px;border-radius:4px;width:'+pct+'%;background:'+col+'"></div></div>'+
      '<div style="font-size:.66rem;color:var(--muted);margin-top:3px">Hits avg: '+d.hAvg.toFixed(1)+' | Misses avg: '+d.mAvg.toFixed(1)+'</div>'+
    '</div>';
  }).join('');

  const monthly = {};
  evals.forEach(function(e) {
    const m = e.date.slice(0,7);
    if (!monthly[m]) monthly[m] = {hits:0,total:0,pos:0};
    monthly[m].hits  += e.evaluation.hit_count;
    monthly[m].total += 10;
    monthly[m].pos   += e.evaluation.positive_count;
  });
  const mRows = Object.keys(monthly).sort().map(function(m) {
    const d = monthly[m];
    const hr = (d.hits/d.total*100).toFixed(1);
    const pr = (d.pos/d.total*100).toFixed(1);
    const days = evals.filter(function(e){return e.date.startsWith(m);}).length;
    const parts = m.split('-');
    const lbl = new Date(+parts[0],+parts[1]-1,1).toLocaleDateString('en-US',{month:'short',year:'2-digit'});
    const hCol = +hr>=20?'#00f5a0':'#ff4d6d';
    return '<tr><td>'+lbl+'</td><td>'+days+'d</td><td style="color:'+hCol+'">'+hr+'%</td><td style="color:#ffd60a">'+pr+'%</td></tr>';
  }).join('');
  document.getElementById('ins-monthly').innerHTML =
    '<table class="ins-table"><tr><th>Month</th><th>Days</th><th>Hit%</th><th>Pos%</th></tr>'+mRows+'</table>';

  const coinCount = {}, coinMeta = {};
  evals.forEach(function(e) {
    e.predicted_coins.forEach(function(c) {
      coinCount[c.id] = (coinCount[c.id]||0)+1;
      coinMeta[c.id] = {name:c.name, sym:c.symbol};
    });
  });
  const topCoins = Object.keys(coinCount).sort(function(a,b){return coinCount[b]-coinCount[a];}).slice(0,10);
  const totalDays = evals.length;
  const cRows = topCoins.map(function(id) {
    const m = coinMeta[id]; const cnt = coinCount[id];
    const pct = (cnt/totalDays*100).toFixed(0);
    return '<tr><td>'+m.name+'<br><span style="font-size:.6rem;color:var(--muted)">'+m.sym.toUpperCase()+'</span></td>'+
      '<td>'+cnt+'/'+totalDays+' <span style="color:var(--blue);font-size:.7rem">('+pct+'%)</span></td></tr>';
  }).join('');
  document.getElementById('ins-coins').innerHTML =
    '<table class="ins-table"><tr><th>Coin</th><th>Appearances</th></tr>'+cRows+'</table>';

  const days = evals.map(function(e){return {date:e.date,hr:e.evaluation.hit_rate,hc:e.evaluation.hit_count};})
    .sort(function(a,b){return b.hr-a.hr;});
  const best5  = days.slice(0,5);
  const worst5 = days.slice(-5).reverse();
  function dayRows(arr,col) {
    return arr.map(function(d) {
      const parts = d.date.split('-');
      const lbl = new Date(+parts[0],+parts[1]-1,+parts[2]).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'2-digit'});
      return '<tr><td>'+lbl+'</td><td style="color:'+col+'">'+d.hc+'/10 ('+d.hr+'%)</td></tr>';
    }).join('');
  }
  document.getElementById('ins-bestworst').innerHTML =
    '<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#00f5a0;margin-bottom:6px">Top Days</div>'+
    '<table class="ins-table" style="margin-bottom:14px"><tr><th>Date</th><th>Hit Rate</th></tr>'+dayRows(best5,'#00f5a0')+'</table>'+
    '<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#ff4d6d;margin-bottom:6px">Toughest Days</div>'+
    '<table class="ins-table"><tr><th>Date</th><th>Hit Rate</th></tr>'+dayRows(worst5,'#ff4d6d')+'</table>';

  const posDiffs = fdiffs.map(function(d){return Math.max(d.diff, 0.001);});
  const tot = posDiffs.reduce(function(s,v){return s+v;},0)||1;
  const wBars = fdiffs.map(function(d,i) {
    const w = (posDiffs[i]/tot*100).toFixed(1);
    return '<div class="factor-row">'+
      '<div class="factor-label"><span>'+d.label+'</span><span style="color:var(--blue)">'+w+'%</span></div>'+
      '<div class="factor-track"><div style="height:8px;border-radius:4px;width:'+w+'%;background:linear-gradient(90deg,#4facfe,#00d4ff)"></div></div>'+
    '</div>';
  }).join('');
  document.getElementById('ins-weights').innerHTML =
    '<p style="font-size:.79rem;color:var(--text2);margin-bottom:18px">Weights derived from correlation between each factor score and actual hit outcomes.</p>'+
    wBars+
    '<div style="margin-top:16px;padding:14px;background:rgba(79,172,254,.06);border:1px solid rgba(79,172,254,.12);border-radius:12px;font-size:.75rem;color:var(--text2)">'+
    '&#128161; <strong style="color:var(--blue)">Current live model</strong> uses __MODEL_TEXT__.'+
    '</div>';
}
</script>
</body>
</html>"""
    return (html
        .replace('__H__',           str(horizon))
        .replace('__ACTIVE7__',     'active' if horizon == 7 else '')
        .replace('__ACTIVE3__',     'active' if horizon == 3 else '')
        .replace('__BADGES__',      badges)
        .replace('__MODEL_TEXT__',  model_text)
        .replace('__BT_NOTE__',     note)
        .replace('__UPDATED_AT__',  updated_at)
        .replace('__TOTAL_PREDS__', str(total_preds))
        .replace('__HIT_DISP__',    hit_disp)
        .replace('__POS_DISP__',    pos_disp)
        .replace('__BEST_DISP__',   best_disp)
        .replace('__TL_JSON__;',    tl_json + ';')
        .replace('__CHART_JSON__;', chart_json + ';'))


if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    print(f"[{today}] Loading data...")
    data = load_data()

    print("  Fetching coins from CoinMarketCap...")
    coins = fetch_coins()
    print(f"  Fetched {len(coins)} coins.")

    if seven_days_ago in data.get('predictions', {}) and seven_days_ago not in data.get('evaluations', {}):
        print(f"  Evaluating prediction from {seven_days_ago}...")
        pred_entry = data['predictions'][seven_days_ago]
        missing = unresolved_picks(pred_entry, coins)
        extra = {}
        if missing:
            try:
                extra = prices_by_symbol(missing)
                print(f"  Priced {len(extra)}/{len(missing)} picks outside today's top 200 via CMC quotes.")
            except Exception as e:
                print(f"  Could not price picks outside the top 200: {e}")
        ev = evaluate_prediction(pred_entry, coins, extra)
        data.setdefault('evaluations', {})[seven_days_ago] = ev
        print(f"  Hit rate: {ev['hit_rate']}%  |  Positive rate: {ev['positive_rate']}%")

    if today not in data.get('predictions', {}):
        print("  Running prediction model v2 (18 signals + market regime)...")
        model_used, model_error = 'v2', None
        try:
            regime = market_regime()
            top10 = run_prediction_v2(coins, regime)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  Model v2 unavailable ({e}); falling back to the v1 formula.")
            model_used, regime, model_error = 'v1', None, f"{type(e).__name__}: {e}"[:400]
            top10 = run_prediction(coins)
        if not top10:
            print("  ERROR: prediction model returned no results.")
            sys.exit(1)

        price_snapshot = {c['id']: c['current_price'] for c in coins if c.get('current_price')}
        maturity = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        data.setdefault('predictions', {})[today] = {
            'date':            today,
            'maturity_date':   maturity,
            'predicted_coins': top10,
            'price_snapshot':  price_snapshot,
            'fetched_at':      datetime.now().isoformat(),
            'model':           model_used,
            'regime':          regime,
            'model_error':     model_error,
        }

        print(f"  Today's top 10 (model {model_used}):")
        for c in top10:
            print(f"    {c['rank']:>2}. {c['symbol'].upper():<10} confidence={c['confidence']}%")
    else:
        print(f"  Prediction for {today} already exists — skipping.")

    # -- 3-day track: same signals, tuned 3-day weights, graded 3 days later --
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    p3, e3 = data.setdefault('predictions_3d', {}), data.setdefault('evaluations_3d', {})
    if three_days_ago in p3 and three_days_ago not in e3:
        print(f"  Evaluating 3-day prediction from {three_days_ago}...")
        entry = dict(p3[three_days_ago])
        entry.setdefault('price_snapshot', data['predictions'].get(three_days_ago, {}).get('price_snapshot', {}))
        missing = unresolved_picks(entry, coins)
        extra = {}
        if missing:
            try:
                extra = prices_by_symbol(missing)
            except Exception as e:
                print(f"  Could not price picks outside the top 200: {e}")
        ev3 = evaluate_prediction(entry, coins, extra)
        e3[three_days_ago] = ev3
        print(f"  3-day hit rate: {ev3['hit_rate']}%  |  Positive rate: {ev3['positive_rate']}%")
    if today not in p3:
        top3 = run_prediction(coins, weights_3d())
        if top3:
            p3[today] = {
                'date':            today,
                'maturity_date':   (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                'predicted_coins': top3,
                'fetched_at':      datetime.now().isoformat(),
            }
            # reuse today's 7-day snapshot; store one only if it is missing
            if today not in data['predictions']:
                p3[today]['price_snapshot'] = {c['id']: c['current_price'] for c in coins if c.get('current_price')}
            print("  Today's 3-day top 10: " + ", ".join(c['symbol'].upper() for c in top3))

    print("  Saving data...")
    save_data(data)

    print("  Regenerating dashboard...")
    generate_dashboard(data)

    print("Done ✓")
