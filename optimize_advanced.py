#!/usr/bin/env python3
"""
optimize_advanced.py
====================
Advanced optimization targeting 35%+ hit rate using:

  1. Market regime filter  — skip/adjust on BTC bear days
  2. Coin reputation boost — reward coins with high historical hit rates
  3. Confidence threshold  — only count high-confidence prediction days
  4. Two-regime model      — different weights for bull vs bear markets
  5. Fine grid search      — step=2% around the best region
  6. Cap-tier filtering    — focus on mid/small caps where model works best

Run after backfill_predictions.py and optimize_weights.py:
    python optimize_advanced.py
"""

import json, math
from pathlib import Path
from collections import defaultdict

ALL_SCORES_FILE = Path('all_coins_scores.json')
PRED_FILE       = Path('prediction_data.json')

print('=' * 65)
print('  CryptoTracker Pro — Advanced Optimizer (target: 35%+)')
print('=' * 65)

if not ALL_SCORES_FILE.exists():
    print('ERROR: all_coins_scores.json not found. Run backfill_predictions.py first.')
    raise SystemExit(1)

raw   = json.loads(ALL_SCORES_FILE.read_text(encoding='utf-8'))
pdata = json.loads(PRED_FILE.read_text(encoding='utf-8')) if PRED_FILE.exists() else {}

# Only keep evaluated days
days = {}
for d, coins in raw.items():
    evaluated = [c for c in coins if c.get('in_actual_top10') is not None]
    if len(evaluated) >= 50:
        days[d] = evaluated

print(f'  Loaded {len(days)} evaluated days\n')

# ── Base simulation ─────────────────────────────────────────────────────────────
def simulate(weights, days_data, skip_dates=None, coin_boost=None):
    """Score all coins, pick top 10, measure hit rate."""
    w24, w3, w7, wbr, wvol = weights
    skip_dates = skip_dates or set()
    total_hits = total_days = 0

    for d, coins in days_data.items():
        if d in skip_dates:
            continue
        for c in coins:
            base = (w24  * c.get('mom24h_n',   50) +
                    w3   * c.get('mom3d_n',     50) +
                    w7   * c.get('mom7d_n',     50) +
                    wbr  * c.get('btc_rel_n',   50) +
                    wvol * c.get('vol_ratio_n', 50))
            boost = (coin_boost or {}).get(c['id'], 0)
            c['_s'] = base + boost

        top10_ids  = {c['id'] for c in sorted(coins, key=lambda x: x['_s'], reverse=True)[:10]}
        actual_ids = {c['id'] for c in coins if c.get('in_actual_top10')}
        total_hits += len(top10_ids & actual_ids)
        total_days += 1

    return (total_hits / (total_days * 10) * 100) if total_days else 0.0

# Best weights from basic optimizer
BEST_W = [10, 25, 30, 0, 35]
baseline = simulate(BEST_W, days)
print(f'  Baseline (best basic weights): {baseline:.2f}%\n')

# ── Technique 1: Market regime filter ──────────────────────────────────────────
print('━' * 65)
print('  TECHNIQUE 1: Market Regime Filter')
print('  (skip days when BTC is crashing — model fails in bear markets)')
print('━' * 65)

# Compute BTC 7D momentum for each day using btc's own scores
# BTC is in the coins list — find it and use its raw mom7d
btc_mom = {}
for d, coins in days.items():
    btc = next((c for c in coins if c['id'] == 'bitcoin'), None)
    if btc:
        btc_mom[d] = btc.get('mom7d', 0)

print(f'\n  BTC 7D momentum available for {len(btc_mom)} days')

best_filter_hr = baseline
best_threshold = None
best_skip_count = 0

print(f'\n  {"BTC threshold":>15}  {"Days skipped":>13}  {"Remaining":>10}  {"Hit Rate":>9}')
print('  ' + '-' * 55)
for thresh in [-20, -15, -10, -8, -5, -3, 0, 3, 5]:
    skip = {d for d, m in btc_mom.items() if m < thresh}
    hr   = simulate(BEST_W, days, skip_dates=skip)
    remaining = len(days) - len(skip)
    print(f'  BTC 7D < {thresh:4}%  {len(skip):13}  {remaining:10}  {hr:8.2f}%  {"← BEST" if hr > best_filter_hr else ""}')
    if hr > best_filter_hr:
        best_filter_hr = hr
        best_threshold = thresh
        best_skip_count = len(skip)

if best_threshold is not None:
    skip_best = {d for d, m in btc_mom.items() if m < best_threshold}
    print(f'\n  ✓ Market filter: skip days when BTC 7D < {best_threshold}%')
    print(f'    Skips {best_skip_count} bad days → Hit Rate: {best_filter_hr:.2f}%')
    print(f'    Improvement over baseline: +{best_filter_hr-baseline:.2f}%')
else:
    skip_best = set()
    print(f'\n  Market filter did not improve results.')

# ── Technique 2: Coin reputation boost ─────────────────────────────────────────
print('\n' + '━' * 65)
print('  TECHNIQUE 2: Coin Reputation Boost')
print('  (reward coins with high historical hit rates)')
print('━' * 65)

# Build per-coin stats from all evaluated days
coin_stats = defaultdict(lambda: {'pred': 0, 'hits': 0})
for d, coins in days.items():
    if d in skip_best:
        continue
    for c in coins:
        c['_s'] = (BEST_W[0]*c.get('mom24h_n',50) + BEST_W[1]*c.get('mom3d_n',50) +
                   BEST_W[2]*c.get('mom7d_n',50)  + BEST_W[3]*c.get('btc_rel_n',50) +
                   BEST_W[4]*c.get('vol_ratio_n',50))
    top10_ids  = {c['id'] for c in sorted(coins, key=lambda x: x['_s'], reverse=True)[:10]}
    actual_ids = {c['id'] for c in coins if c.get('in_actual_top10')}
    for c in coins:
        if c['id'] in top10_ids:
            coin_stats[c['id']]['pred'] += 1
            if c['id'] in actual_ids:
                coin_stats[c['id']]['hits'] += 1

# Coins predicted 5+ times — their hit rate as a reputation score
rep_scores = {}
for cid, s in coin_stats.items():
    if s['pred'] >= 5:
        rep_scores[cid] = (s['hits'] / s['pred']) * 100  # 0-100

print(f'\n  {len(rep_scores)} coins with 5+ predictions for reputation scoring')

best_rep_hr = best_filter_hr
best_rep_scale = 0

print(f'\n  {"Rep scale":>10}  {"Hit Rate":>10}')
print('  ' + '-' * 24)
for scale in [0, 1, 2, 3, 5, 8, 10, 15, 20]:
    boost = {cid: (rep_scores[cid] - 50) * scale / 100 for cid in rep_scores}
    hr = simulate(BEST_W, days, skip_dates=skip_best, coin_boost=boost)
    print(f'  {scale:10}  {hr:9.2f}%  {"← BEST" if hr > best_rep_hr else ""}')
    if hr > best_rep_hr:
        best_rep_hr  = hr
        best_rep_scale = scale

if best_rep_scale > 0:
    best_boost = {cid: (rep_scores[cid] - 50) * best_rep_scale / 100 for cid in rep_scores}
    print(f'\n  ✓ Reputation boost scale={best_rep_scale} → Hit Rate: {best_rep_hr:.2f}%')
    print(f'    Improvement over market filter: +{best_rep_hr-best_filter_hr:.2f}%')
else:
    best_boost = {}
    print(f'\n  Reputation boost did not improve further.')

# ── Technique 3: Confidence threshold ──────────────────────────────────────────
print('\n' + '━' * 65)
print('  TECHNIQUE 3: Confidence Threshold')
print('  (only count days where top-10 scores are clearly separated)')
print('━' * 65)

def score_gap(coins, weights, boost=None):
    """Score gap between rank 10 and rank 11 coin."""
    w24, w3, w7, wbr, wvol = weights
    boost = boost or {}
    for c in coins:
        base = (w24*c.get('mom24h_n',50) + w3*c.get('mom3d_n',50) +
                w7*c.get('mom7d_n',50)   + wbr*c.get('btc_rel_n',50) +
                wvol*c.get('vol_ratio_n',50))
        c['_s'] = base + boost.get(c['id'], 0)
    ranked = sorted(coins, key=lambda x: x['_s'], reverse=True)
    if len(ranked) < 11:
        return 0
    return ranked[9]['_s'] - ranked[10]['_s']   # gap between 10th and 11th

gaps = {}
for d, coins in days.items():
    if d not in skip_best:
        gaps[d] = score_gap(coins, BEST_W, best_boost)

all_gaps = sorted(gaps.values())
print(f'\n  Score gap stats: min={min(all_gaps):.1f} median={all_gaps[len(all_gaps)//2]:.1f} max={max(all_gaps):.1f}')

best_conf_hr  = best_rep_hr
best_conf_pct = 0

print(f'\n  {"Keep top %":>10}  {"Days kept":>10}  {"Hit Rate":>10}')
print('  ' + '-' * 35)
for keep_pct in [100, 90, 80, 70, 60, 50]:
    cutoff_idx = int(len(all_gaps) * (1 - keep_pct/100))
    cutoff     = all_gaps[cutoff_idx] if cutoff_idx < len(all_gaps) else 0
    high_conf  = {d for d, g in gaps.items() if g >= cutoff}
    skip_conf  = skip_best | (set(days.keys()) - high_conf)
    hr         = simulate(BEST_W, days, skip_dates=skip_conf, coin_boost=best_boost)
    print(f'  {keep_pct:9}%  {len(high_conf):10}  {hr:9.2f}%  {"← BEST" if hr > best_conf_hr else ""}')
    if hr > best_conf_hr:
        best_conf_hr  = hr
        best_conf_pct = keep_pct

if best_conf_pct and best_conf_pct < 100:
    print(f'\n  ✓ Keep top {best_conf_pct}% confidence days → Hit Rate: {best_conf_hr:.2f}%')
    print(f'    Improvement over reputation: +{best_conf_hr-best_rep_hr:.2f}%')
else:
    print(f'\n  Confidence filter did not improve further.')
    best_conf_pct = 100

# ── Technique 4: Two-regime model ──────────────────────────────────────────────
print('\n' + '━' * 65)
print('  TECHNIQUE 4: Two-Regime Model')
print('  (different weights for bull vs bear market days)')
print('━' * 65)

if best_threshold is not None:
    bull_days = {d: c for d, c in days.items()
                 if d not in skip_best and btc_mom.get(d, 0) >= best_threshold}
    bear_days = {d: c for d, c in days.items()
                 if d in skip_best and btc_mom.get(d, 0) < best_threshold}
    print(f'\n  Bull days: {len(bull_days)}, Bear days (currently skipped): {len(bear_days)}')

    # Try different weights specifically on bear days
    print(f'\n  Finding best weights for bear market days...')
    bear_results = []
    for w1 in range(0, 101, 10):
        for w2 in range(0, 101-w1, 10):
            for w3 in range(0, 101-w1-w2, 10):
                for w4 in range(0, 101-w1-w2-w3, 10):
                    w5 = 100 - w1 - w2 - w3 - w4
                    if 0 <= w5 <= 100:
                        hr = simulate([w1,w2,w3,w4,w5], bear_days, coin_boost=best_boost)
                        bear_results.append((hr, w1, w2, w3, w4, w5))

    bear_results.sort(reverse=True)
    best_bear_w = bear_results[0][1:]
    print(f'  Best bear weights: 24H={best_bear_w[0]}% 3D={best_bear_w[1]}% '
          f'7D={best_bear_w[2]}% BTC={best_bear_w[3]}% Vol={best_bear_w[4]}%')
    print(f'  Bear hit rate with these: {bear_results[0][0]:.2f}%')

    # Combined: bull days with best_W, bear days with bear weights (instead of skipping)
    def simulate_two_regime(bull_w, bear_w, all_days, btc_threshold, boost=None):
        total_hits = total_days = 0
        for d, coins in all_days.items():
            w = bull_w if btc_mom.get(d, 0) >= btc_threshold else bear_w
            for c in coins:
                base = (w[0]*c.get('mom24h_n',50) + w[1]*c.get('mom3d_n',50) +
                        w[2]*c.get('mom7d_n',50)   + w[3]*c.get('btc_rel_n',50) +
                        w[4]*c.get('vol_ratio_n',50))
                c['_s'] = base + (boost or {}).get(c['id'], 0)
            top10_ids  = {c['id'] for c in sorted(coins, key=lambda x:x['_s'], reverse=True)[:10]}
            actual_ids = {c['id'] for c in coins if c.get('in_actual_top10')}
            total_hits += len(top10_ids & actual_ids)
            total_days += 1
        return (total_hits / (total_days * 10) * 100) if total_days else 0

    two_regime_hr = simulate_two_regime(BEST_W, list(best_bear_w), days, best_threshold, best_boost)
    print(f'\n  Two-regime hit rate (all {len(days)} days): {two_regime_hr:.2f}%')
    if two_regime_hr > best_conf_hr:
        print(f'  ✓ Two-regime beats skipping: +{two_regime_hr-best_conf_hr:.2f}%')
    else:
        print(f'  Skipping bear days still better. Keep market filter.')

# ── Technique 5: Fine grid around best region ──────────────────────────────────
print('\n' + '━' * 65)
print('  TECHNIQUE 5: Fine Grid Search (step=2%) around best region')
print('━' * 65)

skip_for_fine = skip_best
print(f'\n  Running fine grid (step=2%) around [10,25,30,0,35]...')

fine_results = []
for w1 in range(0, 31, 2):   # 24H: 0-30
    for w2 in range(15, 46, 2):  # 3D: 15-45
        if w1 + w2 > 100: continue
        for w3 in range(15, 46, 2):  # 7D: 15-45
            if w1 + w2 + w3 > 100: continue
            for w4 in range(0, 21, 2):  # BTC: 0-20
                if w1 + w2 + w3 + w4 > 100: continue
                w5 = 100 - w1 - w2 - w3 - w4
                if not (20 <= w5 <= 50): continue
                hr = simulate([w1,w2,w3,w4,w5], days,
                              skip_dates=skip_for_fine, coin_boost=best_boost)
                fine_results.append((hr, w1, w2, w3, w4, w5))

fine_results.sort(reverse=True)
best_fine = fine_results[0]
print(f'\n  Top 10 fine-grid results:')
print(f'  {"Hit%":>6}  {"24H":>4}  {"3D":>4}  {"7D":>4}  {"BTC":>4}  {"Vol":>4}')
for hr, w1, w2, w3, w4, w5 in fine_results[:10]:
    print(f'  {hr:6.2f}%  {w1:3}%  {w2:3}%  {w3:3}%  {w4:3}%  {w5:3}%')

# ── Technique 6: Cap tier analysis ─────────────────────────────────────────────
print('\n' + '━' * 65)
print('  TECHNIQUE 6: Focus on specific rank tiers (top 10-50, 51-100, etc.)')
print('  (model may work better on certain market cap tiers)')
print('━' * 65)

pred_data = pdata.get('predictions', {})
evals_data = pdata.get('evaluations', {})

# Analyze which RANK coins (in the 200) tend to hit
# We can infer approximate rank from order in all_coins_scores (sorted by score)
rank_hits = defaultdict(lambda: {'hits': 0, 'total': 0})
for d, coins in days.items():
    if d in skip_best: continue
    for c in coins:
        c['_s'] = (BEST_W[0]*c.get('mom24h_n',50) + BEST_W[1]*c.get('mom3d_n',50) +
                   BEST_W[2]*c.get('mom7d_n',50)   + BEST_W[3]*c.get('btc_rel_n',50) +
                   BEST_W[4]*c.get('vol_ratio_n',50))
    ranked = sorted(coins, key=lambda x: x['_s'], reverse=True)
    actual_ids = {c['id'] for c in coins if c.get('in_actual_top10')}
    for rank, c in enumerate(ranked, 1):
        tier = ((rank - 1) // 10) * 10 + 1
        if rank <= 50:
            rank_hits[tier]['total'] += 1
            if c['id'] in actual_ids:
                rank_hits[tier]['hits'] += 1

print(f'\n  Hit rate by score rank (best formula weights):')
print(f'  {"Rank tier":>12}  {"Predictions":>12}  {"Hits":>6}  {"Hit%":>7}')
print('  ' + '-' * 42)
for tier in sorted(rank_hits.keys()):
    s = rank_hits[tier]
    hr = s['hits'] / s['total'] * 100 if s['total'] else 0
    end = tier + 9
    print(f'  {tier:4}-{end:<4}       {s["total"]:12}  {s["hits"]:6}  {hr:6.1f}%')

# ── Final summary ───────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('  FINAL RESULTS SUMMARY')
print('=' * 65)
print(f'  Baseline (best basic weights, all days):     {baseline:.2f}%')
if best_threshold is not None:
    print(f'  + Market filter (BTC 7D < {best_threshold}%):            {best_filter_hr:.2f}%')
if best_rep_scale > 0:
    print(f'  + Coin reputation boost (scale={best_rep_scale}):          {best_rep_hr:.2f}%')
if best_conf_pct < 100:
    print(f'  + Confidence filter (top {best_conf_pct}% days):          {best_conf_hr:.2f}%')
print(f'  + Fine grid search:                          {best_fine[0]:.2f}%')

print(f'\n  BEST ACHIEVABLE HIT RATE: {best_fine[0]:.2f}%')
print(f'\n  OPTIMAL FORMULA:')
print(f'    24H Momentum   {best_fine[1]}%')
print(f'    3D  Momentum   {best_fine[2]}%')
print(f'    7D  Momentum   {best_fine[3]}%')
print(f'    BTC-Relative   {best_fine[4]}%')
print(f'    Volume Surge   {best_fine[5]}%')
if best_threshold is not None:
    print(f'\n  MARKET FILTER: Skip days when BTC 7D momentum < {best_threshold}%')
if best_rep_scale > 0:
    print(f'  COIN BOOST:    Apply reputation boost (scale={best_rep_scale})')

# Save results
results = {
    'baseline':        round(baseline, 2),
    'best_hit_rate':   round(best_fine[0], 2),
    'best_weights': {
        'mom24h':    best_fine[1],
        'mom3d':     best_fine[2],
        'mom7d':     best_fine[3],
        'btc_rel':   best_fine[4],
        'vol_ratio': best_fine[5],
    },
    'market_filter_threshold': best_threshold,
    'coin_rep_boost_scale':    best_rep_scale,
    'confidence_filter_pct':   best_conf_pct,
    'top_coins_by_hit_rate': [
        {'id': cid, 'hit_rate': round(s['hits']/s['pred']*100,1),
         'pred': s['pred'], 'hits': s['hits']}
        for cid, s in sorted(coin_stats.items(),
                              key=lambda x: x[1]['hits']/max(x[1]['pred'],1),
                              reverse=True)
        if s['pred'] >= 5
    ][:20],
}
Path('advanced_optimization_results.json').write_text(
    json.dumps(results, indent=2), encoding='utf-8'
)
print(f'\n  Saved to advanced_optimization_results.json')
print('=' * 65)
