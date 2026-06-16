#!/usr/bin/env python3
"""
optimize_weights.py
===================
Reads all_coins_scores.json (produced by backfill_predictions.py v2)
and finds the optimal factor weights for the 5-factor prediction model.

Run AFTER backfill_predictions.py completes:
    python optimize_weights.py

Takes ~2 minutes. Outputs the best formula weights.
"""

import json, math, itertools
from pathlib import Path
from collections import defaultdict

ALL_SCORES_FILE = Path('all_coins_scores.json')

# ── Load data ───────────────────────────────────────────────────────────────────
print('=' * 60)
print('  CryptoTracker Pro — Weight Optimizer')
print('=' * 60)

if not ALL_SCORES_FILE.exists():
    print('ERROR: all_coins_scores.json not found.')
    print('Run backfill_predictions.py first.')
    raise SystemExit(1)

raw = json.loads(ALL_SCORES_FILE.read_text(encoding='utf-8'))

# Only keep days where we have evaluation results
days = {}
for d, coins in raw.items():
    evaluated = [c for c in coins if c.get('in_actual_top10') is not None]
    if len(evaluated) >= 50:   # need enough coins for meaningful ranking
        days[d] = evaluated

print(f'  Loaded {len(days)} evaluated days '
      f'({sum(len(v) for v in days.values())} coin-day records)\n')

if not days:
    print('ERROR: No evaluated days found. Wait for predictions to mature (7 days).')
    raise SystemExit(1)

# ── Simulation function ─────────────────────────────────────────────────────────
def simulate(weights, days_data):
    """
    For each day: score all coins with given weights, pick top 10,
    return average hit rate across all days.
    weights = [w_mom24h, w_mom3d, w_mom7d, w_btcrel, w_vol]  — must sum to 100
    """
    w24, w3, w7, wbr, wvol = weights
    total_hits = 0
    total_days = 0

    for d, coins in days_data.items():
        # Score each coin
        for c in coins:
            c['_s'] = (
                w24  * c.get('mom24h_n', 50) +
                w3   * c.get('mom3d_n',  50) +
                w7   * c.get('mom7d_n',  50) +
                wbr  * c.get('btc_rel_n',50) +
                wvol * c.get('vol_ratio_n', 50)
            )

        top10_ids = {
            c['id']
            for c in sorted(coins, key=lambda x: x['_s'], reverse=True)[:10]
        }
        actual_top10_ids = {c['id'] for c in coins if c.get('in_actual_top10')}

        hits = len(top10_ids & actual_top10_ids)
        total_hits += hits
        total_days += 1

    if total_days == 0:
        return 0.0
    return total_hits / (total_days * 10) * 100   # hit rate %

# ── Grid search ─────────────────────────────────────────────────────────────────
print('Running grid search (step=5%)...')
print('Factors: mom24h, mom3d, mom7d, btc_rel, vol_ratio\n')

best_results = []
step = 5
options = list(range(0, 101, step))   # 0,5,10,...,100

combo_count = 0
tested = 0

# Count total combinations first
for w1 in options:
    for w2 in options:
        if w1 + w2 > 100: continue
        for w3 in options:
            if w1 + w2 + w3 > 100: continue
            for w4 in options:
                if w1 + w2 + w3 + w4 > 100: continue
                w5 = 100 - w1 - w2 - w3 - w4
                if 0 <= w5 <= 100:
                    combo_count += 1

print(f'  Testing {combo_count:,} weight combinations...')

for w1 in options:
    for w2 in options:
        if w1 + w2 > 100: continue
        for w3 in options:
            if w1 + w2 + w3 > 100: continue
            for w4 in options:
                if w1 + w2 + w3 + w4 > 100: continue
                w5 = 100 - w1 - w2 - w3 - w4
                if not (0 <= w5 <= 100): continue

                hr = simulate([w1, w2, w3, w4, w5], days)
                best_results.append((hr, w1, w2, w3, w4, w5))
                tested += 1
                if tested % 5000 == 0:
                    best_so_far = max(best_results, key=lambda x: x[0])
                    print(f'  {tested:,}/{combo_count:,} tested... best so far: {best_so_far[0]:.2f}%')

best_results.sort(reverse=True)
print(f'\n  Done! {tested:,} combinations tested.\n')

# ── Results ──────────────────────────────────────────────────────────────────────
print('=' * 60)
print('  TOP 20 WEIGHT COMBINATIONS')
print('=' * 60)
print(f'  {"Hit%":>6}  {"24H":>5}  {"3D":>5}  {"7D":>5}  {"BTC":>5}  {"Vol":>5}')
print('  ' + '-' * 42)
for hr, w1, w2, w3, w4, w5 in best_results[:20]:
    print(f'  {hr:6.2f}%  {w1:4}%  {w2:4}%  {w3:4}%  {w4:4}%  {w5:4}%')

best = best_results[0]
print(f'\n  BEST FORMULA:')
print(f'    24H Momentum   {best[1]}%')
print(f'    3D  Momentum   {best[2]}%')
print(f'    7D  Momentum   {best[3]}%')
print(f'    BTC-Relative   {best[4]}%')
print(f'    Volume Surge   {best[5]}%')
print(f'    Expected Hit Rate: {best[0]:.2f}%')

# Current model baseline
current = simulate([28, 0, 22, 22, 6], days)   # no mom3d in old model
print(f'\n  BASELINE (current model, no 3D): {current:.2f}%')
print(f'  IMPROVEMENT:                    +{best[0]-current:.2f}%')

# ── Segment analysis: which days does the model perform best? ──────────────────
print('\n' + '=' * 60)
print('  SEGMENT ANALYSIS — when does the model shine?')
print('=' * 60)

w1, w2, w3, w4, w5 = best[1], best[2], best[3], best[4], best[5]

monthly = defaultdict(lambda: {'hits': 0, 'days': 0})
day_hits = []

for d, coins in days.items():
    for c in coins:
        c['_s'] = (w1*c.get('mom24h_n',50) + w2*c.get('mom3d_n',50) +
                   w3*c.get('mom7d_n',50)  + w4*c.get('btc_rel_n',50) +
                   w5*c.get('vol_ratio_n',50))
    top10_ids   = {c['id'] for c in sorted(coins, key=lambda x: x['_s'], reverse=True)[:10]}
    actual_ids  = {c['id'] for c in coins if c.get('in_actual_top10')}
    hits        = len(top10_ids & actual_ids)
    month       = d[:7]
    monthly[month]['hits'] += hits
    monthly[month]['days'] += 1
    day_hits.append((d, hits))

print(f'\n  Monthly accuracy (best formula):')
print(f'  {"Month":>8}  {"Days":>5}  {"Hit%":>6}')
print('  ' + '-' * 25)
for m in sorted(monthly):
    md = monthly[m]
    hr = md['hits'] / (md['days'] * 10) * 100
    bar = '█' * int(hr / 5)
    print(f'  {m}  {md["days"]:5}  {hr:5.1f}%  {bar}')

# Best and worst individual days
day_hits.sort(key=lambda x: x[1], reverse=True)
print(f'\n  Best 5 days:')
for d, h in day_hits[:5]:
    print(f'    {d}  {h}/10 hits ({h*10}%)')
print(f'\n  Toughest 5 days:')
for d, h in day_hits[-5:]:
    print(f'    {d}  {h}/10 hits ({h*10}%)')

# ── Top coins analysis ──────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  TOP COINS (most frequently in actual top-10 when predicted)')
print('=' * 60)

coin_stats = defaultdict(lambda: {'pred': 0, 'hits': 0, 'name': ''})
for d, coins in days.items():
    for c in coins:
        c['_s'] = (w1*c.get('mom24h_n',50) + w2*c.get('mom3d_n',50) +
                   w3*c.get('mom7d_n',50)  + w4*c.get('btc_rel_n',50) +
                   w5*c.get('vol_ratio_n',50))
    top10 = sorted(coins, key=lambda x: x['_s'], reverse=True)[:10]
    actual_ids = {c['id'] for c in coins if c.get('in_actual_top10')}
    for c in top10:
        coin_stats[c['id']]['pred'] += 1
        coin_stats[c['id']]['name']  = c.get('name', c['id'])
        if c['id'] in actual_ids:
            coin_stats[c['id']]['hits'] += 1

# Filter: predicted 5+ times
freq = [(cid, s) for cid, s in coin_stats.items() if s['pred'] >= 5]
freq.sort(key=lambda x: x[1]['hits'] / x[1]['pred'], reverse=True)

print(f'  {"Coin":22s}  {"Pred":>5}  {"Hits":>5}  {"Hit%":>6}')
print('  ' + '-' * 44)
for cid, s in freq[:15]:
    hr = s['hits'] / s['pred'] * 100
    print(f'  {s["name"][:22]:22s}  {s["pred"]:5}  {s["hits"]:5}  {hr:5.1f}%')

# ── Save recommendation ─────────────────────────────────────────────────────────
rec = {
    'best_weights': {
        'mom24h':    best[1],
        'mom3d':     best[2],
        'mom7d':     best[3],
        'btc_rel':   best[4],
        'vol_ratio': best[5],
    },
    'expected_hit_rate':  round(best[0], 2),
    'baseline_hit_rate':  round(current, 2),
    'improvement':        round(best[0] - current, 2),
    'days_evaluated':     len(days),
    'top20_combos': [
        {'hit_rate': hr, 'mom24h': w1, 'mom3d': w2, 'mom7d': w3, 'btc_rel': w4, 'vol': w5}
        for hr, w1, w2, w3, w4, w5 in best_results[:20]
    ],
}
Path('weight_optimization_results.json').write_text(
    json.dumps(rec, indent=2), encoding='utf-8'
)
print(f'\n  Results saved to weight_optimization_results.json')
print('\n  Next step: update prediction_tracker.py with the best weights above.')
print('=' * 60)
