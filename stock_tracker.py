#!/usr/bin/env python3
"""
Stock Prediction Tracker
────────────────────────
Runs daily to:
  1. Fetch all S&P 500 stocks from Yahoo Finance via yfinance
  2. Score each stock using a 5-factor Jegadeesh-style momentum model:
       * 7-day Momentum          30%  (short-term price change)
       * 30-day Momentum         25%  (medium-term trend)
       * Volume Surge            20%  (today vs 20-day avg volume)
       * RSI Score               15%  (prefer 45-65, penalise extremes)
       * MA Trend                10%  (price vs 50-day moving average)
  3. Save today's top-10 prediction + price snapshot to stock_data.json
  4. Evaluate any prediction made exactly 7 days ago
  5. Regenerate stock_dashboard.html with all data embedded
"""

import json, os, sys, time, math
import requests
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: yfinance and pandas are required. Run: pip install yfinance pandas")
    sys.exit(1)

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
DATA_FILE   = SCRIPT_DIR / "stock_data.json"
DASHBOARD   = SCRIPT_DIR / "stock_dashboard.html"

# -- S&P 500 ticker list -------------------------------------------------------
def get_sp500_tickers():
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        print(f"  Loaded {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), using fallback list.")
        # Fallback: top 100 by market cap
        return [
            'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','AVGO','JPM',
            'LLY','V','UNH','XOM','MA','COST','HD','PG','WMT','NFLX','BAC','CRM',
            'ORCL','AMD','ABBV','KO','CVX','MRK','ACN','PEP','TMO','CSCO','LIN',
            'MCD','ABT','DHR','NKE','ADBE','TXN','AMGN','PM','NEE','RTX','QCOM',
            'UPS','INTU','SPGI','MS','HON','CAT','ISRG','GS','BLK','SYK','ELV',
            'VRTX','AXP','T','MDT','DE','GILD','REGN','CI','PLD','SCHW','ADI',
            'LRCX','MU','PANW','KLAC','BSX','SO','DUK','CB','ITW','ZTS','MDLZ',
            'MMC','WM','APH','AON','HCA','SHW','CME','TGT','PH','FI','MCO','USB',
            'NSC','EMR','ETN','GE','NOC','TJX','ROP','AIG','OKE','ECL','GM','F'
        ]

# -- Data persistence ----------------------------------------------------------
def load_data():
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {'predictions': {}, 'evaluations': {}}

def save_data(data):
    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

# -- Fetch stock data ----------------------------------------------------------
def fetch_stocks(tickers):
    """Download 3-month daily OHLCV for all tickers in one bulk call."""
    print(f"  Downloading data for {len(tickers)} stocks (this takes ~60s)...")
    try:
        raw = yf.download(
            tickers,
            period='3mo',
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")

    stocks = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw
            else:
                df = raw[ticker] if ticker in raw.columns.get_level_values(0) else None
            if df is None or df.empty or len(df) < 10:
                continue
            df = df.dropna(subset=['Close'])
            close  = df['Close'].values.tolist()
            volume = df['Volume'].values.tolist()
            stocks.append({
                'ticker':  ticker,
                'close':   close,
                'volume':  volume,
                'price':   close[-1],
            })
        except Exception:
            continue
    print(f"  Got usable data for {len(stocks)} stocks.")
    return stocks

# -- Technical indicators ------------------------------------------------------
def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains  = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def momentum_pct(prices, days):
    """% change over last `days` trading days."""
    if len(prices) < days + 1:
        return 0.0
    return (prices[-1] - prices[-(days+1)]) / prices[-(days+1)] * 100

def volume_surge(volumes, window=20):
    """Current volume vs N-day average. Returns ratio."""
    if len(volumes) < window + 1:
        return 1.0
    avg = sum(volumes[-(window+1):-1]) / window
    if avg == 0:
        return 1.0
    return volumes[-1] / avg

def ma_trend(prices, short=10, long=50):
    """
    Score based on price vs moving averages.
    Returns 1.0 if above both MAs, 0.5 if mixed, 0.0 if below both.
    """
    if len(prices) < long:
        return 0.5
    ma_short = sum(prices[-short:]) / short
    ma_long  = sum(prices[-long:])  / long
    price    = prices[-1]
    above_short = price > ma_short
    above_long  = price > ma_long
    if above_short and above_long:
        return 1.0
    if above_short or above_long:
        return 0.5
    return 0.0

def rsi_score(rsi):
    """
    Optimal RSI range is 45-65 for momentum continuation.
    Penalise overbought (>70) and oversold (<30) extremes.
    Returns 0-100.
    """
    if 45 <= rsi <= 65:
        return 100.0
    if rsi > 70:
        return max(0, 100 - (rsi - 70) * 5)
    if rsi < 30:
        return max(0, 100 - (30 - rsi) * 5)
    if rsi < 45:
        return 60 + (rsi - 30) * 2.7
    return 100 - (rsi - 65) * 2.0

# -- Normalise a list of values to 0-100 --------------------------------------
def normalise(values):
    mn, mx = min(values), max(values)
    if mx == mn:
        return [50.0] * len(values)
    return [(v - mn) / (mx - mn) * 100 for v in values]

# -- Score & rank all stocks ---------------------------------------------------
def run_prediction(stocks):
    """
    Score each stock on 5 factors, return top-10.
    Weights: mom7=30%, mom30=25%, vol_surge=20%, rsi=15%, ma=10%
    """
    records = []
    for s in stocks:
        c = s['close']
        v = s['volume']
        mom7   = momentum_pct(c, 7)
        mom30  = momentum_pct(c, 21)   # ~1 month of trading days
        vsurge = volume_surge(v, 20)
        rsi    = calc_rsi(c, 14)
        ma     = ma_trend(c, 10, 50)
        records.append({
            'ticker': s['ticker'],
            'price':  s['price'],
            'mom7':   mom7,
            'mom30':  mom30,
            'vsurge': vsurge,
            'rsi':    rsi,
            'ma':     ma,
        })

    if len(records) < 2:
        return []

    mom7_n   = normalise([r['mom7']   for r in records])
    mom30_n  = normalise([r['mom30']  for r in records])
    vsurge_n = normalise([r['vsurge'] for r in records])
    rsi_n    = [rsi_score(r['rsi'])   for r in records]
    ma_n     = [r['ma'] * 100         for r in records]

    scored = []
    for i, r in enumerate(records):
        score = (
            0.30 * mom7_n[i]   +
            0.25 * mom30_n[i]  +
            0.20 * vsurge_n[i] +
            0.15 * rsi_n[i]    +
            0.10 * ma_n[i]
        )
        scored.append({**r, 'score': round(score, 2)})

    scored.sort(key=lambda x: x['score'], reverse=True)
    top10 = []
    for rank, s in enumerate(scored[:10], 1):
        top10.append({
            'rank':       rank,
            'ticker':     s['ticker'],
            'price':      round(s['price'], 4),
            'score':      s['score'],
            'mom7':       round(s['mom7'], 2),
            'mom30':      round(s['mom30'], 2),
            'vsurge':     round(s['vsurge'], 2),
            'rsi':        round(s['rsi'], 1),
            'confidence': round(min(score / 100 * 100, 99), 0),
        })
    return top10

# -- Evaluate a 7-day-old prediction ------------------------------------------
def evaluate_prediction(pred, stocks):
    """
    A 'hit' = predicted stock appears in the actual top-10 gainers 7 days later.
    Positive rate = % of predicted stocks that went up at all.
    """
    snapshot   = pred.get('price_snapshot', {})
    predicted  = [c['ticker'] for c in pred.get('predicted_coins', [])]
    price_now  = {s['ticker']: s['price'] for s in stocks}

    returns = {}
    for t in predicted:
        old = snapshot.get(t)
        now = price_now.get(t)
        if old and now and old > 0:
            returns[t] = (now - old) / old * 100

    if not returns:
        return None

    # actual top gainers from the predicted set + universe
    all_returns = {}
    for s in stocks:
        old = snapshot.get(s['ticker'])
        if old and old > 0 and s['price']:
            all_returns[s['ticker']] = (s['price'] - old) / old * 100

    top10_actual = sorted(all_returns, key=all_returns.get, reverse=True)[:10]

    hits     = sum(1 for t in predicted if t in top10_actual)
    positive = sum(1 for t in predicted if returns.get(t, 0) > 0)

    return {
        'prediction_date': pred['date'],
        'maturity_date':   pred.get('maturity_date', ''),
        'hit_count':       hits,
        'hit_rate':        round(hits / 10 * 100, 1),
        'positive_count':  positive,
        'positive_rate':   round(positive / len(predicted) * 100, 1),
        'returns':         {t: round(v, 2) for t, v in returns.items()},
    }

# -- Dashboard generation ------------------------------------------------------
def generate_dashboard(data):
    predictions  = data.get('predictions', {})
    evaluations  = data.get('evaluations', {})

    total_preds  = len(predictions)
    total_evals  = len(evaluations)
    avg_hit_rate = round(sum(e['hit_rate'] for e in evaluations.values()) / total_evals, 1) if total_evals else 0
    avg_pos_rate = round(sum(e['positive_rate'] for e in evaluations.values()) / total_evals, 1) if total_evals else 0

    best_day = ''
    best_hr  = 0
    for date, ev in evaluations.items():
        if ev['hit_rate'] > best_hr:
            best_hr  = ev['hit_rate']
            best_day = f"{ev['hit_rate']:.0f}% ({date})"

    hit_disp  = f"{avg_hit_rate:.1f}%" if total_evals else '-'
    pos_disp  = f"{avg_pos_rate:.1f}%" if total_evals else '-'
    best_disp = best_day or '-'

    # Timeline for prediction history
    timeline = []
    for date in sorted(predictions.keys(), reverse=True):
        pred = predictions[date]
        ev   = evaluations.get(date)
        timeline.append({
            'date':            date,
            'maturity_date':   pred.get('maturity_date', ''),
            'predicted_coins': pred.get('predicted_coins', []),
            'evaluation':      ev,
        })

    # Chart data — last 30 evaluated days
    chart_data = {'labels': [], 'hit_rates': [], 'pos_rates': []}
    all_evs = sorted(evaluations.values(), key=lambda e: e['prediction_date'])
    for ev in all_evs[-30:]:
        chart_data['labels'].append(ev['prediction_date'])
        chart_data['hit_rates'].append(ev['hit_rate'])
        chart_data['pos_rates'].append(ev['positive_rate'])

    tl_json    = json.dumps(timeline, ensure_ascii=False)
    chart_json = json.dumps(chart_data, ensure_ascii=False)
    updated_at = datetime.now().strftime('%b %d, %Y at %H:%M')

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Prediction Dashboard - StockTracker Pro</title>
<style>
:root {
  --bg:     #060d1f;
  --glass:  rgba(255,255,255,0.03);
  --border: rgba(255,255,255,0.07);
  --text1:  #e8eaf6;
  --text2:  #8b9cc8;
  --muted:  #3d4e72;
  --blue:   #4facfe;
  --green:  #00f5a0;
  --red:    #ff4d6d;
  --gold:   #ffd166;
  --purple: #c77dff;
}
* { box-sizing:border-box; margin:0; padding:0 }
body { background:var(--bg); color:var(--text1); font-family:'Segoe UI',system-ui,sans-serif; min-height:100vh }
a { color:inherit; text-decoration:none }

/* Tab nav */
.tab-nav { display:flex; gap:8px; padding:20px 32px 0; border-bottom:1px solid var(--border); background:rgba(6,13,31,0.95); position:sticky; top:0; z-index:100 }
.tab-btn { padding:10px 24px; border-radius:10px 10px 0 0; font-size:.85rem; font-weight:600; letter-spacing:.04em; cursor:pointer; border:1px solid transparent; border-bottom:none; transition:all .2s; color:var(--text2) }
.tab-btn.active { background:var(--glass); border-color:var(--border); color:var(--text1); border-bottom:1px solid var(--bg) }
.tab-btn:hover:not(.active) { color:var(--text1) }

/* Header */
.header { display:flex; align-items:center; justify-content:space-between; padding:24px 32px; border-bottom:1px solid var(--border) }
.hdr-brand { display:flex; align-items:center; gap:12px }
.hdr-logo { width:40px; height:40px; border-radius:10px; background:linear-gradient(135deg,#00f5a0,#4facfe); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1rem; color:#060d1f }
.hdr-title { font-size:1.1rem; font-weight:700 }
.hdr-sub { font-size:.65rem; letter-spacing:.1em; color:var(--text2); text-transform:uppercase; margin-top:2px }
.hdr-right { font-size:.75rem; color:var(--muted) }

/* Stats */
.stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; padding:32px 32px 0 }
.stat { background:var(--glass); border:1px solid var(--border); border-radius:20px; padding:24px 26px; position:relative; overflow:hidden }
.stat::before { content:''; position:absolute; top:0; left:0; right:0; height:2px }
.stat:nth-child(1)::before { background:linear-gradient(90deg,#4facfe,#00d4ff) }
.stat:nth-child(2)::before { background:linear-gradient(90deg,#00f5a0,#00d4aa) }
.stat:nth-child(3)::before { background:linear-gradient(90deg,#ffd166,#ffaa00) }
.stat:nth-child(4)::before { background:linear-gradient(90deg,#c77dff,#9b5de5) }
.stat-icon { font-size:1.3rem; margin-bottom:12px }
.stat-label { font-size:.65rem; letter-spacing:.08em; text-transform:uppercase; color:var(--text2); margin-bottom:6px }
.stat-value { font-size:2rem; font-weight:800; line-height:1 }
.stat:nth-child(1) .stat-value { color:var(--blue) }
.stat:nth-child(2) .stat-value { color:var(--green) }
.stat:nth-child(3) .stat-value { color:var(--gold) }
.stat:nth-child(4) .stat-value { color:var(--purple) }
.stat-sub { font-size:.72rem; color:var(--text2); margin-top:6px }

/* Content sections */
.content { padding:32px }
.section-label { font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; color:var(--blue); margin-bottom:8px }
.section-h { font-size:1.25rem; font-weight:700; margin-bottom:20px }

/* Model info */
.model-card { background:var(--glass); border:1px solid var(--border); border-radius:20px; padding:24px; margin-bottom:48px }
.model-desc { font-size:.82rem; color:var(--text2); line-height:1.6; margin-bottom:16px }
.factors { display:flex; flex-wrap:wrap; gap:8px }
.factor-badge { padding:5px 12px; border-radius:20px; font-size:.7rem; font-weight:600; background:rgba(79,172,254,.1); border:1px solid rgba(79,172,254,.2); color:var(--blue) }

/* Chart */
.chart-card { background:var(--glass); border:1px solid var(--border); border-radius:20px; padding:28px 28px 24px; margin-bottom:48px }
.chart-wrap { position:relative; height:220px }

/* Prediction history */
.tl-entry { background:var(--glass); border:1px solid var(--border); border-radius:20px; overflow:hidden; margin-bottom:12px }
.tl-header { display:flex; align-items:center; justify-content:space-between; padding:16px 20px; cursor:pointer; user-select:none }
.tl-header:hover { background:rgba(255,255,255,.02) }
.tl-date { font-size:.85rem; font-weight:600 }
.tl-meta { font-size:.72rem; color:var(--text2); margin-top:2px }
.tl-badges { display:flex; gap:8px; align-items:center }
.badge { padding:3px 10px; border-radius:20px; font-size:.68rem; font-weight:700 }
.badge-hit  { background:rgba(0,245,160,.12); color:var(--green); border:1px solid rgba(0,245,160,.2) }
.badge-miss { background:rgba(255,77,109,.10); color:var(--red);   border:1px solid rgba(255,77,109,.2) }
.badge-pend { background:rgba(139,156,200,.1); color:var(--text2); border:1px solid var(--border) }
.tl-body { display:none; border-top:1px solid var(--border) }
.tl-coins { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px; padding:16px 20px }
.coin-card { background:rgba(255,255,255,.03); border:1px solid var(--border); border-radius:12px; padding:12px 14px }
.coin-rank { font-size:.6rem; color:var(--muted); margin-bottom:4px }
.coin-sym  { font-size:.92rem; font-weight:700 }
.coin-price { font-size:.72rem; color:var(--text2); margin-top:2px }
.coin-score { font-size:.7rem; color:var(--blue); margin-top:4px }
.eval-section { padding:12px 20px 16px; border-top:1px solid var(--border) }
.eval-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px }
.eval-bar-label { display:flex; justify-content:space-between; font-size:.72rem; color:var(--text2); margin-bottom:4px }
.eval-bar-track { height:6px; background:rgba(255,255,255,.07); border-radius:3px; overflow:hidden }
.eval-bar-fill  { height:100%; border-radius:3px }

/* No data */
.no-data { text-align:center; padding:60px 20px; color:var(--muted) }
.no-data-icon { font-size:2.5rem; margin-bottom:12px }

@media(max-width:700px) {
  .stats-grid { grid-template-columns:1fr 1fr }
  .content,.header,.stats-grid { padding-left:16px; padding-right:16px }
  .tab-nav { padding:16px 16px 0 }
}
</style>
</head>
<body>

<!-- Tab navigation -->
<div class="tab-nav">
  <a href="prediction_dashboard.html" class="tab-btn">&#128640; Crypto</a>
  <a href="stock_dashboard.html" class="tab-btn active">&#128200; Stocks</a>
</div>

<div class="header">
  <div class="hdr-brand">
    <div class="hdr-logo">ST</div>
    <div>
      <div class="hdr-title">StockTracker Pro</div>
      <div class="hdr-sub">S&amp;P 500 Prediction Dashboard</div>
    </div>
  </div>
  <div class="hdr-right">Last updated: __UPDATED_AT__</div>
</div>

<div class="stats-grid">
  <div class="stat"><div class="stat-icon">&#128202;</div><div class="stat-label">Total Predictions</div><div class="stat-value">__TOTAL_PREDS__</div><div class="stat-sub">Days tracked so far</div></div>
  <div class="stat"><div class="stat-icon">&#9989;</div><div class="stat-label">Avg Hit Rate</div><div class="stat-value">__HIT_DISP__</div><div class="stat-sub">Top-10 intersection accuracy</div></div>
  <div class="stat"><div class="stat-icon">&#128200;</div><div class="stat-label">Avg Positive Rate</div><div class="stat-value">__POS_DISP__</div><div class="stat-sub">% predicted stocks that went up</div></div>
  <div class="stat"><div class="stat-icon">&#127942;</div><div class="stat-label">Best Day</div><div class="stat-value">__BEST_DISP__</div><div class="stat-sub">Highest single-day hit rate</div></div>
</div>

<div class="content">
  <div class="section-label">Model Info</div>
  <h2 class="section-h">Prediction Model</h2>
  <div class="model-card">
    <div class="model-desc">
      Each day the model scores all S&amp;P 500 stocks using a 5-factor Jegadeesh-style momentum model
      and predicts the top 10 stocks most likely to outperform over the next 7 days.
      A "hit" = predicted stock appears in the actual top-10 gainers 7 days later.
    </div>
    <div class="factors">
      <span class="factor-badge">&#9889; 7-Day Momentum 30%</span>
      <span class="factor-badge">&#128200; 30-Day Momentum 25%</span>
      <span class="factor-badge">&#128266; Volume Surge 20%</span>
      <span class="factor-badge">&#128209; RSI Score 15%</span>
      <span class="factor-badge">&#128202; MA Trend 10%</span>
    </div>
  </div>

  <div id="chart-section" style="margin-bottom:48px">
    <div class="section-label">Accuracy Over Time</div>
    <h2 class="section-h" style="margin-bottom:18px">Hit Rate &amp; Positive Rate</h2>
    <div class="chart-card"><div class="chart-wrap"><canvas id="accChart"></canvas></div></div>
  </div>

  <div class="section-label">Prediction History</div>
  <h2 class="section-h">All Predictions</h2>
  <p style="font-size:.78rem;color:var(--text2);margin-bottom:20px">Click any entry to expand. RSI shown per stock.</p>
  <div id="tl-container"></div>
</div>

<script>
const TIMELINE   = __TL_JSON__;
const CHART_DATA = __CHART_JSON__;

window.addEventListener('load', function() {
  // Draw chart
  if (CHART_DATA.labels.length) {
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
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
    [0,25,50,75,100].forEach(v => {
      const y = toY(v);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cw, y); ctx.stroke();
      ctx.fillStyle = '#8b9cc8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(v + '%', PAD.left - 6, y + 4);
    });
    ctx.fillStyle = '#8b9cc8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
    CHART_DATA.labels.forEach((lbl, i) => { ctx.fillText(lbl.slice(5), toX(i), H - 8); });
    function drawLine(vals, color, fillColor) {
      ctx.save();
      ctx.beginPath(); ctx.moveTo(toX(0), toY(vals[0]));
      vals.forEach((v,i) => { if(i>0) ctx.lineTo(toX(i), toY(v)); });
      ctx.lineTo(toX(vals.length-1), PAD.top + ch); ctx.lineTo(toX(0), PAD.top + ch);
      ctx.closePath(); ctx.fillStyle = fillColor; ctx.fill();
      ctx.beginPath(); ctx.moveTo(toX(0), toY(vals[0]));
      vals.forEach((v,i) => { if(i>0) ctx.lineTo(toX(i), toY(v)); });
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.stroke();
      vals.forEach((v,i) => { ctx.beginPath(); ctx.arc(toX(i), toY(v), 4, 0, Math.PI*2); ctx.fillStyle = color; ctx.fill(); });
      ctx.restore();
    }
    drawLine(CHART_DATA.pos_rates, '#00f5a0', 'rgba(0,245,160,0.07)');
    drawLine(CHART_DATA.hit_rates, '#4facfe', 'rgba(79,172,254,0.12)');
    [['#4facfe','Hit Rate'],['#00f5a0','Positive Rate']].forEach(([c,l], i) => {
      const lx = PAD.left + i * 140;
      ctx.fillStyle = c; ctx.fillRect(lx, 4, 14, 3);
      ctx.fillStyle = '#8b9cc8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(l, lx + 18, 11);
    });
  } else {
    document.getElementById('chart-section').style.display = 'none';
  }

  // Render timeline
  const container = document.getElementById('tl-container');
  if (!TIMELINE.length) {
    container.innerHTML = '<div class="no-data"><div class="no-data-icon">&#128202;</div><div style="font-size:1rem;font-weight:700;color:var(--text2);margin-bottom:8px">No predictions yet</div><div style="font-size:.8rem">Run <code>stock_tracker.py</code> to make the first prediction.</div></div>';
    return;
  }

  TIMELINE.forEach((entry, idx) => {
    const ev  = entry.evaluation;
    const coins = entry.predicted_coins || [];
    const isEval = !!ev;
    const hr = ev ? ev.hit_rate : null;
    const badgeCls = isEval ? (hr >= 50 ? 'badge-hit' : 'badge-miss') : 'badge-pend';
    const badgeTxt = isEval ? (hr >= 50 ? '&#9989; ' + hr + '%' : '&#10060; ' + hr + '%') : '&#9203; Pending';

    const d = entry.date;
    const [y,m,dy] = d.split('-');
    const dateStr = new Date(+y,+m-1,+dy).toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric',year:'numeric'});

    const coinCards = coins.map(c => {
      const ret = ev && ev.returns ? ev.returns[c.ticker] : null;
      const retStr = ret != null ? (ret >= 0 ? '<span style="color:var(--green)">+' + ret.toFixed(2) + '%</span>' : '<span style="color:var(--red)">' + ret.toFixed(2) + '%</span>') : '';
      return '<div class="coin-card">'
        + '<div class="coin-rank">#' + c.rank + '</div>'
        + '<div class="coin-sym">' + c.ticker + '</div>'
        + '<div class="coin-price">$' + (c.price || 0).toFixed(2) + '</div>'
        + '<div class="coin-score">Score: ' + c.score + (retStr ? ' &nbsp;' + retStr : '') + '</div>'
        + '</div>';
    }).join('');

    let evalHtml = '';
    if (isEval) {
      const hp = ev.hit_rate, pp = ev.positive_rate;
      evalHtml = '<div class="eval-section">'
        + '<div style="font-size:.72rem;font-weight:700;color:var(--text2);margin-bottom:8px">7-DAY RESULTS</div>'
        + '<div class="eval-grid">'
        + '<div><div class="eval-bar-label"><span>Hit Rate</span><span>' + hp.toFixed(1) + '%</span></div>'
        + '<div class="eval-bar-track"><div class="eval-bar-fill" style="width:' + hp + '%;background:linear-gradient(90deg,#4facfe,#00d4ff)"></div></div></div>'
        + '<div><div class="eval-bar-label"><span>Positive Rate</span><span>' + pp.toFixed(1) + '%</span></div>'
        + '<div class="eval-bar-track"><div class="eval-bar-fill" style="width:' + pp + '%;background:linear-gradient(90deg,#00f5a0,#00d4aa)"></div></div></div>'
        + '</div></div>';
    }

    const el = document.createElement('div');
    el.className = 'tl-entry';
    el.innerHTML = '<div class="tl-header" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'block\'?\'none\':\'block\'">'
      + '<div><div class="tl-date">' + dateStr + '</div>'
      + '<div class="tl-meta">Matures: ' + (entry.maturity_date || '—') + '</div></div>'
      + '<div class="tl-badges"><span class="badge ' + badgeCls + '">' + badgeTxt + '</span></div>'
      + '</div>'
      + '<div class="tl-body">'
      + '<div class="tl-coins">' + coinCards + '</div>'
      + evalHtml
      + '</div>';
    container.appendChild(el);
  });
});
</script>
</body>
</html>"""

    html = (html
        .replace('__UPDATED_AT__',  updated_at)
        .replace('__TOTAL_PREDS__', str(total_preds))
        .replace('__HIT_DISP__',    hit_disp)
        .replace('__POS_DISP__',    pos_disp)
        .replace('__BEST_DISP__',   best_disp)
        .replace('__TL_JSON__;',    tl_json + ';')
        .replace('__CHART_JSON__;', chart_json + ';'))

    DASHBOARD.write_text(html, encoding='utf-8')
    print(f"  Dashboard written to {DASHBOARD}")


# -- Main ----------------------------------------------------------------------
if __name__ == '__main__':
    today           = datetime.now().strftime('%Y-%m-%d')
    seven_days_ago  = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    print(f"[{today}] Loading stock data...")
    data = load_data()

    print("  Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()

    print("  Downloading price data from Yahoo Finance...")
    stocks = fetch_stocks(tickers)

    if not stocks:
        print("ERROR: No stock data retrieved.")
        sys.exit(1)

    # Evaluate 7-day-old prediction
    if seven_days_ago in data.get('predictions', {}) and seven_days_ago not in data.get('evaluations', {}):
        print(f"  Evaluating prediction from {seven_days_ago}...")
        ev = evaluate_prediction(data['predictions'][seven_days_ago], stocks)
        if ev:
            data.setdefault('evaluations', {})[seven_days_ago] = ev
            print(f"  Hit rate: {ev['hit_rate']}%  |  Positive rate: {ev['positive_rate']}%")

    # Run today's prediction
    if today not in data.get('predictions', {}):
        print("  Running prediction model...")
        top10 = run_prediction(stocks)
        if not top10:
            print("ERROR: Prediction model returned no results.")
            sys.exit(1)

        price_snapshot = {s['ticker']: s['price'] for s in stocks if s.get('price')}
        maturity = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        data.setdefault('predictions', {})[today] = {
            'date':            today,
            'maturity_date':   maturity,
            'predicted_coins': top10,
            'price_snapshot':  price_snapshot,
            'fetched_at':      datetime.now().isoformat(),
        }

        print("  Today's top 10 stocks:")
        for c in top10:
            print(f"    {c['rank']:>2}. {c['ticker']:<8} score={c['score']}  mom7={c['mom7']:+.1f}%  rsi={c['rsi']:.0f}")
    else:
        print(f"  Prediction for {today} already exists — skipping.")

    print("  Saving data...")
    save_data(data)

    print("  Regenerating dashboard...")
    generate_dashboard(data)

    print("Done ✓")
