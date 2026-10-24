"""Rebound Radar model: after a month-long decline, which coins come back.

A coin is "in decline" when its price is down 25% or more over 30 days. For every such coin a
logistic regression (trained by backtest/train_rebound.py on 3 years of CoinMarketCap daily
history, coefficients in backtest/rebound_model.json) scores its chance of getting back to its
price from 30 days earlier within the next 60 days. Coins are then ranked against the other
declining coins of the same day:
  top 30%     Comeback candidate
  middle 40%  Watching
  bottom 30%  Unlikely to recover
The same feature function runs in training and in the daily tracker, so production matches
the backtest.

Inputs for one coin on day t (t = last element):
  closes  : 91 daily closes (oldest -> newest)
  volumes : 31 daily 24h volumes
  mcap    : market cap on day t
  btc     : 31 BTC daily closes
  regime  : dict fng, fng_7, dom, dom_7 (today / 7 days ago), as cmc_client.market_regime()
"""
import json
from pathlib import Path
import numpy as np

FEATS = ["r30", "r14", "r7", "r3", "r1", "down", "hi90", "lo90", "vtrend", "vmc", "size", "rsi",
         "vol30", "btc30", "btc7", "rel30", "fng", "fng7", "dom7", "daysfromhi"]
DECLINE = -0.25        # 30-day return that puts a coin on the radar
HORIZON = 60           # days a coin is tracked before it is graded
BOUNCE = 0.25          # "bounced" = +25% above the entry price at some point
FELL = -0.20           # "fell further" = 20% or more below the entry price when graded
HISTORY_DAYS = 91
TOP, BOTTOM = 70, 30   # day-rank percentiles for the verdicts
MODEL_FILE = Path(__file__).resolve().parent / "backtest" / "rebound_model.json"

VERDICTS = {
    "comeback": "Comeback candidate",
    "watch": "Watching",
    "unlikely": "Unlikely to recover",
}


def _rsi14(p):
    d = np.diff(p[-15:])
    g = np.where(d > 0, d, 0).mean(); l = np.where(d < 0, -d, 0).mean()
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def coin_features(closes, volumes, mcap, btc, regime):
    p, v, b = np.asarray(closes, float), np.asarray(volumes, float), np.asarray(btc, float)
    r = lambda k: p[-1] / p[-1 - k] - 1
    r30, b30 = r(30), b[-1] / b[-31] - 1
    return np.array([
        r30, r(14), r(7), r(3), r(1),
        (np.diff(p[-31:]) < 0).mean(),
        p[-1] / p.max() - 1, p[-1] / p.min() - 1,
        np.log(v[-7:].mean() / max(v.mean(), 1)), np.log(max(v[-1], 1) / mcap), np.log(mcap),
        _rsi14(p), np.std(np.diff(np.log(p[-31:]))),
        b30, b[-1] / b[-8] - 1, r30 - b30,
        regime["fng"], regime["fng"] - regime["fng_7"], regime["dom"] - regime["dom_7"],
        len(p) - 1 - int(np.argmax(p)),
    ])


def load(path=MODEL_FILE):
    return json.loads(Path(path).read_text())


def prob(m, x):
    """Comeback probability for one feature vector."""
    z = np.clip((np.asarray(x, float) - np.array(m["mu"])) / np.array(m["sd"]), -4, 4)
    return float(1 / (1 + np.exp(-(z @ np.array(m["coef"]) + m["intercept"]))))


def day_rank(probs):
    """Percentile (0-100) of each score among the same day's declining coins."""
    p = np.asarray(probs, float)
    if len(p) < 2:
        return np.full(len(p), 50.0)
    return (np.argsort(np.argsort(p, kind="mergesort")) + 0.5) / len(p) * 100


def verdict(rank):
    return "comeback" if rank >= TOP else "unlikely" if rank < BOTTOM else "watch"
