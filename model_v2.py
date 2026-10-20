"""Model v2 ("M3"): 18 rank-normalised coin signals + market regime, logistic regression.

Used for the 7-day picks. Trained by backtest/train_model_v2.py on CoinMarketCap daily
history; coefficients live in backtest/model_v2_7d.json. The same functions build the
features in training and in the daily tracker, so production matches the backtest.

Inputs for one prediction day (t = last row):
  closes, volumes : arrays [31 days x n coins], oldest -> newest, all finite and > 0
  mcaps           : array [n] market caps on day t
  btc             : column index of Bitcoin
  regime          : dict fng, fng_7, dom, dom_7, mcap_total, mcap_total_7  (today / 7 days ago)
"""
import json
from pathlib import Path
import numpy as np

COIN_FEATS = ["mom1", "mom3", "mom7", "mom14", "mom30", "btcrel3", "btcrel7", "voladj", "vol7", "vmc",
              "vtrend", "vtrend30", "rsi", "hi30", "size", "peer3", "peer7", "tight"]
REGIME = ["fng", "fng_chg7", "dom_chg7", "btc7", "mkt7"]
MODEL_FILE = Path(__file__).resolve().parent / "backtest" / "model_v2_7d.json"
HISTORY_DAYS = 31

DRIVERS = [  # plain-language summary of what the fitted model rewards (shown on the dashboard)
    ("&#128279;", "Moves on its own (low peer correlation)"),
    ("&#127906;", "High recent volatility"),
    ("&#128200;", "Smaller market cap"),
    ("&#128266;", "Rising volume vs 7 & 30 days"),
    ("&#127777;", "Market mood (Fear &amp; Greed, BTC dominance)"),
]


def pct_rank(x):
    """Cross-sectional percentile rank in [0, 1]."""
    order = x.argsort(kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def _rsi14(closes):
    w = closes[-15:]; d = np.diff(w, axis=0)
    g = np.where(d > 0, d, 0).mean(0); l = np.where(d < 0, -d, 0).mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + g / l)
    return np.where(l == 0, 100.0, r)


def coin_features(closes, volumes, mcaps, btc):
    """[n x 18] percentile-ranked signals for the last day of the window."""
    p, v = np.asarray(closes, float), np.asarray(volumes, float)
    t = p.shape[0] - 1
    R = np.diff(p, axis=0) / p[:-1]
    mom = lambda k: (p[t] / p[t - k] - 1) * 100
    m1, m3, m7, m14, m30 = mom(1), mom(3), mom(7), mom(14), mom(30)
    vol7 = np.maximum(R[-7:].std(0), 1e-4)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(R[-30:].T)
    np.fill_diagonal(C, -np.inf)
    peers = np.argsort(-np.nan_to_num(C, nan=-np.inf), axis=1)[:, :8]
    tight = np.take_along_axis(np.where(np.isfinite(C), C, 0), peers, 1).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = dict(
            mom1=m1, mom3=m3, mom7=m7, mom14=m14, mom30=m30,
            btcrel3=m3 - m3[btc], btcrel7=m7 - m7[btc], voladj=m7 / (vol7 * 100 + 1), vol7=vol7,
            vmc=v[t] / mcaps, vtrend=v[t] / v[t - 7:t].mean(0), vtrend30=v[t - 6:t + 1].mean(0) / v[t - 30:t + 1].mean(0),
            rsi=_rsi14(p), hi30=p[t] / p[-31:].max(0), size=np.log(mcaps),
            peer3=m3[peers].mean(1), peer7=m7[peers].mean(1), tight=tight)
    # identical to backtest/experiment_signals.py: NaN -> median, +-inf -> extremes, then rank
    cols = [pct_rank(np.nan_to_num(np.asarray(raw[k], float), nan=np.nanmedian(raw[k]))) for k in COIN_FEATS]
    return np.column_stack(cols), raw


def regime_vector(regime, btc_mom7):
    return np.array([regime["fng"] / 100, (regime["fng"] - regime["fng_7"]) / 100, regime["dom"] - regime["dom_7"],
                     btc_mom7 / 100, regime["mcap_total"] / regime["mcap_total_7"] - 1])


def design(X, reg):
    """Coin signals, regime, and every signal x regime interaction (matches the experiment)."""
    r = np.broadcast_to(reg, (len(X), len(reg)))
    inter = np.concatenate([X * r[:, [j]] for j in range(r.shape[1])], axis=1)
    return np.hstack([X, r, inter])


def load_model(path=MODEL_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_proba(Xd, model):
    z = Xd @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-z))


def score_day(closes, volumes, mcaps, btc, regime, model=None):
    """Probability (0-1) that each coin lands in the real top 10 over the model's horizon."""
    model = model or load_model()
    X, raw = coin_features(closes, volumes, mcaps, btc)
    reg = regime_vector(regime, raw["mom7"][btc])
    return predict_proba(design(X, reg), model), raw
