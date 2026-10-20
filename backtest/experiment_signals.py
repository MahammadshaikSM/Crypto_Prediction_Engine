"""Experiment: can new signals lift the top-10 hit rate well beyond the 5-signal model?

Same data and split as backtest_weights.py:
  train = prediction days whose outcome is known by 2025-09-30 (no look-ahead)
  test  = prediction days after 2025-09-30, scored once
Metric = hit@10: share of the day's 10 picks that land in the real top-10 gainers H days later.

Models
  M0  current formula: 5 signals, min-max, tuned weights (backtest/results.json)
  M1  same 5 signals, rank-normalised, weights re-tuned (hill climb on train)
  M2  extended signals (rank-normalised) -> logistic regression
  M3  M2 + market-regime interactions (Fear & Greed, BTC dominance trend, BTC trend)
  M4  gradient-boosted trees on extended signals + regime (non-linear)

New signals: 3/14/30-day momentum, 3-day BTC-relative, volume trend (today vs 7d avg,
7d vs 30d), distance from 30-day high, size (log market cap), peer momentum
(avg 3/7-day return of the 8 coins most correlated with this one over 30 days, a
data-driven "memecoin / sector cluster" signal), cluster tightness.

Usage: python backtest/experiment_signals.py --dir <folder with daily.csv, coins.csv, global.csv, fng.csv>
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest_weights as bw  # noqa: E402

TRAIN_END = bw.TRAIN_END
HORIZONS = (3, 7)


def series(path, col):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["date"]] = float(r[col])
            except (TypeError, ValueError):
                pass
    return out


def pct_rank(x):
    """Cross-sectional percentile rank in [0, 1] (ties averaged)."""
    order = x.argsort(kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


COIN_FEATS = ["mom1", "mom3", "mom7", "mom14", "mom30", "btcrel3", "btcrel7", "voladj", "vol7", "vmc",
              "vtrend", "vtrend30", "rsi", "hi30", "size", "peer3", "peer7", "tight"]
REGIME = ["fng", "fng_chg7", "dom_chg7", "btc7", "mkt7"]


def build(dir_):
    args = argparse.Namespace(csv=str(Path(dir_) / "daily.csv"), xlsx=None)
    dates, ids, sym, P, V, M = bw.load_matrices(args)
    btc = ids.index("1")
    glob_dom = series(Path(dir_) / "global.csv", "BTC dominance %")
    glob_mc = series(Path(dir_) / "global.csv", "Total market cap (USD)")
    fng = series(Path(dir_) / "fng.csv", "Index (0-100)")
    R = np.diff(P, axis=0) / P[:-1]                     # daily returns; R[t-1] = return into day t
    days = []
    for t in range(31, len(dates)):
        d = dates[t]; d7 = dates[t - 7]
        if d not in fng or dates[t - 7] not in fng or d not in glob_dom or d7 not in glob_dom:
            continue
        win = P[t - 30:t + 1]
        ok = np.isfinite(win).all(0) & (win > 0).all(0) & (V[t] > 0) & (M[t] > 0) & np.isfinite(V[t - 30:t + 1]).all(0)
        idx = np.where(ok)[0]
        if len(idx) < 40 or not ok[btc]:
            continue
        p = P[:, idx]
        mom = lambda k: (p[t] / p[t - k] - 1) * 100
        m1, m3, m7, m14, m30 = mom(1), mom(3), mom(7), mom(14), mom(30)
        b = list(idx).index(btc)
        r7 = R[t - 7:t, idx]; vol7 = np.maximum(r7.std(0), 1e-4)
        v = V[:, idx]
        # peer momentum from 30-day return correlations
        C = np.corrcoef(R[t - 30:t, idx].T); np.fill_diagonal(C, -np.inf)
        peers = np.argsort(-C, axis=1)[:, :8]
        tight = np.take_along_axis(np.where(np.isfinite(C), C, 0), peers, 1).mean(1)
        raw = dict(
            mom1=m1, mom3=m3, mom7=m7, mom14=m14, mom30=m30,
            btcrel3=m3 - m3[b], btcrel7=m7 - m7[b], voladj=m7 / (vol7 * 100 + 1), vol7=vol7,
            vmc=V[t, idx] / M[t, idx], vtrend=v[t] / v[t - 7:t].mean(0), vtrend30=v[t - 6:t + 1].mean(0) / v[t - 30:t + 1].mean(0),
            rsi=bw.rsi14(P[:, idx], t), hi30=p[t] / win[:, idx].max(0), size=np.log(M[t, idx]),
            peer3=m3[peers].mean(1), peer7=m7[peers].mean(1), tight=tight)
        X = np.column_stack([pct_rank(np.nan_to_num(raw[k], nan=np.nanmedian(raw[k]))) for k in COIN_FEATS])
        reg = np.array([fng[d] / 100, (fng[d] - fng[d7]) / 100, glob_dom[d] - glob_dom[d7],
                        m7[b] / 100, (glob_mc[d] / glob_mc[d7] - 1) if d7 in glob_mc and d in glob_mc else 0.0])
        # the current production features (min-max, as in prediction_tracker.py)
        F5 = np.column_stack([bw.norm(raw["voladj"]), bw.norm(m1), bw.norm(raw["btcrel7"]), bw.norm(raw["vmc"]),
                              bw.rsi_score(raw["rsi"])])
        out = {}
        for h in HORIZONS:
            if t + h >= len(dates):
                continue
            fwd = (P[t + h, idx] / P[t, idx] - 1) * 100
            if not np.isfinite(fwd).all():
                good = np.isfinite(fwd); fwd = np.where(good, fwd, -1e9)
            top = np.zeros(len(idx), bool); top[np.argsort(-fwd)[:10]] = True
            out[h] = dict(top=top, ret=np.where(fwd < -1e8, np.nan, fwd), n=int((fwd > -1e8).sum()), end=dates[t + h])
        days.append(dict(date=d, idx=idx, X=X, reg=reg, F5=F5, out=out))
    return days


def split(days, h):
    tr = [d for d in days if h in d["out"] and d["out"][h]["end"] <= TRAIN_END]
    te = [d for d in days if h in d["out"] and d["date"] > TRAIN_END]
    return tr, te


def score(days, h, scorer):
    hits, base, pos, edge = [], [], [], []
    for d in days:
        o = d["out"][h]; s = scorer(d); pick = np.argsort(-s)[:10]
        hits.append(o["top"][pick].sum() * 10); base.append(1000 / o["n"])
        pos.append(np.nansum(o["ret"][pick] > 0) * 10); edge.append(np.nanmean(o["ret"][pick]) - np.nanmean(o["ret"]))
    return dict(hit=float(np.mean(hits)), base=float(np.mean(base)), pos=float(np.mean(pos)),
                edge=float(np.mean(edge)), days=len(days), daily=hits)


def design(d, regime):
    X = d["X"]
    if not regime:
        return X
    r = np.broadcast_to(d["reg"], (len(X), len(d["reg"])))
    inter = np.concatenate([X * r[:, [j]] for j in range(r.shape[1])], axis=1)
    return np.hstack([X, r, inter])


def stack(days, h, regime):
    Xs = np.vstack([design(d, regime) for d in days]); y = np.concatenate([d["out"][h]["top"] for d in days])
    return Xs, y.astype(int)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    days = build(a.dir)
    print(f"{len(days)} days with full 30-day history, {days[0]['date']} -> {days[-1]['date']}")
    tuned = json.loads((HERE / "results.json").read_text())["horizons"]
    report = {}
    for h in HORIZONS:
        tr, te = split(days, h)
        val = tr[-120:]; fit = tr[:-120]
        res = {}
        w0 = np.array([tuned[str(h)]["weights"][k] for k in bw.FEATURES]) / 100
        res["M0 current formula"] = score(te, h, lambda d: d["F5"] @ w0)
        # M1: rank-normalised 5 signals, hill-climb from the tuned weights on train
        cols5 = [COIN_FEATS.index(k) for k in ("voladj", "mom1", "btcrel7", "vmc", "rsi")]
        def m1(w): return lambda d: d["X"][:, cols5] @ (np.array(w) * np.array([1, 1, 1, 1, -1]))  # low RSI = good
        w = list(w0 * 100); best = score(tr, h, m1(w))["hit"]
        for step in (5.0, 1.0):
            improved = True
            while improved:
                improved = False
                for i in range(5):
                    for j in range(5):
                        if i == j or w[i] < step: continue
                        cand = list(w); cand[i] -= step; cand[j] += step
                        s = score(tr, h, m1(cand))["hit"]
                        if s > best + 1e-9: best, w, improved = s, cand, True
        res["M1 rank-normalised 5"] = score(te, h, m1(w)); res["M1 rank-normalised 5"]["weights"] = [round(x, 1) for x in w]
        # M2 / M3: logistic regression, C chosen on the last 120 train days
        for name, regime in (("M2 extended linear", False), ("M3 extended + regime", True)):
            Xf, yf = stack(fit, h, regime); best_c, best_v = None, -1
            for C in (0.01, 0.1, 1.0):
                mdl = LogisticRegression(C=C, max_iter=2000).fit(Xf, yf)
                v = score(val, h, lambda d: mdl.decision_function(design(d, regime)))["hit"]
                if v > best_v: best_v, best_c = v, C
            Xa, ya = stack(tr, h, regime); mdl = LogisticRegression(C=best_c, max_iter=3000).fit(Xa, ya)
            res[name] = score(te, h, lambda d: mdl.decision_function(design(d, regime)))
            if not regime:
                coefs = dict(zip(COIN_FEATS, np.round(mdl.coef_[0], 2)))
                res[name]["top_signals"] = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:8]
        # M4: gradient-boosted trees (non-linear), early-stopped on the validation slice
        Xf, yf = stack(fit, h, True); best_it, best_v, best_m = None, -1, None
        for depth in (3, 5):
            mdl = HistGradientBoostingClassifier(max_depth=depth, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                                 random_state=0).fit(Xf, yf)
            v = score(val, h, lambda d: mdl.predict_proba(design(d, True))[:, 1])["hit"]
            if v > best_v: best_v, best_m = v, depth
        Xa, ya = stack(tr, h, True)
        mdl = HistGradientBoostingClassifier(max_depth=best_m, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                             random_state=0).fit(Xa, ya)
        res["M4 boosted trees + regime"] = score(te, h, lambda d: mdl.predict_proba(design(d, True))[:, 1])
        print(f"\n===== {h}-day horizon | train {len(tr)} days | test {len(te)} days =====")
        for k, r in res.items():
            extra = f" weights={r['weights']}" if "weights" in r else ""
            print(f" {k:<28} hit {r['hit']:5.2f}%  (random {r['base']:.2f}%) | up {r['pos']:4.1f}% | edge {r['edge']:+.2f}pp{extra}")
            if "top_signals" in r: print("    strongest linear signals:", r["top_signals"])
        report[h] = {k: {kk: (list(map(float, vv)) if kk == "daily" else vv) for kk, vv in r.items()} for k, r in res.items()}
        best_name = max(res, key=lambda k: res[k]["hit"]); daily = np.array(res[best_name]["daily"])
        print(f" best on test: {best_name} | days >=40%: {np.mean(daily >= 40)*100:.1f}% | best 30-day avg: "
              f"{max(daily[i:i+30].mean() for i in range(len(daily)-29)):.1f}%")
    (HERE / "experiment_results.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
