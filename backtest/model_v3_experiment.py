"""Can the 7-day model be improved for traders? Variants of model v2, judged on basket return after costs.

Protocol (no peeking):
  fit       = training days before the last 120 (outcomes known by 2025-09-30)
  validate  = the last 120 training days: variants are RANKED here only
  held-out  = Oct 2025 - Sep 2026: reported once, for every variant, after the ranking is fixed

Variants
  v2            current production model: logistic regression, target = top-10 finish
  v2_ret        same features, ridge regression on the coin's forward return rank that day
  v2_beat       same features, logistic regression, target = beat the day's median coin
  v2_nopump     v2, but skip coins up >100% in 7 days or with RSI >85 (next best coins fill in)
  v2_caution    v2 on TRADE days; on CAUTION days (BTC 7d < 0) skip pumps as above
  v2_invvol     v2 picks, weighted by 1 / 7-day volatility instead of equally
  blend         average of the v2 and v2_ret ranks

Metrics: average basket return after 0.3% costs; edge vs the top-200 average on non-overlapping weeks
with a bootstrap 90% range; worst drawdown of a one-call-a-week equity curve; share of weeks positive.

Usage: python backtest/model_v3_experiment.py --dir <folder with daily.csv, coins.csv, global.csv, fng.csv>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest_weights as bw      # noqa: E402
import experiment_signals as ex    # noqa: E402

H, COST = 7, 0.003
rng = np.random.default_rng(11)


def rank01(x):
    return ex.pct_rank(np.asarray(x, float))


def fit_models(days):
    C = json.loads((HERE / "model_v2_7d.json").read_text())["C"]
    X = np.vstack([ex.design(d, True) for d in days])
    y_top = np.concatenate([d["out"][H]["top"] for d in days]).astype(int)
    rets = [np.nan_to_num(d["out"][H]["ret"], nan=-100.0) for d in days]
    y_rank = np.concatenate([rank01(r) for r in rets])
    y_beat = np.concatenate([(r > np.median(r)).astype(int) for r in rets])
    m_top = LogisticRegression(C=C, max_iter=3000).fit(X, y_top)
    m_ret = Ridge(alpha=50.0).fit(X, y_rank)
    m_beat = LogisticRegression(C=C, max_iter=3000).fit(X, y_beat)
    return m_top, m_ret, m_beat


def day_inputs(d, P, di):
    t = di[d["date"]]; idx = d["idx"]
    mom7 = P[t, idx] / P[t - 7, idx] - 1
    rsi = bw.rsi14(P[:, idx], t)
    R = np.diff(np.log(P[t - 7:t + 1, idx]), axis=0)
    vol7 = np.maximum(np.nanstd(R, axis=0), 1e-3)
    return t, idx, mom7, rsi, vol7


def choose(d, P, di, models, variant):
    """Return (column indexes of the 10 picks, weights) for one day."""
    m_top, m_ret, m_beat = models
    Xd = ex.design(d, True)
    s_top = m_top.decision_function(Xd)
    t, idx, mom7, rsi, vol7 = day_inputs(d, P, di)
    pump = (mom7 > 1.0) | (rsi > 85)
    if variant == "v2":
        s, mask = s_top, None
    elif variant == "v2_ret":
        s, mask = m_ret.predict(Xd), None
    elif variant == "v2_beat":
        s, mask = m_beat.decision_function(Xd), None
    elif variant == "v2_nopump":
        s, mask = s_top, pump
    elif variant == "v2_caution":
        s, mask = s_top, pump if d["reg"][3] < 0 else None
    elif variant == "v2_invvol":
        s, mask = s_top, None
    elif variant == "blend":
        s, mask = rank01(s_top) + rank01(m_ret.predict(Xd)), None
    else:
        raise ValueError(variant)
    s = np.where(mask, -np.inf, s) if mask is not None else s
    pick = np.argsort(-s)[:10]
    w = (1 / vol7[pick]) if variant == "v2_invvol" else np.ones(10)
    return t, idx, pick, w / w.sum()


def evaluate(days, P, di, models, variant):
    baskets, market, dates = [], [], []
    for d in days:
        t, idx, pick, w = choose(d, P, di, models, variant)
        r = P[t + H, idx] / P[t, idx] - 1
        ok = np.isfinite(r[pick])
        if ok.sum() < 8:
            continue
        ww = w[ok] / w[ok].sum()
        baskets.append(float((r[pick][ok] * ww).sum()) - COST)
        market.append(float(np.nanmean(r[np.isfinite(r)])) - COST)
        dates.append(d["date"])
    b, m = np.array(baskets), np.array(market)
    # non-overlapping weeks, every start offset; report the median offset
    res = []
    for off in range(H):
        x = (b - m)[off::H]
        boot = rng.choice(x, (3000, len(x))).mean(1)
        eq = np.cumprod(1 + b[off::H]); dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        res.append((x.mean(), np.percentile(boot, 5), np.percentile(boot, 95), dd, float((b[off::H] > 0).mean()), eq[-1] - 1))
    res.sort(key=lambda z: z[0])
    e, lo, hi, dd, wk, eq = res[len(res) // 2]
    return dict(days=len(b), avg=float(b.mean()), market=float(m.mean()), edge=float(e), lo=float(lo), hi=float(hi),
                drawdown=dd, weeks_positive=wk, equity=float(eq), worst_offset_edge=float(res[0][0]))


VARIANTS = ["v2", "v2_ret", "v2_beat", "v2_nopump", "v2_caution", "v2_invvol", "blend"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    dates, ids, sym, P, V, M = bw.load_matrices(argparse.Namespace(csv=str(Path(a.dir) / "daily.csv"), xlsx=None))
    di = {d: i for i, d in enumerate(dates)}
    days = ex.build(a.dir)
    tr, te = ex.split(days, H)
    fit, val = tr[:-120], tr[-120:]

    print(f"fit {len(fit)} days, validate {len(val)} days ({val[0]['date']} -> {val[-1]['date']})")
    models = fit_models(fit)
    vres = {v: evaluate(val, P, di, models, v) for v in VARIANTS}
    order = sorted(VARIANTS, key=lambda v: -vres[v]["avg"])
    print("\nVALIDATION (ranks the variants; held-out not used yet)")
    for v in order:
        r = vres[v]
        print(f"  {v:<11} avg {r['avg']*100:+6.2f}% | edge vs market {r['edge']*100:+5.2f}% ({r['lo']*100:+.2f}..{r['hi']*100:+.2f})"
              f" | drawdown {r['drawdown']*100:4.0f}% | weeks up {r['weeks_positive']:.0%}")
    winner = order[0]
    print(f"  -> chosen on validation: {winner}")

    models = fit_models(tr)          # refit on all training days, as production would
    hres = {v: evaluate(te, P, di, models, v) for v in VARIANTS}
    print(f"\nHELD-OUT ({te[0]['date']} -> {te[-1]['date']}), reported once")
    for v in VARIANTS:
        r = hres[v]; tag = "  <- chosen" if v == winner else ""
        print(f"  {v:<11} avg {r['avg']*100:+6.2f}% (market {r['market']*100:+.2f}%) | edge {r['edge']*100:+5.2f}%"
              f" ({r['lo']*100:+.2f}..{r['hi']*100:+.2f}, worst start {r['worst_offset_edge']*100:+.2f}%)"
              f" | drawdown {r['drawdown']*100:4.0f}% | weeks up {r['weeks_positive']:.0%} | equity {r['equity']*100:+.0f}%{tag}")
    (HERE / "model_v3_results.json").write_text(json.dumps(dict(validation=vres, heldout=hres, chosen=winner), indent=1))


if __name__ == "__main__":
    main()
