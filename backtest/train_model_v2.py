"""Train model v2 (M3) for the 7-day picks and export it to backtest/model_v2_7d.json.

1. Parity check: model_v2.coin_features() must reproduce experiment_signals.build() exactly.
2. Choose the regularisation C on the last 120 training days (outcomes known by 2025-09-30).
3. Report the held-out score (fit on train only, scored on Oct 2025 - Sep 2026).
4. Refit on every day with a known outcome and export coefficients for production.

Usage: python backtest/train_model_v2.py --dir <folder with daily.csv, coins.csv, global.csv, fng.csv>
"""
import argparse, json, sys
from datetime import date
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import backtest_weights as bw      # noqa: E402
import experiment_signals as ex    # noqa: E402
import model_v2 as mv              # noqa: E402

H = 7


def parity(dir_, days):
    args = argparse.Namespace(csv=str(Path(dir_) / "daily.csv"), xlsx=None)
    dates, ids, sym, P, V, M = bw.load_matrices(args)
    di = {d: i for i, d in enumerate(dates)}; btc = ids.index("1"); worst = 0.0
    for d in days[::50]:
        t = di[d["date"]]; idx = d["idx"]; b = list(idx).index(btc)
        X, _ = mv.coin_features(P[t - 30:t + 1, idx], V[t - 30:t + 1, idx], M[t, idx], b)
        worst = max(worst, float(np.abs(X - d["X"]).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    days = ex.build(a.dir)
    diff = parity(a.dir, days)
    print(f"parity: max |production - experiment| feature difference = {diff:.2e}")
    assert diff < 1e-9, "production features drift from the backtest features"

    tr, te = ex.split(days, H)
    fit, val = tr[:-120], tr[-120:]
    best_c, best_v = None, -1
    for C in (0.01, 0.1, 1.0):
        Xf, yf = ex.stack(fit, H, True)
        m = LogisticRegression(C=C, max_iter=3000).fit(Xf, yf)
        v = ex.score(val, H, lambda d: m.decision_function(ex.design(d, True)))["hit"]
        print(f"  C={C}: validation hit {v:.2f}%")
        if v > best_v: best_v, best_c = v, C

    Xt, yt = ex.stack(tr, H, True)
    m = LogisticRegression(C=best_c, max_iter=3000).fit(Xt, yt)
    held = ex.score(te, H, lambda d: m.decision_function(ex.design(d, True)))
    w0 = np.array([json.loads((HERE / "results.json").read_text())["horizons"]["7"]["weights"][k] for k in bw.FEATURES]) / 100
    base = ex.score(te, H, lambda d: d["F5"] @ w0)
    print(f"held-out (Oct 2025 - Sep 2026): model v2 {held['hit']:.2f}% vs current formula {base['hit']:.2f}% vs random {held['base']:.2f}%")

    allk = [d for d in days if H in d["out"]]
    Xa, ya = ex.stack(allk, H, True)
    final = LogisticRegression(C=best_c, max_iter=4000).fit(Xa, ya)
    out = dict(
        name="model v2 (M3): 18 signals + market regime, logistic regression", horizon_days=H,
        trained_on=[allk[0]["date"], allk[-1]["date"]], trained=date.today().isoformat(), C=best_c,
        coin_features=mv.COIN_FEATS, regime=mv.REGIME, n_features=int(Xa.shape[1]),
        coef=[float(x) for x in final.coef_[0]], intercept=float(final.intercept_[0]),
        heldout=dict(hit=held["hit"], base=held["base"], pos=held["pos"], edge=held["edge"], days=held["days"],
                     current_formula_hit=base["hit"], period="2025-10-01 to 2026-09-30"))
    (HERE / "model_v2_7d.json").write_text(json.dumps(out, indent=1))
    print(f"exported backtest/model_v2_7d.json ({Xa.shape[1]} coefficients, C={best_c}, {len(allk)} training days)")


if __name__ == "__main__":
    main()
