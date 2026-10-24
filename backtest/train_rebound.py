"""Train the Rebound Radar model and export it to backtest/rebound_model.json.

Events: every (day, coin) where the coin is down 25%+ over 30 days (BTC and stablecoins excluded).
Outcomes over the next 60 days:
  comeback = price gets back to its level from 30 days before the event at any point (the target)
  bounce   = price is 25%+ above the event price at some point within 30 days (reported)
  falling  = price on day 60 is 20%+ below the event price (reported)

1. Held-out test: fit on events whose outcome is known by 2025-09-30, score Oct 2025 onwards,
   rank each day's coins and report outcomes by verdict (Comeback / Watching / Unlikely).
2. Refit on every event with a known outcome and export the coefficients.

A second model for "falls another 20%" ranked well across time (AUC ~0.72) but not between the
coins of the same day, and absolute probabilities drifted between years, so it is not used.

Caveat: the history holds the coins that are in today's top 200, so coins that died and
dropped out are missing. Real comeback rates are lower than these; the live tracker grades
every coin it flags, so its scorecard has no such bias.

Usage: python backtest/train_rebound.py --dir <folder with daily.csv, coins.csv, global.csv, fng.csv>
"""
import argparse, csv, json, sys
from datetime import date
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import backtest_weights as bw      # noqa: E402
import rebound_model as rm         # noqa: E402

C = 0.3


def series(path, col):
    with open(path, encoding="utf-8") as fh:
        return {r["date"]: float(r[col]) for r in csv.DictReader(fh) if r[col]}


def events(dir_):
    dates, ids, sym, P, V, M = bw.load_matrices(argparse.Namespace(csv=str(Path(dir_) / "daily.csv"), xlsx=None))
    fng = series(Path(dir_) / "fng.csv", "Index (0-100)")
    dom = series(Path(dir_) / "global.csv", "BTC dominance %")
    T = len(dates); btc = ids.index("1"); H = rm.HORIZON
    X, Y, D = [], {"comeback": [], "bounce": [], "falling": []}, []
    for t in range(rm.HISTORY_DAYS - 1, T):
        d, d7 = dates[t], dates[t - 7]
        if not all(k in s for k in (d, d7) for s in (fng, dom)):
            continue
        w = P[t - 90:t + 1]
        ok = np.isfinite(w).all(0) & (w > 0).all(0) & np.isfinite(V[t - 30:t + 1]).all(0) & (M[t] > 0)
        r30 = P[t] / P[t - 30] - 1
        regime = {"fng": fng[d], "fng_7": fng[d7], "dom": dom[d], "dom_7": dom[d7]}
        for j in np.where(ok & (r30 <= rm.DECLINE))[0]:
            if j == btc:
                continue
            X.append(rm.coin_features(P[t - 90:t + 1, j], V[t - 30:t + 1, j], M[t, j], P[t - 30:t + 1, btc], regime))
            D.append(t)
            fw = P[t + 1:t + H + 1, j]
            if t + H < T and np.isfinite(fw).all():
                Y["comeback"].append(float(fw.max() >= P[t - 30, j]))
                Y["bounce"].append(float(fw[:30].max() / P[t, j] - 1 >= rm.BOUNCE))
                Y["falling"].append(float(fw[-1] / P[t, j] - 1 <= rm.FELL))
            else:
                for k in Y:
                    Y[k].append(np.nan)
    return dates, np.array(X), {k: np.array(v) for k, v in Y.items()}, np.array(D)


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=4000).fit(np.clip((X - mu) / sd, -4, 4), y)
    return dict(mu=mu.tolist(), sd=sd.tolist(), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    dates, X, Y, D = events(a.dir)
    lab = ~np.isnan(Y["comeback"])
    cut = dates.index("2025-09-30")
    tr, te = lab & (D + rm.HORIZON <= cut), lab & (D > cut)
    print(f"events {len(X)}, with known outcome {lab.sum()}, train {tr.sum()}, held-out {te.sum()}")

    m = fit(X[tr], Y["comeback"][tr])
    p = np.array([rm.prob(m, x) for x in X])
    rank = np.zeros(len(X))
    for t in np.unique(D):
        i = np.where(D == t)[0]; rank[i] = rm.day_rank(p[i])
    v = np.array([rm.verdict(r) for r in rank])
    auc = round(float(roc_auc_score(Y["comeback"][te], p[te])), 3)
    print(f"  held-out AUC {auc}")
    groups = {}
    for g in rm.VERDICTS:
        s = te & (v == g)
        groups[g] = dict(n=int(s.sum()), **{k: round(float(Y[k][s].mean()), 3) for k in Y})
        print(f"  {g:9s} n={s.sum():5d}  comeback {groups[g]['comeback']:.1%}  bounce {groups[g]['bounce']:.1%}  falling {groups[g]['falling']:.1%}")
    allg = {k: round(float(Y[k][te].mean()), 3) for k in Y}
    print(f"  all       n={te.sum():5d}  comeback {allg['comeback']:.1%}  bounce {allg['bounce']:.1%}  falling {allg['falling']:.1%}")

    final = fit(X[lab], Y["comeback"][lab])
    top = sorted(zip(rm.FEATS, final["coef"]), key=lambda z: -abs(z[1]))[:6]
    print("  drivers: " + ", ".join(f"{f} {c:+.2f}" for f, c in top))
    out = dict(name="Rebound Radar: logistic regression on 20 decline signals", trained=date.today().isoformat(),
               trained_on=[dates[int(D[lab].min())], dates[int(D[lab].max())]], features=rm.FEATS, C=C,
               decline=rm.DECLINE, horizon_days=rm.HORIZON, bounce=rm.BOUNCE, fell=rm.FELL,
               heldout=dict(period=f"{dates[cut + 1]} to {dates[int(D[te].max())]}", events=int(te.sum()),
                            auc=auc, groups=groups, all=allg),
               model=final)
    (HERE / "rebound_model.json").write_text(json.dumps(out, indent=1))
    print(f"exported backtest/rebound_model.json ({lab.sum()} events)")


if __name__ == "__main__":
    main()
