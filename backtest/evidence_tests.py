"""Evidence tests for the daily calls, on the held-out year (Oct 2025 - Sep 2026), after costs.

1. Matched baseline: for every pick, a random coin from the same day with similar 7-day volatility and
   market cap (one of the 5 nearest non-picks). If the calls only beat random because they pick wild
   coins, the matched basket will do just as well.
2. Independent weeks: overlapping windows are not independent, so results are recomputed on
   non-overlapping windows (every H-th day, all H start offsets) with bootstrap 90% confidence ranges.
3. Calibration (7-day): does a pick rated "30% chance of a top-10 finish" land in the top 10 ~30% of the time?
4. Liquidity: how much 24h volume the picks have, returns with thin coins swapped for the next pick,
   and returns at higher trading costs.

Picks come from models fit on training days only (as in trader_backtest.py), so nothing here was seen
during fitting. Writes backtest/evidence_results.json.

Usage: python backtest/evidence_tests.py --dir <folder with daily.csv, coins.csv, global.csv, fng.csv>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest_weights as bw      # noqa: E402
import experiment_signals as ex    # noqa: E402

COST = 0.003
DRAWS = 200
VOL, SIZE = ex.COIN_FEATS.index("vol7"), ex.COIN_FEATS.index("size")
rng = np.random.default_rng(7)


def scorers(days):
    out = {}
    tr, _ = ex.split(days, 7)
    C = json.loads((HERE / "model_v2_7d.json").read_text())["C"]
    Xt, yt = ex.stack(tr, 7, True)
    m = LogisticRegression(C=C, max_iter=3000).fit(Xt, yt)
    out[7] = (lambda d: m.decision_function(ex.design(d, True)), lambda d: m.predict_proba(ex.design(d, True))[:, 1])
    w = np.array([json.loads((HERE / "results.json").read_text())["horizons"]["3"]["weights"][k] for k in bw.FEATURES]) / 100
    out[3] = (lambda d: d["F5"] @ w, None)
    return out


def ci(x, reps=4000):
    x = np.asarray(x, float)
    b = rng.choice(x, (reps, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 5)), float(np.percentile(b, 95))


def day_returns(P, t, h, cols):
    r = P[t + h, cols] / P[t, cols] - 1
    return r


def run(days, P, V, di, btc, h, score, proba):
    _, te = ex.split(days, h)
    rows = []
    for d in te:
        t = di[d["date"]]; idx = d["idx"]
        s = score(d); order = np.argsort(-s); pick = order[:10]
        ret = day_returns(P, t, h, idx)
        ok = np.isfinite(ret)
        if ok[pick].sum() < 8:
            continue
        # matched baseline: each pick -> random one of the 5 nearest non-picks on (vol7 rank, size rank)
        rest = np.setdiff1d(np.where(ok)[0], pick)
        dist = np.abs(d["X"][pick][:, None, VOL] - d["X"][rest][None, :, VOL]) + \
               np.abs(d["X"][pick][:, None, SIZE] - d["X"][rest][None, :, SIZE])
        near = rest[np.argsort(dist, axis=1)[:, :5]]                       # [10 x 5]
        draws = near[np.arange(10)[None, :], rng.integers(0, 5, (DRAWS, 10))]  # [DRAWS x 10]
        matched = np.nanmean(ret[draws], axis=1) - COST                    # basket return per draw
        top = np.zeros(len(idx), bool); top[np.argsort(-np.where(ok, ret, -np.inf))[:10]] = True
        # liquidity: 24h volume of picks; swap picks under $1M volume for the next-ranked liquid coin
        vol = V[t, idx]
        liquid = order[np.isfinite(vol[order]) & (vol[order] >= 1e6) & ok[order]][:10]
        rows.append(dict(
            date=d["date"], t=t,
            pick=float(np.nanmean(ret[pick]) - COST), pick_raw=float(np.nanmean(ret[pick])),
            market=float(np.nanmean(ret[ok]) - COST), btc=float(P[t + h, btc] / P[t, btc] - 1),
            matched=float(matched.mean()), matched_draws=matched,
            pick_up=float(np.nanmean(ret[pick] > 0)), market_up=float(np.mean(ret[ok] > 0)),
            matched_up=float(np.nanmean(ret[draws] > 0)),
            pick_hit=float(top[pick].mean()), matched_hit=float(top[draws].mean()),
            liquid=float(np.nanmean(ret[liquid]) - COST) if len(liquid) else np.nan,
            vol_picks=vol[pick], vol_all=vol[ok],
            proba=proba(d)[pick] if proba else None, proba_all=proba(d) if proba else None,
            top_all=top, hit_vec=top[pick]))
    return rows


def independent(rows, h, key_a, key_b=None):
    """Mean of (a - b) on non-overlapping windows, per start offset; median offset reported with CI."""
    res = []
    for off in range(h):
        x = [r[key_a] - (r[key_b] if key_b else 0) for r in rows[off::h]]
        res.append(ci(x) + (len(x),))
    res.sort(key=lambda z: z[0])
    m, lo, hi, n = res[len(res) // 2]
    return dict(mean=m, lo=lo, hi=hi, windows=n, offsets_positive=sum(z[0] > 0 for z in res), offsets=h,
                worst_offset=res[0][0], best_offset=res[-1][0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    dates, ids, sym, P, V, M = bw.load_matrices(argparse.Namespace(csv=str(Path(a.dir) / "daily.csv"), xlsx=None))
    di = {d: i for i, d in enumerate(dates)}; btc = ids.index("1")
    days = ex.build(a.dir)
    sc = scorers(days)
    report = {}
    for h in (7, 3):
        rows = run(days, P, V, di, btc, h, *sc[h])
        f = lambda k: float(np.nanmean([r[k] for r in rows]))
        print(f"\n===== {h}-day calls, held-out {rows[0]['date']} -> {rows[-1]['date']} ({len(rows)} calls, after {COST:.1%} costs) =====")
        print("1) Matched baseline (similar volatility and size)")
        print(f"   avg return per call: picks {f('pick')*100:+.2f}% | matched random {f('matched')*100:+.2f}% |"
              f" top-200 average {f('market')*100:+.2f}% | BTC {f('btc')*100:+.2f}%")
        print(f"   share of coins that went up: picks {f('pick_up'):.1%} | matched {f('matched_up'):.1%} | top-200 {f('market_up'):.1%}")
        print(f"   top-10 hit rate: picks {f('pick_hit'):.1%} | matched {f('matched_hit'):.1%} | random {10/np.mean([len(r['top_all']) for r in rows]):.1%}")
        # share of days the picks beat the matched draws
        beat = np.mean([np.mean(r["pick"] > r["matched_draws"]) for r in rows])
        print(f"   picks beat a matched basket on {beat:.0%} of (day, draw) pairs")

        print("2) Independent windows (non-overlapping, median of all start offsets, 90% range)")
        out = {}
        for name, a_, b_ in (("vs nothing", "pick", None), ("vs matched", "pick", "matched"),
                             ("vs top-200 avg", "pick", "market"), ("vs BTC", "pick", "btc")):
            z = independent(rows, h, a_, b_); out[name] = z
            print(f"   {name:<15} {z['mean']*100:+.2f}%  (90% range {z['lo']*100:+.2f}% .. {z['hi']*100:+.2f}%)"
                  f"  windows {z['windows']}  offsets positive {z['offsets_positive']}/{z['offsets']}"
                  f"  (offset range {z['worst_offset']*100:+.2f}% .. {z['best_offset']*100:+.2f}%)")

        cal = None
        if rows[0]["proba"] is not None:
            print("3) Calibration of the 7-day score (all coins, held-out days)")
            pa = np.concatenate([r["proba_all"] for r in rows]); ya = np.concatenate([r["top_all"] for r in rows])
            pp = np.concatenate([r["proba"] for r in rows]); yp = np.concatenate([r["hit_vec"] for r in rows])
            bins = [0, .02, .05, .08, .12, .2, .3, 1.0]
            cal = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = (pa >= lo) & (pa < hi)
                if s.sum() >= 50:
                    cal.append(dict(lo=lo, hi=hi, n=int(s.sum()), predicted=float(pa[s].mean()), actual=float(ya[s].mean())))
                    print(f"   score {lo*100:4.0f}-{hi*100:3.0f}%: n={s.sum():6d} predicted {pa[s].mean():6.1%} actual {ya[s].mean():6.1%}")
            print(f"   the 10 picks: predicted {pp.mean():.1%} on average, actual hit rate {yp.mean():.1%}")
            cal = dict(bins=cal, picks_predicted=float(pp.mean()), picks_actual=float(yp.mean()))

        print("4) Liquidity and costs")
        vp = np.concatenate([r["vol_picks"] for r in rows]); va = np.concatenate([r["vol_all"] for r in rows])
        vp, va = vp[np.isfinite(vp)], va[np.isfinite(va)]
        print(f"   picks' 24h volume: median ${np.median(vp)/1e6:,.1f}M (all coins ${np.median(va)/1e6:,.1f}M);"
              f" under $1M: {np.mean(vp < 1e6):.1%}, under $5M: {np.mean(vp < 5e6):.1%}")
        print(f"   only coins with $1M+ volume (swap thin picks for the next): {f('liquid')*100:+.2f}% per call vs {f('pick')*100:+.2f}%")
        costs = {}
        for c in (0.003, 0.01, 0.02):
            costs[c] = f("pick_raw") - c
            print(f"   at {c:.1%} round-trip cost: {costs[c]*100:+.2f}% per call (top-200 avg {f('market')*100 + (COST - c)*100:+.2f}%)")
        liq = independent([dict(r, liquid=r["liquid"]) for r in rows if np.isfinite(r["liquid"])], h, "liquid", "matched")
        print(f"   liquid picks vs matched, independent windows: {liq['mean']*100:+.2f}% (90% range {liq['lo']*100:+.2f}% .. {liq['hi']*100:+.2f}%)")
        # execution delay: calls are published hours after the daily snapshot; buy a full day late instead
        late = []
        for r in rows:
            t = r["t"] + 1
            if t + h < P.shape[0]:
                d = next(x for x in days if x["date"] == r["date"])
                pick = np.argsort(-sc[h][0](d))[:10]; rr = P[t + h, d["idx"]] / P[t, d["idx"]] - 1; ok = np.isfinite(rr)
                late.append(dict(pick=float(np.nanmean(rr[pick]) - COST), market=float(np.nanmean(rr[ok]) - COST)))
        delay = independent(late, h, "pick", "market")
        print(f"   bought 1 day late: {np.mean([x['pick'] for x in late])*100:+.2f}% per call;"
              f" vs top-200 (independent) {delay['mean']*100:+.2f}% (90% range {delay['lo']*100:+.2f}% .. {delay['hi']*100:+.2f}%)")

        report[h] = dict(calls=len(rows), period=[rows[0]["date"], rows[-1]["date"]], cost=COST,
                         avg=dict(picks=f("pick"), matched=f("matched"), market=f("market"), btc=f("btc")),
                         up=dict(picks=f("pick_up"), matched=f("matched_up"), market=f("market_up")),
                         hit=dict(picks=f("pick_hit"), matched=f("matched_hit")),
                         independent=out, calibration=cal,
                         liquidity=dict(median_volume_picks=float(np.median(vp)), median_volume_all=float(np.median(va)),
                                        under_1m=float(np.mean(vp < 1e6)), under_5m=float(np.mean(vp < 5e6)),
                                        liquid_only=f("liquid"), liquid_vs_matched=liq),
                         delay_1day=dict(avg=float(np.mean([x["pick"] for x in late])), vs_market=delay),
                         cost_sensitivity={str(k): v for k, v in costs.items()})
    (HERE / "evidence_results.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
