"""What would following the calls have earned? Trader metrics, exit rules and a sit-out filter.

For every day in the held-out year (Oct 2025 - Sep 2026) take the day's 10 picks:
  7-day: model v2 (M3), fit on training days only (same C as production)
  3-day: tuned 5-signal formula (weights tuned on training days only, backtest/results.json)
and buy them in equal weight at that day's close, hold H days (or until an exit rule fires).

Reported per horizon:
  basket return (mean / median), basket win rate, beat-BTC rate, coin win rate,
  avg winner / avg loser, expectancy per coin, and an equity curve that trades every H days
  (non-overlapping) with its max drawdown, next to buying and holding BTC.
Costs: 0.3% round trip per coin (fees + slippage).

Exit rules (chosen on training days, then scored on held-out days):
  stop-loss S%: sell at the first daily close at or below -S%
  take-profit T%: sell at the first daily close at or above +T%
Sit-out filters (same procedure): skip the day when the market looks bad.

Usage: python backtest/trader_backtest.py --dir <folder with daily.csv, coins.csv, global.csv, fng.csv>
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
STOPS = [None, 5, 10, 15, 20, 30]
TAKES = [None, 10, 20, 30, 50, 100]


def picks_by_day(days, h):
    tr, te = ex.split(days, h)
    if h == 7:
        C = json.loads((HERE / "model_v2_7d.json").read_text())["C"]
        Xt, yt = ex.stack(tr, h, True)
        m = LogisticRegression(C=C, max_iter=3000).fit(Xt, yt)
        scorer = lambda d: m.decision_function(ex.design(d, True))
    else:
        w = np.array([json.loads((HERE / "results.json").read_text())["horizons"][str(h)]["weights"][k] for k in bw.FEATURES]) / 100
        scorer = lambda d: d["F5"] @ w
    return {d["date"]: d["idx"][np.argsort(-scorer(d))[:10]] for d in tr + te}, tr, te


def coin_paths(P, t, cols, h):
    """[h x n] returns vs day-t close for days t+1..t+h."""
    return P[t + 1:t + h + 1, cols] / P[t, cols] - 1


def exit_returns(paths, stop, take):
    """Per-coin return with stop / take-profit on daily closes; NaN paths are dropped."""
    out = []
    for j in range(paths.shape[1]):
        p = paths[:, j]
        if not np.isfinite(p).all():
            continue
        r = p[-1]
        for x in p:
            if stop is not None and x <= -stop / 100:
                r = x; break
            if take is not None and x >= take / 100:
                r = x; break
        out.append(r - COST)
    return np.array(out)


def evaluate(day_list, picks, P, di, btc, h, stop=None, take=None, skip=None):
    baskets, btcs, coins, dates = [], [], [], []
    for d in day_list:
        if skip and skip(d):
            continue
        t = di[d["date"]]
        r = exit_returns(coin_paths(P, t, picks[d["date"]], h), stop, take)
        if not len(r):
            continue
        baskets.append(r.mean()); coins.extend(r); dates.append(d["date"])
        btcs.append(P[t + h, btc] / P[t, btc] - 1)
    b, c, bt = np.array(baskets), np.array(coins), np.array(btcs)
    # equity: trade every h-th calendar day only (non-overlapping), compounding; skipped days stay in cash.
    # Every start offset is run; the median offset is reported, the worst one too.
    dd = lambda e: float((e / np.maximum.accumulate(e) - 1).min())
    pos = {dd_: k for k, dd_ in enumerate(dates)}
    all_days = [d["date"] for d in day_list]
    runs = []
    for off in range(h):
        eq, eqb = [1.0], [1.0]
        for k in range(off, len(all_days), h):
            j = pos.get(all_days[k])
            eq.append(eq[-1] * (1 + (b[j] if j is not None else 0.0)))
            eqb.append(eqb[-1] * (1 + (bt[j] if j is not None else 0.0)))
        runs.append((np.array(eq), np.array(eqb)))
    runs.sort(key=lambda r: r[0][-1])
    eq, eqb = runs[len(runs) // 2]
    wins, losses = c[c > 0], c[c <= 0]
    return dict(days=len(b), traded_share=len(b) / max(len(day_list), 1),
                basket_mean=float(b.mean()), basket_median=float(np.median(b)), basket_win=float((b > 0).mean()),
                beat_btc=float((b > bt).mean()), btc_mean=float(bt.mean()),
                coin_win=float((c > 0).mean()), avg_win=float(wins.mean()) if len(wins) else 0.0,
                avg_loss=float(losses.mean()) if len(losses) else 0.0, expectancy=float(c.mean()),
                equity=float(eq[-1] - 1), equity_dd=dd(eq), equity_worst=float(runs[0][0][-1] - 1),
                btc_equity=float(eqb[-1] - 1), btc_dd=dd(eqb))


def show(name, r):
    print(f"  {name:<34} days {r['days']:3d} | basket {r['basket_mean']*100:+5.2f}% (med {r['basket_median']*100:+5.2f}%)"
          f" win {r['basket_win']:.0%} beatBTC {r['beat_btc']:.0%} | coin win {r['coin_win']:.0%}"
          f" avg +{r['avg_win']*100:.1f}/{r['avg_loss']*100:.1f}% | equity {r['equity']*100:+6.1f}% (worst start {r['equity_worst']*100:+.0f}%) dd {r['equity_dd']*100:.0f}%"
          f" (BTC {r['btc_equity']*100:+.1f}%, dd {r['btc_dd']*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    dates, ids, sym, P, V, M = bw.load_matrices(argparse.Namespace(csv=str(Path(a.dir) / "daily.csv"), xlsx=None))
    di = {d: i for i, d in enumerate(dates)}; btc = ids.index("1")
    days = ex.build(a.dir)
    report = {}
    for h in (7, 3):
        picks, tr, te = picks_by_day(days, h)
        print(f"\n===== {h}-day calls | train {len(tr)} days | held-out {len(te)} days ({te[0]['date']} -> {te[-1]['date']}) =====")
        base = evaluate(te, picks, P, di, btc, h); show("hold H days (as published)", base)
        # random 10 coins from the same universe, for reference
        rng = np.random.default_rng(0)
        rnd = {d["date"]: rng.choice(d["idx"], 10, replace=False) for d in te}
        show("random 10 coins", evaluate(te, rnd, P, di, btc, h))

        # exit rules: pick on train by expectancy, then score held-out
        grid = [(s, k) for s in STOPS for k in TAKES]
        tr_scores = {g: evaluate(tr, picks, P, di, btc, h, *g)["expectancy"] for g in grid}
        best_exit = max(grid, key=tr_scores.get)
        ex_res = evaluate(te, picks, P, di, btc, h, *best_exit)
        show(f"exit rule stop {best_exit[0]} / take {best_exit[1]} (train-picked)", ex_res)
        for g in [(10, None), (None, 20), (10, 30)]:
            show(f"  for reference: stop {g[0]} / take {g[1]}", evaluate(te, picks, P, di, btc, h, *g))

        # sit-out filters on market regime: d['reg'] = [fng/100, fng chg7, dom chg7, btc7 %/100, total mcap 7d]
        filters = {
            "BTC 7d down": lambda d: d["reg"][3] < 0,
            "total market cap 7d down": lambda d: d["reg"][4] < 0,
            "extreme greed (F&G > 75)": lambda d: d["reg"][0] > .75,
            "extreme fear (F&G < 25)": lambda d: d["reg"][0] < .25,
            "BTC dominance rising 7d": lambda d: d["reg"][2] > 0,
        }
        tr_base = evaluate(tr, picks, P, di, btc, h)
        print("  sit-out filters (train expectancy -> held-out):")
        for name, f in filters.items():
            r_tr = evaluate(tr, picks, P, di, btc, h, skip=f); r_te = evaluate(te, picks, P, di, btc, h, skip=f)
            print(f"    skip when {name:<28} train {tr_base['expectancy']*100:+.2f}% -> {r_tr['expectancy']*100:+.2f}%"
                  f" | held-out {base['expectancy']*100:+.2f}% -> {r_te['expectancy']*100:+.2f}% (trades {r_te['traded_share']:.0%} of days)"
                  f" | equity {r_te['equity']*100:+.0f}% (worst start {r_te['equity_worst']*100:+.0f}%) dd {r_te['equity_dd']*100:.0f}%")
        # monthly consistency of the published calls
        months = {}
        for d in te:
            t = di[d["date"]]; r = exit_returns(coin_paths(P, t, picks[d["date"]], h), None, None)
            if len(r):
                months.setdefault(d["date"][:7], []).append(r.mean() - (P[t + h, btc] / P[t, btc] - 1))
        beat = [np.mean(v) > 0 for v in months.values()]
        print(f"  months where the calls beat BTC on average: {sum(beat)}/{len(beat)}")
        # market light: Trade when BTC's 7-day change >= 0, Caution when it is negative
        trade = evaluate(te, picks, P, di, btc, h, skip=lambda d: d["reg"][3] < 0)
        caution = evaluate(te, picks, P, di, btc, h, skip=lambda d: d["reg"][3] >= 0)
        show("market light TRADE days", trade); show("market light CAUTION days", caution)
        report[h] = dict(hold=base, exit_rule=dict(stop=best_exit[0], take=best_exit[1], **ex_res),
                         months_beat_btc=[int(sum(beat)), len(beat)],
                         light=dict(trade=trade, caution=caution),
                         period=[te[0]["date"], te[-1]["date"]], cost=COST)
    (HERE / "trader_results.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
