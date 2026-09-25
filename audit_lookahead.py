"""Look-ahead audit.

Test 1 - point-in-time recomputation: for every signal date d in the backtest, cut ALL data
         (prices and VolVue skew) at d, recompute features and picks from scratch, and check
         they are identical to the picks the backtest made. If any feature peeked past d, the
         truncated run would disagree.
Test 2 - the same for the SPY 120-day betas used by the beta-matched hedge.
Test 3 - extra-lag robustness: form baskets from data one day OLDER (signal at d-1, same entry
         at d+1). Results should barely move; a big drop would suggest the edge depends on
         same-day timing.
Test 4 - planted leak (sanity check of the method): a version that peeks 15 days ahead
         must fail Test 1 and produce absurd returns, proving the test can catch a leak.
"""
import numpy as np
import pandas as pd

import backtest as bt
import variants as v
from universe import SECTORS


def picks_equal(a, b):
    return {s: {g: t for g, (t, _) in gp.items()} for s, gp in a.items()} == \
           {s: {g: t for g, (t, _) in gp.items()} for s, gp in b.items()}


def main():
    px, skew = bt.load()
    f = bt.features(px, skew)
    dates = px.index[px.index >= bt.START]
    sig_dates = [dates[i] for i in range(0, len(dates) - 1, bt.REBAL)]

    # Test 1
    mism = 0
    for d in sig_dates:
        pt = bt.features(px.loc[:d], skew.loc[:d])
        if not picks_equal(bt.pick(f, d), bt.pick(pt, d)):
            mism += 1
    print(f"Test 1 point-in-time picks: {len(sig_dates)} signal dates, {mism} mismatches")

    # Test 2
    lr = np.log(px).diff()
    beta_full = lr.rolling(120).cov(lr["SPY"]).div(lr["SPY"].rolling(120).var(), axis=0)
    worst = 0.0
    for d in sig_dates[::5]:
        l2 = np.log(px.loc[:d]).diff()
        b2 = l2.rolling(120).cov(l2["SPY"]).div(l2["SPY"].rolling(120).var(), axis=0).loc[d]
        worst = max(worst, float((beta_full.loc[d] - b2).abs().max(skipna=True)))
    print(f"Test 2 point-in-time betas: max abs difference {worst:.2e}")

    # Test 3 and Test 4: 18 longs + short SPY 0.75x and + sector ETF, 3-week hold
    def run(signal_shift):
        out = []
        for i in range(0, len(dates) - 1 - bt.REBAL, bt.REBAL):
            if i + signal_shift < 0:
                continue
            ds, e, x = dates[i + signal_shift], dates[i + 1], dates[i + 1 + bt.REBAL]
            spy = px.at[x, "SPY"] / px.at[e, "SPY"] - 1
            sec_r, spy_r, n = 0.0, 0.0, 0
            for sec, gp in bt.pick(f, ds).items():
                for g in ("G1", "G3"):
                    if g in gp:
                        t = gp[g][0]
                        r = px.at[x, t] / px.at[e, t] - 1
                        rs = px.at[x, sec] / px.at[e, sec] - 1
                        rs = spy if not np.isfinite(rs) else rs
                        sec_r += r - rs - 0.0024
                        spy_r += r - 0.75 * spy - 0.0023
                        n += 1
            out.append({"exit": x, "sector": sec_r / 18, "spy75": spy_r / 18, "SPY": spy})
        return pd.DataFrame(out).set_index("exit")

    rows = {}
    for name, sh in (("as backtested (signal d, entry d+1)", 0), ("one day older signal (d-1)", -1),
                     ("PLANTED LEAK: signal from d+15", 15)):
        p = run(sh)
        for k in ("sector", "spy75"):
            r = p[k]
            ppy = 252 / bt.REBAL
            eq = (1 + r).cumprod()
            rows[(name, k)] = {"CAGR": eq.iloc[-1] ** (ppy / len(r)) - 1,
                               "Sharpe": r.mean() * ppy / (r.std() * np.sqrt(ppy)),
                               "Max DD": (eq / eq.cummax() - 1).min()}
    res = pd.DataFrame(rows).T
    print("\nTest 3/4 (Sharpe here is not in excess of T-bills):")
    print(res.round(3))

    # Test 4 part 2: the planted leak must fail the point-in-time check
    leak_mism = 0
    for i, d in enumerate(sig_dates[:-2]):
        future = dates[min(dates.get_loc(d) + 15, len(dates) - 1)]
        leaked = bt.pick(f, future)          # a pick made with data 15 days in the future
        honest = bt.pick(bt.features(px.loc[:d], skew.loc[:d]), d)
        leak_mism += not picks_equal(leaked, honest)
    print(f"\nTest 4 planted leak vs point-in-time: {leak_mism}/{len(sig_dates) - 2} mismatches (should be most)")
    res.to_csv("output/audit_lookahead.csv")


if __name__ == "__main__":
    main()
