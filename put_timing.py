"""SPY put timing: hold ATM SPY puts only in baskets where a rule, known on the signal day, is on.

Rules (all use trailing data only; percentiles are over the prior 252 trading days):
  always          : puts every basket (the previous test)
  iv_pct<25/<50   : SPY ATM put IV is in the bottom 25% / 50% of its past year (puts cheap)
  vrp<p25         : SPY IV minus 20d realised vol is in the bottom 25% of its past year
  skew_pct<25     : SPY 30d skew is in the bottom 25% of its past year (puts cheap vs calls)
  spy<sma50       : SPY below its 50-day average (downtrend already running)
  spy<sma200      : SPY below its 200-day average
  calm_top        : SPY within 2% of its 1-year high AND put IV in the bottom 25% (complacency)
  iv<25|spy<sma50 : either cheap IV or downtrend

Each rule is applied to three books: 18 ATM calls + SPY puts 50%, the same with 100%, and the
best stock book (18 longs + short sector ETF) with a 50% SPY put overlay. When the rule is off,
no puts are held.
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
import variants as v

HALF = pd.Timestamp("2022-07-01")


def trailing_pct(s, n=252):
    return s.rolling(n, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)


def signals(px, hold):
    spy = px["SPY"]
    d = pd.read_csv(os.path.join(bt.DATA, "volvue_spy.csv"), parse_dates=["date"]).set_index("date")
    d = d.reindex(px.index).ffill(limit=3)
    put_iv = d["iv_put_20" if hold == "3 weeks" else "iv_put_30"]
    rv = np.log(spy).diff().rolling(20).std() * np.sqrt(252) * 100
    vrp = put_iv - rv
    iv_pct, vrp_pct, skew_pct = trailing_pct(put_iv), trailing_pct(vrp), trailing_pct(d["iv_skew_30"])
    sma50, sma200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    near_high = spy / spy.rolling(252).max() - 1 >= -0.02
    return pd.DataFrame({
        "always": True,
        "iv_pct<25": iv_pct < 25,
        "iv_pct<50": iv_pct < 50,
        "vrp<p25": vrp_pct < 25,
        "skew_pct<25": skew_pct < 25,
        "spy<sma50": spy < sma50,
        "spy<sma200": spy < sma200,
        "calm_top": near_high & (iv_pct < 25),
        "iv<25|spy<sma50": (iv_pct < 25) | (spy < sma50),
    })


def stats(r, spy, cash, ppy):
    m = v.metrics(r, spy, cash, ppy)
    yr = (1 + r).groupby(r.index.year).prod() - 1
    first, second = r[r.index < HALF], r[r.index >= HALF]
    sh = lambda x, c: (x - c).mean() * ppy / (x.std() * np.sqrt(ppy))  # noqa: E731
    return {"CAGR": m["CAGR"], "Ann. vol": m["Ann. vol"], "Sharpe": m["Sharpe"], "Max drawdown": m["Max drawdown"],
            "Worst period": m["Worst period"], "2020": yr.get(2020, np.nan), "2022": yr.get(2022, np.nan),
            "Sharpe 2018-22H1": sh(first, cash[first.index]), "Sharpe 2022H2-26": sh(second, cash[second.index])}


def main():
    px, skew = bt.load()
    f = bt.features(px, skew)
    ivs, rf = v.load_iv()
    rows, diag = [], []
    for hold, (rebal, _, _) in v.HOLDS.items():
        per, _ = v.run_hold(hold, px, f, ivs, rf)
        sig = signals(px, hold).reindex(per["signal"]).fillna(False).astype(bool)
        sig.index = per.index
        ppy = 252 / rebal
        prem = per["spy_put"]
        base = {"18 ATM calls, no puts": per["L18 atm"], "18 longs + sector ETF, no puts": per["L18 sec"]}
        for name, r in base.items():
            rows.append({"hold": hold, "rule": "none", "book": name, "puts on %": 0.0,
                         **stats(r, per["SPY"], per["cash"], ppy)})
        for rule in sig.columns:
            on = sig[rule]
            # SPY put P&L per $ notional in the baskets where the rule is on
            p_on, p_off = prem[on], prem[~on]
            diag.append({"hold": hold, "rule": rule, "baskets on": int(on.sum()), "puts on %": on.mean() * 100,
                         "put P&L % notional (on)": p_on.mean() * 100 if len(p_on) else np.nan,
                         "put P&L % notional (off)": p_off.mean() * 100 if len(p_off) else np.nan,
                         "put paid off % (on)": (p_on > 0).mean() * 100 if len(p_on) else np.nan,
                         "SPY ret % (on)": per["SPY"][on].mean() * 100 if on.any() else np.nan,
                         "SPY ret % (off)": per["SPY"][~on].mean() * 100 if (~on).any() else np.nan})
            books = {"18 ATM calls + SPY puts 50%": per["L18 atm"] + 0.5 * prem * on,
                     "18 ATM calls + SPY puts 100%": per["L18 atm"] + 1.0 * prem * on,
                     "18 longs + sector ETF + SPY puts 50%": per["L18 sec"] + 0.5 * prem * on}
            for name, r in books.items():
                rows.append({"hold": hold, "rule": rule, "book": name, "puts on %": on.mean() * 100,
                             **stats(r, per["SPY"], per["cash"], ppy)})
    res, dg = pd.DataFrame(rows), pd.DataFrame(diag)
    res.to_csv(os.path.join(bt.OUT, "put_timing_books.csv"), index=False)
    dg.to_csv(os.path.join(bt.OUT, "put_timing_diagnostics.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(dg.round(2).to_string(index=False))
    for hold in v.HOLDS:
        print(f"\n== {hold}")
        print(res[res.hold == hold].drop(columns="hold").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
