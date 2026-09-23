"""IV-skew x price-action sector basket backtest.

Every REBAL trading days, for each sector ETF's top-20 stocks, classify on the
signal day's close and pick the single most extreme name per group:

  G1 LONG  momentum    : skew90 <= LOW_SKEW,  20d return > 0, close > SMA50  -> max 20d return
  G2 SHORT momentum    : skew90 <= LOW_SKEW,  20d return < 0, close < SMA50  -> min 20d return
  G3 LONG  mean-revert : skew90 >= HIGH_SKEW, already down >= PULLBACK from
                         60d high (pullback advanced, fear priced in)       -> deepest drawdown
  G4 SHORT mean-revert : skew90 >= HIGH_SKEW, 20d return > 0 and within
                         NEAR_HIGH of 60d high (hedgers bid puts at the top)  -> highest skew

Positions are entered at the NEXT day's close (no look-ahead), held REBAL
trading days, equal-weighted within each group, then a new basket is formed.
"""
import json
import os

import numpy as np
import pandas as pd

from universe import SECTORS, TICKER_SECTOR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")

LOW_SKEW, HIGH_SKEW = 4.0, 5.0      # VolVue iv_skew_90 is in vol points: 4.0 == 0.04
MOM_LB, SMA_LB, HIGH_LB = 20, 50, 60
PULLBACK, NEAR_HIGH = -0.08, -0.03
REBAL = 15                           # 3 weeks
COST = 0.0010                        # 10 bp per side
START = "2018-06-01"

GROUPS = {
    "G1": ("Long momentum (low skew, uptrend)", +1),
    "G2": ("Short momentum (low skew, downtrend)", -1),
    "G3": ("Long mean-revert (high skew, pullback advanced)", +1),
    "G4": ("Short mean-revert (high skew, at highs)", -1),
}


def load():
    px = pd.read_csv(os.path.join(DATA, "prices.csv"), index_col=0, parse_dates=True).sort_index()
    px = px[px.notna().mean(axis=1) > 0.5]
    px = px.ffill()  # delisted / take-private names keep their last price (cash-out)
    v = pd.read_csv(os.path.join(DATA, "volvue_skew90.csv"), parse_dates=["date"])
    v.loc[(v.iv_skew_90 < -10) | (v.iv_skew_90 > 40), "iv_skew_90"] = np.nan   # bad prints
    skew = v.pivot_table(index="date", columns="ticker", values="iv_skew_90")
    skew = skew.reindex(px.index).ffill(limit=3)
    return px, skew


def features(px, skew):
    tick = [t for t in TICKER_SECTOR if t in px.columns]
    p = px[tick]
    return {
        "skew": skew.reindex(columns=tick),
        "mom": p / p.shift(MOM_LB) - 1,
        "above_sma": p > p.rolling(SMA_LB).mean(),
        "dd": p / p.rolling(HIGH_LB).max() - 1,
        "px": p,
    }


def pick(f, d):
    """Return {sector: {group: (ticker, stats)}} for signal date d."""
    out = {}
    for sec, names in SECTORS.items():
        names = [t for t in names if t in f["px"].columns]
        row = pd.DataFrame({k: f[k].loc[d, names] for k in ("skew", "mom", "above_sma", "dd")}).dropna()
        row["above_sma"] = row["above_sma"].astype(bool)
        low, high = row[row["skew"] <= LOW_SKEW], row[row["skew"] >= HIGH_SKEW]
        cands = {
            "G1": low[(low.mom > 0) & low.above_sma].sort_values("mom", ascending=False),
            "G2": low[(low.mom < 0) & ~low.above_sma].sort_values("mom"),
            "G3": high[high.dd <= PULLBACK].sort_values("dd"),
            "G4": high[(high.mom > 0) & (high.dd >= NEAR_HIGH)].sort_values("skew", ascending=False),
        }
        out[sec] = {g: (c.index[0], c.iloc[0].to_dict()) for g, c in cands.items() if len(c)}
    return out


def run():
    px, skew = load()
    f = features(px, skew)
    dates = px.index[px.index >= START]
    sig_idx = list(range(0, len(dates) - 1, REBAL))
    rows, periods = [], []
    for i in sig_idx:
        d = dates[i]
        entry = dates[i + 1]
        exit_ = dates[min(i + 1 + REBAL, len(dates) - 1)]
        if exit_ <= entry:
            break
        picks = pick(f, d)
        per = {"signal": d, "entry": entry, "exit": exit_}
        for g, (_, side) in GROUPS.items():
            rets, rel = [], []
            for sec, gp in picks.items():
                if g not in gp:
                    continue
                t, st = gp[g]
                r = px.at[exit_, t] / px.at[entry, t] - 1
                rs = px.at[exit_, sec] / px.at[entry, sec] - 1
                pnl = side * r - 2 * COST
                rets.append(pnl)
                rel.append(side * (r - rs))
                rows.append({"signal": d.date(), "entry": entry.date(), "exit": exit_.date(), "group": g,
                             "sector": sec, "ticker": t, "skew90": round(st["skew"], 2),
                             "mom20": round(st["mom"], 4), "dd60": round(st["dd"], 4),
                             "stock_ret": round(r, 4), "pnl": round(pnl, 4), "vs_sector": round(side * (r - rs), 4)})
            per[g] = np.mean(rets) if rets else 0.0
            per[g + "_rel"] = np.mean(rel) if rel else 0.0
            per[g + "_n"] = len(rets)
        per["SPY"] = px.at[exit_, "SPY"] / px.at[entry, "SPY"] - 1
        uni = [t for t in TICKER_SECTOR if t in px.columns and pd.notna(px.at[entry, t])]
        per["EW_universe"] = float(np.nanmean(px.loc[exit_, uni] / px.loc[entry, uni] - 1))
        periods.append(per)
    per = pd.DataFrame(periods).set_index("exit")
    per["Momentum L/S"] = 0.5 * (per.G1 + per.G2)
    per["MeanRev L/S"] = 0.5 * (per.G3 + per.G4)
    per["All 4 combined"] = 0.25 * (per.G1 + per.G2 + per.G3 + per.G4)
    per["Longs only (G1+G3)"] = 0.5 * (per.G1 + per.G3)
    trades = pd.DataFrame(rows)
    current = pick(f, px.index[-1])
    return per, trades, current, px.index[-1]


def stats(r, ppy=252 / REBAL):
    eq = (1 + r).cumprod()
    yrs = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ppy)
    dd = (eq / eq.cummax() - 1).min()
    return {"Total return": eq.iloc[-1] - 1, "CAGR": cagr, "Ann. vol": vol,
            "Sharpe": (r.mean() * ppy) / vol if vol else np.nan, "Max drawdown": dd,
            "Hit rate": (r > 0).mean(), "Avg / basket": r.mean(), "Baskets": len(r)}


STRATS = ["G1", "G2", "G3", "G4", "Momentum L/S", "MeanRev L/S", "All 4 combined",
          "Longs only (G1+G3)", "SPY", "EW_universe"]


def main():
    os.makedirs(OUT, exist_ok=True)
    per, trades, current, asof = run()
    st = pd.DataFrame({s: stats(per[s]) for s in STRATS}).T
    rel = pd.DataFrame({g: stats(per[g + "_rel"]) for g in GROUPS}).T
    per.to_csv(os.path.join(OUT, "basket_returns.csv"))
    trades.to_csv(os.path.join(OUT, "all_baskets_trades.csv"), index=False)
    st.to_csv(os.path.join(OUT, "summary_stats.csv"))
    rel.to_csv(os.path.join(OUT, "vs_sector_stats.csv"))
    # yearly
    yearly = per[STRATS].groupby(per.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly.to_csv(os.path.join(OUT, "yearly_returns.csv"))
    cur = [{"sector": s, "group": g, "ticker": t, **{k: round(float(v), 4) for k, v in stv.items()}}
           for s, gp in current.items() for g, (t, stv) in gp.items()]
    pd.DataFrame(cur).to_csv(os.path.join(OUT, "current_basket.csv"), index=False)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump({
            "asof": str(asof.date()),
            "params": {"LOW_SKEW": LOW_SKEW, "HIGH_SKEW": HIGH_SKEW, "MOM_LB": MOM_LB, "SMA_LB": SMA_LB,
                       "HIGH_LB": HIGH_LB, "PULLBACK": PULLBACK, "NEAR_HIGH": NEAR_HIGH,
                       "REBAL": REBAL, "COST_bp_side": COST * 1e4, "START": START},
            "groups": {g: v[0] for g, v in GROUPS.items()},
            "stats": st.reset_index().rename(columns={"index": "strategy"}).to_dict("records"),
            "vs_sector": rel.reset_index().rename(columns={"index": "group"}).to_dict("records"),
            "yearly": yearly.reset_index().rename(columns={"exit": "year"}).to_dict("records"),
            "equity": {"dates": [str(d.date()) for d in per.index],
                       **{s: list(np.round((1 + per[s]).cumprod().values, 4)) for s in STRATS}},
            "current": cur,
            "last_baskets": trades[trades.signal >= trades.signal.unique()[-3]].to_dict("records"),
        }, fh, default=str)
    pd.set_option("display.width", 200)
    print(st.round(3)); print("\nvs sector ETF:\n", rel.round(3)); print("\n", yearly.round(3))
    print("\nCurrent basket as of", asof.date()); print(pd.DataFrame(cur))


if __name__ == "__main__":
    main()
