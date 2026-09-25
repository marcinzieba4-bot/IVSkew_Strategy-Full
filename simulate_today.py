"""Simulate today's basket: positions, ex-ante risk, stress replays and a live mark.

Signal = last date with both a complete close and VolVue skew. Entry = next close.
Two implementations of the 18 long slots (1/18 of CAPITAL each, empty slots in cash):
  A. 18 longs + short own sector ETF (equal $ per slot)
  B. 18 longs + short SPY at 0.75x the total long $
Risk uses current weights applied to history: daily returns of the last 252 days for vol/beta,
and every overlapping 15-day window since 2018 for the 3-week loss distribution.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import requests

import backtest as bt
import variants as v

CAPITAL = 100_000
H = 15  # 3-week hold in trading days


def live_price(t):
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", params={"range": "1d", "interval": "1d"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:  # noqa: BLE001
        return np.nan


def main():
    px, skew = bt.load()
    f = bt.features(px, skew)
    raw = pd.read_csv(os.path.join(bt.DATA, "volvue_skew90.csv"), parse_dates=["date"])
    d = min(px.index[-1], raw.date.max())
    picks = bt.pick(f, d)
    ivs, rf = v.load_iv()
    iv20 = ivs["iv_call_20"].reindex(px.index).ffill(limit=3)
    r = float(rf.reindex(px.index).ffill().iloc[-1])
    slot = CAPITAL / 18

    longs, others = [], []
    for sec, gp in picks.items():
        for g in ("G3", "G1"):
            if g in gp:
                t, st = gp[g]
                p = px.at[d, t]
                vol = iv20.at[d, t] / 100 if pd.notna(iv20.at[d, t]) else np.nan
                prem = v.bs_call(p, p, 21 / 365, vol, r) / p if np.isfinite(vol) else np.nan
                longs.append({"group": g, "sector": sec, "ticker": t, "price": p, "skew90": st["skew"],
                              "mom20": st["mom"], "dd60": st["dd"], "usd": slot, "shares": math.floor(slot / p),
                              "hedge_etf": sec, "hedge_etf_shares": -math.floor(slot / px.at[d, sec]),
                              "call_iv20": vol, "atm_call_prem_pct": prem})
        for g in ("G2", "G4"):
            if g in gp:
                others.append({"group": g, "sector": sec, "ticker": gp[g][0], "skew90": gp[g][1]["skew"]})
    L = pd.DataFrame(longs)
    n = len(L)
    long_usd = n * slot
    spy_short_usd = 0.75 * long_usd

    # weights per $ of capital
    wA = pd.Series(0.0, index=px.columns)
    wB = pd.Series(0.0, index=px.columns)
    for _, row in L.iterrows():
        wA[row.ticker] += slot / CAPITAL
        wA[row.hedge_etf] -= slot / CAPITAL
        wB[row.ticker] += slot / CAPITAL
    wB["SPY"] -= spy_short_usd / CAPITAL
    rets = px.pct_change()
    books = {"A: longs + sector ETF": wA, "B: longs + SPY 0.75x": wB,
             "Longs only (no hedge)": wA.clip(lower=0)}

    last = rets.iloc[-252:].fillna(0)
    win = (px.shift(-H) / px - 1).iloc[:-H]  # forward 15-day returns from each day (history replay)
    win = win[win.index >= "2018-06-01"]
    risk = {}
    for name, w in books.items():
        pr = last @ w
        spy = last["SPY"]
        beta = np.cov(pr, spy)[0, 1] / spy.var()
        h = win.fillna(0) @ w
        risk[name] = {"gross_usd": float(w.abs().sum() * CAPITAL), "net_usd": float(w.sum() * CAPITAL),
                      "vol_3wk_pct": float(pr.std() * math.sqrt(H) * 100), "vol_ann_pct": float(pr.std() * math.sqrt(252) * 100),
                      "beta": float(beta), "var95_3wk_usd": float(np.percentile(h, 5) * CAPITAL),
                      "cvar95_3wk_usd": float(h[h <= np.percentile(h, 5)].mean() * CAPITAL),
                      "worst_3wk_usd": float(h.min() * CAPITAL), "worst_3wk_date": str(h.idxmin().date()),
                      "p_loss_pct": float((h < 0).mean() * 100)}
    # stress: current weights through the worst and best SPY 3-week windows
    spyw = win["SPY"].dropna()
    picks_dates = []
    for dt in spyw.sort_values().index:
        if all(abs((dt - p).days) > 60 for p in picks_dates):
            picks_dates.append(dt)
        if len(picks_dates) == 4:
            break
    picks_dates.append(spyw.idxmax())
    stress = [{"start": str(dt.date()), "SPY_pct": float(spyw[dt] * 100),
               **{name: float((win.loc[dt].fillna(0) @ w) * CAPITAL) for name, w in books.items()}}
              for dt in picks_dates]

    # expected from the backtest (3-week hold)
    p = pd.read_csv(os.path.join(bt.OUT, "variants_returns_15d.csv"), index_col=0)
    exp = {"A: longs + sector ETF": p["L18 sec"], "B: longs + SPY 0.75x": p["L18 spy75"], "Longs only (no hedge)": p["L18 stock"]}
    expected = {k: {"avg_3wk_pct": float(s.mean() * 100), "median_3wk_pct": float(s.median() * 100),
                    "hit_rate_pct": float((s > 0).mean() * 100), "avg_3wk_usd": float(s.mean() * CAPITAL)} for k, s in exp.items()}

    # live mark since the signal close
    live = {t: live_price(t) for t in list(L.ticker) + list(L.hedge_etf.unique()) + ["SPY"]}
    L["live"] = L.ticker.map(live)
    L["chg_since_signal_pct"] = (L["live"] / L["price"] - 1) * 100
    etf_chg = {e: (live[e] / px.at[d, e] - 1) for e in L.hedge_etf.unique()}
    spy_chg = live["SPY"] / px.at[d, "SPY"] - 1
    L["vs_sector_pct"] = L["chg_since_signal_pct"] - L.hedge_etf.map(etf_chg) * 100
    mark = {"A: longs + sector ETF": float((L["chg_since_signal_pct"] / 100 - L.hedge_etf.map(etf_chg)).sum() * slot),
            "B: longs + SPY 0.75x": float((L["chg_since_signal_pct"] / 100).sum() * slot - spy_short_usd * spy_chg),
            "SPY_pct": float(spy_chg * 100)}

    out = {"signal_date": str(d.date()), "capital": CAPITAL, "slot_usd": slot, "n_longs": n,
           "long_usd": long_usd, "spy_short_usd": spy_short_usd, "spy_short_shares": -math.floor(spy_short_usd / px.at[d, "SPY"]),
           "spy_price": float(px.at[d, "SPY"]), "longs": L.replace({np.nan: None}).to_dict("records"),
           "reference_shorts": others, "risk": risk, "stress": stress, "expected": expected, "live_mark": mark}
    json.dump(out, open(os.path.join(bt.OUT, "today_simulation.json"), "w"), default=float, indent=1)
    L.to_csv(os.path.join(bt.OUT, "today_positions.csv"), index=False)
    pd.set_option("display.width", 250)
    print("signal", d.date(), "| longs", n, "| long $", round(long_usd), "| SPY short $", round(spy_short_usd))
    print(L[["group", "sector", "ticker", "price", "skew90", "mom20", "dd60", "shares", "hedge_etf", "hedge_etf_shares",
             "call_iv20", "atm_call_prem_pct", "live", "chg_since_signal_pct", "vs_sector_pct"]].round(3).to_string(index=False))
    print(pd.DataFrame(risk).round(2)); print(pd.DataFrame(stress).round(0)); print(pd.DataFrame(expected).round(2)); print(mark)
    print("reference shorts:", others)


if __name__ == "__main__":
    main()
