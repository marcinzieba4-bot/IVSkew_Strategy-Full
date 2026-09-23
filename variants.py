"""Implementation variants of the long side, run at 3-week and 1-month holding periods.

The stock picks are the same as backtest.py (one name per group per sector, 9 sectors).
The short single-name legs (G2/G4) lost money, so these variants keep only the long
signals and replace the short side with an index hedge, or express the longs with calls.

Books (empty slots sit in cash earning the T-bill rate):
  Long stocks            : 18 long slots (G1+G3), 1/18 of capital each
  + short sector ETF     : each long slot paired with an equal short of its own sector ETF
  + short SPY            : each long slot paired with an equal short of SPY
  Calls ATM / 30-delta   : notional-matched = calls on 1/18 of capital in notional;
                           delta-matched = calls on (1/18 of capital)/delta, so the starting
                           exposure equals the stock slot's. Per slot, expiring at the
                           end of the hold; the unspent cash earns T-bills. Priced with
                           Black-Scholes on VolVue's ATM call IV (iv_call_20 for 3 weeks,
                           iv_call_30 for 1 month) plus a 3% of premium entry spread.
                           The 30-delta strike uses the same ATM IV (no wing vol is available),
                           which slightly overprices OTM calls, so it is conservative.
  Calls vs SPY calls     : delta-matched long calls on each stock, plus short SPY calls
                           sized to the same dollar delta (delta-neutral at entry). SPY
                           calls use VolVue SPY ATM call IV (30-delta: ATM minus SPY_WING_DISCOUNT vol pts
                           for index skew) and a 1% spread on premium received.
                           Variants: ATM/ATM, 30d/30d, long ATM + short 30d SPY.
  ATM calls + short SPY  : ATM calls on N x slot notional (N = 1 or 2) plus short SPY shares
                           worth 0.5 x the call notional (ATM delta ~0.5, so about delta-neutral).
  ATM puts on shorts     : ATM puts on the 18 short slots (G2+G4), priced from VolVue ATM put IV,
                           notional- or delta-matched; the 36-slot book pairs them with ATM calls.
  Calls + SPY puts       : ATM calls on 1x slot notional plus long ATM SPY puts on 50% or 100%
                           of that notional (VolVue SPY ATM put IV, 1% spread).
Each is also run with G3 only (9 slots, 1/9 each), since G3 carries most of the edge.
For comparison: the stock 18L/18S book from backtest.py, and SPY.
"""
import json
import math
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

import backtest as bt
from universe import SECTORS

HOLDS = {"3 weeks": (15, "iv_call_20", 21), "1 month": (21, "iv_call_30", 30)}  # td, iv field, cal days
STOCK_COST, ETF_COST, OPT_SPREAD, SPY_OPT_SPREAD = 0.0010, 0.0002, 0.03, 0.01
# SPY OTM calls trade below ATM IV (index skew); vol points subtracted for the 30-delta SPY call
SPY_WING_DISCOUNT = 2.0


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, vol, r):
    if T <= 0 or vol <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d1 - vol * math.sqrt(T))


def strike_for_delta(S, T, vol, r, delta):
    """Strike whose Black-Scholes call delta N(d1) equals `delta`."""
    d1 = NormalDist().inv_cdf(delta)
    return S * math.exp(-d1 * vol * math.sqrt(T) + (r + 0.5 * vol * vol) * T)


def load_iv():
    v = pd.read_csv(os.path.join(bt.DATA, "volvue_call_iv.csv"), parse_dates=["date"])
    out = {}
    for f in ("iv_call_20", "iv_call_30", "iv_put_20", "iv_put_30"):
        v.loc[(v[f] <= 1) | (v[f] > 400), f] = np.nan
        out[f] = v.pivot_table(index="date", columns="ticker", values=f)
    rf = pd.read_csv(os.path.join(bt.DATA, "tbill.csv"), index_col=0, parse_dates=True)["IRX"] / 100
    return out, rf


def run_hold(label, px, f, ivs, rf):
    rebal, ivf, cal = HOLDS[label]
    iv = ivs[ivf].reindex(px.index).ffill(limit=3)
    ivp = ivs[ivf.replace("call", "put")].reindex(px.index).ffill(limit=3)
    rfx = rf.reindex(px.index).ffill().bfill()
    dates = px.index[px.index >= bt.START]
    rows, trades = [], []
    spy_put_trades = []
    lr = np.log(px).diff()
    betas = lr.rolling(120).cov(lr["SPY"]).div(lr["SPY"].rolling(120).var(), axis=0)
    n = len(SECTORS)
    for i in range(0, len(dates) - 1 - rebal, rebal):
        d, e, x = dates[i], dates[i + 1], dates[i + 1 + rebal]
        T = cal / 365
        r = float(rfx.loc[e])
        cash = (1 + r) ** ((x - e).days / 365) - 1
        spy = px.at[x, "SPY"] / px.at[e, "SPY"] - 1
        # short SPY calls: P&L per $ of dollar-delta sold, for ATM and 30-delta strikes
        spy_short = {}
        vspy = iv.at[e, "SPY"] / 100 if pd.notna(iv.at[e, "SPY"]) else np.nan
        if np.isfinite(vspy):
            P0, PT = px.at[e, "SPY"], px.at[x, "SPY"]
            vw = max(vspy - SPY_WING_DISCOUNT / 100, 0.03)
            for key, vk in (("atm", vspy), ("c30", vw)):
                K = P0 if key == "atm" else strike_for_delta(P0, T, vk, r, 0.30)
                prem = bs_call(P0, K, T, vk, r) * (1 - SPY_OPT_SPREAD)   # premium received
                dlt = ncdf((math.log(P0 / K) + (r + 0.5 * vk ** 2) * T) / (vk * math.sqrt(T)))
                spy_short[key] = (prem - max(PT - K, 0.0)) / P0 / dlt
        # long ATM SPY put: P&L per $ of notional
        spy_put = np.nan
        vsp = ivp.at[e, "SPY"] / 100 if pd.notna(ivp.at[e, "SPY"]) else np.nan
        if np.isfinite(vsp):
            P0, PT = px.at[e, "SPY"], px.at[x, "SPY"]
            pprem = (bs_call(P0, P0, T, vsp, r) - P0 + P0 * math.exp(-r * T)) * (1 + SPY_OPT_SPREAD)
            spy_put = (max(P0 - PT, 0.0) - pprem) / P0
            spy_put_trades.append({"exit": x, "premium": pprem / P0, "ret_on_premium": max(P0 - PT, 0.0) / pprem - 1})
        picks = bt.pick(f, d)
        slots = {k: [] for k in ("stock", "sec", "spy", "atm", "c30", "atm_dm", "c30_dm", "x_atm", "x_c30", "x_mix", "h1", "h2", "pp50", "pp100", "spy50", "spy75", "spyb")}
        g3 = {k: [] for k in slots}
        shorts = []
        puts = {k: [] for k in ("stock", "put", "put_dm")}
        puts_g = {g: [] for g in ("G2", "G4")}
        for sec, gp in picks.items():
            for g in ("G2", "G4"):
                if g in gp:
                    t = gp[g][0]
                    S0, ST = px.at[e, t], px.at[x, t]
                    stk = -(ST / S0 - 1) - 2 * STOCK_COST
                    shorts.append(stk)
                    vp = ivp.at[e, t] / 100 if t in ivp.columns and pd.notna(ivp.at[e, t]) else np.nan
                    if np.isfinite(vp):
                        call = bs_call(S0, S0, T, vp, r)
                        prem = (call - S0 + S0 * math.exp(-r * T)) * (1 + OPT_SPREAD)   # ATM put via parity
                        pay = max(S0 - ST, 0.0)
                        d1 = (r + 0.5 * vp * vp) * T / (vp * math.sqrt(T))
                        pdelta = 1 - ncdf(d1)                                   # |put delta|
                        pn = (pay - prem) / S0 + cash
                        pdm = (pay - prem) / S0 / pdelta + cash
                        trades.append({"hold": label, "entry": e.date(), "exit": x.date(), "group": g, "sector": sec,
                                       "ticker": t, "stock_ret": ST / S0 - 1, "iv_put": vp,
                                       "put_premium": prem / S0, "put_ret_on_premium": pay / prem - 1})
                    else:
                        pn = pdm = cash
                    puts["stock"].append(stk + cash)
                    puts["put"].append(pn)
                    puts["put_dm"].append(pdm)
                    puts_g[g].append(pdm)
            for g in ("G1", "G3"):
                if g not in gp:
                    continue
                t = gp[g][0]
                S0, ST = px.at[e, t], px.at[x, t]
                rs = ST / S0 - 1
                rsec = px.at[x, sec] / px.at[e, sec] - 1
                if not np.isfinite(rsec):  # XLC only trades from 2018-06-19
                    rsec = spy
                res = {"stock": rs - 2 * STOCK_COST,
                       "sec": rs - rsec - 2 * STOCK_COST - 2 * ETF_COST + cash,
                       "spy": rs - spy - 2 * STOCK_COST - 2 * ETF_COST + cash}
                # partial SPY hedges, and a beta hedge using the stock's 120-day beta to SPY before entry
                for key, h in (("spy50", 0.5), ("spy75", 0.75)):
                    res[key] = rs - h * spy - 2 * STOCK_COST - h * 2 * ETF_COST + h * cash
                b = float(betas.at[d, t]) if t in betas.columns and pd.notna(betas.at[d, t]) else 1.0
                b = min(max(b, 0.3), 2.5)
                res["spyb"] = rs - b * spy - 2 * STOCK_COST - b * 2 * ETF_COST + b * cash
                vol = iv.at[e, t] / 100 if t in iv.columns and pd.notna(iv.at[e, t]) else np.nan
                opt = {}
                if np.isfinite(vol):
                    for key, K in (("atm", S0), ("c30", strike_for_delta(S0, T, vol, r, 0.30))):
                        prem = bs_call(S0, K, T, vol, r) * (1 + OPT_SPREAD)
                        pay = max(ST - K, 0.0)
                        # P&L per $ of notional: option P&L + T-bill interest on the notional
                        res[key] = (pay - prem) / S0 + cash
                        # delta-matched: calls on notional/delta, so the starting delta equals the stock slot's
                        dlt = ncdf((math.log(S0 / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T)))
                        res[key + "_dm"] = (pay - prem) / S0 / dlt + cash
                        opt[key] = {"K": K / S0, "prem": prem / S0, "ror": pay / prem - 1}
                    # long ATM calls (1x notional) + long ATM SPY puts on 50% / 100% of that notional
                    for key, h in (("pp50", 0.5), ("pp100", 1.0)):
                        res[key] = res["atm"] + (h * spy_put if np.isfinite(spy_put) else 0.0)
                    # long ATM calls on N x slot notional + short SPY shares worth 0.5 x N (≈ delta-neutral)
                    for key, N in (("h1", 1.0), ("h2", 2.0)):
                        res[key] = N * (res["atm"] - cash) - 0.5 * N * (spy + 2 * ETF_COST) + cash
                    if spy_short:
                        # long stock calls (delta-matched) + short SPY calls with equal dollar delta
                        res["x_atm"] = res["atm_dm"] + spy_short["atm"]
                        res["x_c30"] = res["c30_dm"] + spy_short["c30"]
                        res["x_mix"] = res["atm_dm"] + spy_short["c30"]
                    else:
                        res["x_atm"] = res["x_c30"] = res["x_mix"] = cash
                else:
                    res["atm"] = res["c30"] = res["atm_dm"] = res["c30_dm"] = res["x_atm"] = res["x_c30"] = res["x_mix"] = res["h1"] = res["h2"] = res["pp50"] = res["pp100"] = cash  # no IV -> stay in cash
                for k in slots:
                    slots[k].append(res[k])
                    if g == "G3":
                        g3[k].append(res[k])
                trades.append({"hold": label, "entry": e.date(), "exit": x.date(), "group": g, "sector": sec,
                               "ticker": t, "stock_ret": rs, "iv_call": vol,
                               **{f"{k}_strike": v["K"] for k, v in opt.items()},
                               **{f"{k}_premium": v["prem"] for k, v in opt.items()},
                               **{f"{k}_ret_on_premium": v["ror"] for k, v in opt.items()}})
        row = {"exit": x, "signal": d, "entry": e, "SPY": spy, "cash": cash,
               "spy_put": spy_put if np.isfinite(spy_put) else 0.0}
        for k in slots:
            row[f"L18 {k}"] = (sum(slots[k]) + (2 * n - len(slots[k])) * cash) / (2 * n)
            row[f"G3 {k}"] = (sum(g3[k]) + (n - len(g3[k])) * cash) / n
        long_ = sum(slots["stock"]) / (2 * n)
        fill = lambda lst, m: (sum(lst) + (m - len(lst)) * cash) / m  # noqa: E731
        row["S18 stock"] = fill(puts["stock"], 2 * n)
        row["S18 put"] = fill(puts["put"], 2 * n)
        row["S18 put_dm"] = fill(puts["put_dm"], 2 * n)
        row["G2 put_dm"] = fill(puts_g["G2"], n)
        row["G4 put_dm"] = fill(puts_g["G4"], n)
        # 36-slot book: calls on the 18 long slots, puts on the 18 short slots
        row["CP36 atm"] = 0.5 * (row["L18 atm"] + row["S18 put"])
        row["CP36 atm_dm"] = 0.5 * (row["L18 atm_dm"] + row["S18 put_dm"])
        # stock longs hedged with sector ETF + puts on the short picks (half-weight each side)
        row["SEC+PUT"] = 0.5 * (row["L18 sec"] + row["S18 put_dm"])
        row["Book 18L/18S (stock shorts)"] = 0.5 * (long_ + sum(shorts) / (2 * n))
        rows.append(row)
    pd.DataFrame(spy_put_trades).to_csv(os.path.join(bt.OUT, f"spy_put_trades_{rebal}d.csv"), index=False)
    return pd.DataFrame(rows).set_index("exit"), pd.DataFrame(trades)


NAMES = {
    "L18 stock": "18 longs, stock (unhedged)",
    "L18 sec": "18 longs + short sector ETF",
    "L18 spy": "18 longs + short SPY",
    "L18 spy50": "18 longs + short SPY 0.5x",
    "L18 spy75": "18 longs + short SPY 0.75x",
    "L18 spyb": "18 longs + short SPY beta-matched",
    "L18 atm": "18 longs via ATM calls (notional-matched)",
    "L18 c30": "18 longs via 30-delta calls (notional-matched)",
    "L18 atm_dm": "18 longs via ATM calls (delta-matched)",
    "L18 c30_dm": "18 longs via 30-delta calls (delta-matched)",
    "L18 x_atm": "18 long ATM calls + short ATM SPY calls",
    "L18 x_c30": "18 long 30d calls + short 30d SPY calls",
    "L18 x_mix": "18 long ATM calls + short 30d SPY calls",
    "L18 h1": "18 long ATM calls (1x notional) + short SPY 0.5x",
    "L18 h2": "18 long ATM calls (2x notional) + short SPY 0.5x",
    "L18 pp50": "18 long ATM calls + long ATM SPY puts 50%",
    "L18 pp100": "18 long ATM calls + long ATM SPY puts 100%",
    "G3 stock": "G3 only, stock",
    "G3 sec": "G3 only + short sector ETF",
    "G3 spy": "G3 only + short SPY",
    "G3 spyb": "G3 only + short SPY beta-matched",
    "G3 atm": "G3 only via ATM calls (notional-matched)",
    "G3 c30": "G3 only via 30-delta calls (notional-matched)",
    "G3 atm_dm": "G3 only via ATM calls (delta-matched)",
    "G3 c30_dm": "G3 only via 30-delta calls (delta-matched)",
    "G3 x_atm": "G3 long ATM calls + short ATM SPY calls",
    "G3 x_c30": "G3 long 30d calls + short 30d SPY calls",
    "G3 x_mix": "G3 long ATM calls + short 30d SPY calls",
    "G3 h1": "G3 long ATM calls (1x notional) + short SPY 0.5x",
    "G3 h2": "G3 long ATM calls (2x notional) + short SPY 0.5x",
    "G3 pp50": "G3 long ATM calls + long ATM SPY puts 50%",
    "G3 pp100": "G3 long ATM calls + long ATM SPY puts 100%",
    "S18 stock": "18 shorts, stock (short book alone)",
    "S18 put": "18 shorts via ATM puts (notional-matched)",
    "S18 put_dm": "18 shorts via ATM puts (delta-matched)",
    "G2 put_dm": "G2 only via ATM puts (delta-matched)",
    "G4 put_dm": "G4 only via ATM puts (delta-matched)",
    "CP36 atm": "36-slot book: ATM calls longs + ATM puts shorts (notional)",
    "CP36 atm_dm": "36-slot book: ATM calls longs + ATM puts shorts (delta-matched)",
    "SEC+PUT": "50% longs+sector ETF hedge, 50% puts on shorts",
    "Book 18L/18S (stock shorts)": "Old book: 18L / 18S stocks",
    "SPY": "SPY buy & hold",
}


def metrics(r, spy, cash, ppy):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    ex = r - cash
    vol = r.std() * np.sqrt(ppy)
    dvol = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ppy)
    q = r.quantile(0.05)
    cagr = eq.iloc[-1] ** (ppy / len(r)) - 1
    cov = np.cov(r, spy)
    return {"CAGR": cagr, "Ann. vol": vol, "Sharpe": ex.mean() * ppy / vol,
            "Sortino": ex.mean() * ppy / dvol if dvol else np.nan, "Max drawdown": dd.min(),
            "Calmar": cagr / -dd.min() if dd.min() < 0 else np.nan,
            "CVaR 95%": r[r <= q].mean(), "Worst period": r.min(),
            "Beta to SPY": cov[0, 1] / cov[1, 1], "Hit rate": (r > 0).mean()}


def main():
    px, skew = bt.load()
    f = bt.features(px, skew)
    ivs, rf = load_iv()
    out, eq, all_trades = {}, {}, []
    for label, (rebal, _, _) in HOLDS.items():
        per, tr = run_hold(label, px, f, ivs, rf)
        ppy = 252 / rebal
        out[label] = {k: metrics(per[k], per["SPY"], per["cash"], ppy) for k in NAMES}
        eq[label] = {"dates": [str(d.date()) for d in per.index],
                     **{k: list(np.round((1 + per[k]).cumprod().values, 4)) for k in NAMES}}
        per.to_csv(os.path.join(bt.OUT, f"variants_returns_{rebal}d.csv"))
        all_trades.append(tr)
    tr = pd.concat(all_trades)
    tr.to_csv(os.path.join(bt.OUT, "variants_option_trades.csv"), index=False)
    opt = []
    for (hold, grp), g in tr.groupby(["hold", "group"]):
        for k, nm in (("atm", "ATM call"), ("c30", "30-delta call"), ("put", "ATM put")):
            col = f"{k}_ret_on_premium"
            if col not in g:
                continue
            s = g[col].dropna()
            if s.empty:
                continue
            opt.append({"hold": hold, "group": grp, "option": nm, "trades": len(s),
                        "avg premium % spot": g[f"{k}_premium"].mean(),
                        "avg strike % spot": g[f"{k}_strike"].mean() if f"{k}_strike" in g else 1.0,
                        "avg return on premium": s.mean(), "median": s.median(),
                        "expired worthless": (s <= -0.999).mean(), "hit rate": (s > 0).mean(),
                        "best": s.max()})
    opt = pd.DataFrame(opt)
    opt.to_csv(os.path.join(bt.OUT, "variants_option_stats.csv"), index=False)
    tbl = pd.concat({h: pd.DataFrame(v).T for h, v in out.items()}, axis=1)
    tbl.to_csv(os.path.join(bt.OUT, "variants_stats.csv"))
    json.dump({"names": NAMES, "stats": {h: {k: v for k, v in d.items()} for h, d in out.items()},
               "equity": eq, "options": opt.to_dict("records"),
               "assumptions": {"stock_cost_bp_side": STOCK_COST * 1e4, "etf_cost_bp_side": ETF_COST * 1e4,
                               "option_spread_pct_premium": OPT_SPREAD * 100}},
              open(os.path.join(bt.OUT, "variants.json"), "w"), default=float)
    pd.set_option("display.width", 250)
    for h in HOLDS:
        print(f"\n== {h}"); print(pd.DataFrame(out[h]).T.rename(index=NAMES).round(3))
    print(opt.round(3))


if __name__ == "__main__":
    main()
