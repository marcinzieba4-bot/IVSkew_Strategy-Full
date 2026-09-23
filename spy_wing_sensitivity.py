"""Sensitivity of the 'long stock calls + short SPY calls' books to the SPY 30-delta call IV discount."""
import pandas as pd

import backtest as bt
import variants as v

px, skew = bt.load()
f = bt.features(px, skew)
ivs, rf = v.load_iv()
rows = []
for disc in (0.0, 1.0, 2.0, 3.0, 4.0):
    v.SPY_WING_DISCOUNT = disc
    for label, (rebal, _, _) in v.HOLDS.items():
        per, _ = v.run_hold(label, px, f, ivs, rf)
        for k in ("L18 x_mix", "L18 x_c30", "G3 x_mix"):
            m = v.metrics(per[k], per["SPY"], per["cash"], 252 / rebal)
            rows.append({"SPY 30d IV discount (vol pts)": disc, "hold": label, "variant": v.NAMES[k], **m})
out = pd.DataFrame(rows)
out.to_csv("output/spy_wing_sensitivity.csv", index=False)
pd.set_option("display.width", 250)
print(out[["SPY 30d IV discount (vol pts)", "hold", "variant", "CAGR", "Ann. vol", "Sharpe", "Max drawdown", "Beta to SPY"]].round(3).to_string(index=False))
