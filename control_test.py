"""Control: is G3's edge from the skew signal or just from buying dips in today's mega-caps?

Same dip rule as G3 (>= 8% below 60d high, deepest pick per sector), bucketed by skew.
All buckets share the same survivorship bias, so the *spread* between them isolates skew.
"""
import numpy as np
import pandas as pd

import backtest as bt
from universe import SECTORS

px, skew = bt.load()
f = bt.features(px, skew)
dates = px.index[px.index >= bt.START]
buckets = {"low skew <=4": (-99, bt.LOW_SKEW), "mid 4-5": (bt.LOW_SKEW, bt.HIGH_SKEW),
           "high >=5 (G3)": (bt.HIGH_SKEW, 99), "any skew": (-99, 99)}
res = {b: [] for b in buckets}
for i in range(0, len(dates) - 1 - bt.REBAL, bt.REBAL):
    d, e, x = dates[i], dates[i + 1], dates[i + 1 + bt.REBAL]
    for b, (lo, hi) in buckets.items():
        rel = []
        for sec, names in SECTORS.items():
            names = [t for t in names if t in px.columns]
            row = pd.DataFrame({k: f[k].loc[d, names] for k in ("skew", "dd")}).dropna()
            c = row[(row["skew"] > lo) & (row["skew"] <= hi) & (row.dd <= bt.PULLBACK)].sort_values("dd")
            if len(c):
                t = c.index[0]
                rel.append(px.at[x, t] / px.at[e, t] - px.at[x, sec] / px.at[e, sec])
        res[b].append(np.mean(rel) if rel else np.nan)
out = pd.DataFrame({b: {"avg excess vs sector / 3wk": np.nanmean(v), "hit rate": np.nanmean(np.array(v) > 0),
                        "baskets w/ picks": int(np.sum(~np.isnan(v)))} for b, v in res.items()}).T
print(out.round(4))
out.to_csv("output/control_dip_by_skew.csv")
