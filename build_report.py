"""Render output/report.html from output/results.json + control test."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "output", "results.json")))
ctrl = pd.read_csv(os.path.join(HERE, "output", "control_dip_by_skew.csv"), index_col=0)
res["control"] = [{"bucket": b, "avg": r["avg excess vs sector / 3wk"], "hit": r["hit rate"]} for b, r in ctrl.iterrows()]
res.pop("last_baskets", None)
res["wing"] = pd.read_csv(os.path.join(HERE, "output", "spy_wing_sensitivity.csv")).to_dict("records")
res["pt_diag"] = pd.read_csv(os.path.join(HERE, "output", "put_timing_diagnostics.csv")).to_dict("records")
res["pt_books"] = pd.read_csv(os.path.join(HERE, "output", "put_timing_books.csv")).to_dict("records")
res["today"] = json.load(open(os.path.join(HERE, "output", "today_simulation.json")))
sp = pd.read_csv(os.path.join(HERE, "output", "audit_split_sample.csv"), header=[0, 1], index_col=0)
sp.columns = [f"{h} · {p}" for h, p in sp.columns]
res["audit"] = {"split": sp.reset_index().to_dict("records"), "checks": [
    ["Picks recomputed at every signal date with ALL data cut at that date (140 dates)", "0 mismatches with the backtest's picks", "PASS"],
    ["120-day betas for the beta hedge recomputed point-in-time", "max difference 0", "PASS"],
    ["Signal on the close of day d, trade at the close of d+1; VolVue skew is end-of-day for d", "one full day between information and trade", "PASS"],
    ["Signal one day older (d-1) instead of d", "Sharpe 1.30 → 1.13 (sector hedge), 1.19 → 1.06 (SPY 0.75x): still works", "PASS"],
    ["Planted leak (signal from d+15) caught by the point-in-time test", "138/138 mismatches, Sharpe jumps to 2.2: the test can detect leaks", "PASS"],
    ["Option IVs, T-bill rate, SPY put-timing rules use values at entry or earlier", "checked in code: iv at entry close, rf at entry, trailing percentiles", "PASS"],
    ["Universe = TODAY's top-20 holdings, used back to 2018", "survivorship bias: stocks that fell out of the top 20 are missing", "BIAS"],
    ["Design choices (long side, 3-week hold, hedge type) picked after seeing the full sample", "but the first half alone gives the same choices (table below)", "MINOR"],
    ["VolVue history may be revised; point-in-time vintages are not available", "cannot be tested", "UNKNOWN"],
]}
res["variants"] = json.load(open(os.path.join(HERE, "output", "variants.json")))
html_data = json.dumps(res).replace("NaN", "null")
html = open(os.path.join(HERE, "report_template.html")).read().replace("__DATA__", html_data)
open(os.path.join(HERE, "output", "report.html"), "w").write(html)
print("wrote output/report.html", len(html))
