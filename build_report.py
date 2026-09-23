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
res["variants"] = json.load(open(os.path.join(HERE, "output", "variants.json")))
html_data = json.dumps(res).replace("NaN", "null")
html = open(os.path.join(HERE, "report_template.html")).read().replace("__DATA__", html_data)
open(os.path.join(HERE, "output", "report.html"), "w").write(html)
print("wrote output/report.html", len(html))
