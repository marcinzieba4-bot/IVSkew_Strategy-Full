"""Fetch ATM call IVs (20d / 30d tenor) from VolVue and the 13-week T-bill yield (^IRX) from Yahoo."""
import os

import pandas as pd

from fetch_data import DATA, fetch_yahoo, fetch_volvue_fields
from universe import ALL_TICKERS

if __name__ == "__main__":
    fetch_volvue_fields(ALL_TICKERS + ["SPY"], "ticker,date,iv_call_20,iv_call_30").to_csv(
        os.path.join(DATA, "volvue_call_iv.csv"), index=False)
    fetch_yahoo("^IRX").rename("IRX").to_csv(os.path.join(DATA, "tbill.csv"))
    print("done")
