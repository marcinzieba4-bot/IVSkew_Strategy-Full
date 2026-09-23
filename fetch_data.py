"""Download 90-day IV skew history from VolVue and daily adjusted closes from Yahoo."""
import json
import os
import time

import pandas as pd
import requests

from universe import ALL_TICKERS, BENCHMARKS

START = "2018-01-01"
DATA = os.path.join(os.path.dirname(__file__), "data")
VOLVUE_URL = "https://api.volvue.com/query"
FIELDS = "ticker,date,iv_skew_90,iv_skew_90_perc,iv_mean_90"


def fetch_volvue(tickers, chunk=10):
    return fetch_volvue_fields(tickers, FIELDS, chunk)


def fetch_volvue_fields(tickers, fields, chunk=10):
    key = os.environ["VOLVUE_API_KEY"]
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        sql = (f"SELECT {fields} FROM data WHERE ticker IN ({','.join(repr(t) for t in batch)}) "
               f"AND date>='{START}'")
        r = requests.get(VOLVUE_URL, params={"apiKey": key, "format": "json", "data": sql}, timeout=120)
        r.raise_for_status()
        d = r.json()
        if d.get("error"):
            raise RuntimeError(d["error"])
        frames.append(pd.DataFrame(d["records"], columns=d["columnNames"]))
        print(f"volvue {batch[0]}..{batch[-1]}: {len(frames[-1])} rows")
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_yahoo(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": int(pd.Timestamp(START).timestamp()), "period2": int(time.time()),
              "interval": "1d", "events": "div,splits"}
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    adj = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype=float)
    close = pd.Series(res["indicators"]["quote"][0]["close"], index=idx, dtype=float)
    return adj.fillna(close).rename(ticker)  # latest bar often has no adjclose yet


def main():
    os.makedirs(DATA, exist_ok=True)
    fetch_volvue(ALL_TICKERS).to_csv(os.path.join(DATA, "volvue_skew90.csv"), index=False)
    closes = []
    for t in ALL_TICKERS + BENCHMARKS:
        yt = t.replace(".", "-")
        for attempt in range(4):
            try:
                closes.append(fetch_yahoo(yt).rename(t))
                break
            except Exception as e:  # noqa: BLE001
                print(f"yahoo {t} attempt {attempt}: {e}")
                time.sleep(2 ** attempt)
    px = pd.concat(closes, axis=1).sort_index()
    px = px[~px.index.duplicated()]
    px.to_csv(os.path.join(DATA, "prices.csv"))
    print("prices", px.shape)


if __name__ == "__main__":
    main()
