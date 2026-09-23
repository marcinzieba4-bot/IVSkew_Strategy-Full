# IV Skew Sector Strategy

This project backtests 3-week sector baskets built from VolVue 90-day IV skew (`iv_skew_90`) and price action. The universe is the top 20 holdings of XLK, XLY, XLV, XLI, XLC, XLB and XLE.

| Group | Side | Rule (skew in vol pts; 4.0 = 0.04) | Pick per sector |
|---|---|---|---|
| G1 | Long | skew ≤ 4, 20d return > 0, above 50d SMA | strongest 20d return |
| G2 | Short | skew ≤ 4, 20d return < 0, below 50d SMA | weakest 20d return |
| G3 | Long | skew ≥ 5, ≥ 8% below 60d high (pullback advanced) | deepest drawdown |
| G4 | Short | skew ≥ 5, 20d return > 0, within 3% of 60d high | highest skew |

A position is entered at the close after the signal and held for 15 trading days, with 10 bp cost per side.

```bash
export VOLVUE_API_KEY=...          # GET https://api.volvue.com/query?apiKey=..&format=json&data=<SQL on table `data`>
python fetch_data.py               # VolVue skew + Yahoo adjusted closes -> data/
python backtest.py                 # stats, trades, current basket -> output/
python control_test.py             # dip-buying control by skew bucket
python build_report.py             # output/report.html
```

Caveat: the universe is today's list of holdings applied back to 2018, so the results carry survivorship bias. The fairer read is the sector-relative and control tables.
