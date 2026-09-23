# IV Skew Sector Strategy

This project backtests 3-week sector baskets built from VolVue 90-day IV skew (`iv_skew_90`) and price action. The universe is the top 20 holdings of XLK, XLY, XLV, XLI, XLC, XLB, XLE, XLP and XLU. Each basket holds at most one name per group per sector: up to 18 longs (G1+G3) and 18 shorts (G2+G4). The book gives each of the 36 slots 1/36 of capital, and empty slots sit in cash.

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
python fetch_options_data.py       # VolVue ATM call IVs (20d/30d) + T-bill yield
python variants.py                 # hedges (sector ETF / SPY), ATM & 30-delta calls, 3-week vs 1-month
python build_report.py             # output/report.html
```

Caveat: the universe is today's list of holdings applied back to 2018, so the results carry survivorship bias. The fairer read is the sector-relative and control tables.

## Implementation variants (`variants.py`)

Keeps the 18 long slots (G1+G3) and tests a sector-ETF or SPY short hedge instead of single-stock shorts. It also tests ATM or 30-delta calls instead of stock (notional-matched and delta-matched sizing), each with a 3-week or 1-month hold. Calls are priced with Black-Scholes on VolVue `iv_call_20` / `iv_call_30`, include a 3% premium spread, and are held to expiry.
