"""Top-20 holdings of each SPDR sector ETF (approximate, as of 2026).

NOTE: this is a *static* list of today's largest holdings applied to the whole
backtest, which introduces survivorship / look-ahead bias (today's winners are
over-represented). Treat absolute returns with caution; the relative comparison
between the four skew groups is the more meaningful output.
"""

SECTORS = {
    "XLK": ["NVDA", "MSFT", "AAPL", "AVGO", "ORCL", "PLTR", "AMD", "CSCO", "IBM", "CRM",
            "MU", "LRCX", "AMAT", "INTU", "NOW", "APH", "QCOM", "KLAC", "TXN", "ADBE"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "TJX", "LOW", "SBUX", "ORLY", "DASH",
            "NKE", "GM", "MAR", "RCL", "ABNB", "AZO", "CMG", "HLT", "ROST", "F"],
    "XLV": ["LLY", "JNJ", "ABBV", "UNH", "MRK", "ABT", "TMO", "ISRG", "AMGN", "BSX",
            "GILD", "DHR", "PFE", "SYK", "VRTX", "MDT", "BMY", "MCK", "CVS", "ELV"],
    "XLI": ["GE", "CAT", "RTX", "UBER", "GEV", "BA", "UNP", "HON", "ETN", "DE",
            "LMT", "ADP", "PH", "TT", "MMM", "GD", "WM", "CTAS", "NOC", "TDG"],
    "XLC": ["META", "GOOGL", "NFLX", "T", "VZ", "DIS", "CMCSA", "TMUS", "EA", "TTWO",
            "WBD", "CHTR", "LYV", "OMC", "TKO", "FOXA", "NWSA", "MTCH", "IPG", "FOX"],
    "XLB": ["LIN", "SHW", "NEM", "ECL", "APD", "FCX", "CTVA", "MLM", "VMC", "NUE",
            "DD", "PPG", "IP", "SW", "DOW", "STLD", "PKG", "IFF", "LYB", "AMCR"],
    "XLE": ["XOM", "CVX", "COP", "WMB", "EOG", "KMI", "PSX", "SLB", "MPC", "OKE",
            "VLO", "BKR", "TRGP", "EQT", "OXY", "FANG", "EXE", "HAL", "DVN", "CTRA"],
}

BENCHMARKS = ["SPY"] + list(SECTORS)
ALL_TICKERS = sorted({t for v in SECTORS.values() for t in v})
TICKER_SECTOR = {t: s for s, v in SECTORS.items() for t in v}
