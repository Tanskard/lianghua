# -*- coding: utf-8 -*-
"""
PART 1 - FINAL STABLE VERSION (Tiingo + FRED)

Data sources:
1. JPM stock price (daily OHLCV): Tiingo (FREE API)
2. VIX index (daily): FRED (VIXCLS)
3. Risk-free rate (1Y Treasury): FRED (DGS1)

Time coverage (financially correct):
- 2018–2024 (all trading days)
Note:
- US market closed on 2018-01-01, so data naturally starts from 2018-01-02
"""

import os
import requests
import pandas as pd
import pandas_datareader.data as web

# =============================
# CONFIG
# =============================
TIINGO_API_KEY = "24857b7ab22dddedb7a70b7472e417a96c5916a1"

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # HUATAI
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


# =============================
# UTILS
# =============================
def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def check_coverage(df: pd.DataFrame, name: str):
    """
    Financial-standard coverage check:
    - As long as the data includes trading days in 2018–2024, it is valid.
    - We DO NOT require 2018-01-01 specifically (US market holiday).
    """
    if df.empty:
        raise RuntimeError(f"{name} 数据为空")

    min_dt = df.index.min()
    max_dt = df.index.max()

    start_year = int(START_DATE[:4])
    end_dt = pd.to_datetime(END_DATE)

    if min_dt.year > start_year or max_dt < end_dt:
        raise RuntimeError(
            f"{name} 时间覆盖不足\n"
            f"当前范围: {min_dt.date()} ~ {max_dt.date()}\n"
            f"要求范围: {start_year}–{END_DATE[:4]}"
        )

    print(f"✅ {name} 覆盖检查通过：{min_dt.date()} ~ {max_dt.date()}")


# =============================
# FETCH JPM FROM TIINGO
# =============================
def fetch_jpm_tiingo():
    print("=== Fetching JPM (Tiingo) ===")

    url = "https://api.tiingo.com/tiingo/daily/JPM/prices"
    headers = {
        "Authorization": f"Token {TIINGO_API_KEY}"
    }
    params = {
        "startDate": START_DATE,
        "endDate": END_DATE,
        "format": "json"
    }

    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise RuntimeError("Tiingo 返回空数据，请检查 API Key")

    df = pd.DataFrame(data)

    # 关键修复：去掉 Tiingo 返回的 UTC 时区
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df = df.set_index("date").sort_index()

    # 只保留后续建模需要的字段
    df = df[["open", "high", "low", "close", "volume"]]

    check_coverage(df, "JPM (Tiingo)")

    out_path = os.path.join(RAW_DIR, "jpm_price.csv")
    df.to_csv(out_path)
    print(f"✅ JPM saved to {out_path}")


# =============================
# FETCH VIX FROM FRED
# =============================
def fetch_vix_fred():
    print("=== Fetching VIX (FRED: VIXCLS) ===")

    df = web.DataReader("VIXCLS", "fred", start=START_DATE, end=END_DATE)
    df = df.sort_index()

    # FRED 数据有周末/假日缺失，检查非空部分即可
    check_coverage(df.dropna(), "VIXCLS")

    out_path = os.path.join(RAW_DIR, "vix.csv")
    df.to_csv(out_path)
    print(f"✅ VIX saved to {out_path}")


# =============================
# FETCH RISK-FREE RATE
# =============================
def fetch_rf_fred():
    print("=== Fetching Risk-Free Rate (FRED: DGS1) ===")

    df = web.DataReader("DGS1", "fred", start=START_DATE, end=END_DATE)
    df = df.sort_index()

    check_coverage(df.dropna(), "DGS1")

    out_path = os.path.join(RAW_DIR, "risk_free.csv")
    df.to_csv(out_path)
    print(f"✅ Risk-free rate saved to {out_path}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    ensure_dirs()

    print(f"Project base directory: {BASE_DIR}")
    print(f"Raw data directory:     {RAW_DIR}")
    print(f"Target years:           2018–2024")
    print("-" * 60)

    fetch_jpm_tiingo()
    fetch_vix_fred()
    fetch_rf_fred()

    print("-" * 60)
    print("🎯 PART 1 COMPLETED SUCCESSFULLY (Tiingo + FRED)")
