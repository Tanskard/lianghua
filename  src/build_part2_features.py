# -*- coding: utf-8 -*-
"""
PART 2 - Feature Engineering

Input:
- data/raw/jpm_price.csv
- data/raw/vix.csv
- data/raw/risk_free.csv

Output:
- data/processed/model_input.csv

Features:
- close
- log_return
- vol_5, vol_20, vol_60
- vix, vix_change
- rf_rate, rf_daily
- ma_5, ma_20
- log_volume
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_data():
    jpm = pd.read_csv(
        os.path.join(RAW_DIR, "jpm_price.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    vix = pd.read_csv(
        os.path.join(RAW_DIR, "vix.csv"),
        parse_dates=["DATE"],
        index_col="DATE"
    )

    rf = pd.read_csv(
        os.path.join(RAW_DIR, "risk_free.csv"),
        parse_dates=["DATE"],
        index_col="DATE"
    )

    vix.columns = ["vix"]
    rf.columns = ["rf_rate"]

    return jpm, vix, rf


def build_features(jpm, vix, rf):
    df = jpm.copy()

    # === 收益率 ===
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # === 历史波动率（年化）===
    for window in [5, 20, 60]:
        df[f"vol_{window}"] = (
            df["log_return"]
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

    # === 技术指标 ===
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # === 成交量 ===
    df["log_volume"] = np.log(df["volume"])

    # === 合并 VIX 与利率 ===
    df = df.join(vix, how="left")
    df = df.join(rf, how="left")

    # 对齐交易日（向前填充宏观变量）
    df[["vix", "rf_rate"]] = df[["vix", "rf_rate"]].ffill()

    # === VIX 变化 ===
    df["vix_change"] = df["vix"].diff()

    # === 无风险利率处理 ===
    df["rf_rate"] = df["rf_rate"] / 100.0           # % → 小数
    df["rf_daily"] = df["rf_rate"] / 252.0

    return df


def clean_and_save(df):
    df = df.dropna()

    out_path = os.path.join(PROCESSED_DIR, "model_input.csv")
    df.to_csv(out_path)

    print("✅ PART 2 完成")
    print(f"📄 输出文件: {out_path}")
    print(f"📊 样本区间: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"🧮 特征数量: {df.shape[1]}")
    print(f"📈 样本行数: {df.shape[0]}")


if __name__ == "__main__":
    ensure_dirs()
    jpm, vix, rf = load_data()
    df = build_features(jpm, vix, rf)
    clean_and_save(df)
