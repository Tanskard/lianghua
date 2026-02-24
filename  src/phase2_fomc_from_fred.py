import os
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from datetime import datetime

# ==============================
# 参数设置
# ==============================
START = "2018-01-01"
END = "2024-12-31"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_input.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_with_fed.csv")

# ==============================
# 读取已有数据
# ==============================
df = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date")
df = df.sort_index()

# ==============================
# 从 FRED 下载联邦基金利率
# ==============================
print("Fetching DFF from FRED...")

fed = web.DataReader("DFF", "fred", START, END)
fed = fed.rename(columns={"DFF": "fed_rate_level"})

# ==============================
# 合并到主数据
# ==============================
df = df.merge(fed, left_index=True, right_index=True, how="left")

# 前向填充（FRED 有节假日缺口）
df["fed_rate_level"] = df["fed_rate_level"].ffill()

# ==============================
# 构造特征
# ==============================

# 1️⃣ 利率变化量
df["fed_rate_change"] = df["fed_rate_level"].diff()

# 2️⃣ 利率决议 dummy（是否发生调整）
df["fed_rate_dummy"] = (df["fed_rate_change"] != 0).astype(int)

# ==============================
# 保存
# ==============================
df.to_csv(OUT_PATH)

print("✅ Fed rate variables added.")
print("新增变量：")
print(["fed_rate_level", "fed_rate_change", "fed_rate_dummy"])
print("保存路径：", OUT_PATH)
