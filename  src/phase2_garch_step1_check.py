import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_input.csv")

df = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date").sort_index()

print("✅ 文件读取成功:", INPUT_PATH)
print("时间范围:", df.index.min().date(), "~", df.index.max().date())
print("列名:", list(df.columns))

# 核心列检查
assert "log_return" in df.columns, "❌ 缺少 log_return 列，无法做 GARCH"
print("✅ log_return 存在")

print("log_return 描述统计：")
print(df["log_return"].describe())
