import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_input.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "vix_jpm_corr_output.csv")

# 1) 读取数据
df = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date").sort_index()

# 2) 检查必要列
if "log_return" not in df.columns or "vix" not in df.columns:
    raise RuntimeError("缺少 log_return 或 vix 列")

# 3) 计算 20 日滚动相关性
window = 20
df["vix_jpm_corr_20"] = (
    df["log_return"]
    .rolling(window)
    .corr(df["vix"])
)

# 4) 保存结果
out = df[["log_return", "vix", "vix_jpm_corr_20"]].dropna()
out.to_csv(OUT_PATH)

print("✅ 滚动相关计算完成")
print("输出文件:", OUT_PATH)
print("时间范围:", out.index.min().date(), "~", out.index.max().date())
print("样本量:", len(out))
