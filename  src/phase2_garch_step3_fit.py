import os
import numpy as np
import pandas as pd
from arch import arch_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_input.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "garch_output.csv")

df = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date").sort_index()

if "log_return" not in df.columns:
    raise RuntimeError("model_input.csv 缺少 log_return，无法拟合 GARCH。")

# 1) 准备收益率（百分比尺度更稳）
r_pct = (df["log_return"] * 100).replace([np.inf, -np.inf], np.nan).dropna()

# 2) GARCH(1,1)
model = arch_model(
    r_pct,
    mean="Constant",
    vol="GARCH",
    p=1, q=1,
    dist="normal"
)

res = model.fit(disp="off")
print(res.summary())

# 3) 条件波动率（单位：%），并对齐回 df 的日期索引
cond_vol_pct = res.conditional_volatility.reindex(df.index)

# 4) 保存：日波动率（小数）+ 年化波动率（小数）
df["garch_vol_daily"] = cond_vol_pct / 100.0
df["garch_vol_annual"] = df["garch_vol_daily"] * np.sqrt(252)

# 5) 导出用于对比的列（vol_20 vs garch_vol_annual）
out = df[["log_return", "vol_20", "garch_vol_daily", "garch_vol_annual"]].dropna()
out.to_csv(OUT_PATH)

print("✅ GARCH 输出已保存:", OUT_PATH)
print("输出时间范围:", out.index.min().date(), "~", out.index.max().date())
print("输出行数:", len(out))
