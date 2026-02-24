import os
import time
import random
import requests
import numpy as np
import pandas as pd
from textblob import TextBlob

# =========================
# 基本路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "model_input.csv")

OUT_SENT_PATH = os.path.join(PROCESSED_DIR, "news_sent_daily_gdelt.csv")
OUT_MERGED_PATH = os.path.join(PROCESSED_DIR, "model_input_with_sentiment.csv")

# =========================
# 时间范围
# =========================
START = "2018-01-01"
END = "2024-12-31"

# =========================
# GDELT 查询
# =========================
QUERY = '(JPMorgan OR "JPMorgan Chase" OR "JP Morgan" OR JPM)'

# 降低压力：每天最多抓 50/100 就够做情绪（标题情绪不需要太多）
MAX_RECORDS = 50

# 基础休眠（每次请求后都睡一下）
BASE_SLEEP_SEC = 2.0

# 429 退避参数
MAX_RETRIES = 8            # 单日最多重试次数
BACKOFF_BASE = 5           # 429 第一次等待 5 秒，然后指数增长


def polarity_textblob(text: str) -> float:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    return TextBlob(text).sentiment.polarity


def _request_with_backoff(url, params):
    """
    带指数退避的请求：遇到 429 或网络问题自动重试
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)

            # 429：限流
            if r.status_code == 429:
                sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  ⏳ 429 Rate limited. attempt={attempt}/{MAX_RETRIES}, sleep={sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r

        except requests.RequestException as e:
            # 其他网络问题也退避
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"  ⚠️ Request error attempt={attempt}/{MAX_RETRIES}: {repr(e)}; sleep={sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError("Exceeded max retries (rate limit too strict or network unstable).")


def gdelt_fetch_day(day: str) -> pd.DataFrame:
    start = day.replace("-", "") + "000000"
    end = day.replace("-", "") + "235959"

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "startdatetime": start,
        "enddatetime": end,
        "sourcelang": "English",
        "sort": "datedesc",
    }

    r = _request_with_backoff(url, params)
    data = r.json()
    arts = data.get("articles", [])

    if not arts:
        return pd.DataFrame(columns=["date", "title", "url"])

    rows = []
    for a in arts:
        rows.append({
            "date": day,
            "title": a.get("title", ""),
            "url": a.get("url", ""),
        })
    return pd.DataFrame(rows)


def build_daily_sentiment(start=START, end=END) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")

    # ===== 断点续跑：如果 OUT_SENT_PATH 已存在，就读取已完成的日期 =====
    done_dates = set()
    if os.path.exists(OUT_SENT_PATH):
        try:
            old = pd.read_csv(OUT_SENT_PATH, parse_dates=["date"])
            done_dates = set(old["date"].dt.strftime("%Y-%m-%d").tolist())
            print(f"✅ Resume mode: found existing {len(done_dates)} days in {OUT_SENT_PATH}")
        except Exception:
            done_dates = set()

    out_rows = []
    # 如果已有文件，先把旧数据带上（防止覆盖）
    if os.path.exists(OUT_SENT_PATH) and len(done_dates) > 0:
        old = pd.read_csv(OUT_SENT_PATH, parse_dates=["date"])
        old = old.sort_values("date")
        out_rows = old.to_dict("records")

    for i, d in enumerate(days, 1):
        day = d.strftime("%Y-%m-%d")

        if day in done_dates:
            if i % 50 == 0:
                print(f"[{i}/{len(days)}] Skipping {day} (already done)")
            continue

        print(f"[{i}/{len(days)}] Fetching {day} ...")

        try:
            df_day = gdelt_fetch_day(day)
        except Exception as e:
            print(f"  ⚠️ Fetch failed for {day}: {repr(e)}")
            out_rows.append({
                "date": day,
                "news_count": 0,
                "news_sent_mean": np.nan,
                "news_sent_std": np.nan,
                "news_sent_pos_ratio": np.nan,
                "news_sent_neg_ratio": np.nan,
            })
            # 写入断点文件（即使失败也记一条，避免死循环重跑同一天）
            _save_partial(out_rows)
            time.sleep(BASE_SLEEP_SEC)
            continue

        if len(df_day) == 0:
            out_rows.append({
                "date": day,
                "news_count": 0,
                "news_sent_mean": np.nan,
                "news_sent_std": np.nan,
                "news_sent_pos_ratio": np.nan,
                "news_sent_neg_ratio": np.nan,
            })
        else:
            df_day["polarity"] = df_day["title"].apply(polarity_textblob)
            pol = df_day["polarity"].dropna()

            if len(pol) == 0:
                out_rows.append({
                    "date": day,
                    "news_count": int(len(df_day)),
                    "news_sent_mean": np.nan,
                    "news_sent_std": np.nan,
                    "news_sent_pos_ratio": np.nan,
                    "news_sent_neg_ratio": np.nan,
                })
            else:
                out_rows.append({
                    "date": day,
                    "news_count": int(len(df_day)),
                    "news_sent_mean": float(pol.mean()),
                    "news_sent_std": float(pol.std(ddof=0)) if len(pol) > 1 else 0.0,
                    "news_sent_pos_ratio": float((pol > 0).mean()),
                    "news_sent_neg_ratio": float((pol < 0).mean()),
                })

        # 每天完成后立刻落盘，保证可续跑
        _save_partial(out_rows)

        # 基础限速 + 抖动
        time.sleep(BASE_SLEEP_SEC + random.uniform(0, 1.5))

    df_out = pd.DataFrame(out_rows)
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out = df_out.sort_values("date").drop_duplicates("date").set_index("date")

    # 7 日 EMA 平滑
    df_out["news_sent_ema7"] = df_out["news_sent_mean"].ewm(span=7, adjust=False).mean()

    return df_out


def _save_partial(out_rows):
    df_tmp = pd.DataFrame(out_rows)
    df_tmp.to_csv(OUT_SENT_PATH, index=False)


def merge_to_model(sent_daily: pd.DataFrame):
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"], index_col="date").sort_index()
    df.index = pd.to_datetime(df.index).normalize()

    sent_daily = sent_daily.copy()
    sent_daily.index = pd.to_datetime(sent_daily.index).normalize()

    merged = df.merge(sent_daily, left_index=True, right_index=True, how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    for col in ["news_sent_mean", "news_sent_std", "news_sent_pos_ratio", "news_sent_neg_ratio", "news_sent_ema7"]:
        merged[col] = merged[col].ffill().bfill()

    merged.to_csv(OUT_MERGED_PATH)
    print("✅ merged saved:", OUT_MERGED_PATH)
    print("Added columns:",
          ["news_count", "news_sent_mean", "news_sent_std", "news_sent_pos_ratio", "news_sent_neg_ratio", "news_sent_ema7"])


def main():
    print("=== Step 6: GDELT + TextBlob sentiment (daily) ===")
    print("Input:", INPUT_PATH)
    print("Range:", START, "~", END)
    print("Query:", QUERY)
    print("MAX_RECORDS:", MAX_RECORDS)
    print("-" * 60)

    sent = build_daily_sentiment()

    # 最终再保存一次（index 版）
    sent.to_csv(OUT_SENT_PATH.replace(".csv", "_final.csv"))
    print("✅ daily sentiment (final) saved:", OUT_SENT_PATH.replace(".csv", "_final.csv"))
    print("Sent range:", sent.index.min().date(), "~", sent.index.max().date(), "rows:", len(sent))

    merge_to_model(sent)


if __name__ == "__main__":
    main()