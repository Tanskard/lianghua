import os
import time
import random
import json
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

# 按周抓取：maxrecords 建议调高（因为一次抓 7 天）
MAX_RECORDS_WEEK = 250

# 基础休眠（每次窗口请求后都睡一下）
BASE_SLEEP_SEC = 2.0

# 429 退避参数
MAX_RETRIES = 8
BACKOFF_BASE = 5

# 请求最小间隔（避免过于频繁）
MIN_INTERVAL = 1.6
_last_ts = 0.0

# =========================
# requests Session
# =========================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
})


def throttle():
    """全局节流：保证两次请求至少间隔 MIN_INTERVAL 秒。"""
    global _last_ts
    now = time.time()
    wait = MIN_INTERVAL - (now - _last_ts)
    if wait > 0:
        time.sleep(wait + random.uniform(0.0, 0.4))
    _last_ts = time.time()


def polarity_textblob(text: str) -> float:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    return TextBlob(text).sentiment.polarity


def _get_json_with_backoff(url, params):
    """
    带指数退避的 JSON 请求：
    - 429：读取 Retry-After（若有）否则指数退避
    - 5xx：指数退避
    - 200 但非 JSON：抛出错误并打印 head，便于定位（常见：HTML 网关页）
    """
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        try:
            r = SESSION.get(url, params=params, timeout=30)
            code = r.status_code

            if code == 429:
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    sleep_s = float(ra) + random.uniform(0.5, 2.0)
                else:
                    sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0)
                print(f"  ⏳ 429 Rate limited. attempt={attempt}/{MAX_RETRIES}, sleep={sleep_s:.1f}s")
                time.sleep(min(sleep_s, 120))
                continue

            if 500 <= code < 600:
                sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0)
                print(f"  ⏳ {code} Server error. attempt={attempt}/{MAX_RETRIES}, sleep={sleep_s:.1f}s")
                time.sleep(min(sleep_s, 120))
                continue

            if code != 200:
                head = (r.text or "")[:200].replace("\n", " ")
                raise RuntimeError(f"HTTP {code}, body_head={head}")

            text = (r.text or "").strip()
            if not text:
                raise RuntimeError("Empty response body")

            if not (text.startswith("{") or text.startswith("[")):
                head = text[:200].replace("\n", " ")
                raise RuntimeError(f"Non-JSON response, head={head}")

            return json.loads(text)

        except Exception as e:
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0)
            print(f"  ⚠️ Request/parse error attempt={attempt}/{MAX_RETRIES}: {repr(e)}; sleep={sleep_s:.1f}s")
            time.sleep(min(sleep_s, 120))

    raise RuntimeError("Exceeded max retries (rate limit too strict or network unstable).")


def _parse_article_day(a: dict, fallback_day: str) -> str:
    """
    从 GDELT article 里解析日期（尽量鲁棒）：
    优先 seendate（常见形态：YYYYMMDDHHMMSS 或 ISO），解析失败就用 fallback_day。
    """
    sd = a.get("seendate", "") or a.get("seenDate", "") or ""
    if isinstance(sd, str):
        s = sd.strip()
        # 形如 20180101123456
        if len(s) >= 8 and s[:8].isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        # ISO 形如 2018-01-01T12:34:56Z
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
    return fallback_day


def gdelt_fetch_window(start_day: str, end_day: str) -> pd.DataFrame:
    """
    按时间窗抓取 articles 列表，返回 [date, title, url]（date 为日粒度字符串）
    """
    start_dt = start_day.replace("-", "") + "000000"
    end_dt = end_day.replace("-", "") + "235959"

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS_WEEK,
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "sourcelang": "English",
        "sort": "datedesc",
    }

    data = _get_json_with_backoff(url, params)
    arts = data.get("articles", []) or []

    if not arts:
        return pd.DataFrame(columns=["date", "title", "url"])

    rows = []
    for a in arts:
        day = _parse_article_day(a, fallback_day=start_day)
        rows.append({
            "date": day,
            "title": a.get("title", "") or "",
            "url": a.get("url", "") or "",
        })

    return pd.DataFrame(rows)


def _save_partial(out_rows):
    df_tmp = pd.DataFrame(out_rows)
    df_tmp.to_csv(OUT_SENT_PATH, index=False)


def build_daily_sentiment_weekly(start=START, end=END) -> pd.DataFrame:
    """
    核心函数：
    - 用 7 天窗口请求 GDELT（大幅减少 API 调用次数）
    - 窗口内拿到 articles 后在本地按天聚合情绪
    - 断点续跑：OUT_SENT_PATH 里已有的 date 会被跳过
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    all_days = pd.date_range(start_dt, end_dt, freq="D").strftime("%Y-%m-%d").tolist()

    # ===== 断点续跑：读取已完成日期 =====
    done_dates = set()
    out_rows = []
    if os.path.exists(OUT_SENT_PATH):
        try:
            old = pd.read_csv(OUT_SENT_PATH, parse_dates=["date"])
            done_dates = set(old["date"].dt.strftime("%Y-%m-%d").tolist())
            out_rows = old.to_dict("records")
            print(f"✅ Resume mode: found existing {len(done_dates)} days in {OUT_SENT_PATH}")
        except Exception:
            done_dates = set()
            out_rows = []

    # 构造按 7 天的窗口起点序列
    window_starts = pd.date_range(start_dt, end_dt, freq="7D")
    if len(window_starts) == 0 or window_starts[0] != start_dt:
        window_starts = pd.DatetimeIndex([start_dt]).append(window_starts)

    total_win = len(window_starts)

    for w_i, w_start in enumerate(window_starts, 1):
        w_end = min(w_start + pd.Timedelta(days=6), end_dt)

        start_day = w_start.strftime("%Y-%m-%d")
        end_day = w_end.strftime("%Y-%m-%d")
        win_days = pd.date_range(w_start, w_end, freq="D").strftime("%Y-%m-%d").tolist()

        # 若窗口内全部已完成，则跳过
        if all(d in done_dates for d in win_days):
            if w_i % 10 == 0:
                print(f"[{w_i}/{total_win}] Skipping window {start_day} ~ {end_day} (all done)")
            continue

        print(f"[{w_i}/{total_win}] Fetching window {start_day} ~ {end_day} ...")

        try:
            df_win = gdelt_fetch_window(start_day, end_day)
        except Exception as e:
            print(f"  ⚠️ Window fetch failed {start_day}~{end_day}: {repr(e)}")
            # 失败也补全窗口内未完成日期，避免死循环
            for d in win_days:
                if d in done_dates:
                    continue
                out_rows.append({
                    "date": d,
                    "news_count": 0,
                    "news_sent_mean": np.nan,
                    "news_sent_std": np.nan,
                    "news_sent_pos_ratio": np.nan,
                    "news_sent_neg_ratio": np.nan,
                })
                done_dates.add(d)
            _save_partial(out_rows)
            time.sleep(BASE_SLEEP_SEC + random.uniform(0, 1.5))
            continue

        if len(df_win) > 0:
            df_win["polarity"] = df_win["title"].apply(polarity_textblob)

        # 窗口内逐日聚合（只对未完成日期计算）
        for d in win_days:
            if d in done_dates:
                continue

            if len(df_win) == 0:
                out_rows.append({
                    "date": d,
                    "news_count": 0,
                    "news_sent_mean": np.nan,
                    "news_sent_std": np.nan,
                    "news_sent_pos_ratio": np.nan,
                    "news_sent_neg_ratio": np.nan,
                })
                done_dates.add(d)
                continue

            df_day = df_win[df_win["date"] == d]
            if len(df_day) == 0:
                out_rows.append({
                    "date": d,
                    "news_count": 0,
                    "news_sent_mean": np.nan,
                    "news_sent_std": np.nan,
                    "news_sent_pos_ratio": np.nan,
                    "news_sent_neg_ratio": np.nan,
                })
            else:
                pol = df_day["polarity"].dropna()
                if len(pol) == 0:
                    out_rows.append({
                        "date": d,
                        "news_count": int(len(df_day)),
                        "news_sent_mean": np.nan,
                        "news_sent_std": np.nan,
                        "news_sent_pos_ratio": np.nan,
                        "news_sent_neg_ratio": np.nan,
                    })
                else:
                    out_rows.append({
                        "date": d,
                        "news_count": int(len(df_day)),
                        "news_sent_mean": float(pol.mean()),
                        "news_sent_std": float(pol.std(ddof=0)) if len(pol) > 1 else 0.0,
                        "news_sent_pos_ratio": float((pol > 0).mean()),
                        "news_sent_neg_ratio": float((pol < 0).mean()),
                    })

            done_dates.add(d)

        # 每个窗口完成后立刻落盘
        _save_partial(out_rows)

        # 基础限速 + 抖动
        time.sleep(BASE_SLEEP_SEC + random.uniform(0, 1.5))

    # 输出整理
    df_out = pd.DataFrame(out_rows)
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out = df_out.sort_values("date").drop_duplicates("date").set_index("date")

    # 7 日 EMA 平滑
    df_out["news_sent_ema7"] = df_out["news_sent_mean"].ewm(span=7, adjust=False).mean()

    # 确保全日期覆盖（如果有缺天就补齐）
    full_idx = pd.date_range(start_dt, end_dt, freq="D")
    df_out = df_out.reindex(full_idx)
    df_out.index.name = "date"
    # news_count 缺失补 0，其余后面 merge 时你本来就 ffill/bfill
    if "news_count" in df_out.columns:
        df_out["news_count"] = df_out["news_count"].fillna(0).astype(int)

    return df_out


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
          ["news_count", "news_sent_mean", "news_sent_std",
           "news_sent_pos_ratio", "news_sent_neg_ratio", "news_sent_ema7"])


def main():
    print("=== Step 6 (Weekly windows): GDELT + TextBlob sentiment (daily) ===")
    print("Input:", INPUT_PATH)
    print("Range:", START, "~", END)
    print("Query:", QUERY)
    print("MAX_RECORDS_WEEK:", MAX_RECORDS_WEEK)
    print("-" * 60)

    sent = build_daily_sentiment_weekly()

    # 保存最终版本（带 index）
    final_path = OUT_SENT_PATH.replace(".csv", "_final.csv")
    sent.to_csv(final_path)
    print("✅ daily sentiment (final) saved:", final_path)
    print("Sent range:", sent.index.min().date(), "~", sent.index.max().date(), "rows:", len(sent))

    # 同时也把断点文件整理为最新（可选：覆盖 OUT_SENT_PATH）
    # 这里不强制覆盖，避免你想保留 resume 过程中的原始记录
    # 如果你想覆盖成最终整齐版，取消注释下一行：
    # sent.reset_index().to_csv(OUT_SENT_PATH, index=False)

    merge_to_model(sent)


if __name__ == "__main__":
    main()