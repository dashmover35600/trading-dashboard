"""
NYLO Polygon.io 2-Year Backtest
=================================
Pulls 2 years of 1-min OHLCV data for AAPL and GOOGL from Polygon.io
and runs the same v18 VWAP Reclaim + EMA Pullback strategy logic.

Usage:
    export POLYGON_API_KEY="your_key_here"
    python3 backtest_polygon.py

Free tier: 2 years of 1-min data available.
Paid tier not required for historical data pulls.

API docs: https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to
"""

import os
import sys
import json
import time
import datetime
import pytz
import statistics
import random
import urllib.request
import urllib.parse

import pandas as pd
import ta

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
MARKET_TZ = pytz.timezone("America/New_York")
TICKERS   = ["AAPL", "GOOGL"]
BASE      = os.path.dirname(os.path.abspath(__file__))
OUT       = os.path.join(BASE, "backtest_polygon_results.json")

# Strategy parameters (mirrors v18 / backtest.py)
GAIN_TARGET_PCT   = 0.015
STOP_LOSS_PCT     = 0.0075
PARTIAL_EXIT_PCT  = 0.0075
TRAIL_TRIGGER_PCT = 0.0075
TRAIL_STOP_PCT    = 0.003
SLIPPAGE = {"AAPL": 0.0003, "GOOGL": 0.00025}

RSI_BULL_MIN = 52; RSI_BULL_MAX = 72
RSI_BEAR_MIN = 28; RSI_BEAR_MAX = 48
VOLUME_MIN   = 1.2
MIN_SCORE    = 3
CUTOFF       = datetime.time(12, 0)
DAILY_LOSS_LIMIT = -500.0

EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_DATES = {
    "AAPL":  [datetime.date(2024,2,1), datetime.date(2024,5,2),
               datetime.date(2024,8,1), datetime.date(2024,10,31),
               datetime.date(2025,1,30), datetime.date(2025,5,1),
               datetime.date(2025,7,31), datetime.date(2025,10,30)],
    "GOOGL": [datetime.date(2024,1,30), datetime.date(2024,4,25),
               datetime.date(2024,7,23), datetime.date(2024,10,29),
               datetime.date(2025,2,4),  datetime.date(2025,4,29),
               datetime.date(2025,7,29), datetime.date(2025,10,29)],
}

POS_SIZES = {10:5000, 9:4000, 8:3000, 7:2000, 6:1500, 5:1000, 4:750, 3:500}


# ── Polygon.io fetch ──────────────────────────────────────────────────────────

def polygon_fetch(ticker: str, date_from: str, date_to: str,
                  multiplier: int = 1, timespan: str = "minute") -> pd.DataFrame:
    """
    Fetch OHLCV bars from Polygon.io /v2/aggs endpoint.
    Handles pagination automatically (Polygon returns up to 50,000 results/call).
    Returns a DataFrame with DatetimeIndex in ET, columns: Open High Low Close Volume.
    """
    if not POLYGON_API_KEY:
        print("ERROR: Set POLYGON_API_KEY environment variable.")
        sys.exit(1)

    base_url = "https://api.polygon.io/v2/aggs/ticker"
    frames = []
    url = (f"{base_url}/{ticker}/range/{multiplier}/{timespan}/"
           f"{date_from}/{date_to}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")

    while url:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  Polygon fetch error: {e}")
            break

        if data.get("status") not in ("OK", "DELAYED"):
            print(f"  Polygon error: {data.get('status')} — {data.get('message','')}")
            break

        results = data.get("results", [])
        if results:
            df = pd.DataFrame(results)
            df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume","t":"ts"}, inplace=True)
            df.index = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
            df = df[["Open","High","Low","Close","Volume"]]
            frames.append(df)
        else:
            break

        url = data.get("next_url")
        if url:
            url += f"&apiKey={POLYGON_API_KEY}"
        time.sleep(0.12)  # respect free-tier rate limit (5 req/min)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out.between_time("09:30", "15:30")


def fetch_2y(ticker: str) -> pd.DataFrame:
    """Fetch 2 years of 1-min data, chunked into quarterly segments."""
    print(f"  Fetching {ticker} from Polygon.io (2 years of 1-min bars)...")
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=730)

    # Break into 90-day chunks to stay under response size limits
    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + datetime.timedelta(days=89), end)
        print(f"    {chunk_start} → {chunk_end}")
        df = polygon_fetch(ticker, chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if not df.empty:
            frames.append(df)
        chunk_start = chunk_end + datetime.timedelta(days=1)
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    print(f"    {ticker}: {len(combined):,} bars over {len(combined.index.normalize().unique())} days")
    return combined


# ── Helpers (same as main backtest) ──────────────────────────────────────────

def is_earnings_blackout(date, ticker):
    d = date.date() if hasattr(date, "date") else date
    return any(abs((d - ed).days) <= EARNINGS_BLACKOUT_DAYS for ed in EARNINGS_DATES.get(ticker, []))


def apply_slippage(price, direction, slip):
    return round(price * (1 + slip) if direction == "long" else price * (1 - slip), 4)


def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0,"total_pnl_pct":0,"total_pnl_dollar":0,
                "best":0,"worst":0,"max_drawdown":0,"sharpe":0,"profit_factor":0,"expectancy":0,
                "by_ticker":{},"avg_win":0,"avg_loss":0}
    wins   = [t for t in trades if t["pnl_dollar"] > 0]
    losses = [t for t in trades if t["pnl_dollar"] <= 0]
    pnls   = [t["pnl_pct"] for t in trades]
    dols   = [t["pnl_dollar"] for t in trades]
    peak=0; dd=0; cum=0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if cum - peak < dd: dd = cum - peak
    sharpe = 0
    if len(pnls) > 1:
        try: sharpe = round(statistics.mean(pnls)/statistics.stdev(pnls)*(252**0.5), 3)
        except: pass
    gw = sum(t["pnl_dollar"] for t in wins)
    gl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = round(gw/gl, 3) if gl > 0 else 999
    wr = len(wins)/len(trades)
    avg_win  = round(gw/len(wins),  2) if wins   else 0
    avg_loss = round(gl/len(losses),2) if losses else 0
    exp = round(wr*avg_win - (1-wr)*avg_loss, 2)
    by_ticker = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in by_ticker: by_ticker[tk] = {"trades":0,"pnl":0.0,"wins":0}
        by_ticker[tk]["trades"] += 1; by_ticker[tk]["pnl"] += t["pnl_dollar"]
        if t["pnl_dollar"] > 0: by_ticker[tk]["wins"] += 1
    return {"trades":len(trades),"wins":len(wins),"losses":len(losses),
            "win_rate":round(wr*100,2),"total_pnl_pct":round(sum(pnls),3),
            "total_pnl_dollar":round(sum(dols),2),"best":round(max(pnls),3) if pnls else 0,
            "worst":round(min(pnls),3) if pnls else 0,"max_drawdown":round(dd,3),
            "sharpe":sharpe,"profit_factor":pf,"expectancy":exp,
            "avg_win":avg_win,"avg_loss":avg_loss,"by_ticker":by_ticker}


def monte_carlo(trades, n=1000):
    if len(trades) < 10: return {}
    pnls = [t["pnl_dollar"] for t in trades]
    results = []
    for _ in range(n):
        s = random.choices(pnls, k=len(pnls)); cum=0; peak=0; dd=0
        for p in s:
            cum += p
            if cum > peak: peak = cum
            if cum-peak < dd: dd = cum-peak
        results.append({"total":round(cum,2),"max_dd":round(dd,2)})
    totals = [r["total"] for r in results]; dds = [r["max_dd"] for r in results]
    return {"simulations":n,"median_pnl":round(statistics.median(totals),2),
            "pct_profitable":round(sum(1 for t in totals if t>0)/n*100,1),
            "worst_case_dd":round(min(dds),2),"best_case":round(max(totals),2),
            "worst_case":round(min(totals),2),
            "pct_10":round(sorted(totals)[int(n*.1)],2),"pct_90":round(sorted(totals)[int(n*.9)],2)}


# ── Strategy engine (mirrors backtest.py run_strategy) ───────────────────────

def run_strategy(df: pd.DataFrame, ticker: str) -> list:
    slip   = SLIPPAGE.get(ticker, 0.0003)
    trades = []
    dates  = sorted(df.index.normalize().unique())

    full_cl  = df["Close"].squeeze()
    pre_rsi  = ta.momentum.RSIIndicator(full_cl, window=14).rsi()
    pre_ema9 = full_cl.ewm(span=9,  adjust=False).mean()
    pre_ema21= full_cl.ewm(span=21, adjust=False).mean()

    def lookup(s, ts, default=50.0):
        try:
            v = s.asof(ts)
            return default if pd.isna(v) else float(v)
        except: return default

    for date in dates:
        d_date = date.date()
        if is_earnings_blackout(d_date, ticker): continue

        day_df  = df[df.index.date == d_date]
        n_bars  = len(day_df)
        if n_bars < 15: continue

        closes  = day_df["Close"].squeeze().values.tolist()
        highs   = day_df["High"].squeeze().values.tolist()
        lows    = day_df["Low"].squeeze().values.tolist()
        volumes = day_df["Volume"].squeeze().values.tolist()
        idx     = list(day_df.index)

        # VWAP (cumulative intraday)
        cl = day_df["Close"].squeeze(); hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze();   vo = day_df["Volume"].squeeze()
        tp = (hi + lo + cl) / 3
        vwap_list = ((tp * vo).cumsum() / vo.cumsum()).tolist()

        # Opening drive bias (9:30–9:35)
        drive = day_df.between_time("09:30", "09:35")
        day_bias = "neutral"
        if len(drive) >= 2:
            drv_open  = float(drive["Open"].iloc[0])
            drv_close = float(drive["Close"].iloc[-1])
            move_pct  = (drv_close - drv_open) / drv_open
            avg_vol   = float(day_df["Volume"].squeeze().mean())
            drv_vol   = float(drive["Volume"].sum())
            vr_drv    = drv_vol / avg_vol if avg_vol > 0 else 1.0
            if move_pct > 0.003 and vr_drv > 1.5:
                day_bias = "long"
            elif move_pct < -0.003 and vr_drv > 1.5:
                day_bias = "short"

        in_trade   = False
        entry_data = None
        daily_pnl  = 0.0
        traded_today = False

        for i in range(5, n_bars):
            ts    = idx[i]
            t_now = ts.time()

            if in_trade and entry_data:
                # Exit management
                d       = entry_data["dir"]
                price   = closes[i]
                ep      = entry_data["ep"]
                target  = entry_data["target"]
                stop    = entry_data["stop"]
                trail   = entry_data.get("trail_stop")
                ps      = entry_data["ps"]
                shares  = entry_data["shares"]

                # Trailing stop update
                if d == "long":
                    move = (price - ep) / ep
                    if move >= TRAIL_TRIGGER_PCT and not entry_data.get("trail_active"):
                        entry_data["trail_active"] = True
                        entry_data["trail_peak"]   = price
                        trail = round(price * (1 - TRAIL_STOP_PCT), 4)
                        entry_data["trail_stop"]   = trail
                    elif entry_data.get("trail_active") and price > entry_data.get("trail_peak", price):
                        entry_data["trail_peak"] = price
                        trail = round(price * (1 - TRAIL_STOP_PCT), 4)
                        entry_data["trail_stop"] = trail
                else:
                    move = (ep - price) / ep
                    if move >= TRAIL_TRIGGER_PCT and not entry_data.get("trail_active"):
                        entry_data["trail_active"] = True
                        entry_data["trail_peak"]   = price
                        trail = round(price * (1 + TRAIL_STOP_PCT), 4)
                        entry_data["trail_stop"]   = trail
                    elif entry_data.get("trail_active") and price < entry_data.get("trail_peak", price):
                        entry_data["trail_peak"] = price
                        trail = round(price * (1 + TRAIL_STOP_PCT), 4)
                        entry_data["trail_stop"] = trail

                ht   = (d=="long" and price >= target) or (d=="short" and price <= target)
                hs   = (d=="long" and price <= stop)   or (d=="short" and price >= stop)
                htr  = trail is not None and ((d=="long" and price <= trail) or (d=="short" and price >= trail))
                htim = t_now >= CUTOFF

                if ht or hs or htr or htim:
                    res = "Target Hit" if ht else "Trailing Stop" if htr else "Time Exit" if htim else "Stop Loss Hit"
                    xp  = (target if ht else (trail if htr else (stop if hs else price)))
                    xp  = round(xp * (1-slip if d=="long" else 1+slip), 4)
                    pct = ((xp-ep)/ep*100 if d=="long" else (ep-xp)/ep*100)
                    dol = round(pct/100*ps, 2)
                    trades.append({**entry_data["rec"], "exit":xp, "result":res,
                                   "pnl_pct":round(pct,3), "pnl_dollar":dol})
                    daily_pnl += dol
                    in_trade = False; entry_data = None
                continue

            if traded_today or t_now >= CUTOFF or t_now < datetime.time(9, 35):
                continue
            if daily_pnl <= DAILY_LOSS_LIMIT:
                continue

            vwap = vwap_list[i] if i < len(vwap_list) and vwap_list[i] > 0 else None
            if not vwap: continue

            price      = closes[i]
            prev_price = closes[i-1]
            rsi        = lookup(pre_rsi,  ts, 50.0)
            ema9       = lookup(pre_ema9, ts, price)
            ema21      = lookup(pre_ema21,ts, price)

            vol_window = volumes[max(0,i-20):i]
            avg_v      = sum(vol_window)/len(vol_window) if vol_window else 1.0
            vol_ratio  = round(volumes[i] / avg_v, 2) if avg_v > 0 else 1.0

            signal_type = None
            direction   = None

            # VWAP Reclaim
            if (prev_price < vwap and price > vwap and vol_ratio >= VOLUME_MIN
                    and day_bias in ("long","neutral") and RSI_BULL_MIN <= rsi <= RSI_BULL_MAX):
                signal_type = "vwap_reclaim"; direction = "long"
            elif (prev_price > vwap and price < vwap and vol_ratio >= VOLUME_MIN
                    and day_bias in ("short","neutral") and RSI_BEAR_MIN <= rsi <= RSI_BEAR_MAX):
                signal_type = "vwap_reclaim"; direction = "short"

            # EMA Pullback
            if not signal_type:
                if (prev_price <= ema9*1.001 and price > ema9 and price > ema21
                        and vol_ratio >= VOLUME_MIN
                        and day_bias in ("long","neutral") and RSI_BULL_MIN <= rsi <= RSI_BULL_MAX):
                    signal_type = "ema_pullback"; direction = "long"
                elif (prev_price >= ema9*0.999 and price < ema9 and price < ema21
                        and vol_ratio >= VOLUME_MIN
                        and day_bias in ("short","neutral") and RSI_BEAR_MIN <= rsi <= RSI_BEAR_MAX):
                    signal_type = "ema_pullback"; direction = "short"

            if not signal_type: continue

            # Score
            score = 0
            if day_bias == direction: score += 2
            elif day_bias == "neutral": score += 1
            if direction == "long":
                score += 2 if 52<=rsi<=70 else (1 if 45<=rsi<=78 else 0)
            else:
                score += 2 if 30<=rsi<=48 else (1 if 22<=rsi<=55 else 0)
            if vol_ratio >= 2.0: score += 2
            elif vol_ratio >= 1.3: score += 1
            if vwap and ((direction=="long" and price>vwap) or (direction=="short" and price<vwap)): score += 1
            if (direction=="long" and price>ema9) or (direction=="short" and price<ema9): score += 1
            if signal_type == "vwap_reclaim": score += 1
            if datetime.time(9,35) <= t_now <= datetime.time(10,30): score += 2
            score = min(score, 10)

            if score < MIN_SCORE: continue

            ps      = POS_SIZES.get(min(score,10), 500)
            entry_p = apply_slippage(price, direction, slip)
            target  = round(entry_p*(1+GAIN_TARGET_PCT) if direction=="long" else entry_p*(1-GAIN_TARGET_PCT), 4)
            stop    = round(entry_p*(1-STOP_LOSS_PCT)   if direction=="long" else entry_p*(1+STOP_LOSS_PCT),   4)
            shares  = round(ps / entry_p, 4)

            rec = {"date":d_date.strftime("%Y-%m-%d"), "ticker":ticker,
                   "direction":"Long" if direction=="long" else "Short",
                   "entry":entry_p, "exit":None, "result":None,
                   "pnl_pct":0, "pnl_dollar":0,
                   "rsi":round(rsi,1), "vol_ratio":vol_ratio,
                   "signal_type":signal_type, "signal_score":score,
                   "pos_size":ps, "entry_time":ts.strftime("%I:%M %p"),
                   "hour":ts.hour, "day_bias":day_bias}

            in_trade   = True
            traded_today = True
            entry_data = {"dir":direction,"ep":entry_p,"target":target,"stop":stop,
                          "ps":ps,"shares":shares,"trail_active":False,
                          "trail_peak":entry_p,"trail_stop":None,"rec":rec}

        # EOD force close
        if in_trade and entry_data and closes:
            ep  = closes[-1]
            ep  = round(ep*(1-slip if entry_data["dir"]=="long" else 1+slip), 4)
            pct = ((ep-entry_data["ep"])/entry_data["ep"]*100 if entry_data["dir"]=="long"
                   else (entry_data["ep"]-ep)/entry_data["ep"]*100)
            trades.append({**entry_data["rec"],"exit":ep,"result":"Time Exit",
                           "pnl_pct":round(pct,3),"pnl_dollar":round(pct/100*entry_data["ps"],2)})

    return trades


def walk_forward(trades: list, n_periods: int = 4) -> dict:
    if len(trades) < 20: return {}
    dates  = sorted(set(t["date"] for t in trades))
    chunk  = len(dates) // n_periods
    periods = []
    for i in range(n_periods):
        start = dates[i*chunk]
        end   = dates[min((i+1)*chunk-1, len(dates)-1)]
        pts   = [t for t in trades if start <= t["date"] <= end]
        periods.append({"period":i+1,"start":start,"end":end,
                        "trade_count":len(pts),"stats":calc_stats(pts)})
    wrs = [p["stats"]["win_rate"] for p in periods]
    spread = round(max(wrs) - min(wrs), 2) if wrs else 0
    return {"consistent": spread < 20, "wr_spread": spread, "periods": periods}


def main():
    if not POLYGON_API_KEY:
        print("ERROR: Set POLYGON_API_KEY environment variable first.")
        print("  export POLYGON_API_KEY=your_key")
        sys.exit(1)

    print("="*60)
    print("  NYLO Polygon.io 2-Year Backtest")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Data: Polygon.io 1-min OHLCV, 2 years")
    print(f"  Strategy: VWAP Reclaim + EMA Pullback (v18 logic)")
    print("="*60)

    data = {}
    for t in TICKERS:
        data[t] = fetch_2y(t)

    print("\nRunning strategy on 2-year dataset...")
    all_trades = []
    for t in TICKERS:
        if data[t].empty:
            print(f"  {t}: no data"); continue
        trades = run_strategy(data[t], t)
        wins   = sum(1 for x in trades if x["pnl_dollar"] > 0)
        pnl    = round(sum(x["pnl_dollar"] for x in trades), 2)
        print(f"  {t}: {len(trades)} trades | "
              f"{round(wins/len(trades)*100,1) if trades else 0}% WR | ${pnl:+.2f}")
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: (t["date"], t["entry_time"]))
    st = calc_stats(all_trades)
    mc = monte_carlo(all_trades)
    wf = walk_forward(all_trades, n_periods=8)  # 8 quarters over 2 years

    print(f"\n  TOTAL: {st['trades']} trades | {st['win_rate']}% WR | ${st['total_pnl_dollar']:+.2f}")
    print(f"  Sharpe:{st['sharpe']} | PF:{st['profit_factor']} | E:${st['expectancy']:+.2f}/trade")
    print(f"  Max DD: {st['max_drawdown']}%")

    now_et  = datetime.datetime.now(MARKET_TZ)
    end_date= datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)

    out = {
        "generated_at":  now_et.isoformat(),
        "generated_str": now_et.strftime("%B %d, %Y at %I:%M %p ET"),
        "version":       "polygon_v1",
        "data_source":   "Polygon.io",
        "config": {
            "strategy":    "VWAP Reclaim + EMA Pullback",
            "tickers":     TICKERS,
            "date_from":   str(start_date),
            "date_to":     str(end_date),
            "rsi_bull":    f"{RSI_BULL_MIN}/{RSI_BULL_MAX}",
            "rsi_bear":    f"{RSI_BEAR_MIN}/{RSI_BEAR_MAX}",
            "gain_pct":    GAIN_TARGET_PCT * 100,
            "stop_pct":    STOP_LOSS_PCT * 100,
            "min_score":   MIN_SCORE,
        },
        "stats":         st,
        "monte_carlo":   mc,
        "walk_forward":  wf,
        "trades":        all_trades,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {OUT}")

if __name__ == "__main__":
    main()
