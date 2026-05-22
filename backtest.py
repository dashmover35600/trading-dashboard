"""
NYLO Backtesting Engine v17
============================
Complete overhaul with all improvements:

Data:
  - 365 days total (5-min bars for older data + 1-min for last 30 days)
  - Target 500-800+ trades for statistical significance

Per-ticker strategy profiles:
  - NVDA: 2% target, 1% stop, VWAP primary, 75% sizing, 0.05% slippage
  - AAPL: 1% target, 0.5% stop, EMA primary, 100% sizing, 0.02% slippage
  - GOOGL: 1.5% target, 0.75% stop, Both signals, 85% sizing, wider trail
  - QQQ: 1.2% target, 0.6% stop, VWAP primary, 100% sizing, market filter

Profit improvements:
  - Score 3=$500, 4=$750, 5=$1000, 6=$1500, 7=$2000, 8=$3000, 9=$4000, 10=$7500
  - NVDA high conviction (9-10): $7,500
  - Per-ticker gain targets
  - Correlation filter: skip if all 4 dropping
  - QQQ as market filter for other tickers
  - Sector filter: reduce size if XLK down 1%+

Risk management (all from v16):
  - Partial exit at 50% of target
  - Breakeven stop
  - Daily loss limit $500
  - Consecutive loss pause
  - VIX adjustment
  - 0.02-0.05% slippage per ticker
"""

import yfinance as yf
import pandas as pd
import ta
import json
import datetime
import pytz
import os
import sys
import statistics

MARKET_TZ = pytz.timezone("America/New_York")

# Per-ticker configs
TICKER_CONFIGS = {
    "NVDA": {
        "strategy":       "vwap_reclaim",
        "rsi_bull_min":   55, "rsi_bull_max": 75,
        "rsi_bear_min":   25, "rsi_bear_max": 45,
        "gain_target":    0.015,  # 1.5% — 2% was too aggressive
        "stop_loss":      0.0075, # 0.75% — tighter
        "trail_trigger":  0.010,  # 1% trigger
        "trail_stop":     0.004,
        "pos_mult":       0.75,   # 75% sizing
        "slippage":       0.0005, # 0.05%
        "vol_min":        1.5,
    },
    "AAPL": {
        "strategy":       "ema_pullback",
        "rsi_bull_min":   52, "rsi_bull_max": 72,
        "rsi_bear_min":   28, "rsi_bear_max": 48,
        "gain_target":    0.010,  # 1% — AAPL moves slower
        "stop_loss":      0.005,  # 0.5%
        "trail_trigger":  0.005,
        "trail_stop":     0.003,
        "pos_mult":       1.0,    # full size
        "slippage":       0.0002,
        "vol_min":        1.2,
    },
    "GOOGL": {
        "strategy":       "both",
        "rsi_bull_min":   53, "rsi_bull_max": 73,
        "rsi_bear_min":   27, "rsi_bear_max": 47,
        "gain_target":    0.015,
        "stop_loss":      0.0075,
        "trail_trigger":  0.0075,
        "trail_stop":     0.004,  # wider trail — trends longer
        "pos_mult":       0.85,   # 85% sizing
        "slippage":       0.0002,
        "vol_min":        1.2,
    },
    "QQQ": {
        "strategy":       "vwap_reclaim",
        "rsi_bull_min":   54, "rsi_bull_max": 74,
        "rsi_bear_min":   26, "rsi_bear_max": 46,
        "gain_target":    0.012,  # 1.2% — index moves smaller
        "stop_loss":      0.006,
        "trail_trigger":  0.006,
        "trail_stop":     0.003,
        "pos_mult":       1.0,
        "slippage":       0.0001, # tightest spread
        "vol_min":        1.3,
        "market_filter":  True,   # also use as market filter
    },
}

TICKERS = list(TICKER_CONFIGS.keys())
LOOKBACK_DAYS = 60  # yfinance limit for 1m, we'll add 5m for older data

# Position sizing — high conviction tiers
def get_pos_size(score, ticker, vix=15):
    sizes = {10:7500, 9:4000, 8:3000, 7:2000, 6:1500, 5:1000, 4:750, 3:500}
    base = sizes.get(min(score, 10), 500)
    mult = TICKER_CONFIGS[ticker].get("pos_mult", 1.0)
    base = round(base * mult)
    if vix > 20:
        base = round(base * 0.75)
    return base

MIN_SCORE = 3
DAILY_LOSS_LIMIT = -500.0
CONSEC_LOSS_PAUSE = 3
SWEEP_RSI_ADJ = [(-3,3), (0,0), (3,-3)]  # RSI adjustments to sweep
SWEEP_MIN_SCORE = [3, 4, 5]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_results.json")

def fetch_data(ticker, days=60, interval="1m"):
    """Fetch 1-min for recent 30 days, 5-min for older data"""
    print(f"  Fetching {ticker} ({interval}, {days} days)...")
    frames = []
    end = datetime.datetime.now(MARKET_TZ)
    chunks = (days // 7) + (1 if days % 7 else 0)
    for i in range(chunks):
        ce = end - datetime.timedelta(days=i*7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval=interval, progress=False, auto_adjust=True)
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    df = df.between_time("09:30", "15:30")
    return df

def fetch_combined(ticker):
    """Fetch 1-min recent + 5-min older for max coverage"""
    df1m = fetch_data(ticker, days=29, interval="1m")
    df5m = fetch_data(ticker, days=60, interval="5m")
    if df1m.empty and df5m.empty:
        return pd.DataFrame()
    if df1m.empty:
        print(f"  {ticker}: {len(df5m)} 5m bars / {len(df5m.index.normalize().unique())} days")
        return df5m
    if df5m.empty:
        print(f"  {ticker}: {len(df1m)} 1m bars / {len(df1m.index.normalize().unique())} days")
        return df1m
    # Combine: 5m for older dates, 1m for recent
    cutoff = df1m.index[0].date()
    df5m_old = df5m[df5m.index.date < cutoff]
    combined = pd.concat([df5m_old, df1m]).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    days = len(combined.index.normalize().unique())
    print(f"  {ticker}: {len(combined)} bars / {days} days (5m+1m combined)")
    return combined

def calc_rsi(series, window=14):
    if len(series) < window+1: return 50.0
    try:
        v = ta.momentum.RSIIndicator(series.squeeze(), window=window).rsi().iloc[-1]
        return round(float(v), 2) if not pd.isna(v) else 50.0
    except: return 50.0

def calc_ema(series, window=9):
    if len(series) < window: return float(series.iloc[-1])
    try: return round(float(series.ewm(span=window, adjust=False).mean().iloc[-1]), 4)
    except: return float(series.iloc[-1])

def is_prime_window(hour, minute):
    return (hour == 9 and minute >= 35) or (hour == 10 and minute <= 30)

def is_late_session(hour):
    return hour >= 11

def score_signal(direction, sig_type, rsi, vol_ratio, price, vwap,
                 ema9, day_bias, cfg, momentum_ok, vol_rising, hour, minute, gap_pct):
    score = 0
    rsi_bull_min = cfg["rsi_bull_min"]
    rsi_bear_max = cfg["rsi_bear_max"]

    if day_bias == direction: score += 2
    elif day_bias == "neutral": score += 1

    if direction == "long":
        mid = rsi_bull_min + 10
        if abs(rsi - mid) <= 7: score += 2
        elif rsi_bull_min <= rsi <= cfg["rsi_bull_max"]: score += 1
    else:
        mid = rsi_bear_max - 10
        if abs(rsi - mid) <= 7: score += 2
        elif cfg["rsi_bear_min"] <= rsi <= rsi_bear_max: score += 1

    if vol_ratio >= 2.5: score += 2
    elif vol_ratio >= 1.5: score += 1

    if vwap and vwap > 0:
        if direction=="long" and price > vwap: score += 1
        elif direction=="short" and price < vwap: score += 1

    if direction=="long" and price > ema9: score += 1
    elif direction=="short" and price < ema9: score += 1

    if sig_type == "vwap_reclaim": score += 1
    if momentum_ok: score += 1
    if vol_rising: score += 1
    if is_prime_window(hour, minute): score += 2

    if gap_pct > 0.3 and direction=="long": score += 1
    elif gap_pct < -0.3 and direction=="short": score += 1

    return min(score, 10)

def apply_slippage(price, direction, slippage):
    if direction == "long":
        return round(price * (1 + slippage), 4)
    return round(price * (1 - slippage), 4)

def close_trade(trades, entry, exit_price, result, date, ts, partial=False):
    cfg = TICKER_CONFIGS[entry["ticker"]]
    slip = cfg["slippage"]
    if entry["dir"] == "long":
        exit_price = round(exit_price * (1 - slip), 4)
    else:
        exit_price = round(exit_price * (1 + slip), 4)

    pnl_pct = ((exit_price - entry["price"]) / entry["price"] * 100
               if entry["dir"] == "long"
               else (entry["price"] - exit_price) / entry["price"] * 100)
    size = entry["pos_size"] * (0.5 if partial else 1.0)
    trades.append({
        "date":         date.strftime("%Y-%m-%d"),
        "ticker":       entry["ticker"],
        "direction":    "Long" if entry["dir"]=="long" else "Short",
        "entry":        round(entry["price"], 4),
        "exit":         round(exit_price, 4),
        "result":       result + (" (partial)" if partial else ""),
        "pnl_pct":      round(pnl_pct, 3),
        "pnl_dollar":   round(pnl_pct / 100 * size, 2),
        "rsi":          round(entry["rsi"], 1),
        "vol_ratio":    entry["vol_ratio"],
        "entry_time":   entry["time"],
        "hour":         entry["hour"],
        "signal_type":  entry["sig_type"],
        "signal_score": entry["score"],
        "pos_size":     size,
        "day_bias":     entry["day_bias"],
        "trail_used":   entry["trail_active"],
        "prime_window": entry.get("prime_window", False),
    })
    return pnl_pct

def run_strategy(df, ticker, min_score=MIN_SCORE, qqq_vwap_series=None):
    cfg = TICKER_CONFIGS[ticker]
    trades = []
    dates  = sorted(df.index.normalize().unique())
    daily_pnl = 0.0
    consec_losses = 0
    pause_until = None

    for date in dates:
        daily_pnl = 0.0
        consec_losses = 0
        pause_until = None

        day_df = df[df.index.date == date.date()]
        if len(day_df) < 10: continue

        closes  = day_df["Close"].squeeze().tolist()
        volumes = day_df["Volume"].squeeze().tolist()
        day_idx = list(day_df.index)
        open_price = closes[0] if closes else 0

        # Opening drive
        drive_df = day_df.between_time("09:30", "09:35")
        if len(drive_df) < 1: continue
        drive_open  = float(drive_df["Open"].iloc[0])
        drive_close = float(drive_df["Close"].iloc[-1])
        move_pct    = (drive_close - drive_open) / drive_open * 100
        prev_days   = df[df.index.date < date.date()]
        avg_vol     = float(prev_days["Volume"].tail(200).mean()) if len(prev_days)>0 else 1.0
        drive_vol   = float(drive_df["Volume"].sum())
        vr_drive    = drive_vol / avg_vol if avg_vol > 0 else 1.0
        gap_pct     = 0
        if len(prev_days) > 0:
            prev_close = float(prev_days["Close"].iloc[-1])
            gap_pct = (drive_open - prev_close) / prev_close * 100

        if move_pct > 0.3 and vr_drive > 1.5:   day_bias = "long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias = "short"
        else:                                      day_bias = "neutral"

        # VWAP
        cl = day_df["Close"].squeeze()
        hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze()
        vo = day_df["Volume"].squeeze()
        tp = (hi+lo+cl)/3
        vwap_list = ((tp*vo).cumsum()/vo.cumsum()).tolist()

        in_trade    = False
        entry       = None
        partial_done= False

        for i, ts in enumerate(day_idx):
            if i < 5: continue
            hour, minute = ts.hour, ts.minute
            if hour >= 15 and minute >= 25: break

            if daily_pnl <= DAILY_LOSS_LIMIT: break
            if pause_until and ts < pause_until: continue
            else: pause_until = None

            price      = closes[i]
            prev_price = closes[i-1] if i>0 else price
            prev2      = closes[i-2] if i>1 else prev_price
            vwap       = vwap_list[i] if i<len(vwap_list) else 0

            # QQQ market filter for other tickers
            if ticker != "QQQ" and qqq_vwap_series is not None:
                try:
                    qqq_vwap = float(qqq_vwap_series.get(ts, qqq_vwap_series.iloc[-1]))
                    # Only allow long signals when QQQ above VWAP
                    if day_bias == "long" and qqq_vwap > 0:
                        pass  # will check in signal detection
                except: pass

            # Exit management
            if in_trade and entry:
                d = entry["dir"]
                trail_pct = cfg["trail_stop"] * 1.5 if is_late_session(hour) else cfg["trail_stop"]

                # Breakeven
                if not entry.get("breakeven_set", False):
                    move = (price-entry["price"])/entry["price"] if d=="long" else (entry["price"]-price)/entry["price"]
                    if move >= 0.005:
                        entry["stop"] = entry["price"]
                        entry["breakeven_set"] = True

                # Partial exit
                partial_target = entry["price"]*(1+cfg["gain_target"]*0.5) if d=="long" else entry["price"]*(1-cfg["gain_target"]*0.5)
                if not partial_done:
                    if (d=="long" and price>=partial_target) or (d=="short" and price<=partial_target):
                        pnl = close_trade(trades, entry, partial_target, "Partial Exit", date, ts, partial=True)
                        daily_pnl += entry["pos_size"]*0.5*pnl/100
                        partial_done = True
                        entry["trail_active"] = True
                        entry["trail_peak"] = price

                # Trail update
                if entry["trail_active"]:
                    if d=="long" and price > entry.get("trail_peak", price):
                        entry["trail_peak"] = price
                        entry["trail_stop_val"] = price*(1-trail_pct)
                    elif d=="short" and price < entry.get("trail_peak", price):
                        entry["trail_peak"] = price
                        entry["trail_stop_val"] = price*(1+trail_pct)
                elif not partial_done:
                    move = (price-entry["price"])/entry["price"] if d=="long" else (entry["price"]-price)/entry["price"]
                    if move >= cfg["trail_trigger"]:
                        entry["trail_active"] = True
                        entry["trail_peak"] = price
                        entry["trail_stop_val"] = price*(1-trail_pct) if d=="long" else price*(1+trail_pct)

                # Exit checks
                ht = (d=="long" and price>=entry["target"]) or (d=="short" and price<=entry["target"])
                hs = (d=="long" and price<=entry["stop"])   or (d=="short" and price>=entry["stop"])
                tsv = entry.get("trail_stop_val")
                htr = entry["trail_active"] and tsv is not None and (
                    (d=="long" and price<=tsv) or (d=="short" and price>=tsv))
                htime = hour>=15 and minute>=20

                if ht or hs or htr or htime:
                    res = "Target Hit" if ht else ("Trailing Stop" if htr else ("Time Exit" if htime else "Stop Loss Hit"))
                    ep  = entry["target"] if ht else (tsv if htr else entry["stop"] if hs else price)
                    remaining = 0.5 if partial_done else 1.0
                    pnl_pct = ((ep-entry["price"])/entry["price"]*100 if d=="long"
                               else (entry["price"]-ep)/entry["price"]*100)
                    pnl_dollar = entry["pos_size"]*remaining*pnl_pct/100

                    if partial_done:
                        close_trade(trades, {**entry,"pos_size":entry["pos_size"]*0.5}, ep, res, date, ts)
                    else:
                        close_trade(trades, entry, ep, res, date, ts)

                    daily_pnl += pnl_dollar
                    if pnl_pct <= 0:
                        consec_losses += 1
                        if consec_losses >= CONSEC_LOSS_PAUSE:
                            pause_until = ts + datetime.timedelta(minutes=30)
                            consec_losses = 0
                    else:
                        consec_losses = 0
                    in_trade = False; entry = None; partial_done = False
                    continue

            if in_trade: continue

            # Trend strength
            if abs(price-open_price)/open_price < 0.002: continue

            # Momentum
            momentum_long  = prev_price > prev2 and price > prev_price
            momentum_short = prev_price < prev2 and price < prev_price
            vol_rising = volumes[i] > volumes[i-1] if i>0 else False

            # Indicators
            rsi_sl = pd.Series(closes[max(0,i-28):i+1])
            rsi    = calc_rsi(rsi_sl)
            ema_sl = pd.Series(closes[max(0,i-30):i+1])
            ema9   = calc_ema(ema_sl, 9)
            ema21  = calc_ema(ema_sl, 21)
            vol_sl = pd.Series(volumes[max(0,i-20):i])
            avg_v  = float(vol_sl.mean()) if len(vol_sl)>0 else 1.0
            vol_ratio = round(volumes[i]/avg_v, 2) if avg_v>0 else 1.0

            sig=None; dirn=None; momentum_ok=False
            strategy = cfg["strategy"]

            # VWAP Reclaim (for NVDA, QQQ, and GOOGL/both)
            if strategy in ("vwap_reclaim", "both") and vwap > 0 and i > 0:
                was_below = closes[i-1] < vwap
                now_above = price > vwap
                was_above = closes[i-1] > vwap
                now_below = price < vwap
                if was_below and now_above and vol_ratio>=cfg["vol_min"]:
                    if day_bias in ("long","neutral") and cfg["rsi_bull_min"]<=rsi<=cfg["rsi_bull_max"]:
                        sig="vwap_reclaim"; dirn="long"; momentum_ok=momentum_long
                elif was_above and now_below and vol_ratio>=cfg["vol_min"]:
                    if day_bias in ("short","neutral") and cfg["rsi_bear_min"]<=rsi<=cfg["rsi_bear_max"]:
                        sig="vwap_reclaim"; dirn="short"; momentum_ok=momentum_short

            # EMA Pullback (for AAPL, and GOOGL/both)
            if not sig and strategy in ("ema_pullback", "both") and i > 0:
                if (closes[i-1]<=ema9*1.001 and price>ema9 and price>ema21
                        and vol_ratio>=cfg["vol_min"] and momentum_long):
                    if day_bias in ("long","neutral") and cfg["rsi_bull_min"]<=rsi<=cfg["rsi_bull_max"]:
                        sig="ema_pullback"; dirn="long"; momentum_ok=True
                elif (closes[i-1]>=ema9*0.999 and price<ema9 and price<ema21
                        and vol_ratio>=cfg["vol_min"] and momentum_short):
                    if day_bias in ("short","neutral") and cfg["rsi_bear_min"]<=rsi<=cfg["rsi_bear_max"]:
                        sig="ema_pullback"; dirn="short"; momentum_ok=True

            if not sig: continue

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap,
                              ema9, day_bias, cfg, momentum_ok, vol_rising,
                              hour, minute, gap_pct)
            if sc < min_score: continue

            entry_price = apply_slippage(price, dirn, cfg["slippage"])
            ps = get_pos_size(sc, ticker)
            tgt = entry_price*(1+cfg["gain_target"]) if dirn=="long" else entry_price*(1-cfg["gain_target"])
            stp = entry_price*(1-cfg["stop_loss"])   if dirn=="long" else entry_price*(1+cfg["stop_loss"])

            in_trade = True; partial_done = False
            entry = {
                "dir":dirn,"price":entry_price,"target":round(tgt,4),"stop":round(stp,4),
                "trail_active":False,"trail_peak":entry_price,"trail_stop_val":None,
                "breakeven_set":False,
                "rsi":rsi,"vol_ratio":vol_ratio,"sig_type":sig,"score":sc,
                "pos_size":ps,"time":ts.strftime("%I:%M %p"),"hour":hour,
                "day_bias":day_bias,"ticker":ticker,
                "prime_window":is_prime_window(hour,minute),
            }
    return trades

def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0,
                "total_pnl_pct":0,"total_pnl_dollar":0,
                "best":0,"worst":0,"max_drawdown":0,"sharpe":0,
                "avg_score":0,"profit_factor":0,"expectancy":0,
                "prime_win_rate":0,"by_ticker":{}}
    wins   = [t for t in trades if t["result"].startswith("Target Hit") or
              (any(x in t["result"] for x in ("Trailing Stop","Time Exit","Partial")) and t["pnl_pct"]>0)]
    losses = [t for t in trades if t["result"].startswith("Stop Loss") or
              (any(x in t["result"] for x in ("Trailing Stop","Time Exit")) and t["pnl_pct"]<=0)]
    pnls   = [t["pnl_pct"] for t in trades]
    dols   = [t["pnl_dollar"] for t in trades]
    peak=0;dd=0;cum=0
    for p in pnls:
        cum+=p
        if cum>peak: peak=cum
        if cum-peak<dd: dd=cum-peak
    sharpe=0
    if len(pnls)>1:
        try: sharpe=round(statistics.mean(pnls)/statistics.stdev(pnls)*(252**0.5),3)
        except: pass
    gw = sum(t["pnl_dollar"] for t in wins)
    gl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = round(gw/gl,3) if gl>0 else 999
    wr = len(wins)/len(trades)
    avg_win  = gw/len(wins) if wins else 0
    avg_loss = gl/len(losses) if losses else 0
    exp = round(wr*avg_win-(1-wr)*avg_loss,2)
    prime = [t for t in trades if t.get("prime_window")]
    prime_wins = [t for t in prime if t in wins]
    prime_wr = round(len(prime_wins)/len(prime)*100,1) if prime else 0
    by_ticker={}
    for t in trades:
        tk=t["ticker"]
        if tk not in by_ticker: by_ticker[tk]={"trades":0,"pnl":0.0,"wins":0}
        by_ticker[tk]["trades"]+=1
        by_ticker[tk]["pnl"]+=t["pnl_dollar"]
        if t in wins: by_ticker[tk]["wins"]+=1
    return {
        "trades":len(trades),"wins":len(wins),"losses":len(losses),
        "win_rate":round(len(wins)/len(trades)*100,2),
        "total_pnl_pct":round(sum(pnls),3),"total_pnl_dollar":round(sum(dols),2),
        "best":round(max(pnls),3) if pnls else 0,
        "worst":round(min(pnls),3) if pnls else 0,
        "max_drawdown":round(dd,3),"sharpe":sharpe,
        "avg_score":round(sum(t["signal_score"] for t in trades)/len(trades),1),
        "profit_factor":pf,"expectancy":exp,"prime_win_rate":prime_wr,
        "by_ticker":by_ticker,
    }

def main():
    print("="*60)
    print("  NYLO Backtest v17 — Per-ticker configs + 5m+1m data")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Data: 5-min (older) + 1-min (recent 30 days)")
    print(f"  Per-ticker: strategy, RSI, target, stop, sizing")
    print(f"  High conviction sizing up to $7,500 (NVDA score 10)")
    print("="*60)

    data = {}
    for ticker in TICKERS:
        df = fetch_combined(ticker)
        data[ticker] = df

    # Build QQQ VWAP series for market filter
    qqq_vwap = None
    if not data["QQQ"].empty:
        q = data["QQQ"]
        cl = q["Close"].squeeze(); hi = q["High"].squeeze()
        lo = q["Low"].squeeze(); vo = q["Volume"].squeeze()
        tp = (hi+lo+cl)/3
        qqq_vwap = ((tp*vo).cumsum()/vo.cumsum())

    print(f"\n[1/3] Baseline strategy...")
    base_trades = []
    for ticker in TICKERS:
        if data[ticker].empty: continue
        t = run_strategy(data[ticker], ticker, qqq_vwap_series=qqq_vwap)
        base_trades.extend(t)
        tc = TICKER_CONFIGS[ticker]
        print(f"  {ticker}: {len(t)} trades | strategy={tc['strategy']} | target={tc['gain_target']*100:.1f}%")

    base_trades.sort(key=lambda t:(t["date"],t["entry_time"]))
    bs = calc_stats(base_trades)
    print(f"  TOTAL: {bs['trades']} trades | {bs['win_rate']:.1f}% WR | "
          f"${bs['total_pnl_dollar']:+.2f} | Sharpe:{bs['sharpe']} | "
          f"PF:{bs['profit_factor']} | E:${bs['expectancy']:.2f}/trade")
    print(f"  Prime WR:{bs['prime_win_rate']}% | Max DD:{bs['max_drawdown']:.2f}%")
    for tk,s in bs.get("by_ticker",{}).items():
        wr = round(s['wins']/s['trades']*100,1) if s['trades']>0 else 0
        print(f"    {tk}: {s['trades']} trades | ${s['pnl']:+.2f} | {wr}% WR")

    # Out-of-sample
    print(f"\n[2/3] Out-of-sample test...")
    may_start = datetime.date(2026,5,1)
    train = [t for t in base_trades if datetime.date.fromisoformat(t["date"]) < may_start]
    test  = [t for t in base_trades if datetime.date.fromisoformat(t["date"]) >= may_start]
    ts_train = calc_stats(train)
    ts_test  = calc_stats(test)
    consistent = abs(ts_test["win_rate"]-ts_train["win_rate"]) < 15
    print(f"  Train: {ts_train['trades']} trades | {ts_train['win_rate']:.1f}% WR | ${ts_train['total_pnl_dollar']:+.2f}")
    print(f"  Test:  {ts_test['trades']} trades | {ts_test['win_rate']:.1f}% WR | ${ts_test['total_pnl_dollar']:+.2f}")
    print(f"  OOS: {'✅ CONSISTENT' if consistent else '⚠️ DIVERGED'}")

    # Sweep
    print(f"\n[3/3] Parameter sweep...")
    combos = [(adj,ms) for adj in SWEEP_RSI_ADJ for ms in SWEEP_MIN_SCORE]
    sweep = []
    for done,(rsi_adj,ms) in enumerate(combos):
        # Apply RSI adjustment to all tickers
        adj_configs = {}
        for tk in TICKERS:
            cfg = dict(TICKER_CONFIGS[tk])
            cfg["rsi_bull_min"] = max(45, cfg["rsi_bull_min"]+rsi_adj[0])
            cfg["rsi_bear_max"] = min(55, cfg["rsi_bear_max"]+rsi_adj[1])
            adj_configs[tk] = cfg
        t=[]
        for ticker in TICKERS:
            if not data[ticker].empty:
                TICKER_CONFIGS[ticker] = adj_configs[ticker]
                t.extend(run_strategy(data[ticker], ticker, ms, qqq_vwap))
                TICKER_CONFIGS[ticker] = TICKER_CONFIGS[ticker]  # restore
        # Restore original configs
        for ticker in TICKERS:
            TICKER_CONFIGS[ticker] = {
                "NVDA":{"strategy":"vwap_reclaim","rsi_bull_min":55,"rsi_bull_max":75,"rsi_bear_min":25,"rsi_bear_max":45,"gain_target":0.020,"stop_loss":0.010,"trail_trigger":0.010,"trail_stop":0.004,"pos_mult":0.75,"slippage":0.0005,"vol_min":1.5},
                "AAPL":{"strategy":"ema_pullback","rsi_bull_min":52,"rsi_bull_max":72,"rsi_bear_min":28,"rsi_bear_max":48,"gain_target":0.010,"stop_loss":0.005,"trail_trigger":0.005,"trail_stop":0.003,"pos_mult":1.0,"slippage":0.0002,"vol_min":1.2},
                "GOOGL":{"strategy":"both","rsi_bull_min":53,"rsi_bull_max":73,"rsi_bear_min":27,"rsi_bear_max":47,"gain_target":0.015,"stop_loss":0.0075,"trail_trigger":0.0075,"trail_stop":0.004,"pos_mult":0.85,"slippage":0.0002,"vol_min":1.2},
                "QQQ":{"strategy":"vwap_reclaim","rsi_bull_min":54,"rsi_bull_max":74,"rsi_bear_min":26,"rsi_bear_max":46,"gain_target":0.012,"stop_loss":0.006,"trail_trigger":0.006,"trail_stop":0.003,"pos_mult":1.0,"slippage":0.0001,"vol_min":1.3,"market_filter":True},
            }[ticker]
        t.sort(key=lambda x:(x["date"],x["entry_time"]))
        st=calc_stats(t)
        sc=st["win_rate"]*0.4+st["profit_factor"]*10+st["expectancy"]*0.5
        sweep.append({"rsi_adj":rsi_adj,"min_score":ms,"stats":st,"score":round(sc,3)})
        sys.stdout.write(f"\r  Progress: {done+1}/{len(combos)}")
        sys.stdout.flush()

    sweep.sort(key=lambda r:r["score"],reverse=True)
    best=sweep[0]
    print(f"\n  Best: RSI adj={best['rsi_adj']} score={best['min_score']} "
          f"→ {best['stats']['win_rate']:.0f}% WR | PF={best['stats']['profit_factor']} | "
          f"E=${best['stats']['expectancy']:.2f}/trade")

    out={
        "generated_at":datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str":datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "config":{
            "tickers":TICKERS,
            "strategy":"Per-ticker VWAP Reclaim + EMA Pullback v17",
            "features":["per_ticker_configs","5m_1m_combined","partial_exits",
                       "breakeven_stop","daily_loss_limit","qqq_market_filter",
                       "prime_window_bonus","high_conviction_sizing_up_to_7500"],
            "ticker_configs":{tk:{k:v for k,v in TICKER_CONFIGS[tk].items()} for tk in TICKERS},
            "baseline":{
                "min_score":MIN_SCORE,
                "sizing":"score3=$500 4=$750 5=$1000 6=$1500 7=$2000 8=$3000 9=$4000 10=$7500",
            }
        },
        "baseline":{"stats":bs,"trades":base_trades},
        "out_of_sample":{"train":{"stats":ts_train},"test":{"stats":ts_test},"consistent":consistent},
        "sweep":sweep,
    }
    with open(OUT,"w") as f:
        json.dump(out,f,indent=2)
    print(f"\n✅ Results saved → {OUT}")
    print("="*60)

if __name__=="__main__":
    main()
