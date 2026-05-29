"""
NYLO Backtesting Engine v20 — Full-Year Hybrid Backtest
=========================================================
v20 changes vs v19:
1. Hybrid data fetch: 1h bars for months 3-12 (up to 365 days),
   5m+1m bars for recent 60 days — combined into one dataset
2. Pre-computed cross-bar indicators (RSI/EMA9/EMA21/ATR on full series)
   so hourly bars get proper lookback, not just within-day bars
3. Bar-size adaptive per-day logic: thresholds scale with resolution
   (hourly: min 3 bars, skip 1; minute: min 10 bars, skip 5)
4. Vol-ratio computed per-resolution: hourly uses within-day avg,
   minute uses 20-bar rolling within day
5. OOS split updated to Nov 2025 for ~50/50 split over 1-year window
6. Target: 200-400+ trades across ~300 trading days

Same filters: QQQ 20 EMA, earnings blackout, correlation filter
Same strategy: VWAP Reclaim + EMA Pullback, AAPL + GOOGL, 12 PM cutoff
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import json
import datetime
import pytz
import os
import sys
import statistics
import random

MARKET_TZ = pytz.timezone("America/New_York")

TICKERS = ["AAPL", "GOOGL"]
MARKET_FILTER_TICKER = "QQQ"

TICKER_CONFIGS = {
    "AAPL": {
        "strategy":      "ema_pullback",
        "rsi_bull_min":  52, "rsi_bull_max": 72,
        "rsi_bear_min":  28, "rsi_bear_max": 48,
        "atr_stop_mult": 1.0,
        "atr_target_mult": 1.5,
        "pos_mult":      1.0,
        "slippage":      0.0003,
        "vol_min":       1.2,
        "min_atr_pct":   0.001,
        "gain_target":   0.020,   # 2.0%
        "stop_loss":     0.005,   # 0.50%
    },
    "GOOGL": {
        "strategy":      "both",
        "rsi_bull_min":  53, "rsi_bull_max": 73,
        "rsi_bear_min":  27, "rsi_bear_max": 47,
        "atr_stop_mult": 1.0,
        "atr_target_mult": 1.5,
        "pos_mult":      0.85,
        "slippage":      0.00025,
        "vol_min":       1.2,
        "min_atr_pct":   0.001,
        "gain_target":   0.020,   # 2.0%
        "stop_loss":     0.005,   # 0.50%
    },
}

# ── Earnings blackout ──────────────────────────────────────────────────────────
# Known earnings dates; blackout = 2 days before + 2 days after each date
EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_DATES = {
    "AAPL": [
        datetime.date(2025, 5,  1),
        datetime.date(2025, 7, 31),
        datetime.date(2025, 10, 30),
        datetime.date(2026, 1, 30),
        datetime.date(2026, 5,  1),
    ],
    "GOOGL": [
        datetime.date(2025, 4, 29),
        datetime.date(2025, 7, 29),
        datetime.date(2025, 10, 29),
        datetime.date(2026, 2,  4),
        datetime.date(2026, 4, 29),
    ],
}

def is_earnings_blackout(date, ticker):
    d = date.date() if hasattr(date, "date") else date
    for ed in EARNINGS_DATES.get(ticker, []):
        if abs((d - ed).days) <= EARNINGS_BLACKOUT_DAYS:
            return True
    return False

# ── Position sizing ────────────────────────────────────────────────────────────
def get_pos_size(score, ticker, vix=15):
    sizes = {10:5000, 9:4000, 8:3000, 7:2000, 6:1500, 5:1000, 4:750, 3:500}
    base = sizes.get(min(score, 10), 500)
    base = round(base * TICKER_CONFIGS[ticker].get("pos_mult", 1.0))
    if vix > 20: base = round(base * 0.75)
    return base

MIN_SCORE         = 3
DAILY_LOSS_LIMIT  = -500.0
CONSEC_LOSS_PAUSE = 3
TRAIL_TRIGGER     = 0.010
TRAIL_STOP_PCT    = 0.005
BREAKEVEN_TRIGGER = 0.005
PARTIAL_PCT       = 0.010    # fixed +1.0% partial exit

TRADE_START = datetime.time(9, 30)
TRADE_END   = datetime.time(10, 30)

SWEEP_MIN_SCORE  = [3, 4, 5]
SWEEP_ATR_STOP   = [0.8, 1.0, 1.2]
SWEEP_ATR_TARGET = [1.2, 1.5, 2.0]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_results.json")

# ── Data fetching ──────────────────────────────────────────────────────────────
def _process_frames(frames):
    """Concat, dedup, timezone-convert, market-hours filter."""
    if not frames: return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    return df.between_time("09:30", "15:30")

def fetch_hybrid(ticker):
    """
    Hybrid fetch for ~365 days of data:
      - 1h bars  : months 3-12 (Yahoo allows 2-year 1h history)
      - 5m bars  : recent ~60 days
      - 1m bars  : recent ~30 days
    Combine: 1h for old days, 5m/1m for recent days.
    """
    print(f"  Fetching {ticker} (1y 1h + 60d 5m + 30d 1m)...")
    end      = datetime.datetime.now(MARKET_TZ)
    start_1y = end - datetime.timedelta(days=370)

    # 1h bars — full year
    df_1h = pd.DataFrame()
    try:
        raw = yf.download(ticker,
                          start=start_1y.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          interval="1h", progress=False, auto_adjust=True)
        df_1h = _process_frames([raw]) if not raw.empty else pd.DataFrame()
    except: pass

    # 1m bars — last ~35 days (5 × 7-day chunks)
    frames_1m = []
    for i in range(5):
        ce = end - datetime.timedelta(days=i*7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="1m", progress=False, auto_adjust=True)
            if not df.empty: frames_1m.append(df)
        except: pass
    df_1m = _process_frames(frames_1m)

    # 5m bars — last ~63 days (9 × 7-day chunks)
    frames_5m = []
    for i in range(9):
        ce = end - datetime.timedelta(days=i*7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="5m", progress=False, auto_adjust=True)
            if not df.empty: frames_5m.append(df)
        except: pass
    df_5m = _process_frames(frames_5m)

    # Build recent dataset: 1m takes priority, then 5m fills gaps
    if df_1m.empty and df_5m.empty:
        combined = df_1h
    else:
        if not df_1m.empty:
            recent_cut = df_1m.index[0].date()
            if not df_5m.empty:
                df_5m_mid = df_5m[df_5m.index.date < recent_cut]
                recent = _process_frames([df_5m_mid, df_1m])
            else:
                recent = df_1m
        else:
            recent = df_5m

        recent_cut = recent.index[0].date()
        if not df_1h.empty:
            df_1h_old = df_1h[df_1h.index.date < recent_cut]
            combined  = pd.concat([df_1h_old, recent]).sort_index()
            combined  = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = recent

    if combined.empty: return pd.DataFrame()
    days       = len(combined.index.normalize().unique())
    dates_u    = sorted(combined.index.normalize().unique())
    hourly_d   = sum(1 for d in dates_u
                     if len(combined[combined.index.date == d.date()]) < 15)
    minute_d   = days - hourly_d
    print(f"  {ticker}: {len(combined)} bars / {days} days "
          f"({hourly_d}h-res + {minute_d}m-res days)")
    return combined

# ── Indicators ─────────────────────────────────────────────────────────────────
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

def calc_atr(highs, lows, closes, window=14):
    if len(closes) < window+1: return closes[-1] * 0.01
    try:
        hi  = pd.Series(highs[-window-1:])
        lo  = pd.Series(lows[-window-1:])
        cl  = pd.Series(closes[-window-1:])
        atr = ta.volatility.AverageTrueRange(hi, lo, cl, window=window).average_true_range().iloc[-1]
        return float(atr) if not pd.isna(atr) else closes[-1]*0.01
    except: return closes[-1]*0.01

# ── Signal scoring ─────────────────────────────────────────────────────────────
def score_signal(direction, sig_type, rsi, vol_ratio, price, vwap,
                 ema9, day_bias, cfg, momentum_ok, vol_rising, hour, minute, gap_pct):
    score = 0
    if day_bias == direction:   score += 2
    elif day_bias == "neutral": score += 1
    if direction == "long":
        mid = cfg["rsi_bull_min"] + 10
        if abs(rsi-mid) <= 7:                              score += 2
        elif cfg["rsi_bull_min"] <= rsi <= cfg["rsi_bull_max"]: score += 1
    else:
        mid = cfg["rsi_bear_max"] - 10
        if abs(rsi-mid) <= 7:                              score += 2
        elif cfg["rsi_bear_min"] <= rsi <= cfg["rsi_bear_max"]: score += 1
    if vol_ratio >= 2.5:   score += 2
    elif vol_ratio >= 1.5: score += 1
    if vwap and vwap > 0:
        if direction=="long"  and price > vwap: score += 1
        elif direction=="short" and price < vwap: score += 1
    if direction=="long"  and price > ema9: score += 1
    elif direction=="short" and price < ema9: score += 1
    if sig_type == "vwap_reclaim": score += 1
    if momentum_ok:  score += 1
    if vol_rising:   score += 1
    if (hour==9 and minute>=35) or (hour==10 and minute<=30): score += 2
    if gap_pct >  0.3 and direction=="long":  score += 1
    elif gap_pct < -0.3 and direction=="short": score += 1
    return min(score, 10)

def apply_slippage(price, direction, slippage):
    if direction == "long": return round(price*(1+slippage), 4)
    return round(price*(1-slippage), 4)

def close_trade(trades, entry, exit_price, result, date, ts, partial=False):
    cfg  = TICKER_CONFIGS[entry["ticker"]]
    slip = cfg["slippage"]
    exit_price = round(exit_price*(1-slip if entry["dir"]=="long" else 1+slip), 4)
    pnl_pct = ((exit_price-entry["price"])/entry["price"]*100
               if entry["dir"]=="long"
               else (entry["price"]-exit_price)/entry["price"]*100)
    size = entry["pos_size"] * (0.5 if partial else 1.0)
    trades.append({
        "date":         date.strftime("%Y-%m-%d"),
        "ticker":       entry["ticker"],
        "direction":    "Long" if entry["dir"]=="long" else "Short",
        "entry":        round(entry["price"], 4),
        "exit":         round(exit_price, 4),
        "result":       result + (" (partial)" if partial else ""),
        "pnl_pct":      round(pnl_pct, 3),
        "pnl_dollar":   round(pnl_pct/100*size, 2),
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
        "atr_at_entry": entry.get("atr", 0),
        "earnings_skip":False,
        "corr_skip":    False,
    })
    return pnl_pct

# ── Core strategy ──────────────────────────────────────────────────────────────
def run_strategy(df, ticker, qqq_close=None, qqq_ema20=None,
                 corr_skip_dates=None, min_score=MIN_SCORE,
                 atr_stop_mult=None, atr_target_mult=None):
    cfg    = TICKER_CONFIGS[ticker]
    atr_sm = atr_stop_mult   or cfg["atr_stop_mult"]
    atr_tm = atr_target_mult or cfg["atr_target_mult"]
    trades = []
    dates  = sorted(df.index.normalize().unique())
    corr_skip_dates = corr_skip_dates or set()

    # Pre-compute indicators on the full series so hourly bars get proper lookback
    full_cl = df["Close"].squeeze()
    full_hi = df["High"].squeeze()
    full_lo = df["Low"].squeeze()
    pre_rsi  = ta.momentum.RSIIndicator(full_cl, window=14).rsi()
    pre_ema9 = full_cl.ewm(span=9,  adjust=False).mean()
    pre_ema21= full_cl.ewm(span=21, adjust=False).mean()
    pre_atr  = ta.volatility.AverageTrueRange(
                   full_hi, full_lo, full_cl, window=14).average_true_range()

    def lookup(series, ts, default):
        try:
            v = series.asof(ts)
            return default if pd.isna(v) else float(v)
        except: return default

    for date in dates:
        daily_pnl = 0.0; consec_losses = 0; pause_until = None
        d_date = date.date()

        if is_earnings_blackout(d_date, ticker): continue
        if d_date in corr_skip_dates:            continue

        day_df = df[df.index.date == d_date]
        n_bars = len(day_df)

        # Bar-size detection: hourly sessions have ~6 bars, minute sessions have 200+
        hourly    = n_bars < 15
        min_bars  = 3  if hourly else 10
        skip_bars = 1  if hourly else 5
        if n_bars < min_bars: continue

        closes  = day_df["Close"].squeeze().tolist()
        volumes = day_df["Volume"].squeeze().tolist()
        highs   = day_df["High"].squeeze().tolist()
        lows    = day_df["Low"].squeeze().tolist()
        day_idx = list(day_df.index)
        open_price = closes[0] if closes else 0

        # Opening drive — between_time("09:30","09:35") captures the 9:30 bar
        # for both 1m (first 5 bars) and 1h (the 9:30 hourly bar)
        drive_df = day_df.between_time("09:30", "09:35")
        if len(drive_df) < 1: continue
        drive_open  = float(drive_df["Open"].iloc[0])
        drive_close = float(drive_df["Close"].iloc[-1])
        move_pct    = (drive_close - drive_open) / drive_open * 100
        prev_days   = df[df.index.date < d_date]
        # avg_vol for vr_drive: use recent prev bars (same resolution dominates tail)
        avg_vol = float(prev_days["Volume"].tail(200).mean()) if not prev_days.empty else 1.0
        drive_vol = float(drive_df["Volume"].sum())
        vr_drive  = drive_vol / avg_vol if avg_vol > 0 else 1.0
        gap_pct   = 0
        if not prev_days.empty:
            prev_close = float(prev_days["Close"].squeeze().iloc[-1])
            gap_pct    = (drive_open - prev_close) / prev_close * 100

        if   move_pct >  0.3 and vr_drive > 1.5: day_bias = "long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias = "short"
        else:                                      day_bias = "neutral"

        # VWAP
        cl = day_df["Close"].squeeze(); hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze();   vo = day_df["Volume"].squeeze()
        tp = (hi + lo + cl) / 3
        vwap_list = ((tp * vo).cumsum() / vo.cumsum()).tolist()

        # Within-day avg volume for vol_ratio
        # Hourly: use morning-only avg (9:30-12:00 bars) to avoid afternoon inflation
        if hourly:
            morn_df = day_df.between_time("09:30", "12:00")
            day_avg_vol = float(morn_df["Volume"].mean()) if not morn_df.empty else float(day_df["Volume"].mean())
        else:
            day_avg_vol = None  # minute bars use rolling within-bar calc below

        in_trade = False; entry = None; partial_done = False

        for i, ts in enumerate(day_idx):
            if i < skip_bars: continue
            hour, minute = ts.hour, ts.minute

            t_now = ts.time()
            if t_now < TRADE_START: continue
            if t_now > TRADE_END and not in_trade: continue
            if hour >= 15 and minute >= 25: break

            if daily_pnl <= DAILY_LOSS_LIMIT: break
            if pause_until and ts < pause_until: continue
            else: pause_until = None

            price      = closes[i]
            prev_price = closes[i-1] if i > 0 else price
            prev2      = closes[i-2] if i > 1 else prev_price
            vwap       = vwap_list[i] if i < len(vwap_list) else 0

            # ── Exit management ──────────────────────────────────────────────
            if in_trade and entry:
                d = entry["dir"]
                trail_pct = TRAIL_STOP_PCT

                if not entry.get("breakeven_set", False):
                    move = ((price - entry["price"]) / entry["price"] if d == "long"
                            else (entry["price"] - price) / entry["price"])
                    if move >= BREAKEVEN_TRIGGER:
                        entry["stop"] = entry["price"]
                        entry["breakeven_set"] = True

                partial_tgt = (entry["price"] * (1 + PARTIAL_PCT)
                               if d == "long"
                               else entry["price"] * (1 - PARTIAL_PCT))
                if not partial_done:
                    if (d == "long" and price >= partial_tgt) or (d == "short" and price <= partial_tgt):
                        pnl = close_trade(trades, entry, partial_tgt, "Partial Exit", date, ts, partial=True)
                        daily_pnl += entry["pos_size"] * 0.5 * pnl / 100
                        partial_done = True
                        entry["trail_active"] = True
                        entry["trail_peak"]   = price

                if entry["trail_active"]:
                    if d == "long" and price > entry.get("trail_peak", price):
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = price * (1 - trail_pct)
                    elif d == "short" and price < entry.get("trail_peak", price):
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = price * (1 + trail_pct)
                elif not partial_done:
                    move = ((price - entry["price"]) / entry["price"] if d == "long"
                            else (entry["price"] - price) / entry["price"])
                    if move >= TRAIL_TRIGGER:
                        entry["trail_active"]   = True
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = (price * (1 - trail_pct) if d == "long"
                                                   else price * (1 + trail_pct))

                tsv   = entry.get("trail_stop_val")
                ht    = (d == "long" and price >= entry["target"]) or (d == "short" and price <= entry["target"])
                hs    = (d == "long" and price <= entry["stop"])   or (d == "short" and price >= entry["stop"])
                htr   = (entry["trail_active"] and tsv is not None and
                         ((d == "long" and price <= tsv) or (d == "short" and price >= tsv)))
                htime = hour >= 15 and minute >= 20

                if ht or hs or htr or htime:
                    res = ("Target Hit" if ht else
                           "Trailing Stop" if htr else
                           "Time Exit" if htime else "Stop Loss Hit")
                    ep = (entry["target"] if ht else
                          tsv if htr else
                          entry["stop"] if hs else price)
                    remaining  = 0.5 if partial_done else 1.0
                    pnl_pct    = ((ep - entry["price"]) / entry["price"] * 100 if d == "long"
                                  else (entry["price"] - ep) / entry["price"] * 100)
                    pnl_dollar = entry["pos_size"] * remaining * pnl_pct / 100
                    if partial_done:
                        close_trade(trades, {**entry, "pos_size": entry["pos_size"] * 0.5}, ep, res, date, ts)
                    else:
                        close_trade(trades, entry, ep, res, date, ts)
                    daily_pnl += pnl_dollar
                    if pnl_pct <= 0:
                        consec_losses += 1
                        if consec_losses >= CONSEC_LOSS_PAUSE:
                            pause_until   = ts + datetime.timedelta(minutes=30)
                            consec_losses = 0
                    else:
                        consec_losses = 0
                    in_trade = False; entry = None; partial_done = False
                    continue

            if in_trade: continue

            # ── Indicators (pre-computed cross-bar for accuracy with 1h bars) ─
            rsi  = lookup(pre_rsi,   ts, 50.0)
            ema9 = lookup(pre_ema9,  ts, price)
            ema21= lookup(pre_ema21, ts, price)
            atr  = lookup(pre_atr,   ts, price * 0.01)
            if atr <= 0: atr = price * 0.01

            move_from_open = abs(price - open_price) / open_price if open_price > 0 else 0
            min_move = 0.0005 if hourly else 0.002   # hourly bars naturally move less %
            if move_from_open < min_move: continue

            momentum_long  = prev_price > prev2 and price > prev_price
            momentum_short = prev_price < prev2 and price < prev_price
            vol_rising     = volumes[i] > volumes[i-1] if i > 0 else False

            # vol_ratio: for hourly use morning avg baseline; for minute use 20-bar rolling
            if hourly:
                avg_v = day_avg_vol if day_avg_vol and day_avg_vol > 0 else 1.0
            else:
                vol_sl = pd.Series(volumes[max(0, i-20):i])
                avg_v  = float(vol_sl.mean()) if len(vol_sl) > 0 else 1.0
            vol_ratio = round(volumes[i] / avg_v, 2) if avg_v > 0 else 1.0

            # Hourly bars: skip vol_ratio filter — the 9:30 bar's peak volume always
            # inflates the morning average, making 10:30/11:30 bars look low.
            # Trend + RSI + QQQ + momentum are sufficient filters at this resolution.
            vol_min_eff = 0.0 if hourly else cfg["vol_min"]

            sig = None; dirn = None; momentum_ok = False
            strategy = cfg["strategy"]

            # Hourly RSI: 14-bar RSI spans ~2 trading days — much smoother than
            # 1m RSI, so we widen the acceptance window to capture real signals
            rsi_bull_min = 45 if hourly else cfg["rsi_bull_min"]
            rsi_bull_max = 80 if hourly else cfg["rsi_bull_max"]
            rsi_bear_min = 20 if hourly else cfg["rsi_bear_min"]
            rsi_bear_max = 55 if hourly else cfg["rsi_bear_max"]

            if strategy in ("vwap_reclaim", "both") and vwap > 0 and i > 0:
                if hourly:
                    # Hourly: "above VWAP and rising" — trend continuation, not exact crossover
                    if price > vwap and momentum_long and vol_ratio >= vol_min_eff:
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "vwap_reclaim"; dirn = "long"; momentum_ok = True
                    elif price < vwap and momentum_short and vol_ratio >= vol_min_eff:
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "vwap_reclaim"; dirn = "short"; momentum_ok = True
                else:
                    # Minute: exact VWAP crossover (original logic)
                    if closes[i-1] < vwap and price > vwap and vol_ratio >= vol_min_eff:
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "vwap_reclaim"; dirn = "long"; momentum_ok = momentum_long
                    elif closes[i-1] > vwap and price < vwap and vol_ratio >= vol_min_eff:
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "vwap_reclaim"; dirn = "short"; momentum_ok = momentum_short

            if not sig and strategy in ("ema_pullback", "both") and i > 0:
                if hourly:
                    # Hourly: "above both EMAs and rising" — trend continuation
                    # (tight pullback condition too rare with only 2 bars/day in trade window)
                    if (price > ema9 and price > ema21 and momentum_long
                            and vol_ratio >= vol_min_eff):
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "ema_pullback"; dirn = "long"; momentum_ok = True
                    elif (price < ema9 and price < ema21 and momentum_short
                            and vol_ratio >= vol_min_eff):
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "ema_pullback"; dirn = "short"; momentum_ok = True
                else:
                    # Minute: tight EMA pullback crossover (original logic)
                    if (closes[i-1] <= ema9*1.001 and price > ema9 and price > ema21
                            and vol_ratio >= vol_min_eff and momentum_long):
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "ema_pullback"; dirn = "long"; momentum_ok = True
                    elif (closes[i-1] >= ema9*0.999 and price < ema9 and price < ema21
                            and vol_ratio >= vol_min_eff and momentum_short):
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "ema_pullback"; dirn = "short"; momentum_ok = True

            if not sig: continue

            # ── QQQ 20 EMA regime filter ─────────────────────────────────────
            if qqq_ema20 is not None and qqq_close is not None:
                try:
                    qqq_ema_val = float(qqq_ema20.asof(ts))
                    qqq_px      = float(qqq_close.asof(ts))
                    if dirn == "long"  and qqq_px < qqq_ema_val * 0.998: continue
                    if dirn == "short" and qqq_px > qqq_ema_val * 1.002: continue
                except: pass

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap,
                              ema9, day_bias, cfg, momentum_ok, vol_rising, hour, minute, gap_pct)
            if sc < min_score: continue

            entry_price = apply_slippage(price, dirn, cfg["slippage"])
            ps   = get_pos_size(sc, ticker)
            gain = cfg.get("gain_target", 0.015)
            stop = cfg.get("stop_loss",   0.0075)
            tgt  = round(entry_price * (1 + gain) if dirn == "long" else entry_price * (1 - gain), 4)
            stp  = round(entry_price * (1 - stop)  if dirn == "long" else entry_price * (1 + stop), 4)

            in_trade = True; partial_done = False
            entry = {
                "dir": dirn, "price": entry_price, "target": tgt, "stop": stp,
                "trail_active": False, "trail_peak": entry_price, "trail_stop_val": None,
                "breakeven_set": False, "atr": atr,
                "rsi": rsi, "vol_ratio": vol_ratio, "sig_type": sig, "score": sc,
                "pos_size": ps, "time": ts.strftime("%I:%M %p"), "hour": hour,
                "day_bias": day_bias, "ticker": ticker,
                "prime_window": (hour == 9 and minute >= 35) or (hour == 10 and minute <= 30),
            }
    return trades

# ── Statistics ─────────────────────────────────────────────────────────────────
def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0,
                "total_pnl_pct":0,"total_pnl_dollar":0,"best":0,"worst":0,
                "max_drawdown":0,"sharpe":0,"avg_score":0,"profit_factor":0,
                "expectancy":0,"prime_win_rate":0,"by_ticker":{},"avg_win":0,"avg_loss":0}
    wins   = [t for t in trades if t["result"].startswith("Target Hit") or
              (any(x in t["result"] for x in ("Trailing","Time","Partial")) and t["pnl_pct"]>0)]
    losses = [t for t in trades if t["result"].startswith("Stop Loss") or
              (any(x in t["result"] for x in ("Trailing","Time")) and t["pnl_pct"]<=0)]
    pnls = [t["pnl_pct"] for t in trades]
    dols = [t["pnl_dollar"] for t in trades]
    peak=0; dd=0; cum=0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if cum-peak < dd: dd = cum-peak
    sharpe = 0
    if len(pnls) > 1:
        try: sharpe = round(statistics.mean(pnls)/statistics.stdev(pnls)*(252**0.5), 3)
        except: pass
    gw = sum(t["pnl_dollar"] for t in wins)
    gl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = round(gw/gl, 3) if gl>0 else 999
    wr = len(wins)/len(trades)
    avg_win  = round(gw/len(wins),  2) if wins   else 0
    avg_loss = round(gl/len(losses),2) if losses else 0
    exp      = round(wr*avg_win - (1-wr)*avg_loss, 2)
    prime      = [t for t in trades if t.get("prime_window")]
    prime_wins = [t for t in prime if t in wins]
    prime_wr   = round(len(prime_wins)/len(prime)*100, 1) if prime else 0
    by_ticker  = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in by_ticker: by_ticker[tk] = {"trades":0,"pnl":0.0,"wins":0}
        by_ticker[tk]["trades"] += 1
        by_ticker[tk]["pnl"]    += t["pnl_dollar"]
        if t in wins: by_ticker[tk]["wins"] += 1
    return {
        "trades":len(trades),"wins":len(wins),"losses":len(losses),
        "win_rate":round(wr*100,2),"total_pnl_pct":round(sum(pnls),3),
        "total_pnl_dollar":round(sum(dols),2),
        "best":round(max(pnls),3) if pnls else 0,
        "worst":round(min(pnls),3) if pnls else 0,
        "max_drawdown":round(dd,3),"sharpe":sharpe,
        "avg_score":round(sum(t["signal_score"] for t in trades)/len(trades),1),
        "profit_factor":pf,"expectancy":exp,"prime_win_rate":prime_wr,
        "avg_win":avg_win,"avg_loss":avg_loss,"by_ticker":by_ticker,
    }

# ── Monte Carlo ────────────────────────────────────────────────────────────────
def monte_carlo(trades, n_simulations=1000):
    if len(trades) < 10: return {}
    pnls    = [t["pnl_dollar"] for t in trades]
    results = []
    for _ in range(n_simulations):
        sample = random.choices(pnls, k=len(pnls))
        cum=0; peak=0; dd=0
        for p in sample:
            cum += p
            if cum > peak: peak = cum
            if cum-peak < dd: dd = cum-peak
        results.append({"total":round(cum,2),"max_dd":round(dd,2)})
    totals    = [r["total"]  for r in results]
    dds       = [r["max_dd"] for r in results]
    profitable = sum(1 for t in totals if t>0)/n_simulations*100
    return {
        "simulations":    n_simulations,
        "median_pnl":     round(statistics.median(totals), 2),
        "pct_profitable": round(profitable, 1),
        "worst_case_dd":  round(min(dds), 2),
        "best_case":      round(max(totals), 2),
        "worst_case":     round(min(totals), 2),
        "pct_10":         round(sorted(totals)[int(n_simulations*0.1)], 2),
        "pct_90":         round(sorted(totals)[int(n_simulations*0.9)], 2),
    }

# ── Time-of-day analysis ───────────────────────────────────────────────────────
def time_of_day_analysis(trades):
    buckets = {
        "9:30-10:00": {"trades":[], "wins":0, "losses":0, "pnl":0.0},
        "10:00-11:00":{"trades":[], "wins":0, "losses":0, "pnl":0.0},
        "11:00-12:00":{"trades":[], "wins":0, "losses":0, "pnl":0.0},
        "12:00-13:00":{"trades":[], "wins":0, "losses":0, "pnl":0.0},
    }
    def bucket(t):
        h = t.get("hour", 0)
        time_str = t.get("entry_time","")
        try:
            parts = time_str.replace("AM","").replace("PM","").strip().split(":")
            h2 = int(parts[0]); m2 = int(parts[1]) if len(parts)>1 else 0
            ap = "PM" if "PM" in time_str else "AM"
            if ap=="PM" and h2!=12: h2+=12
            elif ap=="AM" and h2==12: h2=0
        except:
            h2 = h; m2 = 0
        if h2==9:                   return "9:30-10:00"
        elif h2==10:                return "10:00-11:00"
        elif h2==11:                return "11:00-12:00"
        elif h2==12:                return "12:00-13:00"
        return None
    is_win = lambda t: t["result"].startswith("Target Hit") or \
             (any(x in t["result"] for x in ("Trailing","Time","Partial")) and t["pnl_pct"]>0)
    for t in trades:
        b = bucket(t)
        if b and b in buckets:
            buckets[b]["trades"].append(t)
            buckets[b]["pnl"] += t["pnl_dollar"]
            if is_win(t): buckets[b]["wins"] += 1
            else:         buckets[b]["losses"] += 1
    result = {}
    print("\n  ── Time-of-day P&L breakdown ──────────────────────────")
    for label, d in buckets.items():
        n  = len(d["trades"])
        wr = round(d["wins"]/n*100,1) if n>0 else 0
        result[label] = {
            "trades": n, "wins": d["wins"], "losses": d["losses"],
            "win_rate": wr, "pnl_dollar": round(d["pnl"],2),
            "avg_pnl": round(d["pnl"]/n,2) if n>0 else 0,
        }
        print(f"  {label}: {n:3d} trades | {wr:5.1f}% WR | ${d['pnl']:+8.2f} | avg ${round(d['pnl']/n,2) if n>0 else 0:+.2f}/trade")
    best = max(result.items(), key=lambda x:x[1]["pnl_dollar"]) if result else ("—",{})
    print(f"  Best window: {best[0]} (${best[1].get('pnl_dollar',0):+.2f})\n")
    return result

# ── Walk-forward test ──────────────────────────────────────────────────────────
def walk_forward_test(data, qqq_close, qqq_ema20, corr_skip_dates):
    """Split data into 4 equal periods and test each independently."""
    # Gather all trading dates across all tickers
    all_dates = set()
    for ticker in TICKERS:
        if not data[ticker].empty:
            all_dates.update(data[ticker].index.normalize().unique())
    all_dates = sorted(all_dates)
    if len(all_dates) < 8:
        print("  Not enough dates for walk-forward test")
        return []

    n       = len(all_dates)
    chunk   = n // 4
    periods = []
    print("\n  ── Walk-forward test (4 periods) ──────────────────────")
    for p in range(4):
        start = all_dates[p*chunk]
        end   = all_dates[min((p+1)*chunk-1, n-1)]
        period_trades = []
        for ticker in TICKERS:
            if data[ticker].empty: continue
            df_slice = data[ticker][(data[ticker].index.normalize() >= start) &
                                    (data[ticker].index.normalize() <= end)]
            if df_slice.empty: continue
            t = run_strategy(df_slice, ticker, qqq_close=qqq_close,
                             qqq_ema20=qqq_ema20, corr_skip_dates=corr_skip_dates)
            period_trades.extend(t)
        period_trades.sort(key=lambda t:(t["date"],t["entry_time"]))
        st = calc_stats(period_trades)
        label = f"P{p+1}: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
        print(f"  {label} | {st['trades']:3d} trades | {st['win_rate']:.1f}% WR | "
              f"${st['total_pnl_dollar']:+.2f} | PF={st['profit_factor']:.3f} | E=${st['expectancy']:.2f}")
        periods.append({
            "period":      label,
            "start":       start.strftime("%Y-%m-%d"),
            "end":         end.strftime("%Y-%m-%d"),
            "stats":       st,
            "trade_count": len(period_trades),
        })

    # Check consistency: WR should not deviate more than 20% across periods
    wrs = [p["stats"]["win_rate"] for p in periods if p["stats"]["trades"]>0]
    if wrs:
        spread = max(wrs) - min(wrs)
        consistent = spread < 20
        print(f"  WR spread: {spread:.1f}% — {'✅ CONSISTENT' if consistent else '⚠️ DIVERGED'}")
    return periods

# ── Correlation filter builder ─────────────────────────────────────────────────
def build_corr_skip_dates(data):
    """Return a set of dates where AAPL and GOOGL both dropped >1% in first 30 min."""
    skip = set()
    tickers_present = [t for t in ["AAPL","GOOGL"] if t in data and not data[t].empty]
    if len(tickers_present) < 2:
        return skip
    # Find common dates
    dates_a = set(data["AAPL"].index.normalize().unique())
    dates_g = set(data["GOOGL"].index.normalize().unique())
    common  = dates_a & dates_g
    for date in common:
        try:
            d_date = date.date()
            # First 30 minutes
            window_a = data["AAPL"][data["AAPL"].index.date==d_date].between_time("09:30","10:00")
            window_g = data["GOOGL"][data["GOOGL"].index.date==d_date].between_time("09:30","10:00")
            if window_a.empty or window_g.empty: continue
            ret_a = (float(window_a["Close"].squeeze().iloc[-1]) -
                     float(window_a["Open"].squeeze().iloc[0])) / float(window_a["Open"].squeeze().iloc[0]) * 100
            ret_g = (float(window_g["Close"].squeeze().iloc[-1]) -
                     float(window_g["Open"].squeeze().iloc[0])) / float(window_g["Open"].squeeze().iloc[0]) * 100
            if ret_a < -1.0 and ret_g < -1.0:
                skip.add(d_date)
        except: pass
    return skip

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  NYLO Backtest v20 — Full-Year Hybrid (1h + 5m + 1m)")
    print(f"  Tickers : {', '.join(TICKERS)}")
    print(f"  Data    : 1h bars ~months 3-12 + 5m/1m bars recent 60d")
    print(f"  Filters : QQQ 20 EMA (enforced) + Earnings blackout")
    print(f"            + Correlation filter")
    print(f"  Slippage: AAPL 0.03%, GOOGL 0.025%")
    print(f"  Target  : ~300 trading days, 200-400+ trades")
    print("="*60)

    data = {}
    for ticker in TICKERS + [MARKET_FILTER_TICKER]:
        df = fetch_hybrid(ticker)
        data[ticker] = df

    # Build QQQ series for regime filter
    qqq_close  = None
    qqq_ema20  = None
    if not data[MARKET_FILTER_TICKER].empty:
        qqq_cl   = data[MARKET_FILTER_TICKER]["Close"].squeeze()
        qqq_close = qqq_cl
        qqq_ema20 = qqq_cl.ewm(span=20, adjust=False).mean()
        print(f"\n  QQQ 20 EMA built ({len(qqq_ema20)} bars)")

    # Build correlation skip dates
    print("  Computing correlation filter...")
    corr_skip = build_corr_skip_dates(data)
    print(f"  Correlation filter: {len(corr_skip)} days skipped")

    # Earnings blackout summary
    all_dates = set()
    for t in TICKERS:
        if not data[t].empty:
            all_dates.update(d.date() for d in data[t].index.normalize().unique())
    eb_count = sum(1 for d in all_dates for t in TICKERS if is_earnings_blackout(d, t))
    print(f"  Earnings blackout: ~{eb_count} ticker-day combos skipped")

    # ── Baseline ───────────────────────────────────────────────────────────────
    print(f"\n[1/4] Baseline strategy (v19)...")
    base_trades = []
    for ticker in TICKERS:
        if data[ticker].empty: continue
        t = run_strategy(data[ticker], ticker,
                         qqq_close=qqq_close, qqq_ema20=qqq_ema20,
                         corr_skip_dates=corr_skip)
        base_trades.extend(t)
        print(f"  {ticker}: {len(t)} trades | {TICKER_CONFIGS[ticker]['strategy']}")

    base_trades.sort(key=lambda t:(t["date"],t["entry_time"]))
    bs = calc_stats(base_trades)
    print(f"  TOTAL: {bs['trades']} trades | {bs['win_rate']:.1f}% WR | "
          f"${bs['total_pnl_dollar']:+.2f} | Sharpe:{bs['sharpe']} | "
          f"PF:{bs['profit_factor']} | E:${bs['expectancy']:.2f}/trade")
    print(f"  Prime WR:{bs['prime_win_rate']}% | Max DD:{bs['max_drawdown']:.2f}%")
    print(f"  Avg win:${bs['avg_win']} | Avg loss:${bs['avg_loss']}")
    for tk, s in bs.get("by_ticker",{}).items():
        wr = round(s["wins"]/s["trades"]*100,1) if s["trades"]>0 else 0
        print(f"    {tk}: {s['trades']} trades | ${s['pnl']:+.2f} | {wr}% WR")

    # ── Time-of-day analysis ───────────────────────────────────────────────────
    tod = time_of_day_analysis(base_trades)

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    print(f"\n[2/4] Monte Carlo (1000 sims)...")
    mc = monte_carlo(base_trades)
    print(f"  Median P&L: ${mc.get('median_pnl',0)} | "
          f"Profitable: {mc.get('pct_profitable',0)}% of sims")
    print(f"  10th pct: ${mc.get('pct_10',0)} | 90th pct: ${mc.get('pct_90',0)}")
    print(f"  Worst DD: ${mc.get('worst_case_dd',0)}")

    # ── OOS ────────────────────────────────────────────────────────────────────
    # Split at Nov 2025 for ~50/50 over 1-year window (May 2025 – May 2026)
    print(f"\n[3/4] Out-of-sample split...")
    oos_split  = datetime.date(2025, 11, 1)
    train = [t for t in base_trades if datetime.date.fromisoformat(t["date"]) <  oos_split]
    test  = [t for t in base_trades if datetime.date.fromisoformat(t["date"]) >= oos_split]
    ts_train = calc_stats(train); ts_test = calc_stats(test)
    consistent = abs(ts_test["win_rate"]-ts_train["win_rate"]) < 15
    diff       = abs(ts_test["win_rate"]-ts_train["win_rate"])
    print(f"  Train: {ts_train['trades']} trades | {ts_train['win_rate']:.1f}% WR | ${ts_train['total_pnl_dollar']:+.2f}")
    print(f"  Test : {ts_test['trades']} trades | {ts_test['win_rate']:.1f}% WR | ${ts_test['total_pnl_dollar']:+.2f}")
    print(f"  OOS  : {'✅ CONSISTENT' if consistent else '⚠️ DIVERGED'} (diff={diff:.1f}%)")

    # ── Walk-forward ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Walk-forward test (4 periods)...")
    wf_periods = walk_forward_test(data, qqq_close, qqq_ema20, corr_skip)
    wf_wrs = [p["stats"]["win_rate"] for p in wf_periods if p["stats"]["trades"]>0]
    wf_consistent = (max(wf_wrs)-min(wf_wrs)) < 20 if len(wf_wrs)>=2 else False

    # ── Parameter sweep ────────────────────────────────────────────────────────
    print(f"\n  Parameter sweep...")
    combos = [(ms,asm,atm) for ms in SWEEP_MIN_SCORE
              for asm in SWEEP_ATR_STOP for atm in SWEEP_ATR_TARGET if atm>asm]
    sweep  = []
    for done,(ms,asm,atm) in enumerate(combos):
        t = []
        for ticker in TICKERS:
            if not data[ticker].empty:
                t.extend(run_strategy(data[ticker], ticker,
                                      qqq_close=qqq_close, qqq_ema20=qqq_ema20,
                                      corr_skip_dates=corr_skip,
                                      min_score=ms, atr_stop_mult=asm, atr_target_mult=atm))
        t.sort(key=lambda x:(x["date"],x["entry_time"]))
        st = calc_stats(t)
        sc = st["win_rate"]*0.4 + st["profit_factor"]*15 + st["expectancy"]*0.5
        sweep.append({"min_score":ms,"atr_stop":asm,"atr_target":atm,"stats":st,"score":round(sc,3)})
        sys.stdout.write(f"\r  Progress: {done+1}/{len(combos)}")
        sys.stdout.flush()
    sweep.sort(key=lambda r:r["score"], reverse=True)
    best = sweep[0]
    print(f"\n  Best: score={best['min_score']} ATR stop={best['atr_stop']}x "
          f"target={best['atr_target']}x → {best['stats']['win_rate']:.0f}% WR | "
          f"PF={best['stats']['profit_factor']} | E=${best['stats']['expectancy']:.2f}/trade")

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "generated_at":  datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str": datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "version":       "v20",
        "config": {
            "tickers":      TICKERS,
            "market_filter":"QQQ 20 EMA (enforced)",
            "strategy":     "v20 — 1y hybrid (1h+5m+1m), pre-computed indicators, bar-adaptive logic",
            "features":     ["hybrid_1h_5m_1m_dataset","pre_computed_cross_bar_indicators",
                             "bar_size_adaptive_thresholds","qqq_20ema_regime_enforced","earnings_blackout_2d",
                             "correlation_filter_1pct_30min","time_of_day_analysis",
                             "walk_forward_4_periods","slippage_aapl_003_googl_0025",
                             "atr_normalized_stops","partial_exits","breakeven_stop",
                             "daily_loss_limit","monte_carlo_1000"],
            "slippage":     {"AAPL":"0.03%","GOOGL":"0.025%"},
            "baseline":     {"min_score":MIN_SCORE,"atr_stop_mult":1.0,"atr_target_mult":1.5},
            "filters_applied": {
                "earnings_blackout_days": EARNINGS_BLACKOUT_DAYS,
                "corr_days_skipped":      len(corr_skip),
                "earnings_dates":         {k:[str(d) for d in v] for k,v in EARNINGS_DATES.items()},
            },
        },
        "baseline":        {"stats":bs,"trades":base_trades},
        "monte_carlo":     mc,
        "out_of_sample":   {"train":{"stats":ts_train},"test":{"stats":ts_test},"consistent":consistent},
        "time_of_day":     tod,
        "walk_forward":    {"periods":wf_periods,"consistent":wf_consistent,
                            "wr_spread":round(max(wf_wrs)-min(wf_wrs),1) if len(wf_wrs)>=2 else 0},
        "sweep":           sweep[:30],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Results saved → {OUT}")
    print("="*60)

if __name__ == "__main__":
    main()
