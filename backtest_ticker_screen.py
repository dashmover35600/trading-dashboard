"""
NYLO Ticker Screener
====================
Runs the v18 VWAP Reclaim + EMA Pullback + Opening Drive strategy against
a candidate list of tickers using hybrid yfinance data (~180 trading days).

Exit targets (user-specified):
  - Partial exit  : +1.25%
  - Full target   : +2.50%
  - Trail stop    : 0.50%
  - Stop loss     : 1.25% (1:2 R:R)

Pass criteria:
  - Win rate      >= 70%
  - Profit factor >= 1.4
  - Trades/month  >= 10

Ranking: composite score = (WR/100) * PF * trades_per_month

Usage:
    python3 -W ignore backtest_ticker_screen.py

Outputs: backtest_ticker_screen_results.json
"""

import json
import math
import datetime
import statistics
import random
import os

import pandas as pd
import pytz
import ta
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
SCREEN_TICKERS = ["NVDA", "TSLA", "META", "MSFT", "AMD", "AMZN", "SPY"]
MARKET_TZ      = pytz.timezone("America/New_York")
BASE           = os.path.dirname(os.path.abspath(__file__))
OUT            = os.path.join(BASE, "backtest_ticker_screen_results.json")

# v18 exit parameters (user-specified overrides)
PARTIAL_PCT       = 0.0125   # +1.25% partial exit
GAIN_TARGET_PCT   = 0.025    # +2.50% full target
STOP_LOSS_PCT     = 0.0125   # -1.25% stop (1:2 R:R)
TRAIL_TRIGGER_PCT = 0.0125   # activate trail at +1.25%
TRAIL_STOP_PCT    = 0.005    # trail 0.50% from peak
BREAKEVEN_TRIGGER = 0.00625  # move stop to BE at +0.625%

MIN_SCORE        = 3
DAILY_LOSS_LIMIT = -500.0
CONSEC_LOSS_PAUSE = 3
TRADE_START      = datetime.time(9, 30)
TRADE_END        = datetime.time(12, 0)

# Pass criteria
PASS_WR     = 70.0
PASS_PF     = 1.4
PASS_TPM    = 10.0

# Per-ticker config — RSI / slippage / strategy
TICKER_CONFIGS = {
    "NVDA": {"rsi_bull_min":50,"rsi_bull_max":74,"rsi_bear_min":26,"rsi_bear_max":50,
             "slippage":0.00025,"vol_min":1.2,"strategy":"both"},
    "TSLA": {"rsi_bull_min":50,"rsi_bull_max":74,"rsi_bear_min":26,"rsi_bear_max":50,
             "slippage":0.0003,"vol_min":1.2,"strategy":"both"},
    "META": {"rsi_bull_min":52,"rsi_bull_max":72,"rsi_bear_min":28,"rsi_bear_max":48,
             "slippage":0.0002,"vol_min":1.2,"strategy":"both"},
    "MSFT": {"rsi_bull_min":52,"rsi_bull_max":72,"rsi_bear_min":28,"rsi_bear_max":48,
             "slippage":0.00015,"vol_min":1.2,"strategy":"both"},
    "AMD":  {"rsi_bull_min":50,"rsi_bull_max":74,"rsi_bear_min":26,"rsi_bear_max":50,
             "slippage":0.0003,"vol_min":1.2,"strategy":"both"},
    "AMZN": {"rsi_bull_min":52,"rsi_bull_max":72,"rsi_bear_min":28,"rsi_bear_max":48,
             "slippage":0.0002,"vol_min":1.2,"strategy":"both"},
    "SPY":  {"rsi_bull_min":52,"rsi_bull_max":72,"rsi_bear_min":28,"rsi_bear_max":48,
             "slippage":0.0001,"vol_min":1.3,"strategy":"both"},
}

# Earnings blackout — ±2 days around each date
EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_DATES = {
    "NVDA": [datetime.date(2024,2,21), datetime.date(2024,5,22),
             datetime.date(2024,8,28), datetime.date(2024,11,20),
             datetime.date(2025,2,26), datetime.date(2025,5,28),
             datetime.date(2025,8,27), datetime.date(2025,11,19)],
    "TSLA": [datetime.date(2024,1,24), datetime.date(2024,4,23),
             datetime.date(2024,7,23), datetime.date(2024,10,23),
             datetime.date(2025,1,29), datetime.date(2025,4,22),
             datetime.date(2025,7,22), datetime.date(2025,10,22)],
    "META": [datetime.date(2024,2,1),  datetime.date(2024,4,24),
             datetime.date(2024,7,31), datetime.date(2024,10,30),
             datetime.date(2025,1,29), datetime.date(2025,4,30),
             datetime.date(2025,7,30), datetime.date(2025,10,29)],
    "MSFT": [datetime.date(2024,1,30), datetime.date(2024,4,25),
             datetime.date(2024,7,30), datetime.date(2024,10,30),
             datetime.date(2025,1,29), datetime.date(2025,4,30),
             datetime.date(2025,7,30), datetime.date(2025,10,29)],
    "AMD":  [datetime.date(2024,1,30), datetime.date(2024,4,30),
             datetime.date(2024,7,30), datetime.date(2024,10,29),
             datetime.date(2025,1,28), datetime.date(2025,4,29),
             datetime.date(2025,7,29), datetime.date(2025,10,28)],
    "AMZN": [datetime.date(2024,2,1),  datetime.date(2024,4,30),
             datetime.date(2024,8,1),  datetime.date(2024,10,31),
             datetime.date(2025,2,6),  datetime.date(2025,5,1),
             datetime.date(2025,7,31), datetime.date(2025,10,30)],
    "SPY":  [],  # ETF — no earnings blackout
}

# ── Data fetch ─────────────────────────────────────────────────────────────────

def _process(frames):
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    # Drop MultiIndex columns if present (yfinance multi-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df.between_time("09:30", "15:30")


def fetch_hybrid(ticker: str) -> pd.DataFrame:
    """Fetch ~180 trading days via 1h+5m+1m hybrid, same approach as v18."""
    end      = datetime.datetime.now(MARKET_TZ)
    start_1y = end - datetime.timedelta(days=200)

    # 1h — full history (~200 days)
    df_1h = pd.DataFrame()
    try:
        raw = yf.download(ticker,
                          start=start_1y.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          interval="1h", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df_1h = _process([raw]) if not raw.empty else pd.DataFrame()
    except Exception:
        pass

    # 1m — last ~35 days (5 × 7-day chunks)
    frames_1m = []
    for i in range(5):
        ce = end - datetime.timedelta(days=i * 7)
        cs = ce  - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker,
                             start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="1m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                frames_1m.append(df)
        except Exception:
            pass
    df_1m = _process(frames_1m)

    # 5m — last ~63 days (9 × 7-day chunks)
    frames_5m = []
    for i in range(9):
        ce = end - datetime.timedelta(days=i * 7)
        cs = ce  - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker,
                             start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="5m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                frames_5m.append(df)
        except Exception:
            pass
    df_5m = _process(frames_5m)

    # Merge: 1m takes priority > 5m fills gap > 1h covers the rest
    if df_1m.empty and df_5m.empty:
        combined = df_1h
    else:
        if not df_1m.empty:
            recent_cut = df_1m.index[0].date()
            if not df_5m.empty:
                df_5m_mid = df_5m[df_5m.index.date < recent_cut]
                recent = _process([df_5m_mid, df_1m])
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

    return combined


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_earnings_blackout(date, ticker: str) -> bool:
    d = date.date() if hasattr(date, "date") else date
    for ed in EARNINGS_DATES.get(ticker, []):
        if abs((d - ed).days) <= EARNINGS_BLACKOUT_DAYS:
            return True
    return False


def apply_slippage(price: float, direction: str, slip: float) -> float:
    return round(price * (1 + slip) if direction == "long" else price * (1 - slip), 4)


def score_signal(direction, sig_type, rsi, vol_ratio, price, vwap,
                 ema9, day_bias, cfg, momentum_ok, vol_rising, hour, minute) -> int:
    score = 0
    if day_bias == direction:    score += 2
    elif day_bias == "neutral":  score += 1

    if direction == "long":
        mid = (cfg["rsi_bull_min"] + cfg["rsi_bull_max"]) / 2
        if abs(rsi - mid) <= 7:                                    score += 2
        elif cfg["rsi_bull_min"] <= rsi <= cfg["rsi_bull_max"]:    score += 1
    else:
        mid = (cfg["rsi_bear_min"] + cfg["rsi_bear_max"]) / 2
        if abs(rsi - mid) <= 7:                                    score += 2
        elif cfg["rsi_bear_min"] <= rsi <= cfg["rsi_bear_max"]:    score += 1

    if vol_ratio >= 2.5:    score += 2
    elif vol_ratio >= 1.5:  score += 1

    if vwap and vwap > 0:
        if direction == "long"  and price > vwap: score += 1
        elif direction == "short" and price < vwap: score += 1

    if direction == "long"  and price > ema9: score += 1
    elif direction == "short" and price < ema9: score += 1

    if sig_type == "vwap_reclaim": score += 1
    if momentum_ok:  score += 1
    if vol_rising:   score += 1
    if (hour == 9 and minute >= 35) or (hour == 10 and minute <= 30): score += 2
    return min(score, 10)


# ── Core strategy ──────────────────────────────────────────────────────────────

def run_strategy(df: pd.DataFrame, ticker: str) -> list[dict]:
    cfg    = TICKER_CONFIGS[ticker]
    slip   = cfg["slippage"]
    trades = []

    if df.empty:
        return trades

    # Pre-compute indicators on full series for proper cross-bar lookback
    full_cl  = df["Close"].squeeze()
    full_hi  = df["High"].squeeze()
    full_lo  = df["Low"].squeeze()
    pre_rsi  = ta.momentum.RSIIndicator(full_cl, window=14).rsi()
    pre_ema9 = full_cl.ewm(span=9,  adjust=False).mean()
    pre_ema21= full_cl.ewm(span=21, adjust=False).mean()
    pre_atr  = ta.volatility.AverageTrueRange(
                   full_hi, full_lo, full_cl, window=14).average_true_range()

    def lookup(series, ts, default):
        try:
            v = series.asof(ts)
            return default if pd.isna(v) else float(v)
        except Exception:
            return default

    dates = sorted(df.index.normalize().unique())

    for date in dates:
        d_date = date.date()
        if is_earnings_blackout(d_date, ticker):
            continue

        day_df = df[df.index.date == d_date]
        n_bars = len(day_df)

        hourly    = n_bars < 15
        min_bars  = 3 if hourly else 10
        skip_bars = 1 if hourly else 5
        if n_bars < min_bars:
            continue

        closes  = day_df["Close"].squeeze().tolist()
        volumes = day_df["Volume"].squeeze().tolist()
        highs   = day_df["High"].squeeze().tolist()
        lows    = day_df["Low"].squeeze().tolist()
        day_idx = list(day_df.index)
        open_price = closes[0] if closes else 0

        # Opening drive bias
        drive_df = day_df.between_time("09:30", "09:35")
        if len(drive_df) < 1:
            continue
        drive_open  = float(drive_df["Open"].iloc[0])
        drive_close = float(drive_df["Close"].iloc[-1])
        move_pct    = (drive_close - drive_open) / drive_open * 100
        prev_days   = df[df.index.date < d_date]
        avg_vol     = float(prev_days["Volume"].tail(200).mean()) if not prev_days.empty else 1.0
        drive_vol   = float(drive_df["Volume"].sum())
        vr_drive    = drive_vol / avg_vol if avg_vol > 0 else 1.0

        if   move_pct >  0.3 and vr_drive > 1.5: day_bias = "long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias = "short"
        else:                                      day_bias = "neutral"

        # Intraday VWAP
        cl = day_df["Close"].squeeze(); hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze();   vo = day_df["Volume"].squeeze()
        tp = (hi + lo + cl) / 3
        vwap_list = ((tp * vo).cumsum() / vo.cumsum()).tolist()

        if hourly:
            morn_df     = day_df.between_time("09:30", "12:00")
            day_avg_vol = float(morn_df["Volume"].mean()) if not morn_df.empty else float(day_df["Volume"].mean())
        else:
            day_avg_vol = None

        daily_pnl = 0.0; consec_losses = 0; pause_until = None
        in_trade  = False; entry = None; partial_done = False

        for i, ts in enumerate(day_idx):
            if i < skip_bars:
                continue
            hour, minute = ts.hour, ts.minute
            t_now = ts.time()
            if t_now < TRADE_START:
                continue
            if t_now > TRADE_END and not in_trade:
                continue
            if hour >= 15 and minute >= 25:
                break
            if daily_pnl <= DAILY_LOSS_LIMIT:
                break
            if pause_until and ts < pause_until:
                continue
            else:
                pause_until = None

            price      = closes[i]
            prev_price = closes[i - 1] if i > 0 else price
            prev2      = closes[i - 2] if i > 1 else prev_price
            vwap       = vwap_list[i] if i < len(vwap_list) else 0

            # ── Exit logic ─────────────────────────────────────────────────
            if in_trade and entry:
                d         = entry["dir"]
                trail_pct = TRAIL_STOP_PCT * 1.5 if hour >= 11 else TRAIL_STOP_PCT

                # Breakeven stop
                if not entry.get("breakeven_set", False):
                    move = ((price - entry["price"]) / entry["price"] if d == "long"
                            else (entry["price"] - price) / entry["price"])
                    if move >= BREAKEVEN_TRIGGER:
                        entry["stop"]         = entry["price"]
                        entry["breakeven_set"] = True

                # Partial exit at +1.25%
                partial_tgt = (entry["price"] * (1 + PARTIAL_PCT) if d == "long"
                               else entry["price"] * (1 - PARTIAL_PCT))
                if not partial_done:
                    if ((d == "long"  and price >= partial_tgt) or
                            (d == "short" and price <= partial_tgt)):
                        pnl_pct = ((partial_tgt - entry["price"]) / entry["price"] * 100
                                   if d == "long"
                                   else (entry["price"] - partial_tgt) / entry["price"] * 100)
                        pnl_dollar = entry["pos_size"] * 0.5 * pnl_pct / 100
                        daily_pnl += pnl_dollar
                        trades.append(_make_trade(entry, partial_tgt, "Partial Exit",
                                                  d_date, ts, slip, partial=True))
                        partial_done          = True
                        entry["trail_active"] = True
                        entry["trail_peak"]   = price

                # Trail update
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
                    if move >= TRAIL_TRIGGER_PCT:
                        entry["trail_active"]   = True
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = (price * (1 - trail_pct) if d == "long"
                                                   else price * (1 + trail_pct))

                tsv   = entry.get("trail_stop_val")
                ht    = ((d == "long"  and price >= entry["target"]) or
                         (d == "short" and price <= entry["target"]))
                hs    = ((d == "long"  and price <= entry["stop"]) or
                         (d == "short" and price >= entry["stop"]))
                htr   = (entry["trail_active"] and tsv is not None and
                         ((d == "long" and price <= tsv) or (d == "short" and price >= tsv)))
                htime = hour >= 15 and minute >= 20

                if ht or hs or htr or htime:
                    res = ("Target Hit"    if ht   else
                           "Trailing Stop" if htr  else
                           "Time Exit"     if htime else "Stop Loss Hit")
                    ep = (entry["target"] if ht else
                          tsv             if htr  else
                          entry["stop"]   if hs   else price)
                    remaining  = 0.5 if partial_done else 1.0
                    size       = entry["pos_size"] * remaining
                    pnl_pct    = ((ep - entry["price"]) / entry["price"] * 100 if d == "long"
                                  else (entry["price"] - ep) / entry["price"] * 100)
                    pnl_dollar = size * pnl_pct / 100
                    daily_pnl += pnl_dollar
                    trades.append(_make_trade({**entry, "pos_size": size},
                                              ep, res, d_date, ts, slip))
                    if pnl_pct <= 0:
                        consec_losses += 1
                        if consec_losses >= CONSEC_LOSS_PAUSE:
                            pause_until   = ts + datetime.timedelta(minutes=30)
                            consec_losses = 0
                    else:
                        consec_losses = 0
                    in_trade = False; entry = None; partial_done = False
                    continue

            if in_trade:
                continue

            # ── Signal detection ──────────────────────────────────────────
            rsi   = lookup(pre_rsi,   ts, 50.0)
            ema9  = lookup(pre_ema9,  ts, price)
            ema21 = lookup(pre_ema21, ts, price)

            move_from_open = abs(price - open_price) / open_price if open_price > 0 else 0
            min_move = 0.0005 if hourly else 0.002
            if move_from_open < min_move:
                continue

            momentum_long  = prev_price > prev2 and price > prev_price
            momentum_short = prev_price < prev2 and price < prev_price
            vol_rising     = volumes[i] > volumes[i - 1] if i > 0 else False

            if hourly:
                avg_v = day_avg_vol if day_avg_vol and day_avg_vol > 0 else 1.0
            else:
                vol_sl = pd.Series(volumes[max(0, i - 20):i])
                avg_v  = float(vol_sl.mean()) if len(vol_sl) > 0 else 1.0
            vol_ratio = round(volumes[i] / avg_v, 2) if avg_v > 0 else 1.0

            vol_min_eff = 0.0 if hourly else cfg["vol_min"]

            # Widen RSI for hourly resolution
            rsi_bull_min = 45 if hourly else cfg["rsi_bull_min"]
            rsi_bull_max = 80 if hourly else cfg["rsi_bull_max"]
            rsi_bear_min = 20 if hourly else cfg["rsi_bear_min"]
            rsi_bear_max = 55 if hourly else cfg["rsi_bear_max"]

            sig = None; dirn = None; momentum_ok = False
            strategy = cfg["strategy"]

            # VWAP Reclaim
            if strategy in ("vwap_reclaim", "both") and vwap > 0 and i > 0:
                if hourly:
                    if price > vwap and momentum_long and vol_ratio >= vol_min_eff:
                        if day_bias in ("long", "neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "vwap_reclaim"; dirn = "long"; momentum_ok = True
                    elif price < vwap and momentum_short and vol_ratio >= vol_min_eff:
                        if day_bias in ("short", "neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "vwap_reclaim"; dirn = "short"; momentum_ok = True
                else:
                    if closes[i-1] < vwap and price > vwap and vol_ratio >= vol_min_eff:
                        if day_bias in ("long", "neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "vwap_reclaim"; dirn = "long"; momentum_ok = momentum_long
                    elif closes[i-1] > vwap and price < vwap and vol_ratio >= vol_min_eff:
                        if day_bias in ("short", "neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "vwap_reclaim"; dirn = "short"; momentum_ok = momentum_short

            # EMA Pullback
            if not sig and strategy in ("ema_pullback", "both") and i > 0:
                if hourly:
                    if (price > ema9 and price > ema21 and momentum_long and
                            vol_ratio >= vol_min_eff):
                        if day_bias in ("long", "neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "ema_pullback"; dirn = "long"; momentum_ok = True
                    elif (price < ema9 and price < ema21 and momentum_short and
                              vol_ratio >= vol_min_eff):
                        if day_bias in ("short", "neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "ema_pullback"; dirn = "short"; momentum_ok = True
                else:
                    if (closes[i-1] <= ema9 * 1.001 and price > ema9 and price > ema21 and
                            vol_ratio >= vol_min_eff and momentum_long):
                        if day_bias in ("long", "neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "ema_pullback"; dirn = "long"; momentum_ok = True
                    elif (closes[i-1] >= ema9 * 0.999 and price < ema9 and price < ema21 and
                              vol_ratio >= vol_min_eff and momentum_short):
                        if day_bias in ("short", "neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "ema_pullback"; dirn = "short"; momentum_ok = True

            if not sig:
                continue

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap,
                              ema9, day_bias, cfg, momentum_ok, vol_rising, hour, minute)
            if sc < MIN_SCORE:
                continue

            entry_price = apply_slippage(price, dirn, slip)
            pos_size    = _pos_size(sc)
            tgt = round(entry_price * (1 + GAIN_TARGET_PCT) if dirn == "long"
                        else entry_price * (1 - GAIN_TARGET_PCT), 4)
            stp = round(entry_price * (1 - STOP_LOSS_PCT) if dirn == "long"
                        else entry_price * (1 + STOP_LOSS_PCT), 4)

            in_trade = True; partial_done = False
            entry = {
                "ticker": ticker, "dir": dirn, "price": entry_price,
                "target": tgt, "stop": stp,
                "trail_active": False, "trail_peak": entry_price, "trail_stop_val": None,
                "breakeven_set": False,
                "rsi": rsi, "vol_ratio": vol_ratio, "sig_type": sig, "score": sc,
                "pos_size": pos_size, "time": ts.strftime("%I:%M %p"),
                "hour": hour, "day_bias": day_bias,
                "prime_window": (hour == 9 and minute >= 35) or (hour == 10 and minute <= 30),
            }

    return trades


def _pos_size(score: int) -> float:
    return {10:5000,9:4000,8:3000,7:2000,6:1500,5:1000,4:750,3:500}.get(min(score,10), 500)


def _make_trade(entry: dict, exit_price: float, result: str,
                date, ts, slip: float, partial: bool = False) -> dict:
    d         = entry["dir"]
    exit_p    = round(exit_price * (1 - slip if d == "long" else 1 + slip), 4)
    pnl_pct   = ((exit_p - entry["price"]) / entry["price"] * 100 if d == "long"
                 else (entry["price"] - exit_p) / entry["price"] * 100)
    size      = entry["pos_size"]
    return {
        "date":         date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
        "ticker":       entry["ticker"],
        "direction":    "Long" if d == "long" else "Short",
        "entry":        round(entry["price"], 4),
        "exit":         round(exit_p, 4),
        "result":       result + (" (partial)" if partial else ""),
        "pnl_pct":      round(pnl_pct, 3),
        "pnl_dollar":   round(size * pnl_pct / 100, 2),
        "signal_type":  entry["sig_type"],
        "signal_score": entry["score"],
        "pos_size":     size,
        "day_bias":     entry["day_bias"],
        "prime_window": entry.get("prime_window", False),
    }


# ── Statistics ─────────────────────────────────────────────────────────────────

def calc_stats(trades: list[dict], n_trading_days: int) -> dict:
    if not trades:
        return {}

    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    pnls   = [t["pnl_dollar"] for t in trades]
    n      = len(trades)
    wr     = len(wins) / n * 100

    gw = sum(t["pnl_dollar"] for t in wins)
    gl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = round(gw / gl, 3) if gl > 0 else 999.0

    months        = max(n_trading_days / 21, 1)
    trades_pm     = round(n / months, 1)

    total_pnl = sum(pnls)
    exp       = round(total_pnl / n, 2) if n else 0.0

    # Sharpe on per-trade PnL %
    sharpe = 0.0
    if n > 1:
        try:
            pnl_pcts = [t["pnl_pct"] for t in trades]
            sharpe = round(statistics.mean(pnl_pcts) / statistics.stdev(pnl_pcts) * math.sqrt(252), 3)
        except Exception:
            pass

    # Max drawdown
    equity = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Composite score: (WR/100) * PF * trades_per_month
    composite = round((wr / 100) * pf * trades_pm, 3)

    # Pass/fail
    passed = wr >= PASS_WR and pf >= PASS_PF and trades_pm >= PASS_TPM

    return {
        "n_trades":       n,
        "win_rate":       round(wr, 2),
        "profit_factor":  pf,
        "total_pnl":      round(total_pnl, 2),
        "expectancy":     exp,
        "sharpe":         sharpe,
        "max_drawdown":   round(max_dd, 2),
        "avg_win":        round(gw / len(wins), 2)   if wins   else 0.0,
        "avg_loss":       round(gl / len(losses), 2) if losses else 0.0,
        "trades_per_month": trades_pm,
        "n_trading_days": n_trading_days,
        "composite_score": composite,
        "passed":         passed,
        "pass_criteria": {
            "wr_ok":  wr >= PASS_WR,
            "pf_ok":  pf >= PASS_PF,
            "tpm_ok": trades_pm >= PASS_TPM,
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  NYLO Ticker Screener")
    print(f"  Tickers  : {', '.join(SCREEN_TICKERS)}")
    print(f"  Exits    : partial +{PARTIAL_PCT*100:.2f}% | target +{GAIN_TARGET_PCT*100:.2f}%"
          f" | trail {TRAIL_STOP_PCT*100:.2f}%")
    print(f"  Pass     : WR≥{PASS_WR}% · PF≥{PASS_PF} · ≥{PASS_TPM:.0f} trades/mo")
    print("=" * 62)

    results = []

    for ticker in SCREEN_TICKERS:
        print(f"\n[{ticker}] Fetching ~180-day hybrid dataset...")
        df = fetch_hybrid(ticker)
        if df.empty:
            print(f"  [!] No data — skipping.")
            results.append({"ticker": ticker, "error": "no data"})
            continue

        n_days = df.index.normalize().nunique()
        print(f"  {len(df):,} bars · {n_days} trading days")

        print(f"  Running strategy...")
        trades = run_strategy(df, ticker)
        print(f"  {len(trades)} trades generated")

        if not trades:
            results.append({"ticker": ticker, "error": "no trades", "n_trading_days": n_days})
            continue

        stats = calc_stats(trades, n_days)
        results.append({
            "ticker": ticker,
            "stats":  stats,
            "trades": trades,
        })

        p = "✅ PASS" if stats["passed"] else "❌ FAIL"
        print(f"  {p} | WR {stats['win_rate']:.1f}% | PF {stats['profit_factor']:.3f} "
              f"| {stats['trades_per_month']:.1f} trades/mo | "
              f"Composite {stats['composite_score']:.2f} | "
              f"PnL ${stats['total_pnl']:+.0f}")

    # Sort passing tickers by composite score desc, then failing ones
    passing = [r for r in results if r.get("stats", {}).get("passed")]
    failing = [r for r in results if not r.get("stats", {}).get("passed")]
    passing.sort(key=lambda r: r["stats"]["composite_score"], reverse=True)
    failing.sort(key=lambda r: r.get("stats", {}).get("composite_score", 0), reverse=True)
    ranked  = passing + failing

    print(f"\n{'─'*62}")
    print(f"  RANKING (composite = WR × PF × trades/month)")
    print(f"{'─'*62}")
    for i, r in enumerate(ranked, 1):
        if "error" in r:
            print(f"  {i}. {r['ticker']:6s} — {r['error']}")
            continue
        s = r["stats"]
        badge = "✅" if s["passed"] else "❌"
        print(f"  {i}. {r['ticker']:6s} {badge}  "
              f"WR {s['win_rate']:5.1f}%  PF {s['profit_factor']:.3f}  "
              f"{s['trades_per_month']:5.1f} tpm  "
              f"Score {s['composite_score']:.2f}")

    out = {
        "generated":   datetime.datetime.now().isoformat(),
        "parameters": {
            "partial_pct":      PARTIAL_PCT,
            "gain_target_pct":  GAIN_TARGET_PCT,
            "stop_loss_pct":    STOP_LOSS_PCT,
            "trail_stop_pct":   TRAIL_STOP_PCT,
            "min_score":        MIN_SCORE,
            "pass_wr":          PASS_WR,
            "pass_pf":          PASS_PF,
            "pass_tpm":         PASS_TPM,
        },
        "tickers_screened": SCREEN_TICKERS,
        "results":  ranked,
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {OUT}")


if __name__ == "__main__":
    main()
