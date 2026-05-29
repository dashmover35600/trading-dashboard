"""
NYLO Afternoon Session Backtest
================================
Tests VWAP Reclaim + EMA Pullback + Opening Drive strategy in the
1:00 PM – 3:00 PM ET window on AAPL and GOOGL.

Exit parameters (matches live agent):
  Target  : +2.0%
  Stop    : -1.0%
  Partial : +1.25% (50% off)
  Trail   : 0.50% from peak (activates at +1.25%)
  Breakeven: +0.50%

Results broken down by sub-hour window:
  • 1:00–2:00 PM
  • 2:00–3:00 PM

Usage:
    python3 -W ignore backtest_afternoon.py

Output: backtest_afternoon_results.json
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
TICKERS    = ["AAPL", "GOOGL"]
MARKET_TZ  = pytz.timezone("America/New_York")
BASE       = os.path.dirname(os.path.abspath(__file__))
OUT        = os.path.join(BASE, "backtest_afternoon_results.json")

# Entry window
TRADE_START = datetime.time(13, 0)   # 1:00 PM — no new entries before this
TRADE_END   = datetime.time(15, 0)   # 3:00 PM — no new entries after this
HARD_CLOSE  = datetime.time(15, 25)  # force-exit all positions

# Exit parameters (matches live agent)
GAIN_TARGET_PCT   = 0.020    # +2.0%
STOP_LOSS_PCT     = 0.010    # -1.0%
PARTIAL_PCT       = 0.0125   # +1.25% partial exit
TRAIL_TRIGGER_PCT = 0.0125   # activate trail at +1.25%
TRAIL_STOP_PCT    = 0.005    # 0.50% trail from peak
BREAKEVEN_TRIGGER = 0.005    # move stop to BE at +0.5%

MIN_SCORE        = 3
DAILY_LOSS_LIMIT = -500.0
CONSEC_LOSS_PAUSE = 3

# Per-ticker configs (mirrors live agent + backtest.py)
TICKER_CONFIGS = {
    "AAPL":  {"strategy":"ema_pullback","rsi_bull_min":52,"rsi_bull_max":72,
              "rsi_bear_min":28,"rsi_bear_max":48,"vol_min":1.2,
              "slippage":0.0003,"pos_mult":1.0},
    "GOOGL": {"strategy":"both","rsi_bull_min":53,"rsi_bull_max":73,
              "rsi_bear_min":27,"rsi_bear_max":47,"vol_min":1.2,
              "slippage":0.00025,"pos_mult":0.85},
}

EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_DATES = {
    "AAPL":  [datetime.date(2025, 5,  1), datetime.date(2025, 7, 31),
              datetime.date(2025,10, 30), datetime.date(2026, 1, 30),
              datetime.date(2026, 5,  1)],
    "GOOGL": [datetime.date(2025, 4, 29), datetime.date(2025, 7, 29),
              datetime.date(2025,10, 29), datetime.date(2026, 2,  4),
              datetime.date(2026, 4, 29)],
}

# ── Data fetch ─────────────────────────────────────────────────────────────────

def _process(frames):
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    return df.between_time("09:30", "15:30")


def fetch_hybrid(ticker: str) -> pd.DataFrame:
    """Hybrid fetch: 1h ~200 days + 5m last 63d + 1m last 35d."""
    end      = datetime.datetime.now(MARKET_TZ)
    start_1y = end - datetime.timedelta(days=200)

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

    frames_1m = []
    for i in range(5):
        ce = end - datetime.timedelta(days=i * 7)
        cs = ce  - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="1m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                frames_1m.append(df)
        except Exception:
            pass
    df_1m = _process(frames_1m)

    frames_5m = []
    for i in range(9):
        ce = end - datetime.timedelta(days=i * 7)
        cs = ce  - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="5m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                frames_5m.append(df)
        except Exception:
            pass
    df_5m = _process(frames_5m)

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
        if abs(rsi - mid) <= 7:                                 score += 2
        elif cfg["rsi_bull_min"] <= rsi <= cfg["rsi_bull_max"]: score += 1
    else:
        mid = (cfg["rsi_bear_min"] + cfg["rsi_bear_max"]) / 2
        if abs(rsi - mid) <= 7:                                 score += 2
        elif cfg["rsi_bear_min"] <= rsi <= cfg["rsi_bear_max"]: score += 1

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
    # Afternoon bonus: 1:00–1:30 PM is the post-lunch momentum window
    if hour == 13 and minute <= 30: score += 1
    return min(score, 10)


def _pos_size(score: int, ticker: str) -> float:
    base = {10:5000,9:4000,8:3000,7:2000,6:1500,5:1000,4:750,3:500}.get(min(score,10), 500)
    return round(base * TICKER_CONFIGS[ticker].get("pos_mult", 1.0))


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
        hourly = n_bars < 15

        # Need at least some afternoon bars
        pm_df = day_df.between_time("13:00", "15:30")
        if len(pm_df) < 2:
            continue

        closes  = day_df["Close"].squeeze().tolist()
        volumes = day_df["Volume"].squeeze().tolist()
        day_idx = list(day_df.index)

        # Opening drive bias — always computed from 9:30 open
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

        # Full-day cumulative VWAP (anchored at 9:30 open as in live agent)
        cl = day_df["Close"].squeeze(); hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze();   vo = day_df["Volume"].squeeze()
        tp = (hi + lo + cl) / 3
        cum_vol = vo.cumsum()
        vwap_list = ((tp * vo).cumsum() / cum_vol.replace(0, float("nan"))).ffill().tolist()

        # Vol baseline: 20-bar rolling for minute bars; morning avg for hourly
        if hourly:
            morn_df     = day_df.between_time("09:30", "12:00")
            day_avg_vol = float(morn_df["Volume"].mean()) if not morn_df.empty else float(day_df["Volume"].mean())
        else:
            day_avg_vol = None

        daily_pnl = 0.0; consec_losses = 0; pause_until = None
        in_trade  = False; entry = None; partial_done = False
        last_exit_ts = None  # 15-min cooldown between trades

        for i, ts in enumerate(day_idx):
            hour, minute = ts.hour, ts.minute
            t_now = ts.time()

            # Skip pre-afternoon bars (but keep processing for exits)
            if t_now < TRADE_START and not in_trade:
                continue
            if t_now > HARD_CLOSE:
                # Force-exit at hard close
                if in_trade and entry:
                    price_hc  = closes[i] if i < len(closes) else closes[-1]
                    ep        = apply_slippage(price_hc, entry["dir"], slip)
                    d         = entry["dir"]
                    remaining = 0.5 if partial_done else 1.0
                    size      = entry["pos_size"] * remaining
                    pnl_pct   = ((ep - entry["price"]) / entry["price"] * 100 if d == "long"
                                 else (entry["price"] - ep) / entry["price"] * 100)
                    daily_pnl += size * pnl_pct / 100
                    trades.append(_make_trade({**entry, "pos_size": size},
                                              ep, "Time Exit", d_date, ts, slip))
                    in_trade = False; entry = None; partial_done = False
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

            # ── Exit logic ──────────────────────────────────────────────────
            if in_trade and entry:
                d = entry["dir"]

                if not entry.get("breakeven_set", False):
                    move = ((price - entry["price"]) / entry["price"] if d == "long"
                            else (entry["price"] - price) / entry["price"])
                    if move >= BREAKEVEN_TRIGGER:
                        entry["stop"]          = entry["price"]
                        entry["breakeven_set"] = True

                partial_tgt = (entry["price"] * (1 + PARTIAL_PCT) if d == "long"
                               else entry["price"] * (1 - PARTIAL_PCT))
                if not partial_done:
                    if ((d == "long"  and price >= partial_tgt) or
                            (d == "short" and price <= partial_tgt)):
                        pnl_pct    = ((partial_tgt - entry["price"]) / entry["price"] * 100
                                      if d == "long"
                                      else (entry["price"] - partial_tgt) / entry["price"] * 100)
                        daily_pnl += entry["pos_size"] * 0.5 * pnl_pct / 100
                        trades.append(_make_trade(entry, partial_tgt, "Partial Exit",
                                                  d_date, ts, slip, partial=True))
                        partial_done          = True
                        entry["trail_active"] = True
                        entry["trail_peak"]   = price

                if entry["trail_active"]:
                    if d == "long" and price > entry.get("trail_peak", price):
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = price * (1 - TRAIL_STOP_PCT)
                    elif d == "short" and price < entry.get("trail_peak", price):
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = price * (1 + TRAIL_STOP_PCT)
                elif not partial_done:
                    move = ((price - entry["price"]) / entry["price"] if d == "long"
                            else (entry["price"] - price) / entry["price"])
                    if move >= TRAIL_TRIGGER_PCT:
                        entry["trail_active"]   = True
                        entry["trail_peak"]     = price
                        entry["trail_stop_val"] = (price * (1 - TRAIL_STOP_PCT) if d == "long"
                                                   else price * (1 + TRAIL_STOP_PCT))

                tsv   = entry.get("trail_stop_val")
                ht    = ((d == "long"  and price >= entry["target"]) or
                         (d == "short" and price <= entry["target"]))
                hs    = ((d == "long"  and price <= entry["stop"]) or
                         (d == "short" and price >= entry["stop"]))
                htr   = (entry["trail_active"] and tsv is not None and
                         ((d == "long" and price <= tsv) or (d == "short" and price >= tsv)))
                htime = t_now >= TRADE_END and not (ht or hs or htr)

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
                    daily_pnl += size * pnl_pct / 100
                    trades.append(_make_trade({**entry, "pos_size": size},
                                              ep, res, d_date, ts, slip))
                    if pnl_pct <= 0:
                        consec_losses += 1
                        if consec_losses >= CONSEC_LOSS_PAUSE:
                            pause_until   = ts + datetime.timedelta(minutes=30)
                            consec_losses = 0
                    else:
                        consec_losses = 0
                    last_exit_ts = ts
                    in_trade = False; entry = None; partial_done = False
                    continue

            if in_trade:
                continue

            # No new entries after 3:00 PM
            if t_now >= TRADE_END:
                continue

            # 15-min cooldown between trades
            if last_exit_ts and (ts - last_exit_ts).total_seconds() < 900:
                continue

            # ── Signal detection ────────────────────────────────────────────
            rsi   = lookup(pre_rsi,   ts, 50.0)
            ema9  = lookup(pre_ema9,  ts, price)
            ema21 = lookup(pre_ema21, ts, price)

            momentum_long  = prev_price > prev2 and price > prev_price
            momentum_short = prev_price < prev2 and price < prev_price
            vol_rising     = volumes[i] > volumes[i - 1] if i > 0 else False

            if hourly:
                avg_v = day_avg_vol if day_avg_vol and day_avg_vol > 0 else 1.0
            else:
                vol_sl = pd.Series(volumes[max(0, i - 20):i])
                avg_v  = float(vol_sl.mean()) if len(vol_sl) > 0 else 1.0
            vol_ratio = round(volumes[i] / avg_v, 2) if avg_v > 0 else 1.0

            vol_min_eff  = 0.0 if hourly else cfg["vol_min"]
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
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "vwap_reclaim"; dirn = "long"; momentum_ok = True
                    elif price < vwap and momentum_short and vol_ratio >= vol_min_eff:
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "vwap_reclaim"; dirn = "short"; momentum_ok = True
                else:
                    if closes[i-1] < vwap and price > vwap and vol_ratio >= vol_min_eff:
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "vwap_reclaim"; dirn = "long"; momentum_ok = momentum_long
                    elif closes[i-1] > vwap and price < vwap and vol_ratio >= vol_min_eff:
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "vwap_reclaim"; dirn = "short"; momentum_ok = momentum_short

            # EMA Pullback
            if not sig and strategy in ("ema_pullback", "both") and i > 0:
                if hourly:
                    if (price > ema9 and price > ema21 and momentum_long and
                            vol_ratio >= vol_min_eff):
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "ema_pullback"; dirn = "long"; momentum_ok = True
                    elif (price < ema9 and price < ema21 and momentum_short and
                              vol_ratio >= vol_min_eff):
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "ema_pullback"; dirn = "short"; momentum_ok = True
                else:
                    if (closes[i-1] <= ema9 * 1.001 and price > ema9 and price > ema21 and
                            vol_ratio >= vol_min_eff and momentum_long):
                        if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                            sig = "ema_pullback"; dirn = "long"; momentum_ok = True
                    elif (closes[i-1] >= ema9 * 0.999 and price < ema9 and price < ema21 and
                              vol_ratio >= vol_min_eff and momentum_short):
                        if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                            sig = "ema_pullback"; dirn = "short"; momentum_ok = True

            if not sig:
                continue

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap,
                              ema9, day_bias, cfg, momentum_ok, vol_rising, hour, minute)
            if sc < MIN_SCORE:
                continue

            entry_price = apply_slippage(price, dirn, slip)
            pos_size    = _pos_size(sc, ticker)
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
                "hour": hour, "minute": minute, "day_bias": day_bias,
            }

    return trades


def _make_trade(entry, exit_price, result, date, ts, slip, partial=False):
    d       = entry["dir"]
    exit_p  = round(exit_price * (1 - slip if d == "long" else 1 + slip), 4)
    pnl_pct = ((exit_p - entry["price"]) / entry["price"] * 100 if d == "long"
               else (entry["price"] - exit_p) / entry["price"] * 100)
    size    = entry["pos_size"]
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
        "entry_time":   entry["time"],
        "hour":         entry["hour"],
        "day_bias":     entry["day_bias"],
    }


# ── Statistics ─────────────────────────────────────────────────────────────────

def calc_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "expectancy": 0, "sharpe": 0,
                "avg_win": 0, "avg_loss": 0, "max_drawdown": 0}

    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    pnls   = [t["pnl_dollar"] for t in trades]
    n      = len(trades)
    wr     = len(wins) / n * 100

    gw = sum(t["pnl_dollar"] for t in wins)
    gl = abs(sum(t["pnl_dollar"] for t in losses))
    pf = round(gw / gl, 3) if gl > 0 else 999.0

    total_pnl = sum(pnls)
    exp       = round(total_pnl / n, 2)

    sharpe = 0.0
    if n > 1:
        try:
            pnl_pcts = [t["pnl_pct"] for t in trades]
            sharpe = round(statistics.mean(pnl_pcts) /
                           statistics.stdev(pnl_pcts) * math.sqrt(252), 3)
        except Exception:
            pass

    equity = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak: peak = equity
        dd = peak - equity
        if dd > max_dd:   max_dd = dd

    exit_reasons = {}
    for t in trades:
        r = t["result"].replace(" (partial)", "")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        "n_trades":      n,
        "win_rate":      round(wr, 2),
        "profit_factor": pf,
        "total_pnl":     round(total_pnl, 2),
        "expectancy":    exp,
        "sharpe":        sharpe,
        "max_drawdown":  round(max_dd, 2),
        "avg_win":       round(gw / len(wins),   2) if wins   else 0.0,
        "avg_loss":      round(gl / len(losses), 2) if losses else 0.0,
        "exit_reasons":  exit_reasons,
    }


def hour_breakdown(trades: list[dict]) -> dict:
    buckets = {
        "1:00-2:00 PM": [t for t in trades if t["hour"] == 13],
        "2:00-3:00 PM": [t for t in trades if t["hour"] == 14],
    }
    return {label: calc_stats(bucket) for label, bucket in buckets.items()}


def monte_carlo(trades: list[dict], n_sims: int = 1000) -> dict:
    pnls = [t["pnl_dollar"] for t in trades]
    if len(pnls) < 5:
        return {}
    results = []
    for _ in range(n_sims):
        shuffled = random.choices(pnls, k=len(pnls))
        results.append(sum(shuffled))
    results.sort()
    profitable = sum(1 for r in results if r > 0) / n_sims * 100
    return {
        "p10":        round(results[int(0.10 * n_sims)], 2),
        "p50":        round(results[int(0.50 * n_sims)], 2),
        "p90":        round(results[int(0.90 * n_sims)], 2),
        "profitable": round(profitable, 1),
        "n_sims":     n_sims,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  NYLO Afternoon Session Backtest")
    print(f"  Tickers  : {', '.join(TICKERS)}")
    print(f"  Window   : 1:00 PM – 3:00 PM ET")
    print(f"  Exits    : target +{GAIN_TARGET_PCT*100:.1f}% | stop -{STOP_LOSS_PCT*100:.1f}%"
          f" | partial +{PARTIAL_PCT*100:.2f}% | trail {TRAIL_STOP_PCT*100:.1f}%")
    print("=" * 62)

    all_trades = []
    per_ticker = {}

    for ticker in TICKERS:
        print(f"\n[{ticker}] Fetching data...")
        df = fetch_hybrid(ticker)
        if df.empty:
            print(f"  [!] No data — skipping.")
            continue
        n_days = df.index.normalize().nunique()
        print(f"  {len(df):,} bars · {n_days} trading days")

        trades = run_strategy(df, ticker)
        print(f"  {len(trades)} trades in 1–3 PM window")
        all_trades.extend(trades)
        per_ticker[ticker] = trades

    print(f"\n{'─'*62}")
    overall = calc_stats(all_trades)
    by_hour = hour_breakdown(all_trades)
    mc      = monte_carlo(all_trades)

    print(f"\n  OVERALL ({len(all_trades)} trades)")
    print(f"  WR {overall['win_rate']:.1f}% | PF {overall['profit_factor']:.3f} | "
          f"Sharpe {overall['sharpe']:.3f} | PnL ${overall['total_pnl']:+.2f} | "
          f"E ${overall['expectancy']:+.2f}/trade")

    print(f"\n  HOUR BREAKDOWN")
    print(f"  {'Window':<18} {'Trades':>6} {'WR':>7} {'PF':>7} {'PnL':>10} {'Avg Win':>9} {'Avg Loss':>9}")
    print(f"  {'─'*18} {'─'*6} {'─'*7} {'─'*7} {'─'*10} {'─'*9} {'─'*9}")
    for label, s in by_hour.items():
        if s["n_trades"] == 0:
            print(f"  {label:<18} {'0':>6} {'—':>7} {'—':>7} {'—':>10}")
            continue
        print(f"  {label:<18} {s['n_trades']:>6} {s['win_rate']:>6.1f}% "
              f"{s['profit_factor']:>7.3f} ${s['total_pnl']:>+9.2f} "
              f"${s['avg_win']:>8.2f} ${s['avg_loss']:>8.2f}")

    print(f"\n  PER TICKER")
    for ticker, trades in per_ticker.items():
        s = calc_stats(trades)
        if s["n_trades"]:
            print(f"  {ticker}: {s['n_trades']} trades | WR {s['win_rate']:.1f}% | "
                  f"PF {s['profit_factor']:.3f} | PnL ${s['total_pnl']:+.2f}")

    if mc:
        print(f"\n  MONTE CARLO ({mc['n_sims']} sims)")
        print(f"  p10 ${mc['p10']:+.0f} | p50 ${mc['p50']:+.0f} | p90 ${mc['p90']:+.0f} | "
              f"Profitable {mc['profitable']:.1f}%")

    print(f"\n  EXIT REASONS: {overall.get('exit_reasons', {})}")

    # Compare morning (9:30-10:00) vs afternoon directly in output
    print(f"\n  MORNING vs AFTERNOON COMPARISON")
    print(f"  {'Session':<22} {'Trades':>6} {'WR':>7} {'PF':>7} {'PnL':>10}")
    print(f"  {'─'*22} {'─'*6} {'─'*7} {'─'*7} {'─'*10}")
    print(f"  {'9:30-10:00 AM (ref)':<22} {'12':>6} {'66.7':>6}% {'3.291':>7} {'$+146':>10}")
    print(f"  {'1:00-3:00 PM (this)':<22} {overall['n_trades']:>6} "
          f"{overall['win_rate']:>6.1f}% {overall['profit_factor']:>7.3f} "
          f"${overall['total_pnl']:>+9.2f}")

    out = {
        "generated":  datetime.datetime.now().isoformat(),
        "parameters": {
            "tickers":        TICKERS,
            "window":         "1:00 PM – 3:00 PM ET",
            "gain_target":    GAIN_TARGET_PCT,
            "stop_loss":      STOP_LOSS_PCT,
            "partial_pct":    PARTIAL_PCT,
            "trail_stop_pct": TRAIL_STOP_PCT,
            "min_score":      MIN_SCORE,
        },
        "overall":    overall,
        "by_hour":    by_hour,
        "by_ticker":  {tk: calc_stats(tr) for tk, tr in per_ticker.items()},
        "monte_carlo": mc,
        "trades":     all_trades,
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {OUT}")


if __name__ == "__main__":
    main()
