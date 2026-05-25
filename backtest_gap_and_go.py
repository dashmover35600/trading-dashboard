"""
NYLO Gap and Go Backtest
========================
Backtests a Gap and Go strategy on AAPL and GOOGL using the same 60-day
dataset (5m + 1m bars) as the main backtest.

Gap and Go rules:
- Gap >= 0.5% from previous close at open
- First 5-min volume >= 2x prior average
- Long if gap up, short if gap down
- Entry at 9:35 AM ET in gap direction
- Target: entry +/- 1x gap size from entry
- Stop:   entry -/+ 0.5x gap size (2:1 R:R)
- Only trades 9:30-10:30 AM ET
- Signals scored using same scorer as v18
"""

import yfinance as yf
import pandas as pd
import ta
import json
import datetime
import pytz
import os
import statistics
import random

MARKET_TZ = pytz.timezone("America/New_York")
TICKERS   = ["AAPL", "GOOGL"]

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

TICKER_CONFIGS = {
    "AAPL": {
        "slippage":     0.0003,
        "pos_mult":     1.0,
        "rsi_bull_min": 52, "rsi_bull_max": 72,
        "rsi_bear_min": 28, "rsi_bear_max": 48,
    },
    "GOOGL": {
        "slippage":     0.00025,
        "pos_mult":     0.85,
        "rsi_bull_min": 53, "rsi_bull_max": 73,
        "rsi_bear_min": 27, "rsi_bear_max": 47,
    },
}

GAP_MIN_PCT      = 0.5              # minimum gap % from prev close
VOL_RATIO_MIN    = 2.0              # first-5-min vol must be 2x+ avg bar vol
TRADE_END        = datetime.time(10, 30)
MIN_SCORE        = 3
DAILY_LOSS_LIMIT = -500.0

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_gap_results.json")


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_earnings_blackout(date, ticker):
    d = date.date() if hasattr(date, "date") else date
    for ed in EARNINGS_DATES.get(ticker, []):
        if abs((d - ed).days) <= EARNINGS_BLACKOUT_DAYS:
            return True
    return False


def get_pos_size(score, ticker):
    sizes = {10: 5000, 9: 4000, 8: 3000, 7: 2000, 6: 1500, 5: 1000, 4: 750, 3: 500}
    base  = sizes.get(min(score, 10), 500)
    return round(base * TICKER_CONFIGS[ticker].get("pos_mult", 1.0))


def apply_slippage(price, direction, slippage):
    return round(price * (1 + slippage) if direction == "long" else price * (1 - slippage), 4)


# ── Data fetching ──────────────────────────────────────────────────────────────

def _process_frames(frames):
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    return df.between_time("09:30", "15:30")


def fetch_60d(ticker):
    """Fetch ~60 days of 5m + 1m bars — same window as the main backtest's recent data."""
    print(f"  Fetching {ticker} (60d 5m + 30d 1m)...")
    end = datetime.datetime.now(MARKET_TZ)

    frames_1m = []
    for i in range(5):          # 5 × 7-day chunks ≈ 35 days
        ce = end - datetime.timedelta(days=i * 7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="1m", progress=False, auto_adjust=True)
            if not df.empty:
                frames_1m.append(df)
        except Exception:
            pass
    df_1m = _process_frames(frames_1m)

    frames_5m = []
    for i in range(9):          # 9 × 7-day chunks ≈ 63 days
        ce = end - datetime.timedelta(days=i * 7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="5m", progress=False, auto_adjust=True)
            if not df.empty:
                frames_5m.append(df)
        except Exception:
            pass
    df_5m = _process_frames(frames_5m)

    if df_1m.empty and df_5m.empty:
        return pd.DataFrame()
    if df_1m.empty:
        combined = df_5m
    elif df_5m.empty:
        combined = df_1m
    else:
        recent_cut  = df_1m.index[0].date()
        df_5m_older = df_5m[df_5m.index.date < recent_cut]
        combined    = _process_frames([df_5m_older, df_1m])

    days = len(combined.index.normalize().unique())
    print(f"  {ticker}: {len(combined)} bars / {days} trading days")
    return combined


# ── Signal scoring (identical logic to v18) ────────────────────────────────────

def score_signal(direction, rsi, vol_ratio, price, vwap, ema9,
                 day_bias, cfg, momentum_ok, vol_rising, gap_pct):
    score = 0

    if day_bias == direction:
        score += 2
    elif day_bias == "neutral":
        score += 1

    if direction == "long":
        mid = cfg["rsi_bull_min"] + 10
        if abs(rsi - mid) <= 7:
            score += 2
        elif cfg["rsi_bull_min"] <= rsi <= cfg["rsi_bull_max"]:
            score += 1
    else:
        mid = cfg["rsi_bear_max"] - 10
        if abs(rsi - mid) <= 7:
            score += 2
        elif cfg["rsi_bear_min"] <= rsi <= cfg["rsi_bear_max"]:
            score += 1

    if vol_ratio >= 2.5:
        score += 2
    elif vol_ratio >= 1.5:
        score += 1

    if vwap and vwap > 0:
        if direction == "long"  and price > vwap: score += 1
        if direction == "short" and price < vwap: score += 1

    if direction == "long"  and price > ema9: score += 1
    if direction == "short" and price < ema9: score += 1

    # Gap direction confirmation (analogous to vwap_reclaim bonus)
    score += 1

    if momentum_ok:  score += 1
    if vol_rising:   score += 1

    # Prime opening window — Gap & Go entries are always at 9:35
    score += 2

    if gap_pct >  0.3 and direction == "long":  score += 1
    if gap_pct < -0.3 and direction == "short": score += 1

    return min(score, 10)


# ── Core strategy ──────────────────────────────────────────────────────────────

def run_gap_and_go(df, ticker):
    cfg    = TICKER_CONFIGS[ticker]
    trades = []
    dates  = sorted(df.index.normalize().unique())

    # Pre-compute cross-bar indicators for accurate lookbacks
    full_cl  = df["Close"].squeeze()
    full_hi  = df["High"].squeeze()
    full_lo  = df["Low"].squeeze()
    pre_rsi  = ta.momentum.RSIIndicator(full_cl, window=14).rsi()
    pre_ema9 = full_cl.ewm(span=9, adjust=False).mean()

    def lookup(series, ts, default):
        try:
            v = series.asof(ts)
            return default if pd.isna(v) else float(v)
        except Exception:
            return default

    daily_pnl = 0.0

    for date in dates:
        d_date    = date.date()
        daily_pnl = 0.0     # reset each day

        if is_earnings_blackout(d_date, ticker):
            continue

        day_df = df[df.index.date == d_date]
        if len(day_df) < 5:
            continue

        # Previous close
        prev_days = df[df.index.date < d_date]
        if prev_days.empty:
            continue
        prev_close = float(prev_days["Close"].squeeze().iloc[-1])

        # Gap from open
        open_price = float(day_df["Open"].squeeze().iloc[0])
        gap_pct    = (open_price - prev_close) / prev_close * 100

        if abs(gap_pct) < GAP_MIN_PCT:
            continue

        direction = "long" if gap_pct > 0 else "short"

        # First 5-minute volume (9:30–9:34 inclusive)
        first5     = day_df.between_time("09:30", "09:34")
        if first5.empty:
            first5 = day_df.iloc[:1]
        first5_vol = float(first5["Volume"].squeeze().sum())

        # Average per-bar volume from prior days (last 200 bars, same resolution)
        avg_vol   = float(prev_days["Volume"].values[-200:].mean()) if not prev_days.empty else 1.0
        vol_ratio = round(first5_vol / avg_vol, 2) if avg_vol > 0 else 1.0

        if vol_ratio < VOL_RATIO_MIN:
            continue

        # Daily loss guard
        if daily_pnl <= DAILY_LOSS_LIMIT:
            continue

        # Entry bar: first bar at or after 9:35
        after_935 = day_df[day_df.index.time >= datetime.time(9, 35)]
        after_935 = after_935[after_935.index.time <= TRADE_END]
        if after_935.empty:
            continue
        entry_ts  = after_935.index[0]
        entry_bar = after_935.iloc[0]

        _open_val   = float(entry_bar["Open"].iloc[0]) if hasattr(entry_bar["Open"], "iloc") else float(entry_bar["Open"])
        _close_val  = float(entry_bar["Close"].iloc[0]) if hasattr(entry_bar["Close"], "iloc") else float(entry_bar["Close"])
        entry_raw   = _open_val if _open_val > 0 else _close_val
        entry_price = apply_slippage(entry_raw, direction, cfg["slippage"])

        # Gap size anchored to open vs prev close (fixed, not price-dependent)
        gap_size = abs(open_price - prev_close)
        if gap_size <= 0:
            continue

        if direction == "long":
            target = round(entry_price + gap_size, 4)
            stop   = round(entry_price - gap_size * 0.5, 4)
        else:
            target = round(entry_price - gap_size, 4)
            stop   = round(entry_price + gap_size * 0.5, 4)

        # VWAP at entry
        cl = day_df["Close"].squeeze(); hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze();   vo = day_df["Volume"].squeeze()
        tp       = (hi + lo + cl) / 3
        vwap_ser = ((tp * vo).cumsum() / vo.cumsum())
        vwap     = lookup(vwap_ser, entry_ts, 0.0)

        rsi  = lookup(pre_rsi,  entry_ts, 50.0)
        ema9 = lookup(pre_ema9, entry_ts, entry_price)

        # Day bias from opening drive (same as v18)
        drive_df    = day_df.between_time("09:30", "09:35")
        drive_open  = float(drive_df["Open"].squeeze().iloc[0])  if not drive_df.empty else open_price
        drive_close = float(drive_df["Close"].squeeze().iloc[-1]) if not drive_df.empty else open_price
        move_pct    = (drive_close - drive_open) / drive_open * 100
        drive_vol   = float(drive_df["Volume"].squeeze().sum())  if not drive_df.empty else first5_vol
        vr_drive    = drive_vol / avg_vol if avg_vol > 0 else 1.0
        if   move_pct >  0.3 and vr_drive > 1.5: day_bias = "long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias = "short"
        else:                                      day_bias = "neutral"

        # momentum_ok: is price moving in gap direction at entry?
        bars_before = day_df[day_df.index < entry_ts]
        if len(bars_before) >= 2:
            cl_list = bars_before["Close"].squeeze().tolist()
            momentum_ok = (cl_list[-1] > cl_list[-2]) if direction == "long" else (cl_list[-1] < cl_list[-2])
        else:
            momentum_ok = True

        first5_last_vol = first5["Volume"].values[-1] if not first5.empty else 0
        entry_vol       = after_935.iloc[0]["Volume"]
        try:
            vol_rising = float(entry_vol) > float(first5_last_vol)
        except Exception:
            vol_rising = True

        sc = score_signal(direction, rsi, vol_ratio, entry_price, vwap, ema9,
                          day_bias, cfg, momentum_ok, vol_rising, gap_pct)
        if sc < MIN_SCORE:
            continue

        pos_size = get_pos_size(sc, ticker)

        # Simulate: walk bars after entry until target / stop / 10:30
        trade_bars = after_935.iloc[1:]   # skip the entry bar itself
        result     = "Time Exit"
        exit_price = float(after_935.iloc[-1]["Close"]) if not after_935.empty else entry_raw

        for _, row in trade_bars.iterrows():
            hi2 = float(row["High"])
            lo2 = float(row["Low"])

            if direction == "long":
                if hi2 >= target:
                    exit_price = target; result = "Target Hit"; break
                if lo2 <= stop:
                    exit_price = stop;   result = "Stop Loss Hit"; break
            else:
                if lo2 <= target:
                    exit_price = target; result = "Target Hit"; break
                if hi2 >= stop:
                    exit_price = stop;   result = "Stop Loss Hit"; break

        # Exit slippage
        slip = cfg["slippage"]
        exit_price = round(exit_price * (1 - slip if direction == "long" else 1 + slip), 4)

        if direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        pnl_dollar = round(pnl_pct / 100 * pos_size, 2)
        daily_pnl += pnl_dollar

        trades.append({
            "date":         d_date.strftime("%Y-%m-%d"),
            "ticker":       ticker,
            "direction":    "Long" if direction == "long" else "Short",
            "entry":        entry_price,
            "exit":         exit_price,
            "result":       result,
            "pnl_pct":      round(pnl_pct, 3),
            "pnl_dollar":   pnl_dollar,
            "rsi":          round(rsi, 1),
            "vol_ratio":    vol_ratio,
            "gap_pct":      round(gap_pct, 3),
            "gap_size":     round(gap_size, 4),
            "signal_score": sc,
            "pos_size":     pos_size,
            "entry_time":   entry_ts.strftime("%I:%M %p"),
            "hour":         9,
            "day_bias":     day_bias,
            "target":       round(target, 4),
            "stop":         round(stop, 4),
        })

    return trades


# ── Statistics (same logic as v18) ────────────────────────────────────────────

def calc_stats(trades):
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_pct": 0, "total_pnl_dollar": 0, "best": 0, "worst": 0,
                "max_drawdown": 0, "sharpe": 0, "avg_score": 0, "profit_factor": 0,
                "expectancy": 0, "by_ticker": {}, "avg_win": 0, "avg_loss": 0}

    wins   = [t for t in trades if t["result"] == "Target Hit"]
    losses = [t for t in trades if t["result"] == "Stop Loss Hit" or
              (t["result"] == "Time Exit" and t["pnl_pct"] <= 0)]
    pnls   = [t["pnl_pct"]    for t in trades]
    dols   = [t["pnl_dollar"] for t in trades]

    peak = 0; dd = 0; cum = 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if cum - peak < dd: dd = cum - peak

    sharpe = 0
    if len(pnls) > 1:
        try:
            sharpe = round(statistics.mean(pnls) / statistics.stdev(pnls) * (252 ** 0.5), 3)
        except Exception:
            pass

    gw      = sum(t["pnl_dollar"] for t in wins)
    gl      = abs(sum(t["pnl_dollar"] for t in losses))
    pf      = round(gw / gl, 3) if gl > 0 else 999
    wr      = len(wins) / len(trades)
    avg_win  = round(gw / len(wins),   2) if wins   else 0
    avg_loss = round(gl / len(losses), 2) if losses else 0
    exp      = round(wr * avg_win - (1 - wr) * avg_loss, 2)

    by_ticker = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in by_ticker:
            by_ticker[tk] = {"trades": 0, "pnl": 0.0, "wins": 0}
        by_ticker[tk]["trades"] += 1
        by_ticker[tk]["pnl"]    += t["pnl_dollar"]
        if t in wins:
            by_ticker[tk]["wins"] += 1

    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr * 100, 2),
        "total_pnl_pct":    round(sum(pnls), 3),
        "total_pnl_dollar": round(sum(dols), 2),
        "best":  round(max(pnls), 3) if pnls else 0,
        "worst": round(min(pnls), 3) if pnls else 0,
        "max_drawdown":  round(dd, 3),
        "sharpe":        sharpe,
        "avg_score":     round(sum(t["signal_score"] for t in trades) / len(trades), 1),
        "profit_factor": pf,
        "expectancy":    exp,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "by_ticker":     by_ticker,
    }


def monte_carlo(trades, n_simulations=1000):
    if len(trades) < 10:
        return {}
    pnls    = [t["pnl_dollar"] for t in trades]
    results = []
    for _ in range(n_simulations):
        sample = random.choices(pnls, k=len(pnls))
        cum = 0; peak = 0; dd = 0
        for p in sample:
            cum += p
            if cum > peak: peak = cum
            if cum - peak < dd: dd = cum - peak
        results.append({"total": round(cum, 2), "max_dd": round(dd, 2)})
    totals = [r["total"]  for r in results]
    dds    = [r["max_dd"] for r in results]
    return {
        "simulations":    n_simulations,
        "median_pnl":     round(statistics.median(totals), 2),
        "pct_profitable": round(sum(1 for t in totals if t > 0) / n_simulations * 100, 1),
        "worst_case_dd":  round(min(dds), 2),
        "best_case":      round(max(totals), 2),
        "worst_case":     round(min(totals), 2),
        "pct_10":         round(sorted(totals)[int(n_simulations * 0.1)], 2),
        "pct_90":         round(sorted(totals)[int(n_simulations * 0.9)], 2),
    }


def direction_breakdown(trades):
    """P&L split by gap direction (long gaps vs short gaps)."""
    result = {}
    for label, dirn in [("long_gap", "Long"), ("short_gap", "Short")]:
        group = [t for t in trades if t["direction"] == dirn]
        if not group:
            result[label] = {"trades": 0}
            continue
        wins = [t for t in group if t["result"] == "Target Hit"]
        result[label] = {
            "trades":     len(group),
            "wins":       len(wins),
            "win_rate":   round(len(wins) / len(group) * 100, 1),
            "pnl_dollar": round(sum(t["pnl_dollar"] for t in group), 2),
            "avg_gap_pct": round(statistics.mean(abs(t["gap_pct"]) for t in group), 3),
        }
    return result


def gap_size_buckets(trades):
    """Win rate by gap magnitude bucket."""
    buckets = {
        "0.5-1.0%": [], "1.0-2.0%": [], "2.0%+": [],
    }
    for t in trades:
        g = abs(t["gap_pct"])
        if g < 1.0:   buckets["0.5-1.0%"].append(t)
        elif g < 2.0: buckets["1.0-2.0%"].append(t)
        else:         buckets["2.0%+"].append(t)
    result = {}
    for label, group in buckets.items():
        if not group:
            result[label] = {"trades": 0}
            continue
        wins = [t for t in group if t["result"] == "Target Hit"]
        result[label] = {
            "trades":     len(group),
            "wins":       len(wins),
            "win_rate":   round(len(wins) / len(group) * 100, 1),
            "pnl_dollar": round(sum(t["pnl_dollar"] for t in group), 2),
        }
    return result


def print_comparison(gap_stats, v18_stats):
    print("\n  ── Side-by-side comparison ────────────────────────────")
    metrics = [
        ("Trades",       "trades",            "d",    False),
        ("Win Rate %",   "win_rate",          ".1f",  True),
        ("Total P&L $",  "total_pnl_dollar",  "+.2f", True),
        ("Sharpe",       "sharpe",            ".3f",  True),
        ("Profit Factor","profit_factor",     ".3f",  True),
        ("Expectancy $", "expectancy",        "+.2f", True),
        ("Max Drawdown%","max_drawdown",      ".2f",  False),
        ("Avg Win $",    "avg_win",           ".2f",  True),
        ("Avg Loss $",   "avg_loss",          ".2f",  False),
    ]
    print(f"  {'Metric':<18} {'Gap & Go':>12} {'v18 Strategy':>14}")
    print(f"  {'-'*18} {'-'*12} {'-'*14}")
    for label, key, fmt, higher_better in metrics:
        gv = gap_stats.get(key, 0)
        vv = v18_stats.get(key, 0) if v18_stats else "—"
        if v18_stats:
            try:
                g_str = f"{gv:{fmt}}"
                v_str = f"{vv:{fmt}}"
                if higher_better:
                    winner = " <" if gv > vv else (" >" if vv > gv else "  ")
                else:
                    winner = " <" if gv < vv else (" >" if vv < gv else "  ")
                print(f"  {label:<18} {g_str:>12}{winner} {v_str:>14}")
            except Exception:
                print(f"  {label:<18} {str(gv):>12}   {str(vv):>14}")
        else:
            print(f"  {label:<18} {gv:>12}   {'—':>14}")
    print(f"  {'(< = Gap&Go better)':<44}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NYLO Gap and Go Backtest")
    print(f"  Tickers  : {', '.join(TICKERS)}")
    print(f"  Data     : 60-day window (5m + 1m bars)")
    print(f"  Rules    : Gap >= {GAP_MIN_PCT}% | 1st-5m vol >= {VOL_RATIO_MIN}x avg")
    print(f"             Entry 9:35 AM | Target 1x gap | Stop 0.5x (2:1)")
    print(f"             Window 9:30-10:30 AM | Min score {MIN_SCORE}/10")
    print("=" * 60)

    # Fetch data
    data = {}
    for ticker in TICKERS:
        data[ticker] = fetch_60d(ticker)

    # Run Gap and Go
    print("\n[1/3] Running Gap and Go strategy...")
    all_trades = []
    for ticker in TICKERS:
        if data[ticker].empty:
            print(f"  {ticker}: no data fetched")
            continue
        t = run_gap_and_go(data[ticker], ticker)
        all_trades.extend(t)
        wins = sum(1 for x in t if x["result"] == "Target Hit")
        wr   = round(wins / len(t) * 100, 1) if t else 0
        pnl  = round(sum(x["pnl_dollar"] for x in t), 2)
        print(f"  {ticker}: {len(t)} trades | {wr}% WR | ${pnl:+.2f}")

    all_trades.sort(key=lambda t: (t["date"], t["entry_time"]))
    gs = calc_stats(all_trades)

    print(f"\n  Combined totals:")
    print(f"  Trades: {gs['trades']} | WR: {gs['win_rate']}% | "
          f"P&L: ${gs['total_pnl_dollar']:+.2f} | Sharpe: {gs['sharpe']} | "
          f"PF: {gs['profit_factor']} | E: ${gs['expectancy']:+.2f}/trade")
    print(f"  Max DD: {gs['max_drawdown']}% | Avg win: ${gs['avg_win']} | Avg loss: ${gs['avg_loss']}")

    # Direction breakdown
    dir_bk = direction_breakdown(all_trades)
    print(f"\n  By gap direction:")
    for label, d in dir_bk.items():
        if d.get("trades", 0) == 0:
            print(f"    {label}: 0 trades")
        else:
            print(f"    {label}: {d['trades']} trades | {d['win_rate']}% WR | "
                  f"${d['pnl_dollar']:+.2f} | avg gap {d['avg_gap_pct']:.2f}%")

    # Gap size buckets
    size_bk = gap_size_buckets(all_trades)
    print(f"\n  By gap magnitude:")
    for label, d in size_bk.items():
        if d.get("trades", 0) == 0:
            print(f"    {label}: 0 trades")
        else:
            print(f"    {label}: {d['trades']} trades | {d['win_rate']}% WR | ${d['pnl_dollar']:+.2f}")

    # Monte Carlo
    print(f"\n[2/3] Monte Carlo (1000 sims)...")
    mc = monte_carlo(all_trades)
    if mc:
        print(f"  Median P&L: ${mc['median_pnl']} | Profitable sims: {mc['pct_profitable']}%")
        print(f"  10th/90th pct: ${mc['pct_10']} / ${mc['pct_90']}")
        print(f"  Worst draw-down: ${mc['worst_case_dd']}")
    else:
        print("  Not enough trades for Monte Carlo simulation")

    # Load v18 for comparison
    print(f"\n[3/3] Comparing with v18 (VWAP Reclaim + EMA Pullback)...")
    v18_path  = os.path.join(BASE, "backtest_results.json")
    v18_stats = None
    if os.path.exists(v18_path):
        try:
            with open(v18_path) as f:
                raw = json.load(f)
            v18_stats = raw.get("baseline", {}).get("stats", {})
            v18_ver   = raw.get("version", "v18")
            print(f"  Loaded {v18_ver} results ({v18_stats.get('trades', '?')} trades, "
                  f"full-year window)")
            note = ("Note: v18 covers a full-year hybrid dataset; Gap & Go uses 60-day "
                    "window only. Compare directionally, not as exact equivalents.")
            print(f"  ⚠  {note}")
        except Exception as e:
            print(f"  Could not load v18 results: {e}")
    else:
        print(f"  v18 results not found — run backtest.py first")

    print_comparison(gs, v18_stats)

    # Verdict
    verdict = "insufficient_data"
    if v18_stats and gs["trades"] >= 5:
        gap_wr_better = gs["win_rate"]       > v18_stats.get("win_rate",       0)
        gap_pf_better = gs["profit_factor"]  > v18_stats.get("profit_factor",  0)
        gap_ex_better = gs["expectancy"]     > v18_stats.get("expectancy",      0)
        wins_count    = sum([gap_wr_better, gap_pf_better, gap_ex_better])
        if wins_count >= 2:
            verdict = "Gap & Go outperforms v18 on majority of key metrics"
        else:
            verdict = "v18 outperforms Gap & Go on majority of key metrics"
        print(f"\n  Verdict: {verdict}")

    # Save
    out = {
        "generated_at":  datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str": datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "version":       "gap_and_go_v1",
        "config": {
            "tickers":       TICKERS,
            "data_window":   "60-day (5m + 1m bars)",
            "gap_min_pct":   GAP_MIN_PCT,
            "vol_ratio_min": VOL_RATIO_MIN,
            "entry_time":    "9:35 AM ET",
            "target_ratio":  "1x gap size",
            "stop_ratio":    "0.5x gap size (2:1 R:R)",
            "trade_window":  "9:30-10:30 AM ET",
            "min_score":     MIN_SCORE,
            "slippage":      {"AAPL": "0.03%", "GOOGL": "0.025%"},
            "earnings_blackout_days": EARNINGS_BLACKOUT_DAYS,
        },
        "gap_and_go": {
            "stats":               gs,
            "trades":              all_trades,
            "direction_breakdown": dir_bk,
            "gap_size_buckets":    size_bk,
            "monte_carlo":         mc,
        },
        "comparison": {
            "gap_and_go":  gs,
            "v18_strategy": v18_stats or {},
            "note": ("v18 covers full-year hybrid dataset; Gap & Go uses 60-day window. "
                     "Compare directionally."),
            "verdict": verdict,
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Results saved → {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
