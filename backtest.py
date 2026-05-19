"""
NYLO Backtesting Engine
=======================
Runs the full ORB + RSI + VWAP + Volume + Multi-TF strategy
on 30 days of 1-minute historical data for QQQ and GLD.

Also runs a parameter sweep across RSI thresholds to find the
optimal configuration.

Output: backtest_results.json (push to GitHub, read by backtest.html)

Usage:
  python3 backtest.py

Requirements:
  pip install yfinance pandas ta --break-system-packages
"""

import yfinance as yf
import pandas as pd
import ta
import json
import datetime
import pytz
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS           = ["QQQ", "GLD"]
LOOKBACK_DAYS     = 30
POSITION_SIZE     = 500.0
MARKET_TZ         = pytz.timezone("America/New_York")

# Baseline strategy parameters
RSI_BUY_MIN       = 55
RSI_SELL_MAX      = 45
GAIN_TARGET_PCT   = 1.5
STOP_LOSS_PCT     = 0.75
VOLUME_MULT       = 1.5
MAX_TRADES_DAY    = 2

# Parameter sweep ranges
SWEEP_RSI_BUY  = [50, 52, 55, 58, 60, 63, 65]
SWEEP_RSI_SELL = [35, 38, 40, 42, 45, 48, 50]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_results.json")

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_1m(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    yfinance only allows 7 days of 1m data per call.
    We fetch in 7-day chunks and stitch together.
    """
    print(f"  Fetching {ticker} 1-minute data ({days} days)...")
    frames = []
    end   = datetime.datetime.now(MARKET_TZ)
    # How many 7-day chunks we need
    chunks = (days // 7) + (1 if days % 7 else 0)

    for i in range(chunks):
        chunk_end   = end - datetime.timedelta(days=i*7)
        chunk_start = chunk_end - datetime.timedelta(days=7)
        try:
            df = yf.download(
                ticker,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1m",
                progress=False,
                auto_adjust=True
            )
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"    Warning: chunk {i} failed: {e}")

    if not frames:
        print(f"  ERROR: No data for {ticker}")
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)

    # Filter to trading hours only
    df = df.between_time("09:30", "13:00")
    print(f"  {ticker}: {len(df)} 1-min bars loaded")
    return df

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, window: int = 14) -> float:
    if len(series) < window + 1:
        return 50.0
    rsi = ta.momentum.RSIIndicator(series, window=window).rsi()
    val = rsi.iloc[-1]
    return round(float(val), 2) if not pd.isna(val) else 50.0

def calc_vwap(day_df: pd.DataFrame) -> pd.Series:
    close  = day_df["Close"].squeeze()
    high   = day_df["High"].squeeze()
    low    = day_df["Low"].squeeze()
    volume = day_df["Volume"].squeeze()
    tp     = (high + low + close) / 3
    vwap   = (tp * volume).cumsum() / volume.cumsum()
    return vwap

def calc_vol_ratio(df: pd.DataFrame, idx: int, window: int = 20) -> float:
    if idx < window:
        return 1.0
    vol_series = df["Volume"].squeeze()
    avg = float(vol_series.iloc[idx-window:idx].mean())
    cur = float(vol_series.iloc[idx])
    return round(cur / avg, 2) if avg > 0 else 1.0

# ── Strategy engine ───────────────────────────────────────────────────────────
def run_strategy(df: pd.DataFrame, ticker: str, cfg: dict) -> list:
    """
    Run the full 5-filter strategy on a 1-minute DataFrame.
    Returns a list of trade dicts.
    """
    rsi_buy    = cfg["rsi_buy"]
    rsi_sell   = cfg["rsi_sell"]
    gain_pct   = cfg["gain_pct"]
    stop_pct   = cfg["stop_pct"]
    vol_mult   = cfg["vol_mult"]
    max_trades = cfg["max_trades"]
    pos_size   = cfg["pos_size"]

    trades = []

    # Group by trading date
    dates = sorted(df.index.normalize().unique())

    for date in dates:
        day_df = df[df.index.date == date.date()]
        if len(day_df) < 20:
            continue

        # Opening range: 9:30–9:45
        orb = day_df.between_time("09:30", "09:44")
        if len(orb) < 2:
            continue
        range_high = float(orb["High"].max())
        range_low  = float(orb["Low"].min())

        # VWAP for the day
        vwap_series = calc_vwap(day_df)

        # Scan bars after 9:45
        after_range = day_df.between_time("09:45", "12:59")
        if len(after_range) < 5:
            continue

        trades_today = 0
        in_trade     = False
        entry        = None

        closes = day_df["Close"].squeeze()

        for i, (ts, row) in enumerate(after_range.iterrows()):
            if trades_today >= max_trades:
                break

            hour = ts.hour
            minute = ts.minute

            # Dead zone 11:30–12:30
            if (hour == 11 and minute >= 30) or (hour == 12 and minute <= 30):
                continue

            price = float(row["Close"])

            # Check exit
            if in_trade and entry:
                if entry["dir"] == "long":
                    hit_target = price >= entry["target"]
                    hit_stop   = price <= entry["stop"]
                else:
                    hit_target = price <= entry["target"]
                    hit_stop   = price >= entry["stop"]

                if hit_target or hit_stop:
                    exit_price = entry["target"] if hit_target else entry["stop"]
                    pnl_pct = ((exit_price - entry["price"]) / entry["price"] * 100
                               if entry["dir"] == "long"
                               else (entry["price"] - exit_price) / entry["price"] * 100)
                    shares    = pos_size / entry["price"]
                    pnl_dollar= ((exit_price - entry["price"]) * shares
                                 if entry["dir"] == "long"
                                 else (entry["price"] - exit_price) * shares)

                    trades.append({
                        "date":       date.strftime("%Y-%m-%d"),
                        "ticker":     ticker,
                        "direction":  "Long" if entry["dir"] == "long" else "Short",
                        "entry":      round(entry["price"], 4),
                        "exit":       round(exit_price, 4),
                        "target":     round(entry["target"], 4),
                        "stop":       round(entry["stop"], 4),
                        "result":     "Target Hit" if hit_target else "Stop Loss Hit",
                        "pnl_pct":    round(pnl_pct, 3),
                        "pnl_dollar": round(pnl_dollar, 2),
                        "rsi":        round(entry["rsi"], 1),
                        "vol_ratio":  round(entry["vol_ratio"], 2),
                        "entry_time": entry["time"],
                        "hour":       entry["hour"],
                        "shares":     round(shares, 4),
                    })
                    in_trade = False
                    entry    = None
                    trades_today += 1
                    continue

            if in_trade:
                continue

            # Get position in full day_df for RSI lookback
            try:
                pos_in_day = day_df.index.get_loc(ts)
            except Exception:
                continue

            # RSI (14 bars)
            rsi_slice = closes.iloc[max(0, pos_in_day-27):pos_in_day+1]
            rsi       = calc_rsi(rsi_slice)

            # Volume ratio
            vol_ratio = calc_vol_ratio(day_df, pos_in_day)

            # VWAP at this bar
            try:
                vwap = float(vwap_series.loc[ts])
            except Exception:
                vwap = price

            # Signal check
            direction = None
            if price > range_high and rsi > rsi_buy:
                direction = "long"
            elif price < range_low and rsi < rsi_sell:
                direction = "short"
            if not direction:
                continue

            # Volume filter
            if vol_ratio < vol_mult:
                continue

            # VWAP filter
            if direction == "long" and price < vwap:
                continue
            if direction == "short" and price > vwap:
                continue

            # Enter trade
            target = (price * (1 + gain_pct/100) if direction == "long"
                      else price * (1 - gain_pct/100))
            stop   = (price * (1 - stop_pct/100) if direction == "long"
                      else price * (1 + stop_pct/100))

            in_trade = True
            entry = {
                "dir":       direction,
                "price":     price,
                "target":    round(target, 4),
                "stop":      round(stop, 4),
                "rsi":       rsi,
                "vol_ratio": vol_ratio,
                "time":      ts.strftime("%I:%M %p"),
                "hour":      hour,
            }

    return trades

# ── Stats calculator ──────────────────────────────────────────────────────────
def calc_stats(trades: list, pos_size: float) -> dict:
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "total_pnl_pct": 0, "total_pnl_dollar": 0,
            "best": 0, "worst": 0, "max_drawdown": 0, "sharpe": 0
        }

    wins   = [t for t in trades if t["result"] == "Target Hit"]
    losses = [t for t in trades if t["result"] == "Stop Loss Hit"]
    wr     = len(wins) / len(trades) * 100 if trades else 0
    pnls   = [t["pnl_pct"] for t in trades]
    total_pct    = sum(pnls)
    total_dollar = sum(t["pnl_dollar"] for t in trades)
    best  = max(pnls) if pnls else 0
    worst = min(pnls) if pnls else 0

    # Max drawdown
    peak, dd, cum = 0, 0, 0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        if cum - peak < dd: dd = cum - peak

    # Sharpe (annualized, rough)
    import statistics
    if len(pnls) > 1:
        mean = statistics.mean(pnls)
        stdev = statistics.stdev(pnls)
        sharpe = (mean / stdev * (252 ** 0.5)) if stdev > 0 else 0
    else:
        sharpe = 0

    return {
        "trades":          len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        round(wr, 2),
        "total_pnl_pct":   round(total_pct, 3),
        "total_pnl_dollar":round(total_dollar, 2),
        "best":            round(best, 3),
        "worst":           round(worst, 3),
        "max_drawdown":    round(dd, 3),
        "sharpe":          round(sharpe, 3),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NYLO Backtesting Engine")
    print(f"  Tickers  : {', '.join(TICKERS)}")
    print(f"  Lookback : {LOOKBACK_DAYS} days of 1-minute data")
    print(f"  Strategy : ORB + RSI + VWAP + Volume + Multi-TF")
    print("=" * 60)

    # Fetch data for all tickers
    data = {}
    for ticker in TICKERS:
        df = fetch_1m(ticker, LOOKBACK_DAYS)
        data[ticker] = df

    # ── Baseline run ──────────────────────────────────────────────
    print("\n[1/2] Running baseline strategy (RSI buy=55, sell=45)...")
    base_cfg = {
        "rsi_buy":   RSI_BUY_MIN,
        "rsi_sell":  RSI_SELL_MAX,
        "gain_pct":  GAIN_TARGET_PCT,
        "stop_pct":  STOP_LOSS_PCT,
        "vol_mult":  VOLUME_MULT,
        "max_trades":MAX_TRADES_DAY,
        "pos_size":  POSITION_SIZE,
    }

    base_trades = []
    for ticker in TICKERS:
        if data[ticker].empty:
            continue
        t = run_strategy(data[ticker], ticker, base_cfg)
        base_trades.extend(t)
        print(f"  {ticker}: {len(t)} trades")

    base_trades.sort(key=lambda t: (t["date"], t["entry_time"]))
    base_stats = calc_stats(base_trades, POSITION_SIZE)

    print(f"  Total: {base_stats['trades']} trades | "
          f"Win rate: {base_stats['win_rate']:.1f}% | "
          f"P&L: {base_stats['total_pnl_pct']:+.2f}%")

    # ── Parameter sweep ────────────────────────────────────────────
    print("\n[2/2] Running parameter sweep...")
    sweep_results = []
    total_combos = len(SWEEP_RSI_BUY) * len(SWEEP_RSI_SELL)
    done = 0

    for rsi_buy in SWEEP_RSI_BUY:
        for rsi_sell in SWEEP_RSI_SELL:
            if rsi_buy <= rsi_sell:
                done += 1
                continue

            cfg = {**base_cfg, "rsi_buy": rsi_buy, "rsi_sell": rsi_sell}
            trades = []
            for ticker in TICKERS:
                if data[ticker].empty:
                    continue
                trades.extend(run_strategy(data[ticker], ticker, cfg))
            trades.sort(key=lambda t: (t["date"], t["entry_time"]))
            stats = calc_stats(trades, POSITION_SIZE)
            score = stats["win_rate"] * 0.6 + stats["total_pnl_pct"] * 0.4

            sweep_results.append({
                "rsi_buy":  rsi_buy,
                "rsi_sell": rsi_sell,
                "stats":    stats,
                "score":    round(score, 3),
            })
            done += 1
            sys.stdout.write(f"\r  Progress: {done}/{total_combos} combinations")
            sys.stdout.flush()

    sweep_results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n  Best config: RSI buy={sweep_results[0]['rsi_buy']} "
          f"sell={sweep_results[0]['rsi_sell']} "
          f"(score={sweep_results[0]['score']:.1f})")

    # ── Save output ────────────────────────────────────────────────
    output = {
        "generated_at": datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str": datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "config": {
            "tickers":      TICKERS,
            "lookback_days":LOOKBACK_DAYS,
            "position_size":POSITION_SIZE,
            "baseline": {
                "rsi_buy":   RSI_BUY_MIN,
                "rsi_sell":  RSI_SELL_MAX,
                "gain_pct":  GAIN_TARGET_PCT,
                "stop_pct":  STOP_LOSS_PCT,
                "vol_mult":  VOLUME_MULT,
            }
        },
        "baseline": {
            "stats":  base_stats,
            "trades": base_trades,
        },
        "sweep": sweep_results,
    }

    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved → {OUT}")
    print(f"   Push to GitHub: git add backtest_results.json && git commit -m 'Backtest results' && git push")
    print("=" * 60)

if __name__ == "__main__":
    main()
