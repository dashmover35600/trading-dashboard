"""
NYLO TTFM (Time To First Move) Strategy Backtest
=================================================
Captures the first decisive directional move after market open (9:30–9:45 AM ET).

Definition of "first move":
  - Scan 1-min bars from 9:31–9:45 AM (first bar after open)
  - First bar whose |% move from prior close| >= MOVE_THRESHOLD triggers entry
  - Direction = long if move positive, short if negative
  - Stop = opposite side of the bar that triggered entry
  - Target = GAIN_MULT × risk (from entry to stop)
  - Max hold = 30 min from entry; exit at 10:15 AM otherwise

Data source: yfinance (60 days 1-min + 120 days 5-min) — same as backtest.py v18.
Falls back to 5-min bars when 1-min not available for a given day.

Usage:
    python3 backtest_ttfm.py

Outputs: backtest_ttfm_results.json
"""

import json
import datetime
import statistics
import random
import os
import math

import pandas as pd
import pytz
import yfinance as yf

# ── Parameters ────────────────────────────────────────────────────────────────
TICKERS         = ["AAPL", "GOOGL"]
MARKET_TZ       = pytz.timezone("America/New_York")
BASE            = os.path.dirname(os.path.abspath(__file__))
OUT             = os.path.join(BASE, "backtest_ttfm_results.json")

MOVE_THRESHOLD  = 0.0035   # 0.35% move from prior close to trigger entry
GAIN_MULT       = 2.0      # reward:risk ratio for target
MAX_HOLD_BARS   = 30       # bars (1-min) before forced exit
ENTRY_WINDOW_START = datetime.time(9, 31)
ENTRY_WINDOW_END   = datetime.time(9, 45)
HARD_EXIT_TIME     = datetime.time(10, 15)

SLIPPAGE = {"AAPL": 0.0003, "GOOGL": 0.00025}
SIZE_PER_TRADE = 2000.0    # fixed $ position size

EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_DATES = {
    "AAPL":  [datetime.date(2024, 2, 1),  datetime.date(2024, 5, 2),
              datetime.date(2024, 8, 1),  datetime.date(2024, 10, 31),
              datetime.date(2025, 1, 30), datetime.date(2025, 5, 1),
              datetime.date(2025, 7, 31), datetime.date(2025, 10, 30)],
    "GOOGL": [datetime.date(2024, 1, 30), datetime.date(2024, 4, 25),
              datetime.date(2024, 7, 23), datetime.date(2024, 10, 29),
              datetime.date(2025, 2, 4),  datetime.date(2025, 4, 29),
              datetime.date(2025, 7, 29), datetime.date(2025, 10, 28)],
}

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame:
    """Return a combined 1-min + 5-min DataFrame covering ~120 trading days."""
    frames = []
    try:
        df1 = yf.download(ticker, period="60d", interval="1m",
                          auto_adjust=True, progress=False)
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.droplevel(1)
        if not df1.empty:
            frames.append(df1)
    except Exception:
        pass

    try:
        df5 = yf.download(ticker, period="60d", interval="5m",
                          auto_adjust=True, progress=False)
        if isinstance(df5.columns, pd.MultiIndex):
            df5.columns = df5.columns.droplevel(1)
        if not df5.empty:
            # resample to 1-min (forward-fill within bar) to unify resolution
            df5 = df5.resample("1min").ffill()
            frames.append(df5)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(MARKET_TZ)
    return df


# ── Earnings blackout check ───────────────────────────────────────────────────

def in_earnings_blackout(ticker: str, date: datetime.date) -> bool:
    for ed in EARNINGS_DATES.get(ticker, []):
        delta = (date - ed).days
        if -EARNINGS_BLACKOUT_DAYS <= delta <= EARNINGS_BLACKOUT_DAYS:
            return True
    return False


# ── Core strategy ─────────────────────────────────────────────────────────────

def run_ttfm(df: pd.DataFrame, ticker: str) -> list[dict]:
    """Run TTFM on a continuous 1-min DataFrame; return list of trade dicts."""
    if df.empty:
        return []

    slip = SLIPPAGE.get(ticker, 0.0003)
    trades = []

    # Group by trading day
    for date, day_df in df.groupby(df.index.date):
        if in_earnings_blackout(ticker, date):
            continue

        # Need at least the 9:30 open bar and some 9:31+ bars
        day_bars = day_df.between_time("09:30", "10:15")
        if len(day_bars) < 5:
            continue

        # Prior close = last bar's close from prior session (use open as proxy)
        open_bar = day_bars.iloc[0]
        prior_ref = float(open_bar["Open"])  # 9:30 open as reference

        # Scan 9:31–9:45 for first qualifying move
        entry_bars = day_bars.between_time("09:31", "09:45")
        if entry_bars.empty:
            continue

        triggered = False
        for ts, row in entry_bars.iterrows():
            bar_close = float(row["Close"])
            move_pct = (bar_close - prior_ref) / prior_ref

            if abs(move_pct) < MOVE_THRESHOLD:
                continue

            # Entry triggered
            direction = 1 if move_pct > 0 else -1
            entry_price = bar_close * (1 + direction * slip)  # slippage on entry

            # Stop = far side of the trigger bar
            if direction == 1:
                stop_price  = float(row["Low"])  * (1 - slip)
                risk        = entry_price - stop_price
                target_price = entry_price + GAIN_MULT * risk
            else:
                stop_price  = float(row["High"]) * (1 + slip)
                risk        = stop_price - entry_price
                target_price = entry_price - GAIN_MULT * risk

            if risk <= 0:
                continue

            shares = SIZE_PER_TRADE / entry_price

            # Simulate forward from next bar
            remaining = day_bars[day_bars.index > ts]
            remaining = remaining[remaining.index.time <= HARD_EXIT_TIME]

            exit_price  = None
            exit_reason = "time"
            bar_count   = 0

            for ex_ts, ex_row in remaining.iterrows():
                bar_count += 1
                hi = float(ex_row["High"])
                lo = float(ex_row["Low"])

                if direction == 1:
                    if lo <= stop_price:
                        exit_price  = stop_price
                        exit_reason = "stop"
                        break
                    if hi >= target_price:
                        exit_price  = target_price
                        exit_reason = "target"
                        break
                else:
                    if hi >= stop_price:
                        exit_price  = stop_price
                        exit_reason = "stop"
                        break
                    if lo <= target_price:
                        exit_price  = target_price
                        exit_reason = "target"
                        break

                if bar_count >= MAX_HOLD_BARS:
                    exit_price  = float(ex_row["Close"]) * (1 - direction * slip)
                    exit_reason = "time"
                    break

            if exit_price is None:
                if remaining.empty:
                    exit_price  = float(day_bars.iloc[-1]["Close"]) * (1 - direction * slip)
                    exit_reason = "eod"
                else:
                    exit_price  = float(remaining.iloc[-1]["Close"]) * (1 - direction * slip)
                    exit_reason = "time"

            raw_pnl = direction * (exit_price - entry_price) * shares
            pnl     = round(raw_pnl, 2)
            ret_pct = round(direction * (exit_price - entry_price) / entry_price * 100, 4)

            trades.append({
                "date":        str(date),
                "ticker":      ticker,
                "direction":   "long" if direction == 1 else "short",
                "entry":       round(entry_price, 4),
                "exit":        round(exit_price, 4),
                "stop":        round(stop_price, 4),
                "target":      round(target_price, 4),
                "move_pct":    round(move_pct * 100, 3),
                "shares":      round(shares, 2),
                "pnl":         pnl,
                "ret_pct":     ret_pct,
                "exit_reason": exit_reason,
                "win":         pnl > 0,
            })
            triggered = True
            break  # one trade per day per ticker

    return trades


# ── Statistics ────────────────────────────────────────────────────────────────

def calc_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(pnls)
    wr     = len(wins) / n * 100 if n else 0

    gross_profit = sum(wins)   if wins   else 0.0
    gross_loss   = abs(sum(losses)) if losses else 0.0
    pf           = round(gross_profit / gross_loss, 4) if gross_loss else float("inf")

    total_pnl    = sum(pnls)
    expectancy   = total_pnl / n if n else 0.0

    # Sharpe on daily PnL
    daily: dict[str, float] = {}
    for t in trades:
        daily[t["date"]] = daily.get(t["date"], 0.0) + t["pnl"]
    daily_vals = list(daily.values())

    if len(daily_vals) > 1:
        mu  = statistics.mean(daily_vals)
        sd  = statistics.stdev(daily_vals)
        sharpe = round((mu / sd) * math.sqrt(252), 4) if sd else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "n_trades":      n,
        "win_rate":      round(wr, 2),
        "profit_factor": pf,
        "total_pnl":     round(total_pnl, 2),
        "expectancy":    round(expectancy, 2),
        "sharpe":        sharpe,
        "max_drawdown":  round(max_dd, 2),
        "avg_win":       round(statistics.mean(wins), 2)   if wins   else 0.0,
        "avg_loss":      round(statistics.mean(losses), 2) if losses else 0.0,
        "exit_reasons":  {
            "target": sum(1 for t in trades if t["exit_reason"] == "target"),
            "stop":   sum(1 for t in trades if t["exit_reason"] == "stop"),
            "time":   sum(1 for t in trades if t["exit_reason"] == "time"),
            "eod":    sum(1 for t in trades if t["exit_reason"] == "eod"),
        },
    }


def monte_carlo(trades: list[dict], n_sims: int = 1000) -> dict:
    """Shuffle trade order n_sims times; return 5th/95th percentile final PnL."""
    pnls = [t["pnl"] for t in trades]
    if not pnls:
        return {}
    results = []
    for _ in range(n_sims):
        shuffled = pnls[:]
        random.shuffle(shuffled)
        results.append(sum(shuffled))
    results.sort()
    p5  = results[int(0.05 * n_sims)]
    p50 = results[int(0.50 * n_sims)]
    p95 = results[int(0.95 * n_sims)]
    return {"p5": round(p5, 2), "p50": round(p50, 2), "p95": round(p95, 2), "n_sims": n_sims}


def walk_forward(trades: list[dict], n_periods: int = 4) -> list[dict]:
    """Split trades chronologically into n_periods and calc stats per period."""
    if not trades:
        return []
    sorted_trades = sorted(trades, key=lambda t: t["date"])
    chunk = max(1, len(sorted_trades) // n_periods)
    results = []
    for i in range(n_periods):
        start = i * chunk
        end   = start + chunk if i < n_periods - 1 else len(sorted_trades)
        seg   = sorted_trades[start:end]
        stats = calc_stats(seg)
        results.append({
            "period":    i + 1,
            "from":      seg[0]["date"],
            "to":        seg[-1]["date"],
            "n_trades":  len(seg),
            "win_rate":  stats.get("win_rate", 0),
            "pnl":       stats.get("total_pnl", 0),
            "sharpe":    stats.get("sharpe", 0),
            "pf":        stats.get("profit_factor", 0),
        })
    return results


def direction_breakdown(trades: list[dict]) -> dict:
    """Stats split by long vs short."""
    longs  = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    return {
        "long":  calc_stats(longs)  if longs  else {},
        "short": calc_stats(shorts) if shorts else {},
    }


def ticker_breakdown(trades: list[dict]) -> dict:
    result = {}
    for ticker in TICKERS:
        sub = [t for t in trades if t["ticker"] == ticker]
        result[ticker] = calc_stats(sub) if sub else {}
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NYLO TTFM Strategy Backtest")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Move threshold: {MOVE_THRESHOLD*100:.2f}% | R:R = 1:{GAIN_MULT}")
    print(f"Entry window: {ENTRY_WINDOW_START}–{ENTRY_WINDOW_END} ET")
    print("=" * 60)

    all_trades = []
    for ticker in TICKERS:
        print(f"\nFetching {ticker} data...")
        df = fetch_data(ticker)
        if df.empty:
            print(f"  [!] No data for {ticker}, skipping.")
            continue
        days_available = df.index.normalize().nunique()
        print(f"  Got {len(df):,} bars across {days_available} unique days.")

        print(f"  Running TTFM strategy...")
        trades = run_ttfm(df, ticker)
        print(f"  {len(trades)} trades found.")
        all_trades.extend(trades)

    print(f"\n{'─'*60}")
    print(f"Total trades: {len(all_trades)}")

    if not all_trades:
        print("[!] No trades generated — check data availability.")
        return

    overall   = calc_stats(all_trades)
    mc        = monte_carlo(all_trades)
    wf        = walk_forward(all_trades, n_periods=4)
    by_dir    = direction_breakdown(all_trades)
    by_ticker = ticker_breakdown(all_trades)

    print(f"\nWin Rate      : {overall['win_rate']:.1f}%")
    print(f"Profit Factor : {overall['profit_factor']:.3f}")
    print(f"Sharpe        : {overall['sharpe']:.3f}")
    print(f"Total PnL     : ${overall['total_pnl']:.2f}")
    print(f"Expectancy    : ${overall['expectancy']:.2f}/trade")
    print(f"Max Drawdown  : ${overall['max_drawdown']:.2f}")
    print(f"\nExit reasons  : {overall['exit_reasons']}")

    print(f"\nMonte Carlo (p5/p50/p95): ${mc['p5']:.0f} / ${mc['p50']:.0f} / ${mc['p95']:.0f}")

    print(f"\nWalk-Forward periods:")
    for p in wf:
        print(f"  Period {p['period']} ({p['from']} → {p['to']}): "
              f"{p['n_trades']} trades | WR {p['win_rate']:.1f}% | PnL ${p['pnl']:.0f} | Sharpe {p['sharpe']:.2f}")

    print(f"\nDirection breakdown:")
    for d, s in by_dir.items():
        if s:
            print(f"  {d.capitalize():5s}: {s.get('n_trades',0)} trades | "
                  f"WR {s.get('win_rate',0):.1f}% | PnL ${s.get('total_pnl',0):.0f}")

    print(f"\nTicker breakdown:")
    for tk, s in by_ticker.items():
        if s:
            print(f"  {tk}: {s.get('n_trades',0)} trades | "
                  f"WR {s.get('win_rate',0):.1f}% | PnL ${s.get('total_pnl',0):.0f}")

    out = {
        "strategy":    "TTFM",
        "generated":   datetime.datetime.now().isoformat(),
        "parameters": {
            "tickers":         TICKERS,
            "move_threshold":  MOVE_THRESHOLD,
            "gain_mult":       GAIN_MULT,
            "max_hold_bars":   MAX_HOLD_BARS,
            "entry_window":    [str(ENTRY_WINDOW_START), str(ENTRY_WINDOW_END)],
            "hard_exit_time":  str(HARD_EXIT_TIME),
            "size_per_trade":  SIZE_PER_TRADE,
        },
        "overall":          overall,
        "monte_carlo":      mc,
        "walk_forward":     wf,
        "by_direction":     by_dir,
        "by_ticker":        by_ticker,
        "trades":           all_trades,
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {OUT}")


if __name__ == "__main__":
    main()
