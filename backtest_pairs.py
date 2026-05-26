"""
NYLO Pairs Trading Backtest — AAPL vs GOOGL
=============================================
Rules:
- Calculate AAPL/GOOGL price ratio on aligned 1m/5m bars
- Compute 20-period rolling mean and std of ratio
- Signal at ±1.5 std dev from mean:
    ratio HIGH (AAPL outperforms) → Long GOOGL, Short AAPL
    ratio LOW  (GOOGL outperforms)→ Long AAPL,  Short GOOGL
- Exit when ratio reverts to mean (within 0.25 std dev)
- Hard stop: ratio moves another 1.0 std dev against position
- EOD forced exit if still open
- Position size: $1000 each leg
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import datetime
import pytz
import os
import statistics
import random

MARKET_TZ = pytz.timezone("America/New_York")

EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_DATES = {
    "AAPL":  [datetime.date(2025,5,1), datetime.date(2025,7,31),
               datetime.date(2025,10,30), datetime.date(2026,1,30), datetime.date(2026,5,1)],
    "GOOGL": [datetime.date(2025,4,29), datetime.date(2025,7,29),
               datetime.date(2025,10,29), datetime.date(2026,2,4), datetime.date(2026,4,29)],
}
SLIPPAGE_AAPL  = 0.0003
SLIPPAGE_GOOGL = 0.00025

LOOKBACK   = 20      # rolling window for mean/std
Z_ENTRY    = 1.5     # std devs to trigger entry
Z_EXIT     = 0.25    # std devs to trigger exit (reversion to mean)
Z_STOP     = 2.5     # std devs for hard stop (entry + 1.0 extra)
POS_SIZE   = 1000    # $1000 each leg
TRADE_END  = datetime.time(15, 0)
DAILY_LOSS_LIMIT = -1000.0   # two legs, so double the limit

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_pairs_results.json")


def is_earnings_blackout(date):
    d = date.date() if hasattr(date, "date") else date
    for ticker, dates in EARNINGS_DATES.items():
        if any(abs((d - ed).days) <= EARNINGS_BLACKOUT_DAYS for ed in dates):
            return True
    return False


def _process_frames(frames):
    if not frames: return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ) if df.index.tz is None else df.index.tz_convert(MARKET_TZ)
    return df.between_time("09:30", "15:30")


def fetch_60d(ticker):
    print(f"  Fetching {ticker}...")
    end = datetime.datetime.now(MARKET_TZ)
    frames_1m, frames_5m = [], []
    for i in range(5):
        ce, cs = end - datetime.timedelta(days=i*7), end - datetime.timedelta(days=(i+1)*7)
        try:
            r = yf.download(ticker, start=cs.strftime("%Y-%m-%d"), end=ce.strftime("%Y-%m-%d"), interval="1m", progress=False, auto_adjust=True)
            if not r.empty: frames_1m.append(r)
        except: pass
    for i in range(9):
        ce, cs = end - datetime.timedelta(days=i*7), end - datetime.timedelta(days=(i+1)*7)
        try:
            r = yf.download(ticker, start=cs.strftime("%Y-%m-%d"), end=ce.strftime("%Y-%m-%d"), interval="5m", progress=False, auto_adjust=True)
            if not r.empty: frames_5m.append(r)
        except: pass
    df_1m, df_5m = _process_frames(frames_1m), _process_frames(frames_5m)
    if df_1m.empty and df_5m.empty: return pd.DataFrame()
    if df_1m.empty: return df_5m
    if df_5m.empty: return df_1m
    cut = df_1m.index[0].date()
    return _process_frames([df_5m[df_5m.index.date < cut], df_1m])


def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0,"total_pnl_pct":0,"total_pnl_dollar":0,
                "best":0,"worst":0,"max_drawdown":0,"sharpe":0,"avg_score":0,"profit_factor":0,
                "expectancy":0,"by_ticker":{},"avg_win":0,"avg_loss":0}
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
    return {"trades":len(trades),"wins":len(wins),"losses":len(losses),
            "win_rate":round(wr*100,2),"total_pnl_pct":round(sum(pnls),3),
            "total_pnl_dollar":round(sum(dols),2),"best":round(max(pnls),3) if pnls else 0,
            "worst":round(min(pnls),3) if pnls else 0,"max_drawdown":round(dd,3),
            "sharpe":sharpe,"avg_score":0,"profit_factor":pf,"expectancy":exp,
            "avg_win":avg_win,"avg_loss":avg_loss,"by_ticker":{"AAPL/GOOGL":{"trades":len(trades),"pnl":round(sum(dols),2),"wins":len(wins)}}}


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


def build_aligned_series(df_aapl, df_googl):
    """Merge AAPL and GOOGL on common timestamps, return aligned close prices."""
    a = df_aapl["Close"].squeeze().rename("AAPL")
    g = df_googl["Close"].squeeze().rename("GOOGL")
    merged = pd.concat([a, g], axis=1).dropna()
    return merged


def run_pairs(df_aapl, df_googl):
    trades = []
    merged = build_aligned_series(df_aapl, df_googl)
    if merged.empty or len(merged) < LOOKBACK + 5:
        return trades

    # Price ratio
    merged["ratio"] = merged["AAPL"] / merged["GOOGL"]
    merged["mean"]  = merged["ratio"].rolling(LOOKBACK).mean()
    merged["std"]   = merged["ratio"].rolling(LOOKBACK).std()
    merged["zscore"]= (merged["ratio"] - merged["mean"]) / merged["std"]
    merged = merged.dropna()

    dates = sorted(set(merged.index.date))

    for d_date in dates:
        if is_earnings_blackout(d_date): continue

        day = merged[merged.index.date == d_date]
        if len(day) < 5: continue

        in_trade  = False
        entry_data = None
        daily_pnl = 0.0

        for ts, row in day.iterrows():
            t_now = ts.time()
            if t_now > TRADE_END and not in_trade: break
            if daily_pnl <= DAILY_LOSS_LIMIT: break

            z      = float(row["zscore"])
            ratio  = float(row["ratio"])
            mean_r = float(row["mean"])
            std_r  = float(row["std"]) if float(row["std"]) > 0 else 1e-6
            pa     = float(row["AAPL"])
            pg     = float(row["GOOGL"])

            # ── Exit management ──────────────────────────────────────────────
            if in_trade and entry_data:
                reverted = abs(z) <= Z_EXIT
                # Hard stop: z moved further than Z_STOP in the wrong direction
                if entry_data["side"] == "long_googl":
                    # We expect z to fall (ratio to revert down to mean)
                    hard_stop = z > Z_STOP
                else:
                    # We expect z to rise (ratio to revert up to mean)
                    hard_stop = z < -Z_STOP

                force_eod = t_now >= TRADE_END

                if reverted or hard_stop or force_eod:
                    res = "Target Hit" if reverted else ("Stop Loss Hit" if hard_stop else "Time Exit")

                    # Long GOOGL / Short AAPL PnL
                    if entry_data["side"] == "long_googl":
                        long_pnl  = (pg - entry_data["pg"]) / entry_data["pg"] * POS_SIZE
                        short_pnl = (entry_data["pa"] - pa) / entry_data["pa"] * POS_SIZE
                    else:
                        long_pnl  = (pa - entry_data["pa"]) / entry_data["pa"] * POS_SIZE
                        short_pnl = (entry_data["pg"] - pg) / entry_data["pg"] * POS_SIZE

                    # Slippage on exit (both legs)
                    slip_cost = (POS_SIZE * SLIPPAGE_AAPL + POS_SIZE * SLIPPAGE_GOOGL)
                    total_pnl = round(long_pnl + short_pnl - slip_cost, 2)
                    pct       = round(total_pnl / (POS_SIZE * 2) * 100, 3)

                    trades.append({"date":d_date.strftime("%Y-%m-%d"),
                                   "ticker":"AAPL/GOOGL",
                                   "direction":entry_data["label"],
                                   "entry":round(entry_data["ratio"],4),
                                   "exit":round(ratio,4),
                                   "entry_zscore":round(entry_data["z"],3),
                                   "exit_zscore":round(z,3),
                                   "result":res,"pnl_pct":pct,"pnl_dollar":total_pnl,
                                   "signal_score":6,"pos_size":POS_SIZE*2,
                                   "entry_time":entry_data["ts"].strftime("%I:%M %p"),
                                   "hour":entry_data["ts"].hour,"signal_type":"pairs"})
                    daily_pnl += total_pnl
                    in_trade = False; entry_data = None
                continue

            if in_trade: continue

            # ── Signal detection ─────────────────────────────────────────────
            side = None
            if z > Z_ENTRY:
                side = "long_googl"   # ratio high → AAPL outperforming → long GOOGL, short AAPL
                label = "Long GOOGL / Short AAPL"
            elif z < -Z_ENTRY:
                side = "long_aapl"    # ratio low → GOOGL outperforming → long AAPL, short GOOGL
                label = "Long AAPL / Short GOOGL"

            if side is None: continue

            # Apply entry slippage to both legs
            if side == "long_googl":
                pa_entry = pa * (1 - SLIPPAGE_AAPL)    # short AAPL → sell at lower
                pg_entry = pg * (1 + SLIPPAGE_GOOGL)   # long GOOGL → buy at higher
            else:
                pa_entry = pa * (1 + SLIPPAGE_AAPL)    # long AAPL → buy at higher
                pg_entry = pg * (1 - SLIPPAGE_GOOGL)   # short GOOGL → sell at lower

            in_trade = True
            entry_data = {"side":side,"label":label,"ratio":ratio,"z":z,
                          "pa":pa_entry,"pg":pg_entry,"ts":ts,"mean":mean_r,"std":std_r}

        # EOD forced exit
        if in_trade and entry_data:
            last = day.iloc[-1]
            pa_e = float(last["AAPL"])
            pg_e = float(last["GOOGL"])
            if entry_data["side"] == "long_googl":
                long_pnl  = (pg_e - entry_data["pg"]) / entry_data["pg"] * POS_SIZE
                short_pnl = (entry_data["pa"] - pa_e) / entry_data["pa"] * POS_SIZE
            else:
                long_pnl  = (pa_e - entry_data["pa"]) / entry_data["pa"] * POS_SIZE
                short_pnl = (entry_data["pg"] - pg_e) / entry_data["pg"] * POS_SIZE
            slip_cost = POS_SIZE * SLIPPAGE_AAPL + POS_SIZE * SLIPPAGE_GOOGL
            total_pnl = round(long_pnl + short_pnl - slip_cost, 2)
            pct       = round(total_pnl / (POS_SIZE * 2) * 100, 3)
            z_last    = float(last["zscore"])
            trades.append({"date":d_date.strftime("%Y-%m-%d"),"ticker":"AAPL/GOOGL",
                           "direction":entry_data["label"],
                           "entry":round(entry_data["ratio"],4),"exit":round(float(last["ratio"]),4),
                           "entry_zscore":round(entry_data["z"],3),"exit_zscore":round(z_last,3),
                           "result":"Time Exit","pnl_pct":pct,"pnl_dollar":total_pnl,
                           "signal_score":6,"pos_size":POS_SIZE*2,
                           "entry_time":entry_data["ts"].strftime("%I:%M %p"),
                           "hour":entry_data["ts"].hour,"signal_type":"pairs"})

    return trades


def main():
    print("="*60)
    print("  NYLO Pairs Trading Backtest — AAPL vs GOOGL")
    print(f"  Z-score entry: ±{Z_ENTRY} | Exit: ±{Z_EXIT} | Stop: ±{Z_STOP}")
    print(f"  Lookback: {LOOKBACK} bars | Leg size: ${POS_SIZE} each")
    print("="*60)

    df_aapl  = fetch_60d("AAPL")
    df_googl = fetch_60d("GOOGL")

    print("\nRunning Pairs strategy...")
    trades = run_pairs(df_aapl, df_googl)
    wins   = sum(1 for t in trades if t["pnl_dollar"] > 0)
    pnl    = round(sum(t["pnl_dollar"] for t in trades), 2)
    print(f"  AAPL/GOOGL: {len(trades)} trades | {round(wins/len(trades)*100,1) if trades else 0}% WR | ${pnl:+.2f}")

    trades.sort(key=lambda t:(t["date"],t["entry_time"]))
    st = calc_stats(trades)
    mc = monte_carlo(trades)

    print(f"\n  TOTAL: {st['trades']} trades | {st['win_rate']}% WR | ${st['total_pnl_dollar']:+.2f}")
    print(f"  Sharpe:{st['sharpe']} | PF:{st['profit_factor']} | E:${st['expectancy']:+.2f}/trade")

    v18_path = os.path.join(BASE, "backtest_results.json")
    v18_stats = {}
    if os.path.exists(v18_path):
        with open(v18_path) as f: v18_stats = json.load(f).get("baseline",{}).get("stats",{})

    out = {"generated_at":datetime.datetime.now(MARKET_TZ).isoformat(),
           "generated_str":datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
           "version":"pairs_v1",
           "config":{"strategy":"Pairs Trading","tickers":"AAPL/GOOGL",
                     "lookback":LOOKBACK,"z_entry":Z_ENTRY,"z_exit":Z_EXIT,"z_stop":Z_STOP,
                     "pos_size_each_leg":POS_SIZE},
           "stats":st,"trades":trades,"monte_carlo":mc,
           "comparison":{"pairs":st,"v18":v18_stats}}
    with open(OUT,"w") as f: json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {OUT}")

if __name__ == "__main__":
    main()
