"""
NYLO VWAP Bounce Backtest
==========================
Rules:
- Price must touch VWAP (within 0.1%) at least twice in session —
  a "touch" is counted once per approach (new entry into the zone)
- On 3rd touch: enter in direction of opening-drive day bias
- RSI must confirm: > 55 for longs, < 45 for shorts
- Target: 0.75% from entry
- Stop:   0.40% from entry  (≈1.875:1 R:R)
- Window: 9:30 AM – 3:00 PM ET  (hold size limits exposure)
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
    "AAPL":  [datetime.date(2025,5,1), datetime.date(2025,7,31),
               datetime.date(2025,10,30), datetime.date(2026,1,30), datetime.date(2026,5,1)],
    "GOOGL": [datetime.date(2025,4,29), datetime.date(2025,7,29),
               datetime.date(2025,10,29), datetime.date(2026,2,4), datetime.date(2026,4,29)],
}
SLIPPAGE = {"AAPL": 0.0003, "GOOGL": 0.00025}

TOUCH_THRESHOLD = 0.001   # within 0.1% of VWAP
MIN_TOUCHES     = 3       # 3rd touch triggers entry
RSI_LONG_MIN    = 55
RSI_SHORT_MAX   = 45
TARGET_PCT      = 0.0075  # 0.75%
STOP_PCT        = 0.0040  # 0.40%
TRADE_END       = datetime.time(15, 0)
DAILY_LOSS_LIMIT = -500.0

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_vwap_bounce_results.json")


def is_earnings_blackout(date, ticker):
    d = date.date() if hasattr(date, "date") else date
    return any(abs((d - ed).days) <= EARNINGS_BLACKOUT_DAYS for ed in EARNINGS_DATES.get(ticker, []))


def apply_slippage(price, direction, slip):
    return round(price * (1 + slip) if direction == "long" else price * (1 - slip), 4)


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
            "sharpe":sharpe,"avg_score":0,"profit_factor":pf,"expectancy":exp,
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


def run_vwap_bounce(df, ticker):
    slip   = SLIPPAGE[ticker]
    trades = []
    dates  = sorted(df.index.normalize().unique())

    full_cl = df["Close"].squeeze()
    pre_rsi  = ta.momentum.RSIIndicator(full_cl, window=14).rsi()

    def lookup(s, ts, default):
        try:
            v = s.asof(ts); return default if pd.isna(v) else float(v)
        except: return default

    for date in dates:
        d_date = date.date()
        if is_earnings_blackout(d_date, ticker): continue

        day_df = df[df.index.date == d_date]
        n_bars = len(day_df)
        if n_bars < 10: continue

        closes  = day_df["Close"].squeeze().values.tolist()
        highs   = day_df["High"].squeeze().values.tolist()
        lows    = day_df["Low"].squeeze().values.tolist()
        volumes = day_df["Volume"].squeeze().values.tolist()
        idx     = list(day_df.index)

        # VWAP series
        cl = day_df["Close"].squeeze(); hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze();   vo = day_df["Volume"].squeeze()
        tp = (hi + lo + cl) / 3
        vwap_list = ((tp * vo).cumsum() / vo.cumsum()).tolist()

        # Opening drive day bias (same as main backtest)
        drive_df = day_df.between_time("09:30","09:35")
        avg_vol_prev = float(df[df.index.date < d_date]["Volume"].values[-200:].mean()) if not df[df.index.date < d_date].empty else 1.0
        day_bias = "neutral"
        if not drive_df.empty:
            d_open  = float(drive_df["Open"].squeeze().iloc[0])
            d_close = float(drive_df["Close"].squeeze().iloc[-1])
            move    = (d_close - d_open) / d_open * 100
            vr      = float(drive_df["Volume"].squeeze().sum()) / avg_vol_prev if avg_vol_prev > 0 else 1.0
            if   move >  0.3 and vr > 1.5: day_bias = "long"
            elif move < -0.3 and vr > 1.5: day_bias = "short"

        if day_bias == "neutral": continue  # skip neutral days — no clear directional edge

        touch_count = 0
        prev_near   = False
        in_trade    = False
        entry_data  = None
        daily_pnl   = 0.0

        for i in range(1, n_bars):
            ts    = idx[i]
            t_now = ts.time()
            if t_now > TRADE_END and not in_trade: break

            if daily_pnl <= DAILY_LOSS_LIMIT: break

            vwap  = vwap_list[i] if i < len(vwap_list) else 0
            price = closes[i]
            if vwap <= 0: continue

            # ── Exit management ──────────────────────────────────────────────
            if in_trade and entry_data:
                d  = entry_data["dir"]
                ht = (d=="long" and price >= entry_data["target"]) or (d=="short" and price <= entry_data["target"])
                hs = (d=="long" and price <= entry_data["stop"])   or (d=="short" and price >= entry_data["stop"])
                htime = t_now >= TRADE_END

                if ht or hs or htime:
                    res = "Target Hit" if ht else ("Stop Loss Hit" if hs else "Time Exit")
                    xp  = (entry_data["target"] if ht else (entry_data["stop"] if hs else price))
                    xp  = round(xp*(1-slip if d=="long" else 1+slip), 4)
                    pct = ((xp-entry_data["ep"])/entry_data["ep"]*100 if d=="long"
                           else (entry_data["ep"]-xp)/entry_data["ep"]*100)
                    dol = round(pct/100*entry_data["ps"], 2)
                    trades.append({**entry_data["rec"],"exit":xp,"result":res,
                                   "pnl_pct":round(pct,3),"pnl_dollar":dol})
                    daily_pnl += dol
                    in_trade = False; entry_data = None
                continue

            if in_trade: continue

            # ── Touch detection ─────────────────────────────────────────────
            near = abs(price - vwap) / vwap < TOUCH_THRESHOLD
            if near and not prev_near:
                touch_count += 1
            prev_near = near

            # ── Entry on 3rd+ touch ─────────────────────────────────────────
            if touch_count >= MIN_TOUCHES and near:
                rsi = lookup(pre_rsi, ts, 50.0)

                # Confirm RSI direction matches day bias
                if day_bias == "long"  and rsi < RSI_LONG_MIN:  continue
                if day_bias == "short" and rsi > RSI_SHORT_MAX: continue

                dirn    = day_bias
                entry_p = apply_slippage(price, dirn, slip)
                if dirn == "long":
                    target = round(entry_p * (1 + TARGET_PCT), 4)
                    stop   = round(entry_p * (1 - STOP_PCT), 4)
                else:
                    target = round(entry_p * (1 - TARGET_PCT), 4)
                    stop   = round(entry_p * (1 + STOP_PCT), 4)

                ps = 1000  # $1000 per trade for comparability
                rec = {"date":d_date.strftime("%Y-%m-%d"),"ticker":ticker,
                       "direction":"Long" if dirn=="long" else "Short",
                       "entry":entry_p,"exit":None,"result":None,"pnl_pct":0,"pnl_dollar":0,
                       "rsi":round(rsi,1),"vol_ratio":1.0,"touch_count":touch_count,
                       "signal_score":6,"pos_size":ps,
                       "entry_time":ts.strftime("%I:%M %p"),"hour":ts.hour,
                       "day_bias":day_bias,"signal_type":"vwap_bounce"}

                in_trade = True
                entry_data = {"dir":dirn,"ep":entry_p,"target":target,"stop":stop,"ps":ps,"rec":rec}
                touch_count = 0  # reset so we can re-enter if needed later

        # EOD close
        if in_trade and entry_data and closes:
            ep  = closes[-1]
            ep  = round(ep*(1-slip if entry_data["dir"]=="long" else 1+slip), 4)
            pct = ((ep-entry_data["ep"])/entry_data["ep"]*100 if entry_data["dir"]=="long"
                   else (entry_data["ep"]-ep)/entry_data["ep"]*100)
            trades.append({**entry_data["rec"],"exit":ep,"result":"Time Exit",
                           "pnl_pct":round(pct,3),"pnl_dollar":round(pct/100*entry_data["ps"],2)})

    return trades


def main():
    print("="*60)
    print("  NYLO VWAP Bounce Backtest")
    print(f"  Touch zone: ±{TOUCH_THRESHOLD*100:.1f}% of VWAP | Triggers on 3rd touch")
    print(f"  Target: {TARGET_PCT*100:.2f}% | Stop: {STOP_PCT*100:.2f}%")
    print("="*60)

    data = {}
    for t in TICKERS:
        data[t] = fetch_60d(t)

    print("\nRunning VWAP Bounce strategy...")
    all_trades = []
    for t in TICKERS:
        if data[t].empty: print(f"  {t}: no data"); continue
        trades = run_vwap_bounce(data[t], t)
        wins   = sum(1 for x in trades if x["pnl_dollar"] > 0)
        pnl    = round(sum(x["pnl_dollar"] for x in trades), 2)
        print(f"  {t}: {len(trades)} trades | {round(wins/len(trades)*100,1) if trades else 0}% WR | ${pnl:+.2f}")
        all_trades.extend(trades)

    all_trades.sort(key=lambda t:(t["date"],t["entry_time"]))
    st = calc_stats(all_trades)
    mc = monte_carlo(all_trades)

    print(f"\n  TOTAL: {st['trades']} trades | {st['win_rate']}% WR | ${st['total_pnl_dollar']:+.2f}")
    print(f"  Sharpe:{st['sharpe']} | PF:{st['profit_factor']} | E:${st['expectancy']:+.2f}/trade")

    v18_path = os.path.join(BASE, "backtest_results.json")
    v18_stats = {}
    if os.path.exists(v18_path):
        with open(v18_path) as f: v18_stats = json.load(f).get("baseline",{}).get("stats",{})

    out = {"generated_at":datetime.datetime.now(MARKET_TZ).isoformat(),
           "generated_str":datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
           "version":"vwap_bounce_v1",
           "config":{"strategy":"VWAP Bounce","touch_threshold_pct":TOUCH_THRESHOLD*100,
                     "min_touches":MIN_TOUCHES,"rsi_long_min":RSI_LONG_MIN,
                     "rsi_short_max":RSI_SHORT_MAX,"target_pct":TARGET_PCT*100,"stop_pct":STOP_PCT*100},
           "stats":st,"trades":all_trades,"monte_carlo":mc,
           "comparison":{"vwap_bounce":st,"v18":v18_stats}}
    with open(OUT,"w") as f: json.dump(out, f, indent=2)
    print(f"\n✅ Saved → {OUT}")

if __name__ == "__main__":
    main()
