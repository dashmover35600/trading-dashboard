"""
NYLO Backtesting Engine v14
============================
Exact replica of trading_agent_14.py strategy logic:
  1. Opening Drive    — first 5 min sets day bias (long/short/neutral)
  2. VWAP Reclaim     — primary signal: price crosses VWAP with volume
  3. EMA Pullback     — secondary: bounce off 9 EMA in trend direction
  4. Signal Scoring   — 0-10 score, only 6+ trades execute
  5. Dynamic sizing   — $500 / $1000 / $1500 based on score
  6. Trailing stop    — activates at +0.5%, trails by 0.4%
  7. Hard cutoff      — 12:45 PM ET, no holding into lunch

Usage:
  python3 -W ignore backtest.py
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
TICKER            = "QQQ"
LOOKBACK_DAYS     = 90
MARKET_TZ         = pytz.timezone("America/New_York")
POS_BASE          = 1000.0
POS_MEDIUM        = 2000.0
POS_HIGH          = 3000.0
GAIN_TARGET_PCT   = 0.015
STOP_LOSS_PCT     = 0.0075
TRAIL_TRIGGER_PCT = 0.0075
TRAIL_STOP_PCT    = 0.003
MIN_SCORE         = 5
RSI_BULL_MIN      = 52
RSI_BULL_MAX      = 75
RSI_BEAR_MIN      = 25
RSI_BEAR_MAX      = 48
VOLUME_MIN        = 1.2
SWEEP_RSI_BUY     = [45, 48, 50, 52, 55, 58, 60]
SWEEP_RSI_SELL    = [40, 42, 45, 48, 50, 52, 55]
SWEEP_MIN_SCORE   = [4, 5, 6, 7]
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_results.json")

def fetch_1m(ticker, days=90):
    print(f"  Fetching {ticker} 1-min data ({days} days)...")
    frames = []
    end = datetime.datetime.now(MARKET_TZ)
    chunks = (days // 7) + (1 if days % 7 else 0)
    for i in range(chunks):
        ce = end - datetime.timedelta(days=i*7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"), end=ce.strftime("%Y-%m-%d"),
                             interval="1m", progress=False, auto_adjust=True)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)
    df = df.between_time("09:30", "13:00")
    print(f"  {ticker}: {len(df)} bars across {len(df.index.normalize().unique())} days")
    return df

def calc_rsi(series, window=14):
    if len(series) < window + 1:
        return 50.0
    try:
        val = ta.momentum.RSIIndicator(series.squeeze(), window=window).rsi().iloc[-1]
        return round(float(val), 2) if not pd.isna(val) else 50.0
    except:
        return 50.0

def calc_ema(series, window=9):
    if len(series) < window:
        return float(series.iloc[-1])
    try:
        return round(float(series.ewm(span=window, adjust=False).mean().iloc[-1]), 4)
    except:
        return float(series.iloc[-1])

def score_signal(direction, signal_type, rsi, vol_ratio, price, vwap, ema9, day_bias, rsi_bull_min):
    score = 0
    if day_bias == direction: score += 2
    elif day_bias == "neutral": score += 1
    if direction == "long":
        if rsi_bull_min + 2 <= rsi <= rsi_bull_min + 20: score += 2
        elif rsi_bull_min <= rsi <= rsi_bull_min + 28: score += 1
    else:
        if RSI_BEAR_MAX - 22 <= rsi <= RSI_BEAR_MAX - 2: score += 2
        elif RSI_BEAR_MAX - 30 <= rsi <= RSI_BEAR_MAX: score += 1
    if vol_ratio >= 2.0: score += 2
    elif vol_ratio >= 1.3: score += 1
    if vwap and vwap > 0:
        if direction == "long" and price > vwap: score += 1
        elif direction == "short" and price < vwap: score += 1
    if direction == "long" and price > ema9: score += 1
    elif direction == "short" and price < ema9: score += 1
    if signal_type == "vwap_reclaim": score += 1
    return min(score, 10)

def get_pos_size(score):
    if score >= 9: return POS_HIGH
    elif score >= 7: return POS_MEDIUM
    return POS_BASE

def close_trade(trades, entry, exit_price, result, date, ts):
    pnl_pct = ((exit_price - entry["price"]) / entry["price"] * 100
               if entry["dir"] == "long"
               else (entry["price"] - exit_price) / entry["price"] * 100)
    trades.append({
        "date": date.strftime("%Y-%m-%d"), "ticker": TICKER,
        "direction": "Long" if entry["dir"]=="long" else "Short",
        "entry": round(entry["price"],4), "exit": round(exit_price,4),
        "result": result, "pnl_pct": round(pnl_pct,3),
        "pnl_dollar": round(pnl_pct/100*entry["pos_size"],2),
        "rsi": round(entry["rsi"],1), "vol_ratio": entry["vol_ratio"],
        "entry_time": entry["time"], "hour": entry["hour"],
        "signal_type": entry["signal_type"], "signal_score": entry["signal_score"],
        "pos_size": entry["pos_size"], "day_bias": entry["day_bias"],
        "trail_used": entry["trail_active"],
    })

def run_strategy(df, cfg):
    rsi_bull_min = cfg["rsi_bull_min"]
    rsi_bull_max = cfg["rsi_bull_max"]
    rsi_bear_min = cfg["rsi_bear_min"]
    rsi_bear_max = cfg["rsi_bear_max"]
    min_sc       = cfg["min_score"]
    gain_pct     = cfg["gain_pct"]
    stop_pct     = cfg["stop_pct"]
    trades = []
    dates  = sorted(df.index.normalize().unique())

    for date in dates:
        day_df = df[df.index.date == date.date()]
        if len(day_df) < 20: continue
        closes  = day_df["Close"].squeeze().tolist()
        volumes = day_df["Volume"].squeeze().tolist()
        day_idx = list(day_df.index)

        # Opening drive bias
        drive_df = day_df.between_time("09:30", "09:35")
        if len(drive_df) < 2: continue
        drive_open  = float(drive_df["Open"].iloc[0])
        drive_close = float(drive_df["Close"].iloc[-1])
        move_pct    = (drive_close - drive_open) / drive_open * 100
        prev_days   = df[df.index.date < date.date()]
        avg_vol     = float(prev_days["Volume"].tail(200).mean()) if len(prev_days) > 0 else 1.0
        drive_vol   = float(drive_df["Volume"].sum())
        vr_drive    = drive_vol / avg_vol if avg_vol > 0 else 1.0
        if move_pct > 0.3 and vr_drive > 1.5: day_bias = "long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias = "short"
        else: day_bias = "neutral"

        # VWAP for the day
        cl = day_df["Close"].squeeze()
        hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze()
        vo = day_df["Volume"].squeeze()
        tp = (hi + lo + cl) / 3
        vwap_list = ((tp * vo).cumsum() / vo.cumsum()).tolist()

        in_trade = False
        entry = None

        for i, ts in enumerate(day_idx):
            hour, minute = ts.hour, ts.minute
            if (hour == 11 and minute >= 30) or (hour == 12 and minute == 0): continue
            if (hour == 12 and minute >= 45) or hour >= 13:
                if in_trade and entry:
                    close_trade(trades, entry, closes[i] if i < len(closes) else entry["price"],
                                "Time Exit", date, ts)
                    in_trade = False; entry = None
                break

            if i == 0: continue
            price      = closes[i]
            prev_price = closes[i-1]
            vwap       = vwap_list[i] if i < len(vwap_list) else 0
            rsi_sl     = pd.Series(closes[max(0,i-28):i+1])
            rsi        = calc_rsi(rsi_sl)
            ema_sl     = pd.Series(closes[max(0,i-30):i+1])
            ema9       = calc_ema(ema_sl, 9)
            ema21      = calc_ema(ema_sl, 21)
            vol_sl     = pd.Series(volumes[max(0,i-20):i])
            avg_v      = float(vol_sl.mean()) if len(vol_sl) > 0 else 1.0
            vol_ratio  = round(volumes[i] / avg_v, 2) if avg_v > 0 else 1.0

            # Exit check
            if in_trade and entry:
                d = entry["dir"]
                if d == "long":
                    m = (price - entry["price"]) / entry["price"]
                    if m >= TRAIL_TRIGGER_PCT and not entry["trail_active"]:
                        entry["trail_active"] = True; entry["trail_peak"] = price
                        entry["trail_stop"] = price * (1 - TRAIL_STOP_PCT)
                    elif entry["trail_active"] and price > entry["trail_peak"]:
                        entry["trail_peak"] = price
                        entry["trail_stop"] = price * (1 - TRAIL_STOP_PCT)
                else:
                    m = (entry["price"] - price) / entry["price"]
                    if m >= TRAIL_TRIGGER_PCT and not entry["trail_active"]:
                        entry["trail_active"] = True; entry["trail_peak"] = price
                        entry["trail_stop"] = price * (1 + TRAIL_STOP_PCT)
                    elif entry["trail_active"] and price < entry["trail_peak"]:
                        entry["trail_peak"] = price
                        entry["trail_stop"] = price * (1 + TRAIL_STOP_PCT)

                ht = (d=="long" and price>=entry["target"]) or (d=="short" and price<=entry["target"])
                hs = (d=="long" and price<=entry["stop"])   or (d=="short" and price>=entry["stop"])
                htr= entry["trail_active"] and (
                    (d=="long" and price<=entry["trail_stop"]) or
                    (d=="short" and price>=entry["trail_stop"]))
                if ht or hs or htr:
                    res = "Target Hit" if ht else ("Trailing Stop" if htr else "Stop Loss Hit")
                    ep  = entry["target"] if ht else (entry["trail_stop"] if htr else entry["stop"])
                    close_trade(trades, entry, ep, res, date, ts)
                    in_trade = False; entry = None
                    continue

            if in_trade: continue

            # Signals
            sig = None; dirn = None
            if vwap > 0:
                if prev_price < vwap and price > vwap and vol_ratio >= VOLUME_MIN:
                    if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                        sig = "vwap_reclaim"; dirn = "long"
                elif prev_price > vwap and price < vwap and vol_ratio >= VOLUME_MIN:
                    if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                        sig = "vwap_reclaim"; dirn = "short"
            if not sig:
                if prev_price <= ema9*1.001 and price > ema9 and price > ema21 and vol_ratio >= VOLUME_MIN:
                    if day_bias in ("long","neutral") and rsi_bull_min <= rsi <= rsi_bull_max:
                        sig = "ema_pullback"; dirn = "long"
                elif prev_price >= ema9*0.999 and price < ema9 and price < ema21 and vol_ratio >= VOLUME_MIN:
                    if day_bias in ("short","neutral") and rsi_bear_min <= rsi <= rsi_bear_max:
                        sig = "ema_pullback"; dirn = "short"
            if not sig: continue

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap, ema9, day_bias, rsi_bull_min)
            if sc < min_sc: continue

            ps   = get_pos_size(sc)
            tgt  = price*(1+gain_pct) if dirn=="long" else price*(1-gain_pct)
            stp  = price*(1-stop_pct) if dirn=="long" else price*(1+stop_pct)
            in_trade = True
            entry = {"dir":dirn,"price":price,"target":round(tgt,4),"stop":round(stp,4),
                     "trail_active":False,"trail_peak":price,"trail_stop":None,
                     "rsi":rsi,"vol_ratio":vol_ratio,"signal_type":sig,"signal_score":sc,
                     "pos_size":ps,"shares":round(ps/price,4),
                     "time":ts.strftime("%I:%M %p"),"hour":hour,"day_bias":day_bias}
    return trades

def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"trail_exits":0,"win_rate":0,
                "total_pnl_pct":0,"total_pnl_dollar":0,"best":0,"worst":0,
                "max_drawdown":0,"sharpe":0,"avg_score":0}
    wins   = [t for t in trades if t["result"]=="Target Hit" or
               (t["result"]=="Trailing Stop" and t["pnl_pct"] > 0)]
    losses = [t for t in trades if t["result"]=="Stop Loss Hit" or
               (t["result"]=="Trailing Stop" and t["pnl_pct"] <= 0)]
    trails = [t for t in trades if t["result"]=="Trailing Stop"]
    pnls   = [t["pnl_pct"] for t in trades]
    dols   = [t["pnl_dollar"] for t in trades]
    peak=0; dd=0; cum=0
    for p in pnls:
        cum+=p
        if cum>peak: peak=cum
        if cum-peak<dd: dd=cum-peak
    import statistics
    sharpe=0
    if len(pnls)>1:
        try:
            sharpe=round(statistics.mean(pnls)/statistics.stdev(pnls)*(252**0.5),3)
        except: pass
    return {"trades":len(trades),"wins":len(wins),"losses":len(losses),
            "trail_exits":len(trails),
            "win_rate":round(len(wins)/len(trades)*100,2),
            "total_pnl_pct":round(sum(pnls),3),"total_pnl_dollar":round(sum(dols),2),
            "best":round(max(pnls),3),"worst":round(min(pnls),3),
            "max_drawdown":round(dd,3),"sharpe":sharpe,
            "avg_score":round(sum(t["signal_score"] for t in trades)/len(trades),1)}

def main():
    print("="*60)
    print(f"  NYLO Backtest v14 — VWAP Reclaim + EMA Pullback")
    print(f"  Ticker: {TICKER} | Lookback: {LOOKBACK_DAYS} days")
    print(f"  Score: min {MIN_SCORE}/10 | Sizes: ${POS_BASE:.0f}/${POS_MEDIUM:.0f}/${POS_HIGH:.0f}")
    print("="*60)

    df = fetch_1m(TICKER, LOOKBACK_DAYS)
    if df.empty: print("ERROR: No data"); return

    print(f"\n[1/2] Baseline strategy...")
    base_cfg = {"rsi_bull_min":RSI_BULL_MIN,"rsi_bull_max":RSI_BULL_MAX,
                "rsi_bear_min":RSI_BEAR_MIN,"rsi_bear_max":RSI_BEAR_MAX,
                "min_score":MIN_SCORE,"gain_pct":GAIN_TARGET_PCT,"stop_pct":STOP_LOSS_PCT}
    bt = run_strategy(df, base_cfg)
    bt.sort(key=lambda t:(t["date"],t["entry_time"]))
    bs = calc_stats(bt)
    print(f"  {bs['trades']} trades | {bs['win_rate']:.1f}% win rate | "
          f"P&L: {bs['total_pnl_pct']:+.2f}% (${bs['total_pnl_dollar']:+.2f}) | "
          f"Avg score: {bs['avg_score']}/10 | Sharpe: {bs['sharpe']}")

    print(f"\n[2/2] Parameter sweep...")
    combos = [(b,s,ms) for b in SWEEP_RSI_BUY for s in SWEEP_RSI_SELL
              for ms in SWEEP_MIN_SCORE if b > s]
    sweep = []
    for done,(rb,rs,ms) in enumerate(combos):
        cfg = {"rsi_bull_min":rb,"rsi_bull_max":rb+25,"rsi_bear_min":rs-25,"rsi_bear_max":rs,
               "min_score":ms,"gain_pct":GAIN_TARGET_PCT,"stop_pct":STOP_LOSS_PCT}
        t = run_strategy(df, cfg)
        t.sort(key=lambda x:(x["date"],x["entry_time"]))
        st = calc_stats(t)
        sc = st["win_rate"]*0.5 + st["total_pnl_pct"]*0.4 + st["trades"]*0.1
        sweep.append({"rsi_buy":rb,"rsi_sell":rs,"min_score":ms,"stats":st,"score":round(sc,3)})
        sys.stdout.write(f"\r  Progress: {done+1}/{len(combos)}")
        sys.stdout.flush()

    sweep.sort(key=lambda r:r["score"],reverse=True)
    best=sweep[0]
    print(f"\n  Best: RSI buy={best['rsi_buy']} sell={best['rsi_sell']} "
          f"min_score={best['min_score']} → "
          f"{best['stats']['win_rate']:.0f}% win rate, "
          f"{best['stats']['total_pnl_pct']:+.2f}% P&L")

    out = {
        "generated_at":  datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str": datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "config": {"ticker":TICKER,"lookback_days":LOOKBACK_DAYS,
                   "strategy":"VWAP Reclaim + EMA Pullback + Opening Drive",
                   "baseline":{"rsi_bull_min":RSI_BULL_MIN,"rsi_bull_max":RSI_BULL_MAX,
                                "rsi_bear_min":RSI_BEAR_MIN,"rsi_bear_max":RSI_BEAR_MAX,
                                "min_score":MIN_SCORE,"gain_pct":GAIN_TARGET_PCT*100,
                                "stop_pct":STOP_LOSS_PCT*100}},
        "baseline":{"stats":bs,"trades":bt},
        "sweep":sweep[:60],
    }
    with open(OUT,"w") as f:
        json.dump(out,f,indent=2)
    print(f"\n✅ Results saved → {OUT}")
    print(f"   git add backtest_results.json && git commit -m 'Backtest v14' && git push")
    print("="*60)

if __name__=="__main__":
    main()
