"""
NYLO Backtesting Engine v15
============================
Strategy: VWAP Reclaim + EMA Pullback + Opening Drive
Tickers: QQQ, SPY, NVDA, TSLA
Lookback: 120 days
No trade limits — every valid signal fires
Position sizing: Maximum calculated risk
  Score 10: $5000 / Score 9: $4000 / Score 8: $3000
  Score 7: $2000 / Score 6: $1500 / Score 5: $1000
  NVDA/TSLA: 75% of above

Refinements vs v14:
  - No dead zone
  - No trade limit per day
  - Momentum confirmation: 2 consecutive bars
  - Tighter RSI: 55-72 longs / 28-45 shorts
  - Trend strength: must move 0.3% from open
  - VWAP 2-bar confirmation
  - 4 tickers
"""

import yfinance as yf
import pandas as pd
import ta
import json
import datetime
import pytz
import os
import sys

TICKERS          = ["QQQ", "NVDA"]  # v15.1: dropped SPY/TSLA — both losing
VOLATILE         = ["NVDA"]  # NVDA gets 75% sizing
LOOKBACK_DAYS    = 120
MARKET_TZ        = pytz.timezone("America/New_York")
GAIN_TARGET_PCT  = 0.015
STOP_LOSS_PCT    = 0.0075
TRAIL_TRIGGER    = 0.0075
TRAIL_STOP       = 0.003
MIN_SCORE        = 5
RSI_BULL_MIN     = 52  # sweep best
RSI_BULL_MAX     = 72
RSI_BEAR_MIN     = 28
RSI_BEAR_MAX     = 40  # sweep best
VOLUME_MIN       = 1.2
TREND_MIN_PCT    = 0.003
SWEEP_RSI_BUY    = [50, 52, 55, 58, 60]
SWEEP_RSI_SELL   = [40, 42, 45, 48, 50]
SWEEP_MIN_SCORE  = [4, 5, 6, 7]
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_results.json")

def get_pos_size(score, ticker):
    sizes = {10:5000,9:4000,8:3000,7:2000,6:1500,5:1000}
    base = sizes.get(min(score,10), 1000)
    return round(base * 0.75) if ticker in VOLATILE else base

def fetch_1m(ticker, days=120):
    print(f"  Fetching {ticker} ({days} days)...")
    frames = []
    end = datetime.datetime.now(MARKET_TZ)
    chunks = (days // 7) + (1 if days % 7 else 0)
    for i in range(chunks):
        ce = end - datetime.timedelta(days=i*7)
        cs = ce - datetime.timedelta(days=7)
        try:
            df = yf.download(ticker, start=cs.strftime("%Y-%m-%d"),
                             end=ce.strftime("%Y-%m-%d"),
                             interval="1m", progress=False, auto_adjust=True)
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
    print(f"  {ticker}: {len(df)} bars / {len(df.index.normalize().unique())} days")
    return df

def calc_rsi(series, window=14):
    if len(series) < window+1: return 50.0
    try:
        v = ta.momentum.RSIIndicator(series.squeeze(), window=window).rsi().iloc[-1]
        return round(float(v),2) if not pd.isna(v) else 50.0
    except: return 50.0

def calc_ema(series, window=9):
    if len(series) < window: return float(series.iloc[-1])
    try: return round(float(series.ewm(span=window,adjust=False).mean().iloc[-1]),4)
    except: return float(series.iloc[-1])

def score_signal(direction, sig_type, rsi, vol_ratio, price, vwap,
                 ema9, day_bias, rsi_bull_min, momentum_ok):
    score = 0
    if day_bias == direction: score += 2
    elif day_bias == "neutral": score += 1
    if direction == "long":
        if rsi_bull_min+5 <= rsi <= rsi_bull_min+15: score += 2
        elif rsi_bull_min <= rsi <= rsi_bull_min+20: score += 1
    else:
        rsi_sell = RSI_BEAR_MAX
        if rsi_sell-15 <= rsi <= rsi_sell-5: score += 2
        elif rsi_sell-20 <= rsi <= rsi_sell: score += 1
    if vol_ratio >= 2.5: score += 2
    elif vol_ratio >= 1.5: score += 1
    if vwap and vwap > 0:
        if direction=="long" and price > vwap: score += 1
        elif direction=="short" and price < vwap: score += 1
    if direction=="long" and price > ema9: score += 1
    elif direction=="short" and price < ema9: score += 1
    if sig_type == "vwap_reclaim": score += 1
    if momentum_ok: score += 1  # bonus for momentum confirmation
    return min(score, 10)

def close_trade(trades, entry, exit_price, result, date, ts):
    pnl_pct = ((exit_price-entry["price"])/entry["price"]*100
               if entry["dir"]=="long"
               else (entry["price"]-exit_price)/entry["price"]*100)
    trades.append({
        "date":         date.strftime("%Y-%m-%d"),
        "ticker":       entry["ticker"],
        "direction":    "Long" if entry["dir"]=="long" else "Short",
        "entry":        round(entry["price"],4),
        "exit":         round(exit_price,4),
        "result":       result,
        "pnl_pct":      round(pnl_pct,3),
        "pnl_dollar":   round(pnl_pct/100*entry["pos_size"],2),
        "rsi":          round(entry["rsi"],1),
        "vol_ratio":    entry["vol_ratio"],
        "entry_time":   entry["time"],
        "hour":         entry["hour"],
        "signal_type":  entry["sig_type"],
        "signal_score": entry["score"],
        "pos_size":     entry["pos_size"],
        "day_bias":     entry["day_bias"],
        "trail_used":   entry["trail_active"],
    })

def run_strategy(df, ticker, cfg):
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
        highs   = day_df["High"].squeeze().tolist()
        lows    = day_df["Low"].squeeze().tolist()
        day_idx = list(day_df.index)

        # Opening drive bias
        drive_df = day_df.between_time("09:30","09:35")
        if len(drive_df) < 2: continue
        drive_open  = float(drive_df["Open"].iloc[0])
        drive_close = float(drive_df["Close"].iloc[-1])
        move_pct    = (drive_close-drive_open)/drive_open*100
        prev_days   = df[df.index.date < date.date()]
        avg_vol     = float(prev_days["Volume"].tail(200).mean()) if len(prev_days)>0 else 1.0
        drive_vol   = float(drive_df["Volume"].sum())
        vr_drive    = drive_vol/avg_vol if avg_vol>0 else 1.0
        if move_pct > 0.3 and vr_drive > 1.5: day_bias="long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias="short"
        else: day_bias="neutral"

        # Open price for trend strength filter
        open_price = float(day_df["Open"].iloc[0])

        # VWAP
        cl = day_df["Close"].squeeze()
        hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze()
        vo = day_df["Volume"].squeeze()
        tp = (hi+lo+cl)/3
        vwap_list = ((tp*vo).cumsum()/vo.cumsum()).tolist()

        in_trade = False
        entry    = None
        vwap_confirm_count = 0  # bars price stayed above/below VWAP after reclaim

        for i, ts in enumerate(day_idx):
            if i < 10: continue
            hour, minute = ts.hour, ts.minute
            if hour >= 15 and minute >= 30: break  # hard cutoff

            price      = closes[i]
            prev_price = closes[i-1] if i>0 else price
            prev2_price= closes[i-2] if i>1 else prev_price
            vwap       = vwap_list[i] if i<len(vwap_list) else 0

            # Exit check
            if in_trade and entry:
                d = entry["dir"]
                if d == "long":
                    m = (price-entry["price"])/entry["price"]
                    if m >= TRAIL_TRIGGER and not entry["trail_active"]:
                        entry["trail_active"]=True; entry["trail_peak"]=price
                        entry["trail_stop"]=price*(1-TRAIL_STOP)
                    elif entry["trail_active"] and price>entry["trail_peak"]:
                        entry["trail_peak"]=price
                        entry["trail_stop"]=price*(1-TRAIL_STOP)
                else:
                    m = (entry["price"]-price)/entry["price"]
                    if m >= TRAIL_TRIGGER and not entry["trail_active"]:
                        entry["trail_active"]=True; entry["trail_peak"]=price
                        entry["trail_stop"]=price*(1+TRAIL_STOP)
                    elif entry["trail_active"] and price<entry["trail_peak"]:
                        entry["trail_peak"]=price
                        entry["trail_stop"]=price*(1+TRAIL_STOP)

                ht = (d=="long" and price>=entry["target"]) or (d=="short" and price<=entry["target"])
                hs = (d=="long" and price<=entry["stop"])   or (d=="short" and price>=entry["stop"])
                htr= entry["trail_active"] and (
                    (d=="long" and price<=entry["trail_stop"]) or
                    (d=="short" and price>=entry["trail_stop"]))
                htime = hour>=15 and minute>=25

                if ht or hs or htr or htime:
                    res = "Target Hit" if ht else ("Trailing Stop" if htr else ("Time Exit" if htime else "Stop Loss Hit"))
                    ep  = entry["target"] if ht else (entry["trail_stop"] if htr else entry["stop"] if hs else price)
                    close_trade(trades, entry, ep, res, date, ts)
                    in_trade=False; entry=None
                    continue

            if in_trade: continue

            # Trend strength filter — price must be 0.3% away from open
            trend_strength = abs(price-open_price)/open_price
            if trend_strength < TREND_MIN_PCT: continue

            # Momentum confirmation — last 2 bars moving in same direction
            momentum_long  = prev_price > prev2_price and price > prev_price
            momentum_short = prev_price < prev2_price and price < prev_price

            # RSI + indicators
            rsi_sl = pd.Series(closes[max(0,i-28):i+1])
            rsi    = calc_rsi(rsi_sl)
            ema_sl = pd.Series(closes[max(0,i-30):i+1])
            ema9   = calc_ema(ema_sl, 9)
            ema21  = calc_ema(ema_sl, 21)
            vol_sl = pd.Series(volumes[max(0,i-20):i])
            avg_v  = float(vol_sl.mean()) if len(vol_sl)>0 else 1.0
            vol_ratio = round(volumes[i]/avg_v,2) if avg_v>0 else 1.0

            # Signals
            sig=None; dirn=None; momentum_ok=False

            # VWAP Reclaim
            if vwap > 0:
                was_below = closes[i-1] < vwap if i>0 else False
                now_above = price > vwap
                was_above = closes[i-1] > vwap if i>0 else False
                now_below = price < vwap

                if was_below and now_above and vol_ratio>=VOLUME_MIN:
                    if day_bias in ("long","neutral") and rsi_bull_min<=rsi<=rsi_bull_max:
                        sig="vwap_reclaim"; dirn="long"; momentum_ok=momentum_long

                elif was_above and now_below and vol_ratio>=VOLUME_MIN:
                    if day_bias in ("short","neutral") and rsi_bear_min<=rsi<=rsi_bear_max:
                        sig="vwap_reclaim"; dirn="short"; momentum_ok=momentum_short

            # EMA Pullback
            if not sig:
                if (closes[i-1]<=ema9*1.001 and price>ema9 and price>ema21
                        and vol_ratio>=VOLUME_MIN and momentum_long):
                    if day_bias in ("long","neutral") and rsi_bull_min<=rsi<=rsi_bull_max:
                        sig="ema_pullback"; dirn="long"; momentum_ok=True
                elif (closes[i-1]>=ema9*0.999 and price<ema9 and price<ema21
                        and vol_ratio>=VOLUME_MIN and momentum_short):
                    if day_bias in ("short","neutral") and rsi_bear_min<=rsi<=rsi_bear_max:
                        sig="ema_pullback"; dirn="short"; momentum_ok=True

            if not sig: continue

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap,
                              ema9, day_bias, rsi_bull_min, momentum_ok)
            if sc < min_sc: continue

            ps  = get_pos_size(sc, ticker)
            tgt = price*(1+gain_pct) if dirn=="long" else price*(1-gain_pct)
            stp = price*(1-stop_pct) if dirn=="long" else price*(1+stop_pct)

            in_trade = True
            entry = {
                "dir":dirn,"price":price,"target":round(tgt,4),"stop":round(stp,4),
                "trail_active":False,"trail_peak":price,"trail_stop":None,
                "rsi":rsi,"vol_ratio":vol_ratio,"sig_type":sig,"score":sc,
                "pos_size":ps,"time":ts.strftime("%I:%M %p"),"hour":hour,
                "day_bias":day_bias,"ticker":ticker,
            }
    return trades

def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"trail_exits":0,"time_exits":0,
                "win_rate":0,"total_pnl_pct":0,"total_pnl_dollar":0,
                "best":0,"worst":0,"max_drawdown":0,"sharpe":0,"avg_score":0}
    wins   = [t for t in trades if t["result"]=="Target Hit" or
              ((t["result"] in ("Trailing Stop","Time Exit")) and t["pnl_pct"]>0)]
    losses = [t for t in trades if t["result"]=="Stop Loss Hit" or
              ((t["result"] in ("Trailing Stop","Time Exit")) and t["pnl_pct"]<=0)]
    trails = [t for t in trades if t["result"]=="Trailing Stop"]
    times  = [t for t in trades if t["result"]=="Time Exit"]
    pnls   = [t["pnl_pct"] for t in trades]
    dols   = [t["pnl_dollar"] for t in trades]
    peak=0;dd=0;cum=0
    for p in pnls:
        cum+=p
        if cum>peak: peak=cum
        if cum-peak<dd: dd=cum-peak
    import statistics
    sharpe=0
    if len(pnls)>1:
        try: sharpe=round(statistics.mean(pnls)/statistics.stdev(pnls)*(252**0.5),3)
        except: pass
    # Per ticker breakdown
    by_ticker={}
    for t in trades:
        tk=t["ticker"]
        if tk not in by_ticker: by_ticker[tk]={"trades":0,"pnl":0.0,"wins":0}
        by_ticker[tk]["trades"]+=1
        by_ticker[tk]["pnl"]+=t["pnl_dollar"]
        if t in wins: by_ticker[tk]["wins"]+=1
    return {
        "trades":len(trades),"wins":len(wins),"losses":len(losses),
        "trail_exits":len(trails),"time_exits":len(times),
        "win_rate":round(len(wins)/len(trades)*100,2) if trades else 0,
        "total_pnl_pct":round(sum(pnls),3),"total_pnl_dollar":round(sum(dols),2),
        "best":round(max(pnls),3) if pnls else 0,
        "worst":round(min(pnls),3) if pnls else 0,
        "max_drawdown":round(dd,3),"sharpe":sharpe,
        "avg_score":round(sum(t["signal_score"] for t in trades)/len(trades),1) if trades else 0,
        "by_ticker":by_ticker,
    }

def main():
    print("="*60)
    print(f"  NYLO Backtest v15 — Maximum Risk / No Limits")
    print(f"  Tickers  : {', '.join(TICKERS)}")
    print(f"  Lookback : {LOOKBACK_DAYS} days")
    print(f"  Sizing   : $1k-$5k (NVDA/TSLA 75%)")
    print(f"  No trade limit, no dead zone")
    print("="*60)

    data = {}
    for ticker in TICKERS:
        df = fetch_1m(ticker, LOOKBACK_DAYS)
        data[ticker] = df

    print(f"\n[1/2] Baseline strategy...")
    base_cfg = {
        "rsi_bull_min":RSI_BULL_MIN,"rsi_bull_max":RSI_BULL_MAX,
        "rsi_bear_min":RSI_BEAR_MIN,"rsi_bear_max":RSI_BEAR_MAX,
        "min_score":MIN_SCORE,"gain_pct":GAIN_TARGET_PCT,"stop_pct":STOP_LOSS_PCT,
    }
    base_trades = []
    for ticker in TICKERS:
        if data[ticker].empty: continue
        t = run_strategy(data[ticker], ticker, base_cfg)
        base_trades.extend(t)
        print(f"  {ticker}: {len(t)} trades")
    base_trades.sort(key=lambda t:(t["date"],t["entry_time"]))
    bs = calc_stats(base_trades)
    print(f"  TOTAL: {bs['trades']} trades | {bs['win_rate']:.1f}% WR | "
          f"P&L: {bs['total_pnl_pct']:+.2f}% (${bs['total_pnl_dollar']:+.2f}) | "
          f"Sharpe: {bs['sharpe']} | Avg score: {bs['avg_score']}/10")
    for tk,s in bs.get("by_ticker",{}).items():
        print(f"    {tk}: {s['trades']} trades | ${s['pnl']:+.2f}")

    print(f"\n[2/2] Parameter sweep...")
    combos = [(b,s,ms) for b in SWEEP_RSI_BUY for s in SWEEP_RSI_SELL
              for ms in SWEEP_MIN_SCORE if b>s]
    sweep = []
    for done,(rb,rs,ms) in enumerate(combos):
        cfg={**base_cfg,"rsi_bull_min":rb,"rsi_bull_max":rb+18,
             "rsi_bear_min":rs-18,"rsi_bear_max":rs,"min_score":ms}
        t=[]
        for ticker in TICKERS:
            if not data[ticker].empty:
                t.extend(run_strategy(data[ticker],ticker,cfg))
        t.sort(key=lambda x:(x["date"],x["entry_time"]))
        st=calc_stats(t)
        sc=st["win_rate"]*0.5+st["total_pnl_pct"]*0.4+st["trades"]*0.1
        sweep.append({"rsi_buy":rb,"rsi_sell":rs,"min_score":ms,"stats":st,"score":round(sc,3)})
        sys.stdout.write(f"\r  Progress: {done+1}/{len(combos)}")
        sys.stdout.flush()

    sweep.sort(key=lambda r:r["score"],reverse=True)
    best=sweep[0]
    print(f"\n  Best: RSI {best['rsi_buy']}/{best['rsi_sell']} score={best['min_score']} "
          f"→ {best['stats']['win_rate']:.0f}% WR, ${best['stats']['total_pnl_dollar']:+.2f}")

    out={
        "generated_at":datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str":datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "config":{
            "tickers":TICKERS,"lookback_days":LOOKBACK_DAYS,
            "strategy":"VWAP Reclaim + EMA Pullback + Opening Drive + Momentum",
            "baseline":{
                "rsi_bull_min":RSI_BULL_MIN,"rsi_bull_max":RSI_BULL_MAX,
                "rsi_bear_min":RSI_BEAR_MIN,"rsi_bear_max":RSI_BEAR_MAX,
                "min_score":MIN_SCORE,"gain_pct":GAIN_TARGET_PCT*100,
                "stop_pct":STOP_LOSS_PCT*100,"no_trade_limit":True,
                "no_dead_zone":True,"momentum_confirmation":True,
            }
        },
        "baseline":{"stats":bs,"trades":base_trades},
        "sweep":sweep[:60],
    }
    with open(OUT,"w") as f:
        json.dump(out,f,indent=2)
    print(f"\n✅ Results saved → {OUT}")
    print(f"   git add backtest_results.json && git commit -m 'v15 backtest' && git push")
    print("="*60)

if __name__=="__main__":
    main()
