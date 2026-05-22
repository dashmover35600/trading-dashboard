"""
NYLO Backtesting Engine v16
============================
Complete strategy overhaul with all improvements:

Entry filters:
  - Momentum strength (price velocity acceleration)
  - Volume rising confirmation (not just above average)
  - Time of day weighting (9:35-10:30 AM = +2 score bonus)
  - Pre-market gap filter

Exit management:
  - Partial exit: 50% at +0.75%, 50% runs to +1.5%
  - Time-based trail tightening after 11 AM
  - Breakeven stop at +0.5%

Risk management:
  - Daily loss limit: stop at -$500
  - Consecutive loss pause: 3 losses = 30 min pause
  - VIX-adjusted sizing
  - Transaction costs: 0.02% slippage per trade

Out-of-sample:
  - Train Feb-April, test May separately

Tickers: QQQ + NVDA
Lookback: 120 days
"""

import yfinance as yf
import pandas as pd
import ta
import json
import datetime
import pytz
import os
import sys
import statistics

TICKERS         = ["QQQ", "NVDA", "AMD", "MSFT", "AAPL", "META", "GOOGL", "SPY"]
VOLATILE        = ["NVDA", "AMD", "META"]  # 75% sizing on volatile names
LOOKBACK_DAYS   = 120  # keep 120 days, more tickers = more trades
MARKET_TZ       = pytz.timezone("America/New_York")

# Exit management
GAIN_TARGET_PCT    = 0.015   # full target 1.5%
PARTIAL_EXIT_PCT   = 0.0075  # take 50% at 0.75%
STOP_LOSS_PCT      = 0.0075
BREAKEVEN_TRIGGER  = 0.005   # move stop to breakeven at +0.5%
TRAIL_TRIGGER      = 0.0075
TRAIL_STOP         = 0.003
TRAIL_STOP_LATE    = 0.002   # tighter trail after 11 AM

# Entry filters
MIN_SCORE       = 5
RSI_BULL_MIN    = 52
RSI_BULL_MAX    = 72
RSI_BEAR_MIN    = 28
RSI_BEAR_MAX    = 48
VOLUME_MIN      = 1.2
TREND_MIN_PCT   = 0.002

# Risk management
DAILY_LOSS_LIMIT    = -500.0
CONSEC_LOSS_PAUSE   = 3       # pause after 3 consecutive losses
SLIPPAGE_PCT        = 0.0002  # 0.02% slippage per trade
VIX_HIGH            = 20.0    # reduce size above this

# Sweep
SWEEP_RSI_BUY   = [50, 52, 55, 58]
SWEEP_RSI_SELL  = [40, 42, 45, 48]
SWEEP_MIN_SCORE = [3, 4, 5]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "backtest_results.json")

def get_pos_size(score, ticker, vix=15):
    sizes = {10:5000,9:4000,8:3000,7:2000,6:1500,5:1000,4:750}
    base = sizes.get(min(score,10), 1000)
    if ticker in VOLATILE:
        base = round(base * 0.75)
    # VIX adjustment
    if vix > VIX_HIGH:
        base = round(base * 0.75)
    return base

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

def is_prime_window(hour, minute):
    """9:35 AM - 10:30 AM is the prime trading window"""
    if hour == 9 and minute >= 35: return True
    if hour == 10 and minute <= 30: return True
    return False

def is_late_session(hour):
    return hour >= 11

def score_signal(direction, sig_type, rsi, vol_ratio, price, vwap,
                 ema9, day_bias, rsi_bull_min, momentum_ok,
                 vol_rising, hour, minute, gap_pct):
    score = 0

    # Day bias (+2)
    if day_bias == direction: score += 2
    elif day_bias == "neutral": score += 1

    # RSI zone (+2)
    if direction == "long":
        mid = rsi_bull_min + 10
        if abs(rsi - mid) <= 8: score += 2
        elif rsi_bull_min <= rsi <= rsi_bull_min+20: score += 1
    else:
        mid = RSI_BEAR_MAX - 10
        if abs(rsi - mid) <= 8: score += 2
        elif RSI_BEAR_MIN <= rsi <= RSI_BEAR_MAX: score += 1

    # Volume (+2)
    if vol_ratio >= 2.5: score += 2
    elif vol_ratio >= 1.5: score += 1

    # VWAP alignment (+1)
    if vwap and vwap > 0:
        if direction=="long" and price > vwap: score += 1
        elif direction=="short" and price < vwap: score += 1

    # EMA alignment (+1)
    if direction=="long" and price > ema9: score += 1
    elif direction=="short" and price < ema9: score += 1

    # Signal type premium (+1)
    if sig_type == "vwap_reclaim": score += 1

    # Momentum confirmation (+1)
    if momentum_ok: score += 1

    # Volume rising (+1)
    if vol_rising: score += 1

    # Prime window bonus (+2)
    if is_prime_window(hour, minute): score += 2

    # Gap alignment (+1)
    if gap_pct > 0.3 and direction == "long": score += 1
    elif gap_pct < -0.3 and direction == "short": score += 1

    return min(score, 10)

def apply_slippage(price, direction):
    """Apply realistic slippage to entry price"""
    if direction == "long":
        return round(price * (1 + SLIPPAGE_PCT), 4)
    return round(price * (1 - SLIPPAGE_PCT), 4)

def close_trade(trades, entry, exit_price, result, date, ts, partial=False):
    # Apply slippage to exit
    if entry["dir"] == "long":
        exit_price = round(exit_price * (1 - SLIPPAGE_PCT), 4)
    else:
        exit_price = round(exit_price * (1 + SLIPPAGE_PCT), 4)

    pnl_pct = ((exit_price - entry["price"]) / entry["price"] * 100
               if entry["dir"] == "long"
               else (entry["price"] - exit_price) / entry["price"] * 100)
    size = entry["pos_size"] * (0.5 if partial else 1.0)
    trades.append({
        "date":         date.strftime("%Y-%m-%d"),
        "ticker":       entry["ticker"],
        "direction":    "Long" if entry["dir"]=="long" else "Short",
        "entry":        round(entry["price"],4),
        "exit":         round(exit_price,4),
        "result":       result + (" (partial)" if partial else ""),
        "pnl_pct":      round(pnl_pct,3),
        "pnl_dollar":   round(pnl_pct/100*size,2),
        "rsi":          round(entry["rsi"],1),
        "vol_ratio":    entry["vol_ratio"],
        "entry_time":   entry["time"],
        "hour":         entry["hour"],
        "signal_type":  entry["sig_type"],
        "signal_score": entry["score"],
        "pos_size":     size,
        "day_bias":     entry["day_bias"],
        "trail_used":   entry["trail_active"],
        "prime_window": entry.get("prime_window", False),
    })
    return pnl_pct

def run_strategy(df, ticker, cfg, vix=15):
    rsi_bull_min = cfg["rsi_bull_min"]
    rsi_bull_max = cfg["rsi_bull_max"]
    rsi_bear_min = cfg["rsi_bear_min"]
    rsi_bear_max = cfg["rsi_bear_max"]
    min_sc       = cfg["min_score"]
    trades = []
    dates  = sorted(df.index.normalize().unique())

    daily_pnl      = 0.0
    consec_losses  = 0
    pause_until    = None

    for date in dates:
        # Reset daily risk management
        daily_pnl = 0.0
        consec_losses = 0
        pause_until = None

        day_df = df[df.index.date == date.date()]
        if len(day_df) < 20: continue

        closes  = day_df["Close"].squeeze().tolist()
        volumes = day_df["Volume"].squeeze().tolist()
        day_idx = list(day_df.index)
        open_price = closes[0] if closes else 0

        # Opening drive bias
        drive_df = day_df.between_time("09:30","09:35")
        if len(drive_df) < 2: continue
        drive_open  = float(drive_df["Open"].iloc[0])
        drive_close = float(drive_df["Close"].iloc[-1])
        move_pct    = (drive_close - drive_open) / drive_open * 100
        prev_days   = df[df.index.date < date.date()]
        avg_vol     = float(prev_days["Volume"].tail(200).mean()) if len(prev_days)>0 else 1.0
        drive_vol   = float(drive_df["Volume"].sum())
        vr_drive    = drive_vol / avg_vol if avg_vol > 0 else 1.0

        # Gap calculation
        if len(prev_days) > 0:
            prev_close = float(prev_days["Close"].iloc[-1])
            gap_pct = (drive_open - prev_close) / prev_close * 100
        else:
            gap_pct = 0

        if move_pct > 0.3 and vr_drive > 1.5:   day_bias = "long"
        elif move_pct < -0.3 and vr_drive > 1.5: day_bias = "short"
        else:                                      day_bias = "neutral"

        # VWAP
        cl = day_df["Close"].squeeze()
        hi = day_df["High"].squeeze()
        lo = day_df["Low"].squeeze()
        vo = day_df["Volume"].squeeze()
        tp = (hi+lo+cl)/3
        vwap_list = ((tp*vo).cumsum()/vo.cumsum()).tolist()

        in_trade    = False
        entry       = None
        partial_done= False

        for i, ts in enumerate(day_idx):
            if i < 10: continue
            hour, minute = ts.hour, ts.minute
            if (hour >= 15 and minute >= 25): break

            # Daily loss limit check
            if daily_pnl <= DAILY_LOSS_LIMIT:
                break

            # Consecutive loss pause
            if pause_until and ts < pause_until:
                continue
            else:
                pause_until = None

            price      = closes[i]
            prev_price = closes[i-1] if i>0 else price
            prev2      = closes[i-2] if i>1 else prev_price
            vwap       = vwap_list[i] if i<len(vwap_list) else 0

            # Exit management
            if in_trade and entry:
                d = entry["dir"]

                # Update breakeven stop
                if not entry["breakeven_set"]:
                    move = (price-entry["price"])/entry["price"] if d=="long" else (entry["price"]-price)/entry["price"]
                    if move >= BREAKEVEN_TRIGGER:
                        entry["stop"] = entry["price"]  # move to breakeven
                        entry["breakeven_set"] = True

                # Partial exit at 0.75%
                if not partial_done:
                    hit_partial = (d=="long" and price>=entry["price"]*(1+PARTIAL_EXIT_PCT)) or \
                                  (d=="short" and price<=entry["price"]*(1-PARTIAL_EXIT_PCT))
                    if hit_partial:
                        ep = entry["price"]*(1+PARTIAL_EXIT_PCT) if d=="long" else entry["price"]*(1-PARTIAL_EXIT_PCT)
                        pnl = close_trade(trades, entry, ep, "Partial Exit", date, ts, partial=True)
                        daily_pnl += entry["pos_size"]*0.5*pnl/100
                        partial_done = True
                        # Activate trail for remaining 50%
                        entry["trail_active"] = True
                        entry["trail_peak"] = price

                # Update trailing stop
                trail_pct = TRAIL_STOP_LATE if is_late_session(hour) else TRAIL_STOP
                if entry["trail_active"]:
                    if d=="long" and price > entry["trail_peak"]:
                        entry["trail_peak"] = price
                        entry["trail_stop"] = price*(1-trail_pct)
                    elif d=="short" and price < entry["trail_peak"]:
                        entry["trail_peak"] = price
                        entry["trail_stop"] = price*(1+trail_pct)
                elif not partial_done:
                    move = (price-entry["price"])/entry["price"] if d=="long" else (entry["price"]-price)/entry["price"]
                    if move >= TRAIL_TRIGGER:
                        entry["trail_active"] = True
                        entry["trail_peak"] = price
                        entry["trail_stop"] = price*(1-trail_pct) if d=="long" else price*(1+trail_pct)

                # Check exits
                ht = (d=="long" and price>=entry["target"]) or (d=="short" and price<=entry["target"])
                hs = (d=="long" and price<=entry["stop"])   or (d=="short" and price>=entry["stop"])
                htr= entry["trail_active"] and (
                    (d=="long" and price<=entry.get("trail_stop",0)) or
                    (d=="short" and entry.get("trail_stop") and price>=entry["trail_stop"]))
                htime = hour>=15 and minute>=20

                if ht or hs or htr or htime:
                    res = "Target Hit" if ht else ("Trailing Stop" if htr else ("Time Exit" if htime else "Stop Loss Hit"))
                    ep  = entry["target"] if ht else (entry.get("trail_stop", price) if htr else entry["stop"] if hs else price)
                    remaining = 0.5 if partial_done else 1.0
                    pnl_pct = ((ep-entry["price"])/entry["price"]*100 if d=="long"
                               else (entry["price"]-ep)/entry["price"]*100)
                    pnl_dollar = entry["pos_size"]*remaining*pnl_pct/100

                    if partial_done:
                        # Log remaining 50%
                        close_trade(trades, {**entry,"pos_size":entry["pos_size"]*0.5}, ep, res, date, ts)
                    else:
                        close_trade(trades, entry, ep, res, date, ts)

                    daily_pnl += pnl_dollar
                    if pnl_pct <= 0:
                        consec_losses += 1
                        if consec_losses >= CONSEC_LOSS_PAUSE:
                            pause_until = ts + datetime.timedelta(minutes=30)
                            consec_losses = 0
                    else:
                        consec_losses = 0

                    in_trade = False; entry = None; partial_done = False
                    continue

            if in_trade: continue

            # Trend strength
            if abs(price-open_price)/open_price < TREND_MIN_PCT: continue

            # Momentum confirmation (2 consecutive bars)
            momentum_long  = prev_price > prev2 and price > prev_price
            momentum_short = prev_price < prev2 and price < prev_price

            # Volume rising
            vol_rising = (volumes[i] > volumes[i-1]) if i>0 else False

            # RSI
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
            if vwap > 0 and i > 0:
                was_below = closes[i-1] < vwap
                now_above = price > vwap
                was_above = closes[i-1] > vwap
                now_below = price < vwap
                if was_below and now_above and vol_ratio>=VOLUME_MIN:
                    if day_bias in ("long","neutral") and rsi_bull_min<=rsi<=rsi_bull_max:
                        sig="vwap_reclaim"; dirn="long"; momentum_ok=momentum_long
                elif was_above and now_below and vol_ratio>=VOLUME_MIN:
                    if day_bias in ("short","neutral") and rsi_bear_min<=rsi<=rsi_bear_max:
                        sig="vwap_reclaim"; dirn="short"; momentum_ok=momentum_short

            # EMA Pullback
            if not sig and i > 0:
                if (closes[i-1]<=ema9*1.001 and price>ema9 and price>ema21 and vol_ratio>=VOLUME_MIN and momentum_long):
                    if day_bias in ("long","neutral") and rsi_bull_min<=rsi<=rsi_bull_max:
                        sig="ema_pullback"; dirn="long"; momentum_ok=True
                elif (closes[i-1]>=ema9*0.999 and price<ema9 and price<ema21 and vol_ratio>=VOLUME_MIN and momentum_short):
                    if day_bias in ("short","neutral") and rsi_bear_min<=rsi<=rsi_bear_max:
                        sig="ema_pullback"; dirn="short"; momentum_ok=True

            if not sig: continue

            sc = score_signal(dirn, sig, rsi, vol_ratio, price, vwap,
                              ema9, day_bias, rsi_bull_min, momentum_ok,
                              vol_rising, hour, minute, gap_pct)
            if sc < min_sc: continue

            # Apply slippage to entry
            entry_price = apply_slippage(price, dirn)
            ps = get_pos_size(sc, ticker, vix)
            tgt = entry_price*(1+GAIN_TARGET_PCT) if dirn=="long" else entry_price*(1-GAIN_TARGET_PCT)
            stp = entry_price*(1-STOP_LOSS_PCT) if dirn=="long" else entry_price*(1+STOP_LOSS_PCT)

            in_trade = True
            partial_done = False
            entry = {
                "dir":dirn,"price":entry_price,"target":round(tgt,4),"stop":round(stp,4),
                "trail_active":False,"trail_peak":entry_price,"trail_stop":None,
                "breakeven_set":False,
                "rsi":rsi,"vol_ratio":vol_ratio,"sig_type":sig,"score":sc,
                "pos_size":ps,"time":ts.strftime("%I:%M %p"),"hour":hour,
                "day_bias":day_bias,"ticker":ticker,
                "prime_window":is_prime_window(hour,minute),
            }
    return trades

def calc_stats(trades):
    if not trades:
        return {"trades":0,"wins":0,"losses":0,"win_rate":0,
                "total_pnl_pct":0,"total_pnl_dollar":0,
                "best":0,"worst":0,"max_drawdown":0,"sharpe":0,
                "avg_score":0,"profit_factor":0,"expectancy":0,
                "prime_win_rate":0}
    wins   = [t for t in trades if t["result"].startswith("Target Hit") or
              (("Trailing Stop" in t["result"] or "Time Exit" in t["result"] or "Partial" in t["result"]) and t["pnl_pct"]>0)]
    losses = [t for t in trades if t["result"].startswith("Stop Loss") or
              (("Trailing Stop" in t["result"] or "Time Exit" in t["result"]) and t["pnl_pct"]<=0)]
    pnls   = [t["pnl_pct"] for t in trades]
    dols   = [t["pnl_dollar"] for t in trades]

    # Max drawdown
    peak=0;dd=0;cum=0
    for p in pnls:
        cum+=p
        if cum>peak: peak=cum
        if cum-peak<dd: dd=cum-peak

    # Sharpe
    sharpe=0
    if len(pnls)>1:
        try: sharpe=round(statistics.mean(pnls)/statistics.stdev(pnls)*(252**0.5),3)
        except: pass

    # Profit factor
    gross_wins  = sum(t["pnl_dollar"] for t in wins)
    gross_losses= abs(sum(t["pnl_dollar"] for t in losses))
    profit_factor = round(gross_wins/gross_losses,3) if gross_losses>0 else 999

    # Expectancy
    avg_win  = gross_wins/len(wins) if wins else 0
    avg_loss = gross_losses/len(losses) if losses else 0
    wr = len(wins)/len(trades)
    expectancy = round(wr*avg_win - (1-wr)*avg_loss, 2)

    # Prime window win rate
    prime = [t for t in trades if t.get("prime_window")]
    prime_wins = [t for t in prime if t in wins]
    prime_wr = round(len(prime_wins)/len(prime)*100,1) if prime else 0

    # By ticker
    by_ticker={}
    for t in trades:
        tk=t["ticker"]
        if tk not in by_ticker: by_ticker[tk]={"trades":0,"pnl":0.0,"wins":0}
        by_ticker[tk]["trades"]+=1
        by_ticker[tk]["pnl"]+=t["pnl_dollar"]
        if t in wins: by_ticker[tk]["wins"]+=1

    return {
        "trades":len(trades),"wins":len(wins),"losses":len(losses),
        "win_rate":round(len(wins)/len(trades)*100,2),
        "total_pnl_pct":round(sum(pnls),3),"total_pnl_dollar":round(sum(dols),2),
        "best":round(max(pnls),3) if pnls else 0,
        "worst":round(min(pnls),3) if pnls else 0,
        "max_drawdown":round(dd,3),"sharpe":sharpe,
        "avg_score":round(sum(t["signal_score"] for t in trades)/len(trades),1),
        "profit_factor":profit_factor,
        "expectancy":expectancy,
        "prime_win_rate":prime_wr,
        "by_ticker":by_ticker,
    }

def main():
    print("="*60)
    print(f"  NYLO Backtest v16.1 — 10x Trades (8 tickers, min score 3)")
    print(f"  Tickers  : {', '.join(TICKERS)}")
    print(f"  Lookback : {LOOKBACK_DAYS} days")
    print(f"  Slippage : {SLIPPAGE_PCT*100:.2f}% per trade")
    print(f"  Partial exits, breakeven stops, risk management")
    print("="*60)

    data = {}
    for ticker in TICKERS:
        df = fetch_1m(ticker, LOOKBACK_DAYS)
        data[ticker] = df

    # ── Baseline ──────────────────────────────────────────────
    print(f"\n[1/3] Baseline strategy...")
    base_cfg = {
        "rsi_bull_min":RSI_BULL_MIN,"rsi_bull_max":RSI_BULL_MAX,
        "rsi_bear_min":RSI_BEAR_MIN,"rsi_bear_max":RSI_BEAR_MAX,
        "min_score":MIN_SCORE,
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
          f"P&L: ${bs['total_pnl_dollar']:+.2f} | Sharpe: {bs['sharpe']} | "
          f"Profit factor: {bs['profit_factor']} | Expectancy: ${bs['expectancy']:.2f}/trade")
    print(f"  Prime window WR: {bs['prime_win_rate']}% | Max DD: {bs['max_drawdown']:.2f}%")
    for tk,s in bs.get("by_ticker",{}).items():
        wr = round(s['wins']/s['trades']*100,1) if s['trades']>0 else 0
        print(f"    {tk}: {s['trades']} trades | ${s['pnl']:+.2f} | {wr}% WR")

    # ── Out-of-sample test ────────────────────────────────────
    print(f"\n[2/3] Out-of-sample test (train=before May, test=May only)...")
    may_start = datetime.date(2026, 5, 1)
    train_trades, test_trades = [], []
    for t in base_trades:
        d = datetime.date.fromisoformat(t["date"])
        if d >= may_start: test_trades.append(t)
        else: train_trades.append(t)
    ts_train = calc_stats(train_trades)
    ts_test  = calc_stats(test_trades)
    print(f"  Train (pre-May): {ts_train['trades']} trades | {ts_train['win_rate']:.1f}% WR | ${ts_train['total_pnl_dollar']:+.2f}")
    print(f"  Test  (May):     {ts_test['trades']} trades | {ts_test['win_rate']:.1f}% WR | ${ts_test['total_pnl_dollar']:+.2f}")
    oos_consistency = abs(ts_test['win_rate'] - ts_train['win_rate']) < 15
    print(f"  Out-of-sample: {'✅ CONSISTENT' if oos_consistency else '⚠️ DIVERGED'} (diff={abs(ts_test['win_rate']-ts_train['win_rate']):.1f}%)")

    # ── Parameter sweep ───────────────────────────────────────
    print(f"\n[3/3] Parameter sweep...")
    combos = [(b,s,ms) for b in SWEEP_RSI_BUY for s in SWEEP_RSI_SELL
              for ms in SWEEP_MIN_SCORE if b>s]
    sweep = []
    for done,(rb,rs,ms) in enumerate(combos):
        cfg={**base_cfg,"rsi_bull_min":rb,"rsi_bull_max":rb+20,
             "rsi_bear_min":rs-20,"rsi_bear_max":rs,"min_score":ms}
        t=[]
        for ticker in TICKERS:
            if not data[ticker].empty:
                t.extend(run_strategy(data[ticker],ticker,cfg))
        t.sort(key=lambda x:(x["date"],x["entry_time"]))
        st=calc_stats(t)
        sc=st["win_rate"]*0.4+st["profit_factor"]*10+st["expectancy"]*0.5
        sweep.append({"rsi_buy":rb,"rsi_sell":rs,"min_score":ms,"stats":st,"score":round(sc,3)})
        sys.stdout.write(f"\r  Progress: {done+1}/{len(combos)}")
        sys.stdout.flush()

    sweep.sort(key=lambda r:r["score"],reverse=True)
    best=sweep[0]
    print(f"\n  Best: RSI {best['rsi_buy']}/{best['rsi_sell']} score={best['min_score']} "
          f"→ {best['stats']['win_rate']:.0f}% WR | PF={best['stats']['profit_factor']} | "
          f"E=${best['stats']['expectancy']:.2f}/trade")

    out={
        "generated_at":datetime.datetime.now(MARKET_TZ).isoformat(),
        "generated_str":datetime.datetime.now(MARKET_TZ).strftime("%B %d, %Y at %I:%M %p ET"),
        "config":{
            "tickers":TICKERS,"lookback_days":LOOKBACK_DAYS,
            "strategy":"VWAP Reclaim + EMA Pullback + Opening Drive + Momentum v16",
            "features":["partial_exits","breakeven_stop","daily_loss_limit",
                       "consec_loss_pause","vix_sizing","slippage","prime_window_bonus",
                       "volume_rising_filter","time_based_trail"],
            "baseline":{
                "rsi_bull_min":RSI_BULL_MIN,"rsi_bull_max":RSI_BULL_MAX,
                "rsi_bear_min":RSI_BEAR_MIN,"rsi_bear_max":RSI_BEAR_MAX,
                "min_score":MIN_SCORE,"gain_pct":GAIN_TARGET_PCT*100,
                "stop_pct":STOP_LOSS_PCT*100,"slippage_pct":SLIPPAGE_PCT*100,
                "partial_exit_pct":PARTIAL_EXIT_PCT*100,
            }
        },
        "baseline":{"stats":bs,"trades":base_trades},
        "out_of_sample":{"train":{"stats":ts_train},"test":{"stats":ts_test},"consistent":oos_consistency},
        "sweep":sweep[:60],
    }
    with open(OUT,"w") as f:
        json.dump(out,f,indent=2)
    print(f"\n✅ Results saved → {OUT}")
    print("="*60)

if __name__=="__main__":
    main()
