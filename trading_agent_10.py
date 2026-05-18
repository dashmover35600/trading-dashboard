"""
ORB + RSI + Volume Trading Agent v8
New in v8:
  - Multi-timeframe confirmation (1min + 5min must agree before signal fires)
  - Strategy comparison tracker (logs both original and new strategy side by side)
  - Live open trade P&L endpoint for NYLO dashboard real-time updates

Requirements:
    pip install yfinance pandas ta schedule pytz

Morning routine:
    export TRADING_PHONE="+13214378412"
    cd ~/Downloads
    python3 trading_agent_8.py
"""

import yfinance as yf
import pandas as pd
import ta
import schedule
import time
import subprocess
import datetime
import pytz
import csv
import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS           = ["QQQ", "GLD"]
IMESSAGE_TO       = os.environ.get("TRADING_PHONE", "+1XXXXXXXXXX")
MARKET_TZ         = pytz.timezone("America/New_York")
MARKET_OPEN       = datetime.time(9, 30)
RANGE_END         = datetime.time(9, 45)
CUTOFF            = datetime.time(13, 0)

RSI_BUY_MIN       = 55
RSI_SELL_MAX      = 45
GAIN_TARGET       = 0.015
STOP_LOSS         = 0.0075
MAX_TRADES        = 2
VOLUME_MULTIPLIER = 1.5

DEAD_ZONE_START   = datetime.time(11, 30)
DEAD_ZONE_END     = datetime.time(12, 30)

SERVER_PORT       = 8765

# Strategy B (comparison) — slightly different RSI thresholds to compare
STRATEGY_B_RSI_BUY  = 60   # more strict
STRATEGY_B_RSI_SELL = 40   # more strict

POSITION_SIZE     = 500.00  # dollars per trade for paper trading

# ── Subscribers ───────────────────────────────────────────────────────────────
# Auto-managed by dashboard — approve friends there and numbers sync automatically
SUBSCRIBERS = []

# Load saved subscribers from file if it exists
_sub_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subscribers.py")
if os.path.exists(_sub_file):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("subscribers", _sub_file)
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        SUBSCRIBERS.extend(_mod.SUBSCRIBERS)
        print(f"[Subscribers] Loaded {len(SUBSCRIBERS)} from file")
    except Exception as _e:
        print(f"[Subscribers] Load error: {_e}")

# ── File paths ────────────────────────────────────────────────────────────────

BASE         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE     = os.path.join(BASE, "trade_log.csv")
COMPARE_FILE = os.path.join(BASE, "strategy_comparison.csv")
LIVE_FILE    = os.path.join(BASE, "live_trade.json")

# ── Trade log ─────────────────────────────────────────────────────────────────

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date", "Ticker", "Direction",
                "Entry Price", "Exit Price",
                "Target", "Stop Loss",
                "Result", "P&L %", "P&L $", "Shares",
                "Entry Time", "Exit Time",
                "RSI at Entry", "Volume Ratio",
                "5min RSI", "5min Confirmed"
            ])
        print(f"[Tracker] Trade log created → {LOG_FILE}")

    if not os.path.exists(COMPARE_FILE):
        with open(COMPARE_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date", "Ticker", "Direction", "Entry Price",
                "Strategy A Signal", "Strategy B Signal",
                "Strategy A Result", "Strategy A P&L",
                "Strategy B Result", "Strategy B P&L",
                "Winner"
            ])
        print(f"[Tracker] Strategy comparison log created → {COMPARE_FILE}")

def calc_shares(entry_price):
    return round(POSITION_SIZE / entry_price, 4)

def calc_dollar_pnl(entry_price, exit_price, direction, shares):
    if direction == "long":
        return round((exit_price - entry_price) * shares, 2)
    else:
        return round((entry_price - exit_price) * shares, 2)

def log_entry(ticker, direction, entry_price, target, stop, rsi, vol_ratio,
              entry_time, rsi_5m, confirmed_5m):
    shares = calc_shares(entry_price)
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            entry_time.strftime("%Y-%m-%d"), ticker,
            "Long" if direction == "long" else "Short",
            entry_price, "", target, stop, "", "", "", shares,
            entry_time.strftime("%I:%M %p"), "", rsi, f"{vol_ratio}x",
            rsi_5m, "Yes" if confirmed_5m else "No"
        ])

def log_exit(ticker, exit_price, result, pnl, exit_time):
    with open(LOG_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))
    for i in range(len(rows) - 1, 0, -1):
        if rows[i][1] == ticker and rows[i][4] == "":
            entry_price  = float(rows[i][3])
            direction    = "long" if rows[i][2] == "Long" else "short"
            shares       = float(rows[i][10]) if rows[i][10] else calc_shares(entry_price)
            dollar_pnl   = calc_dollar_pnl(entry_price, float(exit_price), direction, shares)
            rows[i][4]   = exit_price
            rows[i][7]   = result
            rows[i][8]   = f"{'+' if pnl > 0 else ''}{pnl}%"
            rows[i][9]   = f"+${dollar_pnl:.2f}" if dollar_pnl >= 0 else f"-${abs(dollar_pnl):.2f}"
            rows[i][12]  = exit_time.strftime("%I:%M %p")
            break
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)

def log_comparison(ticker, direction, entry_price, entry_time,
                   a_signal, b_signal):
    """Log a new comparison row when a signal fires on either strategy."""
    with open(COMPARE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            entry_time.strftime("%Y-%m-%d"), ticker, direction, entry_price,
            "Yes" if a_signal else "No",
            "Yes" if b_signal else "No",
            "", "", "", "", ""   # results filled later
        ])

def read_trades_for_dates(date_list):
    trades = []
    if not os.path.exists(LOG_FILE):
        return trades
    with open(LOG_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["Date"] in date_list:
                trades.append(row)
    return trades

# ── Live trade JSON ───────────────────────────────────────────────────────────

def write_live_trade(ticker=None, direction=None, entry=None,
                     target=None, stop=None, rsi=None, entry_time=None):
    """Write current open trade to a JSON file for the dashboard."""
    data = {
        "active": ticker is not None,
        "ticker": ticker,
        "direction": direction,
        "entry": entry,
        "target": target,
        "stop": stop,
        "rsi": rsi,
        "entry_time": entry_time.strftime("%I:%M %p ET") if entry_time else None,
        "updated": datetime.datetime.now(MARKET_TZ).strftime("%I:%M:%S %p ET")
    }
    with open(LIVE_FILE, "w") as f:
        json.dump(data, f)

def clear_live_trade():
    write_live_trade()

# ── Local web server ──────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/trade_log.csv":
            self._serve_file(LOG_FILE, "text/csv")
        elif self.path == "/strategy_comparison.csv":
            self._serve_file(COMPARE_FILE, "text/csv")
        elif self.path.startswith("/live_trade"):
            self._serve_file(LIVE_FILE, "application/json")
        elif self.path == "/status":
            self._serve_status()
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/update_subscribers":
            self._update_subscribers()
        else:
            self.send_response(404); self.end_headers()

    def _update_subscribers(self):
        """Receive new subscribers list from dashboard and update agent file."""
        try:
            length  = int(self.headers.get('Content-Length', 0))
            content = self.rfile.read(length).decode('utf-8')
            # Write to subscribers file next to agent
            sub_file = os.path.join(BASE, "subscribers.py")
            with open(sub_file, "w") as f:
                f.write("# Auto-generated by NYLO dashboard — do not edit manually\n")
                f.write(content)
            # Reload subscribers into memory
            import importlib.util
            spec = importlib.util.spec_from_file_location("subscribers", sub_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            SUBSCRIBERS.clear()
            SUBSCRIBERS.extend(mod.SUBSCRIBERS)
            print(f"[Server] Subscribers updated → {len(SUBSCRIBERS)} active")
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"ok")
        except Exception as e:
            print(f"[Server] Subscriber update error: {e}")
            self.send_response(500); self.end_headers()

    def _serve_file(self, path, content_type):
        if not os.path.exists(path):
            self.send_response(204); self.end_headers(); return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _serve_status(self):
        status = {
            "running": True,
            "time_et": datetime.datetime.now(MARKET_TZ).strftime("%I:%M %p ET"),
            "trades_today": daily_trade_count["count"],
            "max_trades": MAX_TRADES,
            "tickers": TICKERS,
        }
        data = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass

def start_server():
    server = HTTPServer(("localhost", SERVER_PORT), DashboardHandler)
    print(f"[Server] Dashboard server → http://localhost:{SERVER_PORT}")
    server.serve_forever()

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    ticker: {
        "range_high":  None,
        "range_low":   None,
        "range_set":   False,
        "in_trade":    False,
        "trade_dir":   None,
        "entry_price": None,
        "entry_time":  None,
        "target":      None,
        "stop":        None,
        "rsi_entry":   None,
        "rsi_5m":      None,
    }
    for ticker in TICKERS
}

daily_trade_count = {"count": 0}

# ── iMessage ──────────────────────────────────────────────────────────────────

def send_imessage(message: str, broadcast: bool = False):
    """Send iMessage to owner. If broadcast=True also sends to all subscribers."""
    recipients = [IMESSAGE_TO]
    if broadcast and SUBSCRIBERS:
        recipients += [phone for _, phone in SUBSCRIBERS]

    for recipient in recipients:
        safe = message.replace('"', '\\"').replace("'", "\\'")
        script = f'''
        tell application "Messages"
            set targetService to 1st service whose service type = iMessage
            set targetBuddy to buddy "{recipient}" of targetService
            send "{safe}" to targetBuddy
        end tell
        '''
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            print(f"[iMessage → {recipient}] sent")
        except subprocess.CalledProcessError as e:
            print(f"[iMessage error → {recipient}] {e}")

# ── Market helpers ────────────────────────────────────────────────────────────

def now_et():
    return datetime.datetime.now(MARKET_TZ)

def is_market_hours():
    t = now_et().time()
    return MARKET_OPEN <= t <= CUTOFF

def is_range_window():
    t = now_et().time()
    return MARKET_OPEN <= t <= RANGE_END

def is_dead_zone():
    t = now_et().time()
    return DEAD_ZONE_START <= t <= DEAD_ZONE_END

def fetch_data(ticker, period="5d", interval="1m"):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    df.index = df.index.tz_convert(MARKET_TZ)
    return df

def fetch_data_5m(ticker):
    """Fetch 5-minute candles for multi-timeframe check."""
    df = yf.download(ticker, period="5d", interval="5m",
                     progress=False, auto_adjust=True)
    df.index = df.index.tz_convert(MARKET_TZ)
    return df

def calc_rsi(df, window=14):
    if len(df) < window + 1:
        return 50.0
    rsi_series = ta.momentum.RSIIndicator(df["Close"].squeeze(), window=window).rsi()
    return round(float(rsi_series.iloc[-1]), 2)

def calc_volume_ratio(df):
    if len(df) < 21:
        return 1.0
    vol_series = df["Volume"].squeeze()
    avg_vol    = float(vol_series.iloc[-21:-1].mean())
    cur_vol    = float(vol_series.iloc[-1])
    if avg_vol == 0:
        return 1.0
    return round(cur_vol / avg_vol, 2)

def calc_vwap(df):
    """
    Calculate VWAP (Volume Weighted Average Price) for today only.
    VWAP = cumulative(price * volume) / cumulative(volume)
    Uses today's 1-minute candles only — resets each day.
    """
    try:
        today    = datetime.datetime.now(MARKET_TZ).date()
        today_df = df[df.index.date == today].copy()
        if today_df.empty or len(today_df) < 2:
            return None
        close  = today_df["Close"].squeeze()
        high   = today_df["High"].squeeze()
        low    = today_df["Low"].squeeze()
        volume = today_df["Volume"].squeeze()
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return round(float(vwap.iloc[-1]), 4)
    except Exception as e:
        print(f"[VWAP error] {e}")
        return None

# ── Multi-timeframe confirmation ──────────────────────────────────────────────

def check_5m_confirmation(ticker, direction):
    """
    Fetch the 5-minute chart and check if RSI agrees with the 1-minute signal.
    BUY signal: 5min RSI must also be above 55
    SELL signal: 5min RSI must also be below 45
    Returns (confirmed: bool, rsi_5m: float)
    """
    try:
        df5 = fetch_data_5m(ticker)
        if df5.empty or len(df5) < 15:
            return True, 50.0   # not enough data — allow trade
        rsi_5m = calc_rsi(df5)
        if direction == "long":
            confirmed = rsi_5m > RSI_BUY_MIN
        else:
            confirmed = rsi_5m < RSI_SELL_MAX
        return confirmed, rsi_5m
    except Exception as e:
        print(f"[5m check error] {e}")
        return True, 50.0  # if fetch fails, don't block the trade

# ── Core logic ────────────────────────────────────────────────────────────────

def set_opening_range(ticker, df):
    s = state[ticker]
    if s["range_set"] or not is_range_window():
        return
    today   = now_et().date()
    morning = df[df.index.date == today]
    window  = morning.between_time("09:30", "09:45")
    if window.empty:
        return
    s["range_high"] = round(float(window["High"].max()), 4)
    s["range_low"]  = round(float(window["Low"].min()),  4)
    if now_et().time() >= RANGE_END:
        s["range_set"] = True
        print(f"[{ticker}] Range set → High: {s['range_high']}  Low: {s['range_low']}")

def check_signal(ticker, df):
    s = state[ticker]
    if not s["range_set"] or s["in_trade"]:
        return
    if daily_trade_count["count"] >= MAX_TRADES:
        return
    if now_et().time() > CUTOFF:
        return
    if is_dead_zone():
        print(f"[{ticker}] Dead zone — skipping")
        return

    price     = round(float(df["Close"].iloc[-1]), 4)
    rsi_1m    = calc_rsi(df)
    vol_ratio = calc_volume_ratio(df)
    vol_ok    = vol_ratio >= VOLUME_MULTIPLIER
    vwap      = calc_vwap(df)

    print(f"[{ticker}] Price: {price}  VWAP: {vwap}  1m RSI: {rsi_1m}  Vol: {vol_ratio}x  "
          f"High: {s['range_high']}  Low: {s['range_low']}")

    if not vol_ok:
        print(f"[{ticker}] Volume too low — skipping")
        return

    direction = None
    if price > s["range_high"] and rsi_1m > RSI_BUY_MIN:
        direction = "long"
    elif price < s["range_low"] and rsi_1m < RSI_SELL_MAX:
        direction = "short"

    if not direction:
        return

    # ── VWAP confirmation ──
    vwap_ok = True
    if vwap is not None:
        if direction == "long" and price < vwap:
            vwap_ok = False
        elif direction == "short" and price > vwap:
            vwap_ok = False

    if not vwap_ok:
        print(f"[{ticker}] VWAP check failed — price not on correct side of VWAP ({vwap})")
        send_imessage(
            f"⚠️ SIGNAL SKIPPED — {ticker}\n"
            f"Breakout detected but price is on wrong side of VWAP\n"
            f"Price: ${price}  VWAP: ${vwap}\n"
            f"Waiting for better setup."
        )
        return

    # ── Multi-timeframe confirmation ──
    confirmed_5m, rsi_5m = check_5m_confirmation(ticker, direction)
    b_signal = (direction == "long" and rsi_1m > STRATEGY_B_RSI_BUY) or \
               (direction == "short" and rsi_1m < STRATEGY_B_RSI_SELL)

    print(f"[{ticker}] 5m RSI: {rsi_5m}  5m confirmed: {'✅' if confirmed_5m else '❌'}  VWAP: ✅")

    # Log comparison regardless of whether we trade
    log_comparison(ticker, direction, price, now_et(), True, b_signal)

    if not confirmed_5m:
        send_imessage(
            f"⚠️ SIGNAL SKIPPED — {ticker}\n"
            f"1min RSI: {rsi_1m} ✅  but 5min RSI: {rsi_5m} ❌\n"
            f"Both timeframes must agree — waiting for better setup."
        )
        return

    _enter_trade(ticker, direction, price, rsi_1m, vol_ratio, rsi_5m, confirmed_5m)

def _enter_trade(ticker, direction, price, rsi, vol_ratio, rsi_5m, confirmed_5m):
    s   = state[ticker]
    now = now_et()

    target = round(price * (1 + GAIN_TARGET), 4) if direction == "long" else round(price * (1 - GAIN_TARGET), 4)
    stop   = round(price * (1 - STOP_LOSS),   4) if direction == "long" else round(price * (1 + STOP_LOSS),   4)

    s.update({
        "in_trade": True, "trade_dir": direction, "entry_price": price,
        "entry_time": now, "target": target, "stop": stop,
        "rsi_entry": rsi, "rsi_5m": rsi_5m
    })
    daily_trade_count["count"] += 1

    log_entry(ticker, direction, price, target, stop, rsi, vol_ratio,
              now, rsi_5m, confirmed_5m)

    # Write live trade for dashboard
    write_live_trade(ticker, direction, price, target, stop, rsi, now)

    emoji = "🟢" if direction == "long" else "🔴"
    send_imessage(
        f"{emoji} TRADE SIGNAL — {ticker}\n"
        f"Direction : {'BUY (Long)' if direction == 'long' else 'SELL (Short)'}\n"
        f"Entry     : ${price}\n"
        f"Target    : ${target}  (+{GAIN_TARGET*100:.1f}%)\n"
        f"Stop Loss : ${stop}  (-{STOP_LOSS*100:.2f}%)\n"
        f"1min RSI  : {rsi}  ✅\n"
        f"5min RSI  : {rsi_5m}  ✅\n"
        f"VWAP      : ${calc_vwap(fetch_data(ticker))}  ✅\n"
        f"Volume    : {vol_ratio}x avg  ✅\n"
        f"Time      : {now.strftime('%I:%M %p ET')}\n"
        f"Trades today: {daily_trade_count['count']}/{MAX_TRADES}\n"
        f"— Signal by NYLO Trading",
        broadcast=True
    )

def check_exit(ticker, df):
    s = state[ticker]
    if not s["in_trade"]:
        return

    price     = round(float(df["Close"].iloc[-1]), 4)
    entry     = s["entry_price"]
    direction = s["trade_dir"]
    now       = now_et()

    hit_target = (direction == "long"  and price >= entry * (1 + GAIN_TARGET)) or \
                 (direction == "short" and price <= entry * (1 - GAIN_TARGET))
    hit_stop   = (direction == "long"  and price <= entry * (1 - STOP_LOSS)) or \
                 (direction == "short" and price >= entry * (1 + STOP_LOSS))

    if hit_target or hit_stop:
        pnl    = round((price - entry) / entry * 100, 2) if direction == "long" \
                 else round((entry - price) / entry * 100, 2)
        result = "Target Hit" if hit_target else "Stop Loss Hit"
        emoji  = "✅" if hit_target else "🛑"

        log_exit(ticker, price, result, pnl, now)
        clear_live_trade()

        shares     = calc_shares(entry)
        dollar_pnl = calc_dollar_pnl(entry, price, direction, shares)
        dollar_str = f"+${dollar_pnl:.2f}" if dollar_pnl >= 0 else f"-${abs(dollar_pnl):.2f}"

        send_imessage(
            f"{emoji} {result.upper()} — {ticker}\n"
            f"Entry : ${entry}  →  Exit: ${price}\n"
            f"P&L   : {'+' if pnl > 0 else ''}{pnl}%  ({dollar_str})\n"
            f"Shares: {shares}  ·  Position: ${POSITION_SIZE:.0f}\n"
            f"Time  : {now.strftime('%I:%M %p ET')}\n"
            f"— Signal by NYLO Trading",
            broadcast=True
        )
        s.update({
            "in_trade": False, "trade_dir": None, "entry_price": None,
            "entry_time": None, "target": None, "stop": None,
            "rsi_entry": None, "rsi_5m": None
        })

# ── Daily summary ─────────────────────────────────────────────────────────────

def send_daily_summary():
    today  = now_et().strftime("%Y-%m-%d")
    trades = read_trades_for_dates([today])
    if not trades:
        send_imessage("📋 End of day — no signals fired today.")
        return
    wins      = sum(1 for t in trades if t["Result"] == "Target Hit")
    losses    = sum(1 for t in trades if t["Result"] == "Stop Loss Hit")
    confirmed = sum(1 for t in trades if t.get("5min Confirmed") == "Yes")
    total_pnl = sum(float(t["P&L %"].replace("%","").replace("+","")) for t in trades if t["P&L %"])
    total_dollar = sum(
        float(t.get("P&L $","0").replace("$","").replace("+","").replace("-","") or 0) *
        (1 if not t.get("P&L $","").startswith("-") else -1)
        for t in trades if t.get("P&L $")
    )
    lines = [
        f"📋 Daily Summary — {today}",
        f"Trades : {len(trades)}  |  ✅ {wins} wins  🛑 {losses} losses",
        f"5min confirmed: {confirmed}/{len(trades)} trades",
        f"Total P&L : {'+' if total_pnl >= 0 else ''}{round(total_pnl, 2)}%  "
        f"({'+' if total_dollar >= 0 else ''} ${abs(total_dollar):.2f})", ""
    ]
    for t in trades:
        dollar = f"  ({t.get('P&L $','')})" if t.get("P&L $") else ""
        lines.append(f"{t['Ticker']} {t['Direction']} → {t['Result']} {t['P&L %']}{dollar}")
    send_imessage("\n".join(lines))

# ── Weekly summary ────────────────────────────────────────────────────────────

def send_weekly_summary():
    today      = now_et().date()
    monday     = today - datetime.timedelta(days=today.weekday())
    week_dates = [(monday + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    trades     = read_trades_for_dates(week_dates)
    if not trades:
        send_imessage("📊 Weekly Summary — no trades this week.")
        return
    wins      = sum(1 for t in trades if t["Result"] == "Target Hit")
    losses    = sum(1 for t in trades if t["Result"] == "Stop Loss Hit")
    pending   = sum(1 for t in trades if t["Result"] == "")
    total_pnl = sum(float(t["P&L %"].replace("%","").replace("+","")) for t in trades if t["P&L %"])
    win_rate  = round(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    closed    = [t for t in trades if t["P&L %"]]
    best      = max(closed, key=lambda t: float(t["P&L %"].replace("%","").replace("+","")), default=None)
    worst     = min(closed, key=lambda t: float(t["P&L %"].replace("%","").replace("+","")), default=None)
    lines = [
        f"📊 Weekly Summary — w/e {today.strftime('%b %d')}",
        f"Total trades : {len(trades)}  ({pending} still open)",
        f"✅ Wins: {wins}  🛑 Losses: {losses}  Win rate: {win_rate}%",
        f"Total P&L : {'+' if total_pnl >= 0 else ''}{round(total_pnl, 2)}%",
    ]
    if best:  lines.append(f"Best  : {best['Ticker']} {best['Direction']} {best['P&L %']}")
    if worst: lines.append(f"Worst : {worst['Ticker']} {worst['Direction']} {worst['P&L %']}")
    send_imessage("\n".join(lines))

# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_daily():
    daily_trade_count["count"] = 0
    for ticker in TICKERS:
        state[ticker].update({
            "range_high": None, "range_low": None, "range_set": False,
            "in_trade": False, "trade_dir": None, "entry_price": None,
            "entry_time": None, "target": None, "stop": None,
            "rsi_entry": None, "rsi_5m": None,
        })
    clear_live_trade()
    print("[Agent] Daily state reset.")

# ── Pre-market scanner ────────────────────────────────────────────────────────

def pre_market_scan():
    """
    Runs at 9:00 AM ET every weekday.
    Fetches pre-market data for each ticker and texts a morning briefing
    showing price, overnight change, volume and RSI momentum.
    """
    print("[Pre-market] Running morning scan...")
    lines = ["🌅 NYLO Morning Scan — " + now_et().strftime("%A %b %d"), ""]

    for ticker in TICKERS:
        try:
            # Fetch 5-day 1-minute data to get pre-market and yesterday's close
            df = yf.download(ticker, period="5d", interval="1m",
                             progress=False, auto_adjust=True)
            df.index = df.index.tz_convert(MARKET_TZ)

            # Yesterday's closing price (last candle before today)
            today     = now_et().date()
            yesterday = df[df.index.date < today]
            if yesterday.empty:
                continue
            prev_close = round(float(yesterday["Close"].iloc[-1]), 2)

            # Latest pre-market price
            premarket = df[df.index.date == today]
            if premarket.empty:
                lines.append(f"{ticker} — no pre-market data yet")
                continue
            cur_price  = round(float(premarket["Close"].iloc[-1]), 2)
            change_pct = round((cur_price - prev_close) / prev_close * 100, 2)
            change_dir = "📈" if change_pct >= 0 else "📉"

            # Pre-market volume vs average
            avg_vol    = float(df[df.index.date < today]["Volume"].mean())
            pre_vol    = float(premarket["Volume"].sum())
            vol_ratio  = round(pre_vol / avg_vol * 100, 1) if avg_vol > 0 else 0

            # RSI from recent candles
            rsi = calc_rsi(premarket) if len(premarket) >= 15 else 50.0

            # Bias
            if change_pct > 0.3 and rsi > 55:
                bias = "🟢 Bullish bias"
            elif change_pct < -0.3 and rsi < 45:
                bias = "🔴 Bearish bias"
            else:
                bias = "⚪ Neutral"

            lines.append(
                f"{change_dir} {ticker}\n"
                f"  Price    : ${cur_price}  ({'+' if change_pct >= 0 else ''}{change_pct}% vs yesterday)\n"
                f"  RSI      : {rsi}\n"
                f"  Pre-vol  : {vol_ratio}% of avg\n"
                f"  Bias     : {bias}"
            )
        except Exception as e:
            lines.append(f"{ticker} — scan error: {e}")

    lines.append("")
    lines.append("⏰ Market opens in 30 minutes. Range sets 9:45 AM.")
    send_imessage("\n".join(lines))

# ── Main loop ─────────────────────────────────────────────────────────────────

def run_agent():
    if not is_market_hours():
        return
    for ticker in TICKERS:
        try:
            df = fetch_data(ticker)
            if df.empty:
                continue
            set_opening_range(ticker, df)
            check_signal(ticker, df)
            check_exit(ticker, df)
        except Exception as e:
            print(f"[{ticker}] Error: {e}")

# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_log()
    clear_live_trade()

    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    print("=" * 54)
    print("  ORB + RSI + VWAP Trading Agent v10")
    print("  Watching   : QQQ, GLD")
    print(f"  Alerts     → {IMESSAGE_TO}")
    print(f"  Subscribers: {len(SUBSCRIBERS)} friends following signals")
    print(f"  Log        → {LOG_FILE}")
    print(f"  Volume     → {VOLUME_MULTIPLIER}x avg minimum")
    print(f"  Dead zone  → 11:30 AM – 12:30 PM")
    print(f"  Multi-TF   → 1min + 5min must agree ✅")
    print(f"  VWAP       → price must be on correct side ✅")
    print(f"  Pre-market → scan at 9:00 AM ✅")
    print(f"  Dashboard  → open trading_dashboard.html")
    print("=" * 54)

    send_imessage(
        "🤖 NYLO Trading Agent v10 started!\n"
        "✅ Pre-market scanner at 9:00 AM\n"
        "✅ VWAP + Multi-TF + Volume + RSI + ORB\n"
        "🔵 All systems go — good luck today! 🚀"
    )

    schedule.every(1).minutes.do(run_agent)

    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), day).at("09:00").do(pre_market_scan)
        getattr(schedule.every(), day).at("09:25").do(reset_daily)
        getattr(schedule.every(), day).at("13:05").do(send_daily_summary)

    schedule.every().friday.at("13:05").do(send_weekly_summary)

    while True:
        schedule.run_pending()
        time.sleep(30)
