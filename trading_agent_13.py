"""
ORB + RSI + VWAP + Volume + Gap Direction Filter Trading Agent v13
=============================================================
NYLO Phase 5 — Strategy Improvements

What's new in v12 (strategy overhaul based on backtest results):
  ✅ Gain target     — reduced 1.5% → 1.0% (more achievable in choppy markets)
  ✅ Stop loss       — reduced 0.75% → 0.5% (same 2:1 ratio, tighter exits)
  ✅ Volume filter   — loosened 1.5x → 1.2x (was blocking too many valid signals)
  ✅ 5min MTF check  — REMOVED (was killing signals, too restrictive)
  ✅ SPY trend filter — NEW: only longs when SPY > VWAP, only shorts when SPY < VWAP
  ✅ Max trades      — increased 2 → 3 per day (more opportunities)
  ✅ All v11 reliability features retained

Requirements (unchanged):
  pip install yfinance pandas ta schedule pytz

Morning routine (unchanged):
  export TRADING_PHONE="+1XXXXXXXXXX"
  python3 -W ignore trading_agent_12.py
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
import signal
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS           = ["QQQ"]  # v13: GLD dropped — underperforming in backtest
IMESSAGE_TO       = os.environ.get("TRADING_PHONE", "+1XXXXXXXXXX")
MARKET_TZ         = pytz.timezone("America/New_York")
MARKET_OPEN       = datetime.time(9, 30)
RANGE_END         = datetime.time(9, 45)
CUTOFF            = datetime.time(13, 0)
RSI_BUY_MIN       = 55  # unchanged — backtest showed RSI threshold isn't the issue
RSI_SELL_MAX      = 45  # unchanged
GAIN_TARGET       = 0.020  # v13: 2.0% target — gives trades room to run
STOP_LOSS         = 0.010   # v13: 1.0% stop — 0.5% was noise level for QQQ
MAX_TRADES        = 999     # effectively unlimited — take every valid signal
VOLUME_MULTIPLIER = 1.2     # loosened from 1.5x → 1.2x (was too restrictive)
DEAD_ZONE_START   = datetime.time(11, 30)
DEAD_ZONE_END     = datetime.time(12, 30)
SERVER_PORT       = 8765
AGENT_VERSION     = "13"

# Strategy B comparison thresholds
STRATEGY_B_RSI_BUY  = 60
STRATEGY_B_RSI_SELL = 40

POSITION_SIZE = 500.00  # dollars per trade

# Reliability settings
FETCH_RETRIES    = 3     # how many times to retry yfinance on failure
FETCH_BACKOFF    = 2.0   # seconds, doubles on each retry
HEARTBEAT_INTERVAL = 60  # seconds between heartbeat writes
CRASH_ALERT_COOLDOWN = 300  # seconds — don't spam crash alerts

# ── File paths ────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE     = os.path.join(BASE, "trade_log.csv")
COMPARE_FILE = os.path.join(BASE, "strategy_comparison.csv")
LIVE_FILE    = os.path.join(BASE, "live_trade.json")
HEALTH_FILE  = os.path.join(BASE, "heartbeat.json")
ERROR_LOG    = os.path.join(BASE, "agent_errors.log")

# ── Agent health state ────────────────────────────────────────────────────────
_health = {
    "start_time":      datetime.datetime.now(pytz.timezone("America/New_York")),
    "last_run":        None,
    "last_run_ok":     True,
    "errors_today":    0,
    "last_error":      None,
    "cycles_total":    0,
    "last_crash_alert": None,   # for cooldown
}

# ── Subscribers ───────────────────────────────────────────────────────────────
SUBSCRIBERS = []
_sub_file = os.path.join(BASE, "subscribers.py")
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

# ── Error logging ─────────────────────────────────────────────────────────────
def log_error(context: str, exc: Exception = None):
    """Append a timestamped error entry to agent_errors.log."""
    now_str = datetime.datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
    lines = [f"[{now_str}] {context}"]
    if exc:
        lines.append("  " + traceback.format_exc().strip().replace("\n", "\n  "))
    entry = "\n".join(lines) + "\n"
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(entry)
    except Exception:
        pass
    print(f"[ERROR] {context}" + (f": {exc}" if exc else ""))
    _health["errors_today"] += 1
    _health["last_error"]   = context

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


def calc_shares(entry_price):
    return round(POSITION_SIZE / entry_price, 4)


def calc_dollar_pnl(entry_price, exit_price, direction, shares):
    if direction == "long":
        return round((exit_price - entry_price) * shares, 2)
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
            entry_price = float(rows[i][3])
            direction   = "long" if rows[i][2] == "Long" else "short"
            shares      = float(rows[i][10]) if rows[i][10] else calc_shares(entry_price)
            dollar_pnl  = calc_dollar_pnl(entry_price, float(exit_price), direction, shares)
            rows[i][4]  = exit_price
            rows[i][7]  = result
            rows[i][8]  = f"{'+' if pnl > 0 else ''}{pnl}%"
            rows[i][9]  = f"+${dollar_pnl:.2f}" if dollar_pnl >= 0 else f"-${abs(dollar_pnl):.2f}"
            rows[i][12] = exit_time.strftime("%I:%M %p")
            break
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def log_comparison(ticker, direction, entry_price, entry_time, a_signal, b_signal):
    with open(COMPARE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            entry_time.strftime("%Y-%m-%d"), ticker, direction, entry_price,
            "Yes" if a_signal else "No",
            "Yes" if b_signal else "No",
            "", "", "", "", ""
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
    data = {
        "active":     ticker is not None,
        "ticker":     ticker,
        "direction":  direction,
        "entry":      entry,
        "target":     target,
        "stop":       stop,
        "rsi":        rsi,
        "entry_time": entry_time.strftime("%I:%M %p ET") if entry_time else None,
        "updated":    datetime.datetime.now(MARKET_TZ).strftime("%I:%M:%S %p ET")
    }
    with open(LIVE_FILE, "w") as f:
        json.dump(data, f)


def clear_live_trade():
    write_live_trade()

# ── Heartbeat ─────────────────────────────────────────────────────────────────
def write_heartbeat(status: str = "running"):
    """
    Write heartbeat.json every cycle.
    The dashboard and any external watchdog can check this file.
    If updated_at is >5 min old, the agent is likely hung or dead.
    """
    now = datetime.datetime.now(MARKET_TZ)
    uptime_s = int((now - _health["start_time"]).total_seconds())
    data = {
        "status":        status,                        # "running" | "stopped" | "error"
        "version":       AGENT_VERSION,
        "updated_at":    now.isoformat(),
        "updated_str":   now.strftime("%I:%M:%S %p ET"),
        "uptime_sec":    uptime_s,
        "uptime_str":    _fmt_uptime(uptime_s),
        "last_run":      _health["last_run"],
        "last_run_ok":   _health["last_run_ok"],
        "errors_today":  _health["errors_today"],
        "last_error":    _health["last_error"],
        "cycles_total":  _health["cycles_total"],
        "tickers":       TICKERS,
        "trades_today":  daily_trade_count.get("count", 0),
        "max_trades":    "unlimited",
    }
    try:
        with open(HEALTH_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_error("heartbeat write failed", e)


def _fmt_uptime(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"

# ── Local web server ──────────────────────────────────────────────────────────
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/trade_log.csv":
            self._serve_file(LOG_FILE, "text/csv")
        elif p == "/strategy_comparison.csv":
            self._serve_file(COMPARE_FILE, "text/csv")
        elif p in ("/live_trade", "/live_trade.json"):
            self._serve_file(LIVE_FILE, "application/json")
        elif p in ("/health", "/heartbeat", "/heartbeat.json"):
            self._serve_file(HEALTH_FILE, "application/json")
        elif p == "/agent_errors.log":
            self._serve_file(ERROR_LOG, "text/plain")
        elif p == "/status":
            self._serve_status()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/update_subscribers":
            self._update_subscribers()
        else:
            self.send_response(404)
            self.end_headers()

    def _update_subscribers(self):
        try:
            length  = int(self.headers.get("Content-Length", 0))
            content = self.rfile.read(length).decode("utf-8")
            sub_file = os.path.join(BASE, "subscribers.py")
            with open(sub_file, "w") as f:
                f.write("# Auto-generated by NYLO dashboard — do not edit manually\n")
                f.write(content)
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
            log_error("subscriber update", e)
            self.send_response(500)
            self.end_headers()

    def _serve_file(self, path, content_type):
        if not os.path.exists(path):
            self.send_response(204)
            self.end_headers()
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _serve_status(self):
        """
        /status — quick JSON for the dashboard status widget.
        Includes uptime, last_run, errors, and trade count.
        """
        now = datetime.datetime.now(MARKET_TZ)
        uptime_s = int((now - _health["start_time"]).total_seconds())
        status = {
            "running":       True,
            "version":       AGENT_VERSION,
            "time_et":       now.strftime("%I:%M %p ET"),
            "trades_today":  daily_trade_count.get("count", 0),
            "max_trades":    MAX_TRADES,
            "tickers":       TICKERS,
            "uptime":        _fmt_uptime(uptime_s),
            "errors_today":  _health["errors_today"],
            "last_run":      _health["last_run"],
            "last_run_ok":   _health["last_run_ok"],
        }
        data = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # suppress per-request console spam


def start_server():
    server = HTTPServer(("localhost", SERVER_PORT), DashboardHandler)
    print(f"[Server] Dashboard server → http://localhost:{SERVER_PORT}")
    server.serve_forever()

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    ticker: {
        "range_high": None, "range_low": None, "range_set": False,
        "in_trade":   False, "trade_dir":  None, "entry_price": None,
        "entry_time": None,  "target":     None, "stop":        None,
        "rsi_entry":  None,  "rsi_5m":     None, "allowed_dir": "both",
    }
    for ticker in TICKERS
}
daily_trade_count = {"count": 0}

# ── iMessage ──────────────────────────────────────────────────────────────────
def send_imessage(message: str, broadcast: bool = False):
    recipients = [IMESSAGE_TO]
    if broadcast and SUBSCRIBERS:
        recipients += [phone for _, phone in SUBSCRIBERS]
    for recipient in recipients:
        safe   = message.replace('"', '\\"').replace("'", "\\'")
        script = f'''
tell application "Messages"
  set targetService to 1st service whose service type = iMessage
  set targetBuddy to buddy "{recipient}" of targetService
  send "{safe}" to targetBuddy
end tell
'''
        try:
            subprocess.run(["osascript", "-e", script], check=True, timeout=10)
            print(f"[iMessage → {recipient}] sent")
        except Exception as e:
            log_error(f"iMessage send to {recipient}", e)

# ── Crash alert (with cooldown) ───────────────────────────────────────────────
def maybe_send_crash_alert(context: str, exc: Exception):
    """
    Send an iMessage crash alert, but throttle so we don't spam if errors
    repeat every 60-second cycle.
    """
    now = datetime.datetime.now(MARKET_TZ)
    last = _health["last_crash_alert"]
    if last and (now - last).total_seconds() < CRASH_ALERT_COOLDOWN:
        return  # still in cooldown
    _health["last_crash_alert"] = now
    send_imessage(
        f"⚠️ NYLO AGENT ERROR\n"
        f"Context : {context}\n"
        f"Error   : {type(exc).__name__}: {exc}\n"
        f"Time    : {now.strftime('%I:%M %p ET')}\n"
        f"Errors today: {_health['errors_today']}\n"
        f"Check agent_errors.log for full traceback.\n"
        f"Agent is still running — will retry next cycle."
    )

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

# ── Data fetching with retry ──────────────────────────────────────────────────
def fetch_data(ticker, period="5d", interval="1m"):
    """
    Fetch OHLCV data from yfinance with exponential-backoff retry.
    Returns empty DataFrame only after all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("yfinance returned empty DataFrame")
            df.index = df.index.tz_convert(MARKET_TZ)
            return df
        except Exception as exc:
            last_exc = exc
            wait = FETCH_BACKOFF ** attempt
            print(f"[{ticker}] fetch attempt {attempt}/{FETCH_RETRIES} failed: {exc} — retrying in {wait:.0f}s")
            if attempt < FETCH_RETRIES:
                time.sleep(wait)

    log_error(f"fetch_data({ticker}, {interval}) — all {FETCH_RETRIES} retries failed", last_exc)
    return pd.DataFrame()


def fetch_data_5m(ticker):
    return fetch_data(ticker, period="5d", interval="5m")


def get_spy_trend() -> str:
    """
    Returns 'bullish' if SPY is above its day VWAP, 'bearish' if below.
    Used as a market regime filter — only take longs in bullish regime,
    only shorts in bearish regime.
    Returns 'neutral' on fetch failure (allows all signals through).
    """
    try:
        df = fetch_data("SPY", period="2d", interval="1m")
        if df.empty:
            return "neutral"
        today   = datetime.datetime.now(MARKET_TZ).date()
        day_df  = df[df.index.date == today]
        if len(day_df) < 5:
            return "neutral"
        close  = day_df["Close"].squeeze()
        high   = day_df["High"].squeeze()
        low    = day_df["Low"].squeeze()
        volume = day_df["Volume"].squeeze()
        tp     = (high + low + close) / 3
        vwap   = float((tp * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1])
        price  = float(close.iloc[-1])
        trend  = "bullish" if price > vwap else "bearish"
        print(f"[SPY] Price: {price:.2f} VWAP: {vwap:.2f} → {trend.upper()} market")
        return trend
    except Exception as e:
        log_error("get_spy_trend", e)
        return "neutral"  # fail open — don't block signals

# ── Indicators ────────────────────────────────────────────────────────────────
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
    try:
        today    = datetime.datetime.now(MARKET_TZ).date()
        today_df = df[df.index.date == today].copy()
        if today_df.empty or len(today_df) < 2:
            return None
        close  = today_df["Close"].squeeze()
        high   = today_df["High"].squeeze()
        low    = today_df["Low"].squeeze()
        volume = today_df["Volume"].squeeze()
        tp     = (high + low + close) / 3
        vwap   = (tp * volume).cumsum() / volume.cumsum()
        return round(float(vwap.iloc[-1]), 4)
    except Exception as e:
        log_error("calc_vwap", e)
        return None

# ── Multi-timeframe confirmation ──────────────────────────────────────────────
def check_5m_confirmation(ticker, direction):
    try:
        df5 = fetch_data_5m(ticker)
        if df5.empty or len(df5) < 15:
            return True, 50.0
        rsi_5m    = calc_rsi(df5)
        confirmed = rsi_5m > RSI_BUY_MIN if direction == "long" else rsi_5m < RSI_SELL_MAX
        return confirmed, rsi_5m
    except Exception as e:
        log_error(f"5m confirmation for {ticker}", e)
        return True, 50.0  # don't block on fetch failure

# ── Core strategy ─────────────────────────────────────────────────────────────
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
    s["range_low"]  = round(float(window["Low"].min()), 4)
    if now_et().time() >= RANGE_END:
        s["range_set"] = True
        # Calculate opening gap direction
        yesterday = df[df.index.date < today]
        if not yesterday.empty:
            prev_close = round(float(yesterday["Close"].iloc[-1]), 4)
            first_open = round(float(window["Open"].iloc[0]), 4)
            gap_pct    = (first_open - prev_close) / prev_close * 100
            if gap_pct > 0.2:
                s["allowed_dir"] = "long"
            elif gap_pct < -0.2:
                s["allowed_dir"] = "short"
            else:
                s["allowed_dir"] = "both"
            print(f"[{ticker}] Range set → High: {s['range_high']} Low: {s['range_low']} Gap: {gap_pct:+.2f}% → {s['allowed_dir'].upper()} only")
        else:
            s["allowed_dir"] = "both"
            print(f"[{ticker}] Range set → High: {s['range_high']} Low: {s['range_low']}")


def check_signal(ticker, df):
    s = state[ticker]
    if not s["range_set"] or s["in_trade"]:
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

    print(f"[{ticker}] Price: {price} VWAP: {vwap} 1m RSI: {rsi_1m} Vol: {vol_ratio}x "
          f"High: {s['range_high']} Low: {s['range_low']}")

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

    # Gap direction filter — skip signals against the opening gap
    allowed = s.get("allowed_dir", "both")
    if allowed == "long" and direction == "short":
        print(f"[{ticker}] Skipping SHORT — gap was up, only taking longs today")
        return
    if allowed == "short" and direction == "long":
        print(f"[{ticker}] Skipping LONG — gap was down, only taking shorts today")
        return

    # VWAP filter
    if vwap is not None:
        vwap_ok = (direction == "long" and price >= vwap) or \
                  (direction == "short" and price <= vwap)
        if not vwap_ok:
            print(f"[{ticker}] VWAP check failed — price not on correct side of VWAP ({vwap})")
            send_imessage(
                f"⚠️ SIGNAL SKIPPED — {ticker}\n"
                f"Breakout detected but price is on wrong side of VWAP\n"
                f"Price: ${price} VWAP: ${vwap}\n"
                f"Waiting for better setup."
            )
            return

    # SPY trend filter (replaces 5-min MTF confirmation)
    spy_trend = get_spy_trend()
    spy_ok = (direction == "long"  and spy_trend in ("bullish", "neutral")) or \
             (direction == "short" and spy_trend in ("bearish", "neutral"))

    b_signal = (direction == "long"  and rsi_1m > STRATEGY_B_RSI_BUY) or \
               (direction == "short" and rsi_1m < STRATEGY_B_RSI_SELL)

    print(f"[{ticker}] SPY trend: {spy_trend} {'✅' if spy_ok else '❌'} VWAP: ✅")
    log_comparison(ticker, direction, price, now_et(), True, b_signal)

    if not spy_ok:
        send_imessage(
            f"⚠️ SIGNAL SKIPPED — {ticker}\n"
            f"Breakout detected but SPY trend is {spy_trend} ❌\n"
            f"Only taking {'longs' if direction=='long' else 'shorts'} when SPY is {'bullish' if direction=='long' else 'bearish'}."
        )
        return

    _enter_trade(ticker, direction, price, rsi_1m, vol_ratio, spy_trend)


def _enter_trade(ticker, direction, price, rsi, vol_ratio, spy_trend):
    s   = state[ticker]
    now = now_et()
    target = round(price * (1 + GAIN_TARGET), 4) if direction == "long" \
             else round(price * (1 - GAIN_TARGET), 4)
    stop   = round(price * (1 - STOP_LOSS), 4)   if direction == "long" \
             else round(price * (1 + STOP_LOSS), 4)

    s.update({
        "in_trade": True, "trade_dir": direction, "entry_price": price,
        "entry_time": now, "target": target, "stop": stop,
        "rsi_entry": rsi, "rsi_5m": rsi_5m
    })
    daily_trade_count["count"] += 1
    log_entry(ticker, direction, price, target, stop, rsi, vol_ratio,
              now, spy_trend, True)
    write_live_trade(ticker, direction, price, target, stop, rsi, now)

    emoji = "🟢" if direction == "long" else "🔴"
    send_imessage(
        f"{emoji} TRADE SIGNAL — {ticker}\n"
        f"Direction : {'BUY (Long)' if direction == 'long' else 'SELL (Short)'}\n"
        f"Entry : ${price}\n"
        f"Target : ${target} (+{GAIN_TARGET*100:.1f}%)\n"
        f"Stop Loss : ${stop} (-{STOP_LOSS*100:.2f}%)\n"
        f"1min RSI : {rsi} ✅\n"
        f"SPY Trend : {spy_trend.upper()} ✅\n"
        f"VWAP : ✅\n"
        f"Volume : {vol_ratio}x avg ✅\n"
        f"Time : {now.strftime('%I:%M %p ET')}\n"
        f"Trades today: {daily_trade_count['count']}/{MAX_TRADES}\n"
        f"— Signal by NYLO v{AGENT_VERSION}",
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
            f"Entry : ${entry} → Exit: ${price}\n"
            f"P&L : {'+' if pnl > 0 else ''}{pnl}% ({dollar_str})\n"
            f"Shares: {shares} · Position: ${POSITION_SIZE:.0f}\n"
            f"Time : {now.strftime('%I:%M %p ET')}\n"
            f"— Signal by NYLO v{AGENT_VERSION}",
            broadcast=True
        )
        s.update({
            "in_trade": False, "trade_dir": None, "entry_price": None,
            "entry_time": None, "target": None, "stop": None,
            "rsi_entry": None, "rsi_5m": None,
        })

# ── Main loop (with per-ticker crash isolation) ────────────────────────────────
def run_agent():
    """
    Called every 60 seconds by schedule.
    Each ticker is isolated — an error on QQQ doesn't block GLD.
    Heartbeat and health state are updated on every call.
    """
    if not is_market_hours():
        write_heartbeat("running")
        return

    _health["cycles_total"] += 1
    cycle_had_error = False

    for ticker in TICKERS:
        try:
            df = fetch_data(ticker)
            if df.empty:
                continue
            set_opening_range(ticker, df)
            check_signal(ticker, df)
            check_exit(ticker, df)
        except Exception as exc:
            cycle_had_error = True
            log_error(f"run_agent cycle — {ticker}", exc)
            maybe_send_crash_alert(f"cycle error on {ticker}", exc)

    _health["last_run"]    = now_et().strftime("%I:%M:%S %p ET")
    _health["last_run_ok"] = not cycle_had_error
    write_heartbeat("running")

# ── Daily / weekly summaries ──────────────────────────────────────────────────
def send_daily_summary():
    today  = now_et().strftime("%Y-%m-%d")
    trades = read_trades_for_dates([today])
    if not trades:
        send_imessage("📋 End of day — no signals fired today.")
        return
    wins       = sum(1 for t in trades if t["Result"] == "Target Hit")
    losses     = sum(1 for t in trades if t["Result"] == "Stop Loss Hit")
    confirmed  = sum(1 for t in trades if t.get("5min Confirmed") == "Yes")
    total_pnl  = sum(float(t["P&L %"].replace("%","").replace("+",""))
                     for t in trades if t["P&L %"])
    total_dollar = sum(
        float(t.get("P&L $","0").replace("$","").replace("+","").replace("-","") or 0) *
        (1 if not t.get("P&L $","").startswith("-") else -1)
        for t in trades if t.get("P&L $")
    )
    lines = [
        f"📋 Daily Summary — {today}",
        f"Trades : {len(trades)} | ✅ {wins} wins 🛑 {losses} losses",
        f"5min confirmed: {confirmed}/{len(trades)} trades",
        f"Total P&L : {'+' if total_pnl >= 0 else ''}{round(total_pnl, 2)}% "
        f"({'+' if total_dollar >= 0 else ''}${abs(total_dollar):.2f})",
        f"Errors today: {_health['errors_today']}",
        ""
    ]
    for t in trades:
        dollar = f" ({t.get('P&L $','')})" if t.get("P&L $") else ""
        lines.append(f"{t['Ticker']} {t['Direction']} → {t['Result']} {t['P&L %']}{dollar}")
    send_imessage("\n".join(lines))


def send_weekly_summary():
    today  = now_et().date()
    monday = today - datetime.timedelta(days=today.weekday())
    week_dates = [(monday + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    trades = read_trades_for_dates(week_dates)
    if not trades:
        send_imessage("📊 Weekly Summary — no trades this week.")
        return
    wins    = sum(1 for t in trades if t["Result"] == "Target Hit")
    losses  = sum(1 for t in trades if t["Result"] == "Stop Loss Hit")
    pending = sum(1 for t in trades if t["Result"] == "")
    total_pnl = sum(float(t["P&L %"].replace("%","").replace("+",""))
                    for t in trades if t["P&L %"])
    win_rate  = round(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    closed    = [t for t in trades if t["P&L %"]]
    best  = max(closed, key=lambda t: float(t["P&L %"].replace("%","").replace("+","")), default=None)
    worst = min(closed, key=lambda t: float(t["P&L %"].replace("%","").replace("+","")), default=None)
    lines = [
        f"📊 Weekly Summary — w/e {today.strftime('%b %d')}",
        f"Total trades : {len(trades)} ({pending} still open)",
        f"✅ Wins: {wins} 🛑 Losses: {losses} Win rate: {win_rate}%",
        f"Total P&L : {'+' if total_pnl >= 0 else ''}{round(total_pnl, 2)}%",
    ]
    if best:  lines.append(f"Best  : {best['Ticker']} {best['Direction']} {best['P&L %']}")
    if worst: lines.append(f"Worst : {worst['Ticker']} {worst['Direction']} {worst['P&L %']}")
    send_imessage("\n".join(lines))

# ── Pre-market scan ───────────────────────────────────────────────────────────
def pre_market_scan():
    print("[Pre-market] Running morning scan...")
    lines = ["🌅 NYLO Morning Scan — " + now_et().strftime("%A %b %d"), ""]
    for ticker in TICKERS:
        try:
            df    = fetch_data(ticker, period="5d", interval="1m")
            if df.empty:
                lines.append(f"{ticker} — fetch failed")
                continue
            today     = now_et().date()
            yesterday = df[df.index.date < today]
            if yesterday.empty:
                continue
            prev_close = round(float(yesterday["Close"].iloc[-1]), 2)
            premarket  = df[df.index.date == today]
            if premarket.empty:
                lines.append(f"{ticker} — no pre-market data yet")
                continue
            cur_price  = round(float(premarket["Close"].iloc[-1]), 2)
            change_pct = round((cur_price - prev_close) / prev_close * 100, 2)
            change_dir = "📈" if change_pct >= 0 else "📉"
            avg_vol    = float(df[df.index.date < today]["Volume"].mean())
            pre_vol    = float(premarket["Volume"].sum())
            vol_ratio  = round(pre_vol / avg_vol * 100, 1) if avg_vol > 0 else 0
            rsi        = calc_rsi(premarket) if len(premarket) >= 15 else 50.0
            if change_pct > 0.3 and rsi > 55:
                bias = "🟢 Bullish bias"
            elif change_pct < -0.3 and rsi < 45:
                bias = "🔴 Bearish bias"
            else:
                bias = "⚪ Neutral"
            lines.append(
                f"{change_dir} {ticker}\n"
                f" Price : ${cur_price} ({'+' if change_pct >= 0 else ''}{change_pct}% vs yesterday)\n"
                f" RSI   : {rsi}\n"
                f" Pre-vol: {vol_ratio}% of avg\n"
                f" Bias  : {bias}"
            )
        except Exception as exc:
            log_error(f"pre_market_scan — {ticker}", exc)
            lines.append(f"{ticker} — scan error (see agent_errors.log)")
    lines += ["", "⏰ Market opens in 30 minutes. Range sets 9:45 AM."]
    send_imessage("\n".join(lines))

# ── Market-open guard ─────────────────────────────────────────────────────────
def market_open_guard():
    """
    Runs at 9:25 AM ET.
    Checks that the agent cycled in the last 3 minutes.
    If not — the agent was started late or hung — sends an alert.
    """
    if _health["last_run"] is None:
        send_imessage(
            f"⚠️ NYLO AGENT WARNING\n"
            f"Market opens in 5 minutes but agent has NOT run a cycle yet.\n"
            f"Check that trading_agent_11.py is running!\n"
            f"Time: {now_et().strftime('%I:%M %p ET')}"
        )
    else:
        print(f"[Guard] Agent is live — last run: {_health['last_run']}")

# ── Daily reset ───────────────────────────────────────────────────────────────
def reset_daily():
    daily_trade_count["count"] = 0
    _health["errors_today"]    = 0
    _health["last_error"]      = None
    for ticker in TICKERS:
        state[ticker].update({
            "range_high": None, "range_low": None, "range_set": False,
            "in_trade":   False, "trade_dir":  None, "entry_price": None,
            "entry_time": None,  "target":     None, "stop":        None,
            "rsi_entry":  None,  "rsi_5m":     None, "allowed_dir": "both",
        })
    clear_live_trade()
    print("[Agent] Daily state reset.")

# ── Graceful shutdown ─────────────────────────────────────────────────────────
def _handle_signal(signum, frame):
    print(f"\n[Agent] Caught signal {signum} — shutting down cleanly.")
    write_heartbeat("stopped")
    send_imessage(
        f"🔴 NYLO agent stopped (signal {signum})\n"
        f"Time: {now_et().strftime('%I:%M %p ET')}\n"
        f"Uptime: {_fmt_uptime(int((now_et() - _health['start_time']).total_seconds()))}"
    )
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_log()
    clear_live_trade()
    write_heartbeat("running")

    # Start local HTTP server in background thread
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    print("=" * 60)
    print(f"  NYLO Trading Agent v{AGENT_VERSION}")
    print(f"  Watching    : {', '.join(TICKERS)}")
    print(f"  Alerts →    {IMESSAGE_TO}")
    print(f"  Subscribers : {len(SUBSCRIBERS)} friends following signals")
    print(f"  Trade log   : {LOG_FILE}")
    print(f"  Error log   : {ERROR_LOG}")
    print(f"  Heartbeat   : {HEALTH_FILE}")
    print(f"  Volume      : {VOLUME_MULTIPLIER}x avg minimum")
    print(f"  Dead zone   : 11:30 AM – 12:30 PM")
    print(f"  Gain target : {GAIN_TARGET*100:.1f}% (was 1.0%) ✅  UPDATED")
    print(f"  Stop loss   : {STOP_LOSS*100:.2f}% (was 0.5%) ✅  UPDATED")
    print(f"  Volume min  : {VOLUME_MULTIPLIER}x (was 1.5x) ✅  UPDATED")
    print(f"  Gap filter  : Only trade in direction of opening gap ✅  NEW")
    print(f"  Tickers     : QQQ only (GLD dropped) ✅  UPDATED")
    print(f"  Max trades  : Unlimited — every valid signal taken ✅  UPDATED")
    print(f"  Retry       : {FETCH_RETRIES}x with {FETCH_BACKOFF}s backoff")
    print(f"  Health      : /health endpoint")
    print(f"  Crash alert : iMessage on error")
    print(f"  Shutdown    : SIGTERM/SIGINT handled")
    print(f"  Dashboard   : http://localhost:{SERVER_PORT}")
    print("=" * 60)

    send_imessage(
        f"🤖 NYLO Agent v{AGENT_VERSION} started!\n"
        f"✅ Pre-market scanner at 9:00 AM\n"
        f"✅ ORB + RSI + VWAP + Volume + Gap Filter\n"
        f"✅ Gain: {GAIN_TARGET*100:.0f}% | Stop: {STOP_LOSS*100:.1f}% | QQQ only\n"
        f"✅ Unlimited trades — every valid signal taken\n"
        f"🔵 Strategy v12 — tighter, smarter signals 🚀"
    )

    # Schedule
    schedule.every(1).minutes.do(run_agent)

    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), day).at("09:00").do(pre_market_scan)
        getattr(schedule.every(), day).at("09:25").do(reset_daily)
        getattr(schedule.every(), day).at("09:25").do(market_open_guard)  # NEW
        getattr(schedule.every(), day).at("13:05").do(send_daily_summary)

    schedule.every().friday.at("13:05").do(send_weekly_summary)

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:
            log_error("schedule loop", exc)
            maybe_send_crash_alert("schedule loop crashed", exc)
        time.sleep(30)
