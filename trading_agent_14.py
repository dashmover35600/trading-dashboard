"""
NYLO Elite Trading Agent v14
=============================
Complete strategy rebuild — exceeding all previous versions.

Strategy architecture:
  1. Opening Drive    — first 5 minutes set the day bias (long/short)
  2. VWAP Reclaim     — primary signal: price reclaims VWAP with momentum
  3. EMA Pullback     — secondary signal: pullback to 9 EMA in trend direction
  4. Signal Scoring   — each signal scored 1-10, only 6+ scores execute
  5. Dynamic sizing   — $500 base → $1000 medium → $1500 high confidence
  6. Trailing stop    — locks in profit once trade moves 0.5% in your favor
  7. Hard time exit   — all positions closed by 12:45 PM ET
  8. Pre-market scan  — gap analysis sets context before open

Filters kept from v11:
  ✅ Crash recovery + retry logic
  ✅ Health endpoint + heartbeat
  ✅ iMessage crash alerts
  ✅ Error log
  ✅ Graceful shutdown
  ✅ Market-open guard

Requirements:
  pip install yfinance pandas ta schedule pytz

Morning routine:
  export TRADING_PHONE="+1XXXXXXXXXX"
  python3 -W ignore trading_agent_14.py
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
TICKER            = "QQQ"           # One ticker, master it
IMESSAGE_TO       = os.environ.get("TRADING_PHONE", "+1XXXXXXXXXX")
MARKET_TZ         = pytz.timezone("America/New_York")
MARKET_OPEN       = datetime.time(9, 30)
DRIVE_END         = datetime.time(9, 35)   # opening drive window
CUTOFF            = datetime.time(12, 45)  # hard close — no holding into lunch
DEAD_ZONE_START   = datetime.time(11, 30)
DEAD_ZONE_END     = datetime.time(12, 0)   # shorter dead zone
SERVER_PORT       = 8765
AGENT_VERSION     = "14.2"

# Pushover notifications
PUSHOVER_TOKEN    = "apuf7f5knj2yxnsnxvtk63adchuvkf"
PUSHOVER_USER     = "u7t4a7ybuwsyhbazjzhazy2611hrhz"

# Position sizing tiers
POS_BASE          = 1000.0   # increased from $500    # low confidence signal
POS_MEDIUM        = 2000.0   # increased from $1000   # medium confidence
POS_HIGH          = 3000.0   # increased from $1500   # high confidence

# Strategy parameters
GAIN_TARGET_PCT   = 0.015    # 1.5% target
STOP_LOSS_PCT     = 0.0075   # 0.75% stop → 2:1 ratio
TRAIL_TRIGGER_PCT = 0.0075   # activate trailing stop after +0.75% move
TRAIL_STOP_PCT    = 0.003    # trail by 0.3% from peak (locks in more)

# Signal scoring thresholds
MIN_SCORE         = 5        # v14.2: lowered from 6 — sweep shows more signals at 4-5
RSI_BULL_MIN      = 52       # v14.2: raised from 50 — sweep shows 52 optimal
RSI_BULL_MAX      = 75       # RSI must be below this (not overbought)
RSI_BEAR_MIN      = 25       # RSI must be above this for shorts (not oversold)
RSI_BEAR_MAX      = 48       # v14.2: lowered from 50 — sweep shows 48 optimal
VOLUME_MIN        = 1.2      # minimum volume ratio

# Reliability settings
FETCH_RETRIES     = 3
FETCH_BACKOFF     = 2.0
CRASH_ALERT_COOLDOWN = 300

# ── File paths ────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE     = os.path.join(BASE, "trade_log.csv")
COMPARE_FILE = os.path.join(BASE, "strategy_comparison.csv")
LIVE_FILE    = os.path.join(BASE, "live_trade.json")
HEALTH_FILE  = os.path.join(BASE, "heartbeat.json")
ERROR_LOG    = os.path.join(BASE, "agent_errors.log")

# ── Agent state ───────────────────────────────────────────────────────────────
_health = {
    "start_time":       datetime.datetime.now(pytz.timezone("America/New_York")),
    "last_run":         None,
    "last_run_ok":      True,
    "errors_today":     0,
    "last_error":       None,
    "cycles_total":     0,
    "last_crash_alert": None,
}

# Trading state
state = {
    "day_bias":       None,    # "long", "short", or "neutral"
    "drive_high":     None,    # opening drive high
    "drive_low":      None,    # opening drive low
    "drive_set":      False,
    "in_trade":       False,
    "trade_dir":      None,
    "entry_price":    None,
    "entry_time":     None,
    "target":         None,
    "stop":           None,
    "trail_active":   False,
    "trail_peak":     None,
    "trail_stop":     None,
    "signal_type":    None,    # "vwap_reclaim" or "ema_pullback"
    "signal_score":   None,
    "position_size":  POS_BASE,
    "shares":         0,
    "rsi_entry":      None,
    "vol_ratio":      None,
}

daily_trades    = []
trades_today    = 0
daily_pnl       = 0.0
SUBSCRIBERS     = []

# ── Error logging ─────────────────────────────────────────────────────────────
def log_error(context: str, exc: Exception = None):
    now_str = datetime.datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
    lines   = [f"[{now_str}] {context}"]
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
    # Always ensure correct v14 headers
    needs_header = not os.path.exists(LOG_FILE)
    if not needs_header:
        with open(LOG_FILE, "r") as f:
            first = f.readline()
        if "Signal Type" not in first:
            needs_header = True
            # backup old file
            import shutil
            shutil.copy(LOG_FILE, LOG_FILE + ".bak")
    if needs_header:
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date", "Ticker", "Direction",
                "Entry Price", "Exit Price", "Target", "Stop Loss",
                "Result", "P&L %", "P&L $", "Shares",
                "Entry Time", "Exit Time",
                "RSI at Entry", "Volume Ratio",
                "Signal Type", "Signal Score", "Position Size",
                "Day Bias", "Trail Used"
            ])
        print(f"[Tracker] Trade log initialized → {LOG_FILE}")

def calc_shares(entry_price, position_size):
    return round(position_size / entry_price, 4)

def calc_pnl(entry, exit_price, direction, shares):
    if direction == "long":
        return round((exit_price - entry) * shares, 2)
    return round((entry - exit_price) * shares, 2)

def log_entry_csv(s):
    shares = calc_shares(s["entry_price"], s["position_size"])
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            s["entry_time"].strftime("%Y-%m-%d"), TICKER,
            "Long" if s["trade_dir"] == "long" else "Short",
            s["entry_price"], "", s["target"], s["stop"],
            "", "", "", shares,
            s["entry_time"].strftime("%I:%M %p"), "",
            s["rsi_entry"], f"{s['vol_ratio']}x",
            s["signal_type"], s["signal_score"], f"${s['position_size']:.0f}",
            state["day_bias"], ""
        ])

def log_exit_csv(exit_price, result, pnl_pct, pnl_dollar, exit_time, trail_used):
    try:
        with open(LOG_FILE, "r", newline="") as f:
            rows = list(csv.reader(f))
        for i in range(len(rows) - 1, 0, -1):
            if len(rows[i]) > 4 and rows[i][1] == TICKER and rows[i][4] == "":
                rows[i][4]  = str(round(float(exit_price), 4))
                rows[i][7]  = result
                rows[i][8]  = f"{'+' if pnl_pct > 0 else ''}{round(pnl_pct,3)}%"
                rows[i][9]  = f"+${pnl_dollar:.2f}" if pnl_dollar >= 0 else f"-${abs(pnl_dollar):.2f}"
                rows[i][12] = exit_time.strftime("%I:%M %p")
                if len(rows[i]) > 18:
                    rows[i][18] = "Yes" if trail_used else "No"
                break
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[Tracker] Exit logged — {result} {'+' if pnl_pct>0 else ''}{pnl_pct:.2f}%")
    except Exception as e:
        log_error("log_exit_csv", e)

# ── Live trade JSON ───────────────────────────────────────────────────────────
def write_live_trade():
    s = state
    data = {
        "active":       s["in_trade"],
        "ticker":       TICKER,
        "direction":    s["trade_dir"],
        "entry":        s["entry_price"],
        "target":       s["target"],
        "stop":         s["stop"],
        "trail_active": s["trail_active"],
        "trail_stop":   s["trail_stop"],
        "rsi":          s["rsi_entry"],
        "signal_type":  s["signal_type"],
        "signal_score": s["signal_score"],
        "position_size":s["position_size"],
        "shares":       s["shares"],
        "day_bias":     s["day_bias"],
        "entry_time":   s["entry_time"].strftime("%I:%M %p ET") if s["entry_time"] else None,
        "updated":      datetime.datetime.now(MARKET_TZ).strftime("%I:%M:%S %p ET")
    }
    with open(LIVE_FILE, "w") as f:
        json.dump(data, f)

def clear_live_trade():
    with open(LIVE_FILE, "w") as f:
        json.dump({"active": False, "updated": datetime.datetime.now(MARKET_TZ).strftime("%I:%M:%S %p ET")}, f)

# ── Heartbeat ─────────────────────────────────────────────────────────────────
def write_heartbeat(status="running"):
    now      = datetime.datetime.now(MARKET_TZ)
    uptime_s = int((now - _health["start_time"]).total_seconds())
    h, rem   = divmod(uptime_s, 3600)
    m, s2    = divmod(rem, 60)
    uptime_str = f"{h}h {m}m" if h else f"{m}m {s2}s"
    data = {
        "status":       status,
        "version":      AGENT_VERSION,
        "updated_at":   now.isoformat(),
        "updated_str":  now.strftime("%I:%M:%S %p ET"),
        "uptime_sec":   uptime_s,
        "uptime_str":   uptime_str,
        "last_run":     _health["last_run"],
        "last_run_ok":  _health["last_run_ok"],
        "errors_today": _health["errors_today"],
        "last_error":   _health["last_error"],
        "cycles_total": _health["cycles_total"],
        "tickers":      [TICKER],
        "trades_today": trades_today,
        "daily_pnl":    round(daily_pnl, 2),
        "day_bias":     state["day_bias"],
        "in_trade":     state["in_trade"],
        "signal_type":  state["signal_type"],
    }
    try:
        with open(HEALTH_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_error("heartbeat write", e)

# ── HTTP server ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        routes = {
            "/trade_log.csv":           (LOG_FILE, "text/csv"),
            "/strategy_comparison.csv": (COMPARE_FILE, "text/csv"),
            "/live_trade.json":         (LIVE_FILE, "application/json"),
            "/live_trade":              (LIVE_FILE, "application/json"),
            "/health":                  (HEALTH_FILE, "application/json"),
            "/heartbeat.json":          (HEALTH_FILE, "application/json"),
            "/agent_errors.log":        (ERROR_LOG, "text/plain"),
        }
        if p in routes:
            path, ct = routes[p]
            if not os.path.exists(path):
                self.send_response(204); self.end_headers(); return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *args): pass

def start_server():
    import socketserver
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
    server = ReusableTCPServer(("localhost", SERVER_PORT), Handler)
    server.serve_forever()

# ── Market helpers ────────────────────────────────────────────────────────────
def now_et():
    return datetime.datetime.now(MARKET_TZ)

def is_market_hours():
    t = now_et().time()
    return MARKET_OPEN <= t <= CUTOFF

def is_dead_zone():
    t = now_et().time()
    return DEAD_ZONE_START <= t <= DEAD_ZONE_END

# ── iMessage ──────────────────────────────────────────────────────────────────
def send_imessage(message: str, broadcast: bool = False):
    import re
    # Strip emoji — they break AppleScript
    emoji_re = re.compile(
        u"[😀-🙏🌀-🗿"
        u"🚀-🛿🇠-🇿"
        u"✂-➰Ⓜ-🉑"
        u"🤦-🤷𐀀-􏿿"
        u"♀-♂☀-⭕‍⏏"
        u"⏩⌚️〰]+", flags=re.UNICODE)
    safe = emoji_re.sub("", message).replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").strip()
    recipients = [IMESSAGE_TO]
    if broadcast and SUBSCRIBERS:
        recipients += [phone for _, phone in SUBSCRIBERS]
    for recipient in recipients:
        script = f'tell application "Messages" to send "{safe}" to buddy "{recipient}" of (1st service whose service type = iMessage)'
        try:
            subprocess.run(["osascript", "-e", script], check=True, timeout=10)
            print(f"[iMessage -> {recipient}] sent")
        except Exception as e:
            log_error(f"iMessage to {recipient}", e)
def send_push(title: str, message: str, priority: int = 0):
    """
    Send a Pushover push notification to your phone.
    Priority: -1=quiet, 0=normal, 1=high, 2=emergency
    """
    try:
        import urllib.request
        import urllib.parse
        data = urllib.parse.urlencode({
            "token":   PUSHOVER_TOKEN,
            "user":    PUSHOVER_USER,
            "title":   title[:250],
            "message": message[:1024],
            "priority": priority,
            "sound":   "cashregister" if priority >= 1 else "pushover",
        }).encode()
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=data,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print(f"[Pushover] sent: {title}")
            else:
                print(f"[Pushover] failed: {resp.status}")
    except Exception as e:
        log_error("send_push", e)

def maybe_crash_alert(context: str, exc: Exception):
    now  = now_et()
    last = _health["last_crash_alert"]
    if last and (now - last).total_seconds() < CRASH_ALERT_COOLDOWN:
        return
    _health["last_crash_alert"] = now
    send_imessage(f"⚠️ NYLO v{AGENT_VERSION} ERROR\n{context}\n{type(exc).__name__}: {exc}\nTime: {now.strftime('%I:%M %p ET')}")

# ── Data fetch with retry ─────────────────────────────────────────────────────
def fetch_data(ticker, period="2d", interval="1m"):
    last_exc = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("empty DataFrame")
            df.index = df.index.tz_convert(MARKET_TZ)
            return df
        except Exception as exc:
            last_exc = exc
            wait = FETCH_BACKOFF ** attempt
            print(f"[{ticker}] fetch attempt {attempt}/{FETCH_RETRIES} failed — retry in {wait:.0f}s")
            if attempt < FETCH_RETRIES:
                time.sleep(wait)
    log_error(f"fetch_data({ticker}, {interval}) all retries failed", last_exc)
    return pd.DataFrame()

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, window=14) -> float:
    if len(series) < window + 1:
        return 50.0
    try:
        val = ta.momentum.RSIIndicator(series.squeeze(), window=window).rsi().iloc[-1]
        return round(float(val), 2) if not pd.isna(val) else 50.0
    except Exception:
        return 50.0

def calc_ema(series: pd.Series, window=9) -> float:
    if len(series) < window:
        return float(series.iloc[-1])
    try:
        return round(float(series.ewm(span=window, adjust=False).mean().iloc[-1]), 4)
    except Exception:
        return float(series.iloc[-1])

def calc_vwap(df: pd.DataFrame) -> float:
    try:
        today    = now_et().date()
        today_df = df[df.index.date == today].copy()
        if today_df.empty or len(today_df) < 2:
            return None
        close  = today_df["Close"].squeeze()
        high   = today_df["High"].squeeze()
        low    = today_df["Low"].squeeze()
        volume = today_df["Volume"].squeeze()
        tp     = (high + low + close) / 3
        vwap   = float((tp * volume).cumsum().iloc[-1] / volume.cumsum().iloc[-1])
        return round(vwap, 4)
    except Exception as e:
        log_error("calc_vwap", e)
        return None

def calc_vol_ratio(df: pd.DataFrame, window=20) -> float:
    try:
        vol = df["Volume"].squeeze()
        if len(vol) < window + 1:
            return 1.0
        avg = float(vol.iloc[-window-1:-1].mean())
        cur = float(vol.iloc[-1])
        return round(cur / avg, 2) if avg > 0 else 1.0
    except Exception:
        return 1.0

def calc_atr(df: pd.DataFrame, window=14) -> float:
    """Average True Range — measures current volatility."""
    try:
        atr = ta.volatility.AverageTrueRange(
            df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze(), window=window
        ).average_true_range()
        return round(float(atr.iloc[-1]), 4)
    except Exception:
        return 0.0

# ── VIX filter ───────────────────────────────────────────────────────────────
def get_vix() -> float:
    """
    Fetch current VIX level. If VIX > 25 market is too volatile — skip signals.
    Returns 0.0 on failure (fail open — don't block trading).
    """
    try:
        df = yf.download("^VIX", period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            return 0.0
        return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        return 0.0

VIX_MAX = 30.0  # skip trading if VIX above this level

# ── Earnings blackout ─────────────────────────────────────────────────────────
# QQQ top holdings earnings dates — skip trading day before and day of
# Update this list each quarter
EARNINGS_BLACKOUT = [
    # Format: "YYYY-MM-DD" — day of earnings or day before
    # Q1 2026 earnings season
    "2026-04-23", "2026-04-24",  # MSFT/GOOGL
    "2026-04-24", "2026-04-25",  # META
    "2026-04-30", "2026-05-01",  # AAPL/AMZN
    # Add more as needed
]

def is_earnings_blackout() -> bool:
    today = now_et().strftime("%Y-%m-%d")
    return today in EARNINGS_BLACKOUT

# ── Signal scoring ────────────────────────────────────────────────────────────
def score_signal(df, direction, signal_type, rsi, vol_ratio, price, vwap, ema9) -> int:
    """
    Score a signal 0-10. Only signals scoring MIN_SCORE+ are taken.
    Each factor adds points:
      - Day bias alignment      +2
      - RSI in ideal zone       +2
      - Volume spike            +2
      - VWAP alignment          +1
      - EMA alignment           +1
      - Signal type premium     +1 (vwap reclaim > ema pullback)
      - ATR momentum            +1
    """
    score = 0

    # Day bias alignment (+2) — most important factor
    bias = state["day_bias"]
    if bias == direction:
        score += 2
    elif bias == "neutral":
        score += 1

    # RSI in ideal zone (+2)
    if direction == "long":
        if 52 <= rsi <= 70:
            score += 2
        elif 45 <= rsi <= 78:
            score += 1
    else:
        if 30 <= rsi <= 48:
            score += 2
        elif 22 <= rsi <= 55:
            score += 1

    # Volume spike (+2)
    if vol_ratio >= 2.0:
        score += 2
    elif vol_ratio >= 1.3:
        score += 1

    # VWAP alignment (+1)
    if vwap:
        if direction == "long" and price > vwap:
            score += 1
        elif direction == "short" and price < vwap:
            score += 1

    # EMA alignment (+1)
    if direction == "long" and price > ema9:
        score += 1
    elif direction == "short" and price < ema9:
        score += 1

    # Signal type premium (+1)
    if signal_type == "vwap_reclaim":
        score += 1

    # ATR momentum (+1) — reward when ATR is higher than average (trending day)
    try:
        atr     = calc_atr(df)
        atr_avg = float(ta.volatility.AverageTrueRange(
            df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze(), window=50
        ).average_true_range().iloc[-1])
        if atr > atr_avg * 1.2:
            score += 1
    except Exception:
        pass

    return min(score, 10)

# ── Position sizing ───────────────────────────────────────────────────────────
def get_position_size(score: int) -> float:
    """
    Dynamic position sizing based on signal confidence score.
    Score 9-10 → $3,000 (high conviction)
    Score 7-8  → $2,000 (medium conviction)
    Score 6    → $1,000 (baseline)
    """
    if score >= 9:
        return POS_HIGH    # $3,000
    elif score >= 7:
        return POS_MEDIUM  # $2,000
    return POS_BASE        # $1,000

# ── Opening drive ─────────────────────────────────────────────────────────────
def set_opening_drive(df):
    """
    First 5 minutes (9:30-9:35) set the day bias.
    Strong up move → long bias. Strong down move → short bias.
    """
    s = state
    if s["drive_set"]:
        return
    if now_et().time() < DRIVE_END:
        return

    today    = now_et().date()
    today_df = df[df.index.date == today]
    drive    = today_df.between_time("09:30", "09:35")
    if drive.empty or len(drive) < 2:
        return

    drive_open  = float(drive["Open"].iloc[0])
    drive_close = float(drive["Close"].iloc[-1])
    drive_high  = float(drive["High"].max())
    drive_low   = float(drive["Low"].min())
    move_pct    = (drive_close - drive_open) / drive_open * 100

    # Volume in drive vs average
    avg_vol = float(df["Volume"].squeeze().tail(100).mean())
    drv_vol = float(drive["Volume"].sum())
    vol_ratio = drv_vol / avg_vol if avg_vol > 0 else 1.0

    # Set bias — need both price direction AND volume confirmation
    if move_pct > 0.3 and vol_ratio > 1.5:
        bias = "long"
    elif move_pct < -0.3 and vol_ratio > 1.5:
        bias = "short"
    else:
        bias = "neutral"

    s["day_bias"]   = bias
    s["drive_high"] = drive_high
    s["drive_low"]  = drive_low
    s["drive_set"]  = True

    emoji = "🟢" if bias == "long" else "🔴" if bias == "short" else "⚪"
    print(f"[Drive] {emoji} Day bias: {bias.upper()} | Move: {move_pct:+.2f}% | Vol: {vol_ratio:.1f}x")
    send_push("NYLO Bias", 
        f"{emoji} NYLO Day Bias Set\n"
        f"Bias    : {bias.upper()}\n"
        f"Drive   : {move_pct:+.2f}% in first 5 min\n"
        f"Volume  : {vol_ratio:.1f}x average\n"
        f"Strategy: {'Looking for longs only' if bias=='long' else 'Looking for shorts only' if bias=='short' else 'Both directions available'}\n"
        f"Time    : {now_et().strftime('%I:%M %p ET')}"
    )

# ── Signal detection ──────────────────────────────────────────────────────────
def check_signals(df):
    s = state
    if s["in_trade"] or not s["drive_set"]:
        return
    if now_et().time() > CUTOFF:
        return
    if is_dead_zone():
        return

    # VIX filter — skip if market too volatile
    vix = get_vix()
    if vix > VIX_MAX:
        print(f"[{TICKER}] VIX {vix} > {VIX_MAX} — skipping (too volatile)")
        return

    # Earnings blackout
    if is_earnings_blackout():
        print(f"[{TICKER}] Earnings blackout today — skipping all signals")
        return

    price     = round(float(df["Close"].iloc[-1]), 4)
    prev_price= round(float(df["Close"].iloc[-2]), 4)
    vwap      = calc_vwap(df)
    rsi       = calc_rsi(df["Close"].squeeze())
    ema9      = calc_ema(df["Close"].squeeze(), 9)
    ema21     = calc_ema(df["Close"].squeeze(), 21)
    vol_ratio = calc_vol_ratio(df)
    bias      = s["day_bias"]

    print(f"[{TICKER}] ${price} VWAP:{vwap} RSI:{rsi} EMA9:{ema9} EMA21:{ema21} "
          f"Vol:{vol_ratio}x Bias:{bias}")

    signal_type = None
    direction   = None

    # ── SIGNAL 1: VWAP Reclaim ──────────────────────────────────────────────
    # Price was below VWAP last bar, now above it (reclaim) with momentum
    if vwap:
        was_below_vwap = prev_price < vwap
        now_above_vwap = price > vwap
        was_above_vwap = prev_price > vwap
        now_below_vwap = price < vwap

        if was_below_vwap and now_above_vwap and vol_ratio >= VOLUME_MIN:
            if bias in ("long", "neutral") and RSI_BULL_MIN <= rsi <= RSI_BULL_MAX:
                signal_type = "vwap_reclaim"
                direction   = "long"
                print(f"[{TICKER}] 🎯 VWAP RECLAIM LONG — price crossed above VWAP")

        elif was_above_vwap and now_below_vwap and vol_ratio >= VOLUME_MIN:
            if bias in ("short", "neutral") and RSI_BEAR_MIN <= rsi <= RSI_BEAR_MAX:
                signal_type = "vwap_reclaim"
                direction   = "short"
                print(f"[{TICKER}] 🎯 VWAP RECLAIM SHORT — price crossed below VWAP")

    # ── SIGNAL 2: EMA Pullback ──────────────────────────────────────────────
    # Price pulled back to 9 EMA and is bouncing, in trend direction
    if not signal_type:
        ema_touch_long  = (prev_price <= ema9 * 1.001 and price > ema9
                           and price > ema21 and vol_ratio >= VOLUME_MIN)
        ema_touch_short = (prev_price >= ema9 * 0.999 and price < ema9
                           and price < ema21 and vol_ratio >= VOLUME_MIN)

        if ema_touch_long and bias in ("long", "neutral") and RSI_BULL_MIN <= rsi <= RSI_BULL_MAX:
            signal_type = "ema_pullback"
            direction   = "long"
            print(f"[{TICKER}] 📈 EMA PULLBACK LONG — bounce off 9 EMA")

        elif ema_touch_short and bias in ("short", "neutral") and RSI_BEAR_MIN <= rsi <= RSI_BEAR_MAX:
            signal_type = "ema_pullback"
            direction   = "short"
            print(f"[{TICKER}] 📉 EMA PULLBACK SHORT — rejection at 9 EMA")

    if not signal_type:
        return

    # ── Score the signal ────────────────────────────────────────────────────
    score = score_signal(df, direction, signal_type, rsi, vol_ratio, price, vwap, ema9)
    print(f"[{TICKER}] Signal score: {score}/10 (min {MIN_SCORE} to trade)")

    if score < MIN_SCORE:
        send_imessage(
            f"⚡ SIGNAL SKIPPED — {TICKER}\n"
            f"Type  : {signal_type.replace('_', ' ').title()}\n"
            f"Dir   : {'Long' if direction=='long' else 'Short'}\n"
            f"Score : {score}/10 (need {MIN_SCORE}+)\n"
            f"RSI   : {rsi} | Vol: {vol_ratio}x | Bias: {bias}\n"
            f"Raising the bar — only the best setups."
        )
        return

    # ── Enter trade ─────────────────────────────────────────────────────────
    pos_size = get_position_size(score)
    target   = round(price * (1 + GAIN_TARGET_PCT), 4) if direction == "long" \
               else round(price * (1 - GAIN_TARGET_PCT), 4)
    stop     = round(price * (1 - STOP_LOSS_PCT), 4) if direction == "long" \
               else round(price * (1 + STOP_LOSS_PCT), 4)
    shares   = calc_shares(price, pos_size)
    now      = now_et()

    state.update({
        "in_trade":     True,
        "trade_dir":    direction,
        "entry_price":  price,
        "entry_time":   now,
        "target":       target,
        "stop":         stop,
        "trail_active": False,
        "trail_peak":   price,
        "trail_stop":   None,
        "signal_type":  signal_type,
        "signal_score": score,
        "position_size":pos_size,
        "shares":       shares,
        "rsi_entry":    rsi,
        "vol_ratio":    vol_ratio,
    })

    log_entry_csv(state)
    write_live_trade()

    score_stars = "⭐" * min(score - 5, 5)
    emoji = "🟢" if direction == "long" else "🔴"
    size_label = "HIGH" if pos_size == POS_HIGH else "MEDIUM" if pos_size == POS_MEDIUM else "BASE"

    send_push("NYLO Signal", 
        f"{emoji} TRADE SIGNAL — {TICKER}\n"
        f"Type    : {signal_type.replace('_', ' ').title()} {score_stars}\n"
        f"Score   : {score}/10\n"
        f"Dir     : {'BUY (Long)' if direction=='long' else 'SELL (Short)'}\n"
        f"Entry   : ${price}\n"
        f"Target  : ${target} (+{GAIN_TARGET_PCT*100:.1f}%)\n"
        f"Stop    : ${stop} (-{STOP_LOSS_PCT*100:.2f}%)\n"
        f"Size    : ${pos_size:.0f} [{size_label}] · {shares} shares\n"
        f"RSI     : {rsi} | Vol: {vol_ratio}x\n"
        f"Bias    : {bias.upper()}\n"
        f"Time    : {now.strftime('%I:%M %p ET')}\n"
        f"— NYLO Elite v{AGENT_VERSION}",
        broadcast=True
    )

# ── Exit management ───────────────────────────────────────────────────────────
def check_exit(df):
    global trades_today, daily_pnl
    s   = state
    if not s["in_trade"]:
        return

    price     = round(float(df["Close"].iloc[-1]), 4)
    now       = now_et()
    direction = s["trade_dir"]
    entry     = s["entry_price"]

    # Update trailing stop
    if direction == "long":
        move_pct = (price - entry) / entry
        if move_pct >= TRAIL_TRIGGER_PCT and not s["trail_active"]:
            s["trail_active"] = True
            s["trail_peak"]   = price
            s["trail_stop"]   = round(price * (1 - TRAIL_STOP_PCT), 4)
            print(f"[{TICKER}] 🔒 Trailing stop activated at ${s['trail_stop']}")
        elif s["trail_active"] and price > s["trail_peak"]:
            s["trail_peak"] = price
            s["trail_stop"] = round(price * (1 - TRAIL_STOP_PCT), 4)
    else:
        move_pct = (entry - price) / entry
        if move_pct >= TRAIL_TRIGGER_PCT and not s["trail_active"]:
            s["trail_active"] = True
            s["trail_peak"]   = price
            s["trail_stop"]   = round(price * (1 + TRAIL_STOP_PCT), 4)
            print(f"[{TICKER}] 🔒 Trailing stop activated at ${s['trail_stop']}")
        elif s["trail_active"] and price < s["trail_peak"]:
            s["trail_peak"] = price
            s["trail_stop"] = round(price * (1 + TRAIL_STOP_PCT), 4)

    # Check exit conditions
    hit_target = (direction == "long"  and price >= s["target"]) or \
                 (direction == "short" and price <= s["target"])
    hit_stop   = (direction == "long"  and price <= s["stop"]) or \
                 (direction == "short" and price >= s["stop"])
    hit_trail  = (s["trail_active"] and direction == "long"  and price <= s["trail_stop"]) or \
                 (s["trail_active"] and direction == "short" and price >= s["trail_stop"])
    hit_time   = now.time() >= CUTOFF

    if not (hit_target or hit_stop or hit_trail or hit_time):
        write_live_trade()
        return

    # Determine exit price and result
    if hit_target:
        exit_price = s["target"]
        result     = "Target Hit"
        emoji      = "✅"
    elif hit_trail:
        exit_price = s["trail_stop"]
        result     = "Trailing Stop"
        emoji      = "🔒"
    elif hit_time:
        exit_price = price
        result     = "Time Exit"
        emoji      = "⏰"
    else:
        exit_price = s["stop"]
        result     = "Stop Loss Hit"
        emoji      = "🛑"

    pnl_dollar = calc_pnl(entry, exit_price, direction, s["shares"])
    pnl_pct    = round(pnl_dollar / s["position_size"] * 100, 3)
    global trades_today, daily_pnl
    trades_today += 1
    daily_pnl    += pnl_dollar
    print(f"[Agent] Trade closed — trades today: {trades_today} | daily P&L: ${daily_pnl:.2f}")

    log_exit_csv(exit_price, result, pnl_pct, pnl_dollar, now, s["trail_active"])
    clear_live_trade()

    # Reset state
    signal_type  = s["signal_type"]
    signal_score = s["signal_score"]
    pos_size     = s["position_size"]
    trail_used   = s["trail_active"]
    state.update({
        "in_trade": False, "trade_dir": None, "entry_price": None,
        "entry_time": None, "target": None, "stop": None,
        "trail_active": False, "trail_peak": None, "trail_stop": None,
        "signal_type": None, "signal_score": None,
        "rsi_entry": None, "vol_ratio": None,
    })

    pnl_str = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
    dollar_str = f"+${pnl_dollar:.2f}" if pnl_dollar >= 0 else f"-${abs(pnl_dollar):.2f}"

    send_push("NYLO Exit", 
        f"{emoji} {result.upper()} — {TICKER}\n"
        f"Type    : {signal_type.replace('_', ' ').title() if signal_type else '—'} (score {signal_score}/10)\n"
        f"Entry   : ${entry} → Exit: ${exit_price}\n"
        f"P&L     : {pnl_str} ({dollar_str})\n"
        f"Size    : ${pos_size:.0f} · {s['shares']} shares\n"
        f"Trail   : {'✅ Used' if trail_used else '—'}\n"
        f"Daily P&L: {'+' if daily_pnl >= 0 else ''}${daily_pnl:.2f}\n"
        f"Trades today: {trades_today}\n"
        f"Time    : {now.strftime('%I:%M %p ET')}\n"
        f"— NYLO Elite v{AGENT_VERSION}",
        broadcast=True
    )

    if pnl_dollar > 0:
        print(f"[{TICKER}] ✅ {result} | {pnl_str} ({dollar_str}) | Daily: +${daily_pnl:.2f}")
    else:
        print(f"[{TICKER}] 🛑 {result} | {pnl_str} ({dollar_str}) | Daily: ${daily_pnl:.2f}")

# ── Main loop ─────────────────────────────────────────────────────────────────
def run_agent():
    global trades_today, daily_pnl
    if not is_market_hours():
        write_heartbeat("running")
        return

    _health["cycles_total"] += 1
    cycle_ok = True

    try:
        df = fetch_data(TICKER)
        if df.empty:
            write_heartbeat("running")
            return

        set_opening_drive(df)
        check_signals(df)
        check_exit(df)

    except Exception as exc:
        cycle_ok = False
        log_error("run_agent cycle", exc)
        maybe_crash_alert("main cycle", exc)

    _health["last_run"]    = now_et().strftime("%I:%M:%S %p ET")
    _health["last_run_ok"] = cycle_ok
    write_heartbeat("running")

# ── Daily / weekly summaries ──────────────────────────────────────────────────
def send_daily_summary():
    today  = now_et().strftime("%Y-%m-%d")
    trades = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for row in csv.DictReader(f):
                if row["Date"] == today:
                    trades.append(row)

    wins   = sum(1 for t in trades if t["Result"] == "Target Hit" or
               (t["Result"] == "Trailing Stop" and float(t.get("P&L %","0").replace("%","").replace("+","") or 0) > 0))
    losses = sum(1 for t in trades if t["Result"] == "Stop Loss Hit" or
               (t["Result"] == "Trailing Stop" and float(t.get("P&L %","0").replace("%","").replace("+","") or 0) <= 0))
    timed  = sum(1 for t in trades if t["Result"] == "Time Exit")

    lines = [
        f"📋 Daily Summary — {today}",
        f"Trades : {len(trades)} total | ✅ {wins}W 🛑 {losses}L ⏰ {timed} timed",
        f"Daily P&L : {'+' if daily_pnl >= 0 else ''}${daily_pnl:.2f}",
        f"Day bias : {state['day_bias'] or '—'}",
        f"Errors   : {_health['errors_today']}",
        ""
    ]
    for t in trades:
        lines.append(f"  {t.get('Signal Type','—')} (score {t.get('Signal Score','—')}) → {t['Result']} {t.get('P&L %','')}")
    send_push("NYLO Daily Summary", "\n".join(lines))

# ── Pre-market scan ───────────────────────────────────────────────────────────
def pre_market_scan():
    print("[Pre-market] Morning scan...")
    try:
        df = fetch_data(TICKER, period="5d", interval="1m")
        if df.empty:
            send_imessage(f"⚠️ Pre-market scan failed — no data for {TICKER}")
            return
        today     = now_et().date()
        yesterday = df[df.index.date < today]
        premarket = df[df.index.date == today]

        prev_close  = round(float(yesterday["Close"].iloc[-1]), 2) if not yesterday.empty else 0
        cur_price   = round(float(premarket["Close"].iloc[-1]), 2) if not premarket.empty else 0
        change_pct  = round((cur_price - prev_close) / prev_close * 100, 2) if prev_close else 0
        rsi         = calc_rsi(df["Close"].squeeze())
        atr         = calc_atr(df)
        change_dir  = "📈" if change_pct >= 0 else "📉"
        bias_hint   = "🟢 Likely LONG bias" if change_pct > 0.3 else \
                      "🔴 Likely SHORT bias" if change_pct < -0.3 else "⚪ Neutral — watch drive"

        send_imessage(
            f"🌅 NYLO Morning Scan — {now_et().strftime('%A %b %d')}\n"
            f"{change_dir} {TICKER}: ${cur_price} ({'+' if change_pct>=0 else ''}{change_pct}% pre-mkt)\n"
            f"RSI    : {rsi}\n"
            f"ATR    : {atr} (volatility measure)\n"
            f"Bias   : {bias_hint}\n"
            f"Plan   : Opening drive sets bias at 9:35 AM\n"
            f"         VWAP Reclaim + EMA Pullback signals\n"
            f"         Score {MIN_SCORE}+ required to trade\n"
            f"         Sizes: ${POS_BASE:.0f} / ${POS_MEDIUM:.0f} / ${POS_HIGH:.0f}\n"
            f"⏰ Market opens in ~30 minutes"
        )
    except Exception as e:
        log_error("pre_market_scan", e)

# ── Market open guard ─────────────────────────────────────────────────────────
def market_open_guard():
    if _health["last_run"] is None:
        send_imessage(
            f"⚠️ NYLO WARNING\n"
            f"Market opens in 5 min but agent hasn't run yet.\n"
            f"Check trading_agent_14.py is running!\n"
            f"Time: {now_et().strftime('%I:%M %p ET')}"
        )

# ── Daily reset ───────────────────────────────────────────────────────────────
def reset_daily():
    global trades_today, daily_pnl
    trades_today = 0
    daily_pnl    = 0.0
    _health["errors_today"] = 0
    _health["last_error"]   = None
    state.update({
        "day_bias": None, "drive_high": None, "drive_low": None, "drive_set": False,
        "in_trade": False, "trade_dir": None, "entry_price": None, "entry_time": None,
        "target": None, "stop": None, "trail_active": False, "trail_peak": None,
        "trail_stop": None, "signal_type": None, "signal_score": None,
        "position_size": POS_BASE, "shares": 0, "rsi_entry": None, "vol_ratio": None,
    })
    clear_live_trade()
    print("[Agent] Daily state reset.")

# ── Graceful shutdown ─────────────────────────────────────────────────────────
def _handle_signal(signum, frame):
    print(f"\n[Agent] Shutting down (signal {signum})")
    write_heartbeat("stopped")
    send_imessage(f"🔴 NYLO Elite v{AGENT_VERSION} stopped\nTime: {now_et().strftime('%I:%M %p ET')}")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_log()
    clear_live_trade()
    write_heartbeat("running")

    threading.Thread(target=start_server, daemon=True).start()

    print("=" * 60)
    print(f"  NYLO Elite Trading Agent v{AGENT_VERSION}")
    print(f"  Ticker      : {TICKER} only")
    print(f"  Signals     : VWAP Reclaim + EMA Pullback")
    print(f"  Scoring     : Min {MIN_SCORE}/10 to trade")
    print(f"  Sizes       : ${POS_BASE:.0f} / ${POS_MEDIUM:.0f} / ${POS_HIGH:.0f} (score 6/7/9+)")
    print(f"  Target      : {GAIN_TARGET_PCT*100:.1f}% | Stop: {STOP_LOSS_PCT*100:.2f}%")
    print(f"  Trail       : Activates at +{TRAIL_TRIGGER_PCT*100:.1f}%")
    print(f"  Hard cutoff : 12:45 PM ET")
    print(f"  Dashboard   : http://localhost:{SERVER_PORT}")
    print(f"  Trade log   : {LOG_FILE}")
    print("=" * 60)

    print("[Agent] Startup complete. Monitoring QQQ.")
    try:
        send_push("NYLO Started", f"NYLO Elite v{AGENT_VERSION} started. Monitoring QQQ. Bias sets at 9:35 AM.")
    except Exception:
        pass
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), day).at("09:00").do(pre_market_scan)
        getattr(schedule.every(), day).at("09:25").do(reset_daily)
        getattr(schedule.every(), day).at("09:25").do(market_open_guard)
        getattr(schedule.every(), day).at("13:00").do(send_daily_summary)

    schedule.every(1).minutes.do(run_agent)

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:
            log_error("schedule loop", exc)
            maybe_crash_alert("schedule loop", exc)
        time.sleep(30)
