# NYLO Trading Dashboard

## Project Overview
NYLO is a paper trading signal dashboard for AAPL, GOOGL, AMD, and NVDA built by Giovanni Flores.

## Stack
- Trading agent: trading_agent_14.py (Python, runs locally on Mac)
- Dashboard: trading_dashboard.html (deployed on Vercel)
- Backtest: backtest.py
- GitHub: dashmover35600/trading-dashboard
- Vercel: trading-dashboard-tawny-tau.vercel.app
- Supabase: hwczpidgbtycddiakhnx.supabase.co (auth/members)
- Pushover: token=apuf7f5knj2yxnsnxvtk63adchuvkf / user=u7t4a7ybuwsyhbazjzhazy2611hrhz

## Agent Details
- Version: 18
- Tickers: AAPL, GOOGL, AMD, NVDA
- Strategy: VWAP Reclaim + EMA Pullback + Opening Drive
- Entry window: 9:30–10:00 AM ET (hard cutoff)
- Min score: 3/10
- RSI Bull: 52-72, RSI Bear: 28-48
- Position sizing: score 3=$500 up to score 10=$5000
- Exit params: 2% target | 1% stop | 1.25% partial | 0.5% trail
- Breakeven stop triggers at +0.5%
- Daily loss limit: -$500
- Morning brief Pushover: 8:30 AM ET
- Agent health: localhost:8765/health

## Backtest Results (v18 — current)
- Tickers: AAPL + GOOGL
- Entry window: 9:30–10:00 AM ET
- Win rate: 66.7% (12 trades, 137 days)
- Sharpe: 11.806
- Profit factor: 3.291
- Expectancy: $12.18/trade
- PnL: +$146
- Monte Carlo: 95.5% profitable (1000 sims)
- OOS: CONSISTENT

## Key Files
- ~/Downloads/trading_agent_14.py — live agent
- ~/Downloads/trading_dashboard.html — main dashboard
- ~/Downloads/backtest.py — v18 backtest engine
- ~/Downloads/trade_log.csv — real trade history
- ~/Downloads/backtest_results.json — latest backtest results
- ~/Library/LaunchAgents/com.nylo.trading-agent.plist — auto-start config

## Deploy Pattern
git add -f [files] && git commit -m "message" && git push origin main
Vercel auto-deploys on push to main.

## Agent Restart
pkill -9 -f trading_agent && sleep 2 && nohup python3 -W ignore ~/Downloads/trading_agent_14.py >> /tmp/nylo_v14.log 2>&1 &

## Common Issues
- Agent port conflict: lsof -ti :8765 | xargs kill -9
- Check agent: curl http://localhost:8765/health
- Trade log on GitHub updates every 15 min via cron
