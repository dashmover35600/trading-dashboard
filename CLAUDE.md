# NYLO Trading Dashboard

## Project Overview
NYLO is a paper trading signal dashboard for AAPL and GOOGL built by Giovanni Flores.

## Stack
- Trading agent: trading_agent_14.py (Python, runs locally on Mac)
- Dashboard: trading_dashboard.html (deployed on Vercel)
- Backtest: backtest.py
- GitHub: dashmover35600/trading-dashboard
- Vercel: trading-dashboard-tawny-tau.vercel.app
- Supabase: hwczpidgbtycddiakhnx.supabase.co (auth/members)
- Pushover: token=apuf7f5knj2yxnsnxvtk63adchuvkf / user=u7t4a7ybuwsyhbazjzhazy2611hrhz

## Agent Details
- Version: 17.1
- Tickers: AAPL, GOOGL
- Strategy: VWAP Reclaim + EMA Pullback + Opening Drive
- Min score: 3/10
- RSI Bull: 52-72, RSI Bear: 28-48
- Position sizing: score 3=$500 up to score 10=$5000
- Partial exit at +0.75%, breakeven stop at +0.5%
- Daily loss limit: -$500
- Morning brief Pushover: 8:30 AM ET
- Agent health: localhost:8765/health

## Backtest Results (v18)
- Tickers: AAPL + GOOGL
- Win rate: 79.7% (118 trades, 120 days)
- Sharpe: 3.19
- Profit factor: 1.254
- Expectancy: $0.63/trade
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
