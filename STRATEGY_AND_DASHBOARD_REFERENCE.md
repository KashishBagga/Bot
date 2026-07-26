# Strategy & Dashboard Reference

A snapshot of every strategy currently registered in `indian_trader.py`, every dashboard page available, and the infrastructure underneath both.

**13** strategy classes · **20** registered experiments · **3** dashboard pages · **2** symbols traded (`NSE:NIFTY50-INDEX`, `NSE:NIFTYBANK-INDEX`) · **0** real orders ever placed (paper mode only)

Every strategy runs as an independent `Experiment` against one shared market snapshot per candle — one crashing never affects the others, and several can hold simultaneous positions on the same symbol. Rejected signals aren't discarded: they run as counterfactual "shadow" trades through the identical position-management engine, building a research dataset on whether each filter is actually earning its keep.

---

## Strategies

### Core structural (frozen engine)

| Strategy | Experiments | Thesis |
|---|---|---|
| **Structural v3.2** | `Structural_v3.2_RVOL1.0`, `Structural_v3.2_RVOL0.8` | The flagship, code-frozen engine. Daily→1H→5M bias gate, then one of three setups: liquidity Sweep, structural Breakout, or failed-breakout Trap. Filters: RVOL, bias alignment, move efficiency >0.6, wickiness <0.5, RR ≥1.5, target-zone existence. |

### Pullback & reversion

| Strategy | Experiments | Thesis |
|---|---|---|
| **EMA Pullback 20/50** | `EMA_Pullback_20_50_RVOL1.0` | Enters on a controlled pullback to the 20/50 EMA inside an established trend — continuation, not reversal. |
| **VWAP Reversion** | `VWAP_Reversion_1.5ATR_RVOL1.0` | Fades price stretched ≥1.5×ATR from session VWAP back toward it. Requires a rejection candle and a non-opposed daily bias. |
| **VWAP Reclaim** *(new)* | `VWAP_Reclaim_v1.0` | The opposite thesis from Reversion: trades **continuation** when price crosses back over VWAP, instead of fading a stretch. |

### Level & range breakouts

| Strategy | Experiments | Thesis |
|---|---|---|
| **Previous Day High/Low** | `PrevDay_Extremes_RVOL1.2` | Sweep/fakeout reversal or volume-backed breakout of yesterday's extremes. |
| **Opening Range Breakout** | `ORB_15m_RVOL1.2`, `ORB_30m_RVOL1.2`, `ORB_60m_IB_RVOL1.2` | Breakout of the opening range. The 60-minute "Initial Balance" variant only works correctly after this session's fix to a cutoff-time bug that silently defaulted anything past 30 minutes back to the 15-minute window. |
| **Central Pivot Range** *(new)* | `CPR_v1.0` | Breakout of yesterday's Pivot/TC/BC band — standard floor-trader geometry, popular in Indian intraday trading specifically, not previously in this framework. |
| **Gap Continuation / Fill** *(new)* | `Gap_v1.0` | One strategy, not two contradictory ones — classifies whether an opening gap is extending (Go) or giving itself back (Fill) from price action after the open, decided fresh each candle. |
| **ATR Squeeze** | `ATR_Squeeze_RVOL1.0` | Volatility-compression setup: low ATR percentile plus a 5-minute break of structure fires a directional breakout bet. |

### Structural confluence & order flow

| Strategy | Experiments | Thesis |
|---|---|---|
| **Geometry** | `Geometry_v1.0_Score35`, `Geometry_v1.0_Score50` | Confluence-zone bounce and trendline break-and-retest, gated by narrative bias confidence. |
| **Order Flow** | `OrderFlow_v1.0` | Liquidity stop-sweep reversals and fair-value-gap imbalance pullbacks. This session fixed a real asymmetry: the PUT side of the sweep-reversal bias filter was a silent no-op — CALL signals were checked against opposing narrative bias, PUT signals weren't checked at all. |

### Chart & candlestick pattern recognition

| Strategy | Experiments | Thesis |
|---|---|---|
| **Chart Pattern** *(new)* | `ChartPattern_v1.0_Conf55`, `ChartPattern_v1.0_Conf40` | Wires up a 9-detector pattern engine that already existed in this codebase but was never connected to a trade (Double Top/Bottom, Ascending/Descending Triangle, Bull/Bear Flag, Rectangle, Head & Shoulders, Inverse H&S). Confirmed by one of 15 candlestick patterns (Doji, Engulfing, Hammer, Shooting Star, Marubozu, Morning/Evening Star and others) and cross-checked against real supply/demand zones. Targets: 1.5R partial, nearest zone, 100% measured move, 161.8% extension. |

### Multi-leg options combos

| Strategy | Experiments | Thesis |
|---|---|---|
| **Vertical Spread** *(new)* | `VerticalSpread_v1.0` (Bull Call / Bear Put) | Same directional thesis as any trend strategy here (EMA cross aligned with daily bias) — financed as a debit spread instead of a naked single-leg option. The only combo shape that's still fully directional. Defined risk. |
| **Straddle / Strangle** *(new)* | `Straddle_v1.0_VolCompression`, `Strangle_v1.0_VolCompression` | Long volatility, direction-agnostic: buys both a call and a put on ATR-percentile compression, betting realized vol is about to expand. Defined risk; uses a **realized-vol proxy**, not implied volatility (see caveats). |

---

## Dashboard

Three Streamlit pages, all reading straight from Postgres — none of them hold live broker connections of their own.

### 🔴 Live Trades — `1_🔴_Live_Trades.py`
- Real-time positions across every strategy, auto-refreshing every 20s
- Category filter spanning all strategy families, single-leg and combo alike
- Entry vs. current price, unrealized PnL in R, bars held
- Stop loss — initial vs. current, flagged when trailed
- Multiple targets — 1.5R partial, final target, plus pattern-derived extras (nearest zone / measured move / extension) where applicable
- MFE / MAE excursion tracking
- Combo positions shown separately — each leg's strike, side, entry→current premium and per-leg PnL, net premium paid, max loss/profit, target/stop R
- Stale badge if a position hasn't heartbeat in 10+ minutes

### 📡 Market State — `2_📡_Market_State.py`
- Per-symbol snapshot of daily bias, narrative bias + confidence, market regime
- RVOL, ATR, move efficiency at a glance
- Active supply/demand zones — score, rejection count, freshness
- In-progress & ready chart patterns — type, state, completion %, confidence, projected targets
- Answers "what does the system currently believe" independent of whether any trade is open

### 📊 EOD Trading Analytics & Replay — `dashboard_streamlit.py`
- Session date selector across every day the trader has run
- Executive metrics — realized PnL, win rate, expectancy, shadow PnL, trade count
- Strategy filter across the whole session
- Realized Positions tab — full trade detail, diagnostics, execution-latency audit timeline
- Counterfactual tab — every rejected candidate, why it was rejected, and its simulated outcome

A companion, non-interactive EOD report (12 Markdown/JSON sections) auto-generates at 15:35 IST — market narrative, filter attribution, experiment ranking, thesis analysis, and a research queue of hypotheses the counterfactual data supports.

---

## Infrastructure built underneath

**Live heartbeat** — Current price and unrealized PnL now refresh every candle for open positions — previously the database row only changed when a stop trailed or a target expanded, so it went stale in between. This is what makes the Live Trades page actually live.

**Risk governor visibility** — A daily-loss halt, max-concurrent-position cap, and max-deployed-capital cap gate every real entry — and now survive a restart (previously they silently reset). Signals blocked by these gates are logged to their own table instead of vanishing, so it's possible to tell whether the strategy or the risk governor is the reason a real trade didn't fire.

**Combo PnL model** — Multi-leg positions track combined-premium PnL divided by max loss — keeping them comparable in R-multiple terms to every single-leg strategy, despite the underlying risk model being completely different (defined-risk debit structures only; nothing with unbounded risk is built).

---

## What to know before trusting it

> ⚠️ **Realized vol is not implied vol.** Straddle/Strangle enter on ATR-percentile compression because this system has no implied-volatility or Greeks data anywhere — Fyers quotes return LTP/bid/ask only. Every signal from this strategy is tagged `vol_signal_type: realized_vol_proxy` rather than presented as a real volatility-surface read.

> 🚫 **Intraday only, always.** Every position — real or shadow — force-exits at 15:25 IST. There is no swing-trading or multi-day-hold concept anywhere in this system.

> 🚫 **Paper mode, full stop.** No real order has ever been placed. Every fill, premium, and exit is simulated against real market data, but nothing here has touched live capital.

> ℹ️ **Not yet built.** Iron Condor / Butterfly (naked-ish premium selling) were deliberately deferred — they need proper margin and defined-risk hedging to produce meaningful numbers even in paper mode, and that's a separate design pass.

---

*Structural Paper Trader v3.2 · Strategy Research Framework v4.0 · Reference generated from the active `indian_trader.py` experiment registry.*
