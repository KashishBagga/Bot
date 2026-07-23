# 📚 Strategy Reference

This is the single source of truth for **what every strategy does and exactly what triggers a call**. Each strategy runs as an independent *experiment* every 5-minute candle against one shared `MarketSnapshot` per symbol (`NSE:NIFTY50-INDEX`, `NSE:NIFTYBANK-INDEX`). A strategy emits a **signal dict**; if `accepted == True` it becomes a real paper trade, otherwise it is tracked as a counterfactual (shadow) trade for filter research.

> Signals are options trades: `BUY CALL` / `BUY PUT` resolve to a single ATM option leg; `STRADDLE` / `STRANGLE` resolve to two legs (see [§Volatility combos](#volatility-combos)).

---

## 🧩 The confluence model (how strategies combine)

Every strategy **works on its own**, but they also reinforce each other through a shared **`MarketView`** (`src/core/market_view.py`), computed once per snapshot from:

- **Chart patterns** (`src/core/chart_patterns.py`): H&S, double top/bottom, triangles, flags, RSI divergence, squeeze.
- **Structural daily bias** (BULLISH / BEARISH / NEUTRAL).
- **Regime** (trend/range/vol) and **RVOL** (participation).

`MarketView` produces `bull_score`, `bear_score`, `vol_score` (0–1), a `directional_score` (−1…+1), a `regime_label` (TREND_UP / TREND_DOWN / RANGE / VOLATILE), and the list of active patterns.

Directional strategies call **`view.confluence_boost(side)`** → a **0.7×–1.3× multiplier** applied to their own confidence: agreement with the aggregated view boosts, conflict dampens. So a clean double-bottom raises the confidence of an EMA-pullback BUY CALL firing at the same time, without either strategy knowing about the other. The `MarketView.summary` string is also what the dashboard shows as the live "current market view".

---

## 📈 Directional strategies (single option leg)

Common risk plumbing (shared): take-profit = **nearest opposing H1 zone**, floored at a **2R** projection, capped at **5×ATR**; stops never tighter than **0.5×ATR**; a signal is rejected if **RR < 1.5** (`LOW_RR`).

### 1. Structural Breakout — `Structural_v3.2_RVOL1.0`, `Structural_v3.2_RVOL0.8`
- **Hypothesis:** trade sweeps, breakouts and traps aligned with macro structure.
- **Triggers:** **SWEEP** (price sweeps a supply/demand zone + strong rejection candle), **BREAKOUT** (break of a confirmed 5m swing in the trend direction), **TRAP** (failed breakout — fade the move). *TRAP is evaluated before BREAKOUT.*
- **Filters:** RVOL ≥ threshold (1.0 / 0.8), HTF bias not opposed, move-efficiency > 0.6, wickiness < 0.5, RR ≥ 1.5.
- **Frozen v3.2 engine** (`enhanced_strategy_engine.py`).

### 2. EMA Pullback — `EMA_Pullback_20_50_RVOL1.0`
- **Hypothesis:** established trends resume after a pullback to the 20-EMA.
- **Trigger:** in an EMA-aligned trend, price pulls back to the 20-EMA then resumes → `BUY CALL` (up-trend) / `BUY PUT` (down-trend).
- **Filters:** RVOL ≥ 1.0 (`LOW_RVOL`), daily-bias alignment (`BIAS_MISMATCH`), move-efficiency ≥ 0.6 (`LOW_EFFICIENCY`), RR ≥ 1.5.

### 3. VWAP Reversion — `VWAP_Reversion_1.5ATR_RVOL1.0`
- **Hypothesis:** price overstretched from VWAP mean-reverts when it prints a rejection.
- **Trigger:** price > 1.5×ATR from intraday VWAP **and** a rejection candle → fade back to VWAP (overstretched up → `BUY PUT`; down → `BUY CALL`).
- **Filters:** RVOL, bias, efficiency, RR.

### 4. Previous-Day Extremes — `PrevDay_Extremes_RVOL1.2`
- **Hypothesis:** the prior day's high/low are magnets; they are either swept (fakeout) or broken with volume.
- **Triggers:** **sweep/reversal** of PDH/PDL (reversal RVOL ≥ 1.0) or **volume-backed breakout** of PDH/PDL (breakout RVOL ≥ 1.2), within a proximity band.
- **Filters:** RVOL (tier-dependent), bias, RR.

### 5–6. Opening Range Breakout — `ORB_15m_RVOL1.2`, `ORB_30m_RVOL1.2`
- **Hypothesis:** the opening-auction range sets the day's key levels; a break signals direction.
- **Trigger:** after the opening window (09:30 / 09:45 IST), a close beyond the opening range high/low → `BUY CALL` / `BUY PUT`.
- **Filters:** RVOL ≥ 1.2, bias, move-efficiency ≥ 0.6, no entry after 15:00 (`LATE_SESSION`), RR.

### 7. ATR Squeeze Breakout — `ATR_Squeeze_RVOL1.0`
- **Hypothesis:** momentum breakouts follow low-volatility compression.
- **Trigger:** ATR percentile ≤ 0.20 (compressed) **and** a directional structural break → `BUY CALL` / `BUY PUT`. NEUTRAL structure is rejected (`NEUTRAL_TREND`).
- **Filters:** RVOL ≥ 1.0, bias, RR.

### 8–9. Market Geometry — `Geometry_v1.0_Score35`, `Geometry_v1.0_Score50`
- **Hypothesis:** confluence of geometric levels (composite levels, trendlines, MKE narrative) marks high-probability turns/retests.
- **Trigger:** price at a confluence zone (score ≥ 50) with a body-fraction confirmation, or a trendline break/retest.
- **Filters:** min confluence score, narrative-bias agreement (`NARRATIVE_BIAS_*`), bias confidence ≥ 0.45, RR ≥ 1.5 (1.8 for the tight variant).

### 10. Institutional Order Flow — `OrderFlow_v1.0`
- **Hypothesis:** institutional stop-sweeps and order-flow imbalances precede directional moves.
- **Trigger:** a **liquidity sweep** (confidence ≥ 0.60) or **imbalance pullback** (confidence ≥ 0.55) with a body-fraction filter → `BUY CALL` / `BUY PUT`.
- **Filters:** narrative-bias agreement, min body fraction 0.40, RR ≥ 1.5.

---

## 🆕 New confluence-aware strategies

### 11. Reversal Pattern — `Reversal_Pattern_v1.0`
- **Hypothesis:** completed reversal patterns mark trend exhaustion.
- **Triggers:** `HEAD_SHOULDERS` / `DOUBLE_TOP` confirmed (close below neckline) → `BUY PUT`; `INVERSE_HEAD_SHOULDERS` / `DOUBLE_BOTTOM` confirmed (close above neckline) → `BUY CALL`. Stop at pattern invalidation (head / opposite peak).
- **Filters:** pattern confidence ≥ 0.5 (`LOW_PATTERN_CONFIDENCE`), RVOL ≥ 1.0. **Bias = soft** (reversals may fade the prevailing bias; conflict only dampens confidence).

### 12. Continuation Pattern — `Continuation_Pattern_v1.0`
- **Hypothesis:** consolidations after an impulse resolve in the trend direction.
- **Triggers:** `TRIANGLE_ASC/SYM` upside break or `BULL_FLAG` → `BUY CALL`; `TRIANGLE_DESC/SYM` downside break or `BEAR_FLAG` → `BUY PUT`.
- **Filters:** pattern confidence ≥ 0.45, RVOL ≥ 1.1, move-efficiency ≥ 0.5. **Bias = hard** (continuations must align with the trend → `BIAS_MISMATCH`).

### 13. RSI Divergence — `RSI_Divergence_v1.0`
- **Hypothesis:** price/RSI divergence signals momentum exhaustion and reversion.
- **Triggers:** price lower-low + RSI higher-low (RSI < 45) → `BUY CALL`; price higher-high + RSI lower-high (RSI > 55) → `BUY PUT`.
- **Filters:** pattern confidence ≥ 0.5, RVOL ≤ 3.0 (avoid buying into a `VOLUME_CLIMAX`). **Bias = soft.**

### 14. Squeeze Breakout — `Squeeze_Breakout_v1.0`
- **Hypothesis:** compressed volatility (Bollinger-inside-Keltner) expands; the first range break sets direction.
- **Trigger:** a `VOL_SQUEEZE` is active **and** the close breaks the compression range high/low → `BUY CALL` / `BUY PUT`. Stop at the far range edge.
- **Filters:** RVOL ≥ 1.2. **Bias = off** (expansion can start against prior bias). *Directional sibling of the straddle strategy below.*

### <a name="volatility-combos"></a>15. Volatility Straddle / Strangle — `Volatility_Straddle_v1.0`  *(multi-leg, non-directional)*
- **Hypothesis:** when volatility is compressed **and cheap**, a large move is likely but its direction is unclear — buy both a call and a put.
- **Trigger:** a `VOL_SQUEEZE` is active or ATR percentile ≤ 0.30. Then:
  - **STRADDLE** (ATM call + ATM put) when ATR percentile ≤ 0.15 and `vol_score` ≥ 0.5 (deep compression, move imminent).
  - **STRANGLE** (OTM call + OTM put, ~1% offset) otherwise (cheaper, needs a bigger move).
- **Rejections:** ATR percentile > 0.40 → `VOL_TOO_EXPENSIVE`; |directional_score| > 0.5 → `STRONG_DIRECTIONAL` (a clear direction argues against a non-directional combo — recorded for research).
- **Management (premium space):** exit when the **combined premium** falls **−40%** (stop), rises **+60%** (target), a **24-bar time stop** (~2h), or session end. Sized on the combined premium; both legs exit together.

---

## 🛑 Common rejection codes (why a signal became a shadow trade)

| Code | Meaning |
|---|---|
| `LOW_RVOL` | Participation below the strategy's RVOL threshold |
| `BIAS_MISMATCH` | Direction opposes the higher-timeframe/daily bias |
| `LOW_EFFICIENCY` | Move efficiency below threshold (choppy move) |
| `HIGH_WICKINESS` | Candles too wicky / unstable (structural only) |
| `LOW_RR` | Reward:risk below 1.5 |
| `TP_CAPPED` | Target clamped to 5×ATR (zone was too far) |
| `LOW_PATTERN_CONFIDENCE` | Chart-pattern quality below threshold |
| `VOL_TOO_EXPENSIVE` | ATR percentile too high to buy a combo cheaply |
| `STRONG_DIRECTIONAL` | Clear trend argues against a straddle/strangle |
| `LATE_SESSION` / `NEUTRAL_TREND` / `ZERO_RISK` | Time guard / no direction / degenerate stop |

---

## 🔬 Research loop

Rejected signals run as **counterfactual (shadow) trades** through the *same* position engine, so the counterfactual tables answer "was this filter correct?" (`src/analytics/filter_attribution.py`). Every strategy exposes `metadata` (`id`, `hypothesis_*`, `version`, `maturity`) for attribution, and each experiment's positions are keyed independently by `(symbol, experiment_name)`.

_See `README.md` for the runtime loop and `CLAUDE.md` for architecture._
