# 📈 Weekly Trades Reasoning & Triggers Report

This report lists all trades executed during the week of August 10, 2026, separated by strategy families and experiments to avoid R blending and duplicate trade inflation. It includes capital-normalized metrics and actual ₹ P&L values.

## 📊 Experiment Performance Summary

| Experiment Name | Type | Family | Total Trades | Win Rate | Total R-PnL | Total PnL (₹) | Avg Capital Deployed (₹) | Avg Capital Efficiency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ATR_Squeeze_RVOL1.0` | Underlying-based R | Directional Single-Leg | 3 | 0.0% | -2.15R | -307.86 ₹ | 6,731.25 ₹ | -1.51% |
| `Butterfly_v1.0` | Premium-based R | Options Combo Spreads | 18 | 16.7% | -2.48R | -795.79 ₹ | 176.54 ₹ | -13.76% |
| `CreditSpread_v1.0_PCRFade` | Premium-based R | Options Combo Spreads | 5 | 20.0% | -0.24R | -161.91 ₹ | 1,457.25 ₹ | -4.80% |
| `EMA_Pullback_20_50_RVOL1.0` | Underlying-based R | Directional Single-Leg | 1 | 0.0% | -1.05R | -707.80 ₹ | 9,744.00 ₹ | -7.26% |
| `Geometry_v1.0_Score35` | Underlying-based R | Directional Single-Leg | 6 | 33.3% | -1.27R | -426.70 ₹ | 4,654.79 ₹ | -1.73% |
| `Geometry_v1.0_Score50` | Underlying-based R | Directional Single-Leg | 3 | 66.7% | +1.88R | +150.86 ₹ | 4,485.42 ₹ | +1.01% |
| `IronCondor_v1.0` | Premium-based R | Options Combo Spreads | 7 | 42.9% | +0.84R | +255.73 ₹ | 506.14 ₹ | +12.00% |
| `OIWallReaction_v1.0` | Underlying-based R | Directional Single-Leg | 7 | 14.3% | -3.16R | -876.94 ₹ | 3,671.43 ₹ | -3.30% |
| `OrderFlow_v1.0` | Underlying-based R | Directional Single-Leg | 3 | 66.7% | -0.50R | -229.66 ₹ | 6,069.25 ₹ | -1.98% |
| `PrevDay_Extremes_RVOL1.2` | Underlying-based R | Directional Single-Leg | 1 | 100.0% | +0.77R | +365.39 ₹ | 6,786.00 ₹ | +5.38% |
| `Straddle_v1.0_VolCompression` | Premium-based R | Options Combo Spreads | 8 | 12.5% | -0.91R | -4,934.60 ₹ | 10,949.66 ₹ | -11.35% |
| `Strangle_v1.0_VolCompression` | Premium-based R | Options Combo Spreads | 8 | 12.5% | -0.74R | -3,097.64 ₹ | 8,486.00 ₹ | -9.23% |
| `Structural_v3.2_RVOL0.8` | Underlying-based R | Directional Single-Leg | 8 | 37.5% | -0.46R | -217.28 ₹ | 4,916.59 ₹ | -0.41% |
| `Structural_v3.2_RVOL1.0` | Underlying-based R | Directional Single-Leg | 6 | 33.3% | -0.01R | +465.88 ₹ | 5,266.29 ₹ | +1.78% |
| `VWAP_Reclaim_v1.0` | Underlying-based R | Directional Single-Leg | 2 | 50.0% | -0.04R | -27.94 ₹ | 6,999.00 ₹ | -4.04% |
| `VerticalSpread_v1.0` | Premium-based R | Options Combo Spreads | 7 | 42.9% | -0.04R | -32.96 ₹ | 1,362.64 ₹ | -0.60% |

---

## 🧠 Unique Signal Triggers Summary (Deduplicated)
To understand the core performance of the signal triggers (independent of experiment configurations and double-counted entries):

**Total Unique Trigger Events:** 61
- **Deduplicated Win Rate:** 27.9%
- **Deduplicated Realized ₹ PnL (Representative):** -6,948.91 ₹

---

# Directional Single-Leg Strategy Ledgers

## 🧪 Experiment: `ATR_Squeeze_RVOL1.0`
Total trades: 3

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTYBANK_INDEX_ATR_Squeeze_RVOL1.0_1786432505
**Strategy/Experiment:** `SQUEEZE_BREAKOUT` / `ATR_Squeeze_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 57328.8 → 57347.785 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -149.51 ₹
**Option Deployed (₹):** 7,237.50 ₹ | **Capital Efficiency:** -2.07%
**Stop Loss:** 57347.785 (Initial: 57347.785) | **Take Profit:** 57214.89 (Initial: 57214.89)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300PE` @ premium of `₹482.5` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `SQUEEZE_BREAKOUT` under experiment `ATR_Squeeze_RVOL1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTYBANK_INDEX_ATR_Squeeze_RVOL1.0_1786520105
**Strategy/Experiment:** `SQUEEZE_BREAKOUT` / `ATR_Squeeze_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 57517.1 → 57519.645 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.179R | **Realized PnL (₹):** -26.51 ₹
**Option Deployed (₹):** 6,496.50 ₹ | **Capital Efficiency:** -0.41%
**Stop Loss:** 57519.645 (Initial: 57536.895) | **Take Profit:** 57319.15 (Initial: 57319.15)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500PE` @ premium of `₹433.1` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `SQUEEZE_BREAKOUT` under experiment `ATR_Squeeze_RVOL1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_ATR_Squeeze_RVOL1.0_1786521005
**Strategy/Experiment:** `SQUEEZE_BREAKOUT` / `ATR_Squeeze_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 57510.25 → 57526.87 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.917R | **Realized PnL (₹):** -131.84 ₹
**Option Deployed (₹):** 6,459.75 ₹ | **Capital Efficiency:** -2.04%
**Stop Loss:** 57526.87 (Initial: 57529.42) | **Take Profit:** 57318.55 (Initial: 57318.55)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500PE` @ premium of `₹430.65` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `SQUEEZE_BREAKOUT` under experiment `ATR_Squeeze_RVOL1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `EMA_Pullback_20_50_RVOL1.0`
Total trades: 1

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTYBANK_INDEX_EMA_Pullback_20_50_RVOL1.0_1786506305
**Strategy/Experiment:** `PULLBACK` / `EMA_Pullback_20_50_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 57446.25 → 57356.37 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -707.80 ₹
**Option Deployed (₹):** 9,744.00 ₹ | **Capital Efficiency:** -7.26%
**Stop Loss:** 57356.37 (Initial: 57356.37) | **Take Profit:** 57626.01 (Initial: 57626.01)
**Option Resolved:** `NSE:BANKNIFTY26AUG57400CE` @ premium of `₹649.6` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** The `EmaPullbackStrategy` triggered on a trend-continuation setup. Price pulled back to touch the 20 EMA, and then printed a green/red confirmation body in the direction of the macro EMA trend (bullish/bearish crossover).
**SL/TP Placement Logic:** Stop Loss was set below/above the 50 EMA with a small buffer (`0.2 * ATR`), floored at `0.5 * ATR` from entry. Take Profit was projected to the nearest resistance or fallback R-multiple.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---


## 🧪 Experiment: `Geometry_v1.0_Score35`
Total trades: 6

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786423205
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24479.55 → 24477.43 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.166R | **Realized PnL (₹):** +61.07 ₹
**Option Deployed (₹):** 4,961.25 ₹ | **Capital Efficiency:** +1.23%
**Stop Loss:** 24477.43 (Initial: 24489.38) | **Take Profit:** 24448.23 (Initial: 24448.23)
**Option Resolved:** `NSE:NIFTY2681124500PE` @ premium of `₹66.15` (Lots: 3.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the TRENDLINE RESISTANCE @ 24479.64 (±0.0pts, score=52). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_Geometry_v1.0_Score35_1786433405
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 57327.9 → 57347.99 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -158.21 ₹
**Option Deployed (₹):** 7,005.00 ₹ | **Capital Efficiency:** -2.26%
**Stop Loss:** 57347.99 (Initial: 57347.99) | **Take Profit:** 57295.76 (Initial: 57295.76)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300PE` @ premium of `₹467.0` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the RESISTANCE EMA50 @ 57329.44 (±9.1pts, score=50). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.754.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786506905
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 24423.0 → 24448.16 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +2.673R | **Realized PnL (₹):** +308.73 ₹
**Option Deployed (₹):** 4,352.50 ₹ | **Capital Efficiency:** +7.09%
**Stop Loss:** 24448.16 (Initial: 24413.76) | **Take Profit:** 24460.45 (Initial: 24451.21)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹174.1` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the PWL PDL + TRENDLINE VWAP @ 24423.51 (±3.3pts, score=53). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.768.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786510805
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24371.5 → 24381.95 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -137.16 ₹
**Option Deployed (₹):** 2,525.00 ₹ | **Capital Efficiency:** -5.43%
**Stop Loss:** 24381.95 (Initial: 24381.95) | **Take Profit:** 24350.3 (Initial: 24350.3)
**Option Resolved:** `NSE:NIFTY2681824350PE` @ premium of `₹101.0` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the TRENDLINE RESISTANCE @ 24371.73 (±0.0pts, score=51). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.758.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786517105
**Strategy/Experiment:** `TRENDLINE_RETEST` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (NONE)
**Underlying Entry → Exit:** 24267.1 → 24277.85 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -282.19 ₹
**Option Deployed (₹):** 4,942.50 ₹ | **Capital Efficiency:** -5.71%
**Stop Loss:** 24277.85 (Initial: 24277.85) | **Take Profit:** 24250.0 (Initial: 24250.0)
**Option Resolved:** `NSE:NIFTY2681824250PE` @ premium of `₹98.85` (Lots: 2.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy`. Specifically, price hit the confluence zone. The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### 📅 Date: 2026-08-13
### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786603805
**Strategy/Experiment:** `TRENDLINE_RETEST` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24392.15 → 24400.45 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.957R | **Realized PnL (₹):** -218.94 ₹
**Option Deployed (₹):** 4,142.50 ₹ | **Capital Efficiency:** -5.29%
**Stop Loss:** 24398.7 (Initial: 24401.3) | **Take Profit:** 24363.03 (Initial: 24363.03)
**Option Resolved:** `NSE:NIFTY2681824400PE` @ premium of `₹82.85` (Lots: 2.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy`. Specifically, price hit the confluence zone. The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `Geometry_v1.0_Score50`
Total trades: 3

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score50_1786423205
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score50`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24479.55 → 24477.43 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.166R | **Realized PnL (₹):** +61.07 ₹
**Option Deployed (₹):** 4,961.25 ₹ | **Capital Efficiency:** +1.23%
**Stop Loss:** 24477.43 (Initial: 24489.38) | **Take Profit:** 24448.23 (Initial: 24448.23)
**Option Resolved:** `NSE:NIFTY2681124500PE` @ premium of `₹66.15` (Lots: 3.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the TRENDLINE RESISTANCE @ 24479.64 (±0.0pts, score=52). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score50_1786506905
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score50`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 24423.0 → 24448.16 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +2.673R | **Realized PnL (₹):** +308.73 ₹
**Option Deployed (₹):** 4,352.50 ₹ | **Capital Efficiency:** +7.09%
**Stop Loss:** 24448.16 (Initial: 24413.76) | **Take Profit:** 24460.45 (Initial: 24451.21)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹174.1` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the PWL PDL + TRENDLINE VWAP @ 24423.51 (±3.3pts, score=53). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.768.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-13
### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score50_1786603805
**Strategy/Experiment:** `TRENDLINE_RETEST` / `Geometry_v1.0_Score50`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24392.15 → 24400.45 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.957R | **Realized PnL (₹):** -218.94 ₹
**Option Deployed (₹):** 4,142.50 ₹ | **Capital Efficiency:** -5.29%
**Stop Loss:** 24398.7 (Initial: 24401.3) | **Take Profit:** 24363.03 (Initial: 24363.03)
**Option Resolved:** `NSE:NIFTY2681824400PE` @ premium of `₹82.85` (Lots: 2.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy`. Specifically, price hit the confluence zone. The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `OIWallReaction_v1.0`
Total trades: 7

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786424405
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24479.0 → 24472.84 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.834R | **Realized PnL (₹):** -245.74 ₹
**Option Deployed (₹):** 3,933.75 ₹ | **Capital Efficiency:** -6.25%
**Stop Loss:** 24472.84 (Initial: 24471.14) | **Take Profit:** 24500.0 (Initial: 24500.0)
**Option Resolved:** `NSE:NIFTY2681124500CE` @ premium of `₹52.45` (Lots: 3.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786425905
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24476.7 → 24469.805 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -361.99 ₹
**Option Deployed (₹):** 4,590.00 ₹ | **Capital Efficiency:** -7.89%
**Stop Loss:** 24469.805 (Initial: 24469.805) | **Take Profit:** 24500.0 (Initial: 24500.0)
**Option Resolved:** `NSE:NIFTY2681124500CE` @ premium of `₹45.9` (Lots: 4.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786509005
**Strategy/Experiment:** `OI_WALL_BREAK` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (NONE)
**Underlying Entry → Exit:** 24378.55 → 24387.84 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.418R | **Realized PnL (₹):** -131.91 ₹
**Option Deployed (₹):** 2,997.50 ₹ | **Capital Efficiency:** -4.40%
**Stop Loss:** 24387.84 (Initial: 24403.791) | **Take Profit:** 24302.74 (Initial: 24302.74)
**Option Resolved:** `NSE:NIFTY2681824400PE` @ premium of `₹119.9` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_BREAK` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786515905
**Strategy/Experiment:** `OI_WALL_BREAK` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (NONE)
**Underlying Entry → Exit:** 24282.1 → 24287.855 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.327R | **Realized PnL (₹):** -84.90 ₹
**Option Deployed (₹):** 2,831.25 ₹ | **Capital Efficiency:** -3.00%
**Stop Loss:** 24287.855 (Initial: 24302.855) | **Take Profit:** 24225.01 (Initial: 24225.01)
**Option Resolved:** `NSE:NIFTY2681824300PE` @ premium of `₹113.25` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_BREAK` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786520705
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24288.25 → 24290.37 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.317R | **Realized PnL (₹):** +22.89 ₹
**Option Deployed (₹):** 3,985.00 ₹ | **Capital Efficiency:** +0.57%
**Stop Loss:** 24290.37 (Initial: 24282.47) | **Take Profit:** 24322.93 (Initial: 24322.93)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹159.4` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786524005
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (OI_WALL_FADE)
**Underlying Entry → Exit:** 24284.35 → 24284.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Index points-based):** -0.050R | **Realized PnL (₹):** -3.29 ₹
**Option Deployed (₹):** 3,896.25 ₹ | **Capital Efficiency:** -0.08%
**Stop Loss:** 24279.09 (Initial: 24279.09) | **Take Profit:** 24315.91 (Initial: 24315.91)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹155.85` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-13
### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786604105
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 24399.2 → 24393.8 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.799R | **Realized PnL (₹):** -72.00 ₹
**Option Deployed (₹):** 3,466.25 ₹ | **Capital Efficiency:** -2.08%
**Stop Loss:** 24394.695 (Initial: 24391.994) | **Take Profit:** 24442.43 (Initial: 24442.43)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹138.65` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `OrderFlow_v1.0`
Total trades: 3

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTY50_INDEX_OrderFlow_v1.0_1786431305
**Strategy/Experiment:** `LIQUIDITY_SWEEP` / `OrderFlow_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24431.85 → 24437.564 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -300.04 ₹
**Option Deployed (₹):** 4,390.00 ₹ | **Capital Efficiency:** -6.83%
**Stop Loss:** 24437.564 (Initial: 24437.564) | **Take Profit:** 24397.56 (Initial: 24397.56)
**Option Resolved:** `NSE:NIFTY2681124450PE` @ premium of `₹43.9` (Lots: 4.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle.
**SL/TP Placement Logic:** Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. Take Profit was placed at the nearest opposing liquidity target or FVG imbalance.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_OrderFlow_v1.0_1786432505
**Strategy/Experiment:** `LIQUIDITY_SWEEP` / `OrderFlow_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 57328.8 → 57337.766 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.422R | **Realized PnL (₹):** +60.12 ₹
**Option Deployed (₹):** 10,745.25 ₹ | **Capital Efficiency:** +0.56%
**Stop Loss:** 57337.766 (Initial: 57309.816) | **Take Profit:** 57442.71 (Initial: 57442.71)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300CE` @ premium of `₹716.35` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle.
**SL/TP Placement Logic:** Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. Take Profit was placed at the nearest opposing liquidity target or FVG imbalance.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-13
### Trade trade_NSE_NIFTY50_INDEX_OrderFlow_v1.0_1786602605
**Strategy/Experiment:** `LIQUIDITY_SWEEP` / `OrderFlow_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24386.45 → 24387.6 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.125R | **Realized PnL (₹):** +10.26 ₹
**Option Deployed (₹):** 3,072.50 ₹ | **Capital Efficiency:** +0.33%
**Stop Loss:** 24408.857 (Initial: 24379.86) | **Take Profit:** 24414.13 (Initial: 24400.95)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹122.9` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle.
**SL/TP Placement Logic:** Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. Take Profit was placed at the nearest opposing liquidity target or FVG imbalance.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `PrevDay_Extremes_RVOL1.2`
Total trades: 1

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTYBANK_INDEX_PrevDay_Extremes_RVOL1.2_1786514105
**Strategy/Experiment:** `REVERSAL` / `PrevDay_Extremes_RVOL1.2`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 57587.55 → 57535.668 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.771R | **Realized PnL (₹):** +365.39 ₹
**Option Deployed (₹):** 6,786.00 ₹ | **Capital Efficiency:** +5.38%
**Stop Loss:** 57535.668 (Initial: 57650.777) | **Take Profit:** 57303.35 (Initial: 57303.35)
**Option Resolved:** `NSE:BANKNIFTY26AUG57600PE` @ premium of `₹452.4` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** This trade was triggered by strategy `REVERSAL` under experiment `PrevDay_Extremes_RVOL1.2` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `Structural_v3.2_RVOL0.8`
Total trades: 8

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786430105
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Underlying Entry → Exit:** 24431.05 → 24440.318 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +1.555R | **Realized PnL (₹):** +449.00 ₹
**Option Deployed (₹):** 4,235.00 ₹ | **Capital Efficiency:** +10.60%
**Stop Loss:** 24440.318 (Initial: 24425.275) | **Take Profit:** 24488.8 (Initial: 24488.8)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹42.35` (Lots: 4.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.43 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786434305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24453.15 → 24462.2 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.792R | **Realized PnL (₹):** -724.50 ₹
**Option Deployed (₹):** 4,732.50 ₹ | **Capital Efficiency:** -15.31%
**Stop Loss:** 24462.2 (Initial: 24465.35) | **Take Profit:** 24427.95 (Initial: 24427.95)
**Option Resolved:** `NSE:NIFTY2681124450PE` @ premium of `₹31.55` (Lots: 6.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.97 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786435205
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Underlying Entry → Exit:** 24455.05 → 24464.291 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +1.293R | **Realized PnL (₹):** +444.86 ₹
**Option Deployed (₹):** 4,860.00 ₹ | **Capital Efficiency:** +9.15%
**Stop Loss:** 24464.291 (Initial: 24448.172) | **Take Profit:** 24523.836 (Initial: 24523.836)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹48.6` (Lots: 4.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.57 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786508405
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24402.55 → 24398.16 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.424R | **Realized PnL (₹):** -62.20 ₹
**Option Deployed (₹):** 3,981.25 ₹ | **Capital Efficiency:** -1.56%
**Stop Loss:** 24398.16 (Initial: 24390.81) | **Take Profit:** 24519.943 (Initial: 24519.943)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹159.25` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.42 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786512305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24359.3 → 24355.45 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -50.53 ₹
**Option Deployed (₹):** 4,131.25 ₹ | **Capital Efficiency:** -1.22%
**Stop Loss:** 24355.45 (Initial: 24355.45) | **Take Profit:** 24450.354 (Initial: 24450.354)
**Option Resolved:** `NSE:NIFTY2681824350CE` @ premium of `₹165.25` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 3.15 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786515305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24294.2 → 24277.25 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -222.47 ₹
**Option Deployed (₹):** 4,125.00 ₹ | **Capital Efficiency:** -5.39%
**Stop Loss:** 24277.25 (Initial: 24277.25) | **Take Profit:** 24391.95 (Initial: 24391.95)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹165.0` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.38 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_Structural_v3.2_RVOL0.8_1786518005
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 57516.5 → 57506.0 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.331R | **Realized PnL (₹):** -92.78 ₹
**Option Deployed (₹):** 10,265.25 ₹ | **Capital Efficiency:** -0.90%
**Stop Loss:** 57506.0 (Initial: 57479.1) | **Take Profit:** 57758.16 (Initial: 57758.16)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500CE` @ premium of `₹684.35` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.57 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-13
### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786597805
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 24327.1 → 24330.9 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.336R | **Realized PnL (₹):** +41.34 ₹
**Option Deployed (₹):** 3,002.50 ₹ | **Capital Efficiency:** +1.38%
**Stop Loss:** 24330.9 (Initial: 24317.25) | **Take Profit:** 24408.367 (Initial: 24408.367)
**Option Resolved:** `NSE:NIFTY2681824350CE` @ premium of `₹120.1` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.97 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `Structural_v3.2_RVOL1.0`
Total trades: 6

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786430105
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Underlying Entry → Exit:** 24431.05 → 24440.318 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +1.555R | **Realized PnL (₹):** +449.00 ₹
**Option Deployed (₹):** 4,235.00 ₹ | **Capital Efficiency:** +10.60%
**Stop Loss:** 24440.318 (Initial: 24425.275) | **Take Profit:** 24488.8 (Initial: 24488.8)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹42.35` (Lots: 4.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.43 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786435205
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Underlying Entry → Exit:** 24455.05 → 24464.291 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +1.293R | **Realized PnL (₹):** +444.86 ₹
**Option Deployed (₹):** 4,860.00 ₹ | **Capital Efficiency:** +9.15%
**Stop Loss:** 24464.291 (Initial: 24448.172) | **Take Profit:** 24523.836 (Initial: 24523.836)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹48.6` (Lots: 4.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.57 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-12
### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786508405
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24402.55 → 24398.16 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.424R | **Realized PnL (₹):** -62.20 ₹
**Option Deployed (₹):** 3,981.25 ₹ | **Capital Efficiency:** -1.56%
**Stop Loss:** 24398.16 (Initial: 24390.81) | **Take Profit:** 24519.943 (Initial: 24519.943)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹159.25` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.42 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786512305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24359.3 → 24355.45 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -50.53 ₹
**Option Deployed (₹):** 4,131.25 ₹ | **Capital Efficiency:** -1.22%
**Stop Loss:** 24355.45 (Initial: 24355.45) | **Take Profit:** 24450.354 (Initial: 24450.354)
**Option Resolved:** `NSE:NIFTY2681824350CE` @ premium of `₹165.25` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 3.15 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786515305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 24294.2 → 24277.25 | **Exit Reason:** `INITIAL_SL`
**R-Multiple PnL (Index points-based):** -1.050R | **Realized PnL (₹):** -222.47 ₹
**Option Deployed (₹):** 4,125.00 ₹ | **Capital Efficiency:** -5.39%
**Stop Loss:** 24277.25 (Initial: 24277.25) | **Take Profit:** 24391.95 (Initial: 24391.95)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹165.0` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.38 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_Structural_v3.2_RVOL1.0_1786518005
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (BREAKOUT)
**Underlying Entry → Exit:** 57516.5 → 57506.0 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.331R | **Realized PnL (₹):** -92.78 ₹
**Option Deployed (₹):** 10,265.25 ₹ | **Capital Efficiency:** -0.90%
**Stop Loss:** 57506.0 (Initial: 57479.1) | **Take Profit:** 57758.16 (Initial: 57758.16)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500CE` @ premium of `₹684.35` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.57 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


## 🧪 Experiment: `VWAP_Reclaim_v1.0`
Total trades: 2

### 📅 Date: 2026-08-11
### Trade trade_NSE_NIFTYBANK_INDEX_VWAP_Reclaim_v1.0_1786425905
**Strategy/Experiment:** `VWAP_RECLAIM` / `VWAP_Reclaim_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (NONE)
**Underlying Entry → Exit:** 57301.8 → 57356.03 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** +0.799R | **Realized PnL (₹):** +382.79 ₹
**Option Deployed (₹):** 10,495.50 ₹ | **Capital Efficiency:** +3.65%
**Stop Loss:** 57356.03 (Initial: 57237.91) | **Take Profit:** 57566.45 (Initial: 57566.45)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300CE` @ premium of `₹699.7` (Lots: 1.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 15 * Lots).*
**Why it was triggered:** The `VwapReclaimStrategy` triggered on a trend-continuation crossover. The 5m close crossed over the intraday VWAP line, clearing it by an ATR-scaled buffer to confirm momentum in the reclaim direction (continuation, not reversion).
**SL/TP Placement Logic:** Stop Loss was set at `low/high - 0.15 * ATR`, floored at `0.5 * ATR` from entry. Take Profit was placed at the next opposing zone, floored at `2.0 * R` to ensure positive risk-reward.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### 📅 Date: 2026-08-13
### Trade trade_NSE_NIFTY50_INDEX_VWAP_Reclaim_v1.0_1786607405
**Strategy/Experiment:** `VWAP_RECLAIM` / `VWAP_Reclaim_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Underlying Entry → Exit:** 24355.25 → 24370.7 | **Exit Reason:** `TRAILING_SL`
**R-Multiple PnL (Index points-based):** -0.839R | **Realized PnL (₹):** -410.73 ₹
**Option Deployed (₹):** 3,502.50 ₹ | **Capital Efficiency:** -11.73%
**Stop Loss:** 24369.43 (Initial: 24374.83) | **Take Profit:** 24287.65 (Initial: 24287.65)
**Option Resolved:** `NSE:NIFTY2681824350PE` @ premium of `₹70.05` (Lots: 2.0)
*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of 25 * Lots).*
**Why it was triggered:** The `VwapReclaimStrategy` triggered on a trend-continuation crossover. The 5m close crossed over the intraday VWAP line, clearing it by an ATR-scaled buffer to confirm momentum in the reclaim direction (continuation, not reversion).
**SL/TP Placement Logic:** Stop Loss was set at `low/high - 0.15 * ATR`, floored at `0.5 * ATR` from entry. Take Profit was placed at the next opposing zone, floored at `2.0 * R` to ensure positive risk-reward.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---


# Options Combination Spread Strategy Ledgers

## 🧪 Experiment: `Butterfly_v1.0`
Total trades: 18

### 📅 Date: 2026-08-13
### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57567.55_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → None | **Exit Reason:** `None`
**R-Multiple PnL (Premium-based):** -0.682R | **Realized PnL (₹):** -67.52 ₹
**Risk Capital Deployed (₹):** 99.00 ₹ | **Capital Efficiency:** -68.20%
**Max Loss (Premium points):** 6.6 pts | **Max Profit:** ₹193.4
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BUTTERFLY_SPREAD_24324.35_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → None | **Exit Reason:** `None`
**R-Multiple PnL (Premium-based):** -0.073R | **Realized PnL (₹):** -22.45 ₹
**Risk Capital Deployed (₹):** 307.50 ₹ | **Capital Efficiency:** -7.30%
**Max Loss (Premium points):** 12.3 pts | **Max Profit:** ₹87.7
**Legs Structure:** `BUY CE 24200.0 | SELL CE 24300.0 | SELL CE 24300.0 | BUY CE 24400.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BUTTERFLY_SPREAD_24324.35_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.073R | **Realized PnL (₹):** -22.45 ₹
**Risk Capital Deployed (₹):** 307.50 ₹ | **Capital Efficiency:** -7.30%
**Max Loss (Premium points):** 12.3 pts | **Max Profit:** ₹87.7
**Legs Structure:** `BUY CE 24200.0 | SELL CE 24300.0 | SELL CE 24300.0 | BUY CE 24400.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57567.55_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57669.95 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.682R | **Realized PnL (₹):** -67.52 ₹
**Risk Capital Deployed (₹):** 99.00 ₹ | **Capital Efficiency:** -68.20%
**Max Loss (Premium points):** 6.6 pts | **Max Profit:** ₹193.4
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.68R realized).

---

### Trade cand_NSE_NIFTY50_INDEX_BUTTERFLY_SPREAD_24339.65_20260813_094005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24339.65 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.099R | **Realized PnL (₹):** -37.50 ₹
**Risk Capital Deployed (₹):** 378.75 ₹ | **Capital Efficiency:** -9.90%
**Max Loss (Premium points):** 15.15 pts | **Max Profit:** ₹84.85
**Legs Structure:** `BUY CE 24250.0 | SELL CE 24350.0 | SELL CE 24350.0 | BUY CE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57668.75_20260813_100505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57668.75 → 57611.15 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.508R | **Realized PnL (₹):** -90.68 ₹
**Risk Capital Deployed (₹):** 178.50 ₹ | **Capital Efficiency:** -50.80%
**Max Loss (Premium points):** 11.9 pts | **Max Profit:** ₹188.1
**Legs Structure:** `BUY CE 57500.0 | SELL CE 57700.0 | SELL CE 57700.0 | BUY CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.51R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57586.55_20260813_102005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57586.55 → 57594.85 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.500R | **Realized PnL (₹):** -57.75 ₹
**Risk Capital Deployed (₹):** 115.50 ₹ | **Capital Efficiency:** -50.00%
**Max Loss (Premium points):** 7.7 pts | **Max Profit:** ₹192.3
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.50R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57629.90_20260813_105005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57629.9 → 57638.3 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.685R | **Realized PnL (₹):** -123.81 ₹
**Risk Capital Deployed (₹):** 180.75 ₹ | **Capital Efficiency:** -68.50%
**Max Loss (Premium points):** 12.05 pts | **Max Profit:** ₹187.95
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.69R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57645.60_20260813_105505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57645.6 → 57637.3 | **Exit Reason:** `TARGET_R`
**R-Multiple PnL (Premium-based):** +1.624R | **Realized PnL (₹):** +132.76 ₹
**Risk Capital Deployed (₹):** 81.75 ₹ | **Capital Efficiency:** +162.40%
**Max Loss (Premium points):** 5.45 pts | **Max Profit:** ₹194.55
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target or stop R multiple (+1.62R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57634.20_20260813_110005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57634.2 → 57598.5 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.551R | **Realized PnL (₹):** -105.79 ₹
**Risk Capital Deployed (₹):** 192.00 ₹ | **Capital Efficiency:** -55.10%
**Max Loss (Premium points):** 12.8 pts | **Max Profit:** ₹187.2
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.55R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57595.45_20260813_111005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57595.45 → 57573.95 | **Exit Reason:** `TARGET_R`
**R-Multiple PnL (Premium-based):** +1.774R | **Realized PnL (₹):** +153.01 ₹
**Risk Capital Deployed (₹):** 86.25 ₹ | **Capital Efficiency:** +177.40%
**Max Loss (Premium points):** 5.75 pts | **Max Profit:** ₹194.25
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target or stop R multiple (+1.77R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57576.00_20260813_112505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57576.0 → 57674.5 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.657R | **Realized PnL (₹):** -102.00 ₹
**Risk Capital Deployed (₹):** 155.25 ₹ | **Capital Efficiency:** -65.70%
**Max Loss (Premium points):** 10.35 pts | **Max Profit:** ₹189.65
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.66R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57658.80_20260813_121505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57658.8 → 57620.05 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.515R | **Realized PnL (₹):** -89.22 ₹
**Risk Capital Deployed (₹):** 173.25 ₹ | **Capital Efficiency:** -51.50%
**Max Loss (Premium points):** 11.55 pts | **Max Profit:** ₹188.45
**Legs Structure:** `BUY CE 57500.0 | SELL CE 57700.0 | SELL CE 57700.0 | BUY CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.52R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57622.85_20260813_122005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57622.85 → 57615.25 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -1.078R | **Realized PnL (₹):** -185.96 ₹
**Risk Capital Deployed (₹):** 172.50 ₹ | **Capital Efficiency:** -107.80%
**Max Loss (Premium points):** 11.5 pts | **Max Profit:** ₹188.5
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-1.08R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57599.70_20260813_130505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57599.7 → 57610.9 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.511R | **Realized PnL (₹):** -142.57 ₹
**Risk Capital Deployed (₹):** 279.00 ₹ | **Capital Efficiency:** -51.10%
**Max Loss (Premium points):** 18.6 pts | **Max Profit:** ₹181.4
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.51R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57617.90_20260813_131005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57617.9 → 57636.25 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.593R | **Realized PnL (₹):** -76.50 ₹
**Risk Capital Deployed (₹):** 129.00 ₹ | **Capital Efficiency:** -59.30%
**Max Loss (Premium points):** 8.6 pts | **Max Profit:** ₹191.4
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.59R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57616.10_20260813_135005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57616.1 → 57613.45 | **Exit Reason:** `TARGET_R`
**R-Multiple PnL (Premium-based):** +1.910R | **Realized PnL (₹):** +191.95 ₹
**Risk Capital Deployed (₹):** 100.50 ₹ | **Capital Efficiency:** +191.00%
**Max Loss (Premium points):** 6.7 pts | **Max Profit:** ₹193.3
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target or stop R multiple (+1.91R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57618.30_20260813_135505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57618.3 → 57583.75 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.577R | **Realized PnL (₹):** -81.79 ₹
**Risk Capital Deployed (₹):** 141.75 ₹ | **Capital Efficiency:** -57.70%
**Max Loss (Premium points):** 9.45 pts | **Max Profit:** ₹190.55
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0 | SELL CE 57600.0 | BUY CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.58R realized).

---


## 🧪 Experiment: `CreditSpread_v1.0_PCRFade`
Total trades: 5

### 📅 Date: 2026-08-11
### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24481.75_20260811_094005_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24481.75 → 24450.25 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.358R | **Realized PnL (₹):** +658.72 ₹
**Risk Capital Deployed (₹):** 1,840.00 ₹ | **Capital Efficiency:** +35.80%
**Max Loss (Premium points):** 73.6 pts | **Max Profit:** ₹26.4
**Legs Structure:** `SELL CE 24550.0 | BUY CE 24650.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-12
### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24407.15_20260812_092005_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24407.15 → 24407.15 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -63.50 ₹
**Risk Capital Deployed (₹):** 1,270.00 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 50.8 pts | **Max Profit:** ₹49.2
**Legs Structure:** `SELL CE 24450.0 | BUY CE 24550.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-13
### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24324.35_20260813_093505_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → None | **Exit Reason:** `None`
**R-Multiple PnL (Premium-based):** -0.215R | **Realized PnL (₹):** -292.94 ₹
**Risk Capital Deployed (₹):** 1,362.50 ₹ | **Capital Efficiency:** -21.50%
**Max Loss (Premium points):** 54.5 pts | **Max Profit:** ₹45.5
**Legs Structure:** `SELL CE 24350.0 | BUY CE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24324.35_20260813_093505_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.215R | **Realized PnL (₹):** -292.94 ₹
**Risk Capital Deployed (₹):** 1,362.50 ₹ | **Capital Efficiency:** -21.50%
**Max Loss (Premium points):** 54.5 pts | **Max Profit:** ₹45.5
**Legs Structure:** `SELL CE 24350.0 | BUY CE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24339.65_20260813_094005_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24339.65 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.118R | **Realized PnL (₹):** -171.25 ₹
**Risk Capital Deployed (₹):** 1,451.25 ₹ | **Capital Efficiency:** -11.80%
**Max Loss (Premium points):** 58.05 pts | **Max Profit:** ₹41.95
**Legs Structure:** `SELL CE 24400.0 | BUY CE 24500.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---


## 🧪 Experiment: `IronCondor_v1.0`
Total trades: 7

### 📅 Date: 2026-08-13
### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57567.55_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57604.1 | **Exit Reason:** `TARGET_R`
**R-Multiple PnL (Premium-based):** +0.511R | **Realized PnL (₹):** +158.28 ₹
**Risk Capital Deployed (₹):** 309.75 ₹ | **Capital Efficiency:** +51.10%
**Max Loss (Premium points):** 20.65 pts | **Max Profit:** ₹179.35
**Legs Structure:** `SELL PE 57500.0 | BUY PE 57300.0 | SELL CE 57700.0 | BUY CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target or stop R multiple (+0.51R realized).

---

### Trade cand_NSE_NIFTY50_INDEX_IRON_CONDOR_24324.35_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.000R | **Realized PnL (₹):** +0.00 ₹
**Risk Capital Deployed (₹):** 667.50 ₹ | **Capital Efficiency:** +0.00%
**Max Loss (Premium points):** 26.7 pts | **Max Profit:** ₹73.3
**Legs Structure:** `SELL PE 24250.0 | BUY PE 24150.0 | SELL CE 24350.0 | BUY CE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_IRON_CONDOR_24324.35_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → None | **Exit Reason:** `None`
**R-Multiple PnL (Premium-based):** +0.000R | **Realized PnL (₹):** +0.00 ₹
**Risk Capital Deployed (₹):** 667.50 ₹ | **Capital Efficiency:** +0.00%
**Max Loss (Premium points):** 26.7 pts | **Max Profit:** ₹73.3
**Legs Structure:** `SELL PE 24250.0 | BUY PE 24150.0 | SELL CE 24350.0 | BUY CE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57567.55_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57604.1 | **Exit Reason:** `TARGET_R`
**R-Multiple PnL (Premium-based):** +0.511R | **Realized PnL (₹):** +158.28 ₹
**Risk Capital Deployed (₹):** 309.75 ₹ | **Capital Efficiency:** +51.10%
**Max Loss (Premium points):** 20.65 pts | **Max Profit:** ₹179.35
**Legs Structure:** `SELL PE 57500.0 | BUY PE 57300.0 | SELL CE 57700.0 | BUY CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target or stop R multiple (+0.51R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57604.10_20260813_094005_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57604.1 → None | **Exit Reason:** `None`
**R-Multiple PnL (Premium-based):** -0.157R | **Realized PnL (₹):** -73.48 ₹
**Risk Capital Deployed (₹):** 468.00 ₹ | **Capital Efficiency:** -15.70%
**Max Loss (Premium points):** 31.2 pts | **Max Profit:** ₹168.8
**Legs Structure:** `SELL PE 57500.0 | BUY PE 57300.0 | SELL CE 57700.0 | BUY CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57604.10_20260813_094005_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57604.1 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.157R | **Realized PnL (₹):** -73.48 ₹
**Risk Capital Deployed (₹):** 468.00 ₹ | **Capital Efficiency:** -15.70%
**Max Loss (Premium points):** 31.2 pts | **Max Profit:** ₹168.8
**Legs Structure:** `SELL PE 57500.0 | BUY PE 57300.0 | SELL CE 57700.0 | BUY CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_IRON_CONDOR_24339.65_20260813_094005_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 24339.65 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.132R | **Realized PnL (₹):** +86.13 ₹
**Risk Capital Deployed (₹):** 652.50 ₹ | **Capital Efficiency:** +13.20%
**Max Loss (Premium points):** 26.1 pts | **Max Profit:** ₹73.9
**Legs Structure:** `SELL PE 24300.0 | BUY PE 24200.0 | SELL CE 24400.0 | BUY CE 24500.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---


## 🧪 Experiment: `Straddle_v1.0_VolCompression`
Total trades: 8

### 📅 Date: 2026-08-11
### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24469.00_20260811_110005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24469.0 → 24450.25 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.686R | **Realized PnL (₹):** -1,865.06 ₹
**Risk Capital Deployed (₹):** 2,718.75 ₹ | **Capital Efficiency:** -68.60%
**Max Loss (Premium points):** 108.75 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24450.0 | BUY PE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.69R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57258.00_20260811_120505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57258.0 → 57366.05 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.036R | **Realized PnL (₹):** -648.70 ₹
**Risk Capital Deployed (₹):** 18,019.50 ₹ | **Capital Efficiency:** -3.60%
**Max Loss (Premium points):** 1201.3 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57300.0 | BUY PE 57300.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-12
### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24450.25_20260812_090005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24450.25 → 24450.25 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -356.94 ₹
**Risk Capital Deployed (₹):** 7,138.75 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 285.55 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24450.0 | BUY PE 24450.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57366.05_20260812_090005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57366.05 → 57366.05 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -851.06 ₹
**Risk Capital Deployed (₹):** 17,021.25 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 1134.75 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57400.0 | BUY PE 57400.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-13
### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24324.35_20260813_093505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.020R | **Realized PnL (₹):** -119.58 ₹
**Risk Capital Deployed (₹):** 5,978.75 ₹ | **Capital Efficiency:** -2.00%
**Max Loss (Premium points):** 239.15 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24300.0 | BUY PE 24300.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57567.55_20260813_093505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -795.75 ₹
**Risk Capital Deployed (₹):** 15,915.00 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 1061.0 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57600.0 | BUY PE 57600.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57570.70_20260813_112005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57570.7 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.021R | **Realized PnL (₹):** -324.32 ₹
**Risk Capital Deployed (₹):** 15,444.00 ₹ | **Capital Efficiency:** -2.10%
**Max Loss (Premium points):** 1029.6 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57600.0 | BUY PE 57600.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24327.25_20260813_112505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24327.25 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.005R | **Realized PnL (₹):** +26.81 ₹
**Risk Capital Deployed (₹):** 5,361.25 ₹ | **Capital Efficiency:** +0.50%
**Max Loss (Premium points):** 214.45 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24350.0 | BUY PE 24350.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---


## 🧪 Experiment: `Strangle_v1.0_VolCompression`
Total trades: 8

### 📅 Date: 2026-08-11
### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24469.00_20260811_110005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24469.0 → 24453.15 | **Exit Reason:** `STOP_R`
**R-Multiple PnL (Premium-based):** -0.501R | **Realized PnL (₹):** -454.66 ₹
**Risk Capital Deployed (₹):** 907.50 ₹ | **Capital Efficiency:** -50.10%
**Max Loss (Premium points):** 36.3 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24550.0 | BUY PE 24350.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. Price reached the target or stop R multiple (-0.50R realized).

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57258.00_20260811_120505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57258.0 → 57366.05 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.042R | **Realized PnL (₹):** -636.21 ₹
**Risk Capital Deployed (₹):** 15,147.75 ₹ | **Capital Efficiency:** -4.20%
**Max Loss (Premium points):** 1009.85 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57500.0 | BUY PE 57100.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-12
### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57366.05_20260812_090005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57366.05 → 57366.05 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -712.50 ₹
**Risk Capital Deployed (₹):** 14,250.00 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 950.0 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57600.0 | BUY PE 57200.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24450.25_20260812_090005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24450.25 → 24450.25 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -245.56 ₹
**Risk Capital Deployed (₹):** 4,911.25 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 196.45 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24550.0 | BUY PE 24350.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-13
### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24324.35_20260813_093505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.028R | **Realized PnL (₹):** -106.33 ₹
**Risk Capital Deployed (₹):** 3,797.50 ₹ | **Capital Efficiency:** -2.80%
**Max Loss (Premium points):** 151.9 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24400.0 | BUY PE 24200.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57567.55_20260813_093505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.054R | **Realized PnL (₹):** -703.85 ₹
**Risk Capital Deployed (₹):** 13,034.25 ₹ | **Capital Efficiency:** -5.40%
**Max Loss (Premium points):** 868.95 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57800.0 | BUY PE 57400.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57570.70_20260813_112005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57570.7 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.021R | **Realized PnL (₹):** -264.49 ₹
**Risk Capital Deployed (₹):** 12,594.75 ₹ | **Capital Efficiency:** -2.10%
**Max Loss (Premium points):** 839.65 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 57800.0 | BUY PE 57400.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24327.25_20260813_112505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24327.25 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.008R | **Realized PnL (₹):** +25.96 ₹
**Risk Capital Deployed (₹):** 3,245.00 ₹ | **Capital Efficiency:** +0.80%
**Max Loss (Premium points):** 129.8 pts | **Max Profit:** Unlimited
**Legs Structure:** `BUY CE 24450.0 | BUY PE 24250.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---


## 🧪 Experiment: `VerticalSpread_v1.0`
Total trades: 7

### 📅 Date: 2026-08-11
### Trade cand_NSE_NIFTYBANK_INDEX_BEAR_PUT_SPREAD_57218.40_20260811_094505_VerticalSpread_v1.0
**Strategy/Experiment:** `BEAR_PUT_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BEAR_PUT_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57218.4 → 57366.05 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.171R | **Realized PnL (₹):** -210.33 ₹
**Risk Capital Deployed (₹):** 1,230.00 ₹ | **Capital Efficiency:** -17.10%
**Max Loss (Premium points):** 82.0 pts | **Max Profit:** ₹118.0
**Legs Structure:** `BUY PE 57200.0 | SELL PE 57000.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_PUT_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-12
### Trade cand_NSE_NIFTYBANK_INDEX_BULL_CALL_SPREAD_57446.25_20260812_091505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57446.25 → 57446.25 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -80.59 ₹
**Risk Capital Deployed (₹):** 1,611.75 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 107.45 pts | **Max Profit:** ₹92.55
**Legs Structure:** `BUY CE 57400.0 | SELL CE 57600.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_PUT_SPREAD_24300.30_20260812_110505_VerticalSpread_v1.0
**Strategy/Experiment:** `BEAR_PUT_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_PUT_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24300.3 → 24300.3 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.050R | **Realized PnL (₹):** -44.19 ₹
**Risk Capital Deployed (₹):** 883.75 ₹ | **Capital Efficiency:** -5.00%
**Max Loss (Premium points):** 35.35 pts | **Max Profit:** ₹64.65
**Legs Structure:** `BUY PE 24300.0 | SELL PE 24200.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BEAR_PUT_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### 📅 Date: 2026-08-13
### Trade cand_NSE_NIFTY50_INDEX_BULL_CALL_SPREAD_24324.35_20260813_093505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.215R | **Realized PnL (₹):** +287.56 ₹
**Risk Capital Deployed (₹):** 1,337.50 ₹ | **Capital Efficiency:** +21.50%
**Max Loss (Premium points):** 53.5 pts | **Max Profit:** ₹46.5
**Legs Structure:** `BUY CE 24300.0 | SELL CE 24400.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BULL_CALL_SPREAD_57567.55_20260813_093505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.029R | **Realized PnL (₹):** +47.31 ₹
**Risk Capital Deployed (₹):** 1,631.25 ₹ | **Capital Efficiency:** +2.90%
**Max Loss (Premium points):** 108.75 pts | **Max Profit:** ₹91.25
**Legs Structure:** `BUY CE 57600.0 | SELL CE 57800.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BULL_CALL_SPREAD_24386.45_20260813_120005_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24386.45 → 24353.35 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** +0.018R | **Realized PnL (₹):** +21.58 ₹
**Risk Capital Deployed (₹):** 1,198.75 ₹ | **Capital Efficiency:** +1.80%
**Max Loss (Premium points):** 47.95 pts | **Max Profit:** ₹52.05
**Legs Structure:** `BUY CE 24400.0 | SELL CE 24500.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 25).*
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BULL_CALL_SPREAD_57688.95_20260813_120505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57688.95 → 57591.0 | **Exit Reason:** `SESSION_END`
**R-Multiple PnL (Premium-based):** -0.033R | **Realized PnL (₹):** -54.30 ₹
**Risk Capital Deployed (₹):** 1,645.50 ₹ | **Capital Efficiency:** -3.30%
**Max Loss (Premium points):** 109.7 pts | **Max Profit:** ₹90.3
**Legs Structure:** `BUY CE 57700.0 | SELL CE 57900.0`
*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of 15).*
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

