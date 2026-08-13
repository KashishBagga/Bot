# 📈 Weekly Trades Reasoning & Triggers Report

This report lists all trades executed during the week of August 10, 2026, and provides a detailed structural explanation of the triggers, option strikes selected, stop loss rules, and exit behaviors.

## 📅 Date: 2026-08-11
Total trades executed on this day: 20

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score50_1786423205
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score50`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24479.55 → 24477.43 (PnL: +0.17R)
**Stop Loss:** 24477.43 (Initial: 24489.38) | **Take Profit:** 24448.23 (Initial: 24448.23)
**Option Resolved:** `NSE:NIFTY2681124500PE` @ premium of `₹66.15`
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the TRENDLINE RESISTANCE @ 24479.64 (±0.0pts, score=52). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786423205
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24479.55 → 24477.43 (PnL: +0.17R)
**Stop Loss:** 24477.43 (Initial: 24489.38) | **Take Profit:** 24448.23 (Initial: 24448.23)
**Option Resolved:** `NSE:NIFTY2681124500PE` @ premium of `₹66.15`
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the TRENDLINE RESISTANCE @ 24479.64 (±0.0pts, score=52). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786424405
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24479.0 → 24472.84 (PnL: -0.83R)
**Stop Loss:** 24472.84 (Initial: 24471.14) | **Take Profit:** 24500.0 (Initial: 24500.0)
**Option Resolved:** `NSE:NIFTY2681124500CE` @ premium of `₹52.45`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_VWAP_Reclaim_v1.0_1786425905
**Strategy/Experiment:** `VWAP_RECLAIM` / `VWAP_Reclaim_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (NONE)
**Entry → Exit:** 57301.8 → 57356.03 (PnL: +0.80R)
**Stop Loss:** 57356.03 (Initial: 57237.91) | **Take Profit:** 57566.45 (Initial: 57566.45)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300CE` @ premium of `₹699.7`
**Why it was triggered:** The `VwapReclaimStrategy` triggered on a trend-continuation crossover. The 5m close crossed over the intraday VWAP line, clearing it by an ATR-scaled buffer to confirm momentum in the reclaim direction (continuation, not reversion).
**SL/TP Placement Logic:** Stop Loss was set at `low/high - 0.15 * ATR`, floored at `0.5 * ATR` from entry. Take Profit was placed at the next opposing zone, floored at `2.0 * R` to ensure positive risk-reward.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786425905
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24476.7 → 24469.805 (PnL: -1.05R)
**Stop Loss:** 24469.805 (Initial: 24469.805) | **Take Profit:** 24500.0 (Initial: 24500.0)
**Option Resolved:** `NSE:NIFTY2681124500CE` @ premium of `₹45.9`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786430105
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Entry → Exit:** 24431.05 → 24440.318 (PnL: +1.55R)
**Stop Loss:** 24440.318 (Initial: 24425.275) | **Take Profit:** 24488.8 (Initial: 24488.8)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹42.35`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.43 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786430105
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Entry → Exit:** 24431.05 → 24440.318 (PnL: +1.55R)
**Stop Loss:** 24440.318 (Initial: 24425.275) | **Take Profit:** 24488.8 (Initial: 24488.8)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹42.35`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.43 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OrderFlow_v1.0_1786431305
**Strategy/Experiment:** `LIQUIDITY_SWEEP` / `OrderFlow_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24431.85 → 24437.564 (PnL: -1.05R)
**Stop Loss:** 24437.564 (Initial: 24437.564) | **Take Profit:** 24397.56 (Initial: 24397.56)
**Option Resolved:** `NSE:NIFTY2681124450PE` @ premium of `₹43.9`
**Why it was triggered:** The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle.
**SL/TP Placement Logic:** Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. Take Profit was placed at the nearest opposing liquidity target or FVG imbalance.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_OrderFlow_v1.0_1786432505
**Strategy/Experiment:** `LIQUIDITY_SWEEP` / `OrderFlow_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 57328.8 → 57337.766 (PnL: +0.42R)
**Stop Loss:** 57337.766 (Initial: 57309.816) | **Take Profit:** 57442.71 (Initial: 57442.71)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300CE` @ premium of `₹716.35`
**Why it was triggered:** The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle.
**SL/TP Placement Logic:** Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. Take Profit was placed at the nearest opposing liquidity target or FVG imbalance.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_ATR_Squeeze_RVOL1.0_1786432505
**Strategy/Experiment:** `SQUEEZE_BREAKOUT` / `ATR_Squeeze_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 57328.8 → 57347.785 (PnL: -1.05R)
**Stop Loss:** 57347.785 (Initial: 57347.785) | **Take Profit:** 57214.89 (Initial: 57214.89)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300PE` @ premium of `₹482.5`
**Why it was triggered:** This trade was triggered by strategy `SQUEEZE_BREAKOUT` under experiment `ATR_Squeeze_RVOL1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_Geometry_v1.0_Score35_1786433405
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 57327.9 → 57347.99 (PnL: -1.05R)
**Stop Loss:** 57347.99 (Initial: 57347.99) | **Take Profit:** 57295.76 (Initial: 57295.76)
**Option Resolved:** `NSE:BANKNIFTY26AUG57300PE` @ premium of `₹467.0`
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the RESISTANCE EMA50 @ 57329.44 (±9.1pts, score=50). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.754.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786434305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24453.15 → 24462.2 (PnL: -0.79R)
**Stop Loss:** 24462.2 (Initial: 24465.35) | **Take Profit:** 24427.95 (Initial: 24427.95)
**Option Resolved:** `NSE:NIFTY2681124450PE` @ premium of `₹31.55`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.97 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786435205
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Entry → Exit:** 24455.05 → 24464.291 (PnL: +1.29R)
**Stop Loss:** 24464.291 (Initial: 24448.172) | **Take Profit:** 24523.836 (Initial: 24523.836)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹48.6`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.57 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786435205
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (TRAP)
**Entry → Exit:** 24455.05 → 24464.291 (PnL: +1.29R)
**Stop Loss:** 24464.291 (Initial: 24448.172) | **Take Profit:** 24523.836 (Initial: 24523.836)
**Option Resolved:** `NSE:NIFTY2681124450CE` @ premium of `₹48.6`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.57 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24481.75_20260811_094005_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24481.75 → 24450.25 (PnL: +0.36R)
**Max Loss:** ₹73.6 | **Max Profit:** ₹26.4 | **Net Premium:** -26.4
**Target R:** 0.5R | **Stop R:** -1.0R
**Legs Structure:** SELL CE Strike 24550.0 | BUY CE Strike 24650.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BEAR_PUT_SPREAD_57218.40_20260811_094505_VerticalSpread_v1.0
**Strategy/Experiment:** `BEAR_PUT_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BEAR_PUT_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57218.4 → 57366.05 (PnL: -0.17R)
**Max Loss:** ₹82.0 | **Max Profit:** ₹118.0 | **Net Premium:** 82.0
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY PE Strike 57200.0 | SELL PE Strike 57000.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_PUT_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24469.00_20260811_110005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24469.0 → 24450.25 (PnL: -0.69R)
**Max Loss:** ₹108.75 | **Max Profit:** Unlimited | **Net Premium:** 108.75
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24450.0 | BUY PE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24469.00_20260811_110005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24469.0 → 24453.15 (PnL: -0.50R)
**Max Loss:** ₹36.3 | **Max Profit:** Unlimited | **Net Premium:** 36.3
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24550.0 | BUY PE Strike 24350.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57258.00_20260811_120505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57258.0 → 57366.05 (PnL: -0.04R)
**Max Loss:** ₹1201.3 | **Max Profit:** Unlimited | **Net Premium:** 1201.3
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57300.0 | BUY PE Strike 57300.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57258.00_20260811_120505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57258.0 → 57366.05 (PnL: -0.04R)
**Max Loss:** ₹1009.85 | **Max Profit:** Unlimited | **Net Premium:** 1009.85
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57500.0 | BUY PE Strike 57100.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

## 📅 Date: 2026-08-12
Total trades executed on this day: 27

### Trade trade_NSE_NIFTYBANK_INDEX_EMA_Pullback_20_50_RVOL1.0_1786506305
**Strategy/Experiment:** `PULLBACK` / `EMA_Pullback_20_50_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (NONE)
**Entry → Exit:** 57446.25 → 57356.37 (PnL: -1.05R)
**Stop Loss:** 57356.37 (Initial: 57356.37) | **Take Profit:** 57626.01 (Initial: 57626.01)
**Option Resolved:** `NSE:BANKNIFTY26AUG57400CE` @ premium of `₹649.6`
**Why it was triggered:** The `EmaPullbackStrategy` triggered on a trend-continuation setup. Price pulled back to touch the 20 EMA, and then printed a green/red confirmation body in the direction of the macro EMA trend (bullish/bearish crossover).
**SL/TP Placement Logic:** Stop Loss was set below/above the 50 EMA with a small buffer (`0.2 * ATR`), floored at `0.5 * ATR` from entry. Take Profit was projected to the nearest resistance or fallback R-multiple.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score50_1786506905
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score50`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 24423.0 → 24448.16 (PnL: +2.67R)
**Stop Loss:** 24448.16 (Initial: 24413.76) | **Take Profit:** 24460.45 (Initial: 24451.21)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹174.1`
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the PWL PDL + TRENDLINE VWAP @ 24423.51 (±3.3pts, score=53). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.768.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786506905
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 24423.0 → 24448.16 (PnL: +2.67R)
**Stop Loss:** 24448.16 (Initial: 24413.76) | **Take Profit:** 24460.45 (Initial: 24451.21)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹174.1`
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the PWL PDL + TRENDLINE VWAP @ 24423.51 (±3.3pts, score=53). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.768.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786508405
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24402.55 → 24398.16 (PnL: -0.42R)
**Stop Loss:** 24398.16 (Initial: 24390.81) | **Take Profit:** 24519.943 (Initial: 24519.943)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹159.25`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.42 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786508405
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24402.55 → 24398.16 (PnL: -0.42R)
**Stop Loss:** 24398.16 (Initial: 24390.81) | **Take Profit:** 24519.943 (Initial: 24519.943)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹159.25`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.42 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786509005
**Strategy/Experiment:** `OI_WALL_BREAK` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (NONE)
**Entry → Exit:** 24378.55 → 24387.84 (PnL: -0.42R)
**Stop Loss:** 24387.84 (Initial: 24403.791) | **Take Profit:** 24302.74 (Initial: 24302.74)
**Option Resolved:** `NSE:NIFTY2681824400PE` @ premium of `₹119.9`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_BREAK` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786510805
**Strategy/Experiment:** `CONFLUENCE_BOUNCE` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24371.5 → 24381.95 (PnL: -1.05R)
**Stop Loss:** 24381.95 (Initial: 24381.95) | **Take Profit:** 24350.3 (Initial: 24350.3)
**Option Resolved:** `NSE:NIFTY2681824350PE` @ premium of `₹101.0`
**Why it was triggered:** The system detected a `CONFLUENCE_BOUNCE` setup under the `GeometryStrategy`. Specifically, price hit the TRENDLINE RESISTANCE @ 24371.73 (±0.0pts, score=51). The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.758.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786512305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24359.3 → 24355.45 (PnL: -1.05R)
**Stop Loss:** 24355.45 (Initial: 24355.45) | **Take Profit:** 24450.354 (Initial: 24450.354)
**Option Resolved:** `NSE:NIFTY2681824350CE` @ premium of `₹165.25`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 3.15 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786512305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24359.3 → 24355.45 (PnL: -1.05R)
**Stop Loss:** 24355.45 (Initial: 24355.45) | **Take Profit:** 24450.354 (Initial: 24450.354)
**Option Resolved:** `NSE:NIFTY2681824350CE` @ premium of `₹165.25`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 3.15 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_PrevDay_Extremes_RVOL1.2_1786514105
**Strategy/Experiment:** `REVERSAL` / `PrevDay_Extremes_RVOL1.2`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 57587.55 → 57535.668 (PnL: +0.77R)
**Stop Loss:** 57535.668 (Initial: 57650.777) | **Take Profit:** 57303.35 (Initial: 57303.35)
**Option Resolved:** `NSE:BANKNIFTY26AUG57600PE` @ premium of `₹452.4`
**Why it was triggered:** This trade was triggered by strategy `REVERSAL` under experiment `PrevDay_Extremes_RVOL1.2` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL1.0_1786515305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24294.2 → 24277.25 (PnL: -1.05R)
**Stop Loss:** 24277.25 (Initial: 24277.25) | **Take Profit:** 24391.95 (Initial: 24391.95)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹165.0`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.38 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786515305
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24294.2 → 24277.25 (PnL: -1.05R)
**Stop Loss:** 24277.25 (Initial: 24277.25) | **Take Profit:** 24391.95 (Initial: 24391.95)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹165.0`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.38 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786515905
**Strategy/Experiment:** `OI_WALL_BREAK` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (NONE)
**Entry → Exit:** 24282.1 → 24287.855 (PnL: -0.33R)
**Stop Loss:** 24287.855 (Initial: 24302.855) | **Take Profit:** 24225.01 (Initial: 24225.01)
**Option Resolved:** `NSE:NIFTY2681824300PE` @ premium of `₹113.25`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_BREAK` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786517105
**Strategy/Experiment:** `TRENDLINE_RETEST` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (NONE)
**Entry → Exit:** 24267.1 → 24277.85 (PnL: -1.05R)
**Stop Loss:** 24277.85 (Initial: 24277.85) | **Take Profit:** 24250.0 (Initial: 24250.0)
**Option Resolved:** `NSE:NIFTY2681824250PE` @ premium of `₹98.85`
**Why it was triggered:** The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy`. Specifically, price hit the confluence zone. The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `INITIAL_SL`. Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis.

---

### Trade trade_NSE_NIFTYBANK_INDEX_Structural_v3.2_RVOL0.8_1786518005
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 57516.5 → 57506.0 (PnL: -0.33R)
**Stop Loss:** 57506.0 (Initial: 57479.1) | **Take Profit:** 57758.16 (Initial: 57758.16)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500CE` @ premium of `₹684.35`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.57 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_Structural_v3.2_RVOL1.0_1786518005
**Strategy/Experiment:** `TRAP` / `Structural_v3.2_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 57516.5 → 57506.0 (PnL: -0.33R)
**Stop Loss:** 57506.0 (Initial: 57479.1) | **Take Profit:** 57758.16 (Initial: 57758.16)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500CE` @ premium of `₹684.35`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `TRAP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 1.57 (threshold >= 0.8). Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). Take Profit was set at the opposing zone.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_ATR_Squeeze_RVOL1.0_1786520105
**Strategy/Experiment:** `SQUEEZE_BREAKOUT` / `ATR_Squeeze_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 57517.1 → 57519.645 (PnL: -0.18R)
**Stop Loss:** 57519.645 (Initial: 57536.895) | **Take Profit:** 57319.15 (Initial: 57319.15)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500PE` @ premium of `₹433.1`
**Why it was triggered:** This trade was triggered by strategy `SQUEEZE_BREAKOUT` under experiment `ATR_Squeeze_RVOL1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786520705
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24288.25 → 24290.37 (PnL: +0.32R)
**Stop Loss:** 24290.37 (Initial: 24282.47) | **Take Profit:** 24322.93 (Initial: 24322.93)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹159.4`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTYBANK_INDEX_ATR_Squeeze_RVOL1.0_1786521005
**Strategy/Experiment:** `SQUEEZE_BREAKOUT` / `ATR_Squeeze_RVOL1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 57510.25 → 57526.87 (PnL: -0.92R)
**Stop Loss:** 57526.87 (Initial: 57529.42) | **Take Profit:** 57318.55 (Initial: 57318.55)
**Option Resolved:** `NSE:BANKNIFTY26AUG57500PE` @ premium of `₹430.65`
**Why it was triggered:** This trade was triggered by strategy `SQUEEZE_BREAKOUT` under experiment `ATR_Squeeze_RVOL1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786524005
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (OI_WALL_FADE)
**Entry → Exit:** 24284.35 → 24284.35 (PnL: -0.05R)
**Stop Loss:** 24279.09 (Initial: 24279.09) | **Take Profit:** 24315.91 (Initial: 24315.91)
**Option Resolved:** `NSE:NIFTY2681824300CE` @ premium of `₹155.85`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57366.05_20260812_090005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57366.05 → 57366.05 (PnL: -0.05R)
**Max Loss:** ₹950.0 | **Max Profit:** Unlimited | **Net Premium:** 950.0
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57600.0 | BUY PE Strike 57200.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24450.25_20260812_090005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24450.25 → 24450.25 (PnL: -0.05R)
**Max Loss:** ₹285.55 | **Max Profit:** Unlimited | **Net Premium:** 285.55
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24450.0 | BUY PE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24450.25_20260812_090005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24450.25 → 24450.25 (PnL: -0.05R)
**Max Loss:** ₹196.45 | **Max Profit:** Unlimited | **Net Premium:** 196.45
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24550.0 | BUY PE Strike 24350.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57366.05_20260812_090005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57366.05 → 57366.05 (PnL: -0.05R)
**Max Loss:** ₹1134.75 | **Max Profit:** Unlimited | **Net Premium:** 1134.75
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | BUY PE Strike 57400.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BULL_CALL_SPREAD_57446.25_20260812_091505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57446.25 → 57446.25 (PnL: -0.05R)
**Max Loss:** ₹107.45 | **Max Profit:** ₹92.55 | **Net Premium:** 107.45
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24407.15_20260812_092005_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24407.15 → 24407.15 (PnL: -0.05R)
**Max Loss:** ₹50.8 | **Max Profit:** ₹49.2 | **Net Premium:** -49.2
**Target R:** 0.5R | **Stop R:** -1.0R
**Legs Structure:** SELL CE Strike 24450.0 | BUY CE Strike 24550.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_PUT_SPREAD_24300.30_20260812_110505_VerticalSpread_v1.0
**Strategy/Experiment:** `BEAR_PUT_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_PUT_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24300.3 → 24300.3 (PnL: -0.05R)
**Max Loss:** ₹35.35 | **Max Profit:** ₹64.65 | **Net Premium:** 35.35
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY PE Strike 24300.0 | SELL PE Strike 24200.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_PUT_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

## 📅 Date: 2026-08-13
Total trades executed on this day: 46

### Trade trade_NSE_NIFTY50_INDEX_Structural_v3.2_RVOL0.8_1786597805
**Strategy/Experiment:** `SWEEP` / `Structural_v3.2_RVOL0.8`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 24327.1 → 24330.9 (PnL: +0.34R)
**Stop Loss:** 24330.9 (Initial: 24317.25) | **Take Profit:** 24408.367 (Initial: 24408.367)
**Option Resolved:** `NSE:NIFTY2681824350CE` @ premium of `₹120.1`
**Why it was triggered:** The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` v3.2) triggered a `SWEEP` setup. This occurred under Daily Bias 'NEUTRAL' and Hourly Bias 'NEUTRAL'. The trigger was validated by a Relative Volume (RVOL) of 0.97 (threshold >= 0.8). Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body.
**SL/TP Placement Logic:** Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OrderFlow_v1.0_1786602605
**Strategy/Experiment:** `LIQUIDITY_SWEEP` / `OrderFlow_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (NONE)
**Entry → Exit:** 24386.45 → 24387.6 (PnL: +0.12R)
**Stop Loss:** 24408.857 (Initial: 24379.86) | **Take Profit:** 24414.13 (Initial: 24400.95)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹122.9`
**Why it was triggered:** The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle.
**SL/TP Placement Logic:** Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. Take Profit was placed at the nearest opposing liquidity target or FVG imbalance.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score35_1786603805
**Strategy/Experiment:** `TRENDLINE_RETEST` / `Geometry_v1.0_Score35`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24392.15 → 24400.45 (PnL: -0.96R)
**Stop Loss:** 24398.7 (Initial: 24401.3) | **Take Profit:** 24363.03 (Initial: 24363.03)
**Option Resolved:** `NSE:NIFTY2681824400PE` @ premium of `₹82.85`
**Why it was triggered:** The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy`. Specifically, price hit the confluence zone. The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_Geometry_v1.0_Score50_1786603805
**Strategy/Experiment:** `TRENDLINE_RETEST` / `Geometry_v1.0_Score50`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24392.15 → 24400.45 (PnL: -0.96R)
**Stop Loss:** 24398.7 (Initial: 24401.3) | **Take Profit:** 24363.03 (Initial: 24363.03)
**Option Resolved:** `NSE:NIFTY2681824400PE` @ premium of `₹82.85`
**Why it was triggered:** The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy`. Specifically, price hit the confluence zone. The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. Daily bias was 'CONTINUATION' with confidence 0.6.
**SL/TP Placement Logic:** The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_OIWallReaction_v1.0_1786604105
**Strategy/Experiment:** `OI_WALL_FADE` / `OIWallReaction_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY CALL (BREAKOUT)
**Entry → Exit:** 24399.2 → 24393.8 (PnL: -0.80R)
**Stop Loss:** 24394.695 (Initial: 24391.994) | **Take Profit:** 24442.43 (Initial: 24442.43)
**Option Resolved:** `NSE:NIFTY2681824400CE` @ premium of `₹138.65`
**Why it was triggered:** This trade was triggered by strategy `OI_WALL_FADE` under experiment `OIWallReaction_v1.0` based on default momentum/reversal rules.
**SL/TP Placement Logic:** Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures).
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade trade_NSE_NIFTY50_INDEX_VWAP_Reclaim_v1.0_1786607405
**Strategy/Experiment:** `VWAP_RECLAIM` / `VWAP_Reclaim_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | BUY PUT (BREAKOUT)
**Entry → Exit:** 24355.25 → 24370.7 (PnL: -0.84R)
**Stop Loss:** 24369.43 (Initial: 24374.83) | **Take Profit:** 24287.65 (Initial: 24287.65)
**Option Resolved:** `NSE:NIFTY2681824350PE` @ premium of `₹70.05`
**Why it was triggered:** The `VwapReclaimStrategy` triggered on a trend-continuation crossover. The 5m close crossed over the intraday VWAP line, clearing it by an ATR-scaled buffer to confirm momentum in the reclaim direction (continuation, not reversion).
**SL/TP Placement Logic:** Stop Loss was set at `low/high - 0.15 * ATR`, floored at `0.5 * ATR` from entry. Take Profit was placed at the next opposing zone, floored at `2.0 * R` to ensure positive risk-reward.
**Exit Behavior:** The trade exited due to `TRAILING_SL`. Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24324.35_20260813_093505_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → None (PnL: +0.00R)
**Max Loss:** ₹54.5 | **Max Profit:** ₹45.5 | **Net Premium:** -45.5
**Target R:** 0.5R | **Stop R:** -1.0R
**Legs Structure:** SELL CE Strike 24350.0 | BUY CE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24324.35_20260813_093505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 (PnL: -0.02R)
**Max Loss:** ₹239.15 | **Max Profit:** Unlimited | **Net Premium:** 239.15
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24300.0 | BUY PE Strike 24300.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57567.55_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → None (PnL: +0.00R)
**Max Loss:** ₹6.6 | **Max Profit:** ₹193.4 | **Net Premium:** 6.6
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57567.55_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57604.1 (PnL: +0.51R)
**Max Loss:** ₹20.65 | **Max Profit:** ₹179.35 | **Net Premium:** -179.35
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 57500.0 | BUY PE Strike 57300.0 | SELL CE Strike 57700.0 | BUY CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target profit level (or options combination target R multiple).

---

### Trade cand_NSE_NIFTY50_INDEX_IRON_CONDOR_24324.35_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 (PnL: -0.03R)
**Max Loss:** ₹26.7 | **Max Profit:** ₹73.3 | **Net Premium:** -73.3
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 24250.0 | BUY PE Strike 24150.0 | SELL CE Strike 24350.0 | BUY CE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BUTTERFLY_SPREAD_24324.35_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → None (PnL: +0.00R)
**Max Loss:** ₹12.3 | **Max Profit:** ₹87.7 | **Net Premium:** 12.3
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24200.0 | SELL CE Strike 24300.0 | SELL CE Strike 24300.0 | BUY CE Strike 24400.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24324.35_20260813_093505_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 (PnL: -0.21R)
**Max Loss:** ₹54.5 | **Max Profit:** ₹45.5 | **Net Premium:** -45.5
**Target R:** 0.5R | **Stop R:** -1.0R
**Legs Structure:** SELL CE Strike 24350.0 | BUY CE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_IRON_CONDOR_24324.35_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → None (PnL: +0.00R)
**Max Loss:** ₹26.7 | **Max Profit:** ₹73.3 | **Net Premium:** -73.3
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 24250.0 | BUY PE Strike 24150.0 | SELL CE Strike 24350.0 | BUY CE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BUTTERFLY_SPREAD_24324.35_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 (PnL: -0.07R)
**Max Loss:** ₹12.3 | **Max Profit:** ₹87.7 | **Net Premium:** 12.3
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24200.0 | SELL CE Strike 24300.0 | SELL CE Strike 24300.0 | BUY CE Strike 24400.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57567.55_20260813_093505_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57604.1 (PnL: +0.51R)
**Max Loss:** ₹20.65 | **Max Profit:** ₹179.35 | **Net Premium:** -179.35
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 57500.0 | BUY PE Strike 57300.0 | SELL CE Strike 57700.0 | BUY CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target profit level (or options combination target R multiple).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57567.55_20260813_093505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57669.95 (PnL: -0.68R)
**Max Loss:** ₹6.6 | **Max Profit:** ₹193.4 | **Net Premium:** 6.6
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BULL_CALL_SPREAD_24324.35_20260813_093505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 (PnL: +0.21R)
**Max Loss:** ₹53.5 | **Max Profit:** ₹46.5 | **Net Premium:** 53.5
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY CE Strike 24300.0 | SELL CE Strike 24400.0
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24324.35_20260813_093505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24324.35 → 24353.35 (PnL: -0.03R)
**Max Loss:** ₹151.9 | **Max Profit:** Unlimited | **Net Premium:** 151.9
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24400.0 | BUY PE Strike 24200.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BULL_CALL_SPREAD_57567.55_20260813_093505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57591.0 (PnL: +0.03R)
**Max Loss:** ₹108.75 | **Max Profit:** ₹91.25 | **Net Premium:** 108.75
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY CE Strike 57600.0 | SELL CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57567.55_20260813_093505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57591.0 (PnL: -0.05R)
**Max Loss:** ₹1061.0 | **Max Profit:** Unlimited | **Net Premium:** 1061.0
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57600.0 | BUY PE Strike 57600.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57567.55_20260813_093505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57567.55 → 57591.0 (PnL: -0.05R)
**Max Loss:** ₹868.95 | **Max Profit:** Unlimited | **Net Premium:** 868.95
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57800.0 | BUY PE Strike 57400.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57604.10_20260813_094005_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57604.1 → None (PnL: +0.00R)
**Max Loss:** ₹31.2 | **Max Profit:** ₹168.8 | **Net Premium:** -168.8
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 57500.0 | BUY PE Strike 57300.0 | SELL CE Strike 57700.0 | BUY CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `None`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_IRON_CONDOR_57604.10_20260813_094005_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 57604.1 → 57591.0 (PnL: -0.16R)
**Max Loss:** ₹31.2 | **Max Profit:** ₹168.8 | **Net Premium:** -168.8
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 57500.0 | BUY PE Strike 57300.0 | SELL CE Strike 57700.0 | BUY CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_IRON_CONDOR_24339.65_20260813_094005_IronCondor_v1.0
**Strategy/Experiment:** `IRON_CONDOR` / `IronCondor_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `IRON_CONDOR` (Option Combo Spread)
**Underlying Entry → Exit:** 24339.65 → 24353.35 (PnL: +0.13R)
**Max Loss:** ₹26.1 | **Max Profit:** ₹73.9 | **Net Premium:** -73.9
**Target R:** 0.4R | **Stop R:** -1.0R
**Legs Structure:** SELL PE Strike 24300.0 | BUY PE Strike 24200.0 | SELL CE Strike 24400.0 | BUY CE Strike 24500.0
**Why it was triggered:** This is an options combination spread strategy (`IRON_CONDOR`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.4R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BUTTERFLY_SPREAD_24339.65_20260813_094005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24339.65 → 24353.35 (PnL: -0.10R)
**Max Loss:** ₹15.15 | **Max Profit:** ₹84.85 | **Net Premium:** 15.15
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24250.0 | SELL CE Strike 24350.0 | SELL CE Strike 24350.0 | BUY CE Strike 24450.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_BEAR_CALL_SPREAD_24339.65_20260813_094005_CreditSpread_v1.0_PCRFade
**Strategy/Experiment:** `BEAR_CALL_SPREAD` / `CreditSpread_v1.0_PCRFade`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BEAR_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24339.65 → 24353.35 (PnL: -0.12R)
**Max Loss:** ₹58.05 | **Max Profit:** ₹41.95 | **Net Premium:** -41.95
**Target R:** 0.5R | **Stop R:** -1.0R
**Legs Structure:** SELL CE Strike 24400.0 | BUY CE Strike 24500.0
**Why it was triggered:** This is an options combination spread strategy (`BEAR_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `0.5R` and the stop loss was set at `-1.0R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57668.75_20260813_100505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57668.75 → 57611.15 (PnL: -0.51R)
**Max Loss:** ₹11.9 | **Max Profit:** ₹188.1 | **Net Premium:** 11.9
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57500.0 | SELL CE Strike 57700.0 | SELL CE Strike 57700.0 | BUY CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57586.55_20260813_102005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57586.55 → 57594.85 (PnL: -0.50R)
**Max Loss:** ₹7.7 | **Max Profit:** ₹192.3 | **Net Premium:** 7.7
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57629.90_20260813_105005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57629.9 → 57638.3 (PnL: -0.69R)
**Max Loss:** ₹12.05 | **Max Profit:** ₹187.95 | **Net Premium:** 12.05
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57645.60_20260813_105505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57645.6 → 57637.3 (PnL: +1.62R)
**Max Loss:** ₹5.45 | **Max Profit:** ₹194.55 | **Net Premium:** 5.45
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target profit level (or options combination target R multiple).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57634.20_20260813_110005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57634.2 → 57598.5 (PnL: -0.55R)
**Max Loss:** ₹12.8 | **Max Profit:** ₹187.2 | **Net Premium:** 12.8
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57595.45_20260813_111005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57595.45 → 57573.95 (PnL: +1.77R)
**Max Loss:** ₹5.75 | **Max Profit:** ₹194.25 | **Net Premium:** 5.75
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target profit level (or options combination target R multiple).

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRANGLE_57570.70_20260813_112005_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57570.7 → 57591.0 (PnL: -0.02R)
**Max Loss:** ₹839.65 | **Max Profit:** Unlimited | **Net Premium:** 839.65
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57800.0 | BUY PE Strike 57400.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_LONG_STRADDLE_57570.70_20260813_112005_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 57570.7 → 57591.0 (PnL: -0.02R)
**Max Loss:** ₹1029.6 | **Max Profit:** Unlimited | **Net Premium:** 1029.6
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57600.0 | BUY PE Strike 57600.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRANGLE_24327.25_20260813_112505_Strangle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRANGLE` / `Strangle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRANGLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24327.25 → 24353.35 (PnL: +0.01R)
**Max Loss:** ₹129.8 | **Max Profit:** Unlimited | **Net Premium:** 129.8
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24450.0 | BUY PE Strike 24250.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRANGLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTY50_INDEX_LONG_STRADDLE_24327.25_20260813_112505_Straddle_v1.0_VolCompression
**Strategy/Experiment:** `LONG_STRADDLE` / `Straddle_v1.0_VolCompression`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `LONG_STRADDLE` (Option Combo Spread)
**Underlying Entry → Exit:** 24327.25 → 24353.35 (PnL: +0.01R)
**Max Loss:** ₹214.45 | **Max Profit:** Unlimited | **Net Premium:** 214.45
**Target R:** 1.2R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 24350.0 | BUY PE Strike 24350.0
**Why it was triggered:** This is an options combination spread strategy (`LONG_STRADDLE`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.2R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57576.00_20260813_112505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57576.0 → 57674.5 (PnL: -0.66R)
**Max Loss:** ₹10.35 | **Max Profit:** ₹189.65 | **Net Premium:** 10.35
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTY50_INDEX_BULL_CALL_SPREAD_24386.45_20260813_120005_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTY50-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 24386.45 → 24353.35 (PnL: +0.02R)
**Max Loss:** ₹47.95 | **Max Profit:** ₹52.05 | **Net Premium:** 47.95
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY CE Strike 24400.0 | SELL CE Strike 24500.0
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BULL_CALL_SPREAD_57688.95_20260813_120505_VerticalSpread_v1.0
**Strategy/Experiment:** `BULL_CALL_SPREAD` / `VerticalSpread_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BULL_CALL_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57688.95 → 57591.0 (PnL: -0.03R)
**Max Loss:** ₹109.7 | **Max Profit:** ₹90.3 | **Net Premium:** 109.7
**Target R:** 1.0R | **Stop R:** -0.6R
**Legs Structure:** BUY CE Strike 57700.0 | SELL CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`BULL_CALL_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.0R` and the stop loss was set at `-0.6R` net debit/credit change.
**Exit Behavior:** The trade exited due to `SESSION_END`. The position was closed at the market close (15:25 IST) as a paper-trading session requirement.

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57658.80_20260813_121505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57658.8 → 57620.05 (PnL: -0.52R)
**Max Loss:** ₹11.55 | **Max Profit:** ₹188.45 | **Net Premium:** 11.55
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57500.0 | SELL CE Strike 57700.0 | SELL CE Strike 57700.0 | BUY CE Strike 57900.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57622.85_20260813_122005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57622.85 → 57615.25 (PnL: -1.08R)
**Max Loss:** ₹11.5 | **Max Profit:** ₹188.5 | **Net Premium:** 11.5
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57599.70_20260813_130505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57599.7 → 57610.9 (PnL: -0.51R)
**Max Loss:** ₹18.6 | **Max Profit:** ₹181.4 | **Net Premium:** 18.6
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57617.90_20260813_131005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57617.9 → 57636.25 (PnL: -0.59R)
**Max Loss:** ₹8.6 | **Max Profit:** ₹191.4 | **Net Premium:** 8.6
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57616.10_20260813_135005_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57616.1 → 57613.45 (PnL: +1.91R)
**Max Loss:** ₹6.7 | **Max Profit:** ₹193.3 | **Net Premium:** 6.7
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `TARGET_R`. Price reached the target profit level (or options combination target R multiple).

---

### Trade cand_NSE_NIFTYBANK_INDEX_BUTTERFLY_SPREAD_57618.30_20260813_135505_Butterfly_v1.0
**Strategy/Experiment:** `BUTTERFLY_SPREAD` / `Butterfly_v1.0`
**Symbol & Setup:** NSE:NIFTYBANK-INDEX | `BUTTERFLY_SPREAD` (Option Combo Spread)
**Underlying Entry → Exit:** 57618.3 → 57583.75 (PnL: -0.58R)
**Max Loss:** ₹9.45 | **Max Profit:** ₹190.55 | **Net Premium:** 9.45
**Target R:** 1.5R | **Stop R:** -0.5R
**Legs Structure:** BUY CE Strike 57400.0 | SELL CE Strike 57600.0 | SELL CE Strike 57600.0 | BUY CE Strike 57800.0
**Why it was triggered:** This is an options combination spread strategy (`BUTTERFLY_SPREAD`). It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions (e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction.
**SL/TP Placement Logic:** For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. The target profit was set at `1.5R` and the stop loss was set at `-0.5R` net debit/credit change.
**Exit Behavior:** The trade exited due to `STOP_R`. 

---
