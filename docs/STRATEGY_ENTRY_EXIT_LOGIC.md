# Strategy Entry/Exit Logic Reference

Generated from the actively-registered experiment set in `src/core/experiment_factory.py::build_registry()` (the single source of truth for what runs live/backtest — see `CLAUDE.md`). Retired strategies (ORB 15m/30m, Chart Pattern, Channel) are omitted.

## Shared machinery (applies to almost every strategy below)

**Signal contract** (`src/core/base_strategy.py::StrategyResult`): every strategy's `evaluate()` returns signals carrying fully pre-computed `stop_loss`, `take_profit`, `tp1`, and `rr_ratio` — there is no central risk engine computing these; each strategy computes its own.

**Single-leg exit engine — `_update_position()` in `src/trading/indian_trader.py`** (used by every single-leg strategy, i.e. everything except the 7 multi-leg combo strategies and the scalper):
1. Intrabar SL check first (candle low/high, with gap-fill handling if price gapped through the SL).
2. **TP_EXPANSION**: on hitting `take_profit`, instead of closing, extend `take_profit += stop_loss_distance` and ratchet `stop_loss` up/down to `current_price ∓ stop_loss_distance`, incrementing an expansion counter — unless capped (only `Structural_v3.3_ExitMgmt` caps this, at 3).
3. **TRAILING_SL**: while price makes new highs/lows without hitting TP, trail the stop by the (ATR-adaptive, for the one opt-in experiment) stop distance — 0.75× once `pnl_r >= 1.5R`, else 1.0×.
4. **SESSION_END**: force-exit at 15:25 IST regardless of state.
5. A transaction-cost buffer (`pnl_r -= 0.05`) is applied on index-proxy exits.

**Opt-in "exit management" — currently only `Structural_v3.3_ExitMgmt`** (a shadow-only A/B clone of `Structural_v3.2_RVOL1.0` with identical entries): adds, on top of the above —
- `structure_invalidation`: exit `STRUCTURE_INVALIDATED` if price closes through the swing point (from `StructureEngine`) captured at entry as the invalidation level.
- `atr_adaptive_trailing`: trailing distance becomes Chandelier-style `max(current_ATR×1.5, entry_stop_dist×0.5)`, using live ATR instead of the frozen entry-time distance.
- `tp_expansion_cap: 3`: after 3 expansions, or once the live regime no longer aligns with the trade direction, exit `TP_EXPANSION_CAPPED` instead of expanding further.
- `time_stop_bars: 24`, `time_stop_min_r: 0.3`: force exit `TIME_STOP` if held 24+ bars with `|pnl_r| < 0.3R`.

This exists specifically because `Structural_v3.2` BREAKOUT decayed from +0.8..+2.1R/day to -0.1..-0.5R/day once the market turned choppy — it's being A/B tested against the legacy exit behavior before being promoted to real capital.

**Combo exit engine — `_update_combo_position()`** (used by all 7 multi-leg strategies): re-prices every leg's live premium each candle, `pnl_r = (current_net_value − net_premium_paid) / max_loss`. Exits at `pnl_r >= effective_target_r` (`TARGET_R` — capped at the structure's own `max_profit/max_loss` for bounded-profit combos like credit spreads/condors, so a stated target that's mathematically unreachable can't strand the trade), `pnl_r <= stop_r` (`STOP_R`), or 15:25 `SESSION_END`. There is no zone/ATR logic here at all — combo risk is pure premium-based R.

**Universal "tp1" field**: almost every single-leg strategy also emits `tp1 = entry_price ± 1.5 × risk_distance` alongside the full `take_profit`. This is **not** an enforced partial-exit/scale-out** — it's a stored reference point (used in reporting/attribution), and the actual position is entirely under the shared TP_EXPANSION/trailing engine above. No strategy in this codebase does true multi-target scale-outs (e.g. "sell half at TP1, trail rest") — "multiple targets" in practice means the *dynamic* TP_EXPANSION ratchet, not a fixed TP1/TP2 ladder.

---

## Single-leg directional strategies

### 1. Structural (`StructuralStrategy` → frozen `enhanced_strategy_engine.py`)
**Entry** (5m, gated by Daily→1H bias non-opposition + `RVOL ≥ 1.0` or `0.8`):
- **SWEEP**: price at a demand/supply zone (0.3% tolerance) + strong rejection candle → BUY CALL (demand) / BUY PUT (supply).
- **TRAP**: FFT-detected failed breakout of a 5m BOS level → fade it.
- **BREAKOUT**: 5m BOS with trend, `move_efficiency > 0.6`, `wickiness < 0.5` → trade with the break.

**SL**: SWEEP/TRAP — 1 tick beyond the wick, floored at `0.5×ATR`. BREAKOUT — `0.3×ATR` beyond the broken level, floored at `0.5×ATR`.
**TP**: nearest opposing liquidity zone (not highest-scored, just closest); falls back to 2R if none; capped at `5×ATR`. Rejects if RR < 1.5.
**Multiple targets**: `tp1` at 1.5R; real management is TP_EXPANSION/trailing (base), or the full exit-management stack above for the `v3.3` shadow clone only.

### 2. EMA Pullback
**Entry**: price dips to EMA20 and closes back above it, green candle (mirror for shorts), RVOL≥0.5 (deliberately low — pullbacks are quiet), efficiency≥0.45.
**SL**: below EMA50, `−0.2×ATR` buffer, floored at `0.5×ATR`.
**TP**: next opposing H1 zone, else 2R; capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 3. VWAP Reversion
**Entry**: price stretched `≥1.5×ATR` from VWAP + strong rejection candle → fade back toward VWAP. RVOL≥1.0, efficiency≥0.5.
**SL**: `0.15×ATR` beyond the rejection wick, floored at `0.5×ATR`.
**TP**: VWAP itself; capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 4. Previous Day Extremes
**Entry**: **Reversal** — sweeps PDH/PDL and closes back inside → fade toward the opposite extreme (RVOL≥1.0). **Breakout** — closes through PDH/PDL with bias alignment (RVOL≥1.2).
**SL**: `0.15×ATR` (reversal) / `0.3×ATR` (breakout) beyond the level, floored at `0.5×ATR`.
**TP**: reversal targets the opposite prior-day extreme; breakout defaults to `3×ATR`, overridden by nearest opposing H1 zone if present. Capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 5. ATR Squeeze Breakout
**Entry**: ATR percentile ≤20th (compression) + a 5m BOS in either direction. Counter-daily-bias trades need RVOL≥2.5 + efficiency≥0.5 (else rejected), same-bias trades need RVOL≥1.5.
**SL**: `0.3×ATR` beyond the BOS level, floored at `0.5×ATR`.
**TP**: `3×ATR` default, overridden by nearest opposing H1 zone; capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 6. Geometry (MKE confluence)
**Entry**: reads only MKE Stage-5 `geometry` context (no RVOL/EMA inputs). **Confluence Bounce**: price at a scored support/resistance confluence zone (score≥35 or ≥50 across the two variants) + reversal candle (body≥40% of range), narrative not opposing. **Trendline Break-and-Retest**: a broken trendline retested from the new side with an absorption candle.
**SL**: zone/retest-candle boundary `−0.15×ATR`, floored at `0.5×ATR`.
**TP**: nearest structural level/trendline, capped at `3×ATR`. RR<1.5 (Score35) or <1.8 (Score50) rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 7. Order Flow
**Entry**: **Liquidity Sweep** — active sweep (confidence≥0.60) + reversal candle. **Imbalance Pullback** — price inside an FVG (confidence≥0.55) + reversal candle.
**SL**: swept level / imbalance edge `−0.15×ATR`, floored at `0.5×ATR`.
**TP**: nearest opposing liquidity pool, capped at `3×ATR`. RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 8. VWAP Reclaim
**Entry**: close crosses VWAP by `≥0.10×ATR` in the cross direction (trend-continuation — opposite thesis to VWAP Reversion). RVOL≥1.0, efficiency≥0.55.
**SL**: `0.15×ATR` beyond the cross candle, floored at `0.5×ATR`.
**TP**: `max(2R floor, opposing H1 zone)` — zone can only raise the target, never lower it below 2R. Capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 9. CPR Breakout
**Entry**: close breaks the prior day's Central Pivot Range (TC/BC computed from prior H/L/C). RVOL≥1.1, bias-aligned, efficiency≥0.55.
**SL**: `0.3×ATR` beyond TC/BC, floored at `0.5×ATR`.
**TP**: `max(2R floor, opposing H1 zone)`; capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 10. Gap (v2.0)
**Entry**: only within 45 min of open, gap≥0.4% vs prior close. **Gap-and-Go**: extends further in the gap direction → continuation. **Gap-Fill**: retraces toward prior close → reversion.
**SL**: `0.15×ATR` beyond today's open, floored at `0.5×ATR`.
**TP**: Gap-and-Go uses `max(2R floor, opposing H1 zone)`; Gap-Fill targets `prev_close` directly. Capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 11. Initial Balance Breakout (ORB 60m)
**Entry**: close breaks the 60-minute opening range high/low, RVOL≥1.2, bias-aligned, efficiency≥0.6, rejected if hour≥15.
**SL**: `0.3×ATR` beyond the range boundary, floored at `0.5×ATR`.
**TP**: `max(2R floor, opposing H1 zone)`; capped at 5×ATR; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.
*(15m/30m ORB variants retired 2026-08-08 — 0 real trades in 11 days, net-negative CF pnl_r.)*

### 12. OI-Wall Reaction
**Entry**: reacts to real option-chain OI walls. **Break**: RVOL-confirmed breach (≥1.3) → continuation. **Fade**: price approaches the wall (0.15% tolerance) with a reversal candle → fades to the opposite wall.
**SL**: wall strike `∓0.15×ATR`, floored at `0.5×ATR`.
**TP**: Break — `price ± 3×ATR`; Fade — the opposite wall strike, capped at `3×ATR`. RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 13. PCR-Extreme Reversal
**Entry**: PCR extreme (bullish/bearish OI skew) coinciding with a confluence-zone reversal candle (score≥40).
**SL**: zone boundary `−0.15×ATR`, floored at `0.5×ATR`.
**TP**: fixed `3×ATR` (not zone-based). RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 14. RSI-2 Mean Reversion
**Entry**: RSI(2) ≤10 or ≥90 + a reversal candle, RVOL≤1.5 (exhaustion happens in quiet conditions), not against daily bias.
**SL**: `0.15×ATR` beyond the reversal wick, floored at `0.5×ATR`.
**TP**: EMA20 directly (rejected if EMA20 already passed — no reversion room left); capped at `3×ATR`; RR<1.5 rejected.
**Multiple targets**: tp1@1.5R; base exit only.

### 15. Consolidation Breakout (Standard RVOL1.5 / Tight RVOL2.0)
**Entry**: H1 consolidation zone (ATR in bottom 30th percentile, ≥3 touches) + M5 breakout of the zone boundary + breakout-confidence score ≥60 + RSI momentum confirmation (>50 bullish, <50 bearish). Two variants differ only by RVOL gate (1.5 vs 2.0).
**SL**: `zone.bottom/top ∓ 0.10×zone_ATR`.
**TP**: `entry ± max(2R, 2×zone_range)` — explicit ≥2R guarantee.
**Multiple targets**: none — no `tp1` field emitted by this strategy; base TP_EXPANSION/trailing exit still applies post-entry.

### 16. Momentum Burst (5m, no MTF gating)
**Entry**: deliberately reads only 5m — no Daily/1H gating (fills the gap left by MTF strategies decaying in choppy regimes). Trigger candle: range≥1.8×ATR, body-fraction≥0.55, RVOL≥2.0. Requires same-direction follow-through on the next candle (giveback≤35% of trigger range) or rejected `NO_FOLLOW_THROUGH`.
**SL**: `max(0.5×trigger_range, 0.6×ATR)`.
**TP**: fixed `risk × 2.2` — no zone lookup at all, by design (avoids HTF dependence).
**Multiple targets**: tp1@1.5R; base exit only. Rejects `LATE_SESSION` if hour≥15.

### 17. HTF Pullback Reversal
**Entry**: three-timeframe — Daily bias must be directional (no neutral trades), price pulls back to within 0.6% of the 1H EMA20 from a recent 6-bar extreme, 5m reversal candle in the trend direction.
**SL**: 1H EMA20 `∓0.3×ATR` (anchored to the EMA, not entry price).
**TP**: `risk × 1.8` floor, raised further if a same-direction Daily zone exists beyond it (this strategy *does* use HTF zones, unlike Momentum Burst, since HTF context is its whole premise). RR<1.5 hard-rejected. Rejects `LOW_RVOL` (<0.9) and `LATE_SESSION` (hour≥15).
**Multiple targets**: tp1@1.5R; base exit only.

### 18. NIFTY/BankNifty Relative Value
**Entry**: cross-symbol — rolling z-score (60-bar) of the NIFTY/BankNifty ratio; `|z|≥2.0` fires both legs (fade the rich index, buy the cheap one) as two independent single-leg signals.
**SL**: fixed `±0.5×ATR` (no structural anchor — the thesis is the ratio, not either index's own structure).
**TP**: targets 60% reversion back to the rolling mean (`tp_ratio_reversion_fraction=0.6`), translated to a price via the other leg's current price. Capped at 5×ATR; RR<1.2 rejected.
**Multiple targets**: tp1@1.5R; each leg tracked as an independent single-leg position via base exit.

---

## Multi-leg combo strategies (all use `_update_combo_position()` — premium-based R, no zones/ATR)

All of these skip index-price SL/TP entirely: risk is defined as `target_r`/`stop_r` on **combined premium P&L divided by max loss**, and exit purely on hitting those R thresholds or 15:25 session end. "Multiple targets" doesn't apply — each has exactly one target_r/stop_r pair, no partial scale-out.

| Strategy | Entry thesis | Legs | target_r | stop_r | Notes |
|---|---|---|---|---|---|
| **Vertical Spread** | EMA20/50 cross aligned with daily bias | Buy ATM + sell 2-strikes-OTM same side (debit) | 1.0 | −0.6 | First combo experiment in the framework |
| **Straddle/Strangle** | ATR-percentile realized-vol compression (proxy for IV compression — no direction filter) | Straddle: both ATM. Strangle: both 2 strikes OTM | 1.2 | −0.5 | Unbounded max profit — target not capped |
| **Credit Spread** | PCR-extreme contrarian, wants range (rejects if RVOL>1.3 or efficiency>0.55) | Sell 1-OTM + buy 1+2-OTM, same side as PCR skew | 0.5 | −1.0 | Bounded profit — target capped at reachable max_profit/max_loss (can be as low as ~0.36R in practice) |
| **Iron Condor** | Neutral/range regime, rejects after 14:00 | Sell 1-OTM PE+CE, buy 1+2-OTM PE+CE both sides | 0.4 | −1.0 | Target capped at reachable max |
| **Butterfly** | Same range filters as Iron Condor | Buy 2-ITM CE, sell 2× ATM CE, buy 2-OTM CE (debit) | 1.5 | −0.5 | 30-min loss-cooldown: a losing exit blocks re-entry on that symbol for 30 min (added after 13 same-strike re-fires in one session at 25% win rate); wins re-enter immediately |
| **Iron Butterfly** | Same range filters, tighter (RVOL≤1.2, efficiency≤0.50) | Sell ATM PE+CE, buy 4-strikes-OTM wings both sides | 0.35 | −1.0 | Wider wings than Iron Condor = more premium, tighter profitable band. Same 30-min loss-cooldown |
| **Expiry-Aware Theta** | Iron Condor whose own params scale continuously with time-to-expiry | Same 4-leg shape as Iron Condor, wing width interpolated | 0.25 (near) – 0.5 (far), linear blend over 4 days | −1.0 fixed | `rvol_ceiling` (0.8→1.4) and `wing_width` (5→2) also scale with TTE; late-session cutoff hour is 14 far / 15 near expiry |

---

## Options Scalping (`OI_Scalping_v1.0`, PAPER, single ATM leg)

**Entry**: 4-quadrant OI×premium positioning inference (LONG_BUILDUP/SHORT_BUILDUP/SHORT_COVERING/LONG_UNWIND) on the ATM strike, using seed-calibrated thresholds (`OI_UP=+1.0%`, `OI_DOWN=−0.7%`, `PREM_UP=+1.8%`, `PREM_DOWN=−2.0%`). Requires 3-of-5 recent windows agreeing on direction + matching spot move + RVOL≥1.5 + valid BSM-solved Greeks + >15 min to expiry.

**Stop/target**: premium-based, not index-price-based — `stop_premium = entry_ask × (1 − 0.50)`, `target_premium = entry_ask × 2.0`. Also *defines* a delta-decay exit (`MIN_DELTA_LONG=0.25`) and a 15-minute time stop (`TIME_STOP_MIN_R=0.30`), plus a full transaction-cost model (spread, ₹20×2 brokerage, STT, exchange, GST).

**⚠️ Integration gap**: this strategy emits `stop_premium`/`target_premium`/`risk_per_lot` fields, not the `stop_loss`/`take_profit` that `_update_position()` expects, nor the `legs`/`net_premium_paid` that `_update_combo_position()` expects. No exit-management code path consuming these premium fields was found in `indian_trader.py` — worth verifying directly before trusting this strategy's exits in paper trading; it may currently be running with no active exit logic beyond whatever framework default applies to unrecognized signal shapes.

**Multiple targets**: none defined beyond the single stop/target premium pair.

---

## Summary: what "SL basis" and "TP basis" mean across the system

- **SL basis** is almost universally **ATR-anchored to a structural level** (swing wick, broken BOS level, zone boundary, EMA, VWAP, PDH/PDL, CPR line, opening range, OI wall) with a fixed buffer (`0.10–0.3×ATR`), and always floored at a minimum `0.5×ATR` distance so SL never gets pathologically tight. Combo strategies abandon this entirely for premium-based `stop_r`.
- **TP basis** is a mix of: nearest opposing structural zone (most strategies, when one exists), a fixed ATR-multiple (`2–3×ATR`) as a floor or fallback when no zone exists, and occasionally a specific level (VWAP, EMA20, opposite prior-day extreme, opposite OI wall). Almost everything is capped at `3–5×ATR` to prevent stale/far zones from producing absurd R:R. Every strategy enforces a minimum RR (1.2–1.8, mostly 1.5) at entry, rejecting the signal otherwise.
- **"Multiple targets"** in this codebase does not mean a TP1/TP2 partial-scale-out ladder anywhere. It means: (1) a stored `tp1` reference at 1.5R used for reporting only, and (2) the dynamic **TP_EXPANSION** mechanism in `_update_position()`, which keeps extending the real target and ratcheting the stop up each time price reaches it, functioning like a discrete trailing stop rather than a fixed ladder. Combo strategies have neither — one target_r, one stop_r, full exit.
