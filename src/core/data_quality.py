#!/usr/bin/env python3
"""
Data Quality Validator — tag and quarantine corrupted or mock trade records.
========================================================================
Validates both trade_performance and counterfactual_results data before
saving to the database or when processing research metrics.
"""

from typing import Dict, Any, List, Tuple

def validate_trade_data(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate trade_performance or counterfactual_results fields.
    Returns (is_valid, error_messages).
    """
    errors = []

    # Get values safely, handling possible types and None values
    entry_price = row.get("entry_price")
    stop_loss = row.get("stop_loss")
    take_profit = row.get("take_profit")
    initial_stop_loss = row.get("initial_stop_loss")
    initial_take_profit = row.get("initial_take_profit")
    stop_loss_distance = row.get("stop_loss_distance")
    symbol = row.get("symbol")
    trade_id = row.get("trade_id") or row.get("candidate_id")

    # 1. Entry price must be > 0
    if entry_price is None or entry_price <= 0:
        errors.append(f"Invalid entry_price: {entry_price}")

    # 2. SL and TP must be > 0
    if stop_loss is None or stop_loss <= 0:
        errors.append(f"Invalid stop_loss: {stop_loss}")
    if take_profit is None or take_profit <= 0:
        errors.append(f"Invalid take_profit: {take_profit}")

    # 3. Initial SL and TP must be > 0 if present
    if initial_stop_loss is not None and initial_stop_loss <= 0:
        errors.append(f"Invalid initial_stop_loss: {initial_stop_loss}")
    if initial_take_profit is not None and initial_take_profit <= 0:
        errors.append(f"Invalid initial_take_profit: {initial_take_profit}")

    # 4. SL != Entry (otherwise distance/risk is 0)
    if entry_price is not None and stop_loss is not None:
        if abs(entry_price - stop_loss) < 1e-4:
            errors.append(f"stop_loss equal to entry_price: {entry_price}")
        if initial_stop_loss is not None and abs(entry_price - initial_stop_loss) < 1e-4:
            errors.append(f"initial_stop_loss equal to entry_price: {entry_price}")

    # 5. Quarantine test/mock trades
    if trade_id and any(x in str(trade_id).lower() for x in ("test", "mock", "dummy")):
        errors.append("Test/mock trade quarantined")

    # 6. For index symbols, check if the price levels are reasonable
    if symbol == "NSE:NIFTY50-INDEX":
        if entry_price == 22000.0 or (row.get("exit_price") == 21900.0):
            errors.append("Corrupted test/fallback entry/exit price (e.g. 22000/21900)")
    elif symbol == "NSE:NIFTYBANK-INDEX":
        if entry_price == 50000.0:
            errors.append("Corrupted test entry price (50000.0)")

    # 7. Check high/low bounds (if available)
    highest_price = row.get("highest_price")
    lowest_price = row.get("lowest_price")
    if highest_price is not None and lowest_price is not None:
        if lowest_price > highest_price:
            errors.append(f"lowest_price ({lowest_price}) > highest_price ({highest_price})")
        if entry_price is not None:
            if entry_price < lowest_price - 1.0 or entry_price > highest_price + 1.0:
                errors.append(f"entry_price ({entry_price}) outside range [{lowest_price}, {highest_price}]")

    # 8. Closed trade checks
    exit_price = row.get("exit_price")
    exit_time = row.get("exit_time")
    exit_reason = row.get("exit_reason")
    bars_held = row.get("bars_held")
    final_pnl_r = row.get("final_pnl_r") if row.get("final_pnl_r") is not None else row.get("pnl")

    if exit_time is not None or exit_price is not None or exit_reason is not None:
        if exit_price is None or exit_price <= 0:
            errors.append(f"Invalid exit_price for closed trade: {exit_price}")
        
        # Bars held should be >= 1 for completed trades (per checklist)
        if bars_held is not None and bars_held < 1:
            errors.append(f"Closed trade with bars_held < 1: {bars_held}")

        if highest_price is not None and lowest_price is not None and exit_price is not None:
            if exit_price < lowest_price - 1.0 or exit_price > highest_price + 1.0:
                errors.append(f"exit_price ({exit_price}) outside range [{lowest_price}, {highest_price}]")

        # PnL consistency check
        signal_type = row.get("signal_type") or row.get("signal")
        if signal_type and entry_price and exit_price and stop_loss_distance and stop_loss_distance > 0:
            sig_upper = str(signal_type).upper()
            is_long = "CALL" in sig_upper or ("BUY" in sig_upper and "PUT" not in sig_upper)
            if is_long:
                expected_pnl_raw = (exit_price - entry_price) / stop_loss_distance
            else:
                expected_pnl_raw = (entry_price - exit_price) / stop_loss_distance
            
            if final_pnl_r is not None:
                diff_with_fee = abs(final_pnl_r - (expected_pnl_raw - 0.05))
                diff_without_fee = abs(final_pnl_r - expected_pnl_raw)
                if min(diff_with_fee, diff_without_fee) > 0.25:
                    errors.append(f"PnL inconsistency: stored {final_pnl_r:.2f}R, expected raw {expected_pnl_raw:.2f}R")

    return len(errors) == 0, errors
