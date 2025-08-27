#!/usr/bin/env python3
"""
Options P&L Calculator
Handles both long and short options with proper margin calculations
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OptionsPnLCalculator:
    def __init__(self, margin_requirements: Dict = None):
        """
        Initialize P&L calculator with margin requirements.
        
        Args:
            margin_requirements: Dict of margin requirements for different option types
        """
        self.margin_requirements = margin_requirements or {
            'naked_call': 0.15,  # 15% of underlying value
            'naked_put': 0.15,   # 15% of underlying value
            'covered_call': 0.05, # 5% of underlying value
            'cash_secured_put': 0.10  # 10% of underlying value
        }
    
    def calculate_entry_cost(self, position_type: PositionType, entry_price: float, 
                           quantity: int, lot_size: int, commission_bps: float = 1.0) -> Dict:
        """
        Calculate entry cost for an option position.
        
        Args:
            position_type: LONG or SHORT
            entry_price: Entry price per share
            quantity: Quantity in shares
            lot_size: Lot size for the contract
            commission_bps: Commission in basis points
            
        Returns:
            Dict with cost breakdown
        """
        lots = quantity / lot_size
        notional = entry_price * quantity
        
        # Commission calculation
        commission = notional * (commission_bps / 10000.0)
        
        if position_type == PositionType.LONG:
            # For long options: pay premium + commission
            total_cost = notional + commission
            margin_required = 0  # No margin for long options
        else:
            # For short options: margin required + commission
            margin_required = notional * self.margin_requirements['naked_call']  # Simplified
            total_cost = margin_required + commission
        
        return {
            'position_type': position_type.value,
            'entry_price': entry_price,
            'quantity': quantity,
            'lots': lots,
            'notional': notional,
            'commission': commission,
            'margin_required': margin_required,
            'total_cost': total_cost
        }
    
    def calculate_exit_value(self, position_type: PositionType, exit_price: float,
                           quantity: int, lot_size: int, commission_bps: float = 1.0) -> Dict:
        """
        Calculate exit value for an option position.
        
        Args:
            position_type: LONG or SHORT
            exit_price: Exit price per share
            quantity: Quantity in shares
            lot_size: Lot size for the contract
            commission_bps: Commission in basis points
            
        Returns:
            Dict with exit value breakdown
        """
        lots = quantity / lot_size
        notional = exit_price * quantity
        
        # Commission calculation
        commission = notional * (commission_bps / 10000.0)
        
        if position_type == PositionType.LONG:
            # For long options: receive premium - commission
            total_received = notional - commission
            margin_released = 0
        else:
            # For short options: receive margin back - commission
            margin_released = notional * self.margin_requirements['naked_call']  # Simplified
            total_received = margin_released - commission
        
        return {
            'position_type': position_type.value,
            'exit_price': exit_price,
            'quantity': quantity,
            'lots': lots,
            'notional': notional,
            'commission': commission,
            'margin_released': margin_released,
            'total_received': total_received
        }
    
    def calculate_pnl(self, entry_data: Dict, exit_data: Dict) -> Dict:
        """
        Calculate P&L for an option position.
        
        Args:
            entry_data: Entry cost data from calculate_entry_cost
            exit_data: Exit value data from calculate_exit_value
            
        Returns:
            Dict with P&L breakdown
        """
        position_type = PositionType(entry_data['position_type'])
        
        if position_type == PositionType.LONG:
            # Long options: exit_received - entry_cost
            pnl = exit_data['total_received'] - entry_data['total_cost']
            pnl_per_share = pnl / entry_data['quantity']
            pnl_per_lot = pnl / entry_data['lots']
            returns_pct = (pnl / entry_data['total_cost']) * 100 if entry_data['total_cost'] > 0 else 0
        else:
            # Short options: entry_cost - exit_received (reversed)
            pnl = entry_data['total_cost'] - exit_data['total_received']
            pnl_per_share = pnl / entry_data['quantity']
            pnl_per_lot = pnl / entry_data['lots']
            returns_pct = (pnl / entry_data['total_cost']) * 100 if entry_data['total_cost'] > 0 else 0
        
        return {
            'position_type': position_type.value,
            'pnl': pnl,
            'pnl_per_share': pnl_per_share,
            'pnl_per_lot': pnl_per_lot,
            'returns_pct': returns_pct,
            'entry_cost': entry_data['total_cost'],
            'exit_value': exit_data['total_received'],
            'total_commission': entry_data['commission'] + exit_data['commission']
        }
    
    def calculate_margin_utilization(self, positions: list, available_capital: float) -> Dict:
        """
        Calculate margin utilization across all positions.
        
        Args:
            positions: List of open positions
            available_capital: Available capital for margin
            
        Returns:
            Dict with margin utilization metrics
        """
        total_margin_required = 0
        total_premium_paid = 0
        short_positions = []
        long_positions = []
        
        for position in positions:
            if position.get('position_type') == PositionType.SHORT.value:
                short_positions.append(position)
                total_margin_required += position.get('margin_required', 0)
            else:
                long_positions.append(position)
                total_premium_paid += position.get('entry_cost', 0)
        
        margin_utilization = (total_margin_required / available_capital) * 100 if available_capital > 0 else 0
        capital_utilization = ((total_margin_required + total_premium_paid) / available_capital) * 100 if available_capital > 0 else 0
        
        return {
            'total_margin_required': total_margin_required,
            'total_premium_paid': total_premium_paid,
            'margin_utilization_pct': margin_utilization,
            'capital_utilization_pct': capital_utilization,
            'available_margin': available_capital - total_margin_required,
            'short_positions_count': len(short_positions),
            'long_positions_count': len(long_positions)
        }
    
    def calculate_drawdown_metrics(self, equity_curve: list) -> Dict:
        """
        Calculate comprehensive drawdown metrics.
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            Dict with drawdown metrics
        """
        if not equity_curve:
            return {}
        
        peak = equity_curve[0]
        max_drawdown = 0
        current_drawdown = 0
        drawdown_duration = 0
        max_drawdown_duration = 0
        daily_drawdowns = []
        
        for i, equity in enumerate(equity_curve):
            if equity > peak:
                peak = equity
                current_drawdown = 0
                drawdown_duration = 0
            else:
                current_drawdown = (peak - equity) / peak
                drawdown_duration += 1
                
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    max_drawdown_duration = drawdown_duration
            
            daily_drawdowns.append(current_drawdown)
        
        # Calculate rolling drawdown metrics
        rolling_30d_drawdown = max(daily_drawdowns[-30:]) if len(daily_drawdowns) >= 30 else max(daily_drawdowns)
        rolling_7d_drawdown = max(daily_drawdowns[-7:]) if len(daily_drawdowns) >= 7 else max(daily_drawdowns)
        
        return {
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_duration': max_drawdown_duration,
            'rolling_30d_drawdown_pct': rolling_30d_drawdown * 100,
            'rolling_7d_drawdown_pct': rolling_7d_drawdown * 100,
            'current_drawdown_pct': current_drawdown * 100,
            'peak_equity': peak,
            'current_equity': equity_curve[-1]
        }
    
    def calculate_risk_metrics(self, trades: list, initial_capital: float) -> Dict:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            trades: List of completed trades
            initial_capital: Initial capital
            
        Returns:
            Dict with risk metrics
        """
        if not trades:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Risk metrics
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else float('inf')
        
        # Maximum consecutive losses
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        for trade in trades:
            if trade.get('pnl', 0) <= 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0
        
        # Sharpe ratio (simplified)
        returns = [t.get('returns_pct', 0) for t in trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'sharpe_ratio': sharpe_ratio,
            'total_return_pct': (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
        } 