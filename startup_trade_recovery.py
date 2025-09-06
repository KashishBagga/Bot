#!/usr/bin/env python3
"""
Startup Trade Recovery - Critical for Live Trading
Prevents lost positions on restart by auto-loading open trades from database
"""

def add_startup_trade_recovery():
    """Add startup trade recovery to live_paper_trading.py"""
    
    print("🔄 Adding Startup Trade Recovery...")
    
    with open('live_paper_trading.py', 'r') as f:
        content = f.read()
    
    # Add the recovery method
    recovery_method = '''
    def _recover_open_trades(self):
        """Recover open trades from database on startup - critical for live trading."""
        try:
            logger.info("🔄 Starting trade recovery from database...")
            
            # Get all open trades from database
            open_trades_data = self.db.fetch_open_option_positions()
            
            if not open_trades_data:
                logger.info("✅ No open trades found in database")
                return
            
            recovered_count = 0
            with self._lock:
                for trade_data in open_trades_data:
                    try:
                        # Reconstruct PaperTrade object from database data
                        trade = PaperTrade(
                            id=trade_data['id'],
                            timestamp=pd.to_datetime(trade_data['entry_time']).to_pydatetime().replace(tzinfo=self.tz),
                            contract_symbol=trade_data['contract_symbol'],
                            underlying=trade_data['underlying'],
                            strategy=trade_data['strategy'],
                            signal_type=trade_data.get('signal_type') or trade_data.get('signal', ''),
                            entry_price=trade_data['entry_price'],
                            quantity=trade_data['quantity'],
                            lot_size=trade_data.get('lot_size', 0),
                            strike=trade_data.get('strike'),
                            expiry=pd.to_datetime(trade_data['expiry']).to_pydatetime() if trade_data.get('expiry') else None,
                            option_type=trade_data.get('option_type'),
                            status='OPEN',
                            commission=trade_data.get('commission', 0.0),
                            confidence=trade_data.get('confidence', 0.0),
                            reasoning=trade_data.get('reasoning', ''),
                            entry_value=trade_data.get('entry_value'),
                            entry_commission=trade_data.get('entry_commission'),
                            entry_time=pd.to_datetime(trade_data['entry_time']).to_pydatetime().replace(tzinfo=self.tz)
                        )
                        
                        # Restore to in-memory state
                        self.open_trades[trade.id] = trade
                        key = self._make_open_key(trade.strategy, trade.contract_symbol, trade.option_type)
                        self._open_keys.add(key)
                        
                        recovered_count += 1
                        logger.info(f"✅ Recovered trade: {trade.id[:8]}... | {trade.strategy} {trade.signal_type}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error recovering trade {trade_data.get('id', 'unknown')}: {e}")
                        continue
            
            logger.info(f"🎉 Trade recovery complete: {recovered_count} trades recovered")
            
        except Exception as e:
            logger.error(f"❌ Critical error in trade recovery: {e}")
    '''
    
    # Insert before _validate_production_requirements
    import re
    content = re.sub(
        r'(def _validate_production_requirements\(self\):)',
        recovery_method + r'\n    \1',
        content
    )
    
    # Add recovery call to start_trading
    content = re.sub(
        r'(logger\.info\("🚀 Starting live paper trading\.\.\."\))',
        r'\1\n        \n        # CRITICAL: Recover open trades from database\n        self._recover_open_trades()',
        content
    )
    
    with open('live_paper_trading.py', 'w') as f:
        f.write(content)
    
    print("✅ Startup Trade Recovery Added!")

if __name__ == "__main__":
    add_startup_trade_recovery()
