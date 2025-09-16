#!/usr/bin/env python3
"""
Fix database indentation issues
"""

# Read the file
with open('src/models/consolidated_database.py', 'r') as f:
    content = f.read()

# Fix the calculate_unrealized_pnl method
old_method = '''        def calculate_unrealized_pnl(self, market: str, current_prices: Dict[str, float]) -> float:
            """Calculate unrealized P&L for open trades."""
            try:
                total_unrealized = 0.0
def initialize_connection_pools():'''

new_method = '''    def calculate_unrealized_pnl(self, market: str, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L for open trades."""
        try:
            total_unrealized = 0.0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_id, symbol, signal, entry_price, quantity
                    FROM open_trades 
                    WHERE market = ?
                ''', (market,))
                
                for trade in cursor.fetchall():
                    trade_id, symbol, signal, entry_price, quantity = trade
                    
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        
                        if signal == 'BUY CALL':
                            unrealized = (current_price - entry_price) * quantity
                        else:  # BUY PUT
                            unrealized = (entry_price - current_price) * quantity
                        
                        total_unrealized += unrealized
                
                return total_unrealized
                
        except Exception as e:
            logger.error(f"Failed to calculate unrealized P&L for {market}: {e}")
            return 0.0

def initialize_connection_pools():'''

# Replace the content
content = content.replace(old_method, new_method)

# Write back to file
with open('src/models/consolidated_database.py', 'w') as f:
    f.write(content)

print("✅ Database indentation fixed")
