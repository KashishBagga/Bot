#!/usr/bin/env python3
"""
Fix Advanced Systems Issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def fix_trade_execution_manager():
    """Fix Trade Execution Manager import issues"""
    print("🔧 Fixing Trade Execution Manager...")
    
    # Read the file
    with open('trade_execution_manager.py', 'r') as f:
        content = f.read()
    
    # Add missing import
    if 'from abc import ABC, abstractmethod' not in content:
        content = content.replace(
            'import json',
            'import json\nfrom abc import ABC, abstractmethod'
        )
    
    # Write back
    with open('trade_execution_manager.py', 'w') as f:
        f.write(content)
    
    print("✅ Trade Execution Manager fixed")

def fix_risk_manager():
    """Fix Risk Manager missing method"""
    print("🔧 Fixing Risk Manager...")
    
    # Read the risk manager file
    with open('src/core/risk_manager.py', 'r') as f:
        content = f.read()
    
    # Add missing should_execute_signal method
    if 'def should_execute_signal' not in content:
        method = '''
    def should_execute_signal(self, signal: Dict[str, Any], current_prices: Dict[str, float]) -> bool:
        """Check if signal should be executed based on risk limits"""
        try:
            # Basic risk checks
            confidence = signal.get('confidence', 0)
            if confidence < 50:  # Minimum confidence threshold
                return False
            
            # Check position size
            position_size = signal.get('position_size', 0)
            if position_size <= 0:
                return False
            
            # Check if we have price data
            symbol = signal.get('symbol')
            if symbol not in current_prices:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking signal execution: {e}")
            return False
'''
        
        # Add method before the last closing brace
        content = content.rstrip() + method + '\n'
    
    # Write back
    with open('src/core/risk_manager.py', 'w') as f:
        f.write(content)
    
    print("✅ Risk Manager fixed")

def main():
    """Main function"""
    print("🚀 FIXING ADVANCED SYSTEMS ISSUES")
    print("=" * 50)
    
    fix_trade_execution_manager()
    fix_risk_manager()
    
    print("\n✅ All fixes applied!")

if __name__ == "__main__":
    main()
