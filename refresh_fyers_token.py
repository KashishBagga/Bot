#!/usr/bin/env python3
"""
Fyers Token Refresh Script
Automatically refreshes the Fyers access token when it expires
"""

import os
import sys
from dotenv import load_dotenv
from automated_fyers_auth import AutomatedFyersAuth

def check_and_refresh_token():
    """Check if token is valid, refresh if needed."""
    load_dotenv()
    
    print("🔍 Checking Fyers token validity...")
    
    # Get current token
    current_token = os.getenv("FYERS_ACCESS_TOKEN")
    if not current_token:
        print("❌ No access token found. Running full authentication...")
        return run_full_auth()
    
    # Check if token is still valid
    auth_handler = AutomatedFyersAuth()
    if auth_handler.check_token_validity(current_token):
        print("✅ Current token is still valid!")
        return current_token
    else:
        print("⚠️ Token has expired. Refreshing...")
        return run_full_auth()

def run_full_auth():
    """Run the full authentication process."""
    try:
        auth_handler = AutomatedFyersAuth()
        new_token = auth_handler.run_automated_auth()
        
        if new_token:
            print("✅ Token refreshed successfully!")
            return new_token
        else:
            print("❌ Failed to refresh token!")
            return None
            
    except Exception as e:
        print(f"❌ Error refreshing token: {e}")
        return None

def main():
    """Main function."""
    print("🔄 Fyers Token Refresh Utility")
    print("=" * 40)
    
    token = check_and_refresh_token()
    
    if token:
        print(f"\n✅ Token is ready: {token[:50]}...")
        print("You can now run your trading scripts.")
        return True
    else:
        print("\n❌ Failed to get valid token!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 