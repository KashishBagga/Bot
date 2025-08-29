#!/usr/bin/env python3
"""
Fyers Token Refresh Utility
Automatically refreshes Fyers access token when expired
"""

import os
import sys
from dotenv import load_dotenv
from automated_fyers_auth import AutomatedFyersAuth

def main():
    """Main function to refresh Fyers token."""
    # Load environment variables
    load_dotenv()
    
    print("🔄 Fyers Token Refresh Utility")
    print("=" * 32)
    
    # Check if we already have a valid token
    existing_token = os.getenv("FYERS_ACCESS_TOKEN")
    if existing_token:
        auth_util = AutomatedFyersAuth()
        if auth_util.check_token_validity(existing_token):
            print("✅ Current token is still valid!")
            return
    
    print("⚠️ Token has expired. Refreshing...")
    
    # Initialize auth utility
    auth_util = AutomatedFyersAuth()
    
    # Run authentication
    if auth_util.authenticate():
        print("✅ Token refresh completed successfully!")
    else:
        print("❌ Failed to refresh token!")
        sys.exit(1)

if __name__ == "__main__":
    main() 