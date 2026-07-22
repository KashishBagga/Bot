#!/usr/bin/env python3
"""
Fyers Authentication Script
===========================
This script automates the Fyers login process:
1. Prompts for Client ID and Secret Key if not in .env
2. Opens the Fyers login page in your browser
3. Asks you to paste the URL you're redirected to
4. Extracts the auth_code and generates a fresh Access Token
5. Updates your .env file automatically
"""

import os
import sys
import re
import webbrowser
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv, set_key

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.api.fyers import FyersClient
    from fyers_apiv3 import fyersModel
except ImportError:
    print("❌ Error: Could not import Fyers modules. Make sure 'fyers-apiv3' is installed.")
    print("Run: pip install fyers-apiv3 python-dotenv")
    sys.exit(1)

def update_env(key, value):
    env_path = os.path.join(project_root, ".env")
    set_key(env_path, key, value)
    print(f"✅ Updated {key} in .env")

def main():
    print("🚀 Fyers Authentication Utility")
    print("═" * 30)
    
    # 1. Load existing config
    load_dotenv()
    client_id = os.getenv("FYERS_CLIENT_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/get")

    # 2. Check for missing credentials
    if not client_id or "your_client_id" in client_id:
        client_id = input("👉 Enter your FYERS_CLIENT_ID (App ID): ").strip()
        update_env("FYERS_CLIENT_ID", client_id)
        
    if not secret_key or "your_secret_key" in secret_key:
        secret_key = input("👉 Enter your FYERS_SECRET_KEY: ").strip()
        update_env("FYERS_SECRET_KEY", secret_key)

    # 3. Initialize Session
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type='code',
        grant_type='authorization_code'
    )

    # 4. Open Auth URL
    auth_url = session.generate_authcode()
    print("\n🌐 Opening Fyers Login Page in your browser...")
    print(f"If it doesn't open automatically, click here: {auth_url}")
    webbrowser.open(auth_url)

    # 5. Get the Redirect URL
    print("\n" + "═" * 60)
    print("STEP 1: Log in to Fyers in the browser window that just opened.")
    print("STEP 2: Once you see a 'Success' message, look at the browser's address bar.")
    print("STEP 3: Copy the FULL URL (it starts with " + redirect_uri + ")")
    print("═" * 60 + "\n")
    
    full_url = input("👉 Paste the FULL redirect URL here: ").strip()

    # 6. Extract Auth Code
    try:
        parsed_url = urlparse(full_url)
        auth_code = parse_qs(parsed_url.query).get('auth_code', [None])[0]
        
        if not auth_code:
            # Fallback for when user just pastes the code
            auth_code = full_url
            
        print(f"✨ Extracted Auth Code: {auth_code[:10]}...")
        update_env("FYERS_AUTH_CODE", auth_code)
        
        # 7. Generate Access Token
        session.set_token(auth_code)
        response = session.generate_token()
        
        if 'access_token' in response:
            token = response['access_token']
            print("🚀 Access Token generated successfully!")
            update_env("FYERS_ACCESS_TOKEN", token)
            
            # Cache locally to tokens/
            try:
                import json
                from datetime import date
                token_dir = os.path.join(project_root, "tokens")
                os.makedirs(token_dir, exist_ok=True)
                today_str = date.today().strftime('%Y-%m-%d')
                token_path = os.path.join(token_dir, f"token_{today_str}.json")
                with open(token_path, 'w') as f:
                    json.dump({"access_token": token, "date": today_str}, f)
                print(f"💾 Cached access token to: {token_path}")
            except Exception as ex:
                print(f"❌ Failed to save local token cache: {ex}")
                
            print("\n✅ Setup Complete! You can now run the bot in Live Mode.")
        else:
            print(f"❌ Failed to generate token: {response}")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
