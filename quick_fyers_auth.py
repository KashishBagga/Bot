#!/usr/bin/env python3
"""
Quick Fyers Authentication - Simple and reliable
"""

import os
import sys
import time
import webbrowser
from fyers_apiv3 import fyersModel
from src.config.settings import (
    FYERS_CLIENT_ID, 
    FYERS_SECRET_KEY, 
    FYERS_REDIRECT_URI,
    FYERS_RESPONSE_TYPE,
    FYERS_GRANT_TYPE,
    FYERS_STATE
)

def quick_auth():
    """Quick authentication without hanging the system"""
    try:
        
        # Check if we already have a valid token
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                content = f.read()
                if 'FYERS_ACCESS_TOKEN=' in content:
                    return True
        
        # Generate auth URL
        auth_url = f"https://api-t1.fyers.in/api/v3/generate-authcode?client_id={FYERS_CLIENT_ID}&redirect_uri={FYERS_REDIRECT_URI}&response_type={FYERS_RESPONSE_TYPE}&state={FYERS_STATE}"
        
        
        # Open browser
        webbrowser.open(auth_url, new=1)
        
        
        # Get auth code manually
        auth_code = input("\n🔑 Enter the auth_code from the browser: ").strip()
        
        if not auth_code:
            return False
        
        # Generate access token
        
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_KEY,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type=FYERS_RESPONSE_TYPE,
            grant_type=FYERS_GRANT_TYPE,
            state=FYERS_STATE
        )
        
        session.set_token(auth_code)
        response = session.generate_token()
        
        if 'access_token' in response:
            access_token = response['access_token']
            
            # Update .env file
            update_env_file(access_token, auth_code)
            
            return True
            
        else:
            return False
            
    except Exception as e:
        return False

def update_env_file(access_token, auth_code):
    """Update .env file with new tokens"""
    try:
        # Read existing .env file
        env_lines = []
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_lines = f.readlines()
        
        # Update or add tokens
        updated = False
        auth_updated = False
        
        for i, line in enumerate(env_lines):
            if line.startswith('FYERS_ACCESS_TOKEN='):
                env_lines[i] = f'FYERS_ACCESS_TOKEN={access_token}\n'
                updated = True
            elif line.startswith('FYERS_AUTH_CODE='):
                env_lines[i] = f'FYERS_AUTH_CODE={auth_code}\n'
                auth_updated = True
        
        if not updated:
            env_lines.append(f'FYERS_ACCESS_TOKEN={access_token}\n')
        if not auth_updated:
            env_lines.append(f'FYERS_AUTH_CODE={auth_code}\n')
        
        # Write back to .env file
        with open('.env', 'w') as f:
            f.writelines(env_lines)
        
        
    except Exception as e:

if __name__ == "__main__":
    success = quick_auth()
    if success:
    else:
        sys.exit(1) 