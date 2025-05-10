#!/usr/bin/env python3
"""
Test script for Fyers API authentication
"""
import os
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
import json
import re

# Load environment variables
load_dotenv()

# Get Fyers API credentials from environment variables
client_id = os.getenv("FYERS_CLIENT_ID")
secret_key = os.getenv("FYERS_SECRET_KEY")
redirect_uri = os.getenv("FYERS_REDIRECT_URI")
grant_type = os.getenv("FYERS_GRANT_TYPE")
response_type = os.getenv("FYERS_RESPONSE_TYPE")
state = os.getenv("FYERS_STATE")
auth_code = os.getenv("FYERS_AUTH_CODE")

print(f"Client ID: {client_id}")
print(f"Redirect URI: {redirect_uri}")
print(f"Auth Code: {auth_code[:10]}..." if auth_code and len(auth_code) > 10 else auth_code)

# Initialize Fyers session
appSession = fyersModel.SessionModel(
    client_id=client_id,
    redirect_uri=redirect_uri,
    response_type=response_type,
    state=state,
    secret_key=secret_key,
    grant_type=grant_type
)

# Generate authorization URL
generateTokenUrl = appSession.generate_authcode()
print(f"\nAuthorization URL: {generateTokenUrl}")
print("\nPlease visit this URL in your browser to authorize the application.")
print("After authorization, copy the auth code from the URL and update your .env file.")

# Try to generate token with existing auth code
print("\nTrying to generate token with existing auth code...")
appSession.set_token(auth_code)
response = appSession.generate_token()
print(f"Token response: {json.dumps(response, indent=2)}")

if isinstance(response, dict) and 'access_token' in response:
    access_token = response["access_token"]
    print(f"\nAccess token: {access_token[:10]}...")
    
    # Save access token to .env file
    print("\nSaving access token to .env file...")
    
    # Read current .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    with open(env_path, 'r') as file:
        env_content = file.read()
    
    # Check if FYERS_ACCESS_TOKEN already exists in .env
    if re.search(r'^FYERS_ACCESS_TOKEN=', env_content, re.MULTILINE):
        # Update existing token
        env_content = re.sub(r'^FYERS_ACCESS_TOKEN=.*$', f'FYERS_ACCESS_TOKEN={access_token}', env_content, flags=re.MULTILINE)
    else:
        # Add new token
        env_content += f'\nFYERS_ACCESS_TOKEN={access_token}'
    
    # Write updated content back to .env
    with open(env_path, 'w') as file:
        file.write(env_content)
    
    print("✅ Access token saved to .env file")
    
    # Initialize Fyers model with token
    fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")
    
    # Test API with profile request
    print("\nTesting API with profile request...")
    profile = fyers.get_profile()
    print(f"Profile response: {json.dumps(profile, indent=2)}")
    
    # Test market status
    print("\nTesting API with market status request...")
    market_status = fyers.market_status()
    print(f"Market status response: {json.dumps(market_status, indent=2)}")
else:
    print("\n❌ Failed to generate access token. Please update your auth code in the .env file.") 