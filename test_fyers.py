#!/usr/bin/env python3
"""
Test Fyers API authentication script.
Helps generate auth code and access token for the Fyers API.
"""
import os
import time
import webbrowser
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

def main():
    """Run the Fyers API authentication test."""
    # Load environment variables
    load_dotenv()
    
    # Get Fyers API credentials from environment variables
    client_id = os.getenv("FYERS_CLIENT_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/")
    response_type = os.getenv("FYERS_RESPONSE_TYPE", "code")
    state = os.getenv("FYERS_STATE", "sample")
    
    if not client_id or not secret_key:
        print("❌ Fyers API credentials not found in .env file")
        print("Please set FYERS_CLIENT_ID and FYERS_SECRET_KEY in .env file")
        return False
    
    print(f"Found Fyers API credentials for client ID {client_id}")
    
    try:
        # Initialize session model for generating auth code
        session = fyersModel.SessionModel(
            client_id=client_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type=response_type,
            state=state,
            grant_type="authorization_code"
        )
        
        # Generate auth code URL
        auth_url = session.generate_authcode()
        print("\n=== Step 1: Generate Authorization Code ===")
        print(f"Opening authorization URL in browser: {auth_url}")
        webbrowser.open(auth_url, new=1)
        
        # Wait for user to complete authentication in browser
        print("\nPlease complete authentication in the browser window that just opened.")
        print("After successful authentication, you will be redirected to the Fyers website.")
        print("Look at the URL in your browser address bar after redirection.")
        print("It should contain 'auth_code=' followed by a long string.")
        print("\nPlease copy the entire string AFTER 'auth_code=' and paste it below:")
        
        auth_code = input("Enter auth code: ").strip()
        
        if not auth_code:
            print("❌ Auth code not provided")
            return False
        
        # Update .env file with the new auth code
        update_env_file("FYERS_AUTH_CODE", auth_code)
        
        print("\n=== Step 2: Generate Access Token ===")
        print("Generating access token using the auth code...")
        
        # Set auth code in session
        session.set_token(auth_code)
        
        # Generate token
        response = session.generate_token()
        
        if 'access_token' in response:
            access_token = response['access_token']
            print("✅ Access token generated successfully!")
            
            # Update .env file with the new access token
            update_env_file("FYERS_ACCESS_TOKEN", access_token)
            
            # Test the access token
            print("\n=== Step 3: Test Access Token ===")
            print("Testing the access token by fetching user profile...")
            
            fyers = fyersModel.FyersModel(
                token=access_token,
                is_async=False,
                client_id=client_id,
                log_path=""
            )
            
            profile = fyers.get_profile()
            
            if 'code' in profile and profile['code'] == 200:
                print("✅ Authentication successful!")
                print(f"Logged in as: {profile.get('data', {}).get('name', 'Unknown')}")
                print(f"User ID: {profile.get('data', {}).get('fy_id', 'Unknown')}")
                print(f"Email: {profile.get('data', {}).get('email_id', 'Unknown')}")
                print("\nYou can now use the Fyers API successfully in your trading scripts.")
                return True
            else:
                print(f"❌ Failed to fetch profile: {profile}")
                return False
        else:
            print(f"❌ Failed to generate access token: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error in authentication process: {e}")
        return False

def update_env_file(key, value):
    """Update a value in the .env file."""
    try:
        # Read current .env content
        with open(".env", "r") as file:
            lines = file.readlines()
        
        # Update the value for the key
        updated = False
        with open(".env", "w") as file:
            for line in lines:
                if line.strip().startswith(f"{key}="):
                    file.write(f"{key}={value}\n")
                    updated = True
                else:
                    file.write(line)
            
            # Add key-value if not found
            if not updated:
                file.write(f"\n{key}={value}\n")
        
        print(f"✅ Updated {key} in .env file")
        return True
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False

if __name__ == "__main__":
    main() 