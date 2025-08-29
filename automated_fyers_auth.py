#!/usr/bin/env python3
"""
Automated Fyers API Authentication
Handles the complete authentication flow without manual intervention
"""

import os
import time
import json
import requests
import webbrowser
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import sys

class FyersAuthHandler(BaseHTTPRequestHandler):
    """HTTP server to capture the auth code from Fyers redirect."""
    
    def __init__(self, *args, auth_code_callback=None, **kwargs):
        self.auth_code_callback = auth_code_callback
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle the redirect from Fyers with auth code."""
        try:
            # Parse the URL to extract auth code
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Extract auth code
            auth_code = query_params.get('auth_code', [None])[0]
            state = query_params.get('state', [None])[0]
            
            if auth_code:
                # Store the auth code
                if self.auth_code_callback:
                    self.auth_code_callback(auth_code)
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                success_html = """
                <html>
                <head><title>Authentication Successful</title></head>
                <body>
                <h1>✅ Fyers Authentication Successful!</h1>
                <p>You can close this window now.</p>
                <p>Your access token will be generated automatically.</p>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode())
            else:
                # Send error response
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = """
                <html>
                <head><title>Authentication Failed</title></head>
                <body>
                <h1>❌ Authentication Failed</h1>
                <p>No auth code received. Please try again.</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
                
        except Exception as e:
            print(f"❌ Error handling redirect: {e}")
            self.send_response(500)
            self.end_headers()

class AutomatedFyersAuth:
    """Automated Fyers authentication handler."""
    
    def __init__(self):
        """Initialize the automated auth handler."""
        load_dotenv()
        self.auth_code = None
        self.server = None
        self.server_thread = None
        
        # Get credentials from environment
        self.client_id = os.getenv("FYERS_CLIENT_ID")
        self.secret_key = os.getenv("FYERS_SECRET_KEY")
        self.redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/")
        self.response_type = os.getenv("FYERS_RESPONSE_TYPE", "code")
        self.state = os.getenv("FYERS_STATE", "sample")
        
        if not self.client_id or not self.secret_key:
            raise ValueError("FYERS_CLIENT_ID and FYERS_SECRET_KEY must be set in .env file")
    
    def start_auth_server(self, port=8080):  # Changed back to 8080 since conflicts are resolved
        """Start HTTP server to capture auth code."""
        def auth_code_callback(auth_code):
            self.auth_code = auth_code
            print(f"✅ Auth code received: {auth_code[:20]}...")
        
        # Create custom handler with callback
        class Handler(FyersAuthHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, auth_code_callback=auth_code_callback, **kwargs)
        
        # Start server
        self.server = HTTPServer(('localhost', port), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"✅ Auth server started on port {port}")
    
    def stop_auth_server(self):
        """Stop the auth server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("✅ Auth server stopped")
    
    def generate_auth_url(self):
        """Generate the authentication URL."""
        try:
            # Initialize session model
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type=self.response_type,
                state=self.state,
                grant_type="authorization_code"
            )
            
            # Generate auth URL
            auth_url = session.generate_authcode()
            return auth_url, session
            
        except Exception as e:
            print(f"❌ Error generating auth URL: {e}")
            return None, None
    
    def open_browser_automated(self, auth_url):
        """Open browser automatically and wait for auth."""
        try:
            print(f"🔗 Opening authentication URL: {auth_url}")
            print("📋 IMPORTANT: After login, you'll be redirected to Fyers trade page.")
            print("🔍 Look for the auth code in the URL parameters.")
            print("📝 The URL will contain something like: auth_code=XXXXX&state=sample")
            print("📝 Copy only the auth code part (XXXXX) and paste it below:")
            
            webbrowser.open(auth_url, new=1)
            
            # Wait for manual input instead of automatic capture
            auth_code = input("\n🔑 Enter the auth code from the URL: ").strip()
            
            if auth_code:
                self.auth_code = auth_code
                print("✅ Authentication code captured!")
                return True
            else:
                print("❌ No auth code provided")
                return False
                
        except Exception as e:
            print(f"❌ Error in automated browser auth: {e}")
            return False
    
    def generate_access_token(self, session, auth_code):
        """Generate access token from auth code."""
        try:
            print("🔄 Generating access token...")
            
            # Set auth code in session
            session.set_token(auth_code)
            
            # Generate token
            response = session.generate_token()
            
            if 'access_token' in response:
                access_token = response['access_token']
                print("✅ Access token generated successfully!")
                
                # Update .env file
                self.update_env_file("FYERS_ACCESS_TOKEN", access_token)
                self.update_env_file("FYERS_AUTH_CODE", auth_code)
                
                return access_token
            else:
                print(f"❌ Failed to generate access token: {response}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating access token: {e}")
            return None
    
    def test_access_token(self, access_token):
        """Test the access token by fetching user profile."""
        try:
            print("🧪 Testing access token...")
            
            fyers = fyersModel.FyersModel(
                token=access_token,
                is_async=False,
                client_id=self.client_id,
                log_path=""
            )
            
            profile = fyers.get_profile()
            
            if 'code' in profile and profile['code'] == 200:
                print("✅ Access token test successful!")
                print(f"👤 Logged in as: {profile.get('data', {}).get('name', 'Unknown')}")
                print(f"🆔 User ID: {profile.get('data', {}).get('fy_id', 'Unknown')}")
                print(f"📧 Email: {profile.get('data', {}).get('email_id', 'Unknown')}")
                return True
            else:
                print(f"❌ Access token test failed: {profile}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing access token: {e}")
            return False
    
    def update_env_file(self, key, value):
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
    
    def check_token_validity(self, access_token):
        """Check if the current access token is still valid."""
        try:
            fyers = fyersModel.FyersModel(
                token=access_token,
                is_async=False,
                client_id=self.client_id,
                log_path=""
            )
            
            profile = fyers.get_profile()
            return 'code' in profile and profile['code'] == 200
            
        except Exception:
            return False
    
    def authenticate(self):
        """Perform complete authentication flow."""
        try:
            print("🚀 Starting Fyers Authentication")
            print("=" * 50)
            
            # Generate auth URL
            auth_url, session = self.generate_auth_url()
            if not auth_url:
                return False
            
            # Open browser and get auth code manually
            if not self.open_browser_automated(auth_url):
                return False
            
            # Generate access token
            access_token = self.generate_access_token(session, self.auth_code)
            if not access_token:
                return False
            
            # Test the token
            if not self.test_access_token(access_token):
                return False
            
            print("✅ Authentication completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in authentication flow: {e}")
            return False

def main():
    """Main function to run the authentication utility."""
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