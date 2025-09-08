#!/usr/bin/env python3
"""
Automated Fyers Authentication with automatic auth code capture
"""

import os
import sys
import time
import webbrowser
import logging
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from fyers_apiv3 import fyersModel
from src.config.settings import (
    FYERS_CLIENT_ID, 
    FYERS_SECRET_KEY, 
    FYERS_REDIRECT_URI,
    FYERS_RESPONSE_TYPE,
    FYERS_GRANT_TYPE,
    FYERS_STATE,
    setup_logging
)

# Set up logger
logger = setup_logging('fyers_auth')

class AuthCodeHandler(BaseHTTPRequestHandler):
    """HTTP handler to capture auth code from redirect URL"""
    
    auth_code = None
    server_stopped = False
    
    def do_GET(self):
        """Handle GET request and extract auth code from URL"""
        try:
            # Parse the URL
            parsed_url = urlparse(self.path)
            
            # Extract query parameters
            query_params = parse_qs(parsed_url.query)
            
            # Check if auth_code is present
            if 'auth_code' in query_params:
                AuthCodeHandler.auth_code = query_params['auth_code'][0]
                logger.info(f"✅ Authentication code captured automatically: {AuthCodeHandler.auth_code[:20]}...")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                success_html = """
                <html>
                <head><title>Authentication Successful</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: green;">✅ Authentication Successful!</h1>
                    <p>You can close this window now.</p>
                    <p>The trading system will continue automatically.</p>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode())
                
                # Stop the server after capturing the code
                def stop_server():
                    time.sleep(1)  # Give time for response to be sent
                    AuthCodeHandler.server_stopped = True
                    self.server.shutdown()
                
                threading.Thread(target=stop_server, daemon=True).start()
                
            else:
                # Send error response
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = """
                <html>
                <head><title>Authentication Error</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: red;">❌ Authentication Error</h1>
                    <p>No auth code found in URL.</p>
                    <p>Please try again.</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass

def start_auth_server(port=8080):
    """Start HTTP server to capture auth code"""
    try:
        server = HTTPServer(('localhost', port), AuthCodeHandler)
        logger.info(f"🌐 Auth server started on port {port}")
        
        # Run server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        return server
        
    except Exception as e:
        logger.error(f"Failed to start auth server: {e}")
        return None

def check_and_refresh_token():
    """Check if token exists and is valid, refresh if needed"""
    try:
        # Check if token exists in .env
        if not os.path.exists('.env'):
            logger.warning("⚠️ .env file not found")
            return False
        
        # Read current token
        with open('.env', 'r') as f:
            env_content = f.read()
        
        # Check if access token exists
        if 'FYERS_ACCESS_TOKEN=' not in env_content:
            logger.info("🔑 No access token found, need to authenticate")
            return False
        
        # Extract token
        for line in env_content.split('\n'):
            if line.startswith('FYERS_ACCESS_TOKEN='):
                token = line.split('=')[1].strip()
                if token and token != 'None':
                    logger.info("🔑 Access token found, checking validity...")
                    
                    # Test token validity
                    try:
                        fyers = fyersModel.FyersModel(
                            token=token,
                            is_async=False,
                            client_id=FYERS_CLIENT_ID,
                            log_path=""
                        )
                        
                        profile = fyers.get_profile()
                        if 'code' in profile and profile['code'] == 200:
                            logger.info("✅ Token is valid")
                            return True
                        else:
                            logger.warning("⚠️ Token has expired")
                            return False
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Token validation failed: {e}")
                        return False
        
        logger.info("🔑 No valid access token found")
        return False
        
    except Exception as e:
        logger.error(f"Error checking token: {e}")
        return False

def main():
    """Main authentication function"""
    
    # Check if token is valid
    if check_and_refresh_token():
        return True
    
    
    # Start auth server
    auth_server = start_auth_server()
    if not auth_server:
        return False
    
    try:
        # Initialize session
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type=FYERS_RESPONSE_TYPE,
            state=FYERS_STATE,
            secret_key=FYERS_SECRET_KEY,
            grant_type=FYERS_GRANT_TYPE
        )
        
        # Generate auth URL
        auth_url = session.generate_authcode()
        
        
        # Open browser automatically
        webbrowser.open(auth_url, new=1)
        
        
        # Wait for auth code to be captured
        timeout = 120  # 2 minutes timeout
        start_time = time.time()
        
        while not AuthCodeHandler.auth_code and not AuthCodeHandler.server_stopped:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.1)
        
        if not AuthCodeHandler.auth_code:
            return False
        
        # Generate access token
        
        session.set_token(AuthCodeHandler.auth_code)
        response = session.generate_token()
        
        if 'access_token' in response:
            access_token = response['access_token']
            
            # Update .env file
            update_env_file(access_token, AuthCodeHandler.auth_code)
            
            # Test the token
            test_token(access_token)
            
            return True
            
        else:
            return False
            
    except Exception as e:
        return False
    
    finally:
        # Stop auth server
        if auth_server:
            auth_server.shutdown()

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

def test_token(access_token):
    """Test if the access token works"""
    try:
        fyers = fyersModel.FyersModel(
            token=access_token,
            is_async=False,
            client_id=FYERS_CLIENT_ID,
            log_path=""
        )
        
        profile = fyers.get_profile()
        if 'code' in profile and profile['code'] == 200:
            
            # Get user info
            if 'data' in profile:
                user_data = profile['data']
            
        else:
            
    except Exception as e:

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 