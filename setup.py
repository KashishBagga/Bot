#!/usr/bin/env python
"""
Setup script for the trading bot application.
This script helps set up the necessary directories and configuration files.
"""
import os
import shutil
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "logs",
        "data",
        "src/strategies"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        shutil.copy(".env.example", ".env")
        print("✓ Created .env file from .env.example")
        print("  Please edit .env file with your API credentials")
    elif not os.path.exists(".env.example"):
        print("⚠ .env.example file not found, creating basic .env file")
        with open(".env", "w") as f:
            f.write("""# Fyers API credentials
FYERS_CLIENT_ID=
FYERS_SECRET_KEY=
FYERS_REDIRECT_URI=
FYERS_RESPONSE_TYPE=code
FYERS_GRANT_TYPE=authorization_code
FYERS_STATE=state
FYERS_AUTH_CODE=

# Telegram configuration
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Database configuration
DATABASE_PATH=trading_signals.db

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
""")
        print("✓ Created basic .env file")
        print("  Please edit .env file with your API credentials")
    else:
        print("✓ .env file already exists")

def setup_virtual_environment():
    """Set up a virtual environment and install dependencies."""
    venv_dir = "venv"
    
    if os.path.exists(venv_dir):
        print(f"✓ Virtual environment already exists at {venv_dir}")
        return
    
    try:
        # Create the virtual environment
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print(f"✓ Created virtual environment at {venv_dir}")
        
        # Get the path to the pip executable in the virtual environment
        if sys.platform == "win32":
            pip_executable = os.path.join(venv_dir, "Scripts", "pip")
        else:
            pip_executable = os.path.join(venv_dir, "bin", "pip")
        
        # Upgrade pip
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
        print("✓ Upgraded pip in virtual environment")
        
        # Install dependencies
        if os.path.exists("requirements.txt"):
            subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
            print("✓ Installed dependencies from requirements.txt")
        else:
            print("⚠ requirements.txt not found, skipping dependency installation")
    
    except subprocess.CalledProcessError as e:
        print(f"⚠ Error setting up virtual environment: {e}")
        print("  Please set up the virtual environment manually.")
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        print("  Please set up the virtual environment manually.")

def test_imports():
    """Test importing key modules to verify installation."""
    try:
        # Store the original sys.path
        original_path = sys.path.copy()
        
        # Add the current directory to sys.path
        sys.path.insert(0, os.getcwd())
        
        print("Testing imports...")
        
        # Try importing key modules
        import pandas
        import ta
        import requests
        from src.config.settings import setup_logging
        from src.models.database import db
        from src.core.indicators import indicators
        
        print("✓ All imports successful!")
        
        # Restore the original sys.path
        sys.path = original_path
        
    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print("  Some dependencies may be missing. Please install them manually.")
    except Exception as e:
        print(f"⚠ Unexpected error during import testing: {e}")

def check_strategies():
    """Check if strategies are available."""
    strategies_dir = Path("src/strategies")
    strategies = list(strategies_dir.glob("*.py"))
    if not strategies:
        print("⚠ No strategy modules found in src/strategies/")
        print("  Please make sure to create strategy modules.")
    else:
        print(f"✓ Found {len(strategies)} strategy modules:")
        for strategy in strategies:
            print(f"  - {strategy.name}")

def main():
    """Run the setup process."""
    print("Setting up trading bot environment...")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Set up virtual environment
    setup_virtual_environment()
    
    # Test imports
    test_imports()
    
    # Check strategies
    check_strategies()
    
    print("\nSetup complete!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    
    print("\nTo run the trading bot in real-time mode:")
    print("  python -m src.main --mode realtime")
    
if __name__ == "__main__":
    main() 