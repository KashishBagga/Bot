#!/usr/bin/env python3
"""
Historical Data System Setup & Validation
Verifies system is ready for 20-year data fetching and backtesting
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List
import json

class HistoricalDataSystemSetup:
    """Setup and validation for historical data system"""
    
    def __init__(self):
        self.working_dir = Path.cwd()
        self.env_file = self.working_dir / '.env'
        self.required_files = [
            'fetch_20_year_historical_data.py',
            'validate_historical_data.py',
            'backtest_20_year_engine.py',
            'all_strategies.py'
        ]
        self.required_packages = [
            'pandas',
            'numpy',
            'requests',
            'python-dotenv',
            'pyarrow',  # For parquet files
            'ta-lib',   # Technical analysis
            'backoff'   # For retry logic
        ]
        self.setup_report = {
            'timestamp': datetime.now().isoformat(),
            'working_directory': str(self.working_dir),
            'system_ready': False,
            'checks': {}
        }
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        compatible = version.major >= required_major and version.minor >= required_minor
        
        self.setup_report['checks']['python_version'] = {
            'current': f"{version.major}.{version.minor}.{version.micro}",
            'required': f">= {required_major}.{required_minor}",
            'compatible': compatible,
            'status': 'PASS' if compatible else 'FAIL'
        }
        
        print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}", end='')
        if compatible:
            print(" ✅")
        else:
            print(" ❌ (Requires Python 3.8+)")
        
        return compatible
    
    def check_required_files(self) -> bool:
        """Check if all required files exist"""
        missing_files = []
        
        for file_name in self.required_files:
            file_path = self.working_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        all_present = len(missing_files) == 0
        
        self.setup_report['checks']['required_files'] = {
            'total_required': len(self.required_files),
            'present': len(self.required_files) - len(missing_files),
            'missing': missing_files,
            'all_present': all_present,
            'status': 'PASS' if all_present else 'FAIL'
        }
        
        print(f"📁 Required Files: {len(self.required_files) - len(missing_files)}/{len(self.required_files)}", end='')
        if all_present:
            print(" ✅")
        else:
            print(f" ❌ (Missing: {', '.join(missing_files)})")
        
        return all_present
    
    def check_environment_config(self) -> bool:
        """Check environment configuration"""
        env_exists = self.env_file.exists()
        required_vars = [
            'FYERS_CLIENT_ID',
            'FYERS_SECRET_KEY',
            'FYERS_ACCESS_TOKEN'
        ]
        
        missing_vars = []
        if env_exists:
            try:
                with open(self.env_file, 'r') as f:
                    env_content = f.read()
                
                for var in required_vars:
                    if f"{var}=" not in env_content or f"{var}=" + " " in env_content:
                        missing_vars.append(var)
            except Exception as e:
                env_exists = False
        
        config_ready = env_exists and len(missing_vars) == 0
        
        self.setup_report['checks']['environment_config'] = {
            'env_file_exists': env_exists,
            'required_variables': required_vars,
            'missing_variables': missing_vars,
            'config_ready': config_ready,
            'status': 'PASS' if config_ready else 'FAIL'
        }
        
        print(f"🔑 Environment Config: ", end='')
        if config_ready:
            print("✅")
        elif not env_exists:
            print("❌ (.env file not found)")
        else:
            print(f"❌ (Missing: {', '.join(missing_vars)})")
        
        return config_ready
    
    def check_packages(self) -> bool:
        """Check required Python packages"""
        missing_packages = []
        
        for package in self.required_packages:
            try:
                # Handle special cases
                if package == 'ta-lib':
                    import talib
                elif package == 'python-dotenv':
                    import dotenv
                elif package == 'pyarrow':
                    import pyarrow
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        all_installed = len(missing_packages) == 0
        
        self.setup_report['checks']['packages'] = {
            'total_required': len(self.required_packages),
            'installed': len(self.required_packages) - len(missing_packages),
            'missing': missing_packages,
            'all_installed': all_installed,
            'status': 'PASS' if all_installed else 'FAIL'
        }
        
        print(f"📦 Python Packages: {len(self.required_packages) - len(missing_packages)}/{len(self.required_packages)}", end='')
        if all_installed:
            print(" ✅")
        else:
            print(f" ❌ (Missing: {', '.join(missing_packages)})")
        
        return all_installed
    
    def check_disk_space(self) -> bool:
        """Check available disk space for data storage"""
        try:
            statvfs = os.statvfs(str(self.working_dir))
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            available_gb = available_bytes / (1024**3)
            
            # Estimate 20-year data needs: ~50GB for comprehensive data
            required_gb = 50
            sufficient_space = available_gb >= required_gb
            
            self.setup_report['checks']['disk_space'] = {
                'available_gb': round(available_gb, 2),
                'required_gb': required_gb,
                'sufficient': sufficient_space,
                'status': 'PASS' if sufficient_space else 'WARN'
            }
            
            print(f"💾 Disk Space: {available_gb:.1f}GB available", end='')
            if sufficient_space:
                print(" ✅")
            else:
                print(f" ⚠️ (Recommended: {required_gb}GB+)")
            
            return sufficient_space
            
        except Exception as e:
            print(f"💾 Disk Space: Unable to check ⚠️")
            return True  # Don't fail setup for this
    
    def check_database_access(self) -> bool:
        """Check database accessibility"""
        try:
            import sqlite3
            db_path = self.working_dir / 'trading_signals.db'
            
            # Try to connect and create a test table
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            db_accessible = True
            
            self.setup_report['checks']['database_access'] = {
                'database_path': str(db_path),
                'accessible': db_accessible,
                'existing_tables': len(tables),
                'status': 'PASS'
            }
            
            print(f"🗄️ Database Access: {len(tables)} tables ✅")
            return True
            
        except Exception as e:
            self.setup_report['checks']['database_access'] = {
                'accessible': False,
                'error': str(e),
                'status': 'FAIL'
            }
            print(f"🗄️ Database Access: ❌ ({e})")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        directories = [
            'historical_data_20yr',
            'backtest_results',
            'logs'
        ]
        
        created_dirs = []
        for dir_name in directories:
            dir_path = self.working_dir / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_name)
                except Exception as e:
                    print(f"📂 Failed to create {dir_name}: {e}")
                    return False
        
        self.setup_report['checks']['directories'] = {
            'required': directories,
            'created': created_dirs,
            'status': 'PASS'
        }
        
        if created_dirs:
            print(f"📂 Created directories: {', '.join(created_dirs)} ✅")
        else:
            print(f"📂 All directories exist ✅")
        
        return True
    
    def generate_installation_commands(self) -> List[str]:
        """Generate installation commands for missing packages"""
        missing_packages = self.setup_report['checks'].get('packages', {}).get('missing', [])
        
        if not missing_packages:
            return []
        
        commands = [
            "# Install missing Python packages:",
            f"pip install {' '.join(missing_packages)}",
            "",
            "# Note: TA-Lib may require additional system dependencies:",
            "# On Ubuntu/Debian: sudo apt-get install ta-lib",
            "# On macOS: brew install ta-lib",
            "# On Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/",
        ]
        
        return commands
    
    def run_comprehensive_check(self) -> bool:
        """Run all system checks"""
        print("🔍 HISTORICAL DATA SYSTEM SETUP VALIDATION")
        print("=" * 60)
        
        checks = [
            self.check_python_version(),
            self.check_required_files(),
            self.check_environment_config(),
            self.check_packages(),
            self.check_disk_space(),
            self.check_database_access(),
            self.create_directories()
        ]
        
        # Calculate overall status
        critical_checks = checks[:4]  # First 4 are critical
        system_ready = all(critical_checks)
        
        self.setup_report['system_ready'] = system_ready
        
        print("\n" + "=" * 60)
        if system_ready:
            print("✅ SYSTEM READY FOR HISTORICAL DATA OPERATIONS")
            print("\n🚀 You can now run:")
            print("   python3 fetch_20_year_historical_data.py")
            print("   python3 validate_historical_data.py")
            print("   python3 backtest_20_year_engine.py")
        else:
            print("❌ SYSTEM NOT READY - ISSUES FOUND")
            
            # Show installation commands if needed
            install_commands = self.generate_installation_commands()
            if install_commands:
                print("\n📋 INSTALLATION COMMANDS:")
                for cmd in install_commands:
                    print(f"   {cmd}")
        
        return system_ready
    
    def save_setup_report(self) -> str:
        """Save setup report to file"""
        filename = f"setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.setup_report, f, indent=2)
        
        print(f"\n📄 Setup report saved to: {filename}")
        return filename


def main():
    """Main execution function"""
    setup = HistoricalDataSystemSetup()
    
    system_ready = setup.run_comprehensive_check()
    setup.save_setup_report()
    
    return 0 if system_ready else 1


if __name__ == "__main__":
    exit(main()) 