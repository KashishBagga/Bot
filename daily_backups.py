#!/usr/bin/env python3
"""
Daily Backups - Critical for Data Protection
Protect against corruption and infrastructure failures
"""

def add_daily_backups():
    """Add daily backups to live_paper_trading.py"""
    
    
    with open('live_paper_trading.py', 'r') as f:
        content = f.read()
    
    # Add required imports
    import re
    content = re.sub(
        r'(import requests)',
        r'\1\nimport shutil\nfrom pathlib import Path',
        content
    )
    
    # Add the backup methods
    backup_methods = '''
    def _create_daily_backup(self):
        """Create daily backup of database - critical for data protection."""
        try:
            logger.info("💾 Starting daily database backup...")
            
            # Get current timestamp for backup filename
            timestamp = self.now_kolkata().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"trading_backup_{timestamp}.db"
            
            # Create backup directory if it doesn't exist
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Copy database file
            db_path = Path("unified_trading.db")
            backup_path = backup_dir / backup_filename
            
            if db_path.exists():
                shutil.copy2(db_path, backup_path)
                logger.info(f"✅ Database backup created: {backup_path}")
                
                # Keep only last 30 days of backups
                self._cleanup_old_backups(backup_dir)
                
                # Send backup notification
                self._send_alert(f"💾 Daily backup completed: {backup_filename}")
                
            else:
                logger.warning("⚠️ Database file not found for backup")
                
        except Exception as e:
            logger.error(f"❌ Error creating backup: {e}")
            self._send_alert(f"🚨 CRITICAL: Backup failed: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path, days_to_keep: int = 30):
        """Clean up old backup files to save space."""
        try:
            import time
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            
            for backup_file in backup_dir.glob("trading_backup_*.db"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.debug(f"🗑️ Deleted old backup: {backup_file.name}")
                    
        except Exception as e:
            logger.error(f"❌ Error cleaning up backups: {e}")
    
    def _setup_backup_scheduler(self):
        """Setup daily backup scheduler."""
        try:
            import schedule
            import time
            
            # Schedule daily backup at 11:30 PM IST
            schedule.every().day.at("23:30").do(self._create_daily_backup)
            logger.info("⏰ Daily backup scheduled for 11:30 PM IST")
            
        except Exception as e:
            logger.error(f"❌ Error setting up backup scheduler: {e}")
    
    def _run_backup_scheduler(self):
        """Run backup scheduler in background thread."""
        try:
            import schedule
            import time
            
            while not self._stop_event.is_set():
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"❌ Error in backup scheduler: {e}")
    '''
    
    # Insert before _validate_production_requirements
    content = re.sub(
        r'(def _validate_production_requirements\(self\):)',
        backup_methods + r'\n    \1',
        content
    )
    
    # Add backup scheduler to start_trading
    content = re.sub(
        r'(self\.start_health_server\(\))',
        r'\1\n        \n        # Setup daily backup scheduler\n        self._setup_backup_scheduler()\n        \n        # Start backup scheduler thread\n        backup_thread = threading.Thread(target=self._run_backup_scheduler, daemon=True)\n        backup_thread.start()',
        content
    )
    
    with open('live_paper_trading.py', 'w') as f:
        f.write(content)
    

if __name__ == "__main__":
    add_daily_backups()
