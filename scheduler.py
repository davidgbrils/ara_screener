"""
Auto-scheduler for ARA Bot
Runs bot automatically at 9:30 AM Jakarta time (market open)
Timezone-aware scheduler with data freshness validation
"""

import schedule
import time
from datetime import datetime, timedelta
import pytz
from main import ARABot
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Jakarta timezone
JAKARTA_TZ = pytz.timezone('Asia/Jakarta')

def is_market_open() -> bool:
    """
    Check if market is open (Monday-Friday, 9:30-16:00 WIB)
    
    Returns:
        True if market is open
    """
    now_jakarta = datetime.now(JAKARTA_TZ)
    weekday = now_jakarta.weekday()  # 0=Monday, 6=Sunday
    hour = now_jakarta.hour
    minute = now_jakarta.minute
    
    # Market is open Monday-Friday, 9:30-16:00
    if weekday < 5:  # Monday to Friday
        if hour == 9 and minute >= 30:
            return True
        if 10 <= hour < 16:
            return True
    
    return False

def run_bot():
    """Run ARA Bot scan with fresh data"""
    try:
        now_jakarta = datetime.now(JAKARTA_TZ)
        
        logger.info("=" * 60)
        logger.info("AUTO-SCHEDULED SCAN STARTED")
        logger.info(f"Time: {now_jakarta.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"Market Open: {is_market_open()}")
        logger.info("=" * 60)
        
        # Force refresh data to get today's data
        bot = ARABot()
        results = bot.run(force_refresh=True)
        
        logger.info("=" * 60)
        logger.info("AUTO-SCHEDULED SCAN COMPLETED")
        logger.info(f"Results: {len(results)} signals found")
        logger.info(f"Time: {datetime.now(JAKARTA_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in scheduled scan: {e}", exc_info=True)

def get_next_market_open():
    """
    Calculate next market open time (9:30 AM Jakarta time, Monday-Friday)
    
    Returns:
        Next market open datetime
    """
    now_jakarta = datetime.now(JAKARTA_TZ)
    
    # Target time: 9:30 AM
    target_hour = 9
    target_minute = 30
    
    # Calculate next occurrence
    next_run = now_jakarta.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # If already past 9:30 today, move to tomorrow
    if now_jakarta.time() > next_run.time():
        next_run += timedelta(days=1)
    
    # Skip weekends
    while next_run.weekday() >= 5:  # Saturday or Sunday
        next_run += timedelta(days=1)
    
    return next_run

def setup_scheduler():
    """Setup daily scheduler at 9:30 AM Jakarta time"""
    # Calculate next market open
    next_run = get_next_market_open()
    
    # Schedule at 15:00 PM Jakarta time (using system time conversion)
    # Note: schedule library uses system timezone, so we need to convert
    system_tz = datetime.now().astimezone().tzinfo
    next_run_system = next_run.astimezone(system_tz)
    
    # Schedule for next market open
    schedule.every().day.at("15:00").do(run_bot)
    
    logger.info("Scheduler configured: Daily at 15:00 PM Jakarta time")
    logger.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info("Bot will run automatically when market opens")
    logger.info("Note: Ensure system timezone is correct or use --timezone flag")

def run_scheduler():
    """Run scheduler loop with timezone awareness"""
    setup_scheduler()
    
    logger.info("Scheduler started. Waiting for next run...")
    next_run = get_next_market_open()
    logger.info(f"Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    try:
        while True:
            # Check if it's time to run (9:30 AM Jakarta time)
            now_jakarta = datetime.now(JAKARTA_TZ)
            
            # Run if it's 15:00 PM Jakarta time
            if (now_jakarta.hour == 15 and now_jakarta.minute == 0 and 
                now_jakarta.second < 60):
                run_bot()
                # Wait a bit to avoid multiple runs
                time.sleep(120)  # Wait 2 minutes
            
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds for more accuracy
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)

if __name__ == "__main__":
    # Check if we should run immediately or wait for schedule
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        # Run immediately
        logger.info("Running bot immediately (--now flag)")
        run_bot()
    else:
        # Start scheduler
        run_scheduler()

