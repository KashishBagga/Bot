import time
import pandas as pd
import ta
import gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials
from fyers_apiv3 import fyersModel
import webbrowser
import logging
import math
import pytz
import schedule
import threading
from dotenv import load_dotenv
import os
from db import log_trade_sql


# Load environment variables
load_dotenv()

# Get Fyers API credentials from environment variables
redirect_uri = os.getenv("FYERS_REDIRECT_URI")
client_id = os.getenv("FYERS_CLIENT_ID")
secret_key = os.getenv("FYERS_SECRET_KEY")
grant_type = os.getenv("FYERS_GRANT_TYPE")
response_type = os.getenv("FYERS_RESPONSE_TYPE")
state = os.getenv("FYERS_STATE")
auth_code = os.getenv("FYERS_AUTH_CODE")

# Initialize Fyers session
appSession = fyersModel.SessionModel(
    client_id=client_id,
    redirect_uri=redirect_uri,
    response_type=response_type,
    state=state,
    secret_key=secret_key,
    grant_type=grant_type
)

generateTokenUrl = appSession.generate_authcode()
print((generateTokenUrl))  
# webbrowser.open(generateTokenUrl,new=1)

appSession.set_token(auth_code)
response = appSession.generate_token()

try: 
    access_token = response["access_token"]
except Exception as e:
    print(e,response)  

fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")
# print(fyers.get_profile()) 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_pending_signals(sheet, fyers):
    try:
        pending_records = sheet.get_all_records()
        headers = sheet.row_values(1)

        for row_num, row in enumerate(pending_records, start=2):
            if row.get("Outcome") != "Pending":
                continue

            index = row.get("Index")
            timestamp_str = row.get("Timestamp")
            strike = int(row.get("Strike Price", 0))
            stoploss = int(row.get("Stop Loss", 0))
            target = int(row.get("Target", 0))
            target2 = int(row.get("Target 2", 0))
            target3 = int(row.get("Target 3", 0))
            entry_price = float(row.get("Price", 0.0))

            entry_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            utc_entry_time = pytz.timezone("Asia/Kolkata").localize(entry_time).astimezone(pytz.utc)
            start_time = utc_entry_time + timedelta(minutes=1)
            end_time = start_time + timedelta(minutes=24)

            symbol = "NSE:NIFTY50-INDEX" if index == "NIFTY50" else "NSE:NIFTYBANK-INDEX"
            data = {
                "symbol": symbol,
                "resolution": "1",
                "date_format": "1",
                "range_from": start_time.strftime("%Y-%m-%d"),
                "range_to": end_time.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }

            res = fyers.history(data)
            candles = res.get("candles", [])
            if not candles:
                continue

            outcome = "Pending"
            pnl = 0
            targets_hit = 0
            stoploss_count = 0
            exit_ts1 = exit_ts2 = exit_ts3 = ""

            for c in candles:
                high = c[2]
                low = c[3]
                ts = datetime.fromtimestamp(c[0], pytz.utc).astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

                if low <= (entry_price - stoploss):
                    outcome = "Stoploss Hit"
                    pnl = -stoploss
                    stoploss_count = 1
                    break
                if targets_hit == 0 and high >= (entry_price + target):
                    exit_ts1 = ts
                    pnl += target
                    targets_hit += 1
                elif targets_hit == 1 and high >= (entry_price + target2):
                    exit_ts2 = ts
                    pnl += target2
                    targets_hit += 1
                elif targets_hit == 2 and high >= (entry_price + target3):
                    exit_ts3 = ts
                    pnl += target3
                    targets_hit += 1

            if targets_hit:
                outcome = f"{targets_hit} Targets Hit"

            updates = {
                headers.index("Outcome") + 1: outcome,
                headers.index("P&L (1 Lot)") + 1: pnl,
                headers.index("Targets Hit") + 1: targets_hit,
                headers.index("Stoploss Count") + 1: stoploss_count,
                headers.index("Exit TS 1") + 1: exit_ts1,
                headers.index("Exit TS 2") + 1: exit_ts2,
                headers.index("Exit TS 3") + 1: exit_ts3,
            }

            for col_idx, value in updates.items():
                sheet.update_cell(row_num, col_idx, value)

    except Exception as e:
        print(f"❌ Error in update_pending_signals: {e}")


def start_updater_thread(sheet, fyers):
    def job():
        print("⏱️ Running outcome updater...")
        update_pending_signals(sheet, fyers)

    # Schedule every 15 minutes
    schedule.every(2).minutes.do(job)

    def run_schedule():
        while True:
            schedule.run_pending()
            time.sleep(1)

    updater_thread = threading.Thread(target=run_schedule, daemon=True)
    updater_thread.start()
    print("✅ Updater thread started in background.")


def safe_float(val):
    """Safely convert value to float and handle edge cases"""
    try:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return round(float(val), 2)
        return 0.0
    except (ValueError, TypeError):
        return 0.0



# === 🔐 Google Sheet Setup ===
def setup_sheet():
    """Setup Google Sheets connection and create required sheets"""
    try:
        # Authorize with Google Sheets
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name("analysis-test-458117-64b56a994c55.json", scope)
        client = gspread.authorize(credentials)
        
        # Open the spreadsheet
        spreadsheet = client.open("Trading Signals Tracker")
        
        # Setup main signal sheet
        signal_headers = [
            "Timestamp", "Index", "Signal", "Strike Price", "Stop Loss", "Target",
            "Target 2", "Target 3", "Exit TS 1", "Exit TS 2", "Exit TS 3",
            "Price", "RSI", "MACD", "MACD Signal", "EMA 20", "ATR",
            "Outcome", "RSI Reason", "MACD Reason", "Price Reason",
            "Confidence", "Trade Type", "Option Chain Confirmation", "P&L (1 Lot)",
            "Targets Hit", "Stoploss Count", "Failure Reason"
        ]
        
        try:
            sheet = spreadsheet.worksheet("Signal")
            if sheet.row_values(1) != signal_headers:
                sheet.clear()
                sheet.append_row(signal_headers)
                logger.info("✅ Headers added to Signal Sheet")
        except gspread.exceptions.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(title="Signal", rows="1000", cols="27")
            sheet.append_row(signal_headers)
            logger.info("✅ Created 'Signal' sheet with headers")

        # Setup summary sheet
        try:
            summary_sheet = spreadsheet.worksheet("Summary")
            if summary_sheet.row_values(1) != ["Date", "Index", "P&L (1 Lot)", "Targets Hit", "Stoplosses"]:
                summary_sheet.clear()
                summary_sheet.append_row(["Date", "Index", "P&L (1 Lot)", "Targets Hit", "Stoplosses"])
                logger.info("✅ Headers added to Summary Sheet")
        except gspread.exceptions.WorksheetNotFound:
            summary_sheet = spreadsheet.add_worksheet(title="Summary", rows="1000", cols="5")
            summary_sheet.append_row(["Date", "Index", "P&L (1 Lot)", "Targets Hit", "Stoplosses"])
            logger.info("✅ Created 'Summary' sheet with headers")

        # Setup insights sheet
        try:
            insights_sheet = spreadsheet.worksheet("Insights")
            if insights_sheet.row_values(1) != ["Index", "Total Trades", "Accuracy %", "Total P&L", "Avg Profit", "Avg Loss", "Win Ratio"]:
                insights_sheet.clear()
                insights_sheet.append_row(["Index", "Total Trades", "Accuracy %", "Total P&L", "Avg Profit", "Avg Loss", "Win Ratio"])
                logger.info("✅ Headers added to Insights Sheet")
        except gspread.exceptions.WorksheetNotFound:
            insights_sheet = spreadsheet.add_worksheet(title="Insights", rows="1000", cols="7")
            insights_sheet.append_row(["Index", "Total Trades", "Accuracy %", "Total P&L", "Avg Profit", "Avg Loss", "Win Ratio"])
            logger.info("✅ Created 'Insights' sheet with headers")

        return sheet, summary_sheet, insights_sheet
        
    except Exception as e:
        logger.error(f"Error setting up Google Sheets: {e}")
        return None, None, None

# === 🧠 Analyze Signal ===
def analyze_signal(df):
    """Analyze market data and generate trading signals with extremely loose criteria"""
    try:
        df = df.copy()
        
        # Calculate indicators with safe float handling
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().apply(safe_float)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd().apply(safe_float)
        df['macd_signal'] = macd.macd_signal().apply(safe_float)
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().apply(safe_float)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().apply(safe_float)

        latest = df.iloc[-1]
        
        # Safely extract values with default fallbacks
        rsi = safe_float(latest.get('rsi', 0))
        macd_val = safe_float(latest.get('macd', 0))
        macd_signal = safe_float(latest.get('macd_signal', 0))
        ema_20 = safe_float(latest.get('ema_20', 0))
        price = safe_float(latest.get('close', 0))
        atr = safe_float(latest.get('atr', 0))

        signal = "None"
        confidence = "Low"
        rsi_reason = ""
        macd_reason = ""
        price_reason = ""
        trade_type = "Intraday"
        option_chain_confirmation = "No"

        # Extremely loose criteria for signal generation
        # For CALL signals:
        if (rsi > 50 or  # Extremely loose from 55
            macd_val > macd_signal or  # Extremely loose from +1
            price > ema_20):  # Extremely loose from 1.0001
            signal = "BUY CALL"
            rsi_reason = f"RSI {rsi:.2f} > 50"
            macd_reason = f"MACD {macd_val:.2f} > MACD Signal ({macd_signal:.2f})"
            price_reason = f"Price {price:.2f} > EMA {ema_20:.2f}"
            confidence = "High" if rsi > 55 else "Medium"  # Extremely loose from 60
            option_chain_confirmation = "Yes" if confidence == "High" else "No"

        # For PUT signals:
        elif (rsi < 50 or  # Extremely loose from 45
              macd_val < macd_signal or  # Extremely loose from -1
              price < ema_20):  # Extremely loose from 0.9999
            signal = "BUY PUT"
            rsi_reason = f"RSI {rsi:.2f} < 50"
            macd_reason = f"MACD {macd_val:.2f} < MACD Signal ({macd_signal:.2f})"
            price_reason = f"Price {price:.2f} < EMA {ema_20:.2f}"
            confidence = "High" if rsi < 45 else "Medium"  # Extremely loose from 40
            option_chain_confirmation = "Yes" if confidence == "High" else "No"

        # Log the signal generation details
        logger.info(f"Signal Analysis - RSI: {rsi:.2f}, MACD: {macd_val:.2f}, Price: {price:.2f}, EMA20: {ema_20:.2f}")
        logger.info(f"Generated Signal: {signal} with Confidence: {confidence}")

        return {
            "signal": signal,
            "price": price,
            "rsi": rsi,
            "macd": macd_val,
            "macd_signal": macd_signal,
            "ema_20": ema_20,
            "atr": atr,
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": trade_type,
            "option_chain_confirmation": option_chain_confirmation
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_signal: {e}")
        return {
            "signal": "None",
            "price": 0.0,
            "rsi": 0.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "ema_20": 0.0,
            "atr": 0.0,
            "confidence": "Low",
            "rsi_reason": "",
            "macd_reason": "",
            "price_reason": "",
            "trade_type": "Intraday",
            "option_chain_confirmation": "No"
        }

# === 📊 Fetch 1-minute candles ===
def fetch_candles(fyers, symbol):
    data = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": datetime.now().strftime("%Y-%m-%d"),
        "range_to": datetime.now().strftime("%Y-%m-%d"),
        "cont_flag": "1"
    }
    res = fyers.history(data)
    candles = res.get("candles")
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# === 📈 Log valid trade ===
def log_trade(sheet, index_name, signal_data):
    """Log trade to Google Sheets with safe value handling"""
    try:
        # Calculate targets based on ATR
        atr = signal_data.get('atr', 0)
        stop_loss = int(round(atr))
        target = int(round(1.5 * atr))
        target2 = int(round(2.0 * atr))
        target3 = int(round(2.5 * atr))

        # Get current time in IST
        current_time = datetime.now()
        ist_time = current_time.astimezone(pytz.timezone('Asia/Kolkata'))
        
        # Calculate execution time (next 5-minute candle)
        execution_time = ist_time.replace(minute=(ist_time.minute // 5) * 5, second=0, microsecond=0)
        execution_time = execution_time + timedelta(minutes=5)
        
        # If execution time is after market hours, set to next market open
        if execution_time.hour >= 15 and execution_time.minute >= 15:
            execution_time = execution_time + timedelta(days=1)
            execution_time = execution_time.replace(hour=9, minute=15, second=0)
        elif execution_time.hour < 9 or (execution_time.hour == 9 and execution_time.minute < 15):
            execution_time = execution_time.replace(hour=9, minute=15, second=0)
        
        # Set exit time to market close (3:15 PM) on the same day
        exit_time = execution_time.replace(hour=15, minute=15, second=0)
        if execution_time > exit_time:
            exit_time = exit_time + timedelta(days=1)

        row = [
            execution_time.strftime("%Y-%m-%d %H:%M:%S"),
            index_name,
            signal_data.get('signal', 'None'),
            int(round(signal_data.get('price', 0) / 50) * 50),  # Strike price
            stop_loss,
            target,
            target2,
            target3,
            "",  # Exit TS 1
            "",  # Exit TS 2
            "",  # Exit TS 3
            safe_float(signal_data.get('price', 0)),
            safe_float(signal_data.get('rsi', 0)),
            safe_float(signal_data.get('macd', 0)),
            safe_float(signal_data.get('macd_signal', 0)),
            safe_float(signal_data.get('ema_20', 0)),
            safe_float(signal_data.get('atr', 0)),
            "Pending",  # Outcome
            signal_data.get('rsi_reason', ''),
            signal_data.get('macd_reason', ''),
            signal_data.get('price_reason', ''),
            signal_data.get('confidence', 'Low'),
            signal_data.get('trade_type', 'Intraday'),
            signal_data.get('option_chain_confirmation', 'No'),
            0,  # P&L (1 Lot)
            0,  # Targets Hit
            0,  # Stoploss Count
            ""  # Failure Reason
        ]
        
        sheet.append_row(row, value_input_option="USER_ENTERED")
        logger.info(f"✅ Trade Logged: {signal_data.get('signal', 'None')} at {safe_float(signal_data.get('price', 0))}")
        
    except Exception as e:
        logger.error(f"Error logging trade: {e}")

def is_market_open():
    """Check if the market is currently open"""
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Check if it's a weekday (0-4 = Monday-Friday)
    is_weekday = current_time.weekday() < 5
    
    # Check if current time is within market hours
    is_within_market_hours = market_open <= current_time <= market_close
    
    return is_weekday and is_within_market_hours

# === 🚀 Main Loop ===
def run_realtime_bot():
    sheet, summary_sheet, insights_sheet = setup_sheet()
    if not all([sheet, summary_sheet, insights_sheet]):
        logger.error("Failed to setup Google Sheets")
        return

    # Start updater in background
    start_updater_thread(sheet, fyers)

    symbols = {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }

    logger.info("🔁 Starting real-time signal detection every 1 minute...")
    while True:
        # Check if market is open before processing
        if not is_market_open():
            logger.info("Market is closed. Waiting for next market session...")
            time.sleep(60)
            continue

        for symbol, index_name in symbols.items():
            try:
                df = fetch_candles(fyers, symbol)
                if df.empty:
                    logger.warning(f"No data available for {index_name}")
                    continue

                signal_data = analyze_signal(df)
                if signal_data.get('signal') != "None":
                    log_trade(sheet, index_name, signal_data)

            except Exception as e:
                logger.error(f"❌ Error for {index_name}: {e}")

        time.sleep(60)


if __name__ == "__main__":
    run_realtime_bot()
