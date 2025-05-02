import requests
import pandas as pd
import ta
import time
from fyers_apiv3 import fyersModel
import webbrowser
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import backoff
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv
import math
import pytz
import schedule
import threading
from db import log_trade_sql

"""
In order to get started with Fyers API we would like you to do the following things first.
1. Checkout our API docs :   https://myapi.fyers.in/docsv3
2. Create an APP using our API dashboard :   https://myapi.fyers.in/dashboard/

Once you have created an APP you can start using the below SDK 
"""

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

@dataclass
class Config:
    """Configuration class for the trading bot"""
    client_id: str = os.getenv('FYERS_CLIENT_ID')
    secret_key: str = os.getenv('FYERS_SECRET_KEY')
    redirect_uri: str = os.getenv('FYERS_REDIRECT_URI')
    grant_type: str = os.getenv('FYERS_GRANT_TYPE')
    response_type: str = os.getenv('FYERS_RESPONSE_TYPE')
    state: str = os.getenv('FYERS_STATE')
    symbols: Dict[str, str] = None
    resolution: str = "5"
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_period: int = 20
    atr_period: int = 14
    confirmation_count: int = 2
    rsi_upper: float = 65
    rsi_lower: float = 35
    macd_threshold: float = 5
    price_threshold: float = 0.001

    def __post_init__(self):
        # Validate required environment variables
        required_vars = ['client_id', 'secret_key', 'redirect_uri', 'grant_type', 'response_type', 'state']
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        if self.symbols is None:
            self.symbols = {
                "NSE:NIFTY50-INDEX": "NIFTY50",
                "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
            }

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

# # CONNECTION ESTABLISHED ABOVE


# # 🔥 Function to setup Google Sheet connection
# # 🔥 Setup Sheet

# # 🔧 SETUP GOOGLE SHEET
def setup_sheet(spreadsheet):
    signal_headers = [
        "Timestamp", "Index", "Signal", "Strike Price", "Stop Loss", "Target",
        "Target 2", "Target 3", "Exit TS 1", "Exit TS 2", "Exit TS 3",
        "Price", "RSI", "MACD", "MACD Signal", "EMA 20", "ATR",
        "Outcome", "RSI Reason", "MACD Reason", "Price Reason",
        "Confidence", "Trade Type", "Option Chain Confirmation", "P&L (1 Lot)",
        "Targets Hit", "Stoploss Count", "Failure Reason"
    ]
    sheet = spreadsheet.sheet1
    if sheet.row_values(1) != signal_headers:
        sheet.clear()
        sheet.append_row(signal_headers)
        print("✅ Headers added to Signal Sheet.")

    try:
        summary_sheet = spreadsheet.worksheet("Summary")
    except gspread.exceptions.WorksheetNotFound:
        summary_sheet = spreadsheet.add_worksheet(title="Summary", rows="1000", cols="5")
        summary_sheet.append_row(["Date", "Index", "P&L (1 Lot)", "Targets Hit", "Stoplosses"])
        print("✅ Created 'Summary' sheet.")

    try:
        insights_sheet = spreadsheet.worksheet("Insights")
    except gspread.exceptions.WorksheetNotFound:
        insights_sheet = spreadsheet.add_worksheet(title="Insights", rows="1000", cols="7")
        insights_sheet.append_row(["Index", "Total Trades", "Accuracy %", "Total P&L", "Avg Profit", "Avg Loss", "Win Ratio"])
        print("✅ Created 'Insights' sheet.")

    return sheet, summary_sheet, insights_sheet


def log_signal(sheet, index_name, signal, strike_price, stoploss, target, target2, target3,
               exit_ts1, exit_ts2, exit_ts3, price, rsi, macd, macd_signal, ema_20, atr,
               outcome, rsi_reason, macd_reason, price_reason, confidence, trade_type,
               option_chain_confirmation, pnl_1lot, targets_hit, stoploss_count, failure_reason, timestamp):

    row = [
        timestamp, index_name, signal, strike_price, stoploss, target, target2, target3,
        exit_ts1, exit_ts2, exit_ts3,
        price, rsi, macd, macd_signal, ema_20, atr,
        outcome, rsi_reason, macd_reason, price_reason,
        confidence, trade_type, option_chain_confirmation, pnl_1lot,
        targets_hit, stoploss_count, failure_reason
    ]
    sheet.append_row(row, value_input_option='USER_ENTERED')
    print(f"✅ {timestamp} | {index_name} - {signal} logged with P&L ₹{pnl_1lot}")


def basic_failure_reason(rsi, macd, macd_signal, close, ema_20, targets_hit, outcome):
    if "Stoploss" in outcome:
        reasons = []
        if rsi < 68:
            reasons.append("Weak RSI")
        if macd < macd_signal + 8:
            reasons.append("Weak MACD crossover")
        if close < ema_20 * 1.003:
            reasons.append("Low EMA strength")
        if not reasons:
            return "Sudden reversal or volatility"
        return ", ".join(reasons)
    elif targets_hit == 0:
        return "No momentum after entry"
    return ""


# ✅ You can now call `basic_failure_reason()` from within generate_signal_all()
# and pass the returned value to `log_signal` under the `failure_reason` parameter



# 🔥 Fetch candles
def fetch_candles(symbol, fyers, resolution="5", range_from="2025-04-20", range_to="2025-05-02"):
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"
    }
    response = fyers.history(data)
    candles = response.get('candles')
    if not candles:
        raise Exception(f"No candle data fetched for {symbol}")
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# 🔥 Calculate indicators
def calculate_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    return df

# 🔥 Generate and log signals
def is_volume_spike(candle, df, idx):
    if idx < 20:
        return False
    avg_volume = df['volume'][idx-20:idx].mean()
    return candle['volume'] > avg_volume * 1.5

def is_long_wick_candle(candle):
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['close'], candle['open'])
    lower_wick = min(candle['close'], candle['open']) - candle['low']
    return upper_wick > body or lower_wick > body

def generate_signal_all(df, index_name, sheet, lot_size, summary_sheet, insights_sheet):
    import ta
    df['supertrend'] = ta.trend.stc(df['close'])

    last_signal = "NO TRADE"
    confirmation_counter = 0
    total_signals = 0
    successful_signals = 0
    daily_pnl = {}
    total_pnl = 0
    total_wins = 0
    total_losses = 0
    win_amount = 0
    loss_amount = 0
    targets_hit_count = {}
    stoploss_count = {}

    for idx in range(50, len(df) - 24):
        candle = df.iloc[idx]

        # ✨ Skip low body candles
        body = abs(candle['close'] - candle['open'])
        full_range = candle['high'] - candle['low']
        if full_range == 0 or body / full_range < 0.6:
            continue

        # ✨ Reject if Supertrend not in favor
        if candle['supertrend'] < 0.5:
            continue

        # ✨ Avoid overextended candles
        last2 = df.iloc[idx-2:idx]
        # if (last2['close'] - last2['open']).abs().sum() > 3 * candle['atr']:
        #     continue

        # ✨ Skip trades after 2:45 PM
        ist_time_check = candle['time'].tz_localize("UTC").tz_convert("Asia/Kolkata")
        if ist_time_check.hour >= 14 and ist_time_check.minute >= 45:
            continue

        # ✨ Skip if volume isn't strong
        avg_vol = df['volume'][idx-20:idx].mean()
        # if candle['volume'] < 1.2 * avg_vol:
        #     continue

        current_signal = "NO TRADE"
        rsi_reason = ""
        macd_reason = ""
        price_reason = ""
        confidence = "Medium"
        trade_type = "Intraday"
        option_chain_confirmation = "Pending"

        if (
            candle['rsi'] > 65 and
            candle['macd'] > candle['macd_signal'] + 7 and
            candle['close'] > candle['ema_20'] * 1.001
        ):
            current_signal = "BUY CALL"
            rsi_reason = f"RSI {candle['rsi']:.2f} > 65"
            macd_reason = f"MACD {candle['macd']:.2f} > MACD Signal +7 ({candle['macd_signal'] + 7:.2f})"
            price_reason = f"Price {candle['close']:.2f} > EMA {candle['ema_20']:.2f}"
            confidence = "High" if candle['rsi'] > 70 else "Medium"

        elif (
            candle['rsi'] < 35 and
            candle['macd'] < candle['macd_signal'] - 5 and
            candle['close'] < candle['ema_20'] * 0.999
        ):
            current_signal = "BUY PUT"
            rsi_reason = f"RSI {candle['rsi']:.2f} < 35"
            macd_reason = f"MACD {candle['macd']:.2f} < MACD Signal -5 ({candle['macd_signal'] - 5:.2f})"
            price_reason = f"Price {candle['close']:.2f} < EMA {candle['ema_20']:.2f}"
            confidence = "High" if candle['rsi'] < 30 else "Medium"

        if current_signal == last_signal and current_signal != "NO TRADE":
            confirmation_counter += 1
        else:
            confirmation_counter = 0

        if confirmation_counter == 2:
            price = candle['close']
            atr = candle['atr']
            strike_price = int(round(price / 50) * 50)
            stoploss = int(round(atr))
            target = int(round(1.5 * atr))
            target2 = int(round(2.0 * atr))
            target3 = int(round(2.5 * atr))

            utc_time = candle['time'].tz_localize("UTC")
            ist_time = utc_time.tz_convert("Asia/Kolkata")
            timestamp = ist_time.strftime("%Y-%m-%d %H:%M:%S")
            date_str = ist_time.date().isoformat()

            next_df = df.iloc[idx + 1: idx + 25]
            low_hit = next_df['low'] <= (price - stoploss)
            high_hit1 = next_df['high'] >= (price + target)
            high_hit2 = next_df['high'] >= (price + target2)
            high_hit3 = next_df['high'] >= (price + target3)

            exit_ts1 = next_df[high_hit1].iloc[0]['time'].tz_localize("UTC").tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S") if high_hit1.any() else ""
            exit_ts2 = next_df[high_hit2].iloc[0]['time'].tz_localize("UTC").tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S") if high_hit2.any() else ""
            exit_ts3 = next_df[high_hit3].iloc[0]['time'].tz_localize("UTC").tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S") if high_hit3.any() else ""

            if low_hit.any():
                outcome = "Stoploss Hit"
                pnl = -stoploss * lot_size
                total_losses += 1
                loss_amount += pnl
                stoploss_count[date_str] = stoploss_count.get(date_str, 0) + 1
                targets_hit = 0
            else:
                lots_hit = 0
                pnl = 0
                if high_hit1.any():
                    pnl += target * lot_size
                    lots_hit += 1
                if high_hit2.any():
                    pnl += target2 * lot_size
                    lots_hit += 1
                if high_hit3.any():
                    pnl += target3 * lot_size
                    lots_hit += 1
                outcome = f"{lots_hit} Targets Hit"
                if lots_hit > 0:
                    successful_signals += 1
                    total_wins += 1
                    win_amount += pnl
                targets_hit = lots_hit

            total_pnl += pnl
            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + pnl
            targets_hit_count[date_str] = targets_hit_count.get(date_str, 0) + targets_hit

            option_chain_confirmation = "Yes" if confidence == "High" else "No"

            failure_reason = basic_failure_reason(
                candle['rsi'], candle['macd'], candle['macd_signal'], price,
                candle['ema_20'], targets_hit, outcome
            )

            log_signal(
                sheet, index_name, current_signal, strike_price, stoploss,
                target, target2, target3, exit_ts1, exit_ts2, exit_ts3,
                round(price, 2), round(candle['rsi'], 2), round(candle['macd'], 2),
                round(candle['macd_signal'], 2), round(candle['ema_20'], 2), round(candle['atr'], 2),
                outcome, rsi_reason, macd_reason, price_reason, confidence, trade_type,
                option_chain_confirmation, pnl, targets_hit, stoploss_count.get(date_str, 0),
                failure_reason, timestamp
            )

            confirmation_counter = 0
            total_signals += 1

        last_signal = current_signal

    accuracy = (successful_signals / total_signals * 100) if total_signals else 0
    avg_profit = (win_amount / total_wins) if total_wins else 0
    avg_loss = (loss_amount / total_losses) if total_losses else 0
    win_ratio = (total_wins / total_signals * 100) if total_signals else 0

    print(f"\n📊 Accuracy for {index_name}: {accuracy:.2f}% ({successful_signals}/{total_signals})")
    print(f"✅ Wins: {total_wins}, ❌ Losses: {total_losses}, 💰 Net P&L: ₹{total_pnl:.2f}")
    print("📈 Daily P&L Summary:")
    for d, p in daily_pnl.items():
        print(f"{d}: ₹{p:.2f}")
        summary_sheet.append_row([d, index_name, p, targets_hit_count.get(d, 0), stoploss_count.get(d, 0)])

    insights_sheet.append_row([
        index_name,
        total_signals,
        round(accuracy, 2),
        round(total_pnl, 2),
        round(avg_profit, 2),
        round(avg_loss, 2),
        round(win_ratio, 2)
    ])



# 🔥 Main runner
def run_bot(fyers):
    symbols = {
        "NSE:NIFTY50-INDEX": "NIFTY50",
        "NSE:NIFTYBANK-INDEX": "BANKNIFTY"
    }

    lot_sizes = {
        "NIFTY50": 50,
        "BANKNIFTY": 15
    }

    client = gspread.authorize(
        ServiceAccountCredentials.from_json_keyfile_name(
            "analysis-test-458117-64b56a994c55.json",
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
    )

    spreadsheet = client.open("Trading Signals Tracker")
    
    # ✅ Use setup_sheet to get all 3 sheets
    sheet, summary_sheet, insights_sheet = setup_sheet(spreadsheet)


    for symbol, index_name in symbols.items():
        try:
            df = fetch_candles(symbol, fyers)
            if df.empty:
                print(f"❗ No candles available for {index_name}")
                continue
            df = calculate_indicators(df)
            generate_signal_all(
                df,
                index_name,
                sheet,
                lot_sizes[index_name],
                summary_sheet,
                insights_sheet
            )
        except Exception as e:
            print(f"❌ Error processing {index_name}: {e}")

    print("✅✅✅ Historical Data Processing Finished ✅✅✅")

# 🔥 Entry point
if __name__ == "__main__":
    print("🚀 Bot Started - Auto Refresh Every 5 Minutes!")

    run_bot(fyers)
