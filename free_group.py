import yfinance as yf
import time
import random
from telegram import Bot
import asyncio

async def send_alert(stock_name, price, change_percentage):
    message = f"🚀 {stock_name} breakout!\nCurrent Price: ₹{price}\nChange: {change_percentage:.2f}%"
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)


TELEGRAM_BOT_TOKEN = "7233653035:AAHVNm4ESq5_s9fq-qFbUNN3bHXYerpMsBw"
TELEGRAM_CHAT_ID = "-4760811451"

bot = Bot(token=TELEGRAM_BOT_TOKEN)

stocks = {
    "RELIANCE.NS": "RELIANCE",
    "INFY.NS": "INFY",
    "HDFCBANK.NS": "HDFCBANK",
    "ICICIBANK.NS": "ICICIBANK",
    "AXISBANK.NS": "AXISBANK"   # ✅ instead of TCS
}


def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return info['regularMarketPrice'], info['previousClose']

# def send_alert(stock_name, price, change_percentage):
#     message = f"🚀 {stock_name} breakout!\nCurrent Price: ₹{price}\nChange: {change_percentage:.2f}%"
#     bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

async def check_breakouts():
    for symbol, stock_name in stocks.items():
        try:
            delay = random.uniform(3, 5)
            print(f"Waiting {delay:.2f} seconds before fetching {stock_name}...")
            await asyncio.sleep(delay)

            price, prev_close = get_stock_price(symbol)
            change_percentage = ((price - prev_close) / prev_close) * 100
            # await send_alert(stock_name, price, change_percentage)
            if change_percentage >= 1.0:
                await send_alert(stock_name, price, change_percentage)
                print(f"Alert sent for {stock_name} 🚀")
            else:
                print(f"No breakout yet for {stock_name} 😴")

        except Exception as e:
            print(f"Error fetching {stock_name}: {e}")


if __name__ == "__main__":
    while True:
        print("Checking breakouts...")
        asyncio.run(check_breakouts())
        time.sleep(60)
