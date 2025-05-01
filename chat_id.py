from telegram import Bot

BOT_TOKEN = '7233653035:AAHVNm4ESq5_s9fq-qFbUNN3bHXYerpMsBw'

bot = Bot(token=BOT_TOKEN)
updates = bot.get_updates()

for update in updates:
    print(update.message.chat.id)
