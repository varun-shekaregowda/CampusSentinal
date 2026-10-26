"""
telegram_alert.py
-------------------
Sends a Telegram message (with snapshot photo) when an intrusion or
loitering event is detected. Fully optional — does nothing unless
ENABLE_TELEGRAM is set to True in config.py.

Setup:
    1. In Telegram, message @BotFather -> send /newbot -> follow the
       prompts -> copy the bot token it gives you.
    2. Message @userinfobot -> send /start -> copy your numeric chat ID.
    3. In config.py, set:
           ENABLE_TELEGRAM = True
           TELEGRAM_BOT_TOKEN = "your-token-here"
           TELEGRAM_CHAT_ID = "your-chat-id-here"
    4. Send your bot at least one message first (Telegram bots can't
       message you until you've messaged them once).
"""
import requests

import config


def send_telegram_alert(message: str, image_path: str = None) -> None:
    if not config.ENABLE_TELEGRAM:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[Telegram] Skipped — bot token or chat ID not set in config.py")
        return

    base_url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

    try:
        if image_path:
            with open(image_path, "rb") as photo:
                requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": message},
                    files={"photo": photo},
                    timeout=10,
                )
        else:
            requests.post(
                f"{base_url}/sendMessage",
                data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message},
                timeout=10,
            )
    except Exception as e:
        print(f"[Telegram] Failed to send alert: {e}")
