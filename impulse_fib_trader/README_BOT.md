# Impulse Fib Trader Telegram Bot

This bot integrates the Impulse-Fib trading strategy with Telegram for automated scanning and trading on Binance Spot.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Open `impulse_fib_trader/config/config.py` and fill in your details:
    *   `BOT_TOKEN`: Your Telegram Bot API Token (from @BotFather).
    *   `TELEGRAM_PRIVATE_CHAT_ID`: Your personal Chat ID (use @userinfobot to find it).
    *   `BINANCE_API_KEY`: Your Binance API Key.
    *   `BINANCE_API_SECRET`: Your Binance Secret Key.

    **Important:** Ensure your Binance API keys have **Spot Trading** permissions enabled.

3.  **Train the Model (if not done):**
    Before running the bot, ensure you have a trained model.
    ```bash
    cd impulse_fib_trader
    python main.py
    ```

## Running the Bot

Run the bot script from the `impulse_fib_trader` directory:

```bash
cd impulse_fib_trader
python telegram_bot.py
```

## Features

*   **🔍 Scan Market**: Scans Binance Spot markets for the best setup. If a high-confidence signal is found and you are not in a trade, it will **automatically buy** with your full USDT balance.
*   **📊 Statistics**: Shows total trades, wins, and PnL based on local history.
*   **ℹ️ Status**: Shows the currently active trade details.
*   **Auto-Monitor**: The bot checks the active trade every 15 minutes. If Take Profit or Stop Loss levels are hit, it sells the position and notifies you.

## Safety Notes

*   The bot uses `market` orders for entry and exit.
*   It trades with the **entire** USDT balance available in your spot wallet. Use a dedicated sub-account or limit the balance for safety.
*   State is saved to `trade_state.json`. Do not delete this file while a trade is open, or the bot will lose track of the position.
