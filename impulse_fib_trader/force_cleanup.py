import ccxt
import json
import os
import logging
from datetime import datetime
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_exit_triggered_trades():
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    state_file = 'impulse_fib_trader/trade_state.json'
    history_file = 'impulse_fib_trader/trade_history.json'

    if not os.path.exists(state_file):
        print("Файл активных сделок не найден.")
        return

    with open(state_file, 'r') as f:
        active_trades = json.load(f)

    if not active_trades:
        print("Нет активных сделок для проверки.")
        return

    updated_active = []
    closed_count = 0

    print(f"--- Проверка {len(active_trades)} сделок на пересечение уровней ---")

    for trade in active_trades:
        symbol = trade['symbol']
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            sl = trade['sl']
            tp = trade['tp']
            
            exit_reason = None
            if current_price >= tp:
                exit_reason = "TAKE_PROFIT (LATENT)"
            elif current_price <= sl:
                exit_reason = "STOP_LOSS (LATENT)"
            
            if exit_reason:
                print(f"⚠️ {symbol} отработал уровень! Цена: {current_price}, TP: {tp}, SL: {sl}. ЗАКРЫВАЮ...")
                
                # Продаем по рынку
                order = exchange.create_order(symbol, 'market', 'sell', trade['amount'])
                exit_price = order['average'] if order.get('average') else current_price
                
                pnl = (exit_price - trade['real_entry_price']) * trade['amount']
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_reason'] = exit_reason
                trade['pnl_usdt'] = pnl
                
                # Сохраняем в историю
                save_to_history(history_file, trade)
                closed_count += 1
                print(f"✅ {symbol} закрыт. PnL: {pnl:.2f} USDT")
            else:
                # Цена еще внутри диапазона
                print(f"⏳ {symbol} в игре. Цена: {current_price:.6g} (Диапазон: {sl:.6g} - {tp:.6g})")
                updated_active.append(trade)
                
        except Exception as e:
            print(f"❌ Ошибка при проверке {symbol}: {e}")
            updated_active.append(trade)

    # Сохраняем обновленный стейт
    with open(state_file, 'w') as f:
        json.dump(updated_active, f, indent=4)
    
    print("\n--- Ревизия завершена. Закрыто сделок: {} ---".format(closed_count))

def save_to_history(history_file, trade_data):
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except: pass
    history.append(trade_data)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    force_exit_triggered_trades()
