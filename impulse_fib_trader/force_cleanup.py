
import asyncio
import ccxt
import json
import os
import logging
try:
    from config.config import BINANCE_API_KEY, BINANCE_API_SECRET
except ModuleNotFoundError:
    from impulse_fib_trader.config.config import BINANCE_API_KEY, BINANCE_API_SECRET


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cleanup")

async def force_cleanup():
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    state_file = 'impulse_fib_trader/trade_state.json'
    history_file = 'impulse_fib_trader/trade_history.json'
    
    if not os.path.exists(state_file):
        print("Стейт-файл не найден.")
        return

    with open(state_file, 'r') as f:
        trades = json.load(f)
    
    if not trades:
        print("Нет активных сделок для очистки.")
        return

    print(f"Найдено сделок: {len(trades)}")
    
    remaining_trades = []
    
    for t in trades:
        symbol = t['symbol']
        print(f"Обработка {symbol}...")
        try:
            # Пытаемся продать на бирже
            balance = exchange.fetch_balance()
            base = symbol.split('/')[0]
            qty = balance['free'].get(base, 0)
            
            if qty > 0:
                print(f"Продажа {qty} {base} по рынку...")
                order = exchange.create_order(symbol, 'market', 'sell', exchange.amount_to_precision(symbol, qty))
                exit_p = order.get('average', order.get('price', 0))
                print(f"Успешно продано {symbol} по {exit_p}")
                
                # Записываем в историю как принудительное закрытие
                t['exit_price'] = exit_p
                t['exit_time'] = "FORCED_CLEANUP"
                t['status'] = 'CLOSED'
                t['pnl_usdt'] = (exit_p - t['real_entry_price']) * t['amount']
                
                with open(history_file, 'r') as hf:
                    history = json.load(hf)
                history.append(t)
                with open(history_file, 'w') as hf:
                    json.dump(history, hf, indent=4)
            else:
                print(f"На балансе нет {base}. Просто удаляю из стейта.")
                
        except Exception as e:
            print(f"Ошибка при закрытии {symbol}: {e}")
            print("Сделка будет удалена из списка активных без продажи.")

    # Очищаем стейт
    with open(state_file, 'w') as f:
        json.dump([], f)
    
    print("\n✅ Стейт очищен. Теперь бот сможет искать новые сделки (когда на балансе будут USDT).")

if __name__ == "__main__":
    asyncio.run(force_cleanup())
