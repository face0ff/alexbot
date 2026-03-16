import json
import os
import ccxt
from datetime import datetime
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

def cleanup_ghost_trades():
    state_file = 'impulse_fib_trader/trade_state.json'
    history_file = 'impulse_fib_trader/trade_history.json'
    
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET
    })

    with open(state_file, 'r') as f:
        active = json.load(f)

    ghost_symbols = ['EDU/USDT', 'XAI/USDT', 'AXS/USDT', 'QTUM/USDT']
    
    updated_active = []
    
    for trade in active:
        if trade['symbol'] in ghost_symbols:
            print(f"Фиксирую выход для {trade['symbol']}...")
            try:
                ticker = exchange.fetch_ticker(trade['symbol'])
                exit_price = ticker['last']
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_reason'] = "MANUAL_EXIT_SYNC"
                trade['pnl_usdt'] = (exit_price - trade['real_entry_price']) * trade['amount']
                
                # В историю
                with open(history_file, 'r') as hf:
                    history = json.load(hf)
                history.append(trade)
                with open(history_file, 'w') as hf:
                    json.dump(history, hf, indent=4)
            except:
                print(f"Не удалось получить цену для {trade['symbol']}, пропускаю.")
        else:
            updated_active.append(trade)

    with open(state_file, 'w') as f:
        json.dump(updated_active, f, indent=4)
    print("Очистка завершена.")

if __name__ == "__main__":
    cleanup_ghost_trades()
