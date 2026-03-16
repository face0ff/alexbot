import ccxt
import json
import os
import logging
from config.config import BINANCE_API_KEY, BINANCE_API_SECRET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def attach_oco_to_existing():
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    print("Загрузка рынков Binance...")
    exchange.load_markets()

    state_file = 'impulse_fib_trader/trade_state.json'
    if not os.path.exists(state_file):
        print("Файл сделок не найден.")
        return

    with open(state_file, 'r') as f:
        active_trades = json.load(f)

    updated_trades = []
    for trade in active_trades:
        if trade.get('oco_id'):
            print(f"Сделка {trade['symbol']} уже защищена OCO.")
            updated_trades.append(trade)
            continue

        symbol = trade['symbol']
        sl = trade['sl']
        tp = trade['tp']
        amount = trade['amount']

        print(f"Выставляю OCO для {symbol}: TP={tp}, SL={sl}, Qty={amount}")
        
        try:
            # Binance OCO
            params_oco = {
                'stopPrice': exchange.price_to_precision(symbol, sl),
                'stopLimitPrice': exchange.price_to_precision(symbol, sl * 0.99),
                'stopLimitTimeInForce': 'GTC' # Обязательный параметр!
            }
            oco_res = exchange.private_post_order_oco({
                'symbol': symbol.replace('/', ''),
                'side': 'SELL',
                'quantity': exchange.amount_to_precision(symbol, amount),
                'price': exchange.price_to_precision(symbol, tp),
                'stopPrice': params_oco['stopPrice'],
                'stopLimitPrice': params_oco['stopLimitPrice'],
                'stopLimitTimeInForce': params_oco['stopLimitTimeInForce']
            })
            trade['oco_id'] = oco_res['orderListId']
            print(f"   ✅ OCO успешно выставлен для {symbol}")
        except Exception as e:
            print(f"   ❌ Не удалось выставить OCO для {symbol}: {e}")
        
        updated_trades.append(trade)

    with open(state_file, 'w') as f:
        json.dump(updated_trades, f, indent=4)

if __name__ == "__main__":
    attach_oco_to_existing()
