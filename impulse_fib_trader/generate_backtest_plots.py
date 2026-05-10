
import json
import os
import ccxt
import pandas as pd
import asyncio
import random
from datetime import datetime, timedelta
from data.cleaner import DataCleaner
from reporting_plots import plot_backtest_trade

async def generate_new_plots():
    REPORT_FILE = '../full_backtest_report_2024.json'
    if not os.path.exists(REPORT_FILE):
        REPORT_FILE = 'full_backtest_report_2024.json'
    
    if not os.path.exists(REPORT_FILE):
        print("Отчет не найден. Сначала запустите бэктест.")
        return

    with open(REPORT_FILE, 'r') as f:
        report = json.load(f)
    
    all_trades = report['all_trades']
    if not all_trades:
        print("Сделки не найдены в отчете.")
        return

    # Выберем 12 сделок: 8 прибыльных и 4 убыточных для баланса
    winners = [t for t in all_trades if t['result_r'] > 0]
    losers = [t for t in all_trades if t['result_r'] < 0]
    
    sample_trades = random.sample(winners, min(8, len(winners))) + random.sample(losers, min(4, len(losers)))
    random.shuffle(sample_trades)
    
    os.makedirs('backtest_plots_detailed', exist_ok=True)
    exchange = ccxt.binance()
    cleaner = DataCleaner()
    
    print(f"Генерация {len(sample_trades)} детальных графиков...")
    
    for i, t in enumerate(sample_trades):
        symbol = t['symbol']
        entry_time = datetime.fromisoformat(t['entry_time'])
        exit_time = datetime.fromisoformat(t['exit_time'])
        
        # Скачиваем данные вокруг сделки (с запасом)
        start_fetch = (entry_time - timedelta(days=5)).timestamp() * 1000
        # Ограничим 500 свечей, чтобы захватить и вход и выход
        try:
            ohlcv = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, '1h', int(start_fetch), 500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = cleaner.calculate_indicators(df)
            
            filename = f"backtest_plots_detailed/trade_{i+1}_{symbol.replace('/', '_')}_{entry_time.strftime('%Y%m%d')}.png"
            plot_backtest_trade(df, t, filename)
            print(f"[{i+1}/{len(sample_trades)}] Отрисован {symbol}")
        except Exception as e:
            print(f"Ошибка отрисовки {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(generate_new_plots())
