
import json
import pandas as pd
import numpy as np
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from datetime import datetime, timedelta

def audit():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    with open('impulse_fib_trader/trade_history.json', 'r') as f:
        history = json.load(f)
    
    # Берем последние 15 сделок для анализа
    recent_trades = [t for t in history if t.get('status') == 'CLOSED'][-15:]
    
    print(f"{'Symbol':<12} | {'Result':<10} | {'RSI':<5} | {'MACD Hist':<10} | {'Dist EMA20':<10} | {'Verdict'}")
    print("-" * 80)
    
    for trade in recent_trades:
        symbol = trade['symbol']
        entry_time = pd.to_datetime(trade['entry_time'])
        
        # Качаем данные за день до сделки
        start_date = (entry_time - timedelta(days=2)).strftime('%Y-%m-%d')
        df = fetcher.fetch_ohlcv(symbol, '15m', start_date)
        
        if df.empty: continue
        
        df = cleaner.calculate_indicators(df)
        
        # Находим ближайшую свечу к моменту входа
        df['time_diff'] = abs(df['timestamp'] - entry_time)
        entry_candle = df.sort_values('time_diff').iloc[0]
        
        rsi = entry_candle['rsi']
        macd_hist = entry_candle['macd_hist']
        dist_ema = (entry_candle['close'] / entry_candle['ema_20'] - 1) * 100
        
        verdict = "OK"
        if rsi > 65: verdict = "RSI TOO HIGH"
        if macd_hist < 0: verdict = "MACD NEGATIVE"
        if dist_ema > 3: verdict = "OVEREXTENDED"
        
        res = "LOSS" if trade.get('pnl_usdt', 0) < 0 else "PROFIT"
        
        print(f"{symbol:<12} | {res:<10} | {rsi:<5.1f} | {macd_hist:<10.6f} | {dist_ema:<10.2f}% | {verdict}")

if __name__ == "__main__":
    audit()
