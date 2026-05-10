
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from datetime import timedelta

def simulate():
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    with open('impulse_fib_trader/trade_history.json', 'r') as f:
        history = json.load(f)
    
    # Фильтруем только закрытые сделки
    closed_trades = [t for t in history if t.get('status') == 'CLOSED' and 'pnl_usdt' in t]
    
    results = []
    print(f"Analyzing {len(closed_trades)} trades... This may take a minute.")
    
    for i, trade in enumerate(closed_trades):
        symbol = trade['symbol']
        entry_time = pd.to_datetime(trade['entry_time'])
        pnl = trade['pnl_usdt']
        
        # Получаем тех. контекст (15м свечи)
        start_date = (entry_time - timedelta(days=2)).strftime('%Y-%m-%d')
        df = fetcher.fetch_ohlcv(symbol, '15m', start_date)
        
        if df.empty or len(df) < 50: continue
        
        df = cleaner.calculate_indicators(df)
        df['time_diff'] = abs(df['timestamp'] - entry_time)
        entry_candle = df.sort_values('time_diff').iloc[0]
        
        results.append({
            'symbol': symbol,
            'pnl': pnl,
            'rsi': entry_candle['rsi'],
            'macd_hist': entry_candle['macd_hist'],
            'dist_ema20': (entry_candle['close'] / entry_candle['ema_20'] - 1) * 100,
            'timestamp': entry_time
        })
        if i % 20 == 0: print(f"Processed {i}/{len(closed_trades)} trades...")

    df_results = pd.DataFrame(results).sort_values('timestamp')
    
    # Сценарии
    df_results['pnl_baseline'] = df_results['pnl']
    df_results['pnl_rsi'] = df_results.apply(lambda x: x['pnl'] if x['rsi'] <= 65 else 0, axis=1)
    df_results['pnl_macd'] = df_results.apply(lambda x: x['pnl'] if x['macd_hist'] > 0 else 0, axis=1)
    df_results['pnl_combo'] = df_results.apply(lambda x: x['pnl'] if (x['rsi'] <= 65 and x['macd_hist'] > 0) else 0, axis=1)
    
    # Итоги
    print("\n" + "="*50)
    print(f"{'Strategy':<15} | {'Final PnL':<10} | {'Win Rate':<10} | {'Trades'}")
    print("-" * 50)
    
    for col in ['pnl_baseline', 'pnl_rsi', 'pnl_macd', 'pnl_combo']:
        final_pnl = df_results[col].sum()
        trades_count = len(df_results[df_results[col] != 0])
        wins = len(df_results[(df_results[col] > 0)])
        wr = (wins / trades_count * 100) if trades_count > 0 else 0
        print(f"{col:<15} | {final_pnl:<10.2f} | {wr:<10.1f}% | {trades_count}")

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(df_results['timestamp'], df_results['pnl_baseline'].cumsum(), label='Baseline (No Filters)', alpha=0.7)
    plt.plot(df_results['timestamp'], df_results['pnl_rsi'].cumsum(), label='Only RSI < 65', alpha=0.7)
    plt.plot(df_results['timestamp'], df_results['pnl_macd'].cumsum(), label='Only MACD Hist > 0', alpha=0.7)
    plt.plot(df_results['timestamp'], df_results['pnl_combo'].cumsum(), label='COMBO (RSI + MACD)', linewidth=2, color='black')
    
    plt.title('PnL Comparison: Filters Impact on Real History')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pnl_comparison_filters.png')
    print("\n✅ График сохранен в pnl_comparison_filters.png")

if __name__ == "__main__":
    simulate()
