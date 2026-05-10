
import json

try:
    with open('impulse_fib_trader/config/whitelist.json', 'r') as f:
        whitelist = json.load(f)['whitelist']
    
    with open('total_analysis_2024.json', 'r') as f:
        all_stats = json.load(f)
        
    print(f"{'Symbol':<12} | {'Trades':<7} | {'Winrate':<8} | {'Profit R':<8}")
    print("-" * 45)
    
    filtered = [s for s in all_stats if s['symbol'] in whitelist]
    # Сортируем по количеству сделок
    filtered.sort(key=lambda x: x['trades'], reverse=True)
    
    for s in filtered:
        print(f"{s['symbol']:<12} | {s['trades']:<7} | {s['wr']:.1%} | {s['net_r']:.1f}")
        
except Exception as e:
    print(f"Ошибка: {e}")
