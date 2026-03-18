import json
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.tas_detector import TASDetector

async def test_tas():
    # 1. Загрузка конфига
    with open('impulse_fib_trader/config/pattern_spec_tas.json', 'r') as f:
        config = json.load(f)
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    detector = TASDetector(config)
    
    symbol = 'ETH/USDT'
    print(f"--- Тестирование стратегии TAS_v1 на {symbol} ---")
    
    # Берем побольше данных для истории (последние 30 дней)
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    df = fetcher.fetch_ohlcv(symbol, '1h', start_date)
    
    if df.empty:
        print("Данные не получены.")
        return
        
    df = cleaner.validate_data(df)
    df = cleaner.calculate_indicators(df)
    
    patterns = detector.detect_patterns(df)
    
    print(f"Найдено паттернов: {len(patterns)}")
    
    for p in patterns[-5:]: # Показать последние 5
        print(f"\n✅ ПАТТЕРН ОБНАРУЖЕН: {p['timestamp']}")
        print(f"   Минимум Хвоста: {p['tail_low']:.2f}")
        print(f"   Уровень пробоя: {p['breakout_level']:.2f}")
        print(f"   Минимум полки: {p['shelf_low']:.2f}")
        print(f"   Цена входа: {p['entry_price']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_tas())
