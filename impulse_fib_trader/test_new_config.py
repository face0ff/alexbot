import logging
import json
import pandas as pd
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.detector import PatternDetector
from features.labels import Labeler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_new_config():
    symbol = 'ETH/USDT'
    timeframe = '1h'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    config_path = 'impulse_fib_trader/config/pattern_spec.json'
    detector = PatternDetector(config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    labeler = Labeler(config)
    
    logger.info(f"--- Тестирование новой конфигурации на {symbol} (2023) ---")
    df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
    if df.empty:
        logger.error("Не удалось получить данные.")
        return
        
    df = cleaner.validate_data(df)
    df = cleaner.calculate_indicators(df)
    df = cleaner.identify_swings(df)
    
    patterns = detector.detect_patterns(df)
    logger.info(f"Найдено паттернов: {len(patterns)}")
    
    if patterns:
        y = labeler.create_labels(patterns, df)
        win_rate = y.mean()
        
        print("\n" + "="*50)
        print(f"РЕЗУЛЬТАТЫ ТЕСТА (Config: {config['id']})")
        print("="*50)
        print(f"Символ: {symbol}")
        print(f"Всего паттернов: {len(patterns)}")
        print(f"Win Rate (Labeling R=1.5): {win_rate:.2%}")
        
        # Средняя глубина коррекции
        avg_depth = sum(p['pullback']['depth'] for p in patterns) / len(patterns)
        print(f"Средняя глубина коррекции: {avg_depth:.4f}")
        
        # Типы входов
        confirmations = {}
        for p in patterns:
            c = p['structure']['confirmation']
            confirmations[c] = confirmations.get(c, 0) + 1
        
        print("\nТипы подтверждений входа:")
        for c, count in confirmations.items():
            print(f"- {c}: {count}")
        print("="*50)
    else:
        print("\nПаттерны не найдены с текущим конфигом.")

if __name__ == "__main__":
    test_new_config()
