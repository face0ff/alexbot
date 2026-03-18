import logging
import os
import json
import pandas as pd
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from data.storage import DataStorage
from pattern.tas_detector import TASDetector
from features.engineer import FeatureEngineer
from features.labels import Labeler
from ml.train import MLTrainer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Топ ликвидных монет для обучения новой стратегии
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'LTC/USDT', 
        'LINK/USDT', 'SOL/USDT', 'MATIC/USDT', 'DOT/USDT', 'AVAX/USDT', 'DOGE/USDT'
    ]
    timeframe = '1h'
    # Обучаем на данных за 2023-2024 (свежий рынок)
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    
    # Загружаем конфиг TAS
    config_path = 'impulse_fib_trader/config/pattern_spec_tas.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    detector = TASDetector(config)
    fe = FeatureEngineer()
    labeler = Labeler(config)
    
    all_data_frames = []
    
    for symbol in symbols:
        data_path = f"data_{symbol.replace('/', '_')}_{timeframe}_tas.parquet"
        
        # 1. Загрузка данных
        if not os.path.exists(data_path):
            logger.info(f"--- Скачивание истории {symbol} ---")
            df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            if df.empty: continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            DataStorage.save_to_parquet(df, data_path)
        else:
            df = DataStorage.load_from_parquet(data_path)
            # Пересчет индикаторов для надежности
            df = cleaner.calculate_indicators(df)
        
        # 2. Поиск паттернов TAS
        logger.info(f"Поиск паттернов TAS для {symbol}...")
        df.index.name = symbol
        patterns = detector.detect_patterns(df)
        
        if patterns:
            # 3. Извлечение признаков и создание меток
            X_sym = fe.extract_features(patterns, df)
            y_sym = labeler.create_labels(patterns, df)
            
            all_data_frames.append((X_sym, y_sym))
            logger.info(f"Добавлено {len(patterns)} примеров TAS от {symbol}")

    if not all_data_frames:
        logger.error("Не найдено паттернов TAS для обучения!")
        return

    # 4. Объединение данных
    X = pd.concat([d[0] for d in all_data_frames], ignore_index=True)
    y = pd.concat([d[1] for d in all_data_frames], ignore_index=True)

    logger.info(f"ИТОГО: {len(X)} паттернов TAS. Начинаю обучение...")

    # 5. Обучение модели
    trainer = MLTrainer()
    model, ml_eval = trainer.train(X, y)
    trainer.save_model('trained_model_tas.joblib')
    
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ TAS_v1 ЗАВЕРШЕНО")
    print("="*50)
    print(f"Всего примеров: {len(X)}")
    print(f"Точность (Accuracy): {ml_eval['accuracy']:.2%}")
    print("\nТоп признаков:")
    sorted_features = sorted(ml_eval['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        print(f"- {feat}: {imp:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
