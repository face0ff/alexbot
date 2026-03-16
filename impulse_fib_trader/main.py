import logging
import os
import json
import pandas as pd
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from data.storage import DataStorage
from pattern.detector import PatternDetector
from features.engineer import FeatureEngineer
from features.labels import Labeler
from ml.train import MLTrainer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Расширенный список ликвидных монет для обучения
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'LTC/USDT', 
        'LINK/USDT', 'SOL/USDT', 'MATIC/USDT', 'DOT/USDT', 'AVAX/USDT', 'DOGE/USDT',
        'SHIB/USDT', 'TRX/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT', 'BCH/USDT',
        'FIL/USDT', 'NEAR/USDT', 'ALGO/USDT', 'APE/USDT', 'MANA/USDT', 'SAND/USDT',
        'HBAR/USDT', 'QNT/USDT', 'VET/USDT', 'OP/USDT', 'GRT/USDT', 'EGLD/USDT'
    ]
    timeframe = '1h'
    start_date = '2019-01-01'
    end_date = '2024-12-31'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    config_path = 'config/pattern_spec.json'
    detector = PatternDetector(config_path)
    fe = FeatureEngineer()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    labeler = Labeler(config)
    
    all_data_frames = []
    
    for symbol in symbols:
        data_path = f"data_{symbol.replace('/', '_')}_{timeframe}_longterm.parquet"
        
        # 1. Загрузка данных
        if not os.path.exists(data_path):
            logger.info(f"--- Скачивание истории {symbol} (2019-2024) ---")
            df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            if df.empty:
                logger.warning(f"Данные по {symbol} не получены.")
                continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            df = cleaner.identify_swings(df)
            DataStorage.save_to_parquet(df, data_path)
        else:
            logger.info(f"Загрузка из кэша: {symbol}")
            df = DataStorage.load_from_parquet(data_path)
            # Принудительно пересчитываем индикаторы, так как добавились новые (RSI, EMA)
            df = cleaner.calculate_indicators(df)
            df = cleaner.identify_swings(df)
        
        # 2. Поиск паттернов
        logger.info(f"Поиск паттернов для {symbol}...")
        # Используем стандартный детектор (Breakout), так как он дает точную разметку для обучения
        patterns = detector.detect_patterns(df)
        
        if patterns:
            # 3. Извлечение признаков и создание меток
            X_sym = fe.extract_features(patterns, df)
            y_sym = labeler.create_labels(patterns, df)
            
            all_data_frames.append((X_sym, y_sym))
            logger.info(f"Добавлено {len(patterns)} примеров от {symbol}")

    if not all_data_frames:
        logger.error("Не найдено паттернов для обучения!")
        return

    # 4. Объединение данных
    X = pd.concat([d[0] for d in all_data_frames], ignore_index=True)
    y = pd.concat([d[1] for d in all_data_frames], ignore_index=True)

    logger.info(f"ИТОГО: {len(X)} паттернов. Начинаю обучение...")

    # 5. Обучение модели
    trainer = MLTrainer()
    model, ml_eval = trainer.train(X, y)
    trainer.save_model('trained_model.joblib')
    
    print("\n" + "="*50)
    print("ГЛУБОКОЕ ОБУЧЕНИЕ (2019-2024) ЗАВЕРШЕНО")
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
