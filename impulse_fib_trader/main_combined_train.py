import pandas as pd
import json
import os
import joblib
from datetime import datetime
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.combined_detector import CombinedTASFibDetector
from features.engineer import FeatureEngineer
from features.labels import Labeler
from ml.train import MLTrainer

def train_super_strategy():
    # Загружаем Топ-20 монет
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'LINK/USDT', 'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'TRX/USDT'
    ]
    timeframe = '15m'
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    detector = CombinedTASFibDetector()
    fe = FeatureEngineer()
    labeler = Labeler({'risk_management': {'stop_loss': {'buffer_atr': 0.15}}})
    
    all_X = []
    all_y = []
    
    print(f"--- ЗАГРУЗКА И ОБУЧЕНИЕ 2021-2023 (20 монет) ---")
    
    for symbol in symbols:
        try:
            df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            if df.empty or len(df) < 500: continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            
            patterns = detector.detect_patterns(df)
            if patterns:
                X = fe.extract_features(patterns, df)
                y = labeler.create_labels(patterns, df)
                all_X.append(X)
                all_y.append(y)
                print(f"✅ {symbol}: Найдено {len(patterns)} паттернов")
        except Exception as e:
            print(f"❌ Ошибка {symbol}: {e}")

    if not all_X:
        print("❌ Не найдено данных для обучения! Возможно, параметры детектора слишком строгие.")
        return

    X_full = pd.concat(all_X, ignore_index=True)
    y_full = pd.concat(all_y, ignore_index=True)
    
    print(f"--- ИТОГО ДЛЯ ОБУЧЕНИЯ: {len(X_full)} примеров ---")
    
    trainer = MLTrainer()
    # MLTrainer.train обычно возвращает (model, eval_results)
    model, eval_results = trainer.train(X_full, y_full)
    
    # Сохраняем модель
    joblib.dump(model, 'super_model_v2.joblib')
    print(f"✅ Модель сохранена как super_model_v2.joblib")
    
    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Точность модели: {eval_results.get('accuracy', 0):.2%}")
    print("Топ-5 важных признаков:")
    sorted_f = sorted(eval_results.get('feature_importance', {}).items(), key=lambda x: x[1], reverse=True)
    for f, v in sorted_f[:5]:
        print(f"- {f}: {v:.4f}")

if __name__ == "__main__":
    train_super_strategy()
