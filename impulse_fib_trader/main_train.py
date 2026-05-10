import logging
import os
import json
import pandas as pd
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from data.storage import DataStorage
from pattern.tas_detector import ImpulseRejectionDetector
from features.engineer import FeatureEngineer
from features.labels import Labeler
from ml.train import MLTrainer

def train_tas_longterm():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'LTC/USDT', 'LINK/USDT']
    timeframe = '1h'
    # ОБУЧАЕМ НА 4 ГОДАХ (2021-2024)
    start_date = '2021-01-01'
    end_date = '2024-12-31'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    config_path = 'impulse_fib_trader/config/pattern_spec_tas.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    detector = ImpulseRejectionDetector(config)
    fe = FeatureEngineer()
    labeler = Labeler(config)
    
    all_data = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
        if df.empty: continue
        df = cleaner.validate_data(df)
        df = cleaner.calculate_indicators(df)
        patterns = detector.detect_patterns(df)
        if patterns:
            X = fe.extract_features(patterns, df)
            y = labeler.create_labels(patterns, df)
            all_data.append((X, y))
            print(f"   Added {len(patterns)} patterns")

    X = pd.concat([d[0] for d in all_data], ignore_index=True)
    y = pd.concat([d[1] for d in all_data], ignore_index=True)
    
    trainer = MLTrainer()
    trainer.train(X, y)
    trainer.save_model('trained_model_tas_21_24.joblib')
    print("Model trained on 2021-2024 data.")

if __name__ == "__main__":
    train_tas_longterm()
