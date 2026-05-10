import pandas as pd
import joblib
from datetime import datetime
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.market_structure_smc import SMCMarketStructure
from features.engineer_smc import SMCFeatureEngineer
from features.labels_smc import SMCLabeler
from ml.train import MLTrainer

def train_smc_strategy():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'ADA/USDT']
    timeframe = '15m'
    start_date = '2021-01-01'
    end_date = '2023-12-31'
    
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    smc = SMCMarketStructure(window=5)
    fe = SMCFeatureEngineer()
    labeler = SMCLabeler()
    
    all_X = []
    all_y = []
    
    print(f"--- СБОР ДАННЫХ SMC ДЛЯ ОБУЧЕНИЯ (2021-2023) ---")
    
    for symbol in symbols:
        try:
            df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            if df.empty or len(df) < 500: continue
            
            df = cleaner.validate_data(df)
            df = cleaner.calculate_indicators(df)
            
            # Находим BOS
            df = smc.detect_bos(df)
            
            # Извлекаем признаки и метки
            X = fe.extract_features(df)
            y = labeler.create_labels(df)
            
            # Синхронизируем X и y по индексам
            common_idx = X.index.intersection(y.index)
            all_X.append(X.loc[common_idx])
            all_y.append(y.loc[common_idx])
            
            print(f"✅ {symbol}: Собрано {len(common_idx)} паттернов BOS")
        except Exception as e:
            print(f"❌ Ошибка {symbol}: {e}")

    if not all_X:
        print("Данные не найдены!")
        return

    X_full = pd.concat(all_X)
    y_full = pd.concat(all_y)['target']
    
    print(f"--- ИТОГО ДЛЯ ОБУЧЕНИЯ: {len(X_full)} BOS примеров ---")
    
    trainer = MLTrainer()
    model, eval_results = trainer.train(X_full, y_full)
    
    # Сохраняем модель
    joblib.dump(model, 'smc_model_v1.joblib')
    print(f"✅ Модель SMC сохранена: smc_model_v1.joblib")
    
    print(f"Точность фильтрации: {eval_results['accuracy']:.2%}")
    print("Топ признаков:")
    sorted_f = sorted(eval_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for f, v in sorted_f[:5]:
        print(f"- {f}: {v:.4f}")

if __name__ == "__main__":
    train_smc_strategy()
