import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.detector import PatternDetector
from features.engineer import FeatureEngineer
from ml.train import MLTrainer
import os
import time

# Настройка логирования (делаем тише для сканера)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketScanner:
    def __init__(self, config_path: str, model_path: str):
        self.fetcher = DataFetcher()
        self.cleaner = DataCleaner()
        self.detector = PatternDetector(config_path)
        self.fe = FeatureEngineer()
        self.trainer = MLTrainer()
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        if os.path.exists(model_path):
            self.trainer.load_model(model_path)
            print(f"Успешно загружена ML-модель: {model_path}")
        else:
            print("Ошибка: Файл модели не найден! Сначала запустите python main.py")
            exit(1)

    def scan_market(self, timeframe: str = '1h'):
        symbols = self.fetcher.get_active_symbols()
        print(f"Сканирование {len(symbols)} пар на спотовом рынке ({timeframe})...")
        
        signals = []
        # Берем данные за последние 7 дней, этого достаточно для H1 паттерна
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        for i, symbol in enumerate(symbols):
            if i % 20 == 0:
                print(f"Прогресс: {i}/{len(symbols)}...")
                
            try:
                df = self.fetcher.fetch_ohlcv(symbol, timeframe, start_date)
                if df.empty or len(df) < 40: continue
                
                df = self.cleaner.validate_data(df)
                df = self.cleaner.calculate_indicators(df)
                df = self.cleaner.identify_swings(df)
                
                patterns = self.detector.detect_patterns(df)
                if not patterns: continue
                
                # Ищем паттерны, где вход был в последние 3 часа
                latest_patterns = [p for p in patterns if p['structure']['entry_idx'] >= len(df) - 3]
                
                if latest_patterns:
                    X = self.fe.extract_features(latest_patterns, df)
                    preds = self.trainer.model.predict(X)
                    probs = self.trainer.model.predict_proba(X)
                    
                    for idx, p in enumerate(latest_patterns):
                        if preds[idx] == 1:
                            signals.append({
                                'symbol': symbol,
                                'pattern': p,
                                'ml_prob': float(probs[idx][1])
                            })
                            
            except Exception:
                continue # Игнорируем ошибки для отдельных пар
                
        return signals

    def provide_recommendations(self, signals):
        if not signals:
            print("\n" + "="*50)
            print("СИГНАЛОВ НЕ НАЙДЕНО (Рынок в коррекции или флэте)")
            print("="*50)
            return

        print("\n" + "!"*50)
        print("НАЙДЕНЫ АКТУАЛЬНЫЕ ТОРГОВЫЕ СИГНАЛЫ!")
        print("!"*50)
        
        # Сортируем по уверенности модели
        signals.sort(key=lambda x: x['ml_prob'], reverse=True)
        
        for s in signals:
            symbol = s['symbol']
            p = s['pattern']
            prob = s['ml_prob']
            
            entry = p['structure']['entry_price']
            sl_val = p['pullback']['low'] if p['impulse']['type'] == 'bullish' else p['pullback']['high']
            
            # Расчет уровней
            if p['impulse']['type'] == 'bullish':
                sl = sl_val * 0.997 # 0.3% отступ под лоу
                tp = entry + 2.0 * (entry - sl)
                side = "LONG (Покупка)"
            else:
                sl = sl_val * 1.003
                tp = entry - 2.0 * (sl - entry)
                side = "SHORT (Продажа)"
                
            print(f"\n[{symbol}] -> {side}")
            print(f"Доверие модели ML: {prob:.2%}")
            print(f"ЦЕНА ВХОДА: {entry:.6f}")
            print(f"СТОП-ЛОСС: {sl:.6f}")
            print(f"ТЕЙК-ПРОФИТ: {tp:.6f}")
            print(f"Время формирования: {p['timestamp']}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    scanner = MarketScanner('config/pattern_spec.json', 'trained_model.joblib')
    signals = scanner.scan_market('1h')
    scanner.provide_recommendations(signals)
