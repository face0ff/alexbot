import logging
import json
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from pattern.combined_detector import CombinedTASFibDetector
from features.engineer import FeatureEngineer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketScanner:
    def __init__(self, model_path: str = 'super_model_v2.joblib'):
        self.fetcher = DataFetcher()
        self.cleaner = DataCleaner()
        self.detector = CombinedTASFibDetector()
        self.fe = FeatureEngineer()
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"✅ SUPER-MODEL v2 загружена: {model_path}")
        else:
            logger.error(f"❌ Модель {model_path} не найдена! Сначала обучите её.")
            exit(1)

        # Загружаем белый список монет
        whitelist_path = 'impulse_fib_trader/config/whitelist.json'
        if os.path.exists(whitelist_path):
            with open(whitelist_path, 'r') as f:
                self.symbols = json.load(f)
            logger.info(f"📋 Загружен Whitelist: {len(self.symbols)} монет")
        else:
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            logger.warning("⚠️ Whitelist не найден, использую Топ-3")

    def scan(self, timeframe: str = '15m', lookback_candles: int = 3):
        """
        Сканирует рынок на наличие свежих паттернов.
        lookback_candles: насколько "старым" может быть сигнал (в свечах).
        """
        logger.info(f"🔍 Начинаю сканирование {len(self.symbols)} пар ({timeframe})...")
        signals = []
        
        # Данные за последние 3 дня для M15 более чем достаточно
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

        for symbol in self.symbols:
            try:
                df = self.fetcher.fetch_ohlcv(symbol, timeframe, start_date)
                if df.empty or len(df) < 200: continue
                
                df = self.cleaner.validate_data(df)
                df = self.cleaner.calculate_indicators(df)
                
                patterns = self.detector.detect_patterns(df)
                if not patterns: continue
                
                # Фильтруем только САМЫЕ СВЕЖИЕ паттерны (которые появились в последних N свечах)
                # entry_idx - это индекс свечи, на которой мы должны войти
                latest_idx = len(df) - 1
                fresh_patterns = [p for p in patterns if p['entry_idx'] >= latest_idx - lookback_candles]
                
                if fresh_patterns:
                    X = self.fe.extract_features(fresh_patterns, df)
                    preds = self.model.predict(X)
                    probs = self.model.predict_proba(X)
                    
                    for i, p in enumerate(fresh_patterns):
                        if preds[i] == 1:
                            signals.append({
                                'symbol': symbol,
                                'type': p['type'],
                                'entry_price': p['entry_price'],
                                'sl': p['sl'],
                                'tp': p['tp'],
                                'probability': float(probs[i][1]),
                                'timestamp': p['timestamp']
                            })
                            logger.info(f"🚀 СИГНАЛ: {symbol} | Prob: {probs[i][1]:.2%} | Price: {p['entry_price']}")
                            
            except Exception as e:
                logger.error(f"❌ Ошибка при сканировании {symbol}: {e}")
                
        return signals

    def report_signals(self, signals):
        if not signals:
            print("\n--- Сигналов не обнаружено ---")
            return

        print("\n" + "!"*60)
        print(f"НАЙДЕНО СИГНАЛОВ: {len(signals)}")
        print("!"*60)
        
        # Сортировка по вероятности успеха
        signals.sort(key=lambda x: x['probability'], reverse=True)
        
        for s in signals:
            print(f"\n🪙 МОНЕТА: {s['symbol']}")
            print(f"📈 ТИП: {s['type']}")
            print(f"🎯 ВХОД: {s['entry_price']:.5f}")
            print(f"🛑 СТОП: {s['sl']:.5f}")
            print(f"💰 ТЕЙК: {s['tp']:.5f}")
            print(f"🔥 УВЕРЕННОСТЬ ML: {s['probability']:.1%}")
            print("-" * 30)

if __name__ == "__main__":
    scanner = MarketScanner()
    found_signals = scanner.scan()
    scanner.report_signals(found_signals)
