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
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator
from ml.train import MLTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    symbol = 'ETH/USDT'
    timeframe = '1h'
    start_date = '2022-01-01' # Extended range for ML
    end_date = '2023-12-31'
    data_path = f"data_{symbol.replace('/', '_')}_{timeframe}.parquet"

    # 1. Data Engine
    if not os.path.exists(data_path):
        logger.info(f"Fetching data for {symbol} {timeframe}...")
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
        
        logger.info("Cleaning and calculating indicators...")
        cleaner = DataCleaner()
        df = cleaner.validate_data(df)
        df = cleaner.calculate_indicators(df)
        df = cleaner.identify_swings(df)
        
        DataStorage.save_to_parquet(df, data_path)
    else:
        logger.info(f"Loading data from {data_path}")
        df = DataStorage.load_from_parquet(data_path)

    # 2. Pattern Detector
    logger.info("Running pattern detection...")
    config_path = 'config/pattern_spec.json'
    detector = PatternDetector(config_path)
    patterns = detector.detect_patterns(df)
    
    if not patterns:
        logger.warning("No patterns found!")
        return

    # 3. Backtest (Rule-based)
    logger.info("Running rule-based backtest...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    bt_engine = BacktestEngine(config)
    bt_results = bt_engine.run_backtest(patterns, df)
    
    metrics_calc = MetricsCalculator()
    rule_metrics = metrics_calc.calculate(bt_results)

    # 4. ML Phase
    logger.info("Feature engineering and labeling...")
    fe = FeatureEngineer()
    X = fe.extract_features(patterns, df)
    
    labeler = Labeler(config)
    y = labeler.create_labels(patterns, df)
    
    logger.info("Training ML model...")
    trainer = MLTrainer()
    model, ml_eval = trainer.train(X, y)
    
    # 5. ML-Filtered Backtest
    logger.info("Evaluating ML-filtered results...")
    # Use the model to predict on all patterns (for demonstration)
    # In production, we'd use a proper walk-forward.
    X_preds = model.predict(X)
    ml_filtered_results = bt_results[X_preds == 1]
    ml_metrics = metrics_calc.calculate(ml_filtered_results)

    # Print Comparison
    print("\n" + "="*50)
    print(f"STRATEGY COMPARISON: {symbol} {timeframe}")
    print("="*50)
    print(f"{'Metric':<20} | {'Rule-Based':<12} | {'ML-Filtered':<12}")
    print("-" * 50)
    for key in ['total_trades', 'win_rate', 'expectancy', 'profit_factor', 'net_profit_r']:
        val_rule = f"{rule_metrics.get(key, 0):.3f}" if isinstance(rule_metrics.get(key), float) else str(rule_metrics.get(key, 0))
        val_ml = f"{ml_metrics.get(key, 0):.3f}" if isinstance(ml_metrics.get(key), float) else str(ml_metrics.get(key, 0))
        print(f"{key:<20} | {val_rule:<12} | {val_ml:<12}")
    
    print("="*50)
    print("\nTop 5 Important Features for ML:")
    sorted_features = sorted(ml_eval['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        print(f"- {feat}: {imp:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
