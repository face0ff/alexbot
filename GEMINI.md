{
  "prompt": {
    "context": "Создаю торговый алгоритм для криптовалют на старших таймфреймах (H1+). Нужно реализовать систему обнаружения паттерна 'Impulse-Fib618-Pullback-Continuation' с последующим ML-обучением. Работаю через Gemini CLI для генерации production-ready кода.",
    
    "domain": "algorithmic_trading",
    "market": "crypto_futures",
    "timeframes": ["1h", "4h", "1d"],
    "language": "python",
    "frameworks": ["pandas", "numpy", "ta-lib или чистая реализация", "scikit-learn/xgboost", "ccxt или binance API"],
    
    "pattern_specification": {
      "name": "Impulse_Fib618_Pullback_Continuation_HTF",
      "id": "IFPC_0.618_HTF",
      
      "impulse_detection": {
        "min_atr_multiplier": 2.0,
        "min_candles": 4,
        "min_body_ratio": 0.6,
        "max_internal_retracement": 0.30,
        "displacement_required": true,
        "visual_clarity": "импульс должен быть очевиден без индикаторов"
      },
      
      "pullback_detection": {
        "fib_range": {"min": 0.50, "max": 0.705, "optimal": 0.618, "tolerance": 0.025},
        "max_duration_candles": 12,
        "require_slowdown": true,
        "valid_depth_formula": "abs(high - low) / impulse_range"
      },
      
      "structure_requirements": {
        "swing_confirmation_candles": 2,
        "extremum_type": "higher_high_or_lower_low",
        "structure_break_required": true,
        "break_confirmation": "close_beyond_key_level"
      },
      
      "entry_trigger": {
        "type": "close_below_structure",
        "confirmations": ["rejection_wick", "body_dominance", "range_expansion"]
      },
      
      "risk_management": {
        "stop_loss": {"type": "beyond_extremum", "buffer_atr": 0.15},
        "take_profit": [
          {"type": "impulse_extreme", "rr_min": 2.5},
          {"type": "fib_extension", "level": 1.272}
        ],
        "max_bars_in_trade": 40
      }
    },
    
    "deliverables": [
      {
        "phase": 1,
        "name": "data_engine",
        "description": "Загрузка и подготовка данных H1+ с Binance",
        "requirements": [
          "Функция fetch_ohlcv(symbol, timeframe, start_date, end_date) через ccxt",
          "Валидация: пропуски, выбросы, низкая волатильность",
          "Расчет ATR, свинговых high/low, структуры рынка",
          "Сохранение в parquet для скорости"
        ]
      },
      {
        "phase": 2,
        "name": "pattern_detector",
        "description": "Алгоритм разметки импульсов и коррекций",
        "requirements": [
          "Функция detect_impulses(df) — находит все импульсы по спецификации",
          "Функция measure_pullback(impulse, df) — расчет глубины коррекции по Фибо",
          "Функция validate_structure(impulse, pullback) — проверка слома структуры",
          "Возврат списка размеченных паттернов с метаданными",
          "Логирование почему отбракован (для отладки)"
        ]
      },
      {
        "phase": 3,
        "name": "feature_engineering",
        "description": "Создание признаков для ML из размеченных паттернов",
        "features": [
          "impulse_range_atr: диапазон импульса в единицах ATR",
          "impulse_duration: количество свечей импульса",
          "pullback_depth: фактическая глубина коррекции (0-1)",
          "pullback_duration: свечей в коррекции",
          "volatility_contraction: bool, сжатие волатильности в конце коррекции",
          "extremum_wick_ratio: соотношение тени к телу на экстремуме",
          "structure_break_strength: насколько уверенно сломана структура",
          "momentum_divergence: bool, дивергенция если есть",
          "volume_profile: изменение объема на импульсе vs коррекции"
        ]
      },
      {
        "phase": 4,
        "name": "backtest_engine",
        "description": "Rule-based бэктест без ML для валидации edge",
        "requirements": [
          "Симуляция входов по паттерну с 2019 по 2024",
          "Расчет R-multiple для каждой сделки",
          "Метрики: winrate, expectancy, profit factor, max drawdown, Sharpe",
          "Эквити-кривая, распределение R",
          "Сравнение с buy-and-hold и случайными входами"
        ]
      },
      {
        "phase": 5,
        "name": "ml_classifier",
        "description": "XGBoost для фильтрации ложных паттернов",
        "requirements": [
          "Бинарная классификация: успешная сделка (R >= 1.5) vs неуспешная",
          "Train/test split по времени (не случайно)",
          "Cross-validation с walk-forward анализом",
          "Feature importance, SHAP values для интерпретации",
          "Сохранение модели, pipeline предобработки"
        ]
      }
    ],
    
    "code_structure": {
      "project_root": "impulse_fib_trader/",
      "modules": {
        "data/": ["fetcher.py", "cleaner.py", "storage.py"],
        "pattern/": ["impulse.py", "pullback.py", "structure.py", "detector.py"],
        "features/": ["engineer.py", "labels.py"],
        "backtest/": ["engine.py", "metrics.py", "reporting.py"],
        "ml/": ["train.py", "predict.py", "evaluate.py"],
        "config/": ["settings.yaml", "pattern_spec.json"],
        "notebooks/": ["eda.ipynb", "backtest_results.ipynb"],
        "tests/": ["test_detector.py", "test_features.py"]
      }
    },
    
    "constraints": {
      "no_pandas_loop": "Все расчеты векторизованы, никаких iterrows",
      "no_lookahead_bias": "Строгое разделение past/future в разметке",
      "reproducibility": "Фиксированные random seeds, версии библиотек в requirements.txt",
      "logging": "Подробные логи на каждом этапе для отладки",
      "modularity": "Каждый модуль тестируется изолированно"
    },
    
    "output_format": {
      "code": "Python 3.10+, type hints, docstrings Google style",
      "config": "YAML/JSON для параметров паттерна (легко менять без кода)",
      "tests": "pytest, покрытие критических функций",
      "documentation": "README с запуском, примерами, результатами бэктеста"
    },
    
    "start_with": "Начни с phase 1 и 2: data_engine + pattern_detector. Нужен рабочий код который на исторических данных ETHUSDT H1 находит минимум 50 паттернов за 2023 год и выводит статистику по ним (глубина коррекции, длительность, успешность).",
    
    "important": "Это НЕ high-frequency trading. На H1+ важна точность разметки, а не скорость. Код должен быть понятным и поддерживаемым, с четким разделением ответственности между модулями."
  }
}