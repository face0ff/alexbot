import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Dict, Tuple

class MLTrainer:
    def __init__(self, model_params: Dict = None):
        self.model_params = model_params or {
            'n_estimators': 300, # Больше деревьев
            'max_depth': 6, # Чуть глубже
            'learning_rate': 0.03, # Медленнее учимся для точности
            'objective': 'binary:logistic',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.model_params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Trains the XGBoost classifier using time-based validation.
        """
        # Calculate class scale
        pos_count = sum(y == 1)
        neg_count = sum(y == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Обновляем модель с учетом весов
        self.model.set_params(scale_pos_weight=scale_weight)
        
        # Simple train/test split for final evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Печатаем расширенную метрику для трейдера
        print("\n" + "-"*30)
        print("ДЕТАЛЬНАЯ МЕТРИКА (Test Set):")
        print(f"Точность (Precision) для ПРОФИТА (Class 1): {report_dict['1']['precision']:.2%}")
        print(f"Охват (Recall) для ПРОФИТА (Class 1): {report_dict['1']['recall']:.2%}")
        print("-"*30)

        return self.model, {
            'accuracy': accuracy,
            'report': report_dict,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }


    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
