import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Dict, Tuple

class MLTrainer:
    def __init__(self, model_params: Dict = None):
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.model_params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Trains the XGBoost classifier using time-based validation.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        # Simple train/test split for final evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return self.model, {
            'accuracy': accuracy,
            'report': report,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)
