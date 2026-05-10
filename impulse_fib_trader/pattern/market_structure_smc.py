import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class SMCMarketStructure:
    def __init__(self, window: int = 5):
        """
        window: количество свечей слева и справа для подтверждения свинга (Fractal approach)
        """
        self.window = window

    def find_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Определяет Swing Highs и Swing Lows.
        """
        df = df.copy()
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        
        # Векторизованный поиск фракталов (локальных максимумов/минимумов)
        for i in range(self.window, len(df) - self.window):
            # Swing High
            if df['high'].iloc[i] == df['high'].iloc[i-self.window : i+self.window+1].max():
                df.at[df.index[i], 'swing_high'] = df['high'].iloc[i]
                
            # Swing Low
            if df['low'].iloc[i] == df['low'].iloc[i-self.window : i+self.window+1].min():
                df.at[df.index[i], 'swing_low'] = df['low'].iloc[i]
                
        return df

    def detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Определяет Break of Structure (BOS).
        Bullish BOS: Close > Previous Swing High
        Bearish BOS: Close < Previous Swing Low
        """
        df = self.find_swings(df)
        df['bos_signal'] = 0 # 1 - Bullish, -1 - Bearish
        
        last_high = np.nan
        last_low = np.nan
        
        for i in range(len(df)):
            # Обновляем последние свинги
            if not np.isnan(df['swing_high'].iloc[i]):
                last_high = df['swing_high'].iloc[i]
            if not np.isnan(df['swing_low'].iloc[i]):
                last_low = df['swing_low'].iloc[i]
            
            # Проверяем пробой структуры (BOS)
            if not np.isnan(last_high) and df['close'].iloc[i] > last_high:
                df.at[df.index[i], 'bos_signal'] = 1
                # После BOS старый хай считается "пробитым", сбрасываем его
                last_high = np.nan 
                
            elif not np.isnan(last_low) and df['close'].iloc[i] < last_low:
                df.at[df.index[i], 'bos_signal'] = -1
                last_low = np.nan
                
        return df

    def get_htf_bias(self, htf_df: pd.DataFrame) -> str:
        """
        HTF Bias (4h): Определение общего тренда по последнему BOS.
        """
        htf_df = self.detect_bos(htf_df)
        last_bos = htf_df['bos_signal'].replace(0, np.nan).ffill().iloc[-1]
        
        if last_bos == 1: return "bullish"
        if last_bos == -1: return "bearish"
        return "neutral"
