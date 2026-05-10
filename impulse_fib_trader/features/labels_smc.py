import pandas as pd
import numpy as np

class SMCLabeler:
    def create_labels(self, df: pd.DataFrame, rr: float = 2.0, max_bars: int = 48) -> pd.DataFrame:
        """
        Создает целевую переменную (Y):
        1 - Сделка дошла до TP
        0 - Сделка закрылась в убыток или по времени
        """
        bos_indices = df[df['bos_signal'] != 0].index
        labels = []

        for idx in bos_indices:
            pos = df.index.get_loc(idx)
            candle = df.iloc[pos]
            bos_type = df['bos_signal'].iloc[pos]
            entry_p = candle['close']
            
            # Динамический SL (за ближайший свинг)
            if bos_type == 1: # Bullish BOS
                sl_p = df['swing_low'].iloc[max(0, pos-20):pos].min()
                if np.isnan(sl_p) or sl_p >= entry_p: sl_p = entry_p * 0.99
                tp_p = entry_p + rr * (entry_p - sl_p)
            else: # Bearish BOS
                sl_p = df['swing_high'].iloc[max(0, pos-20):pos].max()
                if np.isnan(sl_p) or sl_p <= entry_p: sl_p = entry_p * 1.01
                tp_p = entry_p - rr * (sl_p - entry_p)
            
            # Симуляция выхода
            label = 0 # По умолчанию убыток/шум
            for i in range(pos + 1, min(pos + max_bars + 1, len(df))):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                
                if bos_type == 1: # Long
                    if low <= sl_p: break
                    if high >= tp_p: label = 1; break
                else: # Short
                    if high >= sl_p: break
                    if low <= tp_p: label = 1; break
            
            labels.append({'timestamp': idx, 'target': label})
            
        return pd.DataFrame(labels).set_index('timestamp')
