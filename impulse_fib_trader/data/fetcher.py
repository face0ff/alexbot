import ccxt
import pandas as pd
import time
from datetime import datetime
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, exchange_id: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} # Переключаем на спот
        })

    def get_active_symbols(self) -> List[str]:
        """Fetches all active USDT spot symbols, excluding leveraged tokens."""
        self.exchange.load_markets()
        symbols = []
        for s in self.exchange.symbols:
            # Берем только пары к USDT
            if '/USDT' in s and ':' not in s:
                # Исключаем токены с плечом (UP, DOWN, BULL, BEAR)
                base = s.split('/')[0]
                if not any(suffix in base for suffix in ['UP', 'DOWN', 'BULL', 'BEAR']):
                    symbols.append(s)
        return symbols

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches OHLCV data from the exchange.
        """
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        limit_end = self.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else self.exchange.milliseconds()
        
        all_ohlcv = []
        current_since = since
        
        retry_count = 0
        while current_since < limit_end and retry_count < 3:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                if len(ohlcv) < 1000:
                    break
                
                time.sleep(self.exchange.rateLimit / 1000)
                    
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                retry_count += 1
                time.sleep(1)
                continue
                
        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(limit_end, unit='ms')]
            
        return df
