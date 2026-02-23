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
            'options': {'defaultType': 'future'}
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches OHLCV data from the exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT').
            timeframe: Timeframe (e.g., '1h', '4h').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: Optional end date in 'YYYY-MM-DD' format.
            
        Returns:
            pd.DataFrame: OHLCV data.
        """
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        limit_end = self.exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else self.exchange.milliseconds()
        
        all_ohlcv = []
        current_since = since
        
        while current_since < limit_end:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
                if len(ohlcv) < 1000:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                time.sleep(5)
                continue
                
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Filter by end_date if provided
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(limit_end, unit='ms')]
            
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    data = fetcher.fetch_ohlcv('ETH/USDT', '1h', '2023-01-01', '2023-12-31')
    print(data.head())
    print(f"Total rows: {len(data)}")
