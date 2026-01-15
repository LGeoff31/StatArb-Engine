import yfinance as yf
import pandas as pd
from typing import List, Tuple
from datetime import datetime, timedelta
import numpy as np


class DataFetcher:
    """
    Fetch and preprocess historical stock price data from Yahoo Finance.
    """

    def __init__(self):
        self.cache = {}

    def fetch_data(self, symbols: List[str], start_date: str = None, end_date: str = None, period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical price data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols (e.g. ['AAPL', 'GOOG', 'MSFT'])
            start_date: Start date (YYYY-MM-DD) or None for period-based
            end_date: End date (YYYY-MM-DD) or None for today
            period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            DataFrame with columns: Date, and one column per symbol (Close prices)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        data_dict = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                if start_date:
                    hist = ticker.history(start=start_date, end=end_date)
                else:
                    hist = ticker.history(period=period)

                if hist.empty:
                    print(f"No data found for {symbol}")
                    continue

                data_dict[symbol] = hist['Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue

        if not data_dict:
            raise ValueError("No data could be fetched for any symbol")

        # Combine into single DataFrame
        df = pd.DataFrame(data_dict)
        df.index.name = 'Date'

        # Remove rows with any NaN values
        df = df.dropna()

        return df

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from prices."""
        return prices.pct_change().dropna()

    def calculate_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns from prices."""
        return pd.DataFrame(
            {col: pd.Series(prices[col]).apply(lambda x: np.log(x)).diff()
             for col in prices.columns}
        ).dropna()
