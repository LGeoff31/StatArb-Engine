"""
Pair selection module using cointegration analysis.
Identifies pairs of stocks that move together over time.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from typing import List, Tuple, Dict
from itertools import combinations


class PairSelector:
    """Selects trading pairs based on cointegration analysis."""

    def __init__(self, pvalue_threshold: float = 0.05):
        """
        Initialize pair selector.

        Args:
            pvalue_threshold: Maximum p-value for cointegration test
        """
        self.pvalue_threshold = pvalue_threshold
        self.pairs = []

    def engle_granger_test(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Perform Engle-Granger cointegration test.

        Args:
            x: First time series
            y: Second time series

        Returns:
            Tuple of (cointegration statistic, p-value, critical values)
        """
        # Run OLS regression: y = alpha + beta * x + error
        ols_result = OLS(y, pd.DataFrame({'x': x, 'const': 1})).fit()

        # Get residuals
        residuals = ols_result.resid

        # Test residuals for stationarity (ADF test)
        adf_result = adfuller(residuals, maxlag=1, autolag='AIC')

        t_stat = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]

        return t_stat, p_value, critical_values

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for the spread.
        Half-life indicates how long it takes for the spread to revert halfway to its mean.

        Args:
            spread: Price spread time series

        Returns:
            Half-life in periods (days)
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag

        # Remove NaN values
        valid_idx = ~(spread_lag.isna() | spread_diff.isna())
        spread_lag = spread_lag[valid_idx]
        spread_diff = spread_diff[valid_idx]

        if len(spread_lag) < 2:
            return np.inf

        # OLS regression: spread_diff = theta * spread_lag + error
        ols_result = OLS(spread_diff, spread_lag).fit()
        theta = ols_result.params.iloc[0]

        if theta >= 0:
            return np.inf

        half_life = -np.log(2) / theta
        return half_life

    def calculate_spread_stats(
        self,
        spread: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate statistics for the spread.

        Args:
            spread: Price spread time series

        Returns:
            Dictionary with spread statistics
        """
        mean = spread.mean()
        std = spread.std()

        # Calculate z-score
        z_scores = (spread - mean) / std

        return {
            'mean': mean,
            'std': std,
            'min_zscore': z_scores.min(),
            'max_zscore': z_scores.max(),
            'half_life': self.calculate_half_life(spread)
        }

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        min_correlation: float = 0.7
    ) -> List[Dict]:
        """
        Find all cointegrated pairs from a set of stocks.

        Args:
            prices: DataFrame with stock prices (columns = symbols)
            min_correlation: Minimum correlation threshold

        Returns:
            List of dictionaries with pair information
        """
        symbols = prices.columns.tolist()
        cointegrated_pairs = []
        all_pairs_info = []  # Store info for all pairs for debugging

        print(
            f"Testing {len(list(combinations(symbols, 2)))} pairs for cointegration...")

        for symbol1, symbol2 in combinations(symbols, 2):
            try:
                # Calculate correlation
                correlation = prices[symbol1].corr(prices[symbol2])

                if abs(correlation) < min_correlation:
                    all_pairs_info.append({
                        'pair': f"{symbol1}-{symbol2}",
                        'correlation': correlation,
                        'pvalue': None,
                        'reason': 'Low correlation'
                    })
                    continue

                # Test for cointegration
                score, pvalue, _ = coint(prices[symbol1], prices[symbol2])

                all_pairs_info.append({
                    'pair': f"{symbol1}-{symbol2}",
                    'correlation': correlation,
                    'pvalue': pvalue,
                    'reason': 'Tested'
                })

                if pvalue < self.pvalue_threshold:
                    # Calculate spread
                    ols_result = OLS(
                        prices[symbol2],
                        pd.DataFrame({'x': prices[symbol1], 'const': 1})
                    ).fit()

                    hedge_ratio = ols_result.params['x']
                    spread = prices[symbol2] - hedge_ratio * prices[symbol1]

                    # Calculate spread statistics
                    spread_stats = self.calculate_spread_stats(spread)

                    pair_info = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'cointegration_pvalue': pvalue,
                        'cointegration_score': score,
                        'hedge_ratio': hedge_ratio,
                        'spread_mean': spread_stats['mean'],
                        'spread_std': spread_stats['std'],
                        'half_life': spread_stats['half_life'],
                        'spread': spread
                    }

                    cointegrated_pairs.append(pair_info)
                    print(f"Found cointegrated pair: {symbol1} - {symbol2} "
                          f"(p-value: {pvalue:.4f}, correlation: {correlation:.3f})")

            except Exception as e:
                print(f"Error testing pair {symbol1}-{symbol2}: {e}")
                continue

        # Print summary if no pairs found
        if not cointegrated_pairs:
            print("\nNo strictly cointegrated pairs found. Summary of tested pairs:")
            tested_pairs = [
                p for p in all_pairs_info if p['pvalue'] is not None]
            tested_pairs.sort(
                key=lambda x: x['pvalue'] if x['pvalue'] else 1.0)
            for p in tested_pairs[:5]:  # Show top 5
                print(f"  {p['pair']}: correlation={p['correlation']:.3f}, "
                      f"p-value={p['pvalue']:.4f}")

        # Sort by p-value (lower is better)
        cointegrated_pairs.sort(key=lambda x: x['cointegration_pvalue'])

        self.pairs = cointegrated_pairs
        return cointegrated_pairs

    def find_correlation_based_pairs(
        self,
        prices: pd.DataFrame,
        min_correlation: float = 0.7,
        max_pvalue: float = 0.20
    ) -> List[Dict]:
        """
        Find pairs based on correlation and mean-reversion characteristics,
        even if they don't pass strict cointegration tests.

        Args:
            prices: DataFrame with stock prices
            min_correlation: Minimum correlation threshold
            max_pvalue: Maximum p-value to consider (more lenient)

        Returns:
            List of dictionaries with pair information
        """
        symbols = prices.columns.tolist()
        candidate_pairs = []

        print(f"\nFinding correlation-based pairs (lenient cointegration test)...")

        for symbol1, symbol2 in combinations(symbols, 2):
            try:
                correlation = prices[symbol1].corr(prices[symbol2])

                if abs(correlation) < min_correlation:
                    continue

                # Test for cointegration with lenient threshold
                score, pvalue, _ = coint(prices[symbol1], prices[symbol2])

                if pvalue < max_pvalue:
                    # Calculate spread
                    ols_result = OLS(
                        prices[symbol2],
                        pd.DataFrame({'x': prices[symbol1], 'const': 1})
                    ).fit()

                    hedge_ratio = ols_result.params['x']
                    spread = prices[symbol2] - hedge_ratio * prices[symbol1]

                    # Calculate spread statistics
                    spread_stats = self.calculate_spread_stats(spread)

                    # Check if spread shows mean-reversion characteristics
                    # (reasonable half-life, not infinite)
                    if spread_stats['half_life'] < 252 and spread_stats['half_life'] > 0:
                        pair_info = {
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'cointegration_pvalue': pvalue,
                            'cointegration_score': score,
                            'hedge_ratio': hedge_ratio,
                            'spread_mean': spread_stats['mean'],
                            'spread_std': spread_stats['std'],
                            'half_life': spread_stats['half_life'],
                            'spread': spread
                        }

                        candidate_pairs.append(pair_info)
                        print(f"Found candidate pair: {symbol1} - {symbol2} "
                              f"(p-value: {pvalue:.4f}, correlation: {correlation:.3f}, "
                              f"half-life: {spread_stats['half_life']:.1f} days)")

            except Exception as e:
                continue

        # Sort by correlation (higher is better) and then by p-value
        candidate_pairs.sort(
            key=lambda x: (-abs(x['correlation']), x['cointegration_pvalue']))

        return candidate_pairs

    def select_best_pair(
        self,
        prices: pd.DataFrame,
        min_correlation: float = 0.7,
        use_fallback: bool = True
    ) -> Dict:
        """
        Select the best cointegrated pair, with fallback to correlation-based pairs.

        Args:
            prices: DataFrame with stock prices
            min_correlation: Minimum correlation threshold
            use_fallback: If True, use correlation-based pairs if no cointegrated pairs found

        Returns:
            Dictionary with best pair information
        """
        pairs = self.find_cointegrated_pairs(prices, min_correlation)

        if not pairs and use_fallback:
            print(
                "\nNo strictly cointegrated pairs found. Using correlation-based selection...")
            pairs = self.find_correlation_based_pairs(
                prices, min_correlation, max_pvalue=0.20)

            if not pairs:
                # Try even more lenient
                print("Trying with even more lenient criteria...")
                pairs = self.find_correlation_based_pairs(
                    prices, min_correlation=0.6, max_pvalue=0.30)

        if not pairs:
            raise ValueError(
                "No suitable pairs found. Try:\n"
                "- Different stocks (e.g., same sector ETFs)\n"
                "- Lower correlation threshold\n"
                "- Longer time period"
            )

        # Select pair with best combination of correlation and p-value
        best_pair = pairs[0]

        print(
            f"\nSelected best pair: {best_pair['symbol1']} - {best_pair['symbol2']}")
        print(
            f"  Cointegration p-value: {best_pair['cointegration_pvalue']:.4f}")
        print(f"  Correlation: {best_pair['correlation']:.3f}")
        print(f"  Half-life: {best_pair['half_life']:.2f} days")

        return best_pair
