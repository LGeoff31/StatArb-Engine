"""
Backtesting engine for pair trading strategy.
Simulates trading with realistic transaction costs and slippage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class Backtester:
    """Backtests pair trading strategies."""

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def calculate_returns(
        self,
        signals: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate strategy returns.

        Args:
            signals: DataFrame with trading signals and positions
            symbol1: First symbol
            symbol2: Second symbol
            prices: DataFrame with prices

        Returns:
            DataFrame with returns and portfolio value
        """
        results = signals.copy()

        # Calculate price returns
        results[f'{symbol1}_returns'] = prices[symbol1].pct_change()
        results[f'{symbol2}_returns'] = prices[symbol2].pct_change()

        # Calculate position changes
        results['position_change'] = results['position'].diff().abs()

        # Calculate strategy returns
        # For long spread: profit when spread increases
        # For short spread: profit when spread decreases
        results['strategy_return'] = 0.0

        for i in range(1, len(results)):
            prev_pos = results['position'].iloc[i-1]
            curr_pos = results['position'].iloc[i]

            if prev_pos == 0:
                # No position, no return
                results.loc[results.index[i], 'strategy_return'] = 0.0
            else:
                # Calculate return based on position
                if prev_pos == 1:  # Long spread
                    # Profit when spread increases (symbol2 outperforms symbol1)
                    spread_return = (
                        results[f'{symbol2}_returns'].iloc[i] -
                        results[f'{symbol1}_returns'].iloc[i]
                    )
                else:  # Short spread (prev_pos == -1)
                    # Profit when spread decreases (symbol1 outperforms symbol2)
                    spread_return = (
                        results[f'{symbol1}_returns'].iloc[i] -
                        results[f'{symbol2}_returns'].iloc[i]
                    )

                results.loc[results.index[i],
                            'strategy_return'] = spread_return

        # Apply transaction costs
        results['transaction_costs'] = (
            results['position_change'] * self.transaction_cost
        )

        # Apply slippage
        results['slippage'] = (
            results['position_change'] * self.slippage
        )

        # Net return after costs
        results['net_return'] = (
            results['strategy_return'] -
            results['transaction_costs'] -
            results['slippage']
        )

        # Calculate cumulative returns
        results['cumulative_return'] = (
            1 + results['net_return']).cumprod() - 1

        # Calculate portfolio value
        results['portfolio_value'] = (
            self.initial_capital * (1 + results['cumulative_return'])
        )

        return results

    def run_backtest(
        self,
        signals: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        prices: pd.DataFrame
    ) -> Dict:
        """
        Run complete backtest.

        Args:
            signals: DataFrame with trading signals
            symbol1: First symbol
            symbol2: Second symbol
            prices: DataFrame with prices

        Returns:
            Dictionary with backtest results
        """
        results = self.calculate_returns(signals, symbol1, symbol2, prices)

        # Calculate statistics
        total_return = results['cumulative_return'].iloc[-1]
        final_value = results['portfolio_value'].iloc[-1]

        # Count trades
        entries = results['entry_signal'].abs().sum()
        exits = results['exit_signal'].abs().sum()

        # Calculate total costs
        total_costs = (
            results['transaction_costs'].sum() +
            results['slippage'].sum()
        )

        return {
            'results': results,
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': entries,
            'total_costs': total_costs,
            'initial_capital': self.initial_capital
        }
