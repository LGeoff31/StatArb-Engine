"""
Performance metrics and risk analysis module.
Calculates key performance indicators for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceAnalyzer:
    """Analyzes performance of trading strategies."""

    def __init__(self):
        pass

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.sqrt(periods_per_year) * \
            excess_returns.mean() / returns.std()

        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation only).

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0

        sortino = (
            np.sqrt(periods_per_year) *
            excess_returns.mean() /
            downside_returns.std()
        )

        return sortino

    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> Dict:
        """
        Calculate maximum drawdown.

        Args:
            portfolio_value: Series of portfolio values

        Returns:
            Dictionary with max drawdown and related metrics
        """
        # Calculate running maximum
        running_max = portfolio_value.expanding().max()

        # Calculate drawdown
        drawdown = (portfolio_value - running_max) / running_max

        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
        max_drawdown_pos = portfolio_value.index.get_loc(max_drawdown_idx)

        # Find peak before drawdown
        peak_idx = portfolio_value[:max_drawdown_idx].idxmax()
        peak_value = portfolio_value.loc[peak_idx]

        # Find recovery point (if any)
        recovery_idx = None
        if max_drawdown_pos < len(portfolio_value) - 1:
            recovery_series = portfolio_value[max_drawdown_idx:]
            recovery_mask = recovery_series >= peak_value
            if recovery_mask.any():
                recovery_idx = recovery_series[recovery_mask].index[0]

        return {
            'max_drawdown': max_drawdown,
            'peak_date': peak_idx,
            'trough_date': max_drawdown_idx,
            'recovery_date': recovery_idx,
            'peak_value': peak_value,
            'trough_value': portfolio_value.loc[max_drawdown_idx]
        }

    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        portfolio_value: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Args:
            returns: Series of returns
            portfolio_value: Series of portfolio values
            periods_per_year: Number of trading periods per year

        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd = abs(self.calculate_max_drawdown(
            portfolio_value)['max_drawdown'])

        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0

        return annual_return / max_dd

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns).

        Args:
            returns: Series of returns

        Returns:
            Win rate as fraction
        """
        if len(returns) == 0:
            return 0.0

        return (returns > 0).sum() / len(returns)

    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Series of returns

        Returns:
            Profit factor
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def analyze_performance(
        self,
        results: pd.DataFrame,
        periods_per_year: int = 252
    ) -> Dict:
        """
        Comprehensive performance analysis.

        Args:
            results: DataFrame with backtest results
            periods_per_year: Number of trading periods per year

        Returns:
            Dictionary with all performance metrics
        """
        returns = results['net_return']
        portfolio_value = results['portfolio_value']

        # Basic metrics
        total_return = results['cumulative_return'].iloc[-1]
        annual_return = returns.mean() * periods_per_year
        annual_volatility = returns.std() * np.sqrt(periods_per_year)

        # Risk-adjusted metrics
        sharpe = self.calculate_sharpe_ratio(
            returns, periods_per_year=periods_per_year)
        sortino = self.calculate_sortino_ratio(
            returns, periods_per_year=periods_per_year)

        # Drawdown analysis
        drawdown_info = self.calculate_max_drawdown(portfolio_value)

        # Calmar ratio
        calmar = self.calculate_calmar_ratio(
            returns, portfolio_value, periods_per_year
        )

        # Trade statistics
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)

        # Count trading days
        total_days = len(results)
        trading_days = (results['position'] != 0).sum()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': drawdown_info['max_drawdown'],
            'peak_date': drawdown_info['peak_date'],
            'trough_date': drawdown_info['trough_date'],
            'recovery_date': drawdown_info['recovery_date'],
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_days': total_days,
            'trading_days': trading_days,
            'days_in_market': trading_days / total_days if total_days > 0 else 0
        }
