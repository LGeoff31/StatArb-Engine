"""
Visualization module for pair trading analysis.
Creates charts for prices, spreads, signals, and performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict


class Visualizer:
    """Creates visualizations for pair trading analysis."""

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_pair_prices(
        self,
        prices: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        save_path: str = None
    ):
        """
        Plot normalized prices of both symbols.

        Args:
            prices: DataFrame with prices
            symbol1: First symbol
            symbol2: Second symbol
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize prices to start at 100
        norm_price1 = (prices[symbol1] / prices[symbol1].iloc[0]) * 100
        norm_price2 = (prices[symbol2] / prices[symbol2].iloc[0]) * 100

        ax.plot(norm_price1.index, norm_price1.values,
                label=symbol1, linewidth=2)
        ax.plot(norm_price2.index, norm_price2.values,
                label=symbol2, linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (Base = 100)')
        ax.set_title(f'Normalized Prices: {symbol1} vs {symbol2}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_spread_and_signals(
        self,
        signals: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot spread, z-score, and trading signals.

        Args:
            signals: DataFrame with signals
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Spread
        axes[0].plot(signals.index, signals['spread'],
                     label='Spread', linewidth=1.5, color='blue')
        axes[0].axhline(y=signals['spread'].mean(), color='red', linestyle='--',
                        label=f"Mean: {signals['spread'].mean():.2f}", alpha=0.7)
        axes[0].fill_between(
            signals.index,
            signals['spread'].mean() - signals['spread'].std(),
            signals['spread'].mean() + signals['spread'].std(),
            alpha=0.2, color='gray', label='±1 Std Dev'
        )
        axes[0].set_ylabel('Spread')
        axes[0].set_title('Price Spread Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Z-score with thresholds
        axes[1].plot(signals.index, signals['zscore'],
                     label='Z-Score', linewidth=1.5, color='purple')
        axes[1].axhline(y=2.0, color='red', linestyle='--',
                        label='Entry Threshold (+2)', alpha=0.7)
        axes[1].axhline(y=-2.0, color='red', linestyle='--',
                        label='Entry Threshold (-2)', alpha=0.7)
        axes[1].axhline(y=0.5, color='green', linestyle='--',
                        label='Exit Threshold (+0.5)', alpha=0.7)
        axes[1].axhline(y=-0.5, color='green', linestyle='--',
                        label='Exit Threshold (-0.5)', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title('Z-Score of Spread')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Positions
        axes[2].plot(signals.index, signals['position'], label='Position',
                     linewidth=2, color='orange', drawstyle='steps-post')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Date')
        axes[2].set_title(
            'Trading Positions (1=Long Spread, -1=Short Spread, 0=No Position)')
        axes[2].set_yticks([-1, 0, 1])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_performance(
        self,
        results: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot portfolio performance over time.

        Args:
            results: DataFrame with backtest results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Portfolio value
        axes[0].plot(results.index, results['portfolio_value'],
                     label='Portfolio Value', linewidth=2, color='green')
        axes[0].axhline(y=results['portfolio_value'].iloc[0], color='red',
                        linestyle='--', label='Initial Capital', alpha=0.7)
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Cumulative returns
        axes[1].plot(results.index, results['cumulative_return'] * 100,
                     label='Cumulative Return', linewidth=2, color='blue')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Cumulative Return (%)')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Cumulative Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_drawdown(
        self,
        results: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot drawdown chart.

        Args:
            results: DataFrame with backtest results
            save_path: Optional path to save figure
        """
        portfolio_value = results['portfolio_value']
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1.5)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.set_title('Portfolio Drawdown Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def print_performance_summary(self, metrics: Dict):
        """
        Print formatted performance summary.

        Args:
            metrics: Dictionary with performance metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Return:           {metrics['total_return']*100:>8.2f}%")
        print(f"Annual Return:          {metrics['annual_return']*100:>8.2f}%")
        print(
            f"Annual Volatility:      {metrics['annual_volatility']*100:>8.2f}%")
        print(f"\nRisk-Adjusted Metrics:")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:>8.2f}")
        print(f"Sortino Ratio:          {metrics['sortino_ratio']:>8.2f}")
        print(f"Calmar Ratio:           {metrics['calmar_ratio']:>8.2f}")
        print(f"\nDrawdown Analysis:")
        print(f"Max Drawdown:           {metrics['max_drawdown']*100:>8.2f}%")
        print(f"Peak Date:              {metrics['peak_date']}")
        print(f"Trough Date:            {metrics['trough_date']}")
        if metrics['recovery_date']:
            print(f"Recovery Date:          {metrics['recovery_date']}")
        print(f"\nTrade Statistics:")
        print(f"Win Rate:               {metrics['win_rate']*100:>8.2f}%")
        print(f"Profit Factor:          {metrics['profit_factor']:>8.2f}")
        print(
            f"Days in Market:         {metrics['days_in_market']*100:>8.2f}%")
        print(f"Total Trading Days:     {metrics['trading_days']:>8.0f}")
        print("="*60 + "\n")
