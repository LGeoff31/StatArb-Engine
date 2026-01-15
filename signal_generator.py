"""
Signal generation module for pair trading.
Generates entry/exit signals based on z-score deviations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class SignalGenerator:
    """Generates trading signals for pair trading strategy."""

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 3.0,
        lookback_window: int = 60
    ):
        """
        Initialize signal generator.

        Args:
            entry_threshold: Z-score threshold for entering a trade
            exit_threshold: Z-score threshold for exiting a trade
            stop_loss: Z-score threshold for stop-loss
            lookback_window: Window for calculating rolling statistics
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.lookback_window = lookback_window

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of the spread.

        Args:
            spread: Price spread time series
            lookback: Lookback window (None uses all history)

        Returns:
            Series of z-scores
        """
        if lookback is None:
            mean = spread.expanding().mean()
            std = spread.expanding().std()
        else:
            mean = spread.rolling(window=lookback).mean()
            std = spread.rolling(window=lookback).std()

        zscore = (spread - mean) / std
        return zscore

    def generate_signals(
        self,
        spread: pd.Series,
        hedge_ratio: float
    ) -> pd.DataFrame:
        """
        Generate trading signals based on z-score.

        Strategy:
        - When z-score > entry_threshold: Spread is too wide, short spread
          (sell symbol2, buy symbol1)
        - When z-score < -entry_threshold: Spread is too narrow, long spread
          (buy symbol2, sell symbol1)
        - Exit when z-score crosses back towards zero (within exit_threshold)
        - Stop loss if z-score exceeds stop_loss threshold

        Args:
            spread: Price spread time series
            hedge_ratio: Hedge ratio for the pair

        Returns:
            DataFrame with signals and positions
        """
        zscore = self.calculate_zscore(spread, self.lookback_window)

        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['zscore'] = zscore

        # Initialize position columns
        # 0: no position, 1: long spread, -1: short spread
        signals['position'] = 0
        # 1: enter long, -1: enter short, 0: no signal
        signals['entry_signal'] = 0
        signals['exit_signal'] = 0  # 1: exit position

        current_position = 0
        entry_zscore = 0

        for i in range(len(signals)):
            z = signals['zscore'].iloc[i]

            # No position
            if current_position == 0:
                # Enter long spread (spread is too negative, expect it to increase)
                if z < -self.entry_threshold:
                    current_position = 1
                    entry_zscore = z
                    signals.loc[signals.index[i], 'entry_signal'] = 1
                    signals.loc[signals.index[i], 'position'] = 1

                # Enter short spread (spread is too positive, expect it to decrease)
                elif z > self.entry_threshold:
                    current_position = -1
                    entry_zscore = z
                    signals.loc[signals.index[i], 'entry_signal'] = -1
                    signals.loc[signals.index[i], 'position'] = -1

            # Long position
            elif current_position == 1:
                # Check stop loss
                if z < -self.stop_loss:
                    current_position = 0
                    signals.loc[signals.index[i], 'exit_signal'] = 1
                    signals.loc[signals.index[i], 'position'] = 0

                # Exit when z-score moves back towards zero
                elif z > -self.exit_threshold:
                    current_position = 0
                    signals.loc[signals.index[i], 'exit_signal'] = 1
                    signals.loc[signals.index[i], 'position'] = 0
                else:
                    signals.loc[signals.index[i], 'position'] = 1

            # Short position
            elif current_position == -1:
                # Check stop loss
                if z > self.stop_loss:
                    current_position = 0
                    signals.loc[signals.index[i], 'exit_signal'] = 1
                    signals.loc[signals.index[i], 'position'] = 0

                # Exit when z-score moves back towards zero
                elif z < self.exit_threshold:
                    current_position = 0
                    signals.loc[signals.index[i], 'exit_signal'] = 1
                    signals.loc[signals.index[i], 'position'] = 0
                else:
                    signals.loc[signals.index[i], 'position'] = -1

        return signals

    def get_position_sizes(
        self,
        signals: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        prices: pd.DataFrame,
        capital: float = 100000
    ) -> pd.DataFrame:
        """
        Calculate position sizes for each symbol.

        Args:
            signals: DataFrame with trading signals
            symbol1: First symbol
            symbol2: Second symbol
            prices: DataFrame with prices
            capital: Total capital to allocate

        Returns:
            DataFrame with position sizes
        """
        positions = signals.copy()

        # Get hedge ratio from spread calculation
        # Assuming we're using the spread = symbol2 - hedge_ratio * symbol1
        # For now, we'll use equal dollar amounts (can be improved)

        positions[f'{symbol1}_shares'] = 0.0
        positions[f'{symbol2}_shares'] = 0.0

        for i in range(len(positions)):
            pos = positions['position'].iloc[i]

            if pos != 0:
                # Allocate capital equally between the two legs
                leg_capital = capital / 2

                price1 = prices[symbol1].iloc[i]
                price2 = prices[symbol2].iloc[i]

                if pos == 1:  # Long spread: buy symbol2, sell symbol1
                    positions.loc[positions.index[i],
                                  f'{symbol2}_shares'] = leg_capital / price2
                    positions.loc[positions.index[i],
                                  f'{symbol1}_shares'] = -leg_capital / price1

                elif pos == -1:  # Short spread: sell symbol2, buy symbol1
                    positions.loc[positions.index[i],
                                  f'{symbol2}_shares'] = -leg_capital / price2
                    positions.loc[positions.index[i],
                                  f'{symbol1}_shares'] = leg_capital / price1

        return positions
