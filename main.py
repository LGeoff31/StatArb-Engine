"""
Main execution script for Statistical Arbitrage / Pair Trading System.

This script demonstrates a complete pair trading workflow:
1. Fetch historical data for potential pairs
2. Find cointegrated pairs
3. Generate trading signals
4. Backtest the strategy
5. Analyze performance
6. Visualize results
"""

from data_fetcher import DataFetcher
from pair_selector import PairSelector
from signal_generator import SignalGenerator
from backtester import Backtester
from performance import PerformanceAnalyzer
from visualizer import Visualizer

import pandas as pd


def main():
    """Main execution function."""

    print("="*60)
    print("STATISTICAL ARBITRAGE / PAIR TRADING SYSTEM")
    print("="*60)
    print("\nThis system implements a pair trading strategy using")
    print("cointegration analysis to identify trading opportunities.\n")

    # Configuration
    # Example: Tech stocks that might be cointegrated
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']

    # Strategy parameters
    ENTRY_THRESHOLD = 2.0  # Z-score threshold for entry
    EXIT_THRESHOLD = 0.5   # Z-score threshold for exit
    STOP_LOSS = 3.0        # Z-score threshold for stop loss

    # Backtest parameters
    INITIAL_CAPITAL = 100000
    TRANSACTION_COST = 0.001  # 0.1%
    SLIPPAGE = 0.0005        # 0.05%

    # Step 1: Fetch data
    print("\n[Step 1] Fetching historical data...")
    print("-" * 60)
    fetcher = DataFetcher()

    try:
        prices = fetcher.fetch_data(
            symbols=SYMBOLS,
            period="2y"  # 2 years of data
        )
        print(f"\nSuccessfully fetched data for {len(prices)} trading days")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Step 2: Find cointegrated pairs
    print("\n[Step 2] Finding cointegrated pairs...")
    print("-" * 60)
    selector = PairSelector(pvalue_threshold=0.05)

    try:
        # This will automatically use fallback if no strictly cointegrated pairs found
        best_pair = selector.select_best_pair(
            prices, min_correlation=0.7, use_fallback=True)
        symbol1 = best_pair['symbol1']
        symbol2 = best_pair['symbol2']
        hedge_ratio = best_pair['hedge_ratio']
        spread = best_pair['spread']
    except Exception as e:
        print(f"Error finding pairs: {e}")
        return

    # Step 3: Generate signals
    print("\n[Step 3] Generating trading signals...")
    print("-" * 60)
    signal_gen = SignalGenerator(
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        stop_loss=STOP_LOSS
    )

    signals = signal_gen.generate_signals(spread, hedge_ratio)

    num_entries = signals['entry_signal'].abs().sum()
    num_exits = signals['exit_signal'].abs().sum()
    print(
        f"Generated {int(num_entries)} entry signals and {int(num_exits)} exit signals")

    # Step 4: Backtest
    print("\n[Step 4] Running backtest...")
    print("-" * 60)
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    backtest_results = backtester.run_backtest(
        signals, symbol1, symbol2, prices)
    results_df = backtest_results['results']

    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Value: ${backtest_results['final_value']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
    print(f"Total Trades: {backtest_results['total_trades']}")
    print(
        f"Total Costs: ${backtest_results['total_costs']*INITIAL_CAPITAL:,.2f}")

    # Step 5: Performance analysis
    print("\n[Step 5] Analyzing performance...")
    print("-" * 60)
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_performance(results_df)

    # Step 6: Visualization
    print("\n[Step 6] Creating visualizations...")
    print("-" * 60)
    visualizer = Visualizer()

    # Plot pair prices
    visualizer.plot_pair_prices(prices, symbol1, symbol2,
                                save_path='pair_prices.png')
    print("Saved: pair_prices.png")

    # Plot spread and signals
    visualizer.plot_spread_and_signals(signals, save_path='signals.png')
    print("Saved: signals.png")

    # Plot performance
    visualizer.plot_performance(results_df, save_path='performance.png')
    print("Saved: performance.png")

    # Plot drawdown
    visualizer.plot_drawdown(results_df, save_path='drawdown.png')
    print("Saved: drawdown.png")

    # Print performance summary
    visualizer.print_performance_summary(metrics)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print(f"- Trading pair: {symbol1} / {symbol2}")
    print(f"- Hedge ratio: {hedge_ratio:.4f}")
    print(f"- Strategy generated {int(num_entries)} trading opportunities")
    print(f"- Final portfolio value: ${backtest_results['final_value']:,.2f}")
    print(f"- Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"- Maximum drawdown: {metrics['max_drawdown']*100:.2f}%")
    print("\nCheck the generated PNG files for detailed visualizations.")


if __name__ == "__main__":
    main()
