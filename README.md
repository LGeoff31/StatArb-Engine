# StatArb Engine: Statistical Arbitrage Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A production-ready statistical arbitrage trading system that automates pair trading strategies using cointegration analysis and mean reversion techniques. Demonstrates advanced quantitative finance concepts used by leading trading firms.

## Overview

Automated system that identifies cointegrated stock pairs and executes market-neutral trades when price spreads deviate from historical norms, profiting from mean reversion.


## 🏗️ Architecture

graph TD
    A[Stock Symbols] --> B[Data Fetcher]
    B --> C[Historical Prices]
    C --> D[Pair Selector]
    D --> E[Cointegration Analysis]
    E --> F[Best Pair Selected]
    F --> G[Signal Generator]
    G --> H[Z-Score Calculation]
    H --> I[Trading Signals]
    I --> J[Backtester]
    J --> K[Simulate Trades]
    K --> L[Apply Costs & Slippage]
    L --> M[Backtest Results]
    M --> N[Performance Analyzer]
    M --> O[Visualizer]
    N --> P[Performance Metrics]
    O --> Q[Charts & Graphs]
    
    style A fill:#e1f5ff
    style F fill:#fff4e1
    style I fill:#ffe1f5
    style P fill:#e1ffe1
    style Q fill:#e1ffe1


## Features

- **Pair Discovery**: Automated cointegration testing using Engle-Granger methodology
- **Signal Generation**: Z-score based entry/exit signals with stop-loss protection
- **Backtesting**: Realistic simulation with transaction costs (0.1%) and slippage (0.05%)
- **Performance Analytics**: Sharpe ratio, Sortino ratio, maximum drawdown analysis
- **Visualization**: Comprehensive charts for prices, spreads, signals, and performance


### Installation
pip install -r requirements.txt

## Quick Start
python3 main.py
