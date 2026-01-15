# StatArb Engine: Statistical Arbitrage Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A production-ready statistical arbitrage trading system that automates pair trading strategies using cointegration analysis and mean reversion techniques. Demonstrates advanced quantitative finance concepts used by leading trading firms.

## Overview

Automated system that identifies cointegrated stock pairs and executes market-neutral trades when price spreads deviate from historical norms, profiting from mean reversion.


## Architecture

flowchart LR
    A[Data] --> B[Pair Selection]
    B --> C[Signal Generation]
    C --> D[Backtesting]
    D --> E[Performance Analysis]
    D --> F[Visualization]
    
    style A fill:#4a90e2
    style B fill:#f5a623
    style C fill:#7ed321
    style D fill:#bd10e0
    style E fill:#50e3c2
    style F fill:#50e3c2



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
