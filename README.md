# StatArb Engine: Statistical Arbitrage Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A production-ready statistical arbitrage trading system that automates pair trading strategies using cointegration analysis and mean reversion techniques. Demonstrates advanced quantitative finance concepts used by leading trading firms.

## Overview

Automated system that identifies cointegrated stock pairs and executes market-neutral trades when price spreads deviate from historical norms, profiting from mean reversion.

**Pair Discovery**: Automated cointegration testing using Engle-Granger methodology
**Signal Generation**: Z-score based entry/exit signals with stop-loss protection
**Backtesting**: Realistic simulation with transaction costs (0.1%) and slippage (0.05%)
**Performance Analytics**: Sharpe ratio, Sortino ratio, maximum drawdown analysis
**Visualization**: Comprehensive charts for prices, spreads, signals, and performance
| Pair Prices | Performance | Signals | Drawdown |
| -- | -- | -- | -- |
| <img width="901" height="447" alt="image" src="https://github.com/user-attachments/assets/f97b9445-0767-4b02-adf9-7f1387986d4d" /> | 
<img width="794" height="566" alt="image" src="https://github.com/user-attachments/assets/3ec07cb7-7e07-4fb3-b78a-bc93894ed9b7" /> |
<img width="796" height="567" alt="image" src="https://github.com/user-attachments/assets/53177146-f725-4970-8967-d225a739c7bd" /> |
<img width="901" height="383" alt="image" src="https://github.com/user-attachments/assets/db53162d-626b-4cd0-b408-85cee3fa2d9b" /> |


https://github.com/user-attachments/assets/f2fb288f-2837-4c6c-8f2f-9e8f10ae6b95


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




### Installation
pip install -r requirements.txt

## Quick Start
python3 main.py
