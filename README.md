# 📈 NSE QuantEdge: Hybrid ML Trading System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **"The bleeding edge of retail quantitative finance."**

Welcome to **NSE QuantEdge** — a complete, modular, end-to-end algorithmic trading system built exclusively for the Indian Stock Market (NSE). 

This is not just a script. It is a fully-fledged quantitative research and trading pipeline that fuses traditional portfolio theory (Sharpe, Sortino, Momentum) with advanced ensemble Machine Learning (XGBoost, LightGBM). It predicts 10-day forward excess returns against the NIFTY 50, ranks the top 20 blue-chip NSE stocks, and executes a simulated paper trading strategy with institutional-grade risk management.

---

## 🚀 Core Features

*   **📡 Institutional Data Pipeline:** Automated ingestion of historical OHLCV data for 20 top NSE stocks and the NIFTY 50 benchmark via Yahoo Finance, stored in an SQLite database.
*   **🧠 40+ Alpha Factors:** Feature engineering factory covering technical indicators (RSI, MACD, Stochastic), risk metrics (Rolling Beta, Max Drawdown, Sortino), and market-relative strength.
*   **🤖 Walk-Forward ML Engine:** Trains Random Forest, Gradient Boosting, XGBoost, and LightGBM models using walk-forward validation (4-year train, 1-year test) with Optuna hyperparameter tuning.
*   **⚖️ Hybrid Decision Engine:** Blends ML probability (30%), Rolling Sharpe (40%), and Momentum (30%) to generate a final ranking score.
*   **📉 Risk-First Paper Trading:** Simulated execution with configurable transaction costs (0.1%), slippage (0.05%), and a strict 15% max drawdown circuit breaker.
*   **🖥️ Real-Time Dashboard:** A dark-mode, 6-page Streamlit UI to monitor equity curves, model performance, daily predictions, and trade history.
*   **🔄 Automated Retraining:** Built-in scheduler for weekly/monthly model retraining.

---

## 🏗️ Project Architecture

```text
ml_trading/
├── config/
│   └── settings.py          # Central configuration (tickers, weights, capital)
├── data/
│   ├── collector.py         # Yahoo Finance downloader & SQLite manager
│   └── trading_system.db    # Local database (OHLCV, predictions, trades)
├── features/
│   └── engineer.py          # 40+ feature generation pipeline
├── models/
│   ├── best_model.pkl       # Saved best-performing model
│   └── xgboost_*.pkl        # Timestamped model artifacts
├── prediction/
│   └── engine.py            # Daily inference engine
├── training/
│   ├── trainer.py           # Walk-forward training & Optuna tuning
│   └── explainer.py         # Model explainability (SHAP)
├── trading/
│   ├── agent.py             # Paper trading execution engine
│   └── scheduler.py         # Automated retraining cron jobs
├── utils/                   # Helper functions
├── dashboard/
│   └── app.py               # Streamlit Dashboard (6 pages)
├── main.py                  # CLI Orchestrator
├── requirements.txt         # Dependencies
└── README.md



 
##Setup Environment
# Clone the repository
git clone https://github.com/yourusername/NSE-QuantEdge-ML.git
cd NSE-QuantEdge-ML

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt











