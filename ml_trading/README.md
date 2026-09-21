# ML Trading System — NSE/NIFTY 50

A postgraduate-level machine learning system that predicts stock outperformance
on India's NSE exchange, with paper trading and a live Streamlit dashboard.

---

## Project Structure

```
ml_trading/
├── config/
│   └── settings.py          ← All configuration (stocks, weights, windows)
├── data/
│   └── collector.py         ← Downloads & stores OHLCV from Yahoo Finance
├── features/
│   └── engineer.py          ← 40+ technical & risk features pipeline
├── training/
│   ├── trainer.py           ← Walk-forward CV, Optuna tuning, 4 models
│   └── explainer.py         ← SHAP global & local explanations
├── prediction/
│   └── engine.py            ← Daily scoring (ML + Sharpe + Momentum)
├── trading/
│   ├── agent.py             ← Paper trading with P&L tracking
│   └── scheduler.py         ← Auto-retraining (weekly/monthly)
├── dashboard/
│   └── app.py               ← Streamlit dashboard
├── models/                  ← Saved .pkl models (auto-created)
├── main.py                  ← CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup in VS Code

### 1. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the folder in VS Code
```
File → Open Folder → select ml_trading/
```

In VS Code, select your venv interpreter:
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `./venv/...`

---

## Running the System

Run all commands from the `ml_trading/` root folder.

### Step 1 — Download 5 years of NSE data
```bash
python main.py --download
```
Downloads OHLCV for 20 NSE stocks + NIFTY 50 from Yahoo Finance.
Stores everything in `trading_system.db` (SQLite).

### Step 2 — Train the models
```bash
python main.py --train
```
- Engineers 40+ features per stock
- Trains Random Forest, Gradient Boosting, XGBoost, LightGBM
- Uses walk-forward cross-validation (no look-ahead bias)
- Runs Optuna hyperparameter search
- Saves best model to `models/best_model.pkl`

> ⚠️ Training takes 10–30 minutes on first run (Optuna runs 50 trials per model).
> Reduce `N_OPTUNA_TRIALS` in `config/settings.py` to speed up.

### Step 3 — Run daily predictions
```bash
python main.py --predict
```
Scores all 20 stocks using the hybrid formula:
```
Final Score = 0.40 × Sharpe + 0.30 × Momentum + 0.30 × ML Probability
```

### Step 4 — Execute paper trades
```bash
python main.py --trade
```
Buys top 5 ranked stocks, sells those dropped from rankings.

### Step 5 — Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`

### Full daily pipeline (Steps 3+4 in one command)
```bash
python main.py --all
```

### Force retrain
```bash
python main.py --force-retrain
```

### System status
```bash
python main.py --status
```

---

## Configuration

Edit `config/settings.py` to customise:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NSE_STOCKS` | 20 NSE tickers | Stocks to track |
| `DEFAULT_WINDOW` | 10 | Prediction horizon (days) |
| `N_OPTUNA_TRIALS` | 50 | Reduce to 10 for faster training |
| `TOP_N_STOCKS` | 5 | How many stocks to hold |
| `INITIAL_CAPITAL` | ₹10,00,000 | Paper trading capital |
| `WEIGHT_SHARPE` | 0.40 | Sharpe weight in hybrid score |
| `WEIGHT_ML` | 0.30 | ML weight in hybrid score |
| `RETRAIN_FREQUENCY` | "weekly" | Auto-retrain schedule |

---

## Dashboard Pages

| Page | Content |
|------|---------|
| 📊 Overview | Equity curve, KPIs, today's top picks |
| 🤖 Model | Metrics radar, version history |
| 📈 Portfolio | P&L chart, drawdown, trade stats |
| 🔮 Predictions | Score breakdown for all stocks |
| 📋 Trades | Full trade history table |
| 🔄 Retrain | Retrain log & manual controls |

---

## Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM
- **Tuning:** Optuna (TPE sampler)
- **Explainability:** SHAP (TreeExplainer)
- **Data:** yfinance, pandas, numpy, scipy
- **Storage:** SQLite (via sqlite3)
- **Dashboard:** Streamlit + Plotly
- **Persistence:** joblib

---

## Notes

- All data uses Yahoo Finance (`.NS` suffix for NSE tickers)
- Walk-forward validation prevents look-ahead bias
- Paper trading only — no real money involved
- SHAP explanations require the model to be trained first
- The system auto-upgrades to incremental data updates after initial download
