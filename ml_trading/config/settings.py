# config/settings.py
"""
Central configuration for the ML Trading System.
All parameters are configurable here.
"""

# ─── Market Configuration ───────────────────────────────────────────────
MARKET = "NSE"
BENCHMARK_TICKER = "^NSEI"          # NIFTY 50 index
BENCHMARK_NAME   = "NIFTY50"

# Top NSE stocks to track
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "ITC.NS", "LT.NS", "AXISBANK.NS", "WIPRO.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS"
]

# ─── Data Configuration ──────────────────────────────────────────────────
DATA_START_DATE   = "2019-01-01"
DATA_END_DATE     = "2024-12-31"
DATA_FREQUENCY    = "1d"
MIN_HISTORY_DAYS  = 252             # ~1 year of trading days required

# ─── Feature Engineering ────────────────────────────────────────────────
SMA_WINDOWS        = [10, 20, 50, 200]
VOLATILITY_WINDOWS = [10, 20, 50]
RSI_PERIOD         = 14
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
MOMENTUM_PERIOD    = 10
ROC_PERIOD         = 10
VOLUME_MA_WINDOW   = 20
SHARPE_WINDOW      = 252            # rolling annual Sharpe
SORTINO_WINDOW     = 252
BETA_WINDOW        = 252

# ─── Label Creation ─────────────────────────────────────────────────────
PREDICTION_WINDOWS = [5, 10, 20]    # trading days
DEFAULT_WINDOW     = 10

# ─── Model Training ─────────────────────────────────────────────────────
MODELS_TO_TRAIN = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]
BEST_METRIC     = "roc_auc"         # metric used to select best model

# Walk-forward validation
TRAIN_YEARS = 4                     # years in training window
TEST_YEARS  = 1                     # years in test window
N_SPLITS    = 3                     # number of walk-forward folds

# Optuna hyperparameter search
N_OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT  = 300               # seconds

# ─── Hybrid Decision Engine ─────────────────────────────────────────────
WEIGHT_SHARPE   = 0.40
WEIGHT_MOMENTUM = 0.30
WEIGHT_ML       = 0.30

TOP_N_STOCKS    = 5                 # number of stocks to trade

# ─── Paper Trading ───────────────────────────────────────────────────────
INITIAL_CAPITAL       = 1_000_000   # ₹10 Lakhs
POSITION_SIZE_PCT     = 0.20        # 20% per stock (max 5 stocks)
TRANSACTION_COST_PCT  = 0.001       # 0.1% per trade
SLIPPAGE_PCT          = 0.0005      # 0.05%
MAX_DRAWDOWN_LIMIT    = 0.15        # stop trading if drawdown > 15%

# ─── Retraining Schedule ────────────────────────────────────────────────
RETRAIN_FREQUENCY = "weekly"        # "weekly" or "monthly"

# ─── Storage ────────────────────────────────────────────────────────────
MODEL_DIR        = "models/"
DATABASE_PATH    = "trading_system.db"  # SQLite (PostgreSQL optional)
LOG_DIR          = "logs/"

# ─── Database (PostgreSQL — optional) ───────────────────────────────────
USE_POSTGRESQL = False
PG_HOST        = "localhost"
PG_PORT        = 5432
PG_DATABASE    = "ml_trading"
PG_USER        = "postgres"
PG_PASSWORD    = "password"
