# main.py
"""
ML Trading System — Main Entry Point
=====================================
Usage:
    python main.py --download    Download 5 years of NSE data
    python main.py --train       Train all models, save best
    python main.py --predict     Run daily predictions
    python main.py --trade       Execute paper trades
    python main.py --retrain     Incremental retrain (checks schedule)
    python main.py --all         Full daily pipeline (predict + trade)
    python main.py --status      Print system status

Dashboard:
    streamlit run dashboard/app.py
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_system.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Imports ───────────────────────────────────────────────────────────────
from config.settings import (
    NSE_STOCKS, BENCHMARK_TICKER, DATABASE_PATH,
    MODEL_DIR, DEFAULT_WINDOW
)
from data.collector       import DataCollector
from features.engineer    import build_feature_dataset, FeatureEngineer
from training.trainer     import ModelTrainer
from prediction.engine    import PredictionEngine
from trading.agent        import PaperTradingAgent
from trading.scheduler    import RetrainingScheduler


# ════════════════════════════════════════════════════════════════════════════
# Pipeline functions
# ════════════════════════════════════════════════════════════════════════════

def pipeline_download():
    """Download 5 years of OHLCV data for all NSE stocks + NIFTY."""
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading historical data")
    logger.info("=" * 60)
    collector = DataCollector(DATABASE_PATH)
    collector.download_all(NSE_STOCKS + [BENCHMARK_TICKER])
    status = collector.get_download_status()
    logger.info(f"\n{status.to_string()}")


def pipeline_train(pred_window: int = DEFAULT_WINDOW):
    """Feature engineering → model training → save best model."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training ML models")
    logger.info("=" * 60)

    # Load data
    collector    = DataCollector(DATABASE_PATH)
    stock_data   = collector.load_all_stocks(NSE_STOCKS)
    benchmark    = collector.load_benchmark()

    if not stock_data:
        logger.error("No stock data found. Run --download first.")
        return None

    if benchmark.empty:
        logger.error("No benchmark data found. Run --download first.")
        return None

    logger.info(f"Loaded {len(stock_data)} stocks")

    # Feature engineering
    logger.info("Engineering features …")
    feature_data = build_feature_dataset(stock_data, benchmark, pred_window)

    if not feature_data:
        logger.error("Feature engineering failed.")
        return None

    # Get feature columns
    first_df  = next(iter(feature_data.values()))
    engineer  = FeatureEngineer(pred_window)
    feat_cols = engineer.get_feature_columns(first_df)
    logger.info(f"Feature columns: {len(feat_cols)}")

    # Train
    trainer      = ModelTrainer(pred_window=pred_window)
    train_result = trainer.train_all(feature_data, feat_cols)

    if train_result and train_result.get("best"):
        best = train_result["best"]
        logger.info(f"\n{'='*40}")
        logger.info(f"BEST MODEL: {best['model_name']}")
        logger.info(f"ROC-AUC  : {best['metrics']['roc_auc']:.4f}")
        logger.info(f"Accuracy : {best['metrics']['accuracy']:.4f}")
        logger.info(f"F1 Score : {best['metrics']['f1']:.4f}")
        logger.info(f"{'='*40}")

    return feature_data, feat_cols


def pipeline_predict(feature_data=None, pred_window: int = DEFAULT_WINDOW):
    """Run daily predictions and score all stocks."""
    logger.info("=" * 60)
    logger.info("STEP 3: Running daily predictions")
    logger.info("=" * 60)

    if feature_data is None:
        collector  = DataCollector(DATABASE_PATH)
        stock_data = collector.load_all_stocks(NSE_STOCKS)
        benchmark  = collector.load_benchmark()
        if not stock_data or benchmark.empty:
            logger.error("No data. Run --download first.")
            return None, None
        feature_data = build_feature_dataset(stock_data, benchmark, pred_window)

    engine = PredictionEngine(DATABASE_PATH, MODEL_DIR)
    engine.load_model()
    scores_df = engine.run_daily(feature_data)

    if scores_df.empty:
        logger.warning("No predictions generated")
        return feature_data, None

    return feature_data, scores_df


def pipeline_trade(feature_data=None, top_picks=None, pred_window: int = DEFAULT_WINDOW):
    """Execute paper trades based on predictions."""
    logger.info("=" * 60)
    logger.info("STEP 4: Executing paper trades")
    logger.info("=" * 60)

    if feature_data is None or top_picks is None:
        feature_data, top_picks = pipeline_predict(pred_window=pred_window)
        if top_picks is None or top_picks.empty:
            logger.error("No predictions to trade on")
            return

    agent = PaperTradingAgent(DATABASE_PATH)
    agent.rebalance(top_picks, feature_data)

    summary = agent.get_performance_summary()
    if summary:
        logger.info(f"\nPortfolio Summary:")
        for k, v in summary.items():
            logger.info(f"  {k:25s}: {v}")


def pipeline_retrain(force: bool = False):
    """Incremental retrain — only runs if schedule dictates."""
    logger.info("=" * 60)
    logger.info("RETRAINING SCHEDULER")
    logger.info("=" * 60)
    scheduler = RetrainingScheduler(DATABASE_PATH)
    scheduler.run(force=force)


def pipeline_all():
    """Full daily pipeline: predict → trade → check retrain schedule."""
    logger.info("=" * 60)
    logger.info("FULL DAILY PIPELINE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # Check if retrain is due
    scheduler = RetrainingScheduler(DATABASE_PATH)
    if scheduler.is_retrain_due():
        logger.info("Retrain is due — running retrain pipeline first …")
        scheduler.run()

    # Predict + trade
    feature_data, top_picks = pipeline_predict()
    if top_picks is not None and not top_picks.empty:
        pipeline_trade(feature_data, top_picks)


def pipeline_status():
    """Print a summary of the system state."""
    import sqlite3
    import pandas as pd

    print("\n" + "=" * 60)
    print(" ML TRADING SYSTEM STATUS")
    print("=" * 60)

    # Model
    payload = ModelTrainer.load_best_model(MODEL_DIR)
    if payload:
        print(f"\n✅ Best Model   : {payload['model_name']}")
        print(f"   Trained On  : {payload.get('train_date', 'N/A')[:19]}")
        print(f"   ROC-AUC     : {payload['metrics']['roc_auc']:.4f}")
        print(f"   # Features  : {len(payload.get('feature_cols', []))}")
    else:
        print("\n❌ No trained model found. Run: python main.py --train")

    # Database tables
    if os.path.exists(DATABASE_PATH):
        conn = sqlite3.connect(DATABASE_PATH)
        for table in ["ohlcv", "daily_predictions", "trades", "portfolio_snapshots"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"\n   {table:30s}: {count:,} rows")
            except Exception:
                print(f"\n   {table:30s}: table not found")
        conn.close()
    else:
        print("\n❌ Database not found. Run: python main.py --download")

    print("\n" + "=" * 60)
    print(" COMMANDS")
    print("=" * 60)
    print("  python main.py --download    Download market data")
    print("  python main.py --train       Train models")
    print("  python main.py --predict     Daily predictions")
    print("  python main.py --trade       Paper trade")
    print("  python main.py --all         Full daily pipeline")
    print("  streamlit run dashboard/app.py  Launch dashboard")
    print("=" * 60 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ML Trading System — NSE/NIFTY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download", action="store_true", help="Download historical data")
    parser.add_argument("--train",    action="store_true", help="Train ML models")
    parser.add_argument("--predict",  action="store_true", help="Run daily predictions")
    parser.add_argument("--trade",    action="store_true", help="Execute paper trades")
    parser.add_argument("--retrain",  action="store_true", help="Retrain (respects schedule)")
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain now")
    parser.add_argument("--all",      action="store_true", help="Full daily pipeline")
    parser.add_argument("--status",   action="store_true", help="System status")
    parser.add_argument("--window",   type=int, default=DEFAULT_WINDOW,
                        help=f"Prediction window in days (default: {DEFAULT_WINDOW})")

    args = parser.parse_args()

    # Default: show status if no args
    if not any(vars(args).values()):
        pipeline_status()
        return

    if args.status:
        pipeline_status()

    if args.download:
        pipeline_download()

    if args.train:
        pipeline_train(args.window)

    if args.predict:
        pipeline_predict(pred_window=args.window)

    if args.trade:
        pipeline_trade(pred_window=args.window)

    if args.retrain:
        pipeline_retrain(force=False)

    if args.force_retrain:
        pipeline_retrain(force=True)

    if args.all:
        pipeline_all()


if __name__ == "__main__":
    main()
