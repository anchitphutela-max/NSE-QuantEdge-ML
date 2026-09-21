# trading/scheduler.py
"""
Automatic Retraining Scheduler.
Runs weekly or monthly: downloads fresh data → rebuilds features
→ retrains models → deploys best model.
"""

import os
import sys
import logging
import warnings
import sqlite3
from datetime import datetime, timedelta
from typing import Dict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    RETRAIN_FREQUENCY, DATABASE_PATH, NSE_STOCKS,
    BENCHMARK_TICKER, DEFAULT_WINDOW
)
from data.collector import DataCollector
from features.engineer import build_feature_dataset, FeatureEngineer
from training.trainer import ModelTrainer


class RetrainingScheduler:
    """
    Manages the full retrain lifecycle.
    Can be triggered manually or via a cron / APScheduler job.
    """

    def __init__(
        self,
        db_path:     str = DATABASE_PATH,
        pred_window: int = DEFAULT_WINDOW,
    ):
        self.db_path     = db_path
        self.pred_window = pred_window
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrain_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date    TEXT,
                status      TEXT,
                best_model  TEXT,
                roc_auc     REAL,
                notes       TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ── Check if retrain is due ───────────────────────────────────────
    def is_retrain_due(self) -> bool:
        conn = sqlite3.connect(self.db_path)
        row  = conn.execute(
            "SELECT MAX(run_date) FROM retrain_log WHERE status='success'"
        ).fetchone()
        conn.close()

        if not row or not row[0]:
            return True  # never trained

        last_run = datetime.fromisoformat(row[0])
        if RETRAIN_FREQUENCY == "weekly":
            return datetime.now() - last_run >= timedelta(weeks=1)
        elif RETRAIN_FREQUENCY == "monthly":
            return datetime.now() - last_run >= timedelta(days=30)
        return False

    # ── Full retrain pipeline ─────────────────────────────────────────
    def run(self, force: bool = False) -> bool:
        """
        Execute the full retrain pipeline.
        Returns True if successful.
        """
        if not force and not self.is_retrain_due():
            logger.info("Retrain not due yet — skipping")
            return False

        logger.info("=" * 60)
        logger.info("RETRAINING PIPELINE STARTED")
        logger.info("=" * 60)
        start_time = datetime.now()

        try:
            # Step 1 — Update data
            logger.info("[1/5] Updating market data …")
            collector = DataCollector(self.db_path)
            collector.update_data(NSE_STOCKS + [BENCHMARK_TICKER])

            # Step 2 — Load data
            logger.info("[2/5] Loading data from database …")
            stock_data = collector.load_all_stocks(NSE_STOCKS)
            benchmark  = collector.load_benchmark()

            if not stock_data or benchmark.empty:
                raise ValueError("Insufficient data for training")

            # Step 3 — Feature engineering
            logger.info("[3/5] Building features …")
            feature_data = build_feature_dataset(stock_data, benchmark, self.pred_window)

            if not feature_data:
                raise ValueError("Feature engineering produced no data")

            # Determine feature columns from first ticker
            first_df = next(iter(feature_data.values()))
            engineer = FeatureEngineer(self.pred_window)
            feat_cols = engineer.get_feature_columns(first_df)

            # Step 4 — Train models
            logger.info("[4/5] Training models …")
            trainer = ModelTrainer(pred_window=self.pred_window)
            train_result = trainer.train_all(feature_data, feat_cols)

            if not train_result or not train_result.get("best"):
                raise ValueError("Training failed — no best model produced")

            best = train_result["best"]

            # Step 5 — Log
            logger.info("[5/5] Logging results …")
            elapsed = (datetime.now() - start_time).seconds
            self._log_run(
                status="success",
                best_model=best["model_name"],
                roc_auc=best["metrics"]["roc_auc"],
                notes=f"Elapsed {elapsed}s | "
                      f"Acc={best['metrics']['accuracy']:.3f} | "
                      f"F1={best['metrics']['f1']:.3f}",
            )

            logger.info(f"RETRAIN COMPLETE in {elapsed}s — "
                        f"Best: {best['model_name']} "
                        f"AUC={best['metrics']['roc_auc']:.4f}")
            return True

        except Exception as exc:
            logger.error(f"Retrain FAILED: {exc}")
            self._log_run(status="failed", notes=str(exc))
            return False

    def _log_run(
        self,
        status:     str,
        best_model: str = "",
        roc_auc:    float = 0.0,
        notes:      str = "",
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO retrain_log (run_date, status, best_model, roc_auc, notes)
            VALUES (?,?,?,?,?)
        """, (datetime.now().isoformat(), status, best_model, roc_auc, notes))
        conn.commit()
        conn.close()

    def get_retrain_history(self):
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query(
            "SELECT * FROM retrain_log ORDER BY run_date DESC", conn
        )
        conn.close()
        return df
