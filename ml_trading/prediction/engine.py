# prediction/engine.py
"""
Prediction Engine.
Every trading day: loads the best model, generates features,
outputs probability-of-outperformance for each stock.
Stores results in the database.
"""

import os
import sys
import sqlite3
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODEL_DIR, DATABASE_PATH, WEIGHT_SHARPE, WEIGHT_MOMENTUM,
    WEIGHT_ML, TOP_N_STOCKS, DEFAULT_WINDOW
)
from training.trainer import ModelTrainer


class PredictionEngine:
    """
    Generates ML predictions and computes the hybrid decision score
    for every stock in the universe.
    """

    def __init__(self, db_path: str = DATABASE_PATH, model_dir: str = MODEL_DIR):
        self.db_path   = db_path
        self.model_dir = model_dir
        self.model_payload = None
        self._init_db()

    # ── Database ──────────────────────────────────────────────────────
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT,
                ticker          TEXT,
                ml_probability  REAL,
                sharpe_score    REAL,
                momentum_score  REAL,
                final_score     REAL,
                selected        INTEGER DEFAULT 0,
                explanation     TEXT,
                UNIQUE(date, ticker)
            )
        """)
        conn.commit()
        conn.close()

    # ── Load model ────────────────────────────────────────────────────
    def load_model(self):
        self.model_payload = ModelTrainer.load_best_model(self.model_dir)
        if self.model_payload:
            logger.info(f"Loaded model: {self.model_payload['model_name']} "
                        f"(trained {self.model_payload.get('train_date', 'unknown')})")
        else:
            logger.warning("No model loaded — predictions will use fallback scoring")

    # ── ML Probability ────────────────────────────────────────────────
    def predict_probability(
        self,
        feature_row: pd.Series,
    ) -> float:
        """Return P(outperform) for a single row of features."""
        if self.model_payload is None:
            return 0.5  # neutral fallback

        model        = self.model_payload["model"]
        imputer      = self.model_payload["imputer"]
        feature_cols = self.model_payload["feature_cols"]

        try:
            x = feature_row[feature_cols].values.reshape(1, -1)
            x = imputer.transform(x)
            prob = model.predict_proba(x)[0][1]
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as exc:
            logger.error(f"Prediction error: {exc}")
            return 0.5

    # ── Score components ──────────────────────────────────────────────
    @staticmethod
    def _normalise_series(s: pd.Series) -> pd.Series:
        """Min-max normalise to [0, 1]."""
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    def compute_scores(
        self,
        feature_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        For each stock, take the LATEST row of features and compute:
          - ML probability
          - Sharpe score (normalised)
          - Momentum score (normalised)
          - Final hybrid score
        """
        records = []
        today   = datetime.today().strftime("%Y-%m-%d")

        for ticker, df in feature_data.items():
            df_clean = df.dropna(subset=["daily_return"])
            if df_clean.empty:
                continue

            latest = df_clean.iloc[-1]

            ml_prob   = self.predict_probability(latest)
            sharpe    = float(latest.get("sharpe_ratio",  0.0) or 0.0)
            momentum  = float(latest.get("momentum",      0.0) or 0.0)

            records.append({
                "date":           today,
                "ticker":         ticker,
                "ml_probability": ml_prob,
                "sharpe_raw":     sharpe,
                "momentum_raw":   momentum,
            })

        if not records:
            return pd.DataFrame()

        scores_df = pd.DataFrame(records)

        # Normalise Sharpe and Momentum to [0,1]
        scores_df["sharpe_score"]   = self._normalise_series(scores_df["sharpe_raw"])
        scores_df["momentum_score"] = self._normalise_series(scores_df["momentum_raw"])

        # Hybrid final score
        scores_df["final_score"] = (
            WEIGHT_SHARPE   * scores_df["sharpe_score"]
          + WEIGHT_MOMENTUM * scores_df["momentum_score"]
          + WEIGHT_ML       * scores_df["ml_probability"]
        )

        scores_df = scores_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        scores_df["selected"] = 0
        scores_df.loc[:TOP_N_STOCKS - 1, "selected"] = 1

        return scores_df

    # ── Explanation text ──────────────────────────────────────────────
    @staticmethod
    def generate_explanation(row: pd.Series) -> str:
        return (
            f"Selected because: "
            f"Sharpe Score = {row['sharpe_score']:.2f} | "
            f"Momentum Score = {row['momentum_score']:.2f} | "
            f"ML Probability = {row['ml_probability'] * 100:.1f}% | "
            f"Final Score = {row['final_score'] * 100:.1f}"
        )

    # ── Store predictions ─────────────────────────────────────────────
    def store_predictions(self, scores_df: pd.DataFrame):
        conn = sqlite3.connect(self.db_path)
        for _, row in scores_df.iterrows():
            explanation = self.generate_explanation(row) if row["selected"] else ""
            conn.execute("""
                INSERT OR REPLACE INTO daily_predictions
                    (date, ticker, ml_probability, sharpe_score,
                     momentum_score, final_score, selected, explanation)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                row["date"], row["ticker"],
                row["ml_probability"], row["sharpe_score"],
                row["momentum_score"], row["final_score"],
                int(row["selected"]), explanation,
            ))
        conn.commit()
        conn.close()
        logger.info(f"Stored predictions for {len(scores_df)} stocks")

    # ── Top picks ─────────────────────────────────────────────────────
    def get_top_picks(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        return scores_df[scores_df["selected"] == 1].copy()

    def get_latest_predictions(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query("""
            SELECT * FROM daily_predictions
            WHERE date = (SELECT MAX(date) FROM daily_predictions)
            ORDER BY final_score DESC
        """, conn)
        conn.close()
        return df

    # ── Full daily run ─────────────────────────────────────────────────
    def run_daily(self, feature_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Load model → score all stocks → store → return top picks."""
        if self.model_payload is None:
            self.load_model()

        scores_df = self.compute_scores(feature_data)
        if scores_df.empty:
            logger.warning("No scores computed — check feature data")
            return pd.DataFrame()

        self.store_predictions(scores_df)
        top = self.get_top_picks(scores_df)

        logger.info("=== TODAY'S TOP PICKS ===")
        for _, row in top.iterrows():
            logger.info(
                f"  {row['ticker']:20s}  ML={row['ml_probability']*100:.1f}%  "
                f"Sharpe={row['sharpe_score']:.2f}  "
                f"Mom={row['momentum_score']:.2f}  "
                f"Final={row['final_score']*100:.1f}"
            )

        return top
