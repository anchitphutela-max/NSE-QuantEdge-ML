# training/explainer.py
"""
SHAP-based explainability for the trained ML models.
Generates global and local feature importance.
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_DIR


class ModelExplainer:
    """
    Wraps SHAP to provide:
    - Global feature importance (summary)
    - Local explanation for a single prediction
    - Top-N feature names with direction
    """

    def __init__(self, model_payload: Dict):
        self.payload      = model_payload
        self.model        = model_payload["model"]
        self.model_name   = model_payload["model_name"]
        self.feature_cols = model_payload["feature_cols"]
        self.explainer    = None
        self.shap_values  = None

    # ── Build explainer ───────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, max_samples: int = 500):
        """Compute SHAP values on a sample of X."""
        X_sample = X[self.feature_cols].dropna()
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)

        logger.info(f"Computing SHAP values for {self.model_name} on {len(X_sample)} samples …")

        try:
            if self.model_name in ("xgboost", "lightgbm", "gradient_boosting", "random_forest"):
                self.explainer   = shap.TreeExplainer(self.model)
                shap_vals        = self.explainer.shap_values(X_sample)
                # For binary classifiers, shap_values can be [neg_class, pos_class]
                if isinstance(shap_vals, list):
                    self.shap_values = shap_vals[1]
                else:
                    self.shap_values = shap_vals
            else:
                self.explainer   = shap.Explainer(self.model, X_sample)
                self.shap_values = self.explainer(X_sample).values

            self.X_sample = X_sample
            logger.info("SHAP values computed successfully")

        except Exception as exc:
            logger.error(f"SHAP computation failed: {exc}")
            self.shap_values = None

    # ── Global importance ─────────────────────────────────────────────
    def global_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return DataFrame of mean |SHAP| per feature, ranked."""
        if self.shap_values is None:
            return pd.DataFrame()

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        return df

    # ── Local explanation ─────────────────────────────────────────────
    def explain_prediction(self, row: pd.Series) -> Dict[str, Any]:
        """
        Explain a single prediction.
        Returns dict with top drivers (feature, value, shap_impact).
        """
        if self.explainer is None:
            return {}

        try:
            x = row[self.feature_cols].values.reshape(1, -1)
            sv = self.explainer.shap_values(x)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv.flatten()

            drivers = pd.DataFrame({
                "feature": self.feature_cols,
                "value":   row[self.feature_cols].values,
                "impact":  sv,
            }).sort_values("impact", key=abs, ascending=False).head(5)

            return {
                "top_drivers": drivers.to_dict("records"),
                "base_value":  float(self.explainer.expected_value[1])
                               if isinstance(self.explainer.expected_value, (list, np.ndarray))
                               else float(self.explainer.expected_value),
            }
        except Exception as exc:
            logger.error(f"Local explanation failed: {exc}")
            return {}

    # ── Plot helpers ──────────────────────────────────────────────────
    def plot_summary(self, output_path: str = "shap_summary.png"):
        if self.shap_values is None:
            return
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values, self.X_sample,
            feature_names=self.feature_cols,
            show=False, max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        logger.info(f"SHAP summary plot saved: {output_path}")

    def plot_bar(self, output_path: str = "shap_bar.png"):
        if self.shap_values is None:
            return
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values, self.X_sample,
            feature_names=self.feature_cols,
            plot_type="bar", show=False, max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        logger.info(f"SHAP bar plot saved: {output_path}")
