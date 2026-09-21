# training/trainer.py
"""
Model Training Pipeline.
Trains Random Forest, Gradient Boosting, XGBoost, LightGBM.
Uses walk-forward validation (no look-ahead bias).
Optimises hyperparameters with Optuna.
Saves best model with Joblib.
"""

import os
import sys
import json
import logging
import warnings
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODEL_DIR, DATABASE_PATH, BEST_METRIC, MODELS_TO_TRAIN,
    N_OPTUNA_TRIALS, OPTUNA_TIMEOUT, TRAIN_YEARS, TEST_YEARS,
    N_SPLITS, DEFAULT_WINDOW
)


class ModelTrainer:
    """
    Full training pipeline:
    - Walk-forward cross-validation
    - Optuna hyperparameter search per model
    - Automatic best-model selection
    - Joblib persistence
    """

    def __init__(
        self,
        model_dir:  str = MODEL_DIR,
        db_path:    str = DATABASE_PATH,
        pred_window:int = DEFAULT_WINDOW,
    ):
        self.model_dir   = model_dir
        self.db_path     = db_path
        self.pred_window = pred_window
        self.label_col   = f"target_{pred_window}d"

        os.makedirs(model_dir, exist_ok=True)
        self._init_db()

    # ── Database ──────────────────────────────────────────────────────
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name      TEXT,
                version         TEXT,
                train_date      TEXT,
                accuracy        REAL,
                precision_score REAL,
                recall          REAL,
                f1              REAL,
                roc_auc         REAL,
                log_loss        REAL,
                n_features      INTEGER,
                params          TEXT,
                is_best         INTEGER DEFAULT 0,
                model_path      TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ── Data Preparation ──────────────────────────────────────────────
    def prepare_data(
        self,
        feature_data: Dict[str, pd.DataFrame],
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Combine all stock feature DataFrames into one X, y pair."""
        dfs = []
        for ticker, df in feature_data.items():
            sub = df[feature_cols + [self.label_col]].copy()
            sub["ticker"] = ticker
            dfs.append(sub)

        combined = pd.concat(dfs).sort_index()
        combined.dropna(subset=[self.label_col], inplace=True)

        X = combined[feature_cols]
        y = combined[self.label_col]
        return X, y

    # ── Walk-Forward Folds ────────────────────────────────────────────
    def walk_forward_splits(
        self, index: pd.DatetimeIndex
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_idx, test_idx) pairs via expanding-window walk-forward.
        """
        dates = pd.Series(index).dt.year.values
        years = sorted(np.unique(dates))

        splits = []
        for i in range(N_SPLITS):
            # Sliding test window from end
            test_end_year   = years[-1] - i * TEST_YEARS
            test_start_year = test_end_year - TEST_YEARS + 1
            train_end_year  = test_start_year - 1
            train_start_year= train_end_year  - TRAIN_YEARS + 1

            train_mask = (dates >= train_start_year) & (dates <= train_end_year)
            test_mask  = (dates >= test_start_year)  & (dates <= test_end_year)

            if train_mask.sum() < 100 or test_mask.sum() < 20:
                continue

            splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

        splits.reverse()  # chronological order
        return splits

    # ── Optuna Objectives ─────────────────────────────────────────────
    def _rf_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "n_jobs": -1, "random_state": 42,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    def _gb_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 300),
            "max_depth":      trial.suggest_int("max_depth", 2, 8),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
        }
        model = GradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    def _xgb_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
            "max_depth":     trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42, "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    def _lgb_objective(self, trial, X_tr, y_tr, X_val, y_val) -> float:
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 500),
            "max_depth":      trial.suggest_int("max_depth", 2, 10),
            "learning_rate":  trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves":     trial.suggest_int("num_leaves", 10, 200),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    # ── Single model training ─────────────────────────────────────────
    def _train_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple],
    ) -> Dict[str, Any]:
        """Train one model type using walk-forward + Optuna."""

        logger.info(f"Training {model_name} …")
        objective_map = {
            "random_forest":    self._rf_objective,
            "gradient_boosting":self._gb_objective,
            "xgboost":          self._xgb_objective,
            "lightgbm":         self._lgb_objective,
        }
        objective_fn = objective_map[model_name]

        # Imputer to handle NaNs
        imputer = SimpleImputer(strategy="median")

        fold_metrics = []
        best_params  = None
        best_val_auc = -np.inf

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr_raw = X.iloc[train_idx]
            y_tr     = y.iloc[train_idx]
            X_val_raw= X.iloc[test_idx]
            y_val    = y.iloc[test_idx]

            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                continue

            X_tr  = pd.DataFrame(imputer.fit_transform(X_tr_raw),  columns=X.columns)
            X_val = pd.DataFrame(imputer.transform(X_val_raw),      columns=X.columns)

            # Optuna search on first fold only (efficiency)
            if fold_idx == 0:
                study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
                study.optimize(
                    lambda trial: objective_fn(trial, X_tr, y_tr, X_val, y_val),
                    n_trials=N_OPTUNA_TRIALS,
                    timeout=OPTUNA_TIMEOUT,
                )
                best_params = study.best_params
                logger.info(f"{model_name} fold {fold_idx}: best AUC={study.best_value:.4f}")

            # Train final model on this fold with best params
            model = self._build_model(model_name, best_params or {})
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            preds = (proba >= 0.5).astype(int)

            fold_metrics.append({
                "fold":      fold_idx,
                "accuracy":  accuracy_score(y_val, preds),
                "precision": precision_score(y_val, preds, zero_division=0),
                "recall":    recall_score(y_val, preds, zero_division=0),
                "f1":        f1_score(y_val, preds, zero_division=0),
                "roc_auc":   roc_auc_score(y_val, proba),
                "log_loss":  log_loss(y_val, proba),
            })

            if fold_metrics[-1]["roc_auc"] > best_val_auc:
                best_val_auc = fold_metrics[-1]["roc_auc"]

        if not fold_metrics:
            return {}

        # Average metrics across folds
        avg = {k: np.mean([m[k] for m in fold_metrics])
               for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]}

        # Retrain on ALL data with best params
        X_all = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        final_model = self._build_model(model_name, best_params or {})
        final_model.fit(X_all, y)

        return {
            "model":       final_model,
            "imputer":     imputer,
            "model_name":  model_name,
            "params":      best_params,
            "metrics":     avg,
            "fold_metrics":fold_metrics,
            "feature_cols":list(X.columns),
        }

    def _build_model(self, model_name: str, params: Dict) -> Any:
        """Instantiate a model with the given params."""
        if model_name == "random_forest":
            defaults = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
            defaults.update(params)
            return RandomForestClassifier(**defaults)

        elif model_name == "gradient_boosting":
            defaults = {"n_estimators": 100, "random_state": 42}
            defaults.update(params)
            return GradientBoostingClassifier(**defaults)

        elif model_name == "xgboost":
            defaults = {"n_estimators": 100, "use_label_encoder": False,
                        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}
            defaults.update(params)
            return xgb.XGBClassifier(**defaults)

        elif model_name == "lightgbm":
            defaults = {"n_estimators": 100, "random_state": 42, "n_jobs": -1, "verbose": -1}
            defaults.update(params)
            return lgb.LGBMClassifier(**defaults)

        raise ValueError(f"Unknown model: {model_name}")

    # ── Run full pipeline ─────────────────────────────────────────────
    def train_all(
        self,
        feature_data: Dict[str, pd.DataFrame],
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """Train all models, compare, save best."""

        X, y = self.prepare_data(feature_data, feature_cols)
        splits = self.walk_forward_splits(X.index)

        if not splits:
            logger.error("No valid walk-forward splits generated")
            return {}

        logger.info(f"Dataset: {X.shape}, {len(splits)} walk-forward folds")

        results    = {}
        best_model = None
        best_auc   = -np.inf

        for name in MODELS_TO_TRAIN:
            result = self._train_model(name, X, y, splits)
            if not result:
                continue
            results[name] = result
            auc = result["metrics"]["roc_auc"]
            logger.info(f"{name}: ROC-AUC={auc:.4f}")

            if auc > best_auc:
                best_auc   = auc
                best_model = result

        if best_model:
            self._save_model(best_model, is_best=True)
            logger.info(f"Best model: {best_model['model_name']} (AUC={best_auc:.4f})")

        # Save all models to DB
        for name, result in results.items():
            self._register_model(result, is_best=(name == best_model["model_name"]))

        return {"results": results, "best": best_model}

    # ── Persistence ───────────────────────────────────────────────────
    def _save_model(self, result: Dict, is_best: bool = False):
        version    = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = result["model_name"]
        path       = os.path.join(self.model_dir, f"{model_name}_{version}.pkl")

        payload = {
            "model":        result["model"],
            "imputer":      result["imputer"],
            "model_name":   model_name,
            "params":       result["params"],
            "metrics":      result["metrics"],
            "feature_cols": result["feature_cols"],
            "train_date":   datetime.now().isoformat(),
            "pred_window":  self.pred_window,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved: {path}")

        if is_best:
            best_path = os.path.join(self.model_dir, "best_model.pkl")
            joblib.dump(payload, best_path)
            logger.info(f"Best model saved: {best_path}")

    def _register_model(self, result: Dict, is_best: bool = False):
        m   = result["metrics"]
        path= os.path.join(self.model_dir, f"{result['model_name']}.pkl")
        conn= sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO model_registry
                (model_name, version, train_date, accuracy, precision_score,
                 recall, f1, roc_auc, log_loss, n_features, params, is_best, model_path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result["model_name"],
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            datetime.now().isoformat(),
            m["accuracy"], m["precision"], m["recall"],
            m["f1"], m["roc_auc"], m["log_loss"],
            len(result["feature_cols"]),
            json.dumps(result["params"]),
            int(is_best), path,
        ))
        conn.commit()
        conn.close()

    # ── Load ──────────────────────────────────────────────────────────
    @staticmethod
    def load_best_model(model_dir: str = MODEL_DIR) -> Optional[Dict]:
        path = os.path.join(model_dir, "best_model.pkl")
        if not os.path.exists(path):
            logger.warning("No best_model.pkl found")
            return None
        return joblib.load(path)

    def get_model_history(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query(
            "SELECT * FROM model_registry ORDER BY train_date DESC", conn
        )
        conn.close()
        return df
