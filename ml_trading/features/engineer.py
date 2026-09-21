# features/engineer.py
"""
Feature Engineering Pipeline.
Generates all technical, risk, momentum, and market features from OHLCV data.
"""

import os
import sys
import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    SMA_WINDOWS, VOLATILITY_WINDOWS, RSI_PERIOD, MACD_FAST, MACD_SLOW,
    MACD_SIGNAL, MOMENTUM_PERIOD, ROC_PERIOD, VOLUME_MA_WINDOW,
    SHARPE_WINDOW, SORTINO_WINDOW, BETA_WINDOW, DEFAULT_WINDOW,
    PREDICTION_WINDOWS
)


class FeatureEngineer:
    """
    Generates a rich feature matrix for each stock.
    Requires both stock OHLCV and benchmark (NIFTY) price series.
    """

    def __init__(self, prediction_window: int = DEFAULT_WINDOW):
        self.prediction_window = prediction_window

    # ── Master pipeline ───────────────────────────────────────────────
    def generate_features(
        self,
        df:        pd.DataFrame,
        benchmark: pd.DataFrame,
        ticker:    str = "",
    ) -> pd.DataFrame:
        """Run the full feature pipeline and return a feature DataFrame."""

        df = df.copy()
        df = df.sort_index()

        # Require a 'close' column
        close_col = "close" if "close" in df.columns else "adj_close"
        df["close"] = df[close_col]

        logger.info(f"{ticker}: engineering features …")

        df = self._price_features(df)
        df = self._volatility_features(df)
        df = self._momentum_features(df)
        df = self._moving_average_features(df)
        df = self._trend_features(df)
        df = self._volume_features(df)
        df = self._risk_features(df, benchmark)
        df = self._market_features(df, benchmark)
        df = self._create_labels(df, benchmark)

        # Drop rows that are all NaN in feature columns
        df.dropna(subset=["daily_return"], inplace=True)

        logger.info(f"{ticker}: {df.shape[1]} features, {len(df)} rows")
        return df

    # ── 1. Price Features ─────────────────────────────────────────────
    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["daily_return"]   = close.pct_change(1)
        df["weekly_return"]  = close.pct_change(5)
        df["monthly_return"] = close.pct_change(21)
        df["annual_return"]  = close.pct_change(252)

        # Log returns (used for Sharpe/Sortino internally)
        df["log_return"]     = np.log(close / close.shift(1))

        # Price normalised to 52-week range
        rolling_52 = close.rolling(252)
        df["price_52w_high_pct"] = close / rolling_52.max() - 1
        df["price_52w_low_pct"]  = close / rolling_52.min() - 1

        return df

    # ── 2. Volatility Features ────────────────────────────────────────
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df["daily_return"]

        for w in VOLATILITY_WINDOWS:
            df[f"volatility_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

        # Realised vs implied vol ratio proxy
        if "volatility_10d" in df.columns and "volatility_50d" in df.columns:
            df["vol_ratio_10_50"] = df["volatility_10d"] / (df["volatility_50d"] + 1e-9)

        return df

    # ── 3. Momentum Features ──────────────────────────────────────────
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # RSI
        df["rsi"] = self._compute_rsi(close, RSI_PERIOD)

        # MACD
        ema_fast   = close.ewm(span=MACD_FAST,   adjust=False).mean()
        ema_slow   = close.ewm(span=MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line= macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()

        df["macd"]          = macd_line
        df["macd_signal"]   = signal_line
        df["macd_histogram"]= macd_line - signal_line
        df["macd_crossover"]= (macd_line > signal_line).astype(int)

        # Momentum score: normalised rate-of-change
        df["momentum"]      = close.pct_change(MOMENTUM_PERIOD)
        df["roc"]           = (close - close.shift(ROC_PERIOD)) / (close.shift(ROC_PERIOD) + 1e-9)

        # Stochastic oscillator
        low_min  = df["low"].rolling(14).min()  if "low"  in df.columns else close.rolling(14).min()
        high_max = df["high"].rolling(14).max() if "high" in df.columns else close.rolling(14).max()
        df["stoch_k"] = 100 * (close - low_min) / (high_max - low_min + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    # ── 4. Moving Average Features ────────────────────────────────────
    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        for w in SMA_WINDOWS:
            sma = close.rolling(w).mean()
            ema = close.ewm(span=w, adjust=False).mean()
            df[f"sma_{w}"]       = sma
            df[f"ema_{w}"]       = ema
            df[f"price_sma_{w}"] = close / (sma + 1e-9) - 1   # distance from SMA

        return df

    # ── 5. Trend Features ─────────────────────────────────────────────
    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sma_50" not in df.columns or "sma_200" not in df.columns:
            return df

        sma50  = df["sma_50"]
        sma200 = df["sma_200"]

        df["golden_cross"] = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).astype(int)
        df["death_cross"]  = ((sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))).astype(int)
        df["above_sma200"] = (df["close"] > sma200).astype(int)
        df["trend_50_200"] = sma50 / (sma200 + 1e-9) - 1

        return df

    # ── 6. Volume Features ────────────────────────────────────────────
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            return df

        vol   = df["volume"].replace(0, np.nan)
        vol_ma= vol.rolling(VOLUME_MA_WINDOW).mean()

        df["relative_volume"] = vol / (vol_ma + 1e-9)
        df["volume_change"]   = vol.pct_change(1)
        df["volume_trend"]    = (vol > vol_ma).astype(int)

        # Volume-price trend
        df["vpt"] = (df["daily_return"] * vol).cumsum()

        return df

    # ── 7. Risk Features ──────────────────────────────────────────────
    def _risk_features(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        ret = df["daily_return"].fillna(0)

        # Rolling Sharpe Ratio (annualised, risk-free = 0 for simplicity)
        df["sharpe_ratio"] = (
            ret.rolling(SHARPE_WINDOW).mean() /
            (ret.rolling(SHARPE_WINDOW).std() + 1e-9)
        ) * np.sqrt(252)

        # Rolling Sortino Ratio
        downside = ret.copy()
        downside[downside > 0] = 0
        df["sortino_ratio"] = (
            ret.rolling(SORTINO_WINDOW).mean() /
            (downside.rolling(SORTINO_WINDOW).std() + 1e-9)
        ) * np.sqrt(252)

        # Max Drawdown (rolling 252-day)
        roll_max = df["close"].rolling(252, min_periods=1).max()
        df["drawdown"]     = df["close"] / roll_max - 1
        df["max_drawdown"] = df["drawdown"].rolling(252).min()

        # Beta & Alpha vs NIFTY
        bench_ret = self._align_returns(df, benchmark)
        if bench_ret is not None:
            df["beta"]  = self._rolling_beta( ret, bench_ret, BETA_WINDOW)
            df["alpha"] = self._rolling_alpha(ret, bench_ret, BETA_WINDOW, df["beta"])

        return df

    # ── 8. Market Features ────────────────────────────────────────────
    def _market_features(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        bench_ret = self._align_returns(df, benchmark)
        if bench_ret is None:
            return df

        bench_close_col = "close" if "close" in benchmark.columns else "adj_close"
        bench_close = benchmark[bench_close_col].reindex(df.index, method="ffill")

        df["nifty_return"]     = bench_ret
        df["nifty_volatility"] = bench_ret.rolling(20).std() * np.sqrt(252)
        df["market_trend"]     = (bench_close > bench_close.rolling(50).mean()).astype(int)

        # Relative strength vs benchmark
        df["relative_return"]  = df["daily_return"] - bench_ret
        df["relative_strength"]= df["monthly_return"] - bench_ret.rolling(21).sum()

        return df

    # ── 9. Labels ─────────────────────────────────────────────────────
    def _create_labels(self, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        bench_ret = self._align_returns(df, benchmark)

        for window in PREDICTION_WINDOWS:
            fwd_stock = df["close"].pct_change(window).shift(-window)
            label_col = f"target_{window}d"

            if bench_ret is not None:
                fwd_bench = bench_ret.rolling(window).sum().shift(-window)
                df[label_col] = (fwd_stock > fwd_bench).astype(float)
            else:
                df[label_col] = (fwd_stock > 0).astype(float)

            df[f"fwd_return_{window}d"] = fwd_stock

        return df

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _align_returns(df: pd.DataFrame, benchmark: pd.DataFrame) -> Optional[pd.Series]:
        if benchmark is None or benchmark.empty:
            return None
        close_col = "close" if "close" in benchmark.columns else "adj_close"
        bench_close = benchmark[close_col].reindex(df.index, method="ffill")
        return bench_close.pct_change(1)

    @staticmethod
    def _rolling_beta(stock_ret: pd.Series, bench_ret: pd.Series, window: int) -> pd.Series:
        def beta_func(x):
            if len(x) < 30:
                return np.nan
            s = x[:, 0]
            b = x[:, 1]
            mask = ~(np.isnan(s) | np.isnan(b))
            if mask.sum() < 30:
                return np.nan
            cov = np.cov(s[mask], b[mask])
            return cov[0, 1] / (cov[1, 1] + 1e-9)

        combined = pd.concat([stock_ret, bench_ret], axis=1).values
        results = []
        for i in range(len(combined)):
            start = max(0, i - window + 1)
            results.append(beta_func(combined[start:i + 1]))
        return pd.Series(results, index=stock_ret.index)

    @staticmethod
    def _rolling_alpha(
        stock_ret: pd.Series,
        bench_ret: pd.Series,
        window: int,
        beta: pd.Series,
        risk_free: float = 0.0,
    ) -> pd.Series:
        stock_ann = stock_ret.rolling(window).mean() * 252
        bench_ann = bench_ret.rolling(window).mean() * 252
        return stock_ann - (risk_free + beta * (bench_ann - risk_free))

    # ── Feature columns (excluding labels) ───────────────────────────
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        exclude = (
            ["open", "high", "low", "close", "volume", "adj_close", "ticker",
             "log_return", "id", "drawdown"]
            + [c for c in df.columns if c.startswith("target_")]
            + [c for c in df.columns if c.startswith("fwd_return_")]
            + [c for c in df.columns if c.startswith("sma_")]
            + [c for c in df.columns if c.startswith("ema_")]
            + ["vpt"]
        )
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


# ── Batch pipeline ────────────────────────────────────────────────────────
def build_feature_dataset(
    stock_data: Dict[str, pd.DataFrame],
    benchmark:  pd.DataFrame,
    prediction_window: int = DEFAULT_WINDOW,
) -> Dict[str, pd.DataFrame]:
    """Build feature DataFrames for all tickers."""
    engineer  = FeatureEngineer(prediction_window)
    feat_data: Dict[str, pd.DataFrame] = {}

    for ticker, df in stock_data.items():
        try:
            feat_df = engineer.generate_features(df, benchmark, ticker)
            feat_data[ticker] = feat_df
        except Exception as exc:
            logger.error(f"{ticker}: feature engineering failed — {exc}")

    logger.info(f"Feature engineering complete: {len(feat_data)} tickers")
    return feat_data
