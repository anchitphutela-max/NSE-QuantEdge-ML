# data/collector.py
"""
Historical data collection from Yahoo Finance for NSE stocks.
Stores OHLCV data in SQLite (or PostgreSQL).
"""

import os
import sqlite3
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    NSE_STOCKS, BENCHMARK_TICKER, DATA_START_DATE, DATA_END_DATE,
    DATA_FREQUENCY, DATABASE_PATH, MIN_HISTORY_DAYS
)


class DataCollector:
    """Downloads and stores historical OHLCV data for NSE stocks."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_database()

    # ── Database Setup ────────────────────────────────────────────────
    def _init_database(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                adj_close   REAL,
                UNIQUE(ticker, date)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS download_log (
                ticker          TEXT PRIMARY KEY,
                last_download   TEXT,
                rows_stored     INTEGER,
                status          TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialised at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Download ──────────────────────────────────────────────────────
    def download_stock(
        self,
        ticker: str,
        start: str = DATA_START_DATE,
        end:   str = DATA_END_DATE,
    ) -> Optional[pd.DataFrame]:
        """Download OHLCV data for a single ticker via yfinance."""
        try:
            logger.info(f"Downloading {ticker} …")
            df = yf.download(ticker, start=start, end=end,
                             interval=DATA_FREQUENCY, progress=False, auto_adjust=True)

            if df.empty or len(df) < MIN_HISTORY_DAYS:
                logger.warning(f"{ticker}: insufficient data ({len(df)} rows)")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            df.index = pd.to_datetime(df.index)
            df["ticker"] = ticker

            # Rename 'close' → adj_close to keep both
            if "close" in df.columns:
                df["adj_close"] = df["close"]

            logger.info(f"{ticker}: {len(df)} rows downloaded")
            return df

        except Exception as exc:
            logger.error(f"{ticker}: download failed — {exc}")
            return None

    def download_all(
        self,
        tickers: List[str] = None,
        start:   str = DATA_START_DATE,
        end:     str = DATA_END_DATE,
    ) -> Dict[str, pd.DataFrame]:
        """Download all configured tickers including the benchmark."""
        if tickers is None:
            tickers = NSE_STOCKS + [BENCHMARK_TICKER]

        results: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.download_stock(ticker, start, end)
            if df is not None:
                self._store_data(ticker, df)
                results[ticker] = df

        logger.info(f"Downloaded {len(results)}/{len(tickers)} tickers successfully")
        return results

    # ── Storage ───────────────────────────────────────────────────────
    def _store_data(self, ticker: str, df: pd.DataFrame):
        """Insert OHLCV rows (ignore duplicates)."""
        conn = self._get_connection()
        rows_inserted = 0

        for date, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ohlcv
                        (ticker, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    str(date.date()),
                    float(row.get("open",  np.nan)),
                    float(row.get("high",  np.nan)),
                    float(row.get("low",   np.nan)),
                    float(row.get("close", np.nan)),
                    float(row.get("volume",np.nan)),
                    float(row.get("adj_close", row.get("close", np.nan))),
                ))
                rows_inserted += 1
            except Exception as exc:
                logger.debug(f"Row insert error for {ticker} on {date}: {exc}")

        conn.execute("""
            INSERT OR REPLACE INTO download_log (ticker, last_download, rows_stored, status)
            VALUES (?, ?, ?, ?)
        """, (ticker, datetime.now().isoformat(), rows_inserted, "success"))

        conn.commit()
        conn.close()
        logger.info(f"{ticker}: {rows_inserted} rows stored")

    # ── Load ──────────────────────────────────────────────────────────
    def load_stock(self, ticker: str) -> pd.DataFrame:
        """Load stored OHLCV data for a ticker as a DataFrame."""
        conn = self._get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date",
            conn, params=(ticker,)
        )
        conn.close()

        if df.empty:
            logger.warning(f"{ticker}: no data in database")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def load_all_stocks(self, tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load all tickers from the database."""
        if tickers is None:
            tickers = NSE_STOCKS

        data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.load_stock(ticker)
            if not df.empty:
                data[ticker] = df

        logger.info(f"Loaded {len(data)} tickers from database")
        return data

    def load_benchmark(self) -> pd.DataFrame:
        """Load NIFTY 50 benchmark data."""
        return self.load_stock(BENCHMARK_TICKER)

    # ── Update (incremental) ─────────────────────────────────────────
    def update_data(self, tickers: List[str] = None):
        """Download only missing recent data for each ticker."""
        if tickers is None:
            tickers = NSE_STOCKS + [BENCHMARK_TICKER]

        today = datetime.today().strftime("%Y-%m-%d")
        conn  = self._get_connection()

        for ticker in tickers:
            row = conn.execute(
                "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
            ).fetchone()

            last_date = row[0] if row and row[0] else DATA_START_DATE
            start     = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

            if start >= today:
                logger.info(f"{ticker}: already up to date")
                continue

            df = self.download_stock(ticker, start=start, end=today)
            if df is not None:
                self._store_data(ticker, df)

        conn.close()
        logger.info("Incremental update complete")

    # ── Status ────────────────────────────────────────────────────────
    def get_download_status(self) -> pd.DataFrame:
        conn = self._get_connection()
        df   = pd.read_sql_query("SELECT * FROM download_log", conn)
        conn.close()
        return df


# ── Standalone run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    collector = DataCollector()
    collector.download_all()
    print(collector.get_download_status())
