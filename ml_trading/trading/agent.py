# trading/agent.py
"""
Paper Trading Agent.
Executes simulated trades based on hybrid decision engine scores.
Tracks portfolio, P&L, drawdown, and logs all decisions.
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
    DATABASE_PATH, INITIAL_CAPITAL, POSITION_SIZE_PCT,
    TRANSACTION_COST_PCT, SLIPPAGE_PCT, MAX_DRAWDOWN_LIMIT, TOP_N_STOCKS
)


class PaperTradingAgent:
    """
    Simulates real trading with:
    - Equal-weight position sizing
    - Transaction costs + slippage
    - Max-drawdown circuit breaker
    - Full trade and portfolio logging
    """

    def __init__(self, db_path: str = DATABASE_PATH, initial_capital: float = INITIAL_CAPITAL):
        self.db_path         = db_path
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.portfolio: Dict[str, Dict] = {}   # ticker → {shares, entry_price, entry_date}
        self.peak_value      = initial_capital
        self.trading_enabled = True
        self._init_db()
        self._load_state()

    # ── Database ──────────────────────────────────────────────────────
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                ticker      TEXT,
                action      TEXT,
                shares      REAL,
                price       REAL,
                value       REAL,
                cost        REAL,
                pnl         REAL DEFAULT 0,
                reason      TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT UNIQUE,
                cash            REAL,
                equity          REAL,
                total_value     REAL,
                drawdown        REAL,
                n_positions     INTEGER
            );

            CREATE TABLE IF NOT EXISTS trade_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                event       TEXT,
                details     TEXT
            );
        """)
        conn.commit()
        conn.close()

    def _load_state(self):
        """Restore cash and positions from last snapshot if available."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT cash FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row:
            self.cash = row[0]
        conn.close()

    # ── Portfolio value ───────────────────────────────────────────────
    def portfolio_value(self, current_prices: Dict[str, float]) -> float:
        equity = sum(
            pos["shares"] * current_prices.get(ticker, pos["entry_price"])
            for ticker, pos in self.portfolio.items()
        )
        return self.cash + equity

    def _get_current_price(self, ticker: str, feature_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        df = feature_data.get(ticker)
        if df is None or df.empty:
            return None
        close_col = "close" if "close" in df.columns else "adj_close"
        return float(df[close_col].iloc[-1])

    # ── Execution helpers ─────────────────────────────────────────────
    def _apply_costs(self, price: float, action: str) -> float:
        """Adjust price for slippage (buy high, sell low) + transaction cost."""
        slip = SLIPPAGE_PCT * price
        if action == "BUY":
            return price * (1 + TRANSACTION_COST_PCT) + slip
        else:
            return price * (1 - TRANSACTION_COST_PCT) - slip

    # ── Circuit breaker ───────────────────────────────────────────────
    def _check_drawdown(self, current_prices: Dict[str, float]) -> bool:
        total = self.portfolio_value(current_prices)
        self.peak_value = max(self.peak_value, total)
        drawdown = (self.peak_value - total) / (self.peak_value + 1e-9)
        if drawdown > MAX_DRAWDOWN_LIMIT:
            logger.warning(f"Max drawdown breached ({drawdown:.1%}). Trading halted.")
            self.trading_enabled = False
            return False
        return True

    # ── Buy / Sell ─────────────────────────────────────────────────────
    def buy(
        self,
        ticker:  str,
        price:   float,
        reason:  str = "",
        date:    str = None,
    ) -> bool:
        if not self.trading_enabled:
            return False
        if ticker in self.portfolio:
            return False  # already holding

        alloc  = self.initial_capital * POSITION_SIZE_PCT
        cost_p = self._apply_costs(price, "BUY")
        shares = alloc / cost_p
        total_cost = shares * cost_p

        if total_cost > self.cash:
            logger.warning(f"Insufficient cash to buy {ticker}")
            return False

        self.cash -= total_cost
        self.portfolio[ticker] = {
            "shares":      shares,
            "entry_price": price,
            "entry_date":  date or datetime.today().strftime("%Y-%m-%d"),
            "cost_basis":  total_cost,
        }

        self._log_trade(ticker, "BUY", shares, price, total_cost, 0.0, reason, date)
        logger.info(f"BUY  {ticker:12s} | {shares:.2f} shares @ ₹{price:,.2f} | cost ₹{total_cost:,.0f}")
        return True

    def sell(
        self,
        ticker:  str,
        price:   float,
        reason:  str = "",
        date:    str = None,
    ) -> bool:
        if ticker not in self.portfolio:
            return False

        pos        = self.portfolio[ticker]
        shares     = pos["shares"]
        sell_price = self._apply_costs(price, "SELL")
        proceeds   = shares * sell_price
        pnl        = proceeds - pos["cost_basis"]

        self.cash += proceeds
        del self.portfolio[ticker]

        self._log_trade(ticker, "SELL", shares, price, proceeds, pnl, reason, date)
        logger.info(f"SELL {ticker:12s} | {shares:.2f} shares @ ₹{price:,.2f} | "
                    f"P&L ₹{pnl:+,.0f}")
        return True

    # ── Rebalance ─────────────────────────────────────────────────────
    def rebalance(
        self,
        top_picks:    pd.DataFrame,
        feature_data: Dict[str, pd.DataFrame],
        date:         str = None,
    ):
        """
        Sell positions not in top picks.
        Buy new top picks.
        """
        if not self.trading_enabled:
            logger.warning("Trading halted — drawdown limit reached")
            return

        today = date or datetime.today().strftime("%Y-%m-%d")
        current_prices = {
            t: self._get_current_price(t, feature_data)
            for t in list(self.portfolio.keys()) + list(top_picks["ticker"].values)
            if self._get_current_price(t, feature_data)
        }

        if not self._check_drawdown(current_prices):
            return

        target_tickers = set(top_picks["ticker"].tolist())
        held_tickers   = set(self.portfolio.keys())

        # Sell positions no longer in top picks
        for ticker in held_tickers - target_tickers:
            price = current_prices.get(ticker)
            if price:
                self.sell(ticker, price, reason="Rebalance: dropped from top picks", date=today)

        # Buy new top picks
        for _, row in top_picks.iterrows():
            ticker = row["ticker"]
            if ticker not in self.portfolio:
                price = current_prices.get(ticker)
                if price:
                    reason = row.get("explanation", f"Final score={row['final_score']:.3f}")
                    self.buy(ticker, price, reason=reason, date=today)

        self._snapshot(current_prices, today)

    # ── Snapshot & Logging ────────────────────────────────────────────
    def _snapshot(self, current_prices: Dict[str, float], date: str):
        total    = self.portfolio_value(current_prices)
        equity   = total - self.cash
        drawdown = (self.peak_value - total) / (self.peak_value + 1e-9)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
                (date, cash, equity, total_value, drawdown, n_positions)
            VALUES (?,?,?,?,?,?)
        """, (date, self.cash, equity, total, drawdown, len(self.portfolio)))
        conn.commit()
        conn.close()

        logger.info(f"Portfolio [{date}]: total=₹{total:,.0f}  "
                    f"cash=₹{self.cash:,.0f}  "
                    f"positions={len(self.portfolio)}  "
                    f"drawdown={drawdown:.1%}")

    def _log_trade(
        self, ticker, action, shares, price, value, pnl, reason, date
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO trades (date, ticker, action, shares, price, value, cost, pnl, reason)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            date or datetime.today().strftime("%Y-%m-%d"),
            ticker, action, shares, price, value,
            value * TRANSACTION_COST_PCT, pnl, reason,
        ))
        conn.commit()
        conn.close()

    # ── Reporting ─────────────────────────────────────────────────────
    def get_trade_history(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query("SELECT * FROM trades ORDER BY date", conn)
        conn.close()
        return df

    def get_portfolio_history(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df   = pd.read_sql_query(
            "SELECT * FROM portfolio_snapshots ORDER BY date", conn
        )
        conn.close()
        return df

    def get_performance_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        history = self.get_portfolio_history()
        if history.empty:
            return {}

        total_now  = history["total_value"].iloc[-1]
        total_ret  = (total_now - self.initial_capital) / self.initial_capital
        max_dd     = history["drawdown"].min()
        n_trades   = len(self.get_trade_history())

        # Annualised return
        n_days     = max(1, len(history))
        ann_ret    = (1 + total_ret) ** (252 / n_days) - 1

        return {
            "initial_capital":   self.initial_capital,
            "current_value":     total_now,
            "total_return_pct":  round(total_ret * 100, 2),
            "annualised_return": round(ann_ret * 100, 2),
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "n_trades":          n_trades,
            "n_positions":       len(self.portfolio),
        }
