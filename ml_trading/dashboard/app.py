# dashboard/app.py
"""
ML Trading Dashboard — Streamlit
Run with: streamlit run dashboard/app.py
"""

import os
import sys
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DATABASE_PATH, MODEL_DIR, INITIAL_CAPITAL,
    WEIGHT_SHARPE, WEIGHT_MOMENTUM, WEIGHT_ML
)
from training.trainer import ModelTrainer

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Trading System — NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .section-header {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
    .stDataFrame { border: 1px solid #30363d; border-radius: 6px; }
    .sidebar .sidebar-content { background: #161b22; }
    div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    .ticker-pill {
        display: inline-block;
        background: #1f6feb33;
        border: 1px solid #1f6feb;
        border-radius: 4px;
        padding: 2px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #58a6ff;
        margin: 3px;
    }
    .buy-pill  { background: #1a7f3733; border-color: #2ea043; color: #3fb950; }
    .sell-pill { background: #b6271733; border-color: #da3633; color: #f85149; }
    h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ── DB helpers ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_predictions():
    if not os.path.exists(DATABASE_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DATABASE_PATH)
    df   = pd.read_sql_query("""
        SELECT * FROM daily_predictions
        WHERE date = (SELECT MAX(date) FROM daily_predictions)
        ORDER BY final_score DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_portfolio_history():
    if not os.path.exists(DATABASE_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DATABASE_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM portfolio_snapshots ORDER BY date", conn
    )
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_trade_history():
    if not os.path.exists(DATABASE_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DATABASE_PATH)
    df   = pd.read_sql_query("SELECT * FROM trades ORDER BY date DESC", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_model_info():
    payload = ModelTrainer.load_best_model(MODEL_DIR)
    return payload


@st.cache_data(ttl=60)
def load_model_history():
    if not os.path.exists(DATABASE_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DATABASE_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM model_registry ORDER BY train_date DESC LIMIT 20", conn
    )
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_retrain_log():
    if not os.path.exists(DATABASE_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DATABASE_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM retrain_log ORDER BY run_date DESC LIMIT 10", conn
    )
    conn.close()
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ ML Trading System")
    st.markdown("**Market:** NSE / NIFTY 50")
    st.markdown(f"**As of:** {datetime.now().strftime('%d %b %Y  %H:%M')}")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🤖 Model", "📈 Portfolio", "🔮 Predictions", "📋 Trades", "🔄 Retrain"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("""
    <div style='font-size:0.75rem; color:#8b949e;'>
    <b>Hybrid Score Weights</b><br>
    Sharpe  : {:.0%}<br>
    Momentum: {:.0%}<br>
    ML Score: {:.0%}
    </div>
    """.format(WEIGHT_SHARPE, WEIGHT_MOMENTUM, WEIGHT_ML), unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("▶️ Run Daily Predictions", use_container_width=True):
        st.info("Launch `python main.py --predict` in terminal")

    if st.button("🏋️ Retrain Now", use_container_width=True):
        st.info("Launch `python main.py --retrain` in terminal")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 System Overview")

    # Portfolio KPIs
    ph = load_portfolio_history()
    preds = load_predictions()
    model_payload = load_model_info()

    col1, col2, col3, col4, col5 = st.columns(5)

    current_value = ph["total_value"].iloc[-1] if not ph.empty else INITIAL_CAPITAL
    total_return  = (current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 if not ph.empty else 0
    max_dd        = ph["drawdown"].min() * 100 if not ph.empty else 0
    n_positions   = ph["n_positions"].iloc[-1] if not ph.empty else 0
    model_auc     = model_payload["metrics"]["roc_auc"] if model_payload else 0

    def metric_card(label, value, color="#58a6ff"):
        return f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>"""

    with col1:
        st.markdown(metric_card("Portfolio Value", f"₹{current_value:,.0f}"), unsafe_allow_html=True)
    with col2:
        c = "#3fb950" if total_return >= 0 else "#f85149"
        st.markdown(metric_card("Total Return", f"{total_return:+.2f}%", c), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Max Drawdown", f"{max_dd:.2f}%", "#f85149"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Open Positions", str(n_positions)), unsafe_allow_html=True)
    with col5:
        st.markdown(metric_card("Model AUC", f"{model_auc:.4f}", "#d2a8ff"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity curve
    if not ph.empty:
        st.markdown('<p class="section-header">Equity Curve</p>', unsafe_allow_html=True)
        ph["date"] = pd.to_datetime(ph["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ph["date"], y=ph["total_value"],
            mode="lines", name="Portfolio",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
        ))
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot",
                      line_color="#8b949e", annotation_text="Initial Capital")
        fig.update_layout(
            height=320, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor="#21262d", showgrid=True),
            yaxis=dict(gridcolor="#21262d", showgrid=True, tickprefix="₹"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Today's picks
    if not preds.empty:
        st.markdown('<p class="section-header">Today\'s Top Picks</p>', unsafe_allow_html=True)
        top = preds[preds["selected"] == 1]
        for _, row in top.iterrows():
            ticker = row["ticker"].replace(".NS", "")
            st.markdown(
                f'<span class="ticker-pill">{ticker}</span>'
                f'  ML: <b>{row["ml_probability"]*100:.1f}%</b>  '
                f'  Sharpe: <b>{row["sharpe_score"]:.2f}</b>  '
                f'  Score: <b>{row["final_score"]*100:.1f}</b>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Model
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model":
    st.markdown("# 🤖 ML Model")

    model_payload = load_model_info()
    model_history = load_model_history()

    if model_payload:
        m = model_payload["metrics"]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Current Model")
            info_df = pd.DataFrame([{
                "Property": k, "Value": v
            } for k, v in {
                "Model Name":   model_payload["model_name"],
                "Trained On":   model_payload.get("train_date", "N/A")[:19],
                "Pred Window":  f"{model_payload.get('pred_window', 10)} days",
                "# Features":   len(model_payload.get("feature_cols", [])),
            }.items()])
            st.dataframe(info_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("### Performance Metrics")
            metrics_df = pd.DataFrame([{
                "Metric": k.upper(), "Value": f"{v:.4f}"
            } for k, v in m.items()])
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        # Metrics radar chart
        st.markdown("### Metrics Radar")
        categories = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        values     = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]]

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="#58a6ff"),
            name="Current Model",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d"),
                angularaxis=dict(gridcolor="#30363d"),
            ),
            paper_bgcolor="#0d1117", font_color="#e6edf3",
            height=350, margin=dict(l=60, r=60, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No trained model found. Run `python main.py --train` first.")

    # Model history table
    if not model_history.empty:
        st.markdown("### Model Version History")
        display_cols = ["model_name", "train_date", "accuracy", "f1", "roc_auc", "n_features", "is_best"]
        available    = [c for c in display_cols if c in model_history.columns]
        st.dataframe(
            model_history[available].rename(columns={
                "model_name": "Model", "train_date": "Trained",
                "roc_auc": "AUC", "n_features": "Features", "is_best": "Best",
            }),
            hide_index=True, use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Portfolio
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Portfolio":
    st.markdown("# 📈 Portfolio Performance")

    ph = load_portfolio_history()
    th = load_trade_history()

    if ph.empty:
        st.info("No portfolio history yet. Run `python main.py --trade` first.")
    else:
        ph["date"] = pd.to_datetime(ph["date"])

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Value",   f"₹{ph['total_value'].iloc[-1]:,.0f}")
        c2.metric("Peak Value",      f"₹{ph['total_value'].max():,.0f}")
        c3.metric("Max Drawdown",    f"{ph['drawdown'].min()*100:.2f}%")
        c4.metric("Avg Positions",   f"{ph['n_positions'].mean():.1f}")

        # Portfolio value + drawdown
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.04)

        fig.add_trace(go.Scatter(
            x=ph["date"], y=ph["total_value"], name="Portfolio Value",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ph["date"], y=ph["drawdown"] * 100, name="Drawdown %",
            line=dict(color="#f85149", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.1)",
        ), row=2, col=1)

        fig.update_layout(
            height=460, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
            fig.update_layout(**{axis: dict(gridcolor="#21262d")})
        fig.update_yaxes(tickprefix="₹", row=1)
        fig.update_yaxes(ticksuffix="%", row=2)

        st.plotly_chart(fig, use_container_width=True)

    # Trade P&L
    if not th.empty:
        st.markdown("### Trade P&L")
        sells = th[th["action"] == "SELL"].copy()
        if not sells.empty:
            sells["date"] = pd.to_datetime(sells["date"])
            fig2 = px.bar(
                sells, x="date", y="pnl", color="pnl",
                color_continuous_scale=["#f85149", "#8b949e", "#3fb950"],
                color_continuous_midpoint=0,
                labels={"pnl": "P&L (₹)", "date": "Date"},
            )
            fig2.update_layout(
                height=260, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e6edf3", margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", tickprefix="₹"),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Predictions
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    st.markdown("# 🔮 Daily Predictions")

    preds = load_predictions()
    if preds.empty:
        st.info("No predictions found. Run `python main.py --predict` first.")
    else:
        st.markdown(f"**Date:** {preds['date'].iloc[0]}")

        # Score breakdown bar chart
        preds["clean_ticker"] = preds["ticker"].str.replace(".NS", "", regex=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Sharpe", x=preds["clean_ticker"], y=preds["sharpe_score"] * WEIGHT_SHARPE,
            marker_color="#58a6ff",
        ))
        fig.add_trace(go.Bar(
            name="Momentum", x=preds["clean_ticker"], y=preds["momentum_score"] * WEIGHT_MOMENTUM,
            marker_color="#d2a8ff",
        ))
        fig.add_trace(go.Bar(
            name="ML Score", x=preds["clean_ticker"], y=preds["ml_probability"] * WEIGHT_ML,
            marker_color="#3fb950",
        ))
        fig.update_layout(
            barmode="stack", height=380,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="Contribution to Final Score"),
        )
        # Highlight selected stocks
        shapes = []
        for i, row in preds.reset_index().iterrows():
            if row["selected"] == 1:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=i - 0.5, x1=i + 0.5, y0=0, y1=1,
                    fillcolor="rgba(63,185,80,0.05)",
                    line=dict(color="#3fb950", width=1, dash="dot"),
                ))
        fig.update_layout(shapes=shapes)
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.markdown("### All Stock Scores")
        display = preds[[
            "clean_ticker", "ml_probability", "sharpe_score",
            "momentum_score", "final_score", "selected",
        ]].copy()
        display.columns = ["Ticker", "ML Prob", "Sharpe", "Momentum", "Final Score", "Selected"]
        display["ML Prob"]    = (display["ML Prob"]    * 100).round(1).astype(str) + "%"
        display["Final Score"]= (display["Final Score"]* 100).round(2)
        display["Selected"]   = display["Selected"].map({1: "✅ BUY", 0: "—"})
        st.dataframe(display, hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Trades
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Trades":
    st.markdown("# 📋 Trade History")

    th = load_trade_history()
    if th.empty:
        st.info("No trades executed yet.")
    else:
        buys  = len(th[th["action"] == "BUY"])
        sells = len(th[th["action"] == "SELL"])
        total_pnl = th["pnl"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trades",  len(th))
        c2.metric("Buy / Sell",    f"{buys} / {sells}")
        c3.metric("Total P&L",     f"₹{total_pnl:+,.0f}")

        st.dataframe(
            th[["date", "ticker", "action", "shares", "price", "value", "pnl", "reason"]]
            .rename(columns={
                "date": "Date", "ticker": "Ticker", "action": "Action",
                "shares": "Shares", "price": "Price (₹)", "value": "Value (₹)",
                "pnl": "P&L (₹)", "reason": "Reason",
            }),
            hide_index=True, use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Retrain
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Retrain":
    st.markdown("# 🔄 Retrain Log")

    retrain_log = load_retrain_log()
    if retrain_log.empty:
        st.info("No retrain history found.")
    else:
        st.dataframe(
            retrain_log.rename(columns={
                "run_date": "Run Date", "status": "Status",
                "best_model": "Best Model", "roc_auc": "AUC", "notes": "Notes",
            }),
            hide_index=True, use_container_width=True,
        )

    st.markdown("### Manual Controls")
    st.code("# Run from terminal:\npython main.py --train    # Full training\npython main.py --retrain  # Incremental retrain\npython main.py --predict  # Daily predictions\npython main.py --trade    # Execute paper trades\npython main.py --all      # Full daily pipeline", language="bash")
