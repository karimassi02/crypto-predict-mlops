"""Chart generation module using Plotly.

Provides reusable chart functions for the Streamlit dashboard
(RNCP C2.2.1 — visualisation des donnees).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(df: pd.DataFrame, title: str = "OHLC") -> go.Figure:
    """Create a candlestick chart with volume bars.

    Args:
        df: DataFrame with columns: date, open, high, low, close, total_volume.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
        ),
        row=1, col=1,
    )

    if "total_volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df["date"], y=df["total_volume"], name="Volume",
                   marker_color="rgba(100,149,237,0.5)"),
            row=2, col=1,
        )

    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False,
        height=600, showlegend=False,
    )
    fig.update_yaxes(title_text="Prix (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def price_line_chart(df: pd.DataFrame, cryptos: list[str],
                     group_col: str = "crypto_id") -> go.Figure:
    """Create a multi-line price chart for comparing cryptocurrencies.

    Args:
        df: DataFrame with date, price, and group column.
        cryptos: List of crypto IDs to include.
        group_col: Column used to identify each crypto.

    Returns:
        Plotly Figure.
    """
    filtered = df[df[group_col].isin(cryptos)]

    fig = px.line(
        filtered, x="date", y="price", color=group_col,
        title="Evolution des prix",
        labels={"price": "Prix (USD)", "date": "Date", group_col: "Crypto"},
    )
    fig.update_layout(height=500)

    return fig


def price_with_sma_chart(df: pd.DataFrame, title: str = "Prix & Moyennes Mobiles") -> go.Figure:
    """Create a price chart with SMA overlays.

    Args:
        df: DataFrame with date, price, sma_7, sma_14, sma_30.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"], name="Prix",
        line=dict(color="white", width=2),
    ))

    sma_styles = {
        "sma_7": ("SMA 7j", "cyan", "dot"),
        "sma_14": ("SMA 14j", "orange", "dash"),
        "sma_30": ("SMA 30j", "magenta", "dashdot"),
    }

    for col, (name, color, dash) in sma_styles.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col], name=name,
                line=dict(color=color, width=1, dash=dash),
            ))

    fig.update_layout(
        title=title, height=500,
        xaxis_title="Date", yaxis_title="Prix (USD)",
        template="plotly_dark",
    )

    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame,
                        title: str = "Matrice de Correlation") -> go.Figure:
    """Create a correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 11},
    ))

    fig.update_layout(title=title, height=500, width=600)

    return fig


def returns_distribution(df: pd.DataFrame, crypto_id: str = None,
                         group_col: str = "crypto_id") -> go.Figure:
    """Create a histogram of daily returns.

    Args:
        df: DataFrame with daily_return column.
        crypto_id: Optional filter for specific crypto.
        group_col: Grouping column.

    Returns:
        Plotly Figure.
    """
    data = df.copy()
    if crypto_id and group_col in data.columns:
        data = data[data[group_col] == crypto_id]

    title = f"Distribution des returns journaliers"
    if crypto_id:
        title += f" — {crypto_id}"

    fig = px.histogram(
        data, x="daily_return", nbins=80,
        title=title, marginal="box",
        labels={"daily_return": "Return journalier"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(height=450)

    return fig


def fear_greed_chart(df: pd.DataFrame) -> go.Figure:
    """Create a Fear & Greed Index time series chart.

    Args:
        df: DataFrame with date, fg_value columns.

    Returns:
        Plotly Figure with colored zones.
    """
    fig = go.Figure()

    # Colored background zones
    zones = [
        (0, 20, "rgba(255,0,0,0.1)", "Extreme Fear"),
        (20, 40, "rgba(255,165,0,0.1)", "Fear"),
        (40, 60, "rgba(255,255,0,0.05)", "Neutral"),
        (60, 80, "rgba(144,238,144,0.1)", "Greed"),
        (80, 100, "rgba(0,128,0,0.1)", "Extreme Greed"),
    ]

    for y0, y1, color, name in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=name, annotation_position="left")

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["fg_value"], name="Fear & Greed",
        line=dict(color="white", width=2), fill="tozeroy",
        fillcolor="rgba(100,149,237,0.2)",
    ))

    if "fg_sma_7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["fg_sma_7"], name="SMA 7j",
            line=dict(color="cyan", width=1, dash="dash"),
        ))

    fig.update_layout(
        title="Fear & Greed Index", height=450,
        yaxis=dict(range=[0, 100], title="Valeur"),
        xaxis_title="Date", template="plotly_dark",
    )

    return fig


def volatility_chart(df: pd.DataFrame, cryptos: list[str],
                     group_col: str = "crypto_id") -> go.Figure:
    """Create a volatility comparison chart.

    Args:
        df: DataFrame with date, volatility_7d, and group column.
        cryptos: List of crypto IDs.
        group_col: Grouping column.

    Returns:
        Plotly Figure.
    """
    filtered = df[df[group_col].isin(cryptos)]

    fig = px.line(
        filtered, x="date", y="volatility_7d", color=group_col,
        title="Volatilite (rolling 7 jours)",
        labels={"volatility_7d": "Volatilite", "date": "Date", group_col: "Crypto"},
    )
    fig.update_layout(height=450, template="plotly_dark")

    return fig
