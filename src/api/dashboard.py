"""Streamlit dashboard for cryptocurrency analysis and visualization.

Covers RNCP competencies:
- C2.1.3: Requetes et calculs (dashboard, SQL, Python)
- C2.2.1: Representation des donnees (visualisations)

Usage:
    streamlit run src/api/dashboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.charts import (
    candlestick_chart,
    correlation_heatmap,
    fear_greed_chart,
    price_line_chart,
    price_with_sma_chart,
    returns_distribution,
    volatility_chart,
)
from src.analysis.statistics import (
    compare_cryptos,
    correlation_matrix,
    descriptive_stats,
    test_correlation,
    test_normality,
)
from src.features.feature_engineering import FeatureEngineer
from src.models.model_registry import ModelRegistry
from src.utils.config import ROOT_DIR, load_config

# --- Page config ---
st.set_page_config(
    page_title="Crypto Predict MLOps",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed data from CSV files."""
    config = load_config()
    processed_dir = ROOT_DIR / config["storage"]["processed_data_dir"]
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]

    # Load combined processed data
    combined_path = processed_dir / "all_cryptos_processed.csv"
    if combined_path.exists():
        market_df = pd.read_csv(combined_path, parse_dates=["date"])
    else:
        # Fallback: load individual raw files
        dfs = []
        for crypto in config["collection"]["cryptocurrencies"]:
            path = raw_dir / f"{crypto['symbol']}_market_data.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["crypto_id"] = crypto["id"]
                dfs.append(df)
        market_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not market_df.empty:
            market_df["date"] = pd.to_datetime(market_df["date"])

    # Load Fear & Greed
    fg_path = processed_dir / "all_cryptos_processed.csv"
    fg_raw_path = raw_dir / "fear_greed_index.csv"
    fg_df = pd.DataFrame()
    if fg_raw_path.exists():
        fg_df = pd.read_csv(fg_raw_path, parse_dates=["date"])

    return market_df, fg_df


def sidebar_filters(df: pd.DataFrame) -> tuple:
    """Render sidebar filters and return selections."""
    st.sidebar.title("Filtres")

    # Crypto selection
    cryptos = sorted(df["crypto_id"].unique()) if "crypto_id" in df.columns else []
    selected_cryptos = st.sidebar.multiselect(
        "Cryptomonnaies", cryptos, default=cryptos,
    )

    # Date range
    if not df.empty and "date" in df.columns:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.sidebar.date_input(
            "Periode", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
        )
    else:
        date_range = None

    return selected_cryptos, date_range


def filter_data(df: pd.DataFrame, cryptos: list, date_range) -> pd.DataFrame:
    """Apply sidebar filters to the DataFrame."""
    filtered = df.copy()

    if cryptos and "crypto_id" in filtered.columns:
        filtered = filtered[filtered["crypto_id"].isin(cryptos)]

    if date_range and len(date_range) == 2 and "date" in filtered.columns:
        start, end = date_range
        filtered = filtered[
            (filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)
        ]

    return filtered


def page_overview(df: pd.DataFrame):
    """Page 1: Vue d'ensemble with KPIs and comparison table."""
    st.header("Vue d'ensemble")

    if df.empty:
        st.warning("Aucune donnee disponible. Lancez d'abord le pipeline ETL.")
        return

    # KPI cards per crypto
    cryptos = df["crypto_id"].unique() if "crypto_id" in df.columns else []
    cols = st.columns(min(len(cryptos), 5)) if len(cryptos) > 0 else []

    for i, crypto in enumerate(cryptos):
        crypto_df = df[df["crypto_id"] == crypto].sort_values("date")
        if crypto_df.empty:
            continue

        with cols[i % len(cols)]:
            latest_price = crypto_df["price"].iloc[-1]
            if len(crypto_df) > 1:
                prev_price = crypto_df["price"].iloc[-2]
                delta = ((latest_price - prev_price) / prev_price) * 100
                delta_str = f"{delta:+.2f}%"
            else:
                delta_str = None

            st.metric(
                label=crypto.upper(),
                value=f"${latest_price:,.2f}",
                delta=delta_str,
            )

    st.divider()

    # Price comparison chart
    if "crypto_id" in df.columns:
        st.plotly_chart(
            price_line_chart(df, list(cryptos)),
            use_container_width=True,
        )

    # Performance comparison table
    st.subheader("Comparaison de performance")
    if "daily_return" in df.columns and "crypto_id" in df.columns:
        comparison = compare_cryptos(df, "daily_return")
        if not comparison.empty:
            st.dataframe(comparison, use_container_width=True)
    else:
        st.info("Lancez le pipeline ETL pour obtenir les returns journaliers.")


def page_technical(df: pd.DataFrame):
    """Page 2: Technical analysis with OHLC and indicators."""
    st.header("Analyse Technique")

    if df.empty:
        st.warning("Aucune donnee disponible.")
        return

    cryptos = df["crypto_id"].unique() if "crypto_id" in df.columns else []
    selected = st.selectbox("Cryptomonnaie", cryptos)

    crypto_df = df[df["crypto_id"] == selected].sort_values("date") if selected else df

    if crypto_df.empty:
        return

    # Candlestick chart
    ohlc_cols = {"open", "high", "low", "close"}
    if ohlc_cols.issubset(set(crypto_df.columns)):
        st.plotly_chart(
            candlestick_chart(crypto_df, f"OHLC — {selected.upper()}"),
            use_container_width=True,
        )

    # Price + SMA
    if "sma_7" in crypto_df.columns:
        st.plotly_chart(
            price_with_sma_chart(crypto_df, f"Prix & SMA — {selected.upper()}"),
            use_container_width=True,
        )

    # Volatility
    if "volatility_7d" in df.columns and "crypto_id" in df.columns:
        st.plotly_chart(
            volatility_chart(df, list(cryptos)),
            use_container_width=True,
        )


def page_sentiment(df: pd.DataFrame, fg_df: pd.DataFrame):
    """Page 3: Sentiment analysis with Fear & Greed Index."""
    st.header("Analyse du Sentiment")

    if fg_df.empty:
        st.warning("Aucune donnee Fear & Greed disponible.")
        return

    # Fear & Greed chart
    st.plotly_chart(fear_greed_chart(fg_df), use_container_width=True)

    # Current value
    latest = fg_df.sort_values("date").iloc[-1]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fear & Greed actuel", int(latest["fg_value"]),
                   delta=latest.get("fg_classification", ""))
    with col2:
        st.metric("Date", str(latest["date"].date() if hasattr(latest["date"], "date") else latest["date"]))

    # Correlation with price
    if not df.empty and "fg_value" in df.columns and "daily_return" in df.columns:
        st.subheader("Correlation Sentiment / Returns")

        cryptos = df["crypto_id"].unique() if "crypto_id" in df.columns else []
        for crypto in cryptos:
            crypto_df = df[df["crypto_id"] == crypto]
            result = test_correlation(crypto_df["fg_value"], crypto_df["daily_return"])
            if result.get("coefficient") is not None:
                st.write(
                    f"**{crypto.upper()}**: r = {result['coefficient']}, "
                    f"p = {result['p_value']} — {result['interpretation']}"
                )


def page_statistics(df: pd.DataFrame):
    """Page 4: Statistical analysis and distributions."""
    st.header("Analyse Statistique")

    if df.empty:
        st.warning("Aucune donnee disponible.")
        return

    # Descriptive statistics
    st.subheader("Statistiques descriptives")
    if "crypto_id" in df.columns:
        for crypto in df["crypto_id"].unique():
            crypto_df = df[df["crypto_id"] == crypto]
            with st.expander(f"{crypto.upper()}", expanded=False):
                numeric_cols = ["price", "daily_return", "total_volume", "volatility_7d"]
                available = [c for c in numeric_cols if c in crypto_df.columns]
                if available:
                    st.dataframe(crypto_df[available].describe().round(4),
                                 use_container_width=True)

    # Returns distribution
    st.subheader("Distribution des returns")
    if "daily_return" in df.columns and "crypto_id" in df.columns:
        selected = st.selectbox("Crypto", df["crypto_id"].unique(), key="stats_crypto")
        st.plotly_chart(
            returns_distribution(df, selected),
            use_container_width=True,
        )

        # Normality test
        crypto_df = df[df["crypto_id"] == selected]
        result = test_normality(crypto_df["daily_return"])
        if result.get("p_value") is not None:
            st.write(f"**Test de normalite (Shapiro-Wilk)**: W = {result['statistic']}, "
                     f"p = {result['p_value']} — "
                     f"{'Distribution normale' if result['is_normal'] else 'Distribution non-normale'}")

    # Correlation matrix
    st.subheader("Matrice de correlation")
    corr_cols = ["price", "total_volume", "daily_return", "volatility_7d",
                 "sma_7", "sma_30", "fg_value"]
    available_corr = [c for c in corr_cols if c in df.columns]

    if len(available_corr) >= 2:
        corr = correlation_matrix(df, available_corr)
        st.plotly_chart(
            correlation_heatmap(corr),
            use_container_width=True,
        )


@st.cache_resource
def load_model_registry() -> ModelRegistry:
    """Charge le model registry (mis en cache)."""
    return ModelRegistry()


def page_predictions(market_df: pd.DataFrame):
    """Page 5: Predictions ML — tendances hausse/baisse par modele.

    Charge les modeles sauvegardes depuis le registry local et affiche
    les predictions pour chaque crypto disponible.

    Competences RNCP :
    - C5.3.1 : Versioning et serialisation des modeles
    - C5.3.2 : Deploiement des modeles via API/UI
    """
    st.header("Predictions ML")

    # --- Chargement du registry ---
    try:
        registry = load_model_registry()
        models_meta = registry.list_models()
    except Exception as e:
        st.error(f"Impossible de charger le registry : {e}")
        return

    if not models_meta:
        st.warning(
            "Aucun modele entraine. Lancez d'abord `python scripts/train_models.py`."
        )
        return

    # --- Selections utilisateur ---
    col_sel1, col_sel2 = st.columns(2)

    available_models = sorted({m["model_name"] for m in models_meta
                                if m["model_name"] != "lstm"})
    available_cryptos = sorted({m["crypto"] for m in models_meta
                                 if m["model_name"] != "lstm"})

    with col_sel1:
        selected_model = st.selectbox("Modele", available_models)
    with col_sel2:
        selected_crypto = st.selectbox("Crypto", available_cryptos)

    st.divider()

    # --- Recherche du meilleur modele matching ---
    model_dir = registry.get_latest_model(
        model_name=selected_model, crypto=selected_crypto
    )
    if model_dir is None:
        st.warning(f"Pas de modele '{selected_model}' pour '{selected_crypto}'.")
        return

    try:
        loaded = registry.load(model_dir)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modele : {e}")
        return

    model = loaded["model"]
    scaler = loaded["scaler"]
    metadata = loaded["metadata"]
    feature_names = metadata.get("feature_names", [])

    # --- Metriques du modele ---
    st.subheader("Performances du modele")
    metrics = metadata.get("metrics", {})
    if metrics:
        metric_cols = st.columns(min(len(metrics), 4))
        for i, (name, value) in enumerate(metrics.items()):
            with metric_cols[i % len(metric_cols)]:
                label = name.replace("_", " ").title()
                st.metric(label, f"{value:.3f}" if isinstance(value, float) else str(value))
    else:
        st.info("Pas de metriques disponibles.")

    st.caption(
        f"Modele entraine le {metadata.get('timestamp', '?')} "
        f"— {len(feature_names)} features"
    )

    st.divider()

    # --- Prediction sur les dernieres donnees ---
    st.subheader("Prediction en temps reel")

    if market_df.empty:
        st.warning("Donnees de marche non disponibles.")
        return

    crypto_df = market_df[market_df["coingecko_id"] == selected_crypto].copy()
    if crypto_df.empty:
        # Essayer avec crypto_id
        crypto_df = market_df[market_df.get("crypto_id", pd.Series()) == selected_crypto].copy()

    if crypto_df.empty:
        st.warning(f"Pas de donnees pour '{selected_crypto}' dans le fichier traite.")
        return

    try:
        engineer = FeatureEngineer(target_horizon=1)
        df_features = engineer.build_features(
            crypto_df.sort_values("date").reset_index(drop=True),
            crypto_id=selected_crypto
        )

        available_feat = [f for f in feature_names if f in df_features.columns]
        if not available_feat:
            st.error("Features du modele incompatibles avec les donnees actuelles.")
            return

        X_last = df_features[available_feat].iloc[[-1]].copy()
        X_last = X_last.replace([np.inf, -np.inf], np.nan).fillna(X_last.median())

        X_input = scaler.transform(X_last) if scaler is not None else X_last.values
        proba = model.predict_proba(X_input)[0]
        proba_hausse = float(proba[1])
        prediction = "hausse" if proba_hausse > 0.5 else "baisse"
        confidence = proba_hausse if proba_hausse > 0.5 else 1 - proba_hausse

    except Exception as e:
        st.error(f"Erreur lors de la prediction : {e}")
        return

    # Affichage de la prediction
    col_pred, col_conf = st.columns(2)
    with col_pred:
        color = "normal" if prediction == "hausse" else "inverse"
        st.metric(
            label=f"Tendance prevue — {selected_crypto.upper()}",
            value=prediction.upper(),
            delta=f"{'↑' if prediction == 'hausse' else '↓'} {confidence:.1%}",
            delta_color=color,
        )
    with col_conf:
        st.metric("Probabilite hausse", f"{proba_hausse:.1%}")

    # Barre de confiance
    st.progress(proba_hausse, text=f"Probabilite hausse : {proba_hausse:.1%}")

    st.divider()

    # --- Tableau de predictions pour toutes les cryptos disponibles ---
    st.subheader("Predictions multi-cryptos")

    rows = []
    for crypto in available_cryptos:
        dir_c = registry.get_latest_model(model_name=selected_model, crypto=crypto)
        if dir_c is None:
            continue
        try:
            loaded_c = registry.load(dir_c)
            model_c = loaded_c["model"]
            scaler_c = loaded_c["scaler"]
            feats_c = loaded_c["metadata"].get("feature_names", [])
            metrics_c = loaded_c["metadata"].get("metrics", {})

            df_c = market_df[market_df["coingecko_id"] == crypto].copy()
            if df_c.empty:
                continue

            eng_c = FeatureEngineer(target_horizon=1)
            df_f_c = eng_c.build_features(
                df_c.sort_values("date").reset_index(drop=True), crypto_id=crypto
            )
            avail_c = [f for f in feats_c if f in df_f_c.columns]
            if not avail_c:
                continue

            X_c = df_f_c[avail_c].iloc[[-1]].copy()
            X_c = X_c.replace([np.inf, -np.inf], np.nan).fillna(X_c.median())
            X_in = scaler_c.transform(X_c) if scaler_c is not None else X_c.values
            p = model_c.predict_proba(X_in)[0]
            ph = float(p[1])

            rows.append({
                "Crypto": crypto.upper(),
                "Tendance": "HAUSSE" if ph > 0.5 else "BAISSE",
                "P(hausse)": f"{ph:.1%}",
                "Accuracy": f"{metrics_c.get('accuracy', 0):.3f}",
                "F1": f"{metrics_c.get('f1', 0):.3f}",
            })
        except Exception:
            continue

    if rows:
        df_preds = pd.DataFrame(rows)
        st.dataframe(df_preds, use_container_width=True, hide_index=True)
    else:
        st.info("Pas assez de modeles disponibles pour le tableau multi-cryptos.")


def main():
    """Main dashboard entry point."""
    st.title("Crypto Predict MLOps — Dashboard")

    # Load data
    market_df, fg_df = load_data()

    # Sidebar filters (pas utilises sur la page Predictions)
    if not market_df.empty:
        selected_cryptos, date_range = sidebar_filters(market_df)
        filtered_df = filter_data(market_df, selected_cryptos, date_range)
    else:
        filtered_df = market_df
        st.sidebar.warning("Pas de donnees chargees")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Vue d'ensemble", "Analyse Technique", "Sentiment", "Statistiques", "Predictions ML"],
    )

    if page == "Vue d'ensemble":
        page_overview(filtered_df)
    elif page == "Analyse Technique":
        page_technical(filtered_df)
    elif page == "Sentiment":
        page_sentiment(filtered_df, fg_df)
    elif page == "Statistiques":
        page_statistics(filtered_df)
    elif page == "Predictions ML":
        page_predictions(market_df)

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Crypto Predict MLOps — M2 Data Science")


if __name__ == "__main__":
    main()
