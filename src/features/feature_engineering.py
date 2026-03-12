"""Module de feature engineering pour la prediction de tendances crypto.

Construit les variables predictives a partir des donnees de marche :
- Indicateurs techniques (RSI, MACD, Bollinger Bands, ATR) via la lib `ta`
- Features temporelles (jour de la semaine, mois, lag features)
- Variable cible : tendance haussiere/baissiere a J+1 (classification binaire)

Competence RNCP C5.2.1 : Construire des variables en utilisant des langages
de programmation et des bibliotheques d'analyse de donnees.
"""

import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Construit les features pour le modele de prediction de tendances.

    Pipeline : indicateurs techniques -> features temporelles -> lags -> cible.
    """

    def __init__(self, target_horizon: int = 1):
        """Initialise le feature engineer.

        Args:
            target_horizon: Nombre de jours pour la prediction de tendance (defaut: 1 = J+1).
        """
        self.target_horizon = target_horizon

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques via la bibliotheque `ta`.

        Indicateurs calcules :
        - RSI (14 periodes) : force relative du mouvement
        - MACD (12, 26, 9) : convergence/divergence des moyennes mobiles
        - Bollinger Bands (20, 2) : bandes de volatilite
        - ATR (14) : Average True Range, mesure de volatilite
        - Stochastic Oscillator (14) : position du prix dans le range

        Args:
            df: DataFrame avec colonnes OHLCV (open, high, low, close, total_volume).

        Returns:
            DataFrame enrichi avec les indicateurs techniques.
        """
        df = df.copy()

        # RSI - Relative Strength Index
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

        # MACD - Moving Average Convergence Divergence
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bollinger.bollinger_hband()
        df["bb_middle"] = bollinger.bollinger_mavg()
        df["bb_lower"] = bollinger.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR - Average True Range
        df["atr_14"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Volume indicators
        df["volume_sma_20"] = df["total_volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["total_volume"] / df["volume_sma_20"]

        logger.info("Indicateurs techniques ajoutes : %d nouvelles colonnes", 14)
        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features temporelles extraites de la date.

        Features :
        - Jour de la semaine (0=lundi, 6=dimanche)
        - Mois (1-12)
        - Jour du mois (1-31)
        - Indicateur weekend

        Args:
            df: DataFrame avec colonne 'date'.

        Returns:
            DataFrame enrichi avec les features temporelles.
        """
        df = df.copy()
        dt = pd.to_datetime(df["date"])

        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["day_of_month"] = dt.dt.day
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        # Encodage cyclique pour jour de la semaine et mois
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        logger.info("Features temporelles ajoutees : 8 colonnes")
        return df

    def add_lag_features(self, df: pd.DataFrame,
                         lags: list[int] = None) -> pd.DataFrame:
        """Ajoute les features decalees (lag features) pour capturer les tendances passees.

        Args:
            df: DataFrame avec colonnes 'price', 'daily_return', 'total_volume'.
            lags: Liste des decalages en jours (defaut: [1, 2, 3, 5, 7]).

        Returns:
            DataFrame enrichi avec les lag features.
        """
        if lags is None:
            lags = [1, 2, 3, 5, 7]

        df = df.copy()

        for lag in lags:
            # Rendements passes
            df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
            # Prix relatif par rapport au prix actuel
            df[f"price_change_{lag}d"] = df["price"].pct_change(lag)
            # Volume relatif
            df[f"volume_change_{lag}d"] = df["total_volume"].pct_change(lag)

        # Rendement cumule sur differentes fenetres
        for window in [3, 5, 7, 14]:
            df[f"cum_return_{window}d"] = df["daily_return"].rolling(window=window).sum()

        logger.info("Lag features ajoutees : %d colonnes", len(lags) * 3 + 4)
        return df

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cree la variable cible : tendance haussiere (1) ou baissiere (0) a J+horizon.

        La cible est definie comme :
        - 1 si le prix a J+horizon est superieur au prix actuel
        - 0 sinon

        Args:
            df: DataFrame avec colonne 'price'.

        Returns:
            DataFrame avec colonne 'target' ajoutee.
        """
        df = df.copy()

        # Rendement futur sur l'horizon defini
        df["future_return"] = df["price"].pct_change(self.target_horizon).shift(
            -self.target_horizon
        )
        # Variable cible binaire : 1 = hausse, 0 = baisse (NaN preserve)
        df["target"] = df["future_return"].apply(
            lambda x: int(x > 0) if pd.notna(x) else np.nan
        )

        # Supprimer la colonne intermediaire
        df = df.drop(columns=["future_return"])

        n_up = df["target"].sum()
        n_total = df["target"].notna().sum()
        logger.info(
            "Variable cible creee (horizon=%d) : %d hausse / %d baisse (%.1f%% hausse)",
            self.target_horizon, n_up, n_total - n_up,
            100 * n_up / n_total if n_total > 0 else 0
        )
        return df

    def build_features(self, df: pd.DataFrame,
                       crypto_id: str = None) -> pd.DataFrame:
        """Pipeline complet de feature engineering pour une crypto.

        Applique sequentiellement : indicateurs techniques -> features temporelles
        -> lag features -> variable cible.

        Args:
            df: DataFrame brut avec colonnes OHLCV + date.
            crypto_id: Identifiant de la crypto (pour le log).

        Returns:
            DataFrame avec toutes les features construites.
        """
        label = crypto_id or "unknown"
        logger.info("Feature engineering pour '%s' (%d lignes)", label, len(df))

        df = self.add_technical_indicators(df)
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df)
        df = self.add_target(df)

        logger.info(
            "Feature engineering termine pour '%s' : %d lignes, %d colonnes",
            label, len(df), len(df.columns)
        )
        return df

    def build_features_all_cryptos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique le feature engineering par crypto puis concatene.

        Args:
            df: DataFrame avec toutes les cryptos (colonne 'coingecko_id').

        Returns:
            DataFrame concatene avec features pour toutes les cryptos.
        """
        frames = []
        for crypto_id, group in df.groupby("coingecko_id"):
            group = group.sort_values("date").reset_index(drop=True)
            enriched = self.build_features(group, crypto_id=crypto_id)
            frames.append(enriched)

        result = pd.concat(frames, ignore_index=True)
        logger.info(
            "Feature engineering complet : %d cryptos, %d lignes, %d colonnes",
            len(frames), len(result), len(result.columns)
        )
        return result
