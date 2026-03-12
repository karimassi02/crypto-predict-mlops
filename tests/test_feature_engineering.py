"""Tests unitaires pour le module de feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_market_data():
    """Genere des donnees de marche synthetiques pour les tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    price = 50000 + np.cumsum(np.random.randn(n) * 500)

    return pd.DataFrame({
        "coingecko_id": "bitcoin",
        "symbol": "BTC",
        "date": dates,
        "price": price,
        "market_cap": price * 19e6,
        "total_volume": np.random.uniform(30e9, 60e9, n),
        "open": price * np.random.uniform(0.98, 1.0, n),
        "high": price * np.random.uniform(1.0, 1.03, n),
        "low": price * np.random.uniform(0.97, 1.0, n),
        "close": price * np.random.uniform(0.99, 1.01, n),
        "crypto_id": "bitcoin",
        "daily_return": np.random.randn(n) * 0.02,
        "fg_value": np.random.randint(10, 90, n),
    })


@pytest.fixture
def engineer():
    return FeatureEngineer(target_horizon=1)


class TestFeatureEngineer:
    """Tests pour la classe FeatureEngineer."""

    def test_add_technical_indicators(self, engineer, sample_market_data):
        """Verifie que les indicateurs techniques sont bien ajoutes."""
        result = engineer.add_technical_indicators(sample_market_data)

        expected_cols = [
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
            "atr_14", "stoch_k", "stoch_d", "volume_sma_20", "volume_ratio"
        ]
        for col in expected_cols:
            assert col in result.columns, f"Colonne manquante : {col}"

        # RSI doit etre entre 0 et 100 (apres la periode de warmup)
        rsi_valid = result["rsi_14"].dropna()
        assert rsi_valid.between(0, 100).all(), "RSI hors limites [0, 100]"

    def test_add_temporal_features(self, engineer, sample_market_data):
        """Verifie les features temporelles."""
        result = engineer.add_temporal_features(sample_market_data)

        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
        assert "day_sin" in result.columns
        assert "month_cos" in result.columns

        # Jour de la semaine entre 0 et 6
        assert result["day_of_week"].between(0, 6).all()
        # is_weekend est binaire
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_add_lag_features(self, engineer, sample_market_data):
        """Verifie les lag features."""
        result = engineer.add_lag_features(sample_market_data)

        # Verifier les lags par defaut [1, 2, 3, 5, 7]
        for lag in [1, 2, 3, 5, 7]:
            assert f"return_lag_{lag}" in result.columns
            assert f"price_change_{lag}d" in result.columns
            assert f"volume_change_{lag}d" in result.columns

        # Verifier les rendements cumules
        for window in [3, 5, 7, 14]:
            assert f"cum_return_{window}d" in result.columns

    def test_add_target(self, engineer, sample_market_data):
        """Verifie la variable cible."""
        result = engineer.add_target(sample_market_data)

        assert "target" in result.columns
        # La cible est binaire (0 ou 1), avec NaN pour les dernieres lignes
        valid_targets = result["target"].dropna()
        assert set(valid_targets.unique()).issubset({0, 1})
        # La derniere ligne doit etre NaN (pas de J+1)
        assert pd.isna(result["target"].iloc[-1])

    def test_build_features_complete(self, engineer, sample_market_data):
        """Verifie le pipeline complet de feature engineering."""
        result = engineer.build_features(sample_market_data, crypto_id="bitcoin")

        # Nombre de colonnes augmente significativement
        assert len(result.columns) > len(sample_market_data.columns)
        # La cible est presente
        assert "target" in result.columns
        # Les indicateurs techniques sont presents
        assert "rsi_14" in result.columns
        assert "macd" in result.columns

    def test_build_features_all_cryptos(self, engineer):
        """Verifie le pipeline multi-cryptos."""
        np.random.seed(42)
        frames = []
        for crypto in ["bitcoin", "ethereum"]:
            n = 50
            frames.append(pd.DataFrame({
                "coingecko_id": crypto,
                "symbol": crypto[:3].upper(),
                "date": pd.date_range("2025-01-01", periods=n, freq="D"),
                "price": 50000 + np.cumsum(np.random.randn(n) * 500),
                "market_cap": np.random.uniform(1e12, 2e12, n),
                "total_volume": np.random.uniform(30e9, 60e9, n),
                "open": np.random.uniform(49000, 51000, n),
                "high": np.random.uniform(50000, 52000, n),
                "low": np.random.uniform(48000, 50000, n),
                "close": np.random.uniform(49500, 51500, n),
                "crypto_id": crypto,
                "daily_return": np.random.randn(n) * 0.02,
                "fg_value": np.random.randint(10, 90, n),
            }))

        df = pd.concat(frames, ignore_index=True)
        result = engineer.build_features_all_cryptos(df)

        # Les deux cryptos sont presentes
        assert set(result["coingecko_id"].unique()) == {"bitcoin", "ethereum"}
        assert len(result) == 100  # 50 + 50

    def test_target_horizon(self, sample_market_data):
        """Verifie que l'horizon de prediction fonctionne."""
        engineer_3d = FeatureEngineer(target_horizon=3)
        result = engineer_3d.add_target(sample_market_data)

        # Les 3 dernieres lignes doivent etre NaN
        assert pd.isna(result["target"].iloc[-1])
        assert pd.isna(result["target"].iloc[-2])
        assert pd.isna(result["target"].iloc[-3])
