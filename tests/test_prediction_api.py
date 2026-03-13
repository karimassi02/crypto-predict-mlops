"""Tests unitaires pour l'API de prediction FastAPI.

Utilise le TestClient de Starlette pour tester les endpoints sans
demarrer un vrai serveur.

Competence RNCP C5.3.2 : Deployer des modeles d'apprentissage automatique
en utilisant des API et des outils CI/CD.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metadata(model_name: str = "xgboost", crypto: str = "bitcoin",
                  n_features: int = 5) -> dict:
    return {
        "model_name": model_name,
        "crypto": crypto,
        "timestamp": "20250101_120000",
        "metrics": {"accuracy": 0.65, "f1": 0.60},
        "feature_names": [f"feat_{i}" for i in range(n_features)],
        "has_scaler": True,
        "path": "/models/fake_dir",
    }


def make_mock_model(proba: list = None):
    """Cree un mock de modele sklearn avec predict_proba."""
    model = MagicMock()
    proba = proba or [0.35, 0.65]
    model.predict_proba.return_value = np.array([proba])
    return model


def make_processed_df(crypto: str = "bitcoin", n: int = 100) -> pd.DataFrame:
    """Cree un DataFrame de donnees traitees synthetiques."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    price = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame({
        "coingecko_id": crypto,
        "symbol": "BTC",
        "date": dates.astype(str),
        "price": price,
        "market_cap": price * 19e6,
        "total_volume": np.random.uniform(30e9, 60e9, n),
        "open": price * 0.99,
        "high": price * 1.02,
        "low": price * 0.97,
        "close": price,
        "daily_return": np.random.randn(n) * 0.02,
        "fg_value": np.random.randint(10, 90, n),
        "crypto_id": crypto,
    })


# ---------------------------------------------------------------------------
# Client fixture — isole completement les dependances externes
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Construit un TestClient avec les dependances mockees."""
    metadata = make_metadata()
    mock_model = make_mock_model()
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 5))

    loaded_model = {
        "model": mock_model,
        "scaler": mock_scaler,
        "metadata": metadata,
    }

    mock_registry = MagicMock()
    mock_registry.list_models.return_value = [metadata]
    mock_registry.load.return_value = loaded_model
    mock_registry.get_latest_model.return_value = Path("/models/fake_dir")

    processed_df = make_processed_df("bitcoin")

    # Feature engineering renvoie un DataFrame avec les features attendues
    feature_cols = [f"feat_{i}" for i in range(5)]
    feature_df = processed_df.copy()
    for col in feature_cols:
        feature_df[col] = np.random.randn(len(feature_df))

    mock_engineer = MagicMock()
    mock_engineer.build_features.return_value = feature_df

    with patch("src.api.prediction_api.ModelRegistry", return_value=mock_registry), \
         patch("src.api.prediction_api.FeatureEngineer", return_value=mock_engineer), \
         patch("src.api.prediction_api.pd.read_csv", return_value=processed_df), \
         patch("src.api.prediction_api.Path.exists", return_value=True):

        # Import apres le patch pour que le lifespan utilise les mocks
        from src.api.prediction_api import app
        app.state.registry = mock_registry
        app.state.loaded_models = {"xgboost_bitcoin": loaded_model}

        # Injecter l'etat dans app_state
        import src.api.prediction_api as api_module
        api_module.app_state["registry"] = mock_registry
        api_module.app_state["loaded_models"] = {
            "xgboost_bitcoin": loaded_model
        }

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_registry, loaded_model, processed_df, mock_engineer


# ---------------------------------------------------------------------------
# Tests /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self, client):
        c, *_ = client
        response = c.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "available_models" in data

    def test_health_contains_model_keys(self, client):
        c, *_ = client
        response = c.get("/health")
        data = response.json()
        assert isinstance(data["available_models"], list)


# ---------------------------------------------------------------------------
# Tests /models
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_list_models_returns_list(self, client):
        c, *_ = client
        response = c.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_model_info_fields(self, client):
        c, *_ = client
        response = c.get("/models")
        data = response.json()
        assert len(data) >= 1
        model = data[0]
        assert "model_name" in model
        assert "crypto" in model
        assert "metrics" in model
        assert "n_features" in model


# ---------------------------------------------------------------------------
# Tests /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_hausse(self, client):
        """Prediction 'hausse' quand proba[1] > 0.5."""
        c, mock_registry, loaded, processed_df, mock_engineer = client

        # proba[1] = 0.65 -> hausse
        response = c.post("/predict", json={"crypto": "bitcoin", "model_name": "xgboost"})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "hausse"
        assert data["probability"] == pytest.approx(0.65, abs=0.01)

    def test_predict_baisse(self, client):
        """Prediction 'baisse' quand proba[1] <= 0.5."""
        c, mock_registry, loaded, processed_df, mock_engineer = client

        # Modifier le mock pour retourner une proba basse
        loaded["model"].predict_proba.return_value = np.array([[0.70, 0.30]])

        response = c.post("/predict", json={"crypto": "bitcoin", "model_name": "xgboost"})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "baisse"
        assert data["probability"] == pytest.approx(0.30, abs=0.01)

    def test_predict_response_fields(self, client):
        """La reponse contient tous les champs attendus."""
        c, *_ = client
        response = c.post("/predict", json={"crypto": "bitcoin", "model_name": "xgboost"})
        assert response.status_code == 200
        data = response.json()
        expected_keys = {"crypto", "model_name", "prediction", "probability", "metrics"}
        assert expected_keys.issubset(set(data.keys()))

    def test_predict_default_model_is_xgboost(self, client):
        """Le modele par defaut est xgboost."""
        c, *_ = client
        response = c.post("/predict", json={"crypto": "bitcoin"})
        assert response.status_code == 200
        assert response.json()["model_name"] == "xgboost"

    def test_predict_model_not_found(self, client):
        """404 si le modele n'existe pas."""
        c, mock_registry, *_ = client
        mock_registry.get_latest_model.return_value = None

        import src.api.prediction_api as api_module
        api_module.app_state["loaded_models"] = {}

        response = c.post("/predict", json={"crypto": "bitcoin", "model_name": "unknown_model"})
        assert response.status_code == 404

    def test_predict_lstm_not_supported(self, client):
        """501 pour les predictions LSTM via API."""
        c, mock_registry, loaded, *_ = client

        import src.api.prediction_api as api_module
        lstm_metadata = make_metadata(model_name="lstm", crypto="bitcoin")
        lstm_loaded = {**loaded, "metadata": lstm_metadata}
        api_module.app_state["loaded_models"]["lstm_bitcoin"] = lstm_loaded

        response = c.post("/predict", json={"crypto": "bitcoin", "model_name": "lstm"})
        assert response.status_code == 501

    def test_predict_crypto_not_in_data(self, client):
        """404 si la crypto n'est pas dans les donnees."""
        c, mock_registry, loaded, *_ = client

        import src.api.prediction_api as api_module
        api_module.app_state["loaded_models"]["xgboost_dogecoin"] = loaded

        with patch("src.api.prediction_api.pd.read_csv") as mock_read:
            # DataFrame ne contient que bitcoin, pas dogecoin
            mock_read.return_value = make_processed_df("bitcoin")
            with patch("src.api.prediction_api.Path.exists", return_value=True):
                response = c.post(
                    "/predict",
                    json={"crypto": "dogecoin", "model_name": "xgboost"}
                )

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests de validation des schemas Pydantic
# ---------------------------------------------------------------------------

class TestRequestValidation:
    def test_missing_crypto_field(self, client):
        """422 si le champ crypto est absent."""
        c, *_ = client
        response = c.post("/predict", json={"model_name": "xgboost"})
        assert response.status_code == 422

    def test_empty_body(self, client):
        """422 si le body est vide."""
        c, *_ = client
        response = c.post("/predict", json={})
        assert response.status_code == 422
