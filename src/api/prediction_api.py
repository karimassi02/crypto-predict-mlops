"""API de prediction de tendances crypto avec FastAPI.

Expose un endpoint /predict qui charge le meilleur modele sauvegarde
et retourne une prediction de tendance (hausse/baisse) pour une crypto.

Competence RNCP C5.3.2 : Deployer des modeles d'apprentissage automatique
en utilisant des API et des outils CI/CD.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.feature_engineering import FeatureEngineer
from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# --- Etat global de l'application ---
app_state = {
    "registry": None,
    "loaded_models": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le registry et les modeles au demarrage."""
    logger.info("Demarrage de l'API de prediction")
    app_state["registry"] = ModelRegistry()

    # Pre-charger les modeles disponibles
    models = app_state["registry"].list_models()
    for meta in models:
        key = f"{meta['model_name']}_{meta['crypto']}"
        if key not in app_state["loaded_models"]:
            try:
                loaded = app_state["registry"].load(Path(meta["path"]))
                app_state["loaded_models"][key] = loaded
                logger.info("Modele pre-charge : %s", key)
            except Exception as e:
                logger.warning("Impossible de charger %s : %s", key, e)

    logger.info("%d modeles charges", len(app_state["loaded_models"]))
    yield
    logger.info("Arret de l'API")


app = FastAPI(
    title="Crypto Predict API",
    description="API de prediction de tendances pour les cryptomonnaies",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schemas Pydantic ---
class PredictionRequest(BaseModel):
    """Schema de requete pour une prediction."""
    crypto: str = Field(
        ..., description="Identifiant de la crypto (ex: bitcoin, ethereum)",
        examples=["bitcoin"]
    )
    model_name: str = Field(
        default="xgboost",
        description="Nom du modele a utiliser",
        examples=["xgboost", "random_forest", "logistic_regression"]
    )


class PredictionResponse(BaseModel):
    """Schema de reponse avec la prediction."""
    crypto: str
    model_name: str
    prediction: str = Field(description="Tendance predite : 'hausse' ou 'baisse'")
    probability: float = Field(description="Probabilite de la prediction (0-1)")
    metrics: dict = Field(description="Metriques du modele utilise")


class ModelInfo(BaseModel):
    """Information sur un modele disponible."""
    model_name: str
    crypto: str
    timestamp: str
    metrics: dict
    n_features: int


class HealthResponse(BaseModel):
    """Reponse du health check."""
    status: str
    models_loaded: int
    available_models: list[str]


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifie l'etat de l'API et les modeles charges."""
    return HealthResponse(
        status="ok",
        models_loaded=len(app_state["loaded_models"]),
        available_models=list(app_state["loaded_models"].keys()),
    )


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """Liste tous les modeles disponibles."""
    registry = app_state["registry"]
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry non initialise")

    models = registry.list_models()
    return [
        ModelInfo(
            model_name=m["model_name"],
            crypto=m["crypto"],
            timestamp=m.get("timestamp", "unknown"),
            metrics=m.get("metrics", {}),
            n_features=len(m.get("feature_names", [])),
        )
        for m in models
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predit la tendance (hausse/baisse) pour une crypto.

    Charge le modele demande et les dernieres donnees de marche
    pour produire une prediction.
    """
    key = f"{request.model_name}_{request.crypto}"

    # Chercher le modele charge
    loaded = app_state["loaded_models"].get(key)
    if loaded is None:
        # Tenter de charger depuis le registry
        registry = app_state["registry"]
        model_dir = registry.get_latest_model(
            model_name=request.model_name, crypto=request.crypto
        )
        if model_dir is None:
            available = list(app_state["loaded_models"].keys())
            raise HTTPException(
                status_code=404,
                detail=f"Modele '{key}' non trouve. Disponibles : {available}"
            )
        loaded = registry.load(model_dir)
        app_state["loaded_models"][key] = loaded

    model = loaded["model"]
    scaler = loaded["scaler"]
    metadata = loaded["metadata"]
    feature_names = metadata.get("feature_names", [])

    # Charger les dernieres donnees de marche
    data_path = ROOT_DIR / "data" / "processed" / "all_cryptos_processed.csv"
    if not data_path.exists():
        raise HTTPException(status_code=500, detail="Donnees non disponibles")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df_crypto = df[df["coingecko_id"] == request.crypto].copy()

    if df_crypto.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Pas de donnees pour '{request.crypto}'"
        )

    # Feature engineering
    engineer = FeatureEngineer(target_horizon=1)
    df_features = engineer.build_features(df_crypto.sort_values("date").reset_index(drop=True),
                                          crypto_id=request.crypto)

    # Prendre la derniere ligne avec des features completes
    available_features = [f for f in feature_names if f in df_features.columns]
    if not available_features:
        raise HTTPException(status_code=500, detail="Features incompatibles")

    X_last = df_features[available_features].iloc[[-1]].copy()
    X_last = X_last.replace([np.inf, -np.inf], np.nan).fillna(X_last.median())

    # Prediction
    if scaler is not None:
        X_last_scaled = scaler.transform(X_last)
    else:
        X_last_scaled = X_last.values

    if request.model_name == "lstm":
        raise HTTPException(
            status_code=501,
            detail="Prediction LSTM via API non encore supportee (necessite une sequence)"
        )

    proba = model.predict_proba(X_last_scaled)[0]
    prediction_class = int(proba[1] > 0.5)
    probability = float(proba[1])

    return PredictionResponse(
        crypto=request.crypto,
        model_name=request.model_name,
        prediction="hausse" if prediction_class == 1 else "baisse",
        probability=probability,
        metrics=metadata.get("metrics", {}),
    )
