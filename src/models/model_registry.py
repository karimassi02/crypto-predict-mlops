"""Module de sauvegarde et versioning des modeles entraines.

Gere la serialisation des modeles avec joblib (sklearn/XGBoost) et
torch.save (LSTM PyTorch), ainsi que le versioning avec MLflow.

Competence RNCP C5.3.1 : Sauvegarder le modele d'apprentissage automatique
entraine a l'aide d'outils de serialisation, versioning afin de pouvoir
le deployer dans des environnements de production.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow

from src.models.trainer import TrainingResult

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


class ModelRegistry:
    """Gere la sauvegarde, le chargement et le versioning des modeles.

    Sauvegarde locale : joblib pour sklearn/XGBoost, torch.save pour LSTM.
    Versioning : MLflow Model Registry pour le tracking des experiences.
    """

    def __init__(self, models_dir: Path = None,
                 mlflow_tracking_uri: str = None):
        """Initialise le registry.

        Args:
            models_dir: Repertoire de sauvegarde locale (defaut: project/models/).
            mlflow_tracking_uri: URI du serveur MLflow.
        """
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            tracking_path = (ROOT_DIR / "mlruns").resolve()
            mlflow.set_tracking_uri(tracking_path.as_uri())

    def save_sklearn_model(self, result: TrainingResult,
                           crypto: str = "all") -> Path:
        """Sauvegarde un modele sklearn/XGBoost avec joblib.

        Sauvegarde le modele, le scaler (si present) et les metadonnees.

        Args:
            result: Resultat d'entrainement contenant le modele.
            crypto: Nom de la crypto associee.

        Returns:
            Chemin du repertoire de sauvegarde.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"{result.model_name}_{crypto}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modele
        model_path = model_dir / "model.joblib"
        joblib.dump(result.model, model_path)
        logger.info("Modele sauvegarde : %s", model_path)

        # Sauvegarder le scaler si present
        if result.scaler is not None:
            scaler_path = model_dir / "scaler.joblib"
            joblib.dump(result.scaler, scaler_path)
            logger.info("Scaler sauvegarde : %s", scaler_path)

        # Sauvegarder les metadonnees
        metadata = {
            "model_name": result.model_name,
            "crypto": crypto,
            "timestamp": timestamp,
            "metrics": result.metrics,
            "feature_names": result.feature_names,
            "has_scaler": result.scaler is not None,
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Metadonnees sauvegardees : %s", metadata_path)
        return model_dir

    def save_lstm_model(self, result: TrainingResult,
                        crypto: str = "all") -> Path:
        """Sauvegarde un modele LSTM PyTorch avec torch.save.

        Args:
            result: Resultat d'entrainement contenant le modele LSTM.
            crypto: Nom de la crypto associee.

        Returns:
            Chemin du repertoire de sauvegarde.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"lstm_{crypto}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modele PyTorch (state_dict pour portabilite)
        import torch
        model_path = model_dir / "model.pt"
        torch.save({
            "model_state_dict": result.model.state_dict(),
            "input_size": result.model.lstm.input_size,
            "hidden_size": result.model.hidden_size,
            "num_layers": result.model.num_layers,
        }, model_path)
        logger.info("Modele LSTM sauvegarde : %s", model_path)

        # Sauvegarder le scaler
        if result.scaler is not None:
            scaler_path = model_dir / "scaler.joblib"
            joblib.dump(result.scaler, scaler_path)

        # Metadonnees
        metadata = {
            "model_name": "lstm",
            "crypto": crypto,
            "timestamp": timestamp,
            "metrics": result.metrics,
            "feature_names": result.feature_names,
            "has_scaler": result.scaler is not None,
            "architecture": {
                "input_size": result.model.lstm.input_size,
                "hidden_size": result.model.hidden_size,
                "num_layers": result.model.num_layers,
            }
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model_dir

    def save(self, result: TrainingResult, crypto: str = "all") -> Path:
        """Sauvegarde un modele (detection automatique du type).

        Args:
            result: Resultat d'entrainement.
            crypto: Nom de la crypto.

        Returns:
            Chemin du repertoire de sauvegarde.
        """
        if result.model_name == "lstm":
            return self.save_lstm_model(result, crypto)
        return self.save_sklearn_model(result, crypto)

    def save_with_mlflow(self, result: TrainingResult,
                         crypto: str = "all",
                         experiment_name: str = "crypto_prediction") -> str:
        """Sauvegarde un modele avec tracking MLflow complet.

        Enregistre le modele, les metriques, les parametres et les artefacts
        dans MLflow pour le versioning.

        Args:
            result: Resultat d'entrainement.
            crypto: Nom de la crypto.
            experiment_name: Nom de l'experience MLflow.

        Returns:
            Run ID MLflow.
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{result.model_name}_{crypto}"):
            # Log des metriques
            for metric_name, value in result.metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log des parametres
            mlflow.log_param("model_name", result.model_name)
            mlflow.log_param("crypto", crypto)
            mlflow.log_param("n_features", len(result.feature_names))

            # Sauvegarde locale + log comme artefact
            model_dir = self.save(result, crypto)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            logger.info("MLflow run ID : %s", run_id)

        return run_id

    def load_sklearn_model(self, model_dir: Path) -> dict:
        """Charge un modele sklearn/XGBoost depuis le disque.

        Args:
            model_dir: Repertoire contenant le modele sauvegarde.

        Returns:
            Dictionnaire avec le modele, le scaler et les metadonnees.
        """
        model_dir = Path(model_dir)

        model = joblib.load(model_dir / "model.joblib")

        scaler = None
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        logger.info("Modele charge : %s (%s)", metadata["model_name"], model_dir)
        return {"model": model, "scaler": scaler, "metadata": metadata}

    def load_lstm_model(self, model_dir: Path) -> dict:
        """Charge un modele LSTM PyTorch depuis le disque.

        Args:
            model_dir: Repertoire contenant le modele sauvegarde.

        Returns:
            Dictionnaire avec le modele, le scaler et les metadonnees.
        """
        import torch
        from src.models.trainer import LSTMModel

        model_dir = Path(model_dir)
        checkpoint = torch.load(model_dir / "model.pt", weights_only=True)

        model = LSTMModel(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        scaler = None
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        logger.info("Modele LSTM charge : %s", model_dir)
        return {"model": model, "scaler": scaler, "metadata": metadata}

    def load(self, model_dir: Path) -> dict:
        """Charge un modele (detection automatique du type).

        Args:
            model_dir: Repertoire contenant le modele.

        Returns:
            Dictionnaire avec le modele, le scaler et les metadonnees.
        """
        model_dir = Path(model_dir)
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        if metadata["model_name"] == "lstm":
            return self.load_lstm_model(model_dir)
        return self.load_sklearn_model(model_dir)

    def get_latest_model(self, model_name: str = None,
                         crypto: str = None) -> Path | None:
        """Retrouve le dernier modele sauvegarde correspondant aux criteres.

        Args:
            model_name: Filtrer par nom de modele (ex: "xgboost").
            crypto: Filtrer par crypto (ex: "bitcoin").

        Returns:
            Chemin du repertoire du dernier modele, ou None.
        """
        candidates = []
        for d in self.models_dir.iterdir():
            if not d.is_dir():
                continue
            metadata_path = d / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path) as f:
                meta = json.load(f)

            if model_name and meta.get("model_name") != model_name:
                continue
            if crypto and meta.get("crypto") != crypto:
                continue

            candidates.append((d, meta.get("timestamp", "")))

        if not candidates:
            return None

        # Trier par timestamp decroissant
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def list_models(self) -> list[dict]:
        """Liste tous les modeles sauvegardes.

        Returns:
            Liste de dictionnaires avec les metadonnees de chaque modele.
        """
        models = []
        for d in sorted(self.models_dir.iterdir()):
            if not d.is_dir():
                continue
            metadata_path = d / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path) as f:
                meta = json.load(f)
            meta["path"] = str(d)
            models.append(meta)

        return models
