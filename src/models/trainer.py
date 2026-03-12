"""Module d'entrainement des modeles de prediction de tendances crypto.

Implemente plusieurs algorithmes :
- Scikit-learn : Logistic Regression, Random Forest, Gradient Boosting
- XGBoost : XGBClassifier
- PyTorch : LSTM pour les dependances temporelles

Utilise un split temporel (pas de shuffle) adapte aux series temporelles.

Competence RNCP C5.2.3 : Entrainer un modele d'apprentissage automatique
a l'aide de librairies (Scikit-learn, XGBoost, PyTorch).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Resultat de l'entrainement d'un modele.

    Attributes:
        model_name: Nom du modele.
        model: Instance du modele entraine.
        metrics: Dictionnaire des metriques de performance.
        feature_names: Liste des features utilisees.
        scaler: Scaler utilise pour la normalisation (si applicable).
    """
    model_name: str
    model: object
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None


class LSTMModel(nn.Module):
    """Reseau LSTM pour la prediction de tendance sur series temporelles.

    Architecture : LSTM multicouche -> Dropout -> Couche dense -> Sigmoid.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        """Initialise le modele LSTM.

        Args:
            input_size: Nombre de features en entree.
            hidden_size: Taille de l'etat cache LSTM.
            num_layers: Nombre de couches LSTM empilees.
            dropout: Taux de dropout entre les couches.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass du LSTM.

        Args:
            x: Tensor de shape (batch_size, seq_length, input_size).

        Returns:
            Tensor de probabilites de shape (batch_size, 1).
        """
        lstm_out, _ = self.lstm(x)
        # Prendre la sortie du dernier pas de temps
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class ModelTrainer:
    """Pipeline d'entrainement unifie pour tous les modeles.

    Gere le split temporel, la normalisation, l'entrainement et l'evaluation
    pour les modeles sklearn, XGBoost et PyTorch LSTM.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialise le trainer.

        Args:
            test_size: Proportion des donnees pour le test (split temporel).
            random_state: Graine aleatoire pour la reproductibilite.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.results: dict[str, TrainingResult] = {}

    def temporal_split(self, X: pd.DataFrame, y: pd.Series):
        """Split temporel sans shuffle pour respecter l'ordre chronologique.

        Args:
            X: DataFrame des features.
            y: Series de la variable cible.

        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        split_idx = int(len(X) * (1 - self.test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(
            "Split temporel : train=%d, test=%d (%.0f%%/%.0f%%)",
            len(X_train), len(X_test),
            100 * (1 - self.test_size), 100 * self.test_size
        )
        return X_train, X_test, y_train, y_test

    def evaluate(self, model_name: str, y_true, y_pred, y_proba=None) -> dict:
        """Calcule les metriques de performance.

        Args:
            model_name: Nom du modele (pour le log).
            y_true: Valeurs reelles.
            y_pred: Predictions.
            y_proba: Probabilites predites (pour AUC-ROC).

        Returns:
            Dictionnaire des metriques.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = 0.0

        logger.info("Resultats %s :", model_name)
        for metric, value in metrics.items():
            logger.info("  %s: %.4f", metric, value)

        return metrics

    def train_logistic_regression(self, X_train, X_test, y_train, y_test,
                                  **kwargs) -> TrainingResult:
        """Entraine un modele de Regression Logistique.

        Args:
            X_train, X_test: Features d'entrainement et test.
            y_train, y_test: Cibles d'entrainement et test.

        Returns:
            TrainingResult avec le modele entraine et ses metriques.
        """
        logger.info("Entrainement : Logistic Regression")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        params = {"max_iter": 1000, "random_state": self.random_state, **kwargs}
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = self.evaluate("Logistic Regression", y_test, y_pred, y_proba)

        result = TrainingResult(
            model_name="logistic_regression",
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns),
            scaler=scaler
        )
        self.results["logistic_regression"] = result
        return result

    def train_random_forest(self, X_train, X_test, y_train, y_test,
                            **kwargs) -> TrainingResult:
        """Entraine un modele Random Forest.

        Args:
            X_train, X_test: Features d'entrainement et test.
            y_train, y_test: Cibles d'entrainement et test.

        Returns:
            TrainingResult avec le modele entraine et ses metriques.
        """
        logger.info("Entrainement : Random Forest")
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": self.random_state,
            "n_jobs": -1,
            **kwargs
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.evaluate("Random Forest", y_test, y_pred, y_proba)

        result = TrainingResult(
            model_name="random_forest",
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns)
        )
        self.results["random_forest"] = result
        return result

    def train_gradient_boosting(self, X_train, X_test, y_train, y_test,
                                **kwargs) -> TrainingResult:
        """Entraine un modele Gradient Boosting.

        Args:
            X_train, X_test: Features d'entrainement et test.
            y_train, y_test: Cibles d'entrainement et test.

        Returns:
            TrainingResult avec le modele entraine et ses metriques.
        """
        logger.info("Entrainement : Gradient Boosting")
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": self.random_state,
            **kwargs
        }
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.evaluate("Gradient Boosting", y_test, y_pred, y_proba)

        result = TrainingResult(
            model_name="gradient_boosting",
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns)
        )
        self.results["gradient_boosting"] = result
        return result

    def train_xgboost(self, X_train, X_test, y_train, y_test,
                      **kwargs) -> TrainingResult:
        """Entraine un modele XGBoost.

        Args:
            X_train, X_test: Features d'entrainement et test.
            y_train, y_test: Cibles d'entrainement et test.

        Returns:
            TrainingResult avec le modele entraine et ses metriques.
        """
        logger.info("Entrainement : XGBoost")
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": self.random_state,
                        "eval_metric": "logloss",
            **kwargs
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.evaluate("XGBoost", y_test, y_pred, y_proba)

        result = TrainingResult(
            model_name="xgboost",
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns)
        )
        self.results["xgboost"] = result
        return result

    def _prepare_lstm_sequences(self, X: np.ndarray, y: np.ndarray,
                                seq_length: int = 10):
        """Prepare les sequences pour le LSTM.

        Args:
            X: Array des features (n_samples, n_features).
            y: Array de la cible.
            seq_length: Longueur des sequences temporelles.

        Returns:
            Tuple (X_seq, y_seq) de tensors PyTorch.
        """
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i - seq_length:i])
            y_seq.append(y[i])

        X_tensor = torch.FloatTensor(np.array(X_seq))
        y_tensor = torch.FloatTensor(np.array(y_seq)).unsqueeze(1)
        return X_tensor, y_tensor

    def train_lstm(self, X_train, X_test, y_train, y_test,
                   seq_length: int = 10, hidden_size: int = 64,
                   num_layers: int = 2, epochs: int = 50,
                   batch_size: int = 32, learning_rate: float = 0.001,
                   **kwargs) -> TrainingResult:
        """Entraine un modele LSTM (PyTorch) pour capturer les dependances temporelles.

        Args:
            X_train, X_test: Features d'entrainement et test.
            y_train, y_test: Cibles d'entrainement et test.
            seq_length: Longueur des sequences temporelles.
            hidden_size: Taille de l'etat cache LSTM.
            num_layers: Nombre de couches LSTM.
            epochs: Nombre d'epoques d'entrainement.
            batch_size: Taille des batchs.
            learning_rate: Taux d'apprentissage.

        Returns:
            TrainingResult avec le modele entraine et ses metriques.
        """
        logger.info("Entrainement : LSTM (PyTorch)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device : %s", device)

        # Normalisation
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        X_test_np = scaler.transform(X_test)

        y_train_np = y_train.values if hasattr(y_train, "values") else y_train
        y_test_np = y_test.values if hasattr(y_test, "values") else y_test

        # Preparation des sequences
        X_train_seq, y_train_seq = self._prepare_lstm_sequences(
            X_train_np, y_train_np, seq_length
        )
        X_test_seq, y_test_seq = self._prepare_lstm_sequences(
            X_test_np, y_test_np, seq_length
        )

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            logger.warning("Pas assez de donnees pour le LSTM (seq_length=%d)", seq_length)
            return TrainingResult(model_name="lstm", model=None, metrics={})

        # Modele
        input_size = X_train.shape[1]
        model = LSTMModel(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Dataset et DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_seq)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )

        # Entrainement
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test_seq.to(device)
            y_proba = model(X_test_device).cpu().numpy().flatten()
            y_pred = (y_proba > 0.5).astype(int)
            y_true = y_test_seq.numpy().flatten().astype(int)

        metrics = self.evaluate("LSTM", y_true, y_pred, y_proba)

        result = TrainingResult(
            model_name="lstm",
            model=model,
            metrics=metrics,
            feature_names=list(X_train.columns) if hasattr(X_train, "columns") else [],
            scaler=scaler
        )
        self.results["lstm"] = result
        return result

    def train_all(self, X_train, X_test, y_train, y_test) -> dict[str, TrainingResult]:
        """Entraine tous les modeles et retourne les resultats compares.

        Args:
            X_train, X_test: Features d'entrainement et test.
            y_train, y_test: Cibles d'entrainement et test.

        Returns:
            Dictionnaire model_name -> TrainingResult.
        """
        logger.info("=" * 60)
        logger.info("Entrainement de tous les modeles")
        logger.info("=" * 60)

        self.train_logistic_regression(X_train, X_test, y_train, y_test)
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, X_test, y_train, y_test)
        self.train_lstm(X_train, X_test, y_train, y_test)

        return self.results

    def get_comparison(self) -> pd.DataFrame:
        """Compare les performances de tous les modeles entraines.

        Returns:
            DataFrame avec les metriques de chaque modele.
        """
        rows = []
        for name, result in self.results.items():
            row = {"model": name, **result.metrics}
            rows.append(row)

        comparison = pd.DataFrame(rows)
        if not comparison.empty:
            comparison = comparison.sort_values("f1", ascending=False).reset_index(drop=True)

        return comparison

    def get_best_model(self, metric: str = "f1") -> TrainingResult:
        """Retourne le meilleur modele selon une metrique donnee.

        Args:
            metric: Metrique de comparaison (defaut: f1).

        Returns:
            TrainingResult du meilleur modele.
        """
        best_name = max(
            self.results,
            key=lambda name: self.results[name].metrics.get(metric, 0)
        )
        best = self.results[best_name]
        logger.info(
            "Meilleur modele (%s) : %s (%.4f)",
            metric, best.model_name, best.metrics.get(metric, 0)
        )
        return best
