"""Module d'optimisation des hyperparametres des modeles ML.

Implemente deux strategies d'optimisation :
- GridSearchCV (scikit-learn) : recherche exhaustive pour modeles classiques
- Optuna : optimisation bayesienne pour XGBoost et LSTM

Tracking des experiences avec MLflow.

Competence RNCP C5.2.4 : Optimiser la performance des modeles en modifiant
les hyperparametres et en analysant les predictions.
"""

import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.trainer import LSTMModel

logger = logging.getLogger(__name__)

# Desactiver les logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Optimise les hyperparametres des modeles de prediction.

    Combine GridSearchCV pour les modeles classiques et Optuna pour XGBoost/LSTM,
    avec tracking MLflow de toutes les experiences.
    """

    def __init__(self, random_state: int = 42,
                 mlflow_tracking_uri: str = "mlruns",
                 experiment_name: str = "crypto_prediction"):
        """Initialise l'optimiseur.

        Args:
            random_state: Graine aleatoire.
            mlflow_tracking_uri: URI du serveur MLflow.
            experiment_name: Nom de l'experience MLflow.
        """
        self.random_state = random_state
        self.best_params: dict[str, dict] = {}
        self.best_scores: dict[str, float] = {}

        # Configuration MLflow (utiliser file:// pour compatibilite Windows)
        tracking_path = Path(mlflow_tracking_uri).resolve()
        mlflow.set_tracking_uri(tracking_path.as_uri())
        mlflow.set_experiment(experiment_name)

    def _get_cv(self, n_splits: int = 5) -> TimeSeriesSplit:
        """Retourne un splitter temporel pour la validation croisee.

        Args:
            n_splits: Nombre de folds.

        Returns:
            TimeSeriesSplit configure.
        """
        return TimeSeriesSplit(n_splits=n_splits)

    def optimize_logistic_regression(self, X_train, y_train,
                                     n_splits: int = 5) -> dict:
        """Optimise la Regression Logistique par GridSearch + TimeSeriesSplit.

        Args:
            X_train: Features d'entrainement.
            y_train: Cible d'entrainement.
            n_splits: Nombre de folds pour la validation croisee.

        Returns:
            Meilleurs hyperparametres trouves.
        """
        logger.info("Optimisation : Logistic Regression (GridSearch)")
        cv = self._get_cv(n_splits)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        best_score = -1
        best_params = {}

        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
        }

        for C in param_grid["C"]:
            for penalty in param_grid["penalty"]:
                model = LogisticRegression(
                    C=C, penalty=penalty, solver="saga",
                    max_iter=1000, random_state=self.random_state
                )
                scores = cross_val_score(model, X_scaled, y_train,
                                         cv=cv, scoring="f1", n_jobs=-1)
                mean_score = scores.mean()

                with mlflow.start_run(nested=True):
                    mlflow.log_params({"model": "logistic_regression",
                                       "C": C, "penalty": penalty})
                    mlflow.log_metric("f1_cv_mean", mean_score)
                    mlflow.log_metric("f1_cv_std", scores.std())

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"C": C, "penalty": penalty, "solver": "saga",
                                   "max_iter": 1000}

        self.best_params["logistic_regression"] = best_params
        self.best_scores["logistic_regression"] = best_score
        logger.info("Meilleurs params LR : %s (F1=%.4f)", best_params, best_score)
        return best_params

    def optimize_random_forest(self, X_train, y_train,
                               n_splits: int = 5) -> dict:
        """Optimise Random Forest par GridSearch + TimeSeriesSplit.

        Args:
            X_train: Features d'entrainement.
            y_train: Cible d'entrainement.
            n_splits: Nombre de folds.

        Returns:
            Meilleurs hyperparametres trouves.
        """
        logger.info("Optimisation : Random Forest (GridSearch)")
        cv = self._get_cv(n_splits)

        best_score = -1
        best_params = {}

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        }

        for n_est in param_grid["n_estimators"]:
            for depth in param_grid["max_depth"]:
                for min_split in param_grid["min_samples_split"]:
                    model = RandomForestClassifier(
                        n_estimators=n_est, max_depth=depth,
                        min_samples_split=min_split,
                        random_state=self.random_state, n_jobs=-1
                    )
                    scores = cross_val_score(model, X_train, y_train,
                                             cv=cv, scoring="f1", n_jobs=-1)
                    mean_score = scores.mean()

                    with mlflow.start_run(nested=True):
                        mlflow.log_params({
                            "model": "random_forest",
                            "n_estimators": n_est,
                            "max_depth": str(depth),
                            "min_samples_split": min_split
                        })
                        mlflow.log_metric("f1_cv_mean", mean_score)

                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {"n_estimators": n_est, "max_depth": depth,
                                       "min_samples_split": min_split}

        self.best_params["random_forest"] = best_params
        self.best_scores["random_forest"] = best_score
        logger.info("Meilleurs params RF : %s (F1=%.4f)", best_params, best_score)
        return best_params

    def optimize_xgboost(self, X_train, y_train,
                         n_trials: int = 50, n_splits: int = 5) -> dict:
        """Optimise XGBoost par Optuna (optimisation bayesienne).

        Args:
            X_train: Features d'entrainement.
            y_train: Cible d'entrainement.
            n_trials: Nombre d'essais Optuna.
            n_splits: Nombre de folds.

        Returns:
            Meilleurs hyperparametres trouves.
        """
        logger.info("Optimisation : XGBoost (Optuna, %d trials)", n_trials)
        cv = self._get_cv(n_splits)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }

            model = XGBClassifier(
                **params, random_state=self.random_state,
                eval_metric="logloss"
            )
            scores = cross_val_score(model, X_train, y_train,
                                     cv=cv, scoring="f1", n_jobs=-1)

            with mlflow.start_run(nested=True):
                mlflow.log_params({"model": "xgboost", **params})
                mlflow.log_metric("f1_cv_mean", scores.mean())

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        self.best_params["xgboost"] = best_params
        self.best_scores["xgboost"] = study.best_value
        logger.info("Meilleurs params XGBoost : %s (F1=%.4f)",
                     best_params, study.best_value)
        return best_params

    def optimize_lstm(self, X_train, y_train,
                      n_trials: int = 20, seq_length: int = 10) -> dict:
        """Optimise le LSTM par Optuna.

        Args:
            X_train: Features d'entrainement.
            y_train: Cible d'entrainement.
            n_trials: Nombre d'essais Optuna.
            seq_length: Longueur des sequences.

        Returns:
            Meilleurs hyperparametres trouves.
        """
        logger.info("Optimisation : LSTM (Optuna, %d trials)", n_trials)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        y_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)

        # Split validation temporel (80/20 du train)
        split = int(len(X_scaled) * 0.8)
        X_tr, X_val = X_scaled[:split], X_scaled[split:]
        y_tr, y_val = y_np[:split], y_np[split:]

        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            # Preparer sequences
            X_tr_seq, y_tr_seq = [], []
            for i in range(seq_length, len(X_tr)):
                X_tr_seq.append(X_tr[i - seq_length:i])
                y_tr_seq.append(y_tr[i])
            X_tr_t = torch.FloatTensor(np.array(X_tr_seq))
            y_tr_t = torch.FloatTensor(np.array(y_tr_seq)).unsqueeze(1)

            X_val_seq, y_val_seq = [], []
            for i in range(seq_length, len(X_val)):
                X_val_seq.append(X_val[i - seq_length:i])
                y_val_seq.append(y_val[i])
            X_val_t = torch.FloatTensor(np.array(X_val_seq))
            y_val_np = np.array(y_val_seq)

            if len(X_tr_t) == 0 or len(X_val_t) == 0:
                return 0.0

            model = LSTMModel(
                input_size=X_tr.shape[1], hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout
            ).to(device)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )

            # Entrainement rapide (30 epochs pour l'optimisation)
            model.train()
            for _ in range(30):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                y_proba = model(X_val_t.to(device)).cpu().numpy().flatten()
                y_pred = (y_proba > 0.5).astype(int)

            score = f1_score(y_val_np.astype(int), y_pred, zero_division=0)

            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    "model": "lstm", "hidden_size": hidden_size,
                    "num_layers": num_layers, "dropout": dropout,
                    "learning_rate": lr, "batch_size": batch_size
                })
                mlflow.log_metric("f1_val", score)

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params["seq_length"] = seq_length
        self.best_params["lstm"] = best_params
        self.best_scores["lstm"] = study.best_value
        logger.info("Meilleurs params LSTM : %s (F1=%.4f)",
                     best_params, study.best_value)
        return best_params

    def optimize_all(self, X_train, y_train) -> dict[str, dict]:
        """Execute l'optimisation pour tous les modeles.

        Args:
            X_train: Features d'entrainement.
            y_train: Cible d'entrainement.

        Returns:
            Dictionnaire model_name -> meilleurs hyperparametres.
        """
        logger.info("=" * 60)
        logger.info("Optimisation de tous les modeles")
        logger.info("=" * 60)

        with mlflow.start_run(run_name="hyperparameter_optimization"):
            self.optimize_logistic_regression(X_train, y_train)
            self.optimize_random_forest(X_train, y_train)
            self.optimize_xgboost(X_train, y_train, n_trials=50)
            self.optimize_lstm(X_train, y_train, n_trials=20)

            # Log des meilleurs scores
            for model_name, score in self.best_scores.items():
                mlflow.log_metric(f"best_f1_{model_name}", score)

        return self.best_params

    def get_summary(self) -> pd.DataFrame:
        """Resume les resultats d'optimisation.

        Returns:
            DataFrame comparatif des meilleurs scores par modele.
        """
        rows = []
        for model_name in self.best_params:
            rows.append({
                "model": model_name,
                "best_f1_cv": self.best_scores.get(model_name, 0),
                "best_params": str(self.best_params[model_name])
            })

        return pd.DataFrame(rows).sort_values("best_f1_cv", ascending=False)
