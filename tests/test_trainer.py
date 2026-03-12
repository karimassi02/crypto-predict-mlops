"""Tests unitaires pour le module d'entrainement des modeles."""

import numpy as np
import pandas as pd
import pytest

from src.models.trainer import LSTMModel, ModelTrainer, TrainingResult


@pytest.fixture
def train_test_data():
    """Genere des donnees train/test synthetiques."""
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features = 10

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_train), name="target")
    y_test = pd.Series(np.random.randint(0, 2, n_test), name="target")

    return X_train, X_test, y_train, y_test


@pytest.fixture
def trainer():
    return ModelTrainer(test_size=0.2, random_state=42)


class TestModelTrainer:
    """Tests pour la classe ModelTrainer."""

    def test_temporal_split(self, trainer):
        """Verifie le split temporel (pas de shuffle)."""
        X = pd.DataFrame({"a": range(100), "b": range(100)})
        y = pd.Series(range(100))

        X_train, X_test, y_train, y_test = trainer.temporal_split(X, y)

        assert len(X_train) == 80
        assert len(X_test) == 20
        # Verifier que l'ordre chronologique est respecte
        assert X_train.iloc[-1]["a"] < X_test.iloc[0]["a"]

    def test_train_logistic_regression(self, trainer, train_test_data):
        """Verifie l'entrainement de la Regression Logistique."""
        X_train, X_test, y_train, y_test = train_test_data
        result = trainer.train_logistic_regression(X_train, X_test, y_train, y_test)

        assert isinstance(result, TrainingResult)
        assert result.model_name == "logistic_regression"
        assert result.model is not None
        assert result.scaler is not None
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert "roc_auc" in result.metrics
        assert 0 <= result.metrics["accuracy"] <= 1

    def test_train_random_forest(self, trainer, train_test_data):
        """Verifie l'entrainement du Random Forest."""
        X_train, X_test, y_train, y_test = train_test_data
        result = trainer.train_random_forest(X_train, X_test, y_train, y_test)

        assert result.model_name == "random_forest"
        assert result.model is not None
        assert "f1" in result.metrics

    def test_train_gradient_boosting(self, trainer, train_test_data):
        """Verifie l'entrainement du Gradient Boosting."""
        X_train, X_test, y_train, y_test = train_test_data
        result = trainer.train_gradient_boosting(X_train, X_test, y_train, y_test)

        assert result.model_name == "gradient_boosting"
        assert result.model is not None

    def test_train_xgboost(self, trainer, train_test_data):
        """Verifie l'entrainement de XGBoost."""
        X_train, X_test, y_train, y_test = train_test_data
        result = trainer.train_xgboost(X_train, X_test, y_train, y_test)

        assert result.model_name == "xgboost"
        assert result.model is not None
        assert "roc_auc" in result.metrics

    def test_train_lstm(self, trainer, train_test_data):
        """Verifie l'entrainement du LSTM."""
        X_train, X_test, y_train, y_test = train_test_data
        result = trainer.train_lstm(
            X_train, X_test, y_train, y_test,
            seq_length=5, epochs=5, hidden_size=16, num_layers=1
        )

        assert result.model_name == "lstm"
        assert result.model is not None
        assert "accuracy" in result.metrics

    def test_train_all(self, trainer, train_test_data):
        """Verifie l'entrainement de tous les modeles."""
        X_train, X_test, y_train, y_test = train_test_data
        results = trainer.train_all(X_train, X_test, y_train, y_test)

        assert len(results) == 5
        assert "logistic_regression" in results
        assert "random_forest" in results
        assert "gradient_boosting" in results
        assert "xgboost" in results
        assert "lstm" in results

    def test_get_comparison(self, trainer, train_test_data):
        """Verifie la comparaison des modeles."""
        X_train, X_test, y_train, y_test = train_test_data
        trainer.train_logistic_regression(X_train, X_test, y_train, y_test)
        trainer.train_random_forest(X_train, X_test, y_train, y_test)

        comparison = trainer.get_comparison()
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "f1" in comparison.columns

    def test_get_best_model(self, trainer, train_test_data):
        """Verifie la selection du meilleur modele."""
        X_train, X_test, y_train, y_test = train_test_data
        trainer.train_logistic_regression(X_train, X_test, y_train, y_test)
        trainer.train_random_forest(X_train, X_test, y_train, y_test)

        best = trainer.get_best_model("f1")
        assert isinstance(best, TrainingResult)
        assert best.model is not None


class TestLSTMModel:
    """Tests pour le modele LSTM PyTorch."""

    def test_forward_pass(self):
        """Verifie le forward pass du LSTM."""
        import torch

        model = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
        x = torch.randn(4, 5, 10)  # batch=4, seq=5, features=10
        output = model(x)

        assert output.shape == (4, 1)
        # Sortie entre 0 et 1 (sigmoid)
        assert (output >= 0).all() and (output <= 1).all()

    def test_different_configs(self):
        """Verifie differentes configurations LSTM."""
        import torch

        for hidden, layers in [(16, 1), (64, 2), (128, 3)]:
            model = LSTMModel(input_size=5, hidden_size=hidden, num_layers=layers)
            x = torch.randn(2, 10, 5)
            output = model(x)
            assert output.shape == (2, 1)
