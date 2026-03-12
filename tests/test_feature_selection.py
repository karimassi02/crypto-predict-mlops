"""Tests unitaires pour le module de feature selection."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection import FeatureSelector, SelectionResult


@pytest.fixture
def classification_data():
    """Genere des donnees de classification synthetiques."""
    np.random.seed(42)
    n = 200

    # Features informatives (correlees avec la cible)
    informative_1 = np.random.randn(n)
    informative_2 = np.random.randn(n)
    target = (informative_1 + informative_2 > 0).astype(int)

    # Features redondantes (tres correlees entre elles)
    redundant = informative_1 * 0.95 + np.random.randn(n) * 0.05

    # Features bruitees (non informatives)
    noise_1 = np.random.randn(n)
    noise_2 = np.random.randn(n)
    noise_3 = np.random.randn(n)

    X = pd.DataFrame({
        "informative_1": informative_1,
        "informative_2": informative_2,
        "redundant": redundant,
        "noise_1": noise_1,
        "noise_2": noise_2,
        "noise_3": noise_3,
    })
    y = pd.Series(target, name="target")

    return X, y


@pytest.fixture
def selector():
    return FeatureSelector(n_features=3, random_state=42)


class TestFeatureSelector:
    """Tests pour la classe FeatureSelector."""

    def test_remove_correlated(self, selector, classification_data):
        """Verifie la suppression des features correlees."""
        X, _ = classification_data
        result = selector.remove_correlated(X, threshold=0.90)

        assert isinstance(result, SelectionResult)
        assert result.method == "correlation"
        # La feature redondante doit etre supprimee (correlees a >0.90)
        # On ne peut pas garder les deux : informative_1 et redundant
        assert not ("informative_1" in result.selected_features
                     and "redundant" in result.selected_features)

    def test_select_kbest(self, selector, classification_data):
        """Verifie SelectKBest avec f_classif."""
        X, y = classification_data
        result = selector.select_kbest(X, y)

        assert result.method == "select_kbest"
        assert result.n_features == 3
        assert len(result.selected_features) == 3
        # Les features informatives doivent avoir de meilleurs scores
        assert result.scores["informative_1"] > result.scores["noise_1"]

    def test_select_rfe(self, selector, classification_data):
        """Verifie RFE avec Random Forest."""
        X, y = classification_data
        result = selector.select_rfe(X, y)

        assert result.method == "rfe"
        assert result.n_features == 3
        assert len(result.selected_features) == 3

    def test_select_by_importance(self, selector, classification_data):
        """Verifie la selection par importance RF."""
        X, y = classification_data
        result = selector.select_by_importance(X, y)

        assert result.method == "rf_importance"
        assert result.n_features == 3
        # Les features informatives doivent avoir une haute importance
        top_features = result.selected_features[:2]
        informative_in_top = sum(
            1 for f in top_features if "informative" in f or "redundant" in f
        )
        assert informative_in_top >= 1, "Au moins 1 feature informative dans le top 2"

    def test_majority_vote(self, selector, classification_data):
        """Verifie le vote majoritaire."""
        X, y = classification_data
        selected = selector.select_by_majority_vote(X, y, min_votes=2)

        assert isinstance(selected, list)
        assert len(selected) > 0
        # Les 4 methodes doivent avoir ete executees
        assert len(selector.results) == 4

    def test_get_summary(self, selector, classification_data):
        """Verifie le resume comparatif."""
        X, y = classification_data
        selector.select_by_majority_vote(X, y, min_votes=2)

        summary = selector.get_summary()
        assert isinstance(summary, pd.DataFrame)
        assert "total_votes" in summary.columns
        assert len(summary) > 0

    def test_n_features_larger_than_available(self, classification_data):
        """Verifie le cas ou n_features > nombre de colonnes."""
        X, y = classification_data
        selector = FeatureSelector(n_features=100, random_state=42)
        result = selector.select_kbest(X, y)

        # Doit selectionner toutes les features disponibles
        assert result.n_features == X.shape[1]
