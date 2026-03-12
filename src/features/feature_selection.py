"""Module de selection de variables pour optimiser la performance du modele.

Implemente plusieurs methodes de selection :
- Analyse de correlation (suppression des features redondantes)
- SelectKBest avec test statistique (f_classif)
- RFE (Recursive Feature Elimination)
- Importance par modele (Random Forest - methode incorporee)

Competence RNCP C5.2.2 : Selectionner les variables en identifiant les
differentes methodes possibles afin d'optimiser la performance du modele.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Resultat d'une methode de selection de variables.

    Attributes:
        method: Nom de la methode utilisee.
        selected_features: Liste des features selectionnees.
        scores: Dictionnaire feature -> score (interpretation depend de la methode).
        n_features: Nombre de features selectionnees.
    """
    method: str
    selected_features: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    n_features: int = 0

    def __post_init__(self):
        self.n_features = len(self.selected_features)


class FeatureSelector:
    """Selectionne les meilleures variables pour le modele de prediction.

    Combine plusieurs methodes et retourne les features les plus pertinentes
    par vote majoritaire.
    """

    def __init__(self, n_features: int = 20, random_state: int = 42):
        """Initialise le selecteur de features.

        Args:
            n_features: Nombre de features a selectionner par methode.
            random_state: Graine aleatoire pour la reproductibilite.
        """
        self.n_features = n_features
        self.random_state = random_state
        self.results: dict[str, SelectionResult] = {}

    def remove_correlated(self, X: pd.DataFrame,
                          threshold: float = 0.90) -> SelectionResult:
        """Supprime les features fortement correlees entre elles.

        Pour chaque paire de features avec |correlation| > threshold,
        la feature ayant la plus faible correlation moyenne avec la cible
        est supprimee.

        Args:
            X: DataFrame des features.
            threshold: Seuil de correlation (defaut: 0.90).

        Returns:
            SelectionResult avec les features non-redondantes.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for column in upper.columns:
            correlated = upper.index[upper[column] > threshold].tolist()
            if correlated:
                # Garder la feature avec la plus grande variance (plus informative)
                variances = X[[column] + correlated].var()
                worst = variances.idxmin()
                to_drop.add(worst)

        selected = [col for col in X.columns if col not in to_drop]
        scores = {col: 1.0 - corr_matrix[col].mean() for col in X.columns}

        result = SelectionResult(
            method="correlation",
            selected_features=selected,
            scores=scores
        )
        self.results["correlation"] = result

        logger.info(
            "Correlation (seuil=%.2f) : %d/%d features retenues (%d supprimees)",
            threshold, len(selected), len(X.columns), len(to_drop)
        )
        return result

    def select_kbest(self, X: pd.DataFrame,
                     y: pd.Series) -> SelectionResult:
        """Selectionne les K meilleures features par test statistique F (ANOVA).

        Utilise f_classif qui mesure la relation lineaire entre chaque feature
        et la variable cible categorielle.

        Args:
            X: DataFrame des features.
            y: Series de la variable cible.

        Returns:
            SelectionResult avec les K meilleures features.
        """
        k = min(self.n_features, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)

        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        scores = dict(zip(X.columns, selector.scores_))

        result = SelectionResult(
            method="select_kbest",
            selected_features=selected,
            scores=scores
        )
        self.results["select_kbest"] = result

        logger.info("SelectKBest (k=%d) : features retenues", k)
        for feat in sorted(selected, key=lambda f: scores[f], reverse=True)[:5]:
            logger.info("  - %s (score=%.2f)", feat, scores[feat])

        return result

    def select_rfe(self, X: pd.DataFrame,
                   y: pd.Series) -> SelectionResult:
        """Selection par elimination recursive (RFE) avec Random Forest.

        Elimine iterativement les features les moins importantes selon
        le modele, jusqu'a atteindre le nombre souhaite.

        Args:
            X: DataFrame des features.
            y: Series de la variable cible.

        Returns:
            SelectionResult avec les features selectionnees par RFE.
        """
        n = min(self.n_features, X.shape[1])
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        rfe = RFE(estimator=estimator, n_features_to_select=n, step=5)
        rfe.fit(X, y)

        mask = rfe.support_
        selected = X.columns[mask].tolist()
        # Le ranking : 1 = selectionne, >1 = ordre d'elimination
        scores = dict(zip(X.columns, 1.0 / rfe.ranking_))

        result = SelectionResult(
            method="rfe",
            selected_features=selected,
            scores=scores
        )
        self.results["rfe"] = result

        logger.info("RFE (n=%d) : features retenues", n)
        return result

    def select_by_importance(self, X: pd.DataFrame,
                             y: pd.Series) -> SelectionResult:
        """Selection par importance des features d'un Random Forest (methode incorporee).

        Entraine un Random Forest et selectionne les features ayant la plus
        grande importance (Mean Decrease Impurity).

        Args:
            X: DataFrame des features.
            y: Series de la variable cible.

        Returns:
            SelectionResult avec les features les plus importantes.
        """
        rf = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        n = min(self.n_features, len(importances))
        selected = importances.head(n).index.tolist()
        scores = importances.to_dict()

        result = SelectionResult(
            method="rf_importance",
            selected_features=selected,
            scores=scores
        )
        self.results["rf_importance"] = result

        logger.info("RF Importance (top %d) :", n)
        for feat in selected[:5]:
            logger.info("  - %s (importance=%.4f)", feat, scores[feat])

        return result

    def select_by_majority_vote(self, X: pd.DataFrame,
                                y: pd.Series,
                                min_votes: int = 2) -> list[str]:
        """Selectionne les features par vote majoritaire des 4 methodes.

        Execute toutes les methodes et ne retient que les features
        selectionnees par au moins `min_votes` methodes.

        Args:
            X: DataFrame des features.
            y: Series de la variable cible.
            min_votes: Nombre minimum de methodes qui doivent selectionner la feature.

        Returns:
            Liste des features selectionnees par vote majoritaire.
        """
        logger.info("Selection par vote majoritaire (min_votes=%d)", min_votes)

        # Executer les 4 methodes
        self.remove_correlated(X)
        self.select_kbest(X, y)
        self.select_rfe(X, y)
        self.select_by_importance(X, y)

        # Compter les votes
        vote_counts = {}
        for result in self.results.values():
            for feat in result.selected_features:
                vote_counts[feat] = vote_counts.get(feat, 0) + 1

        # Selectionner par vote majoritaire
        selected = [feat for feat, votes in vote_counts.items() if votes >= min_votes]

        # Trier par nombre de votes decroissant, puis par score RF importance
        rf_scores = self.results.get("rf_importance", SelectionResult("", [])).scores
        selected.sort(key=lambda f: (-vote_counts[f], -rf_scores.get(f, 0)))

        logger.info(
            "Vote majoritaire : %d features retenues sur %d",
            len(selected), len(X.columns)
        )
        for feat in selected[:10]:
            logger.info("  - %s (%d votes)", feat, vote_counts[feat])

        return selected

    def get_summary(self) -> pd.DataFrame:
        """Retourne un resume comparatif de toutes les methodes executees.

        Returns:
            DataFrame avec une ligne par feature et une colonne par methode
            indiquant si la feature est selectionnee (1) ou non (0).
        """
        if not self.results:
            return pd.DataFrame()

        all_features = set()
        for result in self.results.values():
            all_features.update(result.selected_features)
            all_features.update(result.scores.keys())

        summary = pd.DataFrame(index=sorted(all_features))
        for method, result in self.results.items():
            summary[f"{method}_selected"] = summary.index.isin(
                result.selected_features
            ).astype(int)
            summary[f"{method}_score"] = summary.index.map(
                lambda f, s=result.scores: s.get(f, 0)
            )

        summary["total_votes"] = sum(
            summary[f"{m}_selected"] for m in self.results
        )
        summary = summary.sort_values("total_votes", ascending=False)

        return summary
