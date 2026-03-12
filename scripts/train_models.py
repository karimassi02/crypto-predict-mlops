"""Script principal d'entrainement des modeles de prediction crypto.

Execute le pipeline complet :
1. Chargement des donnees transformees
2. Feature engineering (indicateurs techniques, features temporelles, lags)
3. Feature selection (correlation, SelectKBest, RFE, RF importance)
4. Entrainement des modeles (LogReg, RF, GB, XGBoost, LSTM)
5. Optimisation des hyperparametres (GridSearch + Optuna)
6. Re-entrainement avec les meilleurs parametres
7. Comparaison finale et sauvegarde des resultats

Usage :
    python scripts/train_models.py
    python scripts/train_models.py --crypto bitcoin
    python scripts/train_models.py --skip-optimization
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ajouter le repertoire racine au path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector
from src.models.trainer import ModelTrainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(crypto: str = None) -> pd.DataFrame:
    """Charge les donnees transformees.

    Args:
        crypto: Nom de la crypto a charger (defaut: toutes).

    Returns:
        DataFrame des donnees.
    """
    data_path = ROOT_DIR / "data" / "processed" / "all_cryptos_processed.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    if crypto:
        df = df[df["coingecko_id"] == crypto].reset_index(drop=True)
        logger.info("Donnees chargees pour '%s' : %d lignes", crypto, len(df))
    else:
        logger.info("Donnees chargees (toutes cryptos) : %d lignes", len(df))

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Applique le feature engineering et la selection de variables.

    Args:
        df: DataFrame brut.

    Returns:
        Tuple (DataFrame avec features, liste des features selectionnees).
    """
    # Feature engineering
    engineer = FeatureEngineer(target_horizon=1)
    df_features = engineer.build_features_all_cryptos(df)

    # Supprimer les lignes avec NaN dans la cible
    df_features = df_features.dropna(subset=["target"]).reset_index(drop=True)

    # Colonnes a exclure (identifiants, dates, cible)
    exclude_cols = [
        "coingecko_id", "symbol", "date", "crypto_id",
        "target", "is_outlier", "fg_zone"
    ]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    # Remplacer inf par NaN puis imputer
    X = df_features[feature_cols].replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median())
    y = df_features["target"]

    logger.info("Features disponibles : %d", len(feature_cols))

    # Feature selection
    selector = FeatureSelector(n_features=20, random_state=42)
    selected_features = selector.select_by_majority_vote(X, y, min_votes=2)

    # Sauvegarder le resume de la selection
    summary = selector.get_summary()
    summary_path = ROOT_DIR / "data" / "processed" / "feature_selection_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Resume de la selection sauvegarde : %s", summary_path)

    # Mettre a jour le DataFrame avec les features selectionnees
    df_features["_features_ready"] = True

    return df_features, selected_features


def train_baseline(df: pd.DataFrame, selected_features: list[str]) -> ModelTrainer:
    """Entraine tous les modeles avec les parametres par defaut.

    Args:
        df: DataFrame avec features.
        selected_features: Liste des features a utiliser.

    Returns:
        ModelTrainer avec les resultats.
    """
    X = df[selected_features].replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median())
    y = df["target"]

    trainer = ModelTrainer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = trainer.temporal_split(X, y)

    trainer.train_all(X_train, X_test, y_train, y_test)

    comparison = trainer.get_comparison()
    logger.info("\n=== Comparaison des modeles (baseline) ===")
    logger.info("\n%s", comparison.to_string(index=False))

    return trainer


def train_optimized(df: pd.DataFrame, selected_features: list[str]) -> ModelTrainer:
    """Optimise les hyperparametres puis re-entraine les modeles.

    Args:
        df: DataFrame avec features.
        selected_features: Liste des features a utiliser.

    Returns:
        ModelTrainer avec les resultats optimises.
    """
    from src.models.optimization import HyperparameterOptimizer

    X = df[selected_features].replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median())
    y = df["target"]

    trainer = ModelTrainer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = trainer.temporal_split(X, y)

    # Optimisation des hyperparametres
    optimizer = HyperparameterOptimizer(
        mlflow_tracking_uri=str(ROOT_DIR / "mlruns"),
        experiment_name="crypto_prediction"
    )
    best_params = optimizer.optimize_all(X_train, y_train)

    opt_summary = optimizer.get_summary()
    logger.info("\n=== Resume optimisation ===")
    logger.info("\n%s", opt_summary.to_string(index=False))

    # Re-entrainement avec les meilleurs parametres
    logger.info("\n=== Re-entrainement avec parametres optimises ===")

    lr_params = best_params.get("logistic_regression", {})
    trainer.train_logistic_regression(X_train, X_test, y_train, y_test, **lr_params)

    rf_params = best_params.get("random_forest", {})
    trainer.train_random_forest(X_train, X_test, y_train, y_test, **rf_params)

    xgb_params = best_params.get("xgboost", {})
    trainer.train_xgboost(X_train, X_test, y_train, y_test, **xgb_params)

    lstm_params = best_params.get("lstm", {})
    seq_length = lstm_params.pop("seq_length", 10)
    trainer.train_lstm(X_train, X_test, y_train, y_test, seq_length=seq_length,
                       **lstm_params)

    comparison = trainer.get_comparison()
    logger.info("\n=== Comparaison des modeles (optimises) ===")
    logger.info("\n%s", comparison.to_string(index=False))

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Entrainement des modeles crypto")
    parser.add_argument("--crypto", type=str, default=None,
                        help="Crypto specifique (ex: bitcoin)")
    parser.add_argument("--skip-optimization", action="store_true",
                        help="Passer l'optimisation des hyperparametres")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Pipeline ML - Crypto Predict")
    logger.info("=" * 60)

    # 1. Chargement des donnees
    df = load_data(args.crypto)

    # 2-3. Feature engineering + selection
    df_features, selected_features = prepare_features(df)

    # 4. Entrainement baseline
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 : Entrainement baseline")
    logger.info("=" * 60)
    baseline_trainer = train_baseline(df_features, selected_features)

    # 5. Optimisation + re-entrainement
    if not args.skip_optimization:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2 : Optimisation + Re-entrainement")
        logger.info("=" * 60)
        optimized_trainer = train_optimized(df_features, selected_features)

        # Comparaison baseline vs optimise
        logger.info("\n" + "=" * 60)
        logger.info("COMPARAISON FINALE : Baseline vs Optimise")
        logger.info("=" * 60)

        best_baseline = baseline_trainer.get_best_model("f1")
        best_optimized = optimized_trainer.get_best_model("f1")

        logger.info("Meilleur baseline : %s (F1=%.4f)",
                     best_baseline.model_name,
                     best_baseline.metrics.get("f1", 0))
        logger.info("Meilleur optimise : %s (F1=%.4f)",
                     best_optimized.model_name,
                     best_optimized.metrics.get("f1", 0))

        # Sauvegarder comparaison
        comparison_path = ROOT_DIR / "data" / "processed" / "model_comparison.csv"
        optimized_trainer.get_comparison().to_csv(comparison_path, index=False)
        logger.info("Comparaison sauvegardee : %s", comparison_path)

    # 6. Sauvegarde de tous les modeles
    logger.info("\n" + "=" * 60)
    logger.info("SAUVEGARDE DES MODELES")
    logger.info("=" * 60)

    from src.models.model_registry import ModelRegistry
    registry = ModelRegistry()

    final_trainer = optimized_trainer if not args.skip_optimization else baseline_trainer
    crypto_name = args.crypto or "all"

    for name, result in final_trainer.results.items():
        if result.model is not None:
            model_dir = registry.save(result, crypto=crypto_name)
            logger.info("Sauvegarde %s -> %s", name, model_dir)

    # Sauvegarder aussi avec MLflow
    best = final_trainer.get_best_model("f1")
    run_id = registry.save_with_mlflow(best, crypto=crypto_name)
    logger.info("Meilleur modele enregistre dans MLflow (run_id=%s)", run_id)

    logger.info("\nPipeline ML termine avec succes !")


if __name__ == "__main__":
    main()
