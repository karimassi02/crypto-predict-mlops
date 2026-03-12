"""Script de verification du data drift.

Compare les donnees d'entrainement (reference) avec les donnees recentes
pour detecter si les distributions ont change, ce qui pourrait degrader
les performances du modele.

Usage :
    python scripts/check_drift.py
    python scripts/check_drift.py --crypto bitcoin
    python scripts/check_drift.py --generate-report

Competence RNCP C5.3.3 : Superviser le systeme ML (monitoring, drift).
Competence RNCP C5.3.4 : Automatiser le cycle de vie ML.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.feature_engineering import FeatureEngineer
from src.monitoring.drift_detector import DriftDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Detection de data drift")
    parser.add_argument("--crypto", type=str, default=None,
                        help="Crypto specifique (ex: bitcoin)")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generer un rapport HTML detaille")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Ratio pour separer reference/courant (defaut: 0.8)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Detection de Data Drift")
    logger.info("=" * 60)

    # Charger les donnees
    data_path = ROOT_DIR / "data" / "processed" / "all_cryptos_processed.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    if args.crypto:
        df = df[df["coingecko_id"] == args.crypto].reset_index(drop=True)
        logger.info("Crypto : %s (%d lignes)", args.crypto, len(df))
    else:
        logger.info("Toutes les cryptos (%d lignes)", len(df))

    # Feature engineering
    engineer = FeatureEngineer(target_horizon=1)
    df_features = engineer.build_features_all_cryptos(df)

    # Colonnes de features (exclure identifiants et cible)
    exclude_cols = [
        "coingecko_id", "symbol", "date", "crypto_id",
        "target", "is_outlier", "fg_zone"
    ]
    feature_cols = [c for c in df_features.columns
                    if c not in exclude_cols
                    and df_features[c].dtype in ["float64", "float32", "int64", "int32"]]

    # Split temporel : reference (train) vs courant (recent)
    split_idx = int(len(df_features) * args.split_ratio)
    reference_data = df_features.iloc[:split_idx]
    current_data = df_features.iloc[split_idx:]

    logger.info("Reference : %d lignes (anciennes)", len(reference_data))
    logger.info("Courant : %d lignes (recentes)", len(current_data))
    logger.info("Features analysees : %d", len(feature_cols))

    # Detection de drift
    detector = DriftDetector(drift_threshold=0.5)
    results = detector.check_and_alert(reference_data, current_data, feature_cols)

    # Affichage des resultats
    logger.info("\n" + "=" * 60)
    logger.info("RESULTATS")
    logger.info("=" * 60)
    logger.info("Drift dataset : %s", results["dataset_drift"])
    logger.info("Features en drift : %d/%d (%.0f%%)",
                results["n_drifted_features"],
                results["n_total_features"],
                results["drift_ratio"] * 100)
    logger.info("Alerte : %s", "OUI" if results["alert"] else "NON")

    if results["drifted_features"]:
        logger.info("\nFeatures en drift :")
        for feat in results["drifted_features"]:
            detail = results["feature_details"].get(feat, {})
            logger.info("  - %s (p-value=%.4f, test=%s)",
                        feat,
                        detail.get("p_value", 0),
                        detail.get("stat_test", "?"))

    # Generer le rapport HTML
    if args.generate_report:
        logger.info("\nGeneration du rapport HTML...")
        crypto_label = args.crypto or "all"
        report_path = detector.generate_report(
            reference_data, current_data, feature_cols,
            report_name=f"drift_{crypto_label}"
        )
        logger.info("Rapport disponible : %s", report_path)

    logger.info("\nVerification du drift terminee.")


if __name__ == "__main__":
    main()
