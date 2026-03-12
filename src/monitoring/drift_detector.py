"""Module de detection de data drift et monitoring des modeles.

Utilise Evidently AI pour detecter les derives dans les distributions
des features et de la cible entre les donnees de reference (entrainement)
et les donnees courantes (production).

Competence RNCP C5.3.3 : Superviser le systeme Machine Learning en
selectionnant des outils de monitoring et en les exploitant afin de
detecter les derives et les bugs du modele.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class DriftDetector:
    """Detecte le data drift entre donnees de reference et donnees courantes.

    Genere des rapports de drift avec Evidently AI et remonte des alertes
    lorsque les distributions des features changent significativement.
    """

    def __init__(self, reports_dir: Path = None,
                 drift_threshold: float = 0.5):
        """Initialise le detecteur de drift.

        Args:
            reports_dir: Repertoire de sauvegarde des rapports.
            drift_threshold: Seuil de proportion de features en drift
                             pour declencher une alerte (defaut: 0.5 = 50%).
        """
        self.reports_dir = reports_dir or ROOT_DIR / "data" / "drift_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

    def detect_data_drift(self, reference_data: pd.DataFrame,
                          current_data: pd.DataFrame,
                          feature_columns: list[str] = None) -> dict:
        """Detecte le data drift entre les donnees de reference et courantes.

        Utilise le preset DataDriftPreset d'Evidently qui applique des tests
        statistiques adaptes au type de chaque feature (KS pour numeriques,
        chi2 pour categoriques).

        Args:
            reference_data: Donnees de reference (ex: donnees d'entrainement).
            current_data: Donnees courantes (ex: nouvelles donnees).
            feature_columns: Colonnes a analyser (defaut: toutes les numeriques).

        Returns:
            Dictionnaire avec les resultats de drift.
        """
        if feature_columns:
            ref = reference_data[feature_columns].copy()
            cur = current_data[feature_columns].copy()
        else:
            # Selectionner uniquement les colonnes numeriques
            ref = reference_data.select_dtypes(include=[np.number]).copy()
            cur = current_data.select_dtypes(include=[np.number]).copy()

        # S'assurer que les colonnes sont identiques
        common_cols = list(set(ref.columns) & set(cur.columns))
        ref = ref[common_cols]
        cur = cur[common_cols]

        # Nettoyer les donnees
        ref = ref.replace([np.inf, -np.inf], np.nan).fillna(ref.median())
        cur = cur.replace([np.inf, -np.inf], np.nan).fillna(cur.median())

        logger.info("Detection de drift sur %d features", len(common_cols))

        # Generer le rapport Evidently
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            Dataset.from_pandas(ref),
            Dataset.from_pandas(cur),
        )

        # Extraire les resultats
        result_dict = snapshot.dict()
        drift_results = self._parse_drift_results(result_dict, common_cols)

        return drift_results

    def _parse_drift_results(self, result_dict: dict,
                             feature_columns: list[str]) -> dict:
        """Parse les resultats du rapport Evidently.

        Args:
            result_dict: Dictionnaire brut du rapport.
            feature_columns: Colonnes analysees.

        Returns:
            Dictionnaire structure avec les resultats de drift.
        """
        metrics = result_dict.get("metrics", [])

        # Extraire les informations de drift par feature
        drifted_features = []
        feature_details = {}

        for metric in metrics:
            metric_id = metric.get("metric_id", "")
            # Le DataDriftPreset genere des metriques par colonne
            if "column_name" in metric.get("metric_fields", {}):
                col_name = metric["metric_fields"]["column_name"]
                result = metric.get("result", {})
                is_drifted = result.get("drift_detected", False)
                p_value = result.get("p_value", None)
                stat_test = result.get("stat_test_name", "unknown")

                feature_details[col_name] = {
                    "drift_detected": is_drifted,
                    "p_value": p_value,
                    "stat_test": stat_test,
                }
                if is_drifted:
                    drifted_features.append(col_name)

            # Metriques globales (dataset drift)
            elif metric_id == "evidently:metric:DataDriftTable":
                result = metric.get("result", {})
                dataset_drift = result.get("drift_detected", False)
                n_drifted = result.get("n_drifted_columns", 0)
                n_total = result.get("n_columns", len(feature_columns))
                drift_ratio = n_drifted / n_total if n_total > 0 else 0

        # Si pas de metriques globales trouvees, calculer manuellement
        if not feature_details:
            n_drifted = 0
            n_total = len(feature_columns)
            drift_ratio = 0
            dataset_drift = False

        drift_results = {
            "dataset_drift": dataset_drift if "dataset_drift" in dir() else drift_ratio > self.drift_threshold,
            "drift_ratio": drift_ratio if "drift_ratio" in dir() else len(drifted_features) / max(len(feature_columns), 1),
            "n_drifted_features": n_drifted if "n_drifted" in dir() else len(drifted_features),
            "n_total_features": n_total if "n_total" in dir() else len(feature_columns),
            "drifted_features": drifted_features,
            "feature_details": feature_details,
            "alert": drift_ratio > self.drift_threshold if "drift_ratio" in dir() else len(drifted_features) / max(len(feature_columns), 1) > self.drift_threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return drift_results

    def generate_report(self, reference_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        feature_columns: list[str] = None,
                        report_name: str = None) -> Path:
        """Genere et sauvegarde un rapport HTML de drift.

        Args:
            reference_data: Donnees de reference.
            current_data: Donnees courantes.
            feature_columns: Colonnes a analyser.
            report_name: Nom du rapport (defaut: auto avec timestamp).

        Returns:
            Chemin du fichier HTML genere.
        """
        if feature_columns:
            ref = reference_data[feature_columns].copy()
            cur = current_data[feature_columns].copy()
        else:
            ref = reference_data.select_dtypes(include=[np.number]).copy()
            cur = current_data.select_dtypes(include=[np.number]).copy()

        common_cols = list(set(ref.columns) & set(cur.columns))
        ref = ref[common_cols].replace([np.inf, -np.inf], np.nan).fillna(ref[common_cols].median())
        cur = cur[common_cols].replace([np.inf, -np.inf], np.nan).fillna(cur[common_cols].median())

        report = Report([DataDriftPreset()])
        snapshot = report.run(
            Dataset.from_pandas(ref),
            Dataset.from_pandas(cur),
        )

        # Sauvegarder le rapport HTML
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = report_name or f"drift_report_{timestamp}"
        html_path = self.reports_dir / f"{name}.html"
        snapshot.save_html(str(html_path))

        logger.info("Rapport de drift sauvegarde : %s", html_path)

        # Sauvegarder aussi le JSON
        json_path = self.reports_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(snapshot.dict(), f, indent=2, default=str)

        return html_path

    def check_and_alert(self, reference_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        feature_columns: list[str] = None) -> dict:
        """Verifie le drift et remonte une alerte si necessaire.

        Execute la detection de drift et log une alerte si le seuil
        est depasse. Genere automatiquement un rapport en cas de drift.

        Args:
            reference_data: Donnees de reference.
            current_data: Donnees courantes.
            feature_columns: Colonnes a analyser.

        Returns:
            Dictionnaire avec les resultats et le statut d'alerte.
        """
        results = self.detect_data_drift(reference_data, current_data, feature_columns)

        if results["alert"]:
            logger.warning(
                "ALERTE DRIFT : %.0f%% des features ont derive (%d/%d)",
                results["drift_ratio"] * 100,
                results["n_drifted_features"],
                results["n_total_features"]
            )
            logger.warning("Features en drift : %s", results["drifted_features"])

            # Generer le rapport automatiquement
            report_path = self.generate_report(
                reference_data, current_data, feature_columns
            )
            results["report_path"] = str(report_path)
            logger.warning("Rapport genere : %s", report_path)
        else:
            logger.info(
                "Pas de drift significatif detecte (%d/%d features)",
                results["n_drifted_features"],
                results["n_total_features"]
            )

        return results
