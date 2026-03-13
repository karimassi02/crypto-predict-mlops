"""Tests unitaires pour le module de detection de data drift."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector


@pytest.fixture
def tmp_reports_dir(tmp_path):
    """Repertoire temporaire pour les rapports de test."""
    return tmp_path / "drift_reports"


@pytest.fixture
def detector(tmp_reports_dir):
    """Instance de DriftDetector avec repertoire temporaire."""
    return DriftDetector(reports_dir=tmp_reports_dir, drift_threshold=0.5)


def make_dataframe(n: int, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Cree un DataFrame numerique synthetique.

    Args:
        n: Nombre de lignes.
        seed: Graine aleatoire.
        shift: Decalage applique aux valeurs (simule un drift).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_price": rng.normal(50000 + shift, 500, n),
        "feat_volume": rng.normal(1e9 + shift * 100, 1e8, n),
        "feat_rsi": np.clip(rng.normal(50 + shift * 0.1, 15, n), 0, 100),
        "feat_return": rng.normal(0.0 + shift * 0.001, 0.02, n),
    })


class TestDriftDetectorInit:
    """Tests d'initialisation."""

    def test_default_reports_dir_created(self, tmp_path):
        """Le repertoire de rapports est cree a l'initialisation."""
        reports_dir = tmp_path / "my_reports"
        assert not reports_dir.exists()
        DriftDetector(reports_dir=reports_dir)
        assert reports_dir.exists()

    def test_custom_threshold(self, tmp_reports_dir):
        """Le seuil de drift est correctement stocke."""
        det = DriftDetector(reports_dir=tmp_reports_dir, drift_threshold=0.3)
        assert det.drift_threshold == 0.3


class TestDetectDataDrift:
    """Tests de la methode detect_data_drift."""

    def _mock_evidently(self, drift_ratio: float = 0.0, n_drifted: int = 0,
                        n_total: int = 4):
        """Construit un mock du snapshot Evidently."""
        snapshot = MagicMock()
        snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_id": "evidently:metric:DataDriftTable",
                    "metric_fields": {},
                    "result": {
                        "drift_detected": drift_ratio >= 0.5,
                        "n_drifted_columns": n_drifted,
                        "n_columns": n_total,
                    }
                }
            ]
        }
        return snapshot

    def test_returns_required_keys(self, detector):
        """Le resultat contient toutes les cles attendues."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._mock_evidently()
            MockReport.return_value = mock_report_instance

            result = detector.detect_data_drift(ref, cur)

        required_keys = {
            "dataset_drift", "drift_ratio", "n_drifted_features",
            "n_total_features", "drifted_features", "feature_details",
            "alert", "timestamp"
        }
        assert required_keys.issubset(set(result.keys()))

    def test_no_drift_scenario(self, detector):
        """Pas d'alerte quand le drift est faible."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._mock_evidently(
                drift_ratio=0.0, n_drifted=0, n_total=4
            )
            MockReport.return_value = mock_report_instance

            result = detector.detect_data_drift(ref, cur)

        assert result["alert"] is False
        assert result["drift_ratio"] == 0.0

    def test_drift_scenario(self, detector):
        """Alerte declenchee quand le drift depasse le seuil."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._mock_evidently(
                drift_ratio=0.75, n_drifted=3, n_total=4
            )
            MockReport.return_value = mock_report_instance

            result = detector.detect_data_drift(ref, cur)

        assert result["alert"] is True
        assert result["n_drifted_features"] == 3

    def test_feature_columns_filter(self, detector):
        """Seules les colonnes specifiees sont analysees."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)
        ref["extra_col"] = 1
        cur["extra_col"] = 2

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset") as MockDataset:
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._mock_evidently()
            MockReport.return_value = mock_report_instance

            detector.detect_data_drift(ref, cur, feature_columns=["feat_price", "feat_rsi"])

            # Verifier que from_pandas a ete appele avec les bonnes colonnes
            calls = MockDataset.from_pandas.call_args_list
            assert len(calls) == 2
            passed_ref = calls[0][0][0]
            assert set(passed_ref.columns) == {"feat_price", "feat_rsi"}

    def test_handles_infinite_values(self, detector):
        """Les valeurs infinies sont remplacees avant l'analyse."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)
        ref.iloc[0, 0] = np.inf
        cur.iloc[0, 0] = -np.inf

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset") as MockDataset:
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._mock_evidently()
            MockReport.return_value = mock_report_instance

            # Ne doit pas lever d'exception
            detector.detect_data_drift(ref, cur)

            passed_ref = MockDataset.from_pandas.call_args_list[0][0][0]
            assert not passed_ref.isin([np.inf, -np.inf]).any().any()

    def test_numeric_columns_only(self, detector):
        """Les colonnes non-numeriques sont automatiquement ignorees."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)
        ref["category"] = "A"
        cur["category"] = "B"

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset") as MockDataset:
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._mock_evidently()
            MockReport.return_value = mock_report_instance

            detector.detect_data_drift(ref, cur)

            passed_ref = MockDataset.from_pandas.call_args_list[0][0][0]
            assert "category" not in passed_ref.columns


class TestGenerateReport:
    """Tests de la methode generate_report."""

    def test_html_file_created(self, detector, tmp_reports_dir):
        """Un fichier HTML est cree dans le repertoire de rapports."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_snapshot = MagicMock()
            mock_snapshot.dict.return_value = {"metrics": []}
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = mock_snapshot
            MockReport.return_value = mock_report_instance

            html_path = detector.generate_report(ref, cur, report_name="test_report")

        assert html_path.name == "test_report.html"
        mock_snapshot.save_html.assert_called_once()

    def test_json_file_created(self, detector, tmp_reports_dir):
        """Un fichier JSON est cree en meme temps que le HTML."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_snapshot = MagicMock()
            mock_snapshot.dict.return_value = {"metrics": [], "version": "1.0"}
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = mock_snapshot
            MockReport.return_value = mock_report_instance

            html_path = detector.generate_report(ref, cur, report_name="json_test")

        json_path = html_path.parent / "json_test.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "metrics" in data

    def test_auto_report_name_contains_timestamp(self, detector):
        """Le nom de rapport auto-genere contient un timestamp."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_snapshot = MagicMock()
            mock_snapshot.dict.return_value = {"metrics": []}
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = mock_snapshot
            MockReport.return_value = mock_report_instance

            html_path = detector.generate_report(ref, cur)

        assert "drift_report_" in html_path.name


class TestCheckAndAlert:
    """Tests de la methode check_and_alert."""

    def _make_snapshot(self, n_drifted: int = 0, n_total: int = 4):
        snapshot = MagicMock()
        snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_id": "evidently:metric:DataDriftTable",
                    "metric_fields": {},
                    "result": {
                        "drift_detected": n_drifted / n_total >= 0.5,
                        "n_drifted_columns": n_drifted,
                        "n_columns": n_total,
                    }
                }
            ]
        }
        return snapshot

    def test_no_alert_no_report(self, detector):
        """Sans drift, pas de rapport genere automatiquement."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._make_snapshot(0, 4)
            MockReport.return_value = mock_report_instance

            result = detector.check_and_alert(ref, cur)

        assert result["alert"] is False
        assert "report_path" not in result

    def test_alert_generates_report(self, detector):
        """En cas de drift, un rapport est genere et son chemin est dans le resultat."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        snapshot_no_drift = self._make_snapshot(0, 4)
        snapshot_with_drift = self._make_snapshot(3, 4)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_report_instance = MagicMock()
            # Premier appel (detect) -> drift, deuxieme appel (generate_report) -> pas de drift
            mock_report_instance.run.side_effect = [
                snapshot_with_drift, snapshot_no_drift
            ]
            MockReport.return_value = mock_report_instance

            result = detector.check_and_alert(ref, cur)

        assert result["alert"] is True
        assert "report_path" in result

    def test_result_has_timestamp(self, detector):
        """Le timestamp est present dans le resultat."""
        ref = make_dataframe(200)
        cur = make_dataframe(100)

        with patch("src.monitoring.drift_detector.Report") as MockReport, \
             patch("src.monitoring.drift_detector.Dataset"):
            mock_report_instance = MagicMock()
            mock_report_instance.run.return_value = self._make_snapshot()
            MockReport.return_value = mock_report_instance

            result = detector.check_and_alert(ref, cur)

        assert "timestamp" in result
        assert "T" in result["timestamp"]  # Format ISO 8601
