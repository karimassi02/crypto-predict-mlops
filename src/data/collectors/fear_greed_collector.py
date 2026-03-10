"""Fear & Greed Index collector for crypto market sentiment."""

import logging

import pandas as pd
import requests

from src.utils.config import ROOT_DIR, load_config

logger = logging.getLogger(__name__)


class FearGreedCollector:
    """Collects the Crypto Fear & Greed Index from alternative.me.

    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed)
    and reflects overall market sentiment.
    """

    def __init__(self):
        config = load_config()
        self.base_url = config["collection"]["fear_greed"]["base_url"]
        self.history_days = config["collection"]["fear_greed"]["history_days"]
        self.output_dir = ROOT_DIR / config["storage"]["raw_data_dir"]

    def collect(self) -> pd.DataFrame:
        """Collect historical Fear & Greed Index data.

        Returns:
            DataFrame with columns: date, fg_value, fg_classification.
        """
        logger.info("Collecting Fear & Greed Index (%d days)", self.history_days)

        response = requests.get(
            self.base_url,
            params={"limit": self.history_days, "format": "json"},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()["data"]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.date
        df["fg_value"] = df["value"].astype(int)
        df["fg_classification"] = df["value_classification"]

        df = df[["date", "fg_value", "fg_classification"]]
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("Collected %d records", len(df))

        self._save_csv(df)
        return df

    def _save_csv(self, df: pd.DataFrame):
        """Save DataFrame to CSV in the raw data directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / "fear_greed_index.csv"
        df.to_csv(filepath, index=False)
        logger.info("Saved %s (%d rows)", filepath, len(df))
