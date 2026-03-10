"""CoinGecko API collector for cryptocurrency market data."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import ROOT_DIR, get_coingecko_api_key, load_config

logger = logging.getLogger(__name__)


class CoinGeckoCollector:
    """Collects cryptocurrency data from the CoinGecko API.

    Retrieves historical price, volume, market cap and OHLC data
    for configured cryptocurrencies.
    """

    def __init__(self):
        config = load_config()
        self.api_key = get_coingecko_api_key()
        self.base_url = config["collection"]["coingecko"]["base_url"]
        self.currency = config["collection"]["coingecko"]["currency"]
        self.history_days = config["collection"]["coingecko"]["history_days"]
        self.rate_limit = config["collection"]["coingecko"]["rate_limit_seconds"]
        self.cryptos = config["collection"]["cryptocurrencies"]
        self.output_dir = ROOT_DIR / config["storage"]["raw_data_dir"]
        self.headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": self.api_key,
        }

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limited request to the CoinGecko API."""
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        time.sleep(self.rate_limit)
        return response.json()

    def collect_market_history(self, crypto_id: str) -> pd.DataFrame:
        """Collect historical market data (price, volume, market cap).

        Args:
            crypto_id: CoinGecko cryptocurrency ID (e.g. 'bitcoin').

        Returns:
            DataFrame with columns: timestamp, price, market_cap, total_volume.
        """
        logger.info("Collecting market history for %s (%d days)", crypto_id, self.history_days)

        data = self._request(
            f"coins/{crypto_id}/market_chart",
            params={"vs_currency": self.currency, "days": self.history_days},
        )

        df = pd.DataFrame({
            "timestamp": [p[0] for p in data["prices"]],
            "price": [p[1] for p in data["prices"]],
            "market_cap": [p[1] for p in data["market_caps"]],
            "total_volume": [p[1] for p in data["total_volumes"]],
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.date
        df["crypto_id"] = crypto_id

        # Keep one row per day (last value of the day)
        df = df.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
        df = df.reset_index(drop=True)

        logger.info("Collected %d daily records for %s", len(df), crypto_id)
        return df

    def collect_ohlc(self, crypto_id: str) -> pd.DataFrame:
        """Collect OHLC (Open, High, Low, Close) data.

        Args:
            crypto_id: CoinGecko cryptocurrency ID.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close.
        """
        logger.info("Collecting OHLC for %s", crypto_id)

        data = self._request(
            f"coins/{crypto_id}/ohlc",
            params={"vs_currency": self.currency, "days": self.history_days},
        )

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.date
        df["crypto_id"] = crypto_id

        # Keep one row per day (last candle of the day)
        df = df.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
        df = df.reset_index(drop=True)

        logger.info("Collected %d OHLC records for %s", len(df), crypto_id)
        return df

    def collect_all(self) -> dict[str, pd.DataFrame]:
        """Collect market history and OHLC for all configured cryptocurrencies.

        Returns:
            Dict with keys '{crypto_id}_market' and '{crypto_id}_ohlc'.
        """
        all_data = {}

        for crypto in self.cryptos:
            crypto_id = crypto["id"]
            symbol = crypto["symbol"]
            logger.info("--- Collecting %s (%s) ---", crypto_id, symbol)

            market_df = self.collect_market_history(crypto_id)
            ohlc_df = self.collect_ohlc(crypto_id)

            # Merge market + OHLC on date
            merged = pd.merge(
                market_df, ohlc_df.drop(columns=["timestamp", "crypto_id"]),
                on="date", how="left",
            )

            all_data[crypto_id] = merged
            self._save_csv(merged, f"{symbol}_market_data.csv")

        return all_data

    def _save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV in the raw data directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info("Saved %s (%d rows)", filepath, len(df))
        return filepath
