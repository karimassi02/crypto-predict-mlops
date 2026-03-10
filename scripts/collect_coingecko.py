"""Script to run CoinGecko data collection."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collectors.coingecko_collector import CoinGeckoCollector
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )

    collector = CoinGeckoCollector()
    data = collector.collect_all()

    for crypto_id, df in data.items():
        print(f"\n{crypto_id}: {len(df)} rows")
        print(df.head(3).to_string())
