"""Script to run Fear & Greed Index collection."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collectors.fear_greed_collector import FearGreedCollector
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )

    collector = FearGreedCollector()
    df = collector.collect()

    print(f"\nFear & Greed Index: {len(df)} rows")
    print(df.head(5).to_string())
    print("\n...")
    print(df.tail(5).to_string())
