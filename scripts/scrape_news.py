"""Script to run crypto news scraping."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collectors.news_scraper import CryptoNewsScraper
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )

    scraper = CryptoNewsScraper()
    df = scraper.collect()

    print(f"\nCrypto News: {len(df)} headlines")
    if not df.empty:
        print(df[["title"]].head(10).to_string())
