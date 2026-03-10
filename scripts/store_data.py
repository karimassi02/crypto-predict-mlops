"""Store collected CSV data into PostgreSQL and MongoDB.

Reads raw CSV files from data/raw/ and inserts them into the databases.
Can be run after collect_all.py or independently.

Usage:
    python scripts/store_data.py              # Store all data
    python scripts/store_data.py --source pg  # PostgreSQL only
    python scripts/store_data.py --source mongo  # MongoDB only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage.mongo_connector import MongoConnector
from src.data.storage.postgres_connector import PostgresConnector
from src.utils.config import ROOT_DIR, load_config

logger = logging.getLogger(__name__)


def store_to_postgres():
    """Store market data and Fear & Greed Index into PostgreSQL."""
    config = load_config()
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]
    cryptos = config["collection"]["cryptocurrencies"]

    with PostgresConnector() as pg:
        # Store market data for each crypto
        for crypto in cryptos:
            symbol = crypto["symbol"]
            csv_path = raw_dir / f"{symbol}_market_data.csv"
            if not csv_path.exists():
                logger.warning("File not found: %s — skipping", csv_path)
                continue

            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            pg.insert_market_data(df, crypto["id"])

        # Store Fear & Greed Index
        fg_path = raw_dir / "fear_greed_index.csv"
        if fg_path.exists():
            df = pd.read_csv(fg_path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            pg.insert_fear_greed(df)
        else:
            logger.warning("File not found: %s — skipping", fg_path)


def store_to_mongo():
    """Store news articles into MongoDB."""
    config = load_config()
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]

    news_path = raw_dir / "crypto_news.csv"
    if not news_path.exists():
        logger.warning("File not found: %s — skipping", news_path)
        return

    df = pd.read_csv(news_path)
    articles = df.to_dict(orient="records")

    with MongoConnector() as mongo:
        mongo.insert_news(articles)


def main():
    parser = argparse.ArgumentParser(description="Store collected data into databases")
    parser.add_argument(
        "--source",
        choices=["pg", "mongo", "all"],
        default="all",
        help="Which database to populate (default: all)",
    )
    args = parser.parse_args()

    start = datetime.now()
    logger.info("=== Starting data storage at %s ===", start.isoformat())

    if args.source in ("pg", "all"):
        try:
            store_to_postgres()
            logger.info("PostgreSQL storage completed")
        except Exception as e:
            logger.error("PostgreSQL storage failed: %s", e)

    if args.source in ("mongo", "all"):
        try:
            store_to_mongo()
            logger.info("MongoDB storage completed")
        except Exception as e:
            logger.error("MongoDB storage failed: %s", e)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== Storage completed in %.1fs ===", elapsed)


if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    main()
