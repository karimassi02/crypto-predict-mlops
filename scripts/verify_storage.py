"""Verify that data was correctly stored in PostgreSQL and MongoDB.

Usage:
    python scripts/verify_storage.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage.mongo_connector import MongoConnector
from src.data.storage.postgres_connector import PostgresConnector
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def verify_postgres():
    """Verify PostgreSQL data and print summary."""
    print("\n=== PostgreSQL ===")

    with PostgresConnector() as pg:
        # Cryptocurrencies table
        df = pg.query("SELECT coingecko_id, symbol FROM cryptocurrencies ORDER BY id")
        print(f"\nCryptocurrencies: {len(df)} entries")
        print(df.to_string(index=False))

        # Market data summary
        df = pg.query("""
            SELECT c.symbol, COUNT(*) as rows,
                   MIN(m.date) as first_date, MAX(m.date) as last_date,
                   ROUND(AVG(m.price)::numeric, 2) as avg_price
            FROM market_data m
            JOIN cryptocurrencies c ON m.crypto_id = c.id
            GROUP BY c.symbol ORDER BY c.symbol
        """)
        print(f"\nMarket data:")
        print(df.to_string(index=False))

        # Fear & Greed summary
        df = pg.query("""
            SELECT COUNT(*) as rows,
                   MIN(date) as first_date, MAX(date) as last_date,
                   ROUND(AVG(value)::numeric, 1) as avg_value
            FROM fear_greed_index
        """)
        print(f"\nFear & Greed Index:")
        print(df.to_string(index=False))


def verify_mongo():
    """Verify MongoDB data and print summary."""
    print("\n=== MongoDB ===")

    with MongoConnector() as mongo:
        stats = mongo.get_collection_stats()
        for collection, count in stats.items():
            print(f"  {collection}: {count} documents")

        news = mongo.get_news(limit=5)
        if news:
            print(f"\nLatest {len(news)} news articles:")
            for article in news:
                print(f"  - {article.get('title', 'N/A')[:80]}")


def main():
    print("=" * 50)
    print("  Storage Verification Report")
    print("=" * 50)

    try:
        verify_postgres()
    except Exception as e:
        print(f"\nPostgreSQL verification failed: {e}")

    try:
        verify_mongo()
    except Exception as e:
        print(f"\nMongoDB verification failed: {e}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    main()
