"""MongoDB connector for semi-structured data storage."""

import logging
import os
from datetime import datetime

from pymongo import MongoClient, UpdateOne

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class MongoConnector:
    """Manages MongoDB connections and document operations.

    Stores semi-structured data (news articles, raw API responses)
    in a document-oriented database.
    """

    def __init__(self):
        self.uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGO_DB", "crypto_predict")
        self._client = None
        self._db = None

    def connect(self):
        """Establish a connection to MongoDB."""
        if self._client is None:
            self._client = MongoClient(self.uri)
            self._db = self._client[self.db_name]
            # Test connection
            self._client.admin.command("ping")
            logger.info("Connected to MongoDB (%s, db=%s)", self.uri, self.db_name)
        return self._db

    def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed")

    def insert_news(self, articles: list[dict]) -> int:
        """Insert news articles with UPSERT on title to avoid duplicates.

        Args:
            articles: List of dicts with keys: title, url, scraped_at.

        Returns:
            Number of documents inserted/updated.
        """
        db = self.connect()
        collection = db["news_articles"]

        operations = []
        for article in articles:
            article["updated_at"] = datetime.now().isoformat()
            operations.append(
                UpdateOne(
                    {"title": article["title"]},
                    {"$set": article, "$setOnInsert": {"created_at": datetime.now().isoformat()}},
                    upsert=True,
                )
            )

        if not operations:
            logger.warning("No articles to insert")
            return 0

        result = collection.bulk_write(operations)
        total = result.upserted_count + result.modified_count
        logger.info("News: %d upserted, %d modified",
                    result.upserted_count, result.modified_count)

        # Create index on title for fast lookups
        collection.create_index("title", unique=True)

        return total

    def insert_raw_response(self, source: str, data: dict) -> str:
        """Store a raw API response for traceability.

        Args:
            source: Data source identifier (e.g. 'coingecko', 'fear_greed').
            data: Raw response data.

        Returns:
            Inserted document ID as string.
        """
        db = self.connect()
        collection = db["raw_responses"]

        doc = {
            "source": source,
            "data": data,
            "collected_at": datetime.now().isoformat(),
        }

        result = collection.insert_one(doc)
        logger.info("Stored raw response from %s (id=%s)", source, result.inserted_id)
        return str(result.inserted_id)

    def get_news(self, limit: int = 100, skip: int = 0) -> list[dict]:
        """Retrieve news articles sorted by most recent.

        Args:
            limit: Maximum number of articles to return.
            skip: Number of articles to skip (for pagination).

        Returns:
            List of article documents.
        """
        db = self.connect()
        collection = db["news_articles"]

        cursor = collection.find(
            {}, {"_id": 0}
        ).sort("scraped_at", -1).skip(skip).limit(limit)

        return list(cursor)

    def get_collection_stats(self) -> dict:
        """Get statistics about stored collections.

        Returns:
            Dict with collection names and document counts.
        """
        db = self.connect()
        stats = {}
        for name in db.list_collection_names():
            stats[name] = db[name].count_documents({})
        return stats

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
