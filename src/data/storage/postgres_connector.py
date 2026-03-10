"""PostgreSQL connector for structured data storage."""

import logging
import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class PostgresConnector:
    """Manages PostgreSQL connections and data operations.

    Stores structured market data (prices, OHLC) and Fear & Greed Index
    in a relational database with UPSERT logic to avoid duplicates.
    """

    def __init__(self):
        self.conn_params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "dbname": os.getenv("POSTGRES_DB", "crypto_predict"),
            "user": os.getenv("POSTGRES_USER", "admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "changeme"),
        }
        self._conn = None

    def connect(self):
        """Establish a connection to PostgreSQL."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.conn_params)
            logger.info("Connected to PostgreSQL (%s:%s/%s)",
                        self.conn_params["host"],
                        self.conn_params["port"],
                        self.conn_params["dbname"])
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("PostgreSQL connection closed")

    def _get_crypto_id(self, coingecko_id: str) -> int:
        """Get the internal crypto ID from the coingecko_id."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM cryptocurrencies WHERE coingecko_id = %s",
                (coingecko_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Unknown cryptocurrency: {coingecko_id}")
            return row[0]

    def insert_market_data(self, df: pd.DataFrame, coingecko_id: str) -> int:
        """Insert market data into PostgreSQL with UPSERT.

        Args:
            df: DataFrame with columns: date, price, market_cap, total_volume,
                open, high, low, close.
            coingecko_id: CoinGecko cryptocurrency ID.

        Returns:
            Number of rows inserted/updated.
        """
        conn = self.connect()
        crypto_id = self._get_crypto_id(coingecko_id)

        records = []
        for _, row in df.iterrows():
            records.append((
                crypto_id,
                row["date"],
                row.get("price"),
                row.get("market_cap"),
                row.get("total_volume"),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
            ))

        sql = """
            INSERT INTO market_data (crypto_id, date, price, market_cap, total_volume,
                                     open, high, low, close)
            VALUES %s
            ON CONFLICT (crypto_id, date) DO UPDATE SET
                price = EXCLUDED.price,
                market_cap = EXCLUDED.market_cap,
                total_volume = EXCLUDED.total_volume,
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                collected_at = CURRENT_TIMESTAMP
        """

        with conn.cursor() as cur:
            execute_values(cur, sql, records)
        conn.commit()

        logger.info("Inserted/updated %d market rows for %s", len(records), coingecko_id)
        return len(records)

    def insert_fear_greed(self, df: pd.DataFrame) -> int:
        """Insert Fear & Greed Index data with UPSERT.

        Args:
            df: DataFrame with columns: date, fg_value, fg_classification.

        Returns:
            Number of rows inserted/updated.
        """
        conn = self.connect()

        records = []
        for _, row in df.iterrows():
            records.append((
                row["date"],
                int(row["fg_value"]),
                row["fg_classification"],
            ))

        sql = """
            INSERT INTO fear_greed_index (date, value, classification)
            VALUES %s
            ON CONFLICT (date) DO UPDATE SET
                value = EXCLUDED.value,
                classification = EXCLUDED.classification,
                collected_at = CURRENT_TIMESTAMP
        """

        with conn.cursor() as cur:
            execute_values(cur, sql, records)
        conn.commit()

        logger.info("Inserted/updated %d Fear & Greed rows", len(records))
        return len(records)

    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: SQL query string.
            params: Optional query parameters.

        Returns:
            Query results as a DataFrame.
        """
        conn = self.connect()
        return pd.read_sql_query(sql, conn, params=params)

    def get_market_data(self, coingecko_id: str = None,
                        start_date: str = None,
                        end_date: str = None) -> pd.DataFrame:
        """Retrieve market data with optional filters.

        Args:
            coingecko_id: Filter by cryptocurrency (optional).
            start_date: Start date filter YYYY-MM-DD (optional).
            end_date: End date filter YYYY-MM-DD (optional).

        Returns:
            DataFrame with market data.
        """
        sql = """
            SELECT c.coingecko_id, c.symbol, m.date, m.price, m.market_cap,
                   m.total_volume, m.open, m.high, m.low, m.close
            FROM market_data m
            JOIN cryptocurrencies c ON m.crypto_id = c.id
            WHERE 1=1
        """
        params = []

        if coingecko_id:
            sql += " AND c.coingecko_id = %s"
            params.append(coingecko_id)
        if start_date:
            sql += " AND m.date >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND m.date <= %s"
            params.append(end_date)

        sql += " ORDER BY c.symbol, m.date"

        return self.query(sql, tuple(params) if params else None)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
