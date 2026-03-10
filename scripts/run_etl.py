"""Run the full ETL pipeline: extract from DB/CSV, transform, load processed data.

Usage:
    python scripts/run_etl.py                # From CSV files
    python scripts/run_etl.py --from-db      # From PostgreSQL
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.etl.transformers import DataMerger, FearGreedTransformer, MarketDataTransformer
from src.utils.config import ROOT_DIR, load_config

logger = logging.getLogger(__name__)


def extract_from_csv(config: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Extract data from raw CSV files."""
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]
    cryptos = config["collection"]["cryptocurrencies"]

    market_dfs = {}
    for crypto in cryptos:
        symbol = crypto["symbol"]
        csv_path = raw_dir / f"{symbol}_market_data.csv"
        if csv_path.exists():
            market_dfs[crypto["id"]] = pd.read_csv(csv_path)
            logger.info("Loaded %s (%d rows)", csv_path.name, len(market_dfs[crypto["id"]]))
        else:
            logger.warning("File not found: %s", csv_path)

    fg_path = raw_dir / "fear_greed_index.csv"
    fg_df = pd.read_csv(fg_path) if fg_path.exists() else pd.DataFrame()

    return market_dfs, fg_df


def extract_from_db() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Extract data from PostgreSQL."""
    from src.data.storage.postgres_connector import PostgresConnector

    market_dfs = {}
    with PostgresConnector() as pg:
        # Get all cryptos
        cryptos = pg.query("SELECT coingecko_id FROM cryptocurrencies")
        for _, row in cryptos.iterrows():
            crypto_id = row["coingecko_id"]
            df = pg.get_market_data(coingecko_id=crypto_id)
            if not df.empty:
                df["crypto_id"] = crypto_id
                market_dfs[crypto_id] = df
                logger.info("Extracted %d rows for %s from PostgreSQL", len(df), crypto_id)

        fg_df = pg.query("""
            SELECT date, value as fg_value, classification as fg_classification
            FROM fear_greed_index ORDER BY date
        """)

    return market_dfs, fg_df


def run_etl(from_db: bool = False):
    """Execute the full ETL pipeline."""
    config = load_config()
    start = datetime.now()
    logger.info("=== Starting ETL pipeline at %s ===", start.isoformat())

    # --- Extract ---
    if from_db:
        logger.info("Extracting from PostgreSQL")
        market_dfs, fg_df = extract_from_db()
    else:
        logger.info("Extracting from CSV files")
        market_dfs, fg_df = extract_from_csv(config)

    if not market_dfs:
        logger.error("No market data found. Run collection first.")
        return

    # --- Transform ---
    market_transformer = MarketDataTransformer()
    fg_transformer = FearGreedTransformer()
    merger = DataMerger()

    # Transform Fear & Greed
    fg_transformed = fg_transformer.transform(fg_df) if not fg_df.empty else pd.DataFrame()

    # Transform and merge each crypto
    processed_dir = ROOT_DIR / config["storage"]["processed_data_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_processed = []

    for crypto_id, df in market_dfs.items():
        logger.info("--- Transforming %s ---", crypto_id)

        # Transform market data
        transformed = market_transformer.transform(df)

        # Merge with Fear & Greed
        if not fg_transformed.empty:
            transformed = merger.merge(transformed, fg_transformed)

        # --- Load (save processed) ---
        symbol = crypto_id.upper()
        for crypto in config["collection"]["cryptocurrencies"]:
            if crypto["id"] == crypto_id:
                symbol = crypto["symbol"]
                break

        output_path = processed_dir / f"{symbol}_processed.csv"
        transformed.to_csv(output_path, index=False)
        logger.info("Saved %s (%d rows, %d columns)",
                    output_path.name, len(transformed), len(transformed.columns))

        all_processed.append(transformed)

    # Save combined dataset
    if all_processed:
        combined = pd.concat(all_processed, ignore_index=True)
        combined_path = processed_dir / "all_cryptos_processed.csv"
        combined.to_csv(combined_path, index=False)
        logger.info("Saved combined dataset: %d rows, %d columns",
                    len(combined), len(combined.columns))

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== ETL pipeline completed in %.1fs ===", elapsed)


if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )

    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument("--from-db", action="store_true",
                        help="Extract from PostgreSQL instead of CSV files")
    args = parser.parse_args()

    run_etl(from_db=args.from_db)
