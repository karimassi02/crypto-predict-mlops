"""Main collection script that orchestrates all data collectors.

Can be run manually or scheduled via APScheduler for automated daily collection.

Usage:
    python scripts/collect_all.py              # Run once
    python scripts/collect_all.py --schedule   # Run on daily schedule
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collectors.coingecko_collector import CoinGeckoCollector
from src.data.collectors.fear_greed_collector import FearGreedCollector
from src.data.collectors.news_scraper import CryptoNewsScraper
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_collection():
    """Execute all data collectors and log results."""
    start = datetime.now()
    logger.info("=== Starting data collection at %s ===", start.isoformat())

    results = {}

    # 1. CoinGecko market data
    try:
        collector = CoinGeckoCollector()
        data = collector.collect_all()
        results["coingecko"] = {crypto: len(df) for crypto, df in data.items()}
        logger.info("CoinGecko: collected data for %d cryptocurrencies", len(data))
    except Exception as e:
        logger.error("CoinGecko collection failed: %s", e)
        results["coingecko"] = f"FAILED: {e}"

    # 2. Fear & Greed Index
    try:
        collector = FearGreedCollector()
        df = collector.collect()
        results["fear_greed"] = len(df)
        logger.info("Fear & Greed: collected %d records", len(df))
    except Exception as e:
        logger.error("Fear & Greed collection failed: %s", e)
        results["fear_greed"] = f"FAILED: {e}"

    # 3. Crypto news scraping
    try:
        scraper = CryptoNewsScraper()
        df = scraper.collect()
        results["news"] = len(df)
        logger.info("News: scraped %d headlines", len(df))
    except Exception as e:
        logger.error("News scraping failed: %s", e)
        results["news"] = f"FAILED: {e}"

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=== Collection completed in %.1fs ===", elapsed)
    logger.info("Results: %s", results)

    return results


def run_scheduled():
    """Run collection on a daily schedule using APScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_collection,
        trigger="cron",
        hour=8,
        minute=0,
        id="daily_collection",
    )

    logger.info("Scheduler started — collection runs daily at 08:00")
    logger.info("Press Ctrl+C to stop")

    # Run once immediately, then wait for schedule
    run_collection()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )

    parser = argparse.ArgumentParser(description="Crypto data collection")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on daily schedule (default: run once)",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_collection()
