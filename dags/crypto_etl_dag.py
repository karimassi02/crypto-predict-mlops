"""Airflow DAG for the crypto data ETL pipeline.

Orchestrates: collect data → store in databases → transform → save processed.
Scheduled daily at 08:00 UTC.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments for all tasks
default_args = {
    "owner": "crypto-predict",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="crypto_etl_pipeline",
    default_args=default_args,
    description="Collecte, stockage et transformation des donnees crypto",
    schedule_interval="0 8 * * *",  # Daily at 08:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["crypto", "etl", "mlops"],
)


# ---- Task functions ----

def collect_coingecko(**kwargs):
    """Collect market data from CoinGecko API."""
    from src.data.collectors.coingecko_collector import CoinGeckoCollector

    collector = CoinGeckoCollector()
    data = collector.collect_all()
    return {crypto: len(df) for crypto, df in data.items()}


def collect_fear_greed(**kwargs):
    """Collect Fear & Greed Index."""
    from src.data.collectors.fear_greed_collector import FearGreedCollector

    collector = FearGreedCollector()
    df = collector.collect()
    return len(df)


def collect_news(**kwargs):
    """Scrape crypto news headlines."""
    from src.data.collectors.news_scraper import CryptoNewsScraper

    scraper = CryptoNewsScraper()
    df = scraper.collect()
    return len(df)


def store_postgres(**kwargs):
    """Store collected data into PostgreSQL."""
    import pandas as pd
    from src.data.storage.postgres_connector import PostgresConnector
    from src.utils.config import ROOT_DIR, load_config

    config = load_config()
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]

    with PostgresConnector() as pg:
        for crypto in config["collection"]["cryptocurrencies"]:
            csv_path = raw_dir / f"{crypto['symbol']}_market_data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                pg.insert_market_data(df, crypto["id"])

        fg_path = raw_dir / "fear_greed_index.csv"
        if fg_path.exists():
            df = pd.read_csv(fg_path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            pg.insert_fear_greed(df)


def store_mongo(**kwargs):
    """Store news articles into MongoDB."""
    import pandas as pd
    from src.data.storage.mongo_connector import MongoConnector
    from src.utils.config import ROOT_DIR, load_config

    config = load_config()
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]
    news_path = raw_dir / "crypto_news.csv"

    if news_path.exists():
        df = pd.read_csv(news_path)
        with MongoConnector() as mongo:
            mongo.insert_news(df.to_dict(orient="records"))


def run_transformations(**kwargs):
    """Transform raw data into processed features."""
    import pandas as pd
    from src.data.etl.transformers import DataMerger, FearGreedTransformer, MarketDataTransformer
    from src.utils.config import ROOT_DIR, load_config

    config = load_config()
    raw_dir = ROOT_DIR / config["storage"]["raw_data_dir"]
    processed_dir = ROOT_DIR / config["storage"]["processed_data_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    market_transformer = MarketDataTransformer()
    fg_transformer = FearGreedTransformer()
    merger = DataMerger()

    # Transform Fear & Greed
    fg_path = raw_dir / "fear_greed_index.csv"
    fg_transformed = pd.DataFrame()
    if fg_path.exists():
        fg_df = pd.read_csv(fg_path)
        fg_transformed = fg_transformer.transform(fg_df)

    all_processed = []

    for crypto in config["collection"]["cryptocurrencies"]:
        symbol = crypto["symbol"]
        csv_path = raw_dir / f"{symbol}_market_data.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        transformed = market_transformer.transform(df)

        if not fg_transformed.empty:
            transformed = merger.merge(transformed, fg_transformed)

        output_path = processed_dir / f"{symbol}_processed.csv"
        transformed.to_csv(output_path, index=False)
        all_processed.append(transformed)

    if all_processed:
        combined = pd.concat(all_processed, ignore_index=True)
        combined.to_csv(processed_dir / "all_cryptos_processed.csv", index=False)

    return f"Processed {len(all_processed)} cryptocurrencies"


# ---- Define tasks ----

t_collect_coingecko = PythonOperator(
    task_id="collect_coingecko",
    python_callable=collect_coingecko,
    dag=dag,
)

t_collect_fear_greed = PythonOperator(
    task_id="collect_fear_greed",
    python_callable=collect_fear_greed,
    dag=dag,
)

t_collect_news = PythonOperator(
    task_id="collect_news",
    python_callable=collect_news,
    dag=dag,
)

t_store_postgres = PythonOperator(
    task_id="store_postgres",
    python_callable=store_postgres,
    dag=dag,
)

t_store_mongo = PythonOperator(
    task_id="store_mongo",
    python_callable=store_mongo,
    dag=dag,
)

t_transform = PythonOperator(
    task_id="transform_data",
    python_callable=run_transformations,
    dag=dag,
)

# ---- Dependencies ----
# Collection tasks run in parallel
# Then storage, then transformation
[t_collect_coingecko, t_collect_fear_greed] >> t_store_postgres
t_collect_news >> t_store_mongo
[t_store_postgres, t_store_mongo] >> t_transform
