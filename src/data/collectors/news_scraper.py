"""Web scraper for cryptocurrency news headlines."""

import logging
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.config import ROOT_DIR, load_config

logger = logging.getLogger(__name__)


class CryptoNewsScraper:
    """Scrapes cryptocurrency news headlines from CoinTelegraph.

    Demonstrates web scraping techniques (BeautifulSoup) for the
    RNCP C1.1.2 competency requirement.
    """

    def __init__(self):
        config = load_config()
        self.user_agent = config["collection"]["scraping"]["user_agent"]
        self.rate_limit = config["collection"]["scraping"]["rate_limit_seconds"]
        self.output_dir = ROOT_DIR / config["storage"]["raw_data_dir"]
        self.headers = {"User-Agent": self.user_agent}

    def scrape_cointelegraph(self) -> pd.DataFrame:
        """Scrape latest news headlines from CoinTelegraph.

        Returns:
            DataFrame with columns: title, url, scraped_at.
        """
        logger.info("Scraping CoinTelegraph headlines")

        url = "https://cointelegraph.com/tags/cryptocurrencies"
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        articles = []
        for tag in soup.find_all("a", class_="post-card-inline__title-link"):
            title = tag.get_text(strip=True)
            link = tag.get("href", "")
            if not link.startswith("http"):
                link = f"https://cointelegraph.com{link}"
            articles.append({"title": title, "url": link})

        time.sleep(self.rate_limit)

        if not articles:
            logger.warning("No articles found — page structure may have changed")
            return pd.DataFrame(columns=["title", "url", "scraped_at"])

        df = pd.DataFrame(articles)
        df["scraped_at"] = datetime.now().isoformat()
        df = df.drop_duplicates(subset="title").reset_index(drop=True)

        logger.info("Scraped %d headlines", len(df))
        return df

    def collect(self) -> pd.DataFrame:
        """Run the full scraping pipeline and save results.

        Returns:
            DataFrame with scraped headlines.
        """
        df = self.scrape_cointelegraph()

        if not df.empty:
            self._save_csv(df)

        return df

    def _save_csv(self, df: pd.DataFrame):
        """Save DataFrame to CSV, appending to existing data if present."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / "crypto_news.csv"

        if filepath.exists():
            existing = pd.read_csv(filepath)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset="title").reset_index(drop=True)

        df.to_csv(filepath, index=False)
        logger.info("Saved %s (%d total rows)", filepath, len(df))
