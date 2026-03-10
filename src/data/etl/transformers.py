"""Data transformation module for cleaning and preparing crypto data.

Handles missing values, type casting, normalization, and merging
of market data with sentiment indicators.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataTransformer:
    """Cleans and transforms raw market data for analysis.

    Applies a standardized pipeline: type casting, missing value handling,
    outlier detection, normalization, and daily returns computation.
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full cleaning pipeline to market data.

        Args:
            df: Raw market DataFrame with columns: date, price, market_cap,
                total_volume, open, high, low, close, crypto_id.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning market data (%d rows)", len(df))
        df = df.copy()

        # Ensure correct types
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["price", "market_cap", "total_volume", "open", "high", "low", "close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by crypto and date
        if "crypto_id" in df.columns:
            df = df.sort_values(["crypto_id", "date"]).reset_index(drop=True)
        else:
            df = df.sort_values("date").reset_index(drop=True)

        # Handle missing values: forward fill then backward fill per crypto
        if "crypto_id" in df.columns:
            df[numeric_cols] = df.groupby("crypto_id")[numeric_cols].transform(
                lambda x: x.ffill().bfill()
            )
        else:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Drop rows where price is still NaN (no data at all)
        before = len(df)
        df = df.dropna(subset=["price"])
        if len(df) < before:
            logger.warning("Dropped %d rows with no price data", before - len(df))

        # Remove duplicate dates per crypto
        subset = ["crypto_id", "date"] if "crypto_id" in df.columns else ["date"]
        df = df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

        logger.info("Cleaning complete: %d rows", len(df))
        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily and log returns columns.

        Args:
            df: Cleaned market DataFrame with 'price' column.

        Returns:
            DataFrame with added 'daily_return' and 'log_return' columns.
        """
        df = df.copy()

        if "crypto_id" in df.columns:
            df["daily_return"] = df.groupby("crypto_id")["price"].pct_change()
            df["log_return"] = df.groupby("crypto_id")["price"].transform(
                lambda x: np.log(x / x.shift(1))
            )
        else:
            df["daily_return"] = df["price"].pct_change()
            df["log_return"] = np.log(df["price"] / df["price"].shift(1))

        return df

    def add_volatility(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """Add rolling volatility (standard deviation of returns).

        Args:
            df: DataFrame with 'daily_return' column.
            window: Rolling window size in days.

        Returns:
            DataFrame with added 'volatility_{window}d' column.
        """
        df = df.copy()
        col_name = f"volatility_{window}d"

        if "crypto_id" in df.columns:
            df[col_name] = df.groupby("crypto_id")["daily_return"].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        else:
            df[col_name] = df["daily_return"].rolling(window=window, min_periods=1).std()

        return df

    def add_moving_averages(self, df: pd.DataFrame,
                            windows: list[int] = None) -> pd.DataFrame:
        """Add simple moving averages (SMA) for price.

        Args:
            df: DataFrame with 'price' column.
            windows: List of window sizes (default: [7, 14, 30]).

        Returns:
            DataFrame with added 'sma_{n}' columns.
        """
        if windows is None:
            windows = [7, 14, 30]

        df = df.copy()

        for w in windows:
            col_name = f"sma_{w}"
            if "crypto_id" in df.columns:
                df[col_name] = df.groupby("crypto_id")["price"].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
            else:
                df[col_name] = df["price"].rolling(window=w, min_periods=1).mean()

        return df

    def detect_outliers(self, df: pd.DataFrame, column: str = "daily_return",
                        z_threshold: float = 3.0) -> pd.DataFrame:
        """Flag outliers using Z-score method.

        Args:
            df: DataFrame with the target column.
            column: Column to check for outliers.
            z_threshold: Z-score threshold (default: 3.0).

        Returns:
            DataFrame with added 'is_outlier' boolean column.
        """
        df = df.copy()

        if column not in df.columns:
            logger.warning("Column '%s' not found, skipping outlier detection", column)
            df["is_outlier"] = False
            return df

        mean = df[column].mean()
        std = df[column].std()

        if std == 0 or pd.isna(std):
            df["is_outlier"] = False
        else:
            z_scores = (df[column] - mean).abs() / std
            df["is_outlier"] = z_scores > z_threshold

        n_outliers = df["is_outlier"].sum()
        if n_outliers > 0:
            logger.info("Detected %d outliers in '%s' (z > %.1f)", n_outliers, column, z_threshold)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full transformation pipeline.

        Pipeline: clean → returns → volatility → moving averages → outlier detection.

        Args:
            df: Raw market DataFrame.

        Returns:
            Fully transformed DataFrame.
        """
        df = self.clean(df)
        df = self.add_returns(df)
        df = self.add_volatility(df, window=7)
        df = self.add_moving_averages(df, windows=[7, 14, 30])
        df = self.detect_outliers(df, column="daily_return")

        logger.info("Full transformation complete: %d rows, %d columns",
                    len(df), len(df.columns))
        return df


class FearGreedTransformer:
    """Cleans and transforms Fear & Greed Index data."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Fear & Greed data.

        Args:
            df: Raw DataFrame with columns: date, fg_value, fg_classification.

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["fg_value"] = pd.to_numeric(df["fg_value"], errors="coerce")
        df = df.dropna(subset=["fg_value"])
        df["fg_value"] = df["fg_value"].astype(int)
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

        logger.info("Fear & Greed cleaned: %d rows", len(df))
        return df

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived sentiment features.

        Args:
            df: Cleaned Fear & Greed DataFrame.

        Returns:
            DataFrame with added sentiment features.
        """
        df = df.copy()

        # Rolling average (7 days)
        df["fg_sma_7"] = df["fg_value"].rolling(window=7, min_periods=1).mean()

        # Sentiment change
        df["fg_change"] = df["fg_value"].diff()

        # Sentiment zone: extreme_fear / fear / neutral / greed / extreme_greed
        bins = [0, 20, 40, 60, 80, 100]
        labels = ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]
        df["fg_zone"] = pd.cut(df["fg_value"], bins=bins, labels=labels, include_lowest=True)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full transformation pipeline."""
        df = self.clean(df)
        df = self.add_sentiment_features(df)
        logger.info("Fear & Greed transformation complete: %d rows", len(df))
        return df


class DataMerger:
    """Merges market data with sentiment data on date."""

    def merge(self, market_df: pd.DataFrame,
              fear_greed_df: pd.DataFrame) -> pd.DataFrame:
        """Merge market data with Fear & Greed Index.

        Args:
            market_df: Transformed market DataFrame (must have 'date' column).
            fear_greed_df: Transformed Fear & Greed DataFrame.

        Returns:
            Merged DataFrame with all features.
        """
        market_df = market_df.copy()
        fear_greed_df = fear_greed_df.copy()

        # Ensure date types match
        market_df["date"] = pd.to_datetime(market_df["date"])
        fear_greed_df["date"] = pd.to_datetime(fear_greed_df["date"])

        # Select relevant Fear & Greed columns
        fg_cols = ["date", "fg_value", "fg_sma_7", "fg_change", "fg_zone"]
        fg_available = [c for c in fg_cols if c in fear_greed_df.columns]
        fg_subset = fear_greed_df[fg_available]

        merged = pd.merge(market_df, fg_subset, on="date", how="left")

        # Forward fill sentiment data for weekends/gaps
        sentiment_cols = [c for c in fg_available if c != "date"]
        merged[sentiment_cols] = merged[sentiment_cols].ffill()

        logger.info("Merged data: %d rows, %d columns", len(merged), len(merged.columns))
        return merged
