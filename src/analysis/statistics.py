"""Statistical analysis module for cryptocurrency data.

Provides descriptive statistics, hypothesis tests, and correlation analysis
for the RNCP C2.1.2 / C2.1.4 competencies.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def descriptive_stats(df: pd.DataFrame, group_col: str = "crypto_id") -> pd.DataFrame:
    """Compute descriptive statistics per cryptocurrency.

    Args:
        df: DataFrame with numeric columns and a grouping column.
        group_col: Column to group by.

    Returns:
        DataFrame with mean, std, min, max, quartiles per group.
    """
    numeric_cols = ["price", "daily_return", "total_volume", "volatility_7d"]
    cols = [c for c in numeric_cols if c in df.columns]

    if group_col in df.columns:
        result = df.groupby(group_col)[cols].describe()
    else:
        result = df[cols].describe()

    return result


def test_normality(series: pd.Series, alpha: float = 0.05) -> dict:
    """Test if a series follows a normal distribution (Shapiro-Wilk).

    Args:
        series: Numeric series to test.
        alpha: Significance level.

    Returns:
        Dict with statistic, p_value, and is_normal.
    """
    clean = series.dropna()
    # Shapiro-Wilk limited to 5000 samples
    if len(clean) > 5000:
        clean = clean.sample(5000, random_state=42)

    if len(clean) < 20:
        return {"statistic": None, "p_value": None, "is_normal": None, "note": "Not enough data"}

    stat, p_value = stats.shapiro(clean)
    return {
        "statistic": round(stat, 6),
        "p_value": round(p_value, 6),
        "is_normal": p_value > alpha,
    }


def test_correlation(series_a: pd.Series, series_b: pd.Series,
                     method: str = "pearson") -> dict:
    """Test correlation between two series.

    Args:
        series_a: First numeric series.
        series_b: Second numeric series.
        method: 'pearson' or 'spearman'.

    Returns:
        Dict with coefficient, p_value, and interpretation.
    """
    mask = series_a.notna() & series_b.notna()
    a = series_a[mask]
    b = series_b[mask]

    if len(a) < 10:
        return {"coefficient": None, "p_value": None, "note": "Not enough data"}

    if method == "spearman":
        coef, p_value = stats.spearmanr(a, b)
    else:
        coef, p_value = stats.pearsonr(a, b)

    # Interpret strength
    abs_coef = abs(coef)
    if abs_coef < 0.3:
        strength = "faible"
    elif abs_coef < 0.7:
        strength = "moderee"
    else:
        strength = "forte"

    direction = "positive" if coef > 0 else "negative"

    return {
        "coefficient": round(coef, 4),
        "p_value": round(p_value, 6),
        "interpretation": f"Correlation {strength} {direction}",
        "significant": p_value < 0.05,
    }


def correlation_matrix(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """Compute correlation matrix for selected columns.

    Args:
        df: DataFrame with numeric columns.
        columns: Columns to include (default: auto-detect numeric).

    Returns:
        Correlation matrix as DataFrame.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].corr()


def compare_cryptos(df: pd.DataFrame, metric: str = "daily_return",
                    group_col: str = "crypto_id") -> pd.DataFrame:
    """Compare performance metrics across cryptocurrencies.

    Args:
        df: DataFrame with the metric column and group column.
        metric: Column to compare.
        group_col: Grouping column.

    Returns:
        Summary DataFrame with mean, std, sharpe ratio, min, max per crypto.
    """
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    summary = df.groupby(group_col)[metric].agg(
        ["mean", "std", "min", "max", "count"]
    )

    # Sharpe ratio (annualized, assuming 365 trading days for crypto)
    summary["sharpe_ratio"] = (summary["mean"] / summary["std"]) * np.sqrt(365)
    summary = summary.round(6)

    return summary
