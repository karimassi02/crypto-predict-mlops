"""Security utilities: input validation, data sanitization, secret management."""

import logging
import os
import re
from html import escape

logger = logging.getLogger(__name__)

# Required environment variables and their descriptions
REQUIRED_SECRETS = {
    "COINGECKO_API_KEY": "CoinGecko API key",
    "POSTGRES_PASSWORD": "PostgreSQL password",
}

OPTIONAL_SECRETS = {
    "POSTGRES_HOST": ("PostgreSQL host", "localhost"),
    "POSTGRES_PORT": ("PostgreSQL port", "5432"),
    "POSTGRES_DB": ("PostgreSQL database", "crypto_predict"),
    "POSTGRES_USER": ("PostgreSQL user", "admin"),
    "MONGO_URI": ("MongoDB URI", "mongodb://localhost:27017"),
    "MONGO_DB": ("MongoDB database", "crypto_predict"),
}

# Weak passwords to reject
WEAK_PASSWORDS = {
    "password", "123456", "admin", "root", "changeme",
    "postgres", "mongo", "default", "test", "secret",
}


def validate_env():
    """Validate that all required environment variables are set and not weak.

    Raises:
        EnvironmentError: If a required variable is missing or uses a placeholder.
    """
    missing = []
    for var, description in REQUIRED_SECRETS.items():
        value = os.getenv(var)
        if not value or value.startswith("your_") or value.endswith("_here"):
            missing.append(f"  - {var}: {description}")

    if missing:
        raise EnvironmentError(
            "Missing or unconfigured environment variables:\n"
            + "\n".join(missing)
            + "\n\nCopy .env.example to .env and fill in the values."
        )

    # Warn about weak passwords
    pg_password = os.getenv("POSTGRES_PASSWORD", "")
    if pg_password.lower() in WEAK_PASSWORDS:
        logger.warning(
            "POSTGRES_PASSWORD is weak ('%s'). Use a strong password in production.",
            pg_password,
        )

    logger.info("Environment validation passed")


def sanitize_text(text: str) -> str:
    """Sanitize text input to prevent injection attacks.

    Escapes HTML entities and removes control characters.

    Args:
        text: Raw text input.

    Returns:
        Sanitized text.
    """
    if not isinstance(text, str):
        return str(text)

    # Remove control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Escape HTML to prevent XSS
    text = escape(text)

    return text.strip()


def sanitize_url(url: str) -> str:
    """Validate and sanitize a URL.

    Args:
        url: Raw URL string.

    Returns:
        Sanitized URL.

    Raises:
        ValueError: If URL scheme is not http/https.
    """
    url = url.strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError(f"Invalid URL scheme (must be http/https): {url}")

    # Remove any embedded credentials
    url = re.sub(r"https?://[^@]+@", lambda m: m.group(0).split("//")[0] + "//", url)

    return url


def validate_crypto_id(crypto_id: str) -> str:
    """Validate a cryptocurrency ID (alphanumeric + hyphens only).

    Args:
        crypto_id: CoinGecko cryptocurrency identifier.

    Returns:
        Validated crypto ID.

    Raises:
        ValueError: If ID contains invalid characters.
    """
    if not re.match(r"^[a-z0-9\-]+$", crypto_id):
        raise ValueError(
            f"Invalid crypto ID: '{crypto_id}'. Only lowercase letters, numbers, and hyphens allowed."
        )
    return crypto_id


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret value for safe logging.

    Args:
        value: Secret value to mask.
        visible_chars: Number of characters to show at the end.

    Returns:
        Masked string (e.g., '****xy2z').
    """
    if not value or len(value) <= visible_chars:
        return "****"
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]
