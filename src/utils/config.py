"""Configuration loader for the project."""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.security import mask_secret, validate_env

logger = logging.getLogger(__name__)

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(ROOT_DIR / ".env")


def load_config() -> dict:
    """Load configuration from config/config.yaml."""
    config_path = ROOT_DIR / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_coingecko_api_key() -> str:
    """Get CoinGecko API key from environment."""
    key = os.getenv("COINGECKO_API_KEY")
    if not key or key == "CG-your_api_key_here":
        raise ValueError(
            "COINGECKO_API_KEY not set. Copy .env.example to .env and add your key."
        )
    logger.debug("Using CoinGecko API key: %s", mask_secret(key))
    return key


def load_config_secure() -> dict:
    """Load config and validate environment variables.

    Combines config file loading with env validation.
    Use this in production entrypoints.
    """
    validate_env()
    return load_config()
