-- ============================================
-- Crypto Predict MLOps - PostgreSQL Schema
-- Phase 2: Stockage des donnees structurees
-- ============================================

-- Table des cryptomonnaies de reference
CREATE TABLE IF NOT EXISTS cryptocurrencies (
    id SERIAL PRIMARY KEY,
    coingecko_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Donnees de marche journalieres (prix, volume, market cap, OHLC)
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    crypto_id INTEGER NOT NULL REFERENCES cryptocurrencies(id),
    date DATE NOT NULL,
    price DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    total_volume DOUBLE PRECISION,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (crypto_id, date)
);

-- Fear & Greed Index
CREATE TABLE IF NOT EXISTS fear_greed_index (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    value INTEGER NOT NULL,
    classification VARCHAR(30) NOT NULL,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour les requetes frequentes
CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data (date);
CREATE INDEX IF NOT EXISTS idx_market_data_crypto_date ON market_data (crypto_id, date);
CREATE INDEX IF NOT EXISTS idx_fear_greed_date ON fear_greed_index (date);

-- Inserer les cryptomonnaies configurees
INSERT INTO cryptocurrencies (coingecko_id, symbol) VALUES
    ('bitcoin', 'BTC'),
    ('ethereum', 'ETH'),
    ('solana', 'SOL'),
    ('binancecoin', 'BNB'),
    ('cardano', 'ADA')
ON CONFLICT (coingecko_id) DO NOTHING;
