-- ============================================
-- Phase 4: Securite - Roles et privileges PostgreSQL
-- ============================================

-- Role lecture seule (pour dashboards, analyses)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_readonly') THEN
        CREATE ROLE app_readonly WITH LOGIN PASSWORD 'readonly_pwd';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE crypto_predict TO app_readonly;
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO app_readonly;

-- Role lecture/ecriture (pour les collecteurs et ETL)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_readwrite') THEN
        CREATE ROLE app_readwrite WITH LOGIN PASSWORD 'readwrite_pwd';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE crypto_predict TO app_readwrite;
GRANT USAGE ON SCHEMA public TO app_readwrite;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO app_readwrite;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO app_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app_readwrite;

-- Revoquer les privileges publics par defaut
REVOKE ALL ON DATABASE crypto_predict FROM PUBLIC;
