-- Subtext Database Initialization
-- PostgreSQL 15 with TimescaleDB extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE subtext TO subtext;
