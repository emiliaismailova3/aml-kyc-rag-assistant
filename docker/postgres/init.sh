#!/bin/bash
# Runs once, the first time the Postgres container initialises an empty data directory.
# Creates the pgvector extension and a read-only role used by the agent's SQL tool.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE ROLE assistant_ro LOGIN PASSWORD '${POSTGRES_RO_PASSWORD:-assistant_ro}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO assistant_ro;
    GRANT USAGE ON SCHEMA public TO assistant_ro;
    -- No default access to tables: src/db.py init_db() grants SELECT on the two views only.
SQL
