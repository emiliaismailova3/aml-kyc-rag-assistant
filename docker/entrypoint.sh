#!/usr/bin/env bash
# Entrypoint for both the API and Streamlit UI containers (see docker-compose.yml).
# Builds the Chroma index on first run (if data/chroma_db doesn't exist yet,
# e.g. on a fresh volume), then starts the requested service.
set -euo pipefail

# Only the API reads the index; the UI just talks to the API over HTTP.
build_index_if_missing() {
    if [ ! -d "/app/data/chroma_db" ] || [ -z "$(ls -A /app/data/chroma_db 2>/dev/null)" ]; then
        echo "No existing Chroma index found -- building it from data/raw/ ..."
        python -m src.vectorstore
    fi
}

case "${1:-api}" in
    api)
        build_index_if_missing
        exec uvicorn src.api:app --host 0.0.0.0 --port 8000
        ;;
    ui)
        exec streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
        ;;
    *)
        echo "Unknown command: $1 (expected 'api' or 'ui')" >&2
        exit 1
        ;;
esac
