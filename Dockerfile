# AML/KYC Knowledge Assistant -- single image, used for both the FastAPI
# backend and the Streamlit UI (see docker-compose.yml for how the two
# services share it with different entrypoint commands).
FROM python:3.11-slim

WORKDIR /app

# System deps: build-essential for any package that needs to compile (e.g.
# tokenizers), curl for the healthcheck in docker-compose.yml.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU-only PyTorch build first: the default PyPI wheel bundles
# CUDA libraries (several GB) that this CPU-only embedding model never uses.
# sentence-transformers then finds torch already satisfied.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/raw/ data/raw/
COPY data/eval_questions.json data/agent_test_scenarios.json data/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Downloading the local embedding model at build time (rather than on first
# request) keeps container startup fast and avoids a network dependency
# at runtime for the embedding step.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
