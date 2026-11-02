# Serving image: the Streamlit scoring dashboard (default) and the batch-scoring CLI.
#   docker build -t churn-scoring .
#   docker run --rm -p 8501:8501 churn-scoring
#   docker run --rm -v "$PWD/data:/data" churn-scoring \
#       python -m src.score --input /data/customers.csv --output /data/scored.csv
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime for LightGBM and CatBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app config/ config/
COPY --chown=app:app src/ src/
COPY --chown=app:app app/ app/
COPY --chown=app:app models/ models/
COPY --chown=app:app data/raw/ data/raw/
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
