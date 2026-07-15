# ─────────────────────────────────────────────
# Blue Zone Explorer — Dockerfile
# Target: Google Cloud Run
# ─────────────────────────────────────────────

FROM python:3.12.7-slim

# Prevent Python from writing .pyc files
# and buffer stdout/stderr for Cloud Run logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# System dependencies for geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code — data artifacts only, never raw/processed data
COPY src/ src/
COPY app/ app/
COPY data/artifacts/ data/artifacts/

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
