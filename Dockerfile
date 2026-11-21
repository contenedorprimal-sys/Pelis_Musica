# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runner

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Dependencias del sistema minimas (libgomp1 requerido por OpenMP de scikit-learn/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias de Python (se copian primero para aprovechar cache de Docker)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Datos crudos, datos limpios, modelos y documento (autocontenido para produccion)
COPY data/ ./data/
COPY models/ ./models/
COPY docs/ ./docs/
COPY src/ ./src/
COPY app.py ./

# Configuracion de Streamlit (headless dentro del contenedor)
COPY .streamlit/config.toml ./.streamlit/config.toml

# Usuario no root
RUN useradd -m -s /bin/bash -u 1000 streamlit && \
    chown -R streamlit:streamlit /app
USER streamlit

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
