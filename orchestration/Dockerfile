FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /bin/uv
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    MALLOC_ARENA_MAX=2 \
    PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root com diretório home explícito
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -d /home/appuser appuser

COPY --from=build /app/.venv /app/.venv
COPY app ./app
COPY frontend ./frontend
COPY data/FAQ_RENOVAI_COMPLETO.txt data/sp_bairros_zonas.csv data/sp_rotas_treinamento.csv ./data/

# Garante permissões adequadas para o appuser
RUN chown -R appuser:appuser /app /home/appuser

USER appuser
EXPOSE 8000
# ponytail: MemorySaver é local; persistir checkpoints antes de adicionar workers/réplicas.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
