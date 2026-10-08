FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /bin/uv
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/.venv /app/.venv
COPY app ./app
COPY data/FAQ_RENOVAI_COMPLETO.txt ./data/FAQ_RENOVAI_COMPLETO.txt
USER 10001:10001
EXPOSE 8000
# ponytail: MemorySaver é local; persistir checkpoints antes de adicionar workers/réplicas.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
