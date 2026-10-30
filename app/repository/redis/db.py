from __future__ import annotations

import logging
from typing import Any

from redis import Redis

from app.core.checkpointer import _obter_connection_args_resilientes
from app.repository.base import ConnectionFactory

log = logging.getLogger("renovai.redis")


class RedisConnectionFactory(ConnectionFactory[Redis | None]):
    """Fábrica de conexões síncronas resilientes para o Redis (Rate Limiting, Fila e Ranking)."""

    def __init__(self, dsn: str):
        super().__init__(dsn)
        self._client: Redis | None = None

    def connect(self, dsn: str | None = None) -> Redis | None:
        target_dsn = (dsn if dsn is not None else self.dsn or "").strip()
        if not target_dsn:
            log.info("REDIS_URL ausente: RedisRealtimeRepository operará em modo fallback em memória.")
            return None
        try:
            conn_args: dict[str, Any] = _obter_connection_args_resilientes(is_async=False)
            self._client = Redis.from_url(
                target_dsn,
                decode_responses=True,
                **conn_args,
            )
            return self._client
        except Exception as exc:
            log.warning(f"Não foi possível inicializar cliente Redis ({exc}). Usando fallback em memória.")
            self._client = None
            return None

    def close(self, conn: Redis | None = None) -> None:
        alvo = conn or self._client
        if alvo is not None:
            try:
                alvo.close()
            except Exception:
                pass
