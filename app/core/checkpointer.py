import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver

from app.core.config import Settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def abrir_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """Abre o checkpointer do grafo durante a vida da aplicação.
    Fallback: REDIS_URL (AsyncShallowRedisSaver) -> MemorySaver"""
    if not settings.REDIS_URL:
        log.warning('REDIS_URL ausente: usando MemorySaver (estado some no restart).')
        yield MemorySaver()
        return

    ttl = {'default_ttl': settings.SESSAO_TTL_MIN, 'refresh_on_read': True}
    async with AsyncShallowRedisSaver.from_conn_string(settings.REDIS_URL, ttl=ttl) as saver:
        await saver.asetup()
        yield saver