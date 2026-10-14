import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from pydantic import BaseModel

from app.core.config import Settings
from app.graph.state import EspecialistaOutput
from app.repository.postgresql.perfil_repository import PerfilContext

log = logging.getLogger(__name__)

# Modelos pydantic que vivem dentro do GraphState.
TIPOS_DO_ESTADO: tuple[type[BaseModel], ...] = (PerfilContext, EspecialistaOutput)


def _id_lc(cls: type) -> tuple[str, ...]:
    """Mesmo formato do campo 'id' do envelope: módulo quebrado em partes + nome."""
    return (*cls.__module__.split('.'), cls.__name__)


class RenovaiRedisSerializer(JsonPlusRedisSerializer):
    """JsonPlusRedisSerializer que reconstrói os modelos pydantic do GraphState.

    Só as classes de TIPOS_DO_ESTADO são reconstruídas (sem import dinâmico
    de nada que venha do Redis); qualquer outro envelope segue o fluxo
    normal do serializador do pacote.
    """

    def __init__(self, tipos: tuple[type[BaseModel], ...] = TIPOS_DO_ESTADO):
        super().__init__()
        self._tipos_por_id = {_id_lc(t): t for t in tipos}

    def _revive_if_needed(self, obj: Any) -> Any:
        if (
            isinstance(obj, dict)
            and obj.get('lc') in (1, 2)
            and obj.get('type') == 'constructor'
        ):
            cls = self._tipos_por_id.get(tuple(obj.get('id') or ()))
            if cls is not None:
                # pydantic converte de volta os tipos que viraram string (UUID etc.)
                kwargs = {
                    k: self._revive_if_needed(v)
                    for k, v in (obj.get('kwargs') or {}).items()
                }
                return cls.model_validate(kwargs)
        return super()._revive_if_needed(obj)


def criar_serializer() -> RenovaiRedisSerializer:
    return RenovaiRedisSerializer()


# Patch idempotente no JsonPlusRedisSerializer e no BaseRedisSaver para que
# QUALQUER instância de AsyncShallowRedisSaver reconstrua os modelos pydantic do GraphState
_orig_jsonplus_revive = JsonPlusRedisSerializer._revive_if_needed

def _patched_jsonplus_revive(self: Any, obj: Any) -> Any:
    if (
        isinstance(obj, dict)
        and obj.get('lc') in (1, 2)
        and obj.get('type') == 'constructor'
    ):
        tipos_map = getattr(self, '_tipos_por_id', None)
        if tipos_map is None:
            tipos_map = {_id_lc(t): t for t in TIPOS_DO_ESTADO}
        cls = tipos_map.get(tuple(obj.get('id') or ()))
        if cls is not None:
            kwargs = {
                k: self._revive_if_needed(v)
                for k, v in (obj.get('kwargs') or {}).items()
            }
            return cls.model_validate(kwargs)
    return _orig_jsonplus_revive(self, obj)

if not getattr(JsonPlusRedisSerializer, '_renovai_patched', False):
    JsonPlusRedisSerializer._revive_if_needed = _patched_jsonplus_revive
    JsonPlusRedisSerializer._renovai_patched = True

_orig_base_redis_init = BaseRedisSaver.__init__

def _patched_base_redis_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _orig_base_redis_init(self, *args, **kwargs)
    self.serde = criar_serializer()

if not getattr(BaseRedisSaver, '_renovai_patched', False):
    BaseRedisSaver.__init__ = _patched_base_redis_init
    BaseRedisSaver._renovai_patched = True


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
        # O BaseRedisSaver já inicializa com criar_serializer(), garantimos explicitamente aqui.
        saver.serde = criar_serializer()
        await saver.asetup()
        yield saver
