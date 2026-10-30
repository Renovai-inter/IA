import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
import dataclasses

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from pydantic import BaseModel
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
from redis.retry import Retry as SyncRetry

from app.core.config import Settings
from app.graph.state import EspecialistaOutput
from app.repository.postgresql.perfil_repository import PerfilContext

log = logging.getLogger(__name__)

# Modelos do GraphState (Pydantic ou dataclass).
TIPOS_DO_ESTADO: tuple[type, ...] = (PerfilContext, EspecialistaOutput)


def _id_lc(cls: type) -> tuple[str, ...]:
    """Mesmo formato do campo 'id' do envelope: módulo quebrado em partes + nome."""
    return (*cls.__module__.split('.'), cls.__name__)


def _reconstruir_objeto(cls: type, kwargs: dict[str, Any]) -> Any:
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return cls.model_validate(kwargs)
    if dataclasses.is_dataclass(cls):
        campos = {f.name for f in dataclasses.fields(cls)}
        kwargs_filtrados = {k: v for k, v in kwargs.items() if k in campos}
        return cls(**kwargs_filtrados)
    return cls(**kwargs)


class RenovaiRedisSerializer(JsonPlusRedisSerializer):
    """JsonPlusRedisSerializer que reconstrói os modelos do GraphState.

    Só as classes de TIPOS_DO_ESTADO são reconstruídas (sem import dinâmico
    de nada que venha do Redis); qualquer outro envelope segue o fluxo
    normal do serializador do pacote.
    """

    def __init__(self, tipos: tuple[type, ...] = TIPOS_DO_ESTADO):
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
                kwargs = {
                    k: self._revive_if_needed(v)
                    for k, v in (obj.get('kwargs') or {}).items()
                }
                return _reconstruir_objeto(cls, kwargs)
        return super()._revive_if_needed(obj)


def _criar_serializer() -> RenovaiRedisSerializer:
    return RenovaiRedisSerializer()


criar_serializer = _criar_serializer


# Patch idempotente no JsonPlusRedisSerializer e no BaseRedisSaver para que
# QUALQUER instância de AsyncShallowRedisSaver reconstrua os modelos do GraphState
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
            return _reconstruir_objeto(cls, kwargs)
    return _orig_jsonplus_revive(self, obj)

if not getattr(JsonPlusRedisSerializer, '_renovai_patched', False):
    JsonPlusRedisSerializer._revive_if_needed = _patched_jsonplus_revive
    JsonPlusRedisSerializer._renovai_patched = True

def _obter_connection_args_resilientes(is_async: bool = True, user_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retorna argumentos de conexão resilientes para o pool do Redis.

    Evita que conexões ociosas caiam silenciosamente (WinError 10054 / ConnectionResetError)
    em Redis remoto/cloud, habilitando TCP keepalive, verificação periódica de saúde (health check)
    e retentativas automáticas com backoff exponencial.
    """
    retry_obj = (
        AsyncRetry(ExponentialBackoff(cap=5, base=1), 3)
        if is_async
        else SyncRetry(ExponentialBackoff(cap=5, base=1), 3)
    )
    args: dict[str, Any] = {
        'health_check_interval': 30,
        'socket_keepalive': True,
        'retry_on_timeout': True,
        'retry_on_error': [RedisConnectionError, RedisTimeoutError, ConnectionResetError],
        'retry': retry_obj,
    }
    if user_args:
        args.update(user_args)
    return args


_orig_base_redis_init = BaseRedisSaver.__init__


def _patched_base_redis_init(self: Any, *args: Any, **kwargs: Any) -> None:
    is_async = isinstance(self, AsyncShallowRedisSaver) or hasattr(self, 'aget_tuple')
    conn_args = kwargs.get('connection_args')
    kwargs['connection_args'] = _obter_connection_args_resilientes(is_async=is_async, user_args=conn_args)
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
    conn_args = _obter_connection_args_resilientes(is_async=True)
    async with AsyncShallowRedisSaver.from_conn_string(
        settings.REDIS_URL,
        connection_args=conn_args,
        ttl=ttl,
    ) as saver:
        # O BaseRedisSaver já inicializa com criar_serializer(), garantimos explicitamente aqui.
        saver.serde = criar_serializer()
        await saver.asetup()
        yield saver
