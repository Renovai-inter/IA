"""Testes de app/core/checkpointer.py — sem Redis de verdade.

Cobre:
- sem REDIS_URL, cai no MemorySaver (dev local);
- com REDIS_URL, abre o AsyncShallowRedisSaver com o TTL configurado e
  chama asetup() antes de entregar o saver.

O comportamento contra um Redis real (ida e volta do estado) fica em
tests/integration/test_checkpointer_redis.py.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

import app.core.checkpointer as checkpointer_module
from app.core.checkpointer import TIPOS_DO_ESTADO, RenovaiRedisSerializer, abrir_checkpointer, criar_serializer
from app.graph.state import EspecialistaOutput
from app.repository.postgresql.perfil_repository import PerfilContext


@pytest.mark.asyncio
async def test_sem_redis_url_usa_memorysaver():
    settings = SimpleNamespace(REDIS_URL='', SESSAO_TTL_MIN=120)

    async with abrir_checkpointer(settings) as checkpointer:
        assert isinstance(checkpointer, MemorySaver)


@pytest.mark.asyncio
async def test_com_redis_url_abre_shallow_saver_com_ttl(monkeypatch: pytest.MonkeyPatch):
    saver = MagicMock()
    saver.asetup = AsyncMock()
    chamadas: dict = {}

    @asynccontextmanager
    async def from_conn_string_fake(url, **kwargs):
        chamadas['url'] = url
        chamadas['kwargs'] = kwargs
        yield saver
        chamadas['fechou'] = True

    monkeypatch.setattr(
        checkpointer_module,
        'AsyncShallowRedisSaver',
        SimpleNamespace(from_conn_string=from_conn_string_fake),
    )
    settings = SimpleNamespace(REDIS_URL='redis://default:senha@host:1234', SESSAO_TTL_MIN=45)

    async with abrir_checkpointer(settings) as checkpointer:
        assert checkpointer is saver
        saver.asetup.assert_awaited_once()
        # serializador trocado para o que reconstrói os modelos do estado
        assert isinstance(saver.serde, RenovaiRedisSerializer)

    assert chamadas['url'] == 'redis://default:senha@host:1234'
    assert chamadas['kwargs']['ttl'] == {'default_ttl': 45, 'refresh_on_read': True}
    assert chamadas.get('fechou') is True  # a conexão é fechada ao sair do lifespan


def test_tipos_do_estado_incluem_os_modelos_pydantic():
    assert PerfilContext in TIPOS_DO_ESTADO
    assert EspecialistaOutput in TIPOS_DO_ESTADO


def _envelope(id_lc: list[str], kwargs: dict) -> dict:
    """Formato em que o saver do Redis devolve um modelo pydantic lido do JSON."""
    return {'lc': 2, 'type': 'constructor', 'id': id_lc, 'kwargs': kwargs}


def test_serializer_reconstroi_perfil_context_com_uuid():
    perfil_id, coop_id = uuid4(), uuid4()
    envelope = _envelope(
        ['app', 'repository', 'postgresql', 'perfil_repository', 'PerfilContext'],
        {'perfil_id': str(perfil_id), 'tipo': 'COOPERATIVA', 'empresa_id': None, 'cooperativa_id': str(coop_id)},
    )

    revivido = criar_serializer()._revive_if_needed(envelope)

    assert revivido == PerfilContext(perfil_id=perfil_id, tipo='COOPERATIVA', empresa_id=None, cooperativa_id=coop_id)


def test_serializer_reconstroi_especialista_output_dentro_de_dict():
    canal = {
        'faq': _envelope(
            ['app', 'graph', 'state', 'EspecialistaOutput'],
            {'conteudo': 'texto', 'fonte': 'faq', 'veredito_juiz': 'pendente', 'feedback_juiz': None, 'tentativas': 0},
        )
    }

    revivido = criar_serializer()._revive_if_needed(canal)

    assert revivido == {'faq': EspecialistaOutput(conteudo='texto', fonte='faq')}


def test_serializer_nao_importa_classe_fora_da_lista():
    envelope = _envelope(['os', 'system'], {})

    revivido = criar_serializer()._revive_if_needed(envelope)

    assert not callable(revivido)  # nunca devolve a função os.system
