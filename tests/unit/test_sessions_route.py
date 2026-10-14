"""Testes da rota POST /sessions/{session_id}/encerrar — sem Mongo, Qdrant ou Redis.

Cobre:
- sessão encerrada: grava o resumo no Qdrant e apaga o estado curto do
  thread no checkpointer (adelete_thread);
- sessão sem resumo: não chama o Qdrant, mas ainda limpa o checkpointer;
- sessão inexistente: 404 e o checkpointer NÃO é tocado.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import sessions


def _montar_app(sessao_encerrada):
    mongo_memory = MagicMock()
    mongo_memory.encerrar_sessao.return_value = sessao_encerrada
    qdrant_memory = MagicMock()
    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock()

    app = FastAPI()
    app.include_router(sessions.router)
    app.state.container = SimpleNamespace(
        memory_stores={'mongo_memory': mongo_memory, 'qdrant_memory': qdrant_memory},
        checkpointer=checkpointer,
    )
    return app, mongo_memory, qdrant_memory, checkpointer


@pytest.fixture
def sessao_com_resumo():
    return SimpleNamespace(resumo='Cooperado consultou o estoque de PET.', agentes_chamados=['router', 'material_estoque'])


def test_encerrar_grava_resumo_e_limpa_checkpointer(sessao_com_resumo):
    app, mongo_memory, qdrant_memory, checkpointer = _montar_app(sessao_com_resumo)

    with TestClient(app) as client:
        response = client.post('/sessions/sessao-123/encerrar')

    assert response.status_code == 200
    corpo = response.json()
    assert corpo['session_id'] == 'sessao-123'
    assert corpo['resumo'] == sessao_com_resumo.resumo
    assert corpo['agentes_chamados'] == ['router', 'material_estoque']

    mongo_memory.encerrar_sessao.assert_called_once_with('sessao-123')
    qdrant_memory.atualizar_memoria.assert_called_once_with(sessao_com_resumo)
    checkpointer.adelete_thread.assert_awaited_once_with('sessao-123')


def test_encerrar_sem_resumo_nao_grava_no_qdrant_mas_limpa_checkpointer():
    sessao = SimpleNamespace(resumo='', agentes_chamados=['router'])
    app, _, qdrant_memory, checkpointer = _montar_app(sessao)

    with TestClient(app) as client:
        response = client.post('/sessions/sessao-456/encerrar')

    assert response.status_code == 200
    qdrant_memory.atualizar_memoria.assert_not_called()
    checkpointer.adelete_thread.assert_awaited_once_with('sessao-456')


def test_encerrar_sessao_inexistente_devolve_404_sem_tocar_no_checkpointer():
    app, _, qdrant_memory, checkpointer = _montar_app(None)

    with TestClient(app) as client:
        response = client.post('/sessions/nao-existe/encerrar')

    assert response.status_code == 404
    qdrant_memory.atualizar_memoria.assert_not_called()
    checkpointer.adelete_thread.assert_not_awaited()
