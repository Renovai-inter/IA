"""Integração do checkpointer com um Redis REAL (o REDIS_URL do seu .env).

Pulado automaticamente quando não há REDIS_URL (ex.: no CI sem secret).
Usa um thread_id único por teste e apaga tudo ao final (adelete_thread);
TTL curto como segurança caso o teste quebre no meio.

Cobre o que os testes unitários não conseguem: o estado faz a ida e volta
pelo Redis (serialização), incluindo os modelos pydantic do GraphState
(PerfilContext, EspecialistaOutput).
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.graph.state import RESET_OUTPUTS, EspecialistaOutput, GraphState
from app.repository.postgresql.perfil_repository import PerfilContext

_ENV_FILE = Path(__file__).resolve().parents[2] / '.env'
REDIS_URL = os.getenv('REDIS_URL') or dotenv_values(_ENV_FILE).get('REDIS_URL') or ''

pytestmark = pytest.mark.skipif(not REDIS_URL, reason='REDIS_URL não definida (env ou .env)')


def _grafo_eco():
    """Grafo mínimo: um nó que responde e grava uma saída de especialista."""
    def eco(state: GraphState) -> dict:
        n = len([m for m in state['messages'] if m.type == 'human'])
        return {
            'messages': [{'role': 'assistant', 'content': f'resposta {n}'}],
            'agentes_chamados': ['eco'],
            'especialista_outputs': {f'eco_{n}': EspecialistaOutput(conteudo=f'saida {n}', fonte='teste')},
        }

    g = StateGraph(GraphState)
    g.add_node('eco', eco)
    g.set_entry_point('eco')
    g.add_edge('eco', END)
    return g


@pytest_asyncio.fixture
async def saver():
    from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver

    async with AsyncShallowRedisSaver.from_conn_string(REDIS_URL, ttl={'default_ttl': 5}) as s:
        await s.asetup()
        yield s


@pytest_asyncio.fixture
async def thread_id(saver):
    tid = f'pytest-checkpointer-{uuid4()}'
    yield tid
    await saver.adelete_thread(tid)


@pytest.mark.asyncio
async def test_estado_sobrevive_entre_turnos(saver, thread_id):
    grafo = _grafo_eco().compile(checkpointer=saver)
    cfg = {'configurable': {'thread_id': thread_id}}
    perfil = PerfilContext(perfil_id=uuid4(), tipo='COOPERATIVA', empresa_id=None, cooperativa_id=uuid4())

    await grafo.ainvoke({'messages': [HumanMessage('oi')], 'perfil_ctx': perfil}, cfg)
    resultado = await grafo.ainvoke({'messages': [HumanMessage('de novo')]}, cfg)

    assert [m.content for m in resultado['messages']] == ['oi', 'resposta 1', 'de novo', 'resposta 2']
    assert resultado['agentes_chamados'] == ['eco', 'eco']
    # pydantic volta como pydantic, com os mesmos valores
    assert resultado['perfil_ctx'] == perfil


@pytest.mark.asyncio
async def test_um_grafo_novo_le_o_estado_gravado_por_outro(saver, thread_id):
    """Simula duas réplicas/workers: cada uma compila o próprio grafo, mas
    ambas leem do mesmo Redis — é o que libera escalar o Deployment."""
    cfg = {'configurable': {'thread_id': thread_id}}

    replica_a = _grafo_eco().compile(checkpointer=saver)
    await replica_a.ainvoke({'messages': [HumanMessage('oi')]}, cfg)

    replica_b = _grafo_eco().compile(checkpointer=saver)
    estado = await replica_b.aget_state(cfg)

    assert [m.content for m in estado.values['messages']] == ['oi', 'resposta 1']


@pytest.mark.asyncio
async def test_reset_de_especialista_outputs_persiste(saver, thread_id):
    grafo = _grafo_eco().compile(checkpointer=saver)
    cfg = {'configurable': {'thread_id': thread_id}}

    await grafo.ainvoke({'messages': [HumanMessage('1')], 'especialista_outputs': {RESET_OUTPUTS: True}}, cfg)
    resultado = await grafo.ainvoke({'messages': [HumanMessage('2')], 'especialista_outputs': {RESET_OUTPUTS: True}}, cfg)

    assert list(resultado['especialista_outputs']) == ['eco_2']
    assert resultado['especialista_outputs']['eco_2'] == EspecialistaOutput(conteudo='saida 2', fonte='teste')


@pytest.mark.asyncio
async def test_adelete_thread_apaga_o_estado(saver, thread_id):
    grafo = _grafo_eco().compile(checkpointer=saver)
    cfg = {'configurable': {'thread_id': thread_id}}
    await grafo.ainvoke({'messages': [HumanMessage('oi')]}, cfg)

    await saver.adelete_thread(thread_id)

    estado = await grafo.aget_state(cfg)
    assert not estado.values
