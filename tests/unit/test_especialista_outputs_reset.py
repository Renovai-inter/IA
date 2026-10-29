"""Testes do reset de `especialista_outputs` entre turnos (app/graph/state.py).

Com o checkpointer persistido (Redis), o estado do thread sobrevive entre
requisições. Sem o reset, as saídas de especialistas de um turno anterior
continuariam no estado e chegariam ao Orquestrador no turno seguinte.
O /chat manda `{RESET_OUTPUTS: True}` na entrada de cada turno.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.graph.state import (
    RESET_OUTPUTS,
    RESET_AGENTES,
    EspecialistaOutput,
    merge_especialista_outputs,
    merge_agentes_chamados,
)

from tests.support import StubAgent


# ------------------------------------------------------------------------------
# Reducer isolado
# ------------------------------------------------------------------------------
def test_merge_acumula_saidas_sem_reset():
    a = EspecialistaOutput(conteudo='a')
    b = EspecialistaOutput(conteudo='b')

    assert merge_especialista_outputs({'faq': a}, {'material_estoque': b}) == {
        'faq': a,
        'material_estoque': b,
    }


def test_merge_com_sentinela_zera_o_estado():
    a = EspecialistaOutput(conteudo='a')

    assert merge_especialista_outputs({'faq': a}, {RESET_OUTPUTS: True}) == {}


def test_merge_tolera_none():
    a = EspecialistaOutput(conteudo='a')

    assert merge_especialista_outputs(None, {'faq': a}) == {'faq': a}
    assert merge_especialista_outputs({'faq': a}, None) == {'faq': a}


def test_merge_agentes_chamados_acumula_sem_reset():
    assert merge_agentes_chamados(['router'], ['material_estoque']) == ['router', 'material_estoque']


def test_merge_agentes_chamados_com_reset_descarta_anteriores():
    assert merge_agentes_chamados(['router', 'material_estoque'], [RESET_AGENTES]) == []
    assert merge_agentes_chamados(['router', 'material_estoque'], [RESET_AGENTES, 'guardrail_entrada']) == ['guardrail_entrada']


# ------------------------------------------------------------------------------
# Grafo com checkpointer: dois turnos no mesmo thread
# ------------------------------------------------------------------------------
def _build_graph():
    router = StubAgent(
        'router',
        [{'messages': [], 'proximos_agentes': ['estoque'], 'agentes_chamados': ['router']}],
    )
    # 1ª chamada grava a saída como 'faq', 2ª como 'material_estoque': assim dá
    # pra ver se a saída do turno 1 "vazou" para o turno 2.
    especialista = StubAgent(
        'material_estoque',
        [
            {
                'messages': [{'role': 'assistant', 'content': 'turno 1'}],
                'agentes_chamados': ['material_estoque'],
                'especialista_outputs': {'faq': EspecialistaOutput(conteudo='turno 1')},
            },
            {
                'messages': [{'role': 'assistant', 'content': 'turno 2'}],
                'agentes_chamados': ['material_estoque'],
                'especialista_outputs': {'material_estoque': EspecialistaOutput(conteudo='turno 2')},
            },
        ],
    )
    orchestrator = StubAgent(
        'orchestrator',
        [{'messages': [{'role': 'assistant', 'content': 'final'}], 'agentes_chamados': ['orchestrator']}],
    )

    return GraphBuilder(
        router_agent=router,
        orchestrator_agent=orchestrator,
        specialists=[
            RegistroEspecialista(
                rota='estoque',
                node_name='material_estoque_agent',
                agente=especialista,
                repositories=(),  # sem repository: não depende de perfil de cooperativa
            )
        ],
        repositories={},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent('guardrail_entrada', [{'passou_guardrail': True}]),
        guardrail_saida=StubAgent('guardrail_saida', [{}]),
        judge_agent=StubAgent('judge', [{'passou_juiz': True, 'agentes_chamados': ['judge:apoiada noul=0.95']}]),
    ).build_graph()


def _turno(graph, texto: str, thread_id: str, resetar: bool) -> dict:
    entrada = {'messages': [HumanMessage(content=texto)]}
    if resetar:
        entrada['especialista_outputs'] = {RESET_OUTPUTS: True}
    return graph.invoke(entrada, config={'configurable': {'thread_id': thread_id}})


def test_reset_no_inicio_do_turno_descarta_saidas_do_turno_anterior():
    graph = _build_graph()

    _turno(graph, 'pergunta 1', 'thread-reset', resetar=True)
    resultado = _turno(graph, 'pergunta 2', 'thread-reset', resetar=True)

    assert set(resultado['especialista_outputs']) == {'material_estoque'}
    assert RESET_OUTPUTS not in resultado['especialista_outputs']


def test_sem_reset_as_saidas_acumulam_entre_turnos():
    """Documenta o bug que o reset corrige."""
    graph = _build_graph()

    _turno(graph, 'pergunta 1', 'thread-sem-reset', resetar=False)
    resultado = _turno(graph, 'pergunta 2', 'thread-sem-reset', resetar=False)

    assert set(resultado['especialista_outputs']) == {'faq', 'material_estoque'}


def test_historico_de_mensagens_continua_acumulando_com_reset():
    graph = _build_graph()

    _turno(graph, 'pergunta 1', 'thread-historico', resetar=True)
    resultado = _turno(graph, 'pergunta 2', 'thread-historico', resetar=True)

    conteudos = [m.content for m in resultado['messages']]
    assert 'pergunta 1' in conteudos and 'pergunta 2' in conteudos
