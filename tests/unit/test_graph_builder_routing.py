"""Testes de roteamento do GraphBuilder — sem nenhum LLM, com agentes-dublê
(tests/support.py:StubAgent).

Cobre:
- uma rota conhecida ('estoque', presente em _ROUTE_NODE_MAP via
  RegistroEspecialista) chega no especialista certo e depois no orquestrador;
- uma rota que o roteador conhece (está no prompt) mas que ainda não tem nó
  registrado (ex.: 'financeiro') cai em 'fim' sem resposta do assistente e
  sem vazar o protocolo cru ROUTE=...;
- perfil EMPRESA (sem cooperativa_id) não dispara snapshot de cooperativa.
"""
from __future__ import annotations

from uuid import uuid4

from app.repository.postgresql.perfil_repository import PerfilContext

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista

from tests.support import StubAgent, StubRepository


def _build_graph(rota_do_roteador: str):
    router = StubAgent(
        "router",
        [
            {
                # Igual ao RouterAgent real: ao encaminhar, não grava o
                # protocolo ROUTE=... no histórico.
                "messages": [],
                "proximo_agente": rota_do_roteador,
                "agentes_chamados": ["router"],
            }
        ],
    )
    especialista = StubAgent(
        "material_estoque",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"material_estoque"}'}],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {},
            }
        ],
    )
    orchestrator = StubAgent(
        "orchestrator",
        [
            {
                "messages": [{"role": "assistant", "content": "resposta final formatada"}],
                "agentes_chamados": ["orchestrator"],
            }
        ],
    )

    builder = GraphBuilder(
        router_agent=router,
        orchestrator_agent=orchestrator,
        specialists=[
            RegistroEspecialista(
                rota="estoque",
                node_name="material_estoque_agent",
                agente=especialista,
                repositories=["material_repository"],
            )
        ],
        repositories={"material_repository": StubRepository(snapshot={"itens": []})},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=StubAgent("guardrail_saida", [{}]),
    )
    return builder.build_graph()


def test_rota_conhecida_chega_no_especialista_e_no_orquestrador():
    graph = _build_graph("estoque")
    cooperativa_id = uuid4()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto de papelão temos?")],
            "perfil_ctx": PerfilContext(
                perfil_id=uuid4(),
                tipo="COOPERATIVA",
                empresa_id=None,
                cooperativa_id=cooperativa_id,
            ),
        },
        config={"configurable": {"thread_id": "teste-rota-conhecida"}},
    )

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert resultado["messages"][-1].content == "resposta final formatada"


def test_rota_nao_mapeada_cai_direto_em_fim():
    graph = _build_graph("financeiro")

    resultado = graph.invoke(
        {"messages": [HumanMessage(content="qual o rateio desse mês?")]},
        config={"configurable": {"thread_id": "teste-rota-desconhecida"}},
    )

    # nenhum especialista nem orquestrador rodou; o grafo encerra em 'fim'
    # sem resposta do assistente (mantém a pergunta), e sem vazar o protocolo.
    assert resultado["agentes_chamados"] == ["router"]
    resposta = resultado["messages"][-1].content
    assert "ROUTE=" not in resposta
    assert resposta == "qual o rateio desse mês?"


def test_perfil_empresa_nao_consulta_snapshot_de_cooperativa():
    graph = _build_graph("estoque")

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto de papelão temos?")],
            "perfil_ctx": PerfilContext(
                perfil_id=uuid4(),
                tipo="EMPRESA",
                empresa_id=uuid4(),
                cooperativa_id=None,
            ),
        },
        config={"configurable": {"thread_id": "teste-perfil-empresa"}},
    )

    # o nó do especialista responde direto (sem chamar o agente) e o
    # orquestrador formata em seguida.
    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]