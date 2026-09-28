"""Testes de roteamento do GraphBuilder — sem nenhum LLM, com agentes-dublê
(tests/support.py:StubAgent).

Cobre:
- uma rota conhecida ('estoque', presente em _ROUTE_NODE_MAP via
  RegistroEspecialista) chega no especialista certo e depois no orquestrador;
- uma rota que o roteador conhece (está no prompt) mas que ainda não tem nó
  registrado (ex.: 'financeiro') cai direto em END — é exatamente o
  mecanismo por trás do bug documentado em test_router_agent.py: a resposta
  crua do roteador (com o protocolo ROUTE=...) vira a "resposta final",
  porque não existe nó de especialista pra essa rota ainda.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

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
                "messages": [{"role": "assistant", "content": f"ROUTE={rota_do_roteador}"}],
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
    )
    return builder.build_graph()


def test_rota_conhecida_chega_no_especialista_e_no_orquestrador():
    graph = _build_graph("estoque")
    cooperativa_id = uuid4()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto de papelão temos?")],
            "perfil_ctx": SimpleNamespace(cooperativa_id=cooperativa_id),
        },
        config={"configurable": {"thread_id": "teste-rota-conhecida"}},
    )

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert resultado["messages"][-1].content == "resposta final formatada"


def test_rota_nao_mapeada_cai_direto_em_end():
    graph = _build_graph("financeiro")

    resultado = graph.invoke(
        {"messages": [HumanMessage(content="qual o rateio desse mês?")]},
        config={"configurable": {"thread_id": "teste-rota-desconhecida"}},
    )

    # nenhum especialista nem orquestrador rodou — a "resposta final" é o
    # protocolo cru do roteador, o mesmo efeito descrito no bug ativo.
    assert resultado["agentes_chamados"] == ["router"]
    assert "ROUTE=financeiro" in resultado["messages"][-1].content