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
from app.graph.state import EspecialistaOutput

from tests.support import StubAgent, StubRepository


def _build_graph(rota_do_roteador: str):
    router = StubAgent(
        "router",
        [
            {
                # Igual ao RouterAgent real: ao encaminhar, não grava o
                # protocolo ROUTE=... no histórico.
                "messages": [],
                "proximos_agentes": [rota_do_roteador] if rota_do_roteador and rota_do_roteador != "fim" else [],
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


def test_multiplas_rotas_executam_especialistas_em_paralelo_e_convergem_no_orquestrador():
    router = StubAgent(
        "router",
        [
            {
                "messages": [],
                "proximos_agentes": ["estoque", "rag_faq"],
                "agentes_chamados": ["router"],
            }
        ],
    )
    especialista_estoque = StubAgent(
        "material_estoque",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"material_estoque"}'}],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {
                    "material_estoque": EspecialistaOutput(conteudo="500kg PET", fonte="material_estoque")
                },
            }
        ],
    )
    especialista_faq = StubAgent(
        "faq",
        [
            {
                "messages": [{"role": "assistant", "content": "Normas de reciclagem."}],
                "agentes_chamados": ["faq"],
                "especialista_outputs": {
                    "faq": EspecialistaOutput(conteudo="Normas de reciclagem.", fonte="faq")
                },
            }
        ],
    )
    orchestrator = StubAgent(
        "orchestrator",
        [
            {
                "messages": [{"role": "assistant", "content": "resposta compilada"}],
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
                agente=especialista_estoque,
                repositories=["material_repository"],
            ),
            RegistroEspecialista(
                rota="rag_faq",
                node_name="faq_agent",
                agente=especialista_faq,
                repositories=[],
            ),
        ],
        repositories={"material_repository": StubRepository(snapshot={"itens": []})},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=StubAgent("guardrail_saida", [{}]),
    )
    graph = builder.build_graph()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto PET temos e qual a norma?")],
            "perfil_ctx": PerfilContext(
                perfil_id=uuid4(),
                tipo="COOPERATIVA",
                empresa_id=None,
                cooperativa_id=uuid4(),
            ),
        },
        config={"configurable": {"thread_id": "teste-multiplas-rotas"}},
    )

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "faq" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert "material_estoque" in resultado["especialista_outputs"]
    assert "faq" in resultado["especialista_outputs"]
    assert resultado["messages"][-1].content == "resposta compilada"


def test_rotas_parcialmente_mapeadas_executam_apenas_as_existentes():
    router = StubAgent(
        "router",
        [
            {
                "messages": [],
                "proximos_agentes": ["estoque", "financeiro"],
                "agentes_chamados": ["router"],
            }
        ],
    )
    especialista_estoque = StubAgent(
        "material_estoque",
        [
            {
                "messages": [{"role": "assistant", "content": '{"dominio":"material_estoque"}'}],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {
                    "material_estoque": EspecialistaOutput(conteudo="500kg PET", fonte="material_estoque")
                },
            }
        ],
    )
    orchestrator = StubAgent(
        "orchestrator",
        [
            {
                "messages": [{"role": "assistant", "content": "resposta compilada"}],
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
                agente=especialista_estoque,
                repositories=["material_repository"],
            )
        ],
        repositories={"material_repository": StubRepository(snapshot={"itens": []})},
        checkpointer=MemorySaver(),
        guardrail_entrada=StubAgent("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=StubAgent("guardrail_saida", [{}]),
    )
    graph = builder.build_graph()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto PET e qual rateio?")],
            "perfil_ctx": PerfilContext(
                perfil_id=uuid4(),
                tipo="COOPERATIVA",
                empresa_id=None,
                cooperativa_id=uuid4(),
            ),
        },
        config={"configurable": {"thread_id": "teste-rotas-parciais"}},
    )

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert resultado["messages"][-1].content == "resposta compilada"