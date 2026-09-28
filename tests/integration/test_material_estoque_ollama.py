"""Fluxo ponta a ponta (router_agent -> material_estoque_agent ->
orchestrator_agent) com LLMs locais via Ollama, sem tocar em
Postgres/Mongo/Qdrant — os repositórios são falsos
(tests/integration/fakes.py) e só o LLM é real.

Isso é o mais perto de "rodar o projeto" que dá pra fazer sem subir toda a
infra (Postgres, Mongo, Qdrant) — útil pra validar prompts e o
comportamento do grafo iterando localmente, sem gastar quota do
Gemini/Groq a cada teste.

Observação: o MaterialEstoqueAgent usa as tools reais de
tools/material_estoque_tools.py (que leem os snapshots injetados por
make_repo_backed_node via config — mesmo mecanismo usado em produção).
Um modelo local pequeno pode não chamar a tool de forma tão confiável
quanto o Gemini HIGH usado em produção; por isso as asserções abaixo são
propositalmente tolerantes ao texto exato da resposta.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.material_estoque_agent import MaterialEstoqueAgent
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.router_agent import RouterAgent
from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.prompts import (
    MATERIAL_ESTOQUE_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO,
    ROUTER_PROMPT_COMPLETO,
)
from app.tools.material_estoque_tools import MaterialEstoqueToolkit

from tests.integration.fakes import cenario_papelao
from tests.integration.ollama_helpers import requires_ollama


@requires_ollama
def test_fluxo_estoque_com_ollama(ollama_llm_factory):
    cooperativa_id = uuid4()
    repositories = cenario_papelao(cooperativa_id)
    toolkit = MaterialEstoqueToolkit()

    router = RouterAgent(
        llm=ollama_llm_factory.get("OLLAMA", "LOW"),
        system_prompt=ROUTER_PROMPT_COMPLETO,
        tools=[],
    )
    especialista = MaterialEstoqueAgent(
        llm=ollama_llm_factory.get("OLLAMA", "HIGH"),
        system_prompt=MATERIAL_ESTOQUE_PROMPT_COMPLETO,
        tools=toolkit.get_tools(),
    )
    orchestrator = OrchestratorAgent(
        llm=ollama_llm_factory.get("OLLAMA", "LOW"),
        system_prompt=ORQUESTRADOR_PROMPT_COMPLETO,
        tools=[],
    )

    graph = GraphBuilder(
        router_agent=router,
        orchestrator_agent=orchestrator,
        specialists=[
            RegistroEspecialista(
                rota="estoque",
                node_name="material_estoque_agent",
                agente=especialista,
                repositories=[
                    "material_repository",
                    "estoque_repository",
                    "movimentacao_estoque_repository",
                ],
            )
        ],
        repositories=repositories,
        checkpointer=MemorySaver(),
    ).build_graph()

    resultado = graph.invoke(
        {
            "messages": [HumanMessage(content="quanto de papelão temos em estoque?")],
            "perfil_ctx": SimpleNamespace(cooperativa_id=cooperativa_id),
        },
        config={"configurable": {"thread_id": "integracao-estoque-ollama"}},
    )

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]

    resposta_final = resultado["messages"][-1].content
    assert resposta_final.strip()
    assert (
        "papel" in resposta_final.lower()
        or "3.450" in resposta_final
        or "3450" in resposta_final
    ), f"resposta não parece falar de papelão/quantidade: {resposta_final!r}"