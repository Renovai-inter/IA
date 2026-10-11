"""Testes de execução assíncrona (arun e ainvoke) para todos os agentes e nós do grafo.

Verifica:
- RouterAgent.arun(): extração de rota e preservação do protocolo/histórico;
- OrchestratorAgent.arun(): síntese e formatação de saídas dos especialistas;
- MaterialEstoqueAgent.arun() e FaqAgent.arun(): execução assíncrona e empacotamento de EspecialistaOutput;
- GuardrailEntradaAgent.arun() e GuardrailSaidaAgent.arun(): avaliação determinística e assíncrona de LLM;
- GraphBuilder.ainvoke(): execução assíncrona de ponta a ponta com múltiplos especialistas concorrentes.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.faq_agent import FaqAgent
from app.agents.guardrail_agent import GuardrailEntradaAgent, GuardrailSaidaAgent
from app.agents.material_estoque_agent import MaterialEstoqueAgent
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.router_agent import RouterAgent
from app.graph.builder import GraphBuilder
from app.graph.registro import RegistroEspecialista
from app.graph.state import EspecialistaOutput
from app.repository.postgresql.perfil_repository import PerfilContext
from tests.support import StubAgent, StubRepository


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {"messages": [AIMessage(content=self._texto)]}

    async def ainvoke(self, inputs, config=None):
        return self.invoke(inputs, config)


class _StubLLM:
    def __init__(self, resposta: str):
        self._resposta = resposta

    def invoke(self, messages, config=None):
        return AIMessage(content=self._resposta)

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)


@pytest.mark.asyncio
async def test_router_agent_arun_extrai_multiplas_rotas(monkeypatch: pytest.MonkeyPatch):
    import app.agents.router_agent as router_module

    stub = _StubRunnable("ROUTES=estoque, rag_faq\nPERGUNTA_ORIGINAL=quanto de PET temos?")
    monkeypatch.setattr(router_module, "create_agent", lambda **kwargs: stub)

    agent = RouterAgent(llm=None, system_prompt="prompt", tools=[])
    resultado = await agent.arun({"messages": [HumanMessage(content="quanto de PET temos?")]}, config={})

    assert resultado["proximos_agentes"] == ["estoque", "rag_faq"]
    assert resultado["messages"] == []
    assert resultado["agentes_chamados"] == ["router"]


@pytest.mark.asyncio
async def test_orchestrator_agent_arun():
    llm = _StubLLM("Aqui está o consolidado: temos 500kg de PET.")
    agent = OrchestratorAgent(llm=llm, system_prompt="prompt compilador")

    resultado = await agent.arun(
        {
            "messages": [HumanMessage(content="quanto PET temos?")],
            "especialista_outputs": {
                "material_estoque": EspecialistaOutput(conteudo="500kg PET", fonte="material_estoque")
            },
        },
        config={},
    )

    assert "orchestrator" in resultado["agentes_chamados"]
    assert resultado["messages"][-1]["content"] == "Aqui está o consolidado: temos 500kg de PET."


@pytest.mark.asyncio
async def test_material_estoque_agent_arun(monkeypatch: pytest.MonkeyPatch):
    import app.agents.material_estoque_agent as mat_module

    stub = _StubRunnable("Temos 300kg de alumínio.")
    monkeypatch.setattr(mat_module, "create_agent", lambda **kwargs: stub)

    agent = MaterialEstoqueAgent(llm=None, system_prompt="prompt", tools=[])
    resultado = await agent.arun(
        {
            "messages": [HumanMessage(content="quanto de alumínio?")],
            "snapshots": {"material_snapshot": []},
        },
        config={},
    )

    assert resultado["agentes_chamados"] == ["material_estoque"]
    assert "material_estoque" in resultado["especialista_outputs"]
    assert resultado["especialista_outputs"]["material_estoque"].conteudo == "Temos 300kg de alumínio."


@pytest.mark.asyncio
async def test_faq_agent_arun(monkeypatch: pytest.MonkeyPatch):
    import app.agents.faq_agent as faq_module

    stub = _StubRunnable("A norma ISO 14001 orienta a gestão ambiental.")
    monkeypatch.setattr(faq_module, "create_agent", lambda **kwargs: stub)

    agent = FaqAgent(llm=None, system_prompt="prompt", tools=[])
    resultado = await agent.arun(
        {"messages": [HumanMessage(content="qual norma seguir?")]},
        config={},
    )

    assert resultado["agentes_chamados"] == ["faq"]
    assert "faq" in resultado["especialista_outputs"]
    assert resultado["especialista_outputs"]["faq"].conteudo == "A norma ISO 14001 orienta a gestão ambiental."


@pytest.mark.asyncio
async def test_guardrail_entrada_agent_arun():
    llm = _StubLLM("CATEGORIA=SEGURO")
    agent = GuardrailEntradaAgent(llm=llm, system_prompt="prompt guardrail")

    # Pergunta legítima: passa pelo determinístico e LLM
    resultado = await agent.arun(
        {"messages": [HumanMessage(content="quanto de papelão temos no estoque?")]},
        config={},
    )

    assert resultado["passou_guardrail"] is True
    assert "guardrail_entrada" in resultado["agentes_chamados"][0]


@pytest.mark.asyncio
async def test_guardrail_saida_agent_arun():
    agent = GuardrailSaidaAgent(llm=None, system_prompt="prompt guardrail saída")

    resultado = await agent.arun(
        {"messages": [AIMessage(content="Resposta sem PII.")]},
        config={},
    )

    assert "guardrail_saida:ok" in resultado["agentes_chamados"]


@pytest.mark.asyncio
async def test_graph_ainvoke_multiplas_rotas_em_paralelo():
    # Cria agentes dublês com pequeno atraso para confirmar execução assíncrona
    class AsyncDelayStub(StubAgent):
        async def arun(self, state: dict, config = None) -> dict:
            await asyncio.sleep(0.05)
            return self.run(state, config)

    router = AsyncDelayStub(
        "router",
        [
            {
                "messages": [],
                "proximos_agentes": ["estoque", "rag_faq"],
                "agentes_chamados": ["router"],
            }
        ],
    )
    especialista_estoque = AsyncDelayStub(
        "material_estoque",
        [
            {
                "messages": [{"role": "assistant", "content": "500kg PET"}],
                "agentes_chamados": ["material_estoque"],
                "especialista_outputs": {
                    "material_estoque": EspecialistaOutput(conteudo="500kg PET", fonte="material_estoque")
                },
            }
        ],
    )
    especialista_faq = AsyncDelayStub(
        "faq",
        [
            {
                "messages": [{"role": "assistant", "content": "Norma 123"}],
                "agentes_chamados": ["faq"],
                "especialista_outputs": {
                    "faq": EspecialistaOutput(conteudo="Norma 123", fonte="faq")
                },
            }
        ],
    )
    orchestrator = AsyncDelayStub(
        "orchestrator",
        [
            {
                "messages": [{"role": "assistant", "content": "Resposta final compilada."}],
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
        guardrail_entrada=AsyncDelayStub("guardrail_entrada", [{"passou_guardrail": True}]),
        guardrail_saida=AsyncDelayStub("guardrail_saida", [{}]),
    )
    graph = builder.build_graph()

    resultado = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quanto PET temos e qual a norma?")],
            "perfil_ctx": PerfilContext(
                perfil_id=uuid4(),
                tipo="COOPERATIVA",
                empresa_id=None,
                cooperativa_id=uuid4(),
            ),
        },
        config={"configurable": {"thread_id": "teste-async-ainvoke"}},
    )

    assert "material_estoque" in resultado["agentes_chamados"]
    assert "faq" in resultado["agentes_chamados"]
    assert "orchestrator" in resultado["agentes_chamados"]
    assert resultado["messages"][-1].content == "Resposta final compilada."
