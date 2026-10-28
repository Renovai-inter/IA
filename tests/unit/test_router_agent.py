"""Testes do RouterAgent.run() — isolados de qualquer LLM real.

Técnica: monkeypatch de `create_agent` (tanto em app.agents.base quanto em
app.agents.router_agent, que importa e chama de novo no __init__) para
devolver um runnable-dublê com resposta fixa. Isso evita qualquer
dependência de bind_tools/tool-calling da versão instalada do LangChain —
testamos só a lógica de parsing do protocolo ROUTE=..., que é o que
realmente pertence ao RouterAgent.

Também cobre a correção do antigo vazamento do protocolo: o texto
ROUTE=.../PERGUNTA_ORIGINAL=... só entra no histórico de mensagens quando o
roteador responde direto ao usuário (rota 'fim'); quando encaminha, a lista
de mensagens volta vazia e as rotas seguem em `proximos_agentes`.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.router_agent as router_agent_module
from app.agents.router_agent import RouterAgent


class _StubRunnable:
    def __init__(self, texto_resposta: str):
        self._texto = texto_resposta

    def invoke(self, inputs, config=None):
        return {"messages": [AIMessage(content=self._texto)]}


def _build_router_agent(monkeypatch: pytest.MonkeyPatch, texto_resposta: str) -> RouterAgent:
    stub = _StubRunnable(texto_resposta)
    monkeypatch.setattr(router_agent_module, "create_agent", lambda **kwargs: stub)
    return RouterAgent(llm=None, system_prompt="prompt de teste", tools=[])


def test_router_agent_extrai_rota_do_protocolo(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTE=estoque\nPERGUNTA_ORIGINAL=quanto de papelão temos?"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="quanto de papelão temos?")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["estoque"]
    assert resultado["agentes_chamados"] == ["router"]


def test_router_agent_sem_protocolo_cai_em_fim(monkeypatch: pytest.MonkeyPatch):
    texto = (
        "Olá! Posso te ajudar com estoque, coletas, pedidos, financeiro, "
        "rotas ou dúvidas sobre o Renovaí."
    )
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="oi")]},
        config={},
    )

    assert resultado["proximos_agentes"] == []


def test_router_agent_nao_vaza_protocolo_quando_encaminha(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTE=estoque\nPERGUNTA_ORIGINAL=quanto de papelão temos?"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="quanto de papelão temos?")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["estoque"]
    assert resultado["messages"] == []


def test_router_agent_mantem_resposta_direta_no_historico(monkeypatch: pytest.MonkeyPatch):
    texto = "Olá! Posso te ajudar com estoque ou dúvidas sobre o Renovaí."
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="oi")]},
        config={},
    )

    assert resultado["proximos_agentes"] == []
    assert resultado["messages"] == [{"role": "assistant", "content": texto}]


def test_router_agent_extrai_multiplas_rotas(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTES=estoque, rag_faq\nPERGUNTA_ORIGINAL=quanto de PET temos e qual a norma de armazenamento?"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="quanto de PET temos e qual a norma?")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["estoque", "rag_faq"]
    assert resultado["messages"] == []
    assert resultado["agentes_chamados"] == ["router"]


def test_router_agent_extrai_multiplas_rotas_com_colchetes(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTES=['estoque', 'rag_faq']\nPERGUNTA_ORIGINAL=pergunta teste"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="pergunta teste")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["estoque", "rag_faq"]
    assert resultado["messages"] == []


def test_router_agent_extrai_rota_de_bloco_markdown_json(monkeypatch: pytest.MonkeyPatch):
    texto = (
        "```json\n"
        "{\n"
        '  "ROUTE": "coleta",\n'
        '  "PERGUNTA_ORIGINAL": "Qual foi a última coleta que foi realizada?"\n'
        "}\n"
        "```"
    )
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="Qual foi a última coleta que foi realizada?")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["coleta"]
    assert resultado["messages"] == []


def test_router_agent_extrai_multiplas_rotas_de_json(monkeypatch: pytest.MonkeyPatch):
    texto = '{"ROUTES": ["estoque", "rag_faq"], "PERGUNTA_ORIGINAL": "teste"}'
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="teste")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["estoque", "rag_faq"]
    assert resultado["messages"] == []


def test_router_agent_extrai_rota_com_dois_pontos(monkeypatch: pytest.MonkeyPatch):
    texto = "ROUTE: coleta\nPERGUNTA_ORIGINAL: Qual a última coleta?"
    agent = _build_router_agent(monkeypatch, texto)

    resultado = agent.run(
        {"messages": [HumanMessage(content="Qual a última coleta?")]},
        config={},
    )

    assert resultado["proximos_agentes"] == ["coleta"]
    assert resultado["messages"] == []